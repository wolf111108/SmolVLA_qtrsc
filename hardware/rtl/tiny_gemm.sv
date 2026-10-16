`timescale 1ns/1ps

// ===========================================================================
// tiny_gemm -- fixed-size signed INT8 matrix multiply-accumulate engine.
//
//   C[m,n] = sum(k = 0..7) A[m,k] * W[k,n]
//   A : 4x8 signed INT8 (row major, element index a_addr = 8*m + k, 0..31)
//   W : 8x4 signed INT8 (row major, element index w_addr = 4*k + n, 0..31)
//   C : 4x4 signed INT32 (row major, element index c_addr = 4*m + n, 0..15)
//
// v0.1 scope: ONE multiplier and ONE accumulator, both reused, one task in
// flight, fixed sizes, one clock domain, synchronous active-low reset.
// No bias, no scale, no FP8, no sparsity skipping, no DMA/AXI, no multiple PEs.
//
// A / W read data and C write data are NOT stored here. The three storage
// models live outside the DUT (see tb/tiny_gemm_mem.sv), which keeps this
// module free of any memory technology and makes the storage models swappable.
//
// Handshake / ordering rules implemented here:
//   * every transfer is `valid && ready` on a rising clock edge
//   * request and response on the A/W ports are two independent handshakes
//   * at most one outstanding request per read port
//   * cmd_ready is low in the same cycle the completion is consumed, and comes
//     back one cycle later
//   * the completion is only raised after the 16th C write handshake, and the
//     last accumulator term is captured one cycle before it drives the bus
//   * while rst_n is sampled low the task is cancelled and every output is idle
// ===========================================================================
module tiny_gemm (
    input  wire        clk,
    input  wire        rst_n,

    // ---- start command --------------------------------------------------
    input  wire        cmd_valid,
    output wire        cmd_ready,

    // ---- completion notification ----------------------------------------
    output wire        done_valid,
    input  wire        done_ready,
    output wire        busy,

    // ---- A read port ----------------------------------------------------
    output wire        a_req_valid,
    input  wire        a_req_ready,
    output wire [4:0]  a_req_addr,
    input  wire        a_rsp_valid,
    output wire        a_rsp_ready,
    input  wire [7:0]  a_rsp_data,

    // ---- W read port ----------------------------------------------------
    output wire        w_req_valid,
    input  wire        w_req_ready,
    output wire [4:0]  w_req_addr,
    input  wire        w_rsp_valid,
    output wire        w_rsp_ready,
    input  wire [7:0]  w_rsp_data,

    // ---- C write port ---------------------------------------------------
    output wire        c_wr_valid,
    input  wire        c_wr_ready,
    output wire [3:0]  c_wr_addr,
    output wire [31:0] c_wr_data
);

    // ------------------------------------------------------------------ sizes
    localparam integer M_DIM = 4;   // A rows / C rows
    localparam integer K_DIM = 8;   // A cols / W rows
    localparam integer N_DIM = 4;   // W cols / C cols

    // ------------------------------------------------------------------ state
    localparam [3:0] S_RESET   = 4'd15;  // only held while rst_n is sampled low
    localparam [3:0] S_IDLE    = 4'd0;
    localparam [3:0] S_INIT_C  = 4'd1;
    localparam [3:0] S_A_REQ   = 4'd2;
    localparam [3:0] S_A_WAIT  = 4'd3;
    localparam [3:0] S_W_REQ   = 4'd4;
    localparam [3:0] S_W_WAIT  = 4'd5;
    localparam [3:0] S_MAC     = 4'd6;
    localparam [3:0] S_C_WRITE = 4'd7;
    localparam [3:0] S_DONE    = 4'd8;

    logic [3:0] state;

    // --------------------------------------------------------------- counters
    logic [1:0] m;      // 0 .. M_DIM-1
    logic [1:0] n;      // 0 .. N_DIM-1
    logic [2:0] k;      // 0 .. K_DIM-1

    // -------------------------------------------------------------- datapath
    logic signed [7:0]  a_reg;      // latched A operand
    logic signed [7:0]  w_reg;      // latched W operand
    logic signed [31:0] acc;        // running accumulator
    logic signed [31:0] c_res_q;    // finished result, held stable for the write

    // Signed arithmetic is made explicit: the operands are sign-extended to
    // 16 bits BEFORE the multiply so 8'hFF (= -1) is never read as 255 and
    // 8'h80 * 8'h80 (= 16384) is not truncated.
    wire signed [15:0] a_ext      = {{8{a_reg[7]}}, a_reg};
    wire signed [15:0] w_ext      = {{8{w_reg[7]}}, w_reg};
    wire signed [15:0] product    = a_ext * w_ext;
    wire signed [31:0] product_ext = {{16{product[15]}}, product};
    wire signed [31:0] acc_next    = acc + product_ext;

    // ------------------------------------------------------- command / done
    assign cmd_ready  = (state == S_IDLE);
    assign done_valid = (state == S_DONE);
    assign busy       = (state != S_IDLE) && (state != S_RESET);

    // ------------------------------------------------------------ read ports
    // {m, k} = 8*m + k and {k, n} = 4*k + n, so the address arithmetic is a
    // plain concatenation and the layout is visible in the port mapping.
    assign a_req_valid = (state == S_A_REQ);
    assign a_req_addr  = {m, k};
    assign a_rsp_ready = (state == S_A_WAIT);

    assign w_req_valid = (state == S_W_REQ);
    assign w_req_addr  = {k, n};
    assign w_rsp_ready = (state == S_W_WAIT);

    // ------------------------------------------------------------- write port
    assign c_wr_valid = (state == S_C_WRITE);
    assign c_wr_addr  = {m, n};     // 4*m + n, ascending 0..15
    assign c_wr_data  = c_res_q;

    // ---------------------------------------------------------------- control
    always @(posedge clk) begin
        if (!rst_n) begin
            // Cancel the task, drop any pending transfer, go idle.
            state   <= S_RESET;
            m       <= 2'd0;
            n       <= 2'd0;
            k       <= 3'd0;
            a_reg   <= 8'sd0;
            w_reg   <= 8'sd0;
            acc     <= 32'sd0;
            c_res_q <= 32'sd0;
        end else begin
            case (state)
                // ---- wait for a start command ---------------------------
                S_IDLE: if (cmd_valid && cmd_ready) begin
                    m     <= 2'd0;
                    n     <= 2'd0;
                    state <= S_INIT_C;
                end

                // ---- start a fresh C element ----------------------------
                S_INIT_C: begin
                    acc   <= 32'sd0;
                    k     <= 3'd0;
                    state <= S_A_REQ;
                end

                // ---- read A, then read W, then accumulate ---------------
                S_A_REQ: if (a_req_valid && a_req_ready)
                    state <= S_A_WAIT;

                S_A_WAIT: if (a_rsp_valid && a_rsp_ready) begin
                    a_reg <= a_rsp_data;
                    state <= S_W_REQ;
                end

                S_W_REQ: if (w_req_valid && w_req_ready)
                    state <= S_W_WAIT;

                S_W_WAIT: if (w_rsp_valid && w_rsp_ready) begin
                    w_reg <= w_rsp_data;
                    state <= S_MAC;
                end

                S_MAC: begin
                    // acc_next already includes the term latched in this
                    // iteration, so the value captured into c_res_q below is
                    // the complete sum -- the write port sees it a cycle later.
                    acc <= acc_next;
                    if (k == K_DIM - 1) begin
                        c_res_q <= acc_next;
                        state   <= S_C_WRITE;
                    end else begin
                        k     <= k + 3'd1;
                        state <= S_A_REQ;
                    end
                end

                // ---- write back, advancing n then m ---------------------
                S_C_WRITE: if (c_wr_valid && c_wr_ready) begin
                    if (n < N_DIM - 1) begin
                        n     <= n + 2'd1;
                        state <= S_INIT_C;
                    end else if (m < M_DIM - 1) begin
                        m     <= m + 2'd1;
                        n     <= 2'd0;
                        state <= S_INIT_C;
                    end else begin
                        state <= S_DONE;
                    end
                end

                // ---- hold the completion until it is consumed -----------
                S_DONE: if (done_valid && done_ready)
                    state <= S_IDLE;

                // covers S_RESET release and any unreachable encoding
                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
