`timescale 1ns/1ps

// ===========================================================================
// Storage models for the tiny_gemm testbench.
//
// The spec keeps A / W / C storage OUTSIDE the DUT. These models are the
// testbench side of that boundary; replacing them with an on-chip RAM wrapper
// later must not require touching the matrix arithmetic.
//
// Both models implement the same convention the DUT was written against:
//   * every transfer is `valid && ready` on a rising clock edge
//   * a stalled sender holds valid and payload stable
//   * the read response can never appear in the same cycle as the request
//     handshake (earliest is the following cycle)
//   * at most one outstanding read request per port
//
// The two stall hooks (req_stall / lat_extra / wr_stall) exist only so the
// testbench can inject bounded random backpressure and read/write latency.
// ===========================================================================

// ---------------------------------------------------------------------------
// Read-only memory model: request/response protocol, one outstanding request.
//
//   req_ready = idle && !req_stall
//   effective response latency = 1 + lat_extra cycles
//
// The response data is a combinational read of the registered response
// address, so the array contents may keep changing while a response is being
// held; only the address is latched, which is what the protocol promises.
// ---------------------------------------------------------------------------
module tiny_gemm_rom #(
    parameter integer DEPTH  = 32,
    parameter integer ADDR_W = 5,
    parameter integer DATA_W = 8,
    parameter integer CNT_W  = 4
) (
    input  wire                clk,
    input  wire                rst_n,

    // testbench load port (only way the array is written)
    input  wire                ld_valid,
    input  wire [ADDR_W-1:0]   ld_addr,
    input  wire [DATA_W-1:0]   ld_data,

    // testbench stall controls
    input  wire                req_stall,
    input  wire [CNT_W-1:0]    lat_extra,

    // read request, driven by the DUT
    input  wire                req_valid,
    output wire                req_ready,
    input  wire [ADDR_W-1:0]   req_addr,

    // read response, consumed by the DUT
    output wire                rsp_valid,
    input  wire                rsp_ready,
    output wire [DATA_W-1:0]   rsp_data,

    // observation counters
    output wire [15:0]         req_count,
    output wire [15:0]         rsp_count
);

    logic [DATA_W-1:0] mem [0:DEPTH-1];

    logic              pend_q;        // request accepted, response not yet issued
    logic [ADDR_W-1:0] pend_addr_q;
    logic [CNT_W-1:0]  cnt_q;
    logic              rsp_valid_q;
    logic [ADDR_W-1:0] rsp_addr_q;
    logic [15:0]       req_cnt_q;
    logic [15:0]       rsp_cnt_q;

    wire idle   = ~pend_q & ~rsp_valid_q;
    wire accept = req_valid & req_ready;

    assign req_ready = idle & ~req_stall;
    assign rsp_valid = rsp_valid_q;
    assign rsp_data  = mem[rsp_addr_q];
    assign req_count = req_cnt_q;
    assign rsp_count = rsp_cnt_q;

    always @(posedge clk) begin
        if (!rst_n) begin
            // Outstanding requests are discarded when reset is sampled.
            pend_q      <= 1'b0;
            pend_addr_q <= {ADDR_W{1'b0}};
            cnt_q       <= {CNT_W{1'b0}};
            rsp_valid_q <= 1'b0;
            rsp_addr_q  <= {ADDR_W{1'b0}};
            req_cnt_q   <= 16'd0;
            rsp_cnt_q   <= 16'd0;
        end else begin
            // Testbench load port. Data is written in the same cycle it is
            // presented, so the caller only needs one cycle per element.
            if (ld_valid)
                mem[ld_addr] <= ld_data;

            // These three branches are mutually exclusive: `accept` requires
            // an idle port, and `pend_q` is cleared as soon as the response
            // has been issued.
            if (rsp_valid_q) begin
                if (rsp_ready) begin
                    rsp_valid_q <= 1'b0;
                    rsp_cnt_q   <= rsp_cnt_q + 16'd1;
                end
            end else if (accept) begin
                req_cnt_q <= req_cnt_q + 16'd1;
                if (lat_extra == {CNT_W{1'b0}}) begin
                    // earliest allowed: visible in the cycle after the handshake
                    rsp_valid_q <= 1'b1;
                    rsp_addr_q  <= req_addr;
                end else begin
                    pend_q      <= 1'b1;
                    pend_addr_q <= req_addr;
                    cnt_q       <= lat_extra;
                end
            end else if (pend_q) begin
                if (cnt_q <= {CNT_W{1'b1}}) begin
                    pend_q      <= 1'b0;
                    rsp_valid_q <= 1'b1;
                    rsp_addr_q  <= pend_addr_q;
                end else begin
                    cnt_q <= cnt_q - {CNT_W{1'b1}};
                end
            end
        end
    end

endmodule

// ---------------------------------------------------------------------------
// Writable C memory model.
//
// There is no write response channel in v0.1: the array is updated on the
// `wr_valid && wr_ready` rising edge, so an accepted write is immediately
// committed. `wr_ready` is driven from the testbench (`wr_stall`) so write
// backpressure can be injected.
//
// Array contents survive reset on purpose: a task that was cancelled may have
// written a few elements, and the next full task overwrites all 16 of them.
// ---------------------------------------------------------------------------
module tiny_gemm_cmem #(
    parameter integer DEPTH  = 16,
    parameter integer ADDR_W = 4,
    parameter integer DATA_W = 32
) (
    input  wire               clk,
    input  wire               rst_n,

    // write port, driven by the DUT
    input  wire               wr_stall,
    input  wire               wr_valid,
    output wire               wr_ready,
    input  wire [ADDR_W-1:0]  wr_addr,
    input  wire [DATA_W-1:0]  wr_data,

    // testbench read-back port
    input  wire [ADDR_W-1:0]  rd_addr,
    output wire [DATA_W-1:0]  rd_data,

    // accepted write counter
    output wire [15:0]        wr_count
);

    logic [DATA_W-1:0] mem [0:DEPTH-1];
    logic [15:0]       cnt_q;

    assign wr_ready = ~wr_stall;
    assign rd_data  = mem[rd_addr];
    assign wr_count = cnt_q;

    always @(posedge clk) begin
        if (!rst_n) begin
            cnt_q <= 16'd0;
        end else if (wr_valid && wr_ready) begin
            mem[wr_addr] <= wr_data;
            cnt_q        <= cnt_q + 16'd1;
        end
    end

endmodule
