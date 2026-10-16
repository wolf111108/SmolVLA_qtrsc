`timescale 1ns/1ps

// ===========================================================================
// Self-checking testbench for the tiny_gemm v0.1 specification.
//
// The testbench owns everything the spec puts outside the DUT:
//   * clock, reset, start command and completion acceptance
//   * A / W read-only storage models and the C writable storage model
//   * an independent software reference for C (triple loop over a_data/w_data)
//   * protocol checking, result checking and timeout protection
//
// The reference model never reads DUT internal state, so a wrong accumulator
// inside the DUT cannot be hidden by a matching reference.
//
// Scenarios covered (spec section 9):
//   all zero / all one / mixed signs / extremes / address directed / random,
//   read stalls and read latency, C write stalls, completion backpressure,
//   consecutive tasks, reset during the read phase, reset during the write
//   phase, and recovery after reset.
//
// Marker lines:
//   PASS: tiny_gemm all tests passed      -> every check succeeded
//   ERROR: TB_FAULT_INJECT ...            -> checker self-test (negative run)
// ===========================================================================
module tb_tiny_gemm;

    // ------------------------------------------------------------------ knobs
    localparam integer CLK_HALF   = 5;      // 10 ns period
    localparam integer MAX_STALL  = 8;      // spec section 10: random wait <= 8 cycles
    localparam integer TASK_LIMIT = 10000;  // spec section 10: per-task cycle limit
    localparam integer A_DEPTH    = 32;
    localparam integer W_DEPTH    = 32;
    localparam integer C_DEPTH    = 16;

    // ------------------------------------------------------------ clock/reset
    logic clk = 1'b0;
    always #CLK_HALF clk = ~clk;

    logic rst_n = 1'b0;

    // ------------------------------------------------------------ DUT signals
    logic        cmd_valid  = 1'b0;
    wire         cmd_ready;
    wire         done_valid;
    logic        done_ready = 1'b0;
    wire         busy;

    wire         a_req_valid;
    wire         a_req_ready;
    wire  [4:0]  a_req_addr;
    wire         a_rsp_valid;
    wire         a_rsp_ready;
    wire  [7:0]  a_rsp_data;

    wire         w_req_valid;
    wire         w_req_ready;
    wire  [4:0]  w_req_addr;
    wire         w_rsp_valid;
    wire         w_rsp_ready;
    wire  [7:0]  w_rsp_data;

    wire         c_wr_valid;
    wire         c_wr_ready;
    wire  [3:0]  c_wr_addr;
    wire [31:0]  c_wr_data;

    // ------------------------------------------------- storage model interfaces
    logic        a_ld_valid = 1'b0;
    logic [4:0]  a_ld_addr  = 5'd0;
    logic [7:0]  a_ld_data  = 8'd0;
    logic        w_ld_valid = 1'b0;
    logic [4:0]  w_ld_addr  = 5'd0;
    logic [7:0]  w_ld_data  = 8'd0;

    logic        a_req_stall = 1'b0;
    logic        w_req_stall = 1'b0;
    logic        c_wr_stall  = 1'b0;
    logic [3:0]  a_lat_extra = 4'd0;
    logic [3:0]  w_lat_extra = 4'd0;

    wire  [15:0] a_req_count;
    wire  [15:0] a_rsp_count;
    wire  [15:0] w_req_count;
    wire  [15:0] w_rsp_count;
    wire  [15:0] c_wr_count;

    logic [3:0]  c_rd_addr = 4'd0;
    wire [31:0]  c_rd_data;

    // ------------------------------------------------------------ DUT instance
    tiny_gemm dut (
        .clk(clk),
        .rst_n(rst_n),

        .cmd_valid(cmd_valid),
        .cmd_ready(cmd_ready),
        .done_valid(done_valid),
        .done_ready(done_ready),
        .busy(busy),

        .a_req_valid(a_req_valid),
        .a_req_ready(a_req_ready),
        .a_req_addr(a_req_addr),
        .a_rsp_valid(a_rsp_valid),
        .a_rsp_ready(a_rsp_ready),
        .a_rsp_data(a_rsp_data),

        .w_req_valid(w_req_valid),
        .w_req_ready(w_req_ready),
        .w_req_addr(w_req_addr),
        .w_rsp_valid(w_rsp_valid),
        .w_rsp_ready(w_rsp_ready),
        .w_rsp_data(w_rsp_data),

        .c_wr_valid(c_wr_valid),
        .c_wr_ready(c_wr_ready),
        .c_wr_addr(c_wr_addr),
        .c_wr_data(c_wr_data)
    );

    // ------------------------------------------------- storage model instances
    tiny_gemm_rom #(
        .DEPTH(A_DEPTH), .ADDR_W(5), .DATA_W(8), .CNT_W(4)
    ) u_rom_a (
        .clk(clk),
        .rst_n(rst_n),
        .ld_valid(a_ld_valid),
        .ld_addr(a_ld_addr),
        .ld_data(a_ld_data),
        .req_stall(a_req_stall),
        .lat_extra(a_lat_extra),
        .req_valid(a_req_valid),
        .req_ready(a_req_ready),
        .req_addr(a_req_addr),
        .rsp_valid(a_rsp_valid),
        .rsp_ready(a_rsp_ready),
        .rsp_data(a_rsp_data),
        .req_count(a_req_count),
        .rsp_count(a_rsp_count)
    );

    tiny_gemm_rom #(
        .DEPTH(W_DEPTH), .ADDR_W(5), .DATA_W(8), .CNT_W(4)
    ) u_rom_w (
        .clk(clk),
        .rst_n(rst_n),
        .ld_valid(w_ld_valid),
        .ld_addr(w_ld_addr),
        .ld_data(w_ld_data),
        .req_stall(w_req_stall),
        .lat_extra(w_lat_extra),
        .req_valid(w_req_valid),
        .req_ready(w_req_ready),
        .req_addr(w_req_addr),
        .rsp_valid(w_rsp_valid),
        .rsp_ready(w_rsp_ready),
        .rsp_data(w_rsp_data),
        .req_count(w_req_count),
        .rsp_count(w_rsp_count)
    );

    tiny_gemm_cmem #(
        .DEPTH(C_DEPTH), .ADDR_W(4), .DATA_W(32)
    ) u_cmem (
        .clk(clk),
        .rst_n(rst_n),
        .wr_stall(c_wr_stall),
        .wr_valid(c_wr_valid),
        .wr_ready(c_wr_ready),
        .wr_addr(c_wr_addr),
        .wr_data(c_wr_data),
        .rd_addr(c_rd_addr),
        .rd_data(c_rd_data),
        .wr_count(c_wr_count)
    );

    // --------------------------------------------------------- stimulus data
    // Signed declarations matter: 8'h80 is -128 and 8'hFF is -1 here.
    logic signed [7:0]  a_data [0:A_DEPTH-1];
    logic signed [7:0]  w_data [0:W_DEPTH-1];
    logic signed [31:0] exp_c  [0:C_DEPTH-1];

    // Fixed, printed seeds so a rerun reproduces the exact same matrices.
    integer seed_data = 32'h5EED_1234;
    integer seed_bp   = 32'h0DDB_A11;

    integer li, lj, lk;             // sequential loop / index helpers
    integer acc_r, prod_r;          // reference model temporaries

    integer cmd_total   = 0;
    integer done_total  = 0;
    integer cycle_count = 0;
    integer task_count  = 0;
    integer wr_idx      = 0;        // next expected C element address
    integer c_wr_total  = 0;
    logic   sb_enable   = 1'b1;     // C write scoreboard enable
    logic   bp_enable   = 1'b0;     // random backpressure enable
    logic   task_seen   = 1'b0;     // a command was accepted and not yet completed

    // performance bookkeeping
    integer exec_start_cycle = 0;
    integer last_write_cycle = 0;
    integer exec_cycles      = 0;
    integer done_wait_cycles = 0;
    integer a_req_wait_cyc   = 0;
    integer w_req_wait_cyc   = 0;
    integer a_rsp_wait_cyc   = 0;
    integer w_rsp_wait_cyc   = 0;
    integer c_wr_wait_cyc    = 0;

    // =======================================================================
    // Data preparation
    // =======================================================================
    task automatic fill_constant(input logic [7:0] aval, input logic [7:0] wval);
        for (li = 0; li < A_DEPTH; li = li + 1) begin
            a_data[li] = aval;
            w_data[li] = wval;
        end
    endtask

    task automatic fill_random;
        for (li = 0; li < A_DEPTH; li = li + 1)
            a_data[li] = $random(seed_data);
        for (li = 0; li < W_DEPTH; li = li + 1)
            w_data[li] = $random(seed_data);
    endtask

    // Program both read-only arrays through their load ports.
    task automatic load_data;
        for (li = 0; li < A_DEPTH; li = li + 1) begin
            @(negedge clk);
            a_ld_valid = 1'b1;
            a_ld_addr  = li;
            a_ld_data  = a_data[li];
            w_ld_valid = 1'b1;
            w_ld_addr  = li;
            w_ld_data  = w_data[li];
        end
        @(negedge clk);
        a_ld_valid = 1'b0;
        w_ld_valid = 1'b0;
        a_ld_addr  = 5'd0;
        w_ld_addr  = 5'd0;
    endtask

    // Independent reference: a plain triple loop, sign-extended products,
    // 32-bit accumulation. It reads only the TB-side arrays.
    task automatic compute_reference;
        for (li = 0; li < 4; li = li + 1) begin
            for (lj = 0; lj < 4; lj = lj + 1) begin
                acc_r = 0;
                for (lk = 0; lk < 8; lk = lk + 1) begin
                    prod_r = $signed(a_data[li*8 + lk]) * $signed(w_data[lk*4 + lj]);
                    acc_r  = acc_r + prod_r;
                end
                exp_c[li*4 + lj] = acc_r;
            end
        end
    endtask

    task automatic dump_c;
        $display("INFO: C matrix (row major, expected | actual)");
        for (li = 0; li < 4; li = li + 1) begin
            c_rd_addr = li*4 + 0; #1;
            $write("%12d | %-12d", exp_c[li*4+0], $signed(c_rd_data));
            c_rd_addr = li*4 + 1; #1;
            $write("%12d | %-12d", exp_c[li*4+1], $signed(c_rd_data));
            c_rd_addr = li*4 + 2; #1;
            $write("%12d | %-12d", exp_c[li*4+2], $signed(c_rd_data));
            c_rd_addr = li*4 + 3; #1;
            $write("%12d | %-12d\n", exp_c[li*4+3], $signed(c_rd_data));
        end
    endtask

    // =======================================================================
    // Driver / checker tasks
    // =======================================================================
    // Drive a start command; `valid` is released right after acceptance so an
    // extended cmd_ready cannot start a second task.
    task automatic start_command;
        @(negedge clk);
        cmd_valid = 1'b1;
        do @(posedge clk); while (cmd_ready !== 1'b1);
        @(negedge clk);
        cmd_valid = 1'b0;
    endtask

    task automatic take_done(input integer delay_cycles);
        do @(negedge clk); while (done_valid !== 1'b1);
        repeat (delay_cycles) @(negedge clk);
        done_ready = 1'b1;
        do @(posedge clk); while (done_valid !== 1'b1);
        @(negedge clk);
        done_ready = 1'b0;
    endtask

    // Read the whole C array back and compare with `!==`, so an X or Z in the
    // storage model is reported as a mismatch rather than passing silently.
    task automatic check_c_memory;
        for (li = 0; li < C_DEPTH; li = li + 1) begin
            c_rd_addr = li;
            #1;
            if (c_rd_data !== exp_c[li])
                $fatal(1, "C[%0d] readback mismatch: got %08x expected %08x (t=%0t)",
                       li, c_rd_data, exp_c[li], $time);
        end
    endtask

    // One complete task: reference, command, completion, exact handshake
    // counts, and a full C read-back.
    integer cmd_base, done_base, areq_base, arsp_base, wreq_base, wrsp_base, cwr_base;

    task automatic do_task(input integer done_delay);
        compute_reference();
        wr_idx     = 0;
        cmd_base   = cmd_total;
        done_base  = done_total;
        areq_base  = a_req_count;
        arsp_base  = a_rsp_count;
        wreq_base  = w_req_count;
        wrsp_base  = w_rsp_count;
        cwr_base   = c_wr_count;

        start_command();
        take_done(done_delay);

        if (cmd_total  - cmd_base  !== 1)
            $fatal(1, "expected 1 command handshake, got %0d", cmd_total - cmd_base);
        if (done_total - done_base !== 1)
            $fatal(1, "expected 1 completion handshake, got %0d", done_total - done_base);
        if (a_req_count - areq_base !== 128)
            $fatal(1, "expected 128 A read requests, got %0d", a_req_count - areq_base);
        if (a_rsp_count - arsp_base !== 128)
            $fatal(1, "expected 128 A read responses, got %0d", a_rsp_count - arsp_base);
        if (w_req_count - wreq_base !== 128)
            $fatal(1, "expected 128 W read requests, got %0d", w_req_count - wreq_base);
        if (w_rsp_count - wrsp_base !== 128)
            $fatal(1, "expected 128 W read responses, got %0d", w_rsp_count - wrsp_base);
        if (c_wr_count - cwr_base !== C_DEPTH)
            $fatal(1, "expected %0d C writes, got %0d", C_DEPTH, c_wr_count - cwr_base);
        if (wr_idx !== C_DEPTH)
            $fatal(1, "C scoreboard only saw %0d writes", wr_idx);

        check_c_memory();
        task_count = task_count + 1;
    endtask

    // Reset asserted in the middle of a task. Checks that the cancelled task
    // leaves every output idle and never delivers a stale completion.
    task automatic reset_dut;
        @(negedge clk);
        rst_n      = 1'b0;
        cmd_valid  = 1'b0;
        done_ready = 1'b0;
        repeat (3) @(posedge clk);
        @(negedge clk);
        if (cmd_ready    !== 1'b0) $fatal(1, "cmd_ready not low during reset (t=%0t)",    $time);
        if (done_valid   !== 1'b0) $fatal(1, "done_valid not low during reset (t=%0t)",   $time);
        if (busy         !== 1'b0) $fatal(1, "busy not low during reset (t=%0t)",         $time);
        if (a_req_valid  !== 1'b0) $fatal(1, "a_req_valid not low during reset (t=%0t)",  $time);
        if (w_req_valid  !== 1'b0) $fatal(1, "w_req_valid not low during reset (t=%0t)",  $time);
        if (a_rsp_ready  !== 1'b0) $fatal(1, "a_rsp_ready not low during reset (t=%0t)",  $time);
        if (w_rsp_ready  !== 1'b0) $fatal(1, "w_rsp_ready not low during reset (t=%0t)",  $time);
        if (c_wr_valid   !== 1'b0) $fatal(1, "c_wr_valid not low during reset (t=%0t)",   $time);
        rst_n = 1'b1;
        repeat (2) @(negedge clk);
    endtask

    task automatic check_no_stale_done;
        repeat (12) begin
            @(negedge clk);
            if (done_valid !== 1'b0)
                $fatal(1, "stale done_valid %0d cycles after reset (t=%0t)", 12, $time);
        end
    endtask

    // =======================================================================
    // Protocol monitor
    // =======================================================================
    logic       was_a_req_stalled, was_w_req_stalled;
    logic       was_a_rsp_stalled, was_w_rsp_stalled;
    logic       was_c_wr_stalled,  was_done_stalled;
    logic [4:0] a_addr_q, w_addr_q;
    logic [7:0] a_rsp_data_q, w_rsp_data_q;
    logic [3:0] c_addr_q;
    logic [31:0] c_data_q;

    always @(posedge clk) begin
        if (!rst_n) begin
            was_a_req_stalled <= 1'b0;
            was_w_req_stalled <= 1'b0;
            was_a_rsp_stalled <= 1'b0;
            was_w_rsp_stalled <= 1'b0;
            was_c_wr_stalled  <= 1'b0;
            was_done_stalled  <= 1'b0;
            task_seen         <= 1'b0;
        end else begin
            cycle_count <= cycle_count + 1;

            if (cmd_valid && cmd_ready) begin
                cmd_total        <= cmd_total + 1;
                task_seen        <= 1'b1;
                exec_start_cycle <= cycle_count;
            end
            if (done_valid && done_ready) begin
                done_total       <= done_total + 1;
                task_seen        <= 1'b0;
                done_wait_cycles <= cycle_count - last_write_cycle;
            end

            // ---- senders must hold valid and payload while a transfer waits
            if (was_a_req_stalled && (a_req_valid !== 1'b1 || a_req_addr !== a_addr_q))
                $fatal(1, "A read request changed while stalled (t=%0t)", $time);
            if (was_w_req_stalled && (w_req_valid !== 1'b1 || w_req_addr !== w_addr_q))
                $fatal(1, "W read request changed while stalled (t=%0t)", $time);
            if (was_a_rsp_stalled && (a_rsp_valid !== 1'b1 || a_rsp_data !== a_rsp_data_q))
                $fatal(1, "A read response changed while stalled (t=%0t)", $time);
            if (was_w_rsp_stalled && (w_rsp_valid !== 1'b1 || w_rsp_data !== w_rsp_data_q))
                $fatal(1, "W read response changed while stalled (t=%0t)", $time);
            if (was_c_wr_stalled &&
                (c_wr_valid !== 1'b1 || c_wr_addr !== c_addr_q || c_wr_data !== c_data_q))
                $fatal(1, "C write changed while stalled (t=%0t)", $time);
            if (was_done_stalled && done_valid !== 1'b1)
                $fatal(1, "done_valid dropped while done_ready was low (t=%0t)", $time);

            // ---- nothing may happen without an accepted command -----------
            if (!task_seen) begin
                if (c_wr_valid)
                    $fatal(1, "C write without an accepted command (t=%0t)", $time);
                if (done_valid)
                    $fatal(1, "completion without an accepted command (t=%0t)", $time);
            end

            // ---- completion must not be consumed together with a command --
            if (done_valid && done_ready && cmd_ready)
                $fatal(1, "cmd_ready high in the done handshake cycle (t=%0t)", $time);

            // ---- the last C write must happen before the completion -------
            if (done_valid && c_wr_valid)
                $fatal(1, "completion reported in the same cycle as a C write (t=%0t)", $time);

            was_a_req_stalled <= a_req_valid && !a_req_ready;
            was_w_req_stalled <= w_req_valid && !w_req_ready;
            was_a_rsp_stalled <= a_rsp_valid && !a_rsp_ready;
            was_w_rsp_stalled <= w_rsp_valid && !w_rsp_ready;
            was_c_wr_stalled  <= c_wr_valid && !c_wr_ready;
            was_done_stalled  <= done_valid && !done_ready;
            a_addr_q          <= a_req_addr;
            w_addr_q          <= w_req_addr;
            a_rsp_data_q      <= a_rsp_data;
            w_rsp_data_q      <= w_rsp_data;
            c_addr_q          <= c_wr_addr;
            c_data_q          <= c_wr_data;
        end
    end

    // =======================================================================
    // C write scoreboard: order and value, plus the latency measurement
    // =======================================================================
    always @(posedge clk) begin
        if (rst_n && sb_enable && c_wr_valid && c_wr_ready) begin
            if (wr_idx >= C_DEPTH) begin
                $fatal(1, "more than %0d C writes in one task (t=%0t)", C_DEPTH, $time);
            end else begin
                if (c_wr_addr !== wr_idx[3:0])
                    $fatal(1, "C write address out of order: got %0d expected %0d (t=%0t)",
                           c_wr_addr, wr_idx, $time);
                if (c_wr_data !== exp_c[wr_idx])
                    $fatal(1, "C write data mismatch at C[%0d]: got %08x expected %08x (t=%0t)",
                           wr_idx, c_wr_data, exp_c[wr_idx], $time);
                wr_idx      = wr_idx + 1;
                c_wr_total  = c_wr_total + 1;
                if (wr_idx == C_DEPTH) begin
                    last_write_cycle = cycle_count;
                    exec_cycles      = cycle_count - exec_start_cycle;
                end
            end
        end
    end

    // =======================================================================
    // Watchdogs and stall accounting
    // =======================================================================
    integer task_cycles = 0;

    always @(posedge clk) begin
        if (!rst_n) begin
            task_cycles <= 0;
        end else if (busy) begin
            if (task_cycles >= TASK_LIMIT)
                $fatal(1, "per-task cycle limit of %0d exceeded (t=%0t)", TASK_LIMIT, $time);
            task_cycles <= task_cycles + 1;
        end else begin
            task_cycles <= 0;
        end
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            a_req_wait_cyc <= 0;
            w_req_wait_cyc <= 0;
            a_rsp_wait_cyc <= 0;
            w_rsp_wait_cyc <= 0;
            c_wr_wait_cyc  <= 0;
        end else if (busy) begin
            if (a_req_valid && !a_req_ready) a_req_wait_cyc <= a_req_wait_cyc + 1;
            if (w_req_valid && !w_req_ready) w_req_wait_cyc <= w_req_wait_cyc + 1;
            if (a_rsp_valid && !a_rsp_ready) a_rsp_wait_cyc <= a_rsp_wait_cyc + 1;
            if (w_rsp_valid && !w_rsp_ready) w_rsp_wait_cyc <= w_rsp_wait_cyc + 1;
            if (c_wr_valid && !c_wr_ready)  c_wr_wait_cyc  <= c_wr_wait_cyc + 1;
        end
    end

    // =======================================================================
    // Bounded random backpressure
    // =======================================================================
    integer bp_r;
    integer hold_a = 0, hold_w = 0, hold_c = 0;

    initial begin : backpressure
        a_req_stall = 1'b0;
        w_req_stall = 1'b0;
        c_wr_stall  = 1'b0;
        a_lat_extra = 4'd0;
        w_lat_extra = 4'd0;
        forever begin
            @(negedge clk);
            if (!bp_enable) begin
                a_req_stall = 1'b0;
                w_req_stall = 1'b0;
                c_wr_stall  = 1'b0;
                a_lat_extra = 4'd0;
                w_lat_extra = 4'd0;
                hold_a      = 0;
                hold_w      = 0;
                hold_c      = 0;
            end else begin
                // ---- A read port -------------------------------------
                if (hold_a > 0) begin
                    hold_a      = hold_a - 1;
                    a_req_stall = 1'b1;
                end else begin
                    bp_r = $random(seed_bp);
                    if (bp_r < 0) bp_r = -bp_r;
                    a_req_stall = ((bp_r % 3) == 0);
                    if (a_req_stall) hold_a = (bp_r % MAX_STALL);
                    bp_r = $random(seed_bp);
                    if (bp_r < 0) bp_r = -bp_r;
                    a_lat_extra = (bp_r % 3);
                end

                // ---- W read port -------------------------------------
                if (hold_w > 0) begin
                    hold_w      = hold_w - 1;
                    w_req_stall = 1'b1;
                end else begin
                    bp_r = $random(seed_bp);
                    if (bp_r < 0) bp_r = -bp_r;
                    w_req_stall = ((bp_r % 3) == 0);
                    if (w_req_stall) hold_w = (bp_r % MAX_STALL);
                    bp_r = $random(seed_bp);
                    if (bp_r < 0) bp_r = -bp_r;
                    w_lat_extra = (bp_r % 3);
                end

                // ---- C write port ------------------------------------
                if (hold_c > 0) begin
                    hold_c     = hold_c - 1;
                    c_wr_stall = 1'b1;
                end else begin
                    bp_r = $random(seed_bp);
                    if (bp_r < 0) bp_r = -bp_r;
                    c_wr_stall = ((bp_r % 2) == 0);
                    if (c_wr_stall) hold_c = (bp_r % MAX_STALL);
                end
            end
        end
    end

    // =======================================================================
    // Scenario sequence
    // =======================================================================
    initial begin : main
`ifdef TB_DUMP
        $dumpfile("tb_tiny_gemm.vcd");
        $dumpvars(0, tb_tiny_gemm);
`endif
        $display("INFO: tiny_gemm v0.1 testbench, data seed=0x%08x bp seed=0x%08x",
                 seed_data, seed_bp);

        reset_dut();

        // ---- TEST 1: all zero ------------------------------------------
        $display("TEST 1: all-zero matrices");
        fill_constant(8'h00, 8'h00);
        load_data();
        do_task(0);

        // ---- TEST 2: all one (every C must be 8) ------------------------
        $display("TEST 2: all-one matrices");
        fill_constant(8'h01, 8'h01);
        load_data();
        do_task(0);

        // ---- TEST 3: mixed signs and zeros ------------------------------
        $display("TEST 3: mixed positive / negative / zero");
        for (li = 0; li < A_DEPTH; li = li + 1) begin
            a_data[li] = (li % 3 == 0) ?  8'd7  :
                         (li % 3 == 1) ? -8'd5  : 8'd0;
        end
        for (li = 0; li < W_DEPTH; li = li + 1) begin
            w_data[li] = (li % 4 == 0) ? -8'd9  :
                         (li % 4 == 1) ?  8'd6  :
                         (li % 4 == 2) ? -8'd1  : 8'd0;
        end
        load_data();
        do_task(0);

        // ---- TEST 4: signed extremes ------------------------------------
        $display("TEST 4: extremes (-128 / 127) and product width");
        for (li = 0; li < A_DEPTH; li = li + 1)
            a_data[li] = (li % 3 == 0) ? 8'h80 : ((li % 3 == 1) ? 8'h7F : 8'h00);
        for (li = 0; li < W_DEPTH; li = li + 1)
            w_data[li] = (li % 4 == 0) ? 8'h80 :
                         ((li % 4 == 1) ? 8'h7F :
                         ((li % 4 == 2) ? 8'hFF : 8'h01));
        load_data();
        do_task(0);
        dump_c();

        // ---- TEST 5: address directed -----------------------------------
        // A[2,5] = +3, W[5,1] = -3  =>  C[2,1] = -9, everything else 0.
        $display("TEST 5: address directed (row/column indexing)");
        fill_constant(8'h00, 8'h00);
        a_data[2*8 + 5] = 8'h03;
        w_data[5*4 + 1] = 8'hFD;
        load_data();
        compute_reference();
        if (exp_c[2*4 + 1] !== -32'sd9)
            $fatal(1, "reference model itself is wrong: C[2,1]=%0d", exp_c[2*4 + 1]);
        do_task(0);

        // ---- TEST 6: random matrices, fixed printed seed -----------------
        $display("TEST 6: pseudo-random matrices");
        for (li = 0; li < 4; li = li + 1) begin
            fill_random();
            load_data();
            do_task(0);
        end

        // ---- TEST 7: read stalls and read latency ------------------------
        $display("TEST 7: backpressure - A/W request stalls and response latency");
        bp_enable = 1'b1;
        fill_random();
        load_data();
        do_task(0);

        // ---- TEST 8: C write stalls and completion backpressure ----------
        $display("TEST 8: backpressure - C write stalls and delayed completion");
        fill_random();
        load_data();
        do_task(7);

        // ---- TEST 9: consecutive tasks -----------------------------------
        $display("TEST 9: three consecutive tasks without reset");
        for (li = 0; li < 3; li = li + 1) begin
            fill_random();
            load_data();
            do_task(0);
        end
        bp_enable = 1'b0;
        repeat (3) @(negedge clk);

        // ---- TEST 10: reset during the read phase ------------------------
        $display("TEST 10: reset during the A/W read phase");
        fill_constant(8'h11, 8'h22);
        load_data();
        sb_enable = 1'b0;
        start_command();
        repeat (10) @(negedge clk);
        reset_dut();
        check_no_stale_done();
        sb_enable = 1'b1;

        // ---- TEST 11: reset during the C write phase ---------------------
        $display("TEST 11: reset during the C write phase");
        fill_constant(8'h33, 8'h44);
        load_data();
        cwr_base  = c_wr_count;
        sb_enable = 1'b0;
        start_command();
        begin : wait_first_write
            integer guard;
            guard = 0;
            while ((c_wr_count <= cwr_base) && (guard < TASK_LIMIT)) begin
                @(negedge clk);
                guard = guard + 1;
            end
            if (c_wr_count <= cwr_base)
                $fatal(1, "no C write observed before the read-phase reset test");
        end
        reset_dut();
        check_no_stale_done();
        sb_enable = 1'b1;

        // ---- TEST 12: recovery after reset -------------------------------
        $display("TEST 12: a fresh task after reset completes correctly");
        fill_random();
        load_data();
        do_task(0);

        // ---- checker self-test (negative run only) -----------------------
`ifdef TB_FAULT_INJECT
        $display("INFO: TB_FAULT_INJECT corrupting one expected C value");
        $display("ERROR: TB_FAULT_INJECT the comparison below must fail now");
        exp_c[5] = exp_c[5] + 32'sd1;
        check_c_memory();
        $display("FAIL: TB_FAULT_INJECT the corrupted value was not detected");
        $finish;
`endif

        // ---- summary -----------------------------------------------------
        if (cycle_count === 0)
            $fatal(1, "cycle counter never advanced");
        $display("INFO: tasks=%0d c_writes=%0d", task_count, c_wr_total);
        $display("INFO: last task cycles cmd->last C write = %0d", exec_cycles);
        $display("INFO: last task cycles waiting for completion acceptance = %0d",
                 done_wait_cycles);
        $display("INFO: stall cycles A req=%0d W req=%0d A rsp=%0d W rsp=%0d C wr=%0d",
                 a_req_wait_cyc, w_req_wait_cyc, a_rsp_wait_cyc, w_rsp_wait_cyc,
                 c_wr_wait_cyc);
        $display("PASS: tiny_gemm all tests passed");
        $finish;
    end

    // A deadlocked test must fail rather than run forever.
    initial begin
        #2000000;   // 2 ms of simulated time
        $fatal(1, "global simulation timeout");
    end

endmodule
