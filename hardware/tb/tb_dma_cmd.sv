`timescale 1ns/1ps

// Teaching example. dma_cmd_model is a protocol stub, NOT a real DMA.
// It does not access AXI, DDR or SRAM and cannot verify data correctness.
module tb_dma_cmd;
    logic clk = 0;
    always #5 clk = ~clk; // 10 ns period; simulation stimulus only
    logic rst_n = 0;

    logic dma_cmd_valid = 0;
    wire  dma_cmd_ready;
    logic dma_cmd_dir = 0;
    logic [63:0] dma_cmd_ext_addr = 0;
    logic [1:0] dma_cmd_region = 0;
    logic [31:0] dma_cmd_local_offset = 0;
    logic [31:0] dma_cmd_byte_count = 0;
    wire dma_rsp_valid;
    logic dma_rsp_ready = 0;
    wire [7:0] dma_rsp_status;
    wire dma_busy;

    dma_cmd_model #(.LATENCY(4)) dut (
        .clk(clk), .rst_n(rst_n),
        .dma_cmd_valid(dma_cmd_valid), .dma_cmd_ready(dma_cmd_ready),
        .dma_cmd_dir(dma_cmd_dir), .dma_cmd_ext_addr(dma_cmd_ext_addr),
        .dma_cmd_region(dma_cmd_region),
        .dma_cmd_local_offset(dma_cmd_local_offset),
        .dma_cmd_byte_count(dma_cmd_byte_count),
        .dma_rsp_valid(dma_rsp_valid), .dma_rsp_ready(dma_rsp_ready),
        .dma_rsp_status(dma_rsp_status), .dma_busy(dma_busy)
    );

    // Driver: drive at falling edges; the DUT samples at rising edges.
    // Holding valid while ready is low is intentional and required.
    task automatic send_cmd(
        input logic dir,
        input logic [1:0] region,
        input logic [63:0] address,
        input logic [31:0] count
    );
        @(negedge clk);
        dma_cmd_dir = dir;
        dma_cmd_region = region;
        dma_cmd_ext_addr = address;
        dma_cmd_local_offset = 0;
        dma_cmd_byte_count = count;
        dma_cmd_valid = 1;

        // valid is already high: an edge with ready=1 is the handshake.
        do @(posedge clk); while (dma_cmd_ready !== 1'b1);

        @(negedge clk);
        dma_cmd_valid = 0;
        // Fields are allowed to change AFTER acceptance. Poison them to
        // catch a model that evaluates live command fields later.
        dma_cmd_dir = 1;
        dma_cmd_region = 3;
        dma_cmd_ext_addr = 64'hdead_beef;
        dma_cmd_local_offset = 32'hffff_ffff;
        dma_cmd_byte_count = 0;
    endtask

    task automatic wait_rsp_visible;
        do @(negedge clk); while (dma_rsp_valid !== 1'b1);
    endtask

    task automatic take_rsp;
        @(negedge clk);
        dma_rsp_ready = 1;
        do @(posedge clk); while (dma_rsp_valid !== 1'b1);
        @(negedge clk);
        dma_rsp_ready = 0;
    endtask

    // Safe only for this model: there are NO outstanding external transactions.
    task automatic reset_model;
        @(negedge clk);
        rst_n = 0;
        dma_cmd_valid = 0;
        dma_rsp_ready = 0;
        repeat (3) @(posedge clk);
        @(negedge clk);
        if (dma_cmd_ready !== 0 || dma_rsp_valid !== 0 || dma_busy !== 0)
            $fatal(1, "Reset state is incorrect");
        rst_n = 1;
    endtask

    // Protocol monitor and independent expected-response scoreboard.
    logic [7:0] expected_status [0:31];
    integer wr = 0, rd = 0;
    integer total_cmd = 0, total_rsp = 0, total_aborted = 0;
    logic was_cmd_stalled = 0, was_rsp_stalled = 0;
    logic [130:0] previous_cmd = 0;
    logic [7:0] previous_status = 0;
    wire [130:0] command_payload = {
        dma_cmd_dir, dma_cmd_ext_addr, dma_cmd_region,
        dma_cmd_local_offset, dma_cmd_byte_count
    };

    always @(posedge clk) begin
        if (!rst_n) begin
            total_aborted = total_aborted + (wr - rd);
            wr = 0;
            rd = 0;
            was_cmd_stalled = 0;
            was_rsp_stalled = 0;
        end else begin
            if (was_cmd_stalled &&
                (dma_cmd_valid !== 1'b1 || command_payload !== previous_cmd))
                $fatal(1, "Command changed while stalled");
            if (was_rsp_stalled &&
                (dma_rsp_valid !== 1'b1 || dma_rsp_status !== previous_status))
                $fatal(1, "Response changed while stalled");
            if ((wr != rd) && dma_cmd_ready !== 1'b0)
                $fatal(1, "New command allowed before old response consumed");
            if (dma_rsp_valid && (wr == rd))
                $fatal(1, "Response without an accepted command");
            if (dma_rsp_valid && dma_busy !== 1'b1)
                $fatal(1, "busy must include response waiting time");

            if (dma_cmd_valid && dma_cmd_ready) begin
                if (wr >= 32) $fatal(1, "Scoreboard capacity exceeded");
                if ((!dma_cmd_dir && dma_cmd_region <= 1) ||
                    (dma_cmd_dir && dma_cmd_region == 2)) begin
                    expected_status[wr] = (dma_cmd_byte_count == 0) ? 8'h01 : 8'h00;
                end else begin
                    expected_status[wr] = 8'h03;
                end
                wr = wr + 1;
                total_cmd = total_cmd + 1;
                $display("%0t accepted command %0d", $time, total_cmd);
            end
            if (dma_rsp_valid && dma_rsp_ready) begin
                if (dma_rsp_status !== expected_status[rd])
                    $fatal(1, "Status mismatch: expected %02x got %02x",
                           expected_status[rd], dma_rsp_status);
                rd = rd + 1;
                total_rsp = total_rsp + 1;
                $display("%0t consumed response %0d status=%02x",
                         $time, total_rsp, dma_rsp_status);
            end
            was_cmd_stalled = dma_cmd_valid && !dma_cmd_ready;
            was_rsp_stalled = dma_rsp_valid && !dma_rsp_ready;
            previous_cmd = command_payload;
            previous_status = dma_rsp_status;
        end
    end

    // Scenario: send and receive are separate operations.
    initial begin
        $dumpfile("tb_dma_cmd.vcd");
        $dumpvars(0, tb_dma_cmd);
        reset_model();

        $display("TEST 1: ordinary command");
        send_cmd(0, 0, 64'h1000, 128);
        take_rsp();

        $display("TEST 2: blocked response and pending next command");
        send_cmd(0, 1, 64'h2000, 64);
        wait_rsp_visible(); // ready remains low
        fork
            begin
                // This driver must hold its payload until the old response
                // is consumed and the model becomes idle again.
                send_cmd(1, 2, 64'h3000, 32);
            end
            begin
                repeat (4) @(negedge clk);
                take_rsp();
            end
        join
        take_rsp(); // complete the queued STORE C command

        $display("TEST 3: zero length returns INVALID_ARG");
        send_cmd(0, 0, 64'h1000, 0);
        take_rsp();

        $display("TEST 4: STORE A returns UNSUPPORTED");
        send_cmd(1, 0, 64'h1000, 16);
        take_rsp();

        $display("TEST 5: reset cancels a model task; no old response");
        send_cmd(0, 0, 64'h1000, 16);
        reset_model();
        repeat (8) @(negedge clk);
        send_cmd(0, 0, 64'h4000, 8);
        take_rsp();

        repeat (3) @(negedge clk);
        if (wr != rd || total_cmd != 7 || total_rsp != 6 || total_aborted != 1)
            $fatal(1, "Count mismatch: cmd=%0d rsp=%0d aborted=%0d",
                   total_cmd, total_rsp, total_aborted);
        $display("PASS: commands=7 responses=6 reset_aborted=1");
        $finish;
    end

    // A deadlocked test must fail, not run forever.
    initial begin
        #10000;
        $fatal(1, "Global simulation timeout");
    end
endmodule
