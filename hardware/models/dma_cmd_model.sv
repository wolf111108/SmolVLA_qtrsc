`timescale 1ns/1ps

// Deliberately simplified behavior model. All addresses are merely captured.
// No address-range checks, memory movement, burst protocol or data scoreboard.
module dma_cmd_model #(
    parameter integer LATENCY = 4
) (
    input wire clk,
    input wire rst_n,
    input wire dma_cmd_valid,
    output wire dma_cmd_ready,
    input wire dma_cmd_dir,
    input wire [63:0] dma_cmd_ext_addr,
    input wire [1:0] dma_cmd_region,
    input wire [31:0] dma_cmd_local_offset,
    input wire [31:0] dma_cmd_byte_count,
    output wire dma_rsp_valid,
    input wire dma_rsp_ready,
    output wire [7:0] dma_rsp_status,
    output wire dma_busy
);
    localparam [1:0] IDLE=0, RUN=1, RESP=2;
    logic [1:0] state;
    logic [7:0] saved_status;
    logic [130:0] saved_command;
    integer remaining;

    initial begin
        if (LATENCY < 1) $fatal(1, "LATENCY must be at least 1");
    end

    // rst_n is sampled synchronously; outputs depend on registered state.
    assign dma_cmd_ready = (state == IDLE);
    assign dma_rsp_valid = (state == RESP);
    assign dma_rsp_status = saved_status;
    assign dma_busy = (state == RUN || state == RESP);

    // RESET state=3 holds ready low during sampled reset.
    always @(posedge clk) begin
        if (!rst_n) begin
            state <= 2'd3;
            saved_status <= 0;
            saved_command <= 0;
            remaining <= 0;
        end else begin
            case (state)
                IDLE: if (dma_cmd_valid && dma_cmd_ready) begin
                    saved_command <= {
                        dma_cmd_dir, dma_cmd_ext_addr, dma_cmd_region,
                        dma_cmd_local_offset, dma_cmd_byte_count
                    };
                    if (!((!dma_cmd_dir && dma_cmd_region <= 1) ||
                          (dma_cmd_dir && dma_cmd_region == 2)))
                        saved_status <= 8'h03;
                    else if (dma_cmd_byte_count == 0)
                        saved_status <= 8'h01;
                    else
                        saved_status <= 8'h00;
                    remaining <= LATENCY;
                    state <= RUN;
                end
                RUN: begin
                    if (remaining <= 1) state <= RESP;
                    else remaining <= remaining - 1;
                end
                RESP: if (dma_rsp_ready) state <= IDLE;
                default: state <= IDLE;
            endcase
        end
    end
endmodule
