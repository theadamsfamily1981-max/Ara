//
// Memcopy DMA Kernel - SNN Core v0 Stub
//
// Simple memory-to-memory copy engine for AI Tile v0 validation.
// This is a placeholder for the full SNN inference core.
//
// Features:
//   - Avalon-MM master interface for DDR4 access
//   - Command-driven operation via CSR registers
//   - Burst transfers for high bandwidth
//   - Configurable transfer size (up to 16MB)
//
// Usage:
//   1. CSR writes CMD_SRC, CMD_DST, CMD_LEN
//   2. CSR writes CMD_CFG with MODE=0x0 (MEMCOPY)
//   3. CSR writes CTRL.START=1
//   4. Kernel reads from SRC, writes to DST
//   5. Kernel sets STATUS.DONE when complete
//
// Author: A10PED Neuromorphic Project
// License: BSD-3-Clause
//

module memcopy_kernel #(
    parameter DATA_WIDTH = 512,          // Match DDR4 interface width
    parameter ADDR_WIDTH = 64,           // Full 64-bit addressing
    parameter BURST_SIZE = 16,           // 16-beat bursts (16 x 512 bits = 1KB)
    parameter MAX_TRANSFER_BYTES = 16777216  // 16MB max
)(
    // Clock and reset
    input  wire                     clk,
    input  wire                     rst_n,

    // Avalon-MM master interface (to DDR4)
    output reg  [ADDR_WIDTH-1:0]    avm_address,
    output reg                      avm_read,
    input  wire [DATA_WIDTH-1:0]    avm_readdata,
    input  wire                     avm_readdatavalid,
    output reg                      avm_write,
    output reg  [DATA_WIDTH-1:0]    avm_writedata,
    input  wire                     avm_waitrequest,
    output reg  [7:0]               avm_burstcount,
    output wire [DATA_WIDTH/8-1:0]  avm_byteenable,

    // Control interface (from CSR)
    input  wire                     cmd_start,
    input  wire [63:0]              cmd_src_addr,
    input  wire [63:0]              cmd_dst_addr,
    input  wire [31:0]              cmd_length,     // Bytes
    input  wire [31:0]              cmd_config,

    // Status interface (to CSR)
    output reg                      core_busy,
    output reg                      core_done,
    output reg                      core_error,
    output reg  [7:0]               error_code,
    output reg  [31:0]              perf_cycles
);

    // Byte enable (all bytes valid for full-width transfers)
    assign avm_byteenable = {(DATA_WIDTH/8){1'b1}};

    // FSM states
    localparam STATE_IDLE        = 3'd0;
    localparam STATE_READ_BURST  = 3'd1;
    localparam STATE_READ_WAIT   = 3'd2;
    localparam STATE_WRITE_BURST = 3'd3;
    localparam STATE_WRITE_WAIT  = 3'd4;
    localparam STATE_DONE        = 3'd5;
    localparam STATE_ERROR       = 3'd6;

    reg [2:0] state, next_state;

    // Internal registers
    reg [63:0] src_addr_reg;
    reg [63:0] dst_addr_reg;
    reg [31:0] bytes_remaining;
    reg [31:0] cycles_count;

    // FIFO for read data buffering
    reg [DATA_WIDTH-1:0] data_fifo [0:BURST_SIZE-1];
    reg [4:0] fifo_wr_ptr;
    reg [4:0] fifo_rd_ptr;
    reg [4:0] fifo_count;

    // Burst management
    reg [7:0] burst_length;
    reg [7:0] read_beats_issued;
    reg [7:0] read_beats_received;
    reg [7:0] write_beats_issued;

    // Calculate burst length based on remaining bytes
    wire [31:0] bytes_this_burst = (bytes_remaining > (BURST_SIZE * (DATA_WIDTH/8))) ?
                                    (BURST_SIZE * (DATA_WIDTH/8)) :
                                    bytes_remaining;
    wire [7:0]  beats_this_burst = (bytes_this_burst + (DATA_WIDTH/8) - 1) / (DATA_WIDTH/8);

    //
    // FSM: State transition
    //
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= STATE_IDLE;
        else
            state <= next_state;
    end

    //
    // FSM: Next state logic
    //
    always @(*) begin
        next_state = state;

        case (state)
            STATE_IDLE: begin
                if (cmd_start)
                    next_state = STATE_READ_BURST;
            end

            STATE_READ_BURST: begin
                if (read_beats_issued == burst_length && !avm_waitrequest)
                    next_state = STATE_READ_WAIT;
            end

            STATE_READ_WAIT: begin
                if (read_beats_received == burst_length)
                    next_state = STATE_WRITE_BURST;
            end

            STATE_WRITE_BURST: begin
                if (write_beats_issued == burst_length && !avm_waitrequest) begin
                    if (bytes_remaining == 0)
                        next_state = STATE_DONE;
                    else
                        next_state = STATE_READ_BURST;
                end
            end

            STATE_DONE: begin
                next_state = STATE_IDLE;
            end

            STATE_ERROR: begin
                next_state = STATE_IDLE;
            end

            default: next_state = STATE_IDLE;
        endcase
    end

    //
    // FSM: Output logic and datapath
    //
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Reset all outputs
            avm_address <= 64'h0;
            avm_read <= 1'b0;
            avm_write <= 1'b0;
            avm_writedata <= {DATA_WIDTH{1'b0}};
            avm_burstcount <= 8'h0;

            core_busy <= 1'b0;
            core_done <= 1'b0;
            core_error <= 1'b0;
            error_code <= 8'h0;
            perf_cycles <= 32'h0;

            src_addr_reg <= 64'h0;
            dst_addr_reg <= 64'h0;
            bytes_remaining <= 32'h0;
            cycles_count <= 32'h0;

            fifo_wr_ptr <= 5'h0;
            fifo_rd_ptr <= 5'h0;
            fifo_count <= 5'h0;

            burst_length <= 8'h0;
            read_beats_issued <= 8'h0;
            read_beats_received <= 8'h0;
            write_beats_issued <= 8'h0;

        end else begin
            // Clear one-shot signals
            avm_read <= 1'b0;
            avm_write <= 1'b0;
            core_done <= 1'b0;

            // Increment cycle counter when busy
            if (core_busy)
                cycles_count <= cycles_count + 1'b1;

            case (state)
                STATE_IDLE: begin
                    if (cmd_start) begin
                        // Latch command parameters
                        src_addr_reg <= cmd_src_addr;
                        dst_addr_reg <= cmd_dst_addr;
                        bytes_remaining <= cmd_length;
                        cycles_count <= 32'h0;

                        // Validate parameters
                        if ((cmd_src_addr[5:0] != 6'h0) ||   // 64-byte aligned
                            (cmd_dst_addr[5:0] != 6'h0) ||
                            (cmd_length[5:0] != 6'h0) ||
                            (cmd_length > MAX_TRANSFER_BYTES)) begin
                            // Error: invalid alignment or size
                            next_state = STATE_ERROR;
                            core_error <= 1'b1;
                            error_code <= 8'h02;  // Alignment error
                        end else begin
                            core_busy <= 1'b1;
                            core_error <= 1'b0;
                            error_code <= 8'h0;
                        end
                    end else begin
                        core_busy <= 1'b0;
                    end
                end

                STATE_READ_BURST: begin
                    // Calculate burst length for this iteration
                    burst_length <= beats_this_burst;
                    read_beats_issued <= 8'h0;
                    read_beats_received <= 8'h0;
                    fifo_wr_ptr <= 5'h0;

                    // Issue read burst
                    if (!avm_waitrequest) begin
                        avm_address <= src_addr_reg;
                        avm_read <= 1'b1;
                        avm_burstcount <= beats_this_burst;
                        read_beats_issued <= 1'b1;
                    end
                end

                STATE_READ_WAIT: begin
                    // Wait for read data to arrive
                    if (avm_readdatavalid) begin
                        // Store in FIFO
                        data_fifo[fifo_wr_ptr] <= avm_readdata;
                        fifo_wr_ptr <= fifo_wr_ptr + 1'b1;
                        read_beats_received <= read_beats_received + 1'b1;
                    end
                end

                STATE_WRITE_BURST: begin
                    write_beats_issued <= 8'h0;
                    fifo_rd_ptr <= 5'h0;

                    // Issue write burst
                    if (!avm_waitrequest && (fifo_rd_ptr < burst_length)) begin
                        avm_address <= dst_addr_reg;
                        avm_write <= 1'b1;
                        avm_writedata <= data_fifo[fifo_rd_ptr];
                        avm_burstcount <= (fifo_rd_ptr == 0) ? burst_length : 8'h0;

                        fifo_rd_ptr <= fifo_rd_ptr + 1'b1;
                        write_beats_issued <= write_beats_issued + 1'b1;

                        if (write_beats_issued == burst_length - 1) begin
                            // Update addresses and remaining bytes
                            src_addr_reg <= src_addr_reg + bytes_this_burst;
                            dst_addr_reg <= dst_addr_reg + bytes_this_burst;
                            bytes_remaining <= bytes_remaining - bytes_this_burst;
                        end
                    end
                end

                STATE_DONE: begin
                    core_busy <= 1'b0;
                    core_done <= 1'b1;
                    perf_cycles <= cycles_count;
                end

                STATE_ERROR: begin
                    core_busy <= 1'b0;
                    core_error <= 1'b1;
                end

                default: begin
                    core_busy <= 1'b0;
                end
            endcase
        end
    end

endmodule
