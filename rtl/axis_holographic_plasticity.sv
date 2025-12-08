//-----------------------------------------------------------------------------
// Ara Holographic Plasticity Controller
// Iteration 34: The Forge
//
// This module orchestrates the full matrix plasticity update.
// It iterates through all rows (layers) of the holographic memory,
// feeding each to the plasticity_row_engine for update.
//
// The update runs in the background using Port B of the dual-port BRAM,
// while Port A continues serving inference requests.
//
// Target: Intel Stratix-10 @ 450 MHz
// Full update latency: ~200 µs for 512 rows × 16384 bits
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps
`default_nettype none

module axis_holographic_plasticity #(
    parameter int DIM        = 16384,  // Weight vector dimension
    parameter int N_LAYERS   = 512,    // Number of rows (layers)
    parameter int CHUNK_BITS = 512,    // Bits per memory access
    parameter int ACC_WIDTH  = 7       // Accumulator precision
)(
    input  wire              clk,
    input  wire              rst_n,

    // ==========================================================================
    // Control Interface (from HAL / ChiefOfStaff)
    // ==========================================================================
    input  wire              i_trigger,         // Pulse to start plasticity update
    input  wire signed [7:0] i_reward,          // Global reward signal
    input  wire [DIM-1:0]    i_context_hv,      // Input HV that caused reward

    output wire              o_busy,            // Update in progress
    output wire              o_done,            // Full matrix update complete

    // ==========================================================================
    // Memory Interface - Port B (Plasticity)
    // Directly connects to dual-port BRAM holding weights and accumulators
    // ==========================================================================

    // Address for weight memory
    output reg  [31:0]                       o_weight_addr,
    output reg                               o_weight_we,
    input  wire [CHUNK_BITS-1:0]             i_weight_rdata,
    output wire [CHUNK_BITS-1:0]             o_weight_wdata,

    // Address for accumulator memory
    output reg  [31:0]                       o_accum_addr,
    output reg                               o_accum_we,
    input  wire [CHUNK_BITS*ACC_WIDTH-1:0]   i_accum_rdata,
    output wire [CHUNK_BITS*ACC_WIDTH-1:0]   o_accum_wdata,

    // ==========================================================================
    // Statistics (for monitoring)
    // ==========================================================================
    output reg  [31:0]       o_update_count,    // Total plasticity events
    output reg  [31:0]       o_last_reward      // Last reward value (debug)
);

    // -------------------------------------------------------------------------
    // Local Parameters
    // -------------------------------------------------------------------------
    localparam int NUM_CHUNKS   = DIM / CHUNK_BITS;
    localparam int CHUNK_ADDR_W = $clog2(NUM_CHUNKS);
    localparam int LAYER_ADDR_W = $clog2(N_LAYERS);

    // -------------------------------------------------------------------------
    // Controller FSM
    // -------------------------------------------------------------------------
    typedef enum logic [2:0] {
        S_IDLE      = 3'b000,
        S_LATCH     = 3'b001,
        S_START_ROW = 3'b010,
        S_WAIT_ROW  = 3'b011,
        S_NEXT_ROW  = 3'b100,
        S_COMPLETE  = 3'b101
    } ctrl_state_t;

    ctrl_state_t ctrl_state;

    // -------------------------------------------------------------------------
    // Internal Registers
    // -------------------------------------------------------------------------
    reg [LAYER_ADDR_W-1:0] current_layer;
    reg signed [7:0]       latched_reward;
    reg [DIM-1:0]          latched_context;
    reg                    matrix_busy;
    reg                    matrix_done;

    // Row engine interface
    wire                    row_busy;
    wire                    row_done;
    reg                     row_start;
    wire [CHUNK_ADDR_W-1:0] row_chunk_addr;
    wire                    row_mem_we;
    wire [CHUNK_BITS-1:0]   row_weight_out;
    wire [CHUNK_BITS*ACC_WIDTH-1:0] row_accum_out;

    // -------------------------------------------------------------------------
    // Row Engine Instance
    // -------------------------------------------------------------------------
    plasticity_row_engine #(
        .DIM        (DIM),
        .CHUNK_BITS (CHUNK_BITS),
        .ACC_WIDTH  (ACC_WIDTH)
    ) u_row_engine (
        .clk           (clk),
        .rst_n         (rst_n),

        .i_start       (row_start),
        .i_reward      (latched_reward),
        .o_busy        (row_busy),
        .o_done        (row_done),

        .i_input_hv    (latched_context),

        .o_chunk_addr  (row_chunk_addr),
        .o_mem_we      (row_mem_we),

        .i_weight_chunk(i_weight_rdata),
        .o_weight_chunk(row_weight_out),

        .i_accum_chunk (i_accum_rdata),
        .o_accum_chunk (row_accum_out)
    );

    // Connect row engine outputs to memory interface
    assign o_weight_wdata = row_weight_out;
    assign o_accum_wdata  = row_accum_out;

    // -------------------------------------------------------------------------
    // Address Generation
    // Memory layout:
    //   Weight[layer][chunk] at address: layer * NUM_CHUNKS + chunk
    //   Accum[layer][chunk]  at address: layer * NUM_CHUNKS + chunk (separate space)
    // -------------------------------------------------------------------------
    wire [31:0] base_addr;
    assign base_addr = current_layer * NUM_CHUNKS + row_chunk_addr;

    always_comb begin
        o_weight_addr = base_addr;
        o_accum_addr  = base_addr;
        o_weight_we   = row_mem_we;
        o_accum_we    = row_mem_we;
    end

    // -------------------------------------------------------------------------
    // Main Controller FSM
    // -------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ctrl_state     <= S_IDLE;
            current_layer  <= '0;
            latched_reward <= '0;
            latched_context<= '0;
            matrix_busy    <= 1'b0;
            matrix_done    <= 1'b0;
            row_start      <= 1'b0;
            o_update_count <= '0;
            o_last_reward  <= '0;
        end else begin
            // Default: clear one-shot signals
            row_start   <= 1'b0;
            matrix_done <= 1'b0;

            case (ctrl_state)
                // -------------------------------------------------------------
                S_IDLE: begin
                    matrix_busy <= 1'b0;

                    if (i_trigger && i_reward != 8'sd0) begin
                        ctrl_state <= S_LATCH;
                    end
                end

                // -------------------------------------------------------------
                S_LATCH: begin
                    // Latch inputs so they remain stable during update
                    latched_reward  <= i_reward;
                    latched_context <= i_context_hv;
                    o_last_reward   <= {{24{i_reward[7]}}, i_reward};

                    current_layer <= '0;
                    matrix_busy   <= 1'b1;
                    ctrl_state    <= S_START_ROW;
                end

                // -------------------------------------------------------------
                S_START_ROW: begin
                    // Trigger the row engine
                    row_start  <= 1'b1;
                    ctrl_state <= S_WAIT_ROW;
                end

                // -------------------------------------------------------------
                S_WAIT_ROW: begin
                    // Wait for row engine to complete
                    if (row_done) begin
                        ctrl_state <= S_NEXT_ROW;
                    end
                end

                // -------------------------------------------------------------
                S_NEXT_ROW: begin
                    if (current_layer == N_LAYERS - 1) begin
                        // All rows done
                        ctrl_state <= S_COMPLETE;
                    end else begin
                        // Move to next row
                        current_layer <= current_layer + 1'b1;
                        ctrl_state    <= S_START_ROW;
                    end
                end

                // -------------------------------------------------------------
                S_COMPLETE: begin
                    matrix_done    <= 1'b1;
                    o_update_count <= o_update_count + 1'b1;
                    ctrl_state     <= S_IDLE;
                end

                default: ctrl_state <= S_IDLE;
            endcase
        end
    end

    // Output assignments
    assign o_busy = matrix_busy;
    assign o_done = matrix_done;

    // -------------------------------------------------------------------------
    // Debug / ILA Signals
    // -------------------------------------------------------------------------
    `ifdef SIMULATION
    initial begin
        $display("axis_holographic_plasticity instantiated:");
        $display("  DIM=%0d, N_LAYERS=%0d, CHUNK_BITS=%0d", DIM, N_LAYERS, CHUNK_BITS);
        $display("  NUM_CHUNKS=%0d, Total bits=%0d", NUM_CHUNKS, DIM * N_LAYERS);
    end
    `endif

endmodule

`default_nettype wire
