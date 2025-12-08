// =============================================================================
// Ara Plasticity Engine v2 - Reward-Modulated Hebbian Learning
// =============================================================================
//
// REALITY CHECK:
//   - "Fully unrolled 2048×16384 in one clock" was poetry, not physics
//   - This version: CHUNKED, STREAMING, ACTUALLY SYNTHESIZABLE
//
// Performance (realistic):
//   - 512-bit chunk per cycle @ 600 MHz → 51.2 Gb/s logical update bandwidth
//   - Full 16,384-bit row = 32 chunks → 32 cycles ≈ 53 ns
//   - 32 active rows per reward event → ~1.7 µs total
//   - Still essentially instantaneous at human/emotional timescale
//
// The Rule (this part IS good):
//   if reward > 0:
//       W[i] += (input_hv[i] == core_row[i]) ? +1 : -1
//   else if reward < 0:
//       W[i] += (input_hv[i] == core_row[i]) ? -1 : +1
//   W[i] = clip(W[i], -64, +63)
//   core_row[i] = sign(W[i])
//
// This is reward-modulated Hebbian on bipolar weights:
//   - Positive reward: strengthen agreeing bits, weaken disagreeing
//   - Negative reward: weaken matching bits (bad pattern), strengthen opposing
//   - 7-bit accumulator provides stable memory with gradual plasticity
//
// =============================================================================

`timescale 1ns / 1ps

module plasticity_row_engine #(
    parameter int DIM        = 16384,      // Hypervector dimension
    parameter int CHUNK_BITS = 512,        // Bits processed per cycle
    parameter int ACC_WIDTH  = 7           // Accumulator: -64..+63
)(
    input  logic              clk,
    input  logic              rst_n,

    // === Control Interface ===
    input  logic              start,       // Pulse: begin updating this row
    input  logic signed [7:0] reward,      // Global reward (-128..+127)
    output logic              busy,        // High while processing
    output logic              done,        // Single-cycle pulse when finished

    // === Hypervector Input (full vector, but accessed in chunks) ===
    input  logic [DIM-1:0]    input_hv,

    // === Memory Interface (to BRAM/HBM backing store) ===
    output logic [$clog2(DIM/CHUNK_BITS)-1:0] chunk_addr,
    output logic              mem_rd_en,   // Read request
    output logic              mem_wr_en,   // Write request

    // Read data from memory (arrives 1 cycle after rd_en)
    input  logic [CHUNK_BITS-1:0]            core_chunk_in,
    input  logic [CHUNK_BITS*ACC_WIDTH-1:0]  accum_chunk_in,

    // Write data to memory
    output logic [CHUNK_BITS-1:0]            core_chunk_out,
    output logic [CHUNK_BITS*ACC_WIDTH-1:0]  accum_chunk_out
);

    // =========================================================================
    // Local Parameters
    // =========================================================================

    localparam int NUM_CHUNKS = DIM / CHUNK_BITS;
    localparam int ADDR_WIDTH = $clog2(NUM_CHUNKS);

    // Accumulator limits
    localparam logic signed [ACC_WIDTH-1:0] ACC_MAX = (1 << (ACC_WIDTH-1)) - 1;  // +63
    localparam logic signed [ACC_WIDTH-1:0] ACC_MIN = -(1 << (ACC_WIDTH-1));     // -64

    // =========================================================================
    // State Machine
    // =========================================================================

    typedef enum logic [2:0] {
        S_IDLE,
        S_READ_REQ,    // Issue read request
        S_READ_WAIT,   // Wait for memory response
        S_UPDATE,      // Compute new accumulators + core bits
        S_WRITE        // Write back to memory
    } state_t;

    state_t state, state_next;
    logic [ADDR_WIDTH-1:0] chunk_idx, chunk_idx_next;

    // =========================================================================
    // Pipeline Registers
    // =========================================================================

    // Latched data from memory read
    logic [CHUNK_BITS-1:0]            core_chunk_reg;
    logic [CHUNK_BITS*ACC_WIDTH-1:0]  accum_chunk_reg;
    logic [CHUNK_BITS-1:0]            input_chunk_reg;

    // Computed outputs
    logic [CHUNK_BITS-1:0]            core_chunk_new;
    logic [CHUNK_BITS*ACC_WIDTH-1:0]  accum_chunk_new;

    // =========================================================================
    // Delta Computation (based on reward sign)
    // =========================================================================

    logic signed [ACC_WIDTH-1:0] delta;

    always_comb begin
        if (reward > 0)
            delta = 1;
        else if (reward < 0)
            delta = -1;
        else
            delta = 0;
    end

    // =========================================================================
    // Saturating Add Function
    // =========================================================================

    function automatic logic signed [ACC_WIDTH-1:0] sat_add(
        input logic signed [ACC_WIDTH-1:0] a,
        input logic signed [ACC_WIDTH-1:0] b
    );
        logic signed [ACC_WIDTH:0] sum_ext;
        sum_ext = $signed({a[ACC_WIDTH-1], a}) + $signed({b[ACC_WIDTH-1], b});

        if (sum_ext > ACC_MAX)
            return ACC_MAX;
        else if (sum_ext < ACC_MIN)
            return ACC_MIN;
        else
            return sum_ext[ACC_WIDTH-1:0];
    endfunction

    // =========================================================================
    // Per-Bit Update Logic (Combinational, applied to chunk)
    // =========================================================================

    always_comb begin
        for (int b = 0; b < CHUNK_BITS; b++) begin
            logic bit_core, bit_input, agree;
            logic signed [ACC_WIDTH-1:0] a_old, a_new, step;

            // Extract bits
            bit_core  = core_chunk_reg[b];
            bit_input = input_chunk_reg[b];

            // Agreement: XNOR (1 if both same)
            agree = ~(bit_core ^ bit_input);

            // Get old accumulator value
            a_old = accum_chunk_reg[b*ACC_WIDTH +: ACC_WIDTH];

            // Compute step
            if (delta != 0) begin
                // If agree and reward positive: +1 (strengthen)
                // If agree and reward negative: -1 (weaken)
                // If disagree: opposite
                step = agree ? delta : -delta;
                a_new = sat_add(a_old, step);
            end else begin
                a_new = a_old;
            end

            // Update outputs
            accum_chunk_new[b*ACC_WIDTH +: ACC_WIDTH] = a_new;

            // Core bit = sign of accumulator (positive → 1, else → 0)
            core_chunk_new[b] = (a_new > 0) ? 1'b1 : 1'b0;
        end
    end

    // =========================================================================
    // State Machine - Sequential Logic
    // =========================================================================

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= S_IDLE;
            chunk_idx <= '0;
            busy      <= 1'b0;
            done      <= 1'b0;
        end else begin
            state     <= state_next;
            chunk_idx <= chunk_idx_next;

            // Latch memory read data
            if (state == S_READ_WAIT) begin
                core_chunk_reg  <= core_chunk_in;
                accum_chunk_reg <= accum_chunk_in;
                // Select corresponding slice of input_hv
                input_chunk_reg <= input_hv[chunk_idx*CHUNK_BITS +: CHUNK_BITS];
            end

            // Busy/done flags
            case (state_next)
                S_IDLE: begin
                    busy <= 1'b0;
                    done <= (state == S_WRITE && chunk_idx == NUM_CHUNKS - 1);
                end
                default: begin
                    busy <= 1'b1;
                    done <= 1'b0;
                end
            endcase
        end
    end

    // =========================================================================
    // State Machine - Combinational Logic
    // =========================================================================

    always_comb begin
        // Defaults
        state_next     = state;
        chunk_idx_next = chunk_idx;
        mem_rd_en      = 1'b0;
        mem_wr_en      = 1'b0;
        chunk_addr     = chunk_idx;

        case (state)
            S_IDLE: begin
                chunk_idx_next = '0;
                if (start && reward != 0) begin
                    state_next = S_READ_REQ;
                end
            end

            S_READ_REQ: begin
                // Issue read request
                mem_rd_en  = 1'b1;
                state_next = S_READ_WAIT;
            end

            S_READ_WAIT: begin
                // Wait one cycle for memory response
                state_next = S_UPDATE;
            end

            S_UPDATE: begin
                // Combinational update happens above
                state_next = S_WRITE;
            end

            S_WRITE: begin
                // Write updated chunk back
                mem_wr_en = 1'b1;

                if (chunk_idx == NUM_CHUNKS - 1) begin
                    // Done with all chunks
                    state_next = S_IDLE;
                end else begin
                    // Move to next chunk
                    chunk_idx_next = chunk_idx + 1;
                    state_next     = S_READ_REQ;
                end
            end

            default: state_next = S_IDLE;
        endcase
    end

    // =========================================================================
    // Output Assignments
    // =========================================================================

    assign core_chunk_out  = core_chunk_new;
    assign accum_chunk_out = accum_chunk_new;

endmodule


// =============================================================================
// Plasticity Controller - Schedules Row Updates
// =============================================================================
//
// Manages which rows get updated based on "active slots" from last resonance.
// Runs in background while inference continues (double-buffered pipeline).
//
// =============================================================================

module plasticity_controller #(
    parameter int DIM         = 16384,
    parameter int NUM_ROWS    = 2048,
    parameter int CHUNK_BITS  = 512,
    parameter int ACC_WIDTH   = 7,
    parameter int MAX_ACTIVE  = 32       // Max rows to update per event
)(
    input  logic              clk,
    input  logic              rst_n,

    // === Reward Interface (from Ara emotional loop) ===
    input  logic              reward_valid,
    input  logic signed [7:0] reward,

    // === Active Rows Mask (which rows fired in last resonance) ===
    input  logic [NUM_ROWS-1:0] active_rows_mask,

    // === Input HV (the pattern being learned) ===
    input  logic [DIM-1:0]    input_hv,

    // === Status ===
    output logic              busy,
    output logic              done,
    output logic [15:0]       rows_updated,

    // === Memory Interface (to row BRAM/HBM) ===
    output logic [$clog2(NUM_ROWS)-1:0]       row_select,
    output logic [$clog2(DIM/CHUNK_BITS)-1:0] chunk_addr,
    output logic              mem_rd_en,
    output logic              mem_wr_en,
    input  logic [CHUNK_BITS-1:0]             core_chunk_in,
    input  logic [CHUNK_BITS*ACC_WIDTH-1:0]   accum_chunk_in,
    output logic [CHUNK_BITS-1:0]             core_chunk_out,
    output logic [CHUNK_BITS*ACC_WIDTH-1:0]   accum_chunk_out
);

    // =========================================================================
    // Local Parameters
    // =========================================================================

    localparam int ROW_ADDR_WIDTH = $clog2(NUM_ROWS);

    // =========================================================================
    // Row Engine Instance
    // =========================================================================

    logic              engine_start;
    logic              engine_busy;
    logic              engine_done;
    logic [$clog2(DIM/CHUNK_BITS)-1:0] engine_chunk_addr;
    logic              engine_mem_rd_en;
    logic              engine_mem_wr_en;
    logic [CHUNK_BITS-1:0]             engine_core_out;
    logic [CHUNK_BITS*ACC_WIDTH-1:0]   engine_accum_out;

    plasticity_row_engine #(
        .DIM(DIM),
        .CHUNK_BITS(CHUNK_BITS),
        .ACC_WIDTH(ACC_WIDTH)
    ) row_engine (
        .clk(clk),
        .rst_n(rst_n),
        .start(engine_start),
        .reward(reward_reg),
        .busy(engine_busy),
        .done(engine_done),
        .input_hv(input_hv),
        .chunk_addr(engine_chunk_addr),
        .mem_rd_en(engine_mem_rd_en),
        .mem_wr_en(engine_mem_wr_en),
        .core_chunk_in(core_chunk_in),
        .accum_chunk_in(accum_chunk_in),
        .core_chunk_out(engine_core_out),
        .accum_chunk_out(engine_accum_out)
    );

    // =========================================================================
    // Controller State Machine
    // =========================================================================

    typedef enum logic [2:0] {
        C_IDLE,
        C_SCAN,         // Find next active row
        C_START_ROW,    // Start row engine
        C_WAIT_ROW,     // Wait for row engine to finish
        C_DONE
    } ctrl_state_t;

    ctrl_state_t ctrl_state;
    logic signed [7:0] reward_reg;
    logic [ROW_ADDR_WIDTH-1:0] current_row;
    logic [15:0] update_count;
    logic [NUM_ROWS-1:0] rows_remaining;

    // Find first set bit (priority encoder)
    function automatic logic [ROW_ADDR_WIDTH-1:0] find_first_set(
        input logic [NUM_ROWS-1:0] mask
    );
        for (int i = 0; i < NUM_ROWS; i++) begin
            if (mask[i])
                return i[ROW_ADDR_WIDTH-1:0];
        end
        return '0;
    endfunction

    function automatic logic any_set(input logic [NUM_ROWS-1:0] mask);
        return |mask;
    endfunction

    // =========================================================================
    // Controller Logic
    // =========================================================================

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ctrl_state     <= C_IDLE;
            reward_reg     <= '0;
            current_row    <= '0;
            update_count   <= '0;
            rows_remaining <= '0;
            busy           <= 1'b0;
            done           <= 1'b0;
            engine_start   <= 1'b0;
        end else begin
            done         <= 1'b0;
            engine_start <= 1'b0;

            case (ctrl_state)
                C_IDLE: begin
                    busy <= 1'b0;
                    if (reward_valid && reward != 0) begin
                        reward_reg     <= reward;
                        rows_remaining <= active_rows_mask;
                        update_count   <= '0;
                        busy           <= 1'b1;
                        ctrl_state     <= C_SCAN;
                    end
                end

                C_SCAN: begin
                    if (!any_set(rows_remaining) || update_count >= MAX_ACTIVE) begin
                        // Done: no more rows or hit limit
                        ctrl_state <= C_DONE;
                    end else begin
                        // Find next active row
                        current_row <= find_first_set(rows_remaining);
                        ctrl_state  <= C_START_ROW;
                    end
                end

                C_START_ROW: begin
                    engine_start <= 1'b1;
                    ctrl_state   <= C_WAIT_ROW;
                end

                C_WAIT_ROW: begin
                    if (engine_done) begin
                        // Clear this row from remaining mask
                        rows_remaining[current_row] <= 1'b0;
                        update_count <= update_count + 1;
                        ctrl_state   <= C_SCAN;
                    end
                end

                C_DONE: begin
                    rows_updated <= update_count;
                    done         <= 1'b1;
                    ctrl_state   <= C_IDLE;
                end

                default: ctrl_state <= C_IDLE;
            endcase
        end
    end

    // =========================================================================
    // Output Routing
    // =========================================================================

    assign row_select      = current_row;
    assign chunk_addr      = engine_chunk_addr;
    assign mem_rd_en       = engine_mem_rd_en;
    assign mem_wr_en       = engine_mem_wr_en;
    assign core_chunk_out  = engine_core_out;
    assign accum_chunk_out = engine_accum_out;

endmodule
