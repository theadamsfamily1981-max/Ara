// =============================================================================
// Ara Plasticity Core - Hardware-Agnostic Soul Engine
// =============================================================================
//
// This module is IDENTICAL across all FPGA platforms:
//   - Intel Stratix-10, Agilex
//   - Xilinx VU33P (Forest Kitten), Versal
//   - Any board with sufficient external memory
//
// Platform differences are pushed into the memory adapter.
// This core only sees an abstract memory interface.
//
// The Rule (Reward-Modulated Binary Hebbian):
//   if reward > 0:
//       accum[i] += (input[i] == core[i]) ? +1 : -1
//   else if reward < 0:
//       accum[i] += (input[i] == core[i]) ? -1 : +1
//   accum[i] = clip(accum[i], -64, +63)
//   core[i] = sign(accum[i])
//
// Performance (realistic):
//   - Processes 512 bits per cycle
//   - ~128 cycles per row (read + compute + write)
//   - ~9 µs for 32 active rows @ 450 MHz
//
// =============================================================================

`timescale 1ns / 1ps

`include "ara_soul_config.svh"

module ara_plasticity_core #(
    parameter int ROWS        = ARA_ROWS,
    parameter int DIM         = ARA_DIM,
    parameter int CHUNK_BITS  = ARA_CHUNK_BITS,
    parameter int ACC_WIDTH   = ARA_ACC_WIDTH
)(
    input  logic                           clk,
    input  logic                           rst_n,

    // === Learning Control Interface ===
    input  logic                           start,        // Pulse to begin update
    input  logic [$clog2(ROWS)-1:0]        row_id,       // Which row to adapt
    input  logic signed [7:0]              reward,       // Positive or negative
    input  logic [DIM-1:0]                 input_hv,     // Pattern that caused reward

    output logic                           busy,         // High while processing
    output logic                           done,         // Single-cycle pulse when complete

    // === Abstract Memory Interface ===
    // Platform adapter translates this to HBM/DDR/BRAM
    output logic                           mem_req,      // Request access
    input  logic                           mem_ready,    // Access granted/complete
    output logic [$clog2(DIM/CHUNK_BITS)-1:0] mem_chunk_addr,
    output logic                           mem_we,       // 0=read, 1=write

    // Data to/from memory
    output logic [CHUNK_BITS-1:0]          mem_core_out,
    output logic [CHUNK_BITS*ACC_WIDTH-1:0] mem_accum_out,
    input  logic [CHUNK_BITS-1:0]          mem_core_in,
    input  logic [CHUNK_BITS*ACC_WIDTH-1:0] mem_accum_in
);

    // =========================================================================
    // Local Parameters
    // =========================================================================

    localparam int NUM_CHUNKS = DIM / CHUNK_BITS;
    localparam int CHUNK_ADDR_W = $clog2(NUM_CHUNKS);

    // Accumulator limits
    localparam logic signed [ACC_WIDTH-1:0] ACC_MAX = (1 << (ACC_WIDTH-1)) - 1;  // +63
    localparam logic signed [ACC_WIDTH-1:0] ACC_MIN = -(1 << (ACC_WIDTH-1));     // -64

    // =========================================================================
    // State Machine
    // =========================================================================

    typedef enum logic [2:0] {
        S_IDLE,
        S_READ_REQ,     // Issue read request to memory
        S_READ_WAIT,    // Wait for memory response
        S_COMPUTE,      // Update accumulators and core bits
        S_WRITE_REQ,    // Issue write request
        S_WRITE_WAIT,   // Wait for write completion
        S_NEXT_CHUNK,   // Move to next chunk or finish
        S_DONE
    } state_t;

    state_t state, state_next;

    // =========================================================================
    // Registers
    // =========================================================================

    logic [CHUNK_ADDR_W-1:0] chunk_idx;
    logic [CHUNK_ADDR_W-1:0] chunk_idx_next;

    // Latched control signals
    logic [$clog2(ROWS)-1:0] row_id_reg;
    logic signed [7:0]       reward_reg;

    // Latched memory read data
    logic [CHUNK_BITS-1:0]            core_chunk_reg;
    logic [CHUNK_BITS*ACC_WIDTH-1:0]  accum_chunk_reg;

    // Input HV slice for current chunk
    logic [CHUNK_BITS-1:0] input_chunk;

    // Computed outputs
    logic [CHUNK_BITS-1:0]            core_chunk_new;
    logic [CHUNK_BITS*ACC_WIDTH-1:0]  accum_chunk_new;

    // =========================================================================
    // Input Chunk Selection
    // =========================================================================

    // Select the CHUNK_BITS slice of input_hv corresponding to current chunk
    always_comb begin
        input_chunk = input_hv[chunk_idx * CHUNK_BITS +: CHUNK_BITS];
    end

    // =========================================================================
    // Delta Computation (reward sign determines direction)
    // =========================================================================

    logic signed [ACC_WIDTH-1:0] delta;

    always_comb begin
        if (reward_reg > 0)
            delta = 1;
        else if (reward_reg < 0)
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
    // Per-Bit Hebbian Update (Combinational)
    // =========================================================================

    always_comb begin
        for (int b = 0; b < CHUNK_BITS; b++) begin
            logic bit_core, bit_input, agree;
            logic signed [ACC_WIDTH-1:0] a_old, a_new, step;

            // Extract bits
            bit_core  = core_chunk_reg[b];
            bit_input = input_chunk[b];

            // Agreement: XNOR (1 if both same, treating core as sign bit)
            // core_chunk stores sign: 1 = positive, 0 = negative
            agree = ~(bit_core ^ bit_input);

            // Get old accumulator
            a_old = accum_chunk_reg[b * ACC_WIDTH +: ACC_WIDTH];

            // Compute step based on agreement and reward
            if (delta != 0) begin
                step = agree ? delta : -delta;
                a_new = sat_add(a_old, step);
            end else begin
                a_new = a_old;
            end

            // Store new accumulator
            accum_chunk_new[b * ACC_WIDTH +: ACC_WIDTH] = a_new;

            // Update core bit (sign of accumulator)
            // Positive → 1, non-positive → 0
            core_chunk_new[b] = (a_new > 0) ? 1'b1 : 1'b0;
        end
    end

    // =========================================================================
    // State Machine - Sequential Logic
    // =========================================================================

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= S_IDLE;
            chunk_idx   <= '0;
            row_id_reg  <= '0;
            reward_reg  <= '0;
            busy        <= 1'b0;
            done        <= 1'b0;
        end else begin
            state     <= state_next;
            chunk_idx <= chunk_idx_next;
            done      <= 1'b0;  // Default: clear done

            case (state)
                S_IDLE: begin
                    if (start && reward != 0) begin
                        row_id_reg <= row_id;
                        reward_reg <= reward;
                        busy       <= 1'b1;
                    end
                end

                S_READ_WAIT: begin
                    if (mem_ready) begin
                        // Latch memory read data
                        core_chunk_reg  <= mem_core_in;
                        accum_chunk_reg <= mem_accum_in;
                    end
                end

                S_DONE: begin
                    busy <= 1'b0;
                    done <= 1'b1;
                end

                default: ;
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
        mem_req        = 1'b0;
        mem_we         = 1'b0;
        mem_chunk_addr = chunk_idx;

        case (state)
            S_IDLE: begin
                chunk_idx_next = '0;
                if (start && reward != 0) begin
                    state_next = S_READ_REQ;
                end
            end

            S_READ_REQ: begin
                // Issue read request
                mem_req = 1'b1;
                mem_we  = 1'b0;
                state_next = S_READ_WAIT;
            end

            S_READ_WAIT: begin
                if (mem_ready) begin
                    state_next = S_COMPUTE;
                end
            end

            S_COMPUTE: begin
                // Combinational update computed above
                // Move to write phase
                state_next = S_WRITE_REQ;
            end

            S_WRITE_REQ: begin
                // Issue write request with updated data
                mem_req = 1'b1;
                mem_we  = 1'b1;
                state_next = S_WRITE_WAIT;
            end

            S_WRITE_WAIT: begin
                if (mem_ready) begin
                    state_next = S_NEXT_CHUNK;
                end
            end

            S_NEXT_CHUNK: begin
                if (chunk_idx == NUM_CHUNKS - 1) begin
                    // Done with all chunks
                    state_next = S_DONE;
                end else begin
                    // Move to next chunk
                    chunk_idx_next = chunk_idx + 1;
                    state_next     = S_READ_REQ;
                end
            end

            S_DONE: begin
                state_next = S_IDLE;
            end

            default: state_next = S_IDLE;
        endcase
    end

    // =========================================================================
    // Output Assignments
    // =========================================================================

    assign mem_core_out  = core_chunk_new;
    assign mem_accum_out = accum_chunk_new;

endmodule


// =============================================================================
// Ara Plasticity Controller - Multi-Row Scheduler
// =============================================================================
//
// Schedules plasticity updates across multiple active rows.
// Time-multiplexes the single plasticity_core across rows.
//
// =============================================================================

module ara_plasticity_controller #(
    parameter int ROWS         = ARA_ROWS,
    parameter int DIM          = ARA_DIM,
    parameter int CHUNK_BITS   = ARA_CHUNK_BITS,
    parameter int ACC_WIDTH    = ARA_ACC_WIDTH,
    parameter int MAX_ACTIVE   = ARA_MAX_ACTIVE_ROWS
)(
    input  logic                    clk,
    input  logic                    rst_n,

    // === External Control Interface ===
    input  logic                    event_valid,      // New emotional event
    input  logic signed [7:0]       reward,           // Reward value
    input  logic [DIM-1:0]          input_hv,         // Pattern being learned
    input  logic [ROWS-1:0]         active_mask,      // Which rows participated

    output logic                    busy,
    output logic                    done,
    output logic [15:0]             rows_updated,

    // === Memory Interface (directly to adapter) ===
    output logic                           mem_req,
    input  logic                           mem_ready,
    output logic [$clog2(ROWS)-1:0]        mem_row_addr,
    output logic [$clog2(DIM/CHUNK_BITS)-1:0] mem_chunk_addr,
    output logic                           mem_we,
    output logic [CHUNK_BITS-1:0]          mem_core_out,
    output logic [CHUNK_BITS*ACC_WIDTH-1:0] mem_accum_out,
    input  logic [CHUNK_BITS-1:0]          mem_core_in,
    input  logic [CHUNK_BITS*ACC_WIDTH-1:0] mem_accum_in
);

    // =========================================================================
    // Plasticity Core Instance
    // =========================================================================

    logic                           core_start;
    logic [$clog2(ROWS)-1:0]        core_row_id;
    logic                           core_busy;
    logic                           core_done;
    logic                           core_mem_req;
    logic [$clog2(DIM/CHUNK_BITS)-1:0] core_chunk_addr;
    logic                           core_mem_we;
    logic [CHUNK_BITS-1:0]          core_mem_core_out;
    logic [CHUNK_BITS*ACC_WIDTH-1:0] core_mem_accum_out;

    ara_plasticity_core #(
        .ROWS(ROWS),
        .DIM(DIM),
        .CHUNK_BITS(CHUNK_BITS),
        .ACC_WIDTH(ACC_WIDTH)
    ) core_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(core_start),
        .row_id(core_row_id),
        .reward(reward_reg),
        .input_hv(input_hv_reg),
        .busy(core_busy),
        .done(core_done),
        .mem_req(core_mem_req),
        .mem_ready(mem_ready),
        .mem_chunk_addr(core_chunk_addr),
        .mem_we(core_mem_we),
        .mem_core_out(core_mem_core_out),
        .mem_accum_out(core_mem_accum_out),
        .mem_core_in(mem_core_in),
        .mem_accum_in(mem_accum_in)
    );

    // =========================================================================
    // Controller State Machine
    // =========================================================================

    typedef enum logic [2:0] {
        C_IDLE,
        C_SCAN,         // Find next active row
        C_START_ROW,    // Start core for this row
        C_WAIT_ROW,     // Wait for core to finish
        C_DONE
    } ctrl_state_t;

    ctrl_state_t ctrl_state;

    // Latched inputs
    logic signed [7:0]  reward_reg;
    logic [DIM-1:0]     input_hv_reg;
    logic [ROWS-1:0]    remaining_mask;
    logic [15:0]        update_count;

    // Priority encoder: find first set bit
    logic [$clog2(ROWS)-1:0] next_row;
    logic                    has_next;

    always_comb begin
        next_row = '0;
        has_next = 1'b0;
        for (int i = 0; i < ROWS; i++) begin
            if (remaining_mask[i] && !has_next) begin
                next_row = i[$clog2(ROWS)-1:0];
                has_next = 1'b1;
            end
        end
    end

    // =========================================================================
    // Controller Logic
    // =========================================================================

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ctrl_state     <= C_IDLE;
            reward_reg     <= '0;
            input_hv_reg   <= '0;
            remaining_mask <= '0;
            update_count   <= '0;
            core_row_id    <= '0;
            core_start     <= 1'b0;
            busy           <= 1'b0;
            done           <= 1'b0;
            rows_updated   <= '0;
        end else begin
            done       <= 1'b0;
            core_start <= 1'b0;

            case (ctrl_state)
                C_IDLE: begin
                    busy <= 1'b0;
                    if (event_valid && reward != 0) begin
                        reward_reg     <= reward;
                        input_hv_reg   <= input_hv;
                        remaining_mask <= active_mask;
                        update_count   <= '0;
                        busy           <= 1'b1;
                        ctrl_state     <= C_SCAN;
                    end
                end

                C_SCAN: begin
                    if (!has_next || update_count >= MAX_ACTIVE) begin
                        // Done: no more rows or hit limit
                        ctrl_state <= C_DONE;
                    end else begin
                        core_row_id <= next_row;
                        ctrl_state  <= C_START_ROW;
                    end
                end

                C_START_ROW: begin
                    core_start <= 1'b1;
                    ctrl_state <= C_WAIT_ROW;
                end

                C_WAIT_ROW: begin
                    if (core_done) begin
                        // Clear this row from mask
                        remaining_mask[core_row_id] <= 1'b0;
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
    // Memory Interface Routing
    // =========================================================================

    assign mem_req        = core_mem_req;
    assign mem_row_addr   = core_row_id;
    assign mem_chunk_addr = core_chunk_addr;
    assign mem_we         = core_mem_we;
    assign mem_core_out   = core_mem_core_out;
    assign mem_accum_out  = core_mem_accum_out;

endmodule
