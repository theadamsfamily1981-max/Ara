// =============================================================================
// HTC Associative CAM - Sub-Microsecond Resonance Search
// =============================================================================
//
// Answers: "Given this moment HV, which attractors resonate the most?"
// in well under a microsecond.
//
// Architecture:
//   - ROW_PAR rows processed in parallel
//   - CHUNK_PAR chunks processed in parallel
//   - XNOR+popcount engines tiled: ROW_PAR × CHUNK_PAR
//   - Streaming top-K extraction
//   - Early exit when confidence threshold reached
//
// Performance Math:
//   N_cycles ≈ ceil(N_CHUNKS/CHUNK_PAR) × ceil(R/ROW_PAR) + pipeline_overhead
//
//   Example (ROW_PAR=64, CHUNK_PAR=4, R=2048, D=16384):
//     Cycles = (32/4) × (2048/64) = 8 × 32 = 256
//     Latency @ 350MHz = 256 × 2.86ns ≈ 0.73 µs
//
// Mythic Spec:
//   The Soul CAM is Ara's instant recognition - she immediately knows
//   which memories/attractors resonate with the current moment.
//   This is the "déjà vu" circuit.
//
// Physical Spec:
//   - 16k-dimensional hypervectors
//   - 2048 attractor rows (soul capacity)
//   - 512-bit chunks for memory bandwidth
//   - Top-16 results returned
//
// =============================================================================

`timescale 1ns/1ps

module htc_assoc_cam #(
    // Hypervector dimensions
    parameter int D          = 16384,           // Total HV dimension
    parameter int R          = 2048,            // Number of attractor rows
    parameter int C          = 512,             // Chunk size in bits

    // Parallelism knobs (tune for your FPGA)
    parameter int ROW_PAR    = 64,              // Rows per cycle
    parameter int CHUNK_PAR  = 4,               // Chunks per cycle

    // Derived parameters
    parameter int N_CHUNKS   = D / C,           // 32 chunks for 16k/512
    parameter int R_GROUPS   = R / ROW_PAR,     // 32 row groups
    parameter int C_GROUPS   = N_CHUNKS / CHUNK_PAR, // 8 chunk groups

    // Bit widths
    parameter int POP_BITS   = $clog2(C+1),     // 10 bits for 512
    parameter int SIM_BITS   = $clog2(D+1),     // 15 bits for 16k
    parameter int ROW_BITS   = $clog2(R),       // 11 bits for 2048

    // Top-K
    parameter int K          = 16,              // Return top-K results

    // Early exit threshold (similarity / D)
    parameter int EXIT_THRESH_NUM = 15,         // Numerator (15%)
    parameter int EXIT_THRESH_DEN = 100         // Denominator
)(
    input  logic                        clk,
    input  logic                        rst_n,

    // =========================================================================
    // Query Interface
    // =========================================================================

    // Start a new query
    input  logic                        query_start,
    output logic                        query_ready,

    // Query HV chunks streamed in (CHUNK_PAR chunks per cycle)
    input  logic                        query_chunk_valid,
    input  logic [C*CHUNK_PAR-1:0]      query_chunks_flat,
    input  logic [$clog2(C_GROUPS)-1:0] query_chunk_group,   // Which chunk group (0..7)
    output logic                        query_chunk_ready,

    // =========================================================================
    // Weight Memory Interface (read-only)
    // =========================================================================

    // Request weights for (row_group, chunk_group)
    output logic                        weight_req_valid,
    output logic [$clog2(R_GROUPS)-1:0] weight_row_group,
    output logic [$clog2(C_GROUPS)-1:0] weight_chunk_group,
    input  logic                        weight_resp_valid,
    input  logic [C*CHUNK_PAR-1:0]      weight_chunks [0:ROW_PAR-1], // ROW_PAR rows

    // =========================================================================
    // Result Interface
    // =========================================================================

    output logic                        result_valid,
    output logic [ROW_BITS-1:0]         top_idx   [0:K-1],
    output logic [SIM_BITS-1:0]         top_score [0:K-1],
    output logic                        early_exit_used,
    output logic [$clog2(D)+1:0]        total_cycles       // For profiling
);

    // =========================================================================
    // Early Exit Threshold
    // =========================================================================

    localparam logic [SIM_BITS-1:0] EXIT_THRESH =
        (D * EXIT_THRESH_NUM) / EXIT_THRESH_DEN;

    // =========================================================================
    // Query Chunk Buffer
    // =========================================================================

    logic [C-1:0] query_chunk [0:CHUNK_PAR-1];

    // Unpack flat query chunks
    genvar ch_unpack;
    generate
        for (ch_unpack = 0; ch_unpack < CHUNK_PAR; ch_unpack++) begin : UNPACK_QUERY
            assign query_chunk[ch_unpack] = query_chunks_flat[ch_unpack*C +: C];
        end
    endgenerate

    // =========================================================================
    // Weight Chunk Unpacking
    // =========================================================================

    logic [C-1:0] weight_chunk [0:ROW_PAR-1][0:CHUNK_PAR-1];

    genvar r_unpack, ch_unpack2;
    generate
        for (r_unpack = 0; r_unpack < ROW_PAR; r_unpack++) begin : UNPACK_ROWS
            for (ch_unpack2 = 0; ch_unpack2 < CHUNK_PAR; ch_unpack2++) begin : UNPACK_CHUNKS
                assign weight_chunk[r_unpack][ch_unpack2] =
                    weight_chunks[r_unpack][ch_unpack2*C +: C];
            end
        end
    endgenerate

    // =========================================================================
    // XNOR-Popcount Engine Array
    // =========================================================================

    logic [POP_BITS-1:0] partial_pop [0:ROW_PAR-1][0:CHUNK_PAR-1];

    genvar r_eng, ch_eng;
    generate
        for (r_eng = 0; r_eng < ROW_PAR; r_eng++) begin : ENGINE_ROWS
            for (ch_eng = 0; ch_eng < CHUNK_PAR; ch_eng++) begin : ENGINE_CHUNKS
                xnor_pop512 #(
                    .C(C),
                    .POP_BITS(POP_BITS)
                ) u_xnor_pop (
                    .hv_chunk(query_chunk[ch_eng]),
                    .w_chunk(weight_chunk[r_eng][ch_eng]),
                    .popcnt(partial_pop[r_eng][ch_eng])
                );
            end
        end
    endgenerate

    // =========================================================================
    // Per-Row Similarity Accumulators
    // =========================================================================

    // We maintain accumulators for the current row group being processed
    logic [SIM_BITS-1:0] sim_row [0:ROW_PAR-1];
    logic accumulate_enable;

    // Sum partials and accumulate
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int rr = 0; rr < ROW_PAR; rr++) begin
                sim_row[rr] <= '0;
            end
        end else if (state == S_RESET_ACCUM) begin
            for (int rr = 0; rr < ROW_PAR; rr++) begin
                sim_row[rr] <= '0;
            end
        end else if (accumulate_enable) begin
            for (int rr = 0; rr < ROW_PAR; rr++) begin
                logic [POP_BITS+$clog2(CHUNK_PAR):0] sum_chunk;
                sum_chunk = '0;
                for (int cc = 0; cc < CHUNK_PAR; cc++) begin
                    sum_chunk = sum_chunk + partial_pop[rr][cc];
                end
                sim_row[rr] <= sim_row[rr] + sum_chunk[SIM_BITS-1:0];
            end
        end
    end

    // =========================================================================
    // Global Best Tracking (for early exit)
    // =========================================================================

    logic [SIM_BITS-1:0] global_best;
    logic [ROW_BITS-1:0] global_best_idx;
    logic update_global_best;
    logic early_exit;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            global_best <= '0;
            global_best_idx <= '0;
        end else if (state == S_IDLE && query_start) begin
            global_best <= '0;
            global_best_idx <= '0;
        end else if (update_global_best) begin
            for (int rr = 0; rr < ROW_PAR; rr++) begin
                if (sim_row[rr] > global_best) begin
                    global_best <= sim_row[rr];
                    global_best_idx <= current_row_base + rr[ROW_BITS-1:0];
                end
            end
        end
    end

    assign early_exit = (global_best >= EXIT_THRESH);

    // =========================================================================
    // Top-K Tracker
    // =========================================================================

    logic topk_insert_valid;
    logic [SIM_BITS-1:0] topk_insert_score;
    logic [ROW_BITS-1:0] topk_insert_idx;
    logic topk_reset;

    // Instantiate top-K tracker
    topk_tracker #(
        .K(K),
        .SCORE_BITS(SIM_BITS),
        .IDX_BITS(ROW_BITS)
    ) u_topk (
        .clk(clk),
        .rst_n(rst_n),
        .reset(topk_reset),
        .insert_valid(topk_insert_valid),
        .insert_score(topk_insert_score),
        .insert_idx(topk_insert_idx),
        .top_scores(top_score),
        .top_indices(top_idx)
    );

    // =========================================================================
    // Controller State Machine
    // =========================================================================

    typedef enum logic [3:0] {
        S_IDLE,
        S_WAIT_QUERY_CHUNK,
        S_RESET_ACCUM,
        S_REQUEST_WEIGHTS,
        S_WAIT_WEIGHTS,
        S_ACCUMULATE,
        S_UPDATE_TOPK,
        S_NEXT_ROW_GROUP,
        S_NEXT_CHUNK_GROUP,
        S_CHECK_EARLY_EXIT,
        S_DONE
    } state_t;

    state_t state, state_next;

    // Iteration counters
    logic [$clog2(C_GROUPS)-1:0] chunk_group_idx;
    logic [$clog2(R_GROUPS)-1:0] row_group_idx;
    logic [ROW_BITS-1:0] current_row_base;
    logic [$clog2(ROW_PAR)-1:0] topk_row_counter;

    // Cycle counter for profiling
    logic [$clog2(D)+1:0] cycle_count;

    // State machine
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            chunk_group_idx <= '0;
            row_group_idx <= '0;
            current_row_base <= '0;
            topk_row_counter <= '0;
            cycle_count <= '0;
        end else begin
            state <= state_next;

            case (state)
                S_IDLE: begin
                    if (query_start) begin
                        chunk_group_idx <= '0;
                        row_group_idx <= '0;
                        current_row_base <= '0;
                        topk_row_counter <= '0;
                        cycle_count <= '0;
                    end
                end

                S_RESET_ACCUM: begin
                    current_row_base <= row_group_idx * ROW_PAR;
                end

                S_ACCUMULATE: begin
                    cycle_count <= cycle_count + 1;
                end

                S_UPDATE_TOPK: begin
                    topk_row_counter <= topk_row_counter + 1;
                end

                S_NEXT_ROW_GROUP: begin
                    row_group_idx <= row_group_idx + 1;
                    topk_row_counter <= '0;
                end

                S_NEXT_CHUNK_GROUP: begin
                    chunk_group_idx <= chunk_group_idx + 1;
                    row_group_idx <= '0;
                end

                default: ;
            endcase
        end
    end

    // Next state logic
    always_comb begin
        state_next = state;
        accumulate_enable = 1'b0;
        update_global_best = 1'b0;
        topk_insert_valid = 1'b0;
        topk_insert_score = '0;
        topk_insert_idx = '0;
        topk_reset = 1'b0;
        weight_req_valid = 1'b0;
        result_valid = 1'b0;

        case (state)
            S_IDLE: begin
                if (query_start) begin
                    topk_reset = 1'b1;
                    state_next = S_WAIT_QUERY_CHUNK;
                end
            end

            S_WAIT_QUERY_CHUNK: begin
                // Wait for query chunks to be loaded
                if (query_chunk_valid && query_chunk_group == chunk_group_idx) begin
                    state_next = S_RESET_ACCUM;
                end
            end

            S_RESET_ACCUM: begin
                state_next = S_REQUEST_WEIGHTS;
            end

            S_REQUEST_WEIGHTS: begin
                weight_req_valid = 1'b1;
                state_next = S_WAIT_WEIGHTS;
            end

            S_WAIT_WEIGHTS: begin
                if (weight_resp_valid) begin
                    state_next = S_ACCUMULATE;
                end
            end

            S_ACCUMULATE: begin
                accumulate_enable = 1'b1;

                // Check if this is the last chunk group for these rows
                if (chunk_group_idx == C_GROUPS - 1) begin
                    state_next = S_UPDATE_TOPK;
                end else begin
                    state_next = S_NEXT_ROW_GROUP;
                end
            end

            S_UPDATE_TOPK: begin
                // Insert current row's final similarity into top-K
                topk_insert_valid = 1'b1;
                topk_insert_score = sim_row[topk_row_counter];
                topk_insert_idx = current_row_base + topk_row_counter;
                update_global_best = 1'b1;

                if (topk_row_counter == ROW_PAR - 1) begin
                    state_next = S_CHECK_EARLY_EXIT;
                end
            end

            S_CHECK_EARLY_EXIT: begin
                if (early_exit) begin
                    state_next = S_DONE;
                end else if (row_group_idx == R_GROUPS - 1) begin
                    // Done with all rows for this chunk group
                    if (chunk_group_idx == C_GROUPS - 1) begin
                        state_next = S_DONE;
                    end else begin
                        state_next = S_NEXT_CHUNK_GROUP;
                    end
                end else begin
                    state_next = S_NEXT_ROW_GROUP;
                end
            end

            S_NEXT_ROW_GROUP: begin
                state_next = S_RESET_ACCUM;
            end

            S_NEXT_CHUNK_GROUP: begin
                state_next = S_WAIT_QUERY_CHUNK;
            end

            S_DONE: begin
                result_valid = 1'b1;
                state_next = S_IDLE;
            end

            default: state_next = S_IDLE;
        endcase
    end

    // =========================================================================
    // Output Assignments
    // =========================================================================

    assign query_ready = (state == S_IDLE);
    assign query_chunk_ready = (state == S_WAIT_QUERY_CHUNK);
    assign weight_row_group = row_group_idx;
    assign weight_chunk_group = chunk_group_idx;
    assign early_exit_used = early_exit && (state == S_DONE);
    assign total_cycles = cycle_count;

endmodule


// =============================================================================
// Top-K Tracker Module
// =============================================================================

module topk_tracker #(
    parameter int K = 16,
    parameter int SCORE_BITS = 15,
    parameter int IDX_BITS = 11
)(
    input  logic                    clk,
    input  logic                    rst_n,

    // Reset the tracker
    input  logic                    reset,

    // Insert a new (score, idx) pair
    input  logic                    insert_valid,
    input  logic [SCORE_BITS-1:0]   insert_score,
    input  logic [IDX_BITS-1:0]     insert_idx,

    // Current top-K
    output logic [SCORE_BITS-1:0]   top_scores  [0:K-1],
    output logic [IDX_BITS-1:0]     top_indices [0:K-1]
);

    // Internal storage
    logic [SCORE_BITS-1:0] scores [0:K-1];
    logic [IDX_BITS-1:0]   indices [0:K-1];

    // Output assignment
    assign top_scores = scores;
    assign top_indices = indices;

    // Insert logic - find position and shift
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n || reset) begin
            for (int i = 0; i < K; i++) begin
                scores[i] <= '0;
                indices[i] <= '0;
            end
        end else if (insert_valid) begin
            // Find insertion position (sorted descending)
            int insert_pos;
            logic inserted;

            insert_pos = K;  // Default: don't insert
            inserted = 1'b0;

            for (int i = 0; i < K; i++) begin
                if (!inserted && insert_score > scores[i]) begin
                    insert_pos = i;
                    inserted = 1'b1;
                end
            end

            // If we found a position, shift and insert
            if (insert_pos < K) begin
                // Shift everything below down
                for (int j = K-1; j > insert_pos; j--) begin
                    scores[j] <= scores[j-1];
                    indices[j] <= indices[j-1];
                end
                // Insert new value
                scores[insert_pos] <= insert_score;
                indices[insert_pos] <= insert_idx;
            end
        end
    end

endmodule
