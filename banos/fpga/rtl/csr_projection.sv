/**
 * CSR PROJECTION - Dense-to-Sparse Vector Converter
 * ==================================================
 *
 * Bio-Affective Neuromorphic Operating System
 * Converts dense LLM embeddings to sparse spike patterns
 *
 * The LLM speaks in Dense Vectors (Float32/BFloat16).
 * The SNN speaks in Sparse Spikes (1-bit events).
 *
 * This module implements a Compressed Sparse Row (CSR) matrix multiply:
 *   spikes_out = threshold(projection_matrix @ vector_in)
 *
 * The projection matrix is stored in CSR format:
 *   - row_ptr[i]: Start of row i in col_idx/values
 *   - col_idx[j]: Column index for non-zero j
 *   - values[j]: Weight value for non-zero j
 *
 * For VU7P: Uses URAM for projection matrix storage
 * Matrix size: 1024 output x 4096 input @ 10% sparsity = 400K non-zeros
 * Storage: 400K * 4 bytes = 1.6MB fits in URAM
 */

module csr_projection
    import kf_pkg::*;
#(
    parameter int INPUT_DIM  = 4096,   // LLM hidden dimension (Llama-2/3)
    parameter int OUTPUT_DIM = 1024,   // Target spike neurons
    parameter int NNZ_MAX    = 409600, // Max non-zeros (10% sparsity)
    parameter int DATA_WIDTH = 16      // BFloat16-ish fixed point
)(
    input  logic        clk,
    input  logic        rst_n,

    // =========================================================================
    // AXI Stream Input (Dense Vector from PCIe DMA)
    // =========================================================================
    input  logic [511:0]         s_axis_tdata,   // 32 x 16-bit values
    input  logic                 s_axis_tvalid,
    output logic                 s_axis_tready,
    input  logic                 s_axis_tlast,

    // =========================================================================
    // Sparse Output (Spike Vector)
    // =========================================================================
    output logic [OUTPUT_DIM-1:0] spikes_out,
    output logic                  done,

    // =========================================================================
    // Configuration Interface (Load projection matrix)
    // =========================================================================
    input  logic                  cfg_we,
    input  logic [1:0]            cfg_sel,    // 0=row_ptr, 1=col_idx, 2=values
    input  logic [19:0]           cfg_addr,
    input  logic [31:0]           cfg_wdata
);

    // =========================================================================
    // STATE MACHINE
    // =========================================================================

    typedef enum logic [2:0] {
        ST_IDLE,
        ST_LOAD_VECTOR,
        ST_COMPUTE,
        ST_THRESHOLD,
        ST_DONE
    } state_t;

    state_t state, state_next;

    // =========================================================================
    // MEMORY ARRAYS (URAM on VU7P+)
    // =========================================================================

    // Input vector buffer (ping-pong for streaming)
    (* ram_style = "ultra" *)
    logic signed [DATA_WIDTH-1:0] input_vec [INPUT_DIM];

    // CSR Matrix Storage
    (* ram_style = "ultra" *)
    logic [19:0] row_ptr [OUTPUT_DIM+1];  // Row pointers (20-bit for 1M max)

    (* ram_style = "ultra" *)
    logic [11:0] col_idx [NNZ_MAX];       // Column indices (12-bit for 4K cols)

    (* ram_style = "ultra" *)
    logic signed [DATA_WIDTH-1:0] values [NNZ_MAX]; // Non-zero weights

    // Output accumulator
    logic signed [31:0] accum [OUTPUT_DIM];

    // =========================================================================
    // LOADING LOGIC
    // =========================================================================

    // Load input vector from AXI stream (32 values per beat)
    logic [7:0] load_ptr;
    logic [31:0] values_loaded;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            load_ptr <= 0;
            values_loaded <= 0;
            s_axis_tready <= 1'b0;
        end else begin
            case (state)
                ST_IDLE: begin
                    load_ptr <= 0;
                    values_loaded <= 0;
                    s_axis_tready <= 1'b1;
                end

                ST_LOAD_VECTOR: begin
                    if (s_axis_tvalid && s_axis_tready) begin
                        // Unpack 32 x 16-bit values from 512-bit bus
                        for (int i = 0; i < 32; i++) begin
                            if (values_loaded + i < INPUT_DIM) begin
                                input_vec[values_loaded + i] <=
                                    s_axis_tdata[i*16 +: 16];
                            end
                        end
                        values_loaded <= values_loaded + 32;

                        if (s_axis_tlast || values_loaded + 32 >= INPUT_DIM) begin
                            s_axis_tready <= 1'b0;
                        end
                    end
                end

                default: s_axis_tready <= 1'b0;
            endcase
        end
    end

    // =========================================================================
    // CSR MATRIX CONFIGURATION
    // =========================================================================

    always_ff @(posedge clk) begin
        if (cfg_we) begin
            case (cfg_sel)
                2'b00: row_ptr[cfg_addr[$clog2(OUTPUT_DIM):0]] <= cfg_wdata[19:0];
                2'b01: col_idx[cfg_addr] <= cfg_wdata[11:0];
                2'b10: values[cfg_addr]  <= cfg_wdata[DATA_WIDTH-1:0];
                default: ; // Reserved
            endcase
        end
    end

    // =========================================================================
    // CSR MATRIX-VECTOR MULTIPLY (SpMV)
    // =========================================================================

    // Current row being computed
    logic [$clog2(OUTPUT_DIM)-1:0] row_idx;
    logic [19:0] col_ptr;
    logic [19:0] row_end;
    logic compute_done;

    // Pipelined accumulation
    logic signed [31:0] partial_sum;
    logic [11:0] current_col;
    logic signed [DATA_WIDTH-1:0] current_val;
    logic signed [DATA_WIDTH-1:0] input_sample;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            row_idx <= 0;
            col_ptr <= 0;
            row_end <= 0;
            partial_sum <= 0;
            compute_done <= 0;
            for (int i = 0; i < OUTPUT_DIM; i++) begin
                accum[i] <= 0;
            end
        end else if (state == ST_COMPUTE) begin
            if (row_idx < OUTPUT_DIM) begin
                if (col_ptr == 0) begin
                    // Start new row
                    col_ptr <= row_ptr[row_idx];
                    row_end <= row_ptr[row_idx + 1];
                    partial_sum <= 0;
                end else if (col_ptr < row_end) begin
                    // Accumulate: accum += values[col_ptr] * input_vec[col_idx[col_ptr]]
                    current_col <= col_idx[col_ptr];
                    current_val <= values[col_ptr];
                    input_sample <= input_vec[col_idx[col_ptr]];

                    // Fixed-point multiply-accumulate
                    partial_sum <= partial_sum +
                        ({{16{current_val[DATA_WIDTH-1]}}, current_val} *
                         {{16{input_sample[DATA_WIDTH-1]}}, input_sample}) >>> 8;

                    col_ptr <= col_ptr + 1;
                end else begin
                    // Row complete, save result
                    accum[row_idx] <= partial_sum;
                    row_idx <= row_idx + 1;
                    col_ptr <= 0;
                    partial_sum <= 0;
                end
            end else begin
                compute_done <= 1'b1;
            end
        end else begin
            row_idx <= 0;
            col_ptr <= 0;
            compute_done <= 0;
        end
    end

    // =========================================================================
    // THRESHOLDING (Accumulator -> Spikes)
    // =========================================================================

    // Adaptive threshold based on mean activation
    logic signed [31:0] threshold;
    logic [$clog2(OUTPUT_DIM)-1:0] thresh_idx;
    logic thresh_done;

    // Simple threshold: fire if accum > THRESH
    localparam logic signed [31:0] SPIKE_THRESH = 32'sh0000_1000;  // 0.0625

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            spikes_out <= 0;
            thresh_idx <= 0;
            thresh_done <= 0;
        end else if (state == ST_THRESHOLD) begin
            if (thresh_idx < OUTPUT_DIM) begin
                // Fire if above threshold
                spikes_out[thresh_idx] <= (accum[thresh_idx] > SPIKE_THRESH);
                thresh_idx <= thresh_idx + 1;
            end else begin
                thresh_done <= 1'b1;
            end
        end else begin
            thresh_idx <= 0;
            thresh_done <= 0;
        end
    end

    // =========================================================================
    // STATE MACHINE LOGIC
    // =========================================================================

    assign done = (state == ST_DONE);

    always_comb begin
        state_next = state;

        case (state)
            ST_IDLE: begin
                if (s_axis_tvalid) begin
                    state_next = ST_LOAD_VECTOR;
                end
            end

            ST_LOAD_VECTOR: begin
                if (values_loaded >= INPUT_DIM || (s_axis_tvalid && s_axis_tlast)) begin
                    state_next = ST_COMPUTE;
                end
            end

            ST_COMPUTE: begin
                if (compute_done) begin
                    state_next = ST_THRESHOLD;
                end
            end

            ST_THRESHOLD: begin
                if (thresh_done) begin
                    state_next = ST_DONE;
                end
            end

            ST_DONE: begin
                state_next = ST_IDLE;
            end

            default: state_next = ST_IDLE;
        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
        end else begin
            state <= state_next;
        end
    end

endmodule : csr_projection
