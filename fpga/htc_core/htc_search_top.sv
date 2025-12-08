// =============================================================================
// HTC Search Top - Complete Sub-Microsecond Resonance Search System
// =============================================================================
//
// This is the top-level module that instantiates:
//   - Weight memory banks
//   - XNOR-popcount CAM core
//   - Query buffer
//   - Result interface
//
// Performance Targets:
//   - Query latency: < 1 µs (early exit typical: ~0.3-0.7 µs)
//   - Throughput: 1 query / 2-3 µs sustained
//   - Power: < 10W for search subsystem
//
// Integration:
//   - AXI-Lite control interface for configuration
//   - AXI-Stream for query HV input
//   - AXI-Stream for result output
//   - Memory-mapped for weight updates
//
// =============================================================================

`timescale 1ns/1ps

module htc_search_top #(
    // HV dimensions
    parameter int D          = 16384,
    parameter int R          = 2048,
    parameter int C          = 512,

    // Parallelism (tune for FPGA)
    parameter int ROW_PAR    = 64,
    parameter int CHUNK_PAR  = 4,

    // Top-K results
    parameter int K          = 16,

    // Derived
    parameter int N_CHUNKS   = D / C,
    parameter int R_GROUPS   = R / ROW_PAR,
    parameter int C_GROUPS   = N_CHUNKS / CHUNK_PAR,
    parameter int SIM_BITS   = $clog2(D+1),
    parameter int ROW_BITS   = $clog2(R)
)(
    input  logic                        clk,
    input  logic                        rst_n,

    // =========================================================================
    // Query Input (AXI-Stream style)
    // =========================================================================

    input  logic                        s_query_valid,
    input  logic [D-1:0]                s_query_hv,
    output logic                        s_query_ready,

    // =========================================================================
    // Result Output (AXI-Stream style)
    // =========================================================================

    output logic                        m_result_valid,
    output logic [ROW_BITS-1:0]         m_result_idx   [0:K-1],
    output logic [SIM_BITS-1:0]         m_result_score [0:K-1],
    output logic                        m_result_early_exit,
    output logic [15:0]                 m_result_cycles,
    input  logic                        m_result_ready,

    // =========================================================================
    // Weight Programming Interface
    // =========================================================================

    input  logic                        weight_prog_en,
    input  logic [ROW_BITS-1:0]         weight_prog_row,
    input  logic [D-1:0]                weight_prog_hv,
    output logic                        weight_prog_ready,

    // =========================================================================
    // Status
    // =========================================================================

    output logic                        busy,
    output logic [31:0]                 query_count,
    output logic [31:0]                 early_exit_count
);

    // =========================================================================
    // Query Buffer - Store incoming HV and chunk it out
    // =========================================================================

    logic [D-1:0] query_buffer;
    logic query_buffered;
    logic query_processing;

    // Chunk extraction
    logic [C*CHUNK_PAR-1:0] query_chunks_flat;
    logic [$clog2(C_GROUPS)-1:0] query_chunk_idx;
    logic query_chunk_valid;

    // Buffer incoming query
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            query_buffer <= '0;
            query_buffered <= 1'b0;
        end else if (s_query_valid && s_query_ready) begin
            query_buffer <= s_query_hv;
            query_buffered <= 1'b1;
        end else if (cam_query_start) begin
            query_buffered <= 1'b0;
        end
    end

    assign s_query_ready = !query_buffered && !query_processing;

    // Extract chunks for CAM
    genvar ci;
    generate
        for (ci = 0; ci < CHUNK_PAR; ci++) begin : CHUNK_EXTRACT
            assign query_chunks_flat[ci*C +: C] =
                query_buffer[(query_chunk_idx * CHUNK_PAR + ci) * C +: C];
        end
    endgenerate

    // =========================================================================
    // CAM Core
    // =========================================================================

    logic cam_query_start;
    logic cam_query_ready;
    logic cam_query_chunk_ready;
    logic cam_result_valid;
    logic [ROW_BITS-1:0] cam_top_idx [0:K-1];
    logic [SIM_BITS-1:0] cam_top_score [0:K-1];
    logic cam_early_exit;
    logic [$clog2(D)+1:0] cam_cycles;

    // Weight memory interface
    logic weight_req_valid;
    logic [$clog2(R_GROUPS)-1:0] weight_row_group;
    logic [$clog2(C_GROUPS)-1:0] weight_chunk_group;
    logic weight_resp_valid;
    logic [C*CHUNK_PAR-1:0] weight_chunks [0:ROW_PAR-1];

    htc_assoc_cam #(
        .D(D),
        .R(R),
        .C(C),
        .ROW_PAR(ROW_PAR),
        .CHUNK_PAR(CHUNK_PAR),
        .K(K)
    ) u_cam (
        .clk(clk),
        .rst_n(rst_n),

        // Query interface
        .query_start(cam_query_start),
        .query_ready(cam_query_ready),
        .query_chunk_valid(query_chunk_valid),
        .query_chunks_flat(query_chunks_flat),
        .query_chunk_group(query_chunk_idx),
        .query_chunk_ready(cam_query_chunk_ready),

        // Weight interface
        .weight_req_valid(weight_req_valid),
        .weight_row_group(weight_row_group),
        .weight_chunk_group(weight_chunk_group),
        .weight_resp_valid(weight_resp_valid),
        .weight_chunks(weight_chunks),

        // Results
        .result_valid(cam_result_valid),
        .top_idx(cam_top_idx),
        .top_score(cam_top_score),
        .early_exit_used(cam_early_exit),
        .total_cycles(cam_cycles)
    );

    // =========================================================================
    // Weight Memory
    // =========================================================================

    htc_weight_bank #(
        .D(D),
        .R(R),
        .C(C),
        .ROW_PAR(ROW_PAR),
        .CHUNK_PAR(CHUNK_PAR)
    ) u_weight_bank (
        .clk(clk),
        .rst_n(rst_n),

        // Read interface (for CAM)
        .read_req(weight_req_valid),
        .read_row_group(weight_row_group),
        .read_chunk_group(weight_chunk_group),
        .read_resp_valid(weight_resp_valid),
        .read_data(weight_chunks),

        // Write interface (for plasticity)
        .write_en(1'b0),
        .write_row('0),
        .write_chunk('0),
        .write_data('0),

        // Bulk load
        .bulk_load_en(weight_prog_en),
        .bulk_row(weight_prog_row),
        .bulk_hv(weight_prog_hv)
    );

    assign weight_prog_ready = !query_processing;

    // =========================================================================
    // Query Controller
    // =========================================================================

    typedef enum logic [2:0] {
        QC_IDLE,
        QC_START,
        QC_STREAM_CHUNKS,
        QC_WAIT_RESULT,
        QC_OUTPUT
    } query_ctrl_state_t;

    query_ctrl_state_t qc_state, qc_state_next;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            qc_state <= QC_IDLE;
            query_chunk_idx <= '0;
        end else begin
            qc_state <= qc_state_next;

            case (qc_state)
                QC_START: begin
                    query_chunk_idx <= '0;
                end

                QC_STREAM_CHUNKS: begin
                    if (cam_query_chunk_ready) begin
                        if (query_chunk_idx < C_GROUPS - 1) begin
                            query_chunk_idx <= query_chunk_idx + 1;
                        end
                    end
                end

                default: ;
            endcase
        end
    end

    always_comb begin
        qc_state_next = qc_state;
        cam_query_start = 1'b0;
        query_chunk_valid = 1'b0;
        query_processing = 1'b0;

        case (qc_state)
            QC_IDLE: begin
                if (query_buffered) begin
                    qc_state_next = QC_START;
                end
            end

            QC_START: begin
                query_processing = 1'b1;
                cam_query_start = 1'b1;
                qc_state_next = QC_STREAM_CHUNKS;
            end

            QC_STREAM_CHUNKS: begin
                query_processing = 1'b1;
                query_chunk_valid = 1'b1;

                if (cam_query_chunk_ready && query_chunk_idx == C_GROUPS - 1) begin
                    qc_state_next = QC_WAIT_RESULT;
                end
            end

            QC_WAIT_RESULT: begin
                query_processing = 1'b1;
                if (cam_result_valid) begin
                    qc_state_next = QC_OUTPUT;
                end
            end

            QC_OUTPUT: begin
                if (m_result_ready) begin
                    qc_state_next = QC_IDLE;
                end
            end

            default: qc_state_next = QC_IDLE;
        endcase
    end

    // =========================================================================
    // Result Output
    // =========================================================================

    logic [ROW_BITS-1:0] result_idx_reg [0:K-1];
    logic [SIM_BITS-1:0] result_score_reg [0:K-1];
    logic result_early_exit_reg;
    logic [15:0] result_cycles_reg;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < K; i++) begin
                result_idx_reg[i] <= '0;
                result_score_reg[i] <= '0;
            end
            result_early_exit_reg <= 1'b0;
            result_cycles_reg <= '0;
        end else if (cam_result_valid) begin
            for (int i = 0; i < K; i++) begin
                result_idx_reg[i] <= cam_top_idx[i];
                result_score_reg[i] <= cam_top_score[i];
            end
            result_early_exit_reg <= cam_early_exit;
            result_cycles_reg <= cam_cycles[15:0];
        end
    end

    assign m_result_valid = (qc_state == QC_OUTPUT);
    assign m_result_idx = result_idx_reg;
    assign m_result_score = result_score_reg;
    assign m_result_early_exit = result_early_exit_reg;
    assign m_result_cycles = result_cycles_reg;

    // =========================================================================
    // Statistics
    // =========================================================================

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            query_count <= '0;
            early_exit_count <= '0;
        end else if (cam_result_valid) begin
            query_count <= query_count + 1;
            if (cam_early_exit) begin
                early_exit_count <= early_exit_count + 1;
            end
        end
    end

    assign busy = query_processing || query_buffered;

endmodule


// =============================================================================
// Testbench
// =============================================================================

`ifdef SIMULATION
module htc_search_top_tb;

    // Use smaller parameters for simulation
    localparam int D = 4096;
    localparam int R = 256;
    localparam int C = 512;
    localparam int ROW_PAR = 16;
    localparam int CHUNK_PAR = 2;
    localparam int K = 8;

    logic clk = 0;
    logic rst_n = 0;

    // Query interface
    logic s_query_valid;
    logic [D-1:0] s_query_hv;
    logic s_query_ready;

    // Result interface
    logic m_result_valid;
    logic [$clog2(R)-1:0] m_result_idx [0:K-1];
    logic [$clog2(D+1)-1:0] m_result_score [0:K-1];
    logic m_result_early_exit;
    logic [15:0] m_result_cycles;
    logic m_result_ready;

    // Weight programming
    logic weight_prog_en;
    logic [$clog2(R)-1:0] weight_prog_row;
    logic [D-1:0] weight_prog_hv;
    logic weight_prog_ready;

    // Status
    logic busy;
    logic [31:0] query_count;
    logic [31:0] early_exit_count;

    // Clock generation (350 MHz)
    always #1.43 clk = ~clk;

    htc_search_top #(
        .D(D),
        .R(R),
        .C(C),
        .ROW_PAR(ROW_PAR),
        .CHUNK_PAR(CHUNK_PAR),
        .K(K)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .s_query_valid(s_query_valid),
        .s_query_hv(s_query_hv),
        .s_query_ready(s_query_ready),
        .m_result_valid(m_result_valid),
        .m_result_idx(m_result_idx),
        .m_result_score(m_result_score),
        .m_result_early_exit(m_result_early_exit),
        .m_result_cycles(m_result_cycles),
        .m_result_ready(m_result_ready),
        .weight_prog_en(weight_prog_en),
        .weight_prog_row(weight_prog_row),
        .weight_prog_hv(weight_prog_hv),
        .weight_prog_ready(weight_prog_ready),
        .busy(busy),
        .query_count(query_count),
        .early_exit_count(early_exit_count)
    );

    // Test
    initial begin
        $dumpfile("htc_search_top.vcd");
        $dumpvars(0, htc_search_top_tb);

        // Initialize
        s_query_valid = 0;
        s_query_hv = '0;
        m_result_ready = 1;
        weight_prog_en = 0;
        weight_prog_row = 0;
        weight_prog_hv = '0;

        // Reset
        #20 rst_n = 1;
        #20;

        // Program some attractors
        for (int i = 0; i < 16; i++) begin
            @(posedge clk);
            weight_prog_en = 1;
            weight_prog_row = i;
            // Create a pattern
            weight_prog_hv = {D{1'b0}};
            weight_prog_hv[i*256 +: 256] = {256{1'b1}};
        end
        @(posedge clk);
        weight_prog_en = 0;
        #100;

        // Submit a query
        @(posedge clk);
        s_query_valid = 1;
        s_query_hv = '0;
        s_query_hv[0*256 +: 256] = {256{1'b1}};  // Should match attractor 0
        @(posedge clk);
        s_query_valid = 0;

        // Wait for result
        wait(m_result_valid);
        #10;

        $display("Query complete!");
        $display("  Cycles: %d", m_result_cycles);
        $display("  Early exit: %b", m_result_early_exit);
        $display("  Top result: idx=%d, score=%d", m_result_idx[0], m_result_score[0]);

        #100;
        $finish;
    end

endmodule
`endif
