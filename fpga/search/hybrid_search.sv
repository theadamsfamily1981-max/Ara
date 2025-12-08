// =============================================================================
// Hybrid Search - Unified Front-End for Soul + LAN Search
// =============================================================================
//
// "One mouth, two tongues" - single interface for three search backends:
//
// 1. XNOR-Popcount CAM (MODE_SOUL)
//    - 16k-dim HDC similarity search
//    - 2048 attractors, sub-µs latency
//    - For: moment resonance, attractor matching
//
// 2. Hash CAM (MODE_FLOW)
//    - 64-bit signature lookup
//    - ~ns latency, high fanout
//    - For: flow candidate generation, episode lookup
//
// 3. LUT-TCAM (MODE_REFLEX)
//    - Exact/masked pattern match
//    - Single-cycle latency
//    - For: reflex rules, pain packets, whitelists
//
// The sovereign loop and NodeAgents see a unified interface;
// mode_select chooses which backend answers.
//
// =============================================================================

`timescale 1ns/1ps

module hybrid_search #(
    // Soul CAM parameters
    parameter int D          = 16384,
    parameter int R          = 2048,
    parameter int C          = 512,
    parameter int ROW_PAR    = 64,
    parameter int CHUNK_PAR  = 4,
    parameter int TOPK       = 16,

    // Hash CAM parameters
    parameter int SIG_BITS   = 64,
    parameter int HASH_TABLES = 4,
    parameter int HASH_SIZE  = 4096,
    parameter int HASH_ID_BITS = 16,

    // TCAM parameters
    parameter int TCAM_ENTRIES = 256,
    parameter int TCAM_KEY_WIDTH = 56,
    parameter int TCAM_ACTION_BITS = 8,

    // Derived
    parameter int N_CHUNKS   = D / C,
    parameter int SIM_BITS   = $clog2(D+1),
    parameter int ROW_BITS   = $clog2(R)
)(
    input  logic                        clk,
    input  logic                        rst_n,

    // =========================================================================
    // Mode Selection
    // =========================================================================

    input  logic [1:0]                  mode_select,   // 0=SOUL, 1=FLOW, 2=REFLEX

    // =========================================================================
    // Query Interface (mode-dependent inputs)
    // =========================================================================

    input  logic                        query_valid,
    output logic                        query_ready,

    // Soul mode: HV chunks
    input  logic [C*CHUNK_PAR-1:0]      soul_chunks,
    input  logic [$clog2(N_CHUNKS/CHUNK_PAR)-1:0] soul_chunk_idx,

    // Flow mode: signature
    input  logic [SIG_BITS-1:0]         flow_sig,

    // Reflex mode: key
    input  logic [TCAM_KEY_WIDTH-1:0]   reflex_key,

    // =========================================================================
    // Result Interface (unified)
    // =========================================================================

    output logic                        result_valid,
    output logic [1:0]                  result_mode,   // Which mode responded

    // Soul results
    output logic [ROW_BITS-1:0]         soul_top_idx   [0:TOPK-1],
    output logic [SIM_BITS-1:0]         soul_top_score [0:TOPK-1],
    output logic                        soul_early_exit,

    // Flow results
    output logic [HASH_ID_BITS-1:0]     flow_cand_id [0:HASH_TABLES-1],
    output logic [HASH_TABLES-1:0]      flow_cand_valid,

    // Reflex results
    output logic                        reflex_matched,
    output logic [TCAM_ACTION_BITS-1:0] reflex_action,
    output logic [7:0]                  reflex_rule_id,

    // =========================================================================
    // Programming Interfaces
    // =========================================================================

    // Soul: weight programming (via separate interface)
    input  logic                        soul_weight_en,
    input  logic [ROW_BITS-1:0]         soul_weight_row,
    input  logic [D-1:0]                soul_weight_hv,

    // Flow: entry programming
    input  logic                        flow_prog_en,
    input  logic [SIG_BITS-1:0]         flow_prog_sig,
    input  logic [HASH_ID_BITS-1:0]     flow_prog_id,

    // Reflex: rule programming
    input  logic                        reflex_prog_en,
    input  logic [7:0]                  reflex_prog_idx,
    input  logic [TCAM_KEY_WIDTH-1:0]   reflex_prog_key,
    input  logic [TCAM_KEY_WIDTH-1:0]   reflex_prog_mask,
    input  logic [TCAM_ACTION_BITS-1:0] reflex_prog_action,

    // =========================================================================
    // Status / Debug
    // =========================================================================

    output logic [31:0]                 query_count,
    output logic [31:0]                 soul_queries,
    output logic [31:0]                 flow_queries,
    output logic [31:0]                 reflex_queries
);

    // =========================================================================
    // Mode Constants
    // =========================================================================

    localparam logic [1:0] MODE_SOUL   = 2'b00;
    localparam logic [1:0] MODE_FLOW   = 2'b01;
    localparam logic [1:0] MODE_REFLEX = 2'b10;

    // =========================================================================
    // Soul CAM (XNOR-Popcount)
    // =========================================================================

    logic soul_query_valid;
    logic soul_query_ready;
    logic soul_result_valid;
    logic [ROW_BITS-1:0] soul_idx_int [0:TOPK-1];
    logic [SIM_BITS-1:0] soul_score_int [0:TOPK-1];
    logic soul_early_exit_int;

    // Weight memory interface
    logic soul_weight_req_valid;
    logic [$clog2(R/ROW_PAR)-1:0] soul_weight_row_group;
    logic [$clog2(N_CHUNKS/CHUNK_PAR)-1:0] soul_weight_chunk_group;
    logic soul_weight_resp_valid;
    logic [C*CHUNK_PAR-1:0] soul_weight_chunks [0:ROW_PAR-1];

    htc_assoc_cam #(
        .D(D),
        .R(R),
        .C(C),
        .ROW_PAR(ROW_PAR),
        .CHUNK_PAR(CHUNK_PAR),
        .K(TOPK)
    ) u_soul_cam (
        .clk(clk),
        .rst_n(rst_n),

        .query_start(soul_query_valid && soul_query_ready),
        .query_ready(soul_query_ready),
        .query_chunk_valid(soul_query_valid),
        .query_chunks_flat(soul_chunks),
        .query_chunk_group(soul_chunk_idx),
        .query_chunk_ready(),

        .weight_req_valid(soul_weight_req_valid),
        .weight_row_group(soul_weight_row_group),
        .weight_chunk_group(soul_weight_chunk_group),
        .weight_resp_valid(soul_weight_resp_valid),
        .weight_chunks(soul_weight_chunks),

        .result_valid(soul_result_valid),
        .top_idx(soul_idx_int),
        .top_score(soul_score_int),
        .early_exit_used(soul_early_exit_int),
        .total_cycles()
    );

    // Soul weight bank
    htc_weight_bank #(
        .D(D),
        .R(R),
        .C(C),
        .ROW_PAR(ROW_PAR),
        .CHUNK_PAR(CHUNK_PAR)
    ) u_soul_weights (
        .clk(clk),
        .rst_n(rst_n),

        .read_req(soul_weight_req_valid),
        .read_row_group(soul_weight_row_group),
        .read_chunk_group(soul_weight_chunk_group),
        .read_resp_valid(soul_weight_resp_valid),
        .read_data(soul_weight_chunks),

        .write_en(1'b0),
        .write_row('0),
        .write_chunk('0),
        .write_data('0),

        .bulk_load_en(soul_weight_en),
        .bulk_row(soul_weight_row),
        .bulk_hv(soul_weight_hv)
    );

    // =========================================================================
    // Hash CAM (Flow Lookup)
    // =========================================================================

    logic flow_query_valid;
    logic flow_query_ready;
    logic flow_result_valid;
    logic [HASH_ID_BITS-1:0] flow_id_int [0:HASH_TABLES-1];
    logic [HASH_TABLES-1:0] flow_valid_int;

    hash_cam #(
        .SIG_BITS(SIG_BITS),
        .TABLE_SIZE(HASH_SIZE),
        .K_HASH(HASH_TABLES),
        .ID_BITS(HASH_ID_BITS)
    ) u_hash_cam (
        .clk(clk),
        .rst_n(rst_n),

        .q_valid(flow_query_valid),
        .q_sig(flow_sig),
        .q_ready(flow_query_ready),

        .result_valid(flow_result_valid),
        .cand_id(flow_id_int),
        .cand_valid(flow_valid_int),

        .prog_en(flow_prog_en),
        .prog_sig(flow_prog_sig),
        .prog_id(flow_prog_id),

        .total_entries(),
        .collision_count()
    );

    // =========================================================================
    // LUT-TCAM (Reflex Rules)
    // =========================================================================

    logic reflex_query_valid;
    logic reflex_result_valid;
    logic reflex_matched_int;
    logic [TCAM_ACTION_BITS-1:0] reflex_action_int;
    logic [$clog2(TCAM_ENTRIES)-1:0] reflex_idx_int;

    lutram_tcam #(
        .N(TCAM_ENTRIES),
        .W(TCAM_KEY_WIDTH),
        .ACTION_BITS(TCAM_ACTION_BITS)
    ) u_reflex_tcam (
        .clk(clk),
        .rst_n(rst_n),

        .search_valid(reflex_query_valid),
        .search_key(reflex_key),
        .match_valid(reflex_result_valid),
        .match_found(reflex_matched_int),
        .match_index(reflex_idx_int),
        .match_action(reflex_action_int),

        .prog_en(reflex_prog_en),
        .prog_index(reflex_prog_idx[$clog2(TCAM_ENTRIES)-1:0]),
        .prog_key(reflex_prog_key),
        .prog_mask(reflex_prog_mask),
        .prog_action(reflex_prog_action),

        .entries_used(),
        .table_full()
    );

    // =========================================================================
    // Mode Routing
    // =========================================================================

    // Query routing
    assign soul_query_valid   = query_valid && (mode_select == MODE_SOUL);
    assign flow_query_valid   = query_valid && (mode_select == MODE_FLOW);
    assign reflex_query_valid = query_valid && (mode_select == MODE_REFLEX);

    // Ready muxing
    always_comb begin
        case (mode_select)
            MODE_SOUL:   query_ready = soul_query_ready;
            MODE_FLOW:   query_ready = flow_query_ready;
            MODE_REFLEX: query_ready = 1'b1;  // TCAM always ready
            default:     query_ready = 1'b0;
        endcase
    end

    // Result muxing
    always_comb begin
        result_valid = 1'b0;
        result_mode = mode_select;

        case (mode_select)
            MODE_SOUL:   result_valid = soul_result_valid;
            MODE_FLOW:   result_valid = flow_result_valid;
            MODE_REFLEX: result_valid = reflex_result_valid;
            default:     result_valid = 1'b0;
        endcase
    end

    // =========================================================================
    // Output Assignment
    // =========================================================================

    // Soul outputs
    assign soul_top_idx = soul_idx_int;
    assign soul_top_score = soul_score_int;
    assign soul_early_exit = soul_early_exit_int;

    // Flow outputs
    assign flow_cand_id = flow_id_int;
    assign flow_cand_valid = flow_valid_int;

    // Reflex outputs
    assign reflex_matched = reflex_matched_int;
    assign reflex_action = reflex_action_int;
    assign reflex_rule_id = reflex_idx_int[7:0];

    // =========================================================================
    // Statistics
    // =========================================================================

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            query_count <= '0;
            soul_queries <= '0;
            flow_queries <= '0;
            reflex_queries <= '0;
        end else begin
            if (result_valid) begin
                query_count <= query_count + 1;
                case (result_mode)
                    MODE_SOUL:   soul_queries <= soul_queries + 1;
                    MODE_FLOW:   flow_queries <= flow_queries + 1;
                    MODE_REFLEX: reflex_queries <= reflex_queries + 1;
                    default: ;
                endcase
            end
        end
    end

endmodule


// =============================================================================
// Simplified Wrapper for AXI-Lite Control
// =============================================================================

module hybrid_search_axi #(
    // Use default parameters from hybrid_search
    parameter int D = 16384,
    parameter int R = 2048
)(
    input  logic        clk,
    input  logic        rst_n,

    // AXI-Lite Slave (control/status)
    input  logic [31:0] s_axil_awaddr,
    input  logic        s_axil_awvalid,
    output logic        s_axil_awready,
    input  logic [31:0] s_axil_wdata,
    input  logic        s_axil_wvalid,
    output logic        s_axil_wready,
    output logic [1:0]  s_axil_bresp,
    output logic        s_axil_bvalid,
    input  logic        s_axil_bready,
    input  logic [31:0] s_axil_araddr,
    input  logic        s_axil_arvalid,
    output logic        s_axil_arready,
    output logic [31:0] s_axil_rdata,
    output logic [1:0]  s_axil_rresp,
    output logic        s_axil_rvalid,
    input  logic        s_axil_rready,

    // AXI-Stream Query (simplified)
    input  logic [511:0] s_axis_query_tdata,
    input  logic         s_axis_query_tvalid,
    output logic         s_axis_query_tready,
    input  logic [7:0]   s_axis_query_tuser,   // [1:0]=mode, [7:2]=chunk_idx

    // AXI-Stream Result
    output logic [511:0] m_axis_result_tdata,
    output logic         m_axis_result_tvalid,
    input  logic         m_axis_result_tready,
    output logic [7:0]   m_axis_result_tuser
);

    // Register map (simplified)
    // 0x00: Control (start, mode)
    // 0x04: Status (busy, result_valid)
    // 0x08: Query count
    // 0x0C-0x1F: Reserved

    // This is a skeleton - full implementation would include
    // proper AXI-Lite state machine and data marshalling

    assign s_axil_awready = 1'b1;
    assign s_axil_wready = 1'b1;
    assign s_axil_bresp = 2'b00;
    assign s_axil_bvalid = 1'b0;
    assign s_axil_arready = 1'b1;
    assign s_axil_rdata = 32'h0;
    assign s_axil_rresp = 2'b00;
    assign s_axil_rvalid = 1'b0;

    assign s_axis_query_tready = 1'b1;
    assign m_axis_result_tdata = '0;
    assign m_axis_result_tvalid = 1'b0;
    assign m_axis_result_tuser = '0;

endmodule
