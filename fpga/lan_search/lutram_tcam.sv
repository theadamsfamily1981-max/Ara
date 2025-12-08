// =============================================================================
// LUTRAM TCAM - Distributed LUT Content-Addressable Memory for Reflex Rules
// =============================================================================
//
// Small, ultra-fast TCAM implemented in distributed LUTRAM for:
// - SmartNIC reflex rules (5-tuple hashes, flow fingerprints)
// - On-FPGA micro-policies (attractor tags, lane routing)
// - Debug watchpoints (HV fingerprints, node IDs)
//
// NOT for HDC similarity search - use XNOR-popcount CAM for that.
//
// Architecture:
//   - LUTRAM storage for keys and masks (MLAB on Intel)
//   - Parallel match across all rows
//   - Carry-chain AND reduction per row
//   - Priority encoder for first match
//   - Single-cycle lookup at high Fmax
//
// Sizing (LUT cost):
//   LUTs_storage ≈ (N × W × 2) / 64
//   Example: 256 × 32 TCAM = 256 LUTs storage
//
// Sweet spot: N ≤ 512, W ≤ 64
//
// =============================================================================

`timescale 1ns/1ps

module lutram_tcam #(
    parameter int N = 256,                      // Number of entries
    parameter int W = 32,                       // Key width in bits
    parameter int ACTION_BITS = 8               // Action/tag bits per entry
)(
    input  logic                    clk,
    input  logic                    rst_n,

    // =========================================================================
    // Search Interface (single-cycle lookup)
    // =========================================================================

    input  logic                    search_valid,
    input  logic [W-1:0]            search_key,
    output logic                    match_valid,
    output logic                    match_found,
    output logic [$clog2(N)-1:0]    match_index,
    output logic [ACTION_BITS-1:0]  match_action,

    // =========================================================================
    // Programming Interface
    // =========================================================================

    input  logic                    prog_en,
    input  logic [$clog2(N)-1:0]    prog_index,
    input  logic [W-1:0]            prog_key,
    input  logic [W-1:0]            prog_mask,     // 1 = care, 0 = don't care
    input  logic [ACTION_BITS-1:0]  prog_action,

    // =========================================================================
    // Status
    // =========================================================================

    output logic [$clog2(N):0]      entries_used,
    output logic                    table_full
);

    // =========================================================================
    // Storage: LUTRAM for keys, masks, actions
    // =========================================================================

    (* ramstyle = "MLAB" *) logic [W-1:0]            tcam_key   [0:N-1];
    (* ramstyle = "MLAB" *) logic [W-1:0]            tcam_mask  [0:N-1];
    (* ramstyle = "MLAB" *) logic [ACTION_BITS-1:0]  tcam_action[0:N-1];
    (* ramstyle = "MLAB" *) logic                    tcam_valid [0:N-1];

    // Entry counter
    logic [$clog2(N):0] entry_count;

    // =========================================================================
    // Programming Logic
    // =========================================================================

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            entry_count <= '0;
            for (int i = 0; i < N; i++) begin
                tcam_key[i]    <= '0;
                tcam_mask[i]   <= '0;
                tcam_action[i] <= '0;
                tcam_valid[i]  <= 1'b0;
            end
        end else if (prog_en) begin
            tcam_key[prog_index]    <= prog_key;
            tcam_mask[prog_index]   <= prog_mask;
            tcam_action[prog_index] <= prog_action;

            if (!tcam_valid[prog_index]) begin
                tcam_valid[prog_index] <= 1'b1;
                entry_count <= entry_count + 1;
            end
        end
    end

    assign entries_used = entry_count;
    assign table_full = (entry_count >= N);

    // =========================================================================
    // Parallel Match Logic
    // =========================================================================
    //
    // For each row i, compute:
    //   row_match[i] = AND over all bits j of:
    //     (mask[i][j] == 0) OR (key[i][j] == search_key[j])
    //
    // This synthesizes to a carry-chain AND tree.
    // =========================================================================

    logic [N-1:0] row_match;

    genvar i;
    generate
        for (i = 0; i < N; i++) begin : MATCH_ROW

            logic [W-1:0] bit_match;

            // Per-bit match: don't care OR exact match
            always_comb begin
                for (int j = 0; j < W; j++) begin
                    if (tcam_mask[i][j] == 1'b0)
                        bit_match[j] = 1'b1;  // Don't care
                    else
                        bit_match[j] = (tcam_key[i][j] == search_key[j]);
                end
            end

            // Reduce to single match bit (AND tree → carry chain)
            logic row_match_i;
            always_comb begin
                row_match_i = tcam_valid[i];  // Must be valid entry
                for (int k = 0; k < W; k++) begin
                    row_match_i = row_match_i & bit_match[k];
                end
            end

            assign row_match[i] = row_match_i;

        end
    endgenerate

    // =========================================================================
    // Priority Encoder (First Match Wins)
    // =========================================================================

    logic any_match_comb;
    logic [$clog2(N)-1:0] match_idx_comb;
    logic [ACTION_BITS-1:0] action_comb;

    always_comb begin
        any_match_comb = 1'b0;
        match_idx_comb = '0;
        action_comb = '0;

        for (int i = 0; i < N; i++) begin
            if (row_match[i] && !any_match_comb) begin
                any_match_comb = 1'b1;
                match_idx_comb = i[$clog2(N)-1:0];
                action_comb = tcam_action[i];
            end
        end
    end

    // =========================================================================
    // Output Pipeline (single cycle latency)
    // =========================================================================

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            match_valid  <= 1'b0;
            match_found  <= 1'b0;
            match_index  <= '0;
            match_action <= '0;
        end else begin
            match_valid  <= search_valid;
            match_found  <= any_match_comb;
            match_index  <= match_idx_comb;
            match_action <= action_comb;
        end
    end

endmodule


// =============================================================================
// TCAM with Multiple Priority Levels
// =============================================================================
//
// Extension with priority field per entry for more flexible matching.
// Higher priority entries checked first.
//
// =============================================================================

module lutram_tcam_priority #(
    parameter int N = 256,
    parameter int W = 32,
    parameter int ACTION_BITS = 8,
    parameter int PRIORITY_BITS = 4             // 16 priority levels
)(
    input  logic                        clk,
    input  logic                        rst_n,

    // Search
    input  logic                        search_valid,
    input  logic [W-1:0]                search_key,
    output logic                        match_valid,
    output logic                        match_found,
    output logic [$clog2(N)-1:0]        match_index,
    output logic [ACTION_BITS-1:0]      match_action,
    output logic [PRIORITY_BITS-1:0]    match_priority,

    // Programming
    input  logic                        prog_en,
    input  logic [$clog2(N)-1:0]        prog_index,
    input  logic [W-1:0]                prog_key,
    input  logic [W-1:0]                prog_mask,
    input  logic [ACTION_BITS-1:0]      prog_action,
    input  logic [PRIORITY_BITS-1:0]    prog_priority
);

    // Storage
    (* ramstyle = "MLAB" *) logic [W-1:0]              tcam_key     [0:N-1];
    (* ramstyle = "MLAB" *) logic [W-1:0]              tcam_mask    [0:N-1];
    (* ramstyle = "MLAB" *) logic [ACTION_BITS-1:0]    tcam_action  [0:N-1];
    (* ramstyle = "MLAB" *) logic [PRIORITY_BITS-1:0]  tcam_priority[0:N-1];
    (* ramstyle = "MLAB" *) logic                      tcam_valid   [0:N-1];

    // Programming
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < N; i++) begin
                tcam_key[i]      <= '0;
                tcam_mask[i]     <= '0;
                tcam_action[i]   <= '0;
                tcam_priority[i] <= '0;
                tcam_valid[i]    <= 1'b0;
            end
        end else if (prog_en) begin
            tcam_key[prog_index]      <= prog_key;
            tcam_mask[prog_index]     <= prog_mask;
            tcam_action[prog_index]   <= prog_action;
            tcam_priority[prog_index] <= prog_priority;
            tcam_valid[prog_index]    <= 1'b1;
        end
    end

    // Match logic
    logic [N-1:0] row_match;

    genvar i;
    generate
        for (i = 0; i < N; i++) begin : MATCH_ROW
            logic [W-1:0] bit_match;

            always_comb begin
                for (int j = 0; j < W; j++) begin
                    if (tcam_mask[i][j] == 1'b0)
                        bit_match[j] = 1'b1;
                    else
                        bit_match[j] = (tcam_key[i][j] == search_key[j]);
                end
            end

            logic row_match_i;
            always_comb begin
                row_match_i = tcam_valid[i];
                for (int k = 0; k < W; k++) begin
                    row_match_i = row_match_i & bit_match[k];
                end
            end

            assign row_match[i] = row_match_i;
        end
    endgenerate

    // Priority-based selection (highest priority among matches)
    logic any_match_comb;
    logic [$clog2(N)-1:0] match_idx_comb;
    logic [ACTION_BITS-1:0] action_comb;
    logic [PRIORITY_BITS-1:0] priority_comb;

    always_comb begin
        any_match_comb = 1'b0;
        match_idx_comb = '0;
        action_comb = '0;
        priority_comb = '0;

        for (int i = 0; i < N; i++) begin
            if (row_match[i]) begin
                if (!any_match_comb || tcam_priority[i] > priority_comb) begin
                    any_match_comb = 1'b1;
                    match_idx_comb = i[$clog2(N)-1:0];
                    action_comb = tcam_action[i];
                    priority_comb = tcam_priority[i];
                end
            end
        end
    end

    // Output pipeline
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            match_valid    <= 1'b0;
            match_found    <= 1'b0;
            match_index    <= '0;
            match_action   <= '0;
            match_priority <= '0;
        end else begin
            match_valid    <= search_valid;
            match_found    <= any_match_comb;
            match_index    <= match_idx_comb;
            match_action   <= action_comb;
            match_priority <= priority_comb;
        end
    end

endmodule


// =============================================================================
// Reflex TCAM - Specialized for SmartNIC/LAN Reflex Path
// =============================================================================
//
// Pre-configured for common reflex use cases:
// - 5-tuple flow hash matching
// - Pain packet detection
// - Service whitelist/blacklist
//
// Actions:
//   0x00 = PASS (default)
//   0x01 = DROP
//   0x02 = THROTTLE
//   0x03 = PRIORITY_BOOST
//   0x04 = GLITCH_TRIGGER
//   0x05 = LOG_ONLY
//   0x06 = REDIRECT
//   0x07 = MARK_SUSPICIOUS
//
// =============================================================================

module reflex_tcam #(
    parameter int N = 256,                      // Number of rules
    parameter int HASH_BITS = 32                // Flow hash width
)(
    input  logic                    clk,
    input  logic                    rst_n,

    // =========================================================================
    // Reflex Lookup Interface
    // =========================================================================

    input  logic                    lookup_valid,
    input  logic [HASH_BITS-1:0]    flow_hash,      // Pre-computed 5-tuple hash
    input  logic [7:0]              proto,          // Protocol (TCP=6, UDP=17)
    input  logic [15:0]             dst_port,       // Destination port

    output logic                    result_valid,
    output logic                    rule_matched,
    output logic [7:0]              action,         // See action codes above
    output logic [7:0]              rule_id,        // For logging

    // =========================================================================
    // Rule Programming
    // =========================================================================

    input  logic                    rule_prog_en,
    input  logic [7:0]              rule_prog_id,
    input  logic [HASH_BITS-1:0]    rule_hash,
    input  logic [HASH_BITS-1:0]    rule_hash_mask,
    input  logic [7:0]              rule_proto,
    input  logic [7:0]              rule_proto_mask,
    input  logic [15:0]             rule_port,
    input  logic [15:0]             rule_port_mask,
    input  logic [7:0]              rule_action
);

    // Action codes
    localparam logic [7:0] ACTION_PASS           = 8'h00;
    localparam logic [7:0] ACTION_DROP           = 8'h01;
    localparam logic [7:0] ACTION_THROTTLE       = 8'h02;
    localparam logic [7:0] ACTION_PRIORITY_BOOST = 8'h03;
    localparam logic [7:0] ACTION_GLITCH         = 8'h04;
    localparam logic [7:0] ACTION_LOG            = 8'h05;
    localparam logic [7:0] ACTION_REDIRECT       = 8'h06;
    localparam logic [7:0] ACTION_SUSPICIOUS     = 8'h07;

    // Composite key: hash + proto + port
    localparam int KEY_WIDTH = HASH_BITS + 8 + 16;  // 56 bits for 32-bit hash

    logic [KEY_WIDTH-1:0] search_key;
    assign search_key = {flow_hash, proto, dst_port};

    logic [KEY_WIDTH-1:0] prog_key;
    logic [KEY_WIDTH-1:0] prog_mask;
    assign prog_key = {rule_hash, rule_proto, rule_port};
    assign prog_mask = {rule_hash_mask, rule_proto_mask, rule_port_mask};

    // Internal TCAM
    logic match_valid_int;
    logic match_found_int;
    logic [$clog2(N)-1:0] match_index_int;
    logic [7:0] match_action_int;

    lutram_tcam #(
        .N(N),
        .W(KEY_WIDTH),
        .ACTION_BITS(8)
    ) u_tcam (
        .clk(clk),
        .rst_n(rst_n),

        .search_valid(lookup_valid),
        .search_key(search_key),
        .match_valid(match_valid_int),
        .match_found(match_found_int),
        .match_index(match_index_int),
        .match_action(match_action_int),

        .prog_en(rule_prog_en),
        .prog_index(rule_prog_id[$clog2(N)-1:0]),
        .prog_key(prog_key),
        .prog_mask(prog_mask),
        .prog_action(rule_action),

        .entries_used(),
        .table_full()
    );

    // Output
    assign result_valid = match_valid_int;
    assign rule_matched = match_found_int;
    assign action = match_found_int ? match_action_int : ACTION_PASS;
    assign rule_id = match_index_int[7:0];

endmodule


// =============================================================================
// Testbench
// =============================================================================

`ifdef SIMULATION
module lutram_tcam_tb;

    localparam int N = 64;
    localparam int W = 32;

    logic clk = 0;
    logic rst_n = 0;

    // Search
    logic search_valid;
    logic [W-1:0] search_key;
    logic match_valid;
    logic match_found;
    logic [$clog2(N)-1:0] match_index;
    logic [7:0] match_action;

    // Programming
    logic prog_en;
    logic [$clog2(N)-1:0] prog_index;
    logic [W-1:0] prog_key;
    logic [W-1:0] prog_mask;
    logic [7:0] prog_action;

    // DUT
    lutram_tcam #(
        .N(N),
        .W(W),
        .ACTION_BITS(8)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .search_valid(search_valid),
        .search_key(search_key),
        .match_valid(match_valid),
        .match_found(match_found),
        .match_index(match_index),
        .match_action(match_action),
        .prog_en(prog_en),
        .prog_index(prog_index),
        .prog_key(prog_key),
        .prog_mask(prog_mask),
        .prog_action(prog_action),
        .entries_used(),
        .table_full()
    );

    // Clock
    always #1.43 clk = ~clk;  // ~350 MHz

    initial begin
        $dumpfile("lutram_tcam.vcd");
        $dumpvars(0, lutram_tcam_tb);

        // Initialize
        search_valid = 0;
        search_key = 0;
        prog_en = 0;
        prog_index = 0;
        prog_key = 0;
        prog_mask = 0;
        prog_action = 0;

        // Reset
        #20 rst_n = 1;
        #20;

        // Program rule 0: exact match for 0xDEADBEEF -> action 0x42
        @(posedge clk);
        prog_en = 1;
        prog_index = 0;
        prog_key = 32'hDEADBEEF;
        prog_mask = 32'hFFFFFFFF;  // All bits care
        prog_action = 8'h42;
        @(posedge clk);
        prog_en = 0;

        // Program rule 1: prefix match for 0xCAFE???? -> action 0x55
        @(posedge clk);
        prog_en = 1;
        prog_index = 1;
        prog_key = 32'hCAFE0000;
        prog_mask = 32'hFFFF0000;  // Only upper 16 bits care
        prog_action = 8'h55;
        @(posedge clk);
        prog_en = 0;
        #20;

        // Test 1: Exact match
        @(posedge clk);
        search_valid = 1;
        search_key = 32'hDEADBEEF;
        @(posedge clk);
        search_valid = 0;
        @(posedge clk);
        assert(match_found == 1) else $error("T1: Expected match");
        assert(match_action == 8'h42) else $error("T1: Wrong action");
        $display("Test 1 PASS: Exact match -> action 0x%02x", match_action);

        // Test 2: Prefix match
        @(posedge clk);
        search_valid = 1;
        search_key = 32'hCAFEBABE;
        @(posedge clk);
        search_valid = 0;
        @(posedge clk);
        assert(match_found == 1) else $error("T2: Expected match");
        assert(match_action == 8'h55) else $error("T2: Wrong action");
        $display("Test 2 PASS: Prefix match -> action 0x%02x", match_action);

        // Test 3: No match
        @(posedge clk);
        search_valid = 1;
        search_key = 32'h12345678;
        @(posedge clk);
        search_valid = 0;
        @(posedge clk);
        assert(match_found == 0) else $error("T3: Expected no match");
        $display("Test 3 PASS: No match");

        $display("\nAll tests PASSED!");
        #100 $finish;
    end

endmodule
`endif
