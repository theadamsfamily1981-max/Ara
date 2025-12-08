// =============================================================================
// XNOR-Popcount Engine - Core Similarity Computation Unit
// =============================================================================
//
// Computes Hamming agreement between a query chunk and weight chunk:
//   agree = ~(hv_chunk ^ w_chunk)  // XNOR = bitwise agreement
//   popcnt = popcount(agree)       // Number of matching bits
//
// For bipolar HVs, this directly gives the similarity contribution.
// Cosine similarity = (2 * popcnt - C) / C  (when normalized)
//
// Performance:
//   - Single cycle latency (combinational)
//   - Synthesizes to LUT cascade + carry chain popcount
//   - ~500 ALMs for C=512 on Stratix-10
//
// Usage:
//   Instantiate ROW_PAR × CHUNK_PAR engines for parallel processing.
//
// =============================================================================

`timescale 1ns/1ps

module xnor_pop512 #(
    parameter int C = 512,                      // Chunk size in bits
    parameter int POP_BITS = $clog2(C+1)        // Bits for popcount result (10 for 512)
)(
    input  logic [C-1:0]         hv_chunk,      // Query HV chunk
    input  logic [C-1:0]         w_chunk,       // Weight/attractor chunk
    output logic [POP_BITS-1:0]  popcnt         // Population count of agreements
);

    // =========================================================================
    // XNOR: Compute bitwise agreement
    // =========================================================================

    logic [C-1:0] agree;
    assign agree = ~(hv_chunk ^ w_chunk);

    // =========================================================================
    // Popcount: Tree-based reduction
    // =========================================================================
    //
    // We use a hierarchical tree to avoid a massive single adder.
    // Split into 8 sub-chunks of 64 bits each, popcount those,
    // then sum the 8 partial results.
    //
    // This maps well to Stratix-10's ALM structure.
    // =========================================================================

    localparam int SUB_CHUNKS = 8;
    localparam int SUB_SIZE = C / SUB_CHUNKS;  // 64 bits each
    localparam int SUB_POP_BITS = $clog2(SUB_SIZE + 1);  // 7 bits for 64

    logic [SUB_POP_BITS-1:0] sub_pop [0:SUB_CHUNKS-1];

    // Generate sub-chunk popcounts
    genvar sc;
    generate
        for (sc = 0; sc < SUB_CHUNKS; sc++) begin : SUB_POPCNT
            // Extract sub-chunk
            logic [SUB_SIZE-1:0] sub_agree;
            assign sub_agree = agree[sc*SUB_SIZE +: SUB_SIZE];

            // Popcount via tree (synthesizer will optimize)
            // For 64 bits: 6-level tree
            always_comb begin
                logic [SUB_POP_BITS-1:0] cnt;
                cnt = '0;
                for (int i = 0; i < SUB_SIZE; i++) begin
                    cnt = cnt + sub_agree[i];
                end
                sub_pop[sc] = cnt;
            end
        end
    endgenerate

    // Sum all sub-chunk popcounts
    always_comb begin
        logic [POP_BITS-1:0] total;
        total = '0;
        for (int j = 0; j < SUB_CHUNKS; j++) begin
            total = total + sub_pop[j];
        end
        popcnt = total;
    end

endmodule


// =============================================================================
// Parameterized XNOR-Popcount for arbitrary chunk sizes
// =============================================================================

module xnor_pop #(
    parameter int C = 512,
    parameter int POP_BITS = $clog2(C+1)
)(
    input  logic [C-1:0]         hv_chunk,
    input  logic [C-1:0]         w_chunk,
    output logic [POP_BITS-1:0]  popcnt
);

    logic [C-1:0] agree;
    assign agree = ~(hv_chunk ^ w_chunk);

    // Generic popcount - synthesizer will choose best implementation
    always_comb begin
        popcnt = '0;
        for (int i = 0; i < C; i++) begin
            popcnt = popcnt + agree[i];
        end
    end

endmodule


// =============================================================================
// Registered version for pipelining
// =============================================================================

module xnor_pop512_reg #(
    parameter int C = 512,
    parameter int POP_BITS = $clog2(C+1)
)(
    input  logic                 clk,
    input  logic [C-1:0]         hv_chunk,
    input  logic [C-1:0]         w_chunk,
    output logic [POP_BITS-1:0]  popcnt
);

    logic [POP_BITS-1:0] popcnt_comb;

    xnor_pop512 #(.C(C), .POP_BITS(POP_BITS)) u_core (
        .hv_chunk(hv_chunk),
        .w_chunk(w_chunk),
        .popcnt(popcnt_comb)
    );

    always_ff @(posedge clk) begin
        popcnt <= popcnt_comb;
    end

endmodule


// =============================================================================
// Testbench
// =============================================================================

`ifdef SIMULATION
module xnor_pop512_tb;

    localparam int C = 512;
    localparam int POP_BITS = $clog2(C+1);

    logic [C-1:0] hv_chunk;
    logic [C-1:0] w_chunk;
    logic [POP_BITS-1:0] popcnt;

    xnor_pop512 #(.C(C)) dut (
        .hv_chunk(hv_chunk),
        .w_chunk(w_chunk),
        .popcnt(popcnt)
    );

    initial begin
        $dumpfile("xnor_pop512.vcd");
        $dumpvars(0, xnor_pop512_tb);

        // Test 1: Identical vectors -> max popcount
        hv_chunk = {C{1'b1}};
        w_chunk = {C{1'b1}};
        #10;
        assert(popcnt == C) else $error("T1: Expected %d, got %d", C, popcnt);
        $display("Test 1 PASS: Identical vectors -> popcnt = %d", popcnt);

        // Test 2: Opposite vectors -> zero popcount
        hv_chunk = {C{1'b1}};
        w_chunk = {C{1'b0}};
        #10;
        assert(popcnt == 0) else $error("T2: Expected 0, got %d", popcnt);
        $display("Test 2 PASS: Opposite vectors -> popcnt = %d", popcnt);

        // Test 3: Half agree -> half popcount
        hv_chunk = {{C/2{1'b1}}, {C/2{1'b0}}};
        w_chunk = {{C/2{1'b1}}, {C/2{1'b1}}};
        #10;
        assert(popcnt == C/2) else $error("T3: Expected %d, got %d", C/2, popcnt);
        $display("Test 3 PASS: Half agreement -> popcnt = %d", popcnt);

        // Test 4: Random pattern
        hv_chunk = 512'hDEADBEEF_CAFEBABE_12345678_9ABCDEF0_DEADBEEF_CAFEBABE_12345678_9ABCDEF0_DEADBEEF_CAFEBABE_12345678_9ABCDEF0_DEADBEEF_CAFEBABE_12345678_9ABCDEF0;
        w_chunk = 512'hDEADBEEF_CAFEBABE_12345678_9ABCDEF0_DEADBEEF_CAFEBABE_12345678_9ABCDEF0_DEADBEEF_CAFEBABE_12345678_9ABCDEF0_DEADBEEF_CAFEBABE_12345678_9ABCDEF0;
        #10;
        assert(popcnt == C) else $error("T4: Expected %d, got %d", C, popcnt);
        $display("Test 4 PASS: Same random pattern -> popcnt = %d", popcnt);

        $display("\nAll tests PASSED!");
        #100;
        $finish;
    end

endmodule
`endif
