// =============================================================================
// Multiple Hash CAM - High-Fanout Flow/Episode Candidate Generator
// =============================================================================
//
// Fast approximate lookup for LAN flows and episode signatures.
// Uses multiple hash functions to index into BRAM tables.
//
// Architecture:
//   - K independent hash tables, each TABLE_SIZE entries
//   - Each entry stores a candidate ID
//   - Query returns up to K candidate IDs (may have duplicates/collisions)
//   - 2-3 cycle latency (BRAM read pipeline)
//
// Use Cases:
//   - Flow signature lookup (~ns latency)
//   - Episode/pattern candidate generation
//   - LAN nervous system reflex index
//
// NOT for exact HDC similarity - use XNOR-popcount CAM for that.
// This is the "have I seen something like this before?" fast path.
//
// Sizing:
//   - K=4 tables × TABLE_SIZE=4096 × ID_BITS=16 = 256 Kbits
//   - Fits in a few M20K BRAMs
//
// =============================================================================

`timescale 1ns/1ps

module hash_cam #(
    parameter int SIG_BITS      = 64,           // Signature width
    parameter int TABLE_SIZE    = 4096,         // Entries per table (power-of-2)
    parameter int K_HASH        = 4,            // Number of hash tables
    parameter int ID_BITS       = 16,           // ID bits per entry
    parameter int ADDR_BITS     = $clog2(TABLE_SIZE)
)(
    input  logic                    clk,
    input  logic                    rst_n,

    // =========================================================================
    // Query Interface
    // =========================================================================

    input  logic                    q_valid,
    input  logic [SIG_BITS-1:0]     q_sig,          // Precomputed signature
    output logic                    q_ready,

    // Result: up to K candidates
    output logic                    result_valid,
    output logic [ID_BITS-1:0]      cand_id [0:K_HASH-1],
    output logic [K_HASH-1:0]       cand_valid,     // Which candidates are valid

    // =========================================================================
    // Programming Interface (insert/update)
    // =========================================================================

    input  logic                    prog_en,
    input  logic [SIG_BITS-1:0]     prog_sig,
    input  logic [ID_BITS-1:0]      prog_id,

    // =========================================================================
    // Status
    // =========================================================================

    output logic [31:0]             total_entries,
    output logic [31:0]             collision_count
);

    // =========================================================================
    // Hash Seeds (fixed constants, designed for good distribution)
    // =========================================================================

    localparam logic [SIG_BITS-1:0] HASH_SEED [0:K_HASH-1] = '{
        64'h9E3779B97F4A7C15,  // Golden ratio derived
        64'hC2B2AE3D27D4EB4F,  // Prime-based
        64'h165667B19E3779F9,  // Mixed
        64'h85EBCA6B27D4A7F3   // Another prime
    };

    // =========================================================================
    // BRAM Tables
    // =========================================================================

    // Each table: TABLE_SIZE entries of (valid bit + ID)
    localparam int ENTRY_BITS = 1 + ID_BITS;

    (* ramstyle = "M20K" *) logic [ENTRY_BITS-1:0] table_mem [0:K_HASH-1][0:TABLE_SIZE-1];

    // =========================================================================
    // Hash Function: XOR-fold signature with seed
    // =========================================================================

    function automatic logic [ADDR_BITS-1:0] compute_hash(
        input logic [SIG_BITS-1:0] sig,
        input logic [SIG_BITS-1:0] seed
    );
        logic [SIG_BITS-1:0] mixed;
        mixed = sig ^ seed;

        // Fold down to ADDR_BITS via XOR
        logic [ADDR_BITS-1:0] addr;
        addr = '0;
        for (int i = 0; i < SIG_BITS; i += ADDR_BITS) begin
            addr = addr ^ mixed[i +: ADDR_BITS];
        end

        return addr;
    endfunction

    // =========================================================================
    // State Machine
    // =========================================================================

    typedef enum logic [1:0] {
        S_IDLE,
        S_READ,
        S_RESULT
    } state_t;

    state_t state, state_next;

    // Pipeline registers
    logic [SIG_BITS-1:0] sig_reg;
    logic [ADDR_BITS-1:0] addr_reg [0:K_HASH-1];
    logic [ENTRY_BITS-1:0] entry_reg [0:K_HASH-1];

    // =========================================================================
    // Address Computation (combinational)
    // =========================================================================

    logic [ADDR_BITS-1:0] hash_addr [0:K_HASH-1];

    genvar k;
    generate
        for (k = 0; k < K_HASH; k++) begin : HASH_COMPUTE
            assign hash_addr[k] = compute_hash(sig_reg, HASH_SEED[k]);
        end
    endgenerate

    // =========================================================================
    // State Machine Logic
    // =========================================================================

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            sig_reg <= '0;
            for (int i = 0; i < K_HASH; i++) begin
                addr_reg[i] <= '0;
                entry_reg[i] <= '0;
            end
        end else begin
            state <= state_next;

            case (state)
                S_IDLE: begin
                    if (q_valid && q_ready) begin
                        sig_reg <= q_sig;
                    end
                end

                S_READ: begin
                    // Latch addresses and read from BRAMs
                    for (int i = 0; i < K_HASH; i++) begin
                        addr_reg[i] <= hash_addr[i];
                        entry_reg[i] <= table_mem[i][hash_addr[i]];
                    end
                end

                default: ;
            endcase
        end
    end

    always_comb begin
        state_next = state;
        q_ready = 1'b0;
        result_valid = 1'b0;

        case (state)
            S_IDLE: begin
                q_ready = 1'b1;
                if (q_valid) begin
                    state_next = S_READ;
                end
            end

            S_READ: begin
                state_next = S_RESULT;
            end

            S_RESULT: begin
                result_valid = 1'b1;
                state_next = S_IDLE;
            end

            default: state_next = S_IDLE;
        endcase
    end

    // =========================================================================
    // Output: Extract candidates from entries
    // =========================================================================

    genvar ko;
    generate
        for (ko = 0; ko < K_HASH; ko++) begin : OUTPUT
            assign cand_valid[ko] = entry_reg[ko][ID_BITS];  // Valid bit
            assign cand_id[ko] = entry_reg[ko][ID_BITS-1:0]; // ID
        end
    endgenerate

    // =========================================================================
    // Programming: Insert entry into all K tables
    // =========================================================================

    logic [ADDR_BITS-1:0] prog_addr [0:K_HASH-1];

    generate
        for (k = 0; k < K_HASH; k++) begin : PROG_HASH
            assign prog_addr[k] = compute_hash(prog_sig, HASH_SEED[k]);
        end
    endgenerate

    // Statistics
    logic [31:0] entry_counter;
    logic [31:0] collision_counter;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            entry_counter <= '0;
            collision_counter <= '0;
            // Initialize tables to invalid
            for (int t = 0; t < K_HASH; t++) begin
                for (int e = 0; e < TABLE_SIZE; e++) begin
                    table_mem[t][e] <= '0;
                end
            end
        end else if (prog_en) begin
            entry_counter <= entry_counter + 1;

            for (int t = 0; t < K_HASH; t++) begin
                // Check for collision
                if (table_mem[t][prog_addr[t]][ID_BITS]) begin
                    collision_counter <= collision_counter + 1;
                end
                // Write entry
                table_mem[t][prog_addr[t]] <= {1'b1, prog_id};
            end
        end
    end

    assign total_entries = entry_counter;
    assign collision_count = collision_counter;

endmodule


// =============================================================================
// Hash CAM with Verification (stores truncated signature for verification)
// =============================================================================

module hash_cam_verified #(
    parameter int SIG_BITS      = 64,
    parameter int TABLE_SIZE    = 4096,
    parameter int K_HASH        = 4,
    parameter int ID_BITS       = 16,
    parameter int TAG_BITS      = 16,           // Truncated signature for verification
    parameter int ADDR_BITS     = $clog2(TABLE_SIZE)
)(
    input  logic                    clk,
    input  logic                    rst_n,

    // Query
    input  logic                    q_valid,
    input  logic [SIG_BITS-1:0]     q_sig,
    output logic                    q_ready,

    // Result with verification
    output logic                    result_valid,
    output logic [ID_BITS-1:0]      cand_id [0:K_HASH-1],
    output logic [K_HASH-1:0]       cand_verified,  // Tag matched

    // Programming
    input  logic                    prog_en,
    input  logic [SIG_BITS-1:0]     prog_sig,
    input  logic [ID_BITS-1:0]      prog_id
);

    // Entry: valid + tag + id
    localparam int ENTRY_BITS = 1 + TAG_BITS + ID_BITS;

    // Hash seeds
    localparam logic [SIG_BITS-1:0] HASH_SEED [0:K_HASH-1] = '{
        64'h9E3779B97F4A7C15,
        64'hC2B2AE3D27D4EB4F,
        64'h165667B19E3779F9,
        64'h85EBCA6B27D4A7F3
    };

    (* ramstyle = "M20K" *) logic [ENTRY_BITS-1:0] table_mem [0:K_HASH-1][0:TABLE_SIZE-1];

    // Hash function
    function automatic logic [ADDR_BITS-1:0] compute_hash(
        input logic [SIG_BITS-1:0] sig,
        input logic [SIG_BITS-1:0] seed
    );
        logic [SIG_BITS-1:0] mixed;
        mixed = sig ^ seed;
        logic [ADDR_BITS-1:0] addr;
        addr = '0;
        for (int i = 0; i < SIG_BITS; i += ADDR_BITS) begin
            addr = addr ^ mixed[i +: ADDR_BITS];
        end
        return addr;
    endfunction

    // Extract tag from signature
    function automatic logic [TAG_BITS-1:0] compute_tag(
        input logic [SIG_BITS-1:0] sig
    );
        // Use upper bits as tag
        return sig[SIG_BITS-1 -: TAG_BITS];
    endfunction

    // State machine
    typedef enum logic [1:0] { S_IDLE, S_READ, S_RESULT } state_t;
    state_t state, state_next;

    logic [SIG_BITS-1:0] sig_reg;
    logic [TAG_BITS-1:0] tag_reg;
    logic [ENTRY_BITS-1:0] entry_reg [0:K_HASH-1];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            sig_reg <= '0;
            tag_reg <= '0;
        end else begin
            state <= state_next;

            if (state == S_IDLE && q_valid) begin
                sig_reg <= q_sig;
                tag_reg <= compute_tag(q_sig);
            end

            if (state == S_READ) begin
                for (int i = 0; i < K_HASH; i++) begin
                    logic [ADDR_BITS-1:0] addr;
                    addr = compute_hash(sig_reg, HASH_SEED[i]);
                    entry_reg[i] <= table_mem[i][addr];
                end
            end
        end
    end

    always_comb begin
        state_next = state;
        q_ready = 1'b0;
        result_valid = 1'b0;

        case (state)
            S_IDLE: begin
                q_ready = 1'b1;
                if (q_valid) state_next = S_READ;
            end
            S_READ: state_next = S_RESULT;
            S_RESULT: begin
                result_valid = 1'b1;
                state_next = S_IDLE;
            end
        endcase
    end

    // Output with verification
    genvar ko;
    generate
        for (ko = 0; ko < K_HASH; ko++) begin : OUT
            logic entry_valid;
            logic [TAG_BITS-1:0] entry_tag;
            logic [ID_BITS-1:0] entry_id;

            assign entry_valid = entry_reg[ko][ENTRY_BITS-1];
            assign entry_tag = entry_reg[ko][ENTRY_BITS-2 -: TAG_BITS];
            assign entry_id = entry_reg[ko][ID_BITS-1:0];

            assign cand_id[ko] = entry_id;
            assign cand_verified[ko] = entry_valid && (entry_tag == tag_reg);
        end
    endgenerate

    // Programming
    always_ff @(posedge clk) begin
        if (prog_en) begin
            logic [TAG_BITS-1:0] prog_tag;
            prog_tag = compute_tag(prog_sig);

            for (int t = 0; t < K_HASH; t++) begin
                logic [ADDR_BITS-1:0] addr;
                addr = compute_hash(prog_sig, HASH_SEED[t]);
                table_mem[t][addr] <= {1'b1, prog_tag, prog_id};
            end
        end
    end

endmodule


// =============================================================================
// Flow Signature Generator - Compute 64-bit signature from flow fields
// =============================================================================

module flow_sig_gen #(
    parameter int SIG_BITS = 64
)(
    input  logic [31:0]     src_ip,
    input  logic [31:0]     dst_ip,
    input  logic [15:0]     src_port,
    input  logic [15:0]     dst_port,
    input  logic [7:0]      proto,
    output logic [SIG_BITS-1:0] sig
);

    // Mix using XOR and rotation
    // Simple but effective for flow discrimination

    logic [63:0] mixed;

    always_comb begin
        // Combine fields with rotations to avoid alignment issues
        mixed = '0;
        mixed ^= {src_ip, dst_ip};
        mixed ^= {dst_ip[15:0], src_port, dst_port, proto, 8'h00};
        mixed ^= {16'h0000, src_port, 16'h0000, dst_port};

        // Additional mixing (simple)
        mixed ^= (mixed >> 17);
        mixed ^= (mixed << 31);

        sig = mixed;
    end

endmodule


// =============================================================================
// Testbench
// =============================================================================

`ifdef SIMULATION
module hash_cam_tb;

    localparam int SIG_BITS = 64;
    localparam int TABLE_SIZE = 256;  // Smaller for test
    localparam int K_HASH = 4;
    localparam int ID_BITS = 16;

    logic clk = 0;
    logic rst_n = 0;

    logic q_valid;
    logic [SIG_BITS-1:0] q_sig;
    logic q_ready;
    logic result_valid;
    logic [ID_BITS-1:0] cand_id [0:K_HASH-1];
    logic [K_HASH-1:0] cand_valid;

    logic prog_en;
    logic [SIG_BITS-1:0] prog_sig;
    logic [ID_BITS-1:0] prog_id;

    hash_cam #(
        .SIG_BITS(SIG_BITS),
        .TABLE_SIZE(TABLE_SIZE),
        .K_HASH(K_HASH),
        .ID_BITS(ID_BITS)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .q_valid(q_valid),
        .q_sig(q_sig),
        .q_ready(q_ready),
        .result_valid(result_valid),
        .cand_id(cand_id),
        .cand_valid(cand_valid),
        .prog_en(prog_en),
        .prog_sig(prog_sig),
        .prog_id(prog_id),
        .total_entries(),
        .collision_count()
    );

    always #1.43 clk = ~clk;

    initial begin
        $dumpfile("hash_cam.vcd");
        $dumpvars(0, hash_cam_tb);

        q_valid = 0;
        q_sig = 0;
        prog_en = 0;
        prog_sig = 0;
        prog_id = 0;

        #20 rst_n = 1;
        #20;

        // Insert some entries
        for (int i = 0; i < 10; i++) begin
            @(posedge clk);
            prog_en = 1;
            prog_sig = 64'hDEADBEEF00000000 + i;
            prog_id = 100 + i;
            @(posedge clk);
            prog_en = 0;
        end
        #20;

        // Query for inserted entry
        @(posedge clk);
        q_valid = 1;
        q_sig = 64'hDEADBEEF00000005;
        @(posedge clk);
        q_valid = 0;

        wait(result_valid);
        #10;

        $display("Query result:");
        for (int k = 0; k < K_HASH; k++) begin
            $display("  Table %0d: valid=%b, id=%0d", k, cand_valid[k], cand_id[k]);
        end

        // Query for non-existent
        @(posedge clk);
        q_valid = 1;
        q_sig = 64'h1234567890ABCDEF;
        @(posedge clk);
        q_valid = 0;

        wait(result_valid);
        #10;

        $display("Non-existent query result:");
        for (int k = 0; k < K_HASH; k++) begin
            $display("  Table %0d: valid=%b, id=%0d", k, cand_valid[k], cand_id[k]);
        end

        #100 $finish;
    end

endmodule
`endif
