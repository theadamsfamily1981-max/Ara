// =============================================================================
// HTC Weight Memory Bank - Banked BRAM for Parallel Attractor Access
// =============================================================================
//
// Provides high-bandwidth access to attractor weights for the CAM core.
//
// Architecture:
//   - Weights stored as: weights[row][chunk][bit]
//   - Banked to allow ROW_PAR × CHUNK_PAR parallel reads
//   - Each bank is a M20K BRAM in width-heavy mode
//
// Memory Layout:
//   - Total: R × N_CHUNKS × C bits = 2048 × 32 × 512 = 32 Mbit
//   - Banks: ROW_PAR banks, each storing R/ROW_PAR rows
//   - Each bank: (R/ROW_PAR) × (N_CHUNKS × C) bits
//
// Access Pattern:
//   - For (row_group, chunk_group):
//     - Read ROW_PAR rows simultaneously
//     - Each row provides CHUNK_PAR chunks
//
// =============================================================================

`timescale 1ns/1ps

module htc_weight_bank #(
    parameter int D          = 16384,
    parameter int R          = 2048,
    parameter int C          = 512,
    parameter int N_CHUNKS   = D / C,           // 32
    parameter int ROW_PAR    = 64,
    parameter int CHUNK_PAR  = 4,
    parameter int R_GROUPS   = R / ROW_PAR,     // 32
    parameter int C_GROUPS   = N_CHUNKS / CHUNK_PAR  // 8
)(
    input  logic                        clk,
    input  logic                        rst_n,

    // =========================================================================
    // Read Interface (for CAM queries)
    // =========================================================================

    input  logic                        read_req,
    input  logic [$clog2(R_GROUPS)-1:0] read_row_group,
    input  logic [$clog2(C_GROUPS)-1:0] read_chunk_group,
    output logic                        read_resp_valid,
    output logic [C*CHUNK_PAR-1:0]      read_data [0:ROW_PAR-1],

    // =========================================================================
    // Write Interface (for plasticity updates)
    // =========================================================================

    input  logic                        write_en,
    input  logic [$clog2(R)-1:0]        write_row,
    input  logic [$clog2(N_CHUNKS)-1:0] write_chunk,
    input  logic [C-1:0]                write_data,

    // =========================================================================
    // Bulk Load Interface (for initialization)
    // =========================================================================

    input  logic                        bulk_load_en,
    input  logic [$clog2(R)-1:0]        bulk_row,
    input  logic [D-1:0]                bulk_hv  // Full HV to load
);

    // =========================================================================
    // Memory Declaration
    // =========================================================================
    //
    // We organize memory as ROW_PAR banks.
    // Each bank stores R/ROW_PAR complete rows.
    // Each row has N_CHUNKS chunks of C bits.
    //
    // Bank[b] contains rows where row % ROW_PAR == b
    //
    // Address within bank: (row / ROW_PAR) * N_CHUNKS + chunk
    // =========================================================================

    localparam int BANK_DEPTH = R_GROUPS * N_CHUNKS;  // Entries per bank
    localparam int BANK_WIDTH = C;                     // Bits per entry

    // BRAM banks
    (* ram_style = "block" *)
    logic [BANK_WIDTH-1:0] mem_bank [0:ROW_PAR-1][0:BANK_DEPTH-1];

    // =========================================================================
    // Read Pipeline (2-cycle latency)
    // =========================================================================

    // Stage 1: Address computation
    logic read_req_r1;
    logic [$clog2(R_GROUPS)-1:0] read_row_group_r1;
    logic [$clog2(C_GROUPS)-1:0] read_chunk_group_r1;

    // Stage 2: Data output
    logic read_req_r2;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            read_req_r1 <= 1'b0;
            read_req_r2 <= 1'b0;
        end else begin
            read_req_r1 <= read_req;
            read_row_group_r1 <= read_row_group;
            read_chunk_group_r1 <= read_chunk_group;
            read_req_r2 <= read_req_r1;
        end
    end

    assign read_resp_valid = read_req_r2;

    // =========================================================================
    // Parallel Read from All Banks
    // =========================================================================

    genvar bank_idx, chunk_idx;
    generate
        for (bank_idx = 0; bank_idx < ROW_PAR; bank_idx++) begin : BANK_READ
            // Read CHUNK_PAR consecutive chunks for this bank
            for (chunk_idx = 0; chunk_idx < CHUNK_PAR; chunk_idx++) begin : CHUNK_READ
                logic [$clog2(BANK_DEPTH)-1:0] read_addr;
                logic [BANK_WIDTH-1:0] read_chunk_data;

                // Compute address: row_group * N_CHUNKS + chunk_group * CHUNK_PAR + chunk_idx
                assign read_addr = read_row_group_r1 * N_CHUNKS +
                                   read_chunk_group_r1 * CHUNK_PAR +
                                   chunk_idx;

                // BRAM read
                always_ff @(posedge clk) begin
                    if (read_req_r1) begin
                        read_chunk_data <= mem_bank[bank_idx][read_addr];
                    end
                end

                // Pack into output
                assign read_data[bank_idx][chunk_idx*C +: C] = read_chunk_data;
            end
        end
    endgenerate

    // =========================================================================
    // Single-Chunk Write (for plasticity)
    // =========================================================================

    always_ff @(posedge clk) begin
        if (write_en) begin
            // Determine which bank this row belongs to
            logic [$clog2(ROW_PAR)-1:0] bank;
            logic [$clog2(BANK_DEPTH)-1:0] addr;

            bank = write_row[$clog2(ROW_PAR)-1:0];
            addr = (write_row / ROW_PAR) * N_CHUNKS + write_chunk;

            mem_bank[bank][addr] <= write_data;
        end
    end

    // =========================================================================
    // Bulk HV Load (for initialization)
    // =========================================================================

    always_ff @(posedge clk) begin
        if (bulk_load_en) begin
            logic [$clog2(ROW_PAR)-1:0] bank;
            logic [$clog2(R_GROUPS)-1:0] row_in_bank;

            bank = bulk_row[$clog2(ROW_PAR)-1:0];
            row_in_bank = bulk_row / ROW_PAR;

            // Write all chunks for this row
            for (int ch = 0; ch < N_CHUNKS; ch++) begin
                logic [$clog2(BANK_DEPTH)-1:0] addr;
                addr = row_in_bank * N_CHUNKS + ch;
                mem_bank[bank][addr] <= bulk_hv[ch*C +: C];
            end
        end
    end

endmodule


// =============================================================================
// Simplified Weight Bank for Testing (single-port)
// =============================================================================

module htc_weight_bank_simple #(
    parameter int D          = 16384,
    parameter int R          = 2048,
    parameter int C          = 512,
    parameter int N_CHUNKS   = D / C,
    parameter int ROW_PAR    = 64,
    parameter int CHUNK_PAR  = 4
)(
    input  logic                        clk,
    input  logic                        rst_n,

    // Read interface
    input  logic                        read_req,
    input  logic [$clog2(R)-1:0]        read_row_start,
    input  logic [$clog2(N_CHUNKS)-1:0] read_chunk_start,
    output logic                        read_resp_valid,
    output logic [C*CHUNK_PAR-1:0]      read_data [0:ROW_PAR-1],

    // Write interface (full HV at a time)
    input  logic                        write_en,
    input  logic [$clog2(R)-1:0]        write_row,
    input  logic [D-1:0]                write_hv
);

    // Simple flat storage for simulation
    logic [D-1:0] weights [0:R-1];

    // Read pipeline
    logic read_req_r1, read_req_r2;
    logic [$clog2(R)-1:0] read_row_r1;
    logic [$clog2(N_CHUNKS)-1:0] read_chunk_r1;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            read_req_r1 <= 1'b0;
            read_req_r2 <= 1'b0;
        end else begin
            read_req_r1 <= read_req;
            read_row_r1 <= read_row_start;
            read_chunk_r1 <= read_chunk_start;
            read_req_r2 <= read_req_r1;
        end
    end

    assign read_resp_valid = read_req_r2;

    // Generate read data
    genvar r, ch;
    generate
        for (r = 0; r < ROW_PAR; r++) begin : GEN_ROWS
            for (ch = 0; ch < CHUNK_PAR; ch++) begin : GEN_CHUNKS
                always_ff @(posedge clk) begin
                    if (read_req_r1) begin
                        logic [$clog2(R)-1:0] row_addr;
                        logic [$clog2(N_CHUNKS)-1:0] chunk_addr;
                        row_addr = read_row_r1 + r;
                        chunk_addr = read_chunk_r1 + ch;
                        read_data[r][ch*C +: C] <= weights[row_addr][chunk_addr*C +: C];
                    end
                end
            end
        end
    endgenerate

    // Write
    always_ff @(posedge clk) begin
        if (write_en) begin
            weights[write_row] <= write_hv;
        end
    end

endmodule
