// =============================================================================
// uram_wrapper.sv
//
// UltraRAM Wrapper for SB-852 (VU7P) - Large Weight Storage
// Ara-SYNERGY Kitten Fabric - Synthesizable RTL
//
// This wrapper ensures weight storage uses UltraRAM (URAM) instead of
// distributed LUTRAM or standard BRAM. The VU7P has 1920 URAM blocks,
// each 288Kbit (36KB). Using BRAM would consume excessive logic resources.
//
// Usage:
//   Replace standard BRAM instantiation with this wrapper for:
//   - CSR values (weight storage)
//   - CSR indices (column indices)
//   - Large LUT arrays
//
// NOTE: URAM is 4K x 72b natively. This wrapper handles width conversion.
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

// -----------------------------------------------------------------------------
// Simple Dual-Port URAM Wrapper (1 read, 1 write port)
// -----------------------------------------------------------------------------
module uram_sdp_wrapper #(
    parameter DATA_WIDTH = 32,
    parameter ADDR_WIDTH = 18,       // 2^18 = 256K entries
    parameter DEPTH      = 262144,   // Number of entries
    parameter INIT_FILE  = ""        // Optional hex file for initialization
) (
    input  wire                    clk,
    input  wire                    rst,

    // Write port
    input  wire                    wr_en,
    input  wire [ADDR_WIDTH-1:0]   wr_addr,
    input  wire [DATA_WIDTH-1:0]   wr_data,

    // Read port
    input  wire                    rd_en,
    input  wire [ADDR_WIDTH-1:0]   rd_addr,
    output reg  [DATA_WIDTH-1:0]   rd_data
);

    // Synthesis attribute: Force UltraRAM inference
    // This is CRITICAL for VU7P/VU9P/VU13P with URAM
    (* ram_style = "ultra" *)
    reg [DATA_WIDTH-1:0] mem [0:DEPTH-1];

    // Optional initialization
    generate
        if (INIT_FILE != "") begin : gen_init
            initial begin
                $readmemh(INIT_FILE, mem);
            end
        end
    endgenerate

    // Write logic
    always @(posedge clk) begin
        if (wr_en) begin
            mem[wr_addr] <= wr_data;
        end
    end

    // Read logic (registered output for timing closure)
    always @(posedge clk) begin
        if (rd_en) begin
            rd_data <= mem[rd_addr];
        end
    end

endmodule


// -----------------------------------------------------------------------------
// True Dual-Port URAM Wrapper (2 independent R/W ports)
// Note: URAM is natively TDP, so this maps efficiently
// -----------------------------------------------------------------------------
module uram_tdp_wrapper #(
    parameter DATA_WIDTH = 32,
    parameter ADDR_WIDTH = 18,
    parameter DEPTH      = 262144,
    parameter INIT_FILE  = ""
) (
    input  wire                    clk,
    input  wire                    rst,

    // Port A (Read/Write)
    input  wire                    a_en,
    input  wire                    a_we,
    input  wire [ADDR_WIDTH-1:0]   a_addr,
    input  wire [DATA_WIDTH-1:0]   a_din,
    output reg  [DATA_WIDTH-1:0]   a_dout,

    // Port B (Read/Write)
    input  wire                    b_en,
    input  wire                    b_we,
    input  wire [ADDR_WIDTH-1:0]   b_addr,
    input  wire [DATA_WIDTH-1:0]   b_din,
    output reg  [DATA_WIDTH-1:0]   b_dout
);

    (* ram_style = "ultra" *)
    reg [DATA_WIDTH-1:0] mem [0:DEPTH-1];

    generate
        if (INIT_FILE != "") begin : gen_init
            initial begin
                $readmemh(INIT_FILE, mem);
            end
        end
    endgenerate

    // Port A
    always @(posedge clk) begin
        if (a_en) begin
            if (a_we) begin
                mem[a_addr] <= a_din;
            end
            a_dout <= mem[a_addr];
        end
    end

    // Port B
    always @(posedge clk) begin
        if (b_en) begin
            if (b_we) begin
                mem[b_addr] <= b_din;
            end
            b_dout <= mem[b_addr];
        end
    end

endmodule


// -----------------------------------------------------------------------------
// CSR Weight Storage URAM - Specialized for Kitten Fabric
// Stores quantized synaptic weights in CSR format
// -----------------------------------------------------------------------------
module kitten_csr_weight_uram #(
    parameter NNZ_MAX    = 16777216,  // Max non-zero entries (16M default)
    parameter ADDRW      = 24,        // Address width for NNZ_MAX
    parameter WEIGHT_Q   = 16         // Quantized weight width (Q1.14)
) (
    input  wire              clk,
    input  wire              rst,

    // Host write port (for loading weights from host memory)
    input  wire              host_wr_en,
    input  wire [ADDRW-1:0]  host_wr_addr,
    input  wire [WEIGHT_Q-1:0] host_wr_data,

    // Projection read port (used during SNN forward pass)
    input  wire              proj_rd_en,
    input  wire [ADDRW-1:0]  proj_rd_addr,
    output wire [WEIGHT_Q-1:0] proj_rd_data
);

    // Use multiple URAM blocks for large nnz
    // VU7P URAM: 288Kb per block = 36KB = 18K x 16-bit entries
    // For 16M entries at 16-bit: ~32MB = ~900 URAM blocks

    (* ram_style = "ultra" *)
    reg [WEIGHT_Q-1:0] weight_mem [0:NNZ_MAX-1];

    reg [WEIGHT_Q-1:0] rd_data_reg;

    // Write port
    always @(posedge clk) begin
        if (host_wr_en) begin
            weight_mem[host_wr_addr] <= host_wr_data;
        end
    end

    // Read port with register for timing
    always @(posedge clk) begin
        if (proj_rd_en) begin
            rd_data_reg <= weight_mem[proj_rd_addr];
        end
    end

    assign proj_rd_data = rd_data_reg;

endmodule


// -----------------------------------------------------------------------------
// CSR Index Storage URAM - Column indices for CSR format
// -----------------------------------------------------------------------------
module kitten_csr_index_uram #(
    parameter NNZ_MAX    = 16777216,
    parameter ADDRW      = 24,
    parameter INDEX_W    = 32          // Column index width
) (
    input  wire              clk,
    input  wire              rst,

    input  wire              host_wr_en,
    input  wire [ADDRW-1:0]  host_wr_addr,
    input  wire [INDEX_W-1:0] host_wr_data,

    input  wire              proj_rd_en,
    input  wire [ADDRW-1:0]  proj_rd_addr,
    output wire [INDEX_W-1:0] proj_rd_data
);

    (* ram_style = "ultra" *)
    reg [INDEX_W-1:0] index_mem [0:NNZ_MAX-1];

    reg [INDEX_W-1:0] rd_data_reg;

    always @(posedge clk) begin
        if (host_wr_en) begin
            index_mem[host_wr_addr] <= host_wr_data;
        end
    end

    always @(posedge clk) begin
        if (proj_rd_en) begin
            rd_data_reg <= index_mem[proj_rd_addr];
        end
    end

    assign proj_rd_data = rd_data_reg;

endmodule


// -----------------------------------------------------------------------------
// CSR Indptr Storage - Row pointers (smaller, can use BRAM)
// -----------------------------------------------------------------------------
module kitten_csr_indptr_bram #(
    parameter N_ROWS     = 65536,
    parameter ADDRW      = 17,         // log2(N_ROWS+1)
    parameter PTR_W      = 32
) (
    input  wire              clk,
    input  wire              rst,

    input  wire              host_wr_en,
    input  wire [ADDRW-1:0]  host_wr_addr,
    input  wire [PTR_W-1:0]  host_wr_data,

    input  wire              proj_rd_en,
    input  wire [ADDRW-1:0]  proj_rd_addr,
    output wire [PTR_W-1:0]  proj_rd_data
);

    // Indptr is small enough for BRAM (N_ROWS+1 entries)
    (* ram_style = "block" *)
    reg [PTR_W-1:0] indptr_mem [0:N_ROWS];

    reg [PTR_W-1:0] rd_data_reg;

    always @(posedge clk) begin
        if (host_wr_en) begin
            indptr_mem[host_wr_addr] <= host_wr_data;
        end
    end

    always @(posedge clk) begin
        if (proj_rd_en) begin
            rd_data_reg <= indptr_mem[proj_rd_addr];
        end
    end

    assign proj_rd_data = rd_data_reg;

endmodule


`default_nettype wire
