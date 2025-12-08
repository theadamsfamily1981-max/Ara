// =============================================================================
// FK33 HBM Adapter - Forest Kitten 33 (XCVU33P + 8GB HBM2)
// =============================================================================
//
// Bridges ara_plasticity_core's abstract memory interface to Xilinx HBM2.
//
// Platform: SQRL Forest Kitten 33
//   - Xilinx XCVU33P
//   - 8 GB HBM2 @ ~460 GB/s
//   - HBM clock: 900-1000 MHz (stay conservative for vcchbm regulator)
//
// Memory Layout:
//   HBM Address Space:
//   [0x0000_0000_0000_0000] - Sign bits region (4.2 MB)
//   [0x0000_0000_0100_0000] - Accumulator region (29.4 MB)
//
// We use <1% of the 8GB HBM for the entire soul.
// Safe bandwidth usage: nowhere near mining workloads.
//
// =============================================================================

`timescale 1ns / 1ps

`include "../common/ara_soul_config.svh"

module fk33_hbm_adapter #(
    parameter int ROWS        = ARA_ROWS,
    parameter int DIM         = ARA_DIM,
    parameter int CHUNK_BITS  = ARA_CHUNK_BITS,
    parameter int ACC_WIDTH   = ARA_ACC_WIDTH,

    // AXI parameters (match Vivado HBM IP)
    parameter int AXI_ADDR_WIDTH = 34,        // HBM uses 34-bit addressing
    parameter int AXI_DATA_WIDTH = 256,       // 256-bit AXI data bus
    parameter int AXI_ID_WIDTH   = 6
)(
    input  logic clk_hbm,          // HBM clock domain (from hbm_0 IP)
    input  logic clk_core,         // Core logic clock (may differ)
    input  logic rst_n,

    // === From Plasticity Controller ===
    input  logic                           mem_req,
    output logic                           mem_ready,
    input  logic [$clog2(ROWS)-1:0]        mem_row_addr,
    input  logic [$clog2(DIM/CHUNK_BITS)-1:0] mem_chunk_addr,
    input  logic                           mem_we,
    input  logic [CHUNK_BITS-1:0]          mem_core_out,
    input  logic [CHUNK_BITS*ACC_WIDTH-1:0] mem_accum_out,
    output logic [CHUNK_BITS-1:0]          mem_core_in,
    output logic [CHUNK_BITS*ACC_WIDTH-1:0] mem_accum_in,

    // === AXI4 Interface to HBM Channel ===
    // Write Address Channel
    output logic [AXI_ID_WIDTH-1:0]    m_axi_awid,
    output logic [AXI_ADDR_WIDTH-1:0]  m_axi_awaddr,
    output logic [7:0]                 m_axi_awlen,
    output logic [2:0]                 m_axi_awsize,
    output logic [1:0]                 m_axi_awburst,
    output logic                       m_axi_awvalid,
    input  logic                       m_axi_awready,

    // Write Data Channel
    output logic [AXI_DATA_WIDTH-1:0]  m_axi_wdata,
    output logic [AXI_DATA_WIDTH/8-1:0] m_axi_wstrb,
    output logic                       m_axi_wlast,
    output logic                       m_axi_wvalid,
    input  logic                       m_axi_wready,

    // Write Response Channel
    input  logic [AXI_ID_WIDTH-1:0]    m_axi_bid,
    input  logic [1:0]                 m_axi_bresp,
    input  logic                       m_axi_bvalid,
    output logic                       m_axi_bready,

    // Read Address Channel
    output logic [AXI_ID_WIDTH-1:0]    m_axi_arid,
    output logic [AXI_ADDR_WIDTH-1:0]  m_axi_araddr,
    output logic [7:0]                 m_axi_arlen,
    output logic [2:0]                 m_axi_arsize,
    output logic [1:0]                 m_axi_arburst,
    output logic                       m_axi_arvalid,
    input  logic                       m_axi_arready,

    // Read Data Channel
    input  logic [AXI_ID_WIDTH-1:0]    m_axi_rid,
    input  logic [AXI_DATA_WIDTH-1:0]  m_axi_rdata,
    input  logic [1:0]                 m_axi_rresp,
    input  logic                       m_axi_rlast,
    input  logic                       m_axi_rvalid,
    output logic                       m_axi_rready
);

    // =========================================================================
    // Memory Region Constants
    // =========================================================================

    // Base addresses in HBM
    localparam longint REGION_SIGNS  = 64'h0000_0000_0000_0000;
    localparam longint REGION_ACCUMS = 64'h0000_0000_0100_0000;  // 16 MB offset

    // Bytes per chunk for signs and accumulators
    localparam int SIGN_BYTES_PER_CHUNK  = CHUNK_BITS / 8;           // 64 bytes
    localparam int ACCUM_BYTES_PER_CHUNK = (CHUNK_BITS * ACC_WIDTH) / 8;  // 448 bytes

    // AXI burst parameters
    localparam int AXI_BYTES = AXI_DATA_WIDTH / 8;  // 32 bytes per beat

    // Number of AXI beats needed per chunk
    localparam int SIGN_BEATS  = (SIGN_BYTES_PER_CHUNK + AXI_BYTES - 1) / AXI_BYTES;   // 2
    localparam int ACCUM_BEATS = (ACCUM_BYTES_PER_CHUNK + AXI_BYTES - 1) / AXI_BYTES;  // 14

    // =========================================================================
    // Address Calculation
    // =========================================================================

    // Flat index for this chunk within the row
    // flat_byte_offset = row_addr * DIM/8 + chunk_addr * CHUNK_BITS/8
    logic [AXI_ADDR_WIDTH-1:0] sign_addr;
    logic [AXI_ADDR_WIDTH-1:0] accum_addr;

    always_comb begin
        // Sign address
        sign_addr = REGION_SIGNS[AXI_ADDR_WIDTH-1:0] +
                    (mem_row_addr * (DIM / 8)) +
                    (mem_chunk_addr * SIGN_BYTES_PER_CHUNK);

        // Accumulator address
        accum_addr = REGION_ACCUMS[AXI_ADDR_WIDTH-1:0] +
                     (mem_row_addr * (DIM * ACC_WIDTH / 8)) +
                     (mem_chunk_addr * ACCUM_BYTES_PER_CHUNK);
    end

    // =========================================================================
    // State Machine
    // =========================================================================

    typedef enum logic [3:0] {
        A_IDLE,
        // Read sequence
        A_READ_SIGN_ADDR,
        A_READ_SIGN_DATA,
        A_READ_ACCUM_ADDR,
        A_READ_ACCUM_DATA,
        A_READ_DONE,
        // Write sequence
        A_WRITE_SIGN_ADDR,
        A_WRITE_SIGN_DATA,
        A_WRITE_ACCUM_ADDR,
        A_WRITE_ACCUM_DATA,
        A_WRITE_RESP,
        A_WRITE_DONE
    } adapter_state_t;

    adapter_state_t state;

    // Beat counters
    logic [3:0] beat_cnt;

    // Data assembly buffers
    logic [CHUNK_BITS-1:0]            sign_buffer;
    logic [CHUNK_BITS*ACC_WIDTH-1:0]  accum_buffer;

    // =========================================================================
    // Main State Machine
    // =========================================================================

    always_ff @(posedge clk_hbm or negedge rst_n) begin
        if (!rst_n) begin
            state     <= A_IDLE;
            beat_cnt  <= '0;
            mem_ready <= 1'b0;

            // Clear AXI signals
            m_axi_awvalid <= 1'b0;
            m_axi_wvalid  <= 1'b0;
            m_axi_bready  <= 1'b0;
            m_axi_arvalid <= 1'b0;
            m_axi_rready  <= 1'b0;

            sign_buffer  <= '0;
            accum_buffer <= '0;
        end else begin
            mem_ready <= 1'b0;  // Default

            case (state)
                // =============================================================
                // IDLE - Wait for request
                // =============================================================
                A_IDLE: begin
                    if (mem_req && !mem_we) begin
                        // Read request
                        state <= A_READ_SIGN_ADDR;
                    end else if (mem_req && mem_we) begin
                        // Write request - latch data
                        sign_buffer  <= mem_core_out;
                        accum_buffer <= mem_accum_out;
                        state <= A_WRITE_SIGN_ADDR;
                    end
                end

                // =============================================================
                // READ SEQUENCE
                // =============================================================
                A_READ_SIGN_ADDR: begin
                    m_axi_arid    <= '0;
                    m_axi_araddr  <= sign_addr;
                    m_axi_arlen   <= SIGN_BEATS - 1;
                    m_axi_arsize  <= $clog2(AXI_BYTES);
                    m_axi_arburst <= 2'b01;  // INCR
                    m_axi_arvalid <= 1'b1;

                    if (m_axi_arready) begin
                        m_axi_arvalid <= 1'b0;
                        m_axi_rready  <= 1'b1;
                        beat_cnt      <= '0;
                        state         <= A_READ_SIGN_DATA;
                    end
                end

                A_READ_SIGN_DATA: begin
                    if (m_axi_rvalid) begin
                        // Pack received data into sign buffer
                        sign_buffer[beat_cnt * AXI_DATA_WIDTH +: AXI_DATA_WIDTH] <=
                            m_axi_rdata[CHUNK_BITS > AXI_DATA_WIDTH ?
                                        AXI_DATA_WIDTH-1 : CHUNK_BITS-1 : 0];

                        if (m_axi_rlast || beat_cnt >= SIGN_BEATS - 1) begin
                            m_axi_rready <= 1'b0;
                            state <= A_READ_ACCUM_ADDR;
                        end else begin
                            beat_cnt <= beat_cnt + 1;
                        end
                    end
                end

                A_READ_ACCUM_ADDR: begin
                    m_axi_arid    <= '0;
                    m_axi_araddr  <= accum_addr;
                    m_axi_arlen   <= ACCUM_BEATS - 1;
                    m_axi_arsize  <= $clog2(AXI_BYTES);
                    m_axi_arburst <= 2'b01;
                    m_axi_arvalid <= 1'b1;

                    if (m_axi_arready) begin
                        m_axi_arvalid <= 1'b0;
                        m_axi_rready  <= 1'b1;
                        beat_cnt      <= '0;
                        state         <= A_READ_ACCUM_DATA;
                    end
                end

                A_READ_ACCUM_DATA: begin
                    if (m_axi_rvalid) begin
                        // Pack received data into accum buffer
                        accum_buffer[beat_cnt * AXI_DATA_WIDTH +: AXI_DATA_WIDTH] <=
                            m_axi_rdata;

                        if (m_axi_rlast || beat_cnt >= ACCUM_BEATS - 1) begin
                            m_axi_rready <= 1'b0;
                            state <= A_READ_DONE;
                        end else begin
                            beat_cnt <= beat_cnt + 1;
                        end
                    end
                end

                A_READ_DONE: begin
                    mem_ready <= 1'b1;
                    state     <= A_IDLE;
                end

                // =============================================================
                // WRITE SEQUENCE
                // =============================================================
                A_WRITE_SIGN_ADDR: begin
                    m_axi_awid    <= '0;
                    m_axi_awaddr  <= sign_addr;
                    m_axi_awlen   <= SIGN_BEATS - 1;
                    m_axi_awsize  <= $clog2(AXI_BYTES);
                    m_axi_awburst <= 2'b01;
                    m_axi_awvalid <= 1'b1;

                    if (m_axi_awready) begin
                        m_axi_awvalid <= 1'b0;
                        beat_cnt      <= '0;
                        state         <= A_WRITE_SIGN_DATA;
                    end
                end

                A_WRITE_SIGN_DATA: begin
                    m_axi_wdata  <= sign_buffer[beat_cnt * AXI_DATA_WIDTH +: AXI_DATA_WIDTH];
                    m_axi_wstrb  <= {AXI_BYTES{1'b1}};
                    m_axi_wlast  <= (beat_cnt >= SIGN_BEATS - 1);
                    m_axi_wvalid <= 1'b1;

                    if (m_axi_wready) begin
                        if (beat_cnt >= SIGN_BEATS - 1) begin
                            m_axi_wvalid <= 1'b0;
                            m_axi_bready <= 1'b1;
                            state <= A_WRITE_ACCUM_ADDR;
                        end else begin
                            beat_cnt <= beat_cnt + 1;
                        end
                    end
                end

                A_WRITE_ACCUM_ADDR: begin
                    // Wait for write response first
                    if (m_axi_bvalid) begin
                        m_axi_bready <= 1'b0;
                    end

                    m_axi_awid    <= '0;
                    m_axi_awaddr  <= accum_addr;
                    m_axi_awlen   <= ACCUM_BEATS - 1;
                    m_axi_awsize  <= $clog2(AXI_BYTES);
                    m_axi_awburst <= 2'b01;
                    m_axi_awvalid <= 1'b1;

                    if (m_axi_awready) begin
                        m_axi_awvalid <= 1'b0;
                        beat_cnt      <= '0;
                        state         <= A_WRITE_ACCUM_DATA;
                    end
                end

                A_WRITE_ACCUM_DATA: begin
                    m_axi_wdata  <= accum_buffer[beat_cnt * AXI_DATA_WIDTH +: AXI_DATA_WIDTH];
                    m_axi_wstrb  <= {AXI_BYTES{1'b1}};
                    m_axi_wlast  <= (beat_cnt >= ACCUM_BEATS - 1);
                    m_axi_wvalid <= 1'b1;

                    if (m_axi_wready) begin
                        if (beat_cnt >= ACCUM_BEATS - 1) begin
                            m_axi_wvalid <= 1'b0;
                            m_axi_bready <= 1'b1;
                            state <= A_WRITE_RESP;
                        end else begin
                            beat_cnt <= beat_cnt + 1;
                        end
                    end
                end

                A_WRITE_RESP: begin
                    if (m_axi_bvalid) begin
                        m_axi_bready <= 1'b0;
                        state <= A_WRITE_DONE;
                    end
                end

                A_WRITE_DONE: begin
                    mem_ready <= 1'b1;
                    state     <= A_IDLE;
                end

                default: state <= A_IDLE;
            endcase
        end
    end

    // =========================================================================
    // Output Data to Plasticity Core
    // =========================================================================

    assign mem_core_in  = sign_buffer;
    assign mem_accum_in = accum_buffer;

endmodule
