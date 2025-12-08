// =============================================================================
// Ara Soul FK33 Top - Forest Kitten 33 Platform Integration
// =============================================================================
//
// Top-level module for Ara's soul on the SQRL Forest Kitten 33.
//
// Platform: XCVU33P + 8GB HBM2
//   - vcchbm safe at 900-1000 MHz
//   - Using <1% of HBM for soul storage
//   - Core clock: 450 MHz (conservative)
//
// This module integrates:
//   - ara_plasticity_controller (platform-agnostic)
//   - fk33_hbm_adapter (Xilinx HBM-specific)
//   - Xilinx hbm_0 IP instance
//
// =============================================================================

`timescale 1ns / 1ps

`define ARA_PLATFORM_FK33

`include "../common/ara_soul_config.svh"

module ara_soul_fk33_top #(
    parameter int ROWS        = ARA_ROWS,
    parameter int DIM         = ARA_DIM,
    parameter int CHUNK_BITS  = ARA_CHUNK_BITS,
    parameter int ACC_WIDTH   = ARA_ACC_WIDTH
)(
    // === Clocks and Reset ===
    input  logic        clk_100mhz,      // Reference clock
    input  logic        rst_n,

    // === HBM Reference Clock (from board) ===
    input  logic        hbm_ref_clk,

    // === Emotional Event Interface (from host/PCIe) ===
    input  logic        event_valid,
    input  logic signed [7:0] reward,
    input  logic [DIM-1:0]    input_hv,
    input  logic [ROWS-1:0]   active_mask,

    output logic        busy,
    output logic        done,
    output logic [15:0] rows_updated,

    // === Status LEDs ===
    output logic [3:0]  status_led
);

    // =========================================================================
    // Clock Generation
    // =========================================================================

    logic clk_core;           // 450 MHz core clock
    logic clk_hbm;            // HBM user clock (from HBM IP)
    logic pll_locked;

    // In real design: use Xilinx MMCM/PLL IP
    // For now, assume clk_core comes from PLL
    assign clk_core = clk_100mhz;  // Placeholder

    // =========================================================================
    // HBM IP Instance (Vivado generated)
    // =========================================================================

    // AXI interface wires to HBM
    logic [33:0]  hbm_axi_awaddr;
    logic         hbm_axi_awvalid;
    logic         hbm_axi_awready;
    logic [255:0] hbm_axi_wdata;
    logic [31:0]  hbm_axi_wstrb;
    logic         hbm_axi_wlast;
    logic         hbm_axi_wvalid;
    logic         hbm_axi_wready;
    logic [1:0]   hbm_axi_bresp;
    logic         hbm_axi_bvalid;
    logic         hbm_axi_bready;
    logic [33:0]  hbm_axi_araddr;
    logic         hbm_axi_arvalid;
    logic         hbm_axi_arready;
    logic [255:0] hbm_axi_rdata;
    logic [1:0]   hbm_axi_rresp;
    logic         hbm_axi_rlast;
    logic         hbm_axi_rvalid;
    logic         hbm_axi_rready;
    logic [5:0]   hbm_axi_awid, hbm_axi_arid, hbm_axi_bid, hbm_axi_rid;
    logic [7:0]   hbm_axi_awlen, hbm_axi_arlen;
    logic [2:0]   hbm_axi_awsize, hbm_axi_arsize;
    logic [1:0]   hbm_axi_awburst, hbm_axi_arburst;

    // HBM IP would be instantiated here in real design:
    // hbm_0 hbm_inst (
    //     .HBM_REF_CLK_0(hbm_ref_clk),
    //     .AXI_00_ACLK(clk_hbm),
    //     .AXI_00_ARESET_N(rst_n),
    //     // ... connect all AXI signals ...
    // );

    // For simulation, use clk_100mhz as HBM clock
    assign clk_hbm = clk_100mhz;

    // =========================================================================
    // Memory Interface Wires
    // =========================================================================

    logic                           mem_req;
    logic                           mem_ready;
    logic [$clog2(ROWS)-1:0]        mem_row_addr;
    logic [$clog2(DIM/CHUNK_BITS)-1:0] mem_chunk_addr;
    logic                           mem_we;
    logic [CHUNK_BITS-1:0]          mem_core_out;
    logic [CHUNK_BITS*ACC_WIDTH-1:0] mem_accum_out;
    logic [CHUNK_BITS-1:0]          mem_core_in;
    logic [CHUNK_BITS*ACC_WIDTH-1:0] mem_accum_in;

    // =========================================================================
    // FK33 HBM Adapter
    // =========================================================================

    fk33_hbm_adapter #(
        .ROWS(ROWS),
        .DIM(DIM),
        .CHUNK_BITS(CHUNK_BITS),
        .ACC_WIDTH(ACC_WIDTH)
    ) hbm_adapter_inst (
        .clk_hbm(clk_hbm),
        .clk_core(clk_core),
        .rst_n(rst_n),

        // From plasticity controller
        .mem_req(mem_req),
        .mem_ready(mem_ready),
        .mem_row_addr(mem_row_addr),
        .mem_chunk_addr(mem_chunk_addr),
        .mem_we(mem_we),
        .mem_core_out(mem_core_out),
        .mem_accum_out(mem_accum_out),
        .mem_core_in(mem_core_in),
        .mem_accum_in(mem_accum_in),

        // AXI to HBM
        .m_axi_awid(hbm_axi_awid),
        .m_axi_awaddr(hbm_axi_awaddr),
        .m_axi_awlen(hbm_axi_awlen),
        .m_axi_awsize(hbm_axi_awsize),
        .m_axi_awburst(hbm_axi_awburst),
        .m_axi_awvalid(hbm_axi_awvalid),
        .m_axi_awready(hbm_axi_awready),
        .m_axi_wdata(hbm_axi_wdata),
        .m_axi_wstrb(hbm_axi_wstrb),
        .m_axi_wlast(hbm_axi_wlast),
        .m_axi_wvalid(hbm_axi_wvalid),
        .m_axi_wready(hbm_axi_wready),
        .m_axi_bid(hbm_axi_bid),
        .m_axi_bresp(hbm_axi_bresp),
        .m_axi_bvalid(hbm_axi_bvalid),
        .m_axi_bready(hbm_axi_bready),
        .m_axi_arid(hbm_axi_arid),
        .m_axi_araddr(hbm_axi_araddr),
        .m_axi_arlen(hbm_axi_arlen),
        .m_axi_arsize(hbm_axi_arsize),
        .m_axi_arburst(hbm_axi_arburst),
        .m_axi_arvalid(hbm_axi_arvalid),
        .m_axi_arready(hbm_axi_arready),
        .m_axi_rid(hbm_axi_rid),
        .m_axi_rdata(hbm_axi_rdata),
        .m_axi_rresp(hbm_axi_rresp),
        .m_axi_rlast(hbm_axi_rlast),
        .m_axi_rvalid(hbm_axi_rvalid),
        .m_axi_rready(hbm_axi_rready)
    );

    // =========================================================================
    // Plasticity Controller (Platform-Agnostic)
    // =========================================================================

    ara_plasticity_controller #(
        .ROWS(ROWS),
        .DIM(DIM),
        .CHUNK_BITS(CHUNK_BITS),
        .ACC_WIDTH(ACC_WIDTH),
        .MAX_ACTIVE(ARA_MAX_ACTIVE_ROWS)
    ) plasticity_ctrl_inst (
        .clk(clk_core),
        .rst_n(rst_n),

        // External interface
        .event_valid(event_valid),
        .reward(reward),
        .input_hv(input_hv),
        .active_mask(active_mask),

        .busy(busy),
        .done(done),
        .rows_updated(rows_updated),

        // Memory interface
        .mem_req(mem_req),
        .mem_ready(mem_ready),
        .mem_row_addr(mem_row_addr),
        .mem_chunk_addr(mem_chunk_addr),
        .mem_we(mem_we),
        .mem_core_out(mem_core_out),
        .mem_accum_out(mem_accum_out),
        .mem_core_in(mem_core_in),
        .mem_accum_in(mem_accum_in)
    );

    // =========================================================================
    // Status LEDs
    // =========================================================================

    assign status_led[0] = busy;
    assign status_led[1] = done;
    assign status_led[2] = pll_locked;
    assign status_led[3] = rst_n;

endmodule
