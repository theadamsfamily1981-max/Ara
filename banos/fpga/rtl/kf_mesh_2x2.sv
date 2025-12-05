/**
 * Kitten Fabric (FK33) - 2x2 Mesh Proof of Concept
 * =================================================
 *
 * Bio-Affective Neuromorphic Operating System
 * Minimal mesh for simulation and verification
 *
 * Topology:
 *   +-------+-------+
 *   | (0,1) | (1,1) |
 *   +-------+-------+
 *   | (0,0) | (1,0) |
 *   +-------+-------+
 *
 * Y=0 at bottom, X=0 at left (standard coordinates)
 * XY routing: route X first, then Y
 *
 * Resources (4 tiles):
 * - 4 * 256 = 1024 neurons
 * - 4 * 4096 = 16384 synapses
 * - 4 * 5-port routers
 *
 * External interface:
 * - Spike injection port (for testbench stimuli)
 * - Spike capture port (for monitoring)
 * - Configuration bus (broadcast to all tiles)
 */

module kf_mesh_2x2
    import kf_pkg::*;
(
    input  logic        clk,
    input  logic        rst_n,

    // =========================================================================
    // External Spike Injection (to tile 0,0)
    // =========================================================================
    input  logic        ext_spike_in_valid,
    output logic        ext_spike_in_ready,
    input  spike_flit_t ext_spike_in_flit,

    // =========================================================================
    // External Spike Capture (from tile 1,1 - diagonal corner)
    // =========================================================================
    output logic        ext_spike_out_valid,
    input  logic        ext_spike_out_ready,
    output spike_flit_t ext_spike_out_flit,

    // =========================================================================
    // Configuration Interface (broadcast to all tiles)
    // =========================================================================
    input  logic                          cfg_we,
    input  logic [1:0]                    cfg_tile_sel,  // Which tile (0-3)
    input  logic [1:0]                    cfg_sel,       // synapse/index/vmem
    input  logic [KF_SYNAPSE_ID_BITS-1:0] cfg_addr,
    input  logic [31:0]                   cfg_wdata,

    // =========================================================================
    // Debug/Status
    // =========================================================================
    output logic [7:0]  tile_active_count [4],
    output logic [3:0]  tile_busy
);

    // =========================================================================
    // INTER-TILE WIRING
    // =========================================================================

    // Horizontal links (East-West)
    // tile_00 <-> tile_10
    logic        link_00_10_e_valid, link_00_10_e_ready;
    spike_flit_t link_00_10_e_flit;
    logic        link_10_00_w_valid, link_10_00_w_ready;
    spike_flit_t link_10_00_w_flit;

    // tile_01 <-> tile_11
    logic        link_01_11_e_valid, link_01_11_e_ready;
    spike_flit_t link_01_11_e_flit;
    logic        link_11_01_w_valid, link_11_01_w_ready;
    spike_flit_t link_11_01_w_flit;

    // Vertical links (North-South)
    // tile_00 <-> tile_01
    logic        link_00_01_n_valid, link_00_01_n_ready;
    spike_flit_t link_00_01_n_flit;
    logic        link_01_00_s_valid, link_01_00_s_ready;
    spike_flit_t link_01_00_s_flit;

    // tile_10 <-> tile_11
    logic        link_10_11_n_valid, link_10_11_n_ready;
    spike_flit_t link_10_11_n_flit;
    logic        link_11_10_s_valid, link_11_10_s_ready;
    spike_flit_t link_11_10_s_flit;

    // =========================================================================
    // CONFIGURATION DEMUX
    // =========================================================================

    logic cfg_we_tile [4];

    always_comb begin
        for (int i = 0; i < 4; i++) begin
            cfg_we_tile[i] = cfg_we && (cfg_tile_sel == i[1:0]);
        end
    end

    // =========================================================================
    // TILE (0,0) - Bottom-Left
    // =========================================================================

    // Edge ports tied off (no south, no west neighbor)
    // External injection comes in via west port

    kitten_fabric_tile #(
        .TILE_X (0),
        .TILE_Y (0)
    ) u_tile_00 (
        .clk                    (clk),
        .rst_n                  (rst_n),

        // North -> tile_01
        .noc_north_in_valid     (link_01_00_s_valid),
        .noc_north_in_ready     (link_01_00_s_ready),
        .noc_north_in_flit      (link_01_00_s_flit),
        .noc_north_out_valid    (link_00_01_n_valid),
        .noc_north_out_ready    (link_00_01_n_ready),
        .noc_north_out_flit     (link_00_01_n_flit),

        // South -> edge (tied off)
        .noc_south_in_valid     (1'b0),
        .noc_south_in_ready     (),
        .noc_south_in_flit      ('0),
        .noc_south_out_valid    (),
        .noc_south_out_ready    (1'b1),
        .noc_south_out_flit     (),

        // East -> tile_10
        .noc_east_in_valid      (link_10_00_w_valid),
        .noc_east_in_ready      (link_10_00_w_ready),
        .noc_east_in_flit       (link_10_00_w_flit),
        .noc_east_out_valid     (link_00_10_e_valid),
        .noc_east_out_ready     (link_00_10_e_ready),
        .noc_east_out_flit      (link_00_10_e_flit),

        // West -> external injection
        .noc_west_in_valid      (ext_spike_in_valid),
        .noc_west_in_ready      (ext_spike_in_ready),
        .noc_west_in_flit       (ext_spike_in_flit),
        .noc_west_out_valid     (),
        .noc_west_out_ready     (1'b1),
        .noc_west_out_flit      (),

        // Config
        .cfg_we                 (cfg_we_tile[0]),
        .cfg_sel                (cfg_sel),
        .cfg_addr               (cfg_addr),
        .cfg_wdata              (cfg_wdata),

        // Debug
        .active_neuron_count    (tile_active_count[0]),
        .core_busy              (tile_busy[0])
    );

    // =========================================================================
    // TILE (1,0) - Bottom-Right
    // =========================================================================

    kitten_fabric_tile #(
        .TILE_X (1),
        .TILE_Y (0)
    ) u_tile_10 (
        .clk                    (clk),
        .rst_n                  (rst_n),

        // North -> tile_11
        .noc_north_in_valid     (link_11_10_s_valid),
        .noc_north_in_ready     (link_11_10_s_ready),
        .noc_north_in_flit      (link_11_10_s_flit),
        .noc_north_out_valid    (link_10_11_n_valid),
        .noc_north_out_ready    (link_10_11_n_ready),
        .noc_north_out_flit     (link_10_11_n_flit),

        // South -> edge (tied off)
        .noc_south_in_valid     (1'b0),
        .noc_south_in_ready     (),
        .noc_south_in_flit      ('0),
        .noc_south_out_valid    (),
        .noc_south_out_ready    (1'b1),
        .noc_south_out_flit     (),

        // East -> edge (tied off)
        .noc_east_in_valid      (1'b0),
        .noc_east_in_ready      (),
        .noc_east_in_flit       ('0),
        .noc_east_out_valid     (),
        .noc_east_out_ready     (1'b1),
        .noc_east_out_flit      (),

        // West -> tile_00
        .noc_west_in_valid      (link_00_10_e_valid),
        .noc_west_in_ready      (link_00_10_e_ready),
        .noc_west_in_flit       (link_00_10_e_flit),
        .noc_west_out_valid     (link_10_00_w_valid),
        .noc_west_out_ready     (link_10_00_w_ready),
        .noc_west_out_flit      (link_10_00_w_flit),

        // Config
        .cfg_we                 (cfg_we_tile[1]),
        .cfg_sel                (cfg_sel),
        .cfg_addr               (cfg_addr),
        .cfg_wdata              (cfg_wdata),

        // Debug
        .active_neuron_count    (tile_active_count[1]),
        .core_busy              (tile_busy[1])
    );

    // =========================================================================
    // TILE (0,1) - Top-Left
    // =========================================================================

    kitten_fabric_tile #(
        .TILE_X (0),
        .TILE_Y (1)
    ) u_tile_01 (
        .clk                    (clk),
        .rst_n                  (rst_n),

        // North -> edge (tied off)
        .noc_north_in_valid     (1'b0),
        .noc_north_in_ready     (),
        .noc_north_in_flit      ('0),
        .noc_north_out_valid    (),
        .noc_north_out_ready    (1'b1),
        .noc_north_out_flit     (),

        // South -> tile_00
        .noc_south_in_valid     (link_00_01_n_valid),
        .noc_south_in_ready     (link_00_01_n_ready),
        .noc_south_in_flit      (link_00_01_n_flit),
        .noc_south_out_valid    (link_01_00_s_valid),
        .noc_south_out_ready    (link_01_00_s_ready),
        .noc_south_out_flit     (link_01_00_s_flit),

        // East -> tile_11
        .noc_east_in_valid      (link_11_01_w_valid),
        .noc_east_in_ready      (link_11_01_w_ready),
        .noc_east_in_flit       (link_11_01_w_flit),
        .noc_east_out_valid     (link_01_11_e_valid),
        .noc_east_out_ready     (link_01_11_e_ready),
        .noc_east_out_flit      (link_01_11_e_flit),

        // West -> edge (tied off)
        .noc_west_in_valid      (1'b0),
        .noc_west_in_ready      (),
        .noc_west_in_flit       ('0),
        .noc_west_out_valid     (),
        .noc_west_out_ready     (1'b1),
        .noc_west_out_flit      (),

        // Config
        .cfg_we                 (cfg_we_tile[2]),
        .cfg_sel                (cfg_sel),
        .cfg_addr               (cfg_addr),
        .cfg_wdata              (cfg_wdata),

        // Debug
        .active_neuron_count    (tile_active_count[2]),
        .core_busy              (tile_busy[2])
    );

    // =========================================================================
    // TILE (1,1) - Top-Right
    // =========================================================================

    // External capture comes out via east port

    kitten_fabric_tile #(
        .TILE_X (1),
        .TILE_Y (1)
    ) u_tile_11 (
        .clk                    (clk),
        .rst_n                  (rst_n),

        // North -> edge (tied off)
        .noc_north_in_valid     (1'b0),
        .noc_north_in_ready     (),
        .noc_north_in_flit      ('0),
        .noc_north_out_valid    (),
        .noc_north_out_ready    (1'b1),
        .noc_north_out_flit     (),

        // South -> tile_10
        .noc_south_in_valid     (link_10_11_n_valid),
        .noc_south_in_ready     (link_10_11_n_ready),
        .noc_south_in_flit      (link_10_11_n_flit),
        .noc_south_out_valid    (link_11_10_s_valid),
        .noc_south_out_ready    (link_11_10_s_ready),
        .noc_south_out_flit     (link_11_10_s_flit),

        // East -> external capture
        .noc_east_in_valid      (1'b0),
        .noc_east_in_ready      (),
        .noc_east_in_flit       ('0),
        .noc_east_out_valid     (ext_spike_out_valid),
        .noc_east_out_ready     (ext_spike_out_ready),
        .noc_east_out_flit      (ext_spike_out_flit),

        // West -> tile_01
        .noc_west_in_valid      (link_01_11_e_valid),
        .noc_west_in_ready      (link_01_11_e_ready),
        .noc_west_in_flit       (link_01_11_e_flit),
        .noc_west_out_valid     (link_11_01_w_valid),
        .noc_west_out_ready     (link_11_01_w_ready),
        .noc_west_out_flit      (link_11_01_w_flit),

        // Config
        .cfg_we                 (cfg_we_tile[3]),
        .cfg_sel                (cfg_sel),
        .cfg_addr               (cfg_addr),
        .cfg_wdata              (cfg_wdata),

        // Debug
        .active_neuron_count    (tile_active_count[3]),
        .core_busy              (tile_busy[3])
    );

endmodule : kf_mesh_2x2
