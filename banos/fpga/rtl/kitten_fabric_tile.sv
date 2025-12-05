/**
 * Kitten Fabric (FK33) - Complete Tile
 * =====================================
 *
 * Bio-Affective Neuromorphic Operating System
 * Single tile combining SNN core + NoC router
 *
 * Architecture:
 * - kf_snn_core: Event-driven LIF neuron array
 * - kf_noc_router: XY dimension-order routing
 * - Spike flit conversion between core and router
 *
 * Each tile is a self-contained processing element that:
 * - Receives spikes from neighbors via NoC
 * - Processes them through the SNN core
 * - Emits output spikes back to the NoC
 */

module kitten_fabric_tile
    import kf_pkg::*;
#(
    parameter int TILE_X = 0,
    parameter int TILE_Y = 0
)(
    input  logic        clk,
    input  logic        rst_n,

    // =========================================================================
    // NoC Ports (to/from neighboring tiles)
    // =========================================================================

    // North
    input  logic        noc_north_in_valid,
    output logic        noc_north_in_ready,
    input  spike_flit_t noc_north_in_flit,
    output logic        noc_north_out_valid,
    input  logic        noc_north_out_ready,
    output spike_flit_t noc_north_out_flit,

    // South
    input  logic        noc_south_in_valid,
    output logic        noc_south_in_ready,
    input  spike_flit_t noc_south_in_flit,
    output logic        noc_south_out_valid,
    input  logic        noc_south_out_ready,
    output spike_flit_t noc_south_out_flit,

    // East
    input  logic        noc_east_in_valid,
    output logic        noc_east_in_ready,
    input  spike_flit_t noc_east_in_flit,
    output logic        noc_east_out_valid,
    input  logic        noc_east_out_ready,
    output spike_flit_t noc_east_out_flit,

    // West
    input  logic        noc_west_in_valid,
    output logic        noc_west_in_ready,
    input  spike_flit_t noc_west_in_flit,
    output logic        noc_west_out_valid,
    input  logic        noc_west_out_ready,
    output spike_flit_t noc_west_out_flit,

    // =========================================================================
    // Configuration Interface (directly to SNN core)
    // =========================================================================
    input  logic                          cfg_we,
    input  logic [1:0]                    cfg_sel,
    input  logic [KF_SYNAPSE_ID_BITS-1:0] cfg_addr,
    input  logic [31:0]                   cfg_wdata,

    // =========================================================================
    // Debug/Status
    // =========================================================================
    output logic [7:0]  active_neuron_count,
    output logic        core_busy
);

    // =========================================================================
    // INTERNAL SIGNALS
    // =========================================================================

    // Router local port <-> SNN core connection
    logic        router_to_core_valid;
    logic        router_to_core_ready;
    spike_flit_t router_to_core_flit;

    logic        core_to_router_valid;
    logic        core_to_router_ready;
    spike_flit_t core_to_router_flit;

    // SNN core spike interface (neuron-level)
    logic                          spike_in_valid;
    logic                          spike_in_ready;
    logic [KF_NEURON_ID_BITS-1:0]  spike_in_pre_id;
    logic [7:0]                    spike_in_payload;

    logic                          spike_out_valid;
    logic                          spike_out_ready;
    logic [KF_NEURON_ID_BITS-1:0]  spike_out_post_id;
    logic [7:0]                    spike_out_payload;

    // =========================================================================
    // SPIKE FLIT <-> CORE CONVERSION
    // =========================================================================

    // Router delivers flit to core: extract neuron ID and payload
    assign spike_in_valid   = router_to_core_valid;
    assign router_to_core_ready = spike_in_ready;
    assign spike_in_pre_id  = router_to_core_flit.neuron_id;
    assign spike_in_payload = router_to_core_flit.payload;

    // Core emits spike: need destination from routing table
    // For v1, output spikes broadcast to all tiles (payload carries dest info)
    // In production, we'd have a routing table per neuron

    // Simple v1 approach: output spikes use payload as destination tile
    // payload[7:4] = dest_x, payload[3:0] = dest_y (4x4 mesh max)
    assign core_to_router_valid = spike_out_valid;
    assign spike_out_ready = core_to_router_ready;

    always_comb begin
        core_to_router_flit.dest_x    = {4'b0, spike_out_payload[7:4]};
        core_to_router_flit.dest_y    = {4'b0, spike_out_payload[3:0]};
        core_to_router_flit.neuron_id = spike_out_post_id;
        core_to_router_flit.payload   = spike_out_payload;
    end

    // =========================================================================
    // SNN CORE INSTANCE
    // =========================================================================

    kf_snn_core #(
        .N_NEURONS  (KF_NEURONS_PER_TILE),
        .N_SYNAPSES (KF_SYNAPSES_PER_TILE)
    ) u_snn_core (
        .clk                (clk),
        .rst_n              (rst_n),

        // Spike input
        .spike_in_valid     (spike_in_valid),
        .spike_in_ready     (spike_in_ready),
        .spike_in_pre_id    (spike_in_pre_id),
        .spike_in_payload   (spike_in_payload),

        // Spike output
        .spike_out_valid    (spike_out_valid),
        .spike_out_ready    (spike_out_ready),
        .spike_out_post_id  (spike_out_post_id),
        .spike_out_payload  (spike_out_payload),

        // Configuration
        .cfg_we             (cfg_we),
        .cfg_sel            (cfg_sel),
        .cfg_addr           (cfg_addr),
        .cfg_wdata          (cfg_wdata),

        // Debug
        .active_neuron_count(active_neuron_count),
        .core_busy          (core_busy)
    );

    // =========================================================================
    // NOC ROUTER INSTANCE
    // =========================================================================

    kf_noc_router #(
        .TILE_X (TILE_X),
        .TILE_Y (TILE_Y)
    ) u_noc_router (
        .clk            (clk),
        .rst_n          (rst_n),

        // Local port (to/from SNN core)
        .in_local_valid (core_to_router_valid),
        .in_local_ready (core_to_router_ready),
        .in_local_flit  (core_to_router_flit),
        .out_local_valid(router_to_core_valid),
        .out_local_ready(router_to_core_ready),
        .out_local_flit (router_to_core_flit),

        // North
        .in_north_valid (noc_north_in_valid),
        .in_north_ready (noc_north_in_ready),
        .in_north_flit  (noc_north_in_flit),
        .out_north_valid(noc_north_out_valid),
        .out_north_ready(noc_north_out_ready),
        .out_north_flit (noc_north_out_flit),

        // South
        .in_south_valid (noc_south_in_valid),
        .in_south_ready (noc_south_in_ready),
        .in_south_flit  (noc_south_in_flit),
        .out_south_valid(noc_south_out_valid),
        .out_south_ready(noc_south_out_ready),
        .out_south_flit (noc_south_out_flit),

        // East
        .in_east_valid  (noc_east_in_valid),
        .in_east_ready  (noc_east_in_ready),
        .in_east_flit   (noc_east_in_flit),
        .out_east_valid (noc_east_out_valid),
        .out_east_ready (noc_east_out_ready),
        .out_east_flit  (noc_east_out_flit),

        // West
        .in_west_valid  (noc_west_in_valid),
        .in_west_ready  (noc_west_in_ready),
        .in_west_flit   (noc_west_in_flit),
        .out_west_valid (noc_west_out_valid),
        .out_west_ready (noc_west_out_ready),
        .out_west_flit  (noc_west_out_flit)
    );

endmodule : kitten_fabric_tile
