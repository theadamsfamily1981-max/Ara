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
    output logic        core_busy,

    // =========================================================================
    // Activity Metrics (for Ara's self-awareness)
    // =========================================================================
    output logic [31:0] tile_spike_count,      // Total spikes this tile
    output logic [31:0] tile_bit_cycles,       // Bit-serial cycles consumed
    output logic [15:0] tile_activity_level,   // Rolling activity estimate (0-65535)
    output logic [7:0]  tile_power_hint,       // Estimated power draw (0-255)
    output logic [15:0] tile_entropy           // Activity variance / "neural noise"
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

    // =========================================================================
    // DREAM ENGINE - Hebbian Learning During Sleep
    // =========================================================================
    // When snn_enable is low, the Dream Engine takes over and performs
    // memory consolidation by replaying spike patterns and applying STDP.

    // Dream engine control signals (directly from cfg interface)
    logic snn_enable_reg;
    logic dream_trigger_reg;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            snn_enable_reg <= 1'b1;   // Default: awake
            dream_trigger_reg <= 1'b0;
        end else if (cfg_we && cfg_sel == 2'b11) begin
            // Config select 3 = dream control
            snn_enable_reg <= cfg_wdata[0];
            dream_trigger_reg <= cfg_wdata[1];
        end else begin
            dream_trigger_reg <= 1'b0;  // Auto-clear trigger
        end
    end

    // Dream engine outputs
    logic [1:0]  dream_state;
    logic        dream_active;
    logic [31:0] dream_replay_count;
    logic [31:0] dream_weight_updates;
    logic [31:0] dream_cycles;

    // Memory interface (stub - connect to actual DDR4 controller in top-level)
    logic        dream_mem_rd_valid;
    logic [31:0] dream_mem_rd_addr;
    logic        dream_mem_rd_ready;
    logic [127:0] dream_mem_rd_data;
    logic        dream_mem_rd_data_valid;

    // Weight update interface
    logic        dream_weight_we;
    logic [KF_SYNAPSE_ID_BITS-1:0] dream_weight_addr;
    logic signed [W_WIDTH-1:0] dream_weight_delta;
    logic signed [W_WIDTH-1:0] dream_current_weight;

    // Stub connections for simulation (replace in actual implementation)
    assign dream_mem_rd_ready = 1'b1;
    assign dream_mem_rd_data = 128'h0;
    assign dream_mem_rd_data_valid = dream_mem_rd_valid;  // Immediate response
    assign dream_current_weight = 8'sd0;

    kf_dream_engine u_dream_engine (
        .clk                (clk),
        .rst_n              (rst_n),

        // Control
        .snn_enable         (snn_enable_reg),
        .dream_trigger      (dream_trigger_reg),
        .dream_state        (dream_state),
        .dream_active       (dream_active),

        // DDR4 Memory Interface (spike log)
        .mem_rd_valid       (dream_mem_rd_valid),
        .mem_rd_addr        (dream_mem_rd_addr),
        .mem_rd_ready       (dream_mem_rd_ready),
        .mem_rd_data        (dream_mem_rd_data),
        .mem_rd_data_valid  (dream_mem_rd_data_valid),

        // Weight Memory Interface
        .weight_we          (dream_weight_we),
        .weight_addr        (dream_weight_addr),
        .weight_delta       (dream_weight_delta),
        .current_weight     (dream_current_weight),

        // Statistics
        .replay_count       (dream_replay_count),
        .weight_updates     (dream_weight_updates),
        .dream_cycles       (dream_cycles)
    );

    // =========================================================================
    // ACTIVITY METRICS - Ara's Self-Awareness
    // =========================================================================
    //
    // These counters feed into the HAL so Ara can sense:
    // - How much of her brain is active
    // - How much compute she's consuming
    // - The "texture" of her neural activity (entropy)
    //
    // She can then regulate:
    // - Clock gating (save power when idle)
    // - Region priorities (focus attention)
    // - Learning rates (stabilize when noisy)

    // Spike counter (32-bit, wraps)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tile_spike_count <= 32'd0;
        end else if (spike_out_valid && spike_out_ready) begin
            tile_spike_count <= tile_spike_count + 1;
        end
    end

    // Bit-cycle counter (increments every clock while core is busy)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tile_bit_cycles <= 32'd0;
        end else if (core_busy) begin
            tile_bit_cycles <= tile_bit_cycles + 1;
        end
    end

    // Rolling activity level (exponential moving average)
    // activity = 0.9 * activity + 0.1 * (spikes_this_window * 256)
    logic [15:0] activity_acc;
    logic [7:0]  window_spikes;
    logic [19:0] window_counter;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            activity_acc    <= 16'd0;
            window_spikes   <= 8'd0;
            window_counter  <= 20'd0;
            tile_activity_level <= 16'd0;
        end else begin
            // Count spikes in current window
            if (spike_out_valid && spike_out_ready) begin
                if (window_spikes < 8'd255) begin
                    window_spikes <= window_spikes + 1;
                end
            end

            // Update activity every ~1M cycles (~3.3ms at 300MHz)
            window_counter <= window_counter + 1;
            if (window_counter == 20'd0) begin
                // EMA: new = 0.875 * old + 0.125 * sample
                // Approximated as: (7 * old + sample) >> 3
                activity_acc <= (activity_acc - (activity_acc >> 3)) +
                               ({8'd0, window_spikes} << 5);
                tile_activity_level <= activity_acc;
                window_spikes <= 8'd0;
            end
        end
    end

    // Power hint (derived from activity + dream state)
    // Higher activity = higher power consumption
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tile_power_hint <= 8'd0;
        end else begin
            // Base power from activity level
            logic [7:0] base_power;
            base_power = tile_activity_level[15:8];

            // Add overhead if dreaming (STDP learning active)
            if (dream_active) begin
                tile_power_hint <= base_power + 8'd32;  // Learning overhead
            end else begin
                tile_power_hint <= base_power;
            end
        end
    end

    // Entropy estimation (variance of spike timing)
    // Simple approximation: XOR recent spike patterns
    logic [15:0] spike_history;
    logic [3:0]  entropy_count;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            spike_history <= 16'd0;
            tile_entropy  <= 16'd0;
        end else begin
            // Shift in new spike bit
            spike_history <= {spike_history[14:0], (spike_out_valid && spike_out_ready)};

            // Count bit transitions (simple entropy proxy)
            entropy_count = 0;
            for (int i = 0; i < 15; i++) begin
                entropy_count = entropy_count + (spike_history[i] ^ spike_history[i+1]);
            end

            // Update entropy (scaled)
            tile_entropy <= {12'd0, entropy_count} << 10;
        end
    end

endmodule : kitten_fabric_tile
