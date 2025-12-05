/**
 * Kitten Fabric (FK33) - Common Types and Parameters
 * ===================================================
 *
 * Bio-Affective Neuromorphic Operating System
 * Spatial, event-driven spiking neural network fabric
 *
 * This package defines the core types and parameters for the
 * Kitten Fabric architecture:
 * - Spike packet (AER) format
 * - Synapse structure
 * - Fixed-point formats for membrane potential and weights
 * - Configurable tile parameters
 */

package kf_pkg;

    // =========================================================================
    // TILE CONFIGURATION (v1: conservative for 4x4 mesh)
    // =========================================================================

    parameter int KF_NEURONS_PER_TILE  = 256;    // Neurons per tile
    parameter int KF_SYNAPSES_PER_TILE = 4096;   // Synapses per tile
    parameter int KF_MESH_X            = 4;      // Mesh width (tiles)
    parameter int KF_MESH_Y            = 4;      // Mesh height (tiles)

    // Derived parameters
    parameter int KF_NEURON_ID_BITS    = $clog2(KF_NEURONS_PER_TILE);  // 8
    parameter int KF_SYNAPSE_ID_BITS   = $clog2(KF_SYNAPSES_PER_TILE); // 12
    parameter int KF_COORD_BITS        = 8;      // Tile coordinates (0-255)

    // =========================================================================
    // FIXED-POINT FORMATS
    // =========================================================================

    // Membrane potential: Q8.8 signed (-128.0 to +127.996)
    // Range chosen to allow accumulation of many small weights
    parameter int V_INT_BITS  = 8;
    parameter int V_FRAC_BITS = 8;
    parameter int V_WIDTH     = V_INT_BITS + V_FRAC_BITS;  // 16

    // Synaptic weight: Q1.7 signed (-1.0 to +0.992)
    // Small weights that sum to produce membrane changes
    parameter int W_INT_BITS  = 1;
    parameter int W_FRAC_BITS = 7;
    parameter int W_WIDTH     = W_INT_BITS + W_FRAC_BITS;  // 8

    // Threshold and reset values (Q8.8)
    parameter logic signed [V_WIDTH-1:0] V_THRESH = 16'sh1000;  // +16.0
    parameter logic signed [V_WIDTH-1:0] V_RESET  = 16'sh0000;  // 0.0
    parameter logic signed [V_WIDTH-1:0] V_LEAK   = 16'sh0010;  // +0.0625 (leak per cycle)

    // =========================================================================
    // SPIKE PACKET (Address-Event Representation)
    // =========================================================================

    // 32-bit spike flit for NoC routing
    typedef struct packed {
        logic [KF_COORD_BITS-1:0]    dest_x;     // Destination tile X [31:24]
        logic [KF_COORD_BITS-1:0]    dest_y;     // Destination tile Y [23:16]
        logic [KF_NEURON_ID_BITS-1:0] neuron_id; // Neuron ID within tile [15:8]
        logic [7:0]                   payload;    // Context/class/timestamp [7:0]
    } spike_flit_t;

    parameter int FLIT_WIDTH = $bits(spike_flit_t);  // 32

    // =========================================================================
    // SYNAPSE STRUCTURE
    // =========================================================================

    // Synapse entry in BRAM
    // Stored in synapse_ram[0:KF_SYNAPSES_PER_TILE-1]
    typedef struct packed {
        logic [KF_NEURON_ID_BITS-1:0] pre_id;    // Presynaptic neuron ID
        logic [KF_NEURON_ID_BITS-1:0] post_id;   // Postsynaptic neuron ID
        logic signed [W_WIDTH-1:0]    weight;    // Synaptic weight (Q1.7)
        logic [7:0]                   flags;     // Plasticity flags, delay, etc.
    } synapse_t;

    parameter int SYNAPSE_WIDTH = $bits(synapse_t);  // 32

    // Index entry: range of synapses for a given presynaptic neuron
    typedef struct packed {
        logic [KF_SYNAPSE_ID_BITS-1:0] start_idx;  // First synapse index
        logic [KF_SYNAPSE_ID_BITS-1:0] end_idx;    // Last synapse index (inclusive)
    } syn_index_t;

    parameter int SYN_INDEX_WIDTH = $bits(syn_index_t);  // 24

    // =========================================================================
    // PLASTICITY FLAGS
    // =========================================================================

    // flags[7:0] interpretation:
    // [0]   : Enable STDP for this synapse
    // [1]   : Excitatory (1) vs Inhibitory (0)
    // [3:2] : Delay class (0-3 cycles)
    // [7:4] : Reserved for future plasticity rules

    parameter int FLAG_STDP_EN    = 0;
    parameter int FLAG_EXCITATORY = 1;
    parameter int FLAG_DELAY_LO   = 2;
    parameter int FLAG_DELAY_HI   = 3;

    // =========================================================================
    // ROUTER DIRECTIONS
    // =========================================================================

    typedef enum logic [2:0] {
        DIR_LOCAL = 3'd0,
        DIR_NORTH = 3'd1,
        DIR_SOUTH = 3'd2,
        DIR_EAST  = 3'd3,
        DIR_WEST  = 3'd4
    } direction_t;

    parameter int NUM_PORTS = 5;  // Local + N/S/E/W

    // =========================================================================
    // UTILITY FUNCTIONS
    // =========================================================================

    // Sign-extend weight to membrane potential width for accumulation
    function automatic logic signed [V_WIDTH-1:0] weight_to_vmem(
        input logic signed [W_WIDTH-1:0] w
    );
        // W is Q1.7, V is Q8.8
        // Shift left by 1 to align fractional points, then sign-extend
        return {{(V_INT_BITS-W_INT_BITS){w[W_WIDTH-1]}}, w, 1'b0};
    endfunction

    // Saturating add for membrane potential
    function automatic logic signed [V_WIDTH-1:0] vmem_add_sat(
        input logic signed [V_WIDTH-1:0] a,
        input logic signed [V_WIDTH-1:0] b
    );
        logic signed [V_WIDTH:0] sum;
        sum = a + b;
        // Saturate on overflow
        if (sum > $signed({1'b0, {(V_WIDTH-1){1'b1}}}))
            return {1'b0, {(V_WIDTH-1){1'b1}}};  // Max positive
        else if (sum < $signed({1'b1, {(V_WIDTH-1){1'b0}}}))
            return {1'b1, {(V_WIDTH-1){1'b0}}};  // Max negative
        else
            return sum[V_WIDTH-1:0];
    endfunction

endpackage : kf_pkg
