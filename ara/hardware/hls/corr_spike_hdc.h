/*
 * CorrSpike-HDC Core - Header
 * ===========================
 *
 * Minimal, synthesizable hyperdimensional correlation + spiking inference
 * for FPGA deployment.
 *
 * This is the "subcortical reflex" kernel that runs on-chip:
 * - Takes encoded telemetry as a hypervector
 * - Correlates against stored prototypes
 * - Decides "normal" vs "novel" (escalate?)
 * - Applies Hebbian learning when escalating
 * - Optionally runs spiking inference for local policy
 *
 * Target: Artix-7 / Zynq-7000 / UltraScale+ (Xilinx)
 *         Can be ported to Intel HLS for Stratix-10
 *
 * Compile:
 *   vitis_hls -f run_hls.tcl
 *   OR
 *   vivado_hls -f run_hls.tcl
 */

#ifndef CORR_SPIKE_HDC_H
#define CORR_SPIKE_HDC_H

#include <ap_int.h>
#include <ap_fixed.h>

// ============================================================================
// Configuration
// ============================================================================

// Hypervector dimension - start small, scale later
// 1024 fits comfortably on Artix-7; scale to 4k-16k on bigger parts
#define HV_DIM      1024

// Number of stored prototypes (concept codebook)
#define N_PROTOS    16

// Spiking inference timesteps
#define SPIKE_STEPS 4

// Threshold for similarity (0.7 * HV_DIM by default)
#define THRESH_SIM  ((HV_DIM * 7) / 10)

// Threshold for LIF spike generation
#define THRESH_SPIKE 32

// ============================================================================
// Data Types
// ============================================================================

// Binary HV element: -1 or +1 encoded as 2-bit signed
typedef ap_int<2>  hv_t;

// Small integer weights for synapses / policy deltas
typedef ap_int<8>  w_t;

// Accumulator for similarity computation
typedef ap_int<16> sim_t;

// Fixed-point for decay (Q1.7 format: 0.0 to ~2.0)
typedef ap_fixed<8, 1> decay_t;

// LIF membrane potential
typedef ap_int<16> mem_t;

// Policy code output
typedef ap_uint<8> policy_t;

// ============================================================================
// Kernel Interface
// ============================================================================

extern "C" {

/**
 * Main CorrSpike-HDC kernel
 *
 * @param hv_in         Input hypervector (encoded telemetry)
 * @param prototypes    Stored prototype HVs (concept codebook)
 * @param synapses      Learnable synapse weights
 * @param hv_out        Output hypervector (updated state)
 * @param policy_delta  Hebbian delta for each dimension
 * @param escalate_out  True if input is "novel" (no good prototype match)
 * @param policy_out    Local policy decision (when escalating)
 * @param best_proto    Index of best-matching prototype
 * @param similarity    Similarity score of best match
 */
void corr_spike_hdc(
    hv_t hv_in[HV_DIM],                 // input HV
    hv_t prototypes[N_PROTOS][HV_DIM],  // stored prototype HVs
    w_t  synapses[HV_DIM],              // learnable synapse vector
    hv_t hv_out[HV_DIM],                // updated HV
    w_t  policy_delta[HV_DIM],          // Hebbian delta (for logging)
    bool *escalate_out,                 // did we escalate?
    policy_t *policy_out,               // local policy code
    ap_uint<8> *best_proto,             // best matching prototype index
    sim_t *similarity                   // similarity score
);

/**
 * Extended kernel with rolling state
 *
 * Maintains internal state HPV with decay, enabling temporal integration.
 */
void corr_spike_hdc_stateful(
    hv_t hv_in[HV_DIM],
    hv_t prototypes[N_PROTOS][HV_DIM],
    w_t  synapses[HV_DIM],
    hv_t hv_out[HV_DIM],
    w_t  policy_delta[HV_DIM],
    bool *escalate_out,
    policy_t *policy_out,
    ap_uint<8> *best_proto,
    sim_t *similarity,
    decay_t decay_factor,               // state decay (e.g., 0.9)
    bool reset_state                    // reset internal state
);

} // extern "C"

#endif // CORR_SPIKE_HDC_H
