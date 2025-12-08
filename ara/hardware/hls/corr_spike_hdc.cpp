/*
 * CorrSpike-HDC Core - Implementation
 * ====================================
 *
 * Synthesizable HLS kernel for hyperdimensional correlation + spiking inference.
 *
 * Architecture:
 *   1. Copy input HV to local state
 *   2. Correlate against N_PROTOS prototypes (XNOR + popcount style)
 *   3. Decide escalate based on similarity threshold
 *   4. If escalating: run spiking inference head for local policy
 *   5. Apply Hebbian learning to synapses
 *   6. Output updated HV and policy
 *
 * This is the "subcortical reflex" - always on, low power, fast decisions.
 */

#include "corr_spike_hdc.h"

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Compute similarity between two binary HVs.
 * Similarity = count(a == b) - count(a != b) ∈ [-HV_DIM, +HV_DIM]
 */
static sim_t compute_similarity(hv_t a[HV_DIM], hv_t b[HV_DIM]) {
#pragma HLS INLINE
    sim_t sim = 0;

    SIM_LOOP: for (int d = 0; d < HV_DIM; ++d) {
#pragma HLS UNROLL factor=16
        sim += (a[d] == b[d]) ? 1 : -1;
    }

    return sim;
}

/**
 * Clamp a weight to [-63, +63] range (fits in 7 bits signed)
 */
static w_t clamp_weight(int val) {
#pragma HLS INLINE
    if (val > 63) return 63;
    if (val < -63) return -63;
    return (w_t)val;
}

// ============================================================================
// Main Kernel
// ============================================================================

extern "C" {

void corr_spike_hdc(
    hv_t hv_in[HV_DIM],
    hv_t prototypes[N_PROTOS][HV_DIM],
    w_t  synapses[HV_DIM],
    hv_t hv_out[HV_DIM],
    w_t  policy_delta[HV_DIM],
    bool *escalate_out,
    policy_t *policy_out,
    ap_uint<8> *best_proto,
    sim_t *similarity
) {
    // ========================================================================
    // Interface Pragmas
    // ========================================================================
#pragma HLS INTERFACE m_axi port=hv_in        offset=slave bundle=gmem0 depth=HV_DIM
#pragma HLS INTERFACE m_axi port=prototypes   offset=slave bundle=gmem1 depth=N_PROTOS*HV_DIM
#pragma HLS INTERFACE m_axi port=synapses     offset=slave bundle=gmem2 depth=HV_DIM
#pragma HLS INTERFACE m_axi port=hv_out       offset=slave bundle=gmem3 depth=HV_DIM
#pragma HLS INTERFACE m_axi port=policy_delta offset=slave bundle=gmem4 depth=HV_DIM
#pragma HLS INTERFACE s_axilite port=escalate_out bundle=control
#pragma HLS INTERFACE s_axilite port=policy_out   bundle=control
#pragma HLS INTERFACE s_axilite port=best_proto   bundle=control
#pragma HLS INTERFACE s_axilite port=similarity   bundle=control
#pragma HLS INTERFACE s_axilite port=return       bundle=control

    // ========================================================================
    // Local Buffers
    // ========================================================================

    hv_t hv_state[HV_DIM];
#pragma HLS ARRAY_PARTITION variable=hv_state cyclic factor=16 dim=1

    hv_t proto_buf[HV_DIM];
#pragma HLS ARRAY_PARTITION variable=proto_buf cyclic factor=16 dim=1

    // LIF membrane for spiking inference
    static mem_t lif_mem[HV_DIM];
#pragma HLS ARRAY_PARTITION variable=lif_mem cyclic factor=16 dim=1

    // ========================================================================
    // 1. Copy Input HV to Local State
    // ========================================================================

    COPY_STATE: for (int d = 0; d < HV_DIM; ++d) {
#pragma HLS PIPELINE II=1
        hv_state[d] = hv_in[d];
    }

    // ========================================================================
    // 2. Correlate Against Prototypes
    // ========================================================================

    sim_t max_sim = -HV_DIM;
    ap_uint<8> best_id = 0;

    PROTO_LOOP: for (int p = 0; p < N_PROTOS; ++p) {
        // Load prototype to local buffer
        LOAD_PROTO: for (int d = 0; d < HV_DIM; ++d) {
#pragma HLS PIPELINE II=1
            proto_buf[d] = prototypes[p][d];
        }

        // Compute similarity
        sim_t sim = 0;
        SIM_INNER: for (int d = 0; d < HV_DIM; ++d) {
#pragma HLS UNROLL factor=16
            sim += (hv_state[d] == proto_buf[d]) ? 1 : -1;
        }

        // Track best match
        if (sim > max_sim) {
            max_sim = sim;
            best_id = p;
        }
    }

    // ========================================================================
    // 3. Escalation Decision
    // ========================================================================

    bool escalate = (max_sim < THRESH_SIM);
    *escalate_out = escalate;
    *best_proto = best_id;
    *similarity = max_sim;

    // ========================================================================
    // 4. Spiking Inference (when escalating)
    // ========================================================================

    policy_t policy_code = 0;

    if (escalate) {
        // Run LIF neurons for SPIKE_STEPS timesteps
        SPIKE_TIME: for (int t = 0; t < SPIKE_STEPS; ++t) {
#pragma HLS PIPELINE off

            ap_uint<8> spike_count = 0;

            SPIKE_DIM: for (int d = 0; d < HV_DIM; ++d) {
#pragma HLS PIPELINE II=1

                // Weighted input: hv_state * synapse
                int weighted = hv_state[d] * synapses[d];

                // Integrate
                mem_t acc = lif_mem[d] + weighted;

                // Spike check
                bool spike = (acc > THRESH_SPIKE);

                // Reset on spike, else accumulate
                lif_mem[d] = spike ? (mem_t)0 : acc;

                // Count spikes for policy encoding
                if (spike && d < 8) {
                    // Use first 8 dimensions as policy bits
                    policy_code |= (1 << d);
                }
                spike_count += spike ? 1 : 0;
            }
        }
    }

    *policy_out = policy_code;

    // ========================================================================
    // 5. Hebbian Learning (when escalating)
    // ========================================================================

    HEBB_LOOP: for (int d = 0; d < HV_DIM; ++d) {
#pragma HLS PIPELINE II=1

        w_t dw = 0;

        if (escalate) {
            // Three-factor-ish: if state bit is active, nudge synapse
            // Full three-factor would be: Δw = η * ρ * pre * post
            // Here we simplify: ρ is implicit (escalate), pre = hv_state[d]
            if (hv_state[d] > 0) {
                dw = 1;
            } else if (hv_state[d] < 0) {
                dw = -1;
            }

            // Update synapse with clipping
            synapses[d] = clamp_weight(synapses[d] + dw);
        }

        policy_delta[d] = dw;
    }

    // ========================================================================
    // 6. Output Updated HV
    // ========================================================================

    // For now, passthrough. Later: incorporate synaptic modulation
    OUT_LOOP: for (int d = 0; d < HV_DIM; ++d) {
#pragma HLS PIPELINE II=1
        hv_out[d] = hv_state[d];
    }
}

// ============================================================================
// Stateful Kernel with Rolling State
// ============================================================================

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
    decay_t decay_factor,
    bool reset_state
) {
#pragma HLS INTERFACE m_axi port=hv_in        offset=slave bundle=gmem0 depth=HV_DIM
#pragma HLS INTERFACE m_axi port=prototypes   offset=slave bundle=gmem1 depth=N_PROTOS*HV_DIM
#pragma HLS INTERFACE m_axi port=synapses     offset=slave bundle=gmem2 depth=HV_DIM
#pragma HLS INTERFACE m_axi port=hv_out       offset=slave bundle=gmem3 depth=HV_DIM
#pragma HLS INTERFACE m_axi port=policy_delta offset=slave bundle=gmem4 depth=HV_DIM
#pragma HLS INTERFACE s_axilite port=escalate_out bundle=control
#pragma HLS INTERFACE s_axilite port=policy_out   bundle=control
#pragma HLS INTERFACE s_axilite port=best_proto   bundle=control
#pragma HLS INTERFACE s_axilite port=similarity   bundle=control
#pragma HLS INTERFACE s_axilite port=decay_factor bundle=control
#pragma HLS INTERFACE s_axilite port=reset_state  bundle=control
#pragma HLS INTERFACE s_axilite port=return       bundle=control

    // Persistent state HPV (survives across kernel calls)
    static ap_fixed<16, 8> state_mem[HV_DIM];
#pragma HLS ARRAY_PARTITION variable=state_mem cyclic factor=16 dim=1

    // Local buffers
    hv_t hv_state[HV_DIM];
#pragma HLS ARRAY_PARTITION variable=hv_state cyclic factor=16 dim=1

    hv_t proto_buf[HV_DIM];
#pragma HLS ARRAY_PARTITION variable=proto_buf cyclic factor=16 dim=1

    static mem_t lif_mem[HV_DIM];
#pragma HLS ARRAY_PARTITION variable=lif_mem cyclic factor=16 dim=1

    // Reset state if requested
    if (reset_state) {
        RESET_STATE: for (int d = 0; d < HV_DIM; ++d) {
#pragma HLS PIPELINE II=1
            state_mem[d] = 0;
            lif_mem[d] = 0;
        }
    }

    // ========================================================================
    // 1. Rolling State Integration
    // ========================================================================

    ROLLING_STATE: for (int d = 0; d < HV_DIM; ++d) {
#pragma HLS PIPELINE II=1

        // Decay old state and add new input
        ap_fixed<16, 8> decayed = state_mem[d] * decay_factor;
        ap_fixed<16, 8> updated = decayed + hv_in[d];

        // Store back
        state_mem[d] = updated;

        // Binarize for correlation
        hv_state[d] = (updated > 0) ? (hv_t)1 : (hv_t)-1;
    }

    // ========================================================================
    // 2. Correlate Against Prototypes
    // ========================================================================

    sim_t max_sim = -HV_DIM;
    ap_uint<8> best_id = 0;

    PROTO_LOOP_S: for (int p = 0; p < N_PROTOS; ++p) {
        LOAD_PROTO_S: for (int d = 0; d < HV_DIM; ++d) {
#pragma HLS PIPELINE II=1
            proto_buf[d] = prototypes[p][d];
        }

        sim_t sim = 0;
        SIM_INNER_S: for (int d = 0; d < HV_DIM; ++d) {
#pragma HLS UNROLL factor=16
            sim += (hv_state[d] == proto_buf[d]) ? 1 : -1;
        }

        if (sim > max_sim) {
            max_sim = sim;
            best_id = p;
        }
    }

    // ========================================================================
    // 3. Escalation + Spiking + Hebbian (same as stateless)
    // ========================================================================

    bool escalate = (max_sim < THRESH_SIM);
    *escalate_out = escalate;
    *best_proto = best_id;
    *similarity = max_sim;

    policy_t policy_code = 0;

    if (escalate) {
        SPIKE_TIME_S: for (int t = 0; t < SPIKE_STEPS; ++t) {
            SPIKE_DIM_S: for (int d = 0; d < HV_DIM; ++d) {
#pragma HLS PIPELINE II=1
                int weighted = hv_state[d] * synapses[d];
                mem_t acc = lif_mem[d] + weighted;
                bool spike = (acc > THRESH_SPIKE);
                lif_mem[d] = spike ? (mem_t)0 : acc;
                if (spike && d < 8) {
                    policy_code |= (1 << d);
                }
            }
        }
    }

    *policy_out = policy_code;

    HEBB_LOOP_S: for (int d = 0; d < HV_DIM; ++d) {
#pragma HLS PIPELINE II=1
        w_t dw = 0;
        if (escalate) {
            if (hv_state[d] > 0) dw = 1;
            else if (hv_state[d] < 0) dw = -1;
            synapses[d] = clamp_weight(synapses[d] + dw);
        }
        policy_delta[d] = dw;
    }

    // ========================================================================
    // 4. Output
    // ========================================================================

    OUT_LOOP_S: for (int d = 0; d < HV_DIM; ++d) {
#pragma HLS PIPELINE II=1
        hv_out[d] = hv_state[d];
    }
}

} // extern "C"
