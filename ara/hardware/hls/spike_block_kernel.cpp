/*
 * SpikingBrain-Style Tile - HLS Kernel
 * =====================================
 *
 * High-Level Synthesis implementation for Xilinx Vitis.
 *
 * This kernel processes embeddings through:
 * 1. Sparse spiking neuron layer
 * 2. Linear attention accumulator
 * 3. Optional Hebbian learning
 *
 * Target: Xilinx Alveo U250 / U280
 *
 * Compile:
 *   v++ -c -t hw --platform xilinx_u250 spike_block_kernel.cpp
 */

#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>
#include <hls_math.h>

// ============================================================================
// Configuration
// ============================================================================

#define N_NEURONS   512
#define D_EMBED     128
#define D_HEAD      64
#define MAX_NNZ     8192  // Maximum non-zeros per neuron layer

// Data types
typedef ap_int<8>   state_t;    // Neuron state
typedef ap_int<4>   weight_t;   // Quantized weight
typedef ap_int<24>  accum_t;    // Accumulator
typedef ap_uint<8>  thresh_t;   // Threshold
typedef ap_fixed<16, 8> float16_t;

// ============================================================================
// Sparse Weight Storage
// ============================================================================

struct SparseWeights {
    ap_uint<16> row_ptr[N_NEURONS + 1];
    ap_uint<16> col_idx[MAX_NNZ];
    weight_t    values[MAX_NNZ];
};

// ============================================================================
// Neuron State
// ============================================================================

struct NeuronState {
    state_t  membrane[N_NEURONS];
    thresh_t threshold[N_NEURONS];
};

// ============================================================================
// Linear Attention State
// ============================================================================

struct AttentionState {
    float16_t m[D_HEAD];           // Running key sum
    float16_t u[D_HEAD][D_HEAD];   // Running key⊗value sum
};

// ============================================================================
// Spiking Neuron Layer
// ============================================================================

void spiking_neurons(
    state_t input[D_EMBED],
    ap_uint<1> spikes[N_NEURONS],
    NeuronState& state,
    const SparseWeights& weights,
    state_t rho,
    bool learn_enable
) {
    #pragma HLS INLINE off

    // Fixed-point parameters
    const ap_int<8> ALPHA = 115;  // 0.9 in Q0.7
    const ap_int<8> BETA = 1;     // 0.01 in Q0.7
    const ap_int<8> TAU = 6;      // 0.05 in Q0.7

    // Process each neuron
    NEURON_LOOP: for (int i = 0; i < N_NEURONS; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=512 max=512

        // Sparse MAV
        accum_t accum = 0;
        int start = weights.row_ptr[i];
        int end = weights.row_ptr[i + 1];

        MAV_LOOP: for (int k = start; k < end; k++) {
            #pragma HLS LOOP_TRIPCOUNT min=4 max=32
            int j = weights.col_idx[k];
            weight_t w = weights.values[k];
            accum += w * input[j];
        }

        // Leaky integrate
        accum_t leaked = (state.membrane[i] * ALPHA) >> 7;
        accum_t potential = leaked + accum - state.threshold[i];

        // Spike decision
        if (potential > 0) {
            spikes[i] = 1;
            state.membrane[i] = 0;  // Reset
        } else {
            spikes[i] = 0;
            state.membrane[i] = potential;
        }

        // Threshold adaptation
        if (spikes[i]) {
            if (state.threshold[i] < 255 - BETA)
                state.threshold[i] += BETA;
        } else {
            if (state.threshold[i] > TAU)
                state.threshold[i] -= 1;
        }
    }

    // Hebbian learning (if enabled and rho != 0)
    if (learn_enable && rho != 0) {
        LEARN_LOOP: for (int i = 0; i < N_NEURONS; i++) {
            #pragma HLS PIPELINE II=1
            if (spikes[i]) {
                int start = weights.row_ptr[i];
                int end = weights.row_ptr[i + 1];

                for (int k = start; k < end; k++) {
                    #pragma HLS LOOP_TRIPCOUNT min=4 max=32
                    int j = weights.col_idx[k];
                    // Δw = η * rho * pre[j] (post[i] = 1 since we're here)
                    ap_int<16> delta = (rho * input[j]) >> 11;  // Scale down
                    ap_int<8> new_w = weights.values[k] + delta;
                    // Clamp to 4-bit range
                    if (new_w > 7) new_w = 7;
                    if (new_w < -8) new_w = -8;
                    // Note: In real HW, weights would be in BRAM with write port
                }
            }
        }
    }
}

// ============================================================================
// Linear Attention
// ============================================================================

void linear_attention(
    state_t input[D_EMBED],
    state_t output[D_EMBED],
    AttentionState& state,
    float16_t W_k[D_HEAD][D_EMBED],
    float16_t W_v[D_HEAD][D_EMBED],
    float16_t W_o[D_EMBED][D_HEAD]
) {
    #pragma HLS INLINE off

    float16_t k[D_HEAD], v[D_HEAD];

    // Project to key/value
    KEY_PROJ: for (int h = 0; h < D_HEAD; h++) {
        #pragma HLS PIPELINE II=1
        float16_t sum = 0;
        for (int d = 0; d < D_EMBED; d++) {
            sum += W_k[h][d] * input[d];
        }
        k[h] = sum;
    }

    VALUE_PROJ: for (int h = 0; h < D_HEAD; h++) {
        #pragma HLS PIPELINE II=1
        float16_t sum = 0;
        for (int d = 0; d < D_EMBED; d++) {
            sum += W_v[h][d] * input[d];
        }
        v[h] = sum;
    }

    // Update running statistics: m += k, u += k ⊗ v
    UPDATE_STATS: for (int i = 0; i < D_HEAD; i++) {
        #pragma HLS PIPELINE II=1
        state.m[i] += k[i];
        for (int j = 0; j < D_HEAD; j++) {
            #pragma HLS UNROLL factor=8
            state.u[i][j] += k[i] * v[j];
        }
    }

    // Compute attention output: softmax(m) @ u @ W_o
    // Simplified: just use m as weights directly (linear attention)
    float16_t attn_out[D_HEAD];
    ATTN_OUT: for (int i = 0; i < D_HEAD; i++) {
        #pragma HLS PIPELINE II=1
        float16_t sum = 0;
        for (int j = 0; j < D_HEAD; j++) {
            sum += state.u[i][j] * state.m[j];
        }
        attn_out[i] = sum;
    }

    // Project back to embedding space
    OUTPUT_PROJ: for (int d = 0; d < D_EMBED; d++) {
        #pragma HLS PIPELINE II=1
        float16_t sum = input[d];  // Residual
        for (int h = 0; h < D_HEAD; h++) {
            sum += W_o[d][h] * attn_out[h] * float16_t(0.1);  // Scale down
        }
        output[d] = sum;
    }
}

// ============================================================================
// Top-Level Kernel
// ============================================================================

extern "C" {

void spike_block_kernel(
    // Input embedding stream
    hls::stream<ap_axiu<D_EMBED*8, 0, 0, 0>>& s_axis,
    // Output embedding stream
    hls::stream<ap_axiu<D_EMBED*8, 0, 0, 0>>& m_axis,
    // Spike output (for debug)
    hls::stream<ap_axiu<N_NEURONS, 0, 0, 0>>& spike_axis,
    // Weights (in DDR)
    ap_uint<16>* weight_row_ptr,
    ap_uint<16>* weight_col_idx,
    weight_t* weight_values,
    // Control
    int n_tokens,
    state_t rho,
    bool learn_enable
) {
    #pragma HLS INTERFACE axis port=s_axis
    #pragma HLS INTERFACE axis port=m_axis
    #pragma HLS INTERFACE axis port=spike_axis
    #pragma HLS INTERFACE m_axi port=weight_row_ptr offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=weight_col_idx offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=weight_values offset=slave bundle=gmem0
    #pragma HLS INTERFACE s_axilite port=n_tokens bundle=control
    #pragma HLS INTERFACE s_axilite port=rho bundle=control
    #pragma HLS INTERFACE s_axilite port=learn_enable bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // Local state (persistent across tokens)
    static NeuronState neuron_state;
    static AttentionState attn_state;

    // Load weights to local buffers (simplified - would use streaming in real impl)
    SparseWeights weights;
    LOAD_ROW_PTR: for (int i = 0; i <= N_NEURONS; i++) {
        #pragma HLS PIPELINE II=1
        weights.row_ptr[i] = weight_row_ptr[i];
    }

    int nnz = weights.row_ptr[N_NEURONS];
    LOAD_COL_IDX: for (int i = 0; i < nnz; i++) {
        #pragma HLS PIPELINE II=1
        weights.col_idx[i] = weight_col_idx[i];
        weights.values[i] = weight_values[i];
    }

    // Dummy attention weights (would be loaded from DDR in real impl)
    static float16_t W_k[D_HEAD][D_EMBED];
    static float16_t W_v[D_HEAD][D_EMBED];
    static float16_t W_o[D_EMBED][D_HEAD];
    #pragma HLS ARRAY_PARTITION variable=W_k cyclic factor=8 dim=2
    #pragma HLS ARRAY_PARTITION variable=W_v cyclic factor=8 dim=2
    #pragma HLS ARRAY_PARTITION variable=W_o cyclic factor=8 dim=2

    // Process tokens
    TOKEN_LOOP: for (int t = 0; t < n_tokens; t++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=1024

        // Read input
        ap_axiu<D_EMBED*8, 0, 0, 0> in_pkt = s_axis.read();
        state_t input[D_EMBED];
        #pragma HLS ARRAY_PARTITION variable=input cyclic factor=8
        for (int i = 0; i < D_EMBED; i++) {
            #pragma HLS UNROLL factor=8
            input[i] = in_pkt.data.range((i+1)*8-1, i*8);
        }

        // Spiking neurons
        ap_uint<1> spikes[N_NEURONS];
        #pragma HLS ARRAY_PARTITION variable=spikes cyclic factor=64
        spiking_neurons(input, spikes, neuron_state, weights, rho, learn_enable);

        // Linear attention
        state_t output[D_EMBED];
        #pragma HLS ARRAY_PARTITION variable=output cyclic factor=8
        linear_attention(input, output, attn_state, W_k, W_v, W_o);

        // Combine: output + spike contribution
        for (int i = 0; i < D_EMBED; i++) {
            #pragma HLS UNROLL factor=8
            // Add spike signal (simplified: just check if any spike in corresponding range)
            int spike_range = N_NEURONS / D_EMBED;
            int spike_count = 0;
            for (int s = 0; s < spike_range; s++) {
                spike_count += spikes[i * spike_range + s];
            }
            output[i] += spike_count * 4;  // Scale spike contribution
        }

        // Write output
        ap_axiu<D_EMBED*8, 0, 0, 0> out_pkt;
        for (int i = 0; i < D_EMBED; i++) {
            #pragma HLS UNROLL factor=8
            out_pkt.data.range((i+1)*8-1, i*8) = output[i];
        }
        out_pkt.last = (t == n_tokens - 1);
        m_axis.write(out_pkt);

        // Write spikes (debug)
        ap_axiu<N_NEURONS, 0, 0, 0> spike_pkt;
        for (int i = 0; i < N_NEURONS; i++) {
            #pragma HLS UNROLL factor=64
            spike_pkt.data[i] = spikes[i];
        }
        spike_pkt.last = (t == n_tokens - 1);
        spike_axis.write(spike_pkt);
    }
}

} // extern "C"
