// =============================================================================
// PGU Cache Kernel - Vitis HLS Implementation
// Target: p95 < 100ns @ 250MHz (25 cycles)
// =============================================================================

#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>

// Fixed-point types for PAD values
typedef ap_fixed<16, 2, AP_RND, AP_SAT> pad_t;      // Q2.14 for [-2, 2)
typedef ap_fixed<32, 16, AP_RND, AP_SAT> hash_t;    // Hash values
typedef ap_uint<32> key_t;                           // Cache key
typedef ap_uint<64> value_t;                         // Cached value

// Cache configuration
#define CACHE_SIZE 1024
#define CACHE_WAYS 4
#define TAG_BITS 22
#define INDEX_BITS 8

// Cache entry structure
struct cache_entry_t {
    ap_uint<1> valid;
    ap_uint<TAG_BITS> tag;
    value_t data;
    ap_uint<8> lru;
};

// L3 Metacontrol output structure
struct metacontrol_out_t {
    pad_t temperature_mult;
    pad_t memory_mult;
    pad_t attention_gain;
};

// PGU Cache kernel top function
void pgu_cache_kernel(
    // Input PAD state (AXI-Lite)
    pad_t valence,
    pad_t arousal,
    pad_t dominance,
    // Input query key
    key_t query_key,
    // Cache memory (AXI Master)
    cache_entry_t cache[CACHE_SIZE],
    // Outputs
    ap_uint<1> *cache_hit,
    value_t *cached_value,
    metacontrol_out_t *mc_out
) {
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    #pragma HLS INTERFACE s_axilite port=valence bundle=control
    #pragma HLS INTERFACE s_axilite port=arousal bundle=control
    #pragma HLS INTERFACE s_axilite port=dominance bundle=control
    #pragma HLS INTERFACE s_axilite port=query_key bundle=control
    #pragma HLS INTERFACE s_axilite port=cache_hit bundle=control
    #pragma HLS INTERFACE s_axilite port=cached_value bundle=control
    #pragma HLS INTERFACE s_axilite port=mc_out bundle=control

    #pragma HLS INTERFACE m_axi port=cache offset=slave bundle=gmem depth=1024

    #pragma HLS PIPELINE II=1
    #pragma HLS LATENCY max=25

    // Local cache copy for low latency
    cache_entry_t local_cache[CACHE_WAYS];
    #pragma HLS ARRAY_PARTITION variable=local_cache complete

    // =========================================================================
    // Stage 1: Compute cache index and tag
    // =========================================================================
    ap_uint<INDEX_BITS> index = query_key(INDEX_BITS-1, 0);
    ap_uint<TAG_BITS> tag = query_key(31, INDEX_BITS);

    // =========================================================================
    // Stage 2: Parallel cache lookup (4-way associative)
    // =========================================================================
    #pragma HLS PIPELINE
    cache_lookup: for (int way = 0; way < CACHE_WAYS; way++) {
        #pragma HLS UNROLL
        local_cache[way] = cache[index * CACHE_WAYS + way];
    }

    // Check for hit
    ap_uint<1> hit = 0;
    value_t hit_value = 0;
    int hit_way = 0;

    hit_check: for (int way = 0; way < CACHE_WAYS; way++) {
        #pragma HLS UNROLL
        if (local_cache[way].valid && local_cache[way].tag == tag) {
            hit = 1;
            hit_value = local_cache[way].data;
            hit_way = way;
        }
    }

    // =========================================================================
    // Stage 3: L3 Metacontrol computation (parallel with cache)
    // =========================================================================
    // Temperature: higher arousal → higher exploration
    pad_t temp_mult = pad_t(0.8) + arousal * pad_t(0.5);

    // Memory: lower valence → more conservative
    pad_t mem_mult = pad_t(0.95) + (valence + pad_t(1.0)) * pad_t(0.125);

    // Attention: higher dominance → higher gain
    pad_t attn_gain = pad_t(0.8) + dominance * pad_t(0.4);

    // =========================================================================
    // Stage 4: Output results
    // =========================================================================
    *cache_hit = hit;
    *cached_value = hit_value;
    mc_out->temperature_mult = temp_mult;
    mc_out->memory_mult = mem_mult;
    mc_out->attention_gain = attn_gain;

    // Update LRU on hit (simplified)
    if (hit) {
        local_cache[hit_way].lru = 0;
        cache[index * CACHE_WAYS + hit_way] = local_cache[hit_way];
    }
}

// LIF neuron step for L1 homeostatic loop
void lif_step(
    pad_t v_in,
    pad_t current,
    pad_t tau,
    pad_t v_th,
    pad_t *v_out,
    ap_uint<1> *spike
) {
    #pragma HLS INLINE
    #pragma HLS PIPELINE II=1

    // LIF dynamics: dv/dt = -v/tau + I
    pad_t dv = -v_in / tau + current;
    pad_t new_v = v_in + dv;

    // Spike and reset
    if (new_v >= v_th) {
        *spike = 1;
        *v_out = 0;
    } else {
        *spike = 0;
        *v_out = new_v;
    }
}

// L1 Homeostatic PI controller
void l1_homeostat(
    pad_t current_valence,
    pad_t current_arousal,
    pad_t goal_valence,
    pad_t goal_arousal,
    pad_t error_integral_in,
    pad_t kp,
    pad_t ki,
    pad_t *valence_corr,
    pad_t *arousal_corr,
    pad_t *error_integral_out
) {
    #pragma HLS INLINE
    #pragma HLS PIPELINE II=1

    // Compute errors
    pad_t val_error = goal_valence - current_valence;
    pad_t aro_error = goal_arousal - current_arousal;

    // PI control
    *valence_corr = kp * val_error + ki * error_integral_in;
    *arousal_corr = kp * aro_error + ki * error_integral_in;

    // Update integral with anti-windup
    pad_t new_integral = error_integral_in + val_error + aro_error;
    if (new_integral > pad_t(10.0)) new_integral = pad_t(10.0);
    if (new_integral < pad_t(-10.0)) new_integral = pad_t(-10.0);
    *error_integral_out = new_integral;
}
