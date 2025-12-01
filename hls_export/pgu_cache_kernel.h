// =============================================================================
// PGU Cache Kernel Header
// =============================================================================

#ifndef PGU_CACHE_KERNEL_H
#define PGU_CACHE_KERNEL_H

#include <ap_int.h>
#include <ap_fixed.h>

// Fixed-point types
typedef ap_fixed<16, 2, AP_RND, AP_SAT> pad_t;
typedef ap_uint<32> key_t;
typedef ap_uint<64> value_t;

// Cache configuration
#define CACHE_SIZE 1024
#define CACHE_WAYS 4
#define TAG_BITS 22
#define INDEX_BITS 8

// Cache entry
struct cache_entry_t {
    ap_uint<1> valid;
    ap_uint<TAG_BITS> tag;
    value_t data;
    ap_uint<8> lru;
};

// Metacontrol output
struct metacontrol_out_t {
    pad_t temperature_mult;
    pad_t memory_mult;
    pad_t attention_gain;
};

// Top function declaration
void pgu_cache_kernel(
    pad_t valence,
    pad_t arousal,
    pad_t dominance,
    key_t query_key,
    cache_entry_t cache[CACHE_SIZE],
    ap_uint<1> *cache_hit,
    value_t *cached_value,
    metacontrol_out_t *mc_out
);

// Utility functions
void lif_step(
    pad_t v_in, pad_t current, pad_t tau, pad_t v_th,
    pad_t *v_out, ap_uint<1> *spike
);

void l1_homeostat(
    pad_t current_valence, pad_t current_arousal,
    pad_t goal_valence, pad_t goal_arousal,
    pad_t error_integral_in, pad_t kp, pad_t ki,
    pad_t *valence_corr, pad_t *arousal_corr,
    pad_t *error_integral_out
);

#endif // PGU_CACHE_KERNEL_H
