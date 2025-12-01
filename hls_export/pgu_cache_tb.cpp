// =============================================================================
// PGU Cache Kernel Testbench
// =============================================================================

#include <iostream>
#include <cmath>
#include "pgu_cache_kernel.h"

int main() {
    // Initialize cache
    cache_entry_t cache[CACHE_SIZE];
    for (int i = 0; i < CACHE_SIZE; i++) {
        cache[i].valid = 0;
        cache[i].tag = 0;
        cache[i].data = 0;
        cache[i].lru = 0;
    }

    // Test inputs
    pad_t valence = pad_t(0.3);
    pad_t arousal = pad_t(0.6);
    pad_t dominance = pad_t(0.5);
    key_t query_key = 0x12345678;

    // Outputs
    ap_uint<1> cache_hit;
    value_t cached_value;
    metacontrol_out_t mc_out;

    // Pre-populate cache for testing
    cache[query_key & 0xFF].valid = 1;
    cache[query_key & 0xFF].tag = query_key >> 8;
    cache[query_key & 0xFF].data = 0xDEADBEEF;

    // Run kernel
    pgu_cache_kernel(
        valence, arousal, dominance,
        query_key,
        cache,
        &cache_hit, &cached_value, &mc_out
    );

    // Verify results
    std::cout << "=== PGU Cache Kernel Test ===" << std::endl;
    std::cout << "Cache hit: " << cache_hit << std::endl;
    std::cout << "Cached value: 0x" << std::hex << cached_value << std::endl;
    std::cout << "Temperature mult: " << mc_out.temperature_mult.to_float() << std::endl;
    std::cout << "Memory mult: " << mc_out.memory_mult.to_float() << std::endl;
    std::cout << "Attention gain: " << mc_out.attention_gain.to_float() << std::endl;

    // Check expected values
    bool pass = true;

    // Expect cache hit
    if (cache_hit != 1) {
        std::cerr << "ERROR: Expected cache hit" << std::endl;
        pass = false;
    }

    // Check metacontrol values (approximate)
    float expected_temp = 0.8 + 0.6 * 0.5;  // 1.1
    float expected_mem = 0.95 + (0.3 + 1.0) * 0.125;  // 1.1125
    float expected_attn = 0.8 + 0.5 * 0.4;  // 1.0

    if (std::abs(mc_out.temperature_mult.to_float() - expected_temp) > 0.1) {
        std::cerr << "ERROR: Temperature mult mismatch" << std::endl;
        pass = false;
    }

    // Test cache miss
    key_t miss_key = 0xFFFFFFFF;
    pgu_cache_kernel(
        valence, arousal, dominance,
        miss_key,
        cache,
        &cache_hit, &cached_value, &mc_out
    );

    if (cache_hit != 0) {
        std::cerr << "ERROR: Expected cache miss" << std::endl;
        pass = false;
    }

    std::cout << "\n=== Test " << (pass ? "PASSED" : "FAILED") << " ===" << std::endl;
    return pass ? 0 : 1;
}
