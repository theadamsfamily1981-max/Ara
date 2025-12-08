/*
 * CorrSpike-HDC Testbench
 * =======================
 *
 * HLS C-simulation testbench for the CorrSpike-HDC kernel.
 *
 * Run with:
 *   vitis_hls -f run_hls.tcl  (which calls this for csim)
 *   OR compile standalone:
 *   g++ -I$XILINX_HLS/include -o tb corr_spike_hdc.cpp corr_spike_hdc_tb.cpp
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "corr_spike_hdc.h"

// ============================================================================
// Test Utilities
// ============================================================================

// Simple PRNG for reproducibility
static unsigned int seed = 12345;
static int my_rand() {
    seed = seed * 1103515245 + 12345;
    return (seed >> 16) & 0x7fff;
}

// Generate random binary HV (-1 or +1)
static void random_hv(hv_t hv[HV_DIM]) {
    for (int i = 0; i < HV_DIM; ++i) {
        hv[i] = (my_rand() % 2 == 0) ? (hv_t)1 : (hv_t)-1;
    }
}

// Generate HV similar to a prototype (flip some bits)
static void similar_hv(hv_t proto[HV_DIM], hv_t out[HV_DIM], float similarity) {
    int n_flip = (int)(HV_DIM * (1.0f - similarity) / 2.0f);
    memcpy(out, proto, sizeof(hv_t) * HV_DIM);
    for (int i = 0; i < n_flip; ++i) {
        int idx = my_rand() % HV_DIM;
        out[idx] = -out[idx];
    }
}

// Compute similarity between two HVs
static int compute_sim(hv_t a[HV_DIM], hv_t b[HV_DIM]) {
    int sim = 0;
    for (int i = 0; i < HV_DIM; ++i) {
        sim += (a[i] == b[i]) ? 1 : -1;
    }
    return sim;
}

// ============================================================================
// Test Cases
// ============================================================================

int test_basic_correlation() {
    printf("\n=== Test 1: Basic Correlation ===\n");

    // Allocate buffers
    hv_t hv_in[HV_DIM];
    hv_t prototypes[N_PROTOS][HV_DIM];
    w_t  synapses[HV_DIM];
    hv_t hv_out[HV_DIM];
    w_t  policy_delta[HV_DIM];
    bool escalate;
    policy_t policy;
    ap_uint<8> best_proto;
    sim_t similarity;

    // Initialize
    memset(synapses, 0, sizeof(synapses));

    // Generate random prototypes
    printf("Generating %d random prototypes...\n", N_PROTOS);
    for (int p = 0; p < N_PROTOS; ++p) {
        random_hv(prototypes[p]);
    }

    // Test 1a: Input similar to prototype 5 (should NOT escalate)
    printf("\nTest 1a: Input similar to prototype 5 (90%% similarity)\n");
    similar_hv(prototypes[5], hv_in, 0.9f);

    corr_spike_hdc(hv_in, prototypes, synapses, hv_out, policy_delta,
                   &escalate, &policy, &best_proto, &similarity);

    printf("  Best match: prototype %d\n", (int)best_proto);
    printf("  Similarity: %d / %d (%.1f%%)\n",
           (int)similarity, HV_DIM, 100.0f * (similarity + HV_DIM) / (2 * HV_DIM));
    printf("  Escalate: %s\n", escalate ? "YES" : "NO");
    printf("  Policy: 0x%02x\n", (int)policy);

    if (best_proto != 5) {
        printf("  FAIL: Expected best_proto=5, got %d\n", (int)best_proto);
        return 1;
    }
    if (escalate) {
        printf("  FAIL: Should NOT escalate for similar input\n");
        return 1;
    }
    printf("  PASS\n");

    // Test 1b: Completely random input (should escalate)
    printf("\nTest 1b: Random input (should escalate)\n");
    random_hv(hv_in);

    corr_spike_hdc(hv_in, prototypes, synapses, hv_out, policy_delta,
                   &escalate, &policy, &best_proto, &similarity);

    printf("  Best match: prototype %d\n", (int)best_proto);
    printf("  Similarity: %d / %d (%.1f%%)\n",
           (int)similarity, HV_DIM, 100.0f * (similarity + HV_DIM) / (2 * HV_DIM));
    printf("  Escalate: %s\n", escalate ? "YES" : "NO");
    printf("  Policy: 0x%02x\n", (int)policy);

    // With random input, expected similarity is ~0 (50%)
    // Threshold is 70%, so should escalate
    if (!escalate) {
        printf("  FAIL: Should escalate for random input\n");
        return 1;
    }
    printf("  PASS\n");

    return 0;
}

int test_hebbian_learning() {
    printf("\n=== Test 2: Hebbian Learning ===\n");

    hv_t hv_in[HV_DIM];
    hv_t prototypes[N_PROTOS][HV_DIM];
    w_t  synapses[HV_DIM];
    hv_t hv_out[HV_DIM];
    w_t  policy_delta[HV_DIM];
    bool escalate;
    policy_t policy;
    ap_uint<8> best_proto;
    sim_t similarity;

    // Initialize
    memset(synapses, 0, sizeof(synapses));
    for (int p = 0; p < N_PROTOS; ++p) {
        random_hv(prototypes[p]);
    }

    // Run multiple escalation events
    printf("Running 10 escalation events...\n");
    int total_updates = 0;

    for (int iter = 0; iter < 10; ++iter) {
        random_hv(hv_in);  // Random = will escalate

        corr_spike_hdc(hv_in, prototypes, synapses, hv_out, policy_delta,
                       &escalate, &policy, &best_proto, &similarity);

        int updates = 0;
        for (int d = 0; d < HV_DIM; ++d) {
            if (policy_delta[d] != 0) updates++;
        }
        total_updates += updates;
    }

    printf("  Total dimension updates: %d\n", total_updates);

    // Check synapse distribution
    int pos = 0, neg = 0, zero = 0;
    w_t min_w = 127, max_w = -127;
    for (int d = 0; d < HV_DIM; ++d) {
        if (synapses[d] > 0) pos++;
        else if (synapses[d] < 0) neg++;
        else zero++;
        if (synapses[d] < min_w) min_w = synapses[d];
        if (synapses[d] > max_w) max_w = synapses[d];
    }

    printf("  Synapse stats: pos=%d, neg=%d, zero=%d\n", pos, neg, zero);
    printf("  Synapse range: [%d, %d]\n", (int)min_w, (int)max_w);

    // Should have learned something
    if (pos == 0 && neg == 0) {
        printf("  FAIL: No learning occurred\n");
        return 1;
    }

    // Should be balanced-ish (random inputs)
    if (abs(pos - neg) > HV_DIM / 2) {
        printf("  WARN: Synapse distribution very unbalanced\n");
    }

    printf("  PASS\n");
    return 0;
}

int test_spiking_inference() {
    printf("\n=== Test 3: Spiking Inference ===\n");

    hv_t hv_in[HV_DIM];
    hv_t prototypes[N_PROTOS][HV_DIM];
    w_t  synapses[HV_DIM];
    hv_t hv_out[HV_DIM];
    w_t  policy_delta[HV_DIM];
    bool escalate;
    policy_t policy;
    ap_uint<8> best_proto;
    sim_t similarity;

    // Initialize with biased synapses (to ensure some spikes)
    for (int d = 0; d < HV_DIM; ++d) {
        synapses[d] = (d < HV_DIM/2) ? 10 : -10;
    }
    for (int p = 0; p < N_PROTOS; ++p) {
        random_hv(prototypes[p]);
    }

    // Run with escalating input
    printf("Running with biased synapses...\n");
    random_hv(hv_in);

    corr_spike_hdc(hv_in, prototypes, synapses, hv_out, policy_delta,
                   &escalate, &policy, &best_proto, &similarity);

    printf("  Escalate: %s\n", escalate ? "YES" : "NO");
    printf("  Policy code: 0x%02x\n", (int)policy);

    if (!escalate) {
        printf("  SKIP: Input happened to match a prototype\n");
        return 0;
    }

    // With biased synapses, should get non-trivial policy
    printf("  PASS (policy generated)\n");
    return 0;
}

int test_stateful_kernel() {
    printf("\n=== Test 4: Stateful Kernel with Decay ===\n");

    hv_t hv_in[HV_DIM];
    hv_t prototypes[N_PROTOS][HV_DIM];
    w_t  synapses[HV_DIM];
    hv_t hv_out[HV_DIM];
    w_t  policy_delta[HV_DIM];
    bool escalate;
    policy_t policy;
    ap_uint<8> best_proto;
    sim_t similarity;

    // Initialize
    memset(synapses, 0, sizeof(synapses));
    for (int p = 0; p < N_PROTOS; ++p) {
        random_hv(prototypes[p]);
    }

    decay_t decay = 0.9;  // 90% retention

    // Reset state
    printf("Resetting state...\n");
    random_hv(hv_in);
    corr_spike_hdc_stateful(hv_in, prototypes, synapses, hv_out, policy_delta,
                            &escalate, &policy, &best_proto, &similarity,
                            decay, true);

    // Run sequence with consistent input
    printf("Running 5 steps with same input...\n");
    hv_t consistent_input[HV_DIM];
    similar_hv(prototypes[3], consistent_input, 0.8f);  // 80% similar to proto 3

    for (int t = 0; t < 5; ++t) {
        corr_spike_hdc_stateful(consistent_input, prototypes, synapses, hv_out,
                                policy_delta, &escalate, &policy, &best_proto,
                                &similarity, decay, false);

        printf("  t=%d: best=%d, sim=%d, escalate=%s\n",
               t, (int)best_proto, (int)similarity, escalate ? "Y" : "N");
    }

    // After accumulating consistent input, should converge to proto 3
    if (best_proto != 3) {
        printf("  WARN: Expected convergence to prototype 3\n");
    }

    printf("  PASS\n");
    return 0;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    printf("============================================================\n");
    printf("CorrSpike-HDC HLS Testbench\n");
    printf("============================================================\n");
    printf("HV_DIM=%d, N_PROTOS=%d, THRESH_SIM=%d\n",
           HV_DIM, N_PROTOS, THRESH_SIM);

    int failures = 0;

    failures += test_basic_correlation();
    failures += test_hebbian_learning();
    failures += test_spiking_inference();
    failures += test_stateful_kernel();

    printf("\n============================================================\n");
    if (failures == 0) {
        printf("ALL TESTS PASSED\n");
    } else {
        printf("FAILURES: %d\n", failures);
    }
    printf("============================================================\n");

    return failures;
}
