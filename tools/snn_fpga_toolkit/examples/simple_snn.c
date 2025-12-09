/*
 * Simple SNN Example
 *
 * Demonstrates basic usage of the SNN kernel API
 */

#include <snn_kernel/api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NUM_NEURONS 10000
#define NUM_SYNAPSES 1000000
#define TIMESTEPS 100

/* Get current time in nanoseconds */
static uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

int main(int argc, char *argv[]) {
    int ret;
    uint64_t start_time, end_time;

    printf("=== SNN Kernel Example ===\n\n");

    /* Initialize kernel */
    printf("Initializing SNN kernel...\n");
    snn_kernel_init_t config = {
        .gpu_id = 0,
        .fpga_id = 0,
        .pinned_mem_size = 2ULL * 1024 * 1024 * 1024,  // 2GB
        .rt_priority = 80,
        .enable_monitoring = 1,
        .enable_debugging = 0,
        .max_p2p_streams = 4,
        .nvme_queue_depth = 128
    };

    ret = snn_kernel_initialize(&config);
    if (ret < 0) {
        fprintf(stderr, "Failed to initialize: %s\n", snn_get_error_string());
        fprintf(stderr, "Make sure kernel module is loaded: sudo modprobe snn_kernel_core\n");
        return 1;
    }
    printf("Initialized successfully\n\n");

    /* Query device information */
    printf("Device Information:\n");
    snn_device_info_t gpu_info;
    if (snn_get_device_info(SNN_DEV_GPU, 0, &gpu_info) == 0) {
        printf("  GPU: %s\n", gpu_info.name);
        printf("    Memory: %llu MB\n", gpu_info.caps.memory_size / (1024*1024));
        printf("    PCIe: Gen%u x%u (%.1f GB/s)\n",
               gpu_info.caps.pcie_gen,
               gpu_info.caps.pcie_lanes,
               gpu_info.caps.max_bandwidth_mbps / 1000.0);
    }

    snn_device_info_t fpga_info;
    if (snn_get_device_info(SNN_DEV_FPGA, 0, &fpga_info) == 0) {
        printf("  FPGA: %s\n", fpga_info.name);
        printf("    Compute Units: %u\n", fpga_info.caps.compute_units);
    }
    printf("\n");

    /* Allocate memory */
    printf("Allocating memory...\n");
    size_t weight_size = NUM_SYNAPSES * sizeof(float);
    size_t input_size = NUM_NEURONS * TIMESTEPS * sizeof(char);
    size_t output_size = NUM_NEURONS * sizeof(float);

    void *weights = snn_alloc_pinned(weight_size, SNN_MEM_GPU | SNN_MEM_FPGA);
    void *input = snn_alloc_pinned(input_size, SNN_MEM_GPU | SNN_MEM_FPGA);
    void *output = snn_alloc_pinned(output_size, SNN_MEM_GPU | SNN_MEM_FPGA | SNN_MEM_ZERO);

    if (!weights || !input || !output) {
        fprintf(stderr, "Memory allocation failed: %s\n", snn_get_error_string());
        goto cleanup;
    }

    printf("  Weights: %.2f MB\n", weight_size / (1024.0 * 1024.0));
    printf("  Input: %.2f MB\n", input_size / (1024.0 * 1024.0));
    printf("  Output: %.2f MB\n", output_size / (1024.0 * 1024.0));
    printf("\n");

    /* Initialize with random data */
    printf("Initializing data...\n");
    float *w = (float *)weights;
    for (size_t i = 0; i < NUM_SYNAPSES; i++) {
        w[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;  // Random [-1, 1]
    }

    char *inp = (char *)input;
    for (size_t i = 0; i < NUM_NEURONS * TIMESTEPS; i++) {
        inp[i] = (rand() % 100) < 10 ? 1 : 0;  // 10% spike rate
    }
    printf("Data initialized\n\n");

    /* Configure computation */
    snn_compute_params_t params = {
        .num_neurons = NUM_NEURONS,
        .num_synapses = NUM_SYNAPSES,
        .timesteps = TIMESTEPS,
        .batch_size = 1,
        .use_gpu = 1,
        .use_fpga = 1,
        .weight_matrix_addr = (uint64_t)weights,
        .spike_train_addr = (uint64_t)input,
        .output_addr = (uint64_t)output
    };

    /* Execute SNN computation */
    printf("Running SNN inference (%u neurons, %u timesteps)...\n",
           NUM_NEURONS, TIMESTEPS);

    start_time = get_time_ns();
    ret = snn_compute(&params);
    end_time = get_time_ns();

    if (ret < 0) {
        fprintf(stderr, "Computation failed: %s\n", snn_get_error_string());
        goto cleanup;
    }

    double elapsed_ms = (end_time - start_time) / 1e6;
    printf("Computation completed in %.2f ms\n\n", elapsed_ms);

    /* Get performance statistics */
    snn_perf_stats_t stats;
    snn_get_stats(&stats);

    printf("Performance Statistics:\n");
    printf("  P2P Transfers: %llu\n", stats.p2p_transfers);
    printf("  P2P Data: %.2f MB\n", stats.p2p_bytes_transferred / (1024.0 * 1024.0));
    if (stats.p2p_transfers > 0) {
        printf("  Avg P2P Latency: %llu ns\n", stats.avg_p2p_latency_ns);
    }
    printf("  Pinned Memory: %.2f MB\n", stats.pinned_mem_allocated / (1024.0 * 1024.0));
    printf("  RT Tasks: %llu\n", stats.rt_tasks_executed);
    if (stats.rt_deadline_misses > 0) {
        printf("  Deadline Misses: %llu (%.2f%%)\n",
               stats.rt_deadline_misses,
               (double)stats.rt_deadline_misses / stats.rt_tasks_executed * 100.0);
    }
    printf("\n");

    /* Verify output (simple check) */
    float *out = (float *)output;
    size_t nonzero = 0;
    for (size_t i = 0; i < NUM_NEURONS; i++) {
        if (out[i] != 0.0f)
            nonzero++;
    }
    printf("Results: %zu/%u neurons active (%.1f%%)\n",
           nonzero, NUM_NEURONS, (double)nonzero / NUM_NEURONS * 100.0);

    ret = 0;

cleanup:
    /* Cleanup */
    printf("\nCleaning up...\n");
    if (weights) snn_free_pinned(weights);
    if (input) snn_free_pinned(input);
    if (output) snn_free_pinned(output);

    snn_kernel_shutdown();
    printf("Done.\n");

    return ret;
}
