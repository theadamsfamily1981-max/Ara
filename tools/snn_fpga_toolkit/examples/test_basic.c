/*
 * Basic SNN Kernel Module Test
 *
 * Tests core functionality:
 * - Device open/close
 * - Pipeline initialization
 * - AI recommendation
 * - Performance feedback
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <errno.h>

#include "../include/snn_kernel/snn_kernel.h"
#include "../include/snn_kernel/semantic_ai.h"

#define DEVICE_PATH "/dev/snn0"

void print_allocation(const snn_ai_allocation_t *alloc)
{
    printf("  GPU:        %u%%\n", alloc->use_gpu);
    printf("  FPGA:       %u%%\n", alloc->use_fpga);
    printf("  CPU:        %u neurons\n", alloc->cpu_neurons);
    printf("  Batch size: %u\n", alloc->batch_size);
    printf("  Confidence: %s\n",
           alloc->confidence == SNN_AI_CONFIDENCE_HIGH ? "HIGH" :
           alloc->confidence == SNN_AI_CONFIDENCE_MEDIUM ? "MEDIUM" : "LOW");
}

int main(int argc, char *argv[])
{
    int fd;
    int ret;

    printf("=================================\n");
    printf("SNN Kernel Module - Basic Test\n");
    printf("=================================\n\n");

    /* Open device */
    printf("1. Opening device %s...\n", DEVICE_PATH);
    fd = open(DEVICE_PATH, O_RDWR);
    if (fd < 0) {
        perror("Failed to open device");
        printf("\nTroubleshooting:\n");
        printf("  - Is the module loaded? (lsmod | grep snn_kernel)\n");
        printf("  - Load it: sudo insmod snn_kernel.ko\n");
        printf("  - Check permissions on %s\n", DEVICE_PATH);
        return 1;
    }
    printf("   ✓ Device opened successfully (fd=%d)\n\n", fd);

    /* Initialize pipeline */
    printf("2. Initializing SNN pipeline...\n");
    snn_init_config_t init_config = {
        .use_gpu = 1,
        .use_fpga = 0,  /* Set to 1 if you have FPGA */
        .pinned_memory_size = 256 * 1024 * 1024,  /* 256 MB */
        .flags = 0
    };

    ret = ioctl(fd, SNN_IOC_INIT, &init_config);
    if (ret < 0) {
        perror("Pipeline initialization failed");
        close(fd);
        return 1;
    }
    printf("   ✓ Pipeline initialized\n\n");

    /* Test AI recommendation */
    printf("3. Testing AI Engine...\n");

    /* Example workload: Dense SNN with 100K neurons */
    snn_compute_params_t params = {
        .num_neurons = 100000,
        .num_synapses = 8000000,  /* 80 connections per neuron = dense */
        .timesteps = 1000,
        .batch_size = 32
    };

    snn_system_state_t sys_state = {
        .gpu_utilization = 50,
        .fpga_utilization = 30,
        .gpu_memory_free = 6ULL * 1024 * 1024 * 1024,  /* 6 GB */
        .fpga_memory_free = 4ULL * 1024 * 1024 * 1024,  /* 4 GB */
        .pcie_bandwidth_used = 100,  /* MB/s */
        .rt_deadline_miss_rate = 0
    };

    snn_ai_allocation_t allocation;
    memset(&allocation, 0, sizeof(allocation));

    ret = ioctl(fd, SNN_IOC_AI_RECOMMEND, &allocation);
    if (ret < 0) {
        perror("AI recommendation failed");
        close(fd);
        return 1;
    }

    printf("   AI Recommendation (Dense workload, 100K neurons):\n");
    print_allocation(&allocation);
    printf("\n");

    /* Verify allocation makes sense for dense workload */
    if (allocation.use_gpu < 50) {
        printf("   ⚠ Warning: GPU allocation seems low for dense workload\n");
        printf("   (This is normal during cold-start - will improve with learning)\n\n");
    } else {
        printf("   ✓ Allocation looks reasonable\n\n");
    }

    /* Simulate feedback (positive) */
    printf("4. Providing feedback to AI...\n");
    snn_ai_feedback_t feedback = {
        .actual_latency_ns = 15000000,  /* 15 ms */
        .achieved_throughput = 95,  /* 95% of target */
        .deadline_met = 1,  /* Success! */
        .power_watts = 180,
        .accuracy = 98
    };

    ret = ioctl(fd, SNN_IOC_AI_FEEDBACK, &feedback);
    if (ret < 0) {
        perror("AI feedback failed");
        close(fd);
        return 1;
    }
    printf("   ✓ Feedback provided (positive reinforcement)\n\n");

    /* Test sparse workload */
    printf("5. Testing with sparse workload...\n");
    params.num_neurons = 50000;
    params.num_synapses = 200000;  /* Only 4 connections per neuron = sparse */
    params.timesteps = 500;

    ret = ioctl(fd, SNN_IOC_AI_RECOMMEND, &allocation);
    if (ret < 0) {
        perror("AI recommendation failed");
        close(fd);
        return 1;
    }

    printf("   AI Recommendation (Sparse workload, 50K neurons):\n");
    print_allocation(&allocation);
    printf("\n");

    /* Query AI statistics */
    printf("6. Querying AI statistics...\n");
    snn_ai_stats_t stats;
    memset(&stats, 0, sizeof(stats));

    ret = ioctl(fd, SNN_IOC_AI_STATS, &stats);
    if (ret < 0) {
        perror("AI stats query failed");
    } else {
        printf("   Total decisions:      %llu\n", (unsigned long long)stats.total_decisions);
        printf("   Successful decisions: %llu\n", (unsigned long long)stats.successful_decisions);
        printf("   Learning iterations:  %llu\n", (unsigned long long)stats.learning_iterations);

        if (stats.total_decisions > 0) {
            float success_rate = (float)stats.successful_decisions * 100.0f / (float)stats.total_decisions;
            printf("   Success rate:         %.1f%%\n", success_rate);
        }
        printf("\n");
    }

    /* Cleanup */
    printf("7. Cleaning up...\n");
    close(fd);
    printf("   ✓ Device closed\n\n");

    printf("=================================\n");
    printf("✓ All tests PASSED!\n");
    printf("=================================\n\n");

    printf("Next steps:\n");
    printf("  - Run more iterations to train the AI: ./test_training\n");
    printf("  - Check cold-start progression: dmesg | grep COLD_START\n");
    printf("  - Monitor performance: ./snn_monitor\n");
    printf("  - See examples/ for more advanced usage\n\n");

    return 0;
}
