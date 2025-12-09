/*
 * SNN Kernel Monitor
 *
 * Real-time monitoring tool for SNN kernel performance
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <time.h>

#include "snn_kernel/api.h"

static volatile int running = 1;

/* Signal handler */
static void sigint_handler(int sig)
{
    (void)sig;
    running = 0;
}

/* Format bytes to human-readable */
static void format_bytes(char *buf, size_t len, __u64 bytes)
{
    const char *units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit = 0;
    double value = bytes;

    while (value >= 1024.0 && unit < 4) {
        value /= 1024.0;
        unit++;
    }

    snprintf(buf, len, "%.2f %s", value, units[unit]);
}

/* Display statistics */
static void display_stats(const snn_perf_stats_t *stats, __u64 elapsed_ns)
{
    char buf[64];
    double elapsed_sec = elapsed_ns / 1e9;

    printf("\033[2J\033[H"); /* Clear screen */
    printf("=== SNN Kernel Performance Monitor ===\n\n");

    /* P2P Statistics */
    printf("PCIe P2P Transfers:\n");
    printf("  Total Transfers: %llu\n", stats->p2p_transfers);
    format_bytes(buf, sizeof(buf), stats->p2p_bytes_transferred);
    printf("  Total Data:      %s\n", buf);
    if (elapsed_sec > 0) {
        double throughput = stats->p2p_bytes_transferred / elapsed_sec;
        format_bytes(buf, sizeof(buf), (__u64)throughput);
        printf("  Throughput:      %s/s\n", buf);
    }
    printf("  Avg Latency:     %llu ns\n", stats->avg_p2p_latency_ns);
    printf("\n");

    /* Memory Statistics */
    printf("Pinned Memory:\n");
    format_bytes(buf, sizeof(buf), stats->pinned_mem_allocated);
    printf("  Allocated:       %s\n", buf);
    format_bytes(buf, sizeof(buf), stats->pinned_mem_peak);
    printf("  Peak Usage:      %s\n", buf);
    printf("\n");

    /* RT Scheduler Statistics */
    printf("Real-Time Scheduler:\n");
    printf("  Tasks Executed:  %llu\n", stats->rt_tasks_executed);
    printf("  Deadline Misses: %llu", stats->rt_deadline_misses);
    if (stats->rt_tasks_executed > 0) {
        double miss_rate = (double)stats->rt_deadline_misses /
                          stats->rt_tasks_executed * 100.0;
        printf(" (%.2f%%)", miss_rate);
    }
    printf("\n\n");

    /* NVMe Statistics */
    printf("NVMe Direct I/O:\n");
    printf("  Reads:           %llu\n", stats->nvme_reads);
    printf("  Writes:          %llu\n", stats->nvme_writes);
    format_bytes(buf, sizeof(buf), stats->nvme_bytes_transferred);
    printf("  Total Data:      %s\n", buf);
    if (elapsed_sec > 0) {
        double throughput = stats->nvme_bytes_transferred / elapsed_sec;
        format_bytes(buf, sizeof(buf), (__u64)throughput);
        printf("  Throughput:      %s/s\n", buf);
    }
    printf("\n");

    printf("Press Ctrl+C to exit\n");
}

/* Get timestamp in nanoseconds */
static __u64 get_time_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (__u64)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

int main(int argc, char *argv[])
{
    snn_kernel_init_t config;
    snn_perf_stats_t stats, prev_stats;
    __u64 start_time, current_time;
    int ret;
    int interval = 1000; /* 1 second default */

    /* Parse arguments */
    if (argc > 1) {
        interval = atoi(argv[1]);
        if (interval <= 0)
            interval = 1000;
    }

    printf("SNN Kernel Monitor starting...\n");

    /* Initialize SNN kernel */
    memset(&config, 0, sizeof(config));
    config.gpu_id = 0;
    config.fpga_id = 0;
    config.pinned_mem_size = 0; /* Don't allocate pool in monitor */
    config.rt_priority = 0;
    config.enable_monitoring = 1;
    config.max_p2p_streams = 16;
    config.nvme_queue_depth = 128;

    ret = snn_kernel_initialize(&config);
    if (ret < 0) {
        fprintf(stderr, "Failed to initialize SNN kernel: %s\n",
                snn_get_error_string());
        fprintf(stderr, "Make sure the kernel module is loaded: sudo modprobe snn_kernel_core\n");
        return 1;
    }

    /* Setup signal handler */
    signal(SIGINT, sigint_handler);

    /* Initialize statistics */
    memset(&prev_stats, 0, sizeof(prev_stats));
    start_time = get_time_ns();

    /* Monitor loop */
    while (running) {
        current_time = get_time_ns();

        /* Get current statistics */
        ret = snn_get_stats(&stats);
        if (ret < 0) {
            fprintf(stderr, "Failed to get stats: %s\n",
                    snn_get_error_string());
            break;
        }

        /* Display statistics */
        display_stats(&stats, current_time - start_time);

        /* Save for next iteration */
        prev_stats = stats;

        /* Sleep */
        usleep(interval * 1000);
    }

    printf("\nShutting down monitor...\n");
    snn_kernel_shutdown();

    return 0;
}
