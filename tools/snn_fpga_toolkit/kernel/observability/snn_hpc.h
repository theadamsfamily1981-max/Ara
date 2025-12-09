/*
 * Hardware Performance Counter (HPC) Integration
 *
 * Provides low-overhead (<500ns) access to real-time system metrics:
 * - CPU: cache misses, memory loads/stores, FLOPs (via perf_events)
 * - GPU: SM utilization, memory bandwidth, compute throughput
 * - FPGA: resource utilization, memory access patterns
 *
 * Used by AI engine for accurate Arithmetic Intensity calculation
 * and resource-aware decision making.
 */

#ifndef _SNN_HPC_H
#define _SNN_HPC_H

#include <linux/types.h>
#include <linux/perf_event.h>
#include <linux/hw_breakpoint.h>
#include <linux/spinlock.h>

/*
 * Performance counter types
 */
enum snn_hpc_counter_type {
	SNN_HPC_CPU_CYCLES = 0,
	SNN_HPC_CPU_INSTRUCTIONS,
	SNN_HPC_CACHE_MISSES,
	SNN_HPC_CACHE_REFERENCES,
	SNN_HPC_MEM_LOADS,
	SNN_HPC_MEM_STORES,
	SNN_HPC_BRANCH_MISSES,
	SNN_HPC_GPU_SM_ACTIVE,
	SNN_HPC_GPU_MEM_READ,
	SNN_HPC_GPU_MEM_WRITE,
	SNN_HPC_GPU_COMPUTE_UTIL,
	SNN_HPC_FPGA_UTIL,
	SNN_HPC_FPGA_MEM_BANDWIDTH,
	SNN_HPC_MAX_COUNTERS
};

/*
 * Counter sample structure
 */
struct snn_hpc_sample {
	u64 timestamp;           /* Nanoseconds since boot */
	u64 values[SNN_HPC_MAX_COUNTERS];
	bool valid[SNN_HPC_MAX_COUNTERS];
};

/*
 * Arithmetic Intensity metrics
 * AI = FLOPs / (memory_bytes_accessed)
 */
struct snn_arithmetic_intensity {
	u64 flops;               /* Floating-point operations */
	u64 mem_bytes;           /* Memory bytes accessed */
	u32 ai_ratio;            /* AI × 1000 (e.g., 2500 = 2.5) */
	u32 cache_hit_rate;      /* Cache hit rate × 100 (0-10000) */
	u32 mem_bandwidth_util;  /* Memory bandwidth utilization % */
};

/*
 * GPU-specific metrics (from device driver)
 */
struct snn_gpu_metrics {
	u32 sm_active_cycles;    /* SM active cycles % */
	u32 sm_occupancy;        /* Warp occupancy % */
	u64 global_load_bytes;   /* Global memory load throughput */
	u64 global_store_bytes;  /* Global memory store throughput */
	u32 shared_mem_util;     /* Shared memory utilization % */
	u32 l2_hit_rate;         /* L2 cache hit rate % */
	u32 mem_clock_util;      /* Memory clock utilization % */
	u64 flops;               /* FP operations (FP32 + 2×FP16) */
};

/*
 * FPGA-specific metrics
 */
struct snn_fpga_metrics {
	u32 lut_utilization;     /* LUT utilization % */
	u32 dsp_utilization;     /* DSP block utilization % */
	u32 bram_utilization;    /* Block RAM utilization % */
	u64 axi_read_bytes;      /* AXI read throughput */
	u64 axi_write_bytes;     /* AXI write throughput */
	u32 clock_freq_mhz;      /* Current clock frequency */
	u32 power_watts;         /* Power consumption */
};

/*
 * Unified system metrics
 */
struct snn_system_metrics {
	struct snn_hpc_sample cpu_sample;
	struct snn_gpu_metrics gpu;
	struct snn_fpga_metrics fpga;
	struct snn_arithmetic_intensity ai;

	u64 pcie_bandwidth_used; /* PCIe bandwidth used (bytes/sec) */
	u32 rt_deadline_misses;  /* Real-time deadline misses */

	u64 collection_time_ns;  /* Time to collect metrics */
};

/*
 * HPC monitor structure
 */
struct snn_hpc_monitor {
	/* perf_event handles for CPU counters */
	struct perf_event *cpu_events[SNN_HPC_MAX_COUNTERS];
	int num_cpu_events;

	/* GPU monitoring hooks */
	void *gpu_handle;
	int (*gpu_read_counters)(void *handle, struct snn_gpu_metrics *metrics);

	/* FPGA monitoring hooks */
	void *fpga_handle;
	int (*fpga_read_counters)(void *handle, struct snn_fpga_metrics *metrics);

	/* Cache for last sample (to compute deltas) */
	struct snn_system_metrics last_sample;
	u64 last_timestamp;

	/* Statistics */
	atomic64_t total_samples;
	atomic64_t total_overhead_ns;

	spinlock_t lock;
	bool initialized;
};

/*
 * Initialize HPC monitoring
 *
 * Sets up perf_events for CPU counters and registers GPU/FPGA hooks
 * Returns 0 on success, negative errno on failure
 */
int snn_hpc_init(struct snn_hpc_monitor **monitor);

/*
 * Cleanup HPC monitoring
 */
void snn_hpc_cleanup(struct snn_hpc_monitor *monitor);

/*
 * Collect current system metrics
 *
 * Ultra-fast collection (<500ns target) of all performance counters
 * Returns 0 on success, negative errno on failure
 */
int snn_hpc_collect(struct snn_hpc_monitor *monitor,
                    struct snn_system_metrics *metrics);

/*
 * Calculate Arithmetic Intensity from metrics
 *
 * AI = FLOPs / memory_bytes_accessed
 * Higher AI means compute-bound workload (good for GPU)
 * Lower AI means memory-bound workload (may benefit from FPGA)
 */
void snn_hpc_calculate_ai(const struct snn_system_metrics *metrics,
                          struct snn_arithmetic_intensity *ai);

/*
 * Register GPU monitoring callbacks
 */
int snn_hpc_register_gpu(struct snn_hpc_monitor *monitor,
                         void *gpu_handle,
                         int (*read_fn)(void *, struct snn_gpu_metrics *));

/*
 * Register FPGA monitoring callbacks
 */
int snn_hpc_register_fpga(struct snn_hpc_monitor *monitor,
                          void *fpga_handle,
                          int (*read_fn)(void *, struct snn_fpga_metrics *));

/*
 * Get statistics on HPC overhead
 */
void snn_hpc_get_stats(struct snn_hpc_monitor *monitor,
                       u64 *total_samples,
                       u64 *avg_overhead_ns);

/*
 * Helper: perf_event creation wrapper
 */
static inline struct perf_event *
snn_hpc_create_event(struct perf_event_attr *attr, int cpu)
{
	return perf_event_create_kernel_counter(attr, cpu, NULL, NULL, NULL);
}

/*
 * Helper: Read perf_event value
 * Fast read (<50ns) of single counter
 */
static inline u64 snn_hpc_read_event(struct perf_event *event)
{
	u64 enabled, running, value;

	if (!event)
		return 0;

	/* Fast path: direct read */
	value = perf_event_read_value(event, &enabled, &running);

	/* Adjust for multiplexing if needed */
	if (enabled != running && running != 0)
		value = (value * enabled) / running;

	return value;
}

/*
 * Helper: Cache line size
 */
#define SNN_HPC_CACHE_LINE_SIZE 64

/*
 * Helper: Convert cache references/misses to hit rate
 */
static inline u32 snn_hpc_cache_hit_rate(u64 references, u64 misses)
{
	if (references == 0)
		return 10000;  /* 100.00% (no misses) */

	if (misses > references)
		misses = references;

	u64 hits = references - misses;
	return (u32)((hits * 10000) / references);
}

/*
 * Helper: Estimate FLOPs from instructions (very rough)
 * Better to use GPU/FPGA specific counters
 */
static inline u64 snn_hpc_estimate_flops(u64 instructions, u64 cycles)
{
	/* Assume ~10% of instructions are FP operations
	 * This is a crude estimate - real HW counters are better
	 */
	return instructions / 10;
}

/*
 * Helper: Calculate memory bandwidth utilization
 * Returns percentage (0-100)
 */
static inline u32 snn_hpc_mem_bandwidth_util(u64 mem_bytes,
                                              u64 time_ns,
                                              u64 max_bandwidth_gbps)
{
	if (time_ns == 0)
		return 0;

	/* bytes/sec = mem_bytes * 1e9 / time_ns */
	u64 bytes_per_sec = (mem_bytes * 1000000000ULL) / time_ns;

	/* max_bandwidth in bytes/sec */
	u64 max_bytes_per_sec = max_bandwidth_gbps * 1000000000ULL / 8;

	if (max_bytes_per_sec == 0)
		return 0;

	u32 util = (u32)((bytes_per_sec * 100) / max_bytes_per_sec);
	return (util > 100) ? 100 : util;
}

#endif /* _SNN_HPC_H */
