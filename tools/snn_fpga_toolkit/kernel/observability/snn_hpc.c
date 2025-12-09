/*
 * Hardware Performance Counter (HPC) Implementation
 *
 * Low-overhead performance monitoring using perf_events
 * Target: <500ns collection latency
 */

#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/slab.h>
#include <linux/perf_event.h>
#include <linux/time.h>
#include "snn_hpc.h"

/*
 * perf_event configuration for CPU counters
 */
static struct perf_event_attr snn_hpc_cpu_attrs[] = {
	[SNN_HPC_CPU_CYCLES] = {
		.type = PERF_TYPE_HARDWARE,
		.config = PERF_COUNT_HW_CPU_CYCLES,
		.size = sizeof(struct perf_event_attr),
		.pinned = 1,
		.disabled = 0,
		.exclude_kernel = 0,
		.exclude_hv = 1,
	},
	[SNN_HPC_CPU_INSTRUCTIONS] = {
		.type = PERF_TYPE_HARDWARE,
		.config = PERF_COUNT_HW_INSTRUCTIONS,
		.size = sizeof(struct perf_event_attr),
		.pinned = 0,
		.disabled = 0,
		.exclude_kernel = 0,
		.exclude_hv = 1,
	},
	[SNN_HPC_CACHE_REFERENCES] = {
		.type = PERF_TYPE_HARDWARE,
		.config = PERF_COUNT_HW_CACHE_REFERENCES,
		.size = sizeof(struct perf_event_attr),
		.pinned = 0,
		.disabled = 0,
		.exclude_kernel = 0,
		.exclude_hv = 1,
	},
	[SNN_HPC_CACHE_MISSES] = {
		.type = PERF_TYPE_HARDWARE,
		.config = PERF_COUNT_HW_CACHE_MISSES,
		.size = sizeof(struct perf_event_attr),
		.pinned = 0,
		.disabled = 0,
		.exclude_kernel = 0,
		.exclude_hv = 1,
	},
	[SNN_HPC_BRANCH_MISSES] = {
		.type = PERF_TYPE_HARDWARE,
		.config = PERF_COUNT_HW_BRANCH_MISSES,
		.size = sizeof(struct perf_event_attr),
		.pinned = 0,
		.disabled = 0,
		.exclude_kernel = 0,
		.exclude_hv = 1,
	},
};

#define SNN_HPC_NUM_CPU_COUNTERS (sizeof(snn_hpc_cpu_attrs) / sizeof(snn_hpc_cpu_attrs[0]))

/*
 * Initialize HPC monitoring
 */
int snn_hpc_init(struct snn_hpc_monitor **monitor_ptr)
{
	struct snn_hpc_monitor *monitor;
	int cpu, i, ret;

	monitor = kzalloc(sizeof(*monitor), GFP_KERNEL);
	if (!monitor)
		return -ENOMEM;

	spin_lock_init(&monitor->lock);
	atomic64_set(&monitor->total_samples, 0);
	atomic64_set(&monitor->total_overhead_ns, 0);

	/* Create perf_events on current CPU
	 * In production, might create on all CPUs or specific RT CPU
	 */
	cpu = smp_processor_id();

	pr_info("SNN_HPC: Initializing perf_events on CPU %d\n", cpu);

	monitor->num_cpu_events = 0;

	for (i = 0; i < SNN_HPC_NUM_CPU_COUNTERS; i++) {
		struct perf_event *event;

		event = snn_hpc_create_event(&snn_hpc_cpu_attrs[i], cpu);
		if (IS_ERR(event)) {
			pr_warn("SNN_HPC: Failed to create event %d: %ld\n",
			        i, PTR_ERR(event));
			monitor->cpu_events[i] = NULL;
			continue;
		}

		monitor->cpu_events[i] = event;
		monitor->num_cpu_events++;

		pr_debug("SNN_HPC: Created counter %d (type=%u, config=%llu)\n",
		         i, snn_hpc_cpu_attrs[i].type, snn_hpc_cpu_attrs[i].config);
	}

	if (monitor->num_cpu_events == 0) {
		pr_err("SNN_HPC: No perf_events created - initialization failed\n");
		kfree(monitor);
		return -ENODEV;
	}

	pr_info("SNN_HPC: Initialized %d CPU performance counters\n",
	        monitor->num_cpu_events);

	/* GPU and FPGA hooks will be registered later via callbacks */
	monitor->gpu_handle = NULL;
	monitor->gpu_read_counters = NULL;
	monitor->fpga_handle = NULL;
	monitor->fpga_read_counters = NULL;

	monitor->initialized = true;
	*monitor_ptr = monitor;

	return 0;
}

/*
 * Cleanup HPC monitoring
 */
void snn_hpc_cleanup(struct snn_hpc_monitor *monitor)
{
	int i;

	if (!monitor)
		return;

	pr_info("SNN_HPC: Cleaning up...\n");

	/* Release perf_events */
	for (i = 0; i < SNN_HPC_MAX_COUNTERS; i++) {
		if (monitor->cpu_events[i]) {
			perf_event_release_kernel(monitor->cpu_events[i]);
			monitor->cpu_events[i] = NULL;
		}
	}

	kfree(monitor);

	pr_info("SNN_HPC: Cleanup complete\n");
}

/*
 * Collect CPU metrics (fast path)
 */
static inline void snn_hpc_collect_cpu(struct snn_hpc_monitor *monitor,
                                       struct snn_hpc_sample *sample)
{
	int i;

	sample->timestamp = ktime_get_ns();

	for (i = 0; i < SNN_HPC_MAX_COUNTERS; i++) {
		if (monitor->cpu_events[i]) {
			sample->values[i] = snn_hpc_read_event(monitor->cpu_events[i]);
			sample->valid[i] = true;
		} else {
			sample->values[i] = 0;
			sample->valid[i] = false;
		}
	}
}

/*
 * Collect GPU metrics (if registered)
 */
static inline int snn_hpc_collect_gpu(struct snn_hpc_monitor *monitor,
                                      struct snn_gpu_metrics *gpu)
{
	if (!monitor->gpu_read_counters)
		return -ENODEV;

	return monitor->gpu_read_counters(monitor->gpu_handle, gpu);
}

/*
 * Collect FPGA metrics (if registered)
 */
static inline int snn_hpc_collect_fpga(struct snn_hpc_monitor *monitor,
                                       struct snn_fpga_metrics *fpga)
{
	if (!monitor->fpga_read_counters)
		return -ENODEV;

	return monitor->fpga_read_counters(monitor->fpga_handle, fpga);
}

/*
 * Collect current system metrics
 *
 * Ultra-fast collection of all performance counters
 */
int snn_hpc_collect(struct snn_hpc_monitor *monitor,
                    struct snn_system_metrics *metrics)
{
	u64 start_time, end_time;
	int ret;

	if (!monitor || !monitor->initialized)
		return -EINVAL;

	start_time = ktime_get_ns();

	/* Zero out metrics */
	memset(metrics, 0, sizeof(*metrics));

	/* Collect CPU counters (fastest) */
	snn_hpc_collect_cpu(monitor, &metrics->cpu_sample);

	/* Collect GPU counters (if available) */
	ret = snn_hpc_collect_gpu(monitor, &metrics->gpu);
	if (ret < 0 && ret != -ENODEV) {
		pr_debug("SNN_HPC: GPU counter collection failed: %d\n", ret);
	}

	/* Collect FPGA counters (if available) */
	ret = snn_hpc_collect_fpga(monitor, &metrics->fpga);
	if (ret < 0 && ret != -ENODEV) {
		pr_debug("SNN_HPC: FPGA counter collection failed: %d\n", ret);
	}

	/* Calculate Arithmetic Intensity */
	snn_hpc_calculate_ai(metrics, &metrics->ai);

	end_time = ktime_get_ns();
	metrics->collection_time_ns = end_time - start_time;

	/* Update statistics */
	atomic64_inc(&monitor->total_samples);
	atomic64_add(metrics->collection_time_ns, &monitor->total_overhead_ns);

	/* Cache for next delta calculation */
	memcpy(&monitor->last_sample, metrics, sizeof(*metrics));
	monitor->last_timestamp = start_time;

	return 0;
}

/*
 * Calculate Arithmetic Intensity from metrics
 *
 * AI = FLOPs / memory_bytes_accessed
 */
void snn_hpc_calculate_ai(const struct snn_system_metrics *metrics,
                          struct snn_arithmetic_intensity *ai)
{
	u64 flops = 0;
	u64 mem_bytes = 0;
	u64 cache_refs, cache_misses;

	/* Estimate FLOPs from CPU instructions (rough approximation) */
	if (metrics->cpu_sample.valid[SNN_HPC_CPU_INSTRUCTIONS]) {
		u64 instructions = metrics->cpu_sample.values[SNN_HPC_CPU_INSTRUCTIONS];
		flops += snn_hpc_estimate_flops(instructions,
		                                metrics->cpu_sample.values[SNN_HPC_CPU_CYCLES]);
	}

	/* Add GPU FLOPs (much more accurate) */
	flops += metrics->gpu.flops;

	/* Calculate memory bytes accessed
	 * Estimate from cache misses (each miss = cache line fetch)
	 */
	cache_refs = metrics->cpu_sample.values[SNN_HPC_CACHE_REFERENCES];
	cache_misses = metrics->cpu_sample.values[SNN_HPC_CACHE_MISSES];

	mem_bytes = cache_misses * SNN_HPC_CACHE_LINE_SIZE;

	/* Add GPU memory accesses */
	mem_bytes += metrics->gpu.global_load_bytes + metrics->gpu.global_store_bytes;

	/* Add FPGA memory accesses */
	mem_bytes += metrics->fpga.axi_read_bytes + metrics->fpga.axi_write_bytes;

	/* Calculate AI ratio (× 1000 for precision) */
	if (mem_bytes > 0) {
		ai->ai_ratio = (u32)((flops * 1000) / mem_bytes);
	} else {
		ai->ai_ratio = 0;
	}

	ai->flops = flops;
	ai->mem_bytes = mem_bytes;

	/* Calculate cache hit rate */
	ai->cache_hit_rate = snn_hpc_cache_hit_rate(cache_refs, cache_misses);

	/* Memory bandwidth utilization (if we have timing data) */
	/* This is simplified - real implementation would track bandwidth */
	ai->mem_bandwidth_util = 0;

	pr_debug("SNN_HPC: AI = %u.%03u FLOPs/byte (FLOPs=%llu, MemBytes=%llu)\n",
	         ai->ai_ratio / 1000, ai->ai_ratio % 1000,
	         flops, mem_bytes);
	pr_debug("SNN_HPC: Cache hit rate = %u.%02u%%\n",
	         ai->cache_hit_rate / 100, ai->cache_hit_rate % 100);
}

/*
 * Register GPU monitoring callbacks
 */
int snn_hpc_register_gpu(struct snn_hpc_monitor *monitor,
                         void *gpu_handle,
                         int (*read_fn)(void *, struct snn_gpu_metrics *))
{
	if (!monitor || !read_fn)
		return -EINVAL;

	spin_lock(&monitor->lock);
	monitor->gpu_handle = gpu_handle;
	monitor->gpu_read_counters = read_fn;
	spin_unlock(&monitor->lock);

	pr_info("SNN_HPC: GPU monitoring registered\n");

	return 0;
}

/*
 * Register FPGA monitoring callbacks
 */
int snn_hpc_register_fpga(struct snn_hpc_monitor *monitor,
                          void *fpga_handle,
                          int (*read_fn)(void *, struct snn_fpga_metrics *))
{
	if (!monitor || !read_fn)
		return -EINVAL;

	spin_lock(&monitor->lock);
	monitor->fpga_handle = fpga_handle;
	monitor->fpga_read_counters = read_fn;
	spin_unlock(&monitor->lock);

	pr_info("SNN_HPC: FPGA monitoring registered\n");

	return 0;
}

/*
 * Get statistics on HPC overhead
 */
void snn_hpc_get_stats(struct snn_hpc_monitor *monitor,
                       u64 *total_samples,
                       u64 *avg_overhead_ns)
{
	u64 samples, overhead;

	if (!monitor) {
		*total_samples = 0;
		*avg_overhead_ns = 0;
		return;
	}

	samples = atomic64_read(&monitor->total_samples);
	overhead = atomic64_read(&monitor->total_overhead_ns);

	*total_samples = samples;
	*avg_overhead_ns = (samples > 0) ? (overhead / samples) : 0;

	pr_debug("SNN_HPC: Stats - samples=%llu, avg_overhead=%llu ns\n",
	         samples, *avg_overhead_ns);
}

MODULE_LICENSE("GPL");
MODULE_AUTHOR("SNN Kernel Team");
MODULE_DESCRIPTION("Hardware Performance Counter Integration");
