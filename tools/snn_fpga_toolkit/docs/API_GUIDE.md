# API Guide

Complete guide to using the SNN Kernel API for developing high-performance Spiking Neural Network applications.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Initialization](#initialization)
3. [Memory Management](#memory-management)
4. [P2P Transfers](#p2p-transfers)
5. [Real-Time Scheduling](#real-time-scheduling)
6. [SNN Computation](#snn-computation)
7. [NVMe I/O](#nvme-io)
8. [Performance Monitoring](#performance-monitoring)
9. [Error Handling](#error-handling)
10. [Complete Examples](#complete-examples)

## Getting Started

### Installation

```bash
# Build and install kernel module
make
sudo make install

# Build user-space library
make api
sudo make -C api install

# Load kernel module
sudo modprobe snn_kernel_core debug_level=1
```

### Linking Your Application

```bash
gcc -o my_app my_app.c -lsnn
```

Or with CMake:
```cmake
target_link_libraries(my_app snn)
```

### Header Files

```c
#include <snn_kernel/api.h>
```

## Initialization

### Basic Initialization

```c
#include <snn_kernel/api.h>
#include <stdio.h>

int main() {
    snn_kernel_init_t config = {
        .gpu_id = 0,
        .fpga_id = 0,
        .pinned_mem_size = 4ULL * 1024 * 1024 * 1024,  // 4GB
        .rt_priority = 90,
        .enable_monitoring = 1,
        .enable_debugging = 0,
        .max_p2p_streams = 16,
        .nvme_queue_depth = 128
    };

    int ret = snn_kernel_initialize(&config);
    if (ret < 0) {
        fprintf(stderr, "Init failed: %s\n", snn_get_error_string());
        return 1;
    }

    printf("SNN Kernel initialized successfully\n");

    // ... your application code ...

    snn_kernel_shutdown();
    return 0;
}
```

### Query Device Information

```c
snn_device_info_t gpu_info;
if (snn_get_device_info(SNN_DEV_GPU, 0, &gpu_info) == 0) {
    printf("GPU: %s\n", gpu_info.name);
    printf("  Memory: %llu MB\n", gpu_info.caps.memory_size / (1024*1024));
    printf("  PCIe: Gen%u x%u\n",
           gpu_info.caps.pcie_gen, gpu_info.caps.pcie_lanes);
    printf("  P2P: %s\n",
           gpu_info.caps.supports_p2p ? "Supported" : "Not supported");
}

snn_device_info_t fpga_info;
if (snn_get_device_info(SNN_DEV_FPGA, 0, &fpga_info) == 0) {
    printf("FPGA: %s\n", fpga_info.name);
    printf("  Compute Units: %u\n", fpga_info.caps.compute_units);
}
```

## Memory Management

### Allocate Pinned Memory

```c
// Allocate 1GB of pinned memory accessible by GPU and FPGA
size_t size = 1ULL * 1024 * 1024 * 1024;
void *buffer = snn_alloc_pinned(size, SNN_MEM_GPU | SNN_MEM_FPGA);

if (!buffer) {
    fprintf(stderr, "Allocation failed: %s\n", snn_get_error_string());
    return -1;
}

// Use the buffer
memset(buffer, 0, size);

// Free when done
snn_free_pinned(buffer);
```

### Memory Flags

```c
// Different memory types
void *gpu_only = snn_alloc_pinned(size, SNN_MEM_GPU);
void *fpga_only = snn_alloc_pinned(size, SNN_MEM_FPGA);
void *cpu_gpu_fpga = snn_alloc_pinned(size,
                                      SNN_MEM_CPU | SNN_MEM_GPU | SNN_MEM_FPGA);

// Zeroed memory
void *zeroed = snn_alloc_pinned(size, SNN_MEM_GPU | SNN_MEM_ZERO);

// DMA-capable memory
void *dma_buf = snn_alloc_pinned(size, SNN_MEM_GPU | SNN_MEM_DMA);
```

### Get Physical Address

```c
void *buffer = snn_alloc_pinned(size, SNN_MEM_GPU);
__u64 phys_addr = snn_get_physical_addr(buffer);
printf("Virtual: %p, Physical: 0x%llx\n", buffer, phys_addr);
```

## P2P Transfers

### Synchronous Transfer

```c
// Prepare data on CPU
float *cpu_data = (float *)snn_alloc_pinned(size, SNN_MEM_CPU);
// ... fill cpu_data ...

// Transfer to GPU
snn_p2p_transfer_t transfer = {
    .src_dev = SNN_DEV_CPU,
    .dst_dev = SNN_DEV_GPU,
    .src_addr = (u64)cpu_data,
    .dst_addr = 0,  // Kernel will allocate
    .size = size,
    .flags = SNN_TRANSFER_BLOCKING
};

int ret = snn_p2p_transfer(&transfer);
if (ret < 0) {
    fprintf(stderr, "Transfer failed: %s\n", snn_get_error_string());
}

printf("Transfer complete, GPU addr: 0x%llx\n", transfer.dst_addr);
```

### Asynchronous Transfer

```c
// Start async transfer
transfer.flags = SNN_TRANSFER_ASYNC;
transfer.stream_id = 0;

ret = snn_p2p_transfer(&transfer);
if (ret < 0) {
    fprintf(stderr, "Transfer start failed\n");
    return -1;
}

u64 handle = transfer.completion_handle;

// Do other work while transfer is in progress
// ...

// Wait for completion (timeout: 1 second)
ret = snn_p2p_wait(handle, 1000000000ULL);
if (ret < 0) {
    fprintf(stderr, "Transfer timeout\n");
}
```

### GPU to FPGA Transfer

```c
snn_p2p_transfer_t transfer = {
    .src_dev = SNN_DEV_GPU,
    .dst_dev = SNN_DEV_FPGA,
    .src_addr = gpu_buffer_addr,
    .dst_addr = fpga_buffer_addr,
    .size = data_size,
    .flags = SNN_TRANSFER_P2P | SNN_TRANSFER_ASYNC,
    .stream_id = 1
};

snn_p2p_transfer(&transfer);
```

### Bidirectional Transfer

```c
// CPU to GPU
snn_p2p_transfer_t to_gpu = {
    .src_dev = SNN_DEV_CPU,
    .dst_dev = SNN_DEV_GPU,
    .src_addr = (u64)host_input,
    .dst_addr = gpu_input_addr,
    .size = input_size,
    .flags = SNN_TRANSFER_ASYNC,
    .stream_id = 0
};
snn_p2p_transfer(&to_gpu);

// Process on GPU
// ...

// GPU to CPU
snn_p2p_transfer_t from_gpu = {
    .src_dev = SNN_DEV_GPU,
    .dst_dev = SNN_DEV_CPU,
    .src_addr = gpu_output_addr,
    .dst_addr = (u64)host_output,
    .size = output_size,
    .flags = SNN_TRANSFER_ASYNC,
    .stream_id = 1
};
snn_p2p_transfer(&from_gpu);

// Wait for both
snn_p2p_wait(to_gpu.completion_handle, 0);
snn_p2p_wait(from_gpu.completion_handle, 0);
```

## Real-Time Scheduling

### Set RT Parameters for Current Thread

```c
#include <pthread.h>

void* rt_worker(void *arg) {
    // Set RT parameters
    snn_rt_sched_params_t params = {
        .priority = 95,              // High priority
        .cpu_affinity = 0x10,        // CPU 4
        .deadline_ns = 10000000,     // 10ms deadline
        .period_ns = 20000000,       // 20ms period
        .preemptible = 0             // Non-preemptible
    };

    int ret = snn_set_rt_params(&params);
    if (ret < 0) {
        fprintf(stderr, "Failed to set RT params: %s\n",
                snn_get_error_string());
        return NULL;
    }

    // RT processing loop
    while (running) {
        // Time-critical work
        process_snn_batch();
    }

    return NULL;
}

int main() {
    pthread_t thread;
    pthread_create(&thread, NULL, rt_worker, NULL);
    pthread_join(thread, NULL);
    return 0;
}
```

### Multi-Thread RT Configuration

```c
// Critical path thread: highest priority
snn_rt_sched_params_t critical = {
    .priority = 95,
    .cpu_affinity = 0x10,  // CPU 4
    .deadline_ns = 1000000  // 1ms
};

// Normal processing thread: medium priority
snn_rt_sched_params_t normal = {
    .priority = 70,
    .cpu_affinity = 0x60,  // CPUs 5-6
    .deadline_ns = 10000000  // 10ms
};

// Background thread: low priority
snn_rt_sched_params_t background = {
    .priority = 30,
    .cpu_affinity = 0x0F,  // CPUs 0-3
    .deadline_ns = 0  // No hard deadline
};
```

## SNN Computation

### Basic SNN Inference

```c
// Allocate memory for network
size_t num_neurons = 10000;
size_t num_synapses = 1000000;
size_t timesteps = 100;

void *weights = snn_alloc_pinned(
    num_synapses * sizeof(float),
    SNN_MEM_GPU | SNN_MEM_FPGA
);

void *spike_train = snn_alloc_pinned(
    num_neurons * timesteps * sizeof(char),
    SNN_MEM_GPU | SNN_MEM_FPGA
);

void *output = snn_alloc_pinned(
    num_neurons * timesteps * sizeof(float),
    SNN_MEM_GPU | SNN_MEM_FPGA
);

// Configure computation
snn_compute_params_t params = {
    .num_neurons = num_neurons,
    .num_synapses = num_synapses,
    .timesteps = timesteps,
    .batch_size = 1,
    .use_gpu = 1,
    .use_fpga = 1,
    .weight_matrix_addr = (u64)weights,
    .spike_train_addr = (u64)spike_train,
    .output_addr = (u64)output
};

// Execute
int ret = snn_compute(&params);
if (ret < 0) {
    fprintf(stderr, "Computation failed: %s\n", snn_get_error_string());
}

// Read results
// ... process output ...

// Cleanup
snn_free_pinned(weights);
snn_free_pinned(spike_train);
snn_free_pinned(output);
```

### Batch Processing

```c
// Process multiple samples
for (int batch = 0; batch < num_batches; batch++) {
    // Load batch data
    load_batch_data(spike_train, batch);

    // Process
    params.batch_size = batch_size;
    snn_compute(&params);

    // Save results
    save_batch_results(output, batch);
}
```

### GPU-Only vs FPGA-Only vs Hybrid

```c
// GPU only (best for dense operations)
params.use_gpu = 1;
params.use_fpga = 0;

// FPGA only (best for sparse, event-driven)
params.use_gpu = 0;
params.use_fpga = 1;

// Hybrid (automatic partitioning)
params.use_gpu = 1;
params.use_fpga = 1;
```

## NVMe I/O

### Direct I/O Read

```c
// Allocate pinned buffer (required for direct I/O)
size_t io_size = 4096 * 1024;  // 4MB
void *buffer = snn_alloc_pinned(io_size, SNN_MEM_PINNED);

snn_nvme_io_t io = {
    .offset = 0,
    .size = io_size,
    .buffer_addr = (u64)buffer,
    .flags = 0,  // Synchronous
    .queue_id = 0
};

int ret = snn_nvme_read(&io);
if (ret < 0) {
    fprintf(stderr, "NVMe read failed: %s\n", snn_get_error_string());
}

// Data is now in buffer
process_data(buffer, io_size);
```

### Async I/O with Multiple Requests

```c
#define NUM_REQUESTS 8
snn_nvme_io_t ios[NUM_REQUESTS];
u64 handles[NUM_REQUESTS];

// Submit multiple reads
for (int i = 0; i < NUM_REQUESTS; i++) {
    ios[i].offset = i * io_size;
    ios[i].size = io_size;
    ios[i].buffer_addr = (u64)buffers[i];
    ios[i].flags = SNN_TRANSFER_ASYNC;

    snn_nvme_read(&ios[i]);
    handles[i] = ios[i].completion_handle;
}

// Wait for all to complete
for (int i = 0; i < NUM_REQUESTS; i++) {
    snn_nvme_wait(handles[i], 5000000000ULL);  // 5 second timeout
}
```

### Direct I/O Write

```c
snn_nvme_io_t io = {
    .offset = 0,
    .size = io_size,
    .buffer_addr = (u64)buffer,
    .flags = 0
};

int ret = snn_nvme_write(&io);
```

## Performance Monitoring

### Get Statistics

```c
snn_perf_stats_t stats;
snn_get_stats(&stats);

printf("Performance Statistics:\n");
printf("  P2P Transfers: %llu\n", stats.p2p_transfers);
printf("  P2P Bytes: %llu\n", stats.p2p_bytes_transferred);
printf("  P2P Avg Latency: %llu ns\n", stats.avg_p2p_latency_ns);
printf("  Pinned Memory: %llu bytes\n", stats.pinned_mem_allocated);
printf("  RT Tasks: %llu\n", stats.rt_tasks_executed);
printf("  Deadline Misses: %llu\n", stats.rt_deadline_misses);
printf("  NVMe Reads: %llu\n", stats.nvme_reads);
printf("  NVMe Writes: %llu\n", stats.nvme_writes);
```

### Continuous Monitoring

```c
while (running) {
    snn_get_stats(&stats);

    // Calculate throughput
    double p2p_throughput = stats.p2p_bytes_transferred / elapsed_time;
    printf("P2P Throughput: %.2f GB/s\n", p2p_throughput / 1e9);

    // Check deadline miss rate
    if (stats.rt_tasks_executed > 0) {
        double miss_rate = (double)stats.rt_deadline_misses /
                          stats.rt_tasks_executed;
        if (miss_rate > 0.01) {
            fprintf(stderr, "WARNING: High deadline miss rate: %.2f%%\n",
                    miss_rate * 100);
        }
    }

    sleep(1);
}
```

### Reset Statistics

```c
// Reset counters (useful between benchmark runs)
snn_reset_stats();
```

## Error Handling

### Check Return Values

```c
int ret = snn_p2p_transfer(&transfer);
if (ret < 0) {
    fprintf(stderr, "Error: %s\n", snn_get_error_string());

    // Handle specific errors
    switch (-ret) {
    case EINVAL:
        fprintf(stderr, "Invalid parameters\n");
        break;
    case ENOMEM:
        fprintf(stderr, "Out of memory\n");
        break;
    case ETIMEDOUT:
        fprintf(stderr, "Operation timed out\n");
        break;
    default:
        fprintf(stderr, "Unknown error: %d\n", ret);
    }
}
```

### Graceful Shutdown

```c
void cleanup_and_exit(int signum) {
    printf("Shutting down...\n");

    // Free resources
    snn_free_pinned(buffer);

    // Shutdown kernel
    snn_kernel_shutdown();

    exit(0);
}

int main() {
    signal(SIGINT, cleanup_and_exit);
    signal(SIGTERM, cleanup_and_exit);

    // ... application code ...
}
```

## Complete Examples

### Example 1: Simple SNN Inference

```c
#include <snn_kernel/api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUM_NEURONS 10000
#define NUM_SYNAPSES 1000000
#define TIMESTEPS 100

int main() {
    // Initialize
    snn_kernel_init_t config = {
        .gpu_id = 0,
        .fpga_id = 0,
        .pinned_mem_size = 2ULL * 1024 * 1024 * 1024,
        .rt_priority = 80,
        .enable_monitoring = 1,
        .max_p2p_streams = 4,
        .nvme_queue_depth = 128
    };

    if (snn_kernel_initialize(&config) < 0) {
        fprintf(stderr, "Init failed\n");
        return 1;
    }

    // Allocate memory
    void *weights = snn_alloc_pinned(
        NUM_SYNAPSES * sizeof(float),
        SNN_MEM_GPU | SNN_MEM_FPGA
    );

    void *input = snn_alloc_pinned(
        NUM_NEURONS * TIMESTEPS,
        SNN_MEM_GPU | SNN_MEM_FPGA
    );

    void *output = snn_alloc_pinned(
        NUM_NEURONS * sizeof(float),
        SNN_MEM_GPU | SNN_MEM_FPGA
    );

    // Load weights and input
    // ... (load from file or generate) ...

    // Execute
    snn_compute_params_t params = {
        .num_neurons = NUM_NEURONS,
        .num_synapses = NUM_SYNAPSES,
        .timesteps = TIMESTEPS,
        .batch_size = 1,
        .use_gpu = 1,
        .use_fpga = 1,
        .weight_matrix_addr = (u64)weights,
        .spike_train_addr = (u64)input,
        .output_addr = (u64)output
    };

    printf("Running SNN inference...\n");
    if (snn_compute(&params) < 0) {
        fprintf(stderr, "Compute failed\n");
        goto cleanup;
    }

    // Get statistics
    snn_perf_stats_t stats;
    snn_get_stats(&stats);
    printf("Computation complete!\n");
    printf("  P2P transfers: %llu\n", stats.p2p_transfers);

cleanup:
    snn_free_pinned(weights);
    snn_free_pinned(input);
    snn_free_pinned(output);
    snn_kernel_shutdown();

    return 0;
}
```

### Example 2: High-Performance Data Pipeline

See `examples/snn_pipeline.c` for a complete high-performance training pipeline example.

## API Reference Summary

| Function | Description |
|----------|-------------|
| `snn_kernel_initialize()` | Initialize SNN kernel |
| `snn_kernel_shutdown()` | Shutdown SNN kernel |
| `snn_get_device_info()` | Query device capabilities |
| `snn_alloc_pinned()` | Allocate pinned memory |
| `snn_free_pinned()` | Free pinned memory |
| `snn_get_physical_addr()` | Get physical address |
| `snn_p2p_transfer()` | Perform P2P transfer |
| `snn_p2p_wait()` | Wait for transfer completion |
| `snn_set_rt_params()` | Set RT scheduling parameters |
| `snn_compute()` | Execute SNN computation |
| `snn_nvme_read()` | Read from NVMe |
| `snn_nvme_write()` | Write to NVMe |
| `snn_nvme_wait()` | Wait for I/O completion |
| `snn_get_stats()` | Get performance statistics |
| `snn_reset_stats()` | Reset statistics |
| `snn_get_error_string()` | Get last error message |

## Best Practices

1. **Always check return values**
2. **Free all allocated memory**
3. **Use pinned memory for device access**
4. **Set appropriate RT priorities**
5. **Monitor performance statistics**
6. **Handle signals for graceful shutdown**
7. **Align buffers for direct I/O**
8. **Use async operations for overlap**
9. **Batch operations when possible**
10. **Profile before optimizing**
