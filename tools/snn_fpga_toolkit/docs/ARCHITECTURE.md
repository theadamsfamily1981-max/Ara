# SNN Kernel Architecture

## Overview

The SNN-Optimized Kernel is designed to maximize performance for Spiking Neural Networks through efficient GPU-FPGA integration. This document describes the architecture, design decisions, and implementation details.

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Application Layer                       │
│  ┌────────────┐  ┌────────────┐  ┌──────────────────────┐ │
│  │ SNN        │  │ Training   │  │  Management          │ │
│  │ Framework  │  │ Framework  │  │  Applications        │ │
│  └─────┬──────┘  └─────┬──────┘  └──────────┬───────────┘ │
└────────┼────────────────┼────────────────────┼──────────────┘
         │                │                    │
         └────────────────┴────────────────────┘
                          │
┌─────────────────────────┼──────────────────────────────────┐
│            User-Space API (libsnn.so)                      │
│                         │                                   │
│  ┌──────────────────────▼────────────────────────────┐    │
│  │  snn_kernel_initialize()                          │    │
│  │  snn_alloc_pinned()                               │    │
│  │  snn_p2p_transfer()                               │    │
│  │  snn_set_rt_params()                              │    │
│  │  snn_compute()                                    │    │
│  │  snn_nvme_read/write()                            │    │
│  └───────────────────────────────────────────────────┘    │
└────────────────────────┬───────────────────────────────────┘
                         │ (ioctl)
┌────────────────────────┼───────────────────────────────────┐
│                  Kernel Space                              │
│  ┌──────────────────────▼────────────────────────────┐    │
│  │           SNN Core Module (/dev/snn)              │    │
│  │         Character Device + IOCTL Handler          │    │
│  └─┬──────┬──────┬──────┬──────┬──────┬──────┬──────┘    │
│    │      │      │      │      │      │      │            │
│  ┌─▼──┐ ┌─▼──┐ ┌─▼──┐ ┌─▼──┐ ┌─▼──┐ ┌─▼──┐ ┌─▼──┐       │
│  │PCIe│ │Mem │ │RT  │ │CUDA│ │NVMe│ │SNN │ │Mon │       │
│  │P2P │ │Mgr │ │Sch │ │Brg │ │DIO │ │Pipe│ │Dbg │       │
│  └─┬──┘ └─┬──┘ └─┬──┘ └─┬──┘ └─┬──┘ └─┬──┘ └─┬──┘       │
└────┼──────┼──────┼──────┼──────┼──────┼──────┼───────────┘
     │      │      │      │      │      │      │
┌────▼──────▼──────▼──────▼──────▼──────▼──────▼───────────┐
│                    Hardware Layer                         │
│  ┌──────┐  ┌──────────┐  ┌──────┐  ┌──────┐  ┌──────┐  │
│  │ CPU  │  │  Memory  │  │ GPU  │  │ FPGA │  │ NVMe │  │
│  └──────┘  └──────────┘  └──┬───┘  └──┬───┘  └──────┘  │
│                             │         │                  │
│                             └────┬────┘                  │
│                            PCIe 5.0 Bus                   │
└───────────────────────────────────────────────────────────┘
```

## Core Components

### 1. PCIe Subsystem (kernel/pcie/)

**Purpose**: Manages PCIe 5.0 devices and implements peer-to-peer data transfers.

**Key Features**:
- Device discovery and enumeration
- PCIe capability detection (generation, lanes, bandwidth)
- P2P transfer management
- DMA coordination
- IOMMU awareness

**Design Decisions**:
- Uses standard Linux PCI driver framework
- Supports relaxed ordering for P2P performance
- Handles multiple GPU and FPGA devices
- Implements async transfer with completion tracking

**Performance Considerations**:
- PCIe 5.0: 32 GT/s per lane (theoretical: ~64 GB/s for x16)
- Real-world: 50-60 GB/s achievable
- Latency: ~1-2 microseconds for P2P

### 2. Memory Management (kernel/memory/)

**Purpose**: Provides pinned memory allocation for deterministic access.

**Key Features**:
- Page-aligned memory allocation
- Memory pinning (prevents swapping)
- Multi-device access coordination
- Memory pool management
- Physical address tracking

**Design Decisions**:
- Pre-allocated memory pools for low-latency allocation
- Page reservation to prevent swapping
- Support for huge pages (2MB/1GB) for reduced TLB misses
- Reference counting for shared memory regions

**Memory Layout**:
```
Virtual Memory              Physical Memory
┌────────────────┐         ┌────────────────┐
│  User Space    │         │                │
│  Application   │ ◄─map─► │  Pinned Pages  │
├────────────────┤         │                │
│  Kernel Space  │ ◄─────► │  (Locked)      │
│  SNN Driver    │         │                │
├────────────────┤         ├────────────────┤
│  GPU Memory    │ ◄─P2P─► │  GPU BAR       │
├────────────────┤         ├────────────────┤
│  FPGA Memory   │ ◄─P2P─► │  FPGA BAR      │
└────────────────┘         └────────────────┘
```

### 3. Real-Time Scheduler (kernel/rt_sched/)

**Purpose**: Ensures time-critical SNN computations meet deadlines.

**Key Features**:
- SCHED_FIFO policy with priority 1-99
- CPU affinity management
- Deadline tracking and monitoring
- Deadline miss detection

**Design Decisions**:
- Integrates with Linux RT scheduling
- Uses highest priorities (90-99) for critical paths
- Per-task deadline monitoring
- Statistics for deadline misses

**Priority Levels**:
- 99: Critical interrupt handlers
- 90-95: High-priority SNN computations
- 50-89: Normal SNN operations
- 1-49: Background tasks

### 4. CUDA Bridge (kernel/cuda_bridge/)

**Purpose**: Interfaces kernel space with CUDA runtime.

**Key Features**:
- GPU memory mapping to kernel space
- CUDA stream management
- GPU synchronization
- GPUDirect support

**Integration Points**:
- NVIDIA UVM (Unified Virtual Memory) driver
- CUDA Driver API
- GPUDirect RDMA
- Peer memory API

**Note**: Full CUDA integration requires NVIDIA proprietary drivers and APIs.

### 5. NVMe Direct I/O (kernel/nvme_dio/)

**Purpose**: High-speed storage access for training data.

**Key Features**:
- Direct I/O (bypasses page cache)
- Pinned buffer support
- Async I/O with completion tracking
- High queue depth (128-1024)

**Performance**:
- PCIe 4.0 NVMe: ~7 GB/s sequential
- PCIe 5.0 NVMe: ~14 GB/s sequential
- Direct I/O reduces CPU overhead

### 6. SNN Pipeline (kernel/snn_pipeline/)

**Purpose**: Orchestrates SNN computations across GPU and FPGA.

**Key Features**:
- Workload partitioning
- GPU-FPGA coordination
- Result merging
- Batch processing optimization

**Workload Distribution**:
```
SNN Network (N neurons, S synapses)
         │
         ▼
   ┌─────────┐
   │Partition│
   └────┬────┘
        │
   ┌────┴────┐
   ▼         ▼
  GPU       FPGA
 (60%)     (40%)
   │         │
   │ Compute │
   │         │
   └────┬────┘
        ▼
    ┌───────┐
    │ Merge │
    └───┬───┘
        ▼
     Output
```

## Data Flow

### Typical SNN Training Iteration

1. **Data Loading** (NVMe → Pinned Memory)
   - Direct I/O read from NVMe
   - Load spike trains, weights into pinned memory
   - Time: ~10-50ms for 1GB

2. **Data Distribution** (CPU → GPU/FPGA)
   - P2P transfer to GPU memory
   - P2P transfer to FPGA memory
   - Time: ~20ms for 1GB @ 50GB/s

3. **Computation** (GPU + FPGA)
   - GPU: Dense layer computations
   - FPGA: Spike processing
   - Parallel execution
   - Time: ~50-200ms (model dependent)

4. **Result Collection** (GPU/FPGA → CPU)
   - P2P transfer results back
   - Merge and post-process
   - Time: ~10ms

5. **Weight Update** (CPU)
   - Apply gradients
   - Update learning parameters
   - Time: ~5-10ms

**Total Iteration**: ~100-300ms
**Throughput**: 3-10 iterations/second

## Performance Optimization

### PCIe Bandwidth

- Use P2P transfers to bypass CPU
- Enable relaxed ordering
- Batch small transfers
- Pipeline transfers with computation

### Memory Access

- Pre-allocate pinned memory pools
- Use huge pages (2MB/1GB)
- Minimize TLB misses
- Align to cache lines (64 bytes)

### Real-Time Performance

- Isolate CPUs for RT tasks (isolcpus)
- Disable CPU frequency scaling
- Use RT throttling carefully
- Monitor deadline misses

### GPU-FPGA Coordination

- Overlap computation where possible
- Use multiple CUDA streams
- Pipeline stages
- Minimize synchronization points

## Security Considerations

### Kernel Module Security

- All IOCTLs validate inputs
- Memory access bounds checking
- Reference counting prevents use-after-free
- CAP_SYS_ADMIN required for sensitive operations

### Memory Protection

- Pinned memory isolated per process
- DMA buffers validated
- IOMMU protection when available
- Prevents cross-process access

### Real-Time Guarantees

- Priority limits prevent starvation
- Watchdog for runaway tasks
- Deadline monitoring
- Resource accounting

## Scalability

### Multiple Devices

- Supports multiple GPUs and FPGAs
- Per-device command queues
- Load balancing across devices
- Fault isolation

### Large Models

- Memory pooling reduces allocation overhead
- Streaming for models larger than GPU memory
- Batch processing
- Model parallelism support

### High Throughput

- Multiple concurrent streams
- Async operations throughout
- Lock-free data structures where possible
- Per-CPU data structures

## Future Extensions

1. **Multi-GPU Support**: Scale to multiple GPUs with NVLink
2. **Advanced FPGA Features**: Custom SNN kernels on FPGA
3. **Network Integration**: RDMA for multi-node training
4. **Power Management**: Dynamic voltage/frequency scaling
5. **Profiling**: Integrated performance profiling
6. **Auto-tuning**: Automatic workload partitioning

## References

- PCIe 5.0 Base Specification
- Linux Kernel Documentation (DMA, PCI, RT)
- NVIDIA CUDA Programming Guide
- NVIDIA GPUDirect Documentation
- NVMe Specification
- Spiking Neural Network literature
