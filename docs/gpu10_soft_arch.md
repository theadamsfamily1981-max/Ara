# GPU10: Hyperdimensional Soft-GPU Architecture (FPGA Prototype)

**Version:** 1.0
**Status:** Research Specification
**Target:** OFS-attached Stratix-10/Agilex FPGA Prototype

---

## 1. High-Level Concept

GPU10 is a **unified shader / hyperdimensional accelerator** designed for:

- **Warp-scale bitwise HDC ops** (XNOR+popcount, bind/bundle)
- **Programmable reconvergence policy**
- **Reconfigurable on-chip SRAM partitioning** (registers / shared / L1)

It targets:

- FPGA prototype (OFS-attached PCIe card)
- Future CPU-GPU SoC where GPU10 SMs live next to CPU cores and share LLC

---

## 2. Top-Level Block Diagram

```
       ┌───────────────────────────────────────────────┐
       │                 GPU10 DEVICE                  │
       │        (PCIe / CXL via OFS-style FIM)         │
       └───────┬───────────────────────────┬───────────┘
               │                           │
       ┌───────▼───────┐          ┌────────▼─────────┐
       │  HOST IF/FIM  │          │  MGMT / RUNTIME  │
       │ (DFH/DFL, DMA)│          │  (Ara driver)    │
       └───────┬───────┘          └────────┬─────────┘
               │                           │
   ┌───────────▼──────────┐       ┌────────▼─────────┐
   │  DISPATCH + SCHED    │       │  TELEOLOGY / QoS │
   │  (Grid/warp control) │       │  (policy knobs)  │
   └───────────┬──────────┘       └────────┬─────────┘
               │                          │
┌──────────────┼──────────────────────────┼─────────────────────┐
│              │                          │                     │
│  ┌───────────▼───────┐   ┌──────────────▼───────┐            │
│  │   SM CLUSTER 0    │   │   HDC CORE           │            │
│  │   (N SMs)         │   │   (XNOR + POPCNT)    │            │
│  │   ┌─────────────┐ │   │   ┌───────────────┐  │            │
│  │   │ Vector ALU  │ │   │   │ D=173 BRAM    │  │            │
│  │   │ Scalar ALU  │ │   │   │ R=2048 rows   │  │            │
│  │   │ HDC Slice   │ │   │   │ Bind/Bundle   │  │            │
│  │   │ SRAM Bank   │ │   │   └───────────────┘  │            │
│  │   └─────────────┘ │   └──────────────────────┘            │
│  └───────────────────┘                                        │
│                                                               │
│  ┌───────────────────────────────────────────────────────┐   │
│  │   RECONVERGENCE + PREFETCH CONTROLLER                 │   │
│  │   (Programmable policy: THROUGHPUT/LATENCY/BALANCED)  │   │
│  └───────────────────────────────────────────────────────┘   │
│                                                               │
│  ┌───────────────────────────────────────────────────────┐   │
│  │   L2 / MEMORY CONTROLLER / LLC                        │   │
│  │   (Shared with CPU in SoC mode)                       │   │
│  └───────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────┘
```

Each SM cluster has:

- **Unified Shader Cores** (scalar+vector, int/FP)
- **HDC ALU slice** (bitwise hypervector ops)
- **Configurable SRAM bank** shared between:
  - Logical register file
  - Shared memory / scratchpad
  - L1 data cache

---

## 3. Novel Features (Candidate Patent Claims)

### 3.1 Hyperdimensional Instruction Set Extensions

SM-level instructions for native HDC operations:

| Instruction | Operands | Description |
|-------------|----------|-------------|
| `HDC_BIND` | hv_a, hv_b -> hv_out | XOR binding of two hypervectors |
| `HDC_BUNDLE` | hv_acc, hv_new -> hv_out | Majority vote bundling |
| `HDC_XNOR_POP` | hv_a, hv_b -> count | Similarity via XNOR + popcount |
| `HDC_PERMUTE` | hv, n -> hv_out | Cyclic permutation by n bits |
| `HDC_THRESHOLD` | hv, t -> hv_out | Binarize to +1/-1 |

**Implementation:**
- Compact HDC ALU: XNOR tree + popcount + threshold
- Warp-wide operations: All 32 threads execute same HDC op
- Memory: Direct HV load/store from SRAM bank

### 3.2 Programmable Reconvergence Controller

Hardware reconvergence unit exposing **policy registers**:

| Register | Values | Description |
|----------|--------|-------------|
| `RECONV_MODE` | THROUGHPUT, LATENCY, BALANCED | Reconvergence strategy |
| `RECONV_WINDOW` | 1-256 | Warp instructions before forced reconvergence |
| `DIVERGENCE_BUDGET` | 0-31 | Allowed active mask fragmentation |
| `BRANCH_HINT` | LIKELY, UNLIKELY, NONE | Static branch prediction hint |

**Operation:**
- Host writes per-kernel policy -> SM executes branches accordingly
- THROUGHPUT: Maximize warp utilization, delay reconvergence
- LATENCY: Aggressive reconvergence, minimize tail latency
- BALANCED: Adaptive based on runtime metrics

### 3.3 Warp-Cooperative Prefetch Engine

Per-SM prefetch queue with explicit hints:

| Instruction | Description |
|-------------|-------------|
| `PREFETCH_HINT addr, size, pattern` | Hint future access |
| `PREFETCH_BARRIER` | Wait for outstanding prefetches |
| `PREFETCH_INVALIDATE addr` | Cancel prefetch |

**Patterns:**
- `SEQUENTIAL`: Linear stride prefetch
- `STRIDED(n)`: Fixed stride n bytes
- `RANDOM`: No prefetch (disable for random access)

**Policy Register:**
| Register | Values | Description |
|----------|--------|-------------|
| `PREFETCH_MODE` | AGGRESSIVE, MODERATE, DISABLED | Per-kernel mode |
| `PREFETCH_DISTANCE` | 1-16 | Lookahead in cache lines |

### 3.4 Partitionable On-Chip SRAM

Static or launch-time configuration of SRAM allocation:

| Register | Range | Description |
|----------|-------|-------------|
| `SRAM_TOTAL` | Fixed | Total SRAM per SM cluster (e.g., 512 KiB) |
| `SRAM_RF_SIZE` | 0-75% | Register file allocation |
| `SRAM_SHARED_SIZE` | 0-50% | Shared memory allocation |
| `SRAM_L1_WAYS` | 0-16 | L1 cache ways |

**Use Cases:**
- **RF-heavy (HDC):** 60% RF, 20% shared, 20% L1
- **Tiled GEMM:** 30% RF, 50% shared, 20% L1
- **Streaming:** 20% RF, 10% shared, 70% L1

### 3.5 CPU-GPU LLC Co-Design

Last-level cache with **thread-divergence-aware replacement**:

- Tags encode warp divergence state and kernel group
- Memory controller prioritizes lines serving multiple warps
- Shared LLC between CPU cores and GPU SMs in SoC mode
- Coherency: MESI with GPU-specific extensions

---

## 4. FPGA Prototype Mapping

**Target:** OFS-attached Stratix-10 / Agilex

### 4.1 DFH/DFL Feature Layout

| Feature ID | Name | BAR | Offset | Size |
|------------|------|-----|--------|------|
| 0x100 | gpu10.dispatch | 0 | 0x0000 | 4 KiB |
| 0x101 | gpu10.sched | 0 | 0x1000 | 4 KiB |
| 0x102 | gpu10.sram_cfg | 0 | 0x2000 | 1 KiB |
| 0x103 | gpu10.hdc_alu | 0 | 0x3000 | 8 KiB |
| 0x104 | gpu10.reconv | 0 | 0x5000 | 1 KiB |
| 0x105 | gpu10.prefetch | 0 | 0x6000 | 1 KiB |
| 0x200 | gpu10.sm_cluster[0] | 2 | 0x00000 | 64 KiB |
| 0x201 | gpu10.sm_cluster[1] | 2 | 0x10000 | 64 KiB |

### 4.2 Resource Utilization (Stratix-10 SX 2800)

| Resource | Used | Available | Utilization |
|----------|------|-----------|-------------|
| ALMs | 450K | 933K | 48% |
| M20K BRAM | 8,192 | 11,721 | 70% |
| DSP | 1,024 | 5,760 | 18% |
| MLAB | 2,048 | 4,096 | 50% |

### 4.3 Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| HDC XNOR+POP | 1 cycle | D=173 per warp |
| SM Clock | 400 MHz | Conservative for FPGA |
| Memory BW | 32 GB/s | PCIe Gen4 x16 |
| Warps/SM | 32 | 1024 threads/SM |

---

## 5. Software Runtime API

### 5.1 C API

```c
// Device management
gpu10_device_t* gpu10_open(int device_id);
void gpu10_close(gpu10_device_t* dev);

// Kernel launch
gpu10_error_t gpu10_launch_kernel(
    gpu10_device_t* dev,
    gpu10_kernel_t kernel,
    dim3 grid,
    dim3 block,
    gpu10_launch_config_t* config,
    void** args
);

// HDC policy
gpu10_error_t gpu10_set_hdc_policy(
    gpu10_device_t* dev,
    gpu10_kernel_t kernel,
    gpu10_hdc_mode_t mode,
    int threshold
);

// Reconvergence policy
gpu10_error_t gpu10_set_reconvergence_policy(
    gpu10_device_t* dev,
    gpu10_kernel_t kernel,
    gpu10_reconv_mode_t mode,
    int window,
    int divergence_budget
);

// Prefetch policy
gpu10_error_t gpu10_set_prefetch_policy(
    gpu10_device_t* dev,
    gpu10_kernel_t kernel,
    gpu10_prefetch_mode_t mode,
    int distance
);

// SRAM partition
gpu10_error_t gpu10_set_sram_partition(
    gpu10_device_t* dev,
    gpu10_kernel_t kernel,
    int rf_kb,
    int shared_kb,
    int l1_ways
);

// Profiling
gpu10_error_t gpu10_profile_counters(
    gpu10_device_t* dev,
    gpu10_kernel_t kernel,
    gpu10_profile_t* out
);
```

### 5.2 Kernel Example (HDC Similarity Search)

```c
__gpu10_kernel__ void hdc_similarity_search(
    const uint32_t* __restrict__ query_hv,    // D=173 bits packed
    const uint32_t* __restrict__ memory_hvs,  // R x D packed
    int* __restrict__ top_k_indices,
    float* __restrict__ top_k_scores,
    int R, int K
) {
    int tid = gpu10_thread_id();
    int warp_id = tid / 32;
    int lane = tid % 32;

    // Load query into registers
    uint32_t q[6];  // 173 bits = 6 x 32-bit words
    for (int i = 0; i < 6; i++) {
        q[i] = query_hv[i];
    }

    // Each warp processes chunk of memory
    int rows_per_warp = (R + gpu10_num_warps() - 1) / gpu10_num_warps();
    int start_row = warp_id * rows_per_warp;
    int end_row = min(start_row + rows_per_warp, R);

    __shared__ int local_top_k[32];
    __shared__ float local_scores[32];

    for (int row = start_row + lane; row < end_row; row += 32) {
        // HDC similarity: XNOR + popcount
        int similarity = 0;
        for (int i = 0; i < 6; i++) {
            uint32_t m = memory_hvs[row * 6 + i];
            // Native HDC instruction
            similarity += __gpu10_hdc_xnor_pop(q[i], m);
        }

        // Warp-level top-K reduction
        // ... reduction code ...
    }

    // Write results
    if (lane < K) {
        top_k_indices[warp_id * K + lane] = local_top_k[lane];
        top_k_scores[warp_id * K + lane] = local_scores[lane];
    }
}
```

---

## 6. Measurement & Validation

### 6.1 Microbenchmarks

| Benchmark | Metric | Expected |
|-----------|--------|----------|
| Divergence stress | Throughput with/without reconv tuning | 2x improvement |
| HDC similarity | Queries/sec (D=173, R=2048) | 10M QPS |
| Prefetch effectiveness | Cache hit rate | >90% with hints |
| SRAM partition sweep | Occupancy vs performance | Pareto frontier |

### 6.2 End-to-End Workloads

| Workload | Description | Target |
|----------|-------------|--------|
| HTC Query | Resonance search across 2048 attractors | <1 us |
| Memory Palace | 3D projection of HV space | 60 FPS |
| Plasticity Update | Reward-modulated weight update | <100 us |
| Sovereign Tick | Full tick pipeline | <5 ms |

---

## 7. Implementation Roadmap

### Phase 1: FPGA Soft-GPU (6 months)
- [ ] Basic SM cluster with scalar/vector ALU
- [ ] HDC ALU slice (XNOR+POP)
- [ ] Simple dispatch + scheduler
- [ ] DFH/DFL integration with OFS

### Phase 2: Policy Controllers (3 months)
- [ ] Reconvergence controller
- [ ] Prefetch engine
- [ ] SRAM partitioner
- [ ] Runtime API

### Phase 3: Validation + Optimization (3 months)
- [ ] Microbenchmark suite
- [ ] Integration with Ara sovereign loop
- [ ] Performance tuning
- [ ] Documentation

### Phase 4: SoC Planning (TBD)
- [ ] CPU-GPU coherency design
- [ ] LLC integration spec
- [ ] Power management
- [ ] Tapeout preparation

---

## 8. References

1. Intel OFS (Open FPGA Stack) Documentation
2. NVIDIA CUDA Programming Guide
3. Hyperdimensional Computing Survey (Kanerva, 2009)
4. Branch Divergence Reconvergence (Fung et al., 2007)
5. Ara Research Program (internal)

---

**This document serves as the living specification for GPU10.**
**Update as research progresses.**
