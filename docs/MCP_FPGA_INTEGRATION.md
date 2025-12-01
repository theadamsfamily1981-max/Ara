# MCP FPGA Tools Integration Guide

## Overview

This document describes how the FPGA tools from the `mcp-claude-restart` repository integrate with the TF-A-N/Ara autonomous control system.

## Tool Inventory

### 1. Hardware RTL (neuron_core.v)

**Location**: `mcp-tools/*/examples/fpga_ai_workloads/snn_accelerator/rtl/neuron_core.v`

**Purpose**: Leaky Integrate-and-Fire (LIF) neuron array for FPGA

**Specifications**:
- 256 neurons per core (configurable)
- 24-bit membrane potential (16.8 fixed-point)
- Configurable threshold, leak rate, refractory period
- < 10 cycles per update @ 250MHz = 40ns/neuron

**Integration with TF-A-N**:
```
L0 Reactive Layer ←→ neuron_core.v
                      ↑
                cfg_threshold  (from L1 DAU)
                cfg_leak_rate  (from AEPO)
                cfg_enable     (from L3)
```

### 2. Python FPGA Interface (snn_inference.py)

**Location**: `mcp-tools/*/examples/fpga_ai_workloads/snn_accelerator/software/snn_inference.py`

**Purpose**: High-level Python API for FPGA SNN inference

**Key Classes**:
```python
class SNNAccelerator:
    def __init__(device_id, interface="pcie")  # OpenCL/XRT/simulation
    def load_weights(weight_matrix: np.ndarray)  # INT8 quantized
    def encode_spikes(input_data, encoding="rate")  # Rate/temporal coding
    def infer(spike_input) -> spike_counts  # Run inference
```

**Integration with TF-A-N**:
```python
# In tfan/synergy/fpga_device.py
from mcp_tools.snn_inference import SNNAccelerator

class FPGADevice:
    def __init__(self, config: FPGAConfig):
        self.accelerator = SNNAccelerator(device_id=0)

    def run_snn_step(self, spikes, weights):
        return self.accelerator.infer(spikes)
```

### 3. A10PED Python Driver (a10ped.py)

**Location**: `mcp-tools/*/projects/a10ped_neuromorphic/sw/python/a10ped.py`

**Purpose**: Direct register access for real-time control

**Key Registers** (from ai_tile_regs.h):
| Register | Offset | Purpose |
|----------|--------|---------|
| SNN_THRESHOLD | 0x28 | LIF spike threshold (16.16 FP) |
| SNN_LEAK | 0x2C | Membrane leak rate |
| SNN_REFRACT | 0x30 | Refractory cycles |
| TEMPERATURE | 0x40 | FPGA junction temp |
| PERF_CYCLES | 0x38 | Cycle counter |

**Integration for DAU v_th Control**:
```python
# Real-time threshold modulation
class DAUController:
    def __init__(self):
        self.tile = AITile(tile_id=0)

    def adjust_threshold(self, delta_vth: float):
        # Convert to 16.16 fixed-point
        vth_fp = int(delta_vth * 65536)
        self.tile._write_csr32(CSROffset.SNN_THRESHOLD, vth_fp)
```

### 4. Kernel CSR Graph (snn_csr_graph.c)

**Location**: `mcp-tools/*/kernel/semantic_ai/snn_csr_graph.c`

**Purpose**: High-performance graph storage in kernel space

**Features**:
- CSR++ format with dynamic edge insertion
- Tombstone-based deletion (no full rebuild)
- RCU lock-free reads for concurrent access
- Compaction when tombstone ratio > threshold

**Integration with PGU**:
```c
// Verify topological constraints in kernel space
struct snn_csr_graph *graph;
snn_csr_graph_init(&graph, num_neurons, max_synapses, feature_dim);

// After AEPO optimization, verify β₁ preserved
int old_edges = graph->live_edge_count;
snn_csr_graph_add_edge(graph, src, dst, type, weight);
// Check topological invariants via PGU
```

## Integration Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TF-A-N Control Stack                      │
├─────────────────────────────────────────────────────────────┤
│  L3 Metacontrol  │  AdaptiveMetacontrol (adaptive_controller.py)
│                  │  ↓ AEPO-tuned params (curvature_c, jerk_threshold)
├──────────────────┼──────────────────────────────────────────┤
│  L2 Structural   │  TopologicalVerifier (topological_constraints.py)
│                  │  ↓ PGU gates (β₀, β₁, λ₂ constraints)
├──────────────────┼──────────────────────────────────────────┤
│  L1 Homeostatic  │  a10ped.py → SNN_THRESHOLD register
│                  │  ↓ Real-time v_th modulation (<1ms)
├──────────────────┼──────────────────────────────────────────┤
│  L0 Reactive     │  neuron_core.v on FPGA
│                  │  ↓ LIF populations @ 250MHz
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    MCP Hardware Stack                        │
├─────────────────────────────────────────────────────────────┤
│  User Space      │  snn_inference.py (SNNAccelerator)        │
│                  │  ↓ OpenCL/XRT/Ethernet interface          │
├──────────────────┼──────────────────────────────────────────┤
│  Kernel Space    │  a10ped_driver.c + snn_csr_graph.c        │
│                  │  ↓ PCIe BAR0 mmap, ioctl                  │
├──────────────────┼──────────────────────────────────────────┤
│  Hardware        │  neuron_core.v + ai_csr.v                 │
│                  │  Intel Arria 10 / Stratix 10              │
└─────────────────────────────────────────────────────────────┘
```

## Integration Steps

### Step 1: Install MCP Tools

```bash
# Link MCP tools into TF-A-N
ln -s /home/user/Ara/mcp-tools/mcp-claude-restart-*/projects/a10ped_neuromorphic \
      /home/user/Ara/tfan/hw/a10ped

ln -s /home/user/Ara/mcp-tools/mcp-claude-restart-*/examples/fpga_ai_workloads/snn_accelerator \
      /home/user/Ara/tfan/hw/snn_accelerator
```

### Step 2: Build Kernel Driver

```bash
cd /home/user/Ara/tfan/hw/a10ped/sw/driver
make
sudo insmod a10ped_driver.ko
```

### Step 3: Update tfan/synergy/fpga_device.py

```python
from tfan.hw.a10ped.sw.python.a10ped import AITile, CSROffset
from tfan.hw.snn_accelerator.software.snn_inference import SNNAccelerator

class FPGADevice:
    def __init__(self, config):
        self.tile = AITile(tile_id=0)
        self.accelerator = SNNAccelerator(device_id=0)

    def set_threshold(self, vth: float):
        """L1 Homeostatic: real-time v_th control"""
        self.tile._write_csr32(CSROffset.SNN_THRESHOLD,
                               int(vth * 65536))

    def run_snn(self, spikes, weights):
        """L0 Reactive: hardware SNN inference"""
        return self.accelerator.infer(spikes)
```

### Step 4: Add to Triton Deployment

Update `deploy/triton/docker-compose.production.yml`:

```yaml
services:
  triton-server:
    volumes:
      - /dev/a10ped0:/dev/a10ped0  # FPGA device
    devices:
      - /dev/a10ped0
```

## Performance Targets

| Metric | Target | MCP Tool |
|--------|--------|----------|
| SNN inference latency | < 1ms | snn_inference.py |
| v_th modulation latency | < 100μs | a10ped.py CSR write |
| Throughput | ≥ 250k events/s | neuron_core.v @ 250MHz |
| Topology verification | < 10ms | snn_csr_graph.c |

## Semantic AI Kernel for System Optimization

The MCP tools include a comprehensive **Semantic AI Kernel** for autonomous system optimization.

### Components

| File | Purpose |
|------|---------|
| `snn_ai_engine_v2.c` | Production RL engine (Q24.8, INT8 Q-table, <100μs decisions) |
| `snn_gnn.c/h` | Graph Neural Network for topology-aware embedding |
| `snn_knowledge_graph.c` | Domain knowledge encoding (GPU↔FPGA routing) |
| `snn_cold_start.c/h` | Safe exploration during initialization |
| `snn_quantization.h` | INT8 quantized inference |
| `snn_fixed_point.h` | Q24.8 arithmetic for kernel-space |

### Workload Routing (GPU ↔ FPGA)

```c
// Query knowledge graph for device recommendation
float gpu_score, fpga_score;
snn_kg_recommend_device(kg, SNN_WORKLOAD_SPARSE, &gpu_score, &fpga_score);
// Result: gpu_score=0.40, fpga_score=0.85 → Route to FPGA
```

### Adaptive Learning from Feedback

```c
// Update knowledge graph based on actual performance
snn_kg_update_from_feedback(kg, SNN_DEVICE_FPGA, SNN_WORKLOAD_SPARSE,
                            success=true, performance=0.92);
```

### GNN State Embedding for Topology-Aware Decisions

```c
// Compute graph-level embedding
snn_gnn_forward(model, graph_embedding);

// Get node-level embedding
snn_gnn_get_node_embedding(model, node_id, embedding);
```

### Integration with TF-A-N Model Selector

The semantic AI can drive the Model Selector's routing decisions:

```python
# In tfan/model_selector.py
class ModelSelector:
    def route_workload(self, features):
        # Get recommendation from semantic AI kernel
        gpu_score, fpga_score = self.semantic_ai.recommend_device(features)

        if fpga_score > gpu_score and features.sparsity > 0.9:
            return "fpga"
        else:
            return "gpu"
```

## Files to Copy/Adapt

1. **Essential for L0-L1**:
   - `snn_accelerator/rtl/neuron_core.v` → Synthesize for target FPGA
   - `a10ped/sw/python/a10ped.py` → Import in tfan/synergy
   - `a10ped/sw/driver/a10ped_driver.c` → Build kernel module

2. **Essential for L2-L3**:
   - `kernel/semantic_ai/snn_csr_graph.c` → For PGU kernel-space verification
   - `kernel/semantic_ai/snn_gnn.c` → GNN for topology-aware routing
   - `kernel/semantic_ai/snn_knowledge_graph.c` → Domain knowledge encoding
   - `snn_accelerator/software/snn_inference.py` → High-level API

3. **Essential for System Optimization**:
   - `kernel/semantic_ai/snn_ai_engine_v2.c` → Production RL engine
   - `kernel/semantic_ai/snn_cold_start.c` → Safe exploration
   - `include/snn_kernel/semantic_ai.h` → Type definitions

4. **Reference**:
   - `docs/ARCHITECTURE.md` → System design patterns
   - `projects/a10ped_neuromorphic/abi/ai_tile_registers.yaml` → Register spec
