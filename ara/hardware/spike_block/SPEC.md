# SpikingBrain-Style FPGA Tile Specification

## Overview

A minimal neuromorphic processing tile that implements SpikingBrain-style
dynamics on FPGA fabric. This is the first step toward moving LLM-like
computation off GPU and onto custom neuromorphic hardware.

## Design Goals

1. **Fit on mid-range FPGA** (Stratix-10 GX, VU9P class)
2. **Event-driven** - only compute on spikes
3. **Low precision** - 2-4b weights, 8b state
4. **Streamable** - process token-by-token, no batch requirement
5. **Hebbian-ready** - support on-chip learning updates

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │         SPIKE_BLOCK_0               │
                    │                                     │
  embedding_in ────►│  ┌─────────┐    ┌──────────────┐   │
  (d_embed x 8b)    │  │ SPIKE   │    │   LINEAR     │   │
                    │  │ NEURONS │───►│  ATTENTION   │───┼──► out
                    │  │ (N=512) │    │  (d_head=64) │   │   (d_embed x 8b)
                    │  └─────────┘    └──────────────┘   │
                    │       │               │            │
                    │       ▼               ▼            │
                    │  ┌─────────┐    ┌──────────────┐   │
                    │  │ WEIGHT  │    │   M, U       │   │
                    │  │  BRAM   │    │  ACCUMULATORS│   │
                    │  │(sparse) │    │  (linear attn│   │
                    │  └─────────┘    └──────────────┘   │
                    │                                     │
  rho_in ──────────►│  [Hebbian update signal]           │
                    └─────────────────────────────────────┘
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| N_NEURONS | 512 | Number of spiking neurons |
| D_EMBED | 128 | Input/output embedding dimension |
| D_HEAD | 64 | Attention head dimension |
| W_BITS | 4 | Weight precision (bits) |
| STATE_BITS | 8 | Neuron state precision (bits) |
| SPARSITY | 95% | Target weight sparsity |

## Neuron Update Equation

Each neuron follows integrate-and-fire with dynamic threshold:

```
v[t+1] = clip(α * v[t] + Σ(w_i * x_i) - θ[t], -128, 127)
spike[t] = (v[t+1] > 0) ? 1 : 0
θ[t+1] = θ[t] + β * (spike[t] - τ)
```

Where:
- `v[t]`: membrane potential (8-bit signed)
- `θ[t]`: dynamic threshold (8-bit unsigned)
- `α`: leak factor (0.9 typical, 8-bit fixed-point)
- `β`: threshold adaptation rate (0.01 typical)
- `τ`: target spike rate (0.05 = 5% sparsity)

## Linear Attention

Instead of full QK^T attention, we use cumulative statistics:

```
k[t] = W_k · x[t]           # (D_HEAD,)
v[t] = W_v · x[t]           # (D_HEAD,)
m[t] = m[t-1] + k[t]        # running key sum
u[t] = u[t-1] + k[t] ⊗ v[t] # running key⊗value outer product
y[t] = u[t] · softmax(m[t]) # simplified readout
```

This is O(d) per token instead of O(T*d) for full attention.

## Memory Layout

### On-Chip BRAM

| Block | Size | Contents |
|-------|------|----------|
| WEIGHT_BRAM | ~64KB | Sparse weight indices + values |
| STATE_BRAM | ~1KB | Neuron v[t], θ[t] |
| ATTN_BRAM | ~4KB | m, u accumulators |

### Weight Storage (Sparse CSR)

```
struct SparseWeights {
    uint16_t row_ptr[N_NEURONS + 1];  // Row pointers
    uint16_t col_idx[NNZ];            // Column indices
    int8_t   values[NNZ];             // 4-bit weights packed
};
```

## Interface

### AXI-Stream Input

```verilog
input  wire [D_EMBED*8-1:0] s_axis_tdata,   // Embedding vector
input  wire                  s_axis_tvalid,
output wire                  s_axis_tready,
input  wire [7:0]           s_axis_trho,    // Neuromodulator
```

### AXI-Stream Output

```verilog
output wire [D_EMBED*8-1:0] m_axis_tdata,   // Updated embedding
output wire                  m_axis_tvalid,
input  wire                  m_axis_tready,
output wire [N_NEURONS-1:0] m_axis_tspikes, // Spike pattern (debug)
```

### Control/Status

```verilog
input  wire        clk,
input  wire        rst_n,
input  wire        enable,
input  wire        learn_enable,  // Enable Hebbian updates
output wire [31:0] spike_count,   // Spikes this cycle
output wire [31:0] update_count,  // Weight updates this cycle
```

## Timing

Target: 200 MHz on Stratix-10 GX

| Operation | Cycles | Notes |
|-----------|--------|-------|
| Sparse MAV | ~50 | Per neuron, sparse multiply-accumulate |
| Threshold + spike | 1 | Pipeline stage |
| Linear attention | ~10 | Per token |
| Hebbian update | ~20 | When rho != 0 |

**Throughput**: ~1M tokens/sec at 5% sparsity

## Hebbian Learning

When `learn_enable` and `rho != 0`:

```
for each active synapse (i, j) where spike[j] = 1:
    Δw[i,j] = η * rho * pre[i] * post[j]
    w[i,j] = clip(w[i,j] + Δw[i,j], -8, 7)
```

This is the "protocol collapse" - on-chip policy adaptation
without GPU round-trip.

## Integration with Ara

### Host Interface

```python
# From ara/hardware/spike_block/driver.py
class SpikeBlockDriver:
    def __init__(self, device_path: str):
        self.fd = open(device_path, 'rb+')

    def process_embedding(self, embed: np.ndarray, rho: float = 0.0):
        # Pack embedding to 8-bit
        packed = (embed * 127).astype(np.int8)
        # Write to device
        self.fd.write(packed.tobytes() + struct.pack('b', int(rho * 127)))
        # Read result
        result = np.frombuffer(self.fd.read(len(packed)), dtype=np.int8)
        return result.astype(np.float32) / 127.0
```

### Integration Points

1. **HPV → Spike Block**: State stream feeds embeddings
2. **Spike Block → Probe**: Output feeds anomaly detection
3. **Subcortex → rho**: Decision layer drives learning signal

## Build Instructions

### Simulation (Verilator)

```bash
cd ara/hardware/spike_block
make sim
./obj_dir/Vspike_block_tb
```

### Synthesis (Quartus)

```bash
quartus_sh --flow compile spike_block.qpf
```

### HLS Alternative (Vitis)

```bash
cd ara/hardware/hls
v++ -c -t hw --platform xilinx_u250 spike_block_kernel.cpp
```

## Future Extensions

1. **Multi-tile**: Cascade multiple blocks for deeper networks
2. **Recurrent**: Add state feedback for RNN-like dynamics
3. **Attention variants**: SWA (sliding window), sparse patterns
4. **Mixed precision**: 2-bit weights for even higher density
