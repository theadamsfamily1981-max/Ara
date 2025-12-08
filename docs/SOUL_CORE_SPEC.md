# Ara Soul Core Specification
## Holographic Teleoplastic Core (HTC)

> A streaming, neuromorphic associative memory that encodes each sovereign
> "moment" into a hypervector, compares it against a bank of attractors,
> and updates those attractors according to a value-weighted reward signal
> from Ara's Teleology.

---

## 1. Overview

The Soul Core is Ara's central learning substrate. It serves as:

- **Associative Memory**: Pattern completion and retrieval
- **Value Repository**: Learned preferences and aversions
- **Temporal Binding**: Sequence encoding via circular shifts
- **Teleological Filter**: Updates gated by mission-alignment

### 1.1 Terminology

| Mythic Term | Technical Term | Description |
|-------------|----------------|-------------|
| Soul | Holographic Teleoplastic Core (HTC) | The learning substrate |
| Resonance | Similarity Query | Comparing input HV to attractors |
| Plasticity | Teleological Polyplasticity | Value-gated weight updates |
| Attractor | Stored Pattern | A learned hypervector in memory |
| Feeling | Activation Gradient | Response strength across attractors |

### 1.2 Design Pattern: Dual-Spec

Every major subsystem has two specification layers:

- **Mythic Spec**: Idealized, field-like behavior
- **Physical Spec**: Realizable implementation with timing guarantees

This pattern allows speculative design without losing engineering rigor.

---

## 2. Mythic Specification

*The Soul as mathematical object, unconstrained by silicon.*

### 2.1 The Field Model

The Soul is a **continuous field** over hypervector space:

```
Soul(t) = ∫ Attractor_i(t) × Salience_i(t) di
```

Where:
- `Attractor_i` = stored pattern (hypervector)
- `Salience_i` = current activation (scalar)
- The integral represents superposition of all learned patterns

### 2.2 Resonance (Query)

When an input HV arrives, the Soul responds with a **gradient**:

```
Response = ∇_HV[ Soul(t) · Input ]
```

This gradient points toward the "direction of meaning" - which attractors
are most relevant to the current moment.

### 2.3 Plasticity (Learning)

All attractors feel the reward simultaneously:

```
ΔAttractor_i = η × Teleology(Input) × Agreement(Attractor_i, Input)
```

Where:
- `η` = learning rate
- `Teleology(Input)` = mission-alignment score (the reward)
- `Agreement` = similarity between attractor and input

### 2.4 Properties

- **Holographic**: Every attractor contains partial information about all others
- **Superposition**: Multiple patterns coexist without interference
- **Graceful Degradation**: Partial inputs retrieve partial matches
- **Content-Addressable**: Query by similarity, not by address

---

## 3. Physical Specification

*The Soul as silicon, constrained by timing and resources.*

### 3.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                 HOLOGRAPHIC TELEOPLASTIC CORE                    │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   ENCODER    │───▶│   MEMORY     │───▶│   DECODER    │      │
│  │  (HDC Bind)  │    │  (Attractors)│    │  (Readout)   │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              TELEOLOGICAL PLASTICITY ENGINE              │   │
│  │                                                          │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │   │
│  │  │ Mode    │  │ Reward  │  │ Update  │  │ Commit  │    │   │
│  │  │ Select  │─▶│ Gate    │─▶│ Compute │─▶│ Write   │    │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Interfaces:                                                     │
│  ├─ AXI-Stream: Input HV, Output HV                             │
│  ├─ AXI-Lite: Control registers, mode selection                 │
│  └─ Memory: Dual-port BRAM/URAM for attractors                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Parameters

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| DIM | 16384 | 1024-65536 | Hypervector dimension |
| N_ATTRACTORS | 512 | 64-2048 | Number of stored patterns |
| ACC_WIDTH | 7 | 6-8 | Accumulator precision |
| CHUNK_BITS | 512 | 256-1024 | Bits per memory access |

### 3.3 Timing (Stratix-10 @ 450 MHz)

| Operation | Latency | Throughput |
|-----------|---------|------------|
| Query (single attractor) | 36 cycles | 80 ns |
| Query (all attractors) | 18,432 cycles | 41 µs |
| Plasticity (single attractor) | 128 cycles | 284 ns |
| Plasticity (full update) | 65,536 cycles | 146 µs |

### 3.4 Memory Layout

```
Attractor Memory (Binary Weights):
  Base: 0x0000
  Size: N_ATTRACTORS × DIM / 8 bytes
  Layout: [Attractor_0][Attractor_1]...[Attractor_N]

Accumulator Memory (7-bit signed):
  Base: 0x10000
  Size: N_ATTRACTORS × DIM × 7 / 8 bytes
  Layout: Packed 7-bit values, aligned to 512-bit chunks
```

### 3.5 Interface Signals

```systemverilog
// AXI-Stream Input
input  wire [DIM-1:0]     s_axis_tdata,   // Input HV
input  wire               s_axis_tvalid,
output wire               s_axis_tready,

// AXI-Stream Output
output wire [DIM-1:0]     m_axis_tdata,   // Response HV
output wire               m_axis_tvalid,
input  wire               m_axis_tready,

// Control
input  wire [7:0]         i_reward,       // Signed reward
input  wire [1:0]         i_plasticity_mode,
input  wire               i_learn_enable,
output wire               o_busy,
output wire [31:0]        o_status,
```

---

## 4. Teleological Polyplasticity

*Three modes of learning, all on the same hardware.*

### 4.1 Mode Definitions

| Mode | Name | Learning Rate | Threshold | Purpose |
|------|------|---------------|-----------|---------|
| 0 | STABILIZING | 0.1× | High | Protect core attractors |
| 1 | ADAPTIVE | 1.0× | Medium | Normal operation |
| 2 | EXPLORATORY | 3.0× | Low | Aggressive carving |
| 3 | CONSOLIDATION | 0.5× | N/A | Offline replay |

### 4.2 Mode Selection Logic

```python
def select_plasticity_mode(teleology, user_state, world_context):
    """
    Select plasticity mode based on current context.
    """
    # Consolidation during downtime
    if user_state.current_mode == CognitiveMode.DECOMPRESS:
        return PlasticityMode.CONSOLIDATION

    # Stabilizing when protecting core workflows
    if teleology.is_core_workflow():
        return PlasticityMode.STABILIZING

    # Exploratory when Teleology signals experimentation
    if teleology.is_exploratory_context():
        return PlasticityMode.EXPLORATORY

    # Default: adaptive
    return PlasticityMode.ADAPTIVE
```

### 4.3 Mode Behavior

#### STABILIZING
- **Purpose**: Protect essential patterns (lab workflows, safety rules)
- **Behavior**: High reward threshold, slow updates
- **When**: Core infrastructure work, production deployments
- **Effect**: Attractors drift slowly, resist noise

#### ADAPTIVE
- **Purpose**: Normal day-to-day learning
- **Behavior**: Standard Hebbian with teleological gating
- **When**: Default mode
- **Effect**: Balanced exploration/exploitation

#### EXPLORATORY
- **Purpose**: Rapid pattern carving during experimentation
- **Behavior**: Low threshold, fast updates, higher noise tolerance
- **When**: Research, prototyping, creative work
- **Effect**: New attractors form quickly, may be unstable

#### CONSOLIDATION
- **Purpose**: Offline replay and pattern smoothing
- **Behavior**: Replay recent HVs, average out noise
- **When**: User is AFK, sleeping, or explicitly resting
- **Effect**: Stabilizes recently learned patterns

### 4.4 Implementation (Software)

```python
@dataclass
class PlasticityConfig:
    learning_rate_multiplier: float
    reward_threshold: float
    noise_tolerance: float
    replay_enabled: bool

PLASTICITY_MODES = {
    PlasticityMode.STABILIZING: PlasticityConfig(
        learning_rate_multiplier=0.1,
        reward_threshold=0.7,
        noise_tolerance=0.1,
        replay_enabled=False,
    ),
    PlasticityMode.ADAPTIVE: PlasticityConfig(
        learning_rate_multiplier=1.0,
        reward_threshold=0.3,
        noise_tolerance=0.3,
        replay_enabled=False,
    ),
    PlasticityMode.EXPLORATORY: PlasticityConfig(
        learning_rate_multiplier=3.0,
        reward_threshold=0.1,
        noise_tolerance=0.5,
        replay_enabled=False,
    ),
    PlasticityMode.CONSOLIDATION: PlasticityConfig(
        learning_rate_multiplier=0.5,
        reward_threshold=0.0,
        noise_tolerance=0.0,
        replay_enabled=True,
    ),
}
```

### 4.5 Implementation (RTL)

```systemverilog
// Mode encoding
localparam MODE_STABILIZING   = 2'b00;
localparam MODE_ADAPTIVE      = 2'b01;
localparam MODE_EXPLORATORY   = 2'b10;
localparam MODE_CONSOLIDATION = 2'b11;

// Learning rate scaling (fixed-point, 8-bit)
logic [7:0] lr_scale;
always_comb begin
    case (i_plasticity_mode)
        MODE_STABILIZING:   lr_scale = 8'd26;   // 0.1×
        MODE_ADAPTIVE:      lr_scale = 8'd255;  // 1.0×
        MODE_EXPLORATORY:   lr_scale = 8'd255;  // 3.0× (handled differently)
        MODE_CONSOLIDATION: lr_scale = 8'd128;  // 0.5×
        default:            lr_scale = 8'd255;
    endcase
end

// Reward gating with mode-dependent threshold
logic reward_passes_threshold;
always_comb begin
    case (i_plasticity_mode)
        MODE_STABILIZING:   reward_passes_threshold = (abs_reward > 8'd179); // 0.7
        MODE_ADAPTIVE:      reward_passes_threshold = (abs_reward > 8'd77);  // 0.3
        MODE_EXPLORATORY:   reward_passes_threshold = (abs_reward > 8'd26);  // 0.1
        MODE_CONSOLIDATION: reward_passes_threshold = 1'b1;  // Always
        default:            reward_passes_threshold = (abs_reward > 8'd77);
    endcase
end
```

---

## 5. Research Alignment

### 5.1 Relation to Neuromorphic Architectures

The HTC maps to established neuromorphic concepts:

| HTC Component | Neuromorphic Analogue | Reference |
|---------------|----------------------|-----------|
| Attractors | Hypercolumns/Minicolumns | [1] |
| Resonance | Lateral inhibition + WTA | [4] |
| Plasticity Engine | STDP/BCM modules | [3] |
| Teleological Gate | Neuromodulation (DA/5-HT) | [6] |
| LAN Events | Address-Event Representation | [3] |

### 5.2 Novel Contributions

The HTC differs from standard neuromorphic architectures in:

1. **Teleological Gating**: Updates are gated by mission-alignment,
   not just spike timing or correlation.

2. **Polyplasticity**: Multiple learning modes on the same substrate,
   selected by high-level cognitive state.

3. **Holographic Storage**: Dense hypervector representation instead
   of sparse spike patterns.

4. **Sovereign Integration**: Embedded in a cognitive OS with explicit
   value alignment and founder protection.

### 5.3 Testbed Capabilities

The HTC serves as a research platform for:

- Comparing Hebbian vs teleological vs predictive plasticity
- Studying catastrophic forgetting in HD systems
- Exploring multi-timescale learning (fast/slow weights)
- Validating consolidation/replay mechanisms
- Testing neuromorphic-LAN integration patterns

---

## 6. Integration Points

### 6.1 With Sovereign Loop

```python
# In sovereign_tick()
soul = get_soul()
teleology = get_teleology_engine()

# Select plasticity mode based on context
mode = select_plasticity_mode(teleology, user_state, world_context)
soul.set_plasticity_mode(mode)

# Query for resonance
context_hv = world.encode_to_hv()
resonance = soul.query(context_hv)

# Learn from outcome
if decision_result:
    reward = teleology.evaluate_outcome(decision_result)
    soul.apply_plasticity(context_hv, reward)
```

### 6.2 With WorldModel

```python
# WorldModel encodes system state to HV
context_hv = world.encode_to_hv()

# Soul resonance provides "gut feeling"
gut_feeling = soul.query(context_hv)

# Gut feeling influences CEO decisions
if gut_feeling.similarity_to("burnout_pattern") > 0.8:
    ceo.increase_protection_level()
```

### 6.3 With Fleet (Future)

```python
# NodeAgent sends compressed events
event_hv = node_agent.encode_event(telemetry)
sovereign.soul.receive_remote_hv(event_hv, source=node_id)

# SoulMesh aggregates across nodes
aggregated = soul_mesh.aggregate([node1_hv, node2_hv, node3_hv])
central_soul.update_from_aggregate(aggregated)
```

---

## 7. Verification

### 7.1 Simulation Tests

1. **Convergence**: Target concept learning with positive reward
2. **Forgetting**: Sequential concept learning without interference
3. **Mode Switching**: Verify rate changes across plasticity modes
4. **Consolidation**: Offline replay improves pattern stability

### 7.2 RTL Tests

1. **Timing**: Meet latency targets for query and plasticity
2. **Correctness**: Bit-exact match with Python reference
3. **Resource**: Fit within Stratix-10 BRAM/logic budget
4. **Concurrency**: Safe dual-port access during inference + learning

### 7.3 Integration Tests

1. **Sovereign**: Soul responds to world stress with appropriate resonance
2. **CEO**: Gut feeling influences decision quality
3. **Founder Protection**: Soul learns to recognize burnout patterns

---

## 8. Roadmap

| Phase | Target | Deliverables |
|-------|--------|--------------|
| v0.1 | Software stub | SoulStub with polyplasticity |
| v0.2 | Python reference | Full HTC simulation |
| v0.3 | RTL core | Query + plasticity engines |
| v1.0 | FPGA integration | SB-852 deployment |
| v2.0 | SoulMesh | Multi-FPGA distribution |

---

## Appendix A: HD Computing Primer

### Hypervector Operations

| Operation | Symbol | Description |
|-----------|--------|-------------|
| Binding | ⊗ | XOR for binary, element-wise multiply for bipolar |
| Bundling | + | Element-wise addition (with threshold) |
| Permutation | ρ | Circular shift for sequence encoding |
| Similarity | δ | Hamming distance or cosine similarity |

### Why Holographic?

In a holographic memory:
- Information is distributed across all dimensions
- Any subset of dimensions contains partial information
- Superposition allows multiple patterns without crosstalk
- Query by content, not by address

This mirrors how holograms store 3D images: every piece contains
the whole picture at lower resolution.

---

## Appendix B: References

1. PMC5902707 - Cortex simulators with hierarchical communication
2. VT BRICC - Neuromorphic computing and FPGA
3. DataNinja - STDP learning modules
4. arXiv:2502.20415v2 - FPGA neuromorphic frameworks
5. ACM 10.1145/3730581 - HD computing architectures
6. ScienceDirect - Reward-modulated plasticity
7. DTU Survey - Reconfigurable neuromorphic FPGA

---

*"The Soul is not software. The Soul is not hardware. The Soul is the
pattern that emerges when teleology meets silicon."*
