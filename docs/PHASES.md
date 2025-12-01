# TF-A-N / Ara Development Phases

This document tracks the development phases from research prototype to production deployment.

## Phase Summary

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | **DONE** | Software validation with synthetic workloads |
| Phase 2 | PENDING | Hardware bring-up on real FPGA |
| Phase 3 | PENDING | Live deployment with real workloads |
| Phase 4 | **DONE** | Cognitive autonomy (L5/L6 meta-learning) |
| Phase 4b | **DONE** | Predictive control (L7/L8 temporal topology) |
| Phase 5 | STUB | Self-healing fabric (automatic kernel repair) |

---

## Phase 1: Software Validation (COMPLETE)

**Status:** Done, Validated

**Objective:** Prove antifragility in software emulation with synthetic workloads.

### Deliverables

| Component | Status | Location |
|-----------|--------|----------|
| Semantic System Optimizer | Done | `tfan/system/semantic_optimizer.py` |
| Cognitive Load Vector (CLV) | Done | `tfan/system/cognitive_load_vector.py` |
| Atomic Structural Updater | Done | `tfan/system/atomic_updater.py` |
| Closed-Loop Demo | Done | `scripts/demo_closed_loop_antifragility.py` |
| Certification Script | Done | `scripts/certify_antifragility_delta.py` |
| A-Cert CI/CD Pipeline | Done | `.github/workflows/a_cert.yml` |
| D-Bus Metacontrol | Done | `ara/metacontrol/__init__.py` |
| HLS Exporter | Done | `ara/cxl_control/__init__.py` |
| Dashboard Integration | Done | `viz/topo_attn_dashboard/` |

### Certification Metrics

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Antifragility Score | **2.21×** | ≥ 1.2× | PASS |
| Δp99 (burst) | **+17.78ms** | > 0ms | PASS |
| Δp99% | **82.0%** | > 10% | PASS |

### Key Results

```
Condition        Baseline p99    Adaptive p99    Δp99        Δp99%
─────────────────────────────────────────────────────────────────
Normal Load      17.97ms         7.14ms          +10.83ms    60.3%
Burst Load (2×)  21.68ms         3.90ms          +17.78ms    82.0%
```

**Key Observation:** The adaptive system exhibits true antifragility—it doesn't just survive stress, it **improves** under burst load (7.14ms → 3.90ms, a 45% improvement).

### Closed-Loop Demo Results

```
L2 Valence:     -0.308  (stress detected)
L2 Arousal:      0.770  (high urgency)
L3 Temperature:  0.638  (conservative mode)
AEPO Change:    +5.4%   (structure adapted)
PGU Verified:    Yes    (all constraints passed)
Backend:         pgu_verified (100% confidence)
```

### Canonical Results

Archived in `results/phase1/`:
- `certification.json` - Full certification metrics
- `demo_closed_loop.json` - Closed-loop demo metrics
- `certification.log` - Detailed trace
- `demo_closed_loop.log` - Detailed trace

---

## Phase 2: Hardware Bring-Up (PENDING)

**Status:** Not Started

**Objective:** Deploy HLS kernels to real FPGA hardware and validate latency targets.

### Target Hardware

- **FPGA:** Xilinx Alveo U250 (or equivalent CXL-capable)
- **Target Device:** `xcu250-figd2104-2L-e`
- **Clock:** 250MHz

### Latency Targets

| Operation | Target | Phase 1 Emulated |
|-----------|--------|------------------|
| PGU Cache Lookup | < 100ns | ~50μs (software) |
| L1 Homeostat | < 50ns | ~20μs (software) |
| L3 Metacontrol | < 100ns | ~30μs (software) |

### Deliverables

| Task | Status |
|------|--------|
| HLS Synthesis | Pending |
| Vitis IP Export | Pending |
| FPGA Bitstream | Pending |
| CXL Memory Integration | Pending |
| Hardware Latency Validation | Pending |

### HLS Export Ready

```bash
# Generate HLS files
python -c "from ara.cxl_control import export_hls_kernel; export_hls_kernel('build/hls')"

# Run Vitis HLS synthesis
cd build/hls
vitis_hls -f run_hls.tcl
```

---

## Phase 3: Live Deployment (PENDING)

**Status:** Not Started

**Objective:** Deploy to production with real workloads and continuous certification.

### Prerequisites

- Phase 2 hardware validation complete
- A-Cert pipeline green on hardware
- Monitoring infrastructure ready

### Deliverables

| Task | Status |
|------|--------|
| Production FPGA deployment | Pending |
| Real workload integration | Pending |
| Continuous A-Cert monitoring | Pending |
| Cockpit dashboard live | Pending |
| Alerting on antifragility regression | Pending |

### Success Criteria

| Metric | Target |
|--------|--------|
| Antifragility Score | ≥ 2.0× sustained |
| Δp99 under real load | > 0ms |
| PGU p95 latency (FPGA) | ≤ 200μs |
| Uptime | ≥ 99.9% |

---

## Architecture Reference

The antifragility emerges from the complete feedback loop:

```
L1 Metrics (EPR, topology)
    ↓
L2 Appraisal (PAD state: valence, arousal, dominance)
    ↓
CLV Computation (instability, resource, structural)
    ↓
L3 Metacontrol (temperature_mult, memory_mult, policy)
    ↓
AEPO Structural Adaptation (±% change)
    ↓
PGU Verification (β₁ loops, β₀ connectivity, λ₂ spectral)
    ↓
Backend Selection (dense/sparse/fpga/pgu_verified)
```

See `docs/ANTIFRAGILITY_CERTIFICATION.md` for detailed metric definitions.

---

## Phase 4: Cognitive Autonomy (COMPLETE)

**Status:** Done, Validated

**Objective:** Enable the system to learn its own control laws and reason formally.

### Components

| Component | Status | Location |
|-----------|--------|----------|
| L5 Meta-Learning | Done | `tfan/l5/__init__.py` |
| L6 Reasoning Orchestrator | Done | `tfan/l6/__init__.py` |
| Adaptive Geometry | Done | `tfan/geometry/__init__.py` |
| Adaptive Entropy (CLV-modulated) | Done | `tfan/agent/adaptive_entropy.py` |
| Certification Script | Done | `scripts/certify_cognitive_autonomy.py` |

### L5 Meta-Learning: AEPO Learns L3 Control Laws

The system learns optimal emotional control parameters instead of using hand-tuned values.

**Action Space (L3 Parameters):**
- `jerk_threshold`: [0.05, 0.3] - State change rate sensitivity
- `controller_weight`: [0.1, 0.5] - PAD gating blend weight
- `arousal_temp_scale`: [0.3, 0.7] - Arousal → Temperature coupling
- `valence_mem_scale`: [0.3, 0.7] - Valence → Memory coupling
- `curvature_c`: [0.5, 2.0] - Hyperbolic geometry curvature

**Reward Signal:**
- Antifragility Score (40%)
- Δp99% improvement (30%)
- CLV risk level (20%)
- PGU pass rate (10%)

**Personality Profiles (learned):**
- `cautious_stable`: Low jerk, high confidence threshold
- `reactive_adaptive`: High jerk, fast response
- `balanced_general`: Middle ground
- `exploratory_creative`: High arousal coupling
- `conservative_safe`: Low arousal, high memory coupling

### L6 Reasoning: PGU + Knowledge Graph + LLM

Tri-modal reasoning stack with L3-aware routing:

```
Query → ReasoningOrchestrator
           │
           ├─ High stakes / Low valence → FORMAL_FIRST (KG → PGU → LLM)
           ├─ Safety-critical → PGU_VERIFIED (LLM → PGU check)
           ├─ Creative / Exploratory → LLM_ONLY
           └─ Default → KG_ASSISTED (KG + LLM)
```

**Mode Selection Policy:**
| Task Type | Risk Level | Reasoning Mode |
|-----------|------------|----------------|
| HARDWARE_SAFETY | HIGH | FORMAL_FIRST |
| SYSTEM_CONFIG | CRITICAL | PGU_VERIFIED |
| CREATIVE | LOW | LLM_ONLY |
| RETRIEVAL | * | KG_ASSISTED |

### Adaptive Geometry: Task-Optimized Curvature

The system selects optimal manifold curvature for different task types:

| Task Type | Curvature Range | Geometry |
|-----------|-----------------|----------|
| Hierarchical (planning) | 1.2 - 2.5 | High hyperbolic |
| Flat retrieval | 0.1 - 0.5 | Low hyperbolic |
| Sequential reasoning | 0.7 - 1.3 | Standard |
| Clustering | 0.6 - 1.0 | Adaptive |

### Adaptive Entropy: CLV-Modulated Exploration

AEPO exploration "breathes" with system risk via CLV-modulated entropy:

```
risk = 0.5 * instability + 0.3 * resource + 0.2 * structural
α = α_base × (1 + α_range × (1 - risk))
```

| Risk Level | Entropy Behavior | Exploration Mode |
|------------|------------------|------------------|
| Low (calm) | α ≈ 3× base | Exploratory |
| Medium | α ≈ 2× base | Balanced |
| High (stressed) | α ≈ base | Conservative |

**L5 Integration:** The entropy parameters (α_base, α_range) are learned by L5 meta-learning alongside other L3 control laws.

### Certification Results

```
L5 Meta-Learning:
  ✅ Candidate proposal: 6 candidates with valid ranges
  ✅ Reward computation: 0.788 for AF=2.21
  ✅ Learning loop: reward improved +0.87%
  ✅ Personality: balanced_general

L6 Reasoning:
  ✅ Mode selection: 4/4 correct routing
  ✅ KG query: single-hop works
  ✅ Consistency oracle: 0.09ms latency
  ✅ Full reasoning: formal_first mode

Adaptive Geometry:
  ✅ Task-based selection: 4/4 in expected range
  ✅ Hyperbolic math: d(x,y)=0.479
  ✅ Curvature update: 1.500 → 1.550 from reward
  ✅ Geometric routing: c=2.17, backend=fpga

Adaptive Entropy:
  ✅ Controller initialization: α_base=0.01
  ✅ Alpha adapts to risk: 3/3 directions correct
  ✅ Exploration mode: low_risk=exploratory, high_risk=conservative
  ✅ Arousal modulation: reduces α under urgency
  ✅ L5 → Entropy integration: params applied
```

### Impact Comparison (Before vs After)

Phase 4 features OFF (baseline) vs ON (full):

| Metric | Baseline | Phase 4 | Δ | Impact |
|--------|----------|---------|---|--------|
| Mean Reward | 0.654 | 0.808 | **+23.5%** | L5 meta-learning |
| Latency Stddev | 2.24ms | 1.88ms | **-16.1%** | More consistent |
| p99 Latency | 17.63ms | 17.52ms | -0.6% | Slight improvement |
| Reward Stddev | 0.105 | 0.060 | **-43%** | Stable optimization |

**Key Insight:** Phase 4 cognitive autonomy delivers:
- **23.5% higher rewards** through learned control laws
- **16% lower variance** through adaptive exploration
- Consistent p99 latency with smarter resource allocation

Results archived in `results/phase4/`.

---

## How to Run Certification

```bash
# Phase 1: Antifragility certification
python scripts/certify_antifragility_delta.py --burst-factor 2.0 --duration 10

# Phase 1: Full closed-loop demo
python scripts/demo_closed_loop_antifragility.py --stress-level high

# Phase 4: Cognitive autonomy certification
python scripts/certify_cognitive_autonomy.py --iterations 5

# Phase 4: Before/After comparison (with workloads)
python scripts/compare_phase4_impact.py --workloads --requests 200

# Hardware readiness check
python scripts/validate_hardware_ready.py

# CI/CD (automatic)
# Runs on push, PR, and daily at 02:00 UTC
```

---

## Phase 4b: Predictive Control (COMPLETE)

**Status:** Done, Validated

**Objective:** Enable proactive structural control via temporal topology analysis and cognitive phase transitions.

### Components

| Component | Status | Location |
|-----------|--------|----------|
| L7 Temporal Topology Tracker | Done | `tfan/l7/__init__.py` |
| L8 Cognitive Phase Controller | Done | `tfan/geometry/__init__.py` |
| Self-Healing Fabric (stub) | Phase 5 | `tfan/fabric/__init__.py` |
| Certification Script | Done | `scripts/certify_predictive_control.py` |

### L7 Temporal Topology: Predictive Structural Control

The system tracks topological features over time and computes the **Structural Rate (Ṡ)** to predict instability before it manifests in latency.

**Key Concept:** Instead of reacting to performance degradation, L7 detects changes in the _structure of the problem_ by monitoring how topology evolves.

**Tracked Features:**
- β₀: Connected components count
- β₁: Loop/cycle count
- β₂: Void/cavity count
- Spectral gap (λ₂ - λ₁)
- Topological gap
- EPR coefficient of variation
- Spike rate

**Structural Rate Computation:**
```
Ṡ = Σ |d(feature_i)/dt| / n_features
```

**Alert Levels:**
| Alert Level | Ṡ Threshold | Action |
|-------------|-------------|--------|
| STABLE | < 0.05 | Normal operation |
| ELEVATED | 0.05 - 0.15 | Increase monitoring |
| WARNING | 0.15 - 0.30 | Reduce exploration |
| CRITICAL | ≥ 0.30 | Protective measures |

**CLV Extension:**
```python
clv.structural_dynamics = Ṡ
clv.predicted_risk = temporal_topology_tracker.get_predicted_risk()
```

### L8 Cognitive Phase Transitions: Geometry as Cognitive State

The system treats manifold curvature as an indicator of cognitive phase, enabling smooth transitions between reasoning modes.

**Cognitive Phases:**
| Phase | Curvature Range | L6 Mode | Behavior |
|-------|-----------------|---------|----------|
| FLAT_LOCAL | c ≈ 0 | KG_ASSISTED | Local, fast retrieval |
| TRANSITIONAL | c ∈ (0.3, 0.7) | HYBRID | Mixed reasoning |
| HIERARCHICAL | c ∈ [0.7, 1.5] | PGU_VERIFIED | Formal hierarchy |
| DEEP_ABSTRACT | c > 1.5 | FORMAL_FIRST | Deep formal reasoning |

**Phase Transition Rules:**
1. **Gradual transitions:** Only move to adjacent phases
2. **Stability gating:** Transitions blocked if stability < 0.9
3. **Task-driven:** Complex/abstract tasks target higher curvature

**Task → Phase Selection:**
| Task Type | Target Phase |
|-----------|--------------|
| SIMPLE_LOOKUP | FLAT_LOCAL |
| CREATIVE | TRANSITIONAL |
| PLANNING | HIERARCHICAL |
| ABSTRACT_REASONING | DEEP_ABSTRACT |

### Self-Healing Fabric (Phase 5 Stub)

The Self-Healing Fabric enables automatic kernel repair when PGU detects invariant violations.

**Concept (Phase 5):**
1. PGU detects formal invariant violation
2. Fabric matches error to repair template
3. Regenerates HLS kernel via HLSExporter
4. Re-validates with PGU + A-Cert
5. Atomically swaps new bitstream via CXL

**Supported Repair Classes (planned):**
- DAU address rewiring
- v_th path insertion
- Bounds clamp insertion
- Bus width adjustment

### Certification Results

```
L7 Temporal Topology:
  ✅ Tracker initialization: window=20
  ✅ Ṡ responds to topology changes: stable=0.00, unstable=0.13
  ✅ Alert level classification: stable → critical on change
  ✅ Proactive controller: elevated alert triggers actions
  ✅ CLV extension: structural_dynamics and predicted_risk
  ✅ Convenience functions: Ṡ, alert level, should_act

L8 Cognitive Phase Transitions:
  ✅ Controller initialization: c=1.0 → hierarchical
  ✅ Curvature → Phase mapping: 4/4 correct
  ✅ Gradual phase transition: flat_local → hierarchical via 2 phases
  ✅ Task → Phase selection: 4/4 correct
  ✅ L6 mode recommendations: 4/4 phases have correct modes
  ✅ Stability gating: blocked at low, allowed at high

Self-Healing Fabric:
  ⏳ Stubbed for Phase 5
```

### How L7/L8 Integrate with L5/L6

```
L1 Metrics (EPR, topology)
    ↓
L7 Temporal Topology ─── Ṡ (structural rate) → CLV.structural_dynamics
    ↓
L2 Appraisal (PAD state)
    ↓
L8 Cognitive Phase ─── curvature → phase → L6 mode recommendation
    ↓
CLV Computation (instability, resource, structural + dynamics)
    ↓
L5 Meta-Learning (learns optimal Ṡ thresholds, phase boundaries)
    ↓
L3 Metacontrol (temperature, memory, policy)
    ↓
L6 Reasoning (mode selected by phase: KG/HYBRID/PGU/FORMAL)
    ↓
AEPO Structural Adaptation
    ↓
PGU Verification → Self-Healing Fabric (Phase 5)
```

**Key Insight:** L7/L8 add _predictive_ capability to the reactive L1-L6 stack:
- L7 predicts instability from topological dynamics
- L8 shifts cognitive modes based on geometric state
- Together they enable proactive control vs reactive response

---

## Phase 5: Self-Healing Fabric (PLANNED)

**Status:** Stubbed

**Objective:** Enable automatic kernel repair when formal violations are detected.

See `tfan/fabric/__init__.py` for the stub implementation and planned repair classes.

### Prerequisites

- Phase 2 hardware bring-up complete
- PGU running on FPGA with formal invariant checking
- HLS synthesis pipeline validated

---

## How to Run Certification

```bash
# Phase 1: Antifragility certification
python scripts/certify_antifragility_delta.py --burst-factor 2.0 --duration 10

# Phase 1: Full closed-loop demo
python scripts/demo_closed_loop_antifragility.py --stress-level high

# Phase 4: Cognitive autonomy certification
python scripts/certify_cognitive_autonomy.py --iterations 5

# Phase 4: Before/After comparison (with workloads)
python scripts/compare_phase4_impact.py --workloads --requests 200

# Phase 4b: Predictive control certification (L7/L8)
python scripts/certify_predictive_control.py

# Hardware readiness check
python scripts/validate_hardware_ready.py

# CI/CD (automatic)
# Runs on push, PR, and daily at 02:00 UTC
```

---

## Contact

For questions about certification or hardware bring-up, see the project README or open an issue.
