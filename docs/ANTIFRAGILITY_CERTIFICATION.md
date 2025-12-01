# Antifragility Certification (A-Cert)

This document defines the metrics, thresholds, and methodology for certifying system antifragility in the TF-A-N framework.

## Overview

Antifragility is the property of systems that **improve under stress**, as opposed to merely being robust (surviving stress) or fragile (degrading under stress). The A-Cert pipeline quantifies this property through comparative latency analysis under load.

## Core Metrics

### 1. Δp99 (Latency Delta)

**Definition:**
```
Δp99 = baseline_p99 - adaptive_p99
```

Where:
- `baseline_p99`: 99th percentile latency of the baseline executor (dense, static policy)
- `adaptive_p99`: 99th percentile latency of the adaptive executor (sparse, L3 metacontrol, PGU-cached)

**Interpretation:**
- `Δp99 > 0`: Adaptive system is faster (antifragile behavior)
- `Δp99 = 0`: Systems perform equally
- `Δp99 < 0`: Adaptive system is slower (certification fails)

**Measurement Conditions:**
- **Normal Load**: Baseline request rate with standard inter-arrival times
- **Burst Load**: `burst_factor × normal_rate` sustained for measurement duration

### 2. Δp99% (Percentage Improvement)

**Definition:**
```
Δp99% = (Δp99_burst / baseline_p99_burst) × 100
```

**Interpretation:** Percentage of baseline latency saved by the adaptive system under burst conditions.

### 3. Antifragility Score

**Definition:**
```
antifragility_score = baseline_degradation / adaptive_degradation
```

Where degradation is the increase in p99 latency from normal to burst:
```
baseline_degradation = baseline_p99_burst - baseline_p99_normal
adaptive_degradation = adaptive_p99_burst - adaptive_p99_normal
```

**Special Cases:**
- If `adaptive_degradation ≤ 0` (system improves under stress): Score capped at `10.0×`
- If `baseline_degradation ≤ 0`: Score set to `1.0` (no meaningful comparison)

**Interpretation:**
| Score | Classification |
|-------|----------------|
| < 1.0 | Fragile (adaptive degrades more than baseline) |
| = 1.0 | Robust (equal degradation) |
| 1.0 - 2.0 | Resilient (less degradation) |
| > 2.0 | Antifragile (significantly less degradation) |
| > 5.0 | Highly Antifragile |

## Certification Thresholds

### Pass Criteria

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| Δp99 (burst) | > 0 ms | Adaptive must be faster under stress |
| Antifragility Score | ≥ 1.2× | Must show measurable improvement |
| Δp99% | > 10% | Meaningful latency reduction |

### Hard Gates (Stage 1)

These structural gates must pass before performance certification:

| Gate | Threshold | Description |
|------|-----------|-------------|
| Parameter Reduction | ≥ 97% | Low-rank + sparse factorization efficiency |
| EPR-CV | ≤ 0.15 | Training stability (coefficient of variation) |
| Topological Gap | ≤ 2% | Spectral gap preservation |
| Sparsity | ≥ 98% | Mask density threshold |

### Performance Gates (Stage 2)

| Metric | Target | Description |
|--------|--------|-------------|
| PGU p95 Latency | ≤ 120ms | Verification query response time |
| PGU Cache Hit Rate | ≥ 50% | Efficiency of topology caching |
| CXL Memory Latency | < 2μs | Memory tier access time |
| Energy Savings | 10-100× | vs. dense baseline |

## Measurement Methodology

### Test Configuration

```yaml
# Default certification parameters
duration_seconds: 15
burst_factor: 2.0
warmup_iterations: 50
measurement_iterations: 200
```

### Baseline Executor

The baseline executor represents a conventional dense transformer:
- Full dense attention matrices
- Static inference policy
- No topology-aware caching
- No adaptive load shedding

### Adaptive Executor

The adaptive executor represents the full TF-A-N stack:
- Sparse attention via SNN low-rank factorization
- L3 metacontrol with temperature/memory adaptation
- PGU-verified topology caching
- AEPO structural adaptation under stress
- CLV-driven conservative mode triggers

### Workload Generation

```python
# Normal load: Poisson arrivals at base rate
normal_rate = 100  # requests/second

# Burst load: Sustained high rate
burst_rate = burst_factor * normal_rate  # e.g., 200 req/s at 2.0×
```

## Cognitive Load Vector (CLV)

The CLV consolidates L1/L2 metrics into a single vector for L3 control decisions:

### Components

| Component | Formula | Description |
|-----------|---------|-------------|
| CLV_Instability | `0.4×EPR_CV + 0.35×topo_gap + 0.15×neg_valence + 0.1×trend` | Training/inference stability |
| CLV_Resource | `0.5×jerk + 0.3×latency_pressure + 0.2×memory_pressure` | Resource utilization stress |
| CLV_Structural | `0.4×(1-keep_ratio) + 0.35×(1-spectral_norm) + 0.25×entropy_deficit` | Model structure health |

### Risk Levels

| Level | Condition | Action |
|-------|-----------|--------|
| LOW | All components < 0.3 | Normal operation |
| MEDIUM | Any component 0.3-0.6 | Increase monitoring |
| HIGH | Any component 0.6-0.8 | Trigger conservative mode |
| CRITICAL | Any component > 0.8 | Emergency fallback |

## Architecture Integration

The A-Cert pipeline validates the complete feedback loop:

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

The antifragility emerges from:
1. **Early stress detection** via L2 PAD state (negative valence, high arousal)
2. **Preemptive adaptation** via L3 conservative policy before degradation
3. **Structural optimization** via AEPO entropy-regularized updates
4. **Verified correctness** via PGU topological constraints
5. **Optimal routing** via semantic backend selection

This closed-loop architecture ensures the system doesn't merely survive stress but actively reconfigures to perform better under adverse conditions.

## Canonical Results

### Phase 1 Certification (2024)

| Condition | Baseline p99 | Adaptive p99 | Δp99 | Δp99% |
|-----------|--------------|--------------|------|-------|
| Normal Load | 17.97ms | 7.14ms | +10.83ms | 60.3% |
| Burst Load (2×) | 21.68ms | 3.90ms | +17.78ms | 82.0% |

**Antifragility Score: 2.21×**

Key observations:
- Baseline degrades 17.97ms → 21.68ms (+20.6%) under burst
- Adaptive **improves** 7.14ms → 3.90ms (-45.4%) under burst
- The adaptive system exhibits true antifragility: performance improves under stress

### Closed-Loop Demo Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| L2 Valence | -0.308 | Negative (stress detected) |
| L2 Arousal | 0.770 | High (urgent response needed) |
| L3 Temperature Mult | 0.638 | Conservative (reduced exploration) |
| AEPO Structural Change | +5.4% | Adaptation applied |
| PGU Constraints | All passed | β₁, β₀, λ₂ verified |
| Backend Selected | pgu_verified | 100% confidence |

## Running Certification

### Quick Certification

```bash
python scripts/certify_antifragility_delta.py \
    --burst-factor 2.0 \
    --duration 15 \
    --output results/certification.json
```

### Full A-Cert Pipeline

```bash
# Via GitHub Actions
gh workflow run a_cert.yml

# Or manually
python scripts/validate_all_gates.py --config configs/ci/ci_quick.yaml
python scripts/demo_closed_loop_antifragility.py --stress-level high
python scripts/certify_antifragility_delta.py --burst-factor 2.0 --duration 30
python scripts/generate_acert_report.py --artifacts-dir artifacts/a_cert --output reports/acert_report.md
```

## References

- Taleb, N.N. (2012). *Antifragile: Things That Gain from Disorder*
- TF-A-N Architecture: `docs/ARCHITECTURE.md`
- PGU Constraints: `tfan/pgu/README.md`
- L3 Metacontrol: `tfan/l3/README.md`
