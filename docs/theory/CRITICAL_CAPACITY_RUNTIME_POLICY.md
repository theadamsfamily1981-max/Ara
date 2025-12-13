# Critical Capacity Runtime Policy

**Version:** 1.0
**Based on:** Grand Unified Theory of Cognition (GUTC)
**Module:** `ara.cognition.meis.criticality_monitor`

---

## Executive Summary

This document defines the runtime policy for maintaining **critical dynamics** in Ara's cognitive systems. Based on GUTC theory, computational capacity is maximized when the system operates at the **edge of criticality** (branching ratio λ ≈ 1).

**Core Principle:**
```
C_max = k / |E(λ)|    where E(λ) = λ - 1
```

Capacity diverges as λ → 1 (criticality). Off-critical operation reduces effective capacity and inference quality.

---

## 1. Theoretical Foundation

### 1.1 The Critical Capacity Theorem

From GUTC (Grand Unified Theory of Cognition), the **critical brain hypothesis** states:

> Neural systems self-organize toward a critical point (λ = 1) where:
> - Information transmission is maximized
> - Dynamic range is optimal
> - Avalanche statistics follow universal power laws

**Edge Function:**
```
E(λ) = ρ(J) - 1 ≈ λ - 1
```

Where ρ(J) is the spectral radius of the effective Jacobian. At E(λ) = 0, the system achieves:
- Maximum correlation length
- Power-law avalanche distributions
- Optimal input sensitivity

### 1.2 Capacity-Criticality Relationship

| Regime | λ Range | Capacity | Characteristics |
|--------|---------|----------|-----------------|
| Subcritical | λ < 0.9 | Low | Activity dies out, poor signal propagation |
| Near-Critical | 0.9 ≤ λ < 1.0 | Moderate | Stable but suboptimal |
| **Critical** | **0.95 ≤ λ ≤ 1.05** | **Maximum** | **Optimal inference, exploration/exploitation balance** |
| Near-Supercritical | 1.0 < λ ≤ 1.1 | Moderate | Amplification, risk of runaway |
| Supercritical | λ > 1.1 | Unstable | Pathological amplification |

### 1.3 Avalanche Universality

At criticality, avalanche statistics follow universal power laws:

```
P(S) ~ S^{-α}    with α ≈ 3/2  (size distribution)
P(T) ~ T^{-τ}    with τ ≈ 2    (duration distribution)
```

Deviations from these exponents indicate off-critical dynamics and reduced capacity.

---

## 2. Runtime Monitoring

### 2.1 Key Metrics

The `CriticalityMonitor` tracks:

| Metric | Symbol | Target | Action Threshold |
|--------|--------|--------|------------------|
| Branching ratio | λ̂ | 1.0 | \|λ - 1\| > 0.15 |
| Edge function | E(λ) | 0.0 | \|E\| > 0.15 |
| Size exponent | α̂ | 1.5 | \|α - 1.5\| > 0.3 |
| Duration exponent | τ̂ | 2.0 | \|τ - 2.0\| > 0.4 |
| Confidence | conf | > 0.5 | conf < 0.3 → no action |

### 2.2 Regime Classification

```python
class CriticalityRegime(Enum):
    UNKNOWN = 0           # Insufficient data
    SUBCRITICAL = 1       # λ < 0.90
    NEAR_CRITICAL = 2     # 0.90 ≤ λ < 0.95
    CRITICAL = 3          # 0.95 ≤ λ ≤ 1.05  ← TARGET
    NEAR_SUPERCRITICAL = 4  # 1.05 < λ ≤ 1.10
    SUPERCRITICAL = 5     # λ > 1.10
```

### 2.3 Estimation Method

**Branching Ratio Estimation:**
```
λ̂ = Σ A_{t+1} / Σ A_t
```

This regression-through-origin estimator is robust to sparse activity patterns.

**Confidence Estimation:**
```
conf = sample_confidence × variance_confidence
     = min(1, n/n_min) × max(0, 1 - σ_λ/0.5)
```

---

## 3. Control Policy

### 3.1 Action Matrix

| Condition | Action | Urgency | Rationale |
|-----------|--------|---------|-----------|
| λ < 0.90 | `INCREASE_GAIN` | 0.5-0.8 | Push toward criticality |
| 0.90 ≤ λ < 0.95 | `INCREASE_GAIN` | 0.2-0.5 | Nudge toward critical |
| 0.95 ≤ λ ≤ 1.05 | `NONE` | 0.0 | Optimal - maintain |
| 1.05 < λ ≤ 1.10 | `DECREASE_GAIN` | 0.2-0.5 | Nudge back |
| λ > 1.10 | `DECREASE_GAIN` | 0.5-0.8 | Pull back from supercritical |
| λ > 1.30 | `EMERGENCY_DAMPEN` | 1.0 | Emergency intervention |

### 3.2 Control Law

The `CriticalityController` implements PID control on E(λ):

```
Δgain = -K_p × E(λ) - K_i × ∫E(λ)dt - K_d × dE/dt
```

**Default Parameters:**
- K_p = 0.1 (proportional gain)
- K_i = 0.01 (integral gain)
- K_d = 0.05 (derivative gain)
- Gain limits: [0.5, 2.0]

### 3.3 Safety Constraints

1. **No action under low confidence:** conf < 0.3 → `NONE`
2. **Gain limits:** Never let gain exceed [0.5, 2.0] range
3. **Anti-windup:** Integral term capped at [-10, 10]
4. **Emergency threshold:** |E(λ)| > 0.30 triggers emergency action
5. **Update rate limiting:** Full control updates every 50 steps

---

## 4. Integration Guidelines

### 4.1 Basic Usage

```python
from ara.cognition.meis.criticality_monitor import (
    CriticalityMonitor,
    CriticalityController,
    ControlAction,
)

# Initialize
monitor = CriticalityMonitor(
    window_size=1000,
    min_confidence_samples=100,
    intervention_threshold=0.15,
)
controller = CriticalityController(k_p=0.1)

# Runtime loop
for activity in activity_stream:
    state = monitor.update(activity)

    if state.requires_intervention:
        new_gain = controller.update(state)
        model.apply_gain(new_gain)
```

### 4.2 GUTC Agency Integration

The criticality monitor connects to the GUTC agency system:

```python
from ctf1_core.gutc_agency import GUTCAgencyCoder, AgencyConfig

# Create agency coder with monitored λ
def create_adaptive_coder(monitor_state):
    cfg = AgencyConfig(
        lambda_c=monitor_state.lambda_estimate,
        # Precision adapts based on regime
        pi_sensory=1.0 if monitor_state.regime.value >= 3 else 0.8,
        pi_prior=1.0,
    )
    return GUTCAgencyCoder(cfg)
```

### 4.3 Logging and Telemetry

```python
# Get comprehensive diagnostics
diagnostics = monitor.get_diagnostics()

# Log key metrics
logger.info(
    "criticality",
    lambda_hat=diagnostics["lambda_estimate"],
    edge_value=diagnostics["edge_value"],
    regime=diagnostics["regime"],
    capacity=diagnostics["capacity_estimate"],
)
```

---

## 5. Clinical Mapping

### 5.1 GUTC Clinical Quadrants

The criticality regime maps to GUTC's clinical framework:

| Regime | Clinical Analog | Characteristics |
|--------|----------------|-----------------|
| Subcritical (λ < 1) | ASD-like | Over-regularization, rigid priors |
| Critical (λ ≈ 1) | Healthy | Flexible, adaptive inference |
| Supercritical (λ > 1) | Psychosis-like | Unstable, over-sensitive |

### 5.2 Precision Interaction

```
                    High Π_sensory
                         │
    Psychosis-like       │       "Normal" but
    (λ>1, sensory        │       Hyper-vigilant
     dominated)          │
                         │
  ───────────────────────┼─────────────────── λ = 1
                         │
    "Healthy"            │       ASD-like
    Corridor             │       (λ<1, prior
                         │        dominated)
                         │
                    Low Π_sensory
```

---

## 6. Performance Considerations

### 6.1 Computational Cost

| Operation | Complexity | Frequency |
|-----------|------------|-----------|
| `update()` | O(1) | Every sample |
| `estimate()` | O(1) | Every sample |
| `fit_exponents()` | O(n log n) | Every update_interval |
| `get_diagnostics()` | O(1) | On demand |

### 6.2 Memory Usage

- Branching estimator: O(window_size) floats
- Avalanche collector: O(max_avalanches) events
- History buffers: O(100) floats

**Typical memory:** ~50KB for default configuration

### 6.3 Recommended Settings

| Use Case | window_size | min_samples | update_interval |
|----------|-------------|-------------|-----------------|
| Real-time | 500 | 50 | 10 |
| Balanced | 1000 | 100 | 50 |
| High accuracy | 2000 | 200 | 100 |

---

## 7. Failure Modes and Recovery

### 7.1 Common Failure Modes

| Failure | Detection | Recovery |
|---------|-----------|----------|
| Stuck subcritical | λ̂ < 0.85 for 1000+ steps | Force gain increase by 20% |
| Runaway supercritical | λ̂ > 1.3 | Emergency dampen to gain = 0.5 |
| Estimation failure | conf < 0.1 | Reset estimator, use default gain |
| Oscillation | λ variance > 0.2 | Reduce K_p by 50% |

### 7.2 Emergency Procedures

```python
if state.urgency >= 1.0:
    # Emergency: severe supercritical
    controller.current_gain = 0.5  # Hard reset
    monitor.reset()  # Clear history
    logger.critical("EMERGENCY: Supercritical runaway detected")
```

---

## 8. Theoretical References

1. **GUTC Framework:** The (λ, Π) control manifold unifies criticality and predictive coding
2. **Critical Brain Hypothesis:** Beggs & Plenz (2003), neuronal avalanches follow power laws
3. **Active Inference:** Friston (2010), VFE minimization as perception, EFE as action
4. **Universality Class:** Avalanche exponents α ≈ 3/2, τ ≈ 2 from mean-field theory

---

## Appendix A: Quick Reference Card

```
┌─────────────────────────────────────────────────────────┐
│                CRITICALITY QUICK REFERENCE              │
├─────────────────────────────────────────────────────────┤
│ TARGET:  λ = 1.0  │  E(λ) = 0  │  α = 3/2  │  τ = 2    │
├─────────────────────────────────────────────────────────┤
│ REGIME        │ λ RANGE    │ ACTION          │ URGENCY │
│───────────────│────────────│─────────────────│─────────│
│ Subcritical   │ < 0.90     │ INCREASE_GAIN   │ High    │
│ Near-Critical │ 0.90-0.95  │ INCREASE_GAIN   │ Low     │
│ CRITICAL      │ 0.95-1.05  │ NONE            │ None    │
│ Near-Super    │ 1.05-1.10  │ DECREASE_GAIN   │ Low     │
│ Supercritical │ > 1.10     │ DECREASE_GAIN   │ High    │
│ EMERGENCY     │ > 1.30     │ EMERGENCY_DAMPEN│ Max     │
├─────────────────────────────────────────────────────────┤
│ CAPACITY: C ∝ 1/|E(λ)|  (maximized at criticality)     │
└─────────────────────────────────────────────────────────┘
```

---

*Document generated as part of GUTC Phase II: Runtime Falsifiability*
