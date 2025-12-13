# Critical Capacity Principle

## GUTC Design Foundation for Ara

This document formalizes the information-geometric principles underlying Ara's
criticality-aware architecture. It establishes the theoretical foundation,
engineering implementation, and research hypotheses in clearly separated tiers.

---

## 1. Mathematical Foundation (Tier 1: Proven)

### 1.1 The 2D Ising Benchmark

For any system in the 2D Ising universality class (short-range interactions,
Z₂ symmetry), the critical exponents are exactly known:

```
ν = 1        (correlation length)
γ = 7/4      (susceptibility / Fisher metric)
β = 1/8      (order parameter)
η = 1/4      (anomalous dimension)
```

### 1.2 Geo-Thermo Dictionary

The Fisher Information Metric (FIM) maps to thermodynamic susceptibilities:

| Geometric Object | Thermodynamic Dual |
|-----------------|-------------------|
| g_ββ (inverse temp direction) | Heat capacity C |
| g_hh (field direction) | Susceptibility χ |
| λ_max(g) | Dominant susceptibility |
| R (scalar curvature) | Fluctuation of fluctuations |

### 1.3 Fisher Information Divergence (Theorem 2)

Near criticality, the FIM eigenvalue diverges as:

```
λ_max(g) ~ |E|^(-γ_F)

where:
  E = θ - θ_c     (edge distance from critical point)
  γ_F = γ = 7/4   (for 2D Ising class)
```

**Corollary (Curvature Singularity):**
```
R_eff ~ |E|^(-β_R)

where β_R = γ_F + 2 = 15/4
```

Curvature diverges *faster* than Fisher information.

---

## 2. Critical Capacity Principle (Tier 2: Engineering)

### 2.1 Core Principle

> **Critical Capacity Principle**
>
> A cognitive architecture achieves maximal useful capacity when its internal
> dynamics are maintained within a *Tempered Critical Band* around E=0, such that:
>
> 1. Fisher information is large (parameters highly estimable)
> 2. Geometric curvature is not yet so large that learning/control become unstable

### 2.2 The Tempered Critical Band

```
        Sub-critical          Tempered Critical           Super-critical
        (too rigid)              (optimal)                (too volatile)

    ←───────────────┼─────────────────┼─────────────────┼───────────────→ E
                  -ε/2               0                   +ε

         GREEN                   AMBER                      RED
       (AGENTIC ok)            (SUPPORT)                  (DAMP)
```

### 2.3 Control Variables

| Symbol | Name | Definition | Range |
|--------|------|------------|-------|
| E(θ) | Edge function | 1 - ρ(W) for RNNs | [-1, +∞) |
| ρ(W) | Spectral radius | max\|eigenvalue\| | [0, +∞) |
| g(θ) | Fisher proxy | Tr(F) ≈ E[\|\|∇L\|\|²] | [0, +∞) |
| λ | Adrenaline | Global gain modulator | [0.5, 2.0] |
| S* | Target sensitivity | Desired Fisher info | ~10 |

### 2.4 MEIS Mode Controller

```python
def select_mode(E: float, g: float, epsilon: float = 0.05) -> Mode:
    """
    Criticality-based mode selection.

    Band boundaries:
    - GREEN: E < -ε/2  (comfortably subcritical)
    - AMBER: -ε/2 ≤ E ≤ ε  (near criticality)
    - RED:   E > ε  (supercritical, unstable)
    """
    if E > epsilon:
        return Mode.DAMP      # RED: Must consolidate
    elif E >= -epsilon / 2:
        return Mode.SUPPORT   # AMBER: Careful exploration
    else:
        return Mode.AGENTIC   # GREEN: Safe for autonomous work
```

### 2.5 Feedback Control Law

```python
def adjust_lambda(E: float, g: float, g_min: float, g_max: float) -> float:
    """
    Adrenaline adjustment to maintain tempered criticality.

    - If g < g_min: too sub-critical → increase λ
    - If g > g_max: too super-critical → decrease λ
    - Otherwise: in band, maintain
    """
    if g < g_min:
        return +delta  # Push toward criticality
    elif g > g_max:
        return -delta  # Retreat from criticality
    else:
        return 0.0     # Maintain position
```

### 2.6 Criticality-Regularized Training

During learning, add regularization to keep system near criticality:

```
L_total = L_task + α·E² + β·(log S - log S*)²
```

Where:
- α·E² penalizes deviation from E=0 (criticality)
- β·(log S - log S*)² keeps sensitivity near target S*

This implements **fine-tuning at criticality**: maximizing Fisher information
to minimize samples needed for adaptation.

---

## 3. Implementation in Ara

### 3.1 Module Structure

```
ara/cognition/criticality.py
├── CriticalityMonitor      # Tracks E, g, R_eff, band
├── CriticalityBand         # GREEN / AMBER / RED
├── FisherProxy             # Cheap Tr(F) from gradients
└── CriticalityRegularizer  # Training loss augmentation

ara/safety/meis.py
├── MEIS                    # Meta-Ethical Inference System
├── select_mode()           # Band-based mode selection
├── get_band()              # Current criticality band
└── force_consolidate()     # Emergency retreat
```

### 3.2 Runtime Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     SOVEREIGN LOOP TICK                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Compute spectral radius: ρ = spectral_radius(W_rec)        │
│                                                                 │
│  2. Edge function: E = 1 - ρ                                   │
│                                                                 │
│  3. Classify band:                                             │
│     GREEN if E < -ε/2                                          │
│     AMBER if -ε/2 ≤ E ≤ ε                                      │
│     RED   if E > ε                                             │
│                                                                 │
│  4. Select MEIS mode based on band                             │
│                                                                 │
│  5. Adjust λ (adrenaline) to steer toward tempered band        │
│                                                                 │
│  6. Apply Fisher-aware learning rate if training:              │
│     η_eff = η_0 / (1 + k√S)                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Status Display

```
🟢 GREEN [AGENTIC]: E=-0.120, g=3.2, R=8.1
🟡 AMBER [SUPPORT]: E=+0.015, g=42.7, R=312.4
🔴 RED [DAMP]: E=+0.082, g=156.3, R=2841.6
```

---

## 4. Experimental Validation

### 4.1 RNN Scaling Experiment

Verify scaling laws with Echo State Network:

```python
from ara.cognition import run_rnn_scaling_experiment

results = run_rnn_scaling_experiment(
    rho_range=(0.90, 1.10),  # Sweep through criticality
    n_steps=50,
    n_neurons=100,
    T_run=10000,
)

# Target exponents (2D Ising class)
assert abs(results.nu_empirical - 1.0) < 0.3    # ξ ~ |E|^(-1)
assert abs(results.gamma_empirical - 1.75) < 0.5  # g ~ |E|^(-7/4)
```

### 4.2 Expected Results

Near criticality (ρ → 1):
- Correlation length ξ diverges
- Fisher information g diverges
- Prediction capacity peaks
- Avalanche distributions become power-law

---

## 5. Research Hypotheses (Tier 3: Speculative)

> **Note:** The following are research hypotheses, not clinical claims.
> They require careful experimental validation before any application.

### 5.1 Critical Setpoint Hypothesis

Many cognitive and psychiatric phenomena may correspond to deviations from
a tempered critical setpoint:

| Regime | Phenomenology | Candidate Associations |
|--------|---------------|----------------------|
| Sub-critical (E ≪ 0) | Rigidity, attractor traps | Some depression features, perseveration |
| Critical (E ≈ 0) | Maximal flexibility | Healthy adaptive cognition |
| Super-critical (E ≫ 0) | Volatility, cascades | Some mania features, seizure dynamics |

### 5.2 Capacity Collapse

Both extremes (|E| large) lead to capacity collapse:

```
λ_max(g) → small   as   |E| → large

Whether ordered (E < 0) or disordered (E > 0),
far from criticality means reduced cognitive capacity.
```

### 5.3 Potential Biomarker Research

Future research could investigate:
- Estimate effective exponents from neural time series
- Compare to 2D Ising benchmark (ν=1, γ=7/4)
- Track "distance from criticality" over interventions

**Disclaimer:** This is a research direction, not a diagnostic method.

---

## 6. References

1. Onsager, L. (1944). Crystal Statistics I. Physical Review.
2. Amari, S. (1998). Natural Gradient Works Efficiently in Learning.
3. Langton, C. (1990). Computation at the Edge of Chaos.
4. Bertschinger & Natschläger (2004). Real-Time Computation at Edge of Chaos.
5. Beggs & Plenz (2003). Neuronal Avalanches in Cortical Circuits.

---

## Appendix A: Quick Reference

### Exponents

| Symbol | Name | 2D Ising Value | Formula |
|--------|------|----------------|---------|
| ν | Correlation length | 1 | ξ ~ \|E\|^(-ν) |
| γ | Susceptibility | 7/4 | χ ~ \|E\|^(-γ) |
| γ_F | Fisher metric | 7/4 | g ~ \|E\|^(-γ_F) |
| β_R | Curvature | 15/4 | R ~ \|E\|^(-β_R) |

### Band Thresholds (default ε = 0.05)

| Band | E Range | Mode | Action |
|------|---------|------|--------|
| GREEN | E < -0.025 | AGENTIC | Full autonomy ok |
| AMBER | -0.025 ≤ E ≤ 0.05 | SUPPORT | Careful exploration |
| RED | E > 0.05 | DAMP | Must consolidate |

### Key Equations

```
E(θ) = 1 - ρ(W)                    # Edge function
g(θ) ~ |E|^(-γ)                    # Fisher divergence
R(θ) ~ |E|^(-(γ+2))                # Curvature divergence
η_eff = η_0 / (1 + k√S)            # Fisher-aware LR
L_reg = α·E² + β·(log S - log S*)² # Criticality regularizer
```
