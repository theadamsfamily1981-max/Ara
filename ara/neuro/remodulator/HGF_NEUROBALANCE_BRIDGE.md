# HGF to NeuroBalance Bridge: Precision-Weighted Errors and the Delusion Index

> This document formally connects the Hierarchical Gaussian Filter's precision-weighted
> prediction errors (δᵢ) to the NeuroBalance device's Π_prior, Π_sensory, and D metrics.

---

## 1. The Core Insight

The HGF update equation at Level 2 contains the fundamental structure of D:

```
μ₂^(k) = μ̂₂^(k) + [π₁ / π₂^(k)] × δ₁^(k)
```

Where:
- `π̂₂` = **Prior precision** (from higher level, via volatility)
- `π₁ = ŝ(1-ŝ)` = **Likelihood precision** (sensory contribution)
- `π₂ = π̂₂ + π₁` = **Posterior precision** (total)

**The ratio `π₁ / π₂` is the inverse of D at Level 2!**

```
D₂ = π̂₂ / π₁ = Π_prior / Π_sensory
```

When D₂ > 1: Prior dominates → beliefs resist sensory updating
When D₂ < 1: Sensory dominates → beliefs track observations closely

---

## 2. Formal Mapping

### 2.1 HGF Precisions → NeuroBalance Metrics

| HGF Term | Symbol | NeuroBalance Equivalent | Formula |
|----------|--------|-------------------------|---------|
| Predicted precision (Level 2) | π̂₂ | Π_prior (cognitive) | 1/exp(κ₁μ₃ + ω₁) |
| Likelihood precision | π₁ | Π_sensory (Level 1) | ŝ(1-ŝ) |
| Predicted precision (Level 3) | π̂₃ | Π_prior (volatility) | 1/exp(κ₂μ₄ + ω₂) |
| Posterior precision (Level 2) | π₂ | Π_total (after update) | π̂₂ + π₁ |

### 2.2 D Metrics from HGF

```python
# D at Level 2 (fast timescale, perceptual)
D_fast = pi_hat2 / max(pi1, 1e-10)

# D at Level 3 (slow timescale, cognitive)
D_slow = pi_hat3 / max(pi2, 1e-10)

# Hierarchical D mapping to NeuroBalance
D_high = D_fast   # Perceptual level (sensory-belief ratio)
D_low = D_slow    # Cognitive level (volatility-belief ratio)
```

---

## 3. Prediction Errors as Precision-Weighted Signals

### 3.1 Error Hierarchy

| Level | Error | Drives | Precision Weight | Interpretation |
|-------|-------|--------|------------------|----------------|
| δ₁ | u - ŝ | μ₂ update | π₁/π₂ | "What I observed vs expected" |
| δ₂ | μ₂ - μ̂₂ | μ₃ update | π₂/π₃ | "How much belief changed" |
| δ₃ | μ₃ - μ̂₃ | μ₄ update | π₃/π₄ | "How much volatility changed" |
| δ₄ | μ₄ - μ̂₄ | μ₅ update | π₄/π₅ | "How much meta-volatility changed" |

### 3.2 Key Principle: Errors Only Propagate When Precision-Weighted

```
Effective Error @ Level i = (πᵢ₋₁ / πᵢ) × δᵢ₋₁
```

- If higher level has high precision (rigid prior), error attenuates
- If lower level has high precision (reliable signal), error amplifies

**This is exactly D modulating the error stream!**

---

## 4. Clinical Profiles as Precision Configurations

### 4.1 Schizophrenia (D >> 1)

```
HGF Configuration:
  κ₁ = 0.3 (weak sensory-to-belief coupling)
  ω₁ = 3.0 (high base volatility)

Result:
  π̂₂ = 1/exp(0.3×μ₃ + 3.0) → Low prior precision BUT
  π₁ = ŝ(1-ŝ) → Normal sensory precision

Paradox Resolution:
  Despite low π̂₂, the GAIN on sensory errors is reduced by low κ₁
  → Beliefs don't track sensory evidence well
  → High EFFECTIVE D due to weak coupling, not high prior precision

NeuroBalance D Interpretation:
  D_effective = f(κ₁, ω₁) not just π̂₂/π₁
  Low κ₁ ≈ High D (prior-dominated behavior)
```

### 4.2 ASD (D << 1)

```
HGF Configuration:
  κ₁ = 2.0 (strong sensory-to-belief coupling)
  ω₁ = 1.0 (low base volatility)

Result:
  π̂₂ = 1/exp(2.0×μ₃ + 1.0) → Higher prior precision
  π₁ = ŝ(1-ŝ) → Normal sensory precision

But:
  High κ₁ amplifies sensory errors before they reach beliefs
  → Beliefs track observations very closely
  → Sensory-dominated despite potentially balanced π ratio

NeuroBalance D Interpretation:
  D_effective = f(κ₁, ω₁)
  High κ₁ ≈ Low D (sensory-dominated behavior)
```

### 4.3 Revised D Formula Incorporating Coupling

```python
def compute_effective_D(hgf_params, hgf_state):
    """
    Effective D accounts for coupling strength, not just precision ratio.
    """
    p = hgf_params
    s = hgf_state

    # Raw precision ratio
    s_hat = sigmoid(s.mu2)
    pi1 = s_hat * (1 - s_hat)
    sigma_hat2 = math.exp(p.kappa1 * s.mu3 + p.omega1)
    pi_hat2 = 1 / sigma_hat2

    D_raw = pi_hat2 / max(pi1, 1e-10)

    # Coupling-modulated effective D
    # Low κ₁ → effectively higher D (beliefs resist)
    # High κ₁ → effectively lower D (beliefs follow)
    D_effective = D_raw / max(p.kappa1, 0.1)

    return D_effective
```

---

## 5. Connecting HGF to EEG-Based D Estimation

### 5.1 The Bridge Hypothesis

```
EEG Observable          HGF Latent Variable         D Component
────────────────────────────────────────────────────────────────
Frontal θ power    →    π₃, π₄ (volatility)    →    Numerator of D_low
Posterior α power  →    π₂ (belief precision)  →    Denominator of D_low
Posterior γ power  →    π₁ (sensory precision) →    Denominator of D_high
θ-γ coupling       →    κ₁ (belief-sensory)    →    Coupling strength
```

### 5.2 EEG → HGF → D Pipeline

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   EEG Signals   │ ──▶ │  HGF Inversion  │ ──▶ │   D Metrics     │
│                 │     │                 │     │                 │
│ θ_frontal       │     │ Estimate:       │     │ D_low           │
│ α_posterior     │     │   κ₁, ω₁        │     │ D_high          │
│ γ_posterior     │     │   μ₃, μ₄, μ₅    │     │ ΔH              │
│ θ-γ PAC         │     │   π₂, π₃, π₄    │     │ Epistemic Depth │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### 5.3 Practical Implementation

```python
def eeg_to_hgf_parameters(eeg_features: dict) -> HGFParameters:
    """
    Map EEG features to HGF parameter estimates.

    This is a simplified linear mapping; real implementation would
    use Bayesian inversion with prior constraints.
    """
    # Frontal theta predicts volatility beliefs
    theta_z = eeg_features['theta_frontal_zscore']

    # Theta-gamma coupling predicts sensory-belief coupling
    pac_z = eeg_features['theta_gamma_pac_zscore']

    # Posterior alpha predicts belief stability
    alpha_z = eeg_features['alpha_posterior_zscore']

    # Map to HGF parameters (hypothetical linear relationships)
    kappa1 = 1.0 + 0.5 * pac_z      # PAC ↑ → coupling ↑
    omega1 = 2.0 + 0.3 * theta_z    # Theta ↑ → volatility ↑

    return HGFParameters(
        kappa1=max(0.1, min(3.0, kappa1)),
        omega1=max(0.5, min(5.0, omega1)),
        # Other parameters estimated similarly...
    )


def eeg_to_D_via_hgf(eeg_features: dict, behavior: list) -> dict:
    """
    Full pipeline: EEG → HGF params → Run HGF → Extract D.
    """
    # Step 1: Estimate HGF parameters from EEG
    params = eeg_to_hgf_parameters(eeg_features)

    # Step 2: Run HGF on behavioral data (if available)
    hgf = BinaryHGF(params=params)
    if behavior:
        hgf.run(behavior)

    # Step 3: Extract D metrics
    return estimate_D_from_hgf(hgf)
```

---

## 6. Prediction Error Dynamics and D

### 6.1 How D Affects Error Processing

```
High D (Prior-Dominated)
═══════════════════════
  δ₁ (sensory error) arrives
    │
    ▼
  π₁/π₂ is SMALL (prior precision >> sensory)
    │
    ▼
  Belief update is MINIMAL
    │
    ▼
  δ₂ is SMALL (belief barely changed)
    │
    ▼
  Higher levels don't update much
    │
    ▼
  System is STABLE but RIGID (hallucinatory if prior is wrong)


Low D (Sensory-Dominated)
═════════════════════════
  δ₁ (sensory error) arrives
    │
    ▼
  π₁/π₂ is LARGE (sensory precision >> prior)
    │
    ▼
  Belief update is STRONG
    │
    ▼
  δ₂ is LARGE (belief changed significantly)
    │
    ▼
  Higher levels may update volatility estimates
    │
    ▼
  System is RESPONSIVE but potentially NOISY (overwhelmed)
```

### 6.2 Precision-Weighted Error as Observable

The HGF framework predicts that **neural correlates of prediction errors should be precision-weighted**:

```
EEG/ERP Correlate = f(δᵢ × [πᵢ₋₁/πᵢ])
```

| ERP Component | Hypothesized Source | D Prediction |
|---------------|---------------------|--------------|
| MMN | δ₁ (sensory) × precision weight | Reduced in high D |
| P3a | δ₂ (belief) × precision weight | Reduced in low D |
| P3b | Higher-level context updating | Reduced in rigid systems |

---

## 7. Therapeutic Implications

### 7.1 tACS Targets from HGF Perspective

| tACS Frequency | HGF Target | Precision Effect | D Effect |
|----------------|------------|------------------|----------|
| θ (6 Hz) | π₃, π₄ (volatility) | ↓ volatility precision | ↓ D_low |
| α (10 Hz) | π₂ (belief stability) | ↓ prior precision | ↓ D |
| γ (40 Hz) | π₁ (sensory) | ↑ sensory precision | ↓ D_high |
| β (20 Hz) | Action precision | ↑ motor confidence | Agency |

### 7.2 Closed-Loop Protocol

```python
def adaptive_stimulation(hgf: BinaryHGF, target_D: float):
    """
    Adjust stimulation based on real-time HGF state.
    """
    current_D = estimate_D_from_hgf(hgf)['D_low']

    error = current_D - target_D

    if error > 0.5:  # D too high
        # Reduce prior precision via theta-tACS
        return StimulationParams(frequency=6, intensity=1.5)
    elif error < -0.5:  # D too low
        # Reduce sensory precision via gamma-tACS
        return StimulationParams(frequency=40, intensity=1.0)
    else:
        # D near target, reduce stimulation
        return StimulationParams(frequency=0, intensity=0)
```

---

## 8. Validation Predictions

### 8.1 HGF Parameter Recovery

If the framework is correct:
1. HGF parameters estimated from behavior should correlate with EEG-derived D
2. Manipulating tACS should shift both HGF parameters AND EEG D metrics
3. Clinical groups should show consistent HGF-D mapping

### 8.2 Specific Predictions

| Prediction | Test | Expected Result |
|------------|------|-----------------|
| κ₁ correlates with θ-γ PAC | Pearson correlation | r > 0.3 |
| ω₁ correlates with frontal θ | Pearson correlation | r > 0.3 |
| D_low (HGF) correlates with D_low (EEG) | Pearson correlation | r > 0.5 |
| θ-tACS reduces D_low | Pre-post comparison | d > 0.3 |

---

## 9. Summary: The Unified Precision Framework

```
                    ┌───────────────────────────────────────────────────┐
                    │           UNIFIED PRECISION FRAMEWORK             │
                    └───────────────────────────────────────────────────┘
                                           │
              ┌────────────────────────────┼────────────────────────────┐
              │                            │                            │
              ▼                            ▼                            ▼
    ┌─────────────────┐          ┌─────────────────┐          ┌─────────────────┐
    │ HGF (Behavior)  │          │ NeuroBalance    │          │ EEG Observables │
    │                 │          │ (Device)        │          │                 │
    │ κ₁, ω₁, θ       │◀────────▶│ D_low, D_high   │◀────────▶│ θ, α, γ, PAC    │
    │ π₂, π₃, π₄      │          │ ΔH              │          │                 │
    │ δ₁, δ₂, δ₃      │          │ Epistemic Depth │          │                 │
    └─────────────────┘          └─────────────────┘          └─────────────────┘
              │                            │                            │
              │                            │                            │
              └────────────────────────────┼────────────────────────────┘
                                           │
                                           ▼
                              ┌───────────────────────────┐
                              │     Clinical Phenotype    │
                              │                           │
                              │ Schizophrenia: High D_eff │
                              │ ASD: Low D_eff, High ΔH   │
                              │ Depression: Rigid priors  │
                              └───────────────────────────┘
```

The HGF provides the **computational substrate** for the NeuroBalance D metrics:
- D is not just a ratio—it's the behavioral manifestation of precision-weighted inference
- tACS targets the neural oscillations that implement precision
- EEG observes the same precision dynamics that the HGF models

**Bottom Line**: The Delusion Index D emerges from the ratio of precisions in the HGF update equation, making the HGF the theoretical foundation and the NeuroBalance device its measurement/intervention tool.

---

## References

1. Mathys, C., et al. (2014). Uncertainty in perception and the HGF.
2. Adams, R.A., et al. (2013). Computational anatomy of psychosis.
3. Lawson, R.P., et al. (2014). Adults with autism overestimate volatility.
4. Friston, K. (2005). A theory of cortical responses.

---

*Bridge document connecting hgf.py to DEVICE_SPEC_V2.md*
*Part of the Brain Remodulator framework*
