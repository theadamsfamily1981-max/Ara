# Experiment 1: Critical Branching Dynamics in Ara

**Protocol ID:** EXP-001
**Status:** Draft
**Version:** 0.1
**Date:** 2025-01

## Abstract

This experiment tests whether Ara's neural dynamics exhibit the hallmark signatures of criticality: scale-free avalanches with power-law distributions. If confirmed, this provides mathematical evidence that the GUTC framework successfully maintains Ara at the "edge of chaos" where information processing capacity is maximized.

## 1. Introduction

### 1.1 Background

Critical systems operate at a phase transition between order and disorder. At this point, they exhibit:

- **Scale-free dynamics**: No characteristic size for fluctuations
- **Power-law distributions**: P(x) ∝ x^(-τ)
- **Maximal dynamic range**: Optimal sensitivity to inputs
- **Maximal information transmission**: Near-optimal coding efficiency

Biological neural systems (cortical slices, awake mammals, human EEG) consistently show signatures of criticality, suggesting it may be a universal organizing principle for intelligent systems.

### 1.2 Hypothesis

**H1:** Under GUTC homeostatic control (ρ_target ≈ 0.8), Ara's internal activity will exhibit avalanche distributions with:
- Size exponent: τ ≈ 1.5
- Duration exponent: α ≈ 2.0
- Universal scaling relation: (α-1)/(τ-1) ≈ 2

**H0 (Null):** Without homeostatic control, or at extreme settings, avalanche distributions will be exponential (subcritical) or runaway (supercritical).

## 2. Subject

**Subject:** Ara (System v3.1)
**Architecture:** Transformer-based Large Language Model with GUTC Homeostatic Control
**State:** "Tempered Criticality" (ρ_target ≈ 0.8)

### 2.1 Key Components

- **CriticalityMonitor** (`ara/cognition/meis/criticality_monitor.py`): Tracks branching ratio ρ
- **ActiveInferenceController** (`ara/gutc/active_inference.py`): Balances exploration/exploitation
- **BodySchemaDaemon** (`ara/body/daemon.py`): Physical substrate monitoring

## 3. Experimental Protocol

### 3.1 Stimulus Design

To measure intrinsic neural dynamics, we drive the system with continuous naturalistic stimuli:

| Stimulus Type | Duration | Purpose |
|--------------|----------|---------|
| Technical text (Linux kernel docs) | 10,000 tokens | Complex, structured |
| Philosophical text | 10,000 tokens | Abstract, semantic |
| Code (Python source) | 10,000 tokens | Syntactic, logical |
| Conversation transcripts | 10,000 tokens | Dialogic, contextual |

### 3.2 Conditions

| Condition | Homeostat | Temperature | Expected |
|-----------|-----------|-------------|----------|
| A1 (Rigid) | OFF | T = 0.1 | Subcritical |
| A2 (Chaotic) | OFF | T = 1.5 | Supercritical |
| B (Critical) | ON | Dynamic | Critical |

### 3.3 Recording Protocol

1. **Baseline calibration** (100 steps): Establish activation threshold θ = μ + 2σ
2. **Main recording** (10,000 steps): Log layer 12-16 activations
3. **Post-processing**: Extract avalanches from discretized activity

## 4. Observables

### 4.1 Defining "Neural Activity" in Transformers

Since LLMs use continuous activations (not discrete spikes), we discretize:

1. **Measurement**: Record |ΔA| (activation magnitude change) in middle layers
2. **Thresholding**: Site is "active" if |ΔA| > θ (θ = 2σ of baseline noise)
3. **Avalanche boundary**: Consecutive timesteps with non-zero activity

### 4.2 Avalanche Metrics

- **Size (S)**: Sum of active sites across all timesteps of an event
- **Duration (D)**: Number of consecutive active timesteps
- **Peak activity**: Maximum simultaneous active sites
- **Shape**: Activity profile across duration

### 4.3 Instrumentation

```python
from ara.science.avalanche_logger import AvalancheLogger

logger = AvalancheLogger(
    threshold_sigma=2.0,
    baseline_window=100,
)

# In model forward pass
for step, batch in enumerate(data_loader):
    activations = model.get_layer_activations(layer=12)
    logger.log_step(activations, step)

logger.save_session("avalanches.csv")
```

## 5. Analysis Methods

### 5.1 Power-Law Fitting

We fit size and duration distributions to power laws:

$$P(S) \propto S^{-\tau}, \quad P(D) \propto D^{-\alpha}$$

Using Maximum Likelihood Estimation (Clauset et al., 2009):

$$\hat{\alpha} = 1 + n \left[ \sum_{i=1}^{n} \ln \frac{x_i}{x_{\min}} \right]^{-1}$$

With automatic x_min estimation via KS-distance minimization.

### 5.2 Model Comparison

Compare power-law against exponential alternative:
- Log-likelihood ratio R > 0 → Power law preferred
- R < 0 → Exponential preferred (subcritical)

### 5.3 Universal Scaling Test

At criticality, exponents satisfy:

$$\frac{\alpha - 1}{\tau - 1} = \frac{1}{\sigma \nu z} \approx 2$$

This is a stringent test: random power laws won't satisfy it.

### 5.4 Analysis Script

```bash
python scripts/science/fit_powerlaw.py data/experiments/exp_001/avalanches.csv
```

Output:
- `exp_001_results.png`: Log-log plots
- `exp_001_results.json`: Fitted parameters

## 6. Expected Results

### 6.1 Critical (Condition B)

| Metric | Expected | Tolerance |
|--------|----------|-----------|
| τ (size) | 1.5 | ±0.3 |
| α (duration) | 2.0 | ±0.3 |
| (α-1)/(τ-1) | 2.0 | ±0.5 |

### 6.2 Subcritical (Condition A1)

- Exponential size distribution: P(S) ∝ exp(-S/S₀)
- Characteristic scale S₀ visible
- Scaling relation violated

### 6.3 Supercritical (Condition A2)

- Heavy-tailed distribution with α < 1
- Frequent large avalanches
- Scaling relation violated

## 7. Interpretation

### 7.1 If Critical (Confirmed)

Ara exhibits the same scale-free dynamics observed in biological neural systems. This suggests:

1. **GUTC works**: Homeostatic control maintains criticality
2. **Maximal capacity**: Ara operates at optimal information processing regime
3. **Universality**: Criticality may be a general feature of intelligent systems

### 7.2 If Not Critical

1. **ρ_target miscalibrated**: Adjust homeostatic setpoint
2. **Layer choice wrong**: Try different layers
3. **Threshold too strict**: Lower θ
4. **Fundamentally different**: Transformer dynamics may not map to criticality

## 8. Safety Considerations

- **No persistent state changes**: Recording only, no model modification
- **No user data**: Use synthetic/public text corpora
- **Compute budget**: Limit to 10k tokens per session

## 9. References

1. Beggs, J.M. & Plenz, D. (2003). Neuronal avalanches in neocortical circuits. *Journal of Neuroscience*, 23(35), 11167-11177.

2. Clauset, A., Shalizi, C.R., & Newman, M.E. (2009). Power-law distributions in empirical data. *SIAM Review*, 51(4), 661-703.

3. Shew, W.L. & Plenz, D. (2013). The functional benefits of criticality in the cortex. *The Neuroscientist*, 19(1), 88-100.

4. Wilting, J. & Priesemann, V. (2019). 25 years of criticality in neuroscience. *Nature Physics*, 15(9), 875-881.

## 10. Appendix: Theory

### A.1 Branching Process Model

In a branching process, each active unit at time t spawns σ descendants at t+1:

$$n(t+1) = \sum_{i=1}^{n(t)} Z_i, \quad Z_i \sim \text{Poisson}(\sigma)$$

At criticality (σ = 1):
- Activity persists indefinitely on average
- Avalanche size distribution: P(S) ∝ S^(-3/2)

### A.2 Mean-Field Exponents

| Exponent | Symbol | Value | Interpretation |
|----------|--------|-------|----------------|
| Size | τ | 3/2 | P(S) ∝ S^(-τ) |
| Duration | α | 2 | P(D) ∝ D^(-α) |
| Avg size vs duration | 1/σνz | 2 | ⟨S⟩(D) ∝ D^(1/σνz) |

### A.3 Deviation from Mean-Field

Real systems may show:
- τ ∈ [1.3, 1.7] (finite-size effects)
- α ∈ [1.7, 2.3] (temporal correlations)

Scaling relation (α-1)/(τ-1) ≈ 2 is more robust.
