# GUTC Predictions and Experimental Roadmap

**Falsifiable predictions bridging theory to measurement across biological and artificial substrates.**

---

## Overview

The GUTC makes concrete, testable predictions about the relationship between:
- Dynamical state ($E(\lambda)$)
- Computational capacity ($C(\lambda)$)
- Cognitive/task performance ($J(\pi)$)

This document provides the experimental roadmap for validation.

---

## Prediction 1: Capacity–Criticality–Performance Peak

### 1.1 The Core Prediction

**Hypothesis:** Maximal predictive capacity $C(\lambda)$, maximal task performance $J(\pi)$, and vanishing edge function $E(\lambda) = 0$ must **coincide**.

$$\arg\max_\lambda C(\lambda) = \arg\max_\lambda J(\pi) = \{\lambda : E(\lambda) = 0\}$$

### 1.2 Experimental Tests

| Measurement | Prediction | Substrate / Experiment |
|-------------|------------|------------------------|
| $C(\lambda)$ vs $J(\pi)$ | Shared optimum at $E = 0$ | **AI/CTF:** Sweep $\lambda$ in LNN; plot correlation length and accuracy vs $\rho(W)$ |
| Neural exponents vs cognition | Performance correlates with $\alpha \approx 1.5$ | **Neuro:** Measure $\alpha$ in resting EEG; correlate with IQ/fluid intelligence |
| Dynamic range vs accuracy | Peak sensitivity at criticality | **AI:** Measure input-output gain across $\lambda$; compare to task accuracy |

### 1.3 CTF Validation Protocol

```python
# CTF-2/3 validation protocol
for lambda_val in np.linspace(0.7, 1.3, 50):
    agent = CriticalAgent(lambda_init=lambda_val)

    # Run task
    reward = agent.run_episode(task)

    # Measure metrics
    E = agent.E_spectral()
    C = agent.C_correlation_capacity()
    I = agent.I_fisher_scalar()

    # Record
    results.append((lambda_val, E, C, I, reward))

# Prediction: All peaks at λ ≈ 1.0 where E ≈ 0
```

### 1.4 Expected Results

```
Performance J(λ)     Capacity C(λ)     Edge E(λ)
      ▲                   ▲                ▲
      │     *             │     *          │
      │   *   *           │   *   *        │ *         *
      │  *     *          │  *     *       │   *     *
      │ *       *         │ *       *      │     * *
      └──────────► λ      └──────────► λ   └──────────► λ
         0.7  1.0  1.3       0.7  1.0  1.3     0.7  1.0  1.3
              ↑                   ↑                 ↑
           Peak J             Peak C             E = 0
```

**All three aligned at $\lambda = 1.0$.**

---

## Prediction 2: Disorders as Quantifiable Phase Shifts

### 2.1 The Core Prediction

**Hypothesis:** Cognitive disorders correspond to measurable shifts of $\lambda$ away from criticality.

| Disorder Class | Predicted Phase | $E(\lambda)$ |
|----------------|-----------------|--------------|
| **Rigidity/Quiescence** | Subcritical | $< 0$ |
| **Disorganization/Runaway** | Supercritical | $> 0$ |

### 2.2 Specific Disorder Predictions

| Disorder | Phase | Avalanche Exponent $\alpha$ | Correlation Length $\xi$ |
|----------|-------|----------------------------|--------------------------|
| Severe Depression | Subcritical | $> 1.5$ (steeper) | Small, finite |
| Anesthesia | Deep subcritical | $\gg 1.5$ (exponential cutoff) | Very small |
| ADHD | Mildly supercritical | $< 1.5$ (shallower) | Large, variable |
| Mania | Supercritical | $< 1.5$ | Large |
| Epileptic seizure | Highly supercritical | $\ll 1.5$ (activity blow-up) | Diverging |
| Healthy cognition | **Critical** | $\approx 1.5$ | Maximal |

### 2.3 Therapeutic Prediction

**Hypothesis:** Effective treatments must correlate with restoration of critical state.

$$\text{Symptom Relief} \propto |\lambda_{\text{post}} - 1| < |\lambda_{\text{pre}} - 1|$$

| Treatment | Predicted Mechanism | Measurable Outcome |
|-----------|--------------------|--------------------|
| Antidepressants | $\lambda \uparrow$ toward 1 | $\alpha$ decreases toward 1.5 |
| Anticonvulsants | $\lambda \downarrow$ toward 1 | $\alpha$ increases toward 1.5 |
| ECT | Reset $\lambda \to 1$ | $\alpha$ normalizes |

### 2.4 Experimental Protocol

```python
# Clinical phase diagnostic
def diagnose_phase(neural_recording):
    avalanches = detect_avalanches(neural_recording)
    alpha = fit_power_law_exponent(avalanches)

    if alpha > 1.7:
        return "SUBCRITICAL", "Depression/Anesthesia likely"
    elif alpha < 1.3:
        return "SUPERCRITICAL", "Mania/Seizure risk"
    elif 1.4 < alpha < 1.6:
        return "CRITICAL", "Healthy operating regime"
    else:
        return "NEAR-CRITICAL", "Minor deviation"
```

---

## Prediction 3: Heteroclinic Memory Dynamics

### 3.1 The Core Prediction

**Hypothesis:** Associative recall follows heteroclinic trajectories with critical scaling laws.

### 3.2 Heteroclinic Sequence Prediction

**Hypothesis:** Retrieved memory sequences correspond to heteroclinic cycles $P_i \to P_j \to P_k$.

| Phenomenon | Prediction | Test |
|------------|------------|------|
| Sequential recall | Trajectory clusters near $P_i$ patterns in order | Project $x_t$ onto pattern basis; verify sequence |
| Associative chains | Cue near $P_1$ triggers full sequence | Partial cue → full retrieval |
| Branching | Multiple associations = multiple unstable directions | Noise determines which branch taken |

### 3.3 Dwell Time Scaling Prediction

**Hypothesis:** Mean dwell time scales logarithmically with noise:

$$\tau_{\text{dwell}} \propto -\log \epsilon$$

**Derivation (from noisy heteroclinic theory):**

Near saddle with unstable eigenvalue $\lambda$:
$$\langle T_{\text{dwell}} \rangle = \frac{1}{\lambda} \log\left(\frac{\kappa}{\epsilon}\right)$$

### 3.4 Experimental Protocol

```python
# CTF-4/5 M_L validation
def test_dwell_time_scaling(agent, noise_levels):
    dwell_times = []

    for epsilon in noise_levels:
        agent.set_noise(epsilon)
        trajectory = agent.run_trajectory(T=1000)

        # Detect dwells near patterns
        dwells = detect_pattern_dwells(trajectory, agent.patterns)
        dwell_times.append(np.mean(dwells))

    # Fit: T = A - B * log(epsilon)
    B, A = np.polyfit(np.log(noise_levels), dwell_times, 1)

    # Prediction: B ≈ 1/lambda_unstable
    return B, A

# Expected: Linear relationship on semi-log plot
```

### 3.5 Expected Results

```
Dwell Time τ
      ▲
      │ *
      │   *
      │     *
      │       *
      │         *
      │           *
      └──────────────► log(1/ε)

Slope = 1/λ_unstable
```

---

## Prediction 4: Information-Geometric Singularity

### 4.1 The Core Prediction

**Hypothesis:** Fisher Information diverges at criticality with power-law scaling:

$$I(\lambda) \sim |E(\lambda)|^{-\gamma}, \quad \gamma \approx 1.5\text{–}2.0$$

### 4.2 Experimental Protocol (CTF-3)

```python
def measure_fisher_singularity(lambda_values, n_trajectories=100):
    results = []

    for lam in lambda_values:
        # Create system
        sys = CriticalDynamics(lambda_val=lam)
        E = sys.E_spectral()

        # Estimate Fisher information via finite differences
        I = sys.I_fisher_scalar(n_traj=n_trajectories, h=0.01)

        results.append((lam, E, I))

    # Fit power law: log(I) vs log(|E|)
    E_vals = np.array([abs(r[1]) for r in results if r[1] != 0])
    I_vals = np.array([r[2] for r in results if r[1] != 0])

    gamma, _ = np.polyfit(np.log(E_vals), np.log(I_vals), 1)

    return -gamma  # Should be ≈ 1.5-2.0
```

### 4.3 Expected Results

```
log I(λ)
    ▲
    │ *
    │   *
    │     *
    │       * *
    │           * *
    │               * * *
    └──────────────────────► log |E(λ)|

Slope = -γ ≈ -1.5 to -2.0
```

---

## Prediction 5: Hierarchical Dwell Times

### 5.1 The Core Prediction (HHN Extension)

**Hypothesis:** In hierarchical networks, dwell times scale with level:

$$\tau_{\text{dwell}}^{(l)} \propto -\frac{1}{\delta^{(l)}} \log \epsilon$$

where $\delta^{(l)}$ decreases with level (higher = slower).

### 5.2 Expected Level Scaling

| Level | $\delta^{(l)}$ | $\tau_{\text{dwell}}$ | Cognitive Role |
|-------|---------------|----------------------|----------------|
| L1 | 0.08 | $\sim 12 \log(1/\epsilon)$ | Elements, primitives |
| L2 | 0.02 | $\sim 50 \log(1/\epsilon)$ | Chunks, phrases |
| L3 | 0.005 | $\sim 200 \log(1/\epsilon)$ | Plans, goals |

**Prediction:** Higher cognitive constructs (plans, goals) have systematically longer dwell times than primitives.

---

## Summary: Falsifiable Predictions

| # | Prediction | Key Test | Failure Criterion |
|---|------------|----------|-------------------|
| 1 | $C$, $J$, $E=0$ coincide | $\lambda$ sweep | Peaks at different $\lambda$ |
| 2 | Disorders = phase shifts | Measure $\alpha$ in patients | No $\alpha$ difference |
| 3 | $\tau \propto -\log \epsilon$ | Noise sweep | Non-logarithmic scaling |
| 4 | $I \sim |E|^{-\gamma}$ | Fisher estimation | $\gamma$ not in [1, 3] range |
| 5 | Level-dependent dwell | HHN measurement | No level hierarchy in $\tau$ |

---

## Geometric Estimator Variance (Technical Appendix)

### Setup

For independent ratios $r_t = A_{t+1}/A_t$ with logs $\ell_t = \log r_t$:

$$\mu_\ell = \mathbb{E}[\ell_t], \quad \sigma_\ell^2 = \text{Var}[\ell_t]$$

Geometric estimator: $\hat{m}_{\text{geo}} = \exp(\bar{\ell})$

### Variance Derivation

**Step 1:** Variance of sample mean:
$$\text{Var}[\bar{\ell}] = \frac{\sigma_\ell^2}{n}$$

**Step 2:** Delta method for $g(x) = e^x$:
$$\text{Var}[\hat{m}_{\text{geo}}] \approx (e^{\mu_\ell})^2 \cdot \text{Var}[\bar{\ell}] = \frac{\sigma_\ell^2}{n} e^{2\mu_\ell}$$

**Step 3:** Empirical estimator:
$$\boxed{\widehat{\text{Var}}[\hat{m}_{\text{geo}}] \approx \frac{\hat{\sigma}_\ell^2}{n} \hat{m}_{\text{geo}}^2}$$

Standard error:
$$\text{SE}(\hat{m}_{\text{geo}}) \approx \hat{m}_{\text{geo}} \sqrt{\frac{\hat{\sigma}_\ell^2}{n}}$$

---

## References

1. Beggs, J. M., & Plenz, D. (2003). Neuronal avalanches in neocortical circuits. *J. Neurosci.*
2. Shew, W. L., & Plenz, D. (2013). The functional benefits of criticality in the cortex. *Neuroscientist*
3. Wilting, J., & Priesemann, V. (2019). 25 years of criticality in neuroscience. *Front. Physiol.*
4. Muñoz, M. A. (2018). Colloquium: Criticality and dynamical scaling in living systems. *Rev. Mod. Phys.*
