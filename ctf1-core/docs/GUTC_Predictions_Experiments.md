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
| **ASD** | **Subcritical** | $> 1.5$ (steeper) | **Reduced, rigid** |
| ADHD | Mildly supercritical | $< 1.5$ (shallower) | Large, variable |
| Mania | Supercritical | $< 1.5$ | Large |
| Epileptic seizure | Highly supercritical | $\ll 1.5$ (activity blow-up) | Diverging |
| Healthy cognition | **Critical** | $\approx 1.5$ | Maximal |

### 2.2.1 ASD from the GUTC Perspective

**Hypothesis:** Autism Spectrum Disorder exhibits subcritical dynamics with aberrant precision weighting.

| Feature | GUTC Prediction | Mechanism |
|---------|-----------------|-----------|
| Sensory hypersensitivity | $\Pi_{\text{sensory}} \uparrow$ | Over-precision on bottom-up signals |
| Rigid behavior patterns | $E(\lambda) < 0$ | Subcritical → attractor trapping |
| Predictability preference | $\Pi_{\text{prior}} \uparrow$ | Hyper-precision on predictions |
| Reduced flexibility | $\xi$ diminished | Short correlation length |
| Detail focus | Local $\Pi \uparrow$ | High local precision, reduced global |

**Dual Precision Imbalance in ASD:**

$$\frac{\Pi_{\text{local}}}{\Pi_{\text{global}}} \gg 1 \quad \text{(ASD)} \quad \text{vs.} \quad \approx 1 \quad \text{(Neurotypical)}$$

**Testable Prediction:** EEG/MEG recordings in ASD should show:
- Steeper avalanche exponents ($\alpha > 1.7$)
- Reduced long-range temporal correlations
- Branching ratio $\sigma < 1$

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

## Prediction 6: Dynamical Analysis Methods for Clinical Diagnosis

### 6.1 The Core Approach

**Hypothesis:** Dynamical analysis (branching parameter, Hurst exponent, avalanche statistics) provides superior diagnostic specificity compared to correlation-based inference alone.

**Why dynamical > correlational:**

| Approach | Measures | Limitation |
|----------|----------|------------|
| Correlation-based | Statistical associations | Confounded by covariates |
| **Dynamical analysis** | Causal propagation, phase state | Directly measures $E(\lambda)$ |

### 6.2 Branching Parameter Estimation

The branching parameter $\sigma$ directly estimates distance to criticality:

$$\sigma = \left\langle \frac{A_{t+1}}{A_t} \right\rangle, \quad E(\lambda) = \sigma - 1$$

**Protocol:**

```python
def estimate_branching_parameter(neural_time_series, bin_size_ms=4):
    """
    Estimate branching parameter from neural activity.

    Args:
        neural_time_series: Binned spike counts or LFP power
        bin_size_ms: Temporal resolution

    Returns:
        sigma: Branching ratio (critical = 1.0)
        sigma_geo: Geometric estimator (robust to outliers)
    """
    # Detect avalanches
    avalanches = detect_avalanches(neural_time_series, threshold=0)

    # Conventional estimator
    ratios = []
    for avalanche in avalanches:
        for t in range(len(avalanche) - 1):
            if avalanche[t] > 0:
                ratios.append(avalanche[t+1] / avalanche[t])
    sigma_conv = np.mean(ratios)

    # Geometric estimator (robust)
    log_ratios = np.log([r for r in ratios if r > 0])
    sigma_geo = np.exp(np.mean(log_ratios))

    return sigma_conv, sigma_geo

# Interpretation:
# σ < 0.95 → Subcritical (depression, ASD)
# 0.95 < σ < 1.05 → Critical (healthy)
# σ > 1.05 → Supercritical (mania, seizure risk)
```

### 6.3 Hurst Exponent Analysis

The Hurst exponent $H$ measures long-range temporal correlations:

$$\langle |X(t+\tau) - X(t)|^2 \rangle \sim \tau^{2H}$$

| $H$ Value | Interpretation | GUTC Phase |
|-----------|----------------|------------|
| $H < 0.5$ | Anti-persistent (mean-reverting) | Subcritical |
| $H = 0.5$ | Random walk (no memory) | Random |
| $H > 0.5$ | Persistent (long-range correlations) | Near-critical |
| $H \approx 0.7\text{–}0.8$ | **Optimal** | Critical |

**Protocol:**

```python
def estimate_hurst_exponent(time_series, max_lag=100):
    """
    Estimate Hurst exponent via rescaled range (R/S) analysis.
    """
    lags = np.logspace(1, np.log10(max_lag), 20).astype(int)
    rs_values = []

    for lag in lags:
        # Compute R/S statistic
        segments = np.array_split(time_series, len(time_series) // lag)
        rs = []
        for seg in segments:
            if len(seg) < 2:
                continue
            cumdev = np.cumsum(seg - np.mean(seg))
            R = np.max(cumdev) - np.min(cumdev)
            S = np.std(seg)
            if S > 0:
                rs.append(R / S)
        rs_values.append(np.mean(rs))

    # Fit power law: log(R/S) vs log(lag)
    H, _ = np.polyfit(np.log(lags), np.log(rs_values), 1)

    return H

# Clinical interpretation:
# H < 0.5 in ASD → Reduced temporal integration
# H < 0.5 in depression → Diminished persistence
# H ≈ 0.7-0.8 → Healthy criticality
```

### 6.4 Combined Diagnostic Protocol

```python
def gutc_phase_diagnostic(neural_recording):
    """
    Comprehensive GUTC-based phase diagnostic.

    Returns:
        phase: 'subcritical', 'critical', or 'supercritical'
        confidence: Diagnostic confidence score
        metrics: Dict of all measured values
    """
    # 1. Branching parameter
    sigma_conv, sigma_geo = estimate_branching_parameter(neural_recording)

    # 2. Avalanche exponent
    avalanches = detect_avalanches(neural_recording)
    alpha = fit_power_law_exponent([len(a) for a in avalanches])

    # 3. Hurst exponent
    H = estimate_hurst_exponent(neural_recording)

    # 4. Correlation length (from spatial data if available)
    xi = estimate_correlation_length(neural_recording)

    # Combined decision
    metrics = {
        'sigma': sigma_geo,
        'alpha': alpha,
        'H': H,
        'xi': xi
    }

    # Scoring
    subcritical_score = (sigma_geo < 0.95) + (alpha > 1.6) + (H < 0.5)
    supercritical_score = (sigma_geo > 1.05) + (alpha < 1.4) + (H > 0.9)

    if subcritical_score >= 2:
        phase = 'subcritical'
        confidence = subcritical_score / 3
    elif supercritical_score >= 2:
        phase = 'supercritical'
        confidence = supercritical_score / 3
    else:
        phase = 'critical'
        confidence = 1 - max(subcritical_score, supercritical_score) / 3

    return phase, confidence, metrics
```

### 6.5 Disorder-Specific Signatures

| Disorder | $\sigma$ | $\alpha$ | $H$ | Pattern |
|----------|----------|----------|-----|---------|
| Depression | $< 0.9$ | $> 1.7$ | $< 0.5$ | All subcritical |
| ASD | $< 0.95$ | $> 1.6$ | $< 0.55$ | Subcritical + rigid |
| ADHD | $> 1.05$ | $< 1.4$ | Variable | Supercritical |
| Schizophrenia | Variable | Variable | Variable | Aberrant precision |
| Healthy | $\approx 1.0$ | $\approx 1.5$ | $\approx 0.7$ | Critical |

### 6.6 Refined Estimation Methods

**Least-Squares Branching Ratio Estimator:**

```python
import numpy as np

def estimate_branching_ratio_ls(A):
    """
    Least-squares estimator for branching ratio from event counts.

    A: 1D array of nonnegative event counts A_t.
    Returns: lambda_hat, E_hat = lambda_hat - 1

    More stable than ratio-based estimators for sparse data.
    """
    A = np.asarray(A, dtype=float)
    # Only use bins where A_t > 0
    mask = A[:-1] > 0
    At  = A[:-1][mask]
    At1 = A[1:][mask]
    # Least-squares: E[A_{t+1} | A_t] ≈ λ A_t
    num = np.sum(At * At1)
    den = np.sum(At * At)
    lambda_hat = num / den if den > 0 else np.nan
    E_hat = lambda_hat - 1.0
    return lambda_hat, E_hat
```

**Avalanche Extraction and Power-Law Fitting:**

```python
def extract_avalanches(A):
    """
    Extract avalanche sizes and durations from event count series.
    Avalanche = consecutive bins with A_t > 0 bounded by A_t == 0.
    """
    A = np.asarray(A, dtype=float)
    sizes, durations = [], []
    in_avalanche = False
    current_size, current_dur = 0.0, 0

    for a in A:
        if a > 0:
            in_avalanche = True
            current_size += a
            current_dur += 1
        elif in_avalanche:
            sizes.append(current_size)
            durations.append(current_dur)
            in_avalanche = False
            current_size, current_dur = 0.0, 0

    if in_avalanche:  # Catch trailing avalanche
        sizes.append(current_size)
        durations.append(current_dur)

    return np.array(sizes), np.array(durations)

def fit_power_law_tail(x, xmin=None):
    """
    Rough log-log slope for quick phase assessment.
    For publication-grade inference, use proper MLE methods.
    """
    x = np.asarray(x, dtype=float)
    if xmin is None:
        xmin = np.percentile(x, 50)
    x_tail = x[x >= xmin]
    if len(x_tail) < 10:
        return np.nan

    hist, edges = np.histogram(x_tail, bins='auto', density=True)
    centers = 0.5 * (edges[1:] + edges[:-1])
    mask = hist > 0
    slope = np.polyfit(np.log10(centers[mask]), np.log10(hist[mask]), 1)[0]
    return -slope  # alpha_hat
```

**Rescaled-Range Hurst Estimator:**

```python
def hurst_rs(x, chunk_sizes=[8, 16, 32, 64]):
    """
    Rescaled-range estimator for Hurst exponent H.

    Returns H where:
    - H ≈ 0.5: Short memory (Markov-ish)
    - H > 0.5: Long-range dependence (critical-like)
    - H < 0.5: Anti-persistent (over-correcting)
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    if N < 32:
        return np.nan

    # Select appropriate chunk size
    m = max([c for c in chunk_sizes if c * 4 <= N], default=8)
    Ks = N // m
    RS_vals = []

    for k in range(Ks):
        seg = x[k*m:(k+1)*m]
        seg = seg - seg.mean()
        Y = np.cumsum(seg)
        R = Y.max() - Y.min()
        S = seg.std()
        if S > 0:
            RS_vals.append(R / S)

    if len(RS_vals) == 0:
        return np.nan

    H = np.log2(np.mean(RS_vals)) / np.log2(m)
    return H
```

---

## Prediction 7: Clinical Severity Correlations (ASD Research Roadmap)

### 7.1 The Core Hypothesis

**Claim:** ASD clinical severity scores (ADOS, SRS) are macro-level behavioral manifestations of underlying dynamical mis-tuning, quantifiable via $\Delta\lambda$ and $\Delta\Pi$.

### 7.2 Hypothesis 1: Phase Mis-Tuning → Rigidity/Repetitive Behaviors

**Claim:** Restricted/repetitive behaviors (ADOS R.R.B. domain) correlate with subcritical phase shift.

| Parameter | Predicted Correlation with R.R.B. Severity | Mechanistic Rationale |
|-----------|-------------------------------------------|----------------------|
| $\alpha$ (avalanche exponent) | **Positive:** Higher severity → steeper $\alpha$ | Subcritical dynamics reduce state space, forcing repetitive cycles |
| $\lambda$ (branching ratio) | **Negative:** Higher severity → lower $\lambda$ | System is over-damped, suppressing exploratory dynamics |
| $H$ (Hurst exponent) | **Negative:** Higher severity → lower $H$ | Reduced long-range temporal integration |

### 7.3 Hypothesis 2: Precision Mis-Tuning → Social/Communication Deficits

**Claim:** Social responsiveness deficits (SRS/ADOS S.C.I. domain) correlate with aberrant precision weighting.

| Parameter | Predicted Correlation with S.C.I. Severity | Mechanistic Rationale |
|-----------|-------------------------------------------|----------------------|
| $\Pi_{\text{prior}}$ (prior precision) | **Positive:** Higher → rigid internal models resist social updating |
| $\Pi_{\text{sensory}}$ (sensory precision) | **Positive:** Higher → hyper-attention to detail, sensory overload |
| $\Pi_{\text{local}}/\Pi_{\text{global}}$ ratio | **Positive:** Higher → weak central coherence |

### 7.4 Combined Dynamical Predictor

**Ultimate Hypothesis:** The combinatorial deviation $(\Delta\lambda, \Delta\Pi)$ provides maximal predictive power for ASD severity.

$$\text{Severity} \sim f(|E(\lambda)|, |\Pi - \Pi_{\text{optimal}}|) = f(|\lambda - 1|, \Delta\Pi)$$

**Research Protocol:**

```python
def asd_dynamical_assessment(neural_recording, clinical_scores):
    """
    Correlate dynamical parameters with clinical severity.

    Args:
        neural_recording: EEG/MEG time series
        clinical_scores: Dict with 'ADOS_RRB', 'ADOS_SCI', 'SRS' scores

    Returns:
        correlations: Parameter-severity correlations
        combined_model: Regression model for severity prediction
    """
    # Extract dynamical parameters
    lambda_hat, E_hat = estimate_branching_ratio_ls(neural_recording)
    sizes, durations = extract_avalanches(neural_recording)
    alpha = fit_power_law_tail(sizes)
    H = hurst_rs(neural_recording)

    params = {
        'lambda': lambda_hat,
        'E': E_hat,
        'alpha': alpha,
        'H': H
    }

    # Compute correlations
    correlations = {}
    for param_name, param_val in params.items():
        for score_name, score_val in clinical_scores.items():
            r, p = scipy.stats.pearsonr([param_val], [score_val])
            correlations[f'{param_name}_vs_{score_name}'] = (r, p)

    # Combined model: Severity ~ |E| + |H - 0.7|
    X = np.array([[abs(E_hat), abs(H - 0.7)]])
    # ... fit regression model

    return params, correlations
```

### 7.5 Expected Findings

| Domain | Primary Dynamical Marker | Secondary Marker |
|--------|-------------------------|------------------|
| R.R.B. (rigidity) | $\lambda < 1$ (subcritical) | $\alpha > 1.5$ |
| S.C.I. (social) | $\Pi_{\text{prior}} \uparrow$ | $\Pi_{\text{local}}/\Pi_{\text{global}} \gg 1$ |
| Combined | $(\lambda, \Pi)$ deviation from $(1, \Pi_{\text{opt}})$ | — |

**Key insight:** This framework diagnoses *phase state*, not identity. The same analysis applies to any system (brain, AI agent, behavioral sequence).

---

## Summary: Falsifiable Predictions

| # | Prediction | Key Test | Failure Criterion |
|---|------------|----------|-------------------|
| 1 | $C$, $J$, $E=0$ coincide | $\lambda$ sweep | Peaks at different $\lambda$ |
| 2 | Disorders = phase shifts | Measure $\alpha$ in patients | No $\alpha$ difference |
| 3 | $\tau \propto -\log \epsilon$ | Noise sweep | Non-logarithmic scaling |
| 4 | $I \sim |E|^{-\gamma}$ | Fisher estimation | $\gamma$ not in [1, 3] range |
| 5 | Level-dependent dwell | HHN measurement | No level hierarchy in $\tau$ |
| 6 | Dynamical > correlational | Phase diagnostic | No $\sigma$, $H$ disorder difference |
| 7 | ASD severity ~ $(\Delta\lambda, \Delta\Pi)$ | Clinical correlation | No parameter-severity correlation |

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

## Operational Protocol: Reproducible Pipeline Specification

### Phase Parameter Estimation ($\hat{\lambda}, \hat{\alpha}$) from Resting-State Data

Global phase parameters are estimated from electrophysiological time-series (EEG/MEG/ECoG). These metrics quantify the system's distance from the critical set-point, $E(\lambda) = 0$.

#### Preprocessing Pipeline (Activity Binning)

1. **Filtering:** Apply bandpass filtering (e.g., 1–40 Hz) to remove noise and DC offset
2. **Activity Definition ($A_t$):** For each time bin $\Delta t$ (e.g., 4 ms), define $A_t$ as the number of channels exceeding threshold ($3 \times \text{SD}$)
3. **Avalanche Identification:** Avalanche begins when $A_t > 0$, ends when $A_t = 0$
4. **Metric Calculation:** Compute $P(s)$ distribution and bin-wise ratios

#### Parameter Estimation Summary

| Parameter | Estimation Method | Criticality Proxy |
|-----------|-------------------|-------------------|
| $\hat{\lambda}_{\text{phase}}$ | Geometric mean: $\hat{m}_{\text{geo}} = \exp(\langle \log(A_{t+1}/A_t) \rangle)$ | Distance-to-criticality ($m = 1$) |
| $\hat{\alpha}_{\text{phase}}$ | Power-law fit: $P(s) \sim s^{-\alpha}$ | Steepness ($\alpha > 1.5$ → subcritical) |

---

## Generative Model Specification: Precision Estimation

Precision parameters require fitting a hierarchical generative model to behavioral data from prediction-error tasks.

### Hierarchical Gaussian Filter (HGF/AIF) Structure

```
Level 2 (Prior):     Slow-changing environmental volatility
                           ↓
Level 1 (State):     Fast-changing hidden state (e.g., P(stimulus A))
                           ↓
Observation (u):     Sensory input (stimulus identity)
```

### Precision Parameter Estimation

| Parameter | Task Design | Model Parameter | Falsifiable Prediction |
|-----------|-------------|-----------------|------------------------|
| $\hat{\Pi}_{\text{prior}}$ | Probabilistic Reversal Task | $\Pi_{\text{prior}} \propto 1/\sigma^2_{x_1}$ | Higher → higher S.C.I. severity |
| $\hat{\Pi}_{\text{sensory}}$ | Noisy Detection Task | $\Pi_{\text{sensory}} \propto 1/\sigma^2_{u}$ | Higher → higher S.C.I. severity |

---

## Language-Agnostic Pseudocode: CBH/GUTC Metrics

### Avalanche Identification

```pseudo
INPUT: A[0..T-1]   // nonnegative integers, event counts per bin

avalanches_sizes  = []
avalanches_durs   = []

in_avalanche = FALSE
current_size = 0
current_dur  = 0

FOR t = 0 TO T-1:
    IF A[t] > 0:
        in_avalanche = TRUE
        current_size = current_size + A[t]
        current_dur  = current_dur  + 1
    ELSE:
        IF in_avalanche == TRUE:
            APPEND current_size TO avalanches_sizes
            APPEND current_dur  TO avalanches_durs
            in_avalanche = FALSE
            current_size = 0
            current_dur  = 0
        ENDIF
    ENDIF
ENDFOR

// Close trailing avalanche if needed
IF in_avalanche == TRUE:
    APPEND current_size TO avalanches_sizes
    APPEND current_dur  TO avalanches_durs
ENDIF

OUTPUT: avalanches_sizes[], avalanches_durs[]
```

### Branching Ratio (Least-Squares)

```pseudo
INPUT: A[0..T-1]

sum_At_At1 = 0
sum_At2    = 0

FOR t = 0 TO T-2:
    IF A[t] > 0:
        sum_At_At1 = sum_At_At1 + A[t] * A[t+1]
        sum_At2    = sum_At2    + A[t] * A[t]
    ENDIF
ENDFOR

IF sum_At2 > 0:
    lambda_hat = sum_At_At1 / sum_At2
ELSE:
    lambda_hat = NaN
ENDIF

E_hat = lambda_hat - 1.0

OUTPUT: lambda_hat, E_hat
```

### Branching Ratio (Geometric Estimator)

```pseudo
INPUT: A[0..T-1]

log_ratios = []

FOR t = 0 TO T-2:
    IF A[t] > 0 AND A[t+1] > 0:
        r = A[t+1] / A[t]
        APPEND log(r) TO log_ratios
    ENDIF
ENDFOR

IF length(log_ratios) > 0:
    mean_log_r = AVERAGE(log_ratios)
    lambda_hat = exp(mean_log_r)    // geometric mean
    E_hat      = lambda_hat - 1.0
ELSE:
    lambda_hat = NaN
    E_hat      = NaN
ENDIF

OUTPUT: lambda_hat, E_hat
```

### Avalanche Size Exponent (Log-Log Slope)

```pseudo
INPUT: avalanches_sizes[]

// Choose minimum size for tail fit
xmin = MEDIAN(avalanches_sizes)

tail_sizes = []
FOR each s IN avalanches_sizes:
    IF s >= xmin:
        APPEND s TO tail_sizes
    ENDIF
ENDFOR

IF length(tail_sizes) < MIN_TAIL_COUNT:    // e.g., 20
    alpha_hat = NaN
    STOP
ENDIF

// Bin logarithmically
num_bins = 10
hist, bin_edges = HISTOGRAM(tail_sizes, num_bins, density=TRUE)

bin_centers = []
bin_probs   = []
FOR i = 0 TO length(hist)-1:
    IF hist[i] > 0:
        center = 0.5 * (bin_edges[i] + bin_edges[i+1])
        APPEND center TO bin_centers
        APPEND hist[i] TO bin_probs
    ENDIF
ENDFOR

// Linear fit in log-log space
log_s = [log10(c) FOR c IN bin_centers]
log_p = [log10(p) FOR p IN bin_probs]

slope, intercept = LINEAR_REGRESSION(log_s, log_p)
alpha_hat = -slope    // P(s) ~ s^{-alpha}

OUTPUT: alpha_hat
```

**Note:** For publication-grade analysis, use proper maximum-likelihood fitting with goodness-of-fit tests.

---

## References

1. Beggs, J. M., & Plenz, D. (2003). Neuronal avalanches in neocortical circuits. *J. Neurosci.*
2. Shew, W. L., & Plenz, D. (2013). The functional benefits of criticality in the cortex. *Neuroscientist*
3. Wilting, J., & Priesemann, V. (2019). 25 years of criticality in neuroscience. *Front. Physiol.*
4. Muñoz, M. A. (2018). Colloquium: Criticality and dynamical scaling in living systems. *Rev. Mod. Phys.*
