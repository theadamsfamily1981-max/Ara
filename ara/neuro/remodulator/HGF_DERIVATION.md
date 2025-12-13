# Hierarchical Gaussian Filter: Mathematical Derivation

> This document provides the mathematical foundations for the HGF implementation
> in `hgf.py`, deriving update equations from variational Bayesian inference.

---

## 1. Introduction

The Hierarchical Gaussian Filter (HGF) is a Bayesian model for learning in volatile
environments. It represents beliefs in a hierarchy:
- Lower levels track immediate contingencies (probabilities)
- Higher levels model volatilities (rates of change)

The update equations are derived via **variational Bayesian inference**, minimizing
**variational free energy**—a computable upper bound on surprise.

---

## 2. Generative Model

For a 5-level binary HGF generating observations u^(k) ∈ {0, 1}:

### Level 1 (Binary State)
Input likelihood:
```
p(u^(k) | x_1^(k)) = u^(k) × x_1^(k) + (1 - u^(k)) × (1 - x_1^(k))
```

### Link to Level 2
Prior on x_1 via sigmoid:
```
p(x_1^(k) = 1 | x_2^(k)) = σ(x_2^(k)) = 1 / (1 + exp(-x_2^(k)))
```

### Level 2 (Logit-Probability)
Gaussian random walk with variance scaled by Level 3:
```
p(x_2^(k) | x_2^(k-1), x_3^(k)) = N(x_2^(k); x_2^(k-1), exp(κ₁ × x_3^(k) + ω₁))
```

### Level 3 (Volatility)
```
p(x_3^(k) | x_3^(k-1), x_4^(k)) = N(x_3^(k); x_3^(k-1), exp(κ₂ × x_4^(k) + ω₂))
```

### Level 4 (Higher Volatility)
```
p(x_4^(k) | x_4^(k-1), x_5^(k)) = N(x_4^(k); x_4^(k-1), exp(κ₃ × x_5^(k) + ω₃))
```

### Level 5 (Highest Volatility)
```
p(x_5^(k) | x_5^(k-1)) = N(x_5^(k); x_5^(k-1), exp(θ))
```

### Full Joint
```
p(u^(1:K), x^(1:K)) = ∏_k p(u^(k)|x_1^(k)) × p(x_1^(k)|x_2^(k)) ×
                        p(x_2^(k)|x_2^(k-1),x_3^(k)) × p(x_3^(k)|x_3^(k-1),x_4^(k)) ×
                        p(x_4^(k)|x_4^(k-1),x_5^(k)) × p(x_5^(k)|x_5^(k-1))
```

---

## 3. Variational Bayesian Objective

Exact posterior p(x^(k) | u^(1:k)) is intractable. Use variational approximation q(x^(k)).

### Free Energy
```
F = ∫ q(x^(k)) ln [q(x^(k)) / p(u^(k), x^(k) | u^(1:k-1))] dx^(k)
  = D_KL[q(x^(k)) || p(x^(k) | u^(1:k))] - ln p(u^(k) | u^(1:k-1))
```

Minimizing F:
1. Approximates the true posterior (minimizes KL divergence)
2. Maximizes model evidence (minimizes surprise)

### Mean-Field Factorization
```
q(x^(k)) = ∏_i q(x_i^(k))
```
Each q(x_i^(k)) is Gaussian: N(μ_i^(k), 1/π_i^(k))

---

## 4. Quadratic Approximation

For each level i, minimize variational energy:
```
I(x_i^(k)) = -E_q(x_{≠i}) [ln p(u^(k), x^(k) | u^(1:k-1))] + ln q(x_i^(k))
```

Taylor expand around prior mean μ̂_i^(k) = μ_i^(k-1):
```
I(x_i^(k)) ≈ I(μ̂_i) + (∂I/∂x_i)|_{μ̂_i} × (x_i - μ̂_i) + ½ (∂²I/∂x_i²)|_{μ̂_i} × (x_i - μ̂_i)²
```

This yields Gaussian posterior with:
```
π_i^(k) = (∂²I/∂x_i²)|_{μ̂_i}           [Precision = Hessian]
μ_i^(k) = μ̂_i - (1/π_i) × (∂I/∂x_i)|_{μ̂_i}  [Mean = gradient update]
```

---

## 5. Level-Specific Derivations

### Level 1: Prediction
```
ŝ^(k) = σ(μ̂_2^(k)) = σ(μ_2^(k-1))
```

Prediction error:
```
δ_1^(k) = u^(k) - ŝ^(k)
```

### Level 2: Logit-Beliefs

**Variational Energy:**
```
I(x_2^(k)) = ½ (x_2 - μ_2^(k-1))² exp(-κ₁μ_3^(k-1) - ω₁)
           - u^(k) ln σ(x_2) - (1-u^(k)) ln(1-σ(x_2))
```

**Hessian (Precision):**
```
∂²I/∂x_2² = exp(-κ₁μ_3^(k-1) - ω₁) + ŝ^(k)(1 - ŝ^(k))
          = π̂_2^(k) + ŝ(1-ŝ)
```

Where π̂_2^(k) = 1/exp(κ₁μ_3^(k-1) + ω₁) is predicted precision.

**Posterior Precision:**
```
π_2^(k) = π̂_2^(k) + ŝ^(k)(1 - ŝ^(k))
```

**Gradient:**
```
∂I/∂x_2 = (μ̂_2 - μ_2^(k-1)) × π̂_2 + ŝ - u
```

**Posterior Mean:**
```
μ_2^(k) = μ̂_2^(k) + [ŝ(1-ŝ) / π_2^(k)] × (u^(k) - ŝ^(k))
        = μ̂_2^(k) + [ŝ(1-ŝ) / π_2^(k)] × δ_1^(k)
```

The sigmoid derivative ŝ(1-ŝ) acts as **precision weighting** on the prediction error.

### Level 3: Volatility

**Variational Energy:**
```
I(x_3^(k)) = ½ (x_3 - μ_3^(k-1))² exp(-θ)
           + ½ (μ_2^(k) - μ_2^(k-1))² exp(-κ₁x_3 - ω₁)
           + ½ ln(exp(κ₁x_3 + ω₁))
```

**Predicted Precision:**
```
π̂_3^(k) = 1/exp(κ₂μ_4^(k-1) + ω₂)
```

**Prediction Error at Level 2:**
```
δ_2^(k) = μ_2^(k) - μ̂_2^(k)
```

**Posterior Precision:**
```
π_3^(k) = π̂_3^(k) + ½κ₁² π̂_2^(k) (1 + π_2^(k-1) × δ_2²)
```

**Posterior Mean:**
```
μ_3^(k) = μ̂_3^(k) + [π̂_2^(k) / π_3^(k)] × δ_2^(k)
```

### Levels 4-5: Higher Volatilities

Same structure, with errors propagating upward:
```
δ_i^(k) = μ_{i-1}^(k) - μ̂_{i-1}^(k)
π_i^(k) = π̂_i^(k) + π_{i-1}^(k)
μ_i^(k) = μ̂_i^(k) + [π_{i-1}^(k) / π_i^(k)] × δ_{i-1}^(k)
```

---

## 6. Parameter Interpretation

### κ Parameters (Coupling)

σ_i² ∝ exp(κ_i × μ_{i+1} + ω_i)

| Parameter | Effect | Clinical Interpretation |
|-----------|--------|-------------------------|
| Low κ₁ | Rigid learning rate | Beliefs don't adapt to volatility (schizophrenia?) |
| High κ₁ | Flexible learning | Strong sensory coupling (ASD?) |
| Low κ₂ | Stable meta-learning | Rigid higher-level priors |
| High κ₂ | Volatile meta-learning | Rapidly changing learning strategy |

### ω Parameters (Base Volatility)

| Parameter | Effect | Clinical Interpretation |
|-----------|--------|-------------------------|
| High ω₁ | Fast baseline learning | Chronically unstable beliefs |
| Low ω₁ | Slow baseline learning | Rigid, slow-updating beliefs |
| High θ | Fundamental uncertainty | Deep epistemological doubt |
| Low θ | Stable worldview | Strong metaphysical confidence |

---

## 7. Uncertainty Dynamics

### Different Timescales

| Level | What it tracks | Update speed | σ dynamics |
|-------|---------------|--------------|------------|
| 2 | Probability | Fast (per trial) | Drops quickly in stable environments |
| 3 | Volatility | Medium | Shifts at regime changes |
| 4 | Meta-volatility | Slow | Gradual decrease over many trials |
| 5 | Epistemic depth | Very slow | Requires long-term evidence |

### Key Insight
**Uncertainty resolves at different hierarchical speeds.**

- σ₂ responds to immediate sensory evidence
- σ₅ requires hundreds of trials to stabilize
- This mirrors the "Beautiful Loop" concept of epistemic depth

---

## 8. Connection to Free Energy Principle

The HGF update equations are **gradient descent on variational free energy**:

```
μ_i^(k) = μ̂_i^(k) - (1/π_i^(k)) × ∂F/∂μ_i
```

This is the same update form as:
- Predictive coding neural networks
- Active inference agents
- The Brain Remodulator's precision control

The HGF provides a **tractable, behaviorally-validated** model for testing
precision weighting hypotheses.

---

## 9. Clinical Profiles as Parameter Settings

| Condition | κ₁ | ω₁ | Interpretation |
|-----------|----|----|----------------|
| Healthy | 1.0 | 2.0 | Balanced |
| Schizophrenia | 0.3 | 3.0 | Weak sensory coupling, unstable priors |
| ASD | 2.0 | 1.0 | Strong sensory, rigid priors |
| Anxiety | 1.5 | 3.0 | Hypervigilant, everything volatile |
| Depression | 0.5 | 1.5 | Slow updating, rigid negative priors |

These map to hypotheses about D_low and D_high in the Brain Remodulator.

---

## 10. Implementation Notes

The `hgf.py` implementation uses the simplified update form:

```python
# Posterior = Prior + (Error_Precision / Total_Precision) × Error
pi_post = pi_hat + pi_error
mu_post = mu_hat + (pi_error / pi_post) * delta
```

This is exact for Gaussian-Gaussian conjugacy and approximates the
full variational solution for the sigmoid link at Level 1.

---

## References

1. Mathys, C., et al. (2011). A Bayesian foundation for individual learning
   under uncertainty. *Frontiers in Human Neuroscience*.

2. Mathys, C., et al. (2014). Uncertainty in perception and the Hierarchical
   Gaussian Filter. *Frontiers in Human Neuroscience*.

3. Friston, K. (2005). A theory of cortical responses.
   *Philosophical Transactions of the Royal Society B*.

4. Adams, R.A., et al. (2013). The computational anatomy of psychosis.
   *Frontiers in Psychiatry*.

---

*Companion document to hgf.py*
*Part of the Brain Remodulator framework*
