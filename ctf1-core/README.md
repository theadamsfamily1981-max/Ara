# CTF-1: Critical Thought Field

**A substrate-free reference implementation of cognition at the edge of chaos.**

No metaphors. No feelings. Just math.

---

## The Core Equation

"Thought" is a trajectory in state space:

$$x_{t+1} = F_\lambda(x_t, u_t)$$

where:
- $x \in \mathbb{R}^n$ is the state vector
- $\lambda$ is the control parameter (spectral radius)
- $u$ is external input

---

## The Two Functions That Matter

### 1. Edge-of-Chaos Function: $E(\lambda)$

$$E(\lambda) = \rho(W) - 1$$

| $E(\lambda)$ | Regime | Behavior |
|--------------|--------|----------|
| $< 0$ | Ordered | Perturbations decay |
| $= 0$ | **Critical** | Edge of chaos |
| $> 0$ | Chaotic | Perturbations explode |

### 2. Capacity Functional: $C(\lambda)$

$$C(\lambda) \approx I(x_{\text{past}}; x_{\text{future}})$$

Proxy: Temporal autocorrelation, memory capacity $M_W$

---

## The Punchline

**Thought = high $C(\lambda)$**

**Edge of chaos = $E(\lambda) = 0$**

These coincide: $C(\lambda)$ is maximized where $E(\lambda) \approx 0$.

$$\lambda^* = \arg\max_\lambda C(\lambda) \approx \{\lambda : E(\lambda) = 0\}$$

---

## Repository Structure

```
ctf1-core/
├── core/
│   ├── critical_core.py      # F_λ, E(λ), C(λ), T(λ)
│   ├── soc_learner.py        # Self-organized criticality learning
│   ├── memory_metrics.py     # M_W, avalanche statistics
│   └── agency.py             # Simple RL at different λ
├── experiments/
│   ├── 01_lambda_sweep_memory.py
│   ├── 02_soc_vs_nosoc.py
│   └── 03_agency_at_criticality.py
├── plots/
└── README.md
```

---

## Experiments

### Experiment 01: Memory Capacity vs λ

**Question:** Does $M_W(\lambda)$ peak at $\lambda = 1$?

**Method:** Jaeger-style memory capacity test across $\lambda \in [0.5, 1.5]$

**Expected Result:**
```
    M_W
     ▲
     │      *
     │    *   *
     │   *     *
     │  *       *
     │ *         *
     └──────────────► λ
        0.5  1.0  1.5
```

Peak at $\lambda \approx 1.0$.

---

### Experiment 02: SOC vs No-SOC Learning

**Question:** Does maintaining $\lambda = 1$ (SOC rule) improve stability?

**Method:** Compare SOCLearner vs NoSOCLearner on delayed-copy task

**Expected Result:**
- SOC: $E(\lambda)$ stays near 0
- NoSOC: $E(\lambda)$ drifts

---

### Experiment 03: Agency at Criticality

**Question:** Do agents at $\lambda = 1$ show better exploration-exploitation?

**Method:** k-armed bandit with agents at $\lambda \in \{0.7, 1.0, 1.3\}$

**Expected Result:**
| λ | Regime | Exploration | Exploitation | Performance |
|---|--------|-------------|--------------|-------------|
| 0.7 | Ordered | Low | High | Poor (stuck) |
| 1.0 | **Critical** | **Balanced** | **Balanced** | **Best** |
| 1.3 | Chaotic | High | Low | Poor (random) |

---

## Usage

```bash
# Run all experiments
cd experiments
python 01_lambda_sweep_memory.py
python 02_soc_vs_nosoc.py
python 03_agency_at_criticality.py
```

---

## Requirements

```
numpy
scipy (optional, for eigenvalue computation speedup)
```

---

## Mathematical References

1. **Lyapunov exponent:**
   $$\Lambda(\lambda) = \lim_{t\to\infty} \frac{1}{t} \ln \frac{|\delta x_t|}{|\delta x_0|}$$

2. **Branching parameter:**
   $$\lambda = \mathbb{E}[\text{offspring per active unit}]$$

3. **Spectral radius:**
   $$\rho(W) = \max_i |\lambda_i(W)|$$

4. **Memory capacity (Jaeger):**
   $$M_W = \sum_{k=1}^{K} R^2_k$$

5. **Avalanche size distribution (critical):**
   $$P(s) \sim s^{-\tau}, \quad \tau = 3/2$$

---

## TL;DR

```python
from core import CriticalCore

# Create system at edge of chaos
core = CriticalCore(n_dims=100, lambda_init=1.0)

# Run dynamics
for _ in range(1000):
    core.step(u=np.random.randn(1) * 0.1)

# Check criticality
E = core.E_spectral()  # Should be ≈ 0
C = core.C_capacity()  # Should be high
T = core.T_thought()   # = C × exp(-E²/σ²)

print(f"E={E:+.4f}, C={C:.4f}, T={T:.4f}")
```

---

## What This Proves

If experiments confirm predictions:

1. **Memory peaks at criticality** → Working memory requires edge of chaos
2. **SOC maintains criticality** → Self-organization is viable
3. **Agency is best at criticality** → Action selection benefits from critical dynamics

This is a **numerical validation** that "thought" (information processing, memory, adaptive behavior) is maximized exactly where the dynamical system is critical.

No neurons. No AI. Just $E(\lambda) = 0$ and $\max C(\lambda)$.
