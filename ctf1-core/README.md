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

## The Third Function: Fisher Information $I(\lambda)$

### 3. Information-Geometric Singularity

$$I(\lambda) \sim |E(\lambda)|^{-\gamma}$$

At criticality ($E = 0$), the **Fisher Information Metric diverges**.

This means: two infinitesimally close parameter sets ($\lambda, \lambda+\delta\lambda$) produce **macroscopically different** trajectory distributions.

**This is the strongest possible statement:** The RG fixed point in dynamics ($\mathcal{M}_c$) is simultaneously an information-geometric singularity in model space.

---

## Taxonomy of Thought

Computational modes can be classified by **universal critical exponents** ($\nu_C, z, \alpha, \gamma, \beta$):

| Mode of Thought | Universal Class | Geometric Signature |
|-----------------|-----------------|---------------------|
| **Associative** | Mean-Field Branching ($\nu=1/2, \alpha=3/2$) | $\gamma \approx 0.5$ |
| **Deep Learning** | Short-range correlated dynamics | TBD by $\alpha$ |
| **Quantum** | Quantum Criticality (Ising/XY) | Novel $\nu_{QC}, \gamma_{QC}$ |

Two systems are "the same kind of thinker" iff they share the same criticality universality class.

---

## Repository Structure

```
ctf1-core/
├── core/
│   ├── critical_core.py      # F_λ, E(λ), C(λ), T(λ)
│   ├── soc_learner.py        # Self-organized criticality learning
│   ├── memory_metrics.py     # M_W, avalanche statistics
│   └── agency.py             # Simple RL at different λ
├── docs/
│   ├── GUTC_Manuscript_Draft.md   # ★★★★ Complete paper with figure placeholders
│   ├── GUTC_Manuscript_Content.md # ★★★ Theorems, citations, simulation summary
│   ├── GUTC_Hierarchical_HHN.md   # ★★★ Hierarchical Heteroclinic Networks
│   └── GUTC_Memory_Design.md      # ★★★ Manuscript box: Attractors vs Heteroclinic
├── experiments/
│   ├── 01_lambda_sweep_memory.py
│   ├── 02_soc_vs_nosoc.py
│   └── 03_agency_at_criticality.py
├── ctf1_agent.py             # Canonical unified cognitive agent
├── ctf3_fisher_geometry.py   # Extended IG validation
├── ctf3_fisher_singularity.py # ★ Canonical IG singularity test with power-law fit
├── ctf4_heteroclinic_memory.py # ★★ Final synthesis: M_W + M_L validation
├── ctf5_heteroclinic_core.py  # ★★★ Rigorous M_L with exact saddle conditions
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

### CTF-3: Fisher Information Singularity

**Question:** Does $I(\lambda)$ diverge at $E(\lambda) = 0$?

**Method:** Sweep $\lambda \in [0.8, 1.2]$, measure FIM via finite differences on trajectory log-likelihood

**Expected Result:**
```
    I(λ)
     ▲
     │        *
     │      * * *
     │     *     *
     │    *       *
     │   *         *
     │  *           *
     │ *             *
     └──────────────────► λ
        0.8   1.0   1.2
               ↑
         E(λ) = 0
```

Peak/divergence at $\lambda = 1.0$ where $E(\lambda) = 0$.

**Run (canonical test with power-law fit):**
```bash
python ctf3_fisher_singularity.py
```

**Outputs:**
- `ctf3_E_C_I_vs_lambda.png`: Triple-panel E, C, I vs λ
- `ctf3_I_vs_absE_loglog.png`: Log-log scaling plot with γ estimate

---

### CTF-4: Heteroclinic Memory Validation (Final Synthesis)

**Question:** Does Critical + $M_L$ structure outperform other regimes on associative tasks?

**Method:** Compare three agents on a sequential association task ($P_1 \to P_2 \to P_3$):
- **Critical (SOC + M_L)**: E(λ) ≈ 0, heteroclinic links enabled
- **Ordered (No M_L)**: λ = 0.5, no memory structure
- **Chaotic (No M_L)**: λ = 1.5, no memory structure

**Architecture:**
$$W = \lambda \cdot W_{\text{base}} + W_{\text{mem}}$$

where $W_{\text{mem}} = \sum_i W_{P_i} + \sum_{ij} C_{ij} \cdot W_{\Gamma_{ij}}$

**Expected Result:**
| Agent | Avg Reward | E(λ) |
|-------|------------|------|
| Critical (SOC + M_L) | **Highest** | ≈ 0 |
| Ordered (No M_L) | Low | < 0 |
| Chaotic (No M_L) | Low | > 0 |

**Run:**
```bash
python ctf4_heteroclinic_memory.py
```

**Outputs:**
- `CTF-4_Heteroclinic_Validation.png`: 4-panel (Reward, E(λ), C(λ), T(λ))

**Validates:** $M_W$ (power-law dynamics) + $M_L$ (heteroclinic memory) + SOC (criticality) = optimal cognition

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

6. **Fisher Information (score function):**
   $$I(\lambda) = \mathbb{E}\left[\left(\frac{\partial}{\partial\lambda} \log p_\lambda(x_{0:T})\right)^2\right]$$

7. **IG Singularity at criticality:**
   $$I(\lambda) \sim |E(\lambda)|^{-\gamma}$$

8. **Learning rate bound:**
   $$\eta \lesssim I(\lambda)^{-1} \sim |E(\lambda)|^{\gamma}$$

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
4. **FIM diverges at criticality** → RG fixed point = IG singularity (CTF-3)
5. **M_L + criticality = optimal association** → Structured memory needs critical substrate (CTF-4)

This is a **numerical validation** that "thought" (information processing, memory, adaptive behavior) is maximized exactly where the dynamical system is critical.

**The Information-Geometric Singularity Theorem:**
- The critical surface $\mathcal{M}_c$ is simultaneously a phase boundary in dynamics AND a singularity in model space
- At criticality, nearby parameters produce macroscopically different distributions
- This is **maximal distinguishability**: optimal sensitivity to structure (not chaos)
- Two systems are "the same kind of thinker" iff they share the same universality class

**The Heteroclinic Memory Theorem (CTF-4):**
- $M_W$ (power-law dynamics): spontaneous avalanche activity at E(λ) = 0
- $M_L$ (heteroclinic memory): saddle-node patterns + link manifolds
- Both coexist ONLY at criticality: $W = \lambda \cdot W_{\text{base}} + W_{\text{mem}}$
- Associative sequence traversal ($P_1 \to P_2 \to P_3$) is stable IFF E(λ) ≈ 0

No neurons. No AI. Just $E(\lambda) = 0$, $\max C(\lambda)$, $I(\lambda) \to \infty$, and $M_W + M_L$.
