# Noisy Heteroclinic Dynamics: Dwell Times, Branching, and Noise Scaling

**Mathematical theory of stochastic escape from saddles in heteroclinic networks.**

---

## Overview

In real systems, heteroclinic networks operate under noise. This document provides the mathematical theory for:
1. **Dwell time scaling** with noise amplitude
2. **Branching probabilities** with multiple unstable eigenvalues
3. **Noise-regime transitions** from heteroclinic to excitable behavior

These results are essential for understanding how $M_L$ (heteroclinic memory) operates in practice and how noise strength affects cognitive dynamics.

---

## I. Dwell Time Scaling: The Fundamental Result

### 1.1 Local Linear Model

Near a saddle equilibrium $\xi$, linearize dynamics along the unstable direction:

$$\dot{x} = \lambda x + \sigma \xi(t)$$

where:
- $x$: coordinate along unstable eigenvector
- $\lambda > 0$: unstable eigenvalue
- $\xi(t)$: standard white noise
- $\sigma$: noise amplitude

### 1.2 Solution and Escape

The solution is:

$$x(t) = x_0 e^{\lambda t} + \sigma \int_0^t e^{\lambda(t-s)} \xi(s) \, ds$$

The stochastic integral has variance:

$$\text{Var}[x(t)] \sim \frac{\sigma^2}{2\lambda}(e^{2\lambda t} - 1)$$

Escape occurs when $|x|$ reaches threshold $x_c$ (beyond which linearization fails and trajectory follows heteroclinic connection).

### 1.3 The Logarithmic Scaling Law

Setting typical magnitude equal to threshold:

$$x_c \sim K \sigma e^{\lambda t}$$

Solving for escape time $t$:

$$t \sim \frac{1}{\lambda}\left(\log \frac{x_c}{K} - \log \sigma\right)$$

**The Fundamental Dwell Time Formula:**

$$\boxed{\langle T_{\text{dwell}}(\sigma) \rangle \approx A - B \log \sigma, \quad B = \frac{1}{\lambda}}$$

Equivalently:

$$\boxed{\langle T_{\text{dwell}}(\sigma) \rangle \approx \frac{1}{\lambda} \log\left(\frac{\kappa}{\sigma}\right)}$$

where $\kappa$ depends on escape boundary and local geometry.

### 1.4 Network-Level Cycle Period

For a heteroclinic network with multiple saddles, the cycle period is:

$$\langle T_{\text{cycle}}(\sigma) \rangle \approx \sum_j \frac{1}{\lambda_j} \log\left(\frac{\kappa_j}{\sigma}\right)$$

Each saddle contributes logarithmically with prefactor $1/\lambda_j$.

---

## II. Multidimensional Saddles

### 2.1 Multiple Unstable Directions

For a saddle with $k$ unstable directions, linearize:

$$\dot{\mathbf{x}}_u = \Lambda_u \mathbf{x}_u + \sigma \boldsymbol{\xi}_u(t)$$

where $\Lambda_u = \text{diag}(\lambda_1, \ldots, \lambda_k)$ with all $\lambda_i > 0$.

### 2.2 Dominant Eigenvalue Scaling

Let $\lambda_* = \max_i \lambda_i$. The dwell time scaling remains:

$$\langle T_{\text{dwell}}(\sigma) \rangle \approx \frac{1}{\lambda_*} \log\left(\frac{\kappa}{\sigma}\right)$$

but with prefactors and corrections depending on:
- Full spectrum of stable/unstable eigenvalues
- Noise covariance projected onto eigendirections
- Geometry of exit sectors

### 2.3 Role of Stable Eigenvalues

Stable directions $(-\mu_j)$ affect:
- **Prefactors:** Stronger contraction funnels trajectories closer to unstable manifold
- **Subleading terms:** Corrections of order $O(1)$ or $O(1/\log(1/\sigma))$
- **Approach geometry:** How trajectories enter the saddle neighborhood

---

## III. Branching Probabilities with Multiple Unstable Directions

### 3.1 The Branching Problem

Multiple unstable eigenvalues turn escape into a **splitting problem**: which heteroclinic branch does the trajectory follow?

Each branch corresponds to a sector in the unstable subspace (e.g., $x_1 > 0$ vs. $x_2 > 0$).

### 3.2 Asymptotic Exit Probabilities

**Single Dominant Eigenvalue ($\lambda_1 > \lambda_2$):**

$$\mathbb{P}(\text{exit via branch 1}) \to 1 \quad \text{as } \sigma \to 0$$

The dominant direction overwhelmingly determines the exit.

**Degenerate Eigenvalues ($\lambda_1 = \lambda_2$):**

Exit distribution is **non-degenerate**: both branches remain likely even as $\sigma \to 0$.

Limiting probabilities determined by:
- Noise covariance projected onto unstable eigenspace
- Geometry of exit sectors

**General Case:**

$$\mathbb{P}(\text{exit via branch } i) \sim c_i \varepsilon^{\theta_i}$$

where exponents $\theta_i$ and prefactors $c_i$ are set by eigenvalue ratios and boundary geometry.

### 3.3 Cognitive Implications

| Saddle Type | Behavior | Cognitive Role |
|-------------|----------|----------------|
| Single dominant $\lambda$ | Deterministic routing | Automatic associations |
| Multiple comparable $\lambda_i$ | Noise-sensitive branching | Decision points |
| Isotropic unstable manifold | Genuine random choice | Creative exploration |

**Network design principle:** Place deterministic routers (single $\lambda$) for reliable sequences; place decision nodes (multiple $\lambda_i$) for context-dependent branching.

---

## IV. Noise Regime Transitions

### 4.1 Low Noise Regime

- Classic heteroclinic behavior
- Long dwells near saddles ($\propto \log(1/\sigma)$)
- Fast transitions along heteroclinic orbits
- **Dwell-time distribution:** Broad, right-skewed, heavy-tailed

### 4.2 Intermediate Noise Regime

- Heteroclinic structure blurred
- Lower-level cycles collapse into effective fixed points
- Dwells shorten
- **Dwell-time distribution:** Less heavy-tailed

### 4.3 High Noise Regime

- Saddle structure no longer felt
- System behaves like noisy single-well attractor
- **Dwell-time distribution:** Approaches exponential

### 4.4 Scaling Summary

| Regime | Dwell Time Scaling | Distribution |
|--------|-------------------|--------------|
| Low $\sigma$ (heteroclinic) | $\langle T \rangle \sim -\log \sigma$ | Heavy-tailed |
| Intermediate $\sigma$ | Crossover | Mixed |
| High $\sigma$ (excitable) | $\langle T \rangle \sim \sigma^{-p}$ (Kramers) | Exponential |

---

## V. Near Bifurcation Crossover

### 5.1 Heteroclinic–Excitable Bifurcation

When parameters tune through the heteroclinic boundary:

$$\langle T_{\text{dwell}} \rangle \sim
\begin{cases}
C_1 - C_2 \log \sigma & \text{(heteroclinic side)} \\[4pt]
C_3 \sigma^{-p} & \text{(Kramers/excitable side)}
\end{cases}$$

### 5.2 Critical Implications

The **critical regime** ($E(\lambda) \approx 0$) sits precisely at this boundary:
- Heteroclinic-like scaling for small perturbations
- But sensitivity to parameter changes (IG singularity)

This connects dwell-time statistics to the broader GUTC framework.

---

## VI. Hierarchical Networks Under Noise

### 6.1 Level-Dependent Noise Effects

In HHNs, noise affects levels differently:

| Level | Eigenvalue Scale | Dwell Time | Noise Sensitivity |
|-------|------------------|------------|-------------------|
| L1 (fast) | Large $\lambda$ | Short | High |
| L2 (medium) | Medium $\lambda$ | Medium | Medium |
| LL (slow) | Small $\lambda$ | Long | Low |

### 6.2 Hierarchical Collapse

As noise increases:
1. **First:** Lower-level (fast) cycles collapse
2. **Then:** Higher-level structures destabilize
3. **Finally:** Entire hierarchy dissolves

**Practical implication:** Protect higher levels by design (smaller $\lambda$) to maintain long-term memory under noise.

### 6.3 Optimal Noise Level

There exists an **optimal noise regime** where:
- Low enough: Heteroclinic structure preserved
- High enough: Transitions occur in finite time
- Result: Functional itinerant dynamics

$$\sigma_{\text{opt}} \sim \exp(-\lambda_* \cdot T_{\text{desired}})$$

---

## VII. Formulas Summary

### Dwell Time

$$\langle T_{\text{dwell}} \rangle = \frac{1}{\lambda_*} \log\left(\frac{\kappa}{\sigma}\right)$$

### Cycle Period

$$\langle T_{\text{cycle}} \rangle = \sum_j \frac{1}{\lambda_j} \log\left(\frac{\kappa_j}{\sigma}\right)$$

### Branching (single dominant)

$$\mathbb{P}(\text{branch } i) \approx \begin{cases} 1 & \lambda_i = \lambda_* \\ 0 & \lambda_i < \lambda_* \end{cases}$$

### Branching (comparable eigenvalues)

$$\mathbb{P}(\text{branch } i) = f\left(\frac{\lambda_i}{\lambda_*}, \Sigma_{\text{noise}}, \text{geometry}\right)$$

### Noise-Dwell Regimes

$$\langle T \rangle \sim \begin{cases}
-\frac{1}{\lambda} \log \sigma & \sigma \ll \sigma_c \text{ (heteroclinic)} \\
\sigma^{-p} & \sigma \gg \sigma_c \text{ (Kramers)}
\end{cases}$$

---

## VIII. Connection to GUTC

### 8.1 The Noise–Criticality–Memory Triangle

```
                    Criticality (E ≈ 0)
                         /\
                        /  \
                       /    \
                      /      \
                     /        \
          Noise (σ) ────────── Memory (M_L)
```

- **E ≈ 0 + low σ:** Long dwells, stable heteroclinic sequences
- **E ≈ 0 + high σ:** Fast transitions, degraded memory
- **E ≠ 0:** Loss of criticality benefits regardless of noise

### 8.2 Design Principles for Robust $M_L$

1. **Eigenvalue separation:** Use single dominant $\lambda$ for reliable sequences
2. **Noise protection:** Larger patterns (more dimensions) average noise
3. **Hierarchical shielding:** Slow top levels protected from fast noise
4. **SOC maintenance:** Keep $E(\lambda) \approx 0$ despite perturbations

### 8.3 The Learning Rate Connection

Recall: $\eta \lesssim I(\lambda)^{-1} \sim |E|^{\gamma}$

Combined with dwell time scaling:

$$\text{Effective learning} \propto \frac{\eta}{\langle T_{\text{dwell}} \rangle} \propto \frac{|E|^{\gamma}}{\log(1/\sigma)}$$

Near criticality, both Fisher information and dwell times are large, creating a "slow learning, stable memory" regime.

---

## IX. Experimental Predictions

### 9.1 Dwell Time vs. Noise

**Prediction:** $\langle T \rangle$ vs. $\sigma$ on log-linear plot should be linear with slope $-1/\lambda_*$.

### 9.2 Branching Statistics

**Prediction:** With comparable eigenvalues, repeated trials show non-trivial branching distributions even at low noise.

### 9.3 Distribution Shape

**Prediction:** Dwell-time histograms should be:
- Right-skewed at low noise
- Exponential-like at high noise
- Power-law–like at criticality

---

## References

1. Stone, E., & Holmes, P. (1990). Random perturbations of heteroclinic attractors. *SIAM J. Appl. Math.*, 50(3), 726-743.

2. Armbruster, D., Stone, E., & Kirk, V. (2003). Noisy heteroclinic networks. *Chaos*, 13(1), 71-79.

3. Bakhtin, Y. (2011). Noisy heteroclinic networks. *Probability Theory and Related Fields*, 150(1), 1-42.

4. Ashwin, P., & Postlethwaite, C. (2013). On designing heteroclinic networks from graphs. *Physica D*, 265, 26-39.

5. Creaser, J., Tsaneva-Atanasova, K., & Sherwin, S. (2019). Transitions between regimes in neural systems. *Nonlinear Dynamics*, 97(4), 2513-2534.

6. Agarwal, N., & Field, M. (2022). Asymptotics of escape from unstable equilibria. *Annals of Probability*, 50(3), 1185-1234.

7. Muñoz, M. A. (2018). Colloquium: Criticality and dynamical scaling in living systems. *Rev. Mod. Phys.*, 90(3), 031001.

8. Rabinovich, M. I., et al. (2008). Transient cognitive dynamics, metastability, and decision making. *PLoS Comput. Biol.*, 4(5), e1000072.
