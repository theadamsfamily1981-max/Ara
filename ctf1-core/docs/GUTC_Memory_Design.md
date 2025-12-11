# Memory Design in the Critical Thought Field

## Box: Classical Attractors vs GUTC Heteroclinic Networks

| Feature | Classical attractor model (Hopfield-type) | GUTC / critical model (heteroclinic network) |
|--------|--------------------------------------------|---------------------------------------------|
| **Global regime** | Deeply subcritical, $E(\lambda) < 0$; large negative real parts of eigenvalues | Marginally stable, $E(\lambda) \approx 0$; all $\Re\sigma_k$ confined to a small band around zero |
| **Storage element $P_i$** | Stable fixed point with $\Re\sigma_k^{(i)} \ll 0$ for all $k$ (convergent attractor) | Saddle(-focus) equilibrium with $\Re\sigma_k^{(i)} \in [-\epsilon,+\delta]$, where $0<\delta\le\epsilon\ll 1$ |
| **Working memory $M_W$** | Limited by exponential relaxation time $\tau\sim \log(1/\epsilon)$; short-range correlations | Power-law temporal correlations $C(\tau)\sim\tau^{-\alpha}$; correlation length $\xi\sim \|E\|^{-\nu}$ diverges at $E=0$ |
| **Long-term memory $M_L$** | Static attractor basins; patterns compete and can overwrite each other | Patterns as saddles $P_i$ plus heteroclinic connections $\Gamma_{ij} = W^u(P_i)\cap W^s(P_j)$, storing associations as graph structure |
| **Recall / association** | Convergence and locking in a single attractor; switching requires large perturbations | Itinerant traversal along heteroclinic network; small noise or input can trigger reproducible sequences $P_i\to P_j\to P_k$ |
| **Flexibility & interference** | Rigid; prone to catastrophic interference as new patterns reshape energy landscape | Metastable; structured yet flexible sequences, with selective links $\Gamma_{ij}$ that can be strengthened or pruned locally |

---

## Equations: $M_W$, $M_L$, and Criticality

### The Edge Function

Global criticality is controlled by the edge function:

$$E(\lambda) = \lim_{t\to\infty}\frac{1}{t}\log\frac{\|\delta x_t\|}{\|\delta x_0\|}$$

### The Thought Condition

Thought operates where:

$$\text{Thought} \iff \max_\lambda C(\lambda)\ \text{ at }\ E(\lambda)=0$$

---

### The $M_L$ Condition (Local Eigenvalue Band)

Within the critical regime, long-term memory is encoded in the local Jacobians at pattern states $P_i$:

$$\max_{i,k}\big|\Re\sigma_k^{(i)}\big| \approx \epsilon,\quad 0<\epsilon\ll 1$$

where $\sigma_k^{(i)}$ are eigenvalues of $J(P_i)$.

This ensures:

1. **Global criticality preserved:** $|E(\lambda)| \approx 0$ since no local eigenvalue is strongly expanding or contracting

2. **Metastability:** Dwell times near $P_i$ scale as $\tau_{\text{dwell}}\propto -\log\epsilon$, long but finite under noise

3. **Controllable switching:** Small inputs/noise along $W^u(P_i)$ trigger transitions via $\Gamma_{ij}$

---

### The Weight Decomposition

The combined weight structure separates working and long-term memory:

$$W = \underbrace{\lambda W_{\text{base}}}_{\text{$M_W$: critical bulk}} \;+\; \underbrace{\sum_i W_{P_i} + \sum_{i,j} W_{\Gamma_{ij}}}_{\text{$M_L$: heteroclinic structure}}$$

where:

- **$W_{\text{base}}$**: High-dimensional recurrent matrix tuned so $\rho(\lambda W_{\text{base}})\approx 1$ (edge-of-chaos reservoir)

- **$W_{P_i}$**: Low-rank terms that carve saddles at desired pattern states $P_i$

- **$W_{\Gamma_{ij}}$**: Low-rank terms that align unstable and stable manifolds to realize heteroclinic connections $\Gamma_{ij}$

---

### Memory Origins

In this design:

| Memory Type | Origin | Properties |
|-------------|--------|------------|
| **$M_W$ (Working)** | Critical bulk dynamics | High excess entropy, long correlation length, power-law temporal correlations |
| **$M_L$ (Long-term)** | Heteroclinic network embedded in bulk | Structured associative sequences without leaving the critical band |

---

### The Unified Learning Rule

$$\frac{d\theta}{dt} = \eta_{\text{task}}\nabla_\theta\mathcal{L}_{\text{Task}} - \eta_{\text{soc}}\nabla_\theta |E(\theta)|^2 + \eta_{\text{mem}}\nabla_\theta\mathcal{L}_{\text{Mem}}$$

| Term | Purpose | Rate |
|------|---------|------|
| $\nabla_\theta\mathcal{L}_{\text{Task}}$ | Task performance (Agency) | $\eta_{\text{task}}$ (fast) |
| $-\nabla_\theta \|E(\theta)\|^2$ | Criticality maintenance (SOC) | $\eta_{\text{soc}}$ (slow) |
| $\nabla_\theta\mathcal{L}_{\text{Mem}}$ | Memory structure (Heteroclinic links) | $\eta_{\text{mem}}$ (medium) |

---

## Why Heteroclinic Networks Over Classical Attractors?

### Classical Attractors (Hopfield-type)

- **Problem:** Deep basins → rigid, frozen dynamics
- **Interference:** New patterns reshape energy landscape catastrophically
- **Switching:** Requires large perturbations to escape attractors
- **Correlations:** Exponential decay, short memory

### GUTC Heteroclinic Networks

- **Solution:** Saddle equilibria → metastable, flexible dynamics
- **Structure:** Associations stored as manifold connections $\Gamma_{ij}$
- **Switching:** Small inputs trigger reproducible sequences
- **Correlations:** Power-law decay, long memory (criticality)

---

## The Core Insight

Memory in the GUTC is implemented as a **critical heteroclinic network** rather than classical deep subcritical attractors because:

1. **$M_W$ requires $E(\lambda) \approx 0$** for power-law correlations and high capacity

2. **$M_L$ requires saddles, not stable attractors**, for itinerant sequential recall

3. **Both coexist only at criticality**, where eigenvalues are confined to a narrow band around zero

4. **The weight decomposition $W = \lambda W_{\text{base}} + W_{\text{mem}}$** naturally separates the two memory systems while preserving global criticality

---

## References

1. Muñoz, M.A. (2018). Colloquium: Criticality and dynamical scaling in living systems. *Rev. Mod. Phys.* 90, 031001.
2. Boedecker, J. et al. (2012). Information processing in echo state networks at the edge of chaos. *Theory Biosci.* 131, 205-213.
3. Rabinovich, M.I. et al. (2008). Transient cognitive dynamics, metastability, and decision making. *PLoS Comput. Biol.* 4, e1000072.
4. Ashwin, P. & Postlethwaite, C. (2013). Itinerant memory dynamics and global bifurcations. *Chaos* 13, 1122.
5. Afraimovich, V.S. et al. (2004). Heteroclinic contours in neural ensembles. *Int. J. Bifurc. Chaos* 14, 1195-1208.
6. Rabinovich, M.I. & Varona, P. (2011). Robust transient dynamics and brain functions. *Front. Comput. Neurosci.* 5, 24.
