# Hierarchical Heteroclinic Networks (HHNs) for Multi-Scale Cognition

**Advanced structural realization of $M_L$ and Agency within critical systems.**

---

## Overview

Hierarchical Heteroclinic Networks (HHNs) extend the basic heteroclinic memory architecture to **multiple scales**, enabling:
- Chunking and hierarchical sequencing
- Multi-scale attention dynamics
- Nested decision-making
- Structured itinerant cognition ("cognons")

HHNs solve the fundamental problem: *How do transient, scale-free critical dynamics organize into complex, nested cognitive structures?*

---

## I. Mathematical Foundations

### 1.1 Multi-Level Dynamics

HHNs are coupled dynamical systems organized into layers $l \in \{1, 2, \ldots, L\}$. The activity $a_i^{(l)}$ at node $i$ in level $l$ evolves as:

$$\dot{a}_i^{(l)} = a_i^{(l)} \left( \sigma_i^{(l)} - \sum_j \rho_{ij}^{(l)} a_j^{(l)} \right) + \sum_{k \in l-1} \beta_{ik} a_k^{(l-1)} + \eta_i(t)$$

where:
- $\sigma_i^{(l)}$: Growth rate at node $i$, level $l$
- $\rho_{ij}^{(l)}$: Within-level inhibitory coupling (competition)
- $\beta_{ik}$: Bottom-up coupling from level $l-1$ to level $l$
- $\eta_i(t)$: Noise term

### 1.2 Timescale Separation

| Level | Dynamics | Timescale | Cognitive Role |
|-------|----------|-----------|----------------|
| $l = 1$ (Bottom) | Fast cycles: $P_A \to P_B \to P_C$ | $\tau_1 \sim O(1)$ | Elements, primitives |
| $l = 2$ | Medium: $P_X \to P_Y$ | $\tau_2 \sim O(10)$ | Chunks, phrases |
| $l = L$ (Top) | Slow: $P_{\alpha} \to P_{\beta}$ | $\tau_L \sim O(100)$ | Plans, goals |

### 1.3 Hierarchical Compression

A single saddle $P_X^{(L)}$ at level $L$ represents an entire sequence from level $l-1$:

$$P_X^{(L)} \longleftrightarrow \{P_A^{(l-1)} \to P_B^{(l-1)} \to P_C^{(l-1)}\}$$

This achieves **optimal compression**: high-dimensional input sequences → low-dimensional latent variables.

---

## II. Criticality in HHNs

### 2.1 The Marginal Stability Constraint

At each saddle $P_i^{(l)}$, the Jacobian eigenvalues satisfy the HHN criticality condition:

$$\Re \sigma_k^{(i,l)} \in [-\epsilon^{(l)}, +\delta^{(l)}], \quad 0 < \delta^{(l)} \leq \epsilon^{(l)} \ll 1$$

where the stability margins $\epsilon^{(l)}, \delta^{(l)}$ may vary by level.

### 2.2 Level-Dependent Dwell Times

The dwell time near saddle $P_i^{(l)}$ scales as:

$$\tau_{\text{dwell}}^{(l)} \propto -\frac{1}{\delta^{(l)}} \log \epsilon^{(l)}$$

**Design principle:** Higher levels have smaller $\delta^{(l)}$, yielding longer dwell times.

### 2.3 Global Criticality Preservation

The full system remains globally critical:

$$E(\lambda) = \max_{l,i,k} \Re \sigma_k^{(i,l)} \approx 0$$

The hierarchical structure does not break criticality—it **organizes** critical dynamics across scales.

---

## III. Cognitive Functions

### 3.1 Chunking and Sequencing

**Problem:** Working memory is limited (~7±2 items, Miller's Law).

**HHN Solution:** Lower-level sequences are compressed into single higher-level saddles.

```
Level 2:     [P_chunk1] ───────────► [P_chunk2]
                 │                        │
Level 1:   P_a→P_b→P_c              P_d→P_e→P_f
```

**Mechanism:**
- Level 1: Fast transitions between primitive elements
- Level 2: Each chunk is a single saddle with slow dynamics
- Active memory load: Number of Level-2 saddles, not Level-1 elements

### 3.2 Multi-Scale Attention

| Attention Level | HHN Mechanism | Timescale |
|-----------------|---------------|-----------|
| **Salience (Bottom-up)** | Fast Level-1 transitions triggered by sensory input | Fast ($\tau_1$) |
| **Focus (Top-down)** | Slow Level-L stabilization preventing premature switching | Slow ($\tau_L$) |
| **Sustained Attention** | High-level saddle maintains activation despite Level-1 noise | Very slow |

**The attention bottleneck:** The number of simultaneously active Level-L saddles determines attentional capacity.

### 3.3 Hierarchical Decision-Making

```
Level 3 (Goals):     [Goal_A] ─────────────────────► [Goal_B]
                         │                               │
Level 2 (Plans):    [Plan_1]→[Plan_2]           [Plan_3]→[Plan_4]
                      │   │                       │   │
Level 1 (Actions): a→b→c d→e                   f→g   h→i→j
```

**Deliberation time:** The dwell time at Level-3 saddles during decision-making reflects evidence accumulation:

$$T_{\text{decision}} \propto \tau_{\text{dwell}}^{(3)} \propto -\log \epsilon^{(3)} / \delta^{(3)}$$

### 3.4 Cognons: Units of Thought

A **cognon** is a complete hierarchical heteroclinic trajectory—the structured, nested wandering between discrete cognitive states that constitutes a single "thought."

$$\text{Cognon} = \{P_{i_1}^{(1)} \to P_{i_2}^{(1)} \to \ldots\} \cup \{P_{j_1}^{(2)} \to P_{j_2}^{(2)} \to \ldots\} \cup \ldots \cup \{P_{k}^{(L)}\}$$

**Properties:**
- Non-periodic but structured
- Scale-free switching enabled by criticality
- Realizes universal computation at $E(\lambda) = 0$

---

## IV. Weight Structure for HHNs

### 4.1 Full Decomposition

The weight matrix decomposes hierarchically:

$$W = \underbrace{\lambda W_{\text{base}}}_{M_W} + \underbrace{\sum_{l=1}^{L} \left( \sum_i W_{P_i}^{(l)} + \sum_{ij} W_{\Gamma_{ij}}^{(l)} \right)}_{M_L^{\text{intra}}} + \underbrace{\sum_{l=1}^{L-1} \sum_{ik} W_{\beta_{ik}}^{(l \to l+1)}}_{M_L^{\text{inter}}}$$

where:
- $W_{\text{base}}$: Critical bulk (working memory $M_W$)
- $W_{P_i}^{(l)}$: Saddles at level $l$ (intra-level $M_L$)
- $W_{\Gamma_{ij}}^{(l)}$: Connections within level $l$ (intra-level $M_L$)
- $W_{\beta_{ik}}^{(l \to l+1)}$: Bottom-up connections between levels (inter-level $M_L$)

### 4.2 Low-Rank Structure

Each component is low-rank:

$$W_{P_i}^{(l)} = v_i^{(l)} (P_i^{(l)})^T, \quad \text{rank} = 1$$

$$W_{\Gamma_{ij}}^{(l)} = c_{ij}^{(l)} q_{ij}^{(l)} (r_{ij}^{(l)})^T, \quad \text{rank} = 1$$

$$W_{\beta_{ik}}^{(l \to l+1)} = \gamma_{ik} u_i^{(l+1)} (P_k^{(l)})^T, \quad \text{rank} = 1$$

Total parameters scale as $O(n \cdot K \cdot L)$ where $K$ = patterns per level.

---

## V. Learning Rules for HHNs

### 5.1 Extended Unified Learning Rule

$$\frac{d\theta}{dt} = \eta_{\text{task}} \nabla_\theta \mathcal{L}_{\text{Task}} - \eta_{\text{soc}} \nabla_\theta |E(\theta)|^2 + \sum_{l=1}^{L} \eta_{\text{mem}}^{(l)} \nabla_\theta \mathcal{L}_{\text{Mem}}^{(l)}$$

where each level has its own memory loss:

$$\mathcal{L}_{\text{Mem}}^{(l)} = \mathcal{L}_{\text{fix}}^{(l)} + \mathcal{L}_{\text{trans}}^{(l)} + \mathcal{L}_{\text{hierarchy}}^{(l)}$$

### 5.2 Hierarchy Loss

The hierarchy loss ensures proper bottom-up activation:

$$\mathcal{L}_{\text{hierarchy}}^{(l)} = \sum_{i,k} \left( a_i^{(l+1)} - f\left(\sum_k \beta_{ik} a_k^{(l)}\right) \right)^2$$

This enforces: higher-level saddles activate only when lower-level sequences complete.

### 5.3 Timescale Learning

Learning rates decrease with level:

$$\eta_{\text{mem}}^{(l)} = \eta_{\text{mem}}^{(1)} \cdot \tau_1 / \tau_l$$

Slower levels adapt more slowly, preserving hierarchical structure.

---

## VI. Advantages for Critical Systems

### 6.1 Robustness + Flexibility

| Property | Flat HN | Hierarchical HHN |
|----------|---------|------------------|
| Noise resistance | Low (single scale) | High (higher levels integrate noise) |
| Flexibility | High (all saddles equal) | Structured (levels constrain transitions) |
| Capacity | $O(K)$ | $O(K^L)$ combinatorial |

### 6.2 Optimal Compression

HHNs achieve **rate-distortion optimal** compression of temporal sequences:

$$R(D) = \min_{p(\hat{x}|x)} I(X; \hat{X}) \quad \text{s.t.} \quad \mathbb{E}[d(X, \hat{X})] \leq D$$

The hierarchical saddle structure naturally discovers the optimal coding scheme.

### 6.3 Scale-Free Integration

At criticality ($E = 0$), HHNs exhibit **scale-free temporal integration**:

$$C(\tau) \sim \tau^{-\alpha}, \quad \text{across all levels}$$

This enables seamless coordination between fast sensory processing and slow deliberation.

---

## VII. Clinical Implications

### 7.1 HHN Instabilities and Disorders

| Disorder | HHN Mechanism | Level Affected |
|----------|---------------|----------------|
| **ADHD** | $\delta^{(l)}$ too large → premature switching | Levels 2-3 (attention) |
| **OCD** | $\epsilon^{(l)}$ too small → excessive dwell times | Levels 2-3 (plans) |
| **Autism** | Inter-level coupling $\beta_{ik}$ disrupted | $l \to l+1$ transitions |
| **Schizophrenia** | Global criticality lost ($E > 0$) | All levels (chaotic) |

### 7.2 Therapeutic Targets

**Intervention principle:** Restore optimal marginal stability at the affected level:

$$\delta^{(l)} \approx \epsilon^{(l)} \ll 1$$

Pharmacological or neurostimulation interventions could target specific hierarchical levels.

---

## VIII. Implementation Sketch

### 8.1 Minimal HHN (2 Levels, 3+2 Patterns)

```python
class HierarchicalHHN:
    def __init__(self, n_dim=10, n_patterns_L1=3, n_patterns_L2=2):
        # Level 1: Fast primitives
        self.P_L1 = [random_pattern(n_dim) for _ in range(n_patterns_L1)]
        self.delta_L1, self.epsilon_L1 = 0.08, 0.10

        # Level 2: Slow chunks (each represents L1 sequence)
        self.P_L2 = [random_pattern(n_dim) for _ in range(n_patterns_L2)]
        self.delta_L2, self.epsilon_L2 = 0.02, 0.03  # Slower

        # Inter-level coupling
        self.beta = np.zeros((n_patterns_L2, n_patterns_L1))
        self.beta[0, :] = [1.0, 0.5, 0.2]  # Chunk 0 = P0→P1→P2
        self.beta[1, :] = [0.2, 0.5, 1.0]  # Chunk 1 = P2→P1→P0

    def step(self, x, dt=0.01):
        # Level 1 dynamics (fast)
        a_L1 = self.compute_activations(x, self.P_L1)

        # Level 2 dynamics (slow, driven by L1)
        a_L2 = self.beta @ a_L1

        # Combined update preserving criticality
        dx = self.critical_update(x, a_L1, a_L2)
        return x + dt * dx
```

### 8.2 Integration with CTF-5

HHNs extend the CTF-5 rigorous heteroclinic core by adding:
1. Multiple levels of saddles
2. Bottom-up coupling matrices $\beta$
3. Level-dependent stability margins
4. Hierarchy loss in learning

---

## IX. Connection to GUTC

### 9.1 The Extended Memory Equation

$$M_L = M_L^{\text{flat}} + M_L^{\text{hierarchical}}$$

where:
- $M_L^{\text{flat}}$: Single-level heteroclinic connections (CTF-4/5)
- $M_L^{\text{hierarchical}}$: Multi-level HHN structure (this document)

### 9.2 The Full Cognitive Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GUTC COGNITIVE CORE                      │
├─────────────────────────────────────────────────────────────┤
│  M_W: Critical Bulk                                         │
│  └── E(λ) ≈ 0, power-law correlations, scale-free M_W      │
├─────────────────────────────────────────────────────────────┤
│  M_L: Heteroclinic Memory                                   │
│  ├── Level L: Goals, Plans (τ ~ 100)                       │
│  │   └── P_α^(L) ──► P_β^(L)                               │
│  ├── Level 2: Chunks, Phrases (τ ~ 10)                     │
│  │   └── P_X^(2) ──► P_Y^(2) ──► P_Z^(2)                   │
│  └── Level 1: Primitives, Elements (τ ~ 1)                 │
│      └── P_a^(1) ──► P_b^(1) ──► P_c^(1)                   │
├─────────────────────────────────────────────────────────────┤
│  Agency: Policy conditioned on (M_W, M_L)                   │
│  └── a_t = π(x_t, P_{\text{active}}^{(1:L)})               │
└─────────────────────────────────────────────────────────────┘
```

### 9.3 The Ultimate Equation

$$\text{Cognition} = \underbrace{E(\lambda) = 0}_{\text{Criticality}} + \underbrace{M_W}_{\text{Working Memory}} + \underbrace{M_L^{\text{HHN}}}_{\text{Hierarchical Long-Term Memory}} + \underbrace{\pi(\cdot)}_{\text{Agency}}$$

---

## References

1. Rabinovich, M. I., Huerta, R., Varona, P., & Afraimovich, V. S. (2008). Transient cognitive dynamics, metastability, and decision making. *PLoS Computational Biology*, 4(5), e1000072.

2. Rabinovich, M. I., Muezzinoglu, M. K., Strigo, I., & Bystritsky, A. (2010). Dynamical principles of emotion-cognition interaction: Mathematical images of mental disorders. *PLoS ONE*, 5(9), e12547.

3. Rabinovich, M. I., & Varona, P. (2011). Robust transient dynamics and brain functions. *Frontiers in Computational Neuroscience*, 5, 24.

4. Afraimovich, V. S., Zhigulin, V. P., & Rabinovich, M. I. (2004). On the origin of reproducible sequential activity in neural circuits. *Chaos*, 14(4), 1123-1129.

5. Kiebel, S. J., Daunizeau, J., & Friston, K. J. (2008). A hierarchy of time-scales and the brain. *PLoS Computational Biology*, 4(11), e1000209.

6. Miller, G. A. (1956). The magical number seven, plus or minus two. *Psychological Review*, 63(2), 81-97.

7. Golos, M., Jirsa, V., & Daucé, E. (2015). Multistability in large scale models of brain activity. *PLoS Computational Biology*, 11(12), e1004644.

8. Muñoz, M. A. (2018). Colloquium: Criticality and dynamical scaling in living systems. *Reviews of Modern Physics*, 90(3), 031001.
