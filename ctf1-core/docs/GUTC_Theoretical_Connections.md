# GUTC Theoretical Connections: FEP, Predictive Coding, and Active Inference

**Situating GUTC within the broader landscape of computational theories of mind.**

---

## Overview

The Grand Unified Theory of Cognition (GUTC) is not isolated—it connects to and extends major frameworks:

1. **Free Energy Principle (FEP)** — Friston's variational inference view
2. **Predictive Coding** — Hierarchical error minimization
3. **Active Inference** — Action as inference

This document establishes the mathematical relationships and shows how GUTC provides the **dynamical substrate** for these functional theories.

---

## I. The Free Energy Principle (FEP)

### 1.1 FEP Formulation

The FEP posits that adaptive systems minimize variational free energy:

$$\mathcal{F}(q, u) = D_{\text{KL}}[q(x) \| p(x|u)] + \mathbb{E}_q[-\log p(u|x)]$$

where:
- $q(x)$: Approximate posterior over latent states
- $p(x|u)$: True posterior given observations $u$
- $p(u|x)$: Likelihood (generative model)

Decomposition:
$$\mathcal{F} = \underbrace{D_{\text{KL}}[q \| p]}_{\text{Complexity}} + \underbrace{\mathbb{E}_q[-\log p(u|x)]}_{\text{Accuracy}}$$

### 1.2 GUTC ↔ FEP Connection

| FEP Component | GUTC Realization |
|---------------|------------------|
| Generative model $p(x, u)$ | Dynamical system $F_\lambda$ at criticality |
| Free energy $\mathcal{F}$ | Related to $-C(\lambda)$ (capacity) |
| Inference | Trajectory dynamics minimizing prediction error |
| Markov blanket | Boundary between agent and environment |

**Key insight:** GUTC specifies that the **optimal generative model** operates at criticality ($E = 0$), where $\mathcal{F}$ minimization is most efficient due to maximal sensitivity (Fisher information divergence).

### 1.3 The Fisher Information Link

FEP's natural gradient descent uses the Fisher Information Matrix:

$$\dot{\theta} = -F^{-1}(\theta) \nabla_\theta \mathcal{F}$$

GUTC shows that at criticality:
$$I(\lambda) \sim |E(\lambda)|^{-\gamma} \to \infty$$

**Implication:** At $E = 0$, the system is maximally sensitive to parameter changes, enabling efficient inference—but requiring careful learning rate scaling:

$$\eta \lesssim I(\lambda)^{-1} \sim |E(\lambda)|^{\gamma}$$

---

## II. Predictive Coding

### 2.1 Predictive Coding Formulation

Predictive coding posits hierarchical prediction and error minimization:

$$\epsilon_l = u_l - \hat{u}_l(x_{l+1})$$

where level $l+1$ predicts level $l$. The objective:

$$\mathcal{L}_{\text{PC}} = \sum_l \mathbb{E}[\epsilon_l^2]$$

Update dynamics:
- **Bottom-up:** Prediction errors $\epsilon_l$ propagate upward
- **Top-down:** Predictions $\hat{u}_l$ propagate downward

### 2.2 GUTC ↔ Predictive Coding Connection

| Predictive Coding | GUTC Realization |
|-------------------|------------------|
| Prediction error $\epsilon_l$ | Deviation from heteroclinic trajectory |
| Hierarchical levels | HHN levels (L1 → L_L) |
| Precision weighting | Noise amplitude $\sigma$ in heteroclinic dynamics |
| Top-down predictions | High-level saddle activation constraining lower levels |
| Bottom-up errors | Lower-level deviations triggering transitions |

### 2.3 Mathematical Correspondence

**Predictive Coding dynamics:**
$$\dot{x}_l = -\epsilon_l + \text{(top-down prediction)}$$

**GUTC HHN dynamics:**
$$\dot{a}_i^{(l)} = a_i^{(l)} \left( \sigma_i^{(l)} - \sum_j \rho_{ij}^{(l)} a_j^{(l)} \right) + \sum_{k} \beta_{ik} a_k^{(l-1)}$$

**Mapping:**
- $\epsilon_l$ → deviation from saddle pattern $P_i$
- Top-down → inter-level coupling $\beta_{ik}$
- Precision → inverse of noise $1/\sigma$

### 2.4 Precision and Criticality

In predictive coding, **precision** weights errors (attention = high precision on reliable inputs).

GUTC interpretation:
- **High precision** = Low noise $\sigma$ → Long dwell times $\tau \propto -\log \sigma$
- **Low precision** = High noise → Fast transitions, exploration

At criticality, precision weighting is optimally balanced for both stability and flexibility.

---

## III. Active Inference and Expected Free Energy

### 3.1 Variational Free Energy (VFE)

**Box: VFE in One Equation**

VFE upper-bounds "surprise" ($-\ln p(u)$):

$$\mathcal{F}(q, u) = D_{\text{KL}}[q(x) \| p(x|u)] - \mathbb{E}_q[\ln p(u|x)]$$

Equivalently:
$$\mathcal{F} = \underbrace{\text{Complexity}}_{\text{Minimize}} - \underbrace{\text{Accuracy}}_{\text{Maximize}}$$

Agents seek the simplest internal explanation that accurately predicts observations.

### 3.2 Expected Free Energy (EFE) Derivation

For action selection, agents minimize **expected future free energy**:

$$\mathcal{G}(\pi) = \mathbb{E}_{q(u_{t+1}, x_{t+1}|\pi)}[\mathcal{F}(q, u_{t+1})]$$

**Optimal policy:**
$$\pi^* = \arg\min_\pi \mathcal{G}(\pi)$$

**Decomposition into extrinsic and intrinsic:**

$$\mathcal{G}(\pi) = \underbrace{\mathbb{E}_{q(u|\pi)}[-\ln p(u)]}_{\text{Extrinsic (pragmatic)}} + \underbrace{\mathbb{E}_{q(u|\pi)}[D_{\text{KL}}[q(x|u) \| p(x|u)]]}_{\text{Intrinsic (epistemic)}}$$

- **Extrinsic:** Maximize reward by aligning with preferred outcomes
- **Intrinsic:** Maximize information gain (resolve ambiguity)

### 3.3 The Combined GUTC Objective

The complete cognitive system optimizes:

$$\boxed{\min_{\theta, \phi, q} \quad \mathcal{G}(\pi, \theta, q) - \alpha C(\lambda, \theta) + \beta |E(\lambda, \theta)|^2}$$

where:
- $\mathcal{G}(\pi)$: Expected free energy (agency)
- $C(\lambda)$: Computational capacity (thought)
- $|E(\lambda)|^2$: Criticality constraint (SOC)

**Interpretation:**
- SOC minimizes $|E|^2$ → maintains criticality
- Criticality maximizes $C$ → optimal inference substrate
- Optimal substrate makes $\mathcal{G}$ minimization efficient

### 3.4 GUTC ↔ Active Inference Mapping

| Active Inference | GUTC Realization |
|------------------|------------------|
| Expected free energy $\mathcal{G}$ | Policy objective over $M_L$ branches |
| Extrinsic value | Task reward $\mathcal{L}_{\text{Task}}$ |
| Intrinsic value | Capacity $C(\lambda)$ (information about environment) |
| Policy selection | Heteroclinic branch selection |
| Precision | Inverse noise $1/\sigma$ (dwell time control) |

### 3.5 The Unified Learning Rule

$$\frac{d\theta}{dt} = \eta_{\text{task}} \nabla_\theta \mathcal{L}_{\text{Task}} - \eta_{\text{soc}} \nabla_\theta |E|^2 + \eta_{\text{mem}} \nabla_\theta \mathcal{L}_{\text{Mem}}$$

**Mapping to VFE/EFE:**
- $\nabla_\theta \mathcal{L}_{\text{Task}}$ → Extrinsic gradient (goal-directed)
- $\nabla_\theta |E|^2$ → Criticality maintenance (optimal inference)
- $\nabla_\theta \mathcal{L}_{\text{Mem}}$ → Epistemic structure ($M_L$)

---

## IV. Unified View: GUTC as Dynamical Substrate

### 4.1 The Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                    FUNCTIONAL THEORIES                       │
├─────────────────────────────────────────────────────────────┤
│  Free Energy Principle ← Predictive Coding ← Active Inference│
│       (what to optimize)    (how to compute)   (how to act)  │
└─────────────────────────────────────────────────────────────┘
                              ↓
                     GUTC provides the
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   DYNAMICAL SUBSTRATE                        │
├─────────────────────────────────────────────────────────────┤
│  Criticality (E=0) + M_W (working memory) + M_L (long-term) │
│       (optimal phase)   (capacity)          (structure)      │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 What GUTC Adds

| Framework | What it specifies | What GUTC adds |
|-----------|-------------------|----------------|
| FEP | Objective (minimize $\mathcal{F}$) | **Optimal operating point** ($E = 0$) |
| Predictive Coding | Computation (error minimization) | **Dynamical implementation** (HHN) |
| Active Inference | Action selection | **Memory substrate** ($M_L$ heteroclinic) |

### 4.3 The Core Synthesis

**FEP says:** Minimize free energy.
**GUTC says:** This is most efficiently done at criticality, where Fisher information diverges and inference is maximally sensitive.

**Predictive Coding says:** Hierarchical error minimization.
**GUTC says:** Implemented via Hierarchical Heteroclinic Networks with level-dependent timescales.

**Active Inference says:** Actions minimize expected free energy.
**GUTC says:** Policy operates over heteroclinic memory structure, selecting branches via $M_L$ while maintaining criticality via SOC.

---

## V. Mathematical Equivalences

### 5.1 Free Energy and Capacity

Under certain conditions:

$$\mathcal{F} \approx -C(\lambda) + \text{const}$$

Minimizing $\mathcal{F}$ ↔ Maximizing $C(\lambda)$ ↔ Operating at $E(\lambda) = 0$

### 5.2 Precision and Noise

Predictive coding precision $\Pi$:

$$\Pi \propto \frac{1}{\sigma^2}$$

GUTC dwell time:

$$\tau \propto -\log \sigma \propto \log \sqrt{\Pi}$$

High precision → Long pattern dwelling → Stable inference

### 5.3 Inference Dynamics

**Predictive coding:**
$$\dot{x} = -\nabla_x \mathcal{F}$$

**GUTC at criticality:**
$$\dot{x} = F_\lambda(x) + \sigma \xi(t)$$

with $F_\lambda$ implementing error-driven updates through heteroclinic dynamics.

---

## VI. Hierarchical Control Architecture: AIF + MPC

### 6.1 Two-Level Loop Structure

A powerful pattern emerges: slow AIF for planning, fast execution for control.

```
┌─────────────────────────────────────────────────────────────┐
│              HIGH LEVEL: AIF PLANNER (5-20 Hz)              │
│  • Hierarchical generative model over tasks/goals          │
│  • Minimizes G(π) to select reference trajectories         │
│  • Maintains beliefs q(x_t) via variational inference      │
└─────────────────────────────────────────────────────────────┘
                              ↓ References / Goals
                              ↑ State estimates / Constraint feedback
┌─────────────────────────────────────────────────────────────┐
│             LOW LEVEL: MPC CONTROLLER (100-1000 Hz)         │
│  • Tracks AIF-proposed references                          │
│  • Enforces dynamics, safety constraints                   │
│  • Fast optimization: min ||x - x_ref||² + ||a||²          │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Mapping to GUTC/HHN Architecture

| AIF-MPC Level | GUTC Component | Timescale |
|---------------|----------------|-----------|
| High (AIF planner) | HHN Level L (goals/plans) | Slow ($\tau_L$) |
| Mid (skills/options) | HHN Level 2 (chunks) | Medium ($\tau_2$) |
| Low (MPC execution) | HHN Level 1 (primitives) | Fast ($\tau_1$) |

**Key correspondence:**
- AIF's $\mathcal{G}(\pi)$ minimization → Selection among HHN branches
- MPC reference tracking → Heteroclinic orbit following
- Precision weighting → Noise amplitude $\sigma$ (dwell time control)

### 6.3 The Criticality-Agency Bridge

**Why criticality enables optimal agency:**

| Property | Critical State | Agency Benefit |
|----------|---------------|----------------|
| High $C(\lambda)$ | Long-range correlations | Long-horizon planning |
| $I(\lambda) \to \infty$ | Maximal sensitivity | Efficient exploration |
| Power-law $M_W$ | Extended working memory | Temporal integration |
| Metastable $M_L$ | Structured priors | Goal-directed behavior |

**The critical phase maximizes causal efficacy:** agents can efficiently pursue both exploitation (extrinsic) and exploration (epistemic) simultaneously.

### 6.4 AIT Interpretation of GUTC Components

| GUTC Component | AIT Interpretation |
|----------------|-------------------|
| Thought ($\max C$) | Maximally deep generative model $p(x,u)$ |
| Learning (SOC) | Structural learning $\Delta\theta$ maintaining optimal boundary |
| Agency ($a_t$) | Action as inference: select $a_t$ to minimize future $\mathbb{E}[\mathcal{F}]$ |
| Memory ($M_L$) | Structured priors constraining complexity in $\mathcal{F}$ |

### 6.5 Empirical Validation Directions

| Domain | Active Inference Prediction | GUTC Link |
|--------|---------------------------|-----------|
| Perception | Prediction errors in sensory cortex | Error = deviation from heteroclinic orbit |
| Decision-making | Epistemic foraging matches $\mathcal{G}$ | Branch selection in HHN |
| Motor control | Precision-weighted trajectory updates | Dwell time modulation |
| Psychopathology | Disorders = aberrant precision | Phase errors ($E \neq 0$) |

**Key empirical signatures:**
- Mismatch negativity (auditory) → Precision-weighted error signals
- Bistable perception → Heteroclinic orbit competition
- Parkinson's bradykinesia → Dopaminergic precision deficits → SOC failure

### 6.6 Precision Weighting: The Neurobiological Control Signal

**Precision ($\Pi$)** is the inverse variance of prediction errors, acting as gain control:

$$\Pi = \frac{1}{\sigma^2}$$

**Precision-weighted error update:**
$$\Delta q(x) \propto \Pi \cdot \epsilon, \quad \epsilon = u - \hat{u}$$

High precision → error is reliable → strong belief update.

**Dopamine as Precision Modulator:**

| Dopamine Level | Precision Effect | Cognitive Outcome |
|----------------|------------------|-------------------|
| High | $\Pi \uparrow$ on relevant errors | Efficient learning |
| Low | $\Pi \downarrow$ | Errors treated as noise |
| Aberrant | Misallocated $\Pi$ | Pathological inference |

**Schizophrenia as Aberrant Precision Weighting:**

| Component | Aberrant Weighting | Symptom |
|-----------|-------------------|---------|
| Top-down (priors) | Over-precision ($\Pi_{\text{prior}} \uparrow$) | Delusions (rigid beliefs) |
| Bottom-up (sensory) | Under-precision ($\Pi_{\text{sensory}} \downarrow$) | Hallucinations |
| Self-generated actions | Under-precision ($\Pi_{\text{action}} \downarrow$) | Loss of agency |

**Two-Timescale GUTC Control:**
- **Slow (SOC):** Homeostatic mechanisms adjust $\theta$ to maintain $\rho(W) \approx 1$
- **Fast (Precision):** Neuromodulators adjust $\Pi$ to control inference gain

**The GUTC-Precision Connection:**
$$\Pi \propto \frac{1}{\sigma^2} \implies \tau_{\text{dwell}} \propto -\log \sigma \propto \log \sqrt{\Pi}$$

High precision → low noise → long dwell times → stable inference at criticality.

**ASD as Aberrant Precision Weighting:**

| Component | ASD Pattern | Cognitive Consequence |
|-----------|-------------|----------------------|
| Sensory precision | $\Pi_{\text{sensory}} \uparrow$ (hyper-precise) | Hypersensitivity, detail focus |
| Prior precision | $\Pi_{\text{prior}} \uparrow$ (inflexible) | Rigid predictions, routine preference |
| Global/local ratio | $\Pi_{\text{local}} / \Pi_{\text{global}} \gg 1$ | Weak central coherence |
| Context precision | $\Pi_{\text{context}} \downarrow$ | Difficulty with social inference |

**ASD vs. Schizophrenia Precision Contrast:**

| Feature | ASD | Schizophrenia |
|---------|-----|---------------|
| Sensory precision | $\uparrow$ (too high) | Variable |
| Prior precision | $\uparrow$ (rigid) | $\uparrow$ (delusional) |
| Action precision | Normal | $\downarrow$ (agency loss) |
| Phase state | Subcritical ($E < 0$) | Variable |
| Primary deficit | Over-precision everywhere | Mis-allocated precision |

**GUTC Dynamical Signatures in ASD:**
- Branching ratio $\sigma < 1$ (subcritical)
- Avalanche exponent $\alpha > 1.5$ (steeper, truncated)
- Hurst exponent $H < 0.5$ (reduced temporal integration)
- Correlation length $\xi$ diminished (local processing dominant)

---

## VII. Implications

### 7.1 For Neuroscience

- **Predictive coding** is implemented in **critical cortical dynamics**
- **Precision weighting** corresponds to **noise amplitude** in heteroclinic networks
- **Hierarchical inference** maps to **HHN level structure**

### 7.2 For AI

- Implement **FEP-based agents** with explicit criticality constraints
- Use **heteroclinic memory** for structured world models
- Tune $\rho(W) = 1$ for optimal inference efficiency
- Deploy hierarchical AIF + MPC for embodied systems

### 7.3 For Philosophy

All three frameworks (FEP, PC, AI) describe **what** cognition does.
GUTC describes **how** it's physically realized: critical dynamics.

---

## VIII. Summary Table

| Concept | FEP | Predictive Coding | Active Inference | GUTC |
|---------|-----|-------------------|------------------|------|
| **Objective** | Min $\mathcal{F}$ | Min $\sum \epsilon^2$ | Min $\mathbb{E}[\mathcal{F}]$ | Max $C$ at $E=0$ |
| **Computation** | Variational inference | Error propagation | Policy optimization | Critical dynamics |
| **Memory** | Implicit in $q(x)$ | Hierarchical states | World model | $M_W + M_L$ |
| **Action** | Minimize surprise | Reduce errors | Select policy | Branch selection |
| **Substrate** | Unspecified | Cortical hierarchy | Unspecified | **Critical phase** |

**GUTC provides the missing dynamical substrate for functional theories of mind.**

---

## IX. Precision Weighting: Neural Implementation of $\Pi$ and Links to Autism

Predictive coding and active inference models posit that perception, learning, and action are driven by the minimization of prediction errors that are **precision-weighted**—that is, scaled by their estimated reliability (inverse variance). In this framework, precision $(\Pi)$ acts as a context-dependent **gain control** on prediction-error units, implementing a form of attentional selection and belief updating. Within GUTC, $(\Pi)$ is a key dynamical control parameter that modulates how the critical core uses its capacity: it determines which errors the system allows to drive state changes.

Neuroimaging and pharmacological studies converge on a relatively consistent circuit for precision weighting, involving frontal and cingulate cortex, sensory cortices, and dopaminergic midbrain and striatum, with clear relevance to autism spectrum conditions.

### 9.1 Cortical Circuits for Precision-Weighted Prediction Errors

A series of EEG–fMRI and fMRI–pharmacology studies show that **unsigned prediction errors** (deviations from expected outcomes, regardless of sign) are precision-weighted in a set of frontal and cingulate regions:

* **Superior Frontal Cortex (SFC) and dorsal Anterior Cingulate Cortex (dACC)** encode unsigned prediction errors whose amplitude scales with outcome precision in reward-learning tasks [1]. When outcome distributions are more reliable, unsigned PE signals in these regions increase in magnitude; when outcomes are noisy, the same physical errors are down-weighted.
* **Sensory cortices** (auditory, visual) encode modality-specific prediction errors whose gain is modulated by contextual uncertainty [4]. Under high precision (predictable, reliable environments), sensory PE responses are amplified; under low precision, responses are attenuated.

This fits the canonical hierarchical predictive-coding picture:
* **Lower levels (sensory cortex):** encode prediction errors; their gain reflects locally estimated precision (contextual reliability).
* **Higher levels (SFC, dACC, prefrontal):** encode predictions and set precision via top-down control, effectively tuning how strongly lower-level errors influence belief updates.

In GUTC terms, these circuits implement a hierarchical $(\Pi)$ field over the critical core: local $(\Pi_{\text{sensory}})$ modulates bottom-up drive; higher-order $(\Pi_{\text{prior}})$ modulates the confidence of priors and the threshold for belief revision.

### 9.2 Neuromodulatory Implementation: Dopamine as Precision Gain

Pharmacological manipulations provide causal evidence that **dopamine controls cortical precision weighting**:

* **In reward-learning tasks, pharmacological fMRI provides causal evidence for D2 receptor involvement:** The **D2 antagonist sulpiride (600 mg) selectively reduced the precision weighting** of unsigned PE signals in the **Superior Frontal Cortex (SFC)** (and trend-level effects in dACC) compared to placebo [1]. Conversely, the **D2 agonist bromocriptine** did not substantially alter the precision scaling itself, although it increased the overall main effect of unsigned PE [1]. This indicates that cortical precision weighting in SFC/dACC is critically dependent on D2 receptor signaling, and individual differences in this weighting correlate with learning performance [1].
* Broader dopamine-prediction-error reviews argue that dopamine does more than encode scalar reward PEs: it also modulates the **gain of cortical PE encoding**, effectively implementing a precision parameter in the predictive-coding hierarchy [2].
* Additional work implicates cholinergic and dopaminergic neuromodulation in differentiating precision control at different hierarchical levels: acetylcholine more tied to sensory/attentional precision; dopamine more to volatility and motivational precision [3].

Thus, in the GUTC architecture, dopaminergic (and related neuromodulatory) systems implement the dynamic tuning of $(\Pi)$ that determines how the critical field allocates its finite capacity ($C(\lambda)$): they decide which prediction errors "count," and by how much.

### 9.3 Subcortical Circuits: Midbrain, Striatum, and Salience

Meta-analytic work on prediction-error imaging shows consistent activation patterns in:

* **Midbrain (VTA/SNc) and ventral striatum**, which encode reward and salience prediction errors.
* **Insula and anterior cingulate**, implicated in interoceptive prediction, uncertainty, and global salience mapping.

These regions are often interpreted as computing precision-weighted prediction errors that cut across domains (reward, sensory, cognitive). The midbrain–striatal–cingulate loop can thus be viewed as a **precision-control hub** that:
* Tracks environmental volatility and outcome reliability.
* Adjusts cortical PE gain accordingly (via neuromodulators).
* Coordinates global shifts in attention, learning rate, and exploration.

In GUTC language, this loop is part of the SOC controller over $(\Pi)$: it keeps the system's effective learning rate near the optimal zone given its critical capacity and environmental statistics.

### 9.4 Precision Weighting and Autism: Evidence for $\Pi$ Dysregulation

Autism spectrum conditions (ASD) have been increasingly interpreted through the lens of predictive coding and active inference, not as a failure to form predictions, but as a failure to appropriately **weight prediction errors** [5].

Empirical work using EEG and fMRI (e.g., mismatch negativity, repetition suppression, cue-validity paradigms) shows:
* Altered prediction-error markers (e.g., reduced MMN amplitude, atypical repetition suppression) in temporal and frontal regions, suggesting that PE signals are present but **abnormally modulated** by context and reliability.
* Reduced modulation of anticipatory and late components (e.g., contingent negative variation, sustained potentials) by cue validity in ASD, pointing to **impaired tuning of prediction certainty** rather than absent prediction. The implicated networks include frontal–parietal control regions and sensory cortices that normally adjust precision weights.

Meta-analytic clinical reviews highlight patterns consistent with **hyper- or hypo-precision**:
* **Over-precise priors** in some domains → rigid expectations, difficulty updating beliefs (e.g., social cues).
* **Over-precise sensory likelihoods** → hyper-sensitivity to local detail, sensory overload.
* **Under-precision** in higher-order social or contextual priors → difficulty stabilizing global interpretations.

Within GUTC, these findings are naturally interpreted as dynamical **mis-tuning of $(\Pi)$** on top of the critical core:
* Subcritical / rigid dynamics plus **high prior precision** can yield repetitive, inflexible behavior and resistance to updating (restricted interests, insistence on sameness).
* **Abnormal sensory precision** can explain heightened sensitivity and atypical responses to change or noise.

The clinical instruments (ADOS, ADI-R, SRS, etc.) that quantify social communication and repetitive behaviors can be seen as noisy, high-level readouts of these underlying $(\lambda,\Pi)$ deviations.

### 9.5 Role of Precision in the GUTC Phase Diagram

Bringing this back to the global theory:
* $(\lambda)$ (via $E(\lambda)$) controls how close the system is to the critical surface, setting global capacity and correlation structure [6].
* $(\Pi)$ controls which prediction errors exploit that capacity, acting as a dynamically reconfigurable gain field over the critical substrate.

The neuroimaging and pharmacology data show that:
* Specific cortical regions (SFC, dACC, sensory cortex) and subcortical neuromodulatory systems (dopamine, cholinergic) instantiate $(\Pi)$.
* Precision weighting is behaviorally consequential: individual differences in precision weighting predict learning performance [1].
* Clinical alterations in autism and related conditions are consistent with precision mis-tuning, supporting the GUTC view that many psychiatric phenotypes are phase-and-precision errors in an otherwise critical cognitive field.

This situates precision weighting as a central, biologically grounded control knob in GUTC: **$(\lambda)$ sets the phase; $(\Pi)$ decides which errors get to move the system.**

### Section IX References

| Citation | Reference |
|----------|-----------|
| [1] | Haarsma, J., et al. (2020). Precision weighting of cortical unsigned prediction errors is mediated by dopamine and benefits learning. *bioRxiv/J Neurosci*. |
| [2] | Diederen, K. M. J., & Fletcher, P. C. (2021). Dopamine, prediction error and beyond. *Neuroscientist*, 27(1), 30-46. |
| [3] | Iglesias, S., et al. (2021). Cholinergic and dopaminergic effects on prediction error and uncertainty responses during sensory associative learning. *NeuroImage*, 226, 117590. |
| [4] | Samaha, J., & Postle, B. R. (2017). Correlated individual differences suggest a common mechanism underlying metacognition in visual perception and visual short-term memory. *Proc Biol Sci*, 284(1867). |
| [5] | Van de Cruys, S., et al. (2014). Precise minds in uncertain worlds: Predictive coding in autism. *Psychol Rev*, 121(4), 649-675. |
| [6] | Buzsáki, G., et al. (2024). The brain from inside out. *Oxford University Press* (and related criticality reviews). |

---

## X. Compact GUTC Brain Model

The brain can be modeled as a near-critical, hierarchical inference engine that combines predictive coding, active inference, and self-organized criticality, tuned by precision [7][8].

### 10.1 Core Ingredients

| Component | Description | Reference |
|-----------|-------------|-----------|
| **Critical substrate** | Large-scale cortical networks operate near a critical point, supporting rich, multiscale dynamics and flexible reconfiguration | [8][9] |
| **Hierarchical generative model** | Cortical hierarchies implement predictive processing, with top-down predictions and bottom-up prediction errors exchanging information across levels | [10][7] |
| **Active inference loop** | Perception, learning, and action all minimize (expected) variational free energy, casting behavior as Bayesian model selection under uncertainty | [11][12] |
| **Precision control ($\Pi$)** | Neuromodulatory systems (dopamine, acetylcholine, etc.) set the gain on prediction errors, determining which signals influence updates and policy selection | [13][14] |

In such a model, $\lambda$ (criticality) sets the brain's computational capacity and correlation structure, while $\Pi$ (precision) dynamically selects which errors and hypotheses exploit that capacity, yielding a unified account of perception, cognition, action, and psychopathology [15][8].

### Section X References

| Citation | URL |
|----------|-----|
| [7] | https://www.sciencedirect.com/science/article/abs/pii/S0149763423004426 |
| [8] | https://pmc.ncbi.nlm.nih.gov/articles/PMC4171833/ |
| [9] | https://direct.mit.edu/netn/article/6/4/1148/112392/Theoretical-foundations-of-studying-criticality-in |
| [10] | https://pmc.ncbi.nlm.nih.gov/articles/PMC5836998/ |
| [11] | https://onlinelibrary.wiley.com/doi/10.1111/tops.12704 |
| [12] | https://www.biorxiv.org/content/10.1101/2020.02.11.944611v1.full |
| [13] | https://onlinelibrary.wiley.com/doi/10.1111/pcn.13138 |
| [14] | https://pmc.ncbi.nlm.nih.gov/articles/PMC7804370/ |
| [15] | https://pmc.ncbi.nlm.nih.gov/articles/PMC9336647/ |

---

## XI. Laminar Microcircuit Implementation

Model layer-specific connectivity by constraining your network to a canonical cortical microcircuit rather than a generic layered MLP [16][17].

### 11.1 Start from the Canonical Microcircuit

Implement separate excitatory and inhibitory populations for each lamina (e.g., L2/3, L4, L5, L6), with connectivity patterned after anatomical data:

* Strong **L4 → L2/3** feedforward excitation (sensory input to superficial prediction-error units) [18][16]
* Strong **L2/3 ↔ L5** bidirectional connections, with L2/3 → L5 the dominant interlaminar drive and L5 sending integrated output to subcortical and other cortical areas [19][16]
* Recurrent (intralaminar) excitation within L2/3, L5, and L6, plus local inhibition in each layer [17][16]

This gives a minimal **8-population model**: E/I in L2/3, L4, L5, L6 [17].

### 11.2 Functional Roles of Layers

Tie connectivity to predictive-coding roles:

| Layer | Role | Connectivity Pattern |
|-------|------|---------------------|
| **L4** | Primary thalamic/feedforward input layer; drives initial sensory representations and errors | Receives thalamic input [16][18] |
| **L2/3** | Superficial "prediction-error" microcircuit integrating feedforward input with top-down predictions; projects forward to higher areas | FF output [20][21] |
| **L5** | Deep "representation/prediction" microcircuit integrating across layers and sending descending and subcortical outputs | Subcortical/cortical output [22][16] |
| **L6** | Feedback and gain-control layer, modulating thalamic input and intra-column excitability | Thalamic modulation [16][17] |

In code or equations, make sure your weight matrices respect these asymmetries (e.g., no arbitrary L2/3 → L4 feedback, sparse L4 recurrent connections, strong L2/3→L5 and L5→L2/3).

### 11.3 Extrinsic (Between-Area) Connectivity

For a hierarchy of areas:

| Pathway | Origin | Termination | Function |
|---------|--------|-------------|----------|
| **Feedforward (FF)** | Superficial layers (L2/3) of lower areas | Mainly L4 of higher areas | Carries errors up [23][22] |
| **Feedback (FB)** | Deep layers (L5/6) | L1/L6 and apical dendrites of L2/3 and L5 in lower areas | Carries predictions down [23][22] |

This gives biologically grounded FF/FB pathways: FF carries errors up; FB carries predictions down.

### 11.4 Existing Laminar Circuit Formalisms

To keep things tractable, adopt or adapt established laminar models:

* **Canonical microcircuit neural mass models** (as used in Dynamic Causal Modelling) that explicitly parameterize intrinsic connections among L2/3, L4, L5, L6 and can be fit to data [24][25]
* **Predictive-coding laminar models** that place prediction-error units in L2/3 and representation units in L4/5, with recurrent loops between them [20][16]

You can then add neuromodulatory "precision" parameters that scale specific layer gains (e.g., ACh on L4/L2/3 sensory errors, DA on L2/3–L5 error/prediction loops), without changing the anatomical scaffold [26][27].

**Summary:** Represent each lamina with its own excitatory/inhibitory populations, wire them according to canonical microcircuit data (L4→L2/3→L5 with strong intra-laminar recurrence), and route feedforward/feedback signals through the appropriate layers and dendritic compartments.

### Section XI References

| Citation | URL |
|----------|-----|
| [16] | https://pmc.ncbi.nlm.nih.gov/articles/PMC3777738/ |
| [17] | https://pmc.ncbi.nlm.nih.gov/articles/PMC3920768/ |
| [18] | https://elifesciences.org/articles/71103 |
| [19] | https://www.frontiersin.org/journals/neuroanatomy/articles/10.3389/fnana.2014.00165/full |
| [20] | https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1013469 |
| [21] | https://www.pnas.org/doi/10.1073/pnas.2014868117 |
| [22] | https://www.frontiersin.org/journals/neuroanatomy/articles/10.3389/fnana.2017.00071/full |
| [23] | https://elifesciences.org/articles/59551 |
| [24] | https://pmc.ncbi.nlm.nih.gov/articles/PMC6693530/ |
| [25] | https://pubmed.ncbi.nlm.nih.gov/27871922/ |
| [26] | https://www.sciencedirect.com/science/article/pii/S1053811920310752 |
| [27] | https://pmc.ncbi.nlm.nih.gov/articles/PMC10942646/ |

---

## XII. Formal Model Definition: Biologically Constrained Neural Mass

The network is modeled as a system of coupled differential equations based on a **Neural Mass Model** or a **Rate-Unit Model**, where the state variables represent the average firing rates of defined cell populations. The system is defined by $\mathbf{x} = [x_1, x_2, \dots, x_N]$, where $x_i$ is the firing rate of the $i$-th population, and $N$ is the total number of populations.

### 12.1 Laminar Populations as State Variables

Each lamina ($L_{k} \in \{L2/3, L4, L5, L6\}$) contains at least one Excitatory (E) and one Inhibitory (I) population. The generic dynamics for any population $i$ are:

$$\tau \dot{x}_i = -x_i + f\left(\sum_{j} W_{ij} x_j + I_i^{\text{ext}} + I_i^{\text{mod}}\right)$$

Where:
* $x_i$: Firing rate/activity of population $i$
* $\tau$: Time constant
* $f(\cdot)$: Non-linear activation function (e.g., sigmoid/logistic function, or a thresholded linear function)
* $W_{ij}$: The intrinsic (intracolumnar) connectivity weight matrix, strictly constrained by anatomical data (e.g., $W_{\text{L4E} \to \text{L2/3E}}$ is strong, $W_{\text{L2/3E} \to \text{L4E}}$ is near zero)
* $I_i^{\text{ext}}$: External/extrinsic input, comprising feedforward (FF) and feedback (FB) signals, routed through appropriate layers (FF → L4; FB → L1/L6 and apical dendrites)
* $I_i^{\text{mod}}$: Neuromodulatory input, which acts as the precision gain (constrained in Section 12.3)

### 12.2 Cell Types as Sub-Populations

The generic Inhibitory (I) population in each layer $L_k$ is expanded into three sub-populations, yielding a system of equations for the key cell types:

$$\mathbf{x}_{L_k} = \left[x_{L_k}^{\text{Pyr}}, x_{L_k}^{\text{PV}}, x_{L_k}^{\text{SOM}}, x_{L_k}^{\text{VIP}}\right]$$

The connectivity matrix $W$ is further constrained by interneuron roles:
* $W$ reflects **PV's** broad inhibition on $x^{\text{Pyr}}$ and $x^{\text{PV}}$ populations for stability
* $W$ reflects **VIP's** selective inhibition of $x^{\text{SOM}}$ ($\text{VIP} \to \text{SOM}$)

**Precision/Gating Implementation via SOM:**

The crucial role of **SOM** interneurons in gating top-down predictions is implemented by having their activity modulate the effective top-down input to the Pyramidal units (L2/3 and L5). The effective input, $I^{\text{eff}}$, is a function of the total received top-down input, $I^{\text{top-down}}$, and the activity of the SOM population, $x^{\text{SOM}}$:

$$I^{\text{top-down}}_{\text{eff}} = I^{\text{top-down}} - g_{\text{SOM}}(\mathbf{\Pi}) \cdot x^{\text{SOM}}$$

Where the gain $g_{\text{SOM}}(\mathbf{\Pi})$ may itself be modulated by neuromodulators (as detailed below), linking precision to the cell-type specific inhibition.

### 12.3 Neuromodulators as Precision Gains ($\Pi$)

The abstract precision parameter $\mathbf{\Pi}$ is implemented as a set of neuromodulatory-dependent gain factors that modulate the input $I^{\text{mod}}$ or the function $f(\cdot)$ across specific layers.

| Parameter | Neuromodulator | Mechanism | Application (Layer/Circuit) |
|-----------|----------------|-----------|----------------------------|
| $\Pi_{\text{sensory}}$ | **Acetylcholine (ACh)** | **Multiplicative Gain:** Scales the input gain or the slope of the activation function $f(\cdot)$ (e.g., $\Pi_{\text{sensory}} \cdot f(\sum W x + I)$) in sensory layers, sharpening the PE signal | **L4/L2/3 PE populations** (Sensory Cortex), and on the SOM/VIP circuits to control their excitability |
| $\Pi_{\text{prior}}$ | **Dopamine (DA, D2)** | **Multiplicative/Subtractive Gain:** Scales the gain of PE units, potentially by biasing the PV/SOM balance in favor of reduced SOM inhibition (lower precision) | **SFC/dACC** (Frontal/Cingulate), controlling the precision of higher-order priors and volatility estimates |

This formal structure ensures that the system's dynamics (12.1), internal inhibition (12.2), and global modulations (12.3) are all biologically anchored, making the study of the $(\lambda, \Pi)$ control parameters highly interpretable.

### 12.4 Compact Model Definition

The cortical microcircuit is modeled as a set of coupled neural-mass units, one for each excitatory and inhibitory population in layers L2/3, L4, L5, and L6 (and their cell-type-specific subpopulations), with dynamics:

$$\tau \dot{x}_i = -x_i + f\!\left(\sum_j W_{ij} x_j + I_i\right)$$

where the connectivity matrix $W_{ij}$ is constrained by canonical laminar anatomy (strong L4→L2/3, L2/3↔L5, L6→L4; laminar-specific feedforward and feedback projections). Within each layer, inhibitory populations are subdivided into PV, SOM, and VIP interneurons with cell-type-specific connectivity motifs, allowing SOM- and VIP-mediated modulation of effective top-down input to pyramidal cells:

$$I_{\text{top-down}}^{\text{eff}} = g_{\text{SOM}}(\Pi)\,I_{\text{top-down}}$$

to implement precision-dependent gating. Finally, neuromodulators instantiate abstract precision parameters as gain factors on selected populations: acetylcholine defines a sensory precision $\Pi_{\text{sensory}}$ by scaling prediction-error-related inputs and transfer functions in L4/L2/3 and associated interneurons, while dopamine (D2) defines a prior/volatility precision $\Pi_{\text{prior}}$ by scaling PE gain in frontal (SFC/dACC-like) populations and their PV/SOM circuits, in line with pharmacological fMRI evidence for DA-dependent precision weighting [28][29][30][31][32][33][34].

### Section XII References

| Citation | URL |
|----------|-----|
| [28] | https://pmc.ncbi.nlm.nih.gov/articles/PMC3777738/ |
| [29] | https://pmc.ncbi.nlm.nih.gov/articles/PMC3832795/ |
| [30] | https://pmc.ncbi.nlm.nih.gov/articles/PMC7442488/ |
| [31] | https://pmc.ncbi.nlm.nih.gov/articles/PMC4469730/ |
| [32] | https://pmc.ncbi.nlm.nih.gov/articles/PMC10942646/ |
| [33] | https://www.sciencedirect.com/science/article/pii/S1053811920310752 |
| [34] | https://pmc.ncbi.nlm.nih.gov/articles/PMC8589669/ |

---

## XIII. Formal Model: Biologically Constrained Laminar Neural Mass (Methods)

The microcircuit is modeled as a set of coupled neural-mass units governed by first-order ordinary differential equations (ODEs), which approximates the average activity of specific cell populations. The model integrates constraints across laminar, cellular, and neuromodulatory scales.

### 13.1 Populations and State Variables

For each cortical column (or area), the model includes 16 populations derived from 4 cortical layers ($\ell \in \{\text{L4}, \text{L2/3}, \text{L5}, \text{L6}\}$) and 4 cell types ($c \in \{\text{Pyr}, \text{PV}, \text{SOM}, \text{VIP}\}$).

The state of the system is a vector of mean firing rates $\mathbf{r}(t)$ and total synaptic input currents $\mathbf{u}(t)$:

$$\mathbf{r}(t) = \big(r_{\text{L4,Pyr}}, r_{\text{L4,PV}}, \dots, r_{\text{L6,VIP}}\big)^\top$$

The dynamics for the firing rate $r_{\ell,c}(t)$ of any population $(\ell,c)$ are given by:

$$\tau_{\ell,c} \frac{dr_{\ell,c}}{dt} = - r_{\ell,c}(t) + \phi\big(u_{\ell,c}(t)\big)$$

The input current $u_{\ell,c}(t)$ is defined as the weighted sum of activity from all populations plus external and neuromodulatory inputs:

$$u_{\ell,c}(t) = \sum_{\ell',c'} W_{(\ell,c),(\ell',c')}\, r_{\ell',c'}(t) + I_{\ell,c}^{\text{ext}}(t)$$

### 13.2 Connectivity Matrix $W$ and Intrinsic Constraints

The $16 \times 16$ connectivity matrix $W$ is fixed by three levels of biological constraints:

#### A. Laminar Backbone (Excitatory Connections)

The pyramidal (Pyr) connectivity enforces the canonical cortical backbone, primarily controlling **feedforward (FF) → error** and **feedback (FB) → prediction** loops:

* **FF Drive:** Strong $W_{(\text{L2/3,Pyr}),(\text{L4,Pyr})} = w_{42}$ (L4 → L2/3)
* **Prediction Loop:** Strong $W_{(\text{L5,Pyr}),(\text{L2/3,Pyr})} = w_{52}$ (Error → Prediction) and $W_{(\text{L2/3,Pyr}),(\text{L5,Pyr})} = w_{25}$ (Prediction → Error)
* **Recurrence:** Strong $W_{(\ell,\text{Pyr}),(\ell,\text{Pyr})} = w_{\ell\ell}^{\text{EE}}$ for $\ell \in \{\text{L2/3},\text{L5},\text{L6}\}$, with L4 recurrence set near zero

#### B. Inhibitory Subclasses (PV, SOM, VIP)

Inhibitory connectivity reflects cell-type-specific functional motifs:

* **PV (Stability):** Provides broad, fast, perisomatic inhibition, $W_{(\ell,\text{PV}),(\ell,\text{Pyr})} < 0$, ensuring stability and balancing local excitation
* **VIP (Disinhibition):** Inhibits SOM populations, $W_{(\ell,\text{SOM}),(\ell,\text{VIP})} < 0$, providing a mechanism for contextual disinhibition
* **SOM (Precision Gating):** Provides dendritic inhibition, modeled as a **gating function** on the top-down/feedback channel $u_{\ell,\text{FB}}$ of the pyramidal input $u_{\ell,\text{Pyr}}$:

$$\mathbf{u}_{\ell,\text{Pyr}} = u_{\ell,\text{FF}} + g_{\ell}^{\text{FB}}(t) \, u_{\ell,\text{FB}} + u_{\ell,\text{rec}} + u_{\ell,\text{inh}}$$

Where the SOM-mediated gain $g_{\ell}^{\text{FB}}$ controls the influence of top-down predictions:

$$g_{\ell}^{\text{FB}}(t) = g_{\ell,0}^{\text{FB}} \cdot \exp\big(-\kappa_{\ell}^{\text{SOM}} \, r_{\ell,\text{SOM}}(t)\big)$$

### 13.3 Criticality ($\lambda$) and Precision ($\Pi$) Mapping

The core GUTC control parameters are integrated as global coupling strength ($\lambda$) and local, neuromodulator-dependent gains ($\Pi$).

#### A. Global Criticality ($\lambda$)

The criticality parameter $\lambda$ sets the global phase of the system by scaling the strength of all recurrent and long-range coupling:

$$u_{\ell,c}(t) = \mathbf{\lambda} \sum_{\ell',c'} W_{(\ell,c),(\ell',c')}^{\text{recurrent}} r_{\ell',c'}(t) + \dots$$

(where $W^{\text{recurrent}}$ includes all $W_{ij}$ elements that represent recurrent or long-range coupling).

#### B. Local Precision ($\Pi$) as Neuromodulatory Gain

The abstract precision parameters $\Pi$ are implemented as multiplicative gains on the intrinsic neuronal gain $a$ of the activation function $\phi(u) = 1 / (1 + \exp(-a(u - \theta)))$.

| Parameter | Neuromodulator | Target Location | Modulated Mechanism |
|-----------|----------------|-----------------|---------------------|
| $\Pi_{\text{sensory}}$ | **Acetylcholine (ACh)** | L4/L2/3 (Sensory areas) | Multiplies the intrinsic neuronal gain: $a \to \mathbf{\Pi}_{\text{sensory}} \cdot a$. Also scales thalamic input $I_{\text{L4,Pyr}}^{\text{th}}(t) \to \mathbf{\Pi}_{\text{sensory}} \cdot s_{\text{th}}(t)$ |
| $\Pi_{\text{prior}}$ | **Dopamine (DA, D2)** | SFC/dACC-like (Frontal areas) | Multiplies the intrinsic neuronal gain: $a \to \mathbf{\Pi}_{\text{prior}} \cdot a$. Also controls SOM-gating strength $\kappa_{\ell}^{\text{SOM}}$ to regulate the L2/3 ↔ L5 loop |

In this model, $\lambda$ determines the available computational capacity, and $\Pi$ determines how that capacity is allocated by weighting prediction errors and predictions.

---

## XIV. Mapping the $(\lambda, \Pi)$ Control Space to Psychopathology

The Global Unified Theory of Cognition (GUTC) posits that many psychiatric phenotypes arise from dysregulation in the joint space defined by the global criticality parameter ($\lambda$) and the local precision parameter ($\Pi$). $\lambda$ controls global system capacity, while $\Pi$ controls information allocation.

### 14.1 Four Archetypal Dynamical Regimes

By combining the two-dimensional control space, we hypothesize four archetypal dynamical regimes, each potentially corresponding to a cluster of clinical symptoms:

| Regime | $\lambda$ (Criticality) | $\Pi$ (Precision) | Dynamics and Capacity | Predicted Clinical Phenotype |
|--------|-------------------------|-------------------|----------------------|------------------------------|
| **I** | **Subcritical ($\lambda < 1$)** | **High $\Pi_{\text{prior}}$ (Rigid)** | Low Capacity, Low Variability, **High Signal/Noise**. System is rigid; resists change. PEs are strongly weighted but rapidly self-terminating. | **Autism Spectrum Disorder (ASD):** Restricted, repetitive behaviors; insistence on sameness (rigid priors/dynamics); hyper-focus on local details (high $\Pi_{\text{sensory}}$) |
| **II** | **Supercritical ($\lambda > 1$)** | **Low $\Pi_{\text{prior}}$ (Chaotic)** | High Capacity, High Variability, **Low Signal/Noise**. System is unstable; prone to cascade and over-generalization. | **Psychosis (Acute Schizophrenia/Mania):** Chaotic thought patterns; loose associations; over-interpretation of noise as signal (low $\Pi_{\text{prior}}$ allows insignificant PEs to drive widespread change) |
| **III** | **Subcritical ($\lambda < 1$)** | **Low $\Pi_{\text{prior}}$ (Hypo-Plastic)** | Low Capacity, Low Engagement. System is suppressed; learning is blunted because PEs are ignored. | **Apathy/Anhedonia (Severe Depression):** Reduced motivational salience (low $\Pi_{\text{prior}}$ on reward PE); diminished response to environment; cognitive rigidity |
| **IV** | **Supercritical ($\lambda > 1$)** | **High $\Pi_{\text{prior}}$ (Hyper-Plastic)** | High Capacity, High Instability. Learning is too fast; representations are unstable and constantly overwritten. | **Anxiety/Borderline Features:** Heightened emotional volatility; rapid, unstable social interpretations (hyper-plasticity of emotional priors); difficulty stabilizing identity |

### 14.2 Subcritical + High Precision (Regime I: ASD)

* **Dynamical regime:** $\lambda < 1$ (subcritical), reduced flexibility and fewer transitions between brain states; empirically, autism shows reduced state switching and signs of subcritical or overly stable dynamics in some resting-state and E–I balance measures [35][36][37]
* **Precision pattern:** High and often inflexible precision on sensory prediction errors and/or low-level priors ($\Pi_{\text{sensory}}$ high), as in aberrant-precision accounts of autism where sensory channels are "turned up" and priors are not flexibly adjusted [38][39]
* **Phenotypic mapping:** Restricted, repetitive behaviors; insistence on sameness (rigid priors/dynamics); hyper-focus on local details; sensory hypersensitivity [37][40][38]

### 14.3 Supercritical + Low Precision (Regime II: Psychosis)

* **Dynamical regime:** $\lambda > 1$ (supercritical), overly easy propagation and amplification of activity; schizophrenia and psychosis show altered criticality, abnormal complexity, and deviations from healthy critical scaling [41][42][43]
* **Precision pattern:** Low or misallocated precision on appropriate priors or sensory channels ($\Pi$ too low or assigned to the wrong level), so that random fluctuations are over-interpreted and belief updating is unstable—central to predictive-coding accounts of psychosis [44][45][46]
* **Phenotypic mapping:** Delusions, hallucinations, and aberrant salience can be modeled as supercritical dynamics plus insufficiently constrained precision, allowing noisy PEs to drive runaway belief updates or giving undue weight to aberrant priors [47][48][46]

### 14.4 Clinical Implications

In GUTC language: **ASD** corresponds primarily to "subcritical + high, rigid $\Pi$" sectors of the $(\lambda,\Pi)$ plane, whereas **psychosis-spectrum disorders** occupy "near-/supercritical + low or misallocated $\Pi$" regions, with healthy cognition near $\lambda \approx 1$ and context-sensitive, well-calibrated precision [43][46][38].

**Therapeutic Target:** Clinical interventions should aim to restore the appropriate **balance** of $(\lambda, \Pi)$ to return the system to the optimal critical-balance point.

### Section XIV References

| Citation | URL |
|----------|-----|
| [35] | https://www.nature.com/articles/s41598-020-65500-4 |
| [36] | https://pmc.ncbi.nlm.nih.gov/articles/PMC5504272/ |
| [37] | https://direct.mit.edu/netn/article/5/2/295/97536/Atypical-core-periphery-brain-dynamics-in-autism |
| [38] | https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2014.00302/full |
| [39] | https://sandervandecruys.be/pdf/2014-VandeCruysetal-PsychRev-Precise_minds.pdf |
| [40] | https://journal.psych.ac.cn/xlkxjz/EN/10.3724/SP.J.1042.2024.00813 |
| [41] | https://pmc.ncbi.nlm.nih.gov/articles/PMC8995790/ |
| [42] | https://www.sciencedirect.com/science/article/abs/pii/S0920996421003510 |
| [43] | https://pmc.ncbi.nlm.nih.gov/articles/PMC7479292/ |
| [44] | https://pmc.ncbi.nlm.nih.gov/articles/PMC5424073/ |
| [45] | https://www.annualreviews.org/content/journals/10.1146/annurev-neuro-100223-121214 |
| [46] | https://pmc.ncbi.nlm.nih.gov/articles/PMC6169400/ |
| [47] | https://www.nature.com/articles/s41537-025-00643-9 |
| [48] | https://www.frontiersin.org/journals/psychiatry/articles/10.3389/fpsyt.2013.00047/full |

---

## XV. Methods: Estimating the Branching Ratio ($\hat{m}$) from EEG/MEG Avalanches

We estimate the criticality parameter $\lambda$ of the cortical dynamics by calculating the neuronal branching ratio ($\hat{m}$) from EEG/MEG data. This involves converting the continuous signal into binary spatiotemporal "avalanches" and measuring the propagation factor of activity.

### 15.1 Preprocessing and Binarization

Continuous EEG/MEG data are first filtered and segmented into time series for each sensor (or source-localized region).

* **Time Bin Selection ($\Delta t$):** We choose a time bin $\Delta t$ on the order of the dominant oscillatory timescale or the mean inter-event interval.
* **Binarization:** For each channel ($c$) and time bin ($t$), the signal is thresholded (e.g., amplitude above a Z-score threshold) to obtain a binary activity matrix:

$$B(c,t) \in \{0,1\}$$

where $B(c,t)=1$ denotes an "event" (suprathreshold activation).

* **Activity per Bin:** The total number of active channels per bin is defined as:

$$A_t = \sum_c B(c,t)$$

### 15.2 Neuronal Avalanche Definition

Neuronal avalanches are defined as contiguous runs of non-empty bins ($A_t>0$), bounded immediately before and after by empty bins ($A_t=0$).

* **Activity Vector:** For each avalanche, we record the number of active channels in each bin as a vector of activity $n$:

$$(n_1, n_2, \dots, n_T), \quad n_\tau = A_{t_0+\tau-1}$$

where $t_0$ is the onset bin and $T$ is the avalanche duration in bins.

### 15.3 Branching Ratio Estimators

The branching ratio $\hat{m}$ quantifies the average number of "offspring" events in bin $(t+1)$ arising from "parent" events in bin $(t)$.

#### A. Simple Two-Bin Estimator (Beggs–Plenz Style)

For each avalanche $k$ with duration $T\ge2$, the propagation ratio $r^{(k)}$ is computed as the ratio of activity in the second bin to the first:

$$r^{(k)} = \frac{n^{(k)}_2}{n^{(k)}_1}$$

The branching ratio is then estimated by averaging over all $N_{\text{av}}$ avalanches:

$$\hat{m} = \frac{1}{N_{\text{av}}} \sum_{k=1}^{N_{\text{av}}} r^{(k)}$$

#### B. Geometric / Bin-Wise Estimator

To mitigate bias from the variability of $n_1$ and $n_2$ across avalanches, we also use a bin-wise estimator based on all valid successive bin pairs across all avalanches:

$$R = \left\{ \frac{n_{t+1}}{n_t} \mid n_t>0, \text{ all avalanches and bins} \right\}$$

The geometric mean branching ratio is calculated as:

$$\hat{m}_{\text{geo}} = \exp\left( \frac{1}{|R|} \sum_{r \in R} \ln r \right)$$

**Interpretation:**
| $\hat{m}$ Value | Dynamical Regime | Interpretation |
|-----------------|------------------|----------------|
| $\hat{m} \approx 1$ | **Near-critical** | Optimal computational capacity |
| $\hat{m} < 1$ | **Subcritical** | Activity tends to die out |
| $\hat{m} > 1$ | **Supercritical** | Activity tends to grow/cascade |

### 15.4 Practical Considerations

Threshold and bin width are carefully selected by scanning a reasonable range of $\Delta t$ and Z-thresholds to ensure the production of non-trivial avalanche statistics and, ideally, yield scale-free avalanche size distributions over at least 1–2 decades. We report $\hat{m}$ alongside avalanche size exponents and temporal correlation measures for a robust criticality assessment.

### 15.5 Compact Algorithm for Implementation

For direct implementation, the procedure can be summarized as:

**Step 1: Compute Binary Events**
```
For each channel c:
    Bin time into steps of Δt
    Threshold per bin → B(c,t) ∈ {0,1}
    Let A_t = Σ_c B(c,t)
```

**Step 2: Identify Avalanches**
```
Scan A_t:
    Start new avalanche when A_t > 0 AND A_{t-1} = 0
    End avalanche when A_t = 0
    For each avalanche, store (n_1, ..., n_T)
```

**Step 3: Estimate Branching Ratio**
```
Simple estimator:
    m_hat = mean over avalanches of (n_2 / n_1) for T ≥ 2

Geometric estimator:
    Collect all pairs (n_t, n_{t+1}) with n_t > 0
    Compute ratios r = n_{t+1} / n_t
    m_hat_geo = exp(mean(ln(r)))
```

### 15.6 Bridge to GUTC Parameters

This methodology completes the bridge between empirical EEG/MEG data and the model's criticality parameter:

$$\hat{\lambda} \approx \hat{m}$$

$$E(\lambda) \approx \hat{m} - 1$$

The estimated branching ratio thus directly informs the global capacity $C(\lambda)$ of the GUTC model, enabling empirical validation of the theoretical predictions in Sections XIII and XIV.

---

## XVI. Synthesis and Outlook: The GUTC Control Manifold

The **Grand Unified Theory of Cognition (GUTC)** proposes that the brain is a near-critical, precision-tuned inference machine. Cognition and psychopathology can thus be understood in terms of where a system sits—and how it moves—in a low-dimensional control space spanned by **global criticality** ($\lambda$) and **local precision fields** ($\Pi$).

### 16.1 Core Theoretical Synthesis

The integration of the model's component constraints points to three powerful unifying ideas:

#### A. Criticality as a Computational Setpoint

Empirical and modeling work increasingly suggests that healthy brains operate close to a critical point ($\lambda \approx 1$), maximizing information capacity, flexibility, and sensitivity to input (cf. Section IV) [49]. Our $\lambda$ parameter and the branching-ratio pipeline (Section XV) provide a concrete estimator and a way to track these essential deviations in individuals.

#### B. Predictive Coding as the Algorithm on that Substrate

Predictive processing/active inference provides the principled algorithmic structure: hierarchical generative models updated by **precision-weighted prediction errors** (cf. Sections VIII–IX) [50]. Our $\Pi$ fields, implemented via neuromodulators (ACh/DA) acting on specific laminar microcircuits (Section XI), turn this into a physically instantiated *gain map* over the critical core, determining **which** prediction errors drive state updates and by how much.

#### C. Computational Psychiatry as $(\lambda, \Pi)$-Mis-tuning

Computational psychiatry already frames many disorders as failures of uncertainty/precision encoding and hierarchical inference (cf. Section VI) [51]. The $(\lambda, \Pi)$ phase diagram (Section XIV) makes this link explicit: conditions such as ASD and psychosis correspond to **characteristic regions and trajectories** in this low-dimensional space, rather than isolated, mechanistically unrelated entities.

### 16.2 Generative Theoretical Proposals

This synthesis yields several generative hypotheses and research programs:

#### A. ConCrit–GUTC Link

Conscious, flexible cognition arises only when $\lambda$ is near-critical and $\Pi$ is adaptively regulated across the hierarchy. Unconscious, rigid, or psychotic states correspond to specific **deformations of this critical-precision manifold** [52]. This suggests a single continuous manifold for both "normal" and "abnormal" conscious states, rather than a categorical split.

#### B. Low-Dimensional Control Manifold

Despite enormous microscopic complexity, individual brains may live on a low-dimensional manifold parameterized by a few effective control variables (e.g., $\lambda$, $\Pi_{\text{sensory}}$, $\Pi_{\text{prior}}$) [53]. This predicts that large-scale neural recordings combined with behavioral/clinical fits should reveal **clusters and trajectories** in this control space that systematically track development, learning, and illness trajectories.

#### C. Cross-Species, Cross-Substrate Law

Because criticality and precision-weighted inference are substrate-agnostic principles, the same $(\lambda, \Pi)$ control picture should apply to **artificial agents, animal models, and humans** [54]. Similar regions of this control space should correspond to similar functional regimes (e.g., exploration–exploitation balance, robustness vs. flexibility), regardless of the underlying implementation.

### 16.3 Research Program

These directions define a concrete research program:

$$\boxed{\text{Estimate } \hat{\lambda} \text{ from avalanches, fit } \hat{\Pi} \text{ from behavior, test predictions in } (\lambda, \Pi) \text{-space}}$$

| Step | Method | Outcome |
|------|--------|---------|
| **1. Estimate $\hat{\lambda}$** | Branching ratio from EEG/MEG avalanches (Section XV) | Individual criticality parameter |
| **2. Fit $\hat{\Pi}$** | Behavioral modeling (HGF, active inference) | Precision allocation profile |
| **3. Map to $(\lambda, \Pi)$-space** | Joint parameter estimation | Position on control manifold |
| **4. Test predictions** | Correlate with clinical scores (ADOS, PANSS) and cognitive performance | Validate phase-diagram mapping |

### 16.4 The $(\lambda, \Pi)$ Phase Diagram

#### Figure 1: The $(\lambda, \Pi)$ Control Manifold — Cognitive Regimes as Dynamical Phases

**Manuscript-Ready Caption:**

> The horizontal axis spans global criticality ($\lambda$), from subcritical ($\lambda < 1$) through the critical line ($\lambda = 1$) to supercritical ($\lambda > 1$). The vertical axis indexes effective precision ($\Pi$), reflecting the relative weighting of sensory evidence ($\Pi_{\text{sensory}}$) vs. hierarchical priors ($\Pi_{\text{prior}}$). Regions correspond to distinct cognitive regimes: healthy cognition clusters near the critical corridor at moderate $\Pi$; autistic-like states occupy subcritical zones with elevated sensory precision; psychotic-like dynamics emerge in supercritical, prior-weighted territory; anesthetic/unconscious states collapse to low-$\lambda$, low-$\Pi$ quiescence; manic/chaotic regimes flare into hypercritical instability. Example trajectories illustrate developmental maturation (subcritical → critical), acute psychotic episodes (excursions into supercritical high-$\Pi_{\text{prior}}$), and pharmacological recovery (return to critical corridor).

**ASCII Layout (Quadrant View):**

```
                         High Π (Rigid/Hyper-precise)
                                    ▲
                                    │
           ┌────────────────────────┼────────────────────────┐
           │                        │                        │
           │   REGIME I             │        REGIME IV       │
           │   ASD-like             │        Anxiety/        │
           │   • Subcritical        │        Borderline      │
           │   • High Π_sensory     │        • Supercritical │
           │   • Rigid priors       │        • Hyper-plastic │
           │                        │                        │
   λ < 1 ◄─┼────────────────────────┼────────────────────────┼─► λ > 1
 Subcritical                        │                      Supercritical
           │                   ★ HEALTHY ★                   │
           │                   (λ ≈ 1, Π calibrated)         │
           │   REGIME III           │        REGIME II       │
           │   Depression/          │        Psychosis       │
           │   Anhedonia            │        • Supercritical │
           │   • Subcritical        │        • Low Π_prior   │
           │   • Low Π_prior        │        • Chaotic       │
           │                        │                        │
           └────────────────────────┼────────────────────────┘
                                    │
                                    ▼
                         Low Π (Chaotic/Hypo-precise)
```

**ASCII Layout (Continuous View for Illustrator):**

```
                    ↑  Precision (Π)
                    │  ↑ High Π_prior (strong priors)
                    │         Psychotic-like *
                    │                *
                    │                   *
                    │                      *
                    │
                    │
           ---------*--------------------------→  Criticality (λ)
                    λ < 1          λ = 1          λ > 1
                (subcritical)       |          (supercritical)
                    │               |
                    │               |
   ↑ High Π_sensory │   Healthy / near-critical
     (strong input) │      cognitive corridor *
                    │             *
                    │          *
                    │       *
                    │
                    │
    ↓  Low Π (weak precision)
   Deeply subcritical *         Hypercritical *
   (anesthetic/unconscious)     (manic/chaotic)
```

#### Illustrator Blueprint

| Element | Specification |
|---------|---------------|
| **Style** | 2D phase diagram with color-coded regions |
| **Colors** | Blue = healthy, Red = psychotic, Orange = ASD, Gray = anesthetic, Purple = manic |
| **Axes** | Thick critical line at $\lambda = 1$ (vertical dashed band); gradient shading for $\Pi$ transitions |
| **Font** | Sans-serif matching manuscript body |

**Trajectory Arrows:**
| # | Trajectory | Path |
|---|------------|------|
| 1 | **Development** | Lower-left → center (subcritical/low-$\Pi$ → critical/moderate-$\Pi$) |
| 2 | **Psychosis episode** | Center → upper-right → back to center |
| 3 | **Medication** | Upper-right → center (decreasing $\Pi_{\text{prior}}$) |
| 4 | **Meditation/mindfulness** | Fine-tuning toward optimal center |

#### Legend Paragraph

This low-dimensional control manifold unifies healthy and pathological cognition as positions and trajectories in a shared dynamical landscape. Criticality ($\lambda$) tunes the brain's dynamical repertoire from rigid order to chaotic divergence, while precision ($\Pi$) gates the balance between bottom-up sensory evidence and top-down priors. Observable via avalanche estimators ($\hat{\lambda}$) and behavioral/neuromodulatory markers ($\hat{\Pi}$), this framework predicts systematic mappings between control parameters, cognitive performance, and clinical states across species and substrates.

#### TikZ Code for LaTeX Rendering

For direct LaTeX compilation (standalone PDF or `\includegraphics`):

```latex
\documentclass{standalone}
\usepackage{tikz}
\usetikzlibrary{shapes.geometric, arrows.meta, positioning}

\begin{document}
\begin{tikzpicture}[
  scale=1.5,
  axis/.style={thick, black},
  region/.style={draw=black, thick, fill opacity=0.3},
  label/.style={font=\small\bfseries},
  arrow/.style={-Stealth, thick, blue!70!black}
]

% Axes
\draw[axis] (0,0) -- (10,0) node[right] {$\lambda$ (Criticality)};
\draw[axis] (0,0) -- (0,8) node[above] {$\Pi$ (Precision)};

% Critical line (λ=1)
\draw[dashed, red, thick] (5,0) -- (5,8) node[above, red] {$\lambda=1$};

% Regions
% Healthy (central corridor)
\fill[green!30] (4.5,2) rectangle (5.5,5) node[label, midway, above=0.1cm] {Healthy};

% ASD-like (subcritical, high Π_sensory)
\fill[orange!40] (2,4.5) rectangle (4.5,7) node[label, midway] {ASD-like};

% Psychotic (supercritical, high Π_prior)
\fill[red!30] (5.5,4.5) rectangle (8,7) node[label, midway] {Psychotic};

% Anesthetic (low λ, low Π)
\fill[gray!40] (0,0) rectangle (3,2) node[label, midway, above right] {Anesthetic};

% Manic (high λ, unstable)
\fill[purple!40] (7,0) rectangle (10,3) node[label, midway, above left] {Manic};

% Trajectories
\draw[arrow] (1.5,0.8) -- (4.8,3) node[midway, above, font=\footnotesize] {Development};
\draw[arrow] (5,3.5) -- (7.5,6) node[midway, right, font=\footnotesize, rotate=45] {Psychosis};
\draw[arrow] (7.5,6) .. controls (6.5,4.5) and (5.5,3.5) .. (5,3)
  node[midway, left, font=\footnotesize, rotate=-30] {Recovery};

% Axis labels
\node at (2.5, -0.4) [font=\small] {Subcritical};
\node at (5, -0.4) [font=\small] {Critical};
\node at (7.5, -0.4) [font=\small] {Supercritical};

\node at (-0.6,1.5) [rotate=90, font=\small] {Low $\Pi$};
\node at (-0.6,6) [rotate=90, font=\small] {High $\Pi_{\text{prior}}$};
\node at (-1.2,3) [rotate=90, font=\small] {$\Pi_{\text{sensory}}$};

\end{tikzpicture}
\end{document}
```

**Usage:** Compile with `pdflatex` to generate standalone PDF. Include in manuscript via `\includegraphics{gutc_phase_diagram.pdf}`.

### 16.5 Conclusion

The GUTC framework unifies:
- **Physics** (criticality, phase transitions)
- **Information theory** (Fisher information, capacity)
- **Computation** (predictive coding, active inference)
- **Neurobiology** (laminar circuits, neuromodulation)
- **Clinical science** (computational psychiatry)

into a single, coherent theoretical structure. The brain emerges as a **near-critical inference engine** whose computational regime is determined by the interplay of global coupling ($\lambda$) and precision allocation ($\Pi$).

**Central Thesis:**

$$\boxed{\lambda \text{ sets the phase}; \quad \Pi \text{ decides which errors move the system}}$$

If empirically supported, this would elevate GUTC from a unifying narrative to a **quantitative control theory of cognition**, specifying not only *what* healthy and disordered minds are, but *where* they live in a shared dynamical landscape and *how* they move within it.

### Section XVI References

| Citation | URL |
|----------|-----|
| [49] | https://www.sciencedirect.com/science/article/pii/S0896627325003915 |
| [50] | https://www.sciencedirect.com/science/article/abs/pii/S0149763423004426 |
| [51] | https://pmc.ncbi.nlm.nih.gov/articles/PMC7614021/ |
| [52] | https://pmc.ncbi.nlm.nih.gov/articles/PMC9336647/ |
| [53] | https://link.aps.org/doi/10.1103/PhysRevE.111.014410 |
| [54] | https://royalsocietypublishing.org/doi/10.1098/rstb.2020.0531 |

---

## Appendix A: Connecting All the Dots — From Spikes to the $(\lambda, \Pi)$ Control Manifold

This appendix provides a comprehensive walkthrough of the entire GUTC framework, connecting all components from neural spikes to the control manifold.

### A.0 The Big Picture

GUTC proposes:

> **A brain is a near-critical, precision-tuned inference machine.**

Its state lives on a low-dimensional control manifold:
- **Global criticality ($\lambda$):** How close the dynamics are to the critical point
- **Local precision fields ($\Pi$):** How strongly different prediction errors are weighted

Everything else — avalanches, microcircuits, dopamine, autism, psychosis, attention, agency — is "what it looks like" when you move around on this $(\lambda, \Pi)$ manifold.

### A.1 Substrate: Critical Dynamics (CBH → $\lambda$)

At the physical/network level:
- Neurons form huge recurrent networks
- Activity propagates in **neuronal avalanches**: bursts with power-law size/duration distributions
- We estimate a **branching ratio** $\hat{m}$ from EEG/MEG avalanches:

$$\hat{\lambda} \approx \hat{m}, \quad E(\lambda) = \lambda - 1$$

| $\lambda$ Value | Regime | Characteristics |
|-----------------|--------|-----------------|
| $\lambda \approx 1$ | **Critical** | Scale-free avalanches, maximal dynamic range, $C(\lambda)$ peaks |
| $\lambda < 1$ | **Subcritical** | Activity dies out → rigid, low-capacity dynamics |
| $\lambda > 1$ | **Supercritical** | Activity blows up → unstable, noisy, epileptiform |

**Core Principle:** Thought $\iff$ maximize $C(\lambda)$ subject to $E(\lambda) = 0$

### A.2 Architecture: Canonical Cortical Microcircuit

One cortical column, wired like cortex (not a generic MLP):

| Layer | Function | Key Connections |
|-------|----------|-----------------|
| **L4** | Thalamic/feedforward input | Receives external drive |
| **L2/3** | Superficial integrator | Strong recurrence; projects FF to higher areas |
| **L5** | Deep integrator | Sends FB down and to subcortex |
| **L6** | Gain-control | Feedback to thalamus and L4 |

**Key Connectivity:**
- Strong **L4 → L2/3** (input → superficial)
- Strong bidirectional **L2/3 ↔ L5** (surface ↔ deep loop)
- Between areas: **FF** (L2/3 lower → L4 higher), **FB** (L5/6 higher → L1/L6 lower)

### A.3 Cell Types: Inhibition as Precision & Gating

| Cell Type | Function | GUTC Role |
|-----------|----------|-----------|
| **Pyramidal (E)** | Main excitatory output | Carry predictions and errors |
| **PV interneurons** | Fast, perisomatic inhibition | Stabilize firing rate, control overall gain |
| **SOM interneurons** | Target distal dendrites | Gate top-down inputs/predictions |
| **VIP interneurons** | Inhibit SOM | Disinhibit pyramids; context-dependent gating |

This gives hardware for:
- Stabilizing the critical recurrent core (PV)
- Gating which inputs get through (SOM/VIP)
- Implementing **precision weighting** as actual dendritic/somatic gain control

### A.4 Algorithm: Predictive Coding & Active Inference

On the microcircuit, run **predictive coding / active inference**:

- Brain holds a **generative model** $p_\theta(u,x)$ over hidden states $x$ and observations $u$
- Minimizes **Variational Free Energy**:

$$\mathcal{F}(q,u) = D_{\text{KL}}[q(x)\|p(x|u)] - \mathbb{E}_q[\ln p(u|x)]$$

- In active inference, chooses actions/policies $\pi$ to minimize **Expected Free Energy** $G(\pi)$:
  - **Extrinsic value:** avoid bad outcomes (reward/cost)
  - **Epistemic value:** gain information (reduce uncertainty)

**Circuit Mapping:**
| Component | Location | Function |
|-----------|----------|----------|
| Prediction errors (PEs) | L2/3 (superficial) | Send FF signals up |
| Predictions | L5/6 (deep) | Send FB signals down |
| Free energy minimization | Recurrent loop | Predictions vs PEs until mismatch is small |

### A.5 Precision Fields ($\Pi$): Neuromodulators as Gain Maps

All prediction errors are *not* equal. **Precision** weights them:

| Neuromodulator | Target | Effect | GUTC Parameter |
|----------------|--------|--------|----------------|
| **Acetylcholine (ACh)** | L4/L2/3 | Boosts gain of sensory PEs | $\Pi_{\text{sensory}}$ |
| **Dopamine (DA, D2)** | SFC/dACC, striatum | Modulates gain of outcome/reward PEs | $\Pi_{\text{prior}}$ |

**Mathematical Implementation:**

Neuron update with precision:
$$\tau \dot{x}_i = -x_i + f\big(\lambda \sum_j W_{ij}x_j + I_i^{\text{ext}} + I_i^{\text{mod}}\big)$$

where $f$ has gain $a$, and $\Pi$ modulates $a$:
- $a \to a \cdot \Pi_{\text{sensory}}$ for sensory PE units
- $a \to a \cdot \Pi_{\text{prior}}$ for frontal/outcome PE units

SOM gating:
$$I^{\text{top-down}}_{\text{eff}} = I^{\text{top-down}} - \Pi_{\text{prior}} \cdot \alpha_{\text{SOM}} \cdot x^{\text{SOM}}$$

**Summary:** $\lambda$ sets "how loud the whole network can get"; $\Pi$ decides "which microphones are turned up."

### A.6 Memory & Thought: HHNs on a Critical Substrate

On this critical, precision-tuned substrate, embed **Hierarchical Heteroclinic Networks (HHNs)**:

- Attractor-like structures of **saddle points** connected by **heteroclinic trajectories**
- **Hierarchical:** fast cycles (phonemes, micro-movements) feed into slower, higher-level saddles (words, plans)

| Memory Type | GUTC Realization |
|-------------|------------------|
| **Working memory ($M_W$)** | Near-critical, high-capacity dynamics (avalanches, long correlations) |
| **Long-term memory ($M_L$)** | HHNs in the dynamical landscape |
| **Cognons** | Chunks/units of thought = trajectories through HHNs |

### A.7 Agency: Critical Active Inference

GUTC agency functional:

$$\mathcal{J}(\lambda,\theta,\phi) = J(\pi_\phi,\theta) + \alpha \cdot C(\lambda,\theta) - \beta \cdot |E(\lambda,\theta)|$$

| Term | Meaning | Active Inference Equivalent |
|------|---------|----------------------------|
| $J(\pi)$ | Extrinsic value | Reward, preferred outcomes |
| $C(\lambda)$ | Epistemic value | Predictive information, exploration |
| $\|E(\lambda)\|$ | Criticality penalty | Distance from CBH setpoint |

At **criticality** ($E(\lambda) \approx 0$) with **well-tuned $\Pi$**:
- Gradients of free energy are sharp but stable
- Small actions/observations yield large, informative changes
- Exploration–exploitation balance is optimal

### A.8 Psychopathology: Disorders as $(\lambda, \Pi)$-Mis-Tuning

| Region | $\lambda$ | $\Pi$ | Clinical Mapping |
|--------|-----------|-------|------------------|
| Subcritical + over-precise | $< 1$ | High | **ASD-like:** reduced flexibility, repetitive behaviors |
| Supercritical + mis-allocated | $> 1$ | Wrong level | **Psychosis-like:** unstable beliefs, delusions |
| Strongly subcritical | $\ll 1$ | Low | **Anesthesia, depression** |
| Strongly supercritical | $\gg 1$ | - | **Seizures, runaway dynamics** |

**Key insight:** Different clinical phenotypes = different **regions/trajectories** in $(\lambda, \Pi)$ space, not entirely different systems.

### A.9 Measurement & AI: Closing the Loop

**Brain Side:**
1. EEG/MEG → avalanches → $\hat{m} \approx \hat{\lambda}$
2. Behavior + imaging + pharmacology → infer $\hat{\Pi}$
3. Place each subject/condition as point in $(\hat{\lambda}, \hat{\Pi})$ space
4. Test whether coordinates predict: cognitive performance, symptom severity, treatment response

**AI Side:**
1. Build **Critical Thought Fields (CTFs):** artificial recurrent cores tuned to criticality ($\rho(W) \approx 1$)
2. Add low-rank heteroclinic $M_L$ component ($W_{\text{het}}$)
3. Implement $\Pi$-like gain maps (attention/neuromod-style scaling)
4. Train with active-inference-like objectives (EFE) and SOC controllers homing $\lambda$ to 1

**Substrate-Independent Law:** Biology and silicon obey the same geometry, just with different parts lists.

### A.10 Summary: The Whole Cathedral in One Coordinate System

| Component | What It Provides |
|-----------|------------------|
| **CBH** | *Where* brains sit: near $E(\lambda) = 0$ |
| **Microcircuits & cell types** | *What hardware* does the work |
| **Predictive coding / active inference** | *What computations* run on that hardware |
| **Precision (ACh/DA, $\Pi$)** | *Which signals* matter, right now |
| **HHNs** | *How* sequences, chunks, and thoughts are structured |
| **$(\lambda, \Pi)$ manifold** | *How* all of that changes in health, disorder, and artificial systems |

$$\boxed{\text{The brain is a point moving on a } (\lambda, \Pi) \text{ manifold. GUTC is its coordinate system.}}$$

---

## References

1. Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127-138.

2. Rao, R. P., & Ballard, D. H. (1999). Predictive coding in the visual cortex. *Nature Neuroscience*, 2(1), 79-87.

3. Friston, K., et al. (2017). Active inference and learning. *Neuroscience & Biobehavioral Reviews*, 68, 862-879.

4. Parr, T., Pezzulo, G., & Friston, K. J. (2022). *Active Inference: The Free Energy Principle in Mind, Brain, and Behavior*. MIT Press.

5. Clark, A. (2013). Whatever next? Predictive brains, situated agents, and the future of cognitive science. *Behavioral and Brain Sciences*, 36(3), 181-204.

6. Hohwy, J. (2013). *The Predictive Mind*. Oxford University Press.
