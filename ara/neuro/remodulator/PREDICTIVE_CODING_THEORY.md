# Predictive Coding: Theoretical Foundation

> This document provides the theoretical basis for the Brain Remodulator framework,
> grounding the Delusion Index (D) and precision-weighting concepts in established
> neuroscience and computational theory.

---

## 1. Introduction to Predictive Coding

Predictive coding is a theoretical framework in neuroscience and computational modeling that posits the brain as a **hierarchical prediction machine**. It suggests that the brain continuously generates top-down predictions about incoming sensory inputs based on an internal generative model of the world, and then compares these predictions against actual bottom-up sensory data to compute **prediction errors**. These errors are used to update the model, minimizing discrepancies and enabling efficient perception, learning, and action.

Originating from concepts in information theory and Bayesian inference, predictive coding has been formalized under Karl Friston's **Free Energy Principle (FEP)**, where the brain minimizes variational free energy—a proxy for surprise or prediction error—to infer causes of sensations and maintain adaptive behavior. This framework extends beyond mere perception to unify aspects of cognition, providing a biologically plausible account of how neural hierarchies process information.

```
                    PREDICTIVE CODING HIERARCHY

    HIGHER LEVELS                              LOWER LEVELS
    (Abstract)                                 (Concrete)

    ┌─────────────┐                           ┌─────────────┐
    │   "FACE"    │                           │   PIXELS    │
    │  (concept)  │                           │  (sensory)  │
    └──────┬──────┘                           └──────▲──────┘
           │                                         │
           │  PREDICTIONS (top-down)                 │  ERRORS (bottom-up)
           │  "I expect to see eyes here"            │  "Unexpected edge detected"
           ▼                                         │
    ┌──────────────────────────────────────────────────────┐
    │              PREDICTION ERROR UNITS                   │
    │         ε = observed - predicted                      │
    │         Weighted by PRECISION (Π)                     │
    └──────────────────────────────────────────────────────┘
```

---

## 2. Core Components of the Framework

Predictive coding operates through a bidirectional hierarchical structure, often modeled as a multi-layer neural network or cortical column equivalent:

### 2.1 Generative Model

At the heart is a **probabilistic generative model** that simulates how hidden causes (e.g., objects or events) produce sensory observations. This model is hierarchical:

| Level | Represents | Examples |
|-------|------------|----------|
| Higher | Abstract, invariant features | "A face", "danger", "mother" |
| Middle | Compositional structure | Eyes, nose, spatial relations |
| Lower | Raw sensory details | Edges, colors, frequencies |

Predictions flow **top-down**, while errors ascend **bottom-up**.

### 2.2 Prediction Errors

These are the mismatches between predicted and actual inputs:

```
ε = y - g(μ)

where:
  y = actual sensory input
  g(μ) = predicted input from generative model
  ε = prediction error
```

Crucially, errors are **precision-weighted** by their reliability (inverse variance):

```
ε̃ = Π · ε

where:
  Π = precision (confidence in the signal)
  ε̃ = precision-weighted error
```

This is the foundation of the **Delusion Index**:
- High Π on predictions → errors are downweighted → priors dominate
- High Π on sensory → predictions are revised → sensory dominates

### 2.3 Hierarchical Message Passing

In cortical terms:

| Connection Type | Direction | Carries | Neural Correlate |
|-----------------|-----------|---------|------------------|
| Forward | Bottom-up | Prediction errors | Superficial pyramidal cells |
| Backward | Top-down | Predictions | Deep pyramidal cells |

This reciprocal exchange minimizes free energy across layers, with **attention modulating precisions** to prioritize salient errors.

```
CORTICAL MICROCIRCUIT (simplified)

    Layer 2/3 (Superficial)          Layer 5/6 (Deep)
    ┌─────────────────────┐          ┌─────────────────────┐
    │  ERROR UNITS        │          │  PREDICTION UNITS   │
    │  Forward projections│◄─────────│  Backward proj.     │
    │  to higher areas    │          │  from higher areas  │
    └─────────┬───────────┘          └─────────▲───────────┘
              │                                │
              │    ε = y - g(μ)               │
              │                                │
              ▼                                │
         TO HIGHER LEVEL              FROM HIGHER LEVEL
```

### 2.4 Learning and Plasticity

Over time, synaptic weights adjust to better predict regularities:

- **Hebbian learning**: "Neurons that fire together wire together"
- **Gradient descent on errors**: Minimize F by adjusting parameters
- **Structural learning**: Update the model's causal structure

This includes updating **priors** (what we expect) and **parameters** (how causes map to effects).

---

## 3. Mathematical Basis

### 3.1 Variational Free Energy

Predictive coding formalizes perception as **variational Bayesian inference**, minimizing variational free energy F as an upper bound on surprise (negative log evidence).

For a generative model p(o, u) with observations o and hidden causes/states u:

```
F = E_q(u)[ln q(u) - ln p(o, u)]
  = D_KL[q(u) || p(u|o)] - ln p(o)
```

where:
- q(u) is an approximate posterior (e.g., Gaussian N(μ̃, C))
- D_KL is the Kullback-Leibler divergence
- p(o) is the model evidence

**Key insight**: Minimizing F optimizes q(u) to approximate the true posterior while maximizing model evidence.

### 3.2 Derivation

Start with the log evidence:

```
ln p(o) = E_q(u)[ln p(o|u) + ln p(u) - ln q(u)] + D_KL[q(u) || p(u|o)]
```

Since KL divergence is non-negative:

```
F = -E_q(u)[ln p(o|u) + ln p(u) - ln q(u)] ≥ -ln p(o)
```

Therefore F is an **upper bound on surprise**.

### 3.3 Gradient Descent Dynamics

Minimizing F via gradient descent yields neural dynamics:

```
μ̇ = Dμ - ∂F/∂μ
  = Dμ - ε̃ᵀ Π ε̃
```

where:
- ε̃ are precision-weighted prediction errors
- Π are precision matrices
- D is a derivative operator for generalized coordinates

### 3.4 Hierarchical Free Energy

In hierarchies, F sums layer-wise:

```
F = Σ_l [ε̃_lᵀ Π_l ε̃_l + log det(Σ_l)]
```

with updates propagating errors bidirectionally.

### 3.5 Natural Gradient Descent

Under the Laplace approximation:

```
μ̇ = -H⁻¹ ∂F/∂μ
```

where H = ∂²F/∂μ² ≈ Π (the precision matrix).

This is **natural gradient descent**, which accounts for the geometry of the parameter space.

---

## 4. The Delusion Index: Precision Ratio

The Brain Remodulator operationalizes predictive coding through the **Delusion Index**:

```
D = Π_prior / Π_sensory
```

This ratio captures the relative weighting of top-down predictions vs. bottom-up evidence:

| D Value | Interpretation | Cognitive Style |
|---------|----------------|-----------------|
| D ≈ 1.0 | Balanced | Healthy, flexible inference |
| D >> 1 | Prior-dominated | Beliefs override evidence |
| D << 1 | Sensory-dominated | Overwhelmed by input |

### 4.1 Why This Matters

In predictive coding, perception is a weighted combination:

```
Posterior ∝ Prior^Π_prior × Likelihood^Π_sensory
```

When D >> 1:
- Strong priors → expect patterns even in noise
- Weak sensory updating → ignore contradicting evidence
- Result: hallucinations, delusions, false beliefs

When D << 1:
- Weak priors → no predictions to filter input
- Strong sensory → overwhelmed by raw data
- Result: sensory overload, hyper-literalism, no abstraction

### 4.2 Mapping to EEG

The Brain Remodulator estimates D from EEG biomarkers:

| Precision | EEG Signature | Rationale |
|-----------|---------------|-----------|
| Π_prior | Frontal theta (4-8Hz) | Theta oscillations index top-down prediction updating |
| Π_prior | Theta-gamma coupling | Coupling reflects hierarchical message passing |
| Π_sensory | Alpha suppression (8-12Hz) | Alpha gating: suppression = sensory channel open |
| Π_sensory | Posterior gamma (30-80Hz) | Gamma = sensory binding and precision |

---

## 5. Applications and Neuroscience Implications

### 5.1 Perception and Illusions

Predictive coding explains why we perceive stable worlds despite noisy inputs:

- **Bistable figures** (Necker cube): Priors bias error resolution between interpretations
- **Change blindness**: Strong predictions suppress unexpected changes
- **Placebo effects**: Predictions modulate pain perception

### 5.2 Psychiatric Disorders

Disruptions in precision weighting link to specific disorders:

| Disorder | Precision Aberration | Manifestation |
|----------|---------------------|---------------|
| Schizophrenia | Over-strong priors | Hallucinations, delusions |
| Autism | Under-strong priors | Sensory overload, detail focus |
| Anxiety | Aberrant threat precision | Hypervigilance, false alarms |
| Depression | Aberrant self-model precision | Negative self-fulfilling prophecies |

### 5.3 Action and Active Inference

Predictive coding extends to motor control via **active inference**:

```
Actions minimize prediction errors by changing the world
to match predictions (rather than updating beliefs).
```

Example: Reaching for a cup
1. Predict proprioceptive signals for "arm at cup position"
2. Current position creates prediction error
3. Motor system acts to suppress error
4. Arm moves to predicted position

### 5.4 Learning and AI

Predictive coding has influenced machine learning:

- **PredNet**: Video prediction via error propagation
- **Contrastive learning**: Minimize distance to positive samples
- **Variational autoencoders**: Minimize reconstruction + KL terms

Under fixed predictions, predictive coding is equivalent to **backpropagation**.

### 5.5 Experimental Evidence

| Evidence Type | Finding |
|---------------|---------|
| fMRI | Error signals in visual cortex (V1, V2) |
| EEG/ERP | Mismatch negativity (MMN) = precision-weighted error |
| Electrophysiology | End-stopping in receptive fields matches predictions |
| Computational | Simulations replicate neural response properties |

---

## 6. Criticisms and Limitations

While influential, predictive coding faces several challenges:

### 6.1 Theoretical Concerns

| Concern | Description |
|---------|-------------|
| Falsifiability | Framework is flexible enough to fit most data post-hoc |
| Specificity | Hard to distinguish from alternative theories empirically |
| Scope | Claims to explain "everything" may be unfalsifiable |

### 6.2 Biological Challenges

| Challenge | Description |
|-----------|-------------|
| Negative errors | How do rate-coded neurons represent negative values? |
| Weight transport | Backward weights need to match forward (biologically unclear) |
| Frequency bands | Theory predicts beta for predictions, but evidence is mixed |
| Heterarchy | Real brains have lateral and skip connections |

### 6.3 Empirical Limitations

- Some fMRI results don't clearly distinguish predictions from adaptation
- Individual differences in precision weighting poorly characterized
- EEG-based D estimation remains noisy and indirect

---

## 7. Connection to Free Energy Principle

Predictive coding is a **perceptual subset** of the broader Free Energy Principle:

```
FREE ENERGY PRINCIPLE
├── PERCEPTION: Minimize F by updating beliefs (q → p(u|o))
│   └── Predictive Coding
├── ACTION: Minimize F by changing observations (o → match predictions)
│   └── Active Inference
└── LEARNING: Minimize F by updating model parameters
    └── Structure Learning
```

The FEP proposes that all biological systems minimize free energy (or equivalently, maximize evidence for their model of the world) to maintain their existence against entropy.

---

## 8. Implications for the Brain Remodulator

The Brain Remodulator translates predictive coding theory into a practical intervention:

### 8.1 Theoretical Basis

1. **Precision is tunable**: Attention, neuromodulation, and learning adjust Π
2. **EEG reflects precision**: Theta, alpha, gamma bands index hierarchical precision
3. **Feedback can modify precision**: Neurofeedback has established efficacy

### 8.2 Intervention Logic

```
REMODULATOR CONTROL LOOP

1. MEASURE: Estimate D = Π_prior / Π_sensory from EEG

2. DETECT: Classify aberrant pattern
   - D >> 1 → prior-dominated (schizophrenia-like)
   - D << 1 → sensory-dominated (ASD-like)

3. INTERVENE: Apply corrective feedback
   - D >> 1 → boost sensory precision (grounding, external focus)
   - D << 1 → boost prior precision (abstraction, internal model)

4. ITERATE: Track D convergence toward target (D ≈ 1.0)
```

### 8.3 Theoretical Predictions

If the framework is correct:

1. D should correlate with symptom severity in schizophrenia/ASD
2. Normalizing D should reduce symptoms
3. EEG signatures should track D reliably across individuals
4. Feedback should produce lasting changes in precision weighting

These are **testable hypotheses**, not established facts.

---

## References

### Foundational Papers

1. Rao, R.P., & Ballard, D.H. (1999). Predictive coding in the visual cortex.
2. Friston, K. (2005). A theory of cortical responses.
3. Friston, K. (2010). The free-energy principle: A unified brain theory?
4. Clark, A. (2013). Whatever next? Predictive brains, situated agents.

### Precision and Attention

5. Feldman, H., & Friston, K. (2010). Attention, uncertainty, and free-energy.
6. Kanai, R., et al. (2015). Cerebral hierarchies: Predictive processing.
7. Parr, T., & Friston, K. (2019). Attention or salience?

### Clinical Applications

8. Adams, R.A., et al. (2013). The computational anatomy of psychosis.
9. Lawson, R.P., et al. (2014). Adults with autism overestimate precision.
10. Sterzer, P., et al. (2018). The predictive coding account of psychosis.

### EEG Correlates

11. Sedley, W., et al. (2016). Neural signatures of perceptual inference.
12. Bastos, A.M., et al. (2012). Canonical microcircuits for predictive coding.
13. Arnal, L.H., & Giraud, A.L. (2012). Cortical oscillations and sensory predictions.

---

*Document version: 0.1*
*Part of the Brain Remodulator framework*
*Status: Theoretical background*
