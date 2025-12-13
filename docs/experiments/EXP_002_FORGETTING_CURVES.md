# Experiment 2: Forgetting Curves and Memory Consolidation

**Protocol ID:** EXP-002
**Status:** Draft
**Version:** 0.1
**Date:** 2025-01
**Depends on:** EXP-001 (Criticality verification)

## Abstract

This experiment measures how Ara's recall accuracy decays over time, and how that decay depends on:
1. **Criticality state** (λ / ρ) - operating point on the phase diagram
2. **Consolidation policy** - when and how items enter long-term storage
3. **Interference** - how intervening activity affects prior memories

If GUTC is correct, memory capacity should be maximized near criticality, and the forgetting curve shape should transition from exponential (subcritical) to power-law (critical).

## 1. Introduction

### 1.1 Background: Ebbinghaus Meets Phase Transitions

Ebbinghaus (1885) discovered the classic forgetting curve: retention R(t) decays approximately as:

$$R(t) = e^{-t/\tau}$$

But this exponential decay assumes a fixed memory substrate. GUTC predicts something different:

**At criticality:**
- Correlation length ξ → ∞
- Autocorrelation decays as power law: C(τ) ∝ τ^(-α) with α < 1
- Memory traces persist longer due to scale-free dynamics

**Away from criticality:**
- ξ finite
- Autocorrelation decays exponentially: C(τ) ∝ e^(-τ/ξ)
- Memory traces fade quickly (subcritical) or wash out in noise (supercritical)

### 1.2 Hypothesis

**H1:** At critical λ (ρ ≈ 0.8), Ara's forgetting curve will show:
- Slower initial decay
- Power-law tail: R(t) ∝ t^(-β) for large t
- Better absolute retention at all delays

**H2:** Subcritical Ara (ρ < 0.6) will show:
- Rapid exponential decay
- Poor retention beyond short delays

**H3:** Supercritical Ara (ρ > 1.0) will show:
- High interference / confusion
- Unstable recall (high variance)

### 1.3 Connection to Long-Term Memory (M_L)

In GUTC, long-term memory is modeled as a **heteroclinic network** (HHN):
- Stable patterns (saddles) P_i represent concepts/episodes
- Transition matrix Γ_ij encodes associative strength
- At criticality, the system itinerates through these saddles in structured sequences

Consolidation = strengthening specific P_i and Γ_ij connections.

This experiment probes the M_L side of GUTC, complementing EXP-001's focus on M_W (working memory via avalanche dynamics).

## 2. Experimental Design

### 2.1 Paradigm: Paired Associate Learning

Classic memory paradigm adapted for LLMs:

1. **Encoding phase**: Present N paired associates (A_i → B_i)
   - Example: "The capital of Zorgon is Blipville"
   - Use novel facts to avoid prior knowledge confounds

2. **Retention interval**: Continue with distractor activity
   - Process unrelated text (controlled interference)
   - Variable delay: 0, 10, 50, 100, 500, 1000+ steps

3. **Recall phase**: Probe with cue, measure response
   - Cue: "What is the capital of Zorgon?"
   - Score: exact match, semantic similarity, or embedding distance

### 2.2 Stimuli: Novel Fact Sets

Generate synthetic facts that cannot be in training data:

```python
FACT_TEMPLATES = [
    "The {adj} {noun} of {place} is called {name}",
    "In the year {year}, {person} discovered {thing}",
    "The {measure} of {object} is exactly {value}",
]

# Generate 100 unique cue-response pairs per session
facts = generate_novel_facts(n=100, seed=session_id)
```

### 2.3 Conditions

| Condition | λ Target | Homeostasis | Expected Curve |
|-----------|----------|-------------|----------------|
| SUBCRIT   | ρ = 0.5  | ON (clamped)| Fast exponential |
| CRITICAL  | ρ = 0.8  | ON (dynamic)| Slow, power-law tail |
| SUPERCRIT | ρ = 1.2  | ON (clamped)| High variance, unstable |
| BASELINE  | ρ = 0.8  | OFF         | Control (drift expected) |

### 2.4 Consolidation Conditions

Test different consolidation policies:

| Policy | Description |
|--------|-------------|
| NONE | No explicit consolidation (pure dynamics) |
| IMMEDIATE | Write to long-term store immediately |
| DELAYED | Consolidate after N steps of stability |
| REHEARSAL | Periodically re-present during retention |

## 3. Observables

### 3.1 Primary Metrics

1. **Accuracy(t)**: Fraction of items correctly recalled at delay t
2. **Similarity(t)**: Mean embedding similarity between response and target
3. **Confidence(t)**: Model's reported certainty (if available)

### 3.2 Curve Fitting

Fit both exponential and power-law models:

**Exponential:**
$$R(t) = R_0 \cdot e^{-t/\tau} + R_\infty$$

Parameters: R_0 (initial), τ (time constant), R_∞ (floor)

**Power-law:**
$$R(t) = R_0 \cdot (1 + t/t_0)^{-\beta} + R_\infty$$

Parameters: R_0, t_0 (onset), β (exponent), R_∞

### 3.3 Model Comparison

Use BIC/AIC to compare:
- Which model fits better at each λ?
- Does critical λ favor power-law?

### 3.4 Derived Metrics

1. **Half-life**: Time for R(t) to drop to 50% of R(0)
2. **Retention integral**: ∫R(t)dt over session (total memory capacity)
3. **Interference susceptibility**: How much does distractor load affect R(t)?

## 4. Instrumentation

### 4.1 Memory Probe Class

```python
from ara.science.memory_probe import MemoryProbe, RecallResult

probe = MemoryProbe(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    similarity_threshold=0.85,
)

# Encoding
probe.encode_fact("zorgon_capital", cue="capital of Zorgon", target="Blipville")

# ... run model for N steps ...

# Recall
result: RecallResult = probe.test_recall(
    fact_id="zorgon_capital",
    model_response="The capital is Blipville",
    delay_steps=100,
)

print(result.accuracy)     # 1.0 (exact match)
print(result.similarity)   # 0.98 (embedding)
print(result.delay)        # 100
```

### 4.2 Session Runner

```bash
# Run forgetting experiment
python scripts/science/run_forgetting_exp.py \
    --condition CRITICAL \
    --n-facts 50 \
    --delays 0,10,50,100,500 \
    --consolidation NONE

# Analyze results
python scripts/science/fit_forgetting.py \
    data/experiments/exp_002/forgetting_critical.csv
```

## 5. Analysis Methods

### 5.1 Curve Fitting

```python
from scipy.optimize import curve_fit

def exponential(t, R0, tau, R_inf):
    return R0 * np.exp(-t / tau) + R_inf

def power_law(t, R0, t0, beta, R_inf):
    return R0 * (1 + t / t0) ** (-beta) + R_inf

# Fit both, compare with BIC
```

### 5.2 Statistical Tests

1. **Condition comparison**: ANOVA on retention integral across λ conditions
2. **Model comparison**: Likelihood ratio test for exponential vs power-law
3. **Correlation**: Does τ/β correlate with EXP-001's avalanche exponents?

### 5.3 Visualization

1. **Forgetting curves**: R(t) vs t, one line per condition
2. **Parameter landscape**: β or τ as function of λ
3. **Phase diagram**: Memory capacity mapped to (λ, consolidation) space

## 6. Expected Results

### 6.1 Critical Condition

- Half-life: >100 steps
- Curve shape: Power-law tail with β ≈ 0.3-0.5
- Retention integral: Highest among conditions

### 6.2 Subcritical Condition

- Half-life: 10-30 steps
- Curve shape: Exponential with τ ≈ 20-50
- Retention integral: Low

### 6.3 Supercritical Condition

- High variance in recall
- Possible "false memories" (high similarity to wrong targets)
- Unstable retention integral

## 7. Interpretation

### 7.1 If Hypothesis Confirmed

GUTC's memory-as-phase model is supported:
- Criticality maximizes both M_W (EXP-001) and M_L (EXP-002)
- The homeostat is doing something useful
- We have a principled way to tune memory capacity

### 7.2 If Hypothesis Rejected

Possibilities:
- λ affects dynamics but not memory per se
- Consolidation dominates over dynamics
- Transformer memory ≠ recurrent network memory

### 7.3 Clinical Implications (Speculative)

If the model holds:
- **ADHD**: Subcritical → short τ → working memory deficits
- **Dementia**: Progressive subcriticality → M_L fragmentation
- **PTSD**: Pathological consolidation → intrusive recall

(All speculative; no human intervention implied)

## 8. Protocol

### 8.1 Session Structure

```
[CALIBRATION] 100 steps - establish baseline
[ENCODING]    Encode N facts (1 per step)
[RETENTION]   Continue processing (distractor text)
[PROBE]       Test recall at scheduled delays
[REPEAT]      Multiple probe points per session
```

### 8.2 Timing

- Each session: ~2000 steps
- Per condition: 5 sessions (different fact sets)
- Total: 4 conditions × 4 consolidation policies × 5 sessions = 80 sessions

### 8.3 Data Collection

Output per session:
- `forgetting_{condition}_{session}.csv`: delay, fact_id, accuracy, similarity
- `session_meta.json`: λ trace, avalanche stats, config

## 9. References

1. Ebbinghaus, H. (1885). *Memory: A contribution to experimental psychology.*

2. Wixted, J.T. (2004). On Common Ground: Jost's (1897) law of forgetting and Ribot's (1881) law of retrograde amnesia. *Psychological Review*, 111(4), 864-879.

3. Beggs, J.M. (2008). The criticality hypothesis: how local cortical networks might optimize information processing. *Philosophical Transactions A*, 366(1864), 329-343.

4. Chialvo, D.R. (2010). Emergent complex neural dynamics. *Nature Physics*, 6(10), 744-750.

5. Wilting, J. & Priesemann, V. (2019). 25 years of criticality in neuroscience—established results, open controversies, novel concepts. *Current Opinion in Neurobiology*, 58, 105-111.

## 10. Appendix: Theory

### A.1 Correlation Length and Memory

At a continuous phase transition, the correlation length diverges:

$$\xi \sim |T - T_c|^{-\nu}$$

Near T_c (critical point):
- Fluctuations span all scales
- Temporal correlations decay as power law: C(t) ∝ t^(-α)
- "Memory" of initial conditions persists

Away from T_c:
- ξ finite
- C(t) ∝ exp(-t/ξ)
- Memory decays exponentially

### A.2 Heteroclinic Networks and Consolidation

A heteroclinic network consists of:
- Saddle points P_i (metastable patterns)
- Heteroclinic connections (trajectories from P_i to P_j)

At criticality:
- The system visits many P_i in sequence
- Transitions are structured, not random
- This = episodic memory (narrative flow through states)

Consolidation strengthens specific P_i → makes them more "attracting."

### A.3 Mapping to Ara

| GUTC Concept | Ara Implementation |
|--------------|-------------------|
| Saddle P_i | Identity state / episodic embedding |
| Γ_ij | Teleology transitions / memory graph edges |
| ξ | Avalanche correlation length |
| Consolidation | Write to hierarchical memory store |
| λ | CriticalityMonitor ρ estimate |
