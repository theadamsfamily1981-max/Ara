# Edge of Autumn: Unified Empirical Validation

## The Thesis

**Claim**: There exists a "balanced regime" in β-VAE hyperparameter space where:
1. The regime **exists** (non-empty)
2. Representations are **structured** (disentangled)
3. The system **generalizes** (OOD transfer)
4. The system is **antifragile** (gains from moderate noise)
5. Representations are **causally factorized** (not just correlated)
6. Dynamics **mimic brain criticality** (power-law, edge-of-chaos)

This document integrates six experiments into a coherent empirical proof.

---

## Experiment Architecture

```
                    ┌─────────────────────────────────────┐
                    │     EDGE OF AUTUMN THEOREM          │
                    │   "Balanced regime 𝓑 exists and     │
                    │    has desirable properties"        │
                    └─────────────────────────────────────┘
                                     │
           ┌─────────────────────────┼─────────────────────────┐
           │                         │                         │
           ▼                         ▼                         ▼
    ┌─────────────┐          ┌─────────────┐          ┌─────────────┐
    │ EXISTENCE   │          │ STRUCTURE   │          │ DYNAMICS    │
    │             │          │             │          │             │
    │ MIG vs β    │          │ Causal ISS  │          │ Criticality │
    │ Peak at β*  │          │ Modularity  │          │ Power-laws  │
    └─────────────┘          └─────────────┘          └─────────────┘
           │                         │                         │
           │              ┌─────────┴─────────┐               │
           │              ▼                   ▼               │
           │    ┌─────────────┐      ┌─────────────┐         │
           │    │ OOD         │      │ Antifragile │         │
           │    │ Generalize  │      │ Noise gains │         │
           │    └─────────────┘      └─────────────┘         │
           │              │                   │               │
           └──────────────┴─────────┬─────────┴───────────────┘
                                    ▼
                    ┌─────────────────────────────────────┐
                    │     UNIFIED EMPIRICAL PROOF         │
                    │   "The balanced regime is real,     │
                    │    structured, robust, and          │
                    │    brain-like"                      │
                    └─────────────────────────────────────┘
```

---

## The Six Experiments

### 1. MIG Compactness (Existence Proof)

**File**: `experiments/mig_compactness.py`

**Question**: Does a β regime with superior latent structure exist?

**Method**:
- Train β-VAE on dSprites (known factors: shape, scale, rotation, posX, posY)
- Sweep β ∈ {0.1, 0.3, 1.0, 3.0, 10.0}
- Compute MIG (Mutual Information Gap) per factor
- 5+ seeds, ANOVA + Tukey HSD

**Metric**:
```
MIG = mean over factors of (I₁ - I₂) / I₁
where I₁, I₂ are top two MI values between factor and latents
```

**Expected Result**: Inverted-U curve with peak at intermediate β

**Honest Claim**:
> "We observe a β regime where latent representations are maximally compact (high MIG), flanked by under- and over-regularized regimes where compactness degrades."

---

### 2. Causal Disentanglement (Structure Proof)

**File**: `experiments/causal_interventions_v2.py`

**Question**: Are latents *causally* factorized, not just correlated?

**Method**:
- For each latent z_i, traverse ±3σ while fixing others
- Decode and extract factors from reconstructions
- Measure variance of each factor across traversal

**Metric**: Intervention Specificity Score (ISS)
```
ISS_i = max_j Var(f'_j | traverse z_i) / Σ_k Var(f'_k | traverse z_i)

ISS ≈ 1.0 → z_i controls single factor (causal disentanglement)
ISS ≈ 0.0 → z_i controls many factors (causal entanglement)

Modularity = mean(ISS) over active latents
```

**Expected Result**: Modularity peaks at intermediate β; heatmaps show diagonal structure

**Honest Claim**:
> "The Modularity Score was statistically maximal in the intermediate β regime. This proves the balanced setting yields representations that are not only statistically disentangled but *causally factorized*, supporting the analogy to modular brain circuits."

---

### 3. OOD Generalization (Robustness Proof)

**File**: `experiments/ood_generalization.py`

**Question**: Does the balanced regime transfer better to unseen distributions?

**Method**:
- Train on source distribution (e.g., EEG subjects 1-8)
- Test on target distribution (e.g., subjects 9-10 or shifted factors)
- Measure AUC drop: ID_AUC - OOD_AUC

**Metric**:
```
AUC_drop = AUC(in-distribution) - AUC(out-of-distribution)
Lower drop = better generalization
```

**Expected Result**: Minimal AUC drop at intermediate β

**Honest Claim**:
> "The balanced β regime shows minimal performance degradation under distribution shift, indicating that disentangled representations capture invariant structures rather than memorizing training specifics."

---

### 4. Antifragility (Stress Response Proof)

**File**: `experiments/antifragility.py`

**Question**: Does the system *gain* from moderate perturbations?

**Method**:
- Apply escalating noise levels σ ∈ {0.01, 0.03, 0.05, 0.1, 0.15, 0.2}
- Measure ECE (Expected Calibration Error) before/after
- Track latent entropy changes

**Metric**:
```
ΔECE = ECE(noisy) - ECE(clean)
ΔECE < 0 → noise IMPROVED calibration (antifragile)
ΔECE > 0 → noise degraded calibration (fragile)
```

**Expected Result**: Negative ΔECE at low noise for intermediate β only

**Honest Claim**:
> "The balanced regime exhibits Taleb-style antifragility: moderate perturbations improve prediction calibration, unlike fragile (low-β) or robust-but-dull (high-β) regimes."

---

### 5. Biological Criticality (Dynamics Proof)

**File**: `experiments/criticality_signatures.py`

**Question**: Do training dynamics mimic brain criticality?

**Method**:
- Log gradient magnitudes during training
- Detect "avalanches" (consecutive above-threshold updates)
- Fit power-law to avalanche size distribution

**Metrics**:
```
α = power-law exponent (critical ≈ -1.5)
σ = branching ratio (critical ≈ 1.0)
```

**Expected Result**: α ≈ -1.5 and σ ≈ 1.0 at intermediate β

**Honest Claim**:
> "Training dynamics in the balanced regime exhibit power-law avalanches with exponent α ≈ -1.5 and branching ratio σ ≈ 1, matching signatures of cortical criticality (Beggs & Plenz, 2003)."

---

### 6. Edge of Autumn Sweep (Integration Proof)

**Files**: `experiment_edge_of_autumn.py`, `experiment_edge_of_autumn_v2.py`

**Question**: Can we empirically locate the balanced region 𝓑?

**Method**:
- Define S(β) = structure, P(β) = performance, R(β) = robustness
- Sweep β and compute all three metrics
- Find 𝓑 = {β : S ≥ S* ∧ P ≥ P* ∧ R ≥ R*}

**Expected Result**: Non-empty 𝓑 with optimal β* in interior

---

## How the Proofs Connect

```
MIG (Exp 1)                 Causal ISS (Exp 2)
    │                              │
    │ "Latents are                 │ "Latents are
    │  statistically               │  causally
    │  organized"                  │  factorized"
    │                              │
    └──────────┬───────────────────┘
               │
               ▼
        STRUCTURED REPRESENTATIONS
               │
       ┌───────┴───────┐
       │               │
       ▼               ▼
OOD (Exp 3)      Antifragility (Exp 4)
   │                   │
   │ "Structure        │ "Structure
   │  transfers"       │  improves
   │                   │  under stress"
   │                   │
   └─────────┬─────────┘
             │
             ▼
      ROBUST REPRESENTATIONS
             │
             ▼
   Criticality (Exp 5)
             │
             │ "Dynamics match
             │  brain signatures"
             │
             ▼
    BRAIN-LIKE COMPUTATION
             │
             ▼
   ┌─────────────────────────┐
   │  EDGE OF AUTUMN THESIS  │
   │  EMPIRICALLY VALIDATED  │
   └─────────────────────────┘
```

---

## Statistical Requirements

For each experiment, we require:

| Requirement | Standard |
|-------------|----------|
| Seeds | ≥ 5 per β |
| ANOVA | p < 0.05 for β effect |
| Post-hoc | Bonferroni-corrected pairwise |
| Effect size | Cohen's d ≥ 0.5 vs extremes |
| Peak location | Interior of β range (not at boundary) |

---

## What You Can Honestly Say

### If All Six Experiments Support:

> "We present comprehensive empirical evidence for the Edge of Autumn thesis: an intermediate regularization regime exists in β-VAE training that is simultaneously:
>
> 1. **Identifiable** (MIG peaks at β* ≈ 1)
> 2. **Causally structured** (ISS > 0.7, diagonal modularity)
> 3. **Generalizable** (minimal OOD degradation)
> 4. **Antifragile** (gains from moderate noise)
> 5. **Dynamically critical** (α ≈ -1.5, σ ≈ 1)
>
> This convergent evidence supports the analogy between optimal β-VAE representations and the 'edge of chaos' regime in biological neural systems, where balanced excitation/inhibition enables efficient, modular, and robust information processing."

### What You Cannot Claim:

- ❌ "We proved the brain works this way"
- ❌ "This is the only way to achieve these properties"
- ❌ "β = 1.0 is universally optimal"

### What You Can Claim:

- ✓ "An empirically identifiable balanced regime exists"
- ✓ "This regime has measurable advantages on multiple axes"
- ✓ "The pattern is consistent with criticality theories"
- ✓ "Results are reproducible across seeds"

---

## Running the Full Suite

```bash
# All experiments (takes ~30-60 min)
python -m ara.neuro.arabrain.experiments

# Individual experiments
python -m ara.neuro.arabrain.experiments.mig_compactness
python -m ara.neuro.arabrain.experiments.causal_interventions_v2
python -m ara.neuro.arabrain.experiments.ood_generalization
python -m ara.neuro.arabrain.experiments.antifragility
python -m ara.neuro.arabrain.experiments.criticality_signatures

# Fast mode (fewer seeds, epochs)
python -m ara.neuro.arabrain.experiments --fast
```

---

## Citation

If using these experiments, cite:

```
Edge of Autumn: Empirical Validation of Balanced Representational Regimes
in β-VAE Training Dynamics
```

---

## Future Extensions

1. **Real EEG data**: Adapt from dSprites to actual neural recordings (exploratory, not ground-truth)
2. **Cross-architecture**: Test on different encoder architectures
3. **Temporal dynamics**: Extend to sequence models (HGF integration)
4. **Intervention transfer**: Do causal interventions transfer across subjects?
