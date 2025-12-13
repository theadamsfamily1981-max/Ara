# Renormalization Group Perspective on Edge of Autumn

## The Core Insight

RG says: near a critical point, many different microscopic systems flow to the **same macroscopic behavior** when you zoom out. That "same behavior" is a **universality class**.

```
Microscopic (varies)              Macroscopic (stable)
─────────────────────────────────────────────────────────
• Exact nonlinearity              • MIG curve shape
• Architecture details            • Avalanche exponent ≈ -1.5
• Optimizer choice                • Balanced β band location
• Dataset quirks                  • ISS/modularity pattern
                                  • OOD/antifragility sweet spot
```

**RG justifies saying**: "These patterns aren't accidents of my exact code; they're stable features of a broader class of models."

That's the leap from "neat experiment" → "this is a real phenomenon."

---

## Mapping to Edge of Autumn

| RG Concept | AraBrain Equivalent |
|------------|---------------------|
| Control parameter | β (KL weight) |
| Order parameters | MIG, ISS, OOD-AUC, ΔECE, α, σ |
| Critical point | β* (balanced regime) |
| Fixed point | Where S(β), P(β), R(β) all meet thresholds |
| Relevant direction | β moving away from balanced band |
| Irrelevant directions | Architecture details that don't change macro-behavior |
| Universality class | "Critical β-VAE" models with same exponents |

### The Flow Picture

```
                        HIGH β (over-regularized)
                              │
                              │ collapsed latents
                              │ dead dimensions
                              ▼
    ┌─────────────────────────●─────────────────────────┐
    │                    FIXED POINT                    │
    │                                                   │
    │   • MIG peaks                                     │
    │   • ISS > 0.7 (causal modularity)                │
    │   • OOD drop minimal                              │
    │   • Antifragile (ΔECE < 0 at low noise)          │
    │   • α ≈ -1.5, σ ≈ 1.0                            │
    │                                                   │
    └─────────────────────────●─────────────────────────┘
                              ▲
                              │ entangled latents
                              │ memorization
                              │
                        LOW β (under-regularized)
```

---

## Empirical Tests for Universality

### Test A: Architecture Scaling

**Question**: Do different architectures show the same balanced regime?

**Protocol**:
```python
architectures = [
    {"latent_dim": 8,  "depth": 3, "width": 32},
    {"latent_dim": 16, "depth": 4, "width": 64},
    {"latent_dim": 32, "depth": 5, "width": 128},
]

nonlinearities = ["relu", "gelu", "tanh", "silu"]

for arch in architectures:
    for nonlin in nonlinearities:
        model = build_model(arch, nonlin)
        results = run_beta_sweep(model)
        balanced_band = find_edge_of_autumn(results)

        record(arch, nonlin, balanced_band, critical_exponents)
```

**Success Criteria**:
- All configurations have non-empty balanced band
- MIG/ISS/OOD curves have similar shapes
- Critical exponents α within ±0.3 of -1.5
- Balanced β* within same order of magnitude

**RG Interpretation**: If yes → same universality class

### Test B: Coarse-Graining Invariance

**Question**: Do critical signatures survive "zooming out"?

**Protocol**:
```python
def coarse_grain(z, block_size=2):
    """Average adjacent latent dimensions."""
    d = z.shape[-1]
    z_coarse = z.reshape(-1, d // block_size, block_size).mean(axis=-1)
    return z_coarse

# For trained model at β*:
z_fine = model.encode(x)           # (N, 16)
z_coarse = coarse_grain(z_fine)    # (N, 8)
z_coarser = coarse_grain(z_coarse) # (N, 4)

# Measure at each scale:
for z in [z_fine, z_coarse, z_coarser]:
    measure_avalanche_exponent(z)
    measure_entropy_scaling(z)
    measure_correlation_length(z)
```

**Success Criteria**:
- Avalanche exponent α stable under coarse-graining
- Entropy scales predictably (not random)
- Correlations don't suddenly vanish or explode

**RG Interpretation**: If stable → macro-statistics are genuine, not microscopic accidents

### Test C: Dataset Transfer

**Question**: Does balanced regime exist across datasets?

**Protocol**:
- Train on dSprites → find β*
- Train on 3DShapes → find β*
- Train on synthetic EEG → find β*
- Compare balanced bands and exponents

**Success Criteria**:
- Balanced band exists in all
- Relative location of β* similar (within factor of 3)
- Same qualitative curve shapes

---

## What RG Language Lets You Claim

### Honest Claims (with RG backing):

> "Across architectures and datasets we tested, there exists a β band where multiple macro-metrics (MIG, ISS, OOD, stress response) simultaneously peak or plateau."

> "The shape of these curves and the approximate location of the band are stable under architectural changes (width, depth, nonlinearity) and dataset resampling, suggesting RG-style universality of the balanced regime."

> "In this regime we observe critical-like scaling (avalanche exponents ≈ -1.5), connecting our model to the same universality class as simple branching-process / critical-brain models."

### Still Cannot Claim:

- ❌ "We proved the brain works this way" (RG is about classes, not identity)
- ❌ "β = 1 is universally optimal for all systems" (only within this class)
- ❌ "All neural networks have this property" (only those near this fixed point)

---

## The Complete Picture

```
┌─────────────────────────────────────────────────────────────────┐
│                     EDGE OF AUTUMN THESIS                       │
│                                                                 │
│  Level 1: Existence                                             │
│  └─ MIG vs β shows balanced regime exists                       │
│                                                                 │
│  Level 2: Structure                                             │
│  └─ ISS proves causal factorization (not just correlation)      │
│                                                                 │
│  Level 3: Robustness                                            │
│  └─ OOD + Antifragility show functional advantages              │
│                                                                 │
│  Level 4: Dynamics                                              │
│  └─ Critical exponents match brain-like models                  │
│                                                                 │
│  Level 5: Universality (RG)                           ← NEW     │
│  └─ Same patterns across architectures/datasets                 │
│  └─ Stable under coarse-graining                                │
│  └─ "Real phenomenon, not accident"                             │
│                                                                 │
│  Level 6: Self-Organization (SOC)                    ← FUTURE   │
│  └─ System self-tunes to critical regime                        │
│                                                                 │
│  Level 7: Neural Validation                          ← FUTURE   │
│  └─ Match to actual brain data                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Suggested Paper/Discussion Text

### For Background Section:

> "We interpret our results through the lens of Renormalization Group theory. The balanced β regime corresponds to a neighborhood of a fixed point in observable space, where macro-level quantities (disentanglement, modularity, generalization, stress response, critical dynamics) converge regardless of microscopic architectural details. This suggests AraBrain belongs to a universality class of 'critical representation learners' characterized by edge-of-chaos dynamics and efficient information encoding."

### For Discussion Section:

> "The stability of our findings across architectural variations (latent dimension, depth, nonlinearity) and datasets (dSprites, synthetic EEG) provides preliminary evidence for RG-style universality. The balanced regime is not an artifact of our specific implementation but appears to be a robust feature of β-VAE-like models near criticality. This connects our work to the broader theoretical framework of critical phenomena in neural computation, though we emphasize this establishes AraBrain as a principled toy model rather than a literal description of biological neural processing."

---

## Implementation Priority

To add RG evidence to your empirical suite:

1. **Architecture sweep** (1-2 days)
   - 3 latent dims × 4 nonlinearities = 12 configurations
   - Run existing β-sweep on each
   - Tabulate: balanced band, peak MIG, critical exponent

2. **Coarse-graining test** (half day)
   - Add `coarse_grain()` function
   - Measure exponents at 3 scales
   - Check stability

3. **Dataset transfer** (1 day)
   - Already have dSprites and synthetic EEG
   - Add 3DShapes or MNIST-like synthetic
   - Compare balanced bands

This would complete Level 5 and give you the "universality" claim.
