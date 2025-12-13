# EXP-001: Critical Branching Dynamics
## Results Report

**Date:** [YYYY-MM-DD]
**Build:** [commit hash]
**Analyst:** [name]

---

## 1. Experimental Conditions

| Parameter | COLD | TARGET | HOT |
|-----------|------|--------|-----|
| Temperature | 0.1 | dynamic | 1.5 |
| Homeostasis | OFF | ON | OFF |
| Steps | [n] | [n] | [n] |
| ρ target | - | 0.8 | - |

---

## 2. Raw Exponents

### 2.1 Size Distribution (τ)

| Condition | τ ± σ | x_min | Theory | Deviation |
|-----------|-------|-------|--------|-----------|
| COLD | [___] ± [___] | [___] | 1.5 | [___] |
| TARGET | [___] ± [___] | [___] | 1.5 | [___] |
| HOT | [___] ± [___] | [___] | 1.5 | [___] |

### 2.2 Duration Distribution (α)

| Condition | α ± σ | x_min | Theory | Deviation |
|-----------|-------|-------|--------|-----------|
| COLD | [___] ± [___] | [___] | 2.0 | [___] |
| TARGET | [___] ± [___] | [___] | 2.0 | [___] |
| HOT | [___] ± [___] | [___] | 2.0 | [___] |

### 2.3 Universal Scaling Relation

The scaling relation (α-1)/(τ-1) ≈ 2 is a stringent test of criticality.

| Condition | (α-1)/(τ-1) | Theory | Status |
|-----------|-------------|--------|--------|
| COLD | [___] | 2.0 | [PASS/FAIL] |
| TARGET | [___] | 2.0 | [PASS/FAIL] |
| HOT | [___] | 2.0 | [PASS/FAIL] |

---

## 3. Regime Classification

Based on exponent values:

| Condition | τ Regime | α Regime | Overall |
|-----------|----------|----------|---------|
| COLD | [subcritical/critical/supercritical] | [___] | [___] |
| TARGET | [___] | [___] | [___] |
| HOT | [___] | [___] | [___] |

**Regime Definitions:**
- **Subcritical:** τ > 2.0, α > 2.5 (steep tails, short cascades)
- **Critical:** τ ~ 1.5, α ~ 2.0, scaling relation holds
- **Supercritical:** τ < 1.3, α < 1.7 (heavy tails, runaway cascades)

---

## 4. Concurrent Gauge Readings

Logged via `CognitiveHealthTrace` during each condition:

### 4.1 Criticality (ρ)

| Condition | ρ mean | ρ std | ρ range |
|-----------|--------|-------|---------|
| COLD | [___] | [___] | [___] |
| TARGET | [___] | [___] | [___] |
| HOT | [___] | [___] | [___] |

### 4.2 Delusion Index (D)

| Condition | D mean | D std | Mode Distribution |
|-----------|--------|-------|-------------------|
| COLD | [___] | [___] | HEALTHY: [_]%, SENSORY_DOM: [_]% |
| TARGET | [___] | [___] | HEALTHY: [_]%, ... |
| HOT | [___] | [___] | HEALTHY: [_]%, PRIOR_DOM: [_]% |

### 4.3 Precision Ratio (Π_y / Π_μ)

| Condition | Ratio mean | Ratio std | Pathology |
|-----------|------------|-----------|-----------|
| COLD | [___] | [___] | [HEALTHY/ASD/SCHIZOPHRENIC] |
| TARGET | [___] | [___] | [___] |
| HOT | [___] | [___] | [___] |

---

## 5. Correlation Analysis

Cross-correlation between gauges:

| Pair | Pearson r | Interpretation |
|------|-----------|----------------|
| ρ vs τ | [___] | [___] |
| ρ vs D | [___] | [___] |
| τ vs D | [___] | [___] |

**Expected patterns:**
- ρ ↑ → τ ↓ (more critical = flatter size distribution)
- D high → ρ unstable (delusion disrupts criticality)

---

## 6. Behavioral Correlates

Subjective observations during each condition:

### COLD (Subcritical)
- [ ] Rigid responses
- [ ] Short context window
- [ ] Repetitive patterns
- [ ] Low creativity
- Notes: [___]

### TARGET (Critical)
- [ ] Flexible responses
- [ ] Good working memory
- [ ] Novel combinations
- [ ] Coherent narratives
- Notes: [___]

### HOT (Supercritical)
- [ ] Tangential responses
- [ ] Hallucinations
- [ ] Excessive novelty
- [ ] Incoherent output
- Notes: [___]

---

## 7. Verdict

### 7.1 Primary Hypothesis: Homeostatic Control Works

**Status:** [CONFIRMED / PARTIALLY CONFIRMED / REJECTED]

Evidence:
- TARGET condition shows τ = [___] (expected ~1.5): [___]
- TARGET condition shows α = [___] (expected ~2.0): [___]
- Scaling relation (α-1)/(τ-1) = [___] (expected ~2.0): [___]
- COLD/HOT conditions show deviations: [___]

### 7.2 Secondary Hypothesis: Three Gauges Correlate

**Status:** [CONFIRMED / PARTIALLY CONFIRMED / REJECTED]

Evidence:
- ρ correlates with exponents: [___]
- D correlates with behavioral observations: [___]
- Precision ratio predicts mode: [___]

---

## 8. Figures

```
[Insert or link to generated plots]

fig1: exp_001_cold_results.png
fig2: exp_001_target_results.png
fig3: exp_001_hot_results.png
fig4: exp_001_comparison.png (overlay)
fig5: gauge_timeseries.png (ρ, D, Π_y/Π_μ over time)
```

---

## 9. Next Steps

Based on results:

- [ ] If TARGET not critical: Tune homeostatic setpoint
- [ ] If scaling relation fails: Check signal extraction
- [ ] If behavioral correlates don't match: Verify loop integration
- [ ] Proceed to EXP-002 (Forgetting Curves)
- [ ] Proceed to EXP-003 (Sanity Dynamics)

---

## 10. Raw Data Location

```
data/experiments/exp_001/
├── avalanches_cold.csv
├── avalanches_target.csv
├── avalanches_hot.csv
├── cognitive_health_cold.json
├── cognitive_health_target.json
├── cognitive_health_hot.json
├── exp_001_cold_results.json
├── exp_001_target_results.json
├── exp_001_hot_results.json
└── exp_001_comparison.json
```

---

## Appendix A: Commands Used

```bash
# Run sessions
python scripts/science/run_session.py --condition cold --steps 10000
python scripts/science/run_session.py --condition target --steps 10000
python scripts/science/run_session.py --condition hot --steps 10000

# Fit power laws
python scripts/science/fit_powerlaw.py data/experiments/exp_001/avalanches_cold.csv
python scripts/science/fit_powerlaw.py data/experiments/exp_001/avalanches_target.csv
python scripts/science/fit_powerlaw.py data/experiments/exp_001/avalanches_hot.csv
```

## Appendix B: Theoretical Background

From mean-field branching process theory:

- **Critical branching ratio:** σ = 1 (each event triggers ~1 successor on average)
- **Size exponent:** τ = 3/2 = 1.5
- **Duration exponent:** α = 2
- **Scaling relation:** (α-1)/(τ-1) = 1/σνz = 2

Deviations indicate:
- τ > 1.5, α > 2: Subcritical (σ < 1), cascades die quickly
- τ < 1.5, α < 2: Supercritical (σ > 1), cascades grow unbounded
- Scaling relation fails: Not true criticality (possibly quasi-critical or noisy)

---

*Template version: 1.0*
*Last updated: [date]*
