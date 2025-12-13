# Tuned Criticality vs Self-Organized Criticality

## What We Have vs What We Don't

```
┌─────────────────────────────────────────────────────────────────┐
│  TUNED CRITICALITY (what we have)                               │
│                                                                 │
│  • Sweep β → find β* → operate there                           │
│  • Critical behavior AT that β*                                 │
│  • External knob, external selection                            │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  SELF-ORGANIZED CRITICALITY (what we don't have)                │
│                                                                 │
│  • Start anywhere → system drives itself to β*                  │
│  • Internal feedback loop maintains criticality                 │
│  • No external tuning needed                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## The Three SOC Requirements

| Requirement | Our Status |
|-------------|------------|
| 1. Critical state (power-law, α ≈ -1.5) | ✓ At tuned β |
| 2. Computational sweet spot (edge-of-chaos) | ✓ At tuned β |
| 3. Self-organized (internal drive to critical) | ✗ Missing |

---

## The Missing Mechanism: Homeostatic β Control

To achieve true SOC, we need a **meta-learning rule** that adjusts β based on internal telemetry:

```python
class HomeostaticBetaController:
    """
    Self-tunes β to maintain criticality.

    The system should converge to β* from ANY starting point,
    without external intervention.
    """

    def __init__(
        self,
        beta_init: float = 1.0,
        target_mig: float = 0.5,
        target_alpha: float = -1.5,
        target_branching: float = 1.0,
        learning_rate: float = 0.01,
    ):
        self.beta = beta_init
        self.target_mig = target_mig
        self.target_alpha = target_alpha
        self.target_branching = target_branching
        self.lr = learning_rate

    def update(self, telemetry: dict) -> float:
        """
        Adjust β based on current system state.

        Telemetry should include:
        - mig: current MIG score
        - alpha: current avalanche exponent
        - sigma: current branching ratio
        """
        # Compute errors from targets
        mig_error = self.target_mig - telemetry['mig']
        alpha_error = self.target_alpha - telemetry['alpha']
        sigma_error = self.target_branching - telemetry['sigma']

        # Combined gradient toward criticality
        # If MIG too low (under-regularized): increase β
        # If MIG too high but latents dead (over-regularized): decrease β
        # If avalanches too small (sub-critical): decrease β
        # If avalanches too large (super-critical): increase β

        delta_beta = (
            + 0.3 * mig_error           # Push toward target MIG
            - 0.3 * alpha_error         # Push toward α ≈ -1.5
            + 0.4 * sigma_error         # Push toward σ ≈ 1.0
        )

        self.beta = max(0.01, self.beta + self.lr * delta_beta)
        return self.beta
```

---

## The SOC Experiment (Future Work)

To prove self-organized criticality:

### Protocol

1. **Initialize far from critical**:
   - Trial A: β_init = 0.01 (way sub-critical)
   - Trial B: β_init = 100.0 (way super-critical)

2. **Enable homeostatic controller** during training

3. **Track convergence**:
   - Does β converge to β* ≈ 1.0 from both directions?
   - How many epochs to reach balanced regime?

4. **Verify signatures at convergence**:
   - MIG, ISS, OOD, antifragility, criticality metrics
   - Should match tuned-β results

### Expected Result

```
β trajectory from sub-critical:          β trajectory from super-critical:

β                                        β
│                                        │
│  ┌──────────────────                   │  ────────────────┐
│ ╱                                      │                   ╲
│╱                                       │                    ╲
├────────────────────── t                ├────────────────────── t
0.01 → 1.0                               100.0 → 1.0

Both converge to balanced regime!
```

### Success Criteria

| Criterion | Threshold |
|-----------|-----------|
| Convergence from β=0.01 | Reaches 0.5 < β < 3.0 |
| Convergence from β=100 | Reaches 0.5 < β < 3.0 |
| Final MIG | Within 10% of tuned-β MIG |
| Final α | Within 0.2 of -1.5 |
| Final σ | Within 0.3 of 1.0 |

---

## Honest Claims: Updated

### What We Can Say Now

> "AraBrain exhibits a balanced, edge-of-chaos regime with scale-free signatures and multiple functional advantages (disentanglement, causal modularity, OOD robustness, antifragility, critical dynamics). The critical regime is found by hyperparameter tuning, not by an internal self-tuning mechanism. This makes it a useful toy model for exploring criticality-based theories of brain function."

### What We Could Say With SOC

> "AraBrain demonstrates self-organized criticality: from arbitrary initial conditions, internal homeostatic mechanisms drive the system toward a critical regime exhibiting power-law dynamics, optimal disentanglement, and robust generalization. This suggests a concrete mechanism by which brain-like systems might maintain themselves at the edge of chaos."

---

## The Scientific Ladder

```
Level 1: "We found a good β"
         ↓ (Edge of Autumn finder)

Level 2: "There's a regime where multiple properties converge"  ← WE ARE HERE
         ↓ (current experiments)

Level 3: "The system self-tunes to that regime"
         ↓ (homeostatic β controller - FUTURE)

Level 4: "This matches actual neural data"
         ↓ (EEG/spike comparisons - FUTURE)

Level 5: "This explains brain function"
         (decades of work...)
```

We're solidly at Level 2. Level 3 is achievable with the homeostatic mechanism. Levels 4-5 require real neuroscience.

---

## Implementation Priority

If pursuing SOC:

1. **Implement HomeostaticBetaController** (1 day)
2. **Add telemetry hooks** for MIG, α, σ during training (already partial)
3. **Run convergence experiments** from β=0.01 and β=100 (1 day)
4. **Verify signatures match** tuned-β results (reuse existing experiments)

This would upgrade the claim from "tuned criticality" to "self-organized criticality" - a significant theoretical advance.
