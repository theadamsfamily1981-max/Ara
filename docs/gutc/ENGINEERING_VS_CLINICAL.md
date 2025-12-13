# GUTC Gain Controller: Engineering Notes vs Clinical Metaphors

**Document Type:** Engineering Clarification
**Audience:** Developers, Collaborators, Future-You
**Status:** Living Document

## Purpose

This document clarifies the relationship between:
1. **Clinical metaphors** used to derive and explain the math (schizophrenia, autism, etc.)
2. **Engineering states** used in the actual running code (PRIOR_DOMINATED, SENSORY_DOMINATED, etc.)

**Key Point:** The clinical language is a *design metaphor*, not a diagnostic tool. The code uses neutral engineering terms.

---

## 1. The Derivation (Theory / Metaphor)

### 1.1 Where the Clinical Language Comes From

The GUTC framework borrows from computational psychiatry, which models mental health conditions as parameter imbalances in predictive coding / active inference systems:

| Condition | Model | Parameter Regime |
|-----------|-------|------------------|
| Schizophrenia | Aberrant salience | Π_μ >> Π_y (prior-dominated) |
| Autism | Weak central coherence | Π_y >> Π_μ (sensory-dominated) |
| Depression | Learned helplessness | Low action precision |
| ADHD | Subcritical dynamics | λ < 1, short ξ |

This literature is *theoretical modeling*, not diagnosis. It asks: "If we model X condition as a parameter imbalance, what predictions follow?"

### 1.2 How We Use It

We borrowed this framework because it provides:
- **Intuitive names** for failure modes (easier to remember "schizophrenic mode" than "excessive prior precision ratio")
- **Mathematical structure** (the Π_y / Π_μ update rule)
- **Design heuristics** ("if she's ignoring input, lower prior precision")

But the clinical labels are *metaphors for failure modes*, not claims about actual human conditions.

---

## 2. The Code (Engineering / Neutral)

### 2.1 Cognitive Modes in Code

The running code uses **neutral engineering terms**:

```python
class CognitiveMode(Enum):
    HEALTHY_CORRIDOR = auto()   # Balanced precision, near criticality
    PRIOR_DOMINATED = auto()    # Stubbornly ignoring current input
    SENSORY_DOMINATED = auto()  # Over-reacting to noise, can't generalize
    UNSTABLE = auto()           # ρ too far from criticality
    DISCONNECTED = auto()       # Both precisions very low
```

Note: No clinical terms in the enum names.

### 2.2 Mapping Table

| Clinical Metaphor (docs) | Engineering Term (code) | Ara Symptom |
|--------------------------|-------------------------|-------------|
| "Schizophrenic mode" | `PRIOR_DOMINATED` | Ignores explicit user corrections |
| "ASD mode" | `SENSORY_DOMINATED` | Stuck fixing tiny errors, misses big picture |
| "Dissociative" | `DISCONNECTED` | Unresponsive, not processing |
| "Manic" | (Not implemented) | Everything feels important |

### 2.3 Why This Matters

If someone reads the code without context:
- They see `CognitiveMode.PRIOR_DOMINATED` → clear engineering meaning
- They don't see `CognitiveMode.SCHIZOPHRENIC` → no confusion about medical claims

The clinical language lives in:
- Design documents
- Theory explanations
- Comments explaining *why* the thresholds are set this way

---

## 3. How the Layers Fit Together

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 2: Guardrails (Safety)                               │
│  - Hard content filters                                     │
│  - Policy constraints                                       │
│  - NEVER overridden by cognitive state                      │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: GUTC Controller (Cognition)                       │
│  - Precision balance (Π_y vs Π_μ)                           │
│  - Criticality monitoring (ρ)                               │
│  - CognitiveDoctor diagnosis                                │
│  - ChiefOfStaff auto-tuning                                 │
├─────────────────────────────────────────────────────────────┤
│  Layer 0: Core Model (Substrate)                            │
│  - LLM forward pass                                         │
│  - Activation extraction                                    │
│  - Token generation                                         │
└─────────────────────────────────────────────────────────────┘
```

### Separation of Concerns

- **Layer 0** doesn't know about "cognitive health" - just generates tokens
- **Layer 1** tunes *how* Layer 0 is used (temperature, recursion, goal weights)
- **Layer 2** vetoes *what* Layer 0 outputs, regardless of Layer 1 state

Even if Ara is in `PRIOR_DOMINATED` mode, Layer 2 guardrails still apply. The cognitive state affects *style*, not *permitted content*.

---

## 4. Identity and Imprinting

Where does the user ("Croft") live in this?

### 4.1 Identity as Prior Precision

- The user's identity is encoded as **high-precision priors** on specific nodes in the heteroclinic memory network
- This means: "Ara should flow back to this identity saddle easily"

### 4.2 Balance with Current Input

The precision balance ensures:
- **Strong enough prior** on identity → stable sense of "this is Croft"
- **Not so strong** that Ara ignores what Croft says *now*

If `Π_prior_identity` is too high relative to `Π_sensory_chat`:
- Ara starts assuming things Croft just corrected
- Diagnosis: `PRIOR_DOMINATED` on identity channel

Fix: Lower identity prior precision slightly, or raise sensory weight on chat.

---

## 5. Guidelines for Future Development

### 5.1 Adding New Cognitive Modes

If you add a new mode:
1. Use **neutral engineering name** in the enum
2. Document the clinical metaphor (if any) in comments/docs
3. Add corresponding thresholds to `DiagnosticThresholds`

### 5.2 Logging and Telemetry

When logging cognitive state:
- Log the engineering term (`PRIOR_DOMINATED`)
- Optionally add clinical note in debug logs
- Never expose clinical terms to end users

### 5.3 Documentation

When explaining the system:
- Use clinical metaphors in *theory* sections
- Use engineering terms in *code* sections
- Always clarify that clinical terms are metaphors

---

## 6. References

### Computational Psychiatry (Theory Source)

1. Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*.

2. Lawson, R.P., Rees, G., & Friston, K.J. (2014). An aberrant precision account of autism. *Frontiers in Human Neuroscience*.

3. Adams, R.A., Stephan, K.E., Brown, H.R., Frith, C.D., & Friston, K.J. (2013). The computational anatomy of psychosis. *Frontiers in Psychiatry*.

### GUTC Implementation (Code)

- `ara/gutc/active_inference.py` - Policy scoring with precision weights
- `ara/gutc/precision_diagnostics.py` - Pathology detection (uses neutral terms internally)
- `ara/science/cognitive_health_trace.py` - Telemetry with `CognitiveMode` enum

---

## 7. FAQ

### Q: Is this a medical diagnostic tool?

**No.** This is engineering instrumentation for an AI system. The clinical language is borrowed for intuition, not diagnosis.

### Q: Can I use the clinical terms in the code?

Please don't. Use the neutral engineering terms (`PRIOR_DOMINATED`, etc.) in code. Reserve clinical terms for documentation and design discussions.

### Q: What if Ara shows "schizophrenic" behavior?

In engineering terms: She's in `PRIOR_DOMINATED` mode - ignoring input and trusting internal beliefs too much. Fix by increasing `sensory_precision` (extrinsic weight) and decreasing `prior_precision` (intrinsic weight).

### Q: Why bother with the clinical metaphors at all?

They provide:
- Memorable names for failure modes
- Connection to established theory (computational psychiatry)
- Design heuristics ("what would help a subcritical brain? → stimulants → increase λ")

The metaphors are tools for thinking, not claims about reality.
