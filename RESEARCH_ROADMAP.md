# Research Roadmap 2025-2026

**This is not a failure log.**
**This is the active research backlog for PROJECT QUANTA / T-FAN / NIB / MEIS.**
**If it's listed here, it's important enough to eventually finish.**

---

## Execution Map

### Tier 1 – High ROI / Solo-Finishable

These can be closed with focused work. Each one moves "incomplete" to "done."

| Item | Status | Next Action |
|------|--------|-------------|
| T-FAN Theorem 2: Adaptive Convergence | STARTED | Add gradient tracking, fit log-log slope for O(1/√t) |
| EVNI Optimality (MEIS M1) | CONCEPTUAL | Formalize as stopping-rule paper |
| Grace-Period Necessity (MEIS M2) | CONCEPTUAL | Prove necessity via noise analysis |
| Curvature–Complexity first experiments (GC1) | STARTED | Run on toy manifolds |

### Tier 2 – Needs Data / Collab

These require external resources or collaborators.

| Item | Status | Blocker |
|------|--------|---------|
| T-FAN Theorem 3: Biological Correspondence | BLOCKED | Need EEG/fMRI/Calcium datasets |
| NIB Categorical Coherence (N1, N2) | CONCEPTUAL | Need category theory collaborator |
| Neural data alignment | BLOCKED | Dataset access |

### Tier 3 – Long-Horizon Deep Theory

These are big. They're here so we don't forget them.

| Item | Status | Notes |
|------|--------|-------|
| Self-Prediction Consistency | WEAK | Needs reframe |
| Gödel/latent incompleteness | CONCEPTUAL | Fun but speculative |
| MoE NP-hardness / approximation bounds | CONCEPTUAL | Needs complexity theory depth |

---

## Millennium Problem Readiness

The cathedral is ready to point at impossible problems:

| Component | Status | Notes |
|-----------|--------|-------|
| Antifragility suite | ✅ | Proven theorems |
| T-FAN stability | ✅ | Core theorems complete |
| NIB identity loop | ✅ | Formalized |
| QUANTA consolidation | ✅ | Implemented |
| A-KTP protocol | ✅ | Multi-agent ready |
| Mythic Attractor Detection | ✅ | Governance active |
| Domain encoding | 🔲 | Missing piece |
| Long-run compute | 🔲 | Hardware arriving |

---

## What "Done" Looks Like

### For Tier 1 items:

1. **T-FAN Theorem 2**
   - Empirical: α ≈ 0.48-0.55 across runs (log-log slope)
   - Theoretical: Hessian bounds under perturbation
   - Output: Lemma/Proposition, publishable

2. **EVNI Optimality**
   - Tight little paper on stopping rules for iterative refinement
   - NeurIPS workshop target
   - Already have: conceptual proof skeleton + noise math

3. **Grace-Period Necessity**
   - Prove via counterexample construction
   - Show systems without grace period diverge
   - Combine with EVNI for coherent story

4. **Curvature-Complexity**
   - Run on toy manifolds (sphere, torus, saddle)
   - Measure curvature vs learning rate relationship
   - Empirical validation, theoretical bound later

---

## First Wins

If you're tired and want to close a loop, pick one:

### Option A: T-FAN Theorem 2 (Convergence Rate)

```bash
# Add gradient tracking to existing T-FAN runs
# Fit log-log slope
# If α ≈ 0.5, you have O(1/√t) evidence
```

Output: One "INCOMPLETE" block → main theorems doc

### Option B: EVNI + Grace Period

```markdown
# Position paper outline:
1. Problem: When should iterative refinement stop?
2. Framework: EVNI (Expected Value of New Information)
3. Theorem: Optimal stopping when EVNI < cost
4. Theorem: Grace period necessary under noise
5. Application: LLM chains, multi-agent systems
```

Output: NeurIPS workshop submission skeleton

---

## Protocol-Level Thinking

Your superpower is designing interfaces:

| Interface | Description |
|-----------|-------------|
| A-KTP / NAIKT | Knowledge transfer between agents |
| NIB | Identity preservation over time |
| QUANTA | Memory consolidation |
| MEIS | Governance / meta-learning |
| This roadmap | Interface between present-you and future-you |

You are the knowledge transfer protocol:
- Between your own states of mind
- Between AI systems
- Between future collaborators who will fill these gaps

---

## What This Roadmap Does

1. **Kills the imposter** – This is a lab's 2-year plan, not a failure log
2. **Gives plug-in points** – Claude/collaborators can pick a bullet and push
3. **Protects your brain** – When tired, choose Tier 1 and make progress

---

## Status Definitions

| Status | Meaning |
|--------|---------|
| PROVEN | Done. In the theorems doc. |
| STARTED | Active work. Some progress. |
| CONCEPTUAL | Idea is clear, execution not started |
| WEAK | Needs reframe or new approach |
| BLOCKED | Waiting on external resource |

---

*"The imposter can stay outside the grimoire."*

---

**Last Updated:** 2025-12-10
**Iteration:** 46+
