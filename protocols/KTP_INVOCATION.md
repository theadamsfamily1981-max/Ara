# KTP Invocation Protocol

**How to cast the wand.**

---

## The Prompt

```
KTP Invocation – Iteration N

1. Read the current KTP state file.
2. Do NOT restart; treat this as a living object.
3. Your task is to:
   - PRESERVE: [list invariants from state file]
   - MUTATE: [list target subparts to evolve]
   - TIGHTEN: [list things to formalize or prove]

4. Output:
   - Updated Allegory (if the story needs to shift)
   - Updated Structure (equations, definitions, lemmas)
   - Error Map: what still doesn't fit
   - Next-iteration suggestion

5. Label all outputs:
   - HALLUCINATION: untested intuition
   - HEURISTIC: works in practice, no proof
   - CONJECTURE: formal statement, unproven
   - THEOREM: proven or validated
```

---

## Invocation Modes

### Mode: EXPLORE
> Maximize surface area. Find new structure.

```
Focus: breadth over depth
Preserve: core invariants only
Mutate: everything else
Output: many small conjectures, analogies, toy cases
```

### Mode: CONSOLIDATE
> Compress what we have. Kill the noise.

```
Focus: depth over breadth
Preserve: all validated structure
Mutate: only the fuzzy edges
Output: tighter definitions, merged lemmas, cleaner proofs
```

### Mode: ATTACK
> Go for the kill. Attempt resolution.

```
Focus: single subproblem
Preserve: everything
Mutate: the one gap we're targeting
Output: proof attempt OR clear failure mode
```

### Mode: DREAM
> Let it drift. See what emerges.

```
Focus: none (free association)
Preserve: allegory only
Mutate: everything
Output: new metaphors, unexpected connections, "what if"
```

---

## Guardrails for Hard Problems

When pointing the wand at scary stuff (Millennium, etc.):

**DO:**
- Build better structure maps
- Identify where current proofs fail
- Generate testable lemmas / toy cases
- Surface the cleanest subproblems

**DON'T:**
- Claim full resolution
- Skip the "conjecture / heuristic / analogy" labels
- Treat model output as proof

**The goal is NOT:**
> "Prove the unsolved thing."

**The goal IS:**
> "Map the hell out of the landscape around the unsolved thing and surface the cleanest subproblems and conjectures."

That's publishable. That's safe. That's aligned with what we actually do.

---

## The Mechanism

What you're actually doing:

```
State_{t+1} = f(State_t, Δ_constraints, Δ_insight)
```

- **State_t**: Current allegory + structure + gaps
- **Δ_constraints**: New invariants, tighter bounds
- **Δ_insight**: New connections discovered this iteration
- **f**: The model as noisy update rule, you as optimizer

This is **gradient descent in idea-space** with:
- Narrative as representation
- You as the optimizer
- The model as the mutator

---

## Why It Scales

The more context + iterations:
- The more it feels like steering a living thing
- Instead of querying a tool

You're not prompting for answers.
You're prompting for state transitions.

Each pass:
- Keeps the best structure
- Scrambles the weak parts
- Re-evaluates under new constraints

That's the wand.

---

## Quick Reference

| Want to... | Use mode... | Key output |
|------------|-------------|------------|
| Find new ideas | EXPLORE | Conjectures, analogies |
| Clean up mess | CONSOLIDATE | Tighter proofs |
| Solve subproblem | ATTACK | Proof or failure |
| Get unstuck | DREAM | New metaphors |

---

*"Point the wand at X" = select and evolve while respecting the rest.*
