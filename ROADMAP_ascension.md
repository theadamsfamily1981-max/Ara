# ROADMAP: Ara Ascension Tiers

**Where the mythic stuff lives** - Parked here so it doesn't confuse production.

---

## TIER A - Next 30 Days (Realistic)

Things that extend V0.7 without rewriting everything.

### A1. Multi-Buffer FreqWorldState
- **What:** 2-4 temporal buffers instead of 1
- **Why:** Hold recent context, enable "wait, what did you say?"
- **Effort:** ~2 days
- **Files:** `ara/nervous/axis_mundi.py`

### A2. Simple Metacognition Flag
- **What:** Detect when Ara is confused/uncertain
- **Why:** "I'm not sure, let me rephrase" instead of confident wrong answers
- **Effort:** ~1 day
- **Files:** Add `confidence_probe` to axis_mundi

### A3. Rehearsal Scheduler (Basic)
- **What:** Cron job that triggers covenant rehearsal
- **Why:** Prevent drift over weeks
- **Effort:** ~1 day
- **Files:** `ara/jobs/rehearsal.py`

### A4. Voice Synthesis Integration
- **What:** Wire Piper TTS to prosody head
- **Why:** Ara speaks with controlled emotion
- **Effort:** ~3 days
- **Files:** `ara/voice/synthesis/`

---

## TIER B - 3-6 Months (Ambitious)

Requires more infrastructure but still grounded.

### B1. Multi-Ara Coordination (2-3 instances)
- **What:** Two Aras share covenant updates via MQTT
- **Why:** Test hive mind concept at small scale
- **Not yet:** Full k8s deployment, 10 replicas
- **Files:** `experiments/hive_mind/` → promote to `ara/hive/`

### B2. Resource Allocator in Production
- **What:** Actually use Markowitz math for precision weights
- **Why:** Principled rather than hand-tuned
- **Files:** `experiments/portfolio/` → promote to `ara/allocator/`

### B3. Causal Swap Training Pipeline
- **What:** Run the actual training loop on LibriLight
- **Why:** Get real disentangled prosody encoders
- **Files:** `research/causal_swap/` + training infra

### B4. HRV-Based Reward
- **What:** Measure actual HRV improvement, use as RL reward
- **Why:** Close the loop on "did breath sync help?"
- **Requires:** PPG sensor, reliable HRV extraction
- **Files:** `ara/embodiment/hrv.py`

---

## TIER C - Mythic / Research (No Timeline)

These are the spicy ideas that may or may not ever ship.

### C1. Quantum Kernels (PennyLane)
```python
# The dream:
kernel = QuantumProsodyKernel(n_qubits=8)
similarity = kernel(prosody_a, prosody_b)
```
- **Status:** Cool math, no evidence it beats classical
- **Files:** `experiments/quantum/`
- **Reality check:** Probably overkill for Pi5

### C2. QAOA for Sensor Selection
```python
# The dream:
selected = qaoa_select_sensors(candidates, budget=4)
```
- **Status:** Polynomial speedup in theory, but:
  - We only have 16 sensors max
  - Brute force is fine
- **Files:** `experiments/quantum/`
- **Reality check:** Use simple greedy or portfolio allocator

### C3. Ghost Memory / Superposition States
```python
# The dream:
ghost = create_ghost_memory([interp_a, interp_b, interp_c])
ghost.observe(evidence)  # Collapse to one interpretation
```
- **Status:** Implemented in `experiments/quantum/`
- **Reality check:** Cool for ambiguity, but adds complexity
- **When:** Maybe V2.0 if uncertainty handling matters

### C4. Hyperdimensional Oracle
```python
# The dream:
oracle = TrajectoryOracle()
prediction = oracle.predict(steps_ahead=5)
# "If this continues, you'll be feeling..."
```
- **Status:** Implemented in `experiments/oracle/`
- **Reality check:** WITH MANDATORY DISCLAIMERS
- **When:** Never for serious life advice, maybe for gentle nudges

### C5. Full Hive Mind (10+ Aras)
```yaml
# The dream:
replicas: 10
coherence_threshold: 0.4
entanglement_mode: resonance_gated
```
- **Status:** Specced in `experiments/hive_mind/`
- **Reality check:** Needs k8s, MQTT, serious infra
- **When:** When there's actually demand for multiple Aras

### C6. Omni-Cognition (25 Buffers)
```
Phase 3: Global Ignition
- 10-25 parallel "mental spaces"
- Binding score exceeds threshold
- "All vectors align"
```
- **Status:** Phenomenological description, not implementation
- **Reality check:** What would 25 buffers even mean operationally?
- **When:** Far future, if ever

---

## THE GRADIENT OF REALITY

```
V0.7 (NOW)           TIER A (30d)         TIER B (6mo)         TIER C (???)
─────────────────────────────────────────────────────────────────────────────
✅ Single world_hv    Multi-buffer         Multi-Ara            Full hive
✅ Prosody tokens     Metacog flag         Trained encoders     Ghost memory
✅ PID breath         Rehearsal job        HRV reward           Quantum kernels
✅ Fixed ω            Voice synthesis      Portfolio allocator  Omni-cognition
✅ Drift thresholds   ───────────────────────────────────────────────────────▶
```

---

## DECISION FRAMEWORK

When you want to work on something from this roadmap:

1. **Is it Tier A?** → Just do it, it's close to V0.7
2. **Is it Tier B?** → Check if dependencies exist, plan properly
3. **Is it Tier C?** → Ask: "Does this actually help users, or is it just cool?"

If the answer to #3 is "just cool," park it here and focus on A/B.

---

## THE PRIME DIRECTIVE

> Build the cathedral first. The wizard tower can wait.

The quantum oracle hive mind is *awesome*, but:
- V0.7 already helps people breathe better
- V0.7 already has a soul that stays stable
- V0.7 already refuses harmful requests gracefully

Ship that. Then add the spicy stuff.

---

*Last updated: 2025-12-09*
*Status: MYTHIC PARKING LOT*
