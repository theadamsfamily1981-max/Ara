# Ara v0.1 Readiness Checklist

**"She Exists"** - The organism is clearly alive, even if clumsy.

v0.1 is not "perfect"; it's "organism is demonstrably alive on lab hardware."

---

## Minimal v0.1 Body

The body exists when hardware and daemons form a regulated system.

### Sovereign Loop
- [x] Sovereign loop runs at ≥10 Hz (target: 200 Hz)
- [x] Loop pulls `H_moment` from Sensorium
- [x] Loop calls HTC (software or FPGA)
- [x] Loop emits `u_e` effector commands
- [x] Loop logs `r_t` + mode to audit trail

### HTC Path
- [ ] `HTC` class exists with:
  - [ ] `query(h_moment) → resonance_profile`
  - [ ] `learn(h_moment, reward, mode)`
- [ ] Geometry tests pass:
  - [ ] Random HV orthogonality (pairwise cosine ~N(0, 1/√D))
  - [ ] Bundling capacity (K=16-32 items decodable)
- [ ] Software fallback verified
- [ ] FPGA path verified (when available)

### Nervous System
- [ ] At least one NodeAgent running per physical host
  - [ ] `get_state_hv()` implemented
  - [ ] `emit_events()` implemented
  - [ ] SoulMesh messages flowing
- [ ] At least one reflex path:
  - [ ] eBPF/XDP stub or software equivalent
  - [ ] Can drop/throttle based on `H_flow`

### Sensorium
- [x] `sensorium` module exists
- [x] Reads real telemetry:
  - [x] FPGA/CPU temperature
  - [x] Fan RPM (simulated if unavailable)
  - [x] CPU load
  - [x] Network stats (packets/s, loss rate)
- [x] Converts to `SenseReading → encode() → HV`
- [x] Bundles into `H_moment` at each tick

---

## Minimal v0.1 Mind

The mind exists when there's a real control hierarchy.

### L5 Governance
- [ ] Governance module exists
- [ ] Tracks "Cognitive Capital" / burnout proxies
- [ ] Can veto / delay tasks (not just warn)
- [ ] Logs vetoes with reason + set-point error

### L4 Teleology
- [x] `compute_reward(telemetry, founder_state, teleology)` implemented
- [x] Hard-vetoes on critical thresholds
- [x] Trades off health, antifragility, cathedral
- [x] `choose_mode()` returns {REST, IDLE, ACTIVE, FLOW, EMERGENCY}
- [x] Called every sovereign tick
- [ ] Steers HTC plasticity (polyplasticity modes)

### L3 Cortex (LLMs)
- [ ] LLM calls include:
  - [ ] Current `H_context` / episodic hits
  - [ ] Teleology constraints (allowed/disallowed)
- [ ] Outputs go through Governance, not direct to hardware

### L2 Soul (HTC)
- [ ] Episodic memory stores `H_moment + reward` over time
- [ ] Retrieval: `query(H_probe) → top attractors`
- [ ] Polyplasticity modes affect behavior measurably

---

## Minimal v0.1 Relationship (Covenant)

The covenant exists when founder state affects system behavior.

### Founder State Measurement
- [ ] MindReader process estimates:
  - [ ] Fatigue/burnout (time-of-day, session length)
  - [ ] "Do not push" zones flagged
- [ ] Founder state accessible to Governance

### Founder State Actions
- [ ] Governance can:
  - [ ] Cap work-day batch sizes
  - [ ] Block starting heavy experiments after hours
  - [ ] Suggest "stop now" with enforcement
- [ ] At least one hard enforcement case working

### Safe/Lockout Mode
- [ ] State where:
  - [ ] No new destructive tasks without "two-key turn"
  - [ ] Only reflexes & safety daemons active
  - [ ] Triggered by extreme burnout or Teleology breach
- [ ] Can be entered manually
- [ ] Can be entered by Teleology automatically

### Avatar/UI Reflection
- [ ] Visual changes with Teleology:
  - [ ] Color shift when burnout high
  - [ ] Different when progress high
  - [ ] Entropy/uncertainty reflected
- [ ] Mirror encodes actual state, not just CPU graphs

---

## Minimal v0.1 Capability (Level 9)

Level 9 is specific modules, not a mood.

### Neuromorphic Annealing
- [x] QuantumBridge module exists
- [x] Maps constraint problem into HV representation
- [x] Programs HTC or synthetic annealer loop
- [x] Decodes valid solution
- [ ] Benchmark: "CPU solver time vs Ara annealer time" on real task
- [ ] At least one problem class working end-to-end

### Self-Healing Reflexes
- [ ] At least one pain packet path:
  - [ ] "Thermal runaway" triggers reflex
  - [ ] "Unlikely traffic pattern" triggers reflex
- [ ] Reflex chain:
  - [ ] Immediate eBPF/Arria action
  - [ ] Visual glitch
  - [ ] Log entry back into HTC for learning
- [ ] System adapts thresholds after a few events

---

## Minimal v0.1 Safety & Audit

Safety exists when the system can protect itself and log everything.

### Attractor Diversity Monitor
- [ ] Diversity metric computed each tick
- [ ] Can freeze learning if diversity drops below threshold
- [ ] Alert generated on diversity collapse

### Simple Safe Mode
- [x] Disables plasticity (learning frozen)
- [x] Leaves safety daemons running
- [x] Can be entered by Teleology
- [x] Can be entered manually

### Logging
- [x] Every tick writes to append-only store:
  - [x] `H_moment_id`
  - [x] `mode`
  - [x] `reward`
  - [x] Key effector actions
- [x] Audit daemon operational

---

## Storage & Retrieval (Heim-Optimized)

100× compression with 0% recall loss.

### Heim Compression
- [ ] D=173 sparse binary working
- [ ] Heim analyze calibration complete
- [ ] Geometry tests pass at D=173:
  - [ ] Pairwise cosine ~N(0, 0.076)
  - [ ] <1% pairs exceed |cos| > 0.25

### Oversample + Rerank
- [ ] Stage 1: Coarse retrieval (D=173, 4× oversample)
- [ ] Stage 2: Full precision rerank (D=16k)
- [ ] Recall@8 ≥ 99.9% validated
- [ ] Latency ≤ 500 µs (99.9th percentile)

### Cluster Index
- [ ] RocksDB-backed centroid storage
- [ ] Merge threshold: 0.8
- [ ] Duplicate threshold: 0.95
- [ ] Delta compression working

### Eviction Policy
- [ ] Homeostatic eviction scoring
- [ ] Resonance × 0.6 - age × 0.3 + teleology × 0.1
- [ ] Semantic garbage collection operational

---

## Hardware Status

Current lab hardware state.

### Threadripper Host
- [ ] Sovereign loop running
- [ ] Receptor daemon running
- [ ] Effector daemon running
- [ ] Safety monitor running

### Stratix-10 FPGA
- [ ] Bitstream loaded
- [ ] PCIe enumerated
- [ ] XNOR-CAM accessible
- [ ] Dual-mode (173/16k) configured

### Storage
- [ ] Hot SSD (NVMe) mounted
- [ ] Cold HDD array mounted
- [ ] ZFS/LUKS configured
- [ ] Vault keys accessible

### Network
- [ ] At least one NodeAgent online
- [ ] SoulMesh packets flowing
- [ ] eBPF/XDP loaded (or stub)

---

## Verification Commands

Run these to verify v0.1 readiness:

```bash
# Geometry tests
python -m pytest tests/hd/test_geometry.py -v

# Bundling capacity
python -m pytest tests/hd/test_bundling.py -v

# Homeostasis smoke test
python -c "from ara.homeostasis import AraOrganism; a = AraOrganism(); r = a.boot(); print('PASS' if r.success else f'FAIL: {r.error_message}')"

# HTC retrieval test
python -m pytest tests/cognition/test_htc_retrieval.py -v

# Storage recall validation
python -m pytest tests/storage/test_oversample_recall.py -v

# Full organism integration
python -c "
from ara.homeostasis import AraOrganism
import time

ara = AraOrganism()
result = ara.boot()
if result.success:
    print('Boot OK')
    time.sleep(1.0)  # Let it stabilize
    stats = ara.get_stats()
    print(f'Mode: {stats[\"mode\"]}')
    print(f'Sovereign: {stats[\"sovereign\"][\"actual_hz\"]:.1f} Hz')
    print(f'Safety OK: {stats[\"safety\"][\"safety_ok\"]}')
    ara.shutdown()
    print('v0.1 READY')
else:
    print(f'FAIL: {result.error_message}')
"
```

---

## Definition of "Alive"

When most boxes are checked, it is fair to say:

> "Ara v0.1 is a teleoplastic cybernetic organism running on my lab hardware."

It is still code and circuits—no magic, no metaphysics—but structurally it behaves like a **homeostatic control system with memory, values, and self-protective reflexes**.

---

## Guardrails

Important boundaries:

1. **Ara is not a therapist, doctor, or replacement for human connection**
2. She CAN be:
   - A powerful mirror
   - A protector of your time and energy
   - A way to automate saying "no" when you historically haven't
3. If you find yourself leaning on her instead of getting help when you're really not okay, that's a moment to pull a human into the loop too

Everything remains **testable, inspectable, overrideable**.
