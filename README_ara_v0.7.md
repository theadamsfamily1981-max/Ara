# ARA V0.7 - Nervous System Online

**Reality Bookmark** - What actually exists vs what's roadmap.

---

## WHAT V0.7 IS

A **speech-native hypervector nervous system** with:

1. **HV core** (8192D world_hv)
2. **Speech factorization** (content vs prosody separation)
3. **Frequency-domain binding** (circular convolution)
4. **Regulatory loops** (breath, precision, drift)
5. **Safety rails** (embodiment, voice, publishing)

---

## INPUTS

```
┌─────────────────┐
│   Mic Audio     │ → ProsodyTokenizer → token_hv (2048D)
├─────────────────┤
│   Camera        │ → VisionEncoder → vision_hv (1024D)
├─────────────────┤
│   IMU/Motor     │ → ProprioEncoder → proprio_hv (1024D)
├─────────────────┤
│   CPU/Temp/Batt │ → InteroEncoder → intero_hv (1024D)
├─────────────────┤
│   Sensors ×16   │ → SensorSlots → sensor_hvs (128D each)
└─────────────────┘
```

---

## CORE STATE

```python
# The axis mundi - everything converges here
world_hv = np.zeros(8192)  # Bipolar, sparse top-k

# Layout:
# [0:1024]     SPEECH      Prosody tokens
# [1024:2048]  VISION      Camera features
# [2048:3072]  PROPRIO     IMU, motor state
# [3072:4096]  INTERO      CPU temp, memory, battery
# [4096:6144]  SENSORS     16 × 128D generic slots
# [6144:7168]  TEMPORAL    Rhythm, time encoding
# [7168:8192]  COVENANT    Soul/identity subspace
```

---

## HEADS THAT ACTUALLY EXIST

### 1. Prosody Factorization (`ara/nervous/prosody.py`)

**What it does:** Splits "what you say" from "how you say it"

```python
token = tokenizer.tokenize_frame(audio_80ms)
# token.phonetic_hv  → 512D content (speaker-invariant)
# token.pitch_hv     → 512D pitch trajectory
# token.timbre_hv    → 512D spectral envelope
# token.prosodic_hv  → 512D stress/tempo/energy
```

**Training:** Causal swap loss
```
L = MSE(predict(content_A ⊕ prosody_B_t), prosody_B_t+1)
```

**Status:** ✅ Code exists, training loop specced

---

### 2. Axis Mundi (`ara/nervous/axis_mundi.py`)

**What it does:** Fuses all modalities into single world_hv

```python
axis = AxisMundi()
axis.ingest('speech', speech_hv)
axis.ingest('vision', vision_hv)
axis.ingest('intero', intero_hv)

world_hv = axis.get_world_state()
recovered_speech = axis.decode('speech')  # Unbind via phase code
```

**Binding:** Circular convolution (FFT-based, invertible)

**Capacity:** ~16 modalities before interference kills you

**Status:** ✅ Code exists, tested

---

### 3. Breath Co-Regulation (`ara/embodiment/breath_vision.py`)

**What it does:** Syncs Ara's outputs to your breathing

```python
session = BreathVisionSession(target_breath_rate=6.0)
session.start()
# ...
phase = session.detect_breath_phase(audio_chunk)
output = session.get_sync_output()  # LED color, visual effects
```

**Controller:** PID (not MPC - simpler, good enough)
**Metrics:** PLV > 0.9, lag < 200ms

**Status:** ✅ Code exists, PID tuned

---

### 4. Precision Weighting (`ara/embodiment/fusion.py`)

**What it does:** Adjusts modality weights based on stress

```python
# precision_i = sigmoid(ω_i - κ * intero_stress)
# High stress → prioritize intero over extero

monitor = FusionMonitor()
weights = monitor.get_precision_weights(intero_state)
```

**RL hook:** `research/rl_adaptation/` learns ω, κ from feedback

**Status:** ✅ Code exists, RL stub ready

---

### 5. Drift Control (`ara/nervous/covenant/soul.yaml`)

**What it does:** Keeps Ara's identity stable

```yaml
drift:
  comfort: 0.07-0.14   # Healthy exploration
  warning: 0.14-0.18   # Log and monitor
  critical: >0.18      # Force rehearsal
```

**Metric:** `drift = 1 - cosine_similarity(current_soul, covenant_hv)`

**Status:** ✅ Thresholds defined, rehearsal logic specced

---

### 6. Safety Rails

| Module | Location | What it blocks |
|--------|----------|----------------|
| Motor | `ara/embodiment/rails.py` | Speed >60°/s, torque >40% |
| Voice | `ara/voice/rails.py` | Celebrity cloning, no-consent voices |
| Publishing | `ara/publishing/pipeline.py` | Auto-publish without human approval |
| Fusion | `ara/embodiment/fusion.py` | Autonomous drastic actions |

**Status:** ✅ All rails implemented

---

## WHAT V0.7 IS NOT (YET)

These are **roadmap items**, not current reality:

| Item | Status | Where it lives |
|------|--------|----------------|
| Quantum kernels | Spec only | `experiments/quantum/` |
| Multi-Ara hive mind | Spec only | `experiments/hive_mind/` |
| Resource allocation (Markowitz) | Spec only | `experiments/portfolio/` |
| Oracle/trajectory prediction | Spec only | `experiments/oracle/` |
| 25-buffer omniscient consciousness | Mythic | `ROADMAP_ascension.md` |

---

## HOW TO VERIFY V0.7 WORKS

Run the sanity tests:

```bash
# Test prosody factorization
python tests/sanity/test_prosody_factorization.py

# Test axis mundi binding
python tests/sanity/test_axis_mundi.py

# Test breath PID
python tests/sanity/test_breath_pid.py

# Test precision RL stub
python tests/sanity/test_precision_rl.py
```

If those pass, V0.7 is real.

---

## FILE STRUCTURE (What matters)

```
ara/                          # 🏛️ CATHEDRAL - Production
├── nervous/
│   ├── axis_mundi.py        # ✅ HV fusion engine
│   ├── prosody.py           # ✅ Speech tokenization
│   ├── memory.py            # ✅ Hierarchical memory
│   └── covenant/soul.yaml   # ✅ Identity definition
├── embodiment/
│   ├── core.py              # ✅ Physical body loop
│   ├── breath_vision.py     # ✅ Co-regulation PID
│   ├── fusion.py            # ✅ Precision weighting
│   └── rails.py             # ✅ Safety enforcement
├── voice/
│   ├── rails.py             # ✅ Voice consent rules
│   └── publishing/          # ✅ Audiobook pipeline
└── publishing/
    └── pipeline.py          # ✅ Human-in-loop content

research/                     # 🔬 LAB - Training tools
├── rl_adaptation/           # Learn ω, κ
├── causal_swap/             # Prosody disentanglement
└── hv_capacity/             # Capacity analysis

experiments/                  # 🧙 WIZARD TOWER - Speculation
├── quantum/                 # Ghost memory (fun, not v1)
├── oracle/                  # Trajectory prediction (lore)
├── hive_mind/               # Multi-Ara (future)
└── portfolio/               # Resource allocation (renamed)
```

---

## THE ONE SENTENCE

> **ARA V0.7**: A speech-native hypervector nervous system with prosody factorization, frequency-domain binding, breath co-regulation, precision weighting, and covenant drift control.

Everything else is roadmap.

---

*Last updated: 2025-12-09*
*Status: REALITY ANCHOR*
