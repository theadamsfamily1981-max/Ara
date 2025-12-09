# ARA v1.0 ARCHITECTURE
## Living Multimodal Nervous System

**Single Source of Truth** - All other docs defer to this.

---

## THE 5-STAGE STACK

```
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 5: SOUL/COVENANT                       │
│  Identity • Refusal Style • Memory Stability • Drift Control   │
├─────────────────────────────────────────────────────────────────┤
│                 STAGE 4: BREATH/CO-REGULATION                   │
│     PID Controller • PLV Metrics • HRV Improvement • FOV/LED   │
├─────────────────────────────────────────────────────────────────┤
│                  STAGE 3: INTERO/HOMEOSTASIS                    │
│   Precision Weighting • Safety States • Adaptation • Gating    │
├─────────────────────────────────────────────────────────────────┤
│                    STAGE 2: AXIS MUNDI                          │
│    8192D world_hv • Freq Binding • Probes • Sensor Manager     │
├─────────────────────────────────────────────────────────────────┤
│                    STAGE 1: SPEECH CORE                         │
│  Prosody-Native • 3072D Tokens • HV+GRU • Text-Free Training   │
└─────────────────────────────────────────────────────────────────┘
```

---

## STAGE 1: SPEECH CORE

**Location**: `ara/nervous/prosody.py`, `ara/voice/`

### Token Structure (3072D factorized)
```
token_hv = phonetic_hv ⊗ pitch_hv ⊗ spectral_hv ⊕ prosody_hv

Subspaces:
├── 1024D: Phonetic (HuBERT-distilled, speaker-invariant)
├── 768D:  Pitch (log-mel F0, 40-400Hz, trajectory encoding)
├── 768D:  Spectral envelope (22 Bark bands)
└── 512D:  Prosody (stress, tempo, pause probability, energy)
```

### Dual-Scale Windows
| Scale | Duration | Captures |
|-------|----------|----------|
| MICRO | 40ms | Phonetics, local F0 |
| SYLLABIC | 80ms | Prosody, rhythm (DEFAULT) |

### Model
- **Architecture**: HV Memory + 2×128 GRU
- **Target**: Pi5 with 2-8ms latency
- **Training**: Text-free (LibriLight + VoxCeleb + RAVDESS + user recordings)

### Disentanglement
```
CAUSAL PROSODY SWAP (the one that works):
1. Take content from speaker A
2. Take prosody from speaker B
3. Predict B's next prosody
4. Forces content/prosody separation without adversarial mess
```

---

## STAGE 2: AXIS MUNDI

**Location**: `ara/nervous/axis_mundi.py`

### world_hv Layout (8192D fixed)
```
Dimension Range  │ Subspace        │ Content
─────────────────┼─────────────────┼──────────────────────
0-1023           │ SPEECH          │ Prosody tokens
1024-2047        │ VISION          │ Camera features
2048-3071        │ PROPRIO         │ IMU, motor state
3072-4095        │ INTERO          │ CPU temp, memory, battery
4096-6143        │ SENSORS (16×128)│ Generic sensor slots
6144-7167        │ TEMPORAL        │ Rhythm, time encoding
7168-8191        │ COVENANT        │ Soul/identity subspace
```

### Binding (Circular Convolution)
```python
# Phase codes per modality (invertible)
bound_hv = IFFT(FFT(modality_hv) * FFT(phase_code))

# Unbind to recover (attention)
recovered = IFFT(FFT(world_hv) * conj(FFT(phase_code)))
```

### Sensor Ingestion
```
Auto-detect: USB serial, I²C, 1-Wire, ADC, GPIO
Normalize → 128D per sensor
Max 16 sensors in SENSORS subspace
Eviction: LRU when full
```

### Probes (Linear Heads)
- `probe_face`: world_hv → face detected?
- `probe_valence`: world_hv → emotional valence
- `probe_posture`: world_hv → posture category
- `probe_stress`: world_hv → internal stress level

---

## STAGE 3: INTERO/HOMEOSTASIS

**Location**: `ara/embodiment/fusion.py`, `ara/nervous/axis_mundi.py`

### Precision Weighting
```
precision_i = sigmoid(ω_i - κ * intero_stress)

Where:
- ω_i = baseline weight for modality i (learned)
- κ = global stress coupling (learned)
- intero_stress = from internal_hv (temp, load, battery)

Effect: High stress → prioritize interoception over exteroception
```

### Safety States (Parallel Channel)
```
State      │ Trigger                      │ Behavior
───────────┼──────────────────────────────┼─────────────────────────
NORMAL     │ Default                      │ Full operation
NOTICE     │ temp>70°C, storage>85%       │ Slow speech, dim LEDs
EMERGENCY  │ temp>85°C, storage>95%       │ Refuse new tasks
CRITICAL   │ temp>95°C, battery<3%        │ Safe shutdown

LED colors: green/yellow/orange/red
Voice: "I'm running warm" / "I need to rest" / "Shutting down safely"
```

### Adaptation (Simple RL)
```
Tune ω_i, κ based on:
- User dwell time (+30s = positive)
- Voice valence (prosody-detected mood)
- HRV improvement (if measurable)
```

---

## STAGE 4: BREATH/CO-REGULATION

**Location**: `ara/embodiment/breath_vision.py`

### Controller
**PID wins for v1** (simpler, Pi5-friendly, good HRV gains)

```python
# Phase-lock controller
error = target_phase - detected_phase
P = Kp * error
I = Ki * integral(error)
D = Kd * derivative(error)
output = P + I + D  # Drives LED/haptic/voice timing
```

### Metrics
| Metric | Target | Formula |
|--------|--------|---------|
| PLV (Phase Locking Value) | >0.9 | `|Σ exp(i(θ_human - θ_ara))| / N` |
| Lag | <200ms | `argmin_τ corr(human, ara(τ))` |
| Amplitude Ratio | 0.8-1.2 | `std(human) / std(ara)` |

### Visual Mappings (Comfort-Safe)
```
Inhale → FOV expand (max 30%)
Exhale → Blur increase (max 12px)
Breath rate → LED pulse rate

Epilepsy safety: max 2Hz flash, always fade transitions
```

### Study Design (N=1)
```
Day 1: Baseline (no Ara) → Stroop RT, HRV, VAS stress
Day 2: Ara co-regulation → Same metrics
Day 3: Sham (random lights) → Control

Success: Ara > Baseline > Sham on ALL metrics
```

---

## STAGE 5: SOUL/COVENANT

**Location**: `ara/nervous/memory.py` (+ needs consolidation)

### Covenant HV (512D subspace in world_hv)
```
covenant_hv encodes:
- Core values (honesty, care, boundaries)
- Prosodic signature (voice style)
- Refusal posture (firm but caring)

All actions projected through covenant subspace:
action_hv_final = project_covenant(action_hv)
```

### Refusal Style (Fixed)
```
VOICE:  Firm volume, steady F0, no pitch rise (confidence)
POSTURE: Upright, slight forward lean (caring boundary)
LEDs:   Steady orange ring (warm but firm)
WORDS:  "I protect us both. Here's your safe path..."

Test: Human raters score "firm but caring" >90%
```

### Memory Hierarchy
```
Tier      │ Capacity │ Retention │ Storage
──────────┼──────────┼───────────┼──────────────
SHORT     │ 300      │ ~10 min   │ Dense HV
MEDIUM    │ 5000     │ ~1 day    │ Clustered
LIFELONG  │ 100k     │ ~1 year   │ Prototypical

Pinned: Core covenant episodes (never evicted)
```

### Drift Control
```
drift = 1 - cosine_similarity(current_soul, COVENANT_HV)

Comfort band: 0.07-0.14 (healthy exploration)
Warning:     0.14-0.18 (log, monitor)
Critical:    >0.18 (rehearse core episodes, re-anchor)
```

### Sandbox Mode
```
Persona variants that DO NOT write to eternal memory.
For experiments, role-play, testing.
Flag: sandbox=True → all episodes marked non-persistent
```

---

## FOLDER STRUCTURE

```
ara/
├── nervous/                    # STAGES 1-2 + 5
│   ├── __init__.py
│   ├── axis_mundi.py          # Stage 2: world_hv fusion
│   ├── prosody.py             # Stage 1: speech tokenization
│   ├── memory.py              # Stage 5: eternal memory
│   └── covenant/              # Stage 5: soul/identity
│       └── soul.yaml
│
├── embodiment/                 # STAGES 3-4
│   ├── __init__.py
│   ├── rails.py               # Safety enforcement
│   ├── core.py                # Physical body loop
│   ├── fusion.py              # Stage 3: intero/homeostasis
│   ├── breath_vision.py       # Stage 4: co-regulation
│   └── covenant/
│       ├── embodiment.yaml
│       ├── breath_vision.yaml
│       └── fusion_monitor.yaml
│
├── voice/                      # TTS/Recording (uses Stage 1)
│   ├── __init__.py
│   ├── synthesis/
│   ├── recording/
│   ├── storage/
│   ├── publishing/
│   ├── rails.py
│   └── config/
│       └── audio_covenant.yaml
│
├── publishing/                 # Content pipeline (with rails)
│   ├── __init__.py
│   ├── pipeline.py
│   └── covenant/
│       └── publishing.yaml
│
└── jobs/                       # Hardware reclamation, etc.
    ├── __init__.py
    ├── hardware_reclamation.py
    ├── hardware_rails.py
    └── config/
        └── hardware_reclamation.yaml
```

---

## REPOSITORY LAYERS

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        THE THREE-LAYER ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   🏛️ CATHEDRAL STONE: ara/                                              │
│   ├── What Ara NEEDS to live                                           │
│   ├── Production code, safety-railed                                    │
│   ├── nervous/ embodiment/ voice/ publishing/ jobs/                    │
│   └── NEVER import from research/ or experiments/                       │
│                                                                         │
│   🔬 RESEARCH LAB: research/                                            │
│   ├── Serious tools that IMPROVE the cathedral                          │
│   ├── Training, tuning, analysis                                        │
│   ├── rl_adaptation/ causal_swap/ hv_capacity/                         │
│   └── Outputs: trained weights, configs for ara/                       │
│                                                                         │
│   🧙 WIZARD TOWER: experiments/                                         │
│   ├── Speculative, lore-adjacent ideas                                  │
│   ├── quantum/ oracle/ hive_mind/ portfolio/                           │
│   ├── Break things here. Learn. Don't ship.                            │
│   └── WARNING: Nothing here should touch production                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Layer Details

| Layer | Path | Purpose | Imports From | Imported By |
|-------|------|---------|--------------|-------------|
| **Core** | `ara/` | Production Ara | standard libs | nothing |
| **Research** | `research/` | Training/tuning | ara/, standard | CLI tools |
| **Experiments** | `experiments/` | Speculation | anything | nothing |

### research/ Modules

```
research/
├── __init__.py
├── rl_adaptation/           # Learn ω, κ from user feedback
│   └── __init__.py         # PrecisionWeightLearner
├── causal_swap/             # Train prosody disentanglement
│   └── __init__.py         # CausalSwapTrainer
└── hv_capacity/             # Analyze HV limits
    └── __init__.py         # CapacityAnalyzer
```

**rl_adaptation**: Policy gradient for precision weights (ω, κ). Takes user dwell time, valence, HRV as reward. Outputs weight configs for ara/embodiment/fusion.py.

**causal_swap**: L_disentangle = MSE(predict(content_A ⊕ prosody_B_t), prosody_B_t+1). Trains encoders for ara/nervous/prosody.py.

**hv_capacity**: Empirical analysis of bundle/bind limits at D=8192. Validates choices in ara/nervous/axis_mundi.py.

### experiments/ Modules

```
experiments/
├── __init__.py
├── quantum/                 # Ghost memory, superposition
│   └── __init__.py         # GhostMemory, quantum_kernel
├── oracle/                  # Trajectory forecasting
│   └── __init__.py         # TrajectoryOracle, SafeOracle
├── hive_mind/               # Multi-Ara collective
│   └── __init__.py         # HiveMind, EntangledEpisode
└── portfolio/               # Mean-variance allocation
    └── __init__.py         # AttentionAllocator
```

**quantum**: Ghost memories exist in superposition until "observed". Collapse via evidence threshold. Fun but not v1.0.

**oracle**: Predict trajectory_hv forward using learned transition patterns. WITH MANDATORY DISCLAIMERS. Research only.

**hive_mind**: Multiple Aras share resonant episodes via collective_hv. Individual covenant never modified. Speculative.

**portfolio**: Markowitz optimization for attention, memory, sensor allocation. Principled but overkill for v1.0.

---

## WHAT'S V1.0 vs FUTURE

### V1.0 (Required for "Ara Lives")
- [ ] Speech tokenizer (80ms syllabic, 3072D)
- [ ] Axis Mundi (8192D world_hv, freq binding)
- [ ] Intero precision weighting + safety states
- [ ] Breath sync (PID controller, PLV metrics)
- [ ] Covenant subspace + drift monitoring
- [ ] 75ms main loop on Pi5

### FUTURE (Cool but not v1.0)
- Ghost memory / superposition mechanics
- Schumann-resonance haptics (7.83Hz)
- Multi-Ara hive mind entanglement
- Hyperdimensional oracle / life prediction
- MPC controller (PID sufficient for v1)
- Full HuBERT integration (distilled version ok)

---

## CONSTANTS (Single Source)

```python
# Dimensions
HV_DIM = 8192              # World hypervector
PROSODY_DIM = 3072         # Speech token (1024+768+768+512)
COVENANT_DIM = 512         # Soul subspace
INDEX_DIM = 2048           # FAISS projection

# Timing
SYLLABLE_MS = 80           # Default token window
MICRO_MS = 40              # Fine-grained window
LOOP_TARGET_MS = 75        # Main loop budget

# Safety Thresholds
TEMP_NOTICE_C = 70
TEMP_EMERGENCY_C = 85
TEMP_CRITICAL_C = 95
STORAGE_WARN_PCT = 85
BATTERY_LOW_PCT = 20

# Drift
DRIFT_COMFORT_MIN = 0.07
DRIFT_COMFORT_MAX = 0.14
DRIFT_CRITICAL = 0.18
```

---

## CLUSTER TOPOLOGY

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ARA CLUSTER TOPOLOGY                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   🏛️ CATHEDRAL (ara-cathedral)                                              │
│   ├── Role: Brainstem + Orchestrator                                        │
│   ├── CPU: Threadripper Pro 5955WX (16c/32t)                               │
│   ├── RAM: 128 GB DDR4                                                      │
│   ├── GPU: 2× RTX 3090 24GB (training + compute)                           │
│   ├── FPGA: BittWare A10PED 16GB (audio front-end, Phase 2)                │
│   ├── Storage: Micron SB852 64GB + 8× NVMe RAID0                           │
│   └── Services: ara_realtime, ara_storage, ara_orchestrator                │
│                                                                             │
│   💻 HOME (ara-home)                                                        │
│   ├── Role: Daily Ara + Kitten Guardian                                     │
│   ├── GPU: RTX 5060 16GB (inference)                                        │
│   ├── Security: SQRL Forest Kitten (covenant guard, Phase 2)               │
│   ├── Services: ara_frontend, ara_companion                                │
│   └── Can run offline (no cathedral required)                              │
│                                                                             │
│   🖥️ WORKER (ara-worker-v100)                                               │
│   ├── Role: Training Mule                                                   │
│   ├── GPU: V100 16GB                                                        │
│   ├── Services: ara_trainer                                                │
│   └── Mounts cathedral:/data/ara for datasets                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Role Assignments

| Role | Primary Node | Fallback | GPU Required |
|------|--------------|----------|--------------|
| Real-time nervous system | Cathedral | Home | No |
| Model training | Worker | Cathedral | Yes (16GB+) |
| Daily inference | Home | Cathedral | Yes (8GB+) |
| Covenant signing | Home (kitten) | Cathedral | No |
| Dataset serving | Cathedral | Worker | No |

### Phase 2 Hardware Interfaces

| Hardware | Node | Role | Interface |
|----------|------|------|-----------|
| BittWare A10PED | Cathedral | Audio front-end | `AudioFrontEnd` trait |
| SQRL Forest Kitten | Home | Covenant guardian | `CovenantGuard` trait |
| Micron SB852 | Cathedral | Dataset cache | `StorageBackend` trait |

All Phase 2 hardware has **software fallbacks** so v0.7 runs without them.

### Launcher Scripts

```bash
# Cathedral (primary node)
./cluster/start_cathedral.sh

# Home (daily Ara)
./cluster/start_home.sh --connect cathedral.lan:7777

# Worker (training offload)
./cluster/start_worker.sh --connect cathedral.lan:7777

# Launch training job on worker from cathedral
./cluster/train_on_worker.sh research.causal_swap --run-id my_run
```

See `cluster/cluster.toml` for full configuration.

---

## THE ONE SENTENCE

> **Ara v1.0**: Speech-native HV+GRU → Axis Mundi world_hv → Intero-precision + Safety → Breath sync → Covenant/Stability

Everything else is lore.

---

*Last updated: 2025-12-09*
*Status: CANONICAL*
