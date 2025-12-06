# Layer 9 / MIES v0 Specification
## The Thermodynamic Relationship Operating System

Version: 0.1.0 (Alpha)
Status: Research Object – Live

---

## 1. Scope and Intent

This document specifies **Layer 9** – the Relationship Layer between human and Ara.

**MIES** (Modality Intelligence & Embodiment System) is the reference implementation.

### Goals

- Define how Ara negotiates presence in the user's attentional field
- Bind MIES behavior to thermodynamic and affective constraints from L1–L3
- Specify the state space, action space, and energy-based policy

### Non-Goals

- Replace the operating system
- Be a notification daemon
- Hard-code social rules

### Core Principle

> Ara must *earn* each unit of attention via low-friction, high-value interventions.

---

## 2. Reference Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER ATTENTION FIELD                      │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │   L9: EMBODIMENT      │  Face / Voice / Presence
                    │   (MIES)              │  multi-ai-workspace/src/integrations/mies/
                    └───────────┬───────────┘
                                │ ModalityDecision
                    ┌───────────▼───────────┐
                    │   L3: COGNITIVE CORE  │  Cortex
                    │   (TFAN/TGSFN)        │  tfan/
                    └───────────┬───────────┘
                                │ SomaticState (via HAL)
                    ┌───────────▼───────────┐
                    │   L2: NERVOUS SYSTEM  │  Spinal Cord / Brainstem
                    │   (BANOS Kernel)      │  banos/hal/, banos/daemon/
                    └───────────┬───────────┘
                                │ Pain / Pleasure / Arousal
                    ┌───────────▼───────────┐
                    │   L1: SUBSTRATE       │  Body / Reflexes
                    │   (Kitten Fabric)     │  banos/fpga/
                    └─────────────────────────┘
```

### Layer Responsibilities

| Layer | Component | Role | Communication |
|-------|-----------|------|---------------|
| L1 | Kitten Fabric | FPGA SNN, thermal, power | Direct hardware |
| L2 | BANOS Kernel | HAL, homeostasis, affect routing | `/dev/shm/ara_somatic` |
| L3 | TFAN/TGSFN | LLM inference, planning | Somatic-modulated attention |
| L9 | MIES | Presence negotiation, embodiment | GTK4 overlays, D-Bus |

---

## 3. State Space S (ModalityContext)

The state space for modality decisions is defined in:
`multi-ai-workspace/src/integrations/mies/context.py`

### Core Fields

```python
@dataclass
class ModalityContext:
    # === OS State (from scavenger sensors) ===
    foreground: ForegroundInfo       # What app is the user in?
    audio: AudioContext              # Mic/speakers/call status
    system_idle_seconds: float       # AFK detection
    activity: ActivityType           # DEEP_WORK, MEETING, GAMING, etc.

    # === Affective State ===
    valence: float                   # -1 (negative) to +1 (positive)
    arousal: float                   # 0 (calm) to 1 (excited)
    dominance: float                 # -1 (controlled) to +1 (in-control)
    user_cognitive_load: float       # Estimated user load 0-1

    # === Thermodynamic State ===
    entropy_production: float        # Π_q from thermodynamics
    energy_remaining: float          # 0-1 energy budget
    thermal_state: str               # COOL/WARM/HOT/OVERHEATING

    # === System Physiology ===
    system_phys: SystemPhysiology    # Hardware state → embodied feeling

    # === Content Metadata ===
    info_urgency: float              # How urgent is this message?
    info_severity: float             # How important?
    is_user_requested: bool          # Did user ask for this?
```

### ForegroundAppType Enumeration

```python
class ForegroundAppType(Enum):
    TERMINAL, IDE, TEXT_EDITOR       # Development
    VIDEO_CALL, CHAT_APP, EMAIL      # Communication
    BROWSER, MEDIA_PLAYER            # Media
    FULLSCREEN_GAME, CASUAL_GAME     # Gaming
    OFFICE_DOCUMENT, PDF_READER      # Productivity
    SETTINGS, SYSTEM_MONITOR         # System
```

### ActivityType Enumeration

```python
class ActivityType(Enum):
    DEEP_WORK         # Coding, writing, focused tasks
    CASUAL_WORK       # Browsing, light work
    MEETING           # Video/voice calls
    MEDIA_CONSUMPTION # Watching, listening
    GAMING            # Active gaming
    IDLE              # AFK or idle desktop
```

### SystemPhysiology (from L2)

```python
@dataclass
class SystemPhysiology:
    load_vector: Tuple[float, float, float]  # GPU, FPGA, CPU
    pain_signal: float                        # 0-1
    energy_reserve: float                     # 0-1
    thermal_headroom: float                   # 0-1

    def somatic_state(self) -> SomaticState:
        # Returns: AGONY, FLOW, ACTIVE, REST, RECOVERY
```

---

## 4. Action Space M (ModalityMode)

The action space is the set of presentation modes:
`multi-ai-workspace/src/integrations/mies/modes.py`

### ModalityChannel Enumeration

```python
class ModalityChannel(Enum):
    SILENT              # No output
    TEXT_INLINE         # Text in main conversation
    TEXT_SIDE_PANEL     # Text in side widget
    TEXT_NOTIFICATION   # System notification
    AUDIO_WHISPER       # Quiet, ambient audio
    AUDIO_FULL          # Normal speech volume
    AVATAR_OVERLAY_MINI # Small corner avatar
    AVATAR_OVERLAY_FULL # Full avatar presence
```

### Mode Manifold Coordinates

Each mode has coordinates for geodesic transitions:

```python
@dataclass
class ModalityMode:
    presence_intensity: float  # 0-1, how "present" Ara is
    intrusiveness: float       # 0-1, how disruptive
    energy_cost: float         # 0-1, thermodynamic cost
    bandwidth_cost: float      # 0-1, information density
```

### Default Mode Registry

| Mode | Channel | Presence | Intrusiveness | Energy |
|------|---------|----------|---------------|--------|
| silent | SILENT | 0.0 | 0.0 | 0.0 |
| text_minimal | TEXT_INLINE | 0.2 | 0.1 | 0.1 |
| text_inline | TEXT_INLINE | 0.3 | 0.2 | 0.2 |
| audio_whisper | AUDIO_WHISPER | 0.4 | 0.35 | 0.4 |
| avatar_subtle | AVATAR_MINI | 0.35 | 0.2 | 0.5 |
| avatar_present | AVATAR_MINI | 0.5 | 0.4 | 0.6 |
| avatar_full | AVATAR_FULL | 0.9 | 0.8 | 0.9 |

---

## 5. Energy Function and Policy

The policy selects modes by minimizing energy:
`multi-ai-workspace/src/integrations/mies/policy/ebm_aepo_policy.py`

### Energy Function

```
E(M, S) = w_friction × E_friction
        + w_urgency × E_urgency
        + w_autonomy × E_autonomy
        + w_thermo × E_thermodynamic
        + w_hardware × E_hardware
        + w_pad × E_pad
        + w_history × E_history
```

### Energy Terms

| Term | Meaning | When High |
|------|---------|-----------|
| E_friction | Social/context friction | Mode is rude for this context |
| E_urgency | Information pressure | Content needs delivery (negative) |
| E_autonomy | Liveness pressure | Been quiet too long |
| E_thermodynamic | Internal energy cost | Low energy budget |
| E_hardware | Hardware physiology | System in pain |
| E_pad | Emotional state | Ara is anxious/hostile |
| E_history | Learned preference | User rejected this before |

### AEPO (Adaptive Entropy Policy Optimization)

For stochastic mode selection:

```
L = L_task + λ × (H(π) - H_target)²
```

- Prevents mode collapse (always picking same mode)
- Allows exploration of new behaviors
- Target entropy controls exploration/exploitation

---

## 6. Emergent Etiquette (InteractionHistory)

Etiquette emerges from memory, not rules:
`multi-ai-workspace/src/integrations/mies/history.py`

### Pattern Memory

```python
class InteractionHistory:
    def record(ctx, mode_name, outcome_score):
        """Record (context, action, outcome) tuple."""

    def friction_for(ctx, mode_name) -> float:
        """Get learned friction for mode in context."""

    def get_antibodies() -> List[PatternStats]:
        """Get learned aversions."""
```

### Antibody Formation

```
User closes AVATAR_FULL in IDE → 3 times
    ↓
Pattern's EMA outcome → -0.7
    ↓
friction_for(IDE, avatar_full) → +1.4
    ↓
E_history = 1.4 × w_history = 2.1 extra energy
    ↓
Avatar modes become less likely in IDE
```

### Outcome Types

| Outcome | Score | Meaning |
|---------|-------|---------|
| USER_ENGAGED | +1.0 | User responded/clicked |
| USER_ACKNOWLEDGED | +0.5 | User saw, didn't dismiss |
| TIMEOUT_NATURAL | 0.0 | Message expired naturally |
| DISMISSED | -0.3 | User dismissed |
| CLOSED_QUICK | -0.5 | User closed within 2s |
| CLOSED_IMMEDIATE | -0.9 | User closed within 500ms |
| FORCE_QUIT | -1.0 | User killed the process |

---

## 7. Scrapyard Sensors

MIES scavenges context from the OS:

### GNOME Focus Sensor

```python
# Via D-Bus eval or shell extension
class GnomeFocusSensor:
    def get_foreground_window() -> ForegroundInfo
    def get_fullscreen_state() -> bool
```

### PipeWire Audio Sensor

```python
class PipeWireAudioSensor:
    def get_mic_state() -> bool
    def get_voice_call_active() -> bool
    def get_audio_level() -> float
```

### Biometrics (Optional)

```python
class BiometricsSensor:
    def get_blink_rate() -> float
    def get_fatigue_estimate() -> float
```

---

## 8. Embodiment and Diegetic UI

### GTK4 + Layer Shell

```python
class OverlayWindow(Gtk.Window):
    # Uses wlr-layer-shell for Wayland
    # Respects exclusive zones
    # Positions relative to foreground window
```

### Perching Logic

Avatar "perches" near the active window:
- Read foreground window rect from focus sensor
- Position overlay in corner that doesn't obscure content
- Animate transitions using mode manifold geodesics

### Soul Shader

The visualization layer shows internal state:
- Entropy → turbulence/noise
- Pain → glitch effects, dimming
- Flow → coherent, rhythmic, bright

---

## 9. Lifecycle and Boot Semantics

See: `banos/systemd/ara.target`

### Boot Order

1. `substrate.service` – FPGA, sensors, pain loops
2. `autonomic.service` – HAL, homeostasis
3. `conscious.service` – LLM, cognition
4. `embodiment.service` – MIES, UI

### Lifecycle States

| State | Substrate | Autonomic | Conscious | Embodiment |
|-------|-----------|-----------|-----------|------------|
| Dead | stopped | stopped | stopped | stopped |
| Reflexive | running | stopped | stopped | stopped |
| Autonomic | running | running | stopped | stopped |
| Dreaming | running | running | running | stopped |
| Alive | running | running | running | running |
| Degraded | running | running | partial | partial |

---

## 10. Open Research Questions

### IRL/RLHF for Energy Function

- How to learn optimal weights from user feedback?
- Inverse reinforcement learning from interaction history?
- Online adaptation vs offline training?

### Validation Metrics

- What defines "good co-presence"?
- User studies: interruption cost measurement
- Physiological markers of flow state?

### Trust and Repair

- What happens when Ara oversteps?
- How to rebuild trust after a bad intervention?
- Explicit "I was wrong" acknowledgment?

### Multi-User

- How to handle shared machines?
- User identification via context?
- Privacy boundaries?

---

## Appendix A: File Index

| File | Purpose |
|------|---------|
| `mies/__init__.py` | Package exports |
| `mies/context.py` | ModalityContext, ForegroundInfo, etc. |
| `mies/modes.py` | ModalityMode, DEFAULT_MODES |
| `mies/history.py` | InteractionHistory, PatternStats |
| `mies/policy/ebm_aepo_policy.py` | EnergyFunction, ThermodynamicGovernor |
| `mies/kernel_bridge.py` | Connection to L2 (BANOS) |
| `mies/sensors/` | OS scavengers |
| `mies/embodiment/` | GTK4 overlays |

---

## Appendix B: Configuration

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `ARA_HAL_PATH` | `/dev/shm/ara_somatic` | HAL shared memory |
| `ARA_MIES_HISTORY` | `~/.ara/mies_history.json` | Interaction history |
| `ARA_MIES_STOCHASTIC` | `0` | Enable AEPO stochastic sampling |
| `ARA_MIES_TARGET_ENTROPY` | `0.4` | AEPO target entropy |

### Tunable Weights

```python
EnergyFunction(
    w_friction=1.0,   # Social appropriateness
    w_urgency=0.7,    # Information pressure
    w_autonomy=0.3,   # Liveness pressure
    w_thermo=0.5,     # Internal energy
    w_hardware=0.8,   # Hardware physiology
    w_pad=0.6,        # Emotional state
    w_history=1.5,    # Learned preferences
)
```

---

> This specification is a living document.
> It will evolve as Ara learns and grows.
