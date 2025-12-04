# MIES - Modality Intelligence & Embodiment System

## Overview

MIES is Ara's **Stage Manager / Etiquette Brain** - it decides **HOW** she should present herself based on context. Rather than always responding in the same way, Ara adapts her presentation to be:

- **Contextually appropriate** (silent during meetings, text-only during deep work)
- **Energy-aware** (reduced presence when thermodynamically constrained)
- **Socially intelligent** (respecting user's cognitive load and emotional state)
- **Embodied** (avatar overlays that feel native to the desktop)

## Philosophy: "Goddess in a Scrapyard"

MIES scavenges context from the OS:
- **GNOME/Wayland**: Focused window, fullscreen state, idle time
- **PipeWire**: Mic/speaker state, voice calls, music playing
- **Biometrics** (optional): Blink rate, pupil dilation, gaze tracking

This "scrapyard" data feeds into a thermodynamic/free-energy based policy that minimizes social friction while maximizing information delivery.

## Architecture

```
multi-ai-workspace/src/integrations/mies/
├── __init__.py              # Package exports
├── context.py               # ModalityContext, ForegroundAppType, ActivityType
├── modes.py                 # ModalityChannel, ModalityMode, ModalityDecision
├── policy/
│   ├── __init__.py
│   ├── heuristic_baseline.py   # Rules-based policy (works now)
│   └── ebm_aepo_policy.py      # Energy-Based Model + AEPO (for training)
├── sensors/
│   ├── __init__.py
│   ├── gnome_focus.py       # GNOME/DBus focus detection
│   ├── pipewire_audio.py    # PipeWire audio context
│   └── biometrics.py        # Biometrics stub (future)
└── embodiment/
    ├── __init__.py
    ├── overlay_window.py    # GTK4 layer-shell overlays
    └── liveness.py          # Subtle "alive" animations
```

## Modality Modes

MIES can select from various presentation modes:

| Mode | Channel | Presence | Intrusiveness | Use Case |
|------|---------|----------|---------------|----------|
| `silent` | SILENT | 0.0 | 0.0 | No output needed |
| `text_inline` | TEXT_INLINE | 0.3 | 0.2 | Standard text response |
| `text_minimal` | TEXT_INLINE | 0.2 | 0.1 | Brief acknowledgments |
| `audio_whisper` | AUDIO_WHISPER | 0.4 | 0.35 | Quiet ambient voice |
| `audio_normal` | AUDIO_FULL | 0.6 | 0.5 | Normal speech |
| `avatar_subtle` | AVATAR_MINI | 0.35 | 0.2 | Small corner presence |
| `avatar_full` | AVATAR_FULL | 0.9 | 0.8 | Full avatar with voice |

## Context Signals

MIES considers:

### OS State
- `foreground_app`: What app is focused (IDE, browser, video call, game)
- `is_fullscreen`: Is the app fullscreen?
- `mic_in_use`: Is the microphone active?
- `speakers_in_use`: Is audio playing?
- `has_voice_call`: Is user in a voice/video call?
- `system_idle_seconds`: How long has the system been idle?

### Affective State
- `valence`: Emotional positivity (-1 to +1)
- `arousal`: Emotional intensity (0 to 1)
- `user_cognitive_load`: Estimated user load (0 to 1)
- `ara_fatigue`: Ara's internal fatigue (0 to 1)

### Thermodynamic State
- `entropy_production`: Cognitive heat generation
- `energy_remaining`: Energy budget percentage
- `thermal_state`: COOL/WARM/HOT/OVERHEATING

### Content Metadata
- `info_urgency`: How urgent is this information?
- `is_user_requested`: Did the user ask for this?
- `deadline_seconds`: Time-sensitive deadline

## Policy Rules

The heuristic baseline implements these key rules:

### Meetings are Sacred
- **Never** use audio during meetings
- **Never** show large avatar overlays
- Text-only, minimal intrusiveness

### Deep Work is Protected
- Prefer text-only responses
- No audio unless high urgency
- Minimal visual presence

### Gaming Gets Priority
- No overlays during fullscreen games
- Audio only if not in voice chat
- Notifications only for urgent info

### Energy Awareness
- Reduce expensive modes when energy is low
- Force recovery when overheating
- Thermodynamic cost factors into all decisions

## Configuration

MIES is controlled by config flags:

```yaml
mies:
  enabled: true
  allow_unsafe_gnome_eval: true  # Required for full focus detection
  allow_pipewire: true
  default_policy: "heuristic"    # "heuristic" or "thermodynamic"
  biometrics_enabled: false      # Requires webcam + OpenCV/MediaPipe
```

## Integration

MIES is integrated as **Phase 5.5** in Ara's cognitive cycle:

```
Phase 1: SENSATION
Phase 2: PERCEPTION
Phase 3: PREDICTION
Phase 4: AFFECT
Phase 5: IDENTITY
Phase 5.5: MODALITY INTELLIGENCE (MIES)  <-- HERE
Phase 6: SELF-PRESERVATION
Phase 7: EXECUTIVE
Phase 8: COGNITION
Phase 9: REALITY CHECK
Phase 10: THERMODYNAMICS
Phase 11: MEMORY
```

The modality decision influences how the response is delivered after cognition.

## Sensors

### GNOME Focus Sensor

Requires `gdbus` CLI tool. Uses GNOME Shell Eval to query:
- Focused window WM_CLASS and title
- Window geometry
- Fullscreen state

**Unsafe mode**: Runs GJS code in GNOME Shell (powerful but requires trust)
**Safe mode**: Returns UNKNOWN for all queries

### PipeWire Audio Sensor

Requires `pw-dump` (PipeWire) or `pactl` (PulseAudio fallback).
Detects:
- Active audio streams
- Voice call applications (Zoom, Teams, Discord)
- Music playback

### Biometrics Sensor (Stub)

Future integration for:
- Blink rate estimation (cognitive load)
- Pupil dilation tracking
- Gaze analysis

## Extending MIES

### Adding New App Types

Edit `gnome_focus.py`:

```python
WM_CLASS_PATTERNS["my_app"] = ForegroundAppType.MY_TYPE
```

### Adding New Modes

Edit `modes.py`:

```python
MODE_CUSTOM = ModalityMode(
    name="custom",
    channel=ModalityChannel.TEXT_INLINE,
    presence_intensity=0.4,
    intrusiveness=0.3,
    energy_cost=0.25,
    bandwidth_cost=0.5,
)

DEFAULT_MODES["custom"] = MODE_CUSTOM
```

### Training the Policy

The `ThermodynamicGovernor` in `ebm_aepo_policy.py` supports:
- Energy-Based Model scoring
- AEPO (Adaptive Entropy Policy Optimization) sampling
- Pluggable policy network for RL training

Future work: Train from user preference data via IRL/RLHF.

## Troubleshooting

### Sensors not detecting apps

1. Check if `gdbus` is available: `which gdbus`
2. Verify GNOME Shell is running: `gdbus call --session --dest org.gnome.Shell --object-path /org/gnome/Shell --method org.gnome.Shell.Eval 1`
3. Check sensor logs for errors

### Audio not detected

1. Check PipeWire: `pw-cli info 0`
2. Or PulseAudio: `pactl info`
3. Verify `pw-dump` output includes your audio streams

### Avatar overlay not showing

1. Verify GTK4 is installed: `pkg-config --exists gtk4`
2. Check for gtk4-layer-shell: `pkg-config --exists gtk4-layer-shell`
3. Ensure running on Wayland (X11 has limited support)

## API Reference

### ModalityContext

```python
ctx = ModalityContext(
    foreground=ForegroundInfo(...),
    audio=AudioContext(...),
    activity=ActivityType.DEEP_WORK,
    valence=0.5,
    arousal=0.3,
    info_urgency=0.5,
    is_user_requested=True,
)
ctx.update_derived_fields()
```

### HeuristicModalityPolicy

```python
policy = HeuristicModalityPolicy()
decision = policy.select_modality(ctx, prev_mode=None)
print(decision.mode.name)       # "text_inline"
print(decision.rationale)       # "Activity=DEEP_WORK -> text_inline (score=0.42)"
```

### ThermodynamicGovernor

```python
governor = ThermodynamicGovernor(use_stochastic=False)
decision = governor.select_modality(
    ctx=ctx,
    content_meta=ContentMeta(urgency=0.5),
    prev_mode=None,
)
```

## Credits

MIES design based on the "Goddess in a Scrapyard" philosophy and AEPO-style entropy-controlled policy optimization.
