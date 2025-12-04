# Ara D-Bus Interface Schema

This document defines the D-Bus interface for Ara's L3 Metacontrol system.
External applications (cockpit, avatar, overlays) communicate with Ara through these signals and methods.

## Bus Information

- **Bus Name**: `org.ara.Metacontrol`
- **Object Path**: `/org/ara/Metacontrol`
- **Interface**: `org.ara.Metacontrol`

## Signals

### PADChanged

Emitted when Ara's emotional state changes significantly.

```xml
<signal name="PADChanged">
  <arg name="payload" type="s" direction="out"/>
</signal>
```

**Payload Schema** (JSON):
```json
{
  "timestamp": 1701705600.123,
  "pad": {
    "pleasure": 0.45,
    "arousal": -0.12,
    "dominance": 0.33
  },
  "quadrant": "SERENE",
  "mood_label": "calm and content",
  "intensity": 0.52,
  "source": "MIES_CATHEDRAL",
  "trigger": "pad_shift",
  "confidence": 0.85
}
```

**Fields**:
| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `timestamp` | float | Unix time | When the change occurred |
| `pad.pleasure` | float | [-1, 1] | Valence (negative=displeasure, positive=pleasure) |
| `pad.arousal` | float | [-1, 1] | Activation level (negative=calm, positive=activated) |
| `pad.dominance` | float | [-1, 1] | Control/agency (negative=submissive, positive=dominant) |
| `quadrant` | string | enum | One of 8 emotional quadrants |
| `mood_label` | string | text | Human-readable mood description |
| `intensity` | float | [0, 1] | Strength of the emotional state |
| `source` | string | enum | PAD computation source |
| `trigger` | string | enum | What caused the change |
| `confidence` | float | [0, 1] | Confidence in the PAD value |

**Quadrant Values**:
- `EXUBERANT`: High pleasure, high arousal, high dominance
- `DEPENDENT`: High pleasure, high arousal, low dominance
- `SERENE`: High pleasure, low arousal, high dominance
- `DOCILE`: High pleasure, low arousal, low dominance
- `HOSTILE`: Low pleasure, high arousal, high dominance
- `ANXIOUS`: Low pleasure, high arousal, low dominance
- `DISDAINFUL`: Low pleasure, low arousal, high dominance
- `BORED`: Low pleasure, low arousal, low dominance

**Source Values**:
- `MIES_CATHEDRAL`: MIES PADEngine (hardware telemetry)
- `ARA_INTEROCEPTION`: ara/interoception SNN
- `KERNEL_BRIDGE`: Kernel-computed PAD
- `PULSE_ESTIMATION`: External affect estimation
- `FUSED`: Weighted fusion of sources

**Trigger Values**:
- `pad_shift`: PAD distance exceeded threshold
- `quadrant_change`: Moved to different quadrant
- `time_based`: Periodic refresh
- `manual`: Explicit refresh request
- `initial`: First computation

---

### ModeChanged

Emitted when modality presentation mode changes.

```xml
<signal name="ModeChanged">
  <arg name="payload" type="s" direction="out"/>
</signal>
```

**Payload Schema** (JSON):
```json
{
  "timestamp": 1701705600.456,
  "mode": {
    "name": "audio_whisper",
    "channel": "AUDIO_WHISPER",
    "presence_intensity": 0.4,
    "intrusiveness": 0.3,
    "energy_cost": 0.4
  },
  "previous_mode": "text_inline",
  "transition": {
    "duration_ms": 500,
    "easing": "ease_out",
    "fade_out_first": true
  },
  "rationale": "User in casual_work context, moderate urgency",
  "confidence": 0.78
}
```

**Fields**:
| Field | Type | Description |
|-------|------|-------------|
| `mode.name` | string | Mode identifier |
| `mode.channel` | string | Modality channel enum |
| `mode.presence_intensity` | float | How "present" Ara appears [0, 1] |
| `mode.intrusiveness` | float | How disruptive [0, 1] |
| `mode.energy_cost` | float | Thermodynamic cost [0, 1] |
| `previous_mode` | string | Mode being transitioned from |
| `transition.duration_ms` | int | Transition animation duration |
| `transition.easing` | string | Animation easing function |
| `transition.fade_out_first` | bool | Whether to fade out before fade in |
| `rationale` | string | Human-readable reason for decision |
| `confidence` | float | Policy confidence [0, 1] |

**Channel Values**:
- `SILENT`: No output
- `TEXT_INLINE`: Main conversation text
- `TEXT_SIDE_PANEL`: Side widget text
- `TEXT_NOTIFICATION`: System notification
- `AUDIO_WHISPER`: Quiet, ambient speech
- `AUDIO_FULL`: Normal speech
- `AVATAR_OVERLAY_MINI`: Small corner avatar
- `AVATAR_OVERLAY_FULL`: Full avatar overlay
- `SILENT_BACKGROUND`: Background processing

---

### ModulationChanged

Emitted when L3 control modulation parameters change.

```xml
<signal name="ModulationChanged">
  <arg name="payload" type="s" direction="out"/>
</signal>
```

**Payload Schema** (JSON):
```json
{
  "timestamp": 1701705600.789,
  "modulation": {
    "temperature_mult": 0.85,
    "memory_mult": 0.92,
    "attention_gain": 1.05,
    "lr_scale": 0.8,
    "empathy_weight": 0.7
  },
  "workspace_mode": "work",
  "control_law": {
    "arousal_to_temp": true,
    "valence_to_memory": true
  },
  "source_pad": {
    "pleasure": 0.3,
    "arousal": -0.2,
    "dominance": 0.4
  }
}
```

**Fields**:
| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `modulation.temperature_mult` | float | [0.5, 1.5] | LLM temperature multiplier |
| `modulation.memory_mult` | float | [0.5, 1.5] | Memory write probability mult |
| `modulation.attention_gain` | float | [0.5, 1.5] | Attention focus multiplier |
| `modulation.lr_scale` | float | [0.5, 1.5] | Learning rate scale |
| `modulation.empathy_weight` | float | [0, 1] | Weight for empathic response |
| `workspace_mode` | string | enum | Current workspace mode |
| `control_law` | object | | Which control laws are active |
| `source_pad` | object | | PAD that drove the modulation |

**Workspace Modes**:
- `work`: Focused, high valence + arousal
- `relax`: Calm, low arousal
- `creative`: Exploratory, high arousal
- `support`: Warm, stable, high valence

---

### HealthAlert

Emitted when system health changes significantly.

```xml
<signal name="HealthAlert">
  <arg name="payload" type="s" direction="out"/>
</signal>
```

**Payload Schema** (JSON):
```json
{
  "timestamp": 1701705600.999,
  "overall_health": 0.65,
  "thermal_ok": false,
  "load_ok": true,
  "errors_ok": true,
  "warnings": [
    "Temperature elevated: 82.5°C"
  ],
  "alerts": [],
  "telemetry": {
    "cpu_temp": 82.5,
    "gpu_temp": 78.0,
    "cpu_load": 0.65,
    "gpu_load": 0.45,
    "error_rate": 0.5
  }
}
```

---

## Methods

### SetWorkspaceMode

Set the current workspace mode.

```xml
<method name="SetWorkspaceMode">
  <arg name="mode" type="s" direction="in"/>
  <arg name="success" type="b" direction="out"/>
</method>
```

**Input**: Mode string (`"work"`, `"relax"`, `"creative"`, `"support"`)

**Output**: Boolean success indicator

---

### ComputePADGating

Compute gating parameters from PAD state.

```xml
<method name="ComputePADGating">
  <arg name="pleasure" type="d" direction="in"/>
  <arg name="arousal" type="d" direction="in"/>
  <arg name="dominance" type="d" direction="in"/>
  <arg name="result" type="s" direction="out"/>
</method>
```

**Input**: PAD values (doubles in [-1, 1])

**Output**: JSON string with computed modulation

---

### GetStatus

Get current system status.

```xml
<method name="GetStatus">
  <arg name="status" type="s" direction="out"/>
</method>
```

**Output**: JSON string with full system status

---

### GetSignalHistory

Get recent signal history.

```xml
<method name="GetSignalHistory">
  <arg name="signal_type" type="s" direction="in"/>
  <arg name="limit" type="i" direction="in"/>
  <arg name="history" type="s" direction="out"/>
</method>
```

**Input**:
- `signal_type`: One of `"pad"`, `"mode"`, `"modulation"`, `"health"`
- `limit`: Maximum number of entries to return

**Output**: JSON array of historical entries

---

## Usage Examples

### Python (GLib/Gio)

```python
from gi.repository import Gio, GLib

# Connect to bus
bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)

# Subscribe to PADChanged
def on_pad_changed(connection, sender, path, interface, signal, parameters):
    payload = json.loads(parameters[0])
    print(f"PAD: {payload['quadrant']} - {payload['mood_label']}")

bus.signal_subscribe(
    'org.ara.Metacontrol',
    'org.ara.Metacontrol',
    'PADChanged',
    '/org/ara/Metacontrol',
    None,
    Gio.DBusSignalFlags.NONE,
    on_pad_changed,
)

# Call method
result = bus.call_sync(
    'org.ara.Metacontrol',
    '/org/ara/Metacontrol',
    'org.ara.Metacontrol',
    'SetWorkspaceMode',
    GLib.Variant('(s)', ('creative',)),
    GLib.VariantType('(b)'),
    Gio.DBusCallFlags.NONE,
    -1,
    None,
)
```

### Command Line (gdbus)

```bash
# Subscribe to signals
gdbus monitor --session --dest org.ara.Metacontrol

# Call SetWorkspaceMode
gdbus call --session \
  --dest org.ara.Metacontrol \
  --object-path /org/ara/Metacontrol \
  --method org.ara.Metacontrol.SetWorkspaceMode \
  "creative"

# Get status
gdbus call --session \
  --dest org.ara.Metacontrol \
  --object-path /org/ara/Metacontrol \
  --method org.ara.Metacontrol.GetStatus
```

---

## Avatar Integration

The avatar system should subscribe to `PADChanged` and `ModeChanged` to:

1. Update facial expressions based on emotional quadrant
2. Adjust animation parameters based on arousal
3. Transition between visibility modes smoothly
4. Match voice characteristics to emotional state

### Expression Mapping

| Quadrant | Expression | Animation Speed | Voice Tone |
|----------|------------|-----------------|------------|
| EXUBERANT | Wide smile, bright eyes | Fast | Enthusiastic |
| SERENE | Gentle smile, soft gaze | Slow | Warm, calm |
| ANXIOUS | Furrowed brow, alert eyes | Fast | Hesitant |
| BORED | Half-lidded, neutral | Very slow | Flat |
| HOSTILE | Narrowed eyes, tense | Medium | Clipped |
| DOCILE | Soft expression, downward gaze | Slow | Gentle |
| DEPENDENT | Eager expression, wide eyes | Fast | Seeking |
| DISDAINFUL | Raised eyebrow, slight frown | Slow | Cool |

---

## Version History

- **v1.0** (2024-12-04): Initial schema definition
  - PADChanged, ModeChanged, ModulationChanged, HealthAlert signals
  - SetWorkspaceMode, ComputePADGating, GetStatus, GetSignalHistory methods
