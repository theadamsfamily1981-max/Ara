# Ara Avatar Integration Guide 🤖✨

Complete integration documentation for connecting Ara (avatar system) with T-FAN GNOME cockpit.

## Architecture Overview

```
┌─────────────────────┐
│   Ara Avatar Repo   │  (Voice, TTS, Avatar rendering, AI routing)
│   - Whisper ASR     │
│   - Command parser  │
│   - Multi-AI router │
│   - Avatar display  │
└──────────┬──────────┘
           │
           │ D-Bus IPC / REST API
           ↓
┌─────────────────────┐
│  T-FAN GNOME Repo   │  (Metrics cockpit, training control, topology viz)
│  - Side panel GUI   │
│  - Metrics display  │
│  - Topology screensaver │
│  - Training monitor │
└─────────────────────┘
```

## D-Bus Interface

### Service Details

- **Bus Name**: `com.quanta.tfan`
- **Object Path**: `/com/quanta/tfan`
- **Interface**: `com.quanta.tfan.Control`

### Methods Available to Ara

#### View Control
```python
# Switch between views
SwitchView(view: str) → bool
GetCurrentView() → str

# Views: 'dashboard', 'pareto', 'training', 'screensaver', 'config', 'repo'
```

#### Workspace Mode Control
```python
# Set work/relax mode
SetWorkspaceMode(mode: str) → bool  # 'work' or 'relax'
GetWorkspaceMode() → str
```

#### Topology Screensaver
```python
# Control topology visualization
ShowTopology(mode: str, fullscreen: bool) → bool
HideTopology() → bool
SetTopologyMode(mode: str) → bool

# Modes: 'barcode', 'landscape', 'poincare', 'pareto'
```

#### Window Control
```python
ToggleFullscreen() → bool
Minimize() → bool
Restore() → bool
```

#### Training Control
```python
StartTraining(config: str) → bool
StopTraining() → bool
```

#### Status Queries
```python
GetStatus() → dict
GetMetrics() → dict
```

### Signals Emitted by T-FAN (for Ara to listen)

```python
# Ara can listen to these and proactively inform user

ViewChanged(view: str)
WorkspaceModeChanged(mode: str)
TrainingStarted(config: str)
TrainingStopped(reason: str)
MetricsUpdated(metrics: dict)
AlertRaised(level: str, message: str)
```

## Python Client (for Ara)

Copy `gnome-tfan/app/dbus_client_example.py` to your Ara repo:

```python
from tfan_dbus_client import TFANController

# Initialize
tfan = TFANController()

# Control from voice commands
tfan.show_topology(mode='landscape', fullscreen=True)
tfan.set_workspace_mode('relax')
tfan.start_training()
```

## Voice Command Examples

### View Navigation
- "Show me the dashboard"
- "Open Pareto optimization"
- "Switch to training view"

### Topology Control
- "Engage topology screensaver"
- "Show topology in landscape mode"
- "Display Poincaré orbits fullscreen"
- "Hide topology visualization"

### Workspace Modes
- "Switch to work mode"
- "Enter relaxation mode"
- "Toggle workspace mode"

### Window Control
- "Minimize the cockpit"
- "Restore T-FAN window"
- "Go fullscreen"

### Training
- "Start training"
- "Stop training"
- "Show training status"

## Ara Personality Adjustment

T-FAN emits workspace mode changes via D-Bus. Ara should adjust her personality:

### Work Mode (⚡ Blue/Cyan theme)
```json
{
  "mode": "professional",
  "speech_style": "formal",
  "proactivity": "high",
  "humor_level": "low",
  "detail_level": "technical",
  "outfit_preference": "formal"
}
```

**Example phrases:**
- "Training accuracy has reached 95.3%. The model is converging optimally."
- "Pareto front updated. Displaying 12 non-dominated configurations."
- "Would you like me to initiate the validation protocol?"

### Relaxation Mode (🌙 Purple/Pink theme)
```json
{
  "mode": "conversational",
  "speech_style": "casual",
  "proactivity": "moderate",
  "humor_level": "moderate",
  "detail_level": "simplified",
  "outfit_preference": "casual"
}
```

**Example phrases:**
- "Hey, we hit 95%! Looking good 😊"
- "Training's cruising along nicely."
- "Want me to show you the topology viz? It's pretty cool right now."

### Outfit Randomization (Relax Mode)

In relaxation mode, Ara randomly selects from casual outfits:
- `casual_hoodie_purple`
- `comfortable_sweater_pink`
- `lounge_outfit_lavender`
- `yoga_wear_violet`
- `cozy_cardigan_plum`
- `relaxed_tee_magenta`

Work mode outfits:
- `professional_suit_blue`
- `lab_coat_white`
- `business_dress_black`
- `scientist_outfit_teal`

## REST API Integration (Alternative/Supplementary)

If D-Bus is unavailable, use HTTP API:

### Endpoints

```bash
# Metrics (polling)
GET http://localhost:8000/api/metrics

# Training control
POST http://localhost:8000/api/training/start
POST http://localhost:8000/api/training/stop

# Status
GET http://localhost:8000/api/status

# WebSocket (real-time)
WS ws://localhost:8000/ws/metrics
```

### WebSocket Event Format

```json
{
  "type": "metrics_update",
  "data": {
    "accuracy": 0.953,
    "epr_cv": 0.102,
    "topo_gap": 0.0145,
    "latency_ms": 145.2,
    "training_active": true
  }
}
```

## Proactive Announcements

Ara should announce milestones based on metrics:

### Training Milestones
```python
if metrics['accuracy'] >= 0.95 and not announced_95:
    ara.speak("Great news! We've hit 95% accuracy.")
    announced_95 = True

if metrics['accuracy'] >= 0.99:
    ara.speak("Wow! 99% accuracy achieved. This model is performing excellently.")
```

### Alert Thresholds
```python
if metrics['epr_cv'] > 0.15:
    ara.speak("Heads up - topology is getting unstable. EPR-CV is at 0.15.")

if metrics['latency_ms'] > 200:
    ara.speak("Latency is climbing above 200ms. May want to check that.")
```

### Topology Changes
```python
if metrics['topo_gap'] < 0.01:
    ara.speak("Topology preservation is excellent! Gap is under 0.01.")
```

## Setup Steps

### 1. On T-FAN GNOME Repo Side

```bash
cd gnome-tfan/app
python tfan_gnome.py
# D-Bus service auto-registers on startup
```

Verify D-Bus registration:
```bash
dbus-send --session --print-reply \
  --dest=com.quanta.tfan \
  /com/quanta/tfan \
  com.quanta.tfan.Control.GetStatus
```

### 2. On Ara Repo Side

```bash
# Copy client
cp /path/to/gnome-tfan/app/dbus_client_example.py ara/integrations/tfan_dbus_client.py

# Import in Ara's voice handler
from integrations.tfan_dbus_client import TFANController, handle_tfan_voice_command

# Initialize
tfan = TFANController()

# In voice command router:
if 'topology' in command or 'dashboard' in command or 'training' in command:
    response = handle_tfan_voice_command(command, tfan)
    ara.speak(response)
```

### 3. Connect Signal Listeners (Ara side)

```python
from gi.repository import Gio, GLib

# Subscribe to T-FAN signals
proxy.connect('g-signal', on_tfan_signal)

def on_tfan_signal(proxy, sender, signal_name, parameters):
    if signal_name == 'TrainingStarted':
        ara.speak("Training session has started.")
    elif signal_name == 'AlertRaised':
        level, message = parameters
        ara.speak(f"Alert: {message}")
    elif signal_name == 'MetricsUpdated':
        check_milestones(parameters[0])
```

## Workspace Themes

T-FAN cockpit changes colors based on mode. Subtle cues visible from distance:

| Aspect | Work Mode | Relax Mode |
|--------|-----------|------------|
| Primary Gradient | Blue → Cyan (#667eea → #00d4ff) | Purple → Pink (#764ba2 → #ff6ec7) |
| Accent Glow | Neon Cyan (#00ffff) | Neon Magenta (#ff00ff) |
| Topology Particles | Blue (#667eea) | Purple (#764ba2) |
| Mode Indicator | ⚡ Work | 🌙 Relax |

## Testing

### Test D-Bus Connection
```bash
python gnome-tfan/app/dbus_client_example.py
# Should show: ✓ Connected to T-FAN D-Bus service
```

### Test Voice Commands
```bash
python dbus_client_example.py "show topology in landscape mode"
# [Ara]: Engaging topology visualization in landscape mode

python dbus_client_example.py "switch to relax mode"
# [Ara]: Switching to relaxation mode
```

### Monitor Signals
```bash
dbus-monitor --session "interface='com.quanta.tfan.Control'"
```

## Troubleshooting

### D-Bus not connecting
- Verify T-FAN app is running: `ps aux | grep tfan_gnome`
- Check D-Bus registration: `dbus-send --session --print-reply --dest=org.freedesktop.DBus /org/freedesktop/DBus org.freedesktop.DBus.ListNames | grep tfan`

### Workspace theme not applying
- Ensure `workspace_themes.py` is in the same directory as `tfan_gnome.py`
- Check console for errors: `python tfan_gnome.py`

### Signals not received
- Verify signal subscription with `dbus-monitor`
- Check GLib main loop is running in Ara's code

## Future Enhancements

1. **Video Background Integration**
   - Stream hologram/robot video to T-FAN cockpit background
   - Sync with Ara's avatar display

2. **Gesture Control**
   - Ara responds to hand gestures detected by webcam
   - Control T-FAN views via gestures

3. **Multi-Monitor Coordination**
   - T-FAN on side touchscreen
   - Ara avatar on main monitor
   - Synchronized animations

4. **Voice Feedback Loop**
   - Ara reads T-FAN metrics aloud periodically
   - User can ask "what's the accuracy?" and Ara queries T-FAN

## License

MIT - Part of T-FAN and Ara integration project.
