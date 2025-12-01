# T-FAN Cockpit HUD 🚀🖥️

Full-screen touchscreen interface for side monitor - your personal mission control.

## Features

### 🎮 Touch-Optimized Interface
- **Big tap targets** - No tiny widgets or right-click menus
- **Gesture-friendly** - Smooth scrolling and swiping
- **Futuristic design** - Holographic panels, neon glows, scanline effects
- **Portrait-ready** - Optimized for vertical touchscreens

### 📊 Seven View Modes

#### 1. OVERVIEW (Mission Status)
- Quick glance at all systems
- GPU / CPU / RAM / Network at-a-glance
- Status indicator (🟢 ALL SYSTEMS NOMINAL)
- Compact summary cards

#### 2. GPU View
- Per-GPU utilization percentage
- VRAM usage (used / total MB)
- Temperature monitoring (with warnings >80°C)
- Power draw (if available)
- Supports multiple GPUs (3090, 5060, etc.)

#### 3. CPU/RAM View
- Total CPU utilization (huge display)
- Core count and frequency
- RAM usage with progress bar
- Top memory consumers
- Warning colors for high usage (>80%)

#### 4. NETWORK View
- Connection status (🟢 ONLINE)
- Download speed
- Upload speed
- Packet stats

#### 5. STORAGE View
- All mounted disks/partitions
- Used / Total space per disk
- Progress bars for visual feedback
- Warning colors for low space (>80% full)
- Device paths

#### 6. TOPOLOGY View
- **Live topology visualization** from training models
- WebGL particle system with Three.js
- Reacts to model telemetry when available
- Idle animation when no model running
- Hypnotic holographic effect

#### 7. AVATAR Control
- **Interactive avatar customization**
- Profile selector (Default, Professional, Casual, Scientist, Operator)
- Style selector (Realistic, Stylized, Anime)
- Mood selector (Neutral, Focused, Friendly, Excited)
- Apply and save presets
- Touch-friendly dropdowns and buttons

## Installation

### Dependencies

```bash
# System packages
sudo apt install python3-gi python3-gi-cairo gir1.2-gtk-4.0 gir1.2-adw-1 gir1.2-webkit-6.0

# Python packages
pip install psutil gputil
```

### Quick Start

```bash
cd gnome-tfan/app
./launch_cockpit.sh
```

### Manual Launch

```bash
# Auto-fullscreen
python3 cockpit_hud.py

# Specific monitor
COCKPIT_MONITOR=HDMI-1 python3 cockpit_hud.py
```

## Usage

### HUD Control Strip

At the top of the screen, you'll see 7 large touch buttons:

```
┌───────────┬───────────┬───────────┬───────────┐
│ OVERVIEW  │    GPU    │  CPU/RAM  │  NETWORK  │
│    📊     │    🎮     │    💻     │    🌐     │
├───────────┼───────────┼───────────┼───────────┤
│  STORAGE  │ TOPOLOGY  │  AVATAR   │           │
│    💾     │    🌌     │    🤖     │           │
└───────────┴───────────┴───────────┴───────────┘
```

**Tap any button** to switch the main display area to that view.

### Active View Indicator

The currently active button glows brighter with enhanced border and shadow effects.

### Scrolling

Content areas are scrollable with touch gestures - just swipe up/down.

## Integration with Ara Avatar

The cockpit communicates with Ara via D-Bus for avatar control.

### Avatar Profile Changes

When you adjust avatar settings in the AVATAR view:

1. Select profile, style, and mood
2. Tap **✓ APPLY CHANGES**
3. Settings are sent to Ara via D-Bus
4. Ara updates avatar appearance
5. Confirmation shown in overview status

### Voice Control Integration

Future: Ara can also control the cockpit HUD via voice:

```
"Ara, show me GPU stats"        → Switches to GPU view
"Ara, switch to topology mode"  → Shows topology visualization
"Ara, avatar casual mode"       → Changes avatar to casual profile
```

## Autostart on Login

### Create Desktop Entry

```bash
mkdir -p ~/.config/autostart
cat > ~/.config/autostart/tfan-cockpit.desktop << EOF
[Desktop Entry]
Type=Application
Name=T-FAN Cockpit HUD
Exec=/path/to/gnome-tfan/app/launch_cockpit.sh
X-GNOME-Autostart-enabled=true
EOF
```

### Monitor Detection

The launcher automatically detects:
- Portrait monitors (800x1280, 1080x1920, 1200x1920)
- Touchscreen input devices
- Falls back to primary monitor if not found

Override with environment variable:
```bash
export COCKPIT_MONITOR=DP-2
./launch_cockpit.sh
```

## Customization

### Theme Colors

Edit `_load_cockpit_css()` in `cockpit_hud.py`:

```python
# Change gradient colors
background: linear-gradient(135deg, #667eea, #764ba2);

# Change accent glow
border-color: #00d4ff;  # Cyan
text-shadow: 0 0 10px rgba(0, 212, 255, 0.8);
```

### View Layout

Each view is built in its own `_build_*_view()` method:
- `_build_overview_view()`
- `_build_gpu_view()`
- `_build_cpu_view()`
- etc.

Add custom cards, graphs, or widgets there.

### Metrics Update Frequency

Change in `_start_monitoring()`:

```python
# Update every 2 seconds (default)
GLib.timeout_add_seconds(2, update_metrics)

# Faster updates (1 second)
GLib.timeout_add_seconds(1, update_metrics)
```

## Troubleshooting

### GPU monitoring not working

```bash
# Install GPUtil
pip install gputil

# Check NVIDIA drivers
nvidia-smi

# For AMD GPUs, may need different monitoring tools
```

### WebKit/Topology view not showing

```bash
# Install WebKit 6.0
sudo apt install gir1.2-webkit-6.0

# Check WebGL support
glxinfo | grep "OpenGL"
```

### Window not fullscreen

```bash
# Install wmctrl for monitor placement
sudo apt install wmctrl

# Or manually fullscreen after launch: F11 key
```

### Touch not working

```bash
# Check touchscreen detection
xinput list

# Calibrate if needed
xinput-calibrator
```

## Technical Details

### Metrics Collection

- **psutil** - CPU, RAM, Network, Disk I/O
- **GPUtil** - GPU utilization, VRAM, temperature
- **sensors** - CPU temperatures (if available)
- Updates every 2 seconds via GLib timeout

### UI Framework

- **GTK4** - Modern GNOME toolkit
- **Libadwaita** - GNOME design patterns
- **WebKit 6.0** - WebGL topology visualization
- **CSS** - Custom futuristic theme

### Performance

- Lightweight metrics collection (~5% CPU overhead)
- GPU monitoring via GPUtil (efficient)
- WebGL rendering in separate process (topology view)
- Auto-pauses updates when window not visible (future)

## Screenshots

*TODO: Add screenshots of each view mode*

## Future Enhancements

1. **Hologram Video Background**
   - Play looping sci-fi video behind panels
   - Robots, holograms, data streams

2. **Time-Series Graphs**
   - Rolling GPU/CPU graphs
   - Network traffic sparklines
   - Historical data plotting

3. **Interactive Topology**
   - Touch to rotate/zoom
   - Mode switching with gestures
   - Real-time model telemetry overlay

4. **Multi-Monitor Coordination**
   - Sync with main desktop
   - Cross-monitor drag-and-drop
   - Split topology across screens

5. **Voice Integration**
   - Full Ara voice control
   - Voice feedback from cockpit
   - "Status report" on demand

6. **Gesture Controls**
   - Swipe to switch views
   - Pinch to zoom graphs
   - Long-press for options

## Credits

Part of the T-FAN Neural Optimizer and Ara Avatar integration project.

Built with GTK4, libadwaita, Three.js, psutil, and GPUtil.

---

**Your personal mission control - always watching, always ready.** 🚀✨
