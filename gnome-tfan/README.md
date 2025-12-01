# T-FAN GNOME Integration 🚀

**Beautiful, modern GNOME desktop integration for T-FAN neural network training and optimization.**

## Features

✨ **GNOME Shell Extension**
- System tray indicator with live metrics
- Training status in top bar
- Quick actions menu
- Real-time monitoring

📊 **GTK4/libadwaita Dashboard**
- Modern, beautiful UI with gradient cards
- Real-time training metrics
- Pareto optimization interface
- Configuration editor
- GitHub repository loader

🎯 **Auto-Loading**
- Just paste a GitHub URL
- Auto-clones and configures
- One-click training start
- Integrated Pareto optimization

## Installation

### Quick Install

```bash
cd gnome-tfan
chmod +x install.sh
./install.sh
```

### Manual Installation

1. **Install GNOME Shell Extension:**
   ```bash
   mkdir -p ~/.local/share/gnome-shell/extensions/tfan@quanta-meis-nib-cis
   cp extension/* ~/.local/share/gnome-shell/extensions/tfan@quanta-meis-nib-cis/
   ```

2. **Install Main App:**
   ```bash
   mkdir -p ~/.local/share/tfan
   cp app/tfan_gnome.py ~/.local/share/tfan/
   chmod +x ~/.local/share/tfan/tfan_gnome.py

   # Create launcher
   mkdir -p ~/.local/bin
   echo '#!/bin/bash' > ~/.local/bin/tfan-gnome
   echo 'python3 ~/.local/share/tfan/tfan_gnome.py "$@"' >> ~/.local/bin/tfan-gnome
   chmod +x ~/.local/bin/tfan-gnome
   ```

3. **Install Desktop Entry:**
   ```bash
   cp com.quanta.tfan.desktop ~/.local/share/applications/
   update-desktop-database ~/.local/share/applications/
   ```

4. **Enable Extension:**
   ```bash
   # Restart GNOME Shell (Alt+F2, type 'r', press Enter)
   gnome-extensions enable tfan@quanta-meis-nib-cis
   ```

## Usage

### Launch Dashboard

```bash
tfan-gnome
```

Or click the T-FAN icon in your application menu!

### Load GitHub Repository

1. Open T-FAN dashboard
2. Navigate to "Repository" tab
3. Paste GitHub URL:
   ```
   https://github.com/theadamsfamily1981-max/Quanta-meis-nib-cis
   ```
4. Click "Clone & Configure"
5. App auto-installs dependencies and sets up workspace

### Start Training

**From Top Bar:**
- Click T-FAN indicator → "🚀 Start Training"

**From Dashboard:**
- Click "🚀 Start Training" button
- Select config from dropdown
- Monitor live metrics

### Run Pareto Optimization

**From Top Bar:**
- Click T-FAN indicator → "🎯 Pareto Optimization"

**From Dashboard:**
- Navigate to "Pareto" tab
- Set iterations and initial points
- Click "🎯 Run Optimization"
- View results in dashboard

### Watch Your Model's Topology in Real-Time 🌌

The **Screensaver** view visualizes your model's internal topological features as they evolve during training:

**Setup:**
1. Start the T-FAN API server:
   ```bash
   uvicorn api.main:app --reload
   ```

2. Launch training (in another terminal):
   ```bash
   python training/train.py
   ```

3. Open screensaver:
   - Click T-FAN indicator → "🌌 Topology Screensaver"
   - Or open dashboard → "Screensaver" tab
   - Or run: `tfan-gnome --view=screensaver`

**What You'll See:**
- **EPR-CV drives motion** - Higher entanglement fluctuation = faster, more chaotic particles
- **Topo Gap affects size** - Smaller gap (better topology preservation) = larger, more coherent particles
- **Live HUD overlay** - Real-time metrics from your training session
- **Connection status** - 🟢 Training Live (active training) / 🟡 Connected (API running) / ⚫ Demo Mode

**No training running?** The screensaver works in demo mode with simulated metrics. Start training to see your actual model's topology!

## Interface Guide

### Top Bar Indicator

The T-FAN icon in your top bar shows:
- **Idle** - No active training
- **Step 1234** - Current training step number

Click the indicator to access:
- 📊 Open Dashboard
- 🎯 Pareto Optimization
- 🌌 Topology Screensaver (NEW!)
- 🚀 Start Training
- 📈 Live Metrics (accuracy, latency, hypervolume)
- ⚙️ Settings

### Main Dashboard Views

1. **Dashboard** 📊
   - Live metrics cards
   - Training status
   - Quick actions

2. **Pareto** 🎯
   - Configure optimization
   - Run EHVI optimization
   - View Pareto front results

3. **Training** 🚀
   - Select configuration
   - Monitor training logs
   - Real-time metrics

4. **Screensaver** 🌌
   - **Live topology visualization** - Watch your model's internal geometry in real-time
   - **Metrics-driven animation** - Particle behavior reflects actual topology computations
   - 4 visualization modes (Barcode, Landscape, Poincaré, Pareto)
   - Interactive controls (mode, particle count, rotation)
   - HUD overlay showing live metrics:
     - **EPR-CV** → drives particle "tension" (motion speed)
     - **Topo Gap** → affects particle size (coherence indicator)
     - **Accuracy** → training progress
     - **Connection status** → Training Live / Connected / Demo Mode
   - Hardware-accelerated WebGL rendering
   - Fullscreen mode for immersive viewing

5. **Config** ⚙️
   - Edit training configs
   - Model architecture settings
   - Gate thresholds

6. **Repository** 📦
   - Load from GitHub
   - Auto-clone and configure
   - One-click setup

## Keyboard Shortcuts

- `Ctrl+Q` - Quit application
- `Ctrl+,` - Open preferences

## Dependencies

### Required

```bash
# System packages
sudo apt install python3-gi python3-gi-cairo gir1.2-gtk-4.0 gir1.2-adw-1

# Python packages
pip install PyGObject
```

### Optional

```bash
# For full functionality
pip install matplotlib numpy torch pyyaml
```

## File Structure

```
gnome-tfan/
├── extension/
│   ├── extension.js       # GNOME Shell extension
│   ├── metadata.json      # Extension metadata
│   └── assets/           # Icons and images
├── app/
│   └── tfan_gnome.py     # Main GTK4 application
├── install.sh            # Installation script
├── com.quanta.tfan.desktop  # Desktop entry
└── README.md
```

## Metrics Monitoring

The app monitors `~/.cache/tfan/metrics.json` for live updates.

**Example metrics.json:**
```json
{
  "training_active": true,
  "step": 1234,
  "accuracy": 0.923,
  "latency_ms": 145.2,
  "hypervolume": 47500,
  "epr_cv": 0.12
}
```

Create a simple training hook to export metrics:

```python
import json
from pathlib import Path

def export_metrics(step, accuracy, latency_ms, hypervolume, epr_cv):
    metrics_file = Path.home() / ".cache/tfan/metrics.json"
    metrics_file.parent.mkdir(exist_ok=True)

    with open(metrics_file, 'w') as f:
        json.dump({
            'training_active': True,
            'step': step,
            'accuracy': float(accuracy),
            'latency_ms': float(latency_ms),
            'hypervolume': float(hypervolume),
            'epr_cv': float(epr_cv),
        }, f)
```

## Styling

The app uses modern libadwaita components with custom CSS for a sick look:

- Gradient cards with glassmorphism
- Pulsing training indicator
- Smooth transitions
- Adaptive dark/light mode

## Troubleshooting

**Extension not showing up:**
```bash
# Check installation
ls ~/.local/share/gnome-shell/extensions/tfan@quanta-meis-nib-cis/

# Restart GNOME Shell
Alt+F2 → type 'r' → Enter

# Check extension status
gnome-extensions list
gnome-extensions enable tfan@quanta-meis-nib-cis
```

**App won't launch:**
```bash
# Check dependencies
python3 -c "import gi; gi.require_version('Gtk', '4.0'); gi.require_version('Adw', '1')"

# Run with debug output
python3 ~/.local/share/tfan/tfan_gnome.py
```

**Metrics not updating:**
```bash
# Check metrics file
cat ~/.cache/tfan/metrics.json

# Ensure training exports metrics (see above)
```

## Screenshots

*(To be added - the app looks absolutely fire with gradient cards, smooth animations, and modern libadwaita styling!)*

## Contributing

This integration is part of the T-FAN project. See main repository for contribution guidelines.

## License

MIT License - See main repository

---

**Made with ❤️ for the GNOME desktop**

*Topological, Fractal, Affective Neural networks meet beautiful, modern UI*
