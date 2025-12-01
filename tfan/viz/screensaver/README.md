# T-FAN Topology Screensaver 🌌

**Living visualization of T-FAN's mathematical foundations**

Transform your idle screen into a mesmerizing display of topological data analysis, hyperbolic geometry, and multi-objective optimization in real-time.

## 🎯 Two Versions Available

### Python/VisPy Version (Desktop)
High-performance OpenGL screensaver for Linux desktops with RTX GPUs.
- **Files**: `topo_screensaver.py`, `metrics_bridge.py`
- **Best for**: Native Linux screensaver, maximum performance
- **Platforms**: Linux, macOS, Windows (with OpenGL)

### WebGL Version (Browser) 🌐
Cross-platform browser-based screensaver using Three.js.
- **Files**: `web/` directory
- **Best for**: Any device, browser extension, web deployment
- **Platforms**: Any modern browser (Chrome, Firefox, Safari, Edge, mobile)
- **See**: `web/README.md` for documentation

---

---

## 🎨 Visualization Modes

Press **M** to cycle through these modes:

### 1. Barcode Nebula 🌠
Animated **persistence barcodes** from a streaming point cloud. Each horizontal bar represents a topological feature (connected component, hole, void) with:
- **Length** = persistence (how long the feature survives across scales)
- **Color** = persistence strength (longer bars are brighter)
- **Animation** = point cloud drifts, features appear/disappear

*Mathematical foundation: Vietoris-Rips filtration on R²*

### 2. Landscape Waterfall 🌊
Stacked **persistence landscapes** (λ₁, λ₂, ..., λₖ) flowing in time:
- **Height** = prominence of topological features
- **Layers** = persistence landscape levels (1st most prominent → kth)
- **Flow** = temporal evolution as data topology changes

*Mathematical foundation: Persistence landscapes for H₀ homology*

### 3. Poincaré Orbits 🪐
**Hyperbolic embeddings** visualized on the Poincaré disk:
- **Distance from center** = hierarchy level (deeper = higher)
- **Geodesics** = hyperbolic distances (curved paths)
- **Orbits** = entities drift along hierarchy branches
- **Curvature halos** = encode hierarchy strength

*Mathematical foundation: Poincaré disk model of hyperbolic space*

### 4. Pareto Galaxy ⭐
**Non-dominated configurations** as stars in objective space:
- **Position** = 2D projection of 5-objective space
- **Color** = objective mix (accuracy, latency, EPR CV, topo gap, energy)
- **Size** = hypervolume contribution
- **Frontier** = non-dominated points glow brighter

*Mathematical foundation: Pareto optimality in multi-objective optimization*

---

## 🔥 Telemetry Integration

The screensaver reacts to live T-FAN metrics:

| Metric | Effect |
|--------|--------|
| **EPR-CV** | Global "tension" → animation speed + shader noise |
| **TTW p95** | Spike detection → ripple bursts |
| **Topo cosine** | Palette shift cool ↔ warm |
| **Wasserstein gap** | Feature prominence |
| **PGU p95** | Stability rings on Poincaré disk |
| **Accuracy** | Displayed in status bar |
| **Latency** | Displayed in status bar |
| **Hypervolume** | Displayed in status bar |

---

## 🚀 Quick Start

### Installation

```bash
cd tfan/viz/screensaver
./install.sh
```

Or manually:
```bash
# System dependencies (Ubuntu/Debian)
sudo apt-get install python3-dev python3-pip libgl1-mesa-glx libegl1

# Python dependencies
pip install -r requirements-viz.txt
```

### Run Standalone

```bash
# Basic usage
python topo_screensaver.py --mode landscape

# Fullscreen
python topo_screensaver.py --fullscreen

# With live metrics
python topo_screensaver.py --metrics http://localhost:8000/api/metrics --fullscreen

# All modes
python topo_screensaver.py --mode barcode
python topo_screensaver.py --mode landscape
python topo_screensaver.py --mode poincare
python topo_screensaver.py --mode pareto
```

### With Live Metrics

**Option 1: FastAPI Server (Recommended)**

If you have the T-FAN API running:
```bash
# Terminal 1: Start T-FAN API
uvicorn api.main:app --reload

# Terminal 2: Launch screensaver
python topo_screensaver.py --metrics http://localhost:8000/api/metrics --fullscreen
```

**Option 2: Metrics Bridge**

For standalone deployment without the full API:
```bash
# Terminal 1: Start metrics bridge
python metrics_bridge.py --port 9101

# Terminal 2: Launch screensaver
python topo_screensaver.py --metrics http://localhost:9101/metrics --fullscreen
```

The metrics bridge reads from `~/.cache/tfan/metrics.json` and serves it over HTTP.

---

## 🖥️ Linux Screensaver Integration

### xscreensaver (Classic)

1. **Install xscreensaver:**
   ```bash
   sudo apt-get install xscreensaver xscreensaver-data-extra
   ```

2. **Create wrapper script:**
   ```bash
   mkdir -p ~/bin
   cat > ~/bin/tfan-topo-saver << 'EOF'
   #!/usr/bin/env bash
   cd /absolute/path/to/Quanta-meis-nib-cis/tfan/viz/screensaver
   exec python topo_screensaver.py --fullscreen "$@"
   EOF
   chmod +x ~/bin/tfan-topo-saver
   ```

3. **Configure xscreensaver:**
   - Run: `xscreensaver-demo`
   - Click "Settings" → "Advanced" → "Programs"
   - Click "Add Program"
   - Name: "T-FAN Topology"
   - Command: `/home/youruser/bin/tfan-topo-saver`
   - Click "OK"
   - Select "T-FAN Topology" from the list

4. **Enable xscreensaver:**
   ```bash
   xscreensaver -no-splash &
   ```

### GNOME Screensaver

GNOME 3.8+ removed custom screensaver support, but you can use a workaround:

**Option A: Disable lock screen and use xscreensaver**
```bash
gsettings set org.gnome.desktop.lockdown disable-lock-screen true
xscreensaver -no-splash &
```

**Option B: Live Wallpaper (xwinwrap)**
```bash
# Install xwinwrap
sudo apt-get install xwinwrap

# Run as live wallpaper
xwinwrap -ni -fs -s -st -sp -b -nf -- \
  python topo_screensaver.py --mode landscape --fullscreen
```

### KDE Plasma

```bash
# Add to autostart
mkdir -p ~/.config/autostart
cat > ~/.config/autostart/tfan-screensaver.desktop << EOF
[Desktop Entry]
Type=Application
Name=T-FAN Topology Screensaver
Exec=/home/youruser/bin/tfan-topo-saver --fullscreen
X-KDE-autostart-after=panel
EOF
```

---

## ⌨️ Keyboard Controls

| Key | Action |
|-----|--------|
| **M** or **Tab** | Cycle through visualization modes |
| **P** or **Space** | Pause/unpause animation |
| **Q** or **Esc** | Quit screensaver |

---

## 🔧 Configuration

### Telemetry Endpoint

The screensaver polls metrics every 2 seconds from:
- T-FAN API: `http://localhost:8000/api/metrics`
- Metrics Bridge: `http://localhost:9101/metrics`
- Custom endpoint: `--metrics http://your-server/endpoint`

**Expected JSON format:**
```json
{
  "training_active": true,
  "step": 12340,
  "accuracy": 0.923,
  "latency_ms": 145.2,
  "hypervolume": 47500,
  "epr_cv": 0.12,
  "topo_gap": 0.015,
  "energy": 0.8,
  "timestamp": "2025-11-17T12:34:56"
}
```

### Performance Tuning

**High-end GPU (RTX 3090+):**
```python
# In topo_screensaver.py, increase detail:
FPS = 120  # Smoother animation
n_points = 2400  # More points
resolution = 800  # Higher resolution landscapes
```

**Low-end GPU:**
```python
FPS = 30
n_points = 600
resolution = 200
```

---

## 📐 Mathematical Background

### Persistence Barcodes
- Computed via **Ripser** (fast Vietoris-Rips persistence)
- Visualizes birth/death of topological features across scales
- Bar length = persistence = robustness to noise

### Persistence Landscapes
- Functional summary of persistence diagrams
- Stack of piecewise-linear functions λ₁ ≥ λ₂ ≥ ... ≥ λₖ
- Enable statistical analysis of topology

### Poincaré Disk
- Hyperbolic space model (constant negative curvature)
- Geodesics are circular arcs orthogonal to boundary
- Hierarchical data naturally embeds with exponential volume growth

### Pareto Front
- Non-dominated points in multi-objective optimization
- No point strictly better in all objectives
- Visualized via 2D projection from 5D objective space

---

## 🎬 Demo Videos

Generate demo videos with:
```bash
# Install ffmpeg
sudo apt-get install ffmpeg

# Record 60 seconds of barcode mode
python topo_screensaver.py --mode barcode &
PID=$!
sleep 2
ffmpeg -video_size 1920x1080 -framerate 60 -f x11grab -i :0.0 -t 60 barcode_demo.mp4
kill $PID
```

---

## 🐛 Troubleshooting

**OpenGL errors:**
```bash
# Check OpenGL support
glxinfo | grep "OpenGL version"

# If missing, install Mesa drivers
sudo apt-get install mesa-utils libgl1-mesa-dri
```

**VisPy backend issues:**
```bash
# Try different backend
export VISPY_GL_LIB=/usr/lib/x86_64-linux-gnu/libGL.so.1
python topo_screensaver.py
```

**Metrics not updating:**
```bash
# Check metrics file exists
ls -l ~/.cache/tfan/metrics.json

# Emit test metrics
python -c "
import json
from pathlib import Path
p = Path.home() / '.cache/tfan/metrics.json'
p.parent.mkdir(exist_ok=True)
with open(p, 'w') as f:
    json.dump({'training_active': True, 'step': 1000, 'accuracy': 0.85, 'latency_ms': 120.0, 'hypervolume': 45000, 'epr_cv': 0.10, 'topo_gap': 0.012}, f)
"
```

**High CPU usage:**
```bash
# Reduce FPS
python topo_screensaver.py --fullscreen
# Edit FPS = 30 in topo_screensaver.py
```

---

## 🔬 Advanced Usage

### Custom Point Clouds

Replace data generators in `topo_screensaver.py`:
```python
def custom_dataset(n=1200, t=0.0):
    """Load your own time-varying point cloud."""
    # Example: Read from file
    data = np.load(f'pointclouds/frame_{int(t*10)}.npy')
    return data[:n]  # Return (n, 2) array
```

### Custom Colormaps

```python
from vispy import color
PALETTE = color.get_colormap('plasma')  # Try: plasma, magma, inferno, cividis
```

### Custom Telemetry Mapping

```python
# In Telemetry._poll():
self.values.update({
    'epr_cv': data['my_custom_epr'],
    'accuracy': data['val_acc'],
    # Map your own keys
})
```

---

## 🌐 WebGL Version (Coming Soon)

Browser-based screensaver with:
- Three.js renderer
- WebGL shaders for faster particle systems
- WebSocket live metrics
- Runs on any device with browser
- No installation required

---

## 📚 References

- **Ripser:** [ripser.scikit-tda.org](https://ripser.scikit-tda.org/)
- **Persistence Landscapes:** Bubenik (2015), *Statistical Topological Data Analysis*
- **Poincaré Embeddings:** Nickel & Kiela (2017), *Hyperbolic Embeddings*
- **Pareto Optimization:** Miettinen (1998), *Nonlinear Multiobjective Optimization*
- **VisPy:** [vispy.org](https://vispy.org/)

---

## 🤝 Contributing

Contributions welcome! Ideas:
- New visualization modes (e.g., Mapper graphs, UMAP)
- Shader-based rendering for 10x performance
- VR/AR support
- Audio reactivity (pulse with training loss)
- Multi-monitor spanning

---

## 📄 License

Part of the T-FAN project. See main repository LICENSE.

---

**Built with topology, hyperbolic geometry, and pure mathematical beauty** ✨

*"Mathematics is the art of giving the same name to different things." — Henri Poincaré*
