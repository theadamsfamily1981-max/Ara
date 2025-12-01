# T-FAN Google Colab Notebooks 📓

**Zero-installation topology visualization in your browser**

Run T-FAN topology screensavers directly in Google Colab with free GPU access. No installation, no setup - just click and visualize!

---

## 🚀 Quick Start

### WebGL Version (Recommended) 🌐

**Best for:** Immediate interactive visualization, works on any device

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/theadamsfamily1981-max/Quanta-meis-nib-cis/blob/main/notebooks/TFAN_Topology_Screensaver_WebGL.ipynb)

```
1. Click badge above or open: TFAN_Topology_Screensaver_WebGL.ipynb
2. Run first cell (setup - no dependencies!)
3. Run second cell (launch screensaver)
4. Use form controls to adjust settings
5. Interact with mouse/keyboard
```

**Features:**
- ✅ Zero dependencies (pure HTML/JS/Three.js)
- ✅ Interactive controls (cycle modes, pause, camera)
- ✅ Adjustable particle counts and animation speed
- ✅ 60 FPS in browser
- ✅ Works on mobile/tablet
- ✅ Form-based UI (no code needed)

---

### Python Version 🌌

**Best for:** Real topology computation, mathematical rigor

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/theadamsfamily1981-max/Quanta-meis-nib-cis/blob/main/notebooks/TFAN_Topology_Screensaver_Python.ipynb)

```
1. Click badge above or open: TFAN_Topology_Screensaver_Python.ipynb
2. Run setup cell (installs ripser, persim - takes ~30 seconds)
3. Choose a visualization mode section
4. Adjust parameters with form controls
5. Run cell to generate visualization
```

**Features:**
- ✅ Real persistence diagram computation (Ripser)
- ✅ Actual persistence landscapes (Persim)
- ✅ Mathematical rigor
- ✅ Multiple modes (barcode, landscape, poincaré, pareto)
- ✅ Animation generation
- ✅ Export data (diagrams, landscapes)

---

## 📊 Comparison

| Feature | WebGL Version | Python Version |
|---------|---------------|----------------|
| **Setup Time** | Instant | ~30 seconds |
| **Dependencies** | None (CDN) | ripser, persim, scipy |
| **Interactivity** | Real-time 60 FPS | Static images + animation |
| **Topology** | Simplified | Real computation |
| **Controls** | Mouse + keyboard | Form parameters |
| **Best For** | Demos, exploration | Analysis, research |
| **Mobile** | ✅ Excellent | ✅ Works |
| **Export** | Screenshots | Data files, diagrams |

---

## 🎨 Visualization Modes

Both notebooks include all four modes:

### 1. Barcode Nebula 🌠
**WebGL:** 3D floating bars with dynamic lighting
**Python:** Real persistence barcodes from Ripser

- Computes topological features (H₀ connected components, H₁ loops)
- Bar length = persistence = robustness
- Color = feature strength

### 2. Landscape Waterfall 🌊
**WebGL:** Flowing landscape layers with temporal evolution
**Python:** Actual persistence landscapes (Bubenik 2015)

- Functional summaries: λ₁(t) ≥ λ₂(t) ≥ ... ≥ λₖ(t)
- Enables statistical analysis
- Visualizes topology evolution

### 3. Poincaré Orbits 🪐
**WebGL:** Interactive 3D hyperbolic disk
**Python:** Hierarchical embeddings with levels

- Hyperbolic geometry (constant negative curvature)
- Distance from center = hierarchy depth
- Geodesics = curved paths

### 4. Pareto Galaxy ⭐
**WebGL:** 3D star field with rotation
**Python:** Real non-dominated set computation

- Multi-objective optimization
- 5D → 2D/3D projection
- Pareto ratio calculation

---

## 💡 Usage Tips

### WebGL Version

**Customize appearance:**
```python
# In the form controls:
mode = "landscape"          # Choose your favorite mode
particle_count = 1200       # More = prettier, slower
auto_rotate = True          # Automatic camera rotation
show_metrics = True         # Show FPS and mode
animation_speed = 1.5       # Faster/slower evolution
```

**Keyboard shortcuts:**
- **M** - Cycle through modes
- **P** - Pause/unpause
- **R** - Reset camera

**Mouse controls:**
- **Left drag** - Rotate camera
- **Right drag** - Pan view
- **Scroll** - Zoom in/out

### Python Version

**Generate high-quality images:**
```python
# In each mode's form:
n_points = 1200             # More points = better topology
n_circles = 4               # (barcode) More features
noise_level = 0.04          # (barcode) Less noise = clearer
n_layers = 8                # (landscape) More detail
n_levels = 8                # (poincare) Deeper hierarchy
n_configs = 500             # (pareto) Larger search space
```

**Create animations:**
```python
# Run animation cell:
n_frames = 60               # Longer animation
animation_mode = "landscape"  # Your favorite mode
```

**Export data:**
```python
# Run export cell after visualization
# Saves .pkl and .txt files
# Download from Colab Files panel
```

---

## 📐 Mathematical Background

### Persistent Homology
Tracks topological features across scales:
- **H₀**: Connected components
- **H₁**: Loops/holes
- **H₂**: Voids/cavities

**Persistence** = death - birth = robustness to noise

### Persistence Landscapes
Functional summary enabling statistics:
```
λₖ(t) = k-th largest value at parameter t
Properties: stable, vectorizable, allows hypothesis testing
```

### Hyperbolic Geometry
Poincaré disk model:
```
{z ∈ ℂ : |z| < 1}
Hyperbolic distance: d(z,w) = arcosh(1 + 2|z-w|²/((1-|z|²)(1-|w|²)))
Exponential volume growth → natural for hierarchies
```

### Pareto Optimality
Non-dominated set in multi-objective space:
```
x dominates y iff: xᵢ ≤ yᵢ ∀i and xⱼ < yⱼ for some j
Pareto front = {x : no y dominates x}
```

---

## 🔧 Troubleshooting

### WebGL Version

**Blank screen:**
- Check browser console (F12)
- Try Chrome/Firefox latest
- Refresh page

**Slow performance:**
- Reduce `particle_count` to 400-600
- Disable `auto_rotate`
- Lower `animation_speed`

**Controls not working:**
- Click on visualization first
- Use buttons if keyboard doesn't work

### Python Version

**Installation errors:**
```python
# Try installing individually:
!pip install ripser
!pip install persim
!pip install numpy scipy matplotlib
```

**Computation too slow:**
- Reduce `n_points` to 400-600
- Use maxdim=1 instead of maxdim=2
- Simplify point cloud

**Animation fails:**
```python
# Reduce frames:
n_frames = 15

# Or try simpler mode:
animation_mode = "poincare"  # Faster than barcode
```

**No features found:**
- Increase `noise_level` for barcode
- Use more points
- Try different seed

---

## 📥 Saving Your Work

### WebGL Version
1. **Screenshots**: Right-click → Save image
2. **Full page**: Browser's save/print to PDF
3. **Share**: Share Colab notebook URL

### Python Version
1. **Figures**: Right-click on any plot → Save image
2. **Data**: Run export cell → Download from Files panel
3. **Notebook**: File → Download → Download .ipynb
4. **Animation**: Saves as HTML in cell output

---

## 🎓 Learning Resources

### For Beginners
- Start with **WebGL version** - instant gratification
- Explore all 4 modes with different settings
- Try keyboard/mouse controls
- Experiment with particle counts

### For Researchers
- Use **Python version** for real topology
- Export persistence diagrams for analysis
- Generate animations for presentations
- Customize point cloud generators

### For Developers
- Fork notebooks and modify
- Add custom visualization modes
- Integrate your own data
- Deploy as standalone apps

---

## 🔗 Integration

### With T-FAN API

Both notebooks can connect to live T-FAN metrics:

**WebGL version:**
```javascript
// Modify getWebSocketURL() in notebook:
return 'ws://your-api-server:8000/ws/metrics';
```

**Python version:**
```python
# Add metrics integration:
import requests
metrics = requests.get('http://your-api:8000/api/metrics').json()
# Use metrics to drive visualizations
```

### With Your Data

**Python version:**
```python
# Replace point cloud generators:
P = np.loadtxt('your_data.txt')  # Your point cloud
result = ripser(P, maxdim=1)      # Compute persistence
# Continue with visualization
```

**WebGL version:**
```javascript
// Modify data in createMode() functions:
positions.push(your_x, your_y, your_z);
```

---

## 🚀 Advanced Usage

### Batch Processing (Python)

```python
# Process multiple datasets
datasets = ['data1.txt', 'data2.txt', 'data3.txt']

results = []
for dataset in datasets:
    P = np.loadtxt(dataset)
    result = ripser(P, maxdim=1)
    results.append(result)

# Compare persistence diagrams
for i, result in enumerate(results):
    print(f"Dataset {i}: {len(result['dgms'][1])} H₁ features")
```

### Custom Colormaps (Python)

```python
# Use different color schemes
from matplotlib import cm

# Plasma colormap
colors = cm.plasma(persistence / persistence.max())

# Custom gradient
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list('custom', ['#667eea', '#764ba2'])
```

### Performance Optimization (WebGL)

```javascript
// In notebook cell, modify:
const PARTICLE_COUNT = navigator.hardwareConcurrency > 4 ? 1200 : 400;
const QUALITY = window.devicePixelRatio > 1 ? 2 : 1;
renderer.setPixelRatio(QUALITY);
```

---

## 📚 References

### Papers
- Edelsbrunner & Harer (2010), *Computational Topology*
- Bubenik (2015), *Statistical Topological Data Analysis using Persistence Landscapes*
- Bauer et al. (2021), *Ripser: Efficient Computation of Vietoris-Rips Persistence Barcodes*
- Nickel & Kiela (2017), *Poincaré Embeddings*

### Software
- [Ripser](https://ripser.scikit-tda.org/) - Fast persistence computation
- [Persim](https://persim.scikit-tda.org/) - Persistence landscapes
- [Three.js](https://threejs.org/) - WebGL 3D graphics
- [Scikit-TDA](https://scikit-tda.org/) - Topological data analysis

### Courses
- [Computational Topology (Edelsbrunner)](http://courses.csail.mit.edu/18.S096/fall20/)
- [Applied Algebraic Topology (Ghrist)](https://www.math.upenn.edu/~ghrist/notes.html)
- [TDA Course (Carlsson)](http://web.stanford.edu/class/math233/)

---

## 🤝 Contributing

Found a bug or want to add features?

1. **Fork** the repository
2. **Modify** notebooks
3. **Test** in Colab
4. **Submit** pull request

Ideas for contributions:
- New visualization modes
- Additional topological features (H₂, H₃)
- Real-time data streaming
- VR/AR support
- Audio reactivity
- Custom color schemes
- Statistical analysis tools

---

## 📄 License

Part of the T-FAN project. See main repository LICENSE.

---

## 🎉 Get Started Now!

**WebGL (Instant):**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/theadamsfamily1981-max/Quanta-meis-nib-cis/blob/main/notebooks/TFAN_Topology_Screensaver_WebGL.ipynb)

**Python (Rigorous):**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/theadamsfamily1981-max/Quanta-meis-nib-cis/blob/main/notebooks/TFAN_Topology_Screensaver_Python.ipynb)

---

**Experience the living mathematics of topology - in your browser, with zero installation!** 🌌✨

*"Mathematics is the art of giving the same name to different things." — Henri Poincaré*
