# T-FAN WebGL Screensaver - 60 Second Quick Start ⚡

## Option 1: Simple HTTP Server (Fastest)

```bash
cd tfan/viz/screensaver/web
python serve.py
# Open http://localhost:8080
```

Press **M** to cycle modes, **H** for help, **F** for fullscreen.

---

## Option 2: With Live Metrics

**Terminal 1:** Start T-FAN API
```bash
uvicorn api.main:app --reload
```

**Terminal 2:** Start screensaver
```bash
cd tfan/viz/screensaver/web
python serve.py
# Open http://localhost:8080
```

Watch metrics update in real-time!

---

## Option 3: Browser Extension

```bash
# 1. Copy files to extension
cd tfan/viz/screensaver/web
cp index.html screensaver.js styles.css extension/

# 2. Load in Chrome
# - Open chrome://extensions/
# - Enable "Developer mode"
# - Click "Load unpacked"
# - Select extension/ directory

# 3. Open new tab
# Screensaver appears automatically!
```

---

## Option 4: Direct File Access

```bash
# Just double-click
open index.html

# Or from command line
open -a "Google Chrome" index.html
```

**Note:** WebSocket features require HTTP server.

---

## Controls

| Key | Action |
|-----|--------|
| **M** | Cycle through 4 modes |
| **P** | Pause/unpause |
| **H** | Toggle help |
| **F** | Fullscreen |
| **Q** | Exit fullscreen |

---

## 4 Visualization Modes

1. **Barcode Nebula** 🌠 - 3D persistence barcodes
2. **Landscape Waterfall** 🌊 - Flowing persistence landscapes
3. **Poincaré Orbits** 🪐 - Hyperbolic embeddings
4. **Pareto Galaxy** ⭐ - Multi-objective stars

---

## Troubleshooting

**Black screen?**
- Open browser console (F12)
- Check for WebGL support
- Try different browser

**Metrics not updating?**
- Start T-FAN API on localhost:8000
- Or run metrics bridge
- Check WebSocket status (top right)

**Performance issues?**
- Edit screensaver.js
- Reduce particleCount
- Lower setPixelRatio

---

## Next Steps

- **Fullscreen**: Press **F** for immersive experience
- **Kiosk Mode**: `chromium --kiosk --app=http://localhost:8080`
- **Deploy**: Host on Netlify, Vercel, or GitHub Pages
- **Customize**: Edit colors, particle counts, animations

---

**Enjoy the living mathematics!** ✨
