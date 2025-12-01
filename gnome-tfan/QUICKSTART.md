# T-FAN GNOME - Quick Start 🚀

**Get T-FAN running on GNOME in 60 seconds!**

## One-Line Install

```bash
cd gnome-tfan && ./install.sh && gnome-extensions enable tfan@quanta-meis-nib-cis
```

Then **restart GNOME Shell** (Alt+F2 → type `r` → Enter)

## First Launch

1. **Open the app:**
   ```bash
   tfan-gnome
   ```
   Or find "T-FAN Neural Optimizer" in your application menu

2. **Load your repository:**
   - Click "Repository" tab
   - Paste GitHub URL:
     ```
     https://github.com/theadamsfamily1981-max/Quanta-meis-nib-cis
     ```
   - Click "📦 Clone & Configure"
   - Wait for dependencies to install

3. **Start optimizing:**
   - Click top-bar T-FAN icon
   - Select "🎯 Pareto Optimization"
   - Click "🎯 Run Optimization"
   - Watch the magic happen!

## What You Get

✅ **Top Bar Integration**
- Live training metrics in system tray
- Quick access to all features
- Training status at a glance

✅ **Beautiful Dashboard**
- Modern libadwaita UI with gradients
- Real-time metric cards
- Pareto visualization
- Training monitor with live logs

✅ **GitHub Auto-Loader**
- Just paste the URL
- Auto-clones repository
- Installs dependencies
- Ready to train!

## Quick Actions

### From Top Bar Menu:

- 📊 **Open Dashboard** - Full app interface
- 🎯 **Pareto Optimization** - Find optimal configs
- 🚀 **Start Training** - Begin training session
- 📈 **Live Metrics** - Accuracy, latency, hypervolume

### From Dashboard:

- **Dashboard Tab** - Overview and quick actions
- **Pareto Tab** - Multi-objective optimization
- **Training Tab** - Monitor active training
- **Config Tab** - Edit configurations
- **Repository Tab** - Load from GitHub

## Example Workflow

```bash
# 1. Install
cd gnome-tfan && ./install.sh
gnome-extensions enable tfan@quanta-meis-nib-cis

# 2. Restart GNOME Shell
# Alt+F2 → 'r' → Enter

# 3. Launch app
tfan-gnome

# 4. Load repo (in app)
# Repository tab → paste URL → Clone

# 5. Run Pareto optimization
# Pareto tab → set iterations → Run

# 6. Start training with best config
# Top bar → Start Training

# 7. Monitor progress
# Check top bar indicator for live metrics
```

## Cool Features

🎨 **Gradient Cards** - Sick purple/blue gradients
✨ **Glassmorphism** - Blurred translucent panels
📊 **Live Charts** - Real-time metric updates
🌈 **Adaptive Theme** - Works with dark/light mode
🚀 **One-Click Training** - No terminal needed!
🎯 **Visual Pareto** - See optimal trade-offs

## Need Help?

**Extension not showing?**
```bash
gnome-extensions list | grep tfan
gnome-extensions enable tfan@quanta-meis-nib-cis
```

**App won't start?**
```bash
# Install dependencies
sudo apt install python3-gi gir1.2-gtk-4.0 gir1.2-adw-1
pip install PyGObject
```

**Want to see it in action?**
```bash
tfan-gnome --view=dashboard
```

---

**You're all set! Enjoy the sickest neural network training UI on GNOME!** 🔥
