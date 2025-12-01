# T-FAN Futuristic Icon Pack 🎨✨

Modern, neon-accented icon set matching the T-FAN cockpit aesthetic.

## Icon List

### Core Icons

| Icon | File | Description | Colors |
|------|------|-------------|--------|
| 🔷 | `tfan-icon.svg` | Main T-FAN logo | Blue/Cyan gradient |
| 🌌 | `topology-icon.svg` | Topology visualization | Purple/Pink gradient |
| ⚡ | `work-mode-icon.svg` | Work mode indicator | Blue/Cyan (energetic) |
| 🌙 | `relax-mode-icon.svg` | Relax mode indicator | Purple/Pink (calming) |
| 📈 | `metrics-icon.svg` | Metrics/graphs | Green/Cyan |
| ⭐ | `pareto-icon.svg` | Pareto optimization | Gold/Pink |
| 🧠 | `training-icon.svg` | Neural network training | Pink/Gold |

## Design System

### Color Palettes

**Work Mode (Cool Spectrum)**
- Primary: `#667eea` → `#00d4ff`
- Accent: `#00ffff` (neon cyan)
- Glow: Blue/cyan with 4px blur

**Relax Mode (Warm Spectrum)**
- Primary: `#764ba2` → `#ff6ec7`
- Accent: `#ff00ff` (neon magenta)
- Glow: Purple/pink with 5px blur

### Visual Style

- **Geometric**: Hexagons, clean lines, network graphs
- **Glow effects**: Gaussian blur filters for neon look
- **Gradients**: Linear gradients in 135° direction
- **Transparency**: Semi-transparent elements for depth
- **Scalable**: SVG format, crisp at any size

## Usage in GTK App

```python
import gi
gi.require_version('Gtk', '4.0')
from gi.repository import Gtk, Gio

# Load icon
icon = Gio.Icon.new_for_string('icons/scalable/tfan-icon.svg')

# Use in button
button = Gtk.Button()
button.set_icon_name('tfan-icon')

# Use in image
image = Gtk.Image.new_from_file('icons/scalable/work-mode-icon.svg')
```

## Building Icon Theme

To install as system icon theme:

```bash
# Copy to icon theme directory
sudo mkdir -p /usr/share/icons/TFANFuturistic
sudo cp -r icons/* /usr/share/icons/TFANFuturistic/

# Update icon cache
sudo gtk-update-icon-cache /usr/share/icons/TFANFuturistic

# Set as active theme
gsettings set org.gnome.desktop.interface icon-theme 'TFANFuturistic'
```

## Customization

Each icon uses defined color gradients. To customize:

1. Open SVG in text editor
2. Modify `<linearGradient>` stops
3. Adjust `filter` blur values for glow intensity
4. Change stroke-width for line thickness

Example - change work mode to green theme:
```xml
<linearGradient id="workGrad">
  <stop offset="0%" style="stop-color:#00ff88;stop-opacity:1" />
  <stop offset="100%" style="stop-color:#00d4ff;stop-opacity:1" />
</linearGradient>
```

## Icon Requests

Need additional icons? Follow the template:

```xml
<svg width="256" height="256" viewBox="0 0 256 256">
  <defs>
    <linearGradient id="yourGrad">
      <stop offset="0%" style="stop-color:#startColor"/>
      <stop offset="100%" style="stop-color:#endColor"/>
    </linearGradient>
    <filter id="yourGlow">
      <feGaussianBlur stdDeviation="4"/>
      <feMerge>
        <feMergeNode in="coloredBlur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
  </defs>

  <!-- Your icon shapes here -->
  <path d="..." fill="url(#yourGrad)" filter="url(#yourGlow)"/>
</svg>
```

## License

MIT - Free to use and modify for T-FAN project and derivatives.
