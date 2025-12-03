#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
#  ARA COCKPIT - Quick Setup Script
# ═══════════════════════════════════════════════════════════════════════════════
#
#  Fast installation for Ara Cockpit HUD only.
#  For full installation with all options, use: ./install.sh
#
# ═══════════════════════════════════════════════════════════════════════════════

set -e

echo ""
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║           ARA COCKPIT - Quick Setup                               ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check for sudo
if ! command -v sudo &> /dev/null; then
    echo "Error: sudo is required"
    exit 1
fi

echo "[1/4] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3 \
    python3-pip \
    python3-gi \
    python3-gi-cairo \
    gir1.2-gtk-4.0 \
    gir1.2-adw-1 \
    gir1.2-gst-plugins-base-1.0 \
    gir1.2-gstreamer-1.0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-gtk4 \
    2>/dev/null

# Optional WebKit
sudo apt-get install -y -qq gir1.2-webkit-6.0 2>/dev/null || true

echo "✓ System dependencies installed"

echo "[2/4] Installing Python packages..."
pip3 install --user -q psutil requests GPUtil 2>/dev/null || \
    pip3 install --user -q psutil requests

echo "✓ Python packages installed"

echo "[3/4] Setting up launcher..."
chmod +x "$SCRIPT_DIR/gnome-tfan/app/cockpit_hud.py" 2>/dev/null || true

# Create quick launcher
cat > "$SCRIPT_DIR/run-cockpit.sh" << 'EOF'
#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")/gnome-tfan/app"
python3 cockpit_hud.py "$@"
EOF
chmod +x "$SCRIPT_DIR/run-cockpit.sh"

echo "✓ Launcher ready"

echo "[4/4] Verifying installation..."
if python3 -c "import gi; gi.require_version('Gtk', '4.0'); from gi.repository import Gtk" 2>/dev/null; then
    echo "✓ GTK4 working"
else
    echo "✗ GTK4 not working - please check installation"
    exit 1
fi

if python3 -c "import psutil" 2>/dev/null; then
    echo "✓ psutil working"
else
    echo "✗ psutil not installed"
    exit 1
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "  ✓ Setup complete!"
echo ""
echo "  To launch the Ara Cockpit:"
echo "    ./run-cockpit.sh"
echo ""
echo "  Options:"
echo "    ./run-cockpit.sh --fullscreen    # Fullscreen mode"
echo "    ./run-cockpit.sh --help          # Show help"
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo ""
