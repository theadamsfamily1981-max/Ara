#!/bin/bash
#
# Cockpit HUD Launcher
#
# For FULL SYSTEM launch (Brain + Cockpit + Workspaces), use:
#   ~/Ara/launch_spaceship.sh
#
# This script launches just the Cockpit HUD.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARA_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║           ⚛  ARA COCKPIT HUD  ⚛                              ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Detect touchscreen monitor
if [ -z "$COCKPIT_MONITOR" ]; then
    COCKPIT_MONITOR=$(xrandr 2>/dev/null | grep " connected" | grep -E "800x1280|1080x1920|1200x1920" | awk '{print $1}' | head -1)

    if [ -z "$COCKPIT_MONITOR" ]; then
        echo "⚠ No portrait monitor detected, using primary"
        COCKPIT_MONITOR=$(xrandr 2>/dev/null | grep " connected primary" | awk '{print $1}')
    fi
fi

echo "🖥  Monitor: $COCKPIT_MONITOR"

# Install dependencies if needed
if ! python3 -c "import psutil" 2>/dev/null; then
    echo "📦 Installing psutil..."
    pip3 install psutil 2>/dev/null || pip install psutil
fi

if ! python3 -c "import GPUtil" 2>/dev/null; then
    echo "📦 Installing GPUtil for GPU monitoring..."
    pip3 install gputil 2>/dev/null || pip install gputil
fi

# Check GTK4
if ! python3 -c "import gi; gi.require_version('Gtk', '4.0')" 2>/dev/null; then
    echo "❌ ERROR: GTK4 not available"
    echo ""
    echo "Install with:"
    echo "  sudo apt install python3-gi python3-gi-cairo gir1.2-gtk-4.0 gir1.2-adw-1"
    exit 1
fi

cd "$SCRIPT_DIR"
echo "🚀 Launching Cockpit HUD..."
echo ""

# Launch cockpit
python3 cockpit_hud.py &
COCKPIT_PID=$!

sleep 2  # Wait for window to open

# Try to move to specific monitor
if [ ! -z "$COCKPIT_MONITOR" ] && [ "$COCKPIT_MONITOR" != "primary" ]; then
    if command -v wmctrl &> /dev/null; then
        WINDOW_ID=$(wmctrl -lp | grep $COCKPIT_PID | awk '{print $1}' | head -1)
        if [ ! -z "$WINDOW_ID" ]; then
            GEOMETRY=$(xrandr | grep "^$COCKPIT_MONITOR" | grep -o '[0-9]*x[0-9]*+[0-9]*+[0-9]*')
            if [ ! -z "$GEOMETRY" ]; then
                echo "📺 Moving window to $COCKPIT_MONITOR: $GEOMETRY"
                wmctrl -ir $WINDOW_ID -e 0,$GEOMETRY
            fi
        fi
    fi
fi

echo ""
echo "✓ Cockpit HUD running (PID: $COCKPIT_PID)"
echo ""
echo "TIP: For full system launch (Brain + Cockpit), run:"
echo "  $ARA_DIR/launch_spaceship.sh"
echo ""

wait $COCKPIT_PID
