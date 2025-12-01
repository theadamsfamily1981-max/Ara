#!/bin/bash
# T-FAN Cockpit HUD Launcher
# Launches full-screen cockpit on side touchscreen

# Detect touchscreen monitor
# You can override by setting COCKPIT_MONITOR env variable
if [ -z "$COCKPIT_MONITOR" ]; then
    # Try to auto-detect portrait monitor or touchscreen
    COCKPIT_MONITOR=$(xrandr | grep " connected" | grep -E "800x1280|1080x1920|1200x1920" | awk '{print $1}' | head -1)

    if [ -z "$COCKPIT_MONITOR" ]; then
        echo "⚠ No portrait monitor detected, using primary"
        COCKPIT_MONITOR=$(xrandr | grep " connected primary" | awk '{print $1}')
    fi
fi

echo "🚀 Launching T-FAN Cockpit HUD on monitor: $COCKPIT_MONITOR"

# Install dependencies if needed
if ! python3 -c "import psutil" 2>/dev/null; then
    echo "📦 Installing psutil..."
    pip install psutil
fi

if ! python3 -c "import GPUtil" 2>/dev/null; then
    echo "📦 Installing GPUtil for GPU monitoring..."
    pip install gputil
fi

# Launch cockpit in fullscreen on specific monitor
if [ ! -z "$COCKPIT_MONITOR" ]; then
    # Move window to specific monitor (requires wmctrl)
    python3 cockpit_hud.py &
    COCKPIT_PID=$!

    sleep 2  # Wait for window to open

    # Try to move to specific monitor
    if command -v wmctrl &> /dev/null; then
        WINDOW_ID=$(wmctrl -lp | grep $COCKPIT_PID | awk '{print $1}' | head -1)
        if [ ! -z "$WINDOW_ID" ]; then
            # Get monitor geometry
            GEOMETRY=$(xrandr | grep "^$COCKPIT_MONITOR" | grep -o '[0-9]*x[0-9]*+[0-9]*+[0-9]*')
            echo "Moving window to $COCKPIT_MONITOR: $GEOMETRY"
            wmctrl -ir $WINDOW_ID -e 0,$GEOMETRY
        fi
    fi

    wait $COCKPIT_PID
else
    # Just launch normally
    python3 cockpit_hud.py
fi
