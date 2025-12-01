#!/usr/bin/env bash
#
# T-FAN Topology Screensaver - xscreensaver wrapper
#
# This script is called by xscreensaver. It handles:
# - Starting metrics bridge if not running
# - Launching screensaver with proper display
# - Cleanup on exit
#

# Configuration
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
SCREENSAVER_DIR="$REPO_ROOT/tfan/viz/screensaver"
METRICS_BRIDGE_PORT=9101
METRICS_URL="http://localhost:$METRICS_BRIDGE_PORT/metrics"

# Check if metrics bridge is running
check_metrics_bridge() {
    curl -s "$METRICS_URL" > /dev/null 2>&1
    return $?
}

# Start metrics bridge in background
start_metrics_bridge() {
    cd "$SCREENSAVER_DIR"
    python metrics_bridge.py --port $METRICS_BRIDGE_PORT > /dev/null 2>&1 &
    BRIDGE_PID=$!
    echo $BRIDGE_PID > /tmp/tfan-metrics-bridge.pid
    sleep 1  # Give it time to start
}

# Cleanup function
cleanup() {
    if [[ -f /tmp/tfan-metrics-bridge.pid ]]; then
        BRIDGE_PID=$(cat /tmp/tfan-metrics-bridge.pid)
        if ps -p $BRIDGE_PID > /dev/null 2>&1; then
            kill $BRIDGE_PID 2>/dev/null || true
        fi
        rm -f /tmp/tfan-metrics-bridge.pid
    fi
}

# Set trap for cleanup
trap cleanup EXIT INT TERM

# Start metrics bridge if not running
if ! check_metrics_bridge; then
    start_metrics_bridge
fi

# Determine screensaver mode (cycle through modes or random)
MODES=(barcode landscape poincare pareto)
MODE=${MODES[$RANDOM % ${#MODES[@]}]}

# Launch screensaver
cd "$SCREENSAVER_DIR"
exec python topo_screensaver.py \
    --mode "$MODE" \
    --metrics "$METRICS_URL" \
    --fullscreen \
    "$@"
