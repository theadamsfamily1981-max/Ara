#!/bin/bash
#
# Start Ara Host Services
#
# Runs the "Host" side of Ara:
#   - HAL daemon (somatic link, shared memory)
#   - MIES / Cockpit overlay (GTK4 UI)
#
# These need direct OS access (display server, D-Bus, PipeWire)
# so they run on the host, not in Docker.
#
# Usage:
#   ./start_host.sh           # Start both HAL and Cockpit
#   ./start_host.sh --hal     # HAL only
#   ./start_host.sh --cockpit # Cockpit only
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARA_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_PATH="$ARA_ROOT/venv_host"

# Parse arguments
RUN_HAL=true
RUN_COCKPIT=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --hal)
            RUN_HAL=true
            RUN_COCKPIT=false
            shift
            ;;
        --cockpit)
            RUN_HAL=false
            RUN_COCKPIT=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--hal] [--cockpit]"
            exit 1
            ;;
    esac
done

# Check venv
if [ ! -d "$VENV_PATH" ]; then
    echo "venv_host not found at $VENV_PATH"
    echo "Run: ./bootstrap_organism.sh --host"
    exit 1
fi

# Activate venv
source "$VENV_PATH/bin/activate"

# Set environment
export ARA_ROOT="$ARA_ROOT"
export ARA_HAL_PATH="/dev/shm/ara_somatic"
export PYTHONPATH="$ARA_ROOT:$PYTHONPATH"
export PYTHONUNBUFFERED=1

echo "Ara Host Environment Activated"
echo "  ARA_ROOT: $ARA_ROOT"
echo "  Python: $(which python)"
echo "  HAL Path: $ARA_HAL_PATH"

# Cleanup function
cleanup() {
    echo ""
    echo "Shutting down..."
    if [ -n "$HAL_PID" ]; then
        kill $HAL_PID 2>/dev/null || true
        echo "Stopped HAL (PID: $HAL_PID)"
    fi
    if [ -n "$COCKPIT_PID" ]; then
        kill $COCKPIT_PID 2>/dev/null || true
        echo "Stopped Cockpit (PID: $COCKPIT_PID)"
    fi
    exit 0
}
trap cleanup SIGINT SIGTERM

# Start HAL daemon
if [ "$RUN_HAL" = true ]; then
    echo ""
    echo "Starting HAL daemon..."

    # Find HAL entrypoint
    if [ -f "$ARA_ROOT/banos/daemon/ara_daemon.py" ]; then
        python3 -m banos.daemon.ara_daemon &
        HAL_PID=$!
        echo "  HAL started (PID: $HAL_PID)"
    elif [ -f "$ARA_ROOT/banos/hal/ara_hal.py" ]; then
        python3 -m banos.hal.ara_hal &
        HAL_PID=$!
        echo "  HAL started (PID: $HAL_PID)"
    else
        echo "  Warning: HAL entrypoint not found"
        echo "  Expected: banos/daemon/ara_daemon.py or banos/hal/ara_hal.py"
    fi

    # Wait for HAL to initialize
    sleep 1

    # Check if HAL created shared memory
    if [ -f "$ARA_HAL_PATH" ]; then
        echo "  HAL shared memory active"
    else
        echo "  Warning: HAL shared memory not found at $ARA_HAL_PATH"
    fi
fi

# Start Cockpit / MIES
if [ "$RUN_COCKPIT" = true ]; then
    echo ""
    echo "Starting Cockpit / MIES overlay..."

    # Find cockpit entrypoint
    if [ -f "$ARA_ROOT/cockpit/cockpit_hud.py" ]; then
        python3 "$ARA_ROOT/cockpit/cockpit_hud.py" &
        COCKPIT_PID=$!
        echo "  Cockpit started (PID: $COCKPIT_PID)"
    elif [ -f "$ARA_ROOT/multi-ai-workspace/src/integrations/ara_avatar_backend.py" ]; then
        python3 -m multi_ai_workspace.src.integrations.ara_avatar_backend &
        COCKPIT_PID=$!
        echo "  Avatar backend started (PID: $COCKPIT_PID)"
    else
        echo "  Warning: Cockpit entrypoint not found"
        echo "  Expected: cockpit/cockpit_hud.py or ara_avatar_backend.py"
    fi
fi

echo ""
echo "Host services running. Ctrl+C to stop."

# Wait for children
wait
