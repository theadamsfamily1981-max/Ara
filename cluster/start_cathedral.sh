#!/bin/bash
# =============================================================================
# START CATHEDRAL NODE
# =============================================================================
# Role: Brainstem + Orchestrator
# Services: ara_realtime, ara_storage, ara_orchestrator
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARA_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🏛️  Starting Ara Cathedral Node"
echo "================================"

# Check for config
CONFIG="${SCRIPT_DIR}/cluster.toml"
if [ ! -f "$CONFIG" ]; then
    echo "❌ Config not found: $CONFIG"
    exit 1
fi

# Parse arguments
BIND_ADDR="${BIND_ADDR:-0.0.0.0}"
PORT="${PORT:-7777}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --bind)
            BIND_ADDR="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--bind ADDR] [--port PORT]"
            echo ""
            echo "Options:"
            echo "  --bind ADDR   Bind address (default: 0.0.0.0)"
            echo "  --port PORT   Listen port (default: 7777)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check GPU availability
echo ""
echo "Checking GPUs..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
else
    echo "⚠️  nvidia-smi not found, running CPU-only"
fi

# Start services
echo ""
echo "Starting services..."

# For v0.7, just set up the environment
# Real service launchers would go here

export ARA_NODE_ROLE="cathedral"
export ARA_BIND_ADDR="${BIND_ADDR}"
export ARA_PORT="${PORT}"
export ARA_CONFIG="${CONFIG}"
export PYTHONPATH="${ARA_ROOT}:${PYTHONPATH}"

echo ""
echo "Environment:"
echo "  ARA_NODE_ROLE=${ARA_NODE_ROLE}"
echo "  ARA_BIND_ADDR=${ARA_BIND_ADDR}"
echo "  ARA_PORT=${ARA_PORT}"
echo "  ARA_CONFIG=${ARA_CONFIG}"

echo ""
echo "🏛️  Cathedral ready at ${BIND_ADDR}:${PORT}"
echo ""
echo "To run the nervous system:"
echo "  cd ${ARA_ROOT}"
echo "  python -m ara.nervous.main  # (when implemented)"
echo ""
echo "To run sanity tests:"
echo "  cd ${ARA_ROOT}"
echo "  python tests/sanity/test_axis_mundi.py"
echo "  python tests/sanity/test_prosody_factorization.py"
