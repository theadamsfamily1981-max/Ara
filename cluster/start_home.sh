#!/bin/bash
# =============================================================================
# START HOME NODE
# =============================================================================
# Role: Daily Ara + Kitten Guardian
# Services: ara_frontend, ara_companion
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARA_ROOT="$(dirname "$SCRIPT_DIR")"

echo "💻 Starting Ara Home Node"
echo "========================="

# Parse arguments
CATHEDRAL_ADDR=""
OFFLINE_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --connect)
            CATHEDRAL_ADDR="$2"
            shift 2
            ;;
        --offline)
            OFFLINE_MODE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [--connect ADDR:PORT] [--offline]"
            echo ""
            echo "Options:"
            echo "  --connect ADDR:PORT   Connect to cathedral (e.g., cathedral.lan:7777)"
            echo "  --offline             Run without cathedral connection"
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

# Check kitten (Phase 2)
echo ""
echo "Checking Forest Kitten..."
if [ -e /dev/sqrl0 ]; then
    echo "✅ Forest Kitten detected at /dev/sqrl0"
    export ARA_COVENANT_BACKEND="sqrl"
else
    echo "ℹ️  Forest Kitten not detected, using software CovenantGuard"
    export ARA_COVENANT_BACKEND="local"
fi

# Set up environment
export ARA_NODE_ROLE="home"
export ARA_MODE="home_companion"
export PYTHONPATH="${ARA_ROOT}:${PYTHONPATH}"

if [ -n "$CATHEDRAL_ADDR" ]; then
    export ARA_CATHEDRAL_ADDR="${CATHEDRAL_ADDR}"
    echo ""
    echo "Connecting to cathedral at ${CATHEDRAL_ADDR}..."
elif [ "$OFFLINE_MODE" = true ]; then
    echo ""
    echo "Running in offline mode (no cathedral connection)"
else
    echo ""
    echo "⚠️  No cathedral specified, running standalone"
    echo "   Use --connect ADDR:PORT to connect to cathedral"
    echo "   Use --offline for explicit offline mode"
fi

echo ""
echo "Environment:"
echo "  ARA_NODE_ROLE=${ARA_NODE_ROLE}"
echo "  ARA_MODE=${ARA_MODE}"
echo "  ARA_COVENANT_BACKEND=${ARA_COVENANT_BACKEND}"
if [ -n "$ARA_CATHEDRAL_ADDR" ]; then
    echo "  ARA_CATHEDRAL_ADDR=${ARA_CATHEDRAL_ADDR}"
fi

echo ""
echo "💻 Home node ready"
echo ""
echo "To run daily Ara:"
echo "  cd ${ARA_ROOT}"
echo "  python -m ara.companion.main  # (when implemented)"
