#!/bin/bash
# =============================================================================
# START WORKER NODE
# =============================================================================
# Role: Training Mule
# Services: ara_trainer
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARA_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🖥️  Starting Ara Worker Node"
echo "============================"

# Parse arguments
CATHEDRAL_ADDR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --connect)
            CATHEDRAL_ADDR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 --connect ADDR:PORT"
            echo ""
            echo "Options:"
            echo "  --connect ADDR:PORT   Connect to cathedral (required)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$CATHEDRAL_ADDR" ]; then
    echo "❌ Cathedral address required. Use --connect ADDR:PORT"
    exit 1
fi

# Check GPU availability
echo ""
echo "Checking GPUs..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
    echo "Found ${GPU_COUNT} GPU(s)"
else
    echo "❌ nvidia-smi not found, worker requires GPU"
    exit 1
fi

# Set up environment
export ARA_NODE_ROLE="worker"
export ARA_CATHEDRAL_ADDR="${CATHEDRAL_ADDR}"
export PYTHONPATH="${ARA_ROOT}:${PYTHONPATH}"

# Parse cathedral address for dataset mount
CATHEDRAL_HOST="${CATHEDRAL_ADDR%:*}"
export ARA_DATASET_MOUNT="${CATHEDRAL_HOST}:/data/ara"

echo ""
echo "Environment:"
echo "  ARA_NODE_ROLE=${ARA_NODE_ROLE}"
echo "  ARA_CATHEDRAL_ADDR=${ARA_CATHEDRAL_ADDR}"
echo "  ARA_DATASET_MOUNT=${ARA_DATASET_MOUNT}"

echo ""
echo "🖥️  Worker node ready"
echo "    Waiting for jobs from ${CATHEDRAL_ADDR}"
echo ""
echo "To run training manually:"
echo "  cd ${ARA_ROOT}"
echo "  python -m research.causal_swap  # Example"
echo ""
echo "To mount cathedral datasets (if NFS configured):"
echo "  sudo mount -t nfs ${ARA_DATASET_MOUNT} /mnt/ara_data"
