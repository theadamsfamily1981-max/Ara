#!/bin/bash
# =============================================================================
# TRAIN ON WORKER NODE
# =============================================================================
# Launch a training job on the V100 worker via SSH
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration
WORKER_HOST="${WORKER_HOST:-ara-worker-v100}"
WORKER_USER="${WORKER_USER:-ara}"
ARA_REMOTE_PATH="${ARA_REMOTE_PATH:-~/ara}"

# Parse arguments
TRAINING_SCRIPT=""
RUN_ID="run_$(date +%Y%m%d_%H%M%S)"
EXTRA_ARGS=""

print_usage() {
    echo "Usage: $0 <training_script> [--run-id ID] [-- extra_args...]"
    echo ""
    echo "Examples:"
    echo "  $0 research.causal_swap --run-id my_run"
    echo "  $0 research.rl_adaptation -- --epochs 100 --batch-size 32"
    echo ""
    echo "Environment:"
    echo "  WORKER_HOST   Worker hostname (default: ara-worker-v100)"
    echo "  WORKER_USER   SSH user (default: ara)"
    echo "  ARA_REMOTE_PATH   Ara path on worker (default: ~/ara)"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --run-id)
            RUN_ID="$2"
            shift 2
            ;;
        --help|-h)
            print_usage
            exit 0
            ;;
        --)
            shift
            EXTRA_ARGS="$*"
            break
            ;;
        *)
            if [ -z "$TRAINING_SCRIPT" ]; then
                TRAINING_SCRIPT="$1"
            else
                EXTRA_ARGS="$EXTRA_ARGS $1"
            fi
            shift
            ;;
    esac
done

if [ -z "$TRAINING_SCRIPT" ]; then
    echo "❌ Training script required"
    print_usage
    exit 1
fi

echo "🚀 Launching training job on ${WORKER_HOST}"
echo "   Script: ${TRAINING_SCRIPT}"
echo "   Run ID: ${RUN_ID}"
if [ -n "$EXTRA_ARGS" ]; then
    echo "   Args: ${EXTRA_ARGS}"
fi
echo ""

# Build the remote command
REMOTE_CMD="cd ${ARA_REMOTE_PATH} && \
    export PYTHONPATH=\${PWD}:\${PYTHONPATH} && \
    export ARA_RUN_ID=${RUN_ID} && \
    python -m ${TRAINING_SCRIPT} ${EXTRA_ARGS}"

echo "Executing on ${WORKER_USER}@${WORKER_HOST}:"
echo "  ${REMOTE_CMD}"
echo ""

# Execute
ssh "${WORKER_USER}@${WORKER_HOST}" "${REMOTE_CMD}"

echo ""
echo "✅ Training job completed"
