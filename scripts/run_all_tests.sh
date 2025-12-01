#!/bin/bash
#
# Comprehensive Test Runner
#
# Runs all SNN tests, benchmarks, and gate validations.
# This script is used in CI and for local validation.
#
# Usage:
#   ./scripts/run_all_tests.sh              # Run all tests
#   ./scripts/run_all_tests.sh --quick      # Quick smoke tests only
#   ./scripts/run_all_tests.sh --gates-only # Only gate validation
#

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "========================================"
echo "COMPREHENSIVE SNN TEST SUITE"
echo "========================================"
echo ""

# Parse arguments
QUICK_MODE=false
GATES_ONLY=false

for arg in "$@"; do
    case $arg in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --gates-only)
            GATES_ONLY=true
            shift
            ;;
    esac
done

# Create artifacts directory
mkdir -p artifacts

# ============================================
# SECTION 1: Unit Tests
# ============================================

if [ "$GATES_ONLY" = false ]; then
    echo "========================================"
    echo "SECTION 1: SNN Unit Tests"
    echo "========================================"
    echo ""

    echo "[1/5] Parameter audit tests..."
    python -m pytest tests/snn/test_param_audit.py -v --tb=short || exit 1

    echo ""
    echo "[2/5] Forward correctness tests..."
    python -m pytest tests/snn/test_forward_correctness.py -v --tb=short || exit 1

    echo ""
    echo "[3/5] Event queue tests..."
    python -m pytest tests/snn/test_event_queue.py -v --tb=short || exit 1

    echo ""
    echo "[4/5] Gradient stability tests..."
    python -m pytest tests/snn/test_grad_stability.py -v --tb=short || exit 1

    echo ""
    echo "[5/5] TLS mask ablation tests..."
    python -m pytest tests/snn/test_ablate_tls.py -v --tb=short || exit 1

    echo ""
    echo "✓ All unit tests passed!"
    echo ""
fi

# ============================================
# SECTION 2: Benchmarks
# ============================================

if [ "$GATES_ONLY" = false ]; then
    echo "========================================"
    echo "SECTION 2: Benchmarks"
    echo "========================================"
    echo ""

    echo "[1/3] SNN parameter audit..."
    python scripts/bench_snn.py --audit --emit-json artifacts/snn_audit.json || exit 1

    if [ "$QUICK_MODE" = false ]; then
        echo ""
        echo "[2/3] Roofline analysis..."
        python scripts/bench_snn.py --roofline --output-dir artifacts/ || exit 1

        echo ""
        echo "[3/3] Accuracy/energy comparison..."
        python scripts/bench_accuracy_energy.py --quick --output artifacts/comparison.json || exit 1
    else
        echo ""
        echo "⚠ Skipping roofline and accuracy benchmarks (quick mode)"
    fi

    echo ""
    echo "✓ Benchmarks completed!"
    echo ""
fi

# ============================================
# SECTION 3: Gate Validation
# ============================================

echo "========================================"
echo "SECTION 3: Gate Validation"
echo "========================================"
echo ""

# Check if ci_quick config exists
if [ -f "configs/ci/ci_quick.yaml" ]; then
    CONFIG_PATH="configs/ci/ci_quick.yaml"
elif [ -f "configs/snn_emu_4096.yaml" ]; then
    CONFIG_PATH="configs/snn_emu_4096.yaml"
else
    echo "⚠ No config found, creating minimal config..."
    mkdir -p configs/ci
    cat > configs/ci/ci_quick.yaml <<EOF
backend: snn_emu

model:
  N: 2048
  lowrank_rank: 16
  k_per_row: 32

snn:
  v_th: 1.0
  alpha: 0.95
  time_steps: 128

training:
  learning_rate: 1.5e-3
  grad_clip: 1.0
  batch_size: 2

tfan:
  use_fdt: true
  fdt:
    kp: 0.30
    ki: 0.02
    target_epr_cv: 0.15

device: cpu
dtype: float32
EOF
    CONFIG_PATH="configs/ci/ci_quick.yaml"
fi

echo "Using config: $CONFIG_PATH"
echo ""

# Validate SNN gates
echo "[1/2] SNN gate validation..."
python -c "
import json
with open('artifacts/snn_audit.json') as f:
    audit = json.load(f)

gates = {
    'param_reduction_pct >= 97.0': audit['param_reduction_pct'] >= 97.0,
    'avg_degree <= 0.02*N': audit['avg_degree'] <= 0.02 * audit['N'],
    'rank <= 0.02*N': audit['rank'] <= 0.02 * audit['N'],
    'sparsity >= 0.98': audit['sparsity'] >= 0.98,
}

print('\\nSNN Gate Validation:')
for gate, passed in gates.items():
    status = '✓ PASS' if passed else '✗ FAIL'
    print(f'  {status}: {gate}')

if not all(gates.values()):
    print('\\n❌ SNN gates failed!')
    exit(1)
else:
    print('\\n✅ All SNN gates passed!')
" || exit 1

if [ "$QUICK_MODE" = false ] && [ "$GATES_ONLY" = false ]; then
    echo ""
    echo "[2/2] FDT gate validation..."
    python scripts/validate_all_gates.py --config "$CONFIG_PATH" --fdt-steps 500 --output artifacts/gate_validation.json || exit 1
else
    echo ""
    echo "⚠ Skipping FDT gate validation (quick mode)"
fi

echo ""
echo "✓ All gates validated!"
echo ""

# ============================================
# SECTION 4: Summary
# ============================================

echo "========================================"
echo "FINAL SUMMARY"
echo "========================================"
echo ""

if [ -f "artifacts/snn_audit.json" ]; then
    python -c "
import json

with open('artifacts/snn_audit.json') as f:
    audit = json.load(f)

print('SNN Configuration:')
print(f\"  N: {audit['N']:,}\")
print(f\"  Rank: {audit['rank']}\")
print(f\"  Avg degree: {audit['avg_degree']:.1f}\")
print(f\"  Parameter reduction: {audit['param_reduction_pct']:.2f}%\")
print(f\"  Sparsity: {audit['sparsity']:.2%}\")
print(f\"  Parameters: {audit['lowrank_params']:,} vs {audit['dense_params']:,} (dense)\")
"
fi

echo ""

if [ "$GATES_ONLY" = false ]; then
    echo "✅ All tests passed!"
else
    echo "✅ All gates validated!"
fi

echo ""
echo "Artifacts saved to: artifacts/"
echo ""
echo "========================================"
