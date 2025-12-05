#!/bin/bash
#
# Ara Sandbox Test Runner
# =======================
#
# Runs tests in an isolated sandbox environment.
# Used by the orchestrator to validate patches before deployment.
#
# Usage:
#   ./sandbox_run_tests.sh [suite]
#
# Suites:
#   unit        - Fast unit tests only
#   integration - Integration tests
#   tts         - TTS-specific tests
#   fpga        - FPGA simulation tests
#   full        - All tests
#

set -e

SUITE="${1:-unit}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/../.."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in sandbox mode (orchestrator sets this)
if [[ -n "${ARA_SANDBOX}" ]]; then
    log_info "Running in sandbox mode"
    REPO_ROOT="${ARA_SANDBOX}"
fi

cd "${REPO_ROOT}"

log_info "Test suite: ${SUITE}"
log_info "Repository: ${REPO_ROOT}"

# Track test results
TESTS_PASSED=0
TESTS_FAILED=0

run_python_tests() {
    local pattern="$1"
    local name="$2"

    log_info "Running ${name}..."

    if command -v pytest &> /dev/null; then
        if pytest -v --tb=short ${pattern} 2>&1; then
            log_info "${name} PASSED"
            ((TESTS_PASSED++))
        else
            log_error "${name} FAILED"
            ((TESTS_FAILED++))
        fi
    elif command -v python3 &> /dev/null; then
        if python3 -m pytest -v --tb=short ${pattern} 2>&1; then
            log_info "${name} PASSED"
            ((TESTS_PASSED++))
        else
            log_error "${name} FAILED"
            ((TESTS_FAILED++))
        fi
    else
        log_warn "pytest not available, skipping ${name}"
    fi
}

run_shell_tests() {
    local script="$1"
    local name="$2"

    if [[ -x "${script}" ]]; then
        log_info "Running ${name}..."
        if "${script}" 2>&1; then
            log_info "${name} PASSED"
            ((TESTS_PASSED++))
        else
            log_error "${name} FAILED"
            ((TESTS_FAILED++))
        fi
    else
        log_warn "${name} not found or not executable"
    fi
}

run_verilog_lint() {
    log_info "Running Verilog lint..."

    if command -v verilator &> /dev/null; then
        local rtl_files=(banos/fpga/rtl/*.sv)
        if [[ ${#rtl_files[@]} -gt 0 ]]; then
            if verilator --lint-only -Wall "${rtl_files[@]}" 2>&1; then
                log_info "Verilog lint PASSED"
                ((TESTS_PASSED++))
            else
                log_error "Verilog lint FAILED"
                ((TESTS_FAILED++))
            fi
        else
            log_warn "No Verilog files found"
        fi
    else
        log_warn "verilator not available, skipping lint"
    fi
}

check_syntax() {
    log_info "Checking Python syntax..."

    local errors=0
    while IFS= read -r -d '' pyfile; do
        if ! python3 -m py_compile "${pyfile}" 2>&1; then
            log_error "Syntax error in ${pyfile}"
            ((errors++))
        fi
    done < <(find . -name "*.py" -not -path "./.git/*" -print0)

    if [[ ${errors} -eq 0 ]]; then
        log_info "Python syntax check PASSED"
        ((TESTS_PASSED++))
    else
        log_error "Python syntax check FAILED (${errors} errors)"
        ((TESTS_FAILED++))
    fi
}

# =============================================================================
# TEST SUITES
# =============================================================================

case "${SUITE}" in
    unit)
        log_info "=== Unit Test Suite ==="
        check_syntax
        run_python_tests "tests/unit" "Unit tests"
        ;;

    integration)
        log_info "=== Integration Test Suite ==="
        check_syntax
        run_python_tests "tests/integration" "Integration tests"
        ;;

    tts)
        log_info "=== TTS Test Suite ==="
        check_syntax
        run_python_tests "tests/**/test_tts*.py" "TTS tests"
        ;;

    fpga)
        log_info "=== FPGA Test Suite ==="
        run_verilog_lint
        if [[ -d "banos/fpga/sim" ]]; then
            for tb in banos/fpga/sim/tb_*.sv; do
                if [[ -f "${tb}" ]]; then
                    log_info "Found testbench: ${tb}"
                    # Run with iverilog if available
                    if command -v iverilog &> /dev/null; then
                        basename_tb=$(basename "${tb}" .sv)
                        if iverilog -g2012 -o "/tmp/${basename_tb}" \
                           banos/fpga/rtl/*.sv "${tb}" 2>&1; then
                            if vvp "/tmp/${basename_tb}" 2>&1 | head -100; then
                                log_info "${basename_tb} PASSED"
                                ((TESTS_PASSED++))
                            else
                                log_error "${basename_tb} FAILED"
                                ((TESTS_FAILED++))
                            fi
                        else
                            log_error "${basename_tb} compilation FAILED"
                            ((TESTS_FAILED++))
                        fi
                    fi
                fi
            done
        fi
        ;;

    full)
        log_info "=== Full Test Suite ==="
        check_syntax
        run_python_tests "tests/" "All Python tests"
        run_verilog_lint
        ;;

    *)
        log_error "Unknown test suite: ${SUITE}"
        echo "Available suites: unit, integration, tts, fpga, full"
        exit 1
        ;;
esac

# =============================================================================
# SUMMARY
# =============================================================================

echo ""
echo "========================================"
echo "  TEST SUMMARY"
echo "========================================"
echo -e "  Passed: ${GREEN}${TESTS_PASSED}${NC}"
echo -e "  Failed: ${RED}${TESTS_FAILED}${NC}"
echo "========================================"

if [[ ${TESTS_FAILED} -gt 0 ]]; then
    log_error "Some tests failed!"
    exit 1
else
    log_info "All tests passed!"
    exit 0
fi
