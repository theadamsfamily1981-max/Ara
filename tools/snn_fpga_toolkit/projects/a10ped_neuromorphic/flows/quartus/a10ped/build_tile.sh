#!/usr/bin/env bash
# Batch build script for A10PED tile
# Part of the YAML-driven FPGA build system
#
# This script orchestrates the complete build flow:
# 1. Setup Quartus project from YAML specs
# 2. Run full Quartus compile
# 3. Parse timing and utilization reports
# 4. Generate JSON summaries
#
# Usage:
#   cd /path/to/a10ped_neuromorphic/flows/quartus/a10ped
#   ./build_tile.sh

set -euo pipefail

# Determine project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OUT_DIR="$ROOT_DIR/out/a10ped/build"
PROJ_NAME="a10ped_tile0"

echo "========================================"
echo "  A10PED Neuromorphic Tile Build"
echo "========================================"
echo "Root:   $ROOT_DIR"
echo "Output: $OUT_DIR"
echo "Project: $PROJ_NAME"
echo ""

# Create output directory
mkdir -p "$OUT_DIR"

# Step 1: Setup project
echo "[1/4] Setting up Quartus project..."
cd "$OUT_DIR"
if ! quartus_sh -t "$ROOT_DIR/flows/quartus/a10ped/project.tcl"; then
    echo "❌ Project setup failed"
    exit 1
fi
echo "✅ Project setup complete"
echo ""

# Step 2: Run full compile flow
echo "[2/4] Running Quartus compile (this will take a while)..."
if ! quartus_sh --flow compile "$PROJ_NAME"; then
    echo "❌ Compilation failed"
    exit 1
fi
echo "✅ Compilation complete"
echo ""

# Step 3: Parse timing report
echo "[3/4] Parsing timing report..."
TIMING_RPT="$OUT_DIR/output_files/${PROJ_NAME}.sta.rpt"
TIMING_JSON="$OUT_DIR/timing_summary.json"

if [[ -f "$TIMING_RPT" ]]; then
    if python3 "$ROOT_DIR/tools/parse_reports/parse_quartus_timing.py" \
        "$TIMING_RPT" > "$TIMING_JSON"; then
        echo "✅ Timing summary: $TIMING_JSON"
        cat "$TIMING_JSON"
    else
        echo "⚠️  Warning: Could not parse timing report"
    fi
else
    echo "⚠️  Warning: Timing report not found at $TIMING_RPT"
fi
echo ""

# Step 4: Parse utilization report
echo "[4/4] Parsing utilization report..."
UTIL_RPT="$OUT_DIR/output_files/${PROJ_NAME}.fit.summary"
UTIL_JSON="$OUT_DIR/utilization_summary.json"

if [[ -f "$UTIL_RPT" ]]; then
    if python3 "$ROOT_DIR/tools/parse_reports/parse_quartus_util.py" \
        "$UTIL_RPT" > "$UTIL_JSON"; then
        echo "✅ Utilization summary: $UTIL_JSON"
        cat "$UTIL_JSON"
    else
        echo "⚠️  Warning: Could not parse utilization report"
    fi
else
    echo "⚠️  Warning: Utilization report not found at $UTIL_RPT"
fi
echo ""

# Summary
echo "========================================"
echo "  Build Complete!"
echo "========================================"
echo "Bitstream:    $OUT_DIR/output_files/${PROJ_NAME}.sof"
echo "RBF:          $OUT_DIR/output_files/${PROJ_NAME}.rbf"
echo "Timing:       $TIMING_JSON"
echo "Utilization:  $UTIL_JSON"
echo ""
echo "To program the FPGA:"
echo "  quartus_pgm -c 1 -m jtag -o \"p;$OUT_DIR/output_files/${PROJ_NAME}.sof@1\""
