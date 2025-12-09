#!/bin/bash
#
# Example: Using the width detector on A10PED bitstreams
#
# This script demonstrates the complete workflow for algorithmically
# determining optimal frame width for bitstream analysis.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WIDTH_DETECTOR="$SCRIPT_DIR/width_detector.py"

echo "============================================================"
echo "Width Detector Example Workflow"
echo "============================================================"
echo

# Example 1: Detect width from SOF file
echo "Example 1: Analyzing .sof bitstream"
echo "------------------------------------------------------------"

if [ -f "$1" ]; then
    BITSTREAM="$1"
    echo "Using provided bitstream: $BITSTREAM"
    echo

    # Basic detection
    echo "Running width detection..."
    python3 "$WIDTH_DETECTOR" "$BITSTREAM" --top-k 3
    echo

    # With visualization
    echo "Generating visualization of top candidate..."
    python3 "$WIDTH_DETECTOR" "$BITSTREAM" --visualize
    echo

    # JSON output for automation
    echo "Exporting results as JSON..."
    python3 "$WIDTH_DETECTOR" "$BITSTREAM" --json > width_results.json
    echo "Results saved to: width_results.json"
    echo

    # Extract recommended width for downstream processing
    RECOMMENDED_WIDTH=$(python3 -c "import json; print(json.load(open('width_results.json'))['candidates'][0]['width'])")
    echo "Recommended width: $RECOMMENDED_WIDTH bytes"
    echo

else
    echo "No bitstream provided. Showing usage examples."
    echo
    echo "Usage:"
    echo "  $0 <path_to_bitstream.sof>"
    echo
    echo "Or call width_detector.py directly:"
    echo "  python3 $WIDTH_DETECTOR bitstream.sof"
    echo "  python3 $WIDTH_DETECTOR bitstream.sof --top-k 5"
    echo "  python3 $WIDTH_DETECTOR bitstream.sof --visualize"
    echo "  python3 $WIDTH_DETECTOR bitstream.sof --candidates 512 1024 2048"
    echo "  python3 $WIDTH_DETECTOR bitstream.sof --json"
    echo
fi

echo "============================================================"
echo "Integration with Feature Extraction"
echo "============================================================"
echo
echo "Once width is determined, use it for feature extraction:"
echo
cat << 'EOF'
import numpy as np
from width_detector import WidthDetector

# Detect width
detector = WidthDetector('design.sof')
detector.load_bitstream()
results = detector.detect_width()
width, score = results[0]

# Reshape for feature extraction
data = detector.config_data
num_rows = len(data) // width
matrix = np.frombuffer(data[:num_rows*width], dtype=np.uint8).reshape(num_rows, width)

# Now extract features (CNN input, etc.)
print(f"Matrix shape: {matrix.shape}")
print(f"Ready for ML feature extraction")
EOF

echo
echo "============================================================"
echo "Integration with HNTF Build System"
echo "============================================================"
echo
echo "Add to flows/quartus/a10ped/build_tile.sh:"
echo
cat << 'EOF'
# After Quartus compilation, validate bitstream geometry
echo "Validating bitstream geometry..."
DETECTED_WIDTH=$(python3 tools/bitstream_analysis/width_detector.py \
    output_files/project.sof --json | \
    python3 -c "import sys,json; print(json.load(sys.stdin)['candidates'][0]['width'])")

# Compare with expected width from board spec
EXPECTED_WIDTH=1024  # From a10ped_board.yaml analysis

if [ "$DETECTED_WIDTH" != "$EXPECTED_WIDTH" ]; then
    echo "Warning: Detected width $DETECTED_WIDTH != expected $EXPECTED_WIDTH"
    echo "This may indicate:"
    echo "  - Bitstream compression enabled (check Quartus settings)"
    echo "  - Unexpected device configuration"
    echo "  - Need to update board YAML"
fi
EOF

echo
echo "Done!"
