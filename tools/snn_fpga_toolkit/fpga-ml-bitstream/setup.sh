#!/bin/bash
# setup.sh - Quick setup script for fpga-ml-bitstream

set -e

echo "========================================="
echo "FPGA ML Bitstream Analysis - Setup"
echo "========================================="
echo

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Found Python $python_version"

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install -U pip --quiet

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt --quiet

echo
echo "========================================="
echo "Setup complete!"
echo "========================================="
echo
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo
echo "To get started:"
echo "  1. Place .sof files in dataset/raw_sof/"
echo "  2. Run: python preprocess/sof_to_bits.py --input dataset/raw_sof/example.sof --output dataset/parsed_npz/example_bits.npz"
echo "  3. Run: python preprocess/guess_width_autocorr.py --input dataset/parsed_npz/example_bits.npz --output dataset/images/example.png"
echo "  4. Explore notebooks/ for experiments"
echo
