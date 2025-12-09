#!/bin/bash
#
# FPGA Salvage GUI - Easy Setup Script
# =====================================
#
# This script installs all dependencies and launches the web GUI.
#
# Usage: sudo ./setup_gui.sh

set -e

echo "================================================================"
echo "FPGA Salvage Tool - Web GUI Setup"
echo "================================================================"
echo

# Check for root
if [ "$EUID" -ne 0 ]; then
    echo "ERROR: This script must be run as root (for I2C and JTAG access)"
    echo
    echo "Please run:"
    echo "  sudo ./setup_gui.sh"
    exit 1
fi

# Check Python version
echo "[1/4] Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found. Please install:"
    echo "  sudo apt install python3 python3-pip"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}' | cut -d. -f1,2)
echo "  ✓ Python $PYTHON_VERSION found"

# Install system dependencies
echo
echo "[2/4] Installing system dependencies..."
apt-get update -qq
apt-get install -y \
    python3-pip \
    openocd \
    i2c-tools \
    usbutils \
    > /dev/null 2>&1

echo "  ✓ System dependencies installed"

# Install Python dependencies
echo
echo "[3/4] Installing Python dependencies..."
pip3 install -q -r requirements.txt
echo "  ✓ Python dependencies installed"

# Check for JTAG adapter
echo
echo "[4/4] Checking for JTAG adapter..."
if lsusb | grep -i "ftdi\|blaster\|xilinx" > /dev/null; then
    echo "  ✓ JTAG adapter detected"
else
    echo "  ⚠ No JTAG adapter detected (connect one before using)"
fi

# Get IP address
IP_ADDR=$(hostname -I | awk '{print $1}')

echo
echo "================================================================"
echo "Setup Complete!"
echo "================================================================"
echo
echo "Starting Web GUI..."
echo
echo "Access the GUI at:"
echo "  Local:   http://localhost:5000"
echo "  Network: http://$IP_ADDR:5000"
echo
echo "Press Ctrl+C to stop the server"
echo "================================================================"
echo

# Launch GUI
cd "$(dirname "$0")"
python3 fpga_salvage_gui.py
