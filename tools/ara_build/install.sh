#!/bin/bash
#
# Install Ara Build Tools
#
# This script:
# 1. Creates ~/.ara directory structure
# 2. Creates symlinks for ara-build and ara-doctor
# 3. Optionally adds to PATH
#
# Usage:
#   ./install.sh           # Install to ~/.local/bin
#   ./install.sh /usr/local/bin   # Install to custom location
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="${1:-$HOME/.local/bin}"
ARA_DIR="$HOME/.ara"

echo "=== Ara Build Tools Installer ==="
echo ""
echo "Script dir: $SCRIPT_DIR"
echo "Install to: $INSTALL_DIR"
echo ""

# Create directories
echo "[1/4] Creating directories..."
mkdir -p "$ARA_DIR/build_logs"
mkdir -p "$INSTALL_DIR"

# Make scripts executable
echo "[2/4] Making scripts executable..."
chmod +x "$SCRIPT_DIR/ara_build.py"
chmod +x "$SCRIPT_DIR/ara_doctor.py"

# Create symlinks
echo "[3/4] Creating symlinks..."

if [ -L "$INSTALL_DIR/ara-build" ] || [ -f "$INSTALL_DIR/ara-build" ]; then
    rm "$INSTALL_DIR/ara-build"
fi
ln -s "$SCRIPT_DIR/ara_build.py" "$INSTALL_DIR/ara-build"
echo "  ✓ ara-build -> $SCRIPT_DIR/ara_build.py"

if [ -L "$INSTALL_DIR/ara-doctor" ] || [ -f "$INSTALL_DIR/ara-doctor" ]; then
    rm "$INSTALL_DIR/ara-doctor"
fi
ln -s "$SCRIPT_DIR/ara_doctor.py" "$INSTALL_DIR/ara-doctor"
echo "  ✓ ara-doctor -> $SCRIPT_DIR/ara_doctor.py"

# Check PATH
echo "[4/4] Checking PATH..."
if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
    echo ""
    echo "⚠ $INSTALL_DIR is not in your PATH."
    echo ""
    echo "Add this to your ~/.bashrc or ~/.zshrc:"
    echo ""
    echo "    export PATH=\"$INSTALL_DIR:\$PATH\""
    echo ""
else
    echo "  ✓ $INSTALL_DIR is in PATH"
fi

echo ""
echo "=== Installation Complete ==="
echo ""
echo "Commands available:"
echo "  ara-build <cmd>    # Wrap build commands with failure tracking"
echo "  ara-doctor         # Scan system health"
echo ""
echo "Examples:"
echo "  ara-build pip install pycairo"
echo "  ara-build meson setup build"
echo "  ara-doctor --fix"
echo ""

# Quick test
if command -v ara-doctor &> /dev/null; then
    echo "Running quick health check..."
    echo ""
    ara-doctor --quick 2>/dev/null || true
fi
