#!/usr/bin/env bash
#
# T-FAN Topology Screensaver Installer
# Installs system dependencies and Python packages
#

set -e

echo "🎨 Installing T-FAN Topology Screensaver..."

# Detect OS
if [[ -f /etc/os-release ]]; then
    . /etc/os-release
    OS=$ID
else
    OS=$(uname -s)
fi

# Install system dependencies
echo "📦 Installing system dependencies..."
case "$OS" in
    ubuntu|debian)
        sudo apt-get update
        sudo apt-get install -y \
            python3-dev \
            python3-pip \
            libgl1-mesa-glx \
            libegl1 \
            libglib2.0-0 \
            libxcb-icccm4 \
            libxcb-image0 \
            libxcb-keysyms1 \
            libxcb-randr0 \
            libxcb-render-util0 \
            libxcb-xinerama0 \
            libxkbcommon-x11-0
        ;;
    fedora|rhel|centos)
        sudo dnf install -y \
            python3-devel \
            python3-pip \
            mesa-libGL \
            mesa-libEGL
        ;;
    arch|manjaro)
        sudo pacman -Syu --noconfirm \
            python \
            python-pip \
            mesa \
            libglvnd
        ;;
    *)
        echo "⚠️  Unknown OS: $OS"
        echo "   Please install OpenGL and EGL libraries manually"
        ;;
esac

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install -r requirements-viz.txt

# Make screensaver executable
chmod +x topo_screensaver.py
chmod +x metrics_bridge.py

echo ""
echo "✅ Installation complete!"
echo ""
echo "Quick Start:"
echo "  1. Start metrics bridge:  python metrics_bridge.py"
echo "  2. Launch screensaver:    python topo_screensaver.py --fullscreen"
echo "  3. Press M to cycle modes, P to pause, Q to quit"
echo ""
echo "For xscreensaver integration, see README.md"
