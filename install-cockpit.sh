#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
#  ARA COCKPIT - Complete Installation Suite
# ═══════════════════════════════════════════════════════════════════════════════
#
#  One-command installer for the Ara Cockpit HUD with all dependencies.
#
#  Usage:
#    ./install-cockpit.sh          # Full installation
#    ./install-cockpit.sh --quick  # Skip optional components
#
# ═══════════════════════════════════════════════════════════════════════════════

set -e

# Colors for output
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Banner
echo ""
echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}           ${BOLD}⚛  ARA COCKPIT INSTALLER  ⚛${NC}                           ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}           Premium Cyberpunk HUD Experience                       ${CYAN}║${NC}"
echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════════╝${NC}"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
QUICK_MODE=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --quick|-q)
            QUICK_MODE=true
            shift
            ;;
    esac
done

# Check for sudo
if ! command -v sudo &> /dev/null; then
    echo -e "${RED}✗ Error: sudo is required${NC}"
    exit 1
fi

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$NAME
    OS_VERSION=$VERSION_ID
else
    OS=$(uname -s)
    OS_VERSION="unknown"
fi
echo -e "${CYAN}►${NC} Detected: $OS $OS_VERSION"
echo ""

# ───────────────────────────────────────────────────────────────────────────────
# STEP 1: System Dependencies
# ───────────────────────────────────────────────────────────────────────────────
echo -e "${CYAN}[1/5]${NC} ${BOLD}Installing system dependencies...${NC}"

# Core packages
CORE_PACKAGES=(
    python3
    python3-pip
    python3-venv
    python3-gi
    python3-gi-cairo
    gir1.2-gtk-4.0
    gir1.2-adw-1
)

# GStreamer packages (for video background)
GSTREAMER_PACKAGES=(
    gir1.2-gst-plugins-base-1.0
    gir1.2-gstreamer-1.0
    gstreamer1.0-plugins-base
    gstreamer1.0-plugins-good
    gstreamer1.0-gtk4
)

# System monitoring packages
SYSTEM_PACKAGES=(
    python3-psutil
    python3-requests
    lm-sensors
    wmctrl
)

# Optional packages (best effort)
OPTIONAL_PACKAGES=(
    gir1.2-webkit-6.0
)

echo -e "${CYAN}  ├─${NC} Updating package lists..."
sudo apt-get update -qq 2>/dev/null || true

echo -e "${CYAN}  ├─${NC} Installing core packages..."
sudo apt-get install -y -qq "${CORE_PACKAGES[@]}" 2>/dev/null || {
    echo -e "${YELLOW}  │  Some core packages need manual installation${NC}"
}

echo -e "${CYAN}  ├─${NC} Installing GStreamer packages..."
sudo apt-get install -y -qq "${GSTREAMER_PACKAGES[@]}" 2>/dev/null || {
    echo -e "${YELLOW}  │  GStreamer packages optional - video background may not work${NC}"
}

echo -e "${CYAN}  ├─${NC} Installing system monitoring tools..."
sudo apt-get install -y -qq "${SYSTEM_PACKAGES[@]}" 2>/dev/null || true

if [ "$QUICK_MODE" = false ]; then
    echo -e "${CYAN}  ├─${NC} Installing optional packages..."
    for pkg in "${OPTIONAL_PACKAGES[@]}"; do
        sudo apt-get install -y -qq "$pkg" 2>/dev/null || true
    done
fi

echo -e "${GREEN}  └─ ✓ System dependencies installed${NC}"
echo ""

# ───────────────────────────────────────────────────────────────────────────────
# STEP 2: Python Virtual Environment
# ───────────────────────────────────────────────────────────────────────────────
echo -e "${CYAN}[2/5]${NC} ${BOLD}Setting up Python virtual environment...${NC}"

if [ ! -d "$VENV_DIR" ]; then
    echo -e "${CYAN}  ├─${NC} Creating virtual environment..."
    python3 -m venv "$VENV_DIR" --system-site-packages
fi

source "$VENV_DIR/bin/activate"

echo -e "${CYAN}  ├─${NC} Installing Python packages..."
pip install -q --upgrade pip 2>/dev/null || true
pip install -q psutil requests GPUtil 2>/dev/null || \
    pip install -q psutil requests 2>/dev/null || true

echo -e "${GREEN}  └─ ✓ Python environment ready${NC}"
echo ""

# ───────────────────────────────────────────────────────────────────────────────
# STEP 3: Create Launcher Scripts
# ───────────────────────────────────────────────────────────────────────────────
echo -e "${CYAN}[3/5]${NC} ${BOLD}Creating launcher scripts...${NC}"

# Main launcher
cat > "$SCRIPT_DIR/cockpit" << 'LAUNCHER'
#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
#  ARA COCKPIT - Quick Launcher
# ═══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
APP_DIR="$SCRIPT_DIR/gnome-tfan/app"

# Activate virtual environment if it exists
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
fi

# Check for display
if [ -z "$DISPLAY" ] && [ -z "$WAYLAND_DISPLAY" ]; then
    echo "⚠ No display detected. Run with: DISPLAY=:0 ./cockpit"
    exit 1
fi

# Banner
echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║           ⚛  ARA COCKPIT HUD  ⚛                              ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Parse arguments
FULLSCREEN=""
for arg in "$@"; do
    case $arg in
        --fullscreen|-f)
            FULLSCREEN="--fullscreen"
            ;;
        --help|-h)
            echo "Usage: ./cockpit [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --fullscreen, -f    Launch in fullscreen mode"
            echo "  --help, -h          Show this help message"
            echo ""
            echo "Controls:"
            echo "  F11                 Toggle fullscreen"
            echo "  Esc/Q               Exit"
            echo "  Tab                 Cycle topology views"
            echo ""
            exit 0
            ;;
    esac
done

# Auto-detect touchscreen monitor
if [ -z "$COCKPIT_MONITOR" ]; then
    COCKPIT_MONITOR=$(xrandr 2>/dev/null | grep " connected" | grep -E "800x1280|1080x1920|1200x1920" | awk '{print $1}' | head -1)
fi

if [ ! -z "$COCKPIT_MONITOR" ]; then
    echo "🖥  Monitor: $COCKPIT_MONITOR"
fi

echo "🚀 Launching Cockpit HUD..."
echo ""

cd "$APP_DIR"
python3 cockpit_hud.py $FULLSCREEN &
COCKPIT_PID=$!

# Wait briefly for window
sleep 1

# Try to position on touchscreen monitor
if [ ! -z "$COCKPIT_MONITOR" ] && [ "$COCKPIT_MONITOR" != "primary" ]; then
    if command -v wmctrl &> /dev/null; then
        WINDOW_ID=$(wmctrl -lp 2>/dev/null | grep $COCKPIT_PID | awk '{print $1}' | head -1)
        if [ ! -z "$WINDOW_ID" ]; then
            GEOMETRY=$(xrandr 2>/dev/null | grep "^$COCKPIT_MONITOR" | grep -o '[0-9]*x[0-9]*+[0-9]*+[0-9]*')
            if [ ! -z "$GEOMETRY" ]; then
                echo "📺 Moving to $COCKPIT_MONITOR"
                wmctrl -ir $WINDOW_ID -e 0,$GEOMETRY 2>/dev/null || true
            fi
        fi
    fi
fi

echo ""
echo "✓ Cockpit running (PID: $COCKPIT_PID)"
echo "  Press Ctrl+C to stop"
echo ""

wait $COCKPIT_PID 2>/dev/null
LAUNCHER
chmod +x "$SCRIPT_DIR/cockpit"

# Also update run-cockpit.sh for compatibility
cp "$SCRIPT_DIR/cockpit" "$SCRIPT_DIR/run-cockpit.sh"

echo -e "${GREEN}  └─ ✓ Launcher scripts created${NC}"
echo ""

# ───────────────────────────────────────────────────────────────────────────────
# STEP 4: Set Permissions
# ───────────────────────────────────────────────────────────────────────────────
echo -e "${CYAN}[4/5]${NC} ${BOLD}Setting permissions...${NC}"

chmod +x "$SCRIPT_DIR/gnome-tfan/app/cockpit_hud.py" 2>/dev/null || true
chmod +x "$SCRIPT_DIR/gnome-tfan/app/launch_cockpit.sh" 2>/dev/null || true
chmod +x "$SCRIPT_DIR/launch_spaceship.sh" 2>/dev/null || true

echo -e "${GREEN}  └─ ✓ Permissions set${NC}"
echo ""

# ───────────────────────────────────────────────────────────────────────────────
# STEP 5: Verify Installation
# ───────────────────────────────────────────────────────────────────────────────
echo -e "${CYAN}[5/5]${NC} ${BOLD}Verifying installation...${NC}"

ERRORS=0

# Check GTK4
if python3 -c "import gi; gi.require_version('Gtk', '4.0'); from gi.repository import Gtk" 2>/dev/null; then
    echo -e "${GREEN}  ├─ ✓ GTK4 available${NC}"
else
    echo -e "${RED}  ├─ ✗ GTK4 not working${NC}"
    ERRORS=$((ERRORS + 1))
fi

# Check Adwaita
if python3 -c "import gi; gi.require_version('Adw', '1'); from gi.repository import Adw" 2>/dev/null; then
    echo -e "${GREEN}  ├─ ✓ Libadwaita available${NC}"
else
    echo -e "${YELLOW}  ├─ ⚠ Libadwaita not available (optional)${NC}"
fi

# Check psutil
if python3 -c "import psutil" 2>/dev/null; then
    echo -e "${GREEN}  ├─ ✓ psutil available${NC}"
else
    echo -e "${YELLOW}  ├─ ⚠ psutil not available (demo mode will be used)${NC}"
fi

# Check GStreamer
if python3 -c "import gi; gi.require_version('Gst', '1.0'); from gi.repository import Gst" 2>/dev/null; then
    echo -e "${GREEN}  ├─ ✓ GStreamer available${NC}"
else
    echo -e "${YELLOW}  ├─ ⚠ GStreamer not available (animated background will be used)${NC}"
fi

# Check GPUtil
if python3 -c "import GPUtil" 2>/dev/null; then
    echo -e "${GREEN}  └─ ✓ GPUtil available (GPU monitoring enabled)${NC}"
else
    echo -e "${YELLOW}  └─ ⚠ GPUtil not available (demo GPU data will be used)${NC}"
fi

echo ""

# ───────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ───────────────────────────────────────────────────────────────────────────────
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "  ${GREEN}✓ Installation complete!${NC}"
    echo ""
    echo -e "  ${BOLD}To launch the Ara Cockpit:${NC}"
    echo -e "    ${CYAN}./cockpit${NC}"
    echo ""
    echo -e "  ${BOLD}Options:${NC}"
    echo -e "    ${CYAN}./cockpit --fullscreen${NC}    Fullscreen mode"
    echo -e "    ${CYAN}./cockpit --help${NC}          Show help"
    echo ""
    echo -e "  ${BOLD}Controls:${NC}"
    echo -e "    ${CYAN}F11${NC}  Toggle fullscreen"
    echo -e "    ${CYAN}Tab${NC}  Cycle topology views"
    echo -e "    ${CYAN}Esc${NC}  Exit"
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════════${NC}"
else
    echo -e "${RED}═══════════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "  ${RED}✗ Installation completed with errors${NC}"
    echo ""
    echo "  Please install missing dependencies manually:"
    echo "    sudo apt install python3-gi python3-gi-cairo gir1.2-gtk-4.0 gir1.2-adw-1"
    echo ""
    echo -e "${RED}═══════════════════════════════════════════════════════════════════${NC}"
    exit 1
fi
echo ""
