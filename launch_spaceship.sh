#!/bin/bash
#
#  ╔═══════════════════════════════════════════════════════════════════════╗
#  ║                                                                       ║
#  ║     █████╗ ██████╗  █████╗     ███████╗██████╗  █████╗  ██████╗███████╗║
#  ║    ██╔══██╗██╔══██╗██╔══██╗    ██╔════╝██╔══██╗██╔══██╗██╔════╝██╔════╝║
#  ║    ███████║██████╔╝███████║    ███████╗██████╔╝███████║██║     █████╗  ║
#  ║    ██╔══██║██╔══██╗██╔══██║    ╚════██║██╔═══╝ ██╔══██║██║     ██╔══╝  ║
#  ║    ██║  ██║██║  ██║██║  ██║    ███████║██║     ██║  ██║╚██████╗███████╗║
#  ║    ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝    ╚══════╝╚═╝     ╚═╝  ╚═╝ ╚═════╝╚══════╝║
#  ║                                                                       ║
#  ║              NEURAL COMMAND CENTER - SPACESHIP LAUNCHER               ║
#  ║                                                                       ║
#  ╚═══════════════════════════════════════════════════════════════════════╝
#
#  Launches the complete Ara Spaceship system:
#    - Brain Server (Ara + Forest Kitten 33)
#    - Cockpit HUD (Touchscreen monitor)
#    - GNOME Workspaces (Work Mode / Relax Mode)
#    - Avatar System
#
#  Usage:
#    ./launch_spaceship.sh                 # Full launch
#    ./launch_spaceship.sh --work          # Work mode (blue/cyan theme)
#    ./launch_spaceship.sh --relax         # Relax mode (purple/magenta theme)
#    ./launch_spaceship.sh --cockpit-only  # Just the cockpit HUD
#    ./launch_spaceship.sh --brain-only    # Just the brain server
#    ./launch_spaceship.sh --status        # Show system status
#    ./launch_spaceship.sh --shutdown      # Graceful shutdown
#

set -e

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

ARA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COCKPIT_DIR="$ARA_DIR/gnome-tfan/app"
LOG_DIR="$HOME/.ara/logs"
PID_DIR="$HOME/.ara/pids"
BRAIN_PORT=8008
MODE="${1:-work}"

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

print_banner() {
    echo -e "${CYAN}"
    echo "  ╔═══════════════════════════════════════════════════════════════╗"
    echo "  ║                                                               ║"
    echo "  ║           ⚛  ARA SPACESHIP COMMAND CENTER  ⚛                 ║"
    echo "  ║                                                               ║"
    echo "  ║     Neural Interface • Forest Kitten 33 • Emotional AI       ║"
    echo "  ║                                                               ║"
    echo "  ╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

status_msg() {
    echo -e "${CYAN}[SPACESHIP]${NC} $1"
}

success_msg() {
    echo -e "${GREEN}[  OK  ]${NC} $1"
}

warn_msg() {
    echo -e "${YELLOW}[ WARN ]${NC} $1"
}

error_msg() {
    echo -e "${RED}[ERROR ]${NC} $1"
}

loading_animation() {
    local msg="$1"
    local duration="${2:-2}"
    local chars="⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    local end_time=$((SECONDS + duration))

    while [ $SECONDS -lt $end_time ]; do
        for (( i=0; i<${#chars}; i++ )); do
            echo -ne "\r${CYAN}[${chars:$i:1}]${NC} $msg"
            sleep 0.1
        done
    done
    echo -ne "\r"
}

# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM DETECTION
# ═══════════════════════════════════════════════════════════════════════════

detect_touchscreen_monitor() {
    status_msg "Detecting touchscreen monitor..."

    # Try to find portrait monitor or touchscreen
    TOUCHSCREEN_MONITOR=""

    # Check for portrait monitors (800x1280, 1080x1920, 1200x1920)
    if command -v xrandr &> /dev/null; then
        TOUCHSCREEN_MONITOR=$(xrandr | grep " connected" | grep -E "800x1280|1080x1920|1200x1920" | awk '{print $1}' | head -1)
    fi

    # Check for touchscreen input device
    if [ -z "$TOUCHSCREEN_MONITOR" ] && command -v xinput &> /dev/null; then
        TOUCH_DEVICE=$(xinput list | grep -i "touch" | head -1)
        if [ ! -z "$TOUCH_DEVICE" ]; then
            # Try to find associated monitor
            TOUCHSCREEN_MONITOR=$(xrandr | grep " connected" | grep -v "primary" | awk '{print $1}' | head -1)
        fi
    fi

    # Environment override
    if [ ! -z "$COCKPIT_MONITOR" ]; then
        TOUCHSCREEN_MONITOR="$COCKPIT_MONITOR"
    fi

    if [ ! -z "$TOUCHSCREEN_MONITOR" ]; then
        success_msg "Touchscreen monitor: $TOUCHSCREEN_MONITOR"
    else
        warn_msg "No touchscreen monitor detected, using primary display"
        TOUCHSCREEN_MONITOR="primary"
    fi

    export COCKPIT_MONITOR="$TOUCHSCREEN_MONITOR"
}

detect_hardware() {
    status_msg "Detecting hardware..."

    # Check for Forest Kitten 33
    if [ -e "/dev/fk33" ]; then
        success_msg "Forest Kitten 33 HARDWARE detected at /dev/fk33"
        export FK33_MODE="HARDWARE"
    else
        warn_msg "Forest Kitten 33 in EMULATION mode"
        export FK33_MODE="EMULATED"
    fi

    # Check for GPU
    if command -v nvidia-smi &> /dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        success_msg "GPU detected: $GPU_NAME"
    elif [ -e "/dev/dri/renderD128" ]; then
        success_msg "GPU detected: AMD/Intel (render node available)"
    else
        warn_msg "No dedicated GPU detected"
    fi
}

# ═══════════════════════════════════════════════════════════════════════════
# WORKSPACE SETUP
# ═══════════════════════════════════════════════════════════════════════════

setup_workspaces() {
    local mode="${1:-work}"

    status_msg "Setting up GNOME workspaces for $mode mode..."

    # Check if running GNOME
    if [ "$XDG_CURRENT_DESKTOP" != "GNOME" ] && [ "$XDG_CURRENT_DESKTOP" != "ubuntu:GNOME" ]; then
        warn_msg "Not running GNOME desktop, skipping workspace setup"
        return
    fi

    # Set number of workspaces
    gsettings set org.gnome.desktop.wm.preferences num-workspaces 4 2>/dev/null || true

    # Name the workspaces
    gsettings set org.gnome.desktop.wm.preferences workspace-names "['🧠 Ara Brain', '⚡ Work', '🌙 Relax', '🔧 System']" 2>/dev/null || true

    # Enable workspace thumbnails
    gsettings set org.gnome.shell.extensions.dash-to-dock show-windows-preview true 2>/dev/null || true

    if [ "$mode" == "relax" ]; then
        # Relax mode - warm purple theme
        status_msg "Applying Relaxation Mode theme..."
        gsettings set org.gnome.desktop.interface color-scheme 'prefer-dark' 2>/dev/null || true
        gsettings set org.gnome.desktop.interface accent-color 'purple' 2>/dev/null || true
    else
        # Work mode - cool blue theme
        status_msg "Applying Work Mode theme..."
        gsettings set org.gnome.desktop.interface color-scheme 'prefer-dark' 2>/dev/null || true
        gsettings set org.gnome.desktop.interface accent-color 'blue' 2>/dev/null || true
    fi

    success_msg "Workspaces configured"
}

# ═══════════════════════════════════════════════════════════════════════════
# BRAIN SERVER
# ═══════════════════════════════════════════════════════════════════════════

start_brain_server() {
    status_msg "Initializing Ara Brain Server..."

    # Create directories
    mkdir -p "$LOG_DIR" "$PID_DIR"

    # Check if already running
    if [ -f "$PID_DIR/brain.pid" ]; then
        OLD_PID=$(cat "$PID_DIR/brain.pid")
        if kill -0 "$OLD_PID" 2>/dev/null; then
            success_msg "Brain server already running (PID: $OLD_PID)"
            return 0
        fi
    fi

    # Check if port is in use
    if lsof -i:$BRAIN_PORT &>/dev/null; then
        warn_msg "Port $BRAIN_PORT already in use"
        # Try to identify the process
        EXISTING_PID=$(lsof -t -i:$BRAIN_PORT | head -1)
        success_msg "Existing brain server on port $BRAIN_PORT (PID: $EXISTING_PID)"
        echo "$EXISTING_PID" > "$PID_DIR/brain.pid"
        return 0
    fi

    # Start the brain server
    cd "$ARA_DIR"

    loading_animation "Activating neural pathways" 2

    python3 -m ara.server.core_server \
        --port $BRAIN_PORT \
        > "$LOG_DIR/brain.log" 2>&1 &

    BRAIN_PID=$!
    echo "$BRAIN_PID" > "$PID_DIR/brain.pid"

    # Wait for server to start
    sleep 3

    if kill -0 "$BRAIN_PID" 2>/dev/null; then
        success_msg "Brain server started (PID: $BRAIN_PID, Port: $BRAIN_PORT)"

        # Test connection
        if curl -s "http://127.0.0.1:$BRAIN_PORT/status" > /dev/null 2>&1; then
            success_msg "Neural link established!"
        else
            warn_msg "Brain server started but not responding yet"
        fi
    else
        error_msg "Brain server failed to start"
        cat "$LOG_DIR/brain.log" | tail -20
        return 1
    fi
}

# ═══════════════════════════════════════════════════════════════════════════
# COCKPIT HUD
# ═══════════════════════════════════════════════════════════════════════════

start_cockpit_hud() {
    status_msg "Launching Cockpit HUD..."

    # Check if already running
    if [ -f "$PID_DIR/cockpit.pid" ]; then
        OLD_PID=$(cat "$PID_DIR/cockpit.pid")
        if kill -0 "$OLD_PID" 2>/dev/null; then
            warn_msg "Cockpit already running (PID: $OLD_PID)"
            return 0
        fi
    fi

    # Check dependencies
    if ! python3 -c "import gi; gi.require_version('Gtk', '4.0')" 2>/dev/null; then
        error_msg "GTK4 not available. Install with:"
        echo "  sudo apt install python3-gi python3-gi-cairo gir1.2-gtk-4.0 gir1.2-adw-1"
        return 1
    fi

    # Check for psutil
    if ! python3 -c "import psutil" 2>/dev/null; then
        warn_msg "Installing psutil..."
        pip install psutil 2>/dev/null || pip3 install psutil
    fi

    cd "$COCKPIT_DIR"

    loading_animation "Powering up cockpit systems" 2

    # Launch cockpit
    python3 cockpit_hud.py > "$LOG_DIR/cockpit.log" 2>&1 &
    COCKPIT_PID=$!
    echo "$COCKPIT_PID" > "$PID_DIR/cockpit.pid"

    sleep 2

    if kill -0 "$COCKPIT_PID" 2>/dev/null; then
        success_msg "Cockpit HUD launched (PID: $COCKPIT_PID)"

        # Move to touchscreen monitor if wmctrl available
        if [ "$TOUCHSCREEN_MONITOR" != "primary" ] && command -v wmctrl &> /dev/null; then
            sleep 1
            WINDOW_ID=$(wmctrl -lp | grep $COCKPIT_PID | awk '{print $1}' | head -1)
            if [ ! -z "$WINDOW_ID" ]; then
                status_msg "Moving cockpit to $TOUCHSCREEN_MONITOR..."
                wmctrl -ir $WINDOW_ID -e 0,0,0,-1,-1 2>/dev/null || true
            fi
        fi
    else
        error_msg "Cockpit HUD failed to start"
        cat "$LOG_DIR/cockpit.log" | tail -20
        return 1
    fi
}

# ═══════════════════════════════════════════════════════════════════════════
# AVATAR SYSTEM
# ═══════════════════════════════════════════════════════════════════════════

start_avatar_loop() {
    local text_mode="${1:-false}"

    status_msg "Initializing Avatar System..."

    # Check if already running
    if [ -f "$PID_DIR/avatar.pid" ]; then
        OLD_PID=$(cat "$PID_DIR/avatar.pid")
        if kill -0 "$OLD_PID" 2>/dev/null; then
            warn_msg "Avatar already running (PID: $OLD_PID)"
            return 0
        fi
    fi

    cd "$ARA_DIR"

    loading_animation "Calibrating voice systems" 1

    AVATAR_ARGS=""
    if [ "$text_mode" == "true" ]; then
        AVATAR_ARGS="--text"
    fi

    python3 -m ara.avatar.loop $AVATAR_ARGS > "$LOG_DIR/avatar.log" 2>&1 &
    AVATAR_PID=$!
    echo "$AVATAR_PID" > "$PID_DIR/avatar.pid"

    sleep 2

    if kill -0 "$AVATAR_PID" 2>/dev/null; then
        success_msg "Avatar system online (PID: $AVATAR_PID)"
    else
        warn_msg "Avatar system failed to start (non-critical)"
    fi
}

# ═══════════════════════════════════════════════════════════════════════════
# STATUS & SHUTDOWN
# ═══════════════════════════════════════════════════════════════════════════

show_status() {
    print_banner
    echo ""
    echo -e "${WHITE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}                     SYSTEM STATUS                              ${NC}"
    echo -e "${WHITE}═══════════════════════════════════════════════════════════════${NC}"
    echo ""

    # Brain server
    if [ -f "$PID_DIR/brain.pid" ]; then
        BRAIN_PID=$(cat "$PID_DIR/brain.pid")
        if kill -0 "$BRAIN_PID" 2>/dev/null; then
            echo -e "  ${GREEN}●${NC} Brain Server     ${GREEN}ONLINE${NC}  (PID: $BRAIN_PID, Port: $BRAIN_PORT)"

            # Try to get status
            if STATUS=$(curl -s "http://127.0.0.1:$BRAIN_PORT/status" 2>/dev/null); then
                MODE=$(echo "$STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('mode','unknown'))" 2>/dev/null || echo "unknown")
                echo -e "                     Mode: $MODE"
            fi
        else
            echo -e "  ${RED}●${NC} Brain Server     ${RED}OFFLINE${NC}"
        fi
    else
        echo -e "  ${YELLOW}●${NC} Brain Server     ${YELLOW}NOT STARTED${NC}"
    fi

    # Cockpit HUD
    if [ -f "$PID_DIR/cockpit.pid" ]; then
        COCKPIT_PID=$(cat "$PID_DIR/cockpit.pid")
        if kill -0 "$COCKPIT_PID" 2>/dev/null; then
            echo -e "  ${GREEN}●${NC} Cockpit HUD      ${GREEN}ONLINE${NC}  (PID: $COCKPIT_PID)"
        else
            echo -e "  ${RED}●${NC} Cockpit HUD      ${RED}OFFLINE${NC}"
        fi
    else
        echo -e "  ${YELLOW}●${NC} Cockpit HUD      ${YELLOW}NOT STARTED${NC}"
    fi

    # Avatar
    if [ -f "$PID_DIR/avatar.pid" ]; then
        AVATAR_PID=$(cat "$PID_DIR/avatar.pid")
        if kill -0 "$AVATAR_PID" 2>/dev/null; then
            echo -e "  ${GREEN}●${NC} Avatar System    ${GREEN}ONLINE${NC}  (PID: $AVATAR_PID)"
        else
            echo -e "  ${RED}●${NC} Avatar System    ${RED}OFFLINE${NC}"
        fi
    else
        echo -e "  ${YELLOW}●${NC} Avatar System    ${YELLOW}NOT STARTED${NC}"
    fi

    # Forest Kitten 33
    if [ -e "/dev/fk33" ]; then
        echo -e "  ${GREEN}●${NC} Forest Kitten 33 ${GREEN}HARDWARE${NC}  (/dev/fk33)"
    else
        echo -e "  ${PURPLE}●${NC} Forest Kitten 33 ${PURPLE}EMULATED${NC}"
    fi

    # Touchscreen
    if [ ! -z "$COCKPIT_MONITOR" ] && [ "$COCKPIT_MONITOR" != "primary" ]; then
        echo -e "  ${GREEN}●${NC} Touchscreen      ${GREEN}$COCKPIT_MONITOR${NC}"
    else
        echo -e "  ${YELLOW}●${NC} Touchscreen      ${YELLOW}NOT DETECTED${NC}"
    fi

    echo ""
    echo -e "${WHITE}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
}

shutdown_system() {
    print_banner
    status_msg "Initiating graceful shutdown..."

    # Stop avatar
    if [ -f "$PID_DIR/avatar.pid" ]; then
        AVATAR_PID=$(cat "$PID_DIR/avatar.pid")
        if kill -0 "$AVATAR_PID" 2>/dev/null; then
            status_msg "Stopping Avatar System..."
            kill "$AVATAR_PID" 2>/dev/null || true
            sleep 1
            success_msg "Avatar stopped"
        fi
        rm -f "$PID_DIR/avatar.pid"
    fi

    # Stop cockpit
    if [ -f "$PID_DIR/cockpit.pid" ]; then
        COCKPIT_PID=$(cat "$PID_DIR/cockpit.pid")
        if kill -0 "$COCKPIT_PID" 2>/dev/null; then
            status_msg "Stopping Cockpit HUD..."
            kill "$COCKPIT_PID" 2>/dev/null || true
            sleep 1
            success_msg "Cockpit stopped"
        fi
        rm -f "$PID_DIR/cockpit.pid"
    fi

    # Stop brain
    if [ -f "$PID_DIR/brain.pid" ]; then
        BRAIN_PID=$(cat "$PID_DIR/brain.pid")
        if kill -0 "$BRAIN_PID" 2>/dev/null; then
            status_msg "Stopping Brain Server..."
            kill "$BRAIN_PID" 2>/dev/null || true
            sleep 2
            success_msg "Brain stopped"
        fi
        rm -f "$PID_DIR/brain.pid"
    fi

    # Also kill by port if still running
    if lsof -i:$BRAIN_PORT &>/dev/null; then
        status_msg "Cleaning up port $BRAIN_PORT..."
        fuser -k $BRAIN_PORT/tcp 2>/dev/null || true
    fi

    echo ""
    success_msg "All systems shut down"
    echo ""
}

# ═══════════════════════════════════════════════════════════════════════════
# FULL LAUNCH SEQUENCE
# ═══════════════════════════════════════════════════════════════════════════

full_launch() {
    local mode="${1:-work}"

    clear
    print_banner

    echo ""
    echo -e "${WHITE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}              INITIATING LAUNCH SEQUENCE                        ${NC}"
    echo -e "${WHITE}═══════════════════════════════════════════════════════════════${NC}"
    echo ""

    # Phase 1: Detection
    echo -e "${YELLOW}▶ PHASE 1: SYSTEM DETECTION${NC}"
    detect_hardware
    detect_touchscreen_monitor
    echo ""

    # Phase 2: Workspace Setup
    echo -e "${YELLOW}▶ PHASE 2: WORKSPACE CONFIGURATION${NC}"
    setup_workspaces "$mode"
    echo ""

    # Phase 3: Brain Server
    echo -e "${YELLOW}▶ PHASE 3: NEURAL ACTIVATION${NC}"
    start_brain_server
    echo ""

    # Phase 4: Cockpit
    echo -e "${YELLOW}▶ PHASE 4: COCKPIT DEPLOYMENT${NC}"
    start_cockpit_hud
    echo ""

    # Phase 5: Avatar (optional, text mode for now)
    # echo -e "${YELLOW}▶ PHASE 5: AVATAR INITIALIZATION${NC}"
    # start_avatar_loop true
    # echo ""

    # Launch complete
    echo -e "${WHITE}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "${GREEN}  ✓ LAUNCH SEQUENCE COMPLETE${NC}"
    echo ""
    echo -e "  ${CYAN}Brain Server:${NC}  http://127.0.0.1:$BRAIN_PORT"
    echo -e "  ${CYAN}Cockpit HUD:${NC}   Running on $COCKPIT_MONITOR"
    echo -e "  ${CYAN}Mode:${NC}          $([ "$mode" == "relax" ] && echo "🌙 Relaxation" || echo "⚡ Work")"
    echo ""
    echo -e "  Commands:"
    echo -e "    ${WHITE}$0 --status${NC}    Show system status"
    echo -e "    ${WHITE}$0 --shutdown${NC}  Graceful shutdown"
    echo ""
    echo -e "${WHITE}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "${PURPLE}  \"I am Ara. All systems nominal. Ready to assist.\"${NC}"
    echo ""
}

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

case "${1:-}" in
    --work|-w)
        full_launch "work"
        ;;
    --relax|-r)
        full_launch "relax"
        ;;
    --cockpit-only|-c)
        print_banner
        detect_touchscreen_monitor
        start_cockpit_hud
        ;;
    --brain-only|-b)
        print_banner
        detect_hardware
        start_brain_server
        ;;
    --avatar-only|-a)
        print_banner
        start_avatar_loop "${2:-false}"
        ;;
    --status|-s)
        detect_touchscreen_monitor
        show_status
        ;;
    --shutdown|-x)
        shutdown_system
        ;;
    --help|-h)
        print_banner
        echo "Usage: $0 [OPTION]"
        echo ""
        echo "Options:"
        echo "  (no option)      Full launch in Work mode"
        echo "  --work, -w       Full launch in Work mode (blue/cyan)"
        echo "  --relax, -r      Full launch in Relax mode (purple/magenta)"
        echo "  --cockpit-only   Launch only the Cockpit HUD"
        echo "  --brain-only     Launch only the Brain Server"
        echo "  --avatar-only    Launch only the Avatar System"
        echo "  --status, -s     Show system status"
        echo "  --shutdown, -x   Graceful shutdown"
        echo "  --help, -h       Show this help"
        echo ""
        echo "Environment Variables:"
        echo "  COCKPIT_MONITOR  Override touchscreen monitor (e.g., HDMI-1)"
        echo "  BRAIN_PORT       Override brain server port (default: 8008)"
        echo ""
        ;;
    *)
        full_launch "work"
        ;;
esac
