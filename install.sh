#!/bin/bash
#═══════════════════════════════════════════════════════════════════════════════
#  ARA NEURAL COMMAND CENTER - Installation Suite
#═══════════════════════════════════════════════════════════════════════════════
#
#  This script installs all dependencies required for the Ara Cockpit HUD
#  and related AI systems.
#
#  Usage:
#      ./install.sh              # Full installation
#      ./install.sh --minimal    # Core dependencies only
#      ./install.sh --check      # Check dependencies without installing
#      ./install.sh --desktop    # Install desktop integration only
#
#═══════════════════════════════════════════════════════════════════════════════

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Installation directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

#───────────────────────────────────────────────────────────────────────────────
# Utility Functions
#───────────────────────────────────────────────────────────────────────────────

print_header() {
    echo ""
    echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}                                                                   ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}     ${WHITE}█████╗ ██████╗  █████╗     ${PURPLE}NEURAL COMMAND CENTER${NC}           ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}    ${WHITE}██╔══██╗██╔══██╗██╔══██╗${NC}                                    ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}    ${WHITE}███████║██████╔╝███████║${NC}    ${BLUE}Installation Suite v1.0${NC}         ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}    ${WHITE}██╔══██║██╔══██╗██╔══██║${NC}                                    ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}    ${WHITE}██║  ██║██║  ██║██║  ██║${NC}                                    ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}    ${WHITE}╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝${NC}                                    ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}                                                                   ${CYAN}║${NC}"
    echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_step() {
    echo -e "${BLUE}▶${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${CYAN}ℹ${NC} $1"
}

check_root() {
    if [[ $EUID -eq 0 ]]; then
        print_error "Please do not run this script as root"
        print_info "The script will ask for sudo when needed"
        exit 1
    fi
}

check_distro() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        DISTRO=$ID
        DISTRO_VERSION=$VERSION_ID
        print_info "Detected: $NAME $VERSION_ID"
    else
        print_warning "Could not detect distribution, assuming Debian-based"
        DISTRO="debian"
    fi
}

#───────────────────────────────────────────────────────────────────────────────
# Dependency Checking
#───────────────────────────────────────────────────────────────────────────────

check_command() {
    if command -v "$1" &> /dev/null; then
        return 0
    else
        return 1
    fi
}

check_python_package() {
    python3 -c "import $1" 2>/dev/null
    return $?
}

check_system_deps() {
    echo ""
    print_step "Checking system dependencies..."
    echo ""

    local missing=()
    local installed=()

    # Core tools
    for cmd in python3 pip3 git; do
        if check_command "$cmd"; then
            installed+=("$cmd")
        else
            missing+=("$cmd")
        fi
    done

    # GTK4/Adwaita - check via Python
    if python3 -c "import gi; gi.require_version('Gtk', '4.0'); from gi.repository import Gtk" 2>/dev/null; then
        installed+=("GTK4")
    else
        missing+=("GTK4")
    fi

    if python3 -c "import gi; gi.require_version('Adw', '1'); from gi.repository import Adw" 2>/dev/null; then
        installed+=("libadwaita")
    else
        missing+=("libadwaita")
    fi

    # GStreamer
    if python3 -c "import gi; gi.require_version('Gst', '1.0'); from gi.repository import Gst" 2>/dev/null; then
        installed+=("GStreamer")
    else
        missing+=("GStreamer")
    fi

    # WebKit (optional but recommended)
    if python3 -c "import gi; gi.require_version('WebKit', '6.0'); from gi.repository import WebKit" 2>/dev/null; then
        installed+=("WebKit6")
    else
        missing+=("WebKit6 (optional)")
    fi

    # Print results
    echo "  Installed:"
    for dep in "${installed[@]}"; do
        echo -e "    ${GREEN}✓${NC} $dep"
    done

    if [ ${#missing[@]} -gt 0 ]; then
        echo ""
        echo "  Missing:"
        for dep in "${missing[@]}"; do
            echo -e "    ${RED}✗${NC} $dep"
        done
        return 1
    fi

    return 0
}

check_python_deps() {
    echo ""
    print_step "Checking Python dependencies..."
    echo ""

    local missing=()
    local installed=()

    # Core packages
    for pkg in psutil requests; do
        if check_python_package "$pkg"; then
            installed+=("$pkg")
        else
            missing+=("$pkg")
        fi
    done

    # Optional packages
    if check_python_package "GPUtil"; then
        installed+=("GPUtil")
    else
        missing+=("GPUtil (optional - GPU monitoring)")
    fi

    # Print results
    echo "  Installed:"
    for dep in "${installed[@]}"; do
        echo -e "    ${GREEN}✓${NC} $dep"
    done

    if [ ${#missing[@]} -gt 0 ]; then
        echo ""
        echo "  Missing:"
        for dep in "${missing[@]}"; do
            echo -e "    ${YELLOW}○${NC} $dep"
        done
        return 1
    fi

    return 0
}

#───────────────────────────────────────────────────────────────────────────────
# Installation Functions
#───────────────────────────────────────────────────────────────────────────────

install_system_deps() {
    print_step "Installing system dependencies..."
    echo ""

    # Update package lists
    print_info "Updating package lists..."
    sudo apt-get update -qq

    # Core dependencies
    print_info "Installing core packages..."
    sudo apt-get install -y \
        python3 \
        python3-pip \
        python3-venv \
        python3-gi \
        python3-gi-cairo \
        gir1.2-gtk-4.0 \
        gir1.2-adw-1 \
        gir1.2-gst-plugins-base-1.0 \
        gir1.2-gstreamer-1.0 \
        gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good \
        gstreamer1.0-gtk4 \
        libgtk-4-dev \
        libadwaita-1-dev \
        git \
        curl \
        wget

    # WebKit for neural visualization (optional)
    print_info "Installing WebKit (optional - for neural visualization)..."
    sudo apt-get install -y gir1.2-webkit-6.0 2>/dev/null || \
        sudo apt-get install -y gir1.2-webkit2-4.1 2>/dev/null || \
        print_warning "WebKit not available - neural topology view will be disabled"

    # TTS dependencies (optional)
    print_info "Installing TTS support (optional)..."
    sudo apt-get install -y espeak-ng 2>/dev/null || \
        print_warning "espeak-ng not available"

    print_success "System dependencies installed"
}

install_python_deps() {
    print_step "Installing Python dependencies..."
    echo ""

    # Create virtual environment if it doesn't exist
    if [ ! -d "$VENV_DIR" ]; then
        print_info "Creating Python virtual environment..."
        python3 -m venv "$VENV_DIR"
    fi

    # Activate virtual environment
    source "$VENV_DIR/bin/activate"

    # Upgrade pip
    print_info "Upgrading pip..."
    pip install --upgrade pip -q

    # Install from requirements-cockpit.txt (lightweight) if it exists
    if [ -f "$SCRIPT_DIR/requirements-cockpit.txt" ]; then
        print_info "Installing from requirements-cockpit.txt..."
        pip install -r "$SCRIPT_DIR/requirements-cockpit.txt" -q
    elif [ -f "$SCRIPT_DIR/requirements.txt" ]; then
        print_info "Installing from requirements.txt..."
        pip install -r "$SCRIPT_DIR/requirements.txt" -q
    else
        # Install core packages
        print_info "Installing core Python packages..."
        pip install psutil requests -q

        # Install optional packages
        print_info "Installing optional packages..."
        pip install GPUtil 2>/dev/null || print_warning "GPUtil not available (GPU monitoring disabled)"
    fi

    print_success "Python dependencies installed"
}

install_fonts() {
    print_step "Installing premium fonts (optional)..."
    echo ""

    # Check if fonts directory exists
    FONTS_DIR="$HOME/.local/share/fonts"
    mkdir -p "$FONTS_DIR"

    # Install Inter font
    if [ ! -f "$FONTS_DIR/Inter-Regular.ttf" ]; then
        print_info "Downloading Inter font..."
        INTER_URL="https://github.com/rsms/inter/releases/download/v4.0/Inter-4.0.zip"
        TEMP_DIR=$(mktemp -d)

        if curl -sL "$INTER_URL" -o "$TEMP_DIR/inter.zip" 2>/dev/null; then
            unzip -q "$TEMP_DIR/inter.zip" -d "$TEMP_DIR"
            cp "$TEMP_DIR"/Inter-4.0/InterVariable*.ttf "$FONTS_DIR/" 2>/dev/null || \
                cp "$TEMP_DIR"/Inter-4.0/Inter*.ttf "$FONTS_DIR/" 2>/dev/null || true
            rm -rf "$TEMP_DIR"
            fc-cache -f "$FONTS_DIR" 2>/dev/null || true
            print_success "Inter font installed"
        else
            print_warning "Could not download Inter font - using system fonts"
        fi
    else
        print_info "Inter font already installed"
    fi

    # Install JetBrains Mono (for monospace)
    if [ ! -f "$FONTS_DIR/JetBrainsMono-Regular.ttf" ]; then
        print_info "Downloading JetBrains Mono font..."
        JBM_URL="https://github.com/JetBrains/JetBrainsMono/releases/download/v2.304/JetBrainsMono-2.304.zip"
        TEMP_DIR=$(mktemp -d)

        if curl -sL "$JBM_URL" -o "$TEMP_DIR/jbm.zip" 2>/dev/null; then
            unzip -q "$TEMP_DIR/jbm.zip" -d "$TEMP_DIR"
            cp "$TEMP_DIR"/fonts/ttf/*.ttf "$FONTS_DIR/" 2>/dev/null || true
            rm -rf "$TEMP_DIR"
            fc-cache -f "$FONTS_DIR" 2>/dev/null || true
            print_success "JetBrains Mono font installed"
        else
            print_warning "Could not download JetBrains Mono - using system fonts"
        fi
    else
        print_info "JetBrains Mono already installed"
    fi
}

install_desktop_integration() {
    print_step "Setting up desktop integration..."
    echo ""

    # Create applications directory
    APPS_DIR="$HOME/.local/share/applications"
    mkdir -p "$APPS_DIR"

    # Create desktop entry
    cat > "$APPS_DIR/ara-cockpit.desktop" << EOF
[Desktop Entry]
Name=Ara Cockpit
Comment=Ara Neural Command Center - AI Control Interface
Exec=$SCRIPT_DIR/gnome-tfan/app/launch_cockpit.sh
Icon=$SCRIPT_DIR/assets/ara-icon.png
Terminal=false
Type=Application
Categories=Utility;System;Monitor;
Keywords=AI;Neural;Cockpit;Monitor;System;
StartupWMClass=ara-cockpit
EOF

    # Create spaceship launcher entry
    cat > "$APPS_DIR/ara-spaceship.desktop" << EOF
[Desktop Entry]
Name=Ara Spaceship
Comment=Launch Ara Spaceship Command Center
Exec=$SCRIPT_DIR/launch_spaceship.sh
Icon=$SCRIPT_DIR/assets/ara-icon.png
Terminal=true
Type=Application
Categories=Utility;System;
Keywords=AI;Spaceship;Launch;
Actions=work;relax;cockpit;

[Desktop Action work]
Name=Work Mode
Exec=$SCRIPT_DIR/launch_spaceship.sh --work

[Desktop Action relax]
Name=Relax Mode
Exec=$SCRIPT_DIR/launch_spaceship.sh --relax

[Desktop Action cockpit]
Name=Cockpit Only
Exec=$SCRIPT_DIR/launch_spaceship.sh --cockpit-only
EOF

    # Update desktop database
    update-desktop-database "$APPS_DIR" 2>/dev/null || true

    print_success "Desktop integration installed"
    print_info "You can now find 'Ara Cockpit' in your applications menu"
}

create_launcher_scripts() {
    print_step "Creating launcher scripts..."
    echo ""

    # Create cockpit launcher
    cat > "$SCRIPT_DIR/gnome-tfan/app/launch_cockpit.sh" << 'EOF'
#!/bin/bash
# Ara Cockpit Launcher

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/../../.venv"

# Activate virtual environment if it exists
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
fi

# Launch cockpit
cd "$SCRIPT_DIR"
python3 cockpit_hud.py "$@"
EOF
    chmod +x "$SCRIPT_DIR/gnome-tfan/app/launch_cockpit.sh"

    # Create run script in root
    cat > "$SCRIPT_DIR/run-cockpit.sh" << 'EOF'
#!/bin/bash
# Quick launcher for Ara Cockpit

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/gnome-tfan/app/launch_cockpit.sh" "$@"
EOF
    chmod +x "$SCRIPT_DIR/run-cockpit.sh"

    print_success "Launcher scripts created"
}

create_assets_dir() {
    print_step "Setting up assets directory..."

    mkdir -p "$SCRIPT_DIR/assets"

    # Create a simple placeholder icon if none exists
    if [ ! -f "$SCRIPT_DIR/assets/ara-icon.png" ]; then
        # Create a simple SVG icon and convert if possible
        cat > "$SCRIPT_DIR/assets/ara-icon.svg" << 'EOF'
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128">
  <defs>
    <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#0a1628"/>
      <stop offset="100%" style="stop-color:#1a3a5c"/>
    </linearGradient>
    <linearGradient id="glow" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#64b4ff"/>
      <stop offset="100%" style="stop-color:#a078ff"/>
    </linearGradient>
  </defs>
  <circle cx="64" cy="64" r="60" fill="url(#bg)" stroke="url(#glow)" stroke-width="3"/>
  <text x="64" y="78" text-anchor="middle" font-family="sans-serif" font-size="48" font-weight="bold" fill="url(#glow)">A</text>
</svg>
EOF
        # Try to convert to PNG
        if check_command "rsvg-convert"; then
            rsvg-convert -w 128 -h 128 "$SCRIPT_DIR/assets/ara-icon.svg" > "$SCRIPT_DIR/assets/ara-icon.png"
        elif check_command "convert"; then
            convert -background none "$SCRIPT_DIR/assets/ara-icon.svg" -resize 128x128 "$SCRIPT_DIR/assets/ara-icon.png" 2>/dev/null || true
        fi
    fi

    print_success "Assets directory ready"
}

#───────────────────────────────────────────────────────────────────────────────
# Verification
#───────────────────────────────────────────────────────────────────────────────

verify_installation() {
    echo ""
    print_step "Verifying installation..."
    echo ""

    local errors=0

    # Check if cockpit_hud.py exists
    if [ -f "$SCRIPT_DIR/gnome-tfan/app/cockpit_hud.py" ]; then
        print_success "Cockpit HUD found"
    else
        print_error "Cockpit HUD not found"
        ((errors++))
    fi

    # Try importing required modules
    if python3 -c "import gi; gi.require_version('Gtk', '4.0'); from gi.repository import Gtk" 2>/dev/null; then
        print_success "GTK4 working"
    else
        print_error "GTK4 not working"
        ((errors++))
    fi

    if python3 -c "import gi; gi.require_version('Adw', '1'); from gi.repository import Adw" 2>/dev/null; then
        print_success "libadwaita working"
    else
        print_error "libadwaita not working"
        ((errors++))
    fi

    if python3 -c "import psutil" 2>/dev/null; then
        print_success "psutil working"
    else
        print_error "psutil not installed"
        ((errors++))
    fi

    if python3 -c "import requests" 2>/dev/null; then
        print_success "requests working"
    else
        print_error "requests not installed"
        ((errors++))
    fi

    # Optional checks
    if python3 -c "import GPUtil" 2>/dev/null; then
        print_success "GPUtil working (GPU monitoring enabled)"
    else
        print_warning "GPUtil not available (GPU monitoring disabled)"
    fi

    if python3 -c "import gi; gi.require_version('WebKit', '6.0'); from gi.repository import WebKit" 2>/dev/null; then
        print_success "WebKit6 working (neural visualization enabled)"
    else
        print_warning "WebKit6 not available (neural visualization disabled)"
    fi

    echo ""
    if [ $errors -eq 0 ]; then
        print_success "Installation verified successfully!"
        return 0
    else
        print_error "Installation has $errors error(s)"
        return 1
    fi
}

#───────────────────────────────────────────────────────────────────────────────
# Main Installation Flow
#───────────────────────────────────────────────────────────────────────────────

show_help() {
    echo "Ara Neural Command Center - Installation Suite"
    echo ""
    echo "Usage: ./install.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --help, -h       Show this help message"
    echo "  --check          Check dependencies without installing"
    echo "  --minimal        Install core dependencies only"
    echo "  --desktop        Install desktop integration only"
    echo "  --fonts          Install premium fonts only"
    echo "  --no-fonts       Skip font installation"
    echo "  --no-desktop     Skip desktop integration"
    echo "  --verify         Verify existing installation"
    echo ""
}

main() {
    local check_only=false
    local minimal=false
    local desktop_only=false
    local fonts_only=false
    local skip_fonts=false
    local skip_desktop=false
    local verify_only=false

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --help|-h)
                show_help
                exit 0
                ;;
            --check)
                check_only=true
                ;;
            --minimal)
                minimal=true
                ;;
            --desktop)
                desktop_only=true
                ;;
            --fonts)
                fonts_only=true
                ;;
            --no-fonts)
                skip_fonts=true
                ;;
            --no-desktop)
                skip_desktop=true
                ;;
            --verify)
                verify_only=true
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
        shift
    done

    print_header

    check_root
    check_distro

    # Verify only mode
    if $verify_only; then
        verify_installation
        exit $?
    fi

    # Check only mode
    if $check_only; then
        check_system_deps
        check_python_deps
        exit 0
    fi

    # Desktop only mode
    if $desktop_only; then
        install_desktop_integration
        exit 0
    fi

    # Fonts only mode
    if $fonts_only; then
        install_fonts
        exit 0
    fi

    # Full installation
    echo -e "${WHITE}Starting installation...${NC}"
    echo ""

    # System dependencies
    install_system_deps
    echo ""

    # Python dependencies
    install_python_deps
    echo ""

    # Fonts (unless skipped or minimal)
    if ! $skip_fonts && ! $minimal; then
        install_fonts
        echo ""
    fi

    # Create launcher scripts
    create_launcher_scripts
    echo ""

    # Assets
    create_assets_dir
    echo ""

    # Desktop integration (unless skipped)
    if ! $skip_desktop; then
        install_desktop_integration
        echo ""
    fi

    # Verify
    verify_installation

    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
    echo ""
    print_success "Installation complete!"
    echo ""
    echo -e "  ${WHITE}To launch the Ara Cockpit:${NC}"
    echo -e "    ${CYAN}./run-cockpit.sh${NC}"
    echo ""
    echo -e "  ${WHITE}Or use the full spaceship launcher:${NC}"
    echo -e "    ${CYAN}./launch_spaceship.sh${NC}"
    echo ""
    echo -e "  ${WHITE}The Ara Cockpit is also available in your applications menu.${NC}"
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
    echo ""
}

# Run main function
main "$@"
