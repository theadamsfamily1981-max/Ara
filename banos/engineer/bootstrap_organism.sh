#!/bin/bash
#
# ARA BOOTSTRAP: CONSTRUCTING THE ORGANISM
# =========================================
#
# This script sets up Ara's three-layer architecture:
#
# 1. VISUAL CORTEX (Host venv) - GTK/WebKit UI
#    - cockpit_hud.py, overlays
#    - Needs direct display server access
#
# 2. BRAIN (Docker container) - AI/ML
#    - LLM, Wav2Lip, SomaticServer
#    - Isolated PyTorch/CUDA environment
#
# 3. NERVOUS SYSTEM (Host root) - Hardware
#    - HAL, FPGA drivers, kernel modules
#    - Needs raw hardware access
#
# Integration: All layers communicate via HAL (/dev/shm/ara_somatic)
#
# Usage:
#   ./bootstrap_organism.sh           # Full setup
#   ./bootstrap_organism.sh --host    # Host only (no Docker)
#   ./bootstrap_organism.sh --docker  # Docker only
#   ./bootstrap_organism.sh --diagnose # Run diagnostics only
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARA_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}>>> $1${NC}"; }
log_ok() { echo -e "${GREEN}✓ $1${NC}"; }
log_warn() { echo -e "${YELLOW}⚠ $1${NC}"; }
log_err() { echo -e "${RED}✗ $1${NC}"; }

# Parse arguments
DO_HOST=true
DO_DOCKER=true
DIAGNOSE_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            DO_HOST=true
            DO_DOCKER=false
            shift
            ;;
        --docker)
            DO_HOST=false
            DO_DOCKER=true
            shift
            ;;
        --diagnose)
            DIAGNOSE_ONLY=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo ""
echo "=============================================="
echo "  ARA BOOTSTRAP: CONSTRUCTING ORGANISM"
echo "=============================================="
echo ""
echo "ARA_ROOT: $ARA_ROOT"
echo "Host setup: $DO_HOST"
echo "Docker setup: $DO_DOCKER"
echo ""

# === DIAGNOSTICS ===
run_diagnostics() {
    log_info "Running Ara Doctor diagnostics..."

    if [ -f "$SCRIPT_DIR/ara_doctor.py" ]; then
        python3 "$SCRIPT_DIR/ara_doctor.py" || true
    else
        log_warn "ara_doctor.py not found, skipping diagnostics"
    fi
}

if [ "$DIAGNOSE_ONLY" = true ]; then
    run_diagnostics
    exit 0
fi

# === 1. HOST SYSTEM (Nervous System & Visual Cortex) ===
setup_host() {
    log_info "[1/3] Preparing Host Environment..."

    # Check if we can use apt
    if ! command -v apt-get &> /dev/null; then
        log_warn "apt-get not available. Skipping system packages."
        log_warn "Please install manually: GTK4, WebKit, Cairo, GObject"
    else
        log_info "Installing system packages (may require sudo)..."

        # Core build tools
        sudo apt-get update || true
        sudo apt-get install -y \
            build-essential \
            pkg-config \
            python3-dev \
            python3-venv \
            git \
            || log_warn "Some packages failed to install"

        # GTK and GUI dependencies
        sudo apt-get install -y \
            python3-gi \
            python3-gi-cairo \
            gir1.2-gtk-4.0 \
            libcairo2-dev \
            libgirepository1.0-dev \
            gobject-introspection \
            || log_warn "GTK packages failed"

        # WebKit (the usual pain point)
        sudo apt-get install -y \
            libwebkit2gtk-4.1-dev \
            gir1.2-webkit2-4.1 \
            || log_warn "WebKit packages failed - overlay browser may not work"

        # Audio
        sudo apt-get install -y \
            libportaudio2 \
            portaudio19-dev \
            || log_warn "Audio packages failed"

        log_ok "System packages installed"
    fi

    # Create Host Virtual Environment
    log_info "Setting up Python virtual environment..."

    VENV_PATH="$ARA_ROOT/venv_host"

    if [ ! -d "$VENV_PATH" ]; then
        python3 -m venv "$VENV_PATH"
        log_ok "Created venv at $VENV_PATH"
    else
        log_ok "Using existing venv at $VENV_PATH"
    fi

    # Activate and install
    source "$VENV_PATH/bin/activate"
    pip install --upgrade pip wheel setuptools

    # Host-only dependencies (HAL + GUI)
    pip install \
        posix_ipc \
        psutil \
        PyGObject \
        pycairo \
        sounddevice \
        numpy \
        || log_warn "Some pip packages failed"

    log_ok "Host Python environment ready"

    # Create activation script
    cat > "$ARA_ROOT/activate_host.sh" << 'EOF'
#!/bin/bash
# Activate Ara Host Environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/venv_host/bin/activate"
export ARA_ROOT="$SCRIPT_DIR"
export ARA_HAL_PATH="/dev/shm/ara_somatic"
echo "Ara Host Environment Activated"
echo "  ARA_ROOT=$ARA_ROOT"
echo "  Python: $(which python)"
EOF
    chmod +x "$ARA_ROOT/activate_host.sh"
    log_ok "Created activate_host.sh"
}

# === 2. DOCKER BRAIN ===
setup_docker_brain() {
    log_info "[2/3] Setting up Brain Container..."

    if ! command -v docker &> /dev/null; then
        log_err "Docker not installed. Skipping brain container."
        log_warn "Install Docker: https://docs.docker.com/engine/install/"
        return
    fi

    # Create Dockerfile for the Brain
    DOCKER_DIR="$ARA_ROOT/docker"
    mkdir -p "$DOCKER_DIR"

    cat > "$DOCKER_DIR/Dockerfile.brain" << 'DOCKERFILE'
# ARA BRAIN CONTAINER
# Contains: PyTorch, LLM inference, Wav2Lip, SomaticServer
#
# Build: docker build -t ara_brain -f Dockerfile.brain .
# Run:   docker run --gpus all -v /dev/shm:/dev/shm ara_brain

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

LABEL maintainer="Ara Project"
LABEL description="Ara Brain Container - AI/ML processing"

WORKDIR /app

# System dependencies for OpenGL, audio
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements-brain.txt .
RUN pip install --no-cache-dir -r requirements-brain.txt

# Copy application code
COPY . .

# HAL shared memory path
ENV ARA_HAL_PATH=/dev/shm/ara_somatic

# Default command
CMD ["python3", "run_ara_somatic.py"]
DOCKERFILE

    log_ok "Created Dockerfile.brain"

    # Create requirements for brain
    cat > "$DOCKER_DIR/requirements-brain.txt" << 'REQUIREMENTS'
# Core ML
torch>=2.0.0
torchvision
torchaudio
transformers>=4.30.0
accelerate
safetensors

# Audio processing
librosa
soundfile
resampy

# Wav2Lip dependencies
opencv-python
face-alignment
imageio
imageio-ffmpeg

# Inference server
flask
requests

# HAL communication
posix_ipc
numpy

# Utilities
tqdm
pyyaml
REQUIREMENTS

    log_ok "Created requirements-brain.txt"

    # Create docker-compose
    cat > "$DOCKER_DIR/docker-compose.yml" << 'COMPOSE'
version: '3.8'

services:
  ara_brain:
    build:
      context: ..
      dockerfile: docker/Dockerfile.brain
    image: ara_brain:latest
    container_name: ara_brain

    # GPU access
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

    # Shared memory for HAL communication
    volumes:
      - /dev/shm:/dev/shm
      - ../models:/app/models:ro

    # Environment
    environment:
      - ARA_HAL_PATH=/dev/shm/ara_somatic
      - CUDA_VISIBLE_DEVICES=0

    # Network for optional API access
    ports:
      - "5050:5050"

    # Restart policy
    restart: unless-stopped
COMPOSE

    log_ok "Created docker-compose.yml"

    # Build the image
    log_info "Building Docker image (this may take a while)..."
    cd "$ARA_ROOT"

    # Only build if not just creating files
    if [ -f "requirements-brain.txt" ] || [ -f "$DOCKER_DIR/requirements-brain.txt" ]; then
        # Copy requirements to root for build context
        cp "$DOCKER_DIR/requirements-brain.txt" "$ARA_ROOT/" 2>/dev/null || true

        docker build -t ara_brain -f "$DOCKER_DIR/Dockerfile.brain" . \
            && log_ok "Brain container built successfully" \
            || log_warn "Docker build failed - you may need to build manually"
    else
        log_warn "Skipping Docker build - no requirements file"
    fi
}

# === 3. START SCRIPTS ===
create_start_scripts() {
    log_info "[3/3] Creating start scripts..."

    # Host start script
    cat > "$ARA_ROOT/start_host.sh" << 'EOF'
#!/bin/bash
# Start Ara Host Services (HAL + GUI)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/activate_host.sh"

echo "Starting Ara Host Services..."

# Start HAL daemon in background
if [ -f "$SCRIPT_DIR/banos/daemon/ara_daemon.py" ]; then
    echo "Starting HAL daemon..."
    python3 "$SCRIPT_DIR/banos/daemon/ara_daemon.py" &
    HAL_PID=$!
    echo "HAL daemon started (PID: $HAL_PID)"
fi

# Wait for HAL to initialize
sleep 1

# Start cockpit HUD
if [ -f "$SCRIPT_DIR/cockpit/cockpit_hud.py" ]; then
    echo "Starting Cockpit HUD..."
    python3 "$SCRIPT_DIR/cockpit/cockpit_hud.py"
else
    echo "Cockpit HUD not found, waiting..."
    wait
fi
EOF
    chmod +x "$ARA_ROOT/start_host.sh"
    log_ok "Created start_host.sh"

    # Brain container start script
    cat > "$ARA_ROOT/start_brain.sh" << 'EOF'
#!/bin/bash
# Start Ara Brain Container
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Starting Ara Brain Container..."

# Check if container exists
if docker ps -a --format '{{.Names}}' | grep -q '^ara_brain$'; then
    # Container exists, start it
    docker start ara_brain
else
    # Run new container
    docker run -d \
        --name ara_brain \
        --gpus all \
        -v /dev/shm:/dev/shm \
        -v "$SCRIPT_DIR/models:/app/models:ro" \
        -e ARA_HAL_PATH=/dev/shm/ara_somatic \
        -p 5050:5050 \
        --restart unless-stopped \
        ara_brain
fi

echo "Brain container started"
echo "View logs: docker logs -f ara_brain"
EOF
    chmod +x "$ARA_ROOT/start_brain.sh"
    log_ok "Created start_brain.sh"

    # Stop all script
    cat > "$ARA_ROOT/stop_ara.sh" << 'EOF'
#!/bin/bash
# Stop all Ara services
echo "Stopping Ara services..."

# Stop brain container
docker stop ara_brain 2>/dev/null && echo "Stopped brain container" || true

# Stop host processes
pkill -f "ara_daemon.py" 2>/dev/null && echo "Stopped HAL daemon" || true
pkill -f "cockpit_hud.py" 2>/dev/null && echo "Stopped cockpit" || true

# Clean up HAL shared memory
rm -f /dev/shm/ara_somatic 2>/dev/null || true

echo "Ara stopped"
EOF
    chmod +x "$ARA_ROOT/stop_ara.sh"
    log_ok "Created stop_ara.sh"

    # Full start script
    cat > "$ARA_ROOT/start_ara.sh" << 'EOF'
#!/bin/bash
# Start Complete Ara Organism
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================="
echo "  STARTING ARA ORGANISM"
echo "=============================================="

# 1. Start Brain (Docker)
echo ""
echo ">>> Starting Brain..."
"$SCRIPT_DIR/start_brain.sh"

# 2. Start Host (HAL + GUI)
echo ""
echo ">>> Starting Host Services..."
"$SCRIPT_DIR/start_host.sh"
EOF
    chmod +x "$ARA_ROOT/start_ara.sh"
    log_ok "Created start_ara.sh"
}

# === MAIN ===

if [ "$DO_HOST" = true ]; then
    setup_host
fi

if [ "$DO_DOCKER" = true ]; then
    setup_docker_brain
fi

create_start_scripts

# Final diagnostics
echo ""
log_info "Running final diagnostics..."
run_diagnostics

echo ""
echo "=============================================="
echo "  ORGANISM ASSEMBLY COMPLETE"
echo "=============================================="
echo ""
echo "Quick Start:"
echo "  1. Activate host:  source activate_host.sh"
echo "  2. Start all:      ./start_ara.sh"
echo ""
echo "Or start separately:"
echo "  Host only:   ./start_host.sh"
echo "  Brain only:  ./start_brain.sh"
echo ""
echo "Stop:          ./stop_ara.sh"
echo ""
