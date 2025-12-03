#!/bin/bash
# Ara Conversational Avatar - Installation Script
# Usage: ./install.sh [--gpu] [--dev] [--docker]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_MIN_VERSION="3.10"

# Parse arguments
GPU_MODE=false
DEV_MODE=false
DOCKER_MODE=false

for arg in "$@"; do
    case $arg in
        --gpu)
            GPU_MODE=true
            shift
            ;;
        --dev)
            DEV_MODE=true
            shift
            ;;
        --docker)
            DOCKER_MODE=true
            shift
            ;;
        --help|-h)
            echo "Ara Avatar Installation Script"
            echo ""
            echo "Usage: ./install.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --gpu      Install with CUDA GPU support"
            echo "  --dev      Install development dependencies"
            echo "  --docker   Build Docker image instead of local install"
            echo "  --help     Show this help message"
            exit 0
            ;;
    esac
done

echo -e "${CYAN}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║           Ara Conversational Avatar - Installer               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Docker mode
if [ "$DOCKER_MODE" = true ]; then
    echo -e "${CYAN}Building Docker image...${NC}"

    if [ "$GPU_MODE" = true ]; then
        docker build --target gpu -t ara-avatar:gpu .
        echo -e "${GREEN}✓ GPU Docker image built: ara-avatar:gpu${NC}"
        echo ""
        echo "Run with: docker run --gpus all -it ara-avatar:gpu"
    else
        docker build --target production -t ara-avatar:latest .
        echo -e "${GREEN}✓ CPU Docker image built: ara-avatar:latest${NC}"
        echo ""
        echo "Run with: docker run -it ara-avatar:latest"
    fi
    exit 0
fi

# Check Python version
echo -e "${CYAN}[1/6] Checking Python version...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 not found. Please install Python 3.10+${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [ "$(printf '%s\n' "$PYTHON_MIN_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$PYTHON_MIN_VERSION" ]; then
    echo -e "${RED}✗ Python $PYTHON_VERSION found, but $PYTHON_MIN_VERSION+ required${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python $PYTHON_VERSION${NC}"

# Check system dependencies
echo -e "${CYAN}[2/6] Checking system dependencies...${NC}"

MISSING_DEPS=""

if ! command -v ffmpeg &> /dev/null; then
    MISSING_DEPS="$MISSING_DEPS ffmpeg"
fi

if [ -n "$MISSING_DEPS" ]; then
    echo -e "${YELLOW}Missing system dependencies:$MISSING_DEPS${NC}"

    if command -v apt-get &> /dev/null; then
        echo "Installing with apt-get..."
        sudo apt-get update
        sudo apt-get install -y ffmpeg libsndfile1 portaudio19-dev espeak-ng
    elif command -v brew &> /dev/null; then
        echo "Installing with Homebrew..."
        brew install ffmpeg portaudio espeak
    elif command -v dnf &> /dev/null; then
        echo "Installing with dnf..."
        sudo dnf install -y ffmpeg libsndfile portaudio-devel espeak-ng
    else
        echo -e "${RED}Please install manually: ffmpeg, libsndfile, portaudio, espeak${NC}"
        exit 1
    fi
fi
echo -e "${GREEN}✓ System dependencies OK${NC}"

# Create virtual environment
echo -e "${CYAN}[3/6] Creating virtual environment...${NC}"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo -e "${GREEN}✓ Created $VENV_DIR${NC}"
else
    echo -e "${YELLOW}Using existing $VENV_DIR${NC}"
fi

# Activate venv
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo -e "${CYAN}[4/6] Upgrading pip...${NC}"
pip install --upgrade pip wheel setuptools

# Install PyTorch
echo -e "${CYAN}[5/6] Installing PyTorch...${NC}"
if [ "$GPU_MODE" = true ]; then
    echo "Installing PyTorch with CUDA support..."
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "Installing PyTorch (CPU)..."
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
fi
echo -e "${GREEN}✓ PyTorch installed${NC}"

# Install package
echo -e "${CYAN}[6/6] Installing Ara Avatar...${NC}"
if [ "$DEV_MODE" = true ]; then
    pip install -e ".[full,dev]"
else
    pip install -e ".[full]"
fi
echo -e "${GREEN}✓ Ara Avatar installed${NC}"

# Create directories
mkdir -p outputs/audio models voice_samples

# Download models (optional)
echo ""
echo -e "${CYAN}Downloading models...${NC}"
echo "This may take a while on first run."
python3 -c "
try:
    from TTS.api import TTS
    print('Pre-loading TTS model...')
    tts = TTS('tts_models/multilingual/multi-dataset/xtts_v2')
    print('✓ TTS model ready')
except Exception as e:
    print(f'Note: TTS model will download on first use: {e}')

try:
    import whisper
    print('Pre-loading Whisper model...')
    whisper.load_model('base')
    print('✓ Whisper model ready')
except Exception as e:
    print(f'Note: Whisper model will download on first use: {e}')
" 2>/dev/null || true

# Summary
echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    Installation Complete!                     ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Activate the environment:"
echo -e "  ${CYAN}source $VENV_DIR/bin/activate${NC}"
echo ""
echo "Available commands:"
echo -e "  ${CYAN}ara-avatar${NC}     - Run the full conversational avatar"
echo -e "  ${CYAN}ara-tts${NC}        - Text-to-speech CLI"
echo -e "  ${CYAN}ara-asr${NC}        - Speech-to-text CLI"
echo ""
echo "Quick test:"
echo -e "  ${CYAN}ara-tts \"Hello, I am Ara!\" -o hello.wav${NC}"
echo ""
if [ "$GPU_MODE" = true ]; then
    echo -e "${GREEN}GPU mode enabled - using CUDA for inference${NC}"
else
    echo -e "${YELLOW}CPU mode - add --gpu flag for GPU support${NC}"
fi
