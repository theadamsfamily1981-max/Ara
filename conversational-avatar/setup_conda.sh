#!/bin/bash
# Ara - Conda Environment Setup Script
# Creates a Python 3.10 environment with Coqui TTS for high-quality voice synthesis
#
# Usage:
#   ./setup_conda.sh              # CPU only (default)
#   ./setup_conda.sh --gpu        # With NVIDIA GPU support
#   ./setup_conda.sh my_env_name  # Custom environment name

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

# Defaults
ENV_NAME="ara_env"
GPU_MODE=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --gpu)
            GPU_MODE=true
            ;;
        --help|-h)
            echo "Ara Conda Setup Script"
            echo ""
            echo "Usage: ./setup_conda.sh [OPTIONS] [ENV_NAME]"
            echo ""
            echo "Options:"
            echo "  --gpu    Install with NVIDIA CUDA GPU support"
            echo "  --help   Show this help"
            echo ""
            echo "Examples:"
            echo "  ./setup_conda.sh              # CPU only, env name 'ara_env'"
            echo "  ./setup_conda.sh --gpu        # With GPU support"
            echo "  ./setup_conda.sh my_env       # Custom environment name"
            exit 0
            ;;
        *)
            # Assume it's the environment name if not a flag
            if [[ ! "$arg" =~ ^- ]]; then
                ENV_NAME="$arg"
            fi
            ;;
    esac
done

echo -e "${CYAN}"
echo "=========================================="
echo "  Ara - Conda Environment Setup"
echo "=========================================="
echo -e "${NC}"

# Check for conda
if ! command -v conda &> /dev/null; then
    echo -e "${RED}ERROR: Conda not found!${NC}"
    echo ""
    echo "Install Miniconda:"
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh"
    echo "  # Then restart your terminal"
    echo ""
    exit 1
fi

# Check if environment already exists
if conda env list | grep -q "^$ENV_NAME "; then
    echo -e "${YELLOW}Environment '$ENV_NAME' already exists.${NC}"
    read -p "Remove and recreate? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n "$ENV_NAME" -y
    else
        echo "Exiting. To use existing env: conda activate $ENV_NAME"
        exit 0
    fi
fi

echo -e "${CYAN}Creating conda environment: $ENV_NAME (Python 3.10)${NC}"
conda create -n "$ENV_NAME" python=3.10 -y

echo ""
echo -e "${CYAN}Activating environment...${NC}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Detect distro and show system dependency hints
echo ""
echo -e "${CYAN}System dependencies needed (run separately with sudo):${NC}"
if [ -f /etc/os-release ]; then
    . /etc/os-release
    case "$ID" in
        ubuntu|debian)
            echo "  sudo apt install -y portaudio19-dev espeak-ng libespeak-ng1 ffmpeg"
            ;;
        fedora|rhel|centos)
            echo "  sudo dnf install -y portaudio-devel espeak-ng espeak-ng-devel ffmpeg"
            ;;
        arch|manjaro)
            echo "  sudo pacman -S portaudio espeak-ng ffmpeg"
            ;;
        *)
            echo "  Install: portaudio, espeak-ng, ffmpeg"
            ;;
    esac
else
    echo "  Install: portaudio, espeak-ng, ffmpeg"
fi
echo ""

# Install PyTorch
echo -e "${CYAN}Installing PyTorch...${NC}"
if [ "$GPU_MODE" = true ]; then
    # Check for nvidia-smi to detect CUDA version
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
        echo -e "${GREEN}Detected NVIDIA driver: $CUDA_VERSION${NC}"
    fi
    echo "Installing PyTorch with CUDA 12.1 support..."
    pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
    echo -e "${GREEN}✓ PyTorch installed with GPU support${NC}"
    echo ""
    echo -e "${YELLOW}Note: If your CUDA version differs, you may need to reinstall:${NC}"
    echo "  pip uninstall torch torchaudio"
    echo "  pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8"
    echo "  pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1"
else
    echo "Installing PyTorch (CPU only)..."
    pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
    echo -e "${GREEN}✓ PyTorch installed (CPU)${NC}"
fi

echo ""
echo -e "${CYAN}Installing Coqui TTS (neural voice synthesis)...${NC}"
pip install TTS>=0.22.0
echo -e "${GREEN}✓ Coqui TTS installed${NC}"

echo ""
echo -e "${CYAN}Installing other dependencies...${NC}"
pip install sounddevice>=0.4.6
pip install soundfile>=0.12.0
pip install numpy>=1.24.0
pip install ollama>=0.1
pip install SpeechRecognition>=3.10
pip install pyaudio>=0.2.14
pip install opencv-python>=4.8
pip install pillow>=10.0
pip install pyyaml>=6.0
pip install loguru>=0.7
echo -e "${GREEN}✓ Dependencies installed${NC}"

echo ""
echo -e "${GREEN}=========================================="
echo "  Setup Complete!"
echo "==========================================${NC}"
echo ""
echo "To use Ara:"
echo ""
echo "  1. Activate the environment:"
echo -e "     ${CYAN}conda activate $ENV_NAME${NC}"
echo ""
echo "  2. Start Ollama (in another terminal):"
echo -e "     ${CYAN}ollama serve${NC}"
echo ""
echo "  3. Run Ara:"
echo -e "     ${CYAN}python ara_multimodal.py${NC}"
echo ""
echo -e "${YELLOW}For best voice quality, add a voice reference file:${NC}"
echo "  Record 3-10 seconds of clear speech, then:"
echo -e "  ${CYAN}cp your_voice.wav assets/voices/ara_reference.wav${NC}"
echo ""
if [ "$GPU_MODE" = true ]; then
    echo -e "${GREEN}GPU mode enabled - TTS will use CUDA for faster synthesis${NC}"
else
    echo -e "${YELLOW}CPU mode - for faster TTS, reinstall with: ./setup_conda.sh --gpu${NC}"
fi
echo ""
echo "=========================================="
