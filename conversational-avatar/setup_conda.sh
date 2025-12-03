#!/bin/bash
# Ara - Conda Environment Setup Script
# Creates a Python 3.10 environment with Coqui TTS for high-quality voice synthesis

set -e

ENV_NAME="${1:-ara_env}"
echo "=========================================="
echo "  Ara - Conda Environment Setup"
echo "=========================================="
echo ""

# Check for conda
if ! command -v conda &> /dev/null; then
    echo "ERROR: Conda not found!"
    echo ""
    echo "Install Miniconda:"
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh"
    echo ""
    exit 1
fi

echo "Creating conda environment: $ENV_NAME (Python 3.10)"
conda create -n "$ENV_NAME" python=3.10 -y

echo ""
echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo ""
echo "Installing system audio dependencies..."
# These are needed for PyAudio and speech recognition
if command -v apt-get &> /dev/null; then
    echo "  (You may need to run these with sudo separately)"
    echo "  sudo apt install -y portaudio19-dev espeak-ng libespeak-ng1"
fi

echo ""
echo "Installing PyTorch (CPU)..."
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu

echo ""
echo "Installing Coqui TTS..."
pip install TTS>=0.22.0

echo ""
echo "Installing other dependencies..."
pip install ollama>=0.1
pip install SpeechRecognition>=3.10
pip install pyaudio>=0.2.14
pip install opencv-python>=4.8
pip install pillow>=10.0

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "To use Ara:"
echo "  1. Activate the environment:"
echo "     conda activate $ENV_NAME"
echo ""
echo "  2. Start Ollama (in another terminal):"
echo "     ollama serve"
echo ""
echo "  3. Run Ara:"
echo "     python ara_multimodal.py"
echo ""
echo "For best voice quality, add a voice reference file:"
echo "  cp your_voice_sample.wav assets/voices/ara_reference.wav"
echo ""
echo "=========================================="
