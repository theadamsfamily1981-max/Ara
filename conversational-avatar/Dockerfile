# Ara Conversational Avatar - Docker Image
# Uses Bullseye (older stable Debian) to fix 'libgl1' and 'av' conflicts
# This avoids the FFmpeg 7.x API incompatibility issues

FROM python:3.10-slim-bullseye

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    COQUI_TOS_AGREED=1

# Install system dependencies (including the ones that were "missing")
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    portaudio19-dev \
    libportaudio2 \
    libasound2-dev \
    libpulse-dev \
    espeak-ng \
    libespeak-ng1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch FIRST (CPU version - change to cu121 for NVIDIA GPU)
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install av BEFORE anything else to prevent version conflicts
# Bullseye's FFmpeg 4.x is compatible with av 10.x-12.x
RUN pip install --no-cache-dir "av>=10.0.0,<13.0.0"

# Install Coqui TTS (the good voice engine)
RUN pip install --no-cache-dir TTS>=0.22.0

# Copy and install requirements (av already satisfied, won't reinstall)
COPY requirements-minimal.txt .
RUN pip install --no-cache-dir -r requirements-minimal.txt

COPY requirements-multimodal.txt .
RUN pip install --no-cache-dir -r requirements-multimodal.txt || true

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p assets/voices outputs/audio models

# Create non-root user
RUN useradd -m -u 1000 ara && \
    chown -R ara:ara /app

USER ara

# Environment variables
ENV PYTHONPATH=/app \
    ARA_MODELS_DIR=/app/models \
    ARA_OUTPUT_DIR=/app/outputs \
    ARA_VOICE_SAMPLES=/app/assets/voices

EXPOSE 8000

# Default command - multimodal assistant
CMD ["python", "ara_multimodal.py"]
