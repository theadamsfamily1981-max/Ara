# Ara Conversational Avatar - Docker Image
# Multi-stage build for optimized image size
# IMPORTANT: Uses Python 3.10 for Coqui TTS compatibility

# ==============================================================================
# Stage 1: Base with system dependencies
# ==============================================================================
FROM python:3.10-slim as base

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Audio processing
    ffmpeg \
    libsndfile1 \
    portaudio19-dev \
    libportaudio2 \
    libasound2-dev \
    libpulse-dev \
    # For espeak (TTS fallback - required by Coqui)
    espeak-ng \
    libespeak-ng1 \
    # OpenCV dependencies
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    # Build tools
    build-essential \
    git \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ==============================================================================
# Stage 2: Builder with Python dependencies
# ==============================================================================
FROM base as builder

WORKDIR /build

# Copy all requirements files
COPY requirements-minimal.txt .
COPY requirements-multimodal.txt .
COPY requirements-coqui.txt .

# Install PyTorch CPU version first (smaller image)
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install Coqui TTS (requires Python 3.10)
RUN pip install --no-cache-dir TTS>=0.22.0

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements-minimal.txt
RUN pip install --no-cache-dir -r requirements-multimodal.txt || true

# ==============================================================================
# Stage 3: Production image
# ==============================================================================
FROM base as production

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Create directories for data and voice samples
RUN mkdir -p /app/outputs/audio /app/models /app/assets/voices /data

# Create non-root user
RUN useradd -m -u 1000 ara && \
    chown -R ara:ara /app /data

USER ara

# Environment variables
ENV PYTHONPATH=/app \
    ARA_MODELS_DIR=/app/models \
    ARA_OUTPUT_DIR=/app/outputs \
    ARA_VOICE_SAMPLES=/app/assets/voices \
    COQUI_TOS_AGREED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "from TTS.api import TTS; print('OK')" || exit 1

# Default command - multimodal assistant
CMD ["python", "ara_multimodal.py"]

# ==============================================================================
# Stage 4: GPU-enabled image (CUDA 12.1)
# ==============================================================================
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 as gpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Install Python 3.10 and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    portaudio19-dev \
    libasound2-dev \
    espeak-ng \
    libespeak-ng1 \
    libgl1 \
    libglib2.0-0 \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && python -m pip install --upgrade pip

WORKDIR /app

# Install PyTorch with CUDA
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install Coqui TTS
RUN pip install --no-cache-dir TTS>=0.22.0

# Copy and install requirements
COPY requirements-minimal.txt .
COPY requirements-multimodal.txt .
RUN pip install --no-cache-dir -r requirements-minimal.txt
RUN pip install --no-cache-dir -r requirements-multimodal.txt || true

# Copy application
COPY . .

# Setup directories and user
RUN mkdir -p /app/outputs/audio /app/models /app/assets/voices /data && \
    useradd -m -u 1000 ara && \
    chown -R ara:ara /app /data

USER ara

ENV PYTHONPATH=/app \
    ARA_MODELS_DIR=/app/models \
    ARA_OUTPUT_DIR=/app/outputs \
    ARA_VOICE_SAMPLES=/app/assets/voices \
    COQUI_TOS_AGREED=1 \
    NVIDIA_VISIBLE_DEVICES=all

CMD ["python", "ara_multimodal.py"]

# ==============================================================================
# Stage 5: Development image
# ==============================================================================
FROM production as development

USER root

# Install development tools
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    black \
    ruff \
    mypy \
    ipython \
    jupyter

USER ara

# Expose Jupyter port
EXPOSE 8888

CMD ["ipython"]
