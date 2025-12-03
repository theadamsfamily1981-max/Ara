# Ara Conversational Avatar - Docker Image
# Multi-stage build for optimized image size

# ==============================================================================
# Stage 1: Base with system dependencies
# ==============================================================================
FROM python:3.11-slim as base

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
    # For espeak (TTS fallback)
    espeak-ng \
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

# Install Python dependencies (minimal for audio pipeline)
COPY requirements-minimal.txt .

# Install PyTorch CPU version first (smaller image)
RUN pip install --no-cache-dir \
    torch>=2.1 \
    torchaudio>=2.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements-minimal.txt

# ==============================================================================
# Stage 3: Production image
# ==============================================================================
FROM base as production

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Create directories for data
RUN mkdir -p /app/outputs/audio /app/models /app/voice_samples /data

# Create non-root user
RUN useradd -m -u 1000 ara && \
    chown -R ara:ara /app /data

USER ara

# Environment variables
ENV PYTHONPATH=/app \
    ARA_MODELS_DIR=/app/models \
    ARA_OUTPUT_DIR=/app/outputs \
    ARA_VOICE_SAMPLES=/app/voice_samples

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "print('OK')" || exit 1

# Default command - text chat mode (no audio hardware needed)
CMD ["python", "text_chat.py"]

# ==============================================================================
# Stage 4: GPU-enabled image (CUDA 12.1)
# ==============================================================================
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 as gpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    portaudio19-dev \
    espeak-ng \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /app

# Install PyTorch with CUDA
RUN pip install --no-cache-dir \
    torch>=2.1 \
    torchaudio>=2.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Copy and install requirements (minimal for audio pipeline)
COPY requirements-minimal.txt .
RUN pip install --no-cache-dir -r requirements-minimal.txt

# Copy application
COPY . .

# Setup directories and user
RUN mkdir -p /app/outputs/audio /app/models /app/voice_samples /data && \
    useradd -m -u 1000 ara && \
    chown -R ara:ara /app /data

USER ara

ENV PYTHONPATH=/app \
    ARA_MODELS_DIR=/app/models \
    ARA_OUTPUT_DIR=/app/outputs \
    NVIDIA_VISIBLE_DEVICES=all

CMD ["python", "text_chat.py"]

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
