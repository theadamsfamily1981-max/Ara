# Ara Installation Guide

Complete installation guide for Ara - Multimodal Voice & Vision Assistant.

## Quick Decision Tree

```
Do you want the easiest setup?
├── Yes → Use Docker (Solution 1)
│
└── No, I want to edit code easily
    │
    ├── Are you on Ubuntu?
    │   ├── Yes, and I'm comfortable with PPAs → Native Install (Solution 3)
    │   └── Yes, but I prefer isolation → Conda (Solution 2)
    │
    ├── Are you on Fedora/Arch/other Linux?
    │   └── Use Conda (Solution 2) or Docker (Solution 1)
    │
    └── Do you already use pyenv?
        └── Use pyenv + venv (Solution 4)
```

---

## Solution 1: Docker (Recommended)

**Best for:** Quick setup, guaranteed compatibility, deployment.

### Basic Setup

```bash
cd ~/Ara/conversational-avatar

# Build the image
docker build -t ara-avatar .

# Run (foreground, for testing)
docker run -p 8000:8000 --name ara-instance ara-avatar
```

### Production Setup (Recommended)

Run as a background service that restarts automatically:

```bash
docker run -d \
  --restart unless-stopped \
  -p 8000:8000 \
  --name ara-instance \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v $HOME/.cache/torch:/root/.cache/torch \
  -v $PWD/config:/app/config \
  -v $PWD/assets/voices:/app/assets/voices \
  ara-avatar
```

**What those volumes do:**
- `huggingface` & `torch` cache: Persist downloaded models (saves GB of redownloading)
- `config`: Edit configuration without rebuilding
- `assets/voices`: Add voice reference files without rebuilding

### GPU Support (NVIDIA)

If you have an NVIDIA GPU and want faster TTS/inference:

```bash
# Build GPU image
docker build --target gpu -t ara-avatar-gpu .

# Run with GPU access
docker run -d \
  --gpus all \
  --restart unless-stopped \
  -p 8000:8000 \
  --name ara-instance \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  ara-avatar-gpu
```

> **Note:** You must have the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed for `--gpus all` to work.

### Audio in Docker (Important!)

Docker containers don't have direct access to your microphone or speakers. The recommended pattern is:

- **Container**: Runs the AI/API (handles LLM, TTS generation, vision processing)
- **Host**: Handles mic input and speaker output, talks to container via HTTP

If you *must* have audio inside the container (advanced):

```bash
# Linux with PulseAudio
docker run -d \
  --device /dev/snd \
  -v /run/user/$(id -u)/pulse:/run/user/1000/pulse \
  -e PULSE_SERVER=unix:/run/user/1000/pulse/native \
  ara-avatar
```

This is fragile and varies by system. Docker is best for API-only deployments.

### Port Conflicts

If port 8000 is already in use:

```bash
docker run -p 8010:8000 ...  # Maps to port 8010 instead
```

---

## Solution 2: Conda (Most Flexible)

**Best for:** Development, easy code editing, works on any Linux distro.

### Quick Setup

```bash
cd ~/Ara/conversational-avatar

# Make executable and run
chmod +x setup_conda.sh
./setup_conda.sh

# Activate environment
conda activate ara_env

# Run Ara
python ara_multimodal.py
```

### Using environment.yml (Recommended)

For reproducible installs:

```bash
conda env create -f environment.yml
conda activate ara_env
```

### GPU Support

The default setup installs CPU-only PyTorch. For GPU acceleration:

```bash
conda activate ara_env

# Remove CPU torch
pip uninstall torch torchaudio

# Install GPU torch (adjust cu121 for your CUDA version)
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

Common CUDA versions:
- `cu118` - CUDA 11.8
- `cu121` - CUDA 12.1
- `cu124` - CUDA 12.4
- `cpu` - CPU only

Check your CUDA version with: `nvidia-smi`

### System Dependencies by Distro

**Ubuntu/Debian:**
```bash
sudo apt install -y portaudio19-dev espeak-ng libespeak-ng1 ffmpeg
```

**Fedora:**
```bash
sudo dnf install -y portaudio-devel espeak-ng espeak-ng-devel ffmpeg
```

**Arch Linux:**
```bash
sudo pacman -S portaudio espeak-ng ffmpeg
```

---

## Solution 3: Native Install (Ubuntu Only)

> **Warning:** This method is **Ubuntu-only** and intended for advanced users. Adding PPAs (like `deadsnakes`) modifies your system package set. If you're not comfortable with that, use Docker or Conda instead.
>
> Do **not** use this on Debian, Pop!_OS (without checking), or non-Ubuntu distros. The PPA is only safe on supported Ubuntu releases.

### Install Python 3.10 via Deadsnakes PPA

```bash
# Add deadsnakes PPA
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update

# Install Python 3.10
sudo apt install -y python3.10 python3.10-venv python3.10-dev

# Install system dependencies
sudo apt install -y portaudio19-dev espeak-ng libespeak-ng1 ffmpeg \
    libsndfile1 libasound2-dev
```

### Create Virtual Environment

```bash
cd ~/Ara/conversational-avatar

# Create venv with Python 3.10
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements-coqui.txt

# Run Ara
python ara_multimodal.py
```

---

## Solution 4: pyenv + venv (Advanced)

**Best for:** People who already use pyenv and prefer pure Python tooling.

### Install pyenv

See [pyenv installation guide](https://github.com/pyenv/pyenv#installation).

### Setup

```bash
cd ~/Ara/conversational-avatar

# Install Python 3.10
pyenv install 3.10.14
pyenv local 3.10.14

# Create venv
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements-coqui.txt
```

---

## Post-Install: Voice Cloning Setup

For the best voice quality, add a voice reference file:

1. Record a 3-10 second WAV file of clear speech
2. Save it as `assets/voices/ara_reference.wav`

```bash
# Example: record 5 seconds using arecord
arecord -f cd -d 5 assets/voices/ara_reference.wav

# Or convert an existing file
ffmpeg -i your_voice.mp3 -ar 22050 -ac 1 -acodec pcm_s16le assets/voices/ara_reference.wav
```

Without a reference file, Ara uses the default XTTS-v2 voice (still good quality).

---

## Gotchas & Troubleshooting

### Don't Mix System Python and Virtual Envs

- **Never** install this project's deps with `sudo pip`
- **Always** activate your env before running any `pip` or `python` commands:
  ```bash
  conda activate ara_env        # Conda
  source venv/bin/activate      # venv
  source .venv/bin/activate     # pyenv
  ```

### Model Download Size

First run downloads large models (XTTS-v2, Whisper). This can be **several GB** and take a while on slow networks.

- **Docker**: Map cache directories to persist models between rebuilds
- **Conda/Native**: Models cached in `~/.cache` by default

### CPU vs GPU Performance

Running on CPU works but may feel slow for real-time conversation:

| Setup | XTTS Speed | Recommended Whisper |
|-------|-----------|---------------------|
| CPU | ~2-5 sec/sentence | `small.en` or `medium` |
| GPU | ~0.3-1 sec/sentence | `medium` or `large` |

Adjust in config or use `--whisper-model small.en` if CPU-only.

### Common Errors

**`ModuleNotFoundError: No module named 'TTS'`**
- You're not in the right environment. Run `conda activate ara_env` or `source venv/bin/activate`.

**`illegal instruction (core dumped)`**
- Your CPU doesn't support AVX2 instructions needed by PyTorch. Try:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  ```

**`ALSA lib pcm.c:xxx (snd_pcm_open) Unknown PCM cards.pcm.xxx`**
- Normal on Linux. These are warnings, not errors. Ara suppresses most of them.

**`No module named 'pyaudio'` or build fails**
- Install system dependency first:
  ```bash
  sudo apt install portaudio19-dev  # Ubuntu/Debian
  ```

**`RuntimeError: CUDA out of memory`**
- Model too large for your GPU. Try:
  - Smaller Whisper model (`small.en` instead of `large`)
  - Set `device: "cpu"` in config

**`Connection refused` to Ollama**
- Make sure Ollama is running: `ollama serve`
- Check it's listening: `curl http://localhost:11434/api/tags`

---

## Quick Reference

| Method | Python | GPU | Audio | Best For |
|--------|--------|-----|-------|----------|
| Docker | Built-in | Yes (needs toolkit) | API only | Deployment, easy setup |
| Conda | 3.10 | Yes | Full | Development, any distro |
| Native | 3.10 (PPA) | Yes | Full | Ubuntu power users |
| pyenv | 3.10 | Yes | Full | pyenv enthusiasts |

---

## Getting Help

- Check logs: `docker logs ara-instance` (Docker)
- Verbose mode: `python ara_multimodal.py --verbose`
- Issues: https://github.com/theadamsfamily1981-max/Ara/issues
