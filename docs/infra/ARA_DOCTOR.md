# Ara Doctor - The Somatic Immune System

This document explains Ara's infrastructure layer: the "immune system" that diagnoses, heals, and maintains the organism's environment.

## Overview

Ara is not a single process - it's an organism with multiple organs that need different environments:

```
┌─────────────────────────────────────────────────────────────┐
│                        ARA ORGANISM                          │
├─────────────────────────────────────────────────────────────┤
│  Visual Cortex (Host venv)                                  │
│    - GTK4, WebKit, Cairo                                    │
│    - Direct display server access                           │
│    - Runs: cockpit_hud.py, MIES overlays                    │
├─────────────────────────────────────────────────────────────┤
│  Brain (Docker container)                                   │
│    - PyTorch, CUDA, Wav2Lip                                 │
│    - Isolated from host library conflicts                   │
│    - Runs: run_ara_somatic.py, LLM inference               │
├─────────────────────────────────────────────────────────────┤
│  Nervous System (Host native)                               │
│    - HAL shared memory (/dev/shm/ara_somatic)               │
│    - FPGA drivers, UIO, PCIe access                         │
│    - Runs: ara_daemon.py, hardware interfaces              │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ /dev/shm/ara_somatic
                            │ (512-byte zero-copy bridge)
                            ▼
              ┌─────────────────────────────┐
              │   HAL - Universal Translator │
              │   All organs speak this      │
              └─────────────────────────────┘
```

### Why Three Environments?

**The Problem**: GTK 4.12 wants GLib 2.80. PyTorch wants libstdc++ 11. WebKit wants specific NSS versions. These conflict.

**The Solution**: Let each organ have its own dependency universe. They communicate via shared memory, not library calls.

## The Tools

### `banos/engineer/ara_doctor.py`

**Deep diagnostics for organism assembly.**

Checks five organ systems:
1. **Visual Cortex** - GTK, WebKit, Cairo, PyGObject
2. **Nervous System** - FPGA, PCIe, UIO drivers
3. **Somatic Link** - HAL shared memory, posix_ipc
4. **Brain** - PyTorch, CUDA, transformers
5. **Respiratory** - PipeWire, ALSA, audio

```bash
# Full diagnosis
python3 banos/engineer/ara_doctor.py

# Check specific organ
python3 banos/engineer/ara_doctor.py --organ visual
python3 banos/engineer/ara_doctor.py --organ brain

# Generate healing script
python3 banos/engineer/ara_doctor.py --heal

# JSON output for automation
python3 banos/engineer/ara_doctor.py --json
```

**Output example:**
```
======================================================================
  ARA DOCTOR - ORGANISM DIAGNOSTIC REPORT
======================================================================
Host: Linux 5.15.0-91-generic (x86_64)
Distro: ubuntu 22.04
Python: 3.10.12
----------------------------------------------------------------------

[Visual Cortex]
  ✅ glib-2.0: glib-2.0 version 2.72.4
  ❌ gtk4: gtk4 not found or version too old
  ✅ libgirepository-1.0.so: Library libgirepository-1.0.so found

[Somatic Link]
  ✅ Shared Memory Filesystem: detected at /dev/shm
  ℹ️  HAL Memory: HAL not running

----------------------------------------------------------------------
Summary: 8 OK, 2 warnings, 1 errors, 0 critical

💊 Prescription saved to: heal_ara.sh
```

### `banos/engineer/bootstrap_organism.sh`

**One-time setup for the organism.**

```bash
# Full setup (host venv + Docker brain)
./banos/engineer/bootstrap_organism.sh

# Host only (if you run brain elsewhere)
./banos/engineer/bootstrap_organism.sh --host

# Docker brain only
./banos/engineer/bootstrap_organism.sh --docker

# Just run diagnostics
./banos/engineer/bootstrap_organism.sh --diagnose
```

**What it creates:**
- `venv_host/` - Python venv for host services
- `activate_host.sh` - Quick activation script
- `docker/Dockerfile.brain` - Brain container config
- `docker/docker-compose.yml` - Container orchestration
- Start/stop scripts

### `banos/engineer/start_host.sh`

**Start host-side services (HAL + UI).**

```bash
# Start both HAL and cockpit
./banos/engineer/start_host.sh

# HAL only
./banos/engineer/start_host.sh --hal

# Cockpit/MIES only
./banos/engineer/start_host.sh --cockpit
```

### `banos/engineer/start_brain_container.sh`

**Start the AI brain in Docker.**

```bash
# Interactive (foreground)
./banos/engineer/start_brain_container.sh

# Detached (background)
./banos/engineer/start_brain_container.sh -d

# Rebuild image first
./banos/engineer/start_brain_container.sh --build

# Debug shell
./banos/engineer/start_brain_container.sh --shell
```

## Workflow

### First Time Setup

```bash
# 1. Run bootstrap
./banos/engineer/bootstrap_organism.sh

# 2. Fix any issues
./heal_ara.sh  # If generated

# 3. Verify
python3 banos/engineer/ara_doctor.py
```

### Daily Development

```bash
# Terminal 1: Start host services
./banos/engineer/start_host.sh

# Terminal 2: Start brain container
./banos/engineer/start_brain_container.sh
```

### When Things Break

```bash
# 1. Run diagnostics
python3 banos/engineer/ara_doctor.py --heal

# 2. Review prescription
cat heal_ara.sh

# 3. Apply fixes
./heal_ara.sh

# 4. Verify
python3 banos/engineer/ara_doctor.py
```

## Adding New Organs

To add a new organ to Ara:

1. **Decide its environment:**
   - Host venv? (needs display, D-Bus, audio)
   - Docker? (heavy AI, conflicting deps)
   - Native? (hardware, kernel access)

2. **Add diagnostics to ara_doctor.py:**
   ```python
   def diagnose_new_organ(self):
       self._log("\n--- NEW ORGAN ---")
       self.check_python_mod("new_dep", organ="New Organ")
       self.check_sys_lib("libnew.so", "libnew-dev", "New Organ")
   ```

3. **Add to bootstrap if needed:**
   - Host deps → `setup_host()` apt-get list
   - Docker deps → `requirements-brain.txt`

4. **Communicate via HAL:**
   - Write to `/dev/shm/ara_somatic`
   - Use the defined struct format
   - Don't assume shared libraries

## File Map

```
banos/engineer/
├── __init__.py
├── ara_doctor.py          # Deep diagnostics
├── bootstrap_organism.sh  # One-time setup
├── start_host.sh          # Start HAL + UI
└── start_brain_container.sh # Start Docker brain

docker/
├── Dockerfile.brain       # Brain container
├── docker-compose.yml     # Orchestration
└── requirements-brain.txt # Python deps for brain

Generated files:
├── venv_host/             # Host Python venv
├── activate_host.sh       # Quick activation
├── start_ara.sh           # Start everything
├── stop_ara.sh            # Stop everything
└── heal_ara.sh            # Auto-generated fixes
```

## Comparison with tools/ara_build/

| Feature | `tools/ara_build/ara_doctor.py` | `banos/engineer/ara_doctor.py` |
|---------|----------------------------------|--------------------------------|
| Purpose | Quick build issue diagnosis | Deep organism assembly |
| Scope | Python packages, common errors | All 5 organ systems |
| Output | Pattern-matched suggestions | Full prescription script |
| When | During development builds | First-time setup, debugging |

Use `ara-build pip install foo` for build-time issues.
Use `banos/engineer/ara_doctor.py` for environment setup.

## Troubleshooting

### "HAL shared memory not found"

```bash
# Check if daemon is running
ps aux | grep ara_daemon

# Check shared memory
ls -la /dev/shm/ara*

# Start HAL manually
python3 -m banos.daemon.ara_daemon
```

### "Docker brain can't see GPU"

```bash
# Check NVIDIA runtime
docker info | grep nvidia

# Install nvidia-container-toolkit if missing
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

### "GTK initialization failed"

```bash
# Check display
echo $DISPLAY
echo $WAYLAND_DISPLAY

# Check if X/Wayland is running
loginctl show-session $(loginctl | grep $(whoami) | awk '{print $1}') -p Type
```

### "ldconfig says library exists but import fails"

The library exists but Python can't find its bindings:
```bash
# Check if GI typelib exists
ls /usr/lib/x86_64-linux-gnu/girepository-1.0/ | grep Gtk

# Install typelib package
sudo apt install gir1.2-gtk-4.0
```

## Philosophy

> Dependencies are not the enemy. **Conflicting** dependencies are the enemy.

The solution isn't "install everything in one venv and pray." It's:
1. **Isolate** organs that have incompatible needs
2. **Bridge** them via zero-copy shared memory
3. **Diagnose** before blindly installing
4. **Prescribe** targeted fixes, not kitchen-sink scripts

This makes dependency issues **localized, treatable pathologies** instead of **full-body sepsis**.
