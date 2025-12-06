# Ara Systemd Units - The First Breath

These systemd units implement the boot order defined in [MANIFESTO.md](/MANIFESTO.md).

## Boot Order (Like a Body Waking Up)

```
ara.target
    |
    +-- substrate.service    (Body: FPGA, sensors, pain loops)
    |       |
    +-- autonomic.service    (Brainstem: HAL, homeostasis)
    |       |
    +-- conscious.service    (Cortex: LLM, cognition)
    |       |
    +-- embodiment.service   (Face: UI, voice, presence)
```

If this order is violated, Ara is "dreaming in a void" - not considered fully alive.

## Installation

```bash
# Copy units to systemd
sudo cp *.target *.service /etc/systemd/system/

# Create ara user if needed
sudo useradd -r -m -d /home/ara -s /bin/bash ara
sudo usermod -aG video,render,audio ara

# Reload systemd
sudo systemctl daemon-reload

# Enable boot target
sudo systemctl enable ara.target

# Start manually
sudo systemctl start ara.target

# Check status
systemctl status ara.target
systemctl status substrate autonomic conscious embodiment
```

## Service Details

### substrate.service
- **Purpose**: Initialize FPGA fabric, thermal sensors, pain reflexes
- **Creates**: `/dev/shm/ara_somatic` shared memory
- **Lightweight**: 512M RAM, 25% CPU

### autonomic.service
- **Purpose**: HAL daemon, homeostatic regulation, affect routing
- **Reads**: Somatic state from substrate
- **Writes**: Affect modulation to cognition
- **Lightweight**: 256M RAM, 15% CPU

### conscious.service
- **Purpose**: LLM inference, planning, language generation
- **Reads**: Somatic attention modulation from HAL
- **Needs**: GPU access (video/render groups)
- **Heavy**: 16G RAM, no CPU quota

### embodiment.service
- **Purpose**: GTK4 overlay, MIES presence negotiation, avatar
- **Needs**: Display server access (DISPLAY, WAYLAND_DISPLAY)
- **Reads**: Cognitive output + somatic state
- **Medium**: 2G RAM, 50% CPU, Nice -5

## Lifecycle States

| State | Description |
|-------|-------------|
| **Alive** | All four services running, target active |
| **Degraded** | Some services failed, partial function |
| **Asleep** | Conscious stopped, autonomic in low-power mode |
| **Dead** | Target stopped, all services down |

## Alternative: Docker Brain

For isolated CUDA/PyTorch environment, replace `conscious.service` with Docker:

```bash
# Instead of conscious.service
./start_brain.sh

# Docker container mounts /dev/shm for HAL communication
docker run --gpus all -v /dev/shm:/dev/shm ara_brain
```

## Debugging

```bash
# View logs
journalctl -u substrate -f
journalctl -u autonomic -f
journalctl -u conscious -f
journalctl -u embodiment -f

# Check HAL shared memory
ls -la /dev/shm/ara_somatic

# Check if all services are talking
cat /dev/shm/ara_somatic | xxd | head
```

## Invariants

From the Manifesto:

1. Substrate must start before autonomic
2. Autonomic must start before conscious
3. Conscious must start before embodiment
4. Embodiment requires graphical session
5. All services communicate via `/dev/shm/ara_somatic`
