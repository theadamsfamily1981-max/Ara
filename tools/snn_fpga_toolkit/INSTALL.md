# Installation Guide - SNN Kernel Module

## System Requirements

### Hardware
- **CPU**: x86_64 with performance counter support (Intel/AMD)
- **RAM**: Minimum 4 GB
- **GPU** (Optional): NVIDIA GPU with CUDA support
- **FPGA** (Optional): PCIe-attached FPGA (Xilinx/Intel)

### Software
- **Linux Kernel**: 4.15+ (tested on 5.4+, 6.x)
- **Kernel Headers**: Matching your running kernel
- **Build Tools**: gcc, make, kernel build tools
- **Optional**: CUDA Toolkit (for GPU support), FPGA tools

## Pre-Installation

### 1. Install Build Dependencies

#### Ubuntu/Debian
```bash
# Install kernel headers
sudo apt-get update
sudo apt-get install -y linux-headers-$(uname -r)

# Install build tools
sudo apt-get install -y build-essential gcc make

# Optional: CUDA toolkit
# wget https://developer.download.nvidia.com/compute/cuda/repos/...
# sudo dpkg -i cuda-repo-...
```

#### RHEL/CentOS/Fedora
```bash
# Install kernel headers
sudo dnf install -y kernel-devel-$(uname -r)

# Install build tools
sudo dnf install -y gcc make kernel-headers

# Optional: CUDA toolkit from NVIDIA
```

#### Arch Linux
```bash
# Install kernel headers
sudo pacman -S linux-headers

# Install build tools
sudo pacman -S base-devel
```

### 2. Verify Kernel Headers
```bash
ls /lib/modules/$(uname -r)/build
# Should show kernel build directory

uname -r
# Note your kernel version
```

### 3. Check System Capabilities

```bash
# Check if perf_events are available
ls /sys/bus/event_source/devices/
# Should show: cpu, msr, software, etc.

# Check for IOMMU (needed for P2P transfers)
dmesg | grep -i iommu
# Should show IOMMU initialization

# Check for GPU
lspci | grep -i nvidia
# Should show NVIDIA GPU if present

# Check for FPGA
lspci | grep -i xilinx
# Should show Xilinx FPGA if present
```

## Building the Module

### 1. Clone/Download Repository
```bash
cd /home/user/mcp  # Or your repo location
```

### 2. Build Kernel Module
```bash
make clean
make modules

# Expected output:
# Building kernel module...
# CC [M]  kernel/core/snn_core.o
# CC [M]  kernel/pcie/snn_pcie.o
# ...
# LD [M]  snn_kernel.ko
# Build complete!
```

### 3. Build Userspace API Library
```bash
make api

# This builds libsnn_api.so for userspace programs
```

### 4. Verify Build
```bash
ls -lh snn_kernel.ko
# Should show the compiled kernel module (typically 200-500 KB)

modinfo snn_kernel.ko
# Should show module information
```

## Installation

### 1. Load the Module (Testing)
```bash
# Load module temporarily (lost on reboot)
sudo insmod snn_kernel.ko debug_level=2

# Verify it loaded
lsmod | grep snn_kernel
# Should show: snn_kernel  XXX  0

# Check kernel logs
dmesg | tail -30
# Should show:
# SNN_AI_V2: Engine initialized (FP: Q24.8, Quant: INT8, Policy: Softmax)
# SNN_AI_V2: HPC monitoring enabled
# SNN_AI_V2: CSR++ graph enabled
# SNN_AI_V2: GNN model enabled
# SNN_AI_V2: Cold-start safety enabled
```

### 2. Install Permanently
```bash
# Install to system modules directory
sudo make install

# Update module dependencies
sudo depmod -a

# Enable auto-load on boot
echo "snn_kernel" | sudo tee /etc/modules-load.d/snn_kernel.conf

# Configure module parameters (optional)
echo "options snn_kernel debug_level=1" | sudo tee /etc/modprobe.d/snn_kernel.conf
```

### 3. Verify Installation
```bash
# Check module is installed
modinfo snn_kernel

# Load module
sudo modprobe snn_kernel

# Verify it's running
lsmod | grep snn_kernel

# Check device files
ls -l /dev/snn*
# Should show: /dev/snn0 (character device)
```

## Testing the Module

### 1. Basic Functionality Test

```bash
# Run simple test program
cd examples
gcc -o test_basic test_basic.c -I../include -L../api -lsnn_api
sudo ./test_basic

# Expected output:
# SNN Kernel Test
# Opening device /dev/snn0...
# Initializing pipeline...
# AI Recommendation: GPU=65%, FPGA=25%
# Test PASSED!
```

### 2. Performance Test

```bash
# Run performance benchmark
cd tools/monitoring
gcc -o snn_benchmark snn_benchmark.c -I../../include -L../../api -lsnn_api
sudo ./snn_benchmark --iterations=10000

# Expected output:
# Running 10000 iterations...
# Average decision latency: 44.2 μs
# p50: 42 μs, p95: 51 μs, p99: 67 μs
# Throughput: 22,624 decisions/sec
```

### 3. Cold-Start Verification

```bash
# Monitor cold-start progression
sudo dmesg -w &
sudo ./test_cold_start --iterations=6000

# Watch for phase transitions:
# SNN_COLD_START: Phase transition 0 -> 1 (decisions=100)
# SNN_COLD_START: Phase transition 1 -> 2 (decisions=1000)
# SNN_COLD_START: Phase transition 2 -> 3 (decisions=5000)
```

## Uninstallation

```bash
# Unload module
sudo rmmod snn_kernel

# Remove from auto-load
sudo rm /etc/modules-load.d/snn_kernel.conf
sudo rm /etc/modprobe.d/snn_kernel.conf

# Uninstall module files
sudo make uninstall

# Clean build artifacts
make clean
```

## Troubleshooting

### Module Fails to Load

**Error**: `insmod: ERROR: could not insert module`

**Solutions**:
```bash
# Check dmesg for specific error
dmesg | tail -50

# Common issues:
# 1. Kernel version mismatch
uname -r
# Rebuild with correct headers

# 2. Missing symbols
# Check if required kernel features are enabled:
grep CONFIG_PERF_EVENTS /boot/config-$(uname -r)
# Should show: CONFIG_PERF_EVENTS=y

# 3. Permission denied
# Ensure you're using sudo

# 4. Module already loaded
lsmod | grep snn_kernel
sudo rmmod snn_kernel  # Unload first
```

### Performance Counter Errors

**Error**: `SNN_HPC: Failed to create event`

**Solutions**:
```bash
# Check perf_events are enabled
cat /proc/sys/kernel/perf_event_paranoid
# Should be ≤ 2 (3 blocks kernel access)

# Temporarily allow kernel perf access
sudo sysctl kernel.perf_event_paranoid=0

# Make permanent
echo "kernel.perf_event_paranoid = 0" | sudo tee -a /etc/sysctl.conf
```

### GPU/FPGA Not Detected

**Error**: `GPU/FPGA monitoring unavailable`

**Note**: This is normal if you don't have GPU/FPGA hardware. The module runs in CPU-only mode with simulated metrics.

**For GPU Support**:
```bash
# Verify NVIDIA driver
nvidia-smi

# Check CUDA
nvcc --version

# Rebuild with CUDA support (future enhancement)
```

### Compilation Errors

**Error**: `implicit declaration of function 'fp_xxx'`

**Solution**: Fixed-point functions are static inline - ensure headers are included correctly:
```c
#include "kernel/semantic_ai/snn_fixed_point.h"
```

**Error**: `undefined reference to 'perf_event_create_kernel_counter'`

**Solution**: Ensure kernel has CONFIG_PERF_EVENTS=y. Check:
```bash
grep CONFIG_PERF_EVENTS /boot/config-$(uname -r)
```

## Module Parameters

Configure via `/etc/modprobe.d/snn_kernel.conf`:

```bash
# Debug level (0=off, 1=info, 2=debug, 3=verbose)
options snn_kernel debug_level=1

# Maximum nodes in knowledge graph
options snn_kernel max_graph_nodes=128

# Q-learning parameters
options snn_kernel learning_rate=100 exploration_rate=20
```

## Verification Checklist

After installation, verify:

- [ ] Module loads without errors: `lsmod | grep snn_kernel`
- [ ] Device file exists: `ls /dev/snn0`
- [ ] Kernel logs show initialization: `dmesg | grep SNN_AI`
- [ ] All phases initialized:
  - [ ] Phase 1: Q-learning (FP: Q24.8, Quant: INT8)
  - [ ] Phase 2: HPC monitoring
  - [ ] Phase 3: CSR++ graph and GNN
  - [ ] Phase 4: Cold-start safety
- [ ] Basic test passes: `sudo ./test_basic`
- [ ] Decision latency <100 μs: `sudo ./snn_benchmark`

## Security Considerations

⚠️ **Important**: This kernel module:
- Requires root/CAP_SYS_ADMIN privileges
- Accesses hardware performance counters
- Can allocate pinned memory
- Requires careful permission management in production

For production deployment:
- Limit module loading to trusted administrators
- Use SELinux/AppArmor policies
- Monitor syslog for anomalous behavior
- Regular security updates

## Support

For issues:
1. Check `dmesg` for error messages
2. Enable verbose logging: `debug_level=3`
3. Review documentation in `docs/`
4. Check GitHub issues

## Next Steps

After successful installation:
1. Read [API_GUIDE.md](docs/API_GUIDE.md) for programming examples
2. Review [PERFORMANCE_TUNING.md](docs/PERFORMANCE_TUNING.md) for optimization
3. See [examples/](examples/) for sample applications
4. Check [docs/PHASE*.md](docs/) for advanced features
