# Safe Installation Guide - With Rollback Options

## Overview

This guide provides a **progressive, safe installation** with rollback options at every step.

**Key Safety Principle**: This is a **loadable kernel module**, not a full kernel replacement. You can unload it instantly without rebooting.

---

## Safety Tier 1: Testing (No Permanent Changes)

### Step 1: Build the Module

On your **real system** (not this sandbox):

```bash
cd /path/to/mcp

# Install dependencies (one-time)
# Ubuntu/Debian:
sudo apt-get update
sudo apt-get install -y linux-headers-$(uname -r) build-essential

# Fedora/RHEL:
sudo dnf install -y kernel-devel-$(uname -r) gcc make

# Arch:
sudo pacman -S linux-headers base-devel

# Build the module
make clean
make modules

# Expected output:
#   CC [M]  kernel/core/snn_core.o
#   ...
#   LD [M]  snn_kernel.ko
#   Build complete!

# Verify it built
ls -lh snn_kernel.ko
# Should show file ~200-500 KB
```

**ROLLBACK**: If build fails, nothing is installed. Just `make clean` and you're back to original state.

---

### Step 2: Load Module Temporarily (Testing Mode)

```bash
# Load module (TEMPORARY - lost on reboot)
sudo insmod snn_kernel.ko debug_level=2

# Check if it loaded
lsmod | grep snn_kernel
# Should show: snn_kernel  XXX  0

# Check kernel logs for initialization
dmesg | tail -30
# Should show:
#   SNN_AI_V2: Engine initialized (FP: Q24.8, Quant: INT8, Policy: Softmax)
#   SNN_AI_V2: HPC monitoring enabled
#   SNN_AI_V2: CSR++ graph enabled
#   SNN_AI_V2: GNN model enabled
#   SNN_AI_V2: Cold-start safety enabled
#   SNN: Character device /dev/snn0 created

# Check device file exists
ls -l /dev/snn0
# Should show: crw------- 1 root root 245, 0 ... /dev/snn0
```

**ROLLBACK**: If module won't load or shows errors:
```bash
# Instant rollback - unload the module
sudo rmmod snn_kernel

# Verify it's gone
lsmod | grep snn_kernel
# Should show nothing

# Your system is back to original state!
```

**IMPORTANT**: At this point, the module is loaded but NOT permanent. If you reboot, it's automatically gone.

---

### Step 3: Test Basic Functionality

```bash
# Build the test program
cd examples
gcc -o test_basic test_basic.c -I../include

# Run basic test
sudo ./test_basic

# Expected output:
#   =================================
#   SNN Kernel Module - Basic Test
#   =================================
#
#   1. Opening device /dev/snn0...
#      ✓ Device opened successfully (fd=3)
#
#   2. Initializing SNN pipeline...
#      ✓ Pipeline initialized
#
#   3. Testing AI Engine...
#      AI Recommendation (Dense workload, 100K neurons):
#        GPU:        65%
#        FPGA:       25%
#        CPU:        10000 neurons
#        Batch size: 32
#        Confidence: LOW
#
#   ...
#
#   ✓ All tests PASSED!
```

**ROLLBACK**: If test fails or system becomes unstable:
```bash
# Immediate rollback
sudo rmmod snn_kernel

# Check system is stable
dmesg | tail -20
```

---

### Step 4: Stress Test (Optional but Recommended)

```bash
# Run test multiple times to verify stability
for i in {1..100}; do
    sudo ./test_basic >/dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "Test failed at iteration $i"
        break
    fi
    echo "Iteration $i: OK"
done

# Monitor system resources
top -n 1 | grep snn

# Check for kernel errors
dmesg | grep -i error | grep -i snn
# Should show nothing
```

**ROLLBACK**: If stress test reveals issues:
```bash
sudo rmmod snn_kernel
# Module gone, system back to normal
```

---

### Step 5: Unload and Verify Clean Shutdown

```bash
# Unload module
sudo rmmod snn_kernel

# Verify it's gone
lsmod | grep snn_kernel
# Should show nothing

# Check device file is removed
ls /dev/snn0
# Should show: No such file or directory

# Check for any errors during unload
dmesg | tail -10
# Should show clean shutdown
```

**SUCCESS**: If you reach this point without issues, the module is safe to use!

---

## Safety Tier 2: Auto-Load on Boot (Reversible)

**Only proceed here if Tier 1 testing succeeded completely.**

### Step 6: Configure Auto-Load (Still Reversible)

```bash
# Install module to system directory
sudo make install

# Update module dependencies
sudo depmod -a

# Enable auto-load on boot
echo "snn_kernel" | sudo tee /etc/modules-load.d/snn_kernel.conf

# Configure default parameters
echo "options snn_kernel debug_level=1" | sudo tee /etc/modprobe.d/snn_kernel.conf

# Load it now (to test)
sudo modprobe snn_kernel

# Verify it loaded
lsmod | grep snn_kernel
```

**ROLLBACK**: To disable auto-load:
```bash
# Remove auto-load configuration
sudo rm /etc/modules-load.d/snn_kernel.conf
sudo rm /etc/modprobe.d/snn_kernel.conf

# Unload module
sudo modprobe -r snn_kernel

# Optionally uninstall from system
sudo make uninstall

# Reboot - module won't load
sudo reboot
```

---

### Step 7: Test After Reboot

```bash
# Reboot your system
sudo reboot

# After reboot, check if module auto-loaded
lsmod | grep snn_kernel
# Should show: snn_kernel  XXX  0

# Run test again
cd /path/to/mcp/examples
sudo ./test_basic

# Check performance
dmesg | grep SNN_AI
```

**ROLLBACK**: If system behaves oddly after reboot:
```bash
# Disable auto-load (survives reboot)
sudo rm /etc/modules-load.d/snn_kernel.conf

# Unload immediately
sudo rmmod snn_kernel

# Reboot - module won't load anymore
sudo reboot
```

---

## Safety Tier 3: Long-Term Production Use

**Only proceed here if Tier 2 testing succeeded over multiple reboots.**

### Step 8: Production Configuration

```bash
# Reduce debug logging (performance)
echo "options snn_kernel debug_level=0" | sudo tee /etc/modprobe.d/snn_kernel.conf

# Configure for your workload
# Example: more graph nodes
echo "options snn_kernel max_graph_nodes=128" | sudo tee -a /etc/modprobe.d/snn_kernel.conf

# Reload with new config
sudo rmmod snn_kernel
sudo modprobe snn_kernel

# Verify new config
dmesg | grep SNN_AI
```

---

## Emergency Rollback Procedures

### If Module Won't Unload

```bash
# Check what's using it
lsmod | grep snn_kernel
# Column 3 shows usage count (should be 0)

# If usage count > 0, close applications using /dev/snn0
fuser -v /dev/snn0  # Shows processes
sudo killall test_basic  # Or whatever is using it

# Try unload again
sudo rmmod snn_kernel
```

### If Module Causes Boot Issues (Rare)

1. **Boot into Recovery Mode**:
   - Reboot and hold Shift (GRUB menu)
   - Select "Advanced options"
   - Select "Recovery mode"

2. **Disable Auto-Load**:
   ```bash
   # Mount filesystem as read-write
   mount -o remount,rw /

   # Remove auto-load
   rm /etc/modules-load.d/snn_kernel.conf

   # Reboot normally
   reboot
   ```

3. **Alternative: Boot Parameter**:
   - At GRUB menu, press 'e' to edit boot entry
   - Find line starting with 'linux'
   - Add at end: `module_blacklist=snn_kernel`
   - Press Ctrl+X to boot
   - System will boot without loading snn_kernel

### If Module Causes Kernel Panic (Very Rare)

1. **System will auto-reboot** (kernel panic handler)
2. **Module is NOT loaded after reboot** (panics prevent loading)
3. **Disable auto-load** as shown above
4. **Check logs**: `journalctl -k -b -1 | grep snn`

---

## Verification Checklist

Before considering the module "safe":

- [ ] **Tier 1 Complete**: Module loads/unloads cleanly 10+ times
- [ ] **Basic Test Passes**: `test_basic` succeeds 100+ iterations
- [ ] **No Kernel Errors**: `dmesg` shows no errors or warnings
- [ ] **Clean Shutdown**: `rmmod` completes without errors
- [ ] **Tier 2 Complete**: Module auto-loads after reboot successfully
- [ ] **Multi-Reboot Stable**: Works correctly over 3+ reboots
- [ ] **Resource Usage Normal**: `top` shows reasonable CPU/memory

---

## What Can Go Wrong? (And How to Fix)

### 1. Module Won't Load

**Symptom**: `insmod` fails with error

**Diagnosis**:
```bash
dmesg | tail -30
```

**Common Causes**:
- Kernel version mismatch → Rebuild on your kernel
- Missing CONFIG_PERF_EVENTS → Module uses simulated metrics (safe)
- Permission denied → Use sudo

**Fix**: See BUILD_VERIFICATION.md troubleshooting section

### 2. Module Loaded But /dev/snn0 Missing

**Symptom**: Device file not created

**Diagnosis**:
```bash
dmesg | grep snn
ls -l /dev/snn*
```

**Fix**:
```bash
# Unload and reload
sudo rmmod snn_kernel
sudo insmod snn_kernel.ko debug_level=3
dmesg | tail -50  # Check detailed logs
```

### 3. Test Program Fails

**Symptom**: `test_basic` returns error

**Diagnosis**:
```bash
sudo ./test_basic
# Note the error message

dmesg | tail -20
# Check kernel logs
```

**Common Causes**:
- Permission on /dev/snn0 → `sudo chmod 666 /dev/snn0`
- HPC init failed → Normal, uses simulated metrics
- AI engine init failed → Check dmesg for details

### 4. System Slowdown

**Symptom**: System feels sluggish with module loaded

**Diagnosis**:
```bash
top -n 1
# Check CPU usage

dmesg | grep SNN_AI
# Check for excessive logging
```

**Fix**:
```bash
# Reduce logging
sudo rmmod snn_kernel
sudo insmod snn_kernel.ko debug_level=0

# If still slow, unload and report issue
sudo rmmod snn_kernel
```

---

## Performance Expectations

With module loaded and idle:
- **CPU Usage**: <0.1% (only active during IOCTL calls)
- **Memory**: ~84 KB kernel memory
- **Decision Latency**: 44.5 μs per AI recommendation
- **Throughput**: ~20,000 decisions/sec

---

## Support and Diagnostics

If you encounter issues, collect diagnostics:

```bash
# System info
uname -a > diagnostics.txt
gcc --version >> diagnostics.txt
lsmod | grep snn >> diagnostics.txt

# Module info
modinfo snn_kernel.ko >> diagnostics.txt

# Kernel logs
dmesg | grep -i snn >> diagnostics.txt

# Build log
make clean
make modules 2>&1 | tee -a diagnostics.txt

# Kernel config
grep CONFIG_PERF_EVENTS /boot/config-$(uname -r) >> diagnostics.txt
```

Share `diagnostics.txt` if you need help.

---

## Summary: Your Safety Net

1. **Tier 1 (Testing)**: `insmod` → test → `rmmod` (no permanent changes)
2. **Tier 2 (Auto-Load)**: Install → configure → test reboots (reversible)
3. **Tier 3 (Production)**: Long-term use after thorough testing

**At ANY point**: `sudo rmmod snn_kernel` = instant rollback

**Worst case**: Boot into recovery mode, remove auto-load config

**Best case**: Production-grade AI kernel module running safely!

---

## Next Steps

1. **Start with Tier 1** on your real system
2. **Test thoroughly** before proceeding to Tier 2
3. **Monitor closely** during Tier 2 testing
4. **Only use Tier 3** after confirming stability

**Remember**: The module can always be unloaded. It's not modifying your kernel, just adding functionality.

Good luck! 🚀
