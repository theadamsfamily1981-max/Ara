# K10 Hashboard Recovery - Quick Start Checklist

**Goal**: Boot isolated K10/P2 hashboard (Stratix 10 SoC) without control board and gain root access.

**Time**: 2-8 hours (mostly waiting for downloads/builds)

---

## Pre-Flight Checklist

### Hardware Required

- [ ] K10/P2 hashboard with **visible SD card slot**
- [ ] MicroSD card (8GB+, Class 10)
- [ ] USB-to-TTL adapter (**3.3V**, FTDI FT232RL recommended)
- [ ] 12V DC power supply (5-15A, current-limited bench supply ideal)
- [ ] Jumper wires (3 minimum: GND, TX, RX)
- [ ] Multimeter (for pin identification)
- [ ] Linux workstation (for SD card flashing)

### Software Required

- [ ] Stratix 10 GHRD SD card image (download in Step 1)
- [ ] Serial terminal software (minicom, screen, PuTTY)
- [ ] SD card flashing tool (dd, Rufus, BalenaEtcher)

---

## Step 1: Obtain Boot Image (1-6 hours)

### Option A: Download Pre-Built Image ⚡ FASTEST

**Source**: Intel RocketBoards

**URL**: https://rocketboards.org/foswiki/Documentation/Stratix10SoCGSRD

**File**: `s10_gsrd_sdimage_<version>.wic.gz` (2-4 GB)

**Alternative**: GitHub - https://github.com/altera-opensource/meta-intel-fpga-refdes
- Branch: `socfpga-v22.3` or later
- Look in: `releases/` or `prebuilt/` directories

**Fallback**: Ask in Intel FPGA forums or FPGA Discord/Telegram for mirror links

---

### Option B: Build from Source (Advanced, 4-6 hours)

**Requirements**: Ubuntu 20.04/22.04, 50GB free space

```bash
# Install dependencies
sudo apt-get install gawk wget git diffstat unzip texinfo gcc build-essential chrpath socat cpio python3 python3-pip python3-pexpect xz-utils debianutils iputils-ping python3-git python3-jinja2 libegl1-mesa libsdl1.2-dev pylint3 xterm

# Clone Intel SoC FPGA meta layer
git clone https://github.com/altera-opensource/meta-intel-fpga-refdes.git
cd meta-intel-fpga-refdes
git checkout socfpga-v22.3

# Initialize build environment
source poky/oe-init-build-env build-s10

# Configure for Stratix 10
echo 'MACHINE = "stratix10"' >> conf/local.conf

# Build (2-6 hours)
bitbake gsrd-console-image

# Result: tmp/deploy/images/stratix10/gsrd-console-image-stratix10.wic
```

---

## Step 2: Flash SD Card (10 minutes)

### Linux

```bash
# Insert SD card and identify device
lsblk
# Example: /dev/sdb or /dev/mmcblk0

# Unmount any auto-mounted partitions
sudo umount /dev/sdb*

# Decompress (if .gz)
gunzip s10_gsrd_sdimage_22.3.wic.gz

# Flash
sudo dd if=s10_gsrd_sdimage_22.3.wic of=/dev/sdb bs=4M status=progress conv=fsync

# Verify
sudo sync
lsblk /dev/sdb
# Should show 3 partitions:
# sdb1 (Type 0xA2 - Preloader)
# sdb2 (FAT32 - Boot)
# sdb3 (ext4 - Root)
```

### Windows

1. Decompress `.wic.gz` using 7-Zip
2. Use **BalenaEtcher** or **Rufus**
3. Select `.wic` file
4. Select SD card
5. Flash

---

## Step 3: Identify UART Pins (15-30 minutes)

### Visual Inspection

**Look for**:
- 3-4 pin header near Stratix 10 SoC (large BGA chip)
- Labels: J1, J2, UART, CONSOLE, DEBUG, HPS_UART
- Pin 1 marker: Square pad, triangle (▼), or dot (●)

**Standard Pinout** (Intel Reference):
```
Pin 1: GND
Pin 2: RX  (HPS receives - connect to your adapter's TX)
Pin 3: TX  (HPS transmits - connect to your adapter's RX)
Pin 4: VCC (do not connect)
```

---

### Multimeter Test (Powered OFF)

```bash
# Set multimeter to continuity mode (beep)

# Find GND reference (power connector negative pin)

# Test each UART pin for continuity to GND
# Pin with continuity = GND
# Other pins = TX/RX/VCC
```

---

### Voltage Test (Powered ON)

```bash
# Set multimeter to DC voltage

# Measure each pin relative to GND:
# GND: 0.0V
# TX:  1.8-3.3V (idle high)
# RX:  1.8-3.3V (idle high)
# VCC: 1.8V or 3.3V
```

---

## Step 4: Connect UART (5 minutes)

### Wiring

```
USB-TTL Adapter    →    K10 Hashboard
────────────────────────────────────
GND (Black)        →    Pin 1 (GND)
TX  (Green/Yellow) →    Pin 2 (RX)   ← Crossed!
RX  (White)        →    Pin 3 (TX)   ← Crossed!
VCC (Red)          →    NOT CONNECTED
```

**Critical**: TX and RX are **crossed** (adapter TX → board RX).

**Do NOT connect VCC** - board powers itself.

---

### Serial Terminal Setup

**Linux**:
```bash
# Install minicom
sudo apt-get install minicom

# Connect
sudo minicom -D /dev/ttyUSB0 -b 115200

# Settings: 115200 baud, 8N1, no flow control
# Exit: Ctrl+A then X
```

**Windows**:
- Use **PuTTY**
- Port: COM3 (check Device Manager)
- Speed: 115200
- Connection type: Serial
- Flow control: None

**macOS**:
```bash
ls /dev/tty.usb*
screen /dev/tty.usbserial-XXXXX 115200
```

---

## Step 5: Power On and Boot (2 minutes)

### Procedure

1. **Open serial terminal first** (Step 4)
2. **Insert SD card** into hashboard
3. **Connect 12V power** (start at low current limit if using bench supply)
4. **Watch serial console** for output

---

### Expected Output (Within 5 Seconds)

```
U-Boot SPL 2022.10 (Jan 01 2023 - 12:00:00 +0000)
SDRAM: Initializing DDR4 Controller
SDRAM: 4096 MiB
Trying to boot from MMC1
Loading Environment from MMC...

U-Boot 2022.10 (Jan 01 2023 - 12:00:00 +0000)

CPU:   Intel FPGA SoCFPGA Platform
Model: SoCFPGA Stratix 10 SoCDK
DRAM:  4 GiB

Hit any key to stop autoboot:  3
```

**If you see this**: ✅ **SUCCESS!** Hardware is working.

---

### Troubleshooting: No Output

- [ ] Check baud rate (try 9600, 57600, 115200)
- [ ] Swap TX and RX connections
- [ ] Verify GND connection
- [ ] Check power (measure 12V at connector)
- [ ] Try different UART header (board may have multiple)
- [ ] Test adapter with loopback (short TX to RX, type in terminal)

---

## Step 6: Interrupt U-Boot (30 seconds)

### Action

**When you see**:
```
Hit any key to stop autoboot:  3
Hit any key to stop autoboot:  2
Hit any key to stop autoboot:  1
```

**Press Enter or Spacebar** repeatedly (spam it!)

**Result**:
```
SOCFPGA #
```

✅ **You're at the U-Boot command prompt!**

---

## Step 7: Inject Root Shell (2 minutes)

### Method 1: Init=/bin/sh (Simplest)

```bash
# At U-Boot prompt (SOCFPGA #):

# Save current bootargs
setenv bootargs_orig "${bootargs}"

# Add init=/bin/sh
setenv bootargs "${bootargs} init=/bin/sh"

# Boot
boot
```

**Result**: Kernel boots directly to root shell:
```
# (root shell, no password needed)
```

---

### Method 2: Single-User Mode (Alternative)

```bash
# At U-Boot prompt:
setenv bootargs "${bootargs} single"
boot
```

**Result**: Boots to maintenance mode with root shell.

---

## Step 8: Set Root Password (5 minutes)

### Commands

```bash
# Remount root filesystem as read-write
mount -o remount,rw /

# Verify write access
touch /test && rm /test && echo "Write OK"

# Set root password
passwd
# Enter new password twice

# Enable SSH root login
vi /etc/ssh/sshd_config
# Find line: #PermitRootLogin
# Change to: PermitRootLogin yes
# Save and exit (:wq)

# Sync filesystem
sync

# Reboot
reboot -f
```

---

## Step 9: Verify Normal Boot (2 minutes)

### After Reboot

**Watch serial console**:
```
Linux stratix10 5.10.0 #1 SMP PREEMPT aarch64 GNU/Linux

stratix10 login:
```

**Log in**:
```
login: root
password: <your_password>

# Welcome message...

root@stratix10:~#
```

✅ **SUCCESS! You have root access.**

---

## Step 10: Explore System (10 minutes)

### Basic Commands

```bash
# Check system info
uname -a
cat /etc/os-release

# Check FPGA
cat /sys/class/fpga_manager/fpga0/state
# Should show: operating

# Check memory
free -h
# Should show: ~4GB

# Check storage
df -h
# Shows SD card partitions

# Check temperature
cat /sys/class/thermal/thermal_zone0/temp
# Result in millidegrees (e.g., 45000 = 45°C)

# List available bitstreams
ls -lh /lib/firmware/*.rbf
```

---

## Step 11: Load Custom Bitstream (Optional)

### Procedure

```bash
# Copy .rbf file to /lib/firmware
# (via SD card mount on workstation, then reinsert)

# Check current FPGA state
cat /sys/class/fpga_manager/fpga0/state

# Load bitstream
echo "custom_algo.rbf" > /sys/class/fpga_manager/fpga0/firmware

# Monitor state change
watch -n 1 'cat /sys/class/fpga_manager/fpga0/state'
# Should transition: unknown → programming → operating

# Verify
cat /sys/class/fpga_manager/fpga0/state
# Should show: operating
```

---

## Common Issues and Solutions

### Issue: Garbled UART Output

**Symptom**: `����������������`

**Fix**:
- Try baud rate 115200 (most common)
- Check GND connection
- Verify adapter voltage (must be 3.3V)

---

### Issue: Board Doesn't Power On

**Symptom**: No LEDs, no UART output

**Fix**:
- Check 12V power at connector
- Increase current limit on bench supply
- Look for "power enable" pin on connector (may need bridging)

---

### Issue: SD Card Not Detected

**Symptom**: U-Boot error "MMC init failed"

**Fix**:
- Re-flash SD card (may be corrupted)
- Try different SD card
- Check SD slot for damage
- Board may boot from QSPI flash instead (check jumpers)

---

### Issue: Kernel Panic

**Symptom**: "VFS: Unable to mount root fs"

**Fix**:
```bash
# At U-Boot prompt:
setenv bootargs "console=ttyS0,115200 root=/dev/mmcblk0p3 rw rootwait"
saveenv
boot
```

---

### Issue: FPGA Won't Configure

**Symptom**: Stuck in "programming" state

**Fix**:
```bash
# Check dmesg for errors
dmesg | grep fpga

# Verify bitstream is correct device family
# Stratix 10 SX (not GX or other)

# Try known-good bitstream (from GHRD)
echo "ghrd_10as066n2.rbf" > /sys/class/fpga_manager/fpga0/firmware
```

---

## Success Criteria

You have successfully recovered the hashboard when you can:

- [ ] Boot Linux from SD card
- [ ] See U-Boot prompt on UART
- [ ] Log in as root with your password
- [ ] Access FPGA manager interface
- [ ] Load custom bitstreams
- [ ] Monitor system temperature and status

---

## Next Steps

### Development Platform

**Use Case**: General-purpose FPGA development

**Capabilities**:
- ARM Cortex-A53 Linux system
- Direct FPGA fabric control
- Custom bitstream deployment
- Heterogeneous ARM+FPGA computing

**Applications**:
- DSP algorithms
- ML inference acceleration
- Custom hardware protocols
- Parallel computing research

---

### Mining Restoration (Advanced)

**Requirements**:
- Original K10 firmware filesystem (extracted from working miner)
- USB-Ethernet adapter (network on control board)
- Mining pool configuration

**Steps**:
1. Extract `/usr/bin/miner` from working K10
2. Copy to your recovered board
3. Install network adapter drivers
4. Configure pool: `./miner --pool=... --wallet=...`

---

### Bitstream Reverse Engineering

**Use Day 1 Toolkit**:

```bash
# Analyze bitstreams
python k10-forensics/scripts/analyze_k10_sd.py --scan-dir /lib/firmware

# Preprocess for ML analysis
python a10_ml/scripts/a10_build_dataset.py --input kaspa.rbf --output analysis/
```

---

## Safety Reminders

### Electrical

- ⚠️ Use current-limited power supply (start at 2A, increase gradually)
- ⚠️ Monitor temperature (stay below 70°C)
- ⚠️ Never connect 5V to 3.3V UART pins
- ⚠️ Add heatsinks/fans if chip gets hot

### Legal

- ✅ Only modify hardware you own
- ✅ Educational and research use
- ✅ Authorized security testing
- ❌ Don't violate terms of service
- ❌ Don't circumvent DRM for unauthorized access

---

## Resources

### Documentation

- **Full Guide**: `HASHBOARD_RECOVERY_GUIDE.md` (detailed procedures)
- **UART Pinout**: `UART_PINOUT_REFERENCE.md` (visual diagrams)
- **Day 1 Toolkit**: `README.md` (complete jailbreak toolkit)

### Intel Official

- **RocketBoards**: https://rocketboards.org
- **Intel SoC FPGA**: https://www.intel.com/content/www/us/en/products/details/fpga.html
- **Quartus Download**: https://www.intel.com/content/www/us/en/software-kit/782411/intel-quartus-prime-standard-edition-design-software-version-22-1-2-for-linux.html

### Community

- **FPGA Discord**: Zetheron, FPGA Dev servers
- **Forums**: RocketBoards Forum, EEVblog
- **GitHub**: Intel altera-opensource repositories

---

## Timeline Summary

| Step | Task | Time |
|------|------|------|
| 1 | Download GHRD image | 30min - 2hrs |
| 2 | Flash SD card | 10 min |
| 3 | Identify UART pins | 15-30 min |
| 4 | Connect UART | 5 min |
| 5 | Power on and boot | 2 min |
| 6 | Interrupt U-Boot | 30 sec |
| 7 | Inject root shell | 2 min |
| 8 | Set root password | 5 min |
| 9 | Verify normal boot | 2 min |
| 10 | Explore system | 10 min |
| **Total** | **End-to-End Recovery** | **2-4 hours** |

---

## Quick Reference Commands

### U-Boot

```bash
printenv              # Show all variables
setenv var value      # Set variable
saveenv               # Save to persistent storage
boot                  # Boot kernel
```

### Linux

```bash
# FPGA
cat /sys/class/fpga_manager/fpga0/state
echo "file.rbf" > /sys/class/fpga_manager/fpga0/firmware

# System
uname -a
free -h
df -h
dmesg
top

# Temperature
cat /sys/class/thermal/thermal_zone0/temp
```

---

**Good luck with your recovery! This should get you from blank SD card to root shell in 2-4 hours.** 🚀
