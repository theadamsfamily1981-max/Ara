# K10/P2 Hashboard-Only Recovery Guide

**Situation**: Isolated FPGA hashboard (Stratix 10 SoC) without control board, no network interface, blank SD cards.

**Goal**: Boot the Stratix 10 HPS, establish UART console access, gain root shell.

---

## Hardware Architecture Analysis

### Stratix 10 SoC Components

The K10/P2 hashboard contains:

**Intel Stratix 10 SX SoC** (1SX280LU3F50E2VG or similar):
- **HPS (Hard Processor System)**: Quad-core ARM Cortex-A53 @ 1.5 GHz
- **SDM (Secure Device Manager)**: Boot orchestrator and configuration manager
- **FPGA Fabric**: 2.8M logic elements (for mining algorithms)
- **DDR4 Controller**: External memory interface (on hashboard)

**Critical Boot Components**:
- SD Card Slot (if present - K10 revision likely has it)
- QSPI Flash (backup boot source)
- UART Console Header (3-4 pins: TX, RX, GND, VCC)
- Power Regulators (12V → 1.8V, 1.2V, 0.9V rails)

---

## Part 1: Hardware Verification

### Step 1.1: Identify Your Board Revision

**K10 Original** (Standalone Capable):
- SD card slot **visible on hashboard**
- Separate DDR4 memory chips on hashboard
- UART header near ARM processor
- Power connector: 6-pin or 8-pin

**P2 Revised** (Control Board Dependent):
- SD card slot on **control board only**
- Integrated memory module
- UART may be inaccessible
- Different power connector

**Action**: Visually inspect your hashboard for an SD card slot (micro or full-size).

---

### Step 1.2: Locate UART Console Pins

The UART console is **essential** for headless operation. It's typically a 3-4 pin header.

**Intel Stratix 10 Standard UART Pinout**:
```
Header J-XX (usually near HPS chip):

Pin 1: GND    (Ground - connect to USB-TTL GND)
Pin 2: RX     (HPS receives - connect to USB-TTL TX)
Pin 3: TX     (HPS transmits - connect to USB-TTL RX)
Pin 4: VCC    (3.3V - DO NOT CONNECT)
```

**Identification Methods**:

**Method 1: Visual Inspection**
- Look for a 4-pin header near the main SoC chip
- May be labeled: "J1", "UART", "DEBUG", "CONSOLE"
- Pins may be unpopulated (solder pins yourself)

**Method 2: Multimeter Continuity Test** (with board powered off):
- Find GND reference (usually large copper plane or power connector ground pin)
- Test continuity between header pins and GND
- Pin with continuity to GND = GND pin

**Method 3: Voltage Test** (with board powered on):
- Measure voltage between each pin and GND
- GND pin: 0V
- VCC pin: 3.3V or 1.8V
- TX/RX pins: ~1.5-3.3V (idle high state)

**Common UART Header Locations**:
- Near the Stratix 10 SoC (large BGA chip)
- Near Ethernet PHY chip (if present)
- Edge of board marked "DEBUG"

**If you can't find UART**: Send high-resolution PCB photos - I can help identify it.

---

### Step 1.3: Power Requirements

**Typical Hashboard Power**:
- Input: 12V DC (6-pin PCIe or 8-pin EPS connector)
- Current: 5-15A (depends on FPGA activity)
- Rails: 12V → 5V → 3.3V → 1.8V → 1.2V → 0.9V (internal regulators)

**Standalone Considerations**:

**Issue**: Control board may provide "power enable" signal
**Solution**: Bridge PS_ON or ENABLE pins on power connector

**Power Connector Pinout** (typical 6-pin PCIe):
```
Pin 1, 2, 3: +12V
Pin 4, 5, 6: GND
```

**If board doesn't power on**:
- Check for "EN" or "PWR_ON" pin
- Bridge to GND or 12V (check schematic/similar boards)
- Look for status LEDs (power good indicator)

**Safety**: Use bench power supply with **current limiting** (start at 2A, increase gradually).

---

## Part 2: Bootable SD Card Creation

### Step 2.1: Obtain Stratix 10 GHRD Image

Since proprietary K10 firmware is unavailable, use Intel's generic **Golden Hardware Reference Design (GHRD)**.

**Option A: Pre-built Intel Images**

**Source**: Intel RocketBoards - Stratix 10 SoC GSRD (Golden System Reference Design)

**Download Links** (as of 2024):

1. **Intel Developer Zone** (requires free account):
   ```
   https://www.intel.com/content/www/us/en/developer/topic-technology/edge-5g/tools/devcloud.html
   ```
   Search: "Stratix 10 SoC GSRD"

2. **RocketBoards Forum** (community mirrors):
   ```
   https://forum.rocketboards.org/
   ```
   Look for: "s10_gsrd_<version>.tar.gz"

3. **GitHub - Intel SoC FPGA**:
   ```
   https://github.com/altera-opensource/meta-intel-fpga-refdes
   ```
   Branch: `socfpga-v21.4` or later
   Contains: Yocto build recipes for Stratix 10

**What You Need**:
- **sdimage.tar.gz** or **ghrd.sdimage** (bootable SD card image)
- Size: 2-8 GB
- Format: Raw disk image (.img, .wic, .sdcard)

**Typical Filename**: `s10_gsrd_sdimage_21.4.wic.gz`

---

**Option B: Build GHRD from Source** (Advanced)

If pre-built images don't work, build custom image using Intel SoC EDS:

**Requirements**:
- Linux workstation (Ubuntu 20.04/22.04)
- 50+ GB free space
- Intel SoC FPGA Embedded Development Suite (SoC EDS)
- Yocto build tools

**Quick Build Steps**:
```bash
# Install prerequisites
sudo apt-get install gawk wget git diffstat unzip texinfo gcc build-essential chrpath socat cpio python3 python3-pip python3-pexpect xz-utils debianutils iputils-ping python3-git python3-jinja2 libegl1-mesa libsdl1.2-dev pylint3 xterm

# Clone Intel meta layer
git clone https://github.com/altera-opensource/meta-intel-fpga-refdes.git
cd meta-intel-fpga-refdes
git checkout socfpga-v21.4

# Initialize build environment
source poky/oe-init-build-env build-s10

# Edit conf/local.conf
echo 'MACHINE = "stratix10"' >> conf/local.conf

# Build SD card image
bitbake gsrd-console-image

# Result: tmp/deploy/images/stratix10/gsrd-console-image-stratix10.wic
```

**Build Time**: 2-6 hours (first build)

---

### Step 2.2: Flash SD Card

**Tools**:
- **Linux**: `dd` command
- **Windows**: Rufus, Win32DiskImager, or BalenaEtcher
- **macOS**: `dd` command or BalenaEtcher

**Procedure (Linux)**:

```bash
# 1. Insert SD card, identify device
lsblk
# Look for your SD card (e.g., /dev/sdb, /dev/mmcblk0)

# 2. Unmount any auto-mounted partitions
sudo umount /dev/sdb*

# 3. Decompress image (if .gz)
gunzip s10_gsrd_sdimage_21.4.wic.gz

# 4. Write image to SD card
sudo dd if=s10_gsrd_sdimage_21.4.wic of=/dev/sdb bs=4M status=progress conv=fsync

# 5. Sync and eject
sudo sync
sudo eject /dev/sdb
```

**Verification**:
```bash
# Re-insert card and check partitions
lsblk /dev/sdb

# Should see:
# sdb
# ├─sdb1  (Type: 0xA2 - Preloader)
# ├─sdb2  (FAT32 - Boot files)
# └─sdb3  (ext4 - Root filesystem)
```

---

## Part 3: Hardware Connection and Power-On

### Step 3.1: UART Connection

**Hardware**:
- USB-to-TTL adapter (FTDI FT232RL, CP2102, CH340)
- **Voltage**: 3.3V logic (NOT 5V - will damage Stratix 10)
- Jumper wires

**Wiring**:
```
USB-TTL Adapter   →   K10 Hashboard UART Header
─────────────────────────────────────────────
GND (Black)       →   Pin 1 (GND)
TX  (Yellow)      →   Pin 2 (RX - HPS receives)
RX  (White)       →   Pin 3 (TX - HPS transmits)
VCC (Red)         →   DO NOT CONNECT
```

**Important**: TX and RX are **crossed** (adapter TX → board RX, adapter RX → board TX).

---

### Step 3.2: Serial Terminal Setup

**Linux**:
```bash
# Install minicom or screen
sudo apt-get install minicom

# Connect to USB-TTL adapter
ls /dev/ttyUSB*
# Should see: /dev/ttyUSB0

# Open serial console
sudo minicom -D /dev/ttyUSB0 -b 115200

# Settings: 115200 baud, 8N1 (8 data bits, No parity, 1 stop bit)
# Ctrl+A Z for help, Ctrl+A X to exit
```

**Minicom Configuration**:
```
sudo minicom -s
→ Serial port setup
→ A - Serial Device: /dev/ttyUSB0
→ E - Bps/Par/Bits: 115200 8N1
→ F - Hardware Flow Control: No
→ G - Software Flow Control: No
→ Save setup as dfl (default)
→ Exit
```

**Alternative (Screen)**:
```bash
sudo screen /dev/ttyUSB0 115200
# Ctrl+A K to exit
```

**Windows**:
- Use **PuTTY** or **Tera Term**
- Port: COM3 (check Device Manager)
- Baud: 115200, 8N1
- Flow Control: None

**macOS**:
```bash
ls /dev/tty.usb*
screen /dev/tty.usbserial-XXXXX 115200
```

---

### Step 3.3: Power-On Sequence

**Procedure**:

1. **Connect UART first** (with serial terminal open)
2. **Insert SD card** into hashboard slot
3. **Connect 12V power supply**
4. **Watch serial console** for boot messages

**Expected Output** (within 5 seconds):

```
U-Boot SPL 2021.10-00001-g12345678 (Jan 01 2022 - 12:00:00 +0000)
SDRAM: Initializing DDR4 Controller
SDRAM: 4096 MiB
Trying to boot from MMC1
Loading Environment from MMC...
```

**If you see this**: ✅ Hardware is functioning correctly!

**If you see nothing**:
- Check UART wiring (swap TX/RX if needed)
- Verify baud rate (try 9600, 57600, 115200)
- Check power supply (measure 12V at connector)
- Test with multimeter: measure voltage on TX pin (should toggle)

---

## Part 4: U-Boot Jailbreak (Root Access)

### Step 4.1: Interrupt Autoboot

**Watch for this message**:
```
Hit any key to stop autoboot:  3
Hit any key to stop autoboot:  2
Hit any key to stop autoboot:  1
```

**Action**: Press **Enter** or **Spacebar** repeatedly.

**Result**: U-Boot command prompt:
```
SOCFPGA #
```

---

### Step 4.2: Inspect Environment

**Check current boot configuration**:
```
SOCFPGA # printenv

bootargs=console=ttyS0,115200 root=/dev/mmcblk0p3 rw rootwait
bootcmd=run mmcload; run mmcboot
mmcload=load mmc 0:2 ${loadaddr} ${bootfile}
mmcboot=setenv bootargs ${bootargs}; bootz ${loadaddr} - ${fdtaddr}
```

---

### Step 4.3: Inject Root Shell Parameter

**Method 1: Single-User Mode** (Simplest)
```
SOCFPGA # setenv bootargs_orig "${bootargs}"
SOCFPGA # setenv bootargs "${bootargs} single"
SOCFPGA # boot
```

**Result**: Boots to root shell without password.

---

**Method 2: Init=/bin/sh** (Direct Root Shell)
```
SOCFPGA # setenv bootargs "${bootargs} init=/bin/sh"
SOCFPGA # boot
```

**Result**: Kernel drops directly to `sh` shell:
```
# (root shell prompt)
```

---

**Method 3: Disable SELinux/Security** (If above fail)
```
SOCFPGA # setenv bootargs "${bootargs} selinux=0 enforcing=0 init=/bin/bash"
SOCFPGA # boot
```

---

### Step 4.4: Remount Root Filesystem

**Issue**: Root filesystem is mounted read-only initially.

**Fix**:
```bash
# Check mount status
mount | grep mmcblk0p3

# Remount as read-write
mount -o remount,rw /

# Verify
touch /test_write && echo "Write successful" && rm /test_write
```

---

### Step 4.5: Set Root Password

```bash
# Set new root password
passwd

# Enter new password twice
New password: <your_password>
Retype new password: <your_password>

# Verify shadow file
cat /etc/shadow | grep root
```

---

### Step 4.6: Enable SSH Root Login

```bash
# Edit SSH configuration
vi /etc/ssh/sshd_config

# Find and uncomment/add:
PermitRootLogin yes
PasswordAuthentication yes

# Save and exit (:wq in vi)
```

---

### Step 4.7: Reboot to Normal Linux

```bash
# Sync filesystem
sync

# Reboot
reboot -f
```

**Result**: System boots normally, you can now log in as root with your new password (via UART).

---

## Part 5: Post-Boot Configuration

### Step 5.1: Verify System

**Check Linux version**:
```bash
uname -a
# Linux stratix10 5.10.0 #1 SMP PREEMPT arm64 GNU/Linux

cat /etc/os-release
# NAME="Poky (Yocto Project Reference Distro)"
```

**Check FPGA status**:
```bash
cat /sys/class/fpga_manager/fpga0/state
# operating / programming / unknown
```

**Check memory**:
```bash
free -h
# Should show 4GB DDR4
```

**Check storage**:
```bash
df -h
# Should show SD card partitions
```

---

### Step 5.2: Network Configuration (If Available)

**If hashboard has Ethernet PHY** (unlikely without control board):

```bash
# Check network interfaces
ip link show

# If eth0 exists:
ip addr add 192.168.1.100/24 dev eth0
ip link set eth0 up

# Test connectivity
ping 192.168.1.1
```

**More likely**: Network is on control board (unavailable).

---

### Step 5.3: Install Essential Tools

**If internet is available**:
```bash
# Update package manager (if opkg/apt)
opkg update
opkg install python3 vim htop

# Or build from source
```

**Without internet**: Transfer tools via SD card:
1. Mount SD card on workstation
2. Copy binaries to `/root/` partition
3. Make executable: `chmod +x tool_name`

---

## Part 6: FPGA Bitstream Loading

### Step 6.1: Verify FPGA Manager

```bash
# Check FPGA manager sysfs
ls /sys/class/fpga_manager/fpga0/
# firmware  name  state

# Check current state
cat /sys/class/fpga_manager/fpga0/state
# operating / unknown
```

---

### Step 6.2: Load Custom Bitstream

**Copy .rbf file to /lib/firmware**:
```bash
# Via SD card (mount on workstation)
cp custom_algo.rbf /mnt/sd_root/lib/firmware/

# Or via serial transfer (slow)
cat > /lib/firmware/custom.rbf < /dev/ttyS0
# (paste base64-encoded file, decode on device)
```

**Load bitstream**:
```bash
echo "custom_algo.rbf" > /sys/class/fpga_manager/fpga0/firmware

# Check state
cat /sys/class/fpga_manager/fpga0/state
# Should change: unknown → programming → operating
```

---

## Part 7: Troubleshooting

### Issue: No Boot Messages on UART

**Checks**:
1. **Baud rate mismatch**: Try 9600, 19200, 57600, 115200
2. **TX/RX swapped**: Reverse TX and RX connections
3. **Wrong UART**: Board may have multiple UARTs (try different headers)
4. **Voltage level**: Ensure 3.3V adapter (not 5V)
5. **Ground not connected**: GND must be common between adapter and board

**Test**:
```bash
# Send break signal in minicom
Ctrl+A F
# Should see response if UART is working
```

---

### Issue: SD Card Not Detected

**Checks**:
1. **Partition table**: Verify with `fdisk -l` on workstation
2. **Image corruption**: Re-flash SD card
3. **SD slot damage**: Try different card
4. **Boot source**: Board may be set to boot from QSPI flash (check DIP switches/jumpers)

**Alternative Boot**:
- Boot from QSPI flash (if programmed)
- JTAG programming (requires JTAG adapter)

---

### Issue: Kernel Panic / No Root Filesystem

**Error**:
```
Kernel panic - not syncing: VFS: Unable to mount root fs on unknown-block(0,0)
```

**Cause**: Root filesystem path incorrect.

**Fix in U-Boot**:
```
SOCFPGA # setenv bootargs "console=ttyS0,115200 root=/dev/mmcblk0p3 rw rootwait"
SOCFPGA # saveenv
SOCFPGA # boot
```

**Check partition numbering**:
- `/dev/mmcblk0p3` (eMMC/SD with 'p' suffix)
- `/dev/sda3` (USB/SATA)

---

### Issue: FPGA Won't Configure

**Error**:
```
cat /sys/class/fpga_manager/fpga0/state
programming
(stuck in programming state)
```

**Causes**:
1. Bitstream incompatible with device (wrong FPGA model)
2. Bitstream corrupted
3. FPGA voltage rails not powered

**Fix**:
```bash
# Check dmesg for errors
dmesg | grep fpga

# Try known-good bitstream
echo "ghrd_10as066n2.rbf" > /sys/class/fpga_manager/fpga0/firmware
```

---

## Part 8: Next Steps - Full Exploitation

### Option A: Standalone FPGA Development

**Use Case**: Convert mining hashboard into general-purpose FPGA dev board.

**Benefits**:
- Full Linux OS with root access
- Direct FPGA fabric control
- Custom bitstream loading
- ARM + FPGA heterogeneous computing

**Applications**:
- DSP algorithms
- Machine learning inference
- Custom protocols (SDR, etc.)
- Parallel computing research

---

### Option B: Mining Firmware Restoration

**Goal**: Restore original mining functionality without control board.

**Challenges**:
- Need original K10 firmware filesystem
- Network interface likely on control board (need USB-Ethernet adapter)
- Power management may require custom scripts

**Approach**:
1. Extract firmware from working K10 (if available)
2. Copy `/usr/bin/miner` and configs to your board
3. Install USB-Ethernet adapter driver
4. Configure mining pools via CLI
5. Run miner manually: `/usr/bin/miner --pool=... --wallet=...`

---

### Option C: Reverse Engineering K10 Algorithms

**Goal**: Extract and analyze proprietary mining bitstreams.

**Process**:
1. Locate `.rbf` files in original firmware
2. Use **a10_ml toolkit** (from earlier) to preprocess bitstreams
3. Convert to 2D images for visual analysis
4. Use **k10-forensics** tools to identify bitstream structure

**Tools** (from Day 1 Toolkit):
```bash
# Analyze bitstream
python k10-forensics/scripts/analyze_k10_sd.py --scan-dir /lib/firmware

# Preprocess for ML
python a10_ml/scripts/a10_build_dataset.py --input kaspa.rbf --output analysis/
```

---

## Part 9: Hardware Limitations

### What You CAN'T Do (Without Control Board)

1. **Network Mining**: Ethernet interface likely on control board
   - **Workaround**: USB-Ethernet adapter (if USB port exists)

2. **Web UI Access**: Management interface hosted on control board
   - **Workaround**: Build custom web UI on hashboard Linux

3. **Multi-Hashboard Operation**: Control board orchestrates multiple hashboards
   - **Workaround**: Run single-board standalone

4. **Firmware Updates**: Update mechanism expects control board
   - **Workaround**: Manual SD card flashing

5. **Power Efficiency**: Control board manages DVFS (dynamic voltage/frequency scaling)
   - **Workaround**: Script manual voltage control via sysfs

---

## Part 10: Safety and Legal Considerations

### Electrical Safety

**Risks**:
- Short circuits (12V @ 15A = 180W of potential heat)
- Incorrect voltage on UART (3.3V vs 5V)
- Missing thermal management (hashboard may overheat without fans)

**Mitigations**:
- Use **current-limited** bench supply
- Monitor temperature: `cat /sys/class/thermal/thermal_zone0/temp`
- Add fan if chip exceeds 70°C
- Never connect 5V to 3.3V logic pins

---

### Legal Considerations

**Warranty**: Physical modifications void manufacturer warranty.

**Ownership**: Only modify hardware you legally own.

**Intended Use**: This guide is for:
- ✅ Educational research
- ✅ Hardware recovery and reuse
- ✅ FPGA development learning
- ✅ Authorized security research

**Not for**:
- ❌ Theft or unauthorized access
- ❌ Circumventing mining pool restrictions
- ❌ Violating terms of service

---

## Appendix A: Stratix 10 GHRD Image Sources

### Official Intel Sources

1. **Intel FPGA Developer Center**
   - URL: https://www.intel.com/content/www/us/en/products/details/fpga/development-kits.html
   - Search: "Stratix 10 SoC Development Kit"
   - Downloads: GSRD pre-built images

2. **RocketBoards.org**
   - URL: https://rocketboards.org/foswiki/Documentation/Stratix10SoCGSRD
   - Files: `gsrd-console-image-stratix10.wic.gz`

3. **GitHub - Altera OpenSource**
   - URL: https://github.com/altera-opensource/meta-intel-fpga-refdes
   - Branch: `socfpga-v22.3`

---

### Community Mirrors

**Discord/Telegram**:
- Zetheron Discord (FPGA Mining community)
- FPGA development Telegram groups

**Ask for**: "Stratix 10 SoC bootable SD image" or "s10_gsrd_sdimage.wic"

---

## Appendix B: Suggested Hardware

### Essential

**USB-to-TTL Adapter** ($5-15):
- **Recommended**: FTDI FT232RL (genuine, not clone)
- **Alternative**: CP2102, CH340G
- **Critical**: 3.3V logic level

**Bench Power Supply** ($30-100):
- Adjustable 0-30V, 0-10A
- Current limiting feature
- LED display

**SD Card Reader** ($10-20):
- USB 3.0 for speed
- Supports MicroSD and SD

### Optional

**JTAG Adapter** ($50-200):
- Intel USB Blaster II
- Terasic USB Blaster
- For advanced FPGA programming

**Logic Analyzer** ($10-100):
- 8-channel USB logic analyzer
- For debugging UART signals

**Oscilloscope** ($100-500):
- For analyzing power rails
- Debugging analog issues

---

## Appendix C: Quick Reference Commands

### U-Boot Commands

```bash
# Print all environment variables
printenv

# Set variable
setenv var_name value

# Save to persistent storage
saveenv

# Boot kernel
boot

# Load file from SD card
load mmc 0:2 ${loadaddr} ${filename}

# Memory operations
md ${address}  # Memory display
mw ${address} ${value}  # Memory write
```

---

### Linux System Commands

```bash
# Check FPGA state
cat /sys/class/fpga_manager/fpga0/state

# Load bitstream
echo "algo.rbf" > /sys/class/fpga_manager/fpga0/firmware

# Monitor system
top
htop
dmesg
journalctl -f

# Check hardware
lsusb
lspci
cat /proc/cpuinfo
cat /proc/meminfo

# Temperature
cat /sys/class/thermal/thermal_zone*/temp
```

---

## Summary

**What This Guide Achieves**:

✅ Boot Stratix 10 SoC from generic Intel GHRD image
✅ Establish UART console access (headless operation)
✅ Gain root shell via U-Boot parameter injection
✅ Set persistent root password
✅ Load custom FPGA bitstreams
✅ Convert mining hashboard into general-purpose FPGA dev platform

**Key Success Factors**:

1. **Hardware Verification**: Confirm SD slot and UART pins on your specific board
2. **UART Connection**: Essential for headless access (no network interface)
3. **Generic Boot Image**: Use Intel GHRD (don't need K10-specific firmware)
4. **U-Boot Jailbreak**: Inject `init=/bin/sh` to bypass authentication
5. **Safety**: Current-limited power supply, proper voltage levels

**Timeline**:

- Image download/build: 1-6 hours
- Hardware setup: 30 minutes
- First successful boot: 5 minutes
- Root access: 10 minutes
- FPGA testing: 30 minutes

**Total**: ~2-8 hours for complete recovery

---

**Good luck with your hashboard recovery! If you encounter specific issues, I can provide targeted troubleshooting.**
