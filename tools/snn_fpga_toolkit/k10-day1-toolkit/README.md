# K10 Day 1 Toolkit

**Comprehensive operational toolkit for Superscalar K10 / ColEngine P2 FPGA miners.**

Based on the technical analysis of Stratix 10 SoC architecture, this toolkit provides complete Day 1 capabilities for jailbreaking, FPGA management, network automation, and safety monitoring.

---

## Features

### 🔓 Jailbreak Module (Three Methods)

**Method A: Network-Based Credential Exploitation** (Non-invasive)
- Automated SSH/Telnet/Web UI credential brute-force
- Default credential database from Intel GHRD
- Command injection via web diagnostics
- Success rate: ~60% (if default credentials unchanged)

**Method B: Offline Shadow File Modification** (Most Reliable)
- SD card mounting and filesystem modification
- Root password hash removal
- SSH configuration enablement
- Persistent backdoor installation
- Success rate: ~95% (requires physical access)

**Method C: U-Boot UART Injection** (Hardware Fallback)
- Serial console interception
- Boot parameter injection (`init=/bin/sh`)
- Interactive root shell access
- Success rate: ~80% (requires UART access)

### 🔧 FPGA Management Module

- **Bitstream Manager**: Sysfs-based FPGA manager interface
- **Algorithm Switcher**: Automated algorithm switching (Kaspa/Alephium/Radiant/Karlsen)
- **ISO Extractor**: Extract .rbf bitstreams from vendor ISOs

### 🌐 Network Module

- **Discovery Protocol**: UDP broadcast emulation for miner enumeration
- **Management API**: HTTP/JSON-RPC interface automation
- **Mass Configurator**: Batch configuration for large farms

### 🛡️ Safety Module

- **Watchdog Keeper**: Hardware watchdog management
- **Thermal Monitor**: Temperature monitoring and fan control
- **Power Validator**: Power consumption verification

---

## Installation

```bash
# Clone repository
git clone <repo-url>
cd k10-day1-toolkit

# Install Python dependencies
pip install -r requirements.txt

# Install system utilities (Ubuntu/Debian)
sudo apt-get install fdisk parted mount util-linux

# Optional: Serial console tools
sudo apt-get install minicom screen
```

---

## Quick Start

### Jailbreak a K10/P2 Miner

**Option 1: Automated (tries all methods)**
```bash
python -m jailbreak.auto_jailbreak --target 192.168.1.100
```

**Option 2: Network-based (non-invasive)**
```bash
python -m jailbreak.method_a_network 192.168.1.100
```

**Option 3: Offline shadow edit (most reliable)**
```bash
# 1. Power down miner, extract SD card
# 2. Insert into Linux workstation
# 3. Run:
sudo python -m jailbreak.method_b_shadow /dev/sdb

# 4. Reinsert SD card, boot miner
# 5. SSH with blank password:
ssh root@192.168.1.100
```

**Option 4: UART console (hardware method)**
```bash
# 1. Connect USB-to-TTL adapter (GND, TX→RX, RX→TX)
# 2. Run:
python -m jailbreak.method_c_uboot /dev/ttyUSB0

# 3. Power cycle miner
# 4. Follow interactive prompts
```

---

## Usage Examples

### FPGA Bitstream Management

**Check FPGA state**
```bash
python -m fpga.bitstream_manager --state
# Output: operating / programming / unknown
```

**List available bitstreams**
```bash
python -m fpga.bitstream_manager --list
# Searches /lib/firmware, /opt/bitstreams, etc.
```

**Load custom bitstream**
```bash
# Local operation
python -m fpga.bitstream_manager --load alephium_custom.rbf

# Remote operation via SSH
python -m fpga.bitstream_manager --remote root@192.168.1.100 --load kaspa.rbf
```

### Algorithm Switching

**Switch to Alephium**
```bash
python -m fpga.algorithm_switcher --switch alephium \
    --pool stratum+tcp://pool.alephium.com:20032 \
    --wallet YOUR_WALLET_ADDRESS
```

**Check miner status**
```bash
python -m fpga.algorithm_switcher --status
# Output: running, algorithm, hashrate
```

**Remote switching**
```bash
python -m fpga.algorithm_switcher --remote root@192.168.1.100 --switch kaspa
```

### ISO Extraction

**Extract bitstreams from vendor ISO**
```bash
sudo python -m fpga.iso_extractor algo_alephium.iso -o extracted/alephium

# Output:
# extracted/alephium/
#   ├── alephium_v2.1.rbf
#   ├── config.json
#   └── alephium_extraction_report.json
```

**Batch process multiple ISOs**
```bash
for iso in algo_*.iso; do
    sudo python -m fpga.iso_extractor "$iso" -o "extracted/$(basename "$iso" .iso)"
done
```

### Network Discovery

**Discover miners on network**
```bash
python -m network.discovery --broadcast
# Sends UDP broadcast on port 14235

python -m network.discovery --scan 192.168.1.0/24
# Port scanning method
```

**Save results**
```bash
python -m network.discovery --broadcast --output discovered.json
```

---

## Architecture

### Jailbreak Module (`jailbreak/`)

**`method_a_network.py`** - Network exploitation
- SSH credential brute-force with default database
- Web UI login and command injection
- Telnet fallback
- Parallel testing for speed

**`method_b_shadow.py`** - Offline modification
- Partition detection (FAT32 boot + ext3/ext4 root)
- Shadow file parsing and hash removal
- SSH configuration editing
- Safety backups (.backup suffix)

**`method_c_uboot.py`** - UART injection
- Serial console interface (pyserial)
- U-Boot autoboot interruption
- Bootargs modification: `init=/bin/sh`
- Interactive shell session

**`auto_jailbreak.py`** - Orchestrator
- Intelligent method selection
- Success probability ranking
- Interactive guidance
- Comprehensive reporting

### FPGA Module (`fpga/`)

**`bitstream_manager.py`** - Sysfs interface
- `/sys/class/fpga_manager/fpga0/state`
- `/sys/class/fpga_manager/fpga0/firmware`
- Bitstream loading and verification
- Remote operation via SSH

**`algorithm_switcher.py`** - Algorithm automation
- Process management (stop/start miner)
- Bitstream loading
- Configuration file updates
- Hashrate monitoring

**`iso_extractor.py`** - ISO processing
- Mount/unmount automation
- .rbf file extraction
- Full file tree extraction
- JSON reporting

### Network Module (`network/`)

**`discovery.py`** - UDP discovery protocol
- Broadcast packet generation (JSON/binary/string)
- Response parsing and classification
- Port scanning fallback
- CIDR network enumeration

**`management_api.py`** - HTTP API (planned)
- Pool configuration endpoint
- Firmware update endpoint
- Status retrieval
- Wireshark-based protocol analysis

**`mass_config.py`** - Batch operations (planned)
- CSV-based miner inventory
- Parallel configuration via threading
- Error handling and retry logic
- Progress reporting

### Safety Module (`safety/`)

**`watchdog_keeper.py`** - Watchdog management
- `/dev/watchdog` interface
- Periodic keepalive signals
- Crash recovery prevention

**`thermal_monitor.py`** - Temperature monitoring
- Sysfs temperature sensors
- Fan PWM control
- Overheat protection
- Alert thresholds

**`power_validator.py`** - Power monitoring
- Expected vs. actual power consumption
- Algorithm-specific validation
- Anomaly detection

---

## Technical Details

### Hardware Platform

**Intel Stratix 10 SoC**
- Quad-core ARM Cortex-A53 (HPS)
- Secure Device Manager (SDM)
- FPGA fabric: 1M-2M logic elements
- Boot media: MicroSD card (standard) or eMMC (rare)

**Power Specifications**
| Algorithm | Currency | Power | Hashrate |
|-----------|----------|-------|----------|
| Blake3 | Alephium | 1,973W | 54.72 GH/s |
| kHeavyHash | Kaspa | 1,700W | 30.00 GH/s |
| Sha512256D | Radiant | 1,236W | 22.28 GH/s |
| KarlsenHash | Karlsen | 1,484W | 32.08 GH/s |

### SD Card Partition Layout

**Partition 1: Raw Preloader** (Type 0xA2)
- U-Boot SPL + SDM firmware
- Read by BootROM at sector 0
- Cannot be mounted (raw binary)

**Partition 2: Boot** (FAT32)
- u-boot.img, u-boot.scr
- Linux kernel (zImage)
- Device tree (.dtb)
- Initial bitstream (fpga.rbf)

**Partition 3: Root Filesystem** (ext3/ext4)
- /usr/bin/miner - Mining binary
- /etc/shadow - Password hashes
- /etc/init.d/ - Startup scripts
- /lib/firmware/ - FPGA bitstreams

### Boot Sequence

1. **SDM** - Loads FSBL, validates signatures (if secure boot enabled)
2. **U-Boot SPL** - Initializes DDR memory
3. **U-Boot SSBL** - Reads environment, loads kernel
4. **Linux Kernel** - Mounts root, executes init
5. **Init Scripts** - Start miner, web UI, services

### Default Credentials

| Service | Username | Password | Probability |
|---------|----------|----------|-------------|
| SSH | root | password | High |
| SSH | root | (blank) | Medium |
| Web UI | admin | 12345678 | High |
| Telnet | root | password | High |

### FPGA Manager Sysfs

```bash
# Check state
cat /sys/class/fpga_manager/fpga0/state
# operating | programming | unknown

# Load bitstream
echo "algo.rbf" > /sys/class/fpga_manager/fpga0/firmware

# Get manager name
cat /sys/class/fpga_manager/fpga0/name
# Intel Stratix10 FPGA Manager
```

---

## Day 1 Operational Checklist

### Pre-Deployment

- [ ] Download firmware repositories from Jianguoyun cloud storage
- [ ] Archive management tools (K10 Tool, P2 Tool)
- [ ] Collect community bitstreams (Zetheron Discord, Tswift Telegram)
- [ ] Prepare Linux workstation with SD card reader
- [ ] Install toolkit dependencies (`pip install -r requirements.txt`)

### Deployment

**Step 1: Network reconnaissance**
```bash
# Discover miners
python -m network.discovery --scan 192.168.1.0/24 --output miners.json

# Attempt Method A (non-invasive)
python -m jailbreak.auto_jailbreak --target 192.168.1.100
```

**Step 2: Physical access (if Method A fails)**
```bash
# Power down miner
# Extract SD card
# Mount and edit
sudo python -m jailbreak.method_b_shadow /dev/sdb

# Reinsert, boot, SSH
ssh root@192.168.1.100  # (blank password)
```

**Step 3: Initial configuration**
```bash
# Change root password
passwd

# Enable persistent SSH
sed -i 's/#PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config
/etc/init.d/sshd restart

# Install SSH key
mkdir -p /root/.ssh
echo "YOUR_PUBLIC_KEY" >> /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
```

**Step 4: Verify FPGA functionality**
```bash
# Check state
cat /sys/class/fpga_manager/fpga0/state

# List bitstreams
ls -lh /lib/firmware/*.rbf

# Test algorithm switch
python -m fpga.algorithm_switcher --switch alephium \
    --pool stratum+tcp://pool.alephium.com:20032 \
    --wallet YOUR_WALLET
```

**Step 5: Safety validation**
```bash
# Monitor temperatures
watch -n 5 "cat /sys/class/hwmon/hwmon0/temp1_input"

# Check fan speed (PWM)
cat /sys/class/hwmon/hwmon0/pwm1

# Monitor power (via external meter)
```

### Post-Deployment

- [ ] Create SD card backup (`dd if=/dev/sdb of=k10_backup_YYYYMMDD.img bs=4M`)
- [ ] Document custom bitstream locations
- [ ] Set up monitoring (hashrate, temperature, uptime)
- [ ] Configure automatic failover pools
- [ ] Schedule firmware update checks

---

## Safety Warnings

### Critical Risks

**1. Thermal Runaway**
- Loading high-power bitstreams (Alephium: 2000W) without fan control can cause immediate overheating
- **Mitigation**: Always verify fan daemon is running before switching algorithms

**2. Power Supply Overload**
- Different algorithms have vastly different power consumption (1236W - 1973W)
- **Mitigation**: Validate PSU rating and circuit breaker capacity

**3. Watchdog Resets**
- Stopping miner process may trigger hardware watchdog reset
- **Mitigation**: Feed watchdog manually (`echo 1 > /dev/watchdog`) during debugging

**4. Bricking**
- Corrupting bootloader or partition table renders device unbootable
- **Mitigation**: Always create full SD backup before modifications

### Safe Operation Procedures

**Before bitstream loading**:
```bash
# 1. Verify fan is running
pgrep fan_control || /etc/init.d/fan_control start

# 2. Set fans to 100% manually
echo 255 > /sys/class/hwmon/hwmon0/pwm1

# 3. Monitor temperature during switch
watch -n 1 "cat /sys/class/hwmon/hwmon0/temp1_input"
```

**Emergency shutdown**:
```bash
# Graceful
shutdown -h now

# Immediate (if thermal emergency)
echo o > /proc/sysrq-trigger  # Power off
```

---

## Troubleshooting

### Jailbreak Issues

**Problem**: SSH connection refused
- **Cause**: Firewall or SSH daemon not running
- **Solution**: Use Method B (offline shadow edit) or Method C (UART)

**Problem**: SD card not recognized by Linux
- **Cause**: Card reader incompatibility
- **Solution**: Use USB card reader or different workstation

**Problem**: U-Boot autoboot too fast to interrupt
- **Cause**: bootdelay=0 in environment
- **Solution**: Spam Enter key continuously during power-on

### FPGA Issues

**Problem**: Bitstream load fails (state remains "programming")
- **Cause**: Corrupt .rbf file or wrong device family
- **Solution**: Verify .rbf with `--verify`, re-download from source

**Problem**: FPGA state shows "unknown"
- **Cause**: Driver not loaded or hardware failure
- **Solution**: Check `dmesg | grep fpga`, reload driver `modprobe fpga_mgr`

**Problem**: Miner crashes after bitstream load
- **Cause**: Miner binary incompatible with new bitstream
- **Solution**: Update miner binary or use vendor-matched firmware

### Network Issues

**Problem**: Discovery finds no miners
- **Cause**: Wrong UDP port or miners on different subnet
- **Solution**: Try port scanning method, check router VLAN configuration

**Problem**: High packet loss to miners
- **Cause**: Network congestion or faulty switch
- **Solution**: Use dedicated management VLAN

---

## Related Projects

- **k10-forensics**: SD image analysis and firmware extraction tools
- **a10_ml**: Arria 10 bitstream preprocessing for ML analysis
- **fpga-ml-bitstream**: CNN-based hardware trojan detection

---

## Legal and Ethical Considerations

- **Ownership**: Only jailbreak devices you own or have explicit authorization to modify
- **Warranty**: Physical modifications and firmware changes void manufacturer warranties
- **Liability**: Improper power/thermal management can cause hardware damage or fire hazard
- **Intellectual Property**: Respect vendor copyrights on proprietary bitstreams

**This toolkit is intended for**:
- Research and education
- Authorized penetration testing
- Personal device optimization
- Security vulnerability assessment

**Not intended for**:
- Unauthorized access to third-party devices
- Circumventing mining pool anti-cheat mechanisms
- Redistribution of copyrighted firmware

---

## Contributing

Contributions welcome! Areas of interest:
- Management API reverse engineering (Wireshark captures needed)
- Additional algorithm support
- Safety monitoring enhancements
- Automated farm management features

---

## License

Educational and research use only. See LICENSE file.

---

## References

1. Intel Stratix 10 SoC FPGA Boot User Guide
2. RocketBoards.org - Stratix 10 Documentation
3. K10/P2 technical analysis document (included in docs/)
4. Intel FPGA Manager Linux Driver Documentation

---

## Support

- GitHub Issues: <repo-url>/issues
- Community: Zetheron Discord, Tswift Telegram
- Documentation: See `docs/` directory

---

**Day 1 Toolkit Version 1.0.0**
