# A10PED Neuromorphic - AI Tile v0

**Phase 1: Complete AI Tile with PCIe + DDR4 + CSR + Memcopy Kernel**

This is the first production milestone for the A10PED neuromorphic project. AI Tile v0 implements a complete FPGA accelerator tile with host communication (PCIe), local memory (DDR4), command/status registers (CSR), and a simple DMA kernel (memcopy) that serves as a validated foundation for neuromorphic workloads.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Hardware Requirements](#hardware-requirements)
4. [Software Requirements](#software-requirements)
5. [Quick Start](#quick-start)
6. [Building the FPGA Design](#building-the-fpga-design)
7. [Host Software Setup](#host-software-setup)
8. [Validation and Testing](#validation-and-testing)
9. [Performance Benchmarks](#performance-benchmarks)
10. [Troubleshooting](#troubleshooting)
11. [Next Steps](#next-steps)

---

## Overview

AI Tile v0 turns each Arria 10 FPGA on the A10PED into a standalone neuromorphic accelerator tile with:

- **PCIe Gen3 x8 endpoint**: 8 GB/s bidirectional bandwidth to host
- **DDR4 EMIF controller**: 8 GB local memory (SO-DIMM)
- **AI CSR block**: Command/status registers (auto-generated from YAML)
- **Memcopy DMA kernel**: High-bandwidth memory-to-memory transfers

The memcopy kernel is a **SNN core stub** that validates the complete infrastructure before implementing actual neuromorphic algorithms.

### Key Features

✅ **Hybrid Toolchain**: Quartus for PCIe/EMIF, vendor-agnostic CSR logic
✅ **Clean ABI**: Register-based command protocol
✅ **Linux Driver**: Kernel module with character device interface
✅ **Python API**: High-level interface for rapid prototyping
✅ **Validated**: Automated test suite for memcopy operations

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                         HOST (Linux)                          │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  Application (Python/C++)                               │ │
│  │    ↓                                                     │ │
│  │  Python API (a10ped.py)                                 │ │
│  │    ↓                                                     │ │
│  │  Kernel Driver (a10ped_driver.ko)                       │ │
│  └────────────┬────────────────────────────────────────────┘ │
└───────────────┼──────────────────────────────────────────────┘
                │ PCIe Gen3 x8
┌───────────────┼──────────────────────────────────────────────┐
│      FPGA     │   (BittWare A10PED - Arria 10 GX1150)        │
│  ┌────────────▼───────────┐                                  │
│  │   PCIe Hard IP         │                                  │
│  │  ┌─────────────────┐   │                                  │
│  │  │ BAR0: CSR (1MB) │───┼───→ AI CSR Block                │
│  │  │                 │   │        ↓ (commands)              │
│  │  │ BAR2: DMA(256MB)│───┼──────────────────────┐          │
│  │  └─────────────────┘   │                       │          │
│  └────────────┬────────────┘                       │          │
│               │                                    │          │
│          Avalon-MM                                 ▼          │
│               │                       ┌────────────────────┐ │
│  ┌────────────▼────────────┐          │  Memcopy Kernel   │ │
│  │  DDR4 EMIF Controller   │◄─────────│   (SNN stub)      │ │
│  └────────────┬────────────┘          └────────────────────┘ │
│               │                                               │
│  ┌────────────▼────────────┐                                 │
│  │   DDR4 SO-DIMM (8GB)    │                                 │
│  └─────────────────────────┘                                 │
└───────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Host → CSR**: Application writes command parameters to CSR registers via BAR0
2. **CSR → Kernel**: CSR block signals memcopy kernel to start
3. **Kernel ↔ DDR4**: Kernel reads from SRC, writes to DST via Avalon-MM
4. **Kernel → CSR**: Kernel updates STATUS when done
5. **CSR → Host**: Application reads STATUS via BAR0

---

## Hardware Requirements

- **FPGA Board**: BittWare A10PED (dual Arria 10 GX1150)
- **DDR4 Memory**: At least one 8GB SO-DIMM installed
- **PCIe Slot**: Gen3 x8 or x16 (mechanically compatible)
- **Power**: 75W+ PCIe power connector required
- **Host**: x86_64 Linux system (Ubuntu 22.04 recommended)

---

## Software Requirements

### FPGA Development

- **Intel Quartus Prime Pro 23.4+** (or 17.1+)
  - ~35 GB disk space for installation
  - Evaluation/academic license sufficient
  - Download: https://www.intel.com/content/www/us/en/software/programmable/quartus-prime/download.html

- **Platform Designer** (included with Quartus)
- **USB-Blaster II** cable (onboard via micro-USB)

### Host Software

- **Linux Kernel**: 5.x or newer
- **Build Tools**: gcc, make, kernel headers
- **Python**: 3.8+ with numpy (for test scripts)

---

## Quick Start

### Step 1: Build FPGA Design

```bash
cd /path/to/projects/a10ped_neuromorphic/hw/quartus/ai_tile_v0

# Generate Platform Designer system and build (1-2 hours)
make build

# Check reports
make report
```

**Expected build time**: 1-2 hours for Arria 10 GX1150

### Step 2: Program FPGA

```bash
# Check JTAG cable
make check-jtag

# Program FPGA 0 (first Arria 10)
make program
```

### Step 3: Load Linux Driver

```bash
cd ../../../sw/driver

# Build kernel module
make

# Load driver (creates /dev/a10ped0)
sudo insmod a10ped_driver.ko

# Verify
lsmod | grep a10ped
ls -l /dev/a10ped*
```

### Step 4: Run Validation Tests

```bash
cd ../python

# Run automated test suite
python test_tile_v0.py
```

**Expected output**:
```
============================================================
A10PED AI Tile v0 - Validation Test Suite
============================================================

============================================================
TEST 1: Connection and Hardware Info (Tile 0)
============================================================
✅ PASS: Connected to AITile(tile_id=0, version=1.0.0, ddr_ready=True, busy=False)

...

============================================================
Test Summary
============================================================
Total tests:  5
Passed:       5
Failed:       0
Pass rate:    100.0%

✅ ALL TESTS PASSED
AI Tile v0 is operational and ready for SNN workloads!
```

---

## Building the FPGA Design

### Method 1: Makefile (Recommended)

```bash
make clean      # Remove old build artifacts
make qsys       # Generate Platform Designer system only
make build      # Full build (qsys + synthesis + fit + asm + sta)
make report     # Show timing and resource reports
```

### Method 2: Manual Build

```bash
# 1. Generate Platform Designer system
qsys-script --script=create_qsys.tcl
qsys-generate ai_tile_v0_sys.qsys --synthesis=VERILOG

# 2. Build Quartus project
quartus_sh -t build.tcl

# 3. Program FPGA
quartus_pgm -m jtag -o "p;output_files/ai_tile_v0.sof@1"
```

### Build Output Files

- `output_files/ai_tile_v0.sof` - SRAM Object File (JTAG programming)
- `output_files/ai_tile_v0.rbf` - Raw Binary File (flash programming)
- `output_files/ai_tile_v0.fit.summary` - Resource utilization
- `output_files/ai_tile_v0.sta.rpt` - Timing analysis

### Expected Resource Usage

For Arria 10 GX1150:

| Resource | Used | Available | Utilization |
|----------|------|-----------|-------------|
| ALMs | ~50,000 | 427,200 | ~12% |
| RAM Blocks | ~200 | 2,713 | ~7% |
| DSPs | 0 | 1,518 | 0% |

**Note**: DSPs will be used heavily when SNN kernels replace the memcopy stub.

---

## Host Software Setup

### Linux Kernel Driver

The `a10ped_driver.ko` kernel module provides:

- **Character device**: `/dev/a10ped0` (and `/dev/a10ped1` for dual FPGAs)
- **BAR mapping**: mmap() support for direct CSR access
- **ioctl interface**: MEMCOPY, RESET, GET_STATUS commands

#### Building

```bash
cd sw/driver
make
```

#### Loading

```bash
# Load module
sudo insmod a10ped_driver.ko

# Check dmesg
dmesg | tail -20

# Expected output:
# a10ped: Loading driver version 0.1.0
# a10ped: Probing device 8086:09c4
# a10ped: BAR0 mapped at 0xf7e00000 (len=1048576)
# a10ped: BAR2 mapped at 0xc0000000 (len=268435456)
# a10ped: Device a10ped0 registered successfully
```

#### Unloading

```bash
sudo rmmod a10ped_driver
```

### Python API

The `a10ped.py` module provides a high-level Pythonic interface.

#### Example Usage

```python
from a10ped import AITile

# Connect to tile 0
tile = AITile(tile_id=0)

# Check status
print(tile)
# AITile(tile_id=0, version=1.0.0, ddr_ready=True, busy=False)

status = tile.get_status()
print(f"DDR ready: {status.ddr_ready}")
print(f"Temperature: {tile.get_temperature():.1f}°C")

# Perform memcopy
tile.memcopy(
    src=0x00000000,      # Source address in DDR4
    dst=0x10000000,      # Destination address (256MB offset)
    length=4096          # 4KB transfer (must be 64-byte aligned)
)

# Check performance
cycles = tile.get_perf_cycles()
print(f"Completed in {cycles} cycles")
```

---

## Validation and Testing

### Automated Test Suite

The `test_tile_v0.py` script validates:

1. **Connection**: Device enumeration and hardware info
2. **Status Access**: CSR register read operations
3. **Memcopy Basic**: 4KB transfer with error checking
4. **Memcopy Alignment**: Validation of 64-byte alignment rules
5. **Memcopy Performance**: Bandwidth testing (4KB to 4MB)

#### Running Tests

```bash
cd sw/python

# Test tile 0 (default)
python test_tile_v0.py

# Test tile 1 (if dual-tile board)
python test_tile_v0.py --tile-id 1
```

### Manual Testing with Python

```python
from a10ped import AITile
import time

tile = AITile()

# Reset tile
tile.reset()
time.sleep(0.1)

# Check DDR is ready
status = tile.get_status()
assert status.ddr_ready, "DDR4 not ready!"

# Test small transfer
tile.memcopy(src=0, dst=0x1000, length=64)
assert tile.get_status().done, "Transfer did not complete"

print("Manual test passed!")
```

---

## Performance Benchmarks

Typical memcopy performance on A10PED with DDR4-2400:

| Transfer Size | Latency | Bandwidth | Cycles |
|---------------|---------|-----------|--------|
| 4 KB | 0.05 ms | ~80 MB/s | ~12,500 |
| 64 KB | 0.25 ms | ~256 MB/s | ~62,500 |
| 1 MB | 3.5 ms | ~286 MB/s | ~875,000 |
| 4 MB | 14 ms | ~286 MB/s | ~3.5M |

**Notes**:
- Bandwidth is limited by DDR4 configuration (single-channel)
- PCIe Gen3 x8 theoretical max: ~8 GB/s
- Future optimizations: burst length tuning, prefetching, dual-channel DDR4

---

## Troubleshooting

### Issue: Quartus build fails with "Cannot fit design"

**Symptoms**:
```
Error: Can't fit design in device
```

**Solutions**:
1. Verify correct device in `ai_tile_v0.qsf`:
   ```
   set_global_assignment -name DEVICE 10AX115N2F40E2LG
   ```
2. Regenerate Platform Designer system:
   ```bash
   rm -rf ai_tile_v0_sys/
   make qsys
   ```
3. Check for synthesis errors in `output_files/ai_tile_v0.map.rpt`

### Issue: PCIe device not detected by Linux

**Symptoms**:
```
lspci | grep Intel
# No Arria 10 device found
```

**Solutions**:
1. Check PCIe power connector is plugged in
2. Reseat card in PCIe slot
3. Try different PCIe slot (prefer Gen3 x8 or x16)
4. Check BIOS settings:
   - Enable "Above 4G Decoding"
   - Disable "Secure Boot" if needed
5. Reprogram FPGA:
   ```bash
   make program
   # Wait 5 seconds for PCIe link training
   lspci -vvv | grep -A 20 "8086:09c4"
   ```

### Issue: Driver fails to load

**Symptoms**:
```
insmod: ERROR: could not insert module a10ped_driver.ko: Unknown symbol in module
```

**Solutions**:
1. Rebuild driver for current kernel:
   ```bash
   cd sw/driver
   make clean
   make
   ```
2. Check kernel headers are installed:
   ```bash
   sudo apt install linux-headers-$(uname -r)
   ```
3. Check dmesg for details:
   ```bash
   dmesg | tail -50
   ```

### Issue: Memcopy fails with "Alignment error"

**Symptoms**:
```python
tile.memcopy(src=0x1, dst=0x1000, length=4096)
ValueError: src, dst, and length must be 64-byte aligned
```

**Solutions**:
- All addresses and lengths **must be multiples of 64**:
  ```python
  # Good
  tile.memcopy(src=0x0000, dst=0x1000, length=4096)
  tile.memcopy(src=0x0040, dst=0x2000, length=256)

  # Bad
  tile.memcopy(src=0x0001, dst=0x1000, length=4096)  # src misaligned
  tile.memcopy(src=0x0000, dst=0x1000, length=100)   # length misaligned
  ```

### Issue: DDR4 not ready

**Symptoms**:
```
status.ddr_ready == False
```

**Solutions**:
1. Check DDR4 SO-DIMM is properly seated
2. Verify SO-DIMM is compatible (DDR4-2400, non-ECC or ECC)
3. Check EMIF calibration in Quartus report:
   ```bash
   grep -i "calibration" output_files/ai_tile_v0.fit.rpt
   ```
4. Try reducing DDR4 speed in Platform Designer:
   - Open `ai_tile_v0_sys.qsys`
   - Edit `ddr4_emif` parameters
   - Change speed from DDR4-2400 to DDR4-2133
   - Rebuild

---

## Next Steps

After validating AI Tile v0, proceed to **Phase 2: SNN Core v1**:

### Replace Memcopy Kernel with LIF Neuron Engine

1. **Design SNN datapath** (`hw/rtl/snn_core_v1.v`):
   - Leaky Integrate-and-Fire (LIF) neurons
   - Event-driven update logic
   - Configurable thresholds, leak rates

2. **Update CSR configuration**:
   - Add SNN-specific registers (already defined in YAML)
   - `SNN_THRESHOLD`, `SNN_LEAK`, `SNN_REFRACT`

3. **Extend Python API**:
   - `tile.snn_infer(input_spikes, output_buffer, neuron_count, time_steps)`

4. **Benchmark SNN workloads**:
   - MNIST digit classification via rate-coded spikes
   - DVS gesture recognition
   - Topological field networks

### Dual-Tile Operation

5. **Program both FPGAs**:
   ```bash
   make program-dual
   ```

6. **Multi-tile Python API**:
   ```python
   from a10ped import AITile

   tiles = [AITile(0), AITile(1)]

   # Distribute workload across tiles
   tiles[0].snn_infer(spikes_a, out_a, 512, 100)
   tiles[1].snn_infer(spikes_b, out_b, 512, 100)
   ```

### Inter-Tile Communication

7. **Explore QSFP28/optical links** for spike fabric
8. **Implement event routing** between tiles

---

## References

- **A10PED Datasheet**: https://www.bittware.com/files/ds-a10ped.pdf
- **Intel Arria 10 Handbook**: https://www.intel.com/content/www/us/en/docs/programmable/683561/
- **PCIe Design Example**: https://www.intel.com/content/www/us/en/design-example/714948/
- **EMIF User Guide**: https://www.intel.com/content/www/us/en/docs/programmable/683283/
- **OPAE Documentation**: https://opae.github.io/

---

## License

- **Hardware (RTL, Qsys)**: BSD-3-Clause
- **Software (driver, Python)**: MIT / Dual BSD/GPL (driver)
- **Documentation**: CC-BY-4.0

---

**Status**: Phase 1 complete - Ready for SNN development

**Last Updated**: 2024-11-24

**Contributors**: A10PED Neuromorphic Project
