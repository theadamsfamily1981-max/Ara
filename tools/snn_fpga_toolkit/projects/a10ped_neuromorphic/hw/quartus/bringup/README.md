# A10PED Neuromorphic - JTAG Bring-Up

**Milestone 1.2: JTAG-to-Avalon Bridge with On-Chip RAM**

This is the first hardware milestone for the A10PED neuromorphic project. The goal is to verify basic FPGA connectivity and validate the development toolchain before building more complex designs.

## Overview

This Quartus project implements a minimal FPGA design with:

- **JTAG-to-Avalon Master Bridge**: Allows System Console to access FPGA internal memory
- **4KB On-Chip RAM**: Test buffer for read/write verification
- **50 MHz System Clock**: Direct from board oscillator
- **Status LEDs**: Virtual pins (not connected to physical LEDs in bring-up)

## Hardware Requirements

- **FPGA Board**: BittWare A10PED (dual Arria 10 GX1150)
- **Cable**: USB-Blaster II (onboard via micro-USB)
- **Host PC**: Ubuntu 22.04 or Windows 10/11 with Quartus installed

## Software Requirements

- **Intel Quartus Prime Pro 23.4** (or newer)
  - Free evaluation version sufficient
  - Download from: https://www.intel.com/content/www/us/en/software/programmable/quartus-prime/download.html
  - Requires ~35 GB disk space

- **Python 3.8+** (for test scripts)

## Project Files

```
bringup/
├── README.md              # This file
├── bringup.qpf            # Quartus project file
├── bringup.qsf            # Quartus settings (device, pins, etc.)
├── bringup.sdc            # Timing constraints
├── bringup_top.v          # Top-level Verilog module
├── create_qsys.tcl        # Platform Designer system generator
├── build.tcl              # Automated build script
└── output_files/          # Generated bitstreams (after build)
```

## Quick Start (30 Minutes)

### Step 1: Setup Quartus Environment

```bash
# Source Quartus environment
source /opt/intelFPGA_pro/23.4/quartus/quartus_sh_setup.bash

# Verify tools are available
which quartus_sh
which qsys-script
which quartus_pgm
```

### Step 2: Generate Platform Designer System

```bash
cd /path/to/mcp/projects/a10ped_neuromorphic/hw/quartus/bringup

# Generate Qsys system with JTAG master and RAM
qsys-script --script=create_qsys.tcl
```

**Expected output**:
```
✅ Platform Designer system created: bringup_system.qsys

Next steps:
  1. Open bringup_system.qsys in Platform Designer GUI to review
  2. Generate system: qsys-generate bringup_system.qsys --synthesis=VERILOG
  3. Build Quartus project: quartus_sh -t build.tcl
```

### Step 3: Build FPGA Bitstream

```bash
# Automated build (synthesis, fit, assemble, timing)
quartus_sh -t build.tcl
```

**Expected duration**: 15-30 minutes for Arria 10 GX1150

**Expected output**:
```
=========================================
 Build Complete!
=========================================

Output files:
  → output_files/bringup.sof  (SRAM Object File for JTAG programming)
  → output_files/bringup.rbf  (Raw Binary File for flash)
```

### Step 4: Program FPGA via JTAG

```bash
# Connect A10PED to PC via USB (USB-Blaster II)
# Power on the A10PED

# List available JTAG cables
jtagconfig

# Expected output:
# 1) USB-Blaster II [1-2]
#   02E660DD   10AX115(.|ES)/10AX115R4
#   02E660DD   10AX115(.|ES)/10AX115R4

# Program first FPGA (FPGA 0)
quartus_pgm -m jtag -o "p;output_files/bringup.sof@1"
```

**Note**: The A10PED has **two** Arria 10 FPGAs. `@1` targets the first one, `@2` targets the second.

### Step 5: Test with System Console

#### Option A: Automated Test Script (Recommended)

```bash
cd ../../../sw/tools
python test_jtag_ram.py
```

**Expected output**:
```
============================================================
 A10PED Neuromorphic - JTAG RAM Test Suite
 Milestone 1.2: JTAG Bring-Up Validation
============================================================

============================================================
TEST 1: JTAG Cable Connection
============================================================
Available JTAG cables:
  /devices/10AX115@1#USB-1
Using cable: /devices/10AX115@1#USB-1
✅ PASS: JTAG connection successful

============================================================
TEST 2: On-Chip RAM Read/Write
============================================================
Pattern 1: Writing sequential values...
✅ Pattern 1 PASS: All 256 words verified

Pattern 2: Writing alternating 0x55555555 / 0xAAAAAAAA...
✅ Pattern 2 PASS: All 256 words verified

Pattern 3: Writing walking 1s...
✅ Pattern 3 PASS: All 32 words verified

✅ PASS: RAM read/write successful

============================================================
 ✅ ALL TESTS PASSED - MILESTONE 1.2 COMPLETE!
============================================================
```

#### Option B: Manual Test (System Console GUI)

```bash
# Launch System Console
system-console &
```

**In System Console GUI**:

1. **Tools → JTAG Control**
2. Select cable: `USB-Blaster II [1-2]`
3. **Click "Start Service"**
4. In Tcl Console, run:

```tcl
# Get master service path
set masters [get_service_paths master]
puts $masters
# Example output: /devices/10AX115@1#USB-1/master_0

# Open master service
set master [lindex $masters 0]
open_service master $master

# Write test pattern to on-chip RAM
master_write_32 $master 0x0000 0xDEADBEEF
master_write_32 $master 0x0004 0xCAFEBABE
master_write_32 $master 0x0008 0x12345678

# Read back
set val0 [master_read_32 $master 0x0000 1]
set val1 [master_read_32 $master 0x0004 1]
set val2 [master_read_32 $master 0x0008 1]

puts [format "0x0000: 0x%08X (expected 0xDEADBEEF)" $val0]
puts [format "0x0004: 0x%08X (expected 0xCAFEBABE)" $val1]
puts [format "0x0008: 0x%08X (expected 0x12345678)" $val2]

# Close service
close_service master $master
```

**Expected output**:
```
0x0000: 0xDEADBEEF (expected 0xDEADBEEF)
0x0004: 0xCAFEBABE (expected 0xCAFEBABE)
0x0008: 0x12345678 (expected 0x12345678)
```

## Success Criteria

✅ **Milestone 1.2 is COMPLETE when**:

1. Quartus project builds without errors
2. FPGA programs successfully via JTAG
3. System Console detects JTAG Master service
4. Can read/write on-chip RAM at address 0x0000
5. Test script passes all 3 patterns (sequential, alternating, walking 1s)

## Troubleshooting

### Issue: "jtagconfig" shows no devices

**Symptoms**:
```
No JTAG hardware available
```

**Solutions**:
1. Check USB cable connection (micro-USB to A10PED)
2. Verify board is powered on (check fan spin-up)
3. Install USB-Blaster II drivers:
   ```bash
   sudo apt install libudev-dev
   sudo /opt/intelFPGA_pro/23.4/quartus/bin/jtagconfig --add USB-Blaster
   ```
4. Check USB permissions:
   ```bash
   sudo usermod -a -G plugdev $USER
   # Log out and back in
   ```

### Issue: Quartus build fails with "Cannot fit design"

**Symptoms**:
```
Error: Can't fit design in device 10AX115N2F40E2LG
```

**Solutions**:
1. Verify correct device in `bringup.qsf`:
   ```
   set_global_assignment -name DEVICE 10AX115N2F40E2LG
   ```
2. Regenerate Platform Designer system:
   ```bash
   rm -rf bringup_system/
   qsys-script --script=create_qsys.tcl
   qsys-generate bringup_system.qsys --synthesis=VERILOG
   ```

### Issue: System Console cannot find master service

**Symptoms**:
```
Error: No JTAG master services found
```

**Solutions**:
1. Verify FPGA is programmed with **bringup.sof** (not a different bitstream)
2. Check JTAG cable is selected in System Console
3. Try stopping and restarting service:
   ```tcl
   stop_service master [lindex [get_service_paths master] 0]
   refresh_connections
   ```

### Issue: RAM reads return 0xFFFFFFFF or wrong data

**Symptoms**:
- All reads return `0xFFFFFFFF`
- Reads don't match writes

**Solutions**:
1. **Check address alignment**: Addresses must be word-aligned (multiples of 4)
   - ✅ Good: `0x0000, 0x0004, 0x0008`
   - ❌ Bad: `0x0001, 0x0003, 0x0007`

2. **Verify RAM size**: On-chip RAM is 4KB (0x0000 to 0x0FFF)
   - Addresses beyond 0x0FFF are invalid

3. **Check clock and reset**: In System Console:
   ```tcl
   # Read system info
   set masters [get_service_paths master]
   foreach m $masters {
       puts "Master: $m"
       open_service master $m
       # Try a simple read
       set test [master_read_32 $m 0x0 1]
       puts "Test read: $test"
       close_service master $m
   }
   ```

## Next Steps

After **Milestone 1.2** is complete, proceed to:

### **Phase 2: AI Tile v0 Shell (Milestone 2.1)**

Build a complete AI tile with:
- PCIe Hard IP (Gen3 x8) for host communication
- DDR4 EMIF controller for external memory
- `ai_csr` register block (from generated `hw/rtl/ai_csr.v`)
- `snn_core` stub (simple memcopy for testing)

See: `projects/a10ped_neuromorphic/docs/BRINGUP.md` (to be created)

## Design Notes

### Platform Designer System

The `bringup_system.qsys` contains:

| Component | Type | Address Range | Purpose |
|-----------|------|---------------|---------|
| `clk` | Clock Source | N/A | 50 MHz system clock |
| `reset` | Reset Source | N/A | Synchronous reset |
| `jtag_master` | JTAG-to-Avalon Master | N/A | Debug access from System Console |
| `onchip_ram` | On-Chip Memory II | 0x0000-0x0FFF | 4KB test RAM |

### Memory Map

| Address Range | Size | Component | Access |
|---------------|------|-----------|--------|
| 0x0000-0x0FFF | 4KB  | On-Chip RAM | RW via JTAG |

### Clock Domains

- **Input Clock**: 50 MHz from board oscillator (pin AU33 - **VERIFY THIS!**)
- **System Clock**: 50 MHz (direct, no PLL for bring-up simplicity)

**Note**: Future milestones will use PLLs to generate higher frequencies for DDR4 and PCIe.

### Resource Utilization (Estimated)

For this minimal design on Arria 10 GX1150:

| Resource | Used | Available | Utilization |
|----------|------|-----------|-------------|
| ALMs | ~500 | 427,200 | <1% |
| RAM Blocks | 1 | 2,713 | <1% |
| DSPs | 0 | 1,518 | 0% |

**This is a tiny design!** The Arria 10 GX1150 has massive capacity for future AI logic.

## References

- **BittWare A10PED Product Brief**: https://www.bittware.com/fpga/a10ped/
- **Intel Arria 10 Device Handbook**: https://www.intel.com/content/www/us/en/docs/programmable/683561/
- **Platform Designer User Guide**: https://www.intel.com/content/www/us/en/docs/programmable/683364/
- **System Console User Guide**: https://www.intel.com/content/www/us/en/docs/programmable/683222/

## License

- Hardware (RTL, Qsys): BSD-3-Clause
- Software (test scripts): MIT
- Documentation: CC-BY-4.0

---

**Status**: Ready for testing (Milestone 1.2)

**Last Updated**: 2024-11-24
