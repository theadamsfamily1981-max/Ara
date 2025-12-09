# ATCA Board Salvage Guide

## Overview

ATCA (Advanced Telecommunications Computing Architecture) boards from decommissioned telecom infrastructure often contain high-end FPGAs that can be repurposed for AI research. These boards are significantly cheaper on the used market compared to their original cost.

## What is ATCA?

ATCA is a series of specifications for next-generation carrier-grade communications equipment. ATCA boards are:
- **Hot-swappable** blade servers for telecom racks
- Designed for **high availability** (99.999% uptime)
- Contain **powerful FPGAs** for packet processing, switching, and DSP
- Use **48V DC power** from backplane
- Have **10GbE/40GbE/100GbE** networking

### Why Salvage ATCA Boards?

1. **Cost**: ATCA boards retail for $10,000-$50,000. Used boards sell for $200-$2,000 (5-25x savings!)
2. **Performance**: High-end Virtex-7, Virtex UltraScale+, Stratix IV/V/10 FPGAs
3. **Memory**: Often include massive DDR3/DDR4 arrays (64GB-256GB per board)
4. **Availability**: Telecom decommissioning generates steady supply

## Supported ATCA Hardware

### Xilinx-Based ATCA Boards

| Manufacturer | Board | FPGA | Memory | Typical Use | Used Price |
|--------------|-------|------|--------|-------------|------------|
| Emerson/Artesyn | ATCA-7485 | Virtex-7 690T | 64GB DDR3 | LTE baseband | $800-1,500 |
| Kontron | ATCA-4310 | Virtex-6 LX240T | 32GB DDR3 | Packet processing | $300-600 |
| Advantech | ATCA-9301 | Virtex UltraScale+ VU9P | 128GB DDR4 | 5G switching | $1,500-2,500 |
| Radisys | ATCA-5000 | Virtex-7 V2000T | 64GB DDR3 | High-capacity switching | $1,000-2,000 |
| Mercury | CAM-8110 | Virtex-7 690T | 64GB DDR3 | DSP applications | $700-1,200 |

### Intel/Altera-Based ATCA Boards

| Manufacturer | Board | FPGA | Memory | Typical Use | Used Price |
|--------------|-------|------|--------|-------------|------------|
| Mercury | CAM-7000 | Stratix IV GX | 32GB DDR3 | Packet processing | $400-800 |
| Advantech | MIC-3395 | Arria 10 GX | 64GB DDR4 | 40GbE switching | $900-1,500 |
| NAT Semi | NSC-ATCA-3210 | Stratix V GX | 128GB DDR4 | High-end processing | $1,200-2,000 |
| Trenton | TRC-ATCA-7500 | Arria 10 SX (SoC) | 64GB DDR4 | Embedded telecom | $800-1,400 |

## ATCA-Specific Challenges

### 1. Power Supply

ATCA boards require **48V DC power** from the backplane. For standalone testing:

**Option A: ATCA Shelf/Chassis**
- Buy a used ATCA shelf on eBay ($500-$1,500)
- Includes backplane, power, cooling, shelf manager
- Plug-and-play solution

**Option B: DIY Bench Power**
- Use 48V DC bench supply (20-40A required)
- **WARNING**: Must enable specific power pins
- Requires ATCA power connector adapter

**Option C: ATX PSU Mod (Advanced)**
- Stack 4x 12V rails in series (dangerous!)
- Not recommended unless experienced

### 2. Cooling Requirements

ATCA boards are designed for forced-air cooling in racks:
- Typical airflow: 20-40 CFM
- Use high-CFM fans (80mm, 3000+ RPM)
- Monitor temperatures (FPGA can exceed 100°C without cooling!)

### 3. JTAG Access

ATCA boards have several JTAG access methods:

**Method 1: Front Panel RJ45 (Most Common)**
- Many ATCA boards have RJ45 JTAG on front panel
- Requires JTAG-over-Ethernet adapter or:
  - Custom RJ45-to-14pin adapter cable
  - Pinout varies by manufacturer (check docs!)

**Method 2: On-Board Headers**
- Standard 14-pin Xilinx header
- May require removing board from chassis
- Check schematic for location

**Method 3: Shelf Manager (Remote)**
- Some ATCA shelves support JTAG passthrough via IPMI
- Allows remote programming without physical access
- Vendor-specific (e.g., Pigeon Point, Adlink)

### 4. Multiple FPGAs in JTAG Chain

ATCA boards often have 2-3 FPGAs:
- **Fabric FPGA**: Main processing (e.g., Virtex-7 V2000T)
- **Switch FPGA**: Backplane interface (e.g., Virtex-7 690T)
- **Management FPGA**: IPMI, sensors (e.g., Spartan-6)

You'll typically salvage the **Fabric FPGA** for AI workloads.

## Salvage Procedure for ATCA Boards

### Prerequisites

- ATCA board (pulled from shelf or standalone)
- 48V DC power supply (20-40A) OR ATCA chassis
- JTAG adapter (FT2232H or Platform Cable USB)
- RJ45-to-JTAG adapter (if using front panel JTAG)
- High-CFM cooling fan
- Schematic/manual for your specific board (helpful but not required)

### Step 1: Physical Setup

```bash
# 1. Mount board on test bench with adequate ventilation
# 2. Connect 48V power (carefully - check pinout!)
# 3. Connect cooling fan (exhaust side, 3000+ RPM recommended)
# 4. Connect JTAG cable to front panel RJ45 or on-board header
```

### Step 2: JTAG Configuration

ATCA boards require special OpenOCD configs due to multi-FPGA chains:

```bash
# Test JTAG connection
cd /path/to/mcp/tools/fpga_salvage
sudo openocd -f configs/atca_xilinx.cfg -c "init; scan_chain; shutdown"

# Expected output (example for dual Virtex-7 board):
# Info : JTAG tap: virtex7_fabric.tap tap/device found: 0x03651093
# Info : JTAG tap: virtex7_switch.tap tap/device found: 0x03651093
```

### Step 3: Use the Web GUI (Easy Method)

```bash
# Launch the web GUI
cd /path/to/mcp/tools/fpga_salvage/gui
sudo ./setup_gui.sh

# Open browser to http://localhost:5000
# Select "ATCA Virtex-7" (or your board type)
# Click "Start Salvage"
```

### Step 4: Command-Line Method (Advanced)

```bash
# Edit fpga_salvage.py to add ATCA board type
# Then run:
sudo ./fpga_salvage.py --vendor atca-virtex7

# This will:
# 1. Test JTAG chain
# 2. Erase proprietary telecom firmware
# 3. Program diagnostic bitstream
# 4. Run hardware tests
```

### Step 5: Voltage Tuning

ATCA boards often have I2C-accessible PMICs:

```bash
# Use the GUI "Voltage Tuning" tab, or:
sudo ./scripts/pmic_flasher.py --bus 0 --read

# Tune for AI workloads (lower voltage = lower power)
sudo ./scripts/pmic_flasher.py --bus 0 --preset efficient
```

## Common ATCA Board Configurations

### Example 1: Emerson ATCA-7485 (Dual Virtex-7)

**Specifications**:
- 2x Virtex-7 690T (each 693K logic cells, 3,600 DSPs)
- 64GB DDR3 (32GB per FPGA)
- 4x 10GbE ports
- PCIe Gen2 x8 backplane

**JTAG Chain**:
```
FPGA 0: Virtex-7 690T (Fabric) - IDCODE 0x03651093
FPGA 1: Virtex-7 690T (Switch) - IDCODE 0x03651093
FPGA 2: Spartan-6 (Management) - IDCODE 0x04001093
```

**Salvage Target**: FPGA 0 (Fabric) - Best for AI workloads

**Cooling**: Requires 30+ CFM, monitor via I2C temperature sensors

**Power**: ~200W at full load (both FPGAs active)

**AI Use Case**: Dual-FPGA SNN cluster (1.4M logic cells total)

### Example 2: Advantech ATCA-9301 (Virtex UltraScale+ VU9P)

**Specifications**:
- Virtex UltraScale+ VU9P (1.2M logic cells, 6,840 DSPs)
- 128GB DDR4
- 100GbE support
- Modern (2018+), high-performance

**Salvage Notes**:
- Premium board, harder to find cheap
- Excellent for large-scale AI (supports HBM2 in some configs)
- Lower power consumption than Virtex-7 (14nm vs 28nm)

**AI Use Case**: Large language model inference, GNN acceleration

### Example 3: Mercury CAM-8110 (Virtex-7 690T)

**Specifications**:
- Virtex-7 690T (693K logic cells)
- 16GB DDR3
- Quad 10GbE
- Common on surplus market

**Salvage Notes**:
- Good availability ($700-1,200 used)
- Single main FPGA (simpler JTAG chain)
- Well-documented by Mercury Systems

**AI Use Case**: Mid-size SNN models, real-time inference

## Integrating ATCA FPGAs with SNN Kernel

After salvaging, use the ATCA FPGA as a PCIe accelerator:

### Option 1: Use ATCA Chassis with PCIe Backplane

If your ATCA shelf supports PCIe:
1. Keep board in ATCA chassis
2. Connect shelf backplane to host via PCIe cable
3. ATCA board appears as standard PCIe device

```bash
# Check if FPGA is visible
lspci | grep -i xilinx

# Should show something like:
# 05:00.0 Processing accelerators: Xilinx Corporation Device 7028
```

### Option 2: Extract FPGA, Mount on Custom PCB (Advanced)

For experienced users:
1. Desolder FPGA from ATCA board (BGA rework required!)
2. Design custom PCB with PCIe interface
3. Reflow FPGA onto new board

**Note**: This is very advanced and risky. Only attempt if you have BGA rework experience.

### Option 3: Standalone Operation via JTAG

Use JTAG for data transfer (slower but works):

```c
// In your application
snn_kernel_init_t config = {
    .fpga_id = 0,
    .fpga_interface = SNN_FPGA_JTAG,  // JTAG mode
    .bitstream_path = "/path/to/snn_kernel.bit",
    .enable_monitoring = 1
};

snn_kernel_initialize(&config);
```

## Cost-Benefit Analysis

### Example: Virtex-7 V2000T (Radisys ATCA-5000)

| Item | New FPGA Board | ATCA Salvage |
|------|---------------|--------------|
| Virtex-7 V2000T board | $25,000 | - |
| Used ATCA board | - | $1,500 |
| ATCA chassis (optional) | - | $800 |
| Power supply | Included | $200 (if not using chassis) |
| Cooling | Included | $50 (fans) |
| **Total** | **$25,000** | **$2,550** |
| **Savings** | - | **$22,450 (90%!)** |

**Performance Comparison**:
- Logic cells: 2M (same FPGA chip!)
- Memory: 64GB DDR3 (ATCA) vs 8GB typical (new boards)
- Power: ~300W (same)
- **Result**: Same compute performance, 10x lower cost

## Safety Considerations

### Electrical Safety

⚠️ **48V DC can be dangerous!**
- Always double-check power polarity before connecting
- Use current-limited power supply (start at 5A, increase slowly)
- Monitor for smoke/burning smell on first power-on
- Have fire extinguisher nearby (Class C for electrical fires)

### Thermal Safety

⚠️ **FPGAs can exceed 100°C!**
- Always monitor temperature (via I2C sensors or JTAG)
- Ensure adequate cooling before running workloads
- ATCA boards expect 20-40 CFM forced-air cooling
- Add thermal shutdown to your bitstream (use XADC/SYSMON)

### Legal Considerations

✅ **Legal**:
- Repurpose ATCA boards you purchased legitimately
- Remove proprietary telecom firmware for personal research
- Use for education, research, non-commercial purposes

❌ **Illegal**:
- Extract firmware from active telecom infrastructure
- Reverse-engineer proprietary protocols for malicious use
- Resell boards with stolen/pirated telecom software

## Troubleshooting

### Issue: Board doesn't power on (no LEDs)

**Causes**:
1. Incorrect 48V power pinout
2. Power enable pins not asserted
3. Board requires shelf manager signals

**Solutions**:
```bash
# Check ATCA power pinout (varies by board):
# Pin  1: -48V Return (Ground)
# Pin  2: -48V DC
# Pin 13: +12V (management power)
# Pin 22: Power Enable (may need pull-up)

# Some boards need "Power Good" signal asserted
# Check schematic for ENABLE pin (usually needs 3.3V pull-up)
```

### Issue: JTAG chain not detected

**Causes**:
1. RJ45 pinout doesn't match your JTAG adapter
2. Multiple FPGAs in chain (need correct IDCODE)
3. Board in reset state

**Solutions**:
```bash
# 1. Check RJ45 JTAG pinout (example):
#    Pin 1: TCK
#    Pin 2: TDI
#    Pin 3: TDO
#    Pin 4: TMS
#    Pin 7: GND
#    Pin 8: VCC (3.3V)

# 2. Manually specify all IDCODEs in OpenOCD config
# Edit configs/atca_xilinx.cfg:
jtag newtap fpga0 tap -irlen 6 -expected-id 0x03651093
jtag newtap fpga1 tap -irlen 6 -expected-id 0x03651093
jtag newtap fpga2 tap -irlen 6 -expected-id 0x04001093

# 3. Assert reset release (check for RESET_N pin)
```

### Issue: FPGA overheating (>100°C)

**Causes**:
1. Insufficient airflow
2. Missing heatsink
3. Running at mining voltages (too high)

**Solutions**:
```bash
# 1. Increase cooling
# - Use 80mm fan at 4000+ RPM
# - Position for through-hole airflow (not just blowing on chip)

# 2. Reduce voltage
sudo ./scripts/pmic_flasher.py --bus 0 --preset efficient

# 3. Reduce clock frequency in bitstream (300MHz → 200MHz)
```

### Issue: Memory test fails

**Causes**:
1. DDR3/DDR4 calibration failure (different temp/voltage than original)
2. Missing memory training in diagnostic bitstream
3. Faulty DIMM

**Solutions**:
```bash
# 1. Run memory training in bitstream
#    - Xilinx: Use MIG (Memory Interface Generator) with full calibration
#    - Intel: Use EMIF IP with DDR4 training enabled

# 2. Test individual DIMMs
#    - Remove all but one DIMM
#    - Test each separately

# 3. Check memory voltage
#    - DDR3: 1.5V (may have separate PMIC)
#    - DDR4: 1.2V
```

## Where to Buy ATCA Boards

### Online Marketplaces

1. **eBay**: Search "ATCA board FPGA" or "ATCA blade Virtex"
   - Typical: $300-$2,000
   - Check seller reputation

2. **Alibaba/AliExpress**: Bulk lots from Chinese telecoms
   - Often cheaper but longer shipping
   - Buy from sellers with good ratings

3. **Surplus Electronics Sites**:
   - All-Spec (https://www.allspec.com)
   - BGMicro
   - Electronics Goldmine

4. **Telecom Equipment Resellers**:
   - Genuity IT (telecom surplus)
   - Network Hardware Resale
   - Often have tested/working units

### What to Look For

✅ **Good signs**:
- "Tested, working" or "Powers on"
- Includes backplane/chassis (easier setup)
- Clear photos of FPGA chips
- Seller provides part number

⚠️ **Red flags**:
- "As-is, untested" (may be dead)
- Missing heatsinks or damaged connectors
- No returns accepted
- Extremely low price (likely non-functional)

## Next Steps

After salvaging your ATCA FPGA:

1. **Test with Diagnostic Bitstream**
   - Verify all hardware works (DDR, JTAG, thermals)

2. **Develop AI Kernel**
   - Use Vivado/Quartus to create SNN accelerator bitstream
   - See [docs/ARCHITECTURE.md](ARCHITECTURE.md)

3. **Integrate with SNN Kernel**
   - Use PCIe backplane or JTAG interface
   - See [examples/fpga_salvage_integration.c](../examples/fpga_salvage_integration.c)

4. **Benchmark Performance**
   - Compare to GPU (should be 2-5x better for sparse SNNs)
   - Optimize voltage/frequency for your workload

5. **Build a Cluster**
   - ATCA chassis can hold 14 blades
   - Create massive FPGA cluster for <$20K!

## Resources

- **PICMG ATCA Specifications**: https://www.picmg.org/openstandards/advancedtca/
- **OpenOCD Documentation**: https://openocd.org/doc/html/
- **Xilinx Vivado**: https://www.xilinx.com/support/download.html
- **Intel Quartus**: https://www.intel.com/content/www/us/en/software/programmable/quartus-prime/

## Community

- **GitHub Discussions**: [mcp/discussions](https://github.com/your-repo/mcp/discussions)
- **Discord**: FPGA Salvage channel
- **Reddit**: r/FPGA, r/embedded

---

**Remember**: ATCA board salvage is more complex than mining FPGA salvage due to power and cooling requirements. Start with simpler boards first, then tackle ATCA once you have experience.

Happy salvaging! 🚀
