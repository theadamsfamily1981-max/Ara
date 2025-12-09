# Hardware Adapter Designs for FPGA Salvage

This directory contains PCB designs, schematics, and BOMs for adapters that make FPGA salvage easier.

## Quick Reference

| Adapter | Purpose | Difficulty | Cost |
|---------|---------|------------|------|
| [JTAG Breakout](#jtag-breakout-board) | Convert mining JTAG to standard | Easy | $5 |
| [Hashboard Power](#hashboard-power-adapter) | Standalone power for hashboards | Medium | $15 |
| [Multi-Chip JTAG](#multi-chip-jtag-splitter) | Access individual chips in chain | Medium | $10 |
| [PCIe Test Fixture](#pcie-card-test-fixture) | Test PCIe cards without motherboard | Hard | $50 |

## Available Designs

All designs are provided in:
- **KiCad format** (open-source, free)
- **Gerber files** (ready for PCB fab - upload to JLCPCB, PCBWay, OSH Park)
- **BOM (Bill of Materials)** with Digi-Key/Mouser part numbers
- **Assembly instructions** with photos

## Designs

### 1. JTAG Breakout Board

**Purpose**: Convert mining JTAG headers (10-pin, unpopulated) to standard 14-pin Xilinx or 10-pin ARM

**Features**:
- 10-pin input (2x5, 0.1" pitch) for mining boards
- 14-pin output (2x7, 0.1" pitch) for Xilinx Platform Cable
- Optional level shifter (3.3V ↔ 1.8V for older FPGAs)
- Test points for scope probing
- Compact: 1" x 2"

**Cost**: ~$5 per board (PCB $2 + parts $3)

**Files**:
- `jtag_breakout/jtag_breakout.kicad_pcb` - PCB design
- `jtag_breakout/jtag_breakout.sch` - Schematic
- `jtag_breakout/gerbers/` - Gerber files for fab
- `jtag_breakout/BOM.csv` - Bill of materials

**Use with**:
- Hashboards with unpopulated JTAG
- Mining cards with non-standard pinouts
- ATCA boards with RJ45 JTAG (use with RJ45-to-pin adapter)

### 2. Hashboard Power Adapter

**Purpose**: Power mining hashboards standalone (without mining controller)

**Features**:
- 12V barrel jack input (standard PC power brick)
- VRM enable signal generation (pull-up to 3.3V)
- Current monitoring (INA219 sensor)
- Soft-start circuit (prevents inrush current damage)
- LED status indicators (power, enable, fault)
- XT60 output connector (high-current, polarized)

**Specifications**:
- Input: 12V DC, 2.1mm barrel jack (center positive)
- Output: 12V to hashboard, up to 20A
- Enable signal: 3.3V pull-up, 10K resistor
- Over-current protection: 20A polyfuse

**Cost**: ~$15 per board (PCB $3 + parts $12)

**Files**:
- `hashboard_power/hashboard_power.kicad_pcb`
- `hashboard_power/hashboard_power.sch`
- `hashboard_power/gerbers/`
- `hashboard_power/BOM.csv`

**Use with**:
- 4x Agilex hashboards
- Stratix/Virtex hashboards
- Any mining board with 12V input + enable signal

### 3. Multi-Chip JTAG Splitter

**Purpose**: Access individual FPGAs in a JTAG chain (bypass others)

**Features**:
- JTAG multiplexer (74HC4851) for chip selection
- 4-channel input (for 4-chip hashboards)
- Single JTAG output (to your adapter)
- DIP switches to select active chip
- Bypass mode (all chips in chain)
- JTAG buffer (74LVC244) for signal integrity

**Why needed**:
- Hashboards have 4 chips in series JTAG chain
- If one chip is dead, can't access others
- This adapter lets you bypass dead chips
- Faster programming (target one chip at a time)

**Cost**: ~$10 per board (PCB $2 + parts $8)

**Files**:
- `jtag_splitter/jtag_splitter.kicad_pcb`
- `jtag_splitter/jtag_splitter.sch`
- `jtag_splitter/gerbers/`
- `jtag_splitter/BOM.csv`

**Use with**:
- 4x Agilex hashboards
- Dual-FPGA ATCA boards
- Any multi-chip mining board

### 4. PCIe Card Test Fixture

**Purpose**: Test/program PCIe mining cards without full motherboard

**Features**:
- PCIe x16 edge connector (for card)
- 12V input (barrel jack)
- PCIe aux power (6-pin and 8-pin connectors)
- Power sequencing (3.3V, 12V, PERST# timing)
- JTAG passthrough (access on-board JTAG)
- PCIe lane monitoring (LEDs for each lane)
- Optional: PCIe-to-USB bridge (basic enumeration test)

**Specifications**:
- Input: 12V DC, 10A minimum (120W)
- PCIe power: 6-pin (75W) + 8-pin (150W)
- PCIe lanes: x16 mechanical, x4 electrical (sufficient for JTAG)

**Cost**: ~$50 per board (PCB $15 + parts $35)

**Note**: This is advanced! Requires careful PCB design for PCIe signal integrity.

**Files**:
- `pcie_test_fixture/pcie_test_fixture.kicad_pcb`
- `pcie_test_fixture/pcie_test_fixture.sch`
- `pcie_test_fixture/gerbers/`
- `pcie_test_fixture/BOM.csv`

**Use with**:
- VU33P/VU35P/VU37P PCIe mining cards
- BCU1525 clones
- Any PCIe FPGA accelerator

## How to Order PCBs

### Method 1: JLCPCB (Cheap, Fast)

1. Go to https://jlcpcb.com
2. Upload the Gerber ZIP file
3. Select options:
   - **Layers**: 2 (for breakouts) or 4 (for PCIe fixture)
   - **Thickness**: 1.6mm
   - **Color**: Green (cheapest) or your preference
   - **Surface Finish**: HASL (cheapest) or ENIG (better)
   - **Quantity**: 5 minimum (only $2!)
4. Add to cart, checkout
5. Shipping: ~7-14 days to US/EU
6. **Cost**: $2-5 for 5 boards!

### Method 2: OSH Park (USA, High Quality)

1. Go to https://oshpark.com
2. Upload the Gerber ZIP or KiCad file directly
3. OSH Park auto-detects settings
4. **Options**:
   - 2-layer: $5 per square inch (for 3 boards)
   - 4-layer: $10 per square inch (for 3 boards)
5. Made in USA, purple PCBs (iconic!)
6. **Cost**: $10-30 for 3 boards

### Method 3: PCBWay (Good Quality, Options)

1. Go to https://www.pcbway.com
2. Upload Gerbers
3. Similar to JLCPCB, slightly higher quality
4. **Cost**: $5-10 for 5 boards

## How to Assemble

### Tools Needed

- Soldering iron (preferably temperature-controlled, 300-350°C)
- Solder (60/40 or lead-free)
- Flux (makes soldering easier)
- Tweezers (for SMD parts)
- Multimeter (for testing)
- Magnifying glass or loupe (helpful for SMD)

### Skill Levels

| Design | Soldering Difficulty | Time |
|--------|---------------------|------|
| JTAG Breakout | Easy (through-hole only) | 15 min |
| Hashboard Power | Medium (some SMD) | 30 min |
| JTAG Splitter | Medium (SMD ICs) | 45 min |
| PCIe Fixture | Hard (fine-pitch SMD) | 2 hours |

### Assembly Services

If you don't want to solder:

1. **JLCPCB Assembly** (cheap!)
   - Upload BOM and component placement files
   - JLCPCB sources parts and assembles
   - Add $5-20 to board cost
   - Great for 10+ boards

2. **Local Hackerspace**
   - Many hackerspaces have pick-and-place machines
   - Members can help assemble
   - Usually free or small donation

3. **PCB Assembly Services**
   - Screaming Circuits
   - Tempo Automation
   - More expensive ($50-100+ per board)

## Bill of Materials (BOM)

### JTAG Breakout Board

| Part | Description | Quantity | Cost (ea) | Source |
|------|-------------|----------|-----------|--------|
| J1 | Header 2x5, 0.1" pitch | 1 | $0.50 | [Digi-Key: S7035-ND](https://www.digikey.com) |
| J2 | Header 2x7, 0.1" pitch | 1 | $0.70 | [Digi-Key: S7037-ND](https://www.digikey.com) |
| R1-R5 | 10K resistor, 0805 | 5 | $0.02 | [Digi-Key: RMCF0805JT10K0CT-ND](https://www.digikey.com) |
| C1 | 0.1µF cap, 0805 | 1 | $0.05 | [Digi-Key: 399-1168-1-ND](https://www.digikey.com) |
| **Total** | | | **$1.95** | |

### Hashboard Power Adapter

| Part | Description | Quantity | Cost (ea) | Source |
|------|-------------|----------|-----------|--------|
| J1 | DC barrel jack, 2.1mm | 1 | $1.00 | [Digi-Key: CP-002A-ND](https://www.digikey.com) |
| J2 | XT60 connector | 1 | $1.50 | [Amazon: XT60 Male](https://amazon.com) |
| U1 | INA219 current sensor | 1 | $2.50 | [Adafruit: INA219](https://www.adafruit.com/product/904) |
| F1 | 20A polyfuse | 1 | $0.50 | [Digi-Key: F2663CT-ND](https://www.digikey.com) |
| R1 | 10K resistor (enable pull-up) | 1 | $0.02 | |
| LED1-3 | LEDs (red, green, yellow) | 3 | $0.30 | [Digi-Key: 160-1144-1-ND](https://www.digikey.com) |
| R2-R4 | 1K resistor (LED current limit) | 3 | $0.06 | |
| C1-C3 | 100µF electrolytic caps | 3 | $0.60 | [Digi-Key: P5555-ND](https://www.digikey.com) |
| **Total** | | | **$12.48** | |

### Multi-Chip JTAG Splitter

| Part | Description | Quantity | Cost (ea) | Source |
|------|-------------|----------|-----------|--------|
| U1 | 74HC4851 (8:1 mux) | 1 | $0.60 | [Digi-Key: 296-8221-1-ND](https://www.digikey.com) |
| U2 | 74LVC244 (buffer) | 1 | $0.40 | [Digi-Key: 296-8503-1-ND](https://www.digikey.com) |
| SW1 | DIP switch, 4-position | 1 | $1.20 | [Digi-Key: CKN9088CT-ND](https://www.digikey.com) |
| J1-J5 | Header 2x5, 0.1" pitch | 5 | $2.50 | [Digi-Key: S7035-ND](https://www.digikey.com) |
| R1-R4 | 4.7K resistor, 0805 | 4 | $0.08 | |
| C1-C4 | 0.1µF cap, 0805 | 4 | $0.20 | |
| **Total** | | | **$8.98** | |

## Example Projects

### Project 1: Salvage 4x Agilex Hashboard

**Hardware needed**:
- 4x Agilex hashboard ($200-400 on eBay/AliExpress)
- JTAG Breakout Board (this design, $5)
- Hashboard Power Adapter (this design, $15)
- FT2232H JTAG adapter ($15)

**Steps**:
1. Solder 10-pin header to hashboard JTAG pads
2. Connect JTAG Breakout to hashboard
3. Connect Power Adapter to hashboard 12V input
4. Power on, run salvage tool
5. Access all 4x Agilex chips (5.6M logic cells total!)

**Result**: $435 for 5.6M logic cells vs $50,000+ for new equivalent

### Project 2: Test VU35P PCIe Card

**Hardware needed**:
- VU35P PCIe mining card (user's card!)
- PCIe Test Fixture (this design, $50)
- 12V 10A power supply ($20)

**Steps**:
1. Insert VU35P card into test fixture
2. Connect 12V power and PCIe aux power
3. Power on, check LED indicators
4. Access JTAG via fixture passthrough
5. Program diagnostic bitstream
6. Verify PCIe enumeration (use USB-to-PCIe bridge on fixture)

**Result**: Test card without full PC setup, faster iteration

## Contribution

Have a useful adapter design? Submit a PR!

**Guidelines**:
- KiCad 7.0+ format
- Include schematic, PCB, gerbers, BOM
- Document thoroughly
- Test before submitting (or mark as "untested")

## License

All hardware designs in this directory are licensed under:
- **CERN Open Hardware Licence Version 2 - Strongly Reciprocal (CERN-OHL-S-2.0)**

This means:
- ✅ Free to use, modify, distribute
- ✅ Commercial use allowed
- ✅ Must share modifications under same license
- ✅ Must provide source files (schematics, PCBs)

## Support

Questions? Issues with designs?
- Open a GitHub issue
- Tag with "hardware" label
- Include photos of your build

Happy building! 🔧⚡
