# Advanced Hardware Designs & Recommendations

Beyond basic adapters, these advanced hardware solutions make FPGA salvage easier, safer, and more professional.

## 🔥 Cooling Solutions

Mining hardware often comes without heatsinks or with damaged cooling. Here's how to fix it.

### 1. Universal FPGA Heatsink Adapter

**Problem**: Mining FPGAs use non-standard heatsink mounting holes

**Solution**: Custom heatsink adapter plate

```
Design: universal_heatsink_adapter.stl (3D printable)

Features:
- Mounts to any FPGA package (flip chip BGA)
- Adapts to standard CPU coolers (AM4, LGA115x)
- Aluminum or copper (high thermal conductivity)
- Includes spring-loaded screws (even pressure)

Materials:
- Option 1: 3D print in PETG, attach copper shim
- Option 2: CNC aluminum (better, $20 from SendCutSend)

Thermal Performance:
- Stock mining heatsink: 0.5°C/W (poor)
- With adapter + Noctua NH-L9: 0.3°C/W (good)
- With adapter + water block: 0.1°C/W (excellent!)

Cost: $15 (3D print + shim) or $30 (CNC aluminum)
```

### 2. Active Cooling Shroud

**For**: Hashboards with multiple FPGAs in close proximity

```
Design: hashboard_cooling_shroud.stl

Features:
- Directs airflow across all 4 FPGAs
- Mounts 2x 120mm fans (intake + exhaust)
- Creates positive pressure (dust prevention)
- Fits standard 4x Agilex/Stratix hashboards

Airflow:
- 2x Noctua NF-F12 (120mm, 3000 RPM): 180 CFM total
- Temperature reduction: 30-40°C vs passive

Parts:
- 2x 120mm fans ($30)
- PETG shroud (3D print, $5 in filament)
- 4x M3 screws ($1)

Total: $36
```

### 3. Liquid Cooling Loop

**For**: Extreme overclocking or dense multi-board setups

```
Design: fpga_water_block_specs.txt

Universal FPGA Water Block:
- Cold plate: Copper (nickel plated)
- Fits: 45mm x 45mm FPGA packages
- Mounting: Universal spring screws
- Flow rate: 1 GPM minimum
- Thermal resistance: 0.05-0.1°C/W

Compatible Pumps/Radiators:
- Alphacool Eisbaer (all-in-one, $120)
- Custom loop (pump + 240mm rad, $150)

Performance:
- Keeps FPGA at <60°C even at 300W load
- Silent operation (vs loud mining fans)
- Enables +20% overclock safely

Cost: $120-200 per FPGA (worth it for multi-chip boards!)
```

## ⚡ Advanced Power Solutions

### 4. Multi-Output Bench Power Supply Adapter

**Problem**: Need multiple voltages (12V, 5V, 3.3V) for complex boards

```
Design: multi_rail_power_adapter.kicad_pcb

Inputs:
- 12V DC (barrel jack)

Outputs:
- 12V @ 20A (passthrough for FPGAs)
- 5V @ 5A (buck converter for logic)
- 3.3V @ 3A (LDO for JTAG/IO)

Features:
- Switchable outputs (enable/disable per rail)
- LED indicators per rail
- Current monitoring per rail (INA226)
- Over-current protection (polyfuses)
- Reverse polarity protection

Use Case:
- ATCA boards needing multiple voltages
- Boards with missing VRMs
- Bench testing before full integration

Cost: $25 (PCB + parts)
```

### 5. Programmable Power Sequencer

**Problem**: Some FPGAs need specific power-up sequences (VCCINT before VCCAUX, etc.)

```
Design: power_sequencer.kicad_pcb

Features:
- 4 independently controlled MOSFET outputs
- Arduino Nano for sequence control
- Programmable delays (0-5000ms per step)
- Monitors power-good signals
- Emergency shutdown on fault

Sequences Supported:
- Standard: 3.3V → 1.8V → VCCINT → VCCAUX
- ATCA: 12V → 3.3V → Enable → VCCINT
- Custom: User-programmable via USB

Example Sequence (Stratix 10):
  t=0ms:   3.3V ON
  t=10ms:  1.8V ON
  t=50ms:  VCCINT ON (0.85V)
  t=100ms: VCCAUX ON (1.8V)
  t=150ms: FPGA RESET released

Cost: $35 (PCB + Arduino + MOSFETs)
```

### 6. Current Shunt Measurement Board

**For**: Per-chip power monitoring on multi-FPGA hashboards

```
Design: 4channel_current_monitor.kicad_pcb

Features:
- 4x INA226 high-side current sensors
- 0.1mΩ shunt resistors (100A capable!)
- I2C output (connect to Raspberry Pi)
- Real-time power graphing
- Alert on over-current

Measurements:
- Voltage: 0-36V (±1%)
- Current: 0-100A (±0.5%)
- Power: Calculated (V × I)
- Update rate: 100 Hz

Software:
- Python script for logging
- Grafana dashboard (pretty graphs!)
- Email alerts on fault

Use Case:
- Identify which FPGA in hashboard is drawing too much power
- Optimize AI kernels for power efficiency
- Detect failing chips early

Cost: $40 (PCB + 4x INA226 + shunts)
```

## 🔌 Interconnect & Expansion

### 7. Multi-Board Backplane

**For**: Linking 4+ hashboards into a cluster

```
Design: fpga_cluster_backplane.kicad_pcb (4-layer PCB)

Features:
- 8x edge connectors (for 8 hashboards)
- Shared power distribution (48V or 12V)
- High-speed LVDS interconnect between boards
- Ethernet switch (1GbE per board)
- PCIe switch (optional, for advanced setups)

Topology Options:
1. Ring: Board0 → Board1 → ... → Board7 → Board0
2. Star: All boards connect to central hub
3. Mesh: Full connectivity (64 links!)

Bandwidth:
- LVDS: 1 Gbps per lane, 16 lanes = 16 Gbps
- Ethernet: 1 GbE per board (for control)
- PCIe: Gen3 x4 per board (optional)

Use Case:
- Build 32-FPGA cluster (8 boards x 4 chips)
- Distributed SNN training
- Massive graph processing

Cost: $200 (PCB is large + many connectors)
Advanced! Recommended for 4+ hashboards
```

### 8. LVDS Breakout Cable Kit

**For**: Connecting FPGAs on separate boards via high-speed GPIO

```
Kit: lvds_breakout_cables

Includes:
- 4x shielded twisted-pair cables (1m, 3m, 5m)
- LVDS breakout boards (2x per cable)
- SMA connectors for impedance matching
- Termination resistors (100Ω)

Specifications:
- Data rate: Up to 2 Gbps per pair
- Pairs: 8 pairs per cable (16 Gbps total)
- Cable: CAT6A shielded
- Connectors: SMA or U.FL

Use Case:
- Connect 2 hashboards for dual-board workloads
- FPGA-to-FPGA communication without PCIe
- Synchronization signals between boards

Cost: $60 per cable kit
```

## 🛠️ Programming & Debug Tools

### 9. Pogo Pin Programming Jig

**Problem**: Don't want to solder JTAG headers to every board

```
Design: pogo_pin_jtag_jig.stl (3D print)

Features:
- Spring-loaded pogo pins (gold plated)
- Aligns to hashboard JTAG pads
- Clamps in place (no soldering!)
- Works with unpopulated headers

Pogo Pins:
- P75-E2 (Everett Charles, gold)
- 100g actuation force
- 10,000 cycle lifetime

Alignment:
- 3D printed guide (custom per board)
- Locates on mounting holes or ICs
- Ensures perfect alignment every time

Use Case:
- Program multiple hashboards quickly
- Testing boards before purchase
- Non-destructive inspection

Cost: $30 (3D print + 10x pogo pins)
```

### 10. USB-to-Quad-JTAG Adapter

**Problem**: Programming 4 chips sequentially is slow

```
Design: usb_quad_jtag.kicad_pcb

Features:
- FT4232H (4-port USB-to-JTAG IC)
- Program 4 FPGAs simultaneously!
- Individual JTAG chains per port
- USB 2.0 High Speed (480 Mbps)

Performance:
- Sequential programming: 4x bitstreams = 12 minutes
- Parallel programming: 4x bitstreams = 3 minutes
- 4x speed increase!

Compatible With:
- OpenOCD (multi-adapter mode)
- Quartus Programmer (multi-device)
- Vivado Hardware Manager

Cost: $50 (FT4232H + PCB + connectors)
```

### 11. Logic Analyzer Breakout

**For**: Debugging JTAG communication issues

```
Design: jtag_logic_analyzer_tap.kicad_pcb

Features:
- Passive tap (doesn't affect signal)
- Routes to 8-pin logic analyzer
- 10K pull-ups for signal integrity
- Compatible with Saleae, DSLogic

Signals Monitored:
- TCK (clock)
- TMS (mode select)
- TDI (data in)
- TDO (data out)
- nTRST (reset)
- VREF (reference voltage)

Use Case:
- Troubleshoot JTAG failures
- Verify signal integrity on long cables
- Reverse-engineer unknown JTAG pinouts

Cost: $15 (PCB + connectors)
Requires: Logic analyzer ($70-200, one-time purchase)
```

## 📡 Testing & Diagnostics

### 12. FPGA Quick-Test Board

**Problem**: Want to test if hashboard works before full salvage

```
Design: fpga_quicktest.kicad_pcb

Features:
- Presses onto hashboard edge connector (no JTAG needed!)
- Tests power rails (3.3V, 1.8V, VCCINT, VCCAUX)
- Reads JTAG IDCODE via bit-banging
- LED status indicators (power, JTAG, clock)
- Battery powered (no external PSU needed)

Tests Performed:
1. Power rail voltages (pass/fail)
2. JTAG chain integrity (count chips)
3. FPGA clock present (oscilloscope output)
4. Short circuit detection

Results:
- Green LED: Board likely good
- Yellow LED: Partial failure (some chips dead)
- Red LED: Critical failure (no power or all chips dead)

Use Case:
- Test boards on eBay before buying (if seller allows)
- Quick validation of bulk purchases
- Field testing at liquidation sales

Cost: $25 (PCB + battery + test circuitry)
Time to test: 30 seconds!
```

### 13. Thermal Imaging Mount

**For**: Identifying hot spots and failing components

```
Design: smartphone_thermal_adapter.stl

Features:
- Mounts smartphone thermal camera (FLIR One, Seek Thermal)
- Holds at fixed distance from board (30cm)
- Includes diffuser for even lighting
- Tripod mount (timelapse thermal recording)

Compatible Cameras:
- FLIR One Pro ($400) - Best, 0.1°C sensitivity
- Seek Thermal Compact ($200) - Good value
- CAT S62 phone (built-in thermal, $600)

Use Case:
- Identify overheating VRMs
- Find shorted capacitors
- Optimize cooling placement
- Validate thermal simulations

Cost: $10 (3D print + tripod mount)
Requires: Thermal camera ($200-400, very useful long-term!)
```

### 14. Oscilloscope Probe Set

**Recommended**: For debugging power and signals

```
Kit: fpga_salvage_probe_kit

Includes:
- 4x 10:1 oscilloscope probes (for signals)
- 2x 50:1 high-voltage probes (for 12V rails)
- 1x current probe (clamp-on, AC/DC)
- 1x differential probe (for LVDS)
- Probe holders (magnetic base)

Specifications:
- Bandwidth: 100 MHz (sufficient for JTAG/power)
- Input impedance: 10 MΩ (won't load circuits)
- Current probe: 30A max

Use Case:
- Measure VCCINT ripple (should be <50mV)
- Debug JTAG signal quality
- Verify clock frequencies
- Troubleshoot power sequencing

Cost: $150 (or use existing scope probes)
Requires: Oscilloscope (Rigol DS1054Z, $350 recommended)
```

## 📦 Complete Salvage Kits

### 15. "Hashboard Starter Kit" (Everything You Need)

```
Kit Contents:

Power:
- 12V 30A switching PSU ($40)
- XT60 to barrel jack adapter ($5)
- Power cable with switch ($10)

JTAG:
- FT2232H USB-JTAG adapter ($15)
- JTAG breakout board (this project, $5)
- 10-pin headers (5x, for soldering) ($5)

Cooling:
- 2x Noctua NF-F12 fans ($30)
- 4x adhesive heatsinks ($10)
- Thermal paste (Arctic MX-4) ($8)

Tools:
- Soldering iron (Pinecil, USB-C) ($30)
- Solder (lead-free) ($5)
- Flux pen ($3)
- Tweezers ($5)
- Multimeter ($20)

Monitoring:
- Current sensor board (this project, $40)
- USB thermometer (DS18B20 sensor) ($8)

Total: $239

Savings: Buy individually = $280, kit = $239 ($41 saved)
```

### 16. "Professional Lab Kit" (For Serious Salvage Operations)

```
Kit Contents: Starter Kit PLUS:

Advanced Tools:
- Oscilloscope (Rigol DS1054Z) ($350)
- Logic analyzer (DSLogic Plus) ($150)
- Thermal camera (Seek Thermal) ($200)
- Hot air rework station ($80)

Advanced Hardware:
- USB Quad-JTAG adapter (this project, $50)
- Pogo pin jig (this project, $30)
- Power sequencer (this project, $35)
- 4-channel current monitor (this project, $40)

Infrastructure:
- 19" rack mount (for organizing boards) ($60)
- Cable management kit ($30)
- ESD mat + wrist strap ($25)

Total: $1,289 (on top of starter kit)

Use Case:
- Salvaging 10+ boards
- Building FPGA clusters
- Professional repair shop
- Research lab setup
```

## 🏗️ Multi-Board Infrastructure

### 17. 19" Rack Mount System

**For**: Organizing multiple hashboards

```
Design: hashboard_rack_mount.stl (3D print)

Features:
- Fits 8 hashboards in 4U rack space (2 boards per U)
- Vertical mounting (optimal airflow)
- Shared power distribution (single 48V input)
- Integrated cable management
- Hot-swap capability (remove board without power-down)

Airflow:
- Front-to-back (standard rack orientation)
- 4x 80mm fans at rear (exhaust)
- Mesh front panel (intake)

Power:
- Single 48V 1500W server PSU
- Powers up to 8 boards (200W each)
- Redundant PSU optional (dual PSU)

Cost: $120 (3D print + rack rails + fans + PSU)
Holds: 32 FPGAs (8 boards x 4 chips) = 22M logic cells!
```

### 18. Network Switch Integration

**For**: Ethernet-connected FPGA clusters

```
Design: rack_network_switch.md (guide)

Hardware:
- Ubiquiti EdgeSwitch 16 XG (10GbE, $500)
- OR Mikrotik CRS309-1G-8S+ (10GbE, $280)
- SFP+ DAC cables (1m, $10 each)

Topology:
- Each hashboard: 1x 10GbE port (via QSFP+ to SFP+ adapter)
- Switch: 8-16 ports (for 8-16 boards)
- Uplink: 40GbE to host server

Use Case:
- Distributed ML training
- Parameter server architecture
- Real-time result aggregation

Cost: $280-500 (switch) + $80 (8x DAC cables)
```

## 🧰 Recommended Tools (Not Designs, But Essential)

```
Tier 1: Absolute Minimum ($100)
- Soldering iron: Pinecil ($30) or TS100 ($50)
- Multimeter: UNI-T UT139C ($20)
- Screwdriver set: iFixit Mako ($30)
- Wire cutters/strippers ($10)
- Isopropyl alcohol 99% ($10)

Tier 2: Serious Salvage ($500)
- Tier 1 +
- Oscilloscope: Rigol DS1054Z ($350)
- Hot air station: Quick 861DW ($80)
- ESD mat + strap ($25)
- Helping hands with magnifier ($25)
- Kapton tape ($10)

Tier 3: Professional Lab ($2000)
- Tier 2 +
- Logic analyzer: Saleae Logic 8 ($500)
- Thermal camera: FLIR One Pro ($400)
- Microscope: AmScope SM-4TZ ($300)
- BGA rework station: Quick 861DW ($800)
- Fume extractor ($200)
```

## 📋 Shopping Lists

### AliExpress/Amazon Quick Links

```
Power Supplies:
- Mean Well LRS-350-12 (12V 30A): Search "Mean Well 12V 30A"
- Server PSU breakout: Search "HP 1200W PSU breakout"

JTAG Adapters:
- FT2232H module: Search "FT2232H USB JTAG"
- Xilinx Platform Cable clone: Search "USB JTAG Xilinx"

Cooling:
- Noctua NF-F12: Amazon (genuine)
- Generic 120mm 3000RPM: AliExpress ($5 each)
- Heatsink paste: Arctic MX-4 (Amazon)

Components:
- Pogo pins: Search "P75-E2 pogo pin" (Digi-Key)
- XT60 connectors: Search "XT60 male female"
- Headers 2x5: Search "2.54mm 2x5 header"

Tools:
- Pinecil: Search "Pinecil V2" ($30)
- Multimeter: "UNI-T UT139C" ($20)
```

## 🎯 Prioritization Guide

**If you're buying your first hashboard**:
1. ✅ JTAG adapter + breakout ($20)
2. ✅ 12V PSU ($40)
3. ✅ Soldering iron + header ($35)
4. ✅ 2x fans ($30)
**Total: $125** - Everything you need to start!

**If you're scaling to multiple boards**:
5. Current monitor ($40) - Identify problem boards
6. Pogo pin jig ($30) - Speed up programming
7. Rack mount ($120) - Organize boards

**If you're going pro**:
8. Oscilloscope ($350) - Deep debugging
9. Thermal camera ($200) - Thermal optimization
10. Multi-board backplane ($200) - Build cluster

## 📐 3D Printable Files (Coming Soon!)

All 3D-printable designs will be added:
- Universal heatsink adapter (STL)
- Cooling shroud (STL)
- Pogo pin jig (STL, custom per board)
- Smartphone thermal mount (STL)
- Rack mount brackets (STL)

**Print Settings**:
- Material: PETG (heat resistant)
- Layer height: 0.2mm
- Infill: 30% (structural parts), 100% (thermal)
- Supports: Yes (for overhangs)

**Cost**: ~$2-10 in filament per part

---

**Bottom Line**: Start with the $125 basics, add tools as you scale. The hardware adapters turn a $400 hashboard into a $60,000 equivalent FPGA cluster!
