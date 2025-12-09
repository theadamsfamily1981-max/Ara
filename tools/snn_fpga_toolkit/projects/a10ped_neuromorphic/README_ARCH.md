# Quanta Hardware Architecture: YAML-Driven FPGA Build System

**A scalable, multi-vendor FPGA flow driven by YAML specifications**

This document describes the architecture for the `quanta-hw` repository—a hybrid toolchain system that uses YAML as the single source of truth for neuromorphic AI tiles across Intel (Arria 10, Stratix) and Xilinx (Virtex UltraScale+) FPGAs.

---

## Design Philosophy

### Problem

Traditional FPGA development tightly couples RTL, vendor tools, and board constraints, making it hard to:
- Port designs across boards (A10PED → FK33)
- Reuse IP blocks (PCIe, DDR4, SNN cores)
- Automate builds (Quartus GUI workflows are manual)
- Leverage AI assistants (Claude needs structured specs)

### Solution

**YAML-Driven Build System**:
1. **Specs layer**: Tile + board definitions in YAML (single source of truth)
2. **RTL layer**: Vendor-agnostic RTL driven by specs
3. **Flows layer**: Glue scripts to Yosys/Quartus/Vivado
4. **Automation**: Python/Tcl generators create .qsf/.xdc/.sdc from YAML

**Claude Integration**: Structured specs enable AI-assisted:
- RTL generation from tile YAML
- Constraint file generation from board YAML
- Build script customization
- Report parsing and analysis

---

## Repository Layout

```text
quanta-hw/
  specs/
    tiles/
      a10ped_tile.yaml         # A10PED neuromorphic tile spec
      fk33_tile.yaml           # FK33 neuromorphic tile spec (future)
    boards/
      a10ped_board.yaml        # BittWare A10PED pin/clock config
      fk33_board.yaml          # SQRL FK33 board config (future)
    schema/
      tile_schema.yaml         # Schema for tile definitions
      board_schema.yaml        # Schema for board definitions

  rtl/
    common/
      ai_csr.v                 # Auto-generated CSR block
      snn_core_stub.v          # SNN core stub (v0)
      snn_core_v1.v            # LIF neuron engine (future)
    tiles/
      a10ped/
        top_a10ped_tile.v      # Top-level tile wrapper
      fk33/
        top_fk33_tile.v        # Top-level for FK33 (future)

  flows/
    yosys/
      synth_tile.ys            # Yosys synthesis script (sanity check)
    quartus/
      a10ped/
        gen_qsf.py             # Generate .qsf from board YAML
        build_tile.sh          # Batch build script
        project.tcl            # Quartus project setup
    vivado/
      fk33/
        gen_xdc.py             # Generate .xdc from board YAML
        build_tile.tcl         # Vivado batch build

  constraints/
    a10ped/
      base.sdc               # Base timing constraints
      timing_templates/
        pcie_ddr_basic.sdc   # PCIe + DDR4 timing template
    fk33/
      base.xdc               # Base constraints for Vivado
      timing_templates/
        pcie_hbm_basic.xdc   # PCIe + HBM2 timing template

  out/
    a10ped/
      build/                 # Quartus project files, logs
      bitstreams/            # Final .sof/.rbf files
    fk33/
      build/                 # Vivado project files
      bitstreams/            # Final .bit/.bin files

  tools/
    parse_reports/
      parse_quartus_timing.py   # Extract Fmax, slack from .sta.rpt
      parse_quartus_util.py     # Extract resource usage from .fit.rpt
      parse_vivado_timing.py    # Parse Vivado timing reports
    validate/
      check_tile_spec.py        # Validate YAML against schema
```

---

## YAML Specifications

### Tile Specification Schema

**Purpose**: Define the neuromorphic AI tile architecture (memory, PCIe, CSR, clocks) in a vendor-agnostic way.

**File**: `specs/schema/tile_schema.yaml`

```yaml
# Schema for a neuromorphic AI tile

tile_name: string              # Unique identifier (e.g., "a10ped_tile0")
vendor: ["intel", "xilinx"]    # FPGA vendor
family: string                 # e.g., "arria10", "virtex-ultrascale-plus"
fpga_part: string              # Exact part number (e.g., "10AX115U3F45I2SG")

memory:
  type: ["ddr4", "hbm2"]       # Memory technology
  size_gb: float               # Total capacity
  interface: string            # "emif" (Intel), "xilinx_hbm" (Xilinx)
  channels: int                # Number of memory channels

pcie:
  lanes: int                   # PCIe lane count (4, 8, 16)
  gen: int                     # PCIe generation (2, 3, 4)
  endpoint_name: string        # IP block/module name

csr:
  base_addr: int               # Base address in BAR0
  regs:
    - name: string             # Register name (e.g., "CTRL")
      offset: int              # Byte offset from base_addr
      width: int               # Register width in bits
      access: ["rw", "ro", "wo"]
      desc: string             # Description

clocks:
  - name: string               # Clock domain name
    freq_mhz: float            # Target frequency
    role: ["fabric", "pcie", "mem"]  # Clock purpose

snn_core:
  top_module: string           # Module name (e.g., "snn_core")
  interface:
    csr_port: string           # CSR bus interface name
    mem_ports:
      - name: string
        type: ["axi", "avalon", "native_mem"]
        role: ["read", "write", "readwrite"]
```

### Board Specification Schema

**Purpose**: Define board-specific details (pins, banks, clocks) for constraint generation.

**File**: `specs/schema/board_schema.yaml`

```yaml
# Schema for board-level configuration

board_name: string             # Board identifier (e.g., "bittware_a10ped")
vendor: ["intel", "xilinx"]
fpga_part: string              # Must match tile spec

pcie:
  refclk_pin: string           # PCIe reference clock pin
  reset_pin: string            # PCIe reset pin
  lane_pins:
    - lane: int                # Lane number (0-15)
      tx_p: string             # Transmit differential positive
      tx_n: string             # Transmit differential negative
      rx_p: string             # Receive differential positive
      rx_n: string             # Receive differential negative

ddr4:
  controllers:
    - name: string             # Controller instance name
      bank_group: string       # FPGA bank group (e.g., "3A")
      pins_file: string        # Path to pin CSV/table

clocks:
  - name: string               # Clock net name
    pin: string                # Pin location
    freq_mhz: float            # Frequency
    is_diff: bool              # Differential clock

misc_ios:
  - name: string               # Signal name
    pin: string                # Pin location
    io_standard: string        # e.g., "3.3-V LVCMOS", "LVDS"
    role: string               # Purpose (e.g., "status_led0")
```

---

## Example: A10PED Tile Specification

**File**: `specs/tiles/a10ped_tile.yaml`

```yaml
tile_name: "a10ped_tile0"
vendor: "intel"
family: "arria10"
fpga_part: "10AX115N2F40E2LG"   # Arria 10 GX1150

memory:
  type: "ddr4"
  size_gb: 8.0
  interface: "emif"
  channels: 1

pcie:
  lanes: 8
  gen: 3
  endpoint_name: "pcie_a10ped_ep"

csr:
  base_addr: 0x0000
  regs:
    - { name: "CTRL",       offset: 0x00, width: 32, access: "rw", desc: "bit0:START, bit1:RESET, bit2:IRQ_EN" }
    - { name: "STATUS",     offset: 0x04, width: 32, access: "ro", desc: "bit0:BUSY, bit1:DONE, bit2:ERROR" }
    - { name: "CMD_SRC_LO", offset: 0x08, width: 32, access: "rw", desc: "Source address [31:0]" }
    - { name: "CMD_SRC_HI", offset: 0x0C, width: 32, access: "rw", desc: "Source address [63:32]" }
    - { name: "CMD_DST_LO", offset: 0x10, width: 32, access: "rw", desc: "Destination address [31:0]" }
    - { name: "CMD_DST_HI", offset: 0x14, width: 32, access: "rw", desc: "Destination address [63:32]" }
    - { name: "CMD_LEN",    offset: 0x18, width: 32, access: "rw", desc: "Transfer length in bytes" }
    - { name: "CMD_CFG",    offset: 0x1C, width: 32, access: "rw", desc: "Command mode/config" }
    - { name: "VERSION",    offset: 0x20, width: 32, access: "ro", desc: "Hardware version" }
    - { name: "SNN_THRESHOLD", offset: 0x28, width: 32, access: "rw", desc: "LIF neuron threshold" }
    - { name: "SNN_LEAK",      offset: 0x2C, width: 32, access: "rw", desc: "Membrane leak rate" }
    - { name: "SNN_REFRACT",   offset: 0x30, width: 32, access: "rw", desc: "Refractory period" }

clocks:
  - { name: "core_clk",  freq_mhz: 250.0, role: "fabric" }
  - { name: "pcie_clk",  freq_mhz: 250.0, role: "pcie" }
  - { name: "ddr_clk",   freq_mhz: 266.7, role: "mem" }

snn_core:
  top_module: "snn_core"
  interface:
    csr_port: "csr_avs"
    mem_ports:
      - { name: "ddr_avm", type: "avalon", role: "readwrite" }
```

**Usage with Claude**:
```
Prompt: "Read specs/tiles/a10ped_tile.yaml and generate rtl/tiles/a10ped/top_a10ped_tile.v that:
1. Instantiates ai_csr.v with the specified registers
2. Connects snn_core with the Avalon memory interface
3. Wires PCIe and DDR4 controllers
4. Creates clock domains from the spec"
```

---

## Example: A10PED Board Specification

**File**: `specs/boards/a10ped_board.yaml`

```yaml
board_name: "bittware_a10ped"
vendor: "intel"
fpga_part: "10AX115N2F40E2LG"

pcie:
  refclk_pin: "PIN_AR37"
  reset_pin:  "PIN_BB41"
  lane_pins:
    - { lane: 0, tx_p: "PIN_AV31", tx_n: "PIN_AV32", rx_p: "PIN_AU33", rx_n: "PIN_AU34" }
    - { lane: 1, tx_p: "PIN_BA31", tx_n: "PIN_BA32", rx_p: "PIN_AY33", rx_n: "PIN_AY34" }
    - { lane: 2, tx_p: "PIN_BC31", tx_n: "PIN_BC32", rx_p: "PIN_BB33", rx_n: "PIN_BB34" }
    - { lane: 3, tx_p: "PIN_BF31", tx_n: "PIN_BF32", rx_p: "PIN_BE33", rx_n: "PIN_BE34" }
    - { lane: 4, tx_p: "PIN_AW29", tx_n: "PIN_AW30", rx_p: "PIN_AV35", rx_n: "PIN_AV36" }
    - { lane: 5, tx_p: "PIN_BA29", tx_n: "PIN_BA30", rx_p: "PIN_AY35", rx_n: "PIN_AY36" }
    - { lane: 6, tx_p: "PIN_BC29", tx_n: "PIN_BC30", rx_p: "PIN_BB35", rx_n: "PIN_BB36" }
    - { lane: 7, tx_p: "PIN_BF29", tx_n: "PIN_BF30", rx_p: "PIN_BE35", rx_n: "PIN_BE36" }

ddr4:
  controllers:
    - name: "ddr4_ch0"
      bank_group: "3A"
      pins_file: "constraints/a10ped/ddr4_ch0_pins.csv"

clocks:
  - { name: "refclk_100mhz", pin: "PIN_AU33", freq_mhz: 100.0, is_diff: false }

misc_ios:
  - { name: "status_led0", pin: "PIN_AV39", io_standard: "1.8 V", role: "status" }
  - { name: "status_led1", pin: "PIN_AW39", io_standard: "1.8 V", role: "status" }
  - { name: "status_led2", pin: "PIN_AY38", io_standard: "1.8 V", role: "status" }
  - { name: "status_led3", pin: "PIN_AY39", io_standard: "1.8 V", role: "status" }
```

**Usage with Claude**:
```
Prompt: "Read specs/boards/a10ped_board.yaml and generate flows/quartus/a10ped/gen_qsf.py that:
1. Outputs Quartus .qsf pin assignments for all PCIe lanes
2. Adds DDR4 pins from the CSV file
3. Sets IO standards for misc_ios
4. Creates clock constraints for refclk_100mhz"
```

---

## Build Flow: Quartus for A10PED

### Step 1: Yosys Front-End (Optional Sanity Check)

**File**: `flows/yosys/synth_tile.ys`

```tcl
# Yosys synthesis for vendor-agnostic checks

# Read sources
read_verilog -sv rtl/common/ai_csr.v
read_verilog -sv rtl/common/snn_core_stub.v
read_verilog -sv rtl/tiles/a10ped/top_a10ped_tile.v

# Synthesize for Arria 10 (generic Intel target)
synth_intel -family cyclonev -top top_a10ped_tile

# Output for analysis
write_json out/a10ped/build/synth_output.json
write_verilog out/a10ped/build/synth_output.v

# Check for errors
tee -o out/a10ped/build/yosys.log stat
```

**Run**:
```bash
cd quanta-hw
yosys -c flows/yosys/synth_tile.ys
```

### Step 2: Generate Quartus Constraints

**File**: `flows/quartus/a10ped/gen_qsf.py`

```python
#!/usr/bin/env python3
"""
Generate Quartus .qsf file from board YAML specification

Usage:
    python gen_qsf.py specs/boards/a10ped_board.yaml out/a10ped/build/project.qsf
"""
import sys
import yaml

def generate_qsf(board_yaml_path, output_qsf_path):
    """Generate .qsf from board specification"""
    with open(board_yaml_path, 'r') as f:
        board = yaml.safe_load(f)

    lines = []

    # Header
    lines.append("# Auto-generated from board YAML - DO NOT EDIT\n")
    lines.append(f"# Board: {board['board_name']}\n")
    lines.append(f"# Part: {board['fpga_part']}\n\n")

    # Device
    lines.append(f"set_global_assignment -name DEVICE {board['fpga_part']}\n\n")

    # PCIe pins
    pcie = board['pcie']
    lines.append("# PCIe Reference Clock\n")
    lines.append(f"set_location_assignment {pcie['refclk_pin']} -to pcie_refclk\n")
    lines.append(f"set_instance_assignment -name IO_STANDARD HCSL -to pcie_refclk\n\n")

    lines.append("# PCIe Reset\n")
    lines.append(f"set_location_assignment {pcie['reset_pin']} -to pcie_perst_n\n")
    lines.append(f"set_instance_assignment -name IO_STANDARD \"1.8 V\" -to pcie_perst_n\n\n")

    lines.append("# PCIe Lanes (managed by HIP)\n")
    for lane_info in pcie['lane_pins']:
        lane = lane_info['lane']
        lines.append(f"# Lane {lane}: TX={lane_info['tx_p']}/{lane_info['tx_n']}, "
                    f"RX={lane_info['rx_p']}/{lane_info['rx_n']}\n")

    # Clocks
    lines.append("\n# Board Clocks\n")
    for clk in board.get('clocks', []):
        lines.append(f"set_location_assignment {clk['pin']} -to {clk['name']}\n")
        std = "LVDS" if clk['is_diff'] else "3.3-V LVCMOS"
        lines.append(f"set_instance_assignment -name IO_STANDARD \"{std}\" -to {clk['name']}\n")

    # Misc IOs
    lines.append("\n# Miscellaneous IOs\n")
    for io in board.get('misc_ios', []):
        lines.append(f"set_location_assignment {io['pin']} -to {io['name']}\n")
        lines.append(f"set_instance_assignment -name IO_STANDARD \"{io['io_standard']}\" -to {io['name']}\n")

    # Write output
    with open(output_qsf_path, 'w') as f:
        f.writelines(lines)

    print(f"✅ Generated {output_qsf_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: gen_qsf.py <board_yaml> <output_qsf>")
        sys.exit(1)

    generate_qsf(sys.argv[1], sys.argv[2])
```

### Step 3: Quartus Project Setup

**File**: `flows/quartus/a10ped/project.tcl`

```tcl
# Quartus project setup and build for A10PED tile

set proj_name "a10ped_tile0"
set proj_dir  [file normalize "../../../out/a10ped/build"]
set rtl_root  [file normalize "../../.."]

# Create project
project_new $proj_name -overwrite -revision $proj_name
cd $proj_dir

# Device and top-level
set_global_assignment -name FAMILY "Arria 10"
set_global_assignment -name TOP_LEVEL_ENTITY top_a10ped_tile

# Add RTL sources
set_global_assignment -name SYSTEMVERILOG_FILE "$rtl_root/rtl/common/ai_csr.v"
set_global_assignment -name SYSTEMVERILOG_FILE "$rtl_root/rtl/common/snn_core_stub.v"
set_global_assignment -name SYSTEMVERILOG_FILE "$rtl_root/rtl/tiles/a10ped/top_a10ped_tile.v"

# Generate pin assignments from board YAML
exec python3 "$rtl_root/flows/quartus/a10ped/gen_qsf.py" \
     "$rtl_root/specs/boards/a10ped_board.yaml" \
     "$proj_dir/$proj_name.qsf"

# Add timing constraints
set_global_assignment -name SDC_FILE "$rtl_root/constraints/a10ped/base.sdc"

# Optimization settings
set_global_assignment -name OPTIMIZATION_MODE "AGGRESSIVE PERFORMANCE"
set_global_assignment -name SEED 1

# Run full compile flow
puts "Starting Quartus compile..."
execute_flow -compile

puts "✅ Build complete!"
puts "Bitstream: $proj_dir/output_files/$proj_name.sof"
```

### Step 4: Batch Build Script

**File**: `flows/quartus/a10ped/build_tile.sh`

```bash
#!/usr/bin/env bash
# Batch build script for A10PED tile

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT_DIR="$ROOT_DIR/out/a10ped/build"

echo "=== A10PED Tile Build ==="
echo "Root: $ROOT_DIR"
echo "Output: $OUT_DIR"

# Create output directory
mkdir -p "$OUT_DIR"

# Run Quartus
cd "$OUT_DIR"
quartus_sh -t "$ROOT_DIR/flows/quartus/a10ped/project.tcl"

# Parse reports
python3 "$ROOT_DIR/tools/parse_reports/parse_quartus_timing.py" \
    "$OUT_DIR/output_files/a10ped_tile0.sta.rpt" \
    > "$OUT_DIR/timing_summary.json"

python3 "$ROOT_DIR/tools/parse_reports/parse_quartus_util.py" \
    "$OUT_DIR/output_files/a10ped_tile0.fit.summary" \
    > "$OUT_DIR/utilization_summary.json"

echo ""
echo "✅ Build complete!"
echo "Bitstream: $OUT_DIR/output_files/a10ped_tile0.sof"
echo "Timing: $OUT_DIR/timing_summary.json"
echo "Utilization: $OUT_DIR/utilization_summary.json"
```

**Run**:
```bash
cd quanta-hw/flows/quartus/a10ped
./build_tile.sh
```

---

## Using Claude for Code Generation

### Pattern 1: Generate RTL from Tile Spec

**Prompt**:
```
I have a tile specification in specs/tiles/a10ped_tile.yaml.

Please generate rtl/tiles/a10ped/top_a10ped_tile.v that:
1. Instantiates ai_csr module with registers from the spec
2. Connects snn_core with the Avalon memory interface
3. Implements the clock domains (core_clk, pcie_clk, ddr_clk)
4. Wires PCIe endpoint and DDR4 EMIF controller
5. Includes proper resets and error handling

Use Verilog-2001 syntax and follow Intel recommended coding styles.
```

### Pattern 2: Generate Constraint Scripts

**Prompt**:
```
Read specs/boards/a10ped_board.yaml and generate flows/quartus/a10ped/gen_qsf.py.

The script should:
1. Parse the board YAML
2. Generate Quartus .qsf pin assignments for:
   - PCIe lanes (tx_p, tx_n, rx_p, rx_n)
   - DDR4 pins from CSV file
   - Clock inputs with proper IO standards
   - Misc IOs (LEDs, etc.)
3. Handle differential vs single-ended correctly
4. Add comments for clarity

Output should be a complete Python script.
```

### Pattern 3: Parse Reports

**Prompt**:
```
Generate tools/parse_reports/parse_quartus_timing.py that:
1. Reads Quartus .sta.rpt file
2. Extracts:
   - Fmax for each clock domain
   - Worst setup/hold slack
   - Number of failing paths
3. Outputs JSON with this structure:
   {
     "clocks": [{"name": "core_clk", "fmax_mhz": 250.3, "target_mhz": 250.0}],
     "worst_setup_slack_ns": 0.123,
     "worst_hold_slack_ns": 0.456,
     "failing_paths": 0
   }

Handle edge cases like missing clocks or timing not run.
```

### Pattern 4: Expand to FK33 (Xilinx)

**Prompt**:
```
I have specs/tiles/a10ped_tile.yaml for Intel Arria 10.

Create specs/tiles/fk33_tile.yaml for Xilinx Virtex UltraScale+ VU33P with:
- Same CSR register map (for ABI compatibility)
- HBM2 instead of DDR4 (4GB, 2 pseudo-channels)
- PCIe Gen4 x16 (vs Gen3 x8 on A10PED)
- Higher clock targets: core_clk=300MHz, pcie_clk=500MHz
- AXI memory interface instead of Avalon

Maintain the same tile_name structure but adapt vendor-specific fields.
```

---

## Report Parsing Tools

### Quartus Timing Parser

**File**: `tools/parse_reports/parse_quartus_timing.py`

```python
#!/usr/bin/env python3
"""
Parse Quartus .sta.rpt timing report and extract key metrics

Usage:
    python parse_quartus_timing.py output_files/project.sta.rpt
"""
import sys
import re
import json

def parse_timing_report(rpt_path):
    """Extract timing metrics from Quartus .sta.rpt"""
    with open(rpt_path, 'r') as f:
        content = f.read()

    result = {
        "clocks": [],
        "worst_setup_slack_ns": None,
        "worst_hold_slack_ns": None,
        "failing_paths": 0
    }

    # Parse clock summary (example pattern - adjust to actual report format)
    # Look for: "Clock core_clk : 250.00 MHz Fmax = 253.45 MHz"
    clock_pattern = r'Clock\s+(\S+)\s*:\s*(\d+\.\d+)\s*MHz.*Fmax\s*=\s*(\d+\.\d+)\s*MHz'
    for match in re.finditer(clock_pattern, content):
        clk_name = match.group(1)
        target_mhz = float(match.group(2))
        fmax_mhz = float(match.group(3))

        result["clocks"].append({
            "name": clk_name,
            "target_mhz": target_mhz,
            "fmax_mhz": fmax_mhz,
            "margin_mhz": fmax_mhz - target_mhz
        })

    # Parse worst slacks
    # Example: "Worst-case slack = 0.123 ns"
    setup_slack_match = re.search(r'Worst-case\s+setup\s+slack\s*=\s*([-\d.]+)\s*ns', content, re.IGNORECASE)
    if setup_slack_match:
        result["worst_setup_slack_ns"] = float(setup_slack_match.group(1))

    hold_slack_match = re.search(r'Worst-case\s+hold\s+slack\s*=\s*([-\d.]+)\s*ns', content, re.IGNORECASE)
    if hold_slack_match:
        result["worst_hold_slack_ns"] = float(hold_slack_match.group(1))

    return result

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: parse_quartus_timing.py <sta_rpt_file>")
        sys.exit(1)

    metrics = parse_timing_report(sys.argv[1])
    print(json.dumps(metrics, indent=2))
```

### Quartus Utilization Parser

**File**: `tools/parse_reports/parse_quartus_util.py`

```python
#!/usr/bin/env python3
"""
Parse Quartus .fit.summary for resource utilization

Usage:
    python parse_quartus_util.py output_files/project.fit.summary
"""
import sys
import re
import json

def parse_utilization(summary_path):
    """Extract resource usage from Quartus .fit.summary"""
    with open(summary_path, 'r') as f:
        content = f.read()

    result = {
        "alms": {"used": 0, "available": 0, "percent": 0.0},
        "registers": {"used": 0, "available": 0, "percent": 0.0},
        "memory_bits": {"used": 0, "available": 0, "percent": 0.0},
        "dsps": {"used": 0, "available": 0, "percent": 0.0}
    }

    # Example patterns (adjust to actual format)
    # "ALMs: 45,123 / 427,200 ( 11% )"
    patterns = {
        "alms": r'ALM.*:\s*([\d,]+)\s*/\s*([\d,]+)\s*\(\s*(\d+)%\s*\)',
        "registers": r'Registers.*:\s*([\d,]+)\s*/\s*([\d,]+)\s*\(\s*(\d+)%\s*\)',
        "memory_bits": r'Memory.*:\s*([\d,]+)\s*/\s*([\d,]+)\s*\(\s*(\d+)%\s*\)',
        "dsps": r'DSP.*:\s*([\d,]+)\s*/\s*([\d,]+)\s*\(\s*(\d+)%\s*\)'
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            result[key] = {
                "used": int(match.group(1).replace(',', '')),
                "available": int(match.group(2).replace(',', '')),
                "percent": float(match.group(3))
            }

    return result

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: parse_quartus_util.py <fit_summary_file>")
        sys.exit(1)

    metrics = parse_utilization(sys.argv[1])
    print(json.dumps(metrics, indent=2))
```

---

## Validation Tools

**File**: `tools/validate/check_tile_spec.py`

```python
#!/usr/bin/env python3
"""
Validate tile YAML against schema

Usage:
    python check_tile_spec.py specs/tiles/a10ped_tile.yaml
"""
import sys
import yaml

def validate_tile_spec(tile_yaml_path):
    """Basic validation of tile specification"""
    with open(tile_yaml_path, 'r') as f:
        tile = yaml.safe_load(f)

    errors = []

    # Required top-level fields
    required_fields = ['tile_name', 'vendor', 'family', 'fpga_part', 'memory', 'pcie', 'csr']
    for field in required_fields:
        if field not in tile:
            errors.append(f"Missing required field: {field}")

    # Validate CSR registers
    if 'csr' in tile:
        seen_offsets = set()
        for reg in tile['csr'].get('regs', []):
            offset = reg.get('offset')
            if offset in seen_offsets:
                errors.append(f"Duplicate CSR offset: 0x{offset:04X}")
            seen_offsets.add(offset)

            if reg.get('access') not in ['rw', 'ro', 'wo']:
                errors.append(f"Invalid access mode for {reg.get('name')}: {reg.get('access')}")

    # Validate clocks
    if 'clocks' in tile:
        for clk in tile['clocks']:
            if clk.get('freq_mhz', 0) <= 0:
                errors.append(f"Invalid clock frequency for {clk.get('name')}")

    if errors:
        print("❌ Validation failed:")
        for err in errors:
            print(f"  - {err}")
        return False
    else:
        print(f"✅ {tile_yaml_path} is valid")
        return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: check_tile_spec.py <tile_yaml>")
        sys.exit(1)

    valid = validate_tile_spec(sys.argv[1])
    sys.exit(0 if valid else 1)
```

---

## Future: Vivado Flow for FK33

**File**: `flows/vivado/fk33/gen_xdc.py` (skeleton)

```python
#!/usr/bin/env python3
"""
Generate Vivado .xdc constraints from board YAML

Usage:
    python gen_xdc.py specs/boards/fk33_board.yaml out/fk33/build/constraints.xdc
"""
import sys
import yaml

def generate_xdc(board_yaml_path, output_xdc_path):
    """Generate .xdc from board specification"""
    with open(board_yaml_path, 'r') as f:
        board = yaml.safe_load(f)

    lines = []

    # Header
    lines.append("# Auto-generated from board YAML - DO NOT EDIT\n")
    lines.append(f"# Board: {board['board_name']}\n\n")

    # PCIe constraints (Xilinx format)
    pcie = board['pcie']
    lines.append("# PCIe Reference Clock\n")
    lines.append(f"set_property PACKAGE_PIN {pcie['refclk_pin']} [get_ports pcie_refclk_p]\n")
    lines.append(f"create_clock -period 10.000 [get_ports pcie_refclk_p]\n\n")

    # TODO: Add PCIe lanes, HBM pins, etc.

    with open(output_xdc_path, 'w') as f:
        f.writelines(lines)

    print(f"✅ Generated {output_xdc_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: gen_xdc.py <board_yaml> <output_xdc>")
        sys.exit(1)

    generate_xdc(sys.argv[1], sys.argv[2])
```

---

## Summary

This architecture enables:

✅ **Multi-board support**: A10PED, FK33, etc. share CSR ABI
✅ **Vendor abstraction**: YAML specs hide Intel vs Xilinx details
✅ **Automation**: Scripts generate constraints from specs
✅ **Claude integration**: Structured prompts for RTL, constraints, parsers
✅ **Reproducibility**: `build_tile.sh` does complete batch builds
✅ **Extensibility**: Add new boards by creating YAML + constraint generators

**Next steps**:
1. Implement `specs/tiles/fk33_tile.yaml`
2. Create Vivado build flow for FK33
3. Add HBM2 memory controller integration
4. Develop cross-platform validation suite

---

**Status**: Architecture documented - ready for implementation

**Last Updated**: 2024-11-24
