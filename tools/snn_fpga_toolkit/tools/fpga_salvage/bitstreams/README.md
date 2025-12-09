# Diagnostic Bitstream Generation

This directory contains diagnostic bitstreams for FPGA salvage operations.

## Required Bitstreams

### Intel Stratix 10
- **File**: `stratix10_diag.sof`
- **Purpose**: Hardware diagnostics via JTAG
- **Features**:
  - JTAG-to-AXI bridge for register access
  - Temperature sensor interface (On-Chip Thermal Diode)
  - Power monitoring (VCCINT, VCCAUX via on-chip ADC)
  - DDR4 memory test engine
  - HBM2 interface test (if applicable)

### Xilinx Virtex UltraScale+
- **File**: `virtex_ultrascale_diag.bit`
- **Purpose**: Hardware diagnostics via JTAG
- **Features**:
  - BSCAN-based debug interface
  - XADC/SYSMON temperature/voltage monitoring
  - DDR4 BIST (Built-In Self Test)
  - PCIe endpoint test logic

### Xilinx Kintex UltraScale+
- **File**: `kintex_ultrascale_diag.bit`
- **Purpose**: Same as Virtex, scaled for Kintex resources

## How to Generate Diagnostic Bitstreams

### Option 1: Intel Quartus Prime (for Stratix 10)

```bash
# 1. Install Quartus Prime Pro (supports Stratix 10)
# Download from: https://www.intel.com/content/www/us/en/software/programmable/quartus-prime/download.html

# 2. Create new project
quartus_sh --tcl_eval project_new stratix10_diag -part 10SX2800

# 3. Add RTL files (see examples/stratix10_diag/)
quartus_sh --tcl_eval set_global_assignment -name VERILOG_FILE stratix10_diag.v

# 4. Compile
quartus_sh --flow compile stratix10_diag

# 5. Output: output_files/stratix10_diag.sof
```

### Option 2: Xilinx Vivado (for Virtex/Kintex UltraScale+)

```tcl
# 1. Install Vivado (2023.1 or later recommended)
# Download from: https://www.xilinx.com/support/download.html

# 2. Create project (TCL script)
create_project virtex_diag ./virtex_diag -part xcvu9p-flgb2104-2-i

# 3. Add sources
add_files {virtex_diag.v xadc_wrapper.v ddr4_test.v}
add_files -fileset constrs_1 {constraints.xdc}

# 4. Add IP cores
create_ip -name jtag_axi -vendor xilinx.com -library ip -version 1.2
create_ip -name xadc_wiz -vendor xilinx.com -library ip -version 3.3

# 5. Synthesize and implement
launch_runs synth_1
wait_on_run synth_1
launch_runs impl_1 -to_step write_bitstream
wait_on_run impl_1

# 6. Output: virtex_diag.runs/impl_1/virtex_diag.bit
```

### Option 3: Open-Source Toolchain (Limited Support)

For **Xilinx 7-Series only** (NOT UltraScale+):

```bash
# Using SymbiFlow/F4PGA
# Note: UltraScale+ support is experimental/incomplete

# 1. Install F4PGA
conda install -c litex-hub f4pga

# 2. Synthesize with Yosys
yosys -p "synth_xilinx -top diag_core; write_json diag.json" diag.v

# 3. Place & Route with nextpnr
nextpnr-xilinx --json diag.json --xdc constraints.xdc --fasm diag.fasm

# 4. Generate bitstream
fasm2frames diag.fasm diag.frames
xc7frames2bit --part xc7a200t diag.frames diag.bit
```

**Note**: Open-source tools **DO NOT** support Stratix 10 or UltraScale+ yet. You must use vendor tools.

## Example Diagnostic Core (Verilog)

### Minimal Temperature/Voltage Monitor

```verilog
module stratix10_diag (
    input wire clk,
    input wire reset_n,

    // JTAG interface
    input wire tck,
    input wire tms,
    input wire tdi,
    output wire tdo,

    // DDR4 interface (simplified)
    output wire [17:0] ddr4_addr,
    output wire [2:0] ddr4_ba,
    inout wire [71:0] ddr4_dq,
    // ... (full DDR4 interface)

    // Status LEDs
    output wire [3:0] status_led
);

// Instantiate On-Chip Thermal Diode
wire [7:0] temp_data;
fiftyfivenm_tsdblock thermal_sensor (
    .clk(clk),
    .ce(1'b1),
    .clr(~reset_n),
    .tsdcaldone(),
    .tsdcalo(temp_data)
);

// Instantiate JTAG-to-Avalon Bridge
wire [31:0] avl_address;
wire avl_write;
wire avl_read;
wire [31:0] avl_writedata;
wire [31:0] avl_readdata;

altera_jtag_avalon_master jtag_master (
    .clk_clk(clk),
    .clk_reset_reset_n(reset_n),
    .master_address(avl_address),
    .master_write(avl_write),
    .master_read(avl_read),
    .master_writedata(avl_writedata),
    .master_readdata(avl_readdata)
);

// Register Map
// 0x00A: Temperature (read-only)
// 0x00B: Voltage (read-only)
// 0x00C: Memory status
// 0x00D: Thermal limits

reg [31:0] reg_data;

always @(posedge clk) begin
    if (avl_read) begin
        case (avl_address[11:0])
            12'h00A: reg_data <= {24'h0, temp_data};  // Temperature
            12'h00B: reg_data <= voltage_reg;          // Voltage monitoring
            12'h00C: reg_data <= ddr4_status;          // Memory test result
            12'h00D: reg_data <= thermal_limits;       // Max temperature
            default: reg_data <= 32'hDEADBEEF;
        endcase
    end
end

assign avl_readdata = reg_data;

// DDR4 BIST (Built-In Self Test)
// ... (implementation specific to your board)

endmodule
```

## Pre-Built Bitstreams (Community)

If you don't have access to vendor tools, check:

1. **GitHub**: Search for "fpga mining diagnostic" or "stratix10 test bitstream"
2. **FPGA Discord/Forums**: Community members may share diagnostic cores
3. **OpenCores**: Some generic test cores available

## Security Notes

- **DO NOT** use proprietary mining bitstreams from other companies
- **DO NOT** extract encrypted bitstreams (violates DMCA/CFAA)
- **ONLY** use:
  - Self-generated bitstreams
  - Open-source diagnostic cores
  - Bitstreams from boards you own

## Troubleshooting

### "No bitstream found"
- Generate using vendor tools (see above)
- Or skip bitstream step and use JTAG boundary scan only

### "Programming failed"
- Check JTAG connection
- Verify bitstream matches device (e.g., VU9P bitstream for VU9P chip)
- Ensure flash is not write-protected

### "OpenOCD doesn't support my device"
- Use vendor tools (Quartus Programmer, Vivado Hardware Manager)
- Set `export QUARTUS_PGM=/path/to/quartus_pgm` or `VIVADO_LAB=/path/to/vivado_lab`

## Minimal Functional Test (No Custom Bitstream)

If you don't have a diagnostic bitstream yet, you can still test basic functionality:

```bash
# Test JTAG chain detection
./fpga_salvage.py --vendor stratix10 --skip-erase

# This will:
# 1. Detect FPGA via JTAG IDCODE
# 2. Verify JTAG communication
# 3. Skip bitstream programming
# 4. Report JTAG chain status
```

Once you have a diagnostic bitstream, the tool will run full hardware tests.
