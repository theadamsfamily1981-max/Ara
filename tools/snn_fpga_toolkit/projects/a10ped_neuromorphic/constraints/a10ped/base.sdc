# Synopsys Design Constraints (SDC)
# A10PED Neuromorphic Tile - Base Timing Constraints
# Part of the YAML-driven FPGA build system
#
# This file contains base timing constraints for the A10PED tile.
# Board-specific pin assignments are auto-generated from YAML by gen_qsf.py

# =============================================================================
# Primary Clocks
# =============================================================================

# PCIe reference clock (100 MHz)
# Pin assignment from board YAML: PIN_AR37
create_clock -name pcie_refclk -period 10.000 [get_ports pcie_refclk]

# Board reference clock (100 MHz)
# Pin assignment from board YAML: PIN_AU33
create_clock -name refclk_100mhz -period 10.000 [get_ports refclk_100mhz]

# System clock (50 MHz)
# Pin assignment from board YAML: PIN_AV39
create_clock -name sysclk_50mhz -period 20.000 [get_ports sysclk_50mhz]

# =============================================================================
# Derive Clocks
# =============================================================================

# Derive PLL clocks from Platform Designer system
# This will automatically create clocks for:
#   - core_clk (250 MHz fabric clock)
#   - pcie_clk (250 MHz PCIe clock)
#   - ddr_clk (266.7 MHz DDR4 clock)
derive_pll_clocks

# Derive clock uncertainty for setup/hold analysis
derive_clock_uncertainty

# =============================================================================
# False Paths
# =============================================================================

# Status LEDs are asynchronous outputs (no timing constraints)
set_false_path -to [get_ports status_led0]
set_false_path -to [get_ports status_led1]
set_false_path -to [get_ports status_led2]
set_false_path -to [get_ports status_led3]

# PCIe reset is asynchronous
set_false_path -from [get_ports pcie_perst_n]

# UART signals are low-speed and asynchronous
set_false_path -to [get_ports uart_tx]
set_false_path -from [get_ports uart_rx]

# =============================================================================
# Input/Output Delays
# =============================================================================

# PCIe reset input delay constraints
set_input_delay -clock pcie_refclk -max 2.0 [get_ports pcie_perst_n]
set_input_delay -clock pcie_refclk -min 0.0 [get_ports pcie_perst_n]

# =============================================================================
# Clock Groups
# =============================================================================

# PCIe and fabric clocks are related (same PLL source)
# DDR4 clock is independent
set_clock_groups -asynchronous \
    -group [get_clocks pcie_refclk] \
    -group [get_clocks refclk_100mhz] \
    -group [get_clocks sysclk_50mhz]

# =============================================================================
# Notes
# =============================================================================
#
# 1. Pin assignments are auto-generated from specs/boards/a10ped_board.yaml
#    by flows/quartus/a10ped/gen_qsf.py
#
# 2. DDR4 timing constraints are handled by the EMIF IP core
#
# 3. PCIe timing constraints are handled by the PCIe HIP core
#
# 4. For advanced constraints (multicycle paths, false paths for specific
#    paths), create a separate .sdc file and add it to the project
#
