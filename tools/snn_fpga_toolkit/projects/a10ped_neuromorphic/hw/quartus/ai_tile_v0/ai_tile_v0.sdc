# Synopsys Design Constraints (SDC)
# A10PED Neuromorphic - AI Tile v0

# PCIe reference clock (100 MHz)
create_clock -name pcie_refclk -period 10.000 [get_ports pcie_refclk]

# Derive PLL clocks from Platform Designer system
derive_pll_clocks

# Derive clock uncertainty
derive_clock_uncertainty

# False paths to status LEDs
set_false_path -to [get_ports status_led[*]]

# PCIe reset is asynchronous
set_false_path -from [get_ports pcie_perst_n]

# Set input delay for PCIe reset
set_input_delay -clock pcie_refclk -max 2.0 [get_ports pcie_perst_n]
set_input_delay -clock pcie_refclk -min 0.0 [get_ports pcie_perst_n]
