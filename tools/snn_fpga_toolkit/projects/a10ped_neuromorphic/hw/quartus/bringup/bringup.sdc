# Synopsys Design Constraints (SDC)
# A10PED Neuromorphic - Bring-Up

# Create clock constraint for 50 MHz board clock
create_clock -name clk_50 -period 20.000 [get_ports clk_50]

# Derive PLL clocks (if any created by Platform Designer)
derive_pll_clocks

# Derive clock uncertainty
derive_clock_uncertainty

# Set false paths to virtual pins (status LEDs)
set_false_path -to [get_ports status_led[*]]
