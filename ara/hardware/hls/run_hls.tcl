# ============================================================================
# CorrSpike-HDC Vitis HLS Build Script
# ============================================================================
#
# Usage:
#   vitis_hls -f run_hls.tcl
#   OR
#   vivado_hls -f run_hls.tcl
#
# Targets:
#   - C Simulation (csim)
#   - C Synthesis (csynth)
#   - RTL Co-simulation (cosim)
#   - IP Export (export)
#
# ============================================================================

# Project configuration
set PROJECT_NAME "corr_spike_hdc"
set TOP_FUNCTION "corr_spike_hdc"
set CLOCK_PERIOD 5.0   ;# 200 MHz
set PART "xcu250-figd2104-2L-e"  ;# Alveo U250 (change for your board)

# Alternative parts:
# - Artix-7:     xc7a100tcsg324-1
# - Zynq-7020:   xc7z020clg400-1
# - ZU9EG:       xczu9eg-ffvb1156-2-e
# - Alveo U280:  xcu280-fsvh2892-2L-e

# Create project
open_project -reset ${PROJECT_NAME}_proj

# Add source files
add_files corr_spike_hdc.cpp
add_files corr_spike_hdc.h
add_files -tb corr_spike_hdc_tb.cpp

# Set top function
set_top ${TOP_FUNCTION}

# Create solution
open_solution -reset "solution1" -flow_target vitis

# Set target device and clock
set_part ${PART}
create_clock -period ${CLOCK_PERIOD} -name default

# Configuration directives
config_compile -pipeline_loops 64
config_schedule -effort high
config_bind -effort high

# ============================================================================
# C Simulation
# ============================================================================
puts "Running C Simulation..."
csim_design

# ============================================================================
# C Synthesis
# ============================================================================
puts "Running C Synthesis..."
csynth_design

# Print resource estimates
puts ""
puts "============================================================"
puts "Resource Estimates:"
puts "============================================================"

# ============================================================================
# (Optional) RTL Co-simulation
# ============================================================================
# Uncomment to run RTL simulation (slower but verifies Verilog)
# puts "Running RTL Co-simulation..."
# cosim_design -tool xsim

# ============================================================================
# Export IP
# ============================================================================
puts "Exporting IP..."
export_design -format ip_catalog -output ${PROJECT_NAME}_ip.zip

puts ""
puts "============================================================"
puts "Build Complete!"
puts "============================================================"
puts "IP exported to: ${PROJECT_NAME}_ip.zip"
puts ""

exit
