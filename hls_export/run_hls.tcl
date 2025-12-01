# =============================================================================
# Vitis HLS Synthesis Script for PGU Cache Kernel
# =============================================================================

# Create project
open_project pgu_cache_hls
set_top pgu_cache_kernel
add_files pgu_cache_kernel.cpp
add_files -tb pgu_cache_tb.cpp

# Create solution
open_solution "solution1" -flow_target vivado
set_part {xcu250-figd2104-2L-e}
create_clock -period 4.0 -name default

# Synthesis directives
config_compile -pipeline_style flp
config_schedule -effort high -relax_ii_for_timing=0

# Run synthesis
csynth_design

# Run co-simulation
cosim_design -trace_level all -rtl verilog

# Export IP
export_design -format ip_catalog -output pgu_cache_ip.zip

# Generate reports
report_timing -file timing_report.rpt
report_utilization -file utilization_report.rpt

exit
