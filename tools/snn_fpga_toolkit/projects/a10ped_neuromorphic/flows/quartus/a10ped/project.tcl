# Quartus project setup and build for A10PED tile
# Part of the YAML-driven FPGA build system
#
# This Tcl script sets up a Quartus project from scratch using the
# YAML specifications and auto-generated constraints.
#
# Usage:
#   cd /path/to/a10ped_neuromorphic
#   mkdir -p out/a10ped/build
#   cd out/a10ped/build
#   quartus_sh -t ../../../flows/quartus/a10ped/project.tcl

package require ::quartus::project

set proj_name "a10ped_tile0"
set proj_dir  [file normalize "../../../out/a10ped/build"]
set rtl_root  [file normalize "../../.."]

puts "=== Quartus Project Setup ==="
puts "Project name: $proj_name"
puts "Project dir:  $proj_dir"
puts "RTL root:     $rtl_root"

# Create output directory if it doesn't exist
file mkdir $proj_dir

# Create project (overwrite if exists)
cd $proj_dir
if {[project_exists $proj_name]} {
    project_open $proj_name -force
} else {
    project_new $proj_name -overwrite -revision $proj_name
}

# Device and family settings
set_global_assignment -name FAMILY "Arria 10"
set_global_assignment -name DEVICE 10AX115N2F40E2LG

# Top-level entity
set_global_assignment -name TOP_LEVEL_ENTITY ai_csr
puts "Top-level: ai_csr (stub - replace with full tile later)"

# Add RTL sources
set_global_assignment -name SYSTEMVERILOG_FILE "$rtl_root/hw/rtl/ai_csr.v"
set_global_assignment -name SYSTEMVERILOG_FILE "$rtl_root/hw/rtl/memcopy_kernel.v"
# Future: Add full tile top when ready
# set_global_assignment -name SYSTEMVERILOG_FILE "$rtl_root/hw/rtl/top_a10ped_tile.v"

puts "Added RTL sources"

# Generate pin assignments from board YAML
puts "Generating .qsf from board YAML..."
if {[catch {exec python3 "$rtl_root/flows/quartus/a10ped/gen_qsf.py" \
     "$rtl_root/specs/boards/a10ped_board.yaml" \
     "$proj_dir/board_pins.qsf"} result]} {
    puts "Warning: Could not generate board pins: $result"
} else {
    # Source the generated pin assignments
    set_global_assignment -name SOURCE_TCL_SCRIPT_FILE "$proj_dir/board_pins.qsf"
    puts "✅ Generated and sourced board_pins.qsf"
}

# Add timing constraints
set sdc_file "$rtl_root/constraints/a10ped/base.sdc"
if {[file exists $sdc_file]} {
    set_global_assignment -name SDC_FILE $sdc_file
    puts "✅ Added timing constraints: base.sdc"
} else {
    puts "⚠️  Warning: base.sdc not found at $sdc_file"
}

# Optimization settings
set_global_assignment -name OPTIMIZATION_MODE "AGGRESSIVE PERFORMANCE"
set_global_assignment -name SEED 1
set_global_assignment -name SYNTH_TIMING_DRIVEN_SYNTHESIS ON
set_global_assignment -name OPTIMIZATION_TECHNIQUE SPEED
set_global_assignment -name PHYSICAL_SYNTHESIS_COMBO_LOGIC ON
set_global_assignment -name PHYSICAL_SYNTHESIS_REGISTER_DUPLICATION ON
set_global_assignment -name PHYSICAL_SYNTHESIS_REGISTER_RETIMING ON

# Save project
export_assignments
project_close

puts ""
puts "✅ Project setup complete!"
puts "   Project: $proj_dir/$proj_name.qpf"
puts ""
puts "To compile:"
puts "   cd $proj_dir"
puts "   quartus_sh --flow compile $proj_name"
