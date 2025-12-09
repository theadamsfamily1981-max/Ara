#!/usr/bin/env quartus_sh -t
#
# Quartus Build Script
# A10PED Neuromorphic - Bring-Up
#
# Usage:
#   quartus_sh -t build.tcl
#
# Steps:
#   1. Generate Platform Designer system
#   2. Synthesize design
#   3. Fit (place and route)
#   4. Assemble programming file
#   5. Timing analysis
#
# Author: A10PED Neuromorphic Project
# License: BSD-3-Clause
#

puts ""
puts "========================================="
puts " A10PED Neuromorphic - Bring-Up Build"
puts "========================================="
puts ""

# Project settings
set project_name "bringup"
set qsys_system "bringup_system"

# Check if Platform Designer system exists
if {![file exists "${qsys_system}.qsys"]} {
    puts "ERROR: Platform Designer system not found: ${qsys_system}.qsys"
    puts "Run: qsys-script --script=create_qsys.tcl"
    exit 1
}

# Step 1: Generate Platform Designer system
puts "Step 1/5: Generating Platform Designer system..."
puts ""
if {[catch {exec qsys-generate ${qsys_system}.qsys --synthesis=VERILOG} result]} {
    puts "ERROR generating Platform Designer system:"
    puts $result
    exit 1
}
puts "✅ Platform Designer system generated"
puts ""

# Load Quartus project
package require ::quartus::project
if {![project_exists ${project_name}]} {
    puts "ERROR: Project ${project_name} does not exist"
    exit 1
}
project_open ${project_name}

# Step 2: Synthesis
puts "Step 2/5: Running synthesis..."
puts ""
load_package flow
if {[catch {execute_module -tool map} result]} {
    puts "ERROR during synthesis:"
    puts $result
    project_close
    exit 1
}
puts "✅ Synthesis complete"
puts ""

# Step 3: Fit (Place and Route)
puts "Step 3/5: Running fitter (place and route)..."
puts ""
if {[catch {execute_module -tool fit} result]} {
    puts "ERROR during fitting:"
    puts $result
    project_close
    exit 1
}
puts "✅ Fitting complete"
puts ""

# Step 4: Assemble (Generate programming file)
puts "Step 4/5: Assembling programming file..."
puts ""
if {[catch {execute_module -tool asm} result]} {
    puts "ERROR during assembly:"
    puts $result
    project_close
    exit 1
}
puts "✅ Programming file generated"
puts ""

# Step 5: Timing Analysis
puts "Step 5/5: Running timing analysis..."
puts ""
if {[catch {execute_module -tool sta} result]} {
    puts "WARNING during timing analysis:"
    puts $result
    # Don't exit on timing failure for bring-up
}
puts "✅ Timing analysis complete"
puts ""

# Report results
puts "========================================="
puts " Build Complete!"
puts "========================================="
puts ""
puts "Output files:"
puts "  → output_files/${project_name}.sof  (SRAM Object File for JTAG programming)"
puts "  → output_files/${project_name}.rbf  (Raw Binary File for flash)"
puts ""
puts "Next steps:"
puts "  1. Program FPGA via JTAG:"
puts "     quartus_pgm -m jtag -o \"p;output_files/${project_name}.sof\""
puts ""
puts "  2. Test with System Console:"
puts "     system-console"
puts "     Then use JTAG Master service to read/write on-chip RAM at address 0x0000"
puts ""

project_close
