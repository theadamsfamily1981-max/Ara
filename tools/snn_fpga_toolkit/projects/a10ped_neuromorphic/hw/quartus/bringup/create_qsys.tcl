#
# Platform Designer System Creation Script
# A10PED Neuromorphic - Bring-Up System
#
# Creates a minimal Qsys system with:
#   - JTAG-to-Avalon Master Bridge
#   - 4KB On-Chip RAM
#
# Usage:
#   qsys-script --script=create_qsys.tcl
#
# Author: A10PED Neuromorphic Project
# License: BSD-3-Clause
#

package require qsys

# Create new system
create_system bringup_system

# Set device family
set_project_property DEVICE_FAMILY "Arria 10"
set_project_property DEVICE 10AX115N2F40E2LG

# Clock source
add_instance clk clock_source
set_instance_parameter_value clk clockFrequency 50000000
set_instance_parameter_value clk clockFrequencyKnown true
set_instance_parameter_value clk resetSynchronousEdges DEASSERT

# Reset source
add_instance reset reset_source
set_instance_parameter_value reset associatedClock clk
set_instance_parameter_value reset synchronousEdges DEASSERT

# JTAG-to-Avalon Master Bridge
add_instance jtag_master altera_jtag_avalon_master
set_instance_parameter_value jtag_master USE_PLI 0
set_instance_parameter_value jtag_master PLI_PORT 50000

# On-Chip RAM (4KB = 4096 bytes = 1024 words @ 32-bit)
add_instance onchip_ram altera_avalon_onchip_memory2
set_instance_parameter_value onchip_ram allowInSystemMemoryContentEditor false
set_instance_parameter_value onchip_ram blockType "AUTO"
set_instance_parameter_value onchip_ram dataWidth 32
set_instance_parameter_value onchip_ram dualPort false
set_instance_parameter_value onchip_ram initMemContent true
set_instance_parameter_value onchip_ram initializationFileName "onchip_ram.hex"
set_instance_parameter_value onchip_ram instanceID "ORAM"
set_instance_parameter_value onchip_ram memorySize 4096.0
set_instance_parameter_value onchip_ram readDuringWriteMode "DONT_CARE"
set_instance_parameter_value onchip_ram simAllowMRAMContentsFile false
set_instance_parameter_value onchip_ram slave1Latency 1
set_instance_parameter_value onchip_ram slave2Latency 1
set_instance_parameter_value onchip_ram useNonDefaultInitFile false
set_instance_parameter_value onchip_ram useShallowMemBlocks false
set_instance_parameter_value onchip_ram writable true
set_instance_parameter_value onchip_ram ecc_enabled false
set_instance_parameter_value onchip_ram resetrequest_enabled true

# Connect clock and reset
add_connection clk.clk jtag_master.clk
add_connection clk.clk onchip_ram.clk1
add_connection reset.reset jtag_master.clk_reset
add_connection reset.reset onchip_ram.reset1

# Connect JTAG master to on-chip RAM
add_connection jtag_master.master onchip_ram.s1
set_connection_parameter_value jtag_master.master/onchip_ram.s1 arbitrationPriority 1
set_connection_parameter_value jtag_master.master/onchip_ram.s1 baseAddress 0x0000
set_connection_parameter_value jtag_master.master/onchip_ram.s1 defaultConnection 0

# Export clock and reset interfaces
add_interface clk clock sink
set_interface_property clk EXPORT_OF clk.clk_in
add_interface reset reset sink
set_interface_property reset EXPORT_OF reset.reset_in

# Set system info
set_module_property NAME bringup_system
set_module_property DISPLAY_NAME "A10PED Bring-Up System"
set_module_property DESCRIPTION "Minimal JTAG-to-Avalon system with 4KB on-chip RAM for testing"
set_module_property VERSION 1.0

# Save system
save_system bringup_system.qsys

puts "✅ Platform Designer system created: bringup_system.qsys"
puts ""
puts "Next steps:"
puts "  1. Open bringup_system.qsys in Platform Designer GUI to review"
puts "  2. Generate system: qsys-generate bringup_system.qsys --synthesis=VERILOG"
puts "  3. Build Quartus project: quartus_sh -t build.tcl"
