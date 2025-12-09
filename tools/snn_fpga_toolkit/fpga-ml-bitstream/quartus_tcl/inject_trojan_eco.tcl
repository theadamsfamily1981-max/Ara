# inject_trojan_eco.tcl
#
# Purpose:
#   - Use the Quartus ECO (Engineering Change Order) flow to insert
#     synthetic "Trojan" modifications into an already-compiled design.
#   - Example Trojans:
#       * Extra LUTs driven by a high-fanout net.
#       * Additional routing from a secret net to an unused IO.
#
# Usage:
#   quartus_sh -t inject_trojan_eco.tcl \
#       <project> <base_revision> <trojan_revision> <out_dir>
#
# Requirements:
#   - <base_revision> must already be compiled.
#   - Claude should fill in the actual make_connection logic.

if { $argc < 4 } {
    puts "Usage: quartus_sh -t inject_trojan_eco.tcl <project> <base_rev> <trojan_rev> <out_dir>"
    qexit -error
}

set project_name   [lindex $argv 0]
set base_revision  [lindex $argv 1]
set trojan_revision [lindex $argv 2]
set out_dir        [file normalize [lindex $argv 3]]

file mkdir $out_dir

load_package flow
load_package ::quartus::project
load_package ::quartus::netlist
load_package ::quartus::eco

# Open base revision and create a new "trojan" revision
project_open -revision $base_revision $project_name

# Claude: create a new revision that will hold ECO changes.
# Depending on Quartus version, we might set a new REVISION_NAME:
set_global_assignment -name REVISION_NAME $trojan_revision

# Load final netlist into ECO environment
# This may vary by device; Claude should verify:
initialize_eco

# --- EXAMPLE TROJAN PATTERN (Claude to refine) ---
# 1. identify a high-fanout net
set high_fanout_nets [get_nets -high_fanout 1]
set target_net [lindex $high_fanout_nets 0]

# 2. create a new logic cell / LUT instance
#    (pseudo-code; Claude needs to map to actual ECO commands)
#    E.g., create a LUT that toggles or leaks information.
#    Use 'create_node', 'set_instance_assignment', etc.

# 3. connect the Trojan logic to the target net
#    make_connection -from $target_net -to <new_node_pin>

# 4. (optional) route Trojan output to an IO or internal node

# -------------------------------------------------

# Save ECO changes and re-run incremental compile
finalize_eco
project_close

# Run a quick recompile of the trojan revision
execute_flow -compile $project_name -revision $trojan_revision

# Copy resulting .sof into out_dir
set sof_file "${project_name}.${trojan_revision}.sof"
set src_path [file join [pwd] $sof_file]
if { [file exists $src_path] } {
    set dst_path [file join $out_dir $sof_file]
    file copy -force $src_path $dst_path
    puts "Wrote Trojan bitstream: $dst_path"
} else {
    puts "WARNING: .sof not found for trojan revision $trojan_revision"
}
