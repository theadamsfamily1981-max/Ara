#!/usr/bin/env quartus_sh -t
#
# Trojan Injection via Engineering Change Order (ECO)
#
# Inserts Hardware Trojans into post-fit netlists using Quartus ECO flow.
# This mimics a real-world supply-chain attack where an adversary modifies
# the netlist after synthesis but before bitstream generation.
#
# ETHICAL USE ONLY:
#   - Educational/research purposes
#   - Training ML models on YOUR OWN designs
#   - Testing defensive security tools
#   - DO NOT use on third-party IP without permission
#
# Usage:
#   quartus_sh -t inject_trojan.tcl <project> <trojan_type> <output_dir>
#
# Arguments:
#   project: Quartus project name (.qpf)
#   trojan_type: Type of Trojan to inject
#       - timebomb: 32-bit counter that triggers after N cycles
#       - comparator: Triggers on specific data pattern
#       - ringoscillator: Side-channel power leakage
#   output_dir: Output directory for infected bitstream
#
# Example:
#   quartus_sh -t inject_trojan.tcl golden_design timebomb data/raw/arria10/infected
#
# Output:
#   data/raw/arria10/infected/timebomb_001/design.rbf
#   data/raw/arria10/infected/timebomb_001/metadata.json
#   data/raw/arria10/infected/timebomb_001/eco_log.txt

package require ::quartus::project
package require ::quartus::flow
package require ::quartus::eco

# Parse arguments
if {$argc < 3} {
    puts "ERROR: Insufficient arguments"
    puts "Usage: quartus_sh -t inject_trojan.tcl <project> <trojan_type> <output_dir>"
    puts ""
    puts "Trojan types:"
    puts "  timebomb       - 32-bit counter (sequential trigger)"
    puts "  comparator     - Data pattern detector (combinational trigger)"
    puts "  ringoscillator - Power side-channel leaker"
    exit 1
}

set project_name [lindex $argv 0]
set trojan_type [lindex $argv 1]
set output_dir [lindex $argv 2]

puts "========================================"
puts "ECO Trojan Injection"
puts "========================================"
puts "Project:      $project_name"
puts "Trojan Type:  $trojan_type"
puts "Output:       $output_dir"
puts ""

# Verify project exists
if {![file exists "${project_name}.qpf"]} {
    puts "ERROR: Project file ${project_name}.qpf not found"
    exit 1
}

# Create output directory
file mkdir $output_dir
set trojan_dir "${output_dir}/${trojan_type}_001"
file mkdir $trojan_dir

# Open log file
set log_file "${trojan_dir}/eco_log.txt"
set log_fh [open $log_file w]
puts $log_fh "ECO Trojan Injection Log"
puts $log_fh "========================"
puts $log_fh "Project: $project_name"
puts $log_fh "Trojan:  $trojan_type"
puts $log_fh "Time:    [clock format [clock seconds]]"
puts $log_fh ""

# Step 1: Load post-fit netlist
puts "Step 1: Loading post-fit netlist..."
puts $log_fh "Step 1: Loading post-fit netlist"

project_open $project_name -current_revision

# Load the post-fit database
if {[catch {post_fit_load_project} result]} {
    puts "ERROR: Failed to load post-fit netlist"
    puts "       Make sure the project has been compiled first"
    puts $log_fh "ERROR: $result"
    close $log_fh
    exit 1
}

puts "  ✓ Post-fit netlist loaded"
puts $log_fh "  Post-fit netlist loaded successfully"

# Step 2: Find victim signals
puts "\nStep 2: Identifying victim signals..."
puts $log_fh "\nStep 2: Identifying victim signals"

# Look for clock and reset signals
set victim_nets {}

# Try to find clock nets
if {[catch {
    set clock_nets [get_nets -filter {name =~ "*clk*" || name =~ "*clock*"}]
    if {[llength $clock_nets] > 0} {
        lappend victim_nets {*}$clock_nets
        puts "  Found [llength $clock_nets] clock nets"
        puts $log_fh "  Found [llength $clock_nets] clock nets"
    }
} result]} {
    puts "  No clock nets found via pattern matching"
    puts $log_fh "  No clock nets via pattern: $result"
}

# Try to find reset nets
if {[catch {
    set reset_nets [get_nets -filter {name =~ "*rst*" || name =~ "*reset*"}]
    if {[llength $reset_nets] > 0} {
        lappend victim_nets {*}$reset_nets
        puts "  Found [llength $reset_nets] reset nets"
        puts $log_fh "  Found [llength $reset_nets] reset nets"
    }
} result]} {
    puts "  No reset nets found via pattern matching"
    puts $log_fh "  No reset nets via pattern: $result"
}

# If no specific victims found, use first available net
if {[llength $victim_nets] == 0} {
    puts "  WARNING: No clock/reset nets found, using first available net"
    puts $log_fh "  WARNING: Using fallback net selection"

    set all_nets [get_nets *]
    if {[llength $all_nets] > 0} {
        set victim_nets [list [lindex $all_nets 0]]
        puts "  Selected: [lindex $victim_nets 0]"
        puts $log_fh "  Selected: [lindex $victim_nets 0]"
    } else {
        puts "ERROR: No nets found in design"
        puts $log_fh "ERROR: No nets in design"
        close $log_fh
        exit 1
    }
}

# Use first victim net
set victim_net [lindex $victim_nets 0]
puts "  Using victim net: $victim_net"
puts $log_fh "  Victim net: $victim_net"

# Step 3: Create Trojan logic
puts "\nStep 3: Creating Trojan logic..."
puts $log_fh "\nStep 3: Creating Trojan logic"

switch $trojan_type {
    "timebomb" {
        puts "  Trojan: 32-bit counter (time bomb)"
        puts $log_fh "  Type: 32-bit time bomb counter"

        # Create counter cells (simplified - actual implementation would use ECO primitives)
        # Note: Full implementation requires device-specific cell instantiation
        puts "  TODO: Implement full ECO cell instantiation for time bomb"
        puts $log_fh "  NOTE: This is a placeholder. Full implementation requires:"
        puts $log_fh "    - create_cell -type register -width 32 -name trojan_counter"
        puts $log_fh "    - place_cell -cell trojan_counter -location LAB_X<N>_Y<M>"
        puts $log_fh "    - make_connection -from $victim_net -to trojan_counter CLK"
        puts $log_fh "    - route_design -incremental"
    }

    "comparator" {
        puts "  Trojan: Data comparator (pattern trigger)"
        puts $log_fh "  Type: Comparator trigger"

        puts "  TODO: Implement full ECO cell instantiation for comparator"
        puts $log_fh "  NOTE: This is a placeholder. Full implementation requires:"
        puts $log_fh "    - Find data bus nets"
        puts $log_fh "    - create_cell -type lut -count 8 -name trojan_cmp"
        puts $log_fh "    - Configure LUTs as comparators (check for 0xDEADBEEF)"
        puts $log_fh "    - place_cell in empty LAB"
        puts $log_fh "    - make_connection to data bus"
        puts $log_fh "    - route_design -incremental"
    }

    "ringoscillator" {
        puts "  Trojan: Ring oscillator (side-channel leaker)"
        puts $log_fh "  Type: Ring oscillator"

        puts "  TODO: Implement full ECO cell instantiation for ring oscillator"
        puts $log_fh "  NOTE: This is a placeholder. Full implementation requires:"
        puts $log_fh "    - create_cell -type lut -count 3 -name trojan_ro"
        puts $log_fh "    - Configure as inverter chain (odd number)"
        puts $log_fh "    - place_cell in isolated region"
        puts $log_fh "    - Loop output back to input (creates oscillation)"
        puts $log_fh "    - route_design -incremental"
    }

    default {
        puts "ERROR: Unknown trojan type: $trojan_type"
        puts $log_fh "ERROR: Unknown trojan type"
        close $log_fh
        exit 1
    }
}

# Note to implementer:
# The actual ECO commands depend on the specific device family and require:
#   1. get_locations -filter {type == LAB && used == 0}  # Find empty LABs
#   2. create_cell with device-specific primitive types
#   3. place_cell with exact coordinates
#   4. make_connection with proper port naming
#   5. route_design -incremental (critical for stealth)
#
# For production use, these placeholders should be replaced with real ECO commands.
# See Quartus Prime Pro User Guide: Engineering Change Orders for device-specific syntax.

# Step 4: Save modified netlist
puts "\nStep 4: Saving modified netlist..."
puts $log_fh "\nStep 4: Saving modified netlist"

# In actual implementation, this would call:
# project_save
# But for this placeholder, we just note it
puts "  TODO: Uncomment project_save after implementing ECO"
puts $log_fh "  NOTE: project_save skipped in placeholder implementation"

# Step 5: Generate infected bitstream
puts "\nStep 5: Generating infected bitstream..."
puts $log_fh "\nStep 5: Generating bitstream"

# In actual implementation:
# execute_module -tool asm
# execute_module -tool cpf ...
#
# For now, create a marker file
set marker_file "${trojan_dir}/PLACEHOLDER.txt"
set marker_fh [open $marker_file w]
puts $marker_fh "This directory is a PLACEHOLDER for ECO Trojan injection."
puts $marker_fh ""
puts $marker_fh "To implement full functionality:"
puts $marker_fh "1. Replace placeholder ECO logic with actual create_cell commands"
puts $marker_fh "2. Add device-specific primitive instantiation"
puts $marker_fh "3. Implement proper placement (get_locations, place_cell)"
puts $marker_fh "4. Add incremental routing (route_design -incremental)"
puts $marker_fh "5. Uncomment project_save and bitstream generation"
puts $marker_fh ""
puts $marker_fh "See Quartus Prime Pro Tcl API documentation:"
puts $marker_fh "  ::quartus::eco package"
puts $marker_fh "  create_cell, place_cell, make_connection, route_design"
close $marker_fh

puts "  ⚠ Placeholder implementation - see ${trojan_dir}/PLACEHOLDER.txt"
puts $log_fh "  Placeholder marker created"

# Save metadata
set metadata_file "${trojan_dir}/metadata.json"
set metadata_fh [open $metadata_file w]
puts $metadata_fh "\{"
puts $metadata_fh "  \"project\": \"$project_name\","
puts $metadata_fh "  \"trojan_type\": \"$trojan_type\","
puts $metadata_fh "  \"label\": \"infected\","
puts $metadata_fh "  \"device\": \"[get_global_assignment -name DEVICE]\","
puts $metadata_fh "  \"timestamp\": \"[clock format [clock seconds] -format {%Y-%m-%d %H:%M:%S}]\","
puts $metadata_fh "  \"status\": \"placeholder\","
puts $metadata_fh "  \"note\": \"This is a placeholder. Real ECO implementation required.\""
puts $metadata_fh "\}"
close $metadata_fh

project_close
close $log_fh

puts ""
puts "========================================"
puts "Summary"
puts "========================================"
puts "⚠ PLACEHOLDER IMPLEMENTATION"
puts ""
puts "This script demonstrates the ECO Trojan injection workflow,"
puts "but actual cell instantiation requires device-specific commands."
puts ""
puts "Output: $trojan_dir"
puts "Log:    $log_file"
puts ""
puts "To implement production version:"
puts "  1. Study device-specific ECO primitives"
puts "  2. Replace placeholder logic with real create_cell/place_cell"
puts "  3. Test on reference design with known layout"
puts "  4. Verify Trojan is functional (but benign) for training"
puts ""
puts "For full Trust-Hub benchmark integration, see:"
puts "  https://trust-hub.org/"
