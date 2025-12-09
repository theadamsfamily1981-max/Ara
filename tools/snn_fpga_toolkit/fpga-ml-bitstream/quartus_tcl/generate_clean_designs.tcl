# generate_clean_designs.tcl
#
# Purpose:
#   - Automatically build many "clean" variants of a base design
#     (different fitter seeds, minor parameter tweaks) to create
#     a labeled dataset of known-good .sof files.
#
# Usage (from Quartus shell):
#   quartus_sh -t generate_clean_designs.tcl \
#       <project_name> <revision_name> <device_part> <out_dir>
#
# NOTE:
#   - This script assumes the Quartus project already exists
#     with the given project + revision.
#   - Claude should flesh out error handling and more parameters.

if { $argc < 4 } {
    puts "Usage: quartus_sh -t generate_clean_designs.tcl <project> <revision> <device> <out_dir>"
    qexit -error
}

set project_name  [lindex $argv 0]
set revision_name [lindex $argv 1]
set device_part   [lindex $argv 2]
set out_dir       [file normalize [lindex $argv 3]]

file mkdir $out_dir

# List of fitter seeds or parameter variations
# Claude: feel free to expand this, or make it a command-line parameter.
set seeds {1 2 3 4 5 6 7 8 9 10}

load_package flow

foreach seed $seeds {
    set rev "${revision_name}_seed${seed}"

    # Create a temporary revision that reuses the same sources, only changes the fitter seed
    project_open -revision $revision_name $project_name

    # Clone the base revision to a new temporary revision
    # NOTE: Claude: consider using 'project_copy' or 'set_global_assignment' as needed.
    set_global_assignment -name REVISION_NAME $rev
    set_global_assignment -name DEVICE $device_part
    set_global_assignment -name SEED $seed

    project_close

    # Run full compile
    execute_flow -compile $project_name -revision $rev

    # Copy resulting .sof into out_dir with a descriptive name
    set sof_file "${project_name}.${rev}.sof"
    set src_path [file join [pwd] $sof_file]
    if { [file exists $src_path] } {
        set dst_path [file join $out_dir $sof_file]
        file copy -force $src_path $dst_path
        puts "Wrote clean bitstream: $dst_path"
    } else {
        puts "WARNING: .sof not found for $rev"
    }
}
