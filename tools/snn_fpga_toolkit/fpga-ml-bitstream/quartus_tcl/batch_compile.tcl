# batch_compile.tcl
#
# Thin wrapper to:
#   1) generate multiple clean designs
#   2) inject Trojans using ECO
#   3) emit a manifest (CSV) with labels.
#
# Usage:
#   quartus_sh -t batch_compile.tcl <project> <revision> <device> <out_dir>
#
# Claude: this is optional sugar to orchestrate the other scripts.

if { $argc < 4 } {
    puts "Usage: quartus_sh -t batch_compile.tcl <project> <revision> <device> <out_dir>"
    qexit -error
}

set project_name   [lindex $argv 0]
set base_revision  [lindex $argv 1]
set device_part    [lindex $argv 2]
set out_root       [file normalize [lindex $argv 3]]

set clean_dir   [file join $out_root "clean"]
set trojan_dir  [file join $out_root "trojan"]
file mkdir $clean_dir
file mkdir $trojan_dir

# 1) Call generate_clean_designs.tcl
exec quartus_sh -t [file join [pwd] "quartus_tcl/generate_clean_designs.tcl"] \
    $project_name $base_revision $device_part $clean_dir

# 2) For each clean design, generate a Trojan variant
# Claude: list clean .sof files and call inject_trojan_eco.tcl with appropriate args.
# Optionally create a CSV manifest with columns: filename,label

# This script is a placeholder; Claude should implement the loop in Tcl or
# move the orchestration into Python if more convenient.

puts "Batch compile complete. Clean designs in $clean_dir, Trojans in $trojan_dir"
