#!/usr/bin/env quartus_sh -t
#
# Batch Dataset Generation Orchestrator
#
# Orchestrates the complete synthetic data generation pipeline:
#   1. Generate clean bitstreams with seed variations
#   2. Generate infected bitstreams with Trojan injections
#
# Usage:
#   quartus_sh -t batch_run.tcl <project> <output_root> <num_clean> <num_infected>
#
# Example:
#   quartus_sh -t batch_run.tcl golden_design data/raw/arria10 50 50
#
# This will create:
#   data/raw/arria10/clean/seed_001/design.rbf
#   data/raw/arria10/clean/seed_002/design.rbf
#   ...
#   data/raw/arria10/infected/timebomb_001/design.rbf
#   data/raw/arria10/infected/comparator_001/design.rbf
#   ...

if {$argc < 4} {
    puts "ERROR: Insufficient arguments"
    puts "Usage: quartus_sh -t batch_run.tcl <project> <output_root> <num_clean> <num_infected>"
    exit 1
}

set project_name [lindex $argv 0]
set output_root [lindex $argv 1]
set num_clean [lindex $argv 2]
set num_infected [lindex $argv 3]

set script_dir [file dirname [info script]]

puts "========================================"
puts "Batch Dataset Generation"
puts "========================================"
puts "Project:         $project_name"
puts "Output Root:     $output_root"
puts "Clean Samples:   $num_clean"
puts "Infected Samples: $num_infected"
puts ""

# Create output directories
file mkdir "${output_root}/clean"
file mkdir "${output_root}/infected"

# Phase 1: Generate clean bitstreams
puts "========================================"
puts "Phase 1: Clean Bitstreams"
puts "========================================"
puts ""

if {[catch {
    exec quartus_sh -t "${script_dir}/generate_clean.tcl" \
        $project_name \
        "${output_root}/clean" \
        $num_clean
} result]} {
    puts "ERROR: Clean generation failed"
    puts $result
    exit 1
}

# Phase 2: Generate infected bitstreams
puts ""
puts "========================================"
puts "Phase 2: Infected Bitstreams"
puts "========================================"
puts ""

# Distribute infected samples across Trojan types
set trojan_types [list timebomb comparator ringoscillator]
set samples_per_type [expr {$num_infected / [llength $trojan_types]}]

foreach trojan_type $trojan_types {
    puts "\nGenerating $samples_per_type x $trojan_type Trojans..."

    for {set i 0} {$i < $samples_per_type} {incr i} {
        if {[catch {
            exec quartus_sh -t "${script_dir}/inject_trojan.tcl" \
                $project_name \
                $trojan_type \
                "${output_root}/infected"
        } result]} {
            puts "  WARNING: Trojan injection failed for sample $i"
            puts "  $result"
            continue
        }
    }
}

puts ""
puts "========================================"
puts "Batch Complete"
puts "========================================"
puts "Clean samples:    $num_clean"
puts "Infected samples: $num_infected"
puts "Output:           $output_root"
puts ""
puts "Next steps:"
puts "  1. python cli/preprocess_dataset.py --input $output_root --output data/images/arria10"
puts "  2. python cli/train_model.py --data data/images/arria10 --model arria10_cnn"
