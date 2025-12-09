#!/usr/bin/env quartus_sh -t
#
# Generate Clean Bitstreams - Seed Variation
#
# Compiles a golden design multiple times with different fitter seeds to
# generate topological diversity. Essential for preventing ML overfitting.
#
# Usage:
#   quartus_sh -t generate_clean.tcl <project_name> <output_dir> <num_seeds>
#
# Arguments:
#   project_name: Name of Quartus project (.qpf)
#   output_dir: Directory to save generated .rbf files
#   num_seeds: Number of seed variations to generate
#
# Example:
#   quartus_sh -t generate_clean.tcl golden_design data/raw/arria10/clean 10
#
# Output:
#   data/raw/arria10/clean/seed_001/design.rbf
#   data/raw/arria10/clean/seed_002/design.rbf
#   ...
#
# Each seed produces a completely different physical layout (placement & routing)
# from the same HDL source, creating training diversity.

package require ::quartus::project
package require ::quartus::flow

# Parse arguments
if {$argc < 3} {
    puts "ERROR: Insufficient arguments"
    puts "Usage: quartus_sh -t generate_clean.tcl <project> <output_dir> <num_seeds>"
    exit 1
}

set project_name [lindex $argv 0]
set output_dir [lindex $argv 1]
set num_seeds [lindex $argv 2]

puts "========================================"
puts "Clean Bitstream Generation"
puts "========================================"
puts "Project:    $project_name"
puts "Output:     $output_dir"
puts "Seeds:      $num_seeds"
puts ""

# Verify project exists
if {![file exists "${project_name}.qpf"]} {
    puts "ERROR: Project file ${project_name}.qpf not found"
    exit 1
}

# Create output directory
file mkdir $output_dir

# Generate bitstreams for each seed
for {set seed 1} {$seed <= $num_seeds} {incr seed} {
    puts "----------------------------------------"
    puts "Seed $seed / $num_seeds"
    puts "----------------------------------------"

    # Create seed-specific output directory
    set seed_padded [format "%03d" $seed]
    set seed_dir "${output_dir}/seed_${seed_padded}"
    file mkdir $seed_dir

    # Open project
    project_open $project_name -current_revision

    # Set fitter seed
    set_global_assignment -name SEED $seed
    puts "  Set SEED = $seed"

    # Disable bitstream compression (critical for Arria 10 image analysis)
    set_global_assignment -name GENERATE_COMPRESSED_SOF OFF
    puts "  Disabled bitstream compression"

    # Run full compilation
    puts "  Running compilation..."
    if {[catch {execute_flow -compile} result]} {
        puts "  ERROR: Compilation failed"
        puts "  $result"
        project_close
        continue
    }

    # Run Assembler (generates .sof)
    puts "  Running Assembler..."
    if {[catch {execute_module -tool asm} result]} {
        puts "  ERROR: Assembler failed"
        puts "  $result"
        project_close
        continue
    }

    # Convert .sof to .rbf
    set sof_file "output_files/${project_name}.sof"
    set rbf_file "${seed_dir}/design.rbf"

    puts "  Converting to RBF..."
    if {[catch {
        execute_module -tool cpf \
            -c \
            -o bitstream_compression=off \
            $sof_file \
            $rbf_file
    } result]} {
        puts "  ERROR: SOF to RBF conversion failed"
        puts "  $result"
        project_close
        continue
    }

    # Save metadata
    set metadata_file "${seed_dir}/metadata.json"
    set metadata_fh [open $metadata_file w]
    puts $metadata_fh "\{"
    puts $metadata_fh "  \"project\": \"$project_name\","
    puts $metadata_fh "  \"seed\": $seed,"
    puts $metadata_fh "  \"label\": \"clean\","
    puts $metadata_fh "  \"device\": \"[get_global_assignment -name DEVICE]\","
    puts $metadata_fh "  \"timestamp\": \"[clock format [clock seconds] -format {%Y-%m-%d %H:%M:%S}]\","
    puts $metadata_fh "  \"bitstream\": \"design.rbf\""
    puts $metadata_fh "\}"
    close $metadata_fh

    project_close

    puts "  ✓ Complete: $rbf_file"
}

puts ""
puts "========================================"
puts "Summary"
puts "========================================"
puts "Generated $num_seeds clean bitstreams"
puts "Output: $output_dir"
puts ""
puts "Next: Run inject_trojan.tcl to generate infected variants"
