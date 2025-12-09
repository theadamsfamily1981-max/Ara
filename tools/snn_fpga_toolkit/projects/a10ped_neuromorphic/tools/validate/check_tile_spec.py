#!/usr/bin/env python3
"""
Validate tile YAML against schema

This tool performs comprehensive validation of neuromorphic AI tile
specifications to catch errors before attempting hardware builds.

Usage:
    python check_tile_spec.py specs/tiles/a10ped_tile.yaml

Validation checks:
- Required top-level fields
- CSR register uniqueness and validity
- Clock frequency constraints
- Memory configuration
- PCIe configuration

Author: Quanta Hardware Project
License: MIT
"""

import sys
import yaml
from pathlib import Path


def validate_tile_spec(tile_yaml_path):
    """Validate tile specification against schema"""
    errors = []
    warnings = []

    # Load YAML
    try:
        with open(tile_yaml_path, 'r') as f:
            tile = yaml.safe_load(f)
    except Exception as e:
        errors.append(f"Failed to parse YAML: {e}")
        return errors, warnings, None

    # Required top-level fields
    required_fields = [
        'tile_name', 'vendor', 'family', 'fpga_part',
        'memory', 'pcie', 'csr', 'clocks'
    ]

    for field in required_fields:
        if field not in tile:
            errors.append(f"Missing required field: {field}")

    # Validate vendor
    valid_vendors = ['intel', 'xilinx', 'lattice', 'microchip']
    if tile.get('vendor') not in valid_vendors:
        errors.append(f"Invalid vendor: {tile.get('vendor')} (must be one of {valid_vendors})")

    # Validate memory configuration
    if 'memory' in tile:
        memory = tile['memory']

        if 'type' not in memory:
            errors.append("Memory configuration missing 'type' field")
        elif memory['type'] not in ['ddr4', 'ddr3', 'hbm2', 'gddr6']:
            errors.append(f"Invalid memory type: {memory['type']}")

        if memory.get('size_gb', 0) <= 0:
            errors.append(f"Invalid memory size: {memory.get('size_gb')}")

        if memory.get('channels', 0) <= 0:
            errors.append(f"Invalid memory channels: {memory.get('channels')}")

    # Validate PCIe configuration
    if 'pcie' in tile:
        pcie = tile['pcie']

        if pcie.get('lanes') not in [1, 2, 4, 8, 16]:
            errors.append(f"Invalid PCIe lane count: {pcie.get('lanes')} (must be 1, 2, 4, 8, or 16)")

        if pcie.get('gen') not in [1, 2, 3, 4, 5]:
            errors.append(f"Invalid PCIe generation: {pcie.get('gen')} (must be 1-5)")

        # Validate BARs
        if 'bars' in pcie:
            seen_bars = set()
            for bar in pcie['bars']:
                bar_num = bar.get('number')
                if bar_num in seen_bars:
                    errors.append(f"Duplicate BAR number: {bar_num}")
                seen_bars.add(bar_num)

                if bar_num < 0 or bar_num > 5:
                    errors.append(f"Invalid BAR number: {bar_num} (must be 0-5)")

                if bar.get('size_kb', 0) <= 0:
                    errors.append(f"Invalid BAR size for BAR{bar_num}: {bar.get('size_kb')} KB")

    # Validate CSR registers
    if 'csr' in tile:
        csr = tile['csr']

        if 'regs' not in csr:
            errors.append("CSR configuration missing 'regs' field")
        else:
            seen_offsets = set()
            seen_names = set()

            for reg in csr['regs']:
                # Check for required register fields
                if 'name' not in reg:
                    errors.append("CSR register missing 'name' field")
                    continue

                reg_name = reg['name']

                # Check for duplicate names
                if reg_name in seen_names:
                    errors.append(f"Duplicate CSR register name: {reg_name}")
                seen_names.add(reg_name)

                # Check offset
                offset = reg.get('offset')
                if offset is None:
                    errors.append(f"Register {reg_name} missing offset")
                elif offset in seen_offsets:
                    errors.append(f"Duplicate CSR offset: 0x{offset:04X} (register {reg_name})")
                else:
                    seen_offsets.add(offset)

                # Check width
                width = reg.get('width', 0)
                if width not in [8, 16, 32, 64]:
                    errors.append(f"Invalid register width for {reg_name}: {width} (must be 8, 16, 32, or 64)")

                # Check access mode
                access = reg.get('access')
                if access not in ['rw', 'ro', 'wo']:
                    errors.append(f"Invalid access mode for {reg_name}: {access} (must be rw, ro, or wo)")

                # Warn if reset value is missing
                if 'reset_value' not in reg:
                    warnings.append(f"Register {reg_name} missing reset_value")

    # Validate clocks
    if 'clocks' in tile:
        for i, clk in enumerate(tile['clocks']):
            if 'name' not in clk:
                errors.append(f"Clock #{i} missing name")

            freq_mhz = clk.get('freq_mhz', 0)
            if freq_mhz <= 0:
                errors.append(f"Invalid clock frequency for {clk.get('name', f'clock #{i}')}: {freq_mhz} MHz")
            elif freq_mhz > 1000:
                warnings.append(f"Very high clock frequency for {clk.get('name')}: {freq_mhz} MHz")

            role = clk.get('role')
            if role not in ['fabric', 'pcie', 'mem', 'user']:
                warnings.append(f"Unknown clock role for {clk.get('name')}: {role}")

    # Validate SNN core (if present)
    if 'snn_core' in tile:
        snn = tile['snn_core']

        if 'top_module' not in snn:
            warnings.append("SNN core missing top_module field")

        if 'parameters' in snn:
            params = snn['parameters']

            neuron_count = params.get('neuron_count', 0)
            if neuron_count <= 0:
                errors.append(f"Invalid neuron count: {neuron_count}")

            precision = params.get('precision')
            if precision not in ['int8', 'int16', 'fp16', 'fp32']:
                warnings.append(f"Unknown precision: {precision}")

    return errors, warnings, tile


def main():
    """Main entry point"""
    if len(sys.argv) != 2:
        print("Usage: check_tile_spec.py <tile_yaml>", file=sys.stderr)
        print("", file=sys.stderr)
        print("Example:", file=sys.stderr)
        print("  python check_tile_spec.py specs/tiles/a10ped_tile.yaml", file=sys.stderr)
        sys.exit(1)

    tile_yaml_path = Path(sys.argv[1])

    if not tile_yaml_path.exists():
        print(f"❌ Error: File not found: {tile_yaml_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Validating: {tile_yaml_path}")
    print("")

    errors, warnings, tile = validate_tile_spec(tile_yaml_path)

    # Display results
    if errors:
        print("❌ Validation FAILED:")
        print("")
        for err in errors:
            print(f"  ERROR: {err}")
        print("")

    if warnings:
        print("⚠️  Warnings:")
        print("")
        for warn in warnings:
            print(f"  WARN: {warn}")
        print("")

    if not errors and not warnings:
        print("✅ Validation passed with no errors or warnings")
        print("")

    if not errors:
        if tile:
            print("Summary:")
            print(f"  Tile:        {tile.get('tile_name')}")
            print(f"  Vendor:      {tile.get('vendor')}")
            print(f"  Part:        {tile.get('fpga_part')}")
            print(f"  Memory:      {tile.get('memory', {}).get('type')} ({tile.get('memory', {}).get('size_gb')} GB)")
            print(f"  PCIe:        Gen{tile.get('pcie', {}).get('gen')} x{tile.get('pcie', {}).get('lanes')}")
            print(f"  Registers:   {len(tile.get('csr', {}).get('regs', []))}")
            print(f"  Clocks:      {len(tile.get('clocks', []))}")
            if 'snn_core' in tile:
                params = tile['snn_core'].get('parameters', {})
                print(f"  SNN Neurons: {params.get('neuron_count', 'N/A')}")
                print(f"  Precision:   {params.get('precision', 'N/A')}")

    # Exit with error code if validation failed
    sys.exit(0 if not errors else 1)


if __name__ == "__main__":
    main()
