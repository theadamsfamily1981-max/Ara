#!/usr/bin/env python3
"""
Generate Synthetic FPGA Bitstream Data for Testing

Creates realistic-looking .rbf files with spatial structure that mimics
real FPGA configuration bitstreams. Used for testing/validation when
real Quartus-generated bitstreams are not available.

Usage:
    python generate_synthetic_data.py --clean 10 --infected 10 --output test_data/raw
"""

import argparse
import struct
import random
from pathlib import Path


def create_sof_header():
    """Create a minimal SOF header (256 bytes)."""
    header = b'\x00\x09\x0f\xf0'  # SOF magic bytes
    header += struct.pack('>I', 0x00000001)  # Version
    header += struct.pack('>I', 0x00000100)  # Header length (256)
    header += b'\x00' * (0x100 - len(header))  # Pad to 256 bytes
    return header


def create_synthetic_bitstream(
    width: int = 1024,
    num_rows: int = 2048,
    seed: int = 42,
    trojan: bool = False
) -> bytes:
    """
    Create a synthetic FPGA bitstream with realistic spatial structure.

    Args:
        width: Frame width in bits
        num_rows: Number of rows
        seed: Random seed for reproducibility
        trojan: If True, insert anomalous "Trojan" region

    Returns:
        Binary data simulating an FPGA bitstream
    """
    random.seed(seed)

    # Convert width from bits to bytes
    width_bytes = width // 8

    # Create base pattern (simulates LAB columns)
    # Real FPGAs have repeating structures every ~64-128 bits
    column_period = 64  # bits
    col_period_bytes = column_period // 8

    bitstream_data = bytearray()

    # Create template pattern for one "column"
    template = [0] * width_bytes

    # Add column markers (simulates LAB boundaries)
    for col in range(0, width_bytes, col_period_bytes):
        if col + 4 <= width_bytes:
            # Marker pattern: alternating high/low entropy regions
            template[col] = 0xFF
            template[col+1] = 0x00
            template[col+2] = random.randint(0, 255)
            template[col+3] = random.randint(0, 255)

    # Fill rest with medium entropy (simulates configured logic)
    for i in range(width_bytes):
        if template[i] == 0:
            template[i] = random.randint(64, 192)

    # Generate rows with spatial correlation
    for row in range(num_rows):
        # Add small variation to template (simulates different LAB configs)
        row_data = []
        for val in template:
            noise = random.randint(-20, 20)
            new_val = max(0, min(255, val + noise))
            row_data.append(new_val)

        # Every N rows, refresh template (simulates different region types)
        if row % 128 == 0:
            template = row_data.copy()

        bitstream_data.extend(row_data)

    # If Trojan mode, insert anomalous region
    if trojan:
        # Insert dense, high-entropy "Trojan" block
        trojan_row_start = num_rows // 2
        trojan_row_count = 32  # Small anomalous region
        trojan_col_start = width_bytes // 2
        trojan_col_count = 64  # 64 bytes wide

        for row in range(trojan_row_start, trojan_row_start + trojan_row_count):
            offset = row * width_bytes + trojan_col_start

            # High entropy (random) block
            for i in range(trojan_col_count):
                if offset + i < len(bitstream_data):
                    bitstream_data[offset + i] = random.randint(0, 255)

    # Wrap in SOF format
    sof_header = create_sof_header()
    section_header = struct.pack('>I', len(bitstream_data))

    return sof_header + section_header + bytes(bitstream_data)


def generate_dataset(
    output_dir: Path,
    num_clean: int,
    num_infected: int,
    width: int = 1024,
    num_rows: int = 2048
):
    """Generate complete synthetic dataset."""

    print(f"Generating synthetic dataset...")
    print(f"  Output: {output_dir}")
    print(f"  Clean samples: {num_clean}")
    print(f"  Infected samples: {num_infected}")
    print(f"  Bitstream size: {width} bits × {num_rows} rows")
    print()

    # Create directories
    clean_dir = output_dir / 'clean'
    infected_dir = output_dir / 'infected'
    clean_dir.mkdir(parents=True, exist_ok=True)
    infected_dir.mkdir(parents=True, exist_ok=True)

    # Generate clean samples
    print("Generating clean samples...")
    for i in range(1, num_clean + 1):
        sample_dir = clean_dir / f'seed_{i:03d}'
        sample_dir.mkdir(exist_ok=True)

        # Generate bitstream
        bitstream = create_synthetic_bitstream(
            width=width,
            num_rows=num_rows,
            seed=i,
            trojan=False
        )

        # Save .rbf
        rbf_path = sample_dir / 'design.rbf'
        with open(rbf_path, 'wb') as f:
            f.write(bitstream)

        # Save metadata
        metadata = {
            'label': 'clean',
            'seed': i,
            'width': width,
            'num_rows': num_rows,
            'synthetic': True
        }

        import json
        metadata_path = sample_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"  ✓ {rbf_path} ({len(bitstream):,} bytes)")

    # Generate infected samples
    print(f"\nGenerating infected samples...")
    for i in range(1, num_infected + 1):
        sample_dir = infected_dir / f'trojan_{i:03d}'
        sample_dir.mkdir(exist_ok=True)

        # Generate bitstream with Trojan
        bitstream = create_synthetic_bitstream(
            width=width,
            num_rows=num_rows,
            seed=100 + i,  # Different seed space
            trojan=True
        )

        # Save .rbf
        rbf_path = sample_dir / 'design.rbf'
        with open(rbf_path, 'wb') as f:
            f.write(bitstream)

        # Save metadata
        metadata = {
            'label': 'infected',
            'trojan_type': 'synthetic_high_entropy',
            'seed': 100 + i,
            'width': width,
            'num_rows': num_rows,
            'synthetic': True
        }

        import json
        metadata_path = sample_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"  ✓ {rbf_path} ({len(bitstream):,} bytes)")

    print()
    print("=" * 60)
    print("Synthetic dataset generation complete!")
    print(f"Total samples: {num_clean + num_infected}")
    print(f"  Clean: {num_clean}")
    print(f"  Infected: {num_infected}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic FPGA bitstream data for testing'
    )

    parser.add_argument(
        '--clean',
        type=int,
        default=10,
        help='Number of clean samples (default: 10)'
    )

    parser.add_argument(
        '--infected',
        type=int,
        default=10,
        help='Number of infected samples (default: 10)'
    )

    parser.add_argument(
        '--output',
        required=True,
        help='Output directory'
    )

    parser.add_argument(
        '--width',
        type=int,
        default=1024,
        help='Bitstream width in bits (default: 1024)'
    )

    parser.add_argument(
        '--rows',
        type=int,
        default=2048,
        help='Number of rows (default: 2048)'
    )

    args = parser.parse_args()

    generate_dataset(
        output_dir=Path(args.output),
        num_clean=args.clean,
        num_infected=args.infected,
        width=args.width,
        num_rows=args.rows
    )


if __name__ == '__main__':
    main()
