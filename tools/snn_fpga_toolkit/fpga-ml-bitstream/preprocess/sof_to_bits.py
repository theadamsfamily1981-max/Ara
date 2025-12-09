#!/usr/bin/env python3
"""
sof_to_bits.py

Convert a .sof (or raw design binary) into a flat vector of bits and store it
in a .npz file.

Pipeline:
    .sof --> (optional extract_sof_sections) --> design bytes --> bit vector

Bit ordering:
    - For now: MSB-first for each byte.
      i.e., byte 0xA5 => bits [1,0,1,0,0,1,0,1]

Output:
    np.savez(output_path, bits=bits, meta=meta_dict)

Usage:
    python sof_to_bits.py --input example.sof --output example_bits.npz
"""

import argparse
from pathlib import Path
import numpy as np

from extract_sof_sections import extract_design_section  # same directory import


def bytes_to_bits(data: bytes) -> np.ndarray:
    """Convert bytes -> 1D np.uint8 array of {0,1} bits."""
    arr = np.frombuffer(data, dtype=np.uint8)
    # Expand each byte to 8 bits (MSB-first)
    bits = np.unpackbits(arr, bitorder="big")
    return bits.astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw = args.input.read_bytes()
    design = extract_design_section(raw)
    bits = bytes_to_bits(design)

    meta = {
        "input_file": str(args.input),
        "num_bytes": len(design),
        "num_bits": int(bits.size),
    }

    np.savez_compressed(args.output, bits=bits, meta=meta)
    print(f"Saved bits: {args.output} (num_bits={bits.size})")


if __name__ == "__main__":
    main()
