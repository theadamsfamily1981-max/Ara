#!/usr/bin/env python3
"""
extract_sof_sections.py

Goal:
    Parse an Intel .sof (SRAM Object File) and extract the "design" section
    that actually configures the FPGA fabric, stripping SDM firmware headers
    and other non-configuration chunks.

Status:
    - Initial version can treat the .sof as a raw binary blob and pass it
      through unchanged (identity mapping).
    - Claude should incrementally improve this to understand the real .sof
      container format (based on Intel docs + experiments).

Usage:
    python extract_sof_sections.py --input input.sof --output design.bin
"""

import argparse
from pathlib import Path


def extract_design_section(raw: bytes) -> bytes:
    """
    Placeholder extraction logic.

    For now:
        - Return the entire file.
    Later:
        - Parse .sof header structure.
        - Identify pointer block (e.g., 0x18 bytes), SDM firmware vs design section.
        - Return only the design payload.

    Claude:
        - Use the research notes + Intel Stratix 10 / Arria 10 config docs
          to implement real parsing.
    """
    return raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw = args.input.read_bytes()
    design = extract_design_section(raw)
    args.output.write_bytes(design)
    print(f"Wrote design section: {args.output} (len={len(design)} bytes)")


if __name__ == "__main__":
    main()
