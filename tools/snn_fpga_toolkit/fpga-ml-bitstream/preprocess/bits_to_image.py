#!/usr/bin/env python3
"""
bits_to_image.py

Thin wrapper that:
    - Loads bits from .npz
    - Uses a specified width (from CLI or file)
    - Writes a PNG image.

Usage:
    python bits_to_image.py \
        --input example_bits.npz \
        --width 2048 \
        --output example.png
"""

import argparse
from pathlib import Path
import numpy as np
from skimage.io import imsave


def bits_to_image(bits, width: int):
    effective_len = (bits.size // width) * width
    arr = bits[:effective_len].reshape(-1, width)
    img = (arr * 255).astype("uint8")
    return img


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, required=True)
    args = parser.parse_args()

    data = np.load(args.input, allow_pickle=True)
    bits = data["bits"]

    img = bits_to_image(bits, args.width)
    imsave(args.output, img)
    print(f"Wrote image {args.output} with width={args.width}")


if __name__ == "__main__":
    main()
