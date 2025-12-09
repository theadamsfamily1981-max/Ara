#!/usr/bin/env python3
"""
guess_width_autocorr.py

Given a .npz file with a 1D bit vector, heuristically guess a good 2D width
for reshaping into an image.

Idea:
    - Candidate widths in some range (e.g. 128..8192).
    - For each width:
        * Truncate bits to a multiple of width.
        * Reshape to (H, W).
        * Compute:
            - Column-wise entropy
            - Autocorrelation score along rows/cols
        * Choose the width that maximizes "structure" (low entropy, high periodicity).

This is intentionally heuristic; exact frame sizes are proprietary.

Usage:
    python guess_width_autocorr.py \
        --input example_bits.npz \
        --output example.png \
        --width-out example_width.txt
"""

import argparse
from pathlib import Path
import numpy as np
from skimage.io import imsave


def score_width(bits: np.ndarray, width: int) -> float:
    """
    Compute a scalar "structure score" for a given width.

    Claude:
        - Implement something stable here:
            * reshape to (H, width)
            * compute column sums or entropy
            * compute autocorrelation of each row or column
            * combine into a single score.

    For now:
        - Use variance of column-wise means as a crude measure.
    """
    length = bits.size
    effective_len = (length // width) * width
    if effective_len < width * 8:  # require at least 8 rows
        return -1e9

    arr = bits[:effective_len].reshape(-1, width)
    col_mean = arr.mean(axis=0)
    return float(col_mean.var())


def guess_width(bits: np.ndarray, min_width: int = 128, max_width: int = 8192) -> int:
    best_w = None
    best_score = -1e18

    for w in range(min_width, max_width + 1, 64):
        s = score_width(bits, w)
        if s > best_score:
            best_score = s
            best_w = w

    return best_w


def bits_to_image(bits: np.ndarray, width: int) -> np.ndarray:
    effective_len = (bits.size // width) * width
    arr = bits[:effective_len].reshape(-1, width)
    # Map {0,1} -> {0,255} uint8 image
    img = (arr * 255).astype(np.uint8)
    return img


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output PNG path")
    parser.add_argument("--width-out", type=Path, required=False, help="Write chosen width here")
    parser.add_argument("--min-width", type=int, default=128)
    parser.add_argument("--max-width", type=int, default=4096)
    args = parser.parse_args()

    data = np.load(args.input, allow_pickle=True)
    bits = data["bits"]

    width = guess_width(bits, args.min_width, args.max_width)
    img = bits_to_image(bits, width)
    imsave(args.output, img)

    if args.width_out:
        args.width_out.write_text(str(width))

    print(f"Guessed width={width}, wrote image {args.output}")


if __name__ == "__main__":
    main()
