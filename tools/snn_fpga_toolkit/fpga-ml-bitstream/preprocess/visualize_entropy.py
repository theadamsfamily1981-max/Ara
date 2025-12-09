#!/usr/bin/env python3
"""
visualize_entropy.py

Compute and visualize local entropy of a bitstream image to help
identify anomalous regions (potential Trojans).

Usage:
    python visualize_entropy.py \
        --input example.png \
        --output example_entropy.png \
        --window-size 16
"""

import argparse
from pathlib import Path
import numpy as np
from skimage.io import imread, imsave
from scipy.stats import entropy
import matplotlib.pyplot as plt


def compute_local_entropy(img: np.ndarray, window_size: int = 16) -> np.ndarray:
    """
    Compute local Shannon entropy for each window in the image.

    Returns:
        entropy_map: 2D array of entropy values
    """
    h, w = img.shape
    entropy_map = np.zeros((h // window_size, w // window_size))

    for i in range(0, h - window_size + 1, window_size):
        for j in range(0, w - window_size + 1, window_size):
            window = img[i:i+window_size, j:j+window_size]
            hist, _ = np.histogram(window.flatten(), bins=256, range=(0, 255))
            hist = hist / hist.sum()  # normalize
            ent = entropy(hist + 1e-10)  # add small value to avoid log(0)
            entropy_map[i // window_size, j // window_size] = ent

    return entropy_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=16)
    args = parser.parse_args()

    img = imread(args.input, as_gray=True)
    img = (img * 255).astype(np.uint8)

    entropy_map = compute_local_entropy(img, args.window_size)

    # Visualize
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Bitstream Image')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(entropy_map, cmap='hot')
    plt.title(f'Local Entropy (window={args.window_size})')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved entropy visualization: {args.output}")


if __name__ == "__main__":
    main()
