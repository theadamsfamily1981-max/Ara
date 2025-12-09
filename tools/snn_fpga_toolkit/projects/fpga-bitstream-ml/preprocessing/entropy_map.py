#!/usr/bin/env python3
"""
Entropy Mapping for FPGA Bitstreams

Converts 1D bitstreams into 2D entropy maps suitable for CNN analysis.

The core insight: FPGA configuration data exhibits spatial texture patterns.
- Empty regions: Low entropy (all zeros)
- Regular logic (LABs): Medium entropy (structured patterns)
- Dense routing: High entropy (complex interconnects)
- Hardware Trojans: Anomalous local entropy spikes

Process:
1. Fold 1D bitstream into 2D array using detected width
2. Compute local Shannon entropy in sliding windows
3. Normalize and save as grayscale image

This creates visual "heatmaps" where Trojans appear as disruptions in the
regular texture pattern of benign logic.

Reference: Section 3.1.2 of bitstream forensics paper
"""

import numpy as np
from typing import Tuple, Optional
from pathlib import Path


def fold_to_2d(bits: np.ndarray, width: int) -> np.ndarray:
    """
    Fold 1D bit array into 2D image with specified width.

    Args:
        bits: 1D bit array (0/1 values)
        width: Image width in bits (detected via autocorrelation)

    Returns:
        2D array of shape (height, width) with uint8 values (0 or 1)

    Example:
        >>> bits = np.array([1,0,1,0,1,0,1,0,1,0,1,0], dtype=np.uint8)
        >>> img = fold_to_2d(bits, width=3)
        >>> img.shape
        (4, 3)
    """
    # Truncate to multiple of width
    num_rows = len(bits) // width
    truncated_length = num_rows * width

    if truncated_length < len(bits):
        bits = bits[:truncated_length]

    # Reshape into 2D
    image = bits.reshape(num_rows, width)

    return image.astype(np.uint8)


def compute_entropy_map(
    bit_image: np.ndarray,
    window_size: int = 16,
    stride: int = 4,
    normalize: bool = True
) -> np.ndarray:
    """
    Compute local Shannon entropy map over 2D bit image.

    Entropy H = -Σ p(x) log₂ p(x) measures information content.
    For bitstreams:
    - H ≈ 0: Constant data (zeros or ones)
    - H ≈ 1: Maximum randomness for binary data

    Args:
        bit_image: 2D bit array (0/1 values)
        window_size: Size of sliding window (bits)
        stride: Step size for sliding window
        normalize: Scale to [0, 255] for image output

    Returns:
        2D entropy map (float32 if not normalized, uint8 if normalized)

    Example:
        >>> img = np.random.randint(0, 2, (1024, 1024), dtype=np.uint8)
        >>> entropy = compute_entropy_map(img, window_size=16, stride=4)
        >>> entropy.shape
        (252, 252)  # Reduced by windowing
    """
    height, width = bit_image.shape

    # Calculate output dimensions
    out_height = (height - window_size) // stride + 1
    out_width = (width - window_size) // stride + 1

    # Pre-allocate entropy map
    entropy_map = np.zeros((out_height, out_width), dtype=np.float32)

    # Slide window over image
    for i in range(out_height):
        y_start = i * stride
        y_end = y_start + window_size

        for j in range(out_width):
            x_start = j * stride
            x_end = x_start + window_size

            # Extract window
            window = bit_image[y_start:y_end, x_start:x_end]

            # Compute Shannon entropy
            entropy_map[i, j] = _shannon_entropy_binary(window)

    if normalize:
        # Normalize to [0, 255] for grayscale image
        # Binary data has max entropy of 1.0 bit
        entropy_map = (entropy_map * 255).astype(np.uint8)

    return entropy_map


def _shannon_entropy_binary(data: np.ndarray) -> float:
    """
    Compute Shannon entropy for binary (0/1) data.

    H = -p₀ log₂(p₀) - p₁ log₂(p₁)

    Args:
        data: Binary array (0s and 1s)

    Returns:
        Entropy in bits (range [0, 1] for binary data)
    """
    # Count occurrences
    count_0 = np.sum(data == 0)
    count_1 = np.sum(data == 1)
    total = data.size

    if total == 0:
        return 0.0

    # Compute probabilities
    p0 = count_0 / total
    p1 = count_1 / total

    # Shannon entropy (handle zero probabilities)
    entropy = 0.0
    if p0 > 0:
        entropy -= p0 * np.log2(p0)
    if p1 > 0:
        entropy -= p1 * np.log2(p1)

    return entropy


def save_entropy_png(
    entropy_map: np.ndarray,
    output_path: str,
    colormap: str = 'grayscale'
):
    """
    Save entropy map as PNG image.

    Args:
        entropy_map: 2D entropy array (uint8 or float32)
        output_path: Output file path (.png)
        colormap: Color mapping ('grayscale', 'viridis', 'hot')

    Colormaps:
        - grayscale: Black = low entropy, White = high entropy
        - viridis: Blue = low, Yellow = high (good for visual analysis)
        - hot: Black → Red → Yellow → White (highlight hotspots)
    """
    try:
        from PIL import Image
        import matplotlib.pyplot as plt
        from matplotlib import cm
    except ImportError:
        raise ImportError(
            "Pillow and matplotlib required. "
            "Install with: pip install Pillow matplotlib"
        )

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Normalize to uint8 if needed
    if entropy_map.dtype != np.uint8:
        entropy_map = ((entropy_map / entropy_map.max()) * 255).astype(np.uint8)

    if colormap == 'grayscale':
        # Direct grayscale save
        img = Image.fromarray(entropy_map, mode='L')
        img.save(output_path)
    else:
        # Apply matplotlib colormap
        if colormap == 'viridis':
            cmap = cm.viridis
        elif colormap == 'hot':
            cmap = cm.hot
        else:
            raise ValueError(f"Unknown colormap: {colormap}")

        # Normalize to [0, 1]
        normalized = entropy_map.astype(np.float32) / 255.0

        # Apply colormap
        colored = cmap(normalized)

        # Convert to uint8 RGB
        rgb = (colored[:, :, :3] * 255).astype(np.uint8)

        # Save
        img = Image.fromarray(rgb, mode='RGB')
        img.save(output_path)

    print(f"Saved entropy map to: {output_path}")


def visualize_entropy_comparison(
    bit_image: np.ndarray,
    entropy_map: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Create side-by-side comparison of raw bitstream and entropy map.

    Useful for visual verification that entropy mapping is working correctly.

    Args:
        bit_image: Original 2D bit image
        entropy_map: Computed entropy map
        save_path: Optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required. Install with: pip install matplotlib")

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Raw bitstream (subsample if too large)
    if bit_image.shape[0] > 2048 or bit_image.shape[1] > 2048:
        subsample = (bit_image.shape[0] // 2048 + 1, bit_image.shape[1] // 2048 + 1)
        display_bits = bit_image[::subsample[0], ::subsample[1]]
    else:
        display_bits = bit_image

    axes[0].imshow(display_bits, cmap='gray', interpolation='nearest')
    axes[0].set_title('Raw Bitstream (2D Folded)')
    axes[0].set_xlabel('Bit Column')
    axes[0].set_ylabel('Bit Row')

    # Entropy map
    im = axes[1].imshow(entropy_map, cmap='viridis', interpolation='nearest')
    axes[1].set_title('Local Entropy Map')
    axes[1].set_xlabel('Window Column')
    axes[1].set_ylabel('Window Row')

    # Colorbar
    cbar = plt.colorbar(im, ax=axes[1])
    cbar.set_label('Entropy (bits)', rotation=270, labelpad=20)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to: {save_path}")
    else:
        plt.show()


def compute_entropy_statistics(entropy_map: np.ndarray) -> dict:
    """
    Compute statistical summary of entropy map.

    Useful for automated anomaly detection without visual inspection.

    Returns dictionary with:
        - mean: Average entropy
        - std: Standard deviation
        - min, max: Range
        - quartiles: 25th, 50th, 75th percentiles
        - high_entropy_fraction: Fraction of windows with H > 0.8
    """
    stats = {
        'mean': float(np.mean(entropy_map)),
        'std': float(np.std(entropy_map)),
        'min': float(np.min(entropy_map)),
        'max': float(np.max(entropy_map)),
        'q25': float(np.percentile(entropy_map, 25)),
        'q50': float(np.percentile(entropy_map, 50)),  # Median
        'q75': float(np.percentile(entropy_map, 75))
    }

    # Normalize to [0, 1] if needed
    if entropy_map.dtype == np.uint8:
        normalized = entropy_map.astype(np.float32) / 255.0
    else:
        normalized = entropy_map

    # High entropy threshold (H > 0.8 indicates dense, complex regions)
    high_entropy_mask = normalized > 0.8
    stats['high_entropy_fraction'] = float(np.mean(high_entropy_mask))

    return stats


def detect_entropy_anomalies(
    entropy_map: np.ndarray,
    threshold: float = 3.0
) -> Tuple[np.ndarray, int]:
    """
    Detect local anomalies in entropy map using Z-score thresholding.

    Anomalies are regions with significantly higher entropy than the
    surrounding area, which may indicate:
    - Hardware Trojans (dense, unexpected logic)
    - Routing congestion (complex interconnects)
    - Design errors (unintended complexity)

    Args:
        entropy_map: 2D entropy array
        threshold: Z-score threshold (default 3.0 = 99.7th percentile)

    Returns:
        Tuple of (anomaly_mask, num_anomalies)
        - anomaly_mask: Boolean 2D array (True = anomaly)
        - num_anomalies: Count of anomalous windows
    """
    # Normalize to [0, 1]
    if entropy_map.dtype == np.uint8:
        normalized = entropy_map.astype(np.float32) / 255.0
    else:
        normalized = entropy_map

    # Compute Z-scores
    mean = np.mean(normalized)
    std = np.std(normalized)

    z_scores = (normalized - mean) / (std + 1e-8)

    # Threshold
    anomaly_mask = np.abs(z_scores) > threshold
    num_anomalies = int(np.sum(anomaly_mask))

    return anomaly_mask, num_anomalies


if __name__ == '__main__':
    # CLI for testing
    import sys
    from rbf_utils import read_rbf, bytes_to_bits
    from autocorr_width import infer_frame_width

    if len(sys.argv) < 2:
        print("Usage: python entropy_map.py <bitstream.rbf> [--visualize]")
        print("\nGenerates entropy map from bitstream:")
        print("  1. Detects optimal frame width via autocorrelation")
        print("  2. Folds bitstream into 2D image")
        print("  3. Computes local Shannon entropy")
        print("  4. Saves as grayscale PNG")
        sys.exit(1)

    rbf_path = sys.argv[1]
    visualize = '--visualize' in sys.argv

    print(f"Loading bitstream: {rbf_path}")
    rbf_bytes = read_rbf(rbf_path)
    bits = bytes_to_bits(rbf_bytes)

    print(f"Bitstream size: {len(bits):,} bits")

    # Step 1: Detect width
    print("\n[1/4] Detecting frame width via autocorrelation...")
    width = infer_frame_width(bits)
    print(f"      Detected width: {width} bits")

    # Step 2: Fold to 2D
    print(f"\n[2/4] Folding to 2D image...")
    bit_image = fold_to_2d(bits, width)
    print(f"      Image shape: {bit_image.shape} (height={bit_image.shape[0]}, width={width})")

    # Step 3: Compute entropy
    print(f"\n[3/4] Computing local entropy map (16x16 windows)...")
    entropy_map = compute_entropy_map(bit_image, window_size=16, stride=4, normalize=False)
    print(f"      Entropy map shape: {entropy_map.shape}")

    # Statistics
    stats = compute_entropy_statistics(entropy_map)
    print(f"\n      Entropy statistics:")
    print(f"        Mean:   {stats['mean']:.4f} bits")
    print(f"        Std:    {stats['std']:.4f} bits")
    print(f"        Range:  [{stats['min']:.4f}, {stats['max']:.4f}]")
    print(f"        Median: {stats['q50']:.4f} bits")
    print(f"        High-entropy regions: {stats['high_entropy_fraction']*100:.2f}%")

    # Anomaly detection
    anomaly_mask, num_anomalies = detect_entropy_anomalies(entropy_map, threshold=3.0)
    print(f"\n      Anomaly detection (Z-score > 3.0):")
    print(f"        Anomalous windows: {num_anomalies} ({num_anomalies/entropy_map.size*100:.2f}%)")

    # Step 4: Save
    print(f"\n[4/4] Saving entropy map...")
    output_path = Path(rbf_path).stem + '_entropy.png'
    save_entropy_png(entropy_map, output_path, colormap='viridis')

    if visualize:
        print("\nGenerating visualization...")
        visualize_entropy_comparison(
            bit_image,
            entropy_map,
            save_path=Path(rbf_path).stem + '_comparison.png'
        )

    print("\n✓ Complete!")
    print(f"  Entropy map: {output_path}")
    if visualize:
        print(f"  Comparison:  {Path(rbf_path).stem}_comparison.png")
