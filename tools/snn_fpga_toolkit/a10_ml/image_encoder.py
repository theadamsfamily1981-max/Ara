"""
image_encoder.py

Convert FPGA bitstreams to 2D images for CNN training.

The bitstream is treated as a 1D sequence of bits and reshaped into a 2D array
(image) with a specified width. The resulting image can be saved as:
- NumPy array (.npy) for ML training
- PNG image for visualization
- Binary format for verification
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import json


def bitstream_to_image(
    bits: np.ndarray,
    width: int,
    height: Optional[int] = None,
    pad: bool = True
) -> np.ndarray:
    """
    Convert 1D bitstream to 2D image.

    Args:
        bits: 1D array of bits (0 or 1)
        width: Image width in bits
        height: Image height (optional, auto-computed if None)
        pad: If True, pad with zeros to fill last row

    Returns:
        2D array of shape (height, width)

    Example:
        >>> bits = np.array([1,0,1,0,1,0,1,0,1,0,1,0])
        >>> img = bitstream_to_image(bits, width=3)
        >>> print(img.shape)  # (4, 3)
    """
    if height is None:
        # Auto-compute height
        height = len(bits) // width
        remainder = len(bits) % width

        if remainder > 0:
            if pad:
                # Pad to fill last row
                padding = width - remainder
                bits = np.concatenate([bits, np.zeros(padding, dtype=bits.dtype)])
                height += 1
            else:
                # Truncate last incomplete row
                bits = bits[:height * width]

    # Reshape to 2D
    total_bits = height * width
    image = bits[:total_bits].reshape(height, width)

    print(f"🖼️  Created {height}x{width} image ({total_bits:,} bits)")

    return image


def image_to_bitstream(image: np.ndarray) -> np.ndarray:
    """
    Convert 2D image back to 1D bitstream.

    Args:
        image: 2D array of bits

    Returns:
        1D array of bits
    """
    return image.flatten()


def save_image_dataset(
    images: List[np.ndarray],
    labels: List[int],
    output_dir: Path,
    prefix: str = "bitstream"
):
    """
    Save multiple images as a dataset for ML training.

    Creates directory structure:
        output_dir/
            images/
                bitstream_0000.npy
                bitstream_0001.npy
                ...
            labels.json

    Args:
        images: List of 2D bit arrays
        labels: List of integer labels (0=clean, 1=trojan)
        output_dir: Directory to save dataset
        prefix: Filename prefix for images
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Save each image
    image_files = []
    for i, (img, label) in enumerate(zip(images, labels)):
        filename = f"{prefix}_{i:04d}.npy"
        filepath = images_dir / filename
        np.save(filepath, img)
        image_files.append({
            'filename': filename,
            'label': label,
            'shape': list(img.shape),
            'size_bits': int(img.size)
        })

    # Save labels and metadata
    labels_file = output_dir / "labels.json"
    metadata = {
        'num_images': len(images),
        'num_clean': sum(1 for l in labels if l == 0),
        'num_trojan': sum(1 for l in labels if l == 1),
        'images': image_files
    }

    with open(labels_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"💾 Saved {len(images)} images to {output_dir}")
    print(f"   Clean: {metadata['num_clean']}, Trojan: {metadata['num_trojan']}")


def bits_to_grayscale_png(
    bits: np.ndarray,
    width: int,
    output_path: Path,
    scale: int = 8
):
    """
    Save bitstream as grayscale PNG for visualization.

    Bits are packed into bytes (8 bits = 1 grayscale pixel) for efficient storage.

    Args:
        bits: 1D array of bits
        width: Image width in bits (must be multiple of 8 for byte alignment)
        output_path: Path to save PNG
        scale: Upscaling factor for visualization (default: 8x)

    Requires:
        Pillow (PIL)
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow required for PNG export: pip install Pillow")

    # Ensure width is multiple of 8
    if width % 8 != 0:
        new_width = (width // 8 + 1) * 8
        print(f"⚠️  Rounding width {width} → {new_width} for byte alignment")
        width = new_width

    # Convert to 2D
    image = bitstream_to_image(bits, width, pad=True)

    # Pack bits into bytes (8 bits → 1 uint8 pixel)
    height, width_bits = image.shape
    width_bytes = width_bits // 8

    # Reshape to (height, width_bytes, 8) and pack
    image_reshaped = image.reshape(height, width_bytes, 8)
    grayscale = np.packbits(image_reshaped, axis=2).squeeze(-1)

    # Create PIL image
    img = Image.fromarray(grayscale, mode='L')

    # Upscale for visibility
    if scale > 1:
        new_size = (width_bytes * scale, height * scale)
        img = img.resize(new_size, Image.NEAREST)

    img.save(output_path)
    print(f"💾 Saved visualization to {output_path}")


def bits_to_rgb_png(
    bits: np.ndarray,
    width: int,
    output_path: Path,
    colormap: str = 'viridis'
):
    """
    Save bitstream as colored PNG using matplotlib colormap.

    Useful for visualizing patterns and structures.

    Args:
        bits: 1D array of bits
        width: Image width in bits
        output_path: Path to save PNG
        colormap: Matplotlib colormap name

    Requires:
        matplotlib
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib import cm
    except ImportError:
        raise ImportError("matplotlib required for colored PNG: pip install matplotlib")

    # Convert to 2D
    image = bitstream_to_image(bits, width, pad=True)

    # Apply colormap
    cmap = cm.get_cmap(colormap)
    colored = cmap(image.astype(float))

    # Save with matplotlib
    plt.imsave(output_path, colored)
    print(f"💾 Saved colored visualization to {output_path}")


def calculate_image_stats(image: np.ndarray) -> dict:
    """
    Calculate statistics for a bitstream image.

    Args:
        image: 2D bit array

    Returns:
        Dictionary with statistics
    """
    flat = image.flatten()

    stats = {
        'shape': list(image.shape),
        'total_bits': int(image.size),
        'num_ones': int(np.sum(flat)),
        'num_zeros': int(image.size - np.sum(flat)),
        'density': float(np.mean(flat)),  # Fraction of 1s
        'row_entropies': {
            'mean': float(np.mean([_row_entropy(row) for row in image])),
            'std': float(np.std([_row_entropy(row) for row in image])),
            'min': float(np.min([_row_entropy(row) for row in image])),
            'max': float(np.max([_row_entropy(row) for row in image])),
        },
        'col_entropies': {
            'mean': float(np.mean([_row_entropy(col) for col in image.T])),
            'std': float(np.std([_row_entropy(col) for col in image.T])),
        }
    }

    return stats


def _row_entropy(row: np.ndarray) -> float:
    """Calculate Shannon entropy of a binary row."""
    counts = np.bincount(row.astype(int), minlength=2)
    total = len(row)

    entropy = 0.0
    for count in counts:
        if count > 0:
            p = count / total
            entropy -= p * np.log2(p)

    return entropy


def compare_images(img1: np.ndarray, img2: np.ndarray) -> dict:
    """
    Compare two bitstream images (e.g., clean vs trojan).

    Args:
        img1: First image (2D bit array)
        img2: Second image (2D bit array)

    Returns:
        Dictionary with comparison metrics
    """
    if img1.shape != img2.shape:
        return {'error': 'Images have different shapes'}

    # Bit-level differences
    diff = np.abs(img1 - img2)
    num_diffs = np.sum(diff)

    # Hamming distance
    hamming_distance = num_diffs / img1.size

    comparison = {
        'shape': list(img1.shape),
        'num_differences': int(num_diffs),
        'hamming_distance': float(hamming_distance),
        'percent_different': float(hamming_distance * 100),
        'identical': bool(num_diffs == 0)
    }

    return comparison


def create_diff_image(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    Create difference image showing where two bitstreams differ.

    Args:
        img1: First image (2D bit array)
        img2: Second image (2D bit array)

    Returns:
        Difference image (1 where different, 0 where same)
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Shape mismatch: {img1.shape} vs {img2.shape}")

    return np.abs(img1 - img2).astype(np.uint8)


# Example usage and testing
if __name__ == '__main__':
    print("=" * 60)
    print("A10 Image Encoder Test")
    print("=" * 60)
    print()

    # Generate synthetic bitstream
    width = 1024
    height = 500
    total_bits = width * height

    print(f"🔧 Generating synthetic bitstream ({width}x{height} = {total_bits:,} bits)")
    bits = np.random.randint(0, 2, total_bits, dtype=np.uint8)

    # Convert to image
    print("\n" + "=" * 60)
    print("Bitstream → Image Conversion")
    print("=" * 60)
    print()

    image = bitstream_to_image(bits, width)
    print(f"Image shape: {image.shape}")

    # Calculate stats
    print("\n" + "=" * 60)
    print("Image Statistics")
    print("=" * 60)
    print()

    stats = calculate_image_stats(image)
    print(f"Total bits:      {stats['total_bits']:,}")
    print(f"Ones:            {stats['num_ones']:,} ({stats['density']*100:.2f}%)")
    print(f"Zeros:           {stats['num_zeros']:,}")
    print(f"Row entropy:     {stats['row_entropies']['mean']:.4f} ± {stats['row_entropies']['std']:.4f}")
    print(f"Column entropy:  {stats['col_entropies']['mean']:.4f} ± {stats['col_entropies']['std']:.4f}")

    # Test round-trip conversion
    print("\n" + "=" * 60)
    print("Round-Trip Test")
    print("=" * 60)
    print()

    bits_recovered = image_to_bitstream(image)
    matches = np.array_equal(bits, bits_recovered)
    print(f"Original bits:   {len(bits):,}")
    print(f"Recovered bits:  {len(bits_recovered):,}")
    print(f"Match:           {matches} ✅" if matches else f"Match:           {matches} ❌")

    # Test dataset saving
    print("\n" + "=" * 60)
    print("Dataset Creation")
    print("=" * 60)
    print()

    output_dir = Path("/tmp/a10_test_dataset")
    images = [image, image * 0]  # Two images: random and all-zeros
    labels = [0, 1]  # 0=clean, 1=trojan

    save_image_dataset(images, labels, output_dir, prefix="test")
    print(f"✅ Dataset saved to {output_dir}")
