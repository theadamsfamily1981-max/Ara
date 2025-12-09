#!/usr/bin/env python3
"""
Bitstream-to-Image End-to-End Pipeline

Orchestrates the complete preprocessing workflow:
  RBF/SOF → Bits → Width Detection → 2D Folding → Entropy Map → PNG

This is the main entry point for converting raw Intel FPGA bitstreams into
visual representations suitable for CNN analysis.

Usage:
    python bitstream_to_image.py --rbf design.rbf --out entropy.png
    python bitstream_to_image.py --rbf design.rbf --out entropy.png --width 1024 --skip-autocorr

Integration:
    Called by cli/preprocess_dataset.py to batch-process training data.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any

from rbf_utils import read_rbf, bytes_to_bits, detect_format, get_bitstream_info
from autocorr_width import infer_frame_width, get_top_k_widths
from entropy_map import (
    fold_to_2d,
    compute_entropy_map,
    save_entropy_png,
    compute_entropy_statistics,
    detect_entropy_anomalies,
    visualize_entropy_comparison
)


def process_bitstream(
    rbf_path: str,
    output_path: str,
    width: Optional[int] = None,
    window_size: int = 16,
    stride: int = 4,
    colormap: str = 'grayscale',
    save_metadata: bool = True,
    visualize: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Complete bitstream-to-image conversion pipeline.

    Args:
        rbf_path: Input .rbf or .sof file
        output_path: Output .png file path
        width: Manual width override (or None for auto-detection)
        window_size: Entropy window size (bits)
        stride: Window stride (bits)
        colormap: Color mapping for PNG
        save_metadata: Save JSON metadata alongside PNG
        visualize: Generate comparison visualization
        verbose: Print progress messages

    Returns:
        Dictionary with processing metadata:
            - input_file: Input path
            - output_file: Output PNG path
            - detected_width: Frame width used
            - entropy_stats: Statistical summary
            - anomalies: Anomaly detection results
    """
    if verbose:
        print(f"Processing: {rbf_path}")
        print("=" * 60)

    # Step 1: Load and analyze bitstream
    if verbose:
        print("\n[1/5] Loading bitstream...")

    rbf_bytes = read_rbf(rbf_path)
    format_type, format_metadata = detect_format(rbf_bytes)

    if verbose:
        print(f"      Format: {format_type}")
        print(f"      Size: {len(rbf_bytes):,} bytes")

        if format_metadata['estimated_compression']:
            print("      WARNING: High entropy detected - bitstream may be compressed!")
            print("               For Arria 10, recompile with bitstream_compression=off")

    # Convert to bits
    bits = bytes_to_bits(rbf_bytes)

    if verbose:
        print(f"      Total bits: {len(bits):,}")

    # Step 2: Width detection
    if verbose:
        print("\n[2/5] Detecting frame width...")

    if width is None:
        detected_width = infer_frame_width(bits)
        if verbose:
            # Show top candidates
            top_widths = get_top_k_widths(bits, k=3)
            print(f"      Top candidates:")
            for i, (w, score) in enumerate(top_widths, 1):
                marker = "← SELECTED" if w == detected_width else ""
                print(f"        {i}. {w:4d} bits (score: {score:.4f}) {marker}")
    else:
        detected_width = width
        if verbose:
            print(f"      Using manual width: {detected_width} bits")

    # Step 3: Fold to 2D
    if verbose:
        print("\n[3/5] Folding to 2D image...")

    bit_image = fold_to_2d(bits, detected_width)

    if verbose:
        print(f"      Image shape: {bit_image.shape[0]} rows × {bit_image.shape[1]} cols")
        print(f"      Coverage: {bit_image.size / len(bits) * 100:.2f}% of bitstream")

    # Step 4: Compute entropy
    if verbose:
        print(f"\n[4/5] Computing entropy map (window={window_size}, stride={stride})...")

    entropy_map = compute_entropy_map(
        bit_image,
        window_size=window_size,
        stride=stride,
        normalize=False  # Keep as float for statistics
    )

    if verbose:
        print(f"      Entropy map shape: {entropy_map.shape}")

    # Compute statistics
    stats = compute_entropy_statistics(entropy_map)
    anomaly_mask, num_anomalies = detect_entropy_anomalies(entropy_map, threshold=3.0)

    if verbose:
        print(f"\n      Statistics:")
        print(f"        Mean entropy:   {stats['mean']:.4f} bits")
        print(f"        Std deviation:  {stats['std']:.4f} bits")
        print(f"        Range:          [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"        Median:         {stats['q50']:.4f} bits")
        print(f"        High-entropy:   {stats['high_entropy_fraction']*100:.2f}%")
        print(f"\n      Anomaly detection:")
        print(f"        Anomalous windows: {num_anomalies} ({num_anomalies/entropy_map.size*100:.2f}%)")

    # Step 5: Save outputs
    if verbose:
        print(f"\n[5/5] Saving outputs...")

    # Normalize entropy map for saving
    entropy_normalized = ((entropy_map / entropy_map.max()) * 255).astype('uint8')

    # Save PNG
    save_entropy_png(entropy_normalized, output_path, colormap=colormap)

    if verbose:
        print(f"      Entropy map: {output_path}")

    # Save metadata JSON
    metadata = {
        'input_file': str(rbf_path),
        'output_file': str(output_path),
        'format': format_type,
        'bitstream_size_bytes': len(rbf_bytes),
        'bitstream_size_bits': len(bits),
        'detected_width': detected_width,
        'width_detection_method': 'autocorrelation' if width is None else 'manual',
        'image_shape': list(bit_image.shape),
        'entropy_map_shape': list(entropy_map.shape),
        'window_size': window_size,
        'stride': stride,
        'entropy_statistics': stats,
        'anomaly_count': num_anomalies,
        'anomaly_fraction': float(num_anomalies / entropy_map.size)
    }

    if save_metadata:
        metadata_path = Path(output_path).with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        if verbose:
            print(f"      Metadata:    {metadata_path}")

    # Optional visualization
    if visualize:
        comparison_path = Path(output_path).with_stem(Path(output_path).stem + '_comparison')
        comparison_path = comparison_path.with_suffix('.png')

        visualize_entropy_comparison(bit_image, entropy_normalized, save_path=str(comparison_path))

        if verbose:
            print(f"      Comparison:  {comparison_path}")

    if verbose:
        print("\n✓ Pipeline complete!")

    return metadata


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Convert FPGA bitstream to entropy map image for ML analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect width and generate entropy map
  python bitstream_to_image.py --rbf design.rbf --out entropy.png

  # Manual width specification
  python bitstream_to_image.py --rbf design.rbf --out entropy.png --width 1024

  # Generate with comparison visualization
  python bitstream_to_image.py --rbf design.rbf --out entropy.png --visualize

  # Use color map for better visual analysis
  python bitstream_to_image.py --rbf design.rbf --out entropy.png --colormap viridis

  # Batch processing (see cli/preprocess_dataset.py)
        """
    )

    parser.add_argument(
        '--rbf',
        required=True,
        help='Input bitstream file (.rbf or .sof)'
    )

    parser.add_argument(
        '--out',
        required=True,
        help='Output PNG file path'
    )

    parser.add_argument(
        '--width',
        type=int,
        help='Manual frame width (bits). If not specified, auto-detected via autocorrelation.'
    )

    parser.add_argument(
        '--window-size',
        type=int,
        default=16,
        help='Entropy window size in bits (default: 16)'
    )

    parser.add_argument(
        '--stride',
        type=int,
        default=4,
        help='Window stride in bits (default: 4)'
    )

    parser.add_argument(
        '--colormap',
        choices=['grayscale', 'viridis', 'hot'],
        default='grayscale',
        help='Color mapping for output PNG (default: grayscale)'
    )

    parser.add_argument(
        '--no-metadata',
        action='store_true',
        help='Do not save JSON metadata file'
    )

    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate side-by-side comparison visualization'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress messages'
    )

    args = parser.parse_args()

    try:
        process_bitstream(
            rbf_path=args.rbf,
            output_path=args.out,
            width=args.width,
            window_size=args.window_size,
            stride=args.stride,
            colormap=args.colormap,
            save_metadata=not args.no_metadata,
            visualize=args.visualize,
            verbose=not args.quiet
        )

        return 0

    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    except Exception as e:
        print(f"ERROR: Pipeline failed: {e}", file=sys.stderr)
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
