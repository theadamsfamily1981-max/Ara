#!/usr/bin/env python3
"""
a10_build_dataset.py

Batch process Intel FPGA bitstreams into ML training dataset.

Usage:
    # Process single file
    python scripts/a10_build_dataset.py --input design.sof --output datasets/design

    # Process directory
    python scripts/a10_build_dataset.py --input-dir bitstreams/ --output datasets/arria10

    # With custom width range
    python scripts/a10_build_dataset.py --input design.sof --output out/ \
        --min-width 1000 --max-width 2000 --method autocorrelation

    # Generate visualization
    python scripts/a10_build_dataset.py --input design.sof --output out/ --visualize
"""

import sys
import argparse
from pathlib import Path
from typing import List, Tuple
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from a10_ml import (
    load_bitstream,
    strip_intel_headers,
    guess_width,
    bitstream_to_image,
    save_image_dataset
)
from a10_ml.preprocess import bytes_to_bits, analyze_bitstream_header
from a10_ml.image_encoder import (
    calculate_image_stats,
    bits_to_grayscale_png,
    bits_to_rgb_png
)
from a10_ml.width_detection import visualize_width_scan


def process_single_file(
    input_path: Path,
    output_dir: Path,
    min_width: int = 800,
    max_width: int = 4096,
    step: int = 8,
    method: str = 'autocorrelation',
    visualize: bool = False
) -> dict:
    """
    Process a single bitstream file.

    Args:
        input_path: Path to .sof or .rbf file
        output_dir: Output directory
        min_width: Minimum width for detection
        max_width: Maximum width for detection
        step: Width search step
        method: Detection method ('autocorrelation' or 'entropy')
        visualize: Generate visualization plots

    Returns:
        Dictionary with processing results
    """
    print("=" * 70)
    print(f"Processing: {input_path.name}")
    print("=" * 70)
    print()

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load bitstream
    print("📂 Step 1: Loading bitstream...")
    raw = load_bitstream(input_path)

    # Analyze header
    print("\n🔍 Analyzing header...")
    header_info = analyze_bitstream_header(raw)
    for key, value in header_info.items():
        print(f"   {key:20s}: {value}")

    # Step 2: Strip headers
    print("\n✂️  Step 2: Stripping headers...")
    clean = strip_intel_headers(raw)
    print(f"   Original: {len(raw):,} bytes")
    print(f"   Stripped: {len(clean):,} bytes")
    print(f"   Removed:  {len(raw) - len(clean):,} bytes ({100*(len(raw)-len(clean))/len(raw):.1f}%)")

    # Step 3: Convert to bits
    print("\n🔢 Step 3: Converting to bits...")
    bits = bytes_to_bits(clean)

    # Step 4: Detect width
    print("\n📐 Step 4: Detecting optimal width...")
    width = guess_width(bits, min_width, max_width, step, method)

    # Step 5: Generate image
    print("\n🖼️  Step 5: Generating image...")
    image = bitstream_to_image(bits, width)

    # Step 6: Calculate statistics
    print("\n📊 Step 6: Calculating statistics...")
    stats = calculate_image_stats(image)
    print(f"   Shape:        {stats['shape']}")
    print(f"   Bit density:  {stats['density']*100:.2f}%")
    print(f"   Row entropy:  {stats['row_entropies']['mean']:.4f} ± {stats['row_entropies']['std']:.4f}")

    # Step 7: Save outputs
    print("\n💾 Step 7: Saving outputs...")

    # Save image as .npy
    image_path = output_dir / f"{input_path.stem}_image.npy"
    import numpy as np
    np.save(image_path, image)
    print(f"   Saved: {image_path}")

    # Save metadata
    metadata = {
        'source_file': str(input_path),
        'source_size_bytes': len(raw),
        'stripped_size_bytes': len(clean),
        'width': width,
        'height': image.shape[0],
        'total_bits': int(image.size),
        'detection_method': method,
        'header_info': header_info,
        'statistics': stats
    }

    metadata_path = output_dir / f"{input_path.stem}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   Saved: {metadata_path}")

    # Optional: Generate visualizations
    if visualize:
        print("\n📈 Generating visualizations...")

        # Width scan plot
        scan_plot = output_dir / f"{input_path.stem}_width_scan.png"
        visualize_width_scan(bits, min_width, max_width, step, str(scan_plot))

        # Grayscale PNG
        if width % 8 == 0:  # Only if byte-aligned
            gray_png = output_dir / f"{input_path.stem}_grayscale.png"
            try:
                bits_to_grayscale_png(bits, width, gray_png, scale=1)
            except ImportError:
                print("   ⚠️  Skipping PNG (Pillow not installed)")

        # Colored PNG
        color_png = output_dir / f"{input_path.stem}_colored.png"
        try:
            bits_to_rgb_png(bits, width, color_png, colormap='viridis')
        except ImportError:
            print("   ⚠️  Skipping colored PNG (matplotlib not installed)")

    print("\n✅ Processing complete!")
    print("=" * 70)
    print()

    return metadata


def process_directory(
    input_dir: Path,
    output_dir: Path,
    min_width: int = 800,
    max_width: int = 4096,
    step: int = 8,
    method: str = 'autocorrelation',
    pattern: str = "*.sof",
    create_dataset: bool = True
) -> List[dict]:
    """
    Process all bitstreams in a directory.

    Args:
        input_dir: Directory containing .sof/.rbf files
        output_dir: Output directory
        min_width: Minimum width for detection
        max_width: Maximum width for detection
        step: Width search step
        method: Detection method
        pattern: File glob pattern (e.g., "*.sof", "*.rbf", "*.bit")
        create_dataset: Create ML dataset structure

    Returns:
        List of metadata dictionaries
    """
    input_dir = Path(input_dir)
    files = list(input_dir.glob(pattern))

    if not files:
        print(f"❌ No files matching '{pattern}' found in {input_dir}")
        return []

    print(f"🔍 Found {len(files)} files matching '{pattern}'")
    print()

    all_metadata = []

    for i, file_path in enumerate(files, 1):
        print(f"\n{'='*70}")
        print(f"File {i}/{len(files)}")
        print(f"{'='*70}\n")

        try:
            metadata = process_single_file(
                file_path,
                output_dir / file_path.stem,
                min_width,
                max_width,
                step,
                method,
                visualize=False  # Disable per-file viz for batch
            )
            all_metadata.append(metadata)

        except Exception as e:
            print(f"❌ Error processing {file_path.name}: {e}")
            continue

    # Save combined metadata
    combined_path = output_dir / "dataset_metadata.json"
    with open(combined_path, 'w') as f:
        json.dump(all_metadata, f, indent=2)
    print(f"\n💾 Saved combined metadata to {combined_path}")

    # Create ML dataset structure if requested
    if create_dataset and all_metadata:
        print("\n📦 Creating ML dataset structure...")
        create_ml_dataset(output_dir, all_metadata)

    return all_metadata


def create_ml_dataset(output_dir: Path, metadata_list: List[dict]):
    """
    Organize processed bitstreams into ML-ready dataset.

    Creates:
        output_dir/
            images/
                bitstream_0000.npy
                bitstream_0001.npy
                ...
            labels.json
    """
    import numpy as np

    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    image_files = []

    for i, metadata in enumerate(metadata_list):
        # Load image from individual output
        source_image = output_dir / Path(metadata['source_file']).stem / f"{Path(metadata['source_file']).stem}_image.npy"

        if not source_image.exists():
            print(f"⚠️  Image not found: {source_image}")
            continue

        # Copy to dataset directory
        dest_image = images_dir / f"bitstream_{i:04d}.npy"
        np.save(dest_image, np.load(source_image))

        image_files.append({
            'filename': dest_image.name,
            'source': metadata['source_file'],
            'width': metadata['width'],
            'height': metadata['height'],
            'label': 0  # Default: clean (user must manually label trojans)
        })

    # Save labels
    labels_path = output_dir / "labels.json"
    labels_data = {
        'num_images': len(image_files),
        'note': 'All images labeled as 0 (clean) by default. Manually update labels for trojan bitstreams.',
        'images': image_files
    }

    with open(labels_path, 'w') as f:
        json.dump(labels_data, f, indent=2)

    print(f"✅ Created ML dataset with {len(image_files)} images")
    print(f"   Images: {images_dir}")
    print(f"   Labels: {labels_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Build ML dataset from Intel FPGA bitstreams',
        epilog='''
Examples:
  # Single file
  python scripts/a10_build_dataset.py --input design.sof --output datasets/design

  # Batch processing
  python scripts/a10_build_dataset.py --input-dir bitstreams/ --output datasets/arria10

  # Custom width range
  python scripts/a10_build_dataset.py --input design.sof --output out/ \\
      --min-width 1000 --max-width 2000

  # With visualization
  python scripts/a10_build_dataset.py --input design.sof --output out/ --visualize
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', type=Path,
                             help='Single .sof/.rbf file to process')
    input_group.add_argument('--input-dir', type=Path,
                             help='Directory containing bitstreams')

    # Output
    parser.add_argument('--output', type=Path, required=True,
                        help='Output directory')

    # Width detection parameters
    parser.add_argument('--min-width', type=int, default=800,
                        help='Minimum width to test (default: 800)')
    parser.add_argument('--max-width', type=int, default=4096,
                        help='Maximum width to test (default: 4096)')
    parser.add_argument('--step', type=int, default=8,
                        help='Width search step (default: 8)')
    parser.add_argument('--method', choices=['autocorrelation', 'entropy'],
                        default='autocorrelation',
                        help='Width detection method (default: autocorrelation)')

    # Batch processing options
    parser.add_argument('--pattern', type=str, default='*.sof',
                        help='File pattern for batch processing (default: *.sof)')
    parser.add_argument('--no-dataset', action='store_true',
                        help='Skip ML dataset creation for batch processing')

    # Visualization
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization plots')

    args = parser.parse_args()

    # Single file mode
    if args.input:
        if not args.input.exists():
            print(f"❌ Input file not found: {args.input}")
            return 1

        process_single_file(
            args.input,
            args.output,
            args.min_width,
            args.max_width,
            args.step,
            args.method,
            args.visualize
        )

    # Batch mode
    elif args.input_dir:
        if not args.input_dir.exists():
            print(f"❌ Input directory not found: {args.input_dir}")
            return 1

        process_directory(
            args.input_dir,
            args.output,
            args.min_width,
            args.max_width,
            args.step,
            args.method,
            args.pattern,
            create_dataset=not args.no_dataset
        )

    return 0


if __name__ == '__main__':
    sys.exit(main())
