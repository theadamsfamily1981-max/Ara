#!/usr/bin/env python3
"""
Batch Bitstream Preprocessing

Recursively processes all .rbf files in a directory tree, converting them
to entropy map images for ML training/inference.

Input structure:
    data/raw/arria10/
        clean/
            seed_001/design.rbf
            seed_002/design.rbf
        infected/
            timebomb_001/design.rbf
            comparator_001/design.rbf

Output structure:
    data/images/arria10/
        clean/
            seed_001_entropy.png
            seed_001_entropy.json
            seed_002_entropy.png
            ...
        infected/
            timebomb_001_entropy.png
            comparator_001_entropy.png
            ...

Usage:
    python preprocess_dataset.py --input data/raw/arria10 --output data/images/arria10
    python preprocess_dataset.py --input data/raw/arria10 --output data/images/arria10 --parallel 8
"""

import argparse
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import json

# Add preprocessing module to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'preprocessing'))

from bitstream_to_image import process_bitstream


def find_rbf_files(root_dir: Path) -> list:
    """
    Recursively find all .rbf files in directory tree.

    Returns:
        List of tuples: (rbf_path, relative_category)
        - rbf_path: Absolute path to .rbf file
        - relative_category: 'clean' or 'infected' (from parent dir)
    """
    rbf_files = []

    for rbf_path in root_dir.rglob('*.rbf'):
        # Determine category from directory structure
        if 'clean' in rbf_path.parts:
            category = 'clean'
        elif 'infected' in rbf_path.parts:
            category = 'infected'
        else:
            category = 'unknown'

        rbf_files.append((rbf_path, category))

    return rbf_files


def process_single_file(args: tuple) -> dict:
    """
    Process a single .rbf file (worker function for multiprocessing).

    Args:
        args: Tuple of (rbf_path, output_dir, category, index, total)

    Returns:
        Processing results dictionary
    """
    rbf_path, output_dir, category, index, total = args

    # Create output path
    output_dir = Path(output_dir) / category
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate output filename
    stem = rbf_path.stem
    parent = rbf_path.parent.name
    output_name = f"{parent}_{stem}_entropy.png"
    output_path = output_dir / output_name

    # Skip if already processed
    if output_path.exists():
        return {
            'rbf_path': str(rbf_path),
            'output_path': str(output_path),
            'status': 'skipped',
            'message': 'Already exists'
        }

    # Process
    try:
        metadata = process_bitstream(
            rbf_path=str(rbf_path),
            output_path=str(output_path),
            save_metadata=True,
            visualize=False,
            verbose=False
        )

        return {
            'rbf_path': str(rbf_path),
            'output_path': str(output_path),
            'status': 'success',
            'metadata': metadata
        }

    except Exception as e:
        return {
            'rbf_path': str(rbf_path),
            'output_path': str(output_path),
            'status': 'failed',
            'error': str(e)
        }


def preprocess_dataset(
    input_dir: str,
    output_dir: str,
    parallel: int = 1,
    skip_existing: bool = True
) -> dict:
    """
    Batch preprocess all .rbf files in input directory.

    Args:
        input_dir: Root directory with raw .rbf files
        output_dir: Output directory for entropy images
        parallel: Number of parallel workers (default: 1)
        skip_existing: Skip files that already have output (default: True)

    Returns:
        Summary statistics dictionary
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    print(f"Preprocessing dataset...")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Workers: {parallel}")
    print()

    # Find all .rbf files
    print("Scanning for .rbf files...")
    rbf_files = find_rbf_files(input_dir)

    if len(rbf_files) == 0:
        print("ERROR: No .rbf files found")
        return {'error': 'No files found'}

    print(f"Found {len(rbf_files)} .rbf files")

    # Count by category
    categories = {}
    for _, cat in rbf_files:
        categories[cat] = categories.get(cat, 0) + 1

    print("\nDistribution:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")
    print()

    # Prepare arguments for workers
    total = len(rbf_files)
    worker_args = [
        (rbf_path, output_dir, category, i+1, total)
        for i, (rbf_path, category) in enumerate(rbf_files)
    ]

    # Process files
    print("Processing...")
    results = []

    if parallel > 1:
        # Multiprocessing
        with Pool(processes=parallel) as pool:
            results = list(tqdm(
                pool.imap(process_single_file, worker_args),
                total=total,
                desc="Converting"
            ))
    else:
        # Sequential
        for args in tqdm(worker_args, desc="Converting"):
            results.append(process_single_file(args))

    # Summarize
    summary = {
        'total': total,
        'success': sum(1 for r in results if r['status'] == 'success'),
        'skipped': sum(1 for r in results if r['status'] == 'skipped'),
        'failed': sum(1 for r in results if r['status'] == 'failed'),
        'categories': categories,
        'results': results
    }

    return summary


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Batch preprocess FPGA bitstreams to entropy images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic preprocessing
  python preprocess_dataset.py \\
      --input data/raw/arria10 \\
      --output data/images/arria10

  # Parallel processing (8 workers)
  python preprocess_dataset.py \\
      --input data/raw/arria10 \\
      --output data/images/arria10 \\
      --parallel 8

  # Force reprocessing (don't skip existing)
  python preprocess_dataset.py \\
      --input data/raw/arria10 \\
      --output data/images/arria10 \\
      --no-skip
        """
    )

    parser.add_argument(
        '--input',
        required=True,
        help='Input directory with .rbf files'
    )

    parser.add_argument(
        '--output',
        required=True,
        help='Output directory for entropy images'
    )

    parser.add_argument(
        '--parallel',
        type=int,
        default=1,
        help='Number of parallel workers (default: 1, 0 = auto)'
    )

    parser.add_argument(
        '--no-skip',
        action='store_true',
        help='Reprocess files even if output exists'
    )

    parser.add_argument(
        '--summary',
        help='Save summary to JSON file'
    )

    args = parser.parse_args()

    # Determine parallel workers
    if args.parallel == 0:
        args.parallel = max(1, cpu_count() - 1)

    # Preprocess
    summary = preprocess_dataset(
        input_dir=args.input,
        output_dir=args.output,
        parallel=args.parallel,
        skip_existing=not args.no_skip
    )

    if 'error' in summary:
        print(f"\nERROR: {summary['error']}")
        return 1

    # Print summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total files:      {summary['total']}")
    print(f"Successfully processed: {summary['success']}")
    print(f"Skipped (exists): {summary['skipped']}")
    print(f"Failed:           {summary['failed']}")

    if summary['failed'] > 0:
        print("\nFailed files:")
        for result in summary['results']:
            if result['status'] == 'failed':
                print(f"  {result['rbf_path']}: {result['error']}")

    print("=" * 60)

    # Save summary if requested
    if args.summary:
        with open(args.summary, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to: {args.summary}")

    return 0 if summary['failed'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
