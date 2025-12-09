#!/usr/bin/env python3
"""
analyze_k10_sd.py

Command-line tool for analyzing K10 / ColEngine P2 SD card images or firmware ZIPs.

Usage:
    # Analyze SD image
    python scripts/analyze_k10_sd.py --image k10_sd0.img --out reports/k10_sd0_report.json

    # Analyze firmware ZIP
    python scripts/analyze_k10_sd.py --firmware-zip colengine_p2_firmware.zip --out reports/firmware_report.json

    # Scan directory for bitstreams
    python scripts/analyze_k10_sd.py --scan-dir /mnt/k10_extracted --out reports/bitstreams.json
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from k10_tools.sd_image_analyzer import analyze_sd_image
from k10_tools.firmware_extractor import extract_firmware_zip, generate_firmware_report
from k10_tools.bitstream_finder import find_bitstreams


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze K10 / ColEngine P2 firmware and SD images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Analyze SD card image
  python scripts/analyze_k10_sd.py --image k10_sd0.img --out reports/k10_report.json

  # Extract and analyze firmware ZIP
  python scripts/analyze_k10_sd.py --firmware-zip firmware.zip --extract-to out/firmware

  # Scan directory for bitstreams
  python scripts/analyze_k10_sd.py --scan-dir /mnt/k10_extracted --out reports/bitstreams.json

  # All-in-one: extract ZIP and find bitstreams
  python scripts/analyze_k10_sd.py --firmware-zip firmware.zip --extract-to out/ --find-bitstreams
        '''
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', type=Path, help='SD card image file (.img)')
    input_group.add_argument('--firmware-zip', type=Path, help='Firmware ZIP file from vendor')
    input_group.add_argument('--scan-dir', type=Path, help='Directory to scan for bitstreams')

    # Output options
    parser.add_argument('--out', type=Path, help='Output JSON report path')
    parser.add_argument('--extract-to', type=Path, help='Directory to extract firmware to')
    parser.add_argument('--find-bitstreams', action='store_true', help='Scan for bitstreams after extraction')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    print("=" * 70)
    print("K10 / ColEngine P2 Firmware Analysis Tool")
    print("=" * 70)
    print()

    try:
        if args.image:
            # Analyze SD image
            if not args.image.exists():
                print(f"❌ Error: Image file not found: {args.image}")
                return 1

            output_path = args.out or Path('reports') / f"{args.image.stem}_report.json"
            analyze_sd_image(args.image, output_path)

        elif args.firmware_zip:
            # Extract and analyze firmware ZIP
            if not args.firmware_zip.exists():
                print(f"❌ Error: Firmware ZIP not found: {args.firmware_zip}")
                return 1

            extract_dir = args.extract_to or Path('out') / 'firmware'
            extract_dir.mkdir(parents=True, exist_ok=True)

            # Extract
            extract_report = extract_firmware_zip(
                args.firmware_zip,
                extract_dir,
                copy_bitstreams=True
            )

            # Generate report
            output_path = args.out or Path('reports') / f"{args.firmware_zip.stem}_report.json"
            generate_firmware_report(args.firmware_zip, extract_report, output_path)

            # Optional: scan for bitstreams
            if args.find_bitstreams:
                print(f"\n{'=' * 70}")
                print("Scanning for additional bitstreams...")
                print(f"{'=' * 70}\n")

                bitstream_report_path = Path('reports') / f"{args.firmware_zip.stem}_bitstreams.json"
                find_bitstreams(
                    extract_dir / args.firmware_zip.stem,
                    output_json=bitstream_report_path,
                    verbose=args.verbose
                )

        elif args.scan_dir:
            # Scan directory for bitstreams
            if not args.scan_dir.exists():
                print(f"❌ Error: Directory not found: {args.scan_dir}")
                return 1

            output_path = args.out or Path('reports') / 'bitstream_scan.json'
            find_bitstreams(args.scan_dir, output_json=output_path, verbose=args.verbose)

        print(f"\n{'=' * 70}")
        print("✅ Analysis complete!")
        print(f"{'=' * 70}\n")

        return 0

    except KeyboardInterrupt:
        print("\n\n⏹ Interrupted by user")
        return 130

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
