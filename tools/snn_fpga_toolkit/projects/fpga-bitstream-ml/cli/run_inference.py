#!/usr/bin/env python3
"""
End-to-End Bitstream Trojan Analysis

Takes a raw .rbf or .sof bitstream file and runs the complete analysis pipeline:
  1. Load bitstream
  2. Detect frame width (autocorrelation)
  3. Generate entropy map
  4. Run CNN inference
  5. Report Trojan detection results

This is the main tool for analyzing suspicious bitstreams.

Usage:
    python run_inference.py --rbf design.rbf --model arria10_cnn.pt
    python run_inference.py --rbf design.rbf --model arria10_cnn.pt --threshold 0.9
    python run_inference.py --rbf design.rbf --model arria10_cnn.pt --save-entropy
"""

import argparse
import sys
import tempfile
from pathlib import Path
import json

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'preprocessing'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'models' / 'arria10_cnn'))

from bitstream_to_image import process_bitstream
from infer_cnn import analyze_bitstream


def analyze_rbf(
    rbf_path: str,
    model_path: str,
    threshold: float = 0.5,
    save_entropy: bool = False,
    entropy_output: str = None,
    device: str = 'auto',
    verbose: bool = True
) -> dict:
    """
    Complete end-to-end analysis of a single bitstream.

    Args:
        rbf_path: Path to .rbf or .sof bitstream file
        model_path: Path to trained CNN model (.pt)
        threshold: Trojan detection threshold (default 0.5)
        save_entropy: Save entropy map PNG (default False)
        entropy_output: Custom path for entropy map (or None for auto)
        device: Device for inference ('auto', 'cuda', 'cpu')
        verbose: Print detailed progress (default True)

    Returns:
        Complete analysis results dictionary
    """
    if verbose:
        print("=" * 70)
        print("FPGA BITSTREAM TROJAN ANALYSIS")
        print("=" * 70)
        print()

    # Step 1: Preprocess bitstream to entropy map
    if verbose:
        print("[1/2] Preprocessing bitstream to entropy map...")
        print(f"      Input: {rbf_path}")

    # Create temp file for entropy map
    with tempfile.NamedTemporaryFile(suffix='_entropy.png', delete=False) as tmp:
        temp_entropy_path = tmp.name

    try:
        preprocessing_metadata = process_bitstream(
            rbf_path=rbf_path,
            output_path=temp_entropy_path,
            save_metadata=False,
            visualize=False,
            verbose=False
        )

        if verbose:
            print(f"      Format: {preprocessing_metadata['format']}")
            print(f"      Detected width: {preprocessing_metadata['detected_width']} bits")
            print(f"      Entropy stats: mean={preprocessing_metadata['entropy_statistics']['mean']:.4f}, "
                  f"anomalies={preprocessing_metadata['anomaly_count']}")
            print()

        # Step 2: Run CNN inference
        if verbose:
            print("[2/2] Running CNN Trojan detector...")
            print(f"      Model: {model_path}")

        cnn_results = analyze_bitstream(
            model_path=model_path,
            image_path=temp_entropy_path,
            threshold=threshold,
            device=device,
            verbose=False  # We'll print our own summary
        )

        if verbose:
            print(f"      Trojan probability: {cnn_results['trojan_probability']:.4f}")
            print()

        # Combine results
        combined_results = {
            'bitstream_file': str(rbf_path),
            'model_file': str(model_path),
            'preprocessing': preprocessing_metadata,
            'cnn_inference': cnn_results,
            'final_verdict': {
                'prediction': cnn_results['prediction'],
                'label': cnn_results['label'],
                'trojan_probability': cnn_results['trojan_probability'],
                'confidence': cnn_results['confidence'],
                'threshold': threshold
            }
        }

        # Print final verdict
        if verbose:
            print("=" * 70)
            print("FINAL VERDICT")
            print("=" * 70)
            print()

            label = cnn_results['label']
            prob = cnn_results['trojan_probability']
            conf = cnn_results['confidence']

            print(f"  Classification:  {label}")
            print(f"  Probability:     {prob:.4f} ({prob*100:.2f}%)")
            print(f"  Confidence:      {conf:.4f} ({conf*100:.2f}%)")
            print()

            if label == 'INFECTED':
                print("  ⚠ WARNING: Hardware Trojan Detected!")
                print()
                print("  RISK ASSESSMENT:")
                if prob > 0.9:
                    print("    Severity: HIGH (>90% confidence)")
                    print("    Recommendation: DO NOT DEPLOY")
                elif prob > 0.7:
                    print("    Severity: MEDIUM (70-90% confidence)")
                    print("    Recommendation: Manual inspection required")
                else:
                    print("    Severity: LOW (50-70% confidence)")
                    print("    Recommendation: Consider re-compilation")

                print()
                print("  SUGGESTED ACTIONS:")
                print("    1. DO NOT flash this bitstream to hardware")
                print("    2. Review build logs for anomalies")
                print("    3. Verify HDL source integrity")
                print("    4. Check for unauthorized ECO changes")
                print("    5. Re-compile with different seed")
                print("    6. Compare with known-good bitstreams")

            else:
                print("  ✓ Bitstream Appears Clean")
                print()
                print("  No Hardware Trojan detected.")
                print("  Safe to proceed with deployment.")
                print()
                print("  NOTE: This analysis is probabilistic.")
                print("        False negatives are possible for novel/stealthy Trojans.")
                print("        Always combine with other verification methods.")

            print()
            print("=" * 70)

        # Optionally save entropy map
        if save_entropy or entropy_output:
            if entropy_output is None:
                entropy_output = Path(rbf_path).with_suffix('.entropy.png')

            import shutil
            shutil.copy(temp_entropy_path, entropy_output)

            if verbose:
                print(f"\nEntropy map saved to: {entropy_output}")

        return combined_results

    finally:
        # Clean up temp file
        if not save_entropy and not entropy_output:
            Path(temp_entropy_path).unlink(missing_ok=True)


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='End-to-end FPGA bitstream Trojan analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python run_inference.py --rbf design.rbf --model arria10_cnn.pt

  # Higher confidence threshold
  python run_inference.py --rbf design.rbf --model arria10_cnn.pt --threshold 0.9

  # Save entropy map for inspection
  python run_inference.py --rbf design.rbf --model arria10_cnn.pt --save-entropy

  # Save detailed results to JSON
  python run_inference.py --rbf design.rbf --model arria10_cnn.pt --json results.json

  # Batch analysis
  for rbf in bitstreams/*.rbf; do
      python run_inference.py --rbf "$rbf" --model arria10_cnn.pt
  done

Integration with HNTF build system:
  # In flows/quartus/a10ped/build_tile.sh:
  python run_inference.py \\
      --rbf output_files/a10ped_tile0.rbf \\
      --model /path/to/arria10_cnn.pt \\
      --threshold 0.90

  if [ $? -ne 0 ]; then
      echo "ERROR: Trojan detected! Aborting flash."
      exit 1
  fi
        """
    )

    parser.add_argument(
        '--rbf',
        required=True,
        help='Input bitstream file (.rbf or .sof)'
    )

    parser.add_argument(
        '--model',
        required=True,
        help='Path to trained CNN model (.pt)'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Trojan detection threshold (default: 0.5, higher = more strict)'
    )

    parser.add_argument(
        '--save-entropy',
        action='store_true',
        help='Save entropy map PNG alongside bitstream'
    )

    parser.add_argument(
        '--entropy-output',
        help='Custom path for entropy map (implies --save-entropy)'
    )

    parser.add_argument(
        '--device',
        choices=['auto', 'cuda', 'cpu'],
        default='auto',
        help='Device for CNN inference (default: auto)'
    )

    parser.add_argument(
        '--json',
        help='Save detailed results to JSON file'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output (only show verdict)'
    )

    args = parser.parse_args()

    # Run analysis
    try:
        results = analyze_rbf(
            rbf_path=args.rbf,
            model_path=args.model,
            threshold=args.threshold,
            save_entropy=args.save_entropy or (args.entropy_output is not None),
            entropy_output=args.entropy_output,
            device=args.device,
            verbose=not args.quiet
        )

        # Save JSON if requested
        if args.json:
            with open(args.json, 'w') as f:
                json.dump(results, f, indent=2)

            if not args.quiet:
                print(f"\nDetailed results saved to: {args.json}")

        # Exit code: 0 = clean, 1 = infected
        return results['final_verdict']['prediction']

    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    except Exception as e:
        print(f"ERROR: Analysis failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 2


if __name__ == '__main__':
    sys.exit(main())
