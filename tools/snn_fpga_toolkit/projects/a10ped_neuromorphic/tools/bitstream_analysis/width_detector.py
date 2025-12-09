#!/usr/bin/env python3
"""
Algorithmic Width Detector for Intel FPGA Bitstreams

Implements the autocorrelation + entropy minimization approach validated
in BITSTREAM_ANALYSIS_RESEARCH_FINDINGS.md (Section 3).

Based on research showing 78% top-1 accuracy, 94% top-3 accuracy on Arria 10.

Usage:
    python3 width_detector.py bitstream.sof
    python3 width_detector.py bitstream.rbf --top-k 3
"""

import numpy as np
import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import struct
import json


class WidthDetector:
    """
    Algorithmically determines optimal 2D image width for bitstream analysis.

    Approach:
    1. Autocorrelation analysis - finds repeating patterns that indicate frame boundaries
    2. Entropy variance - measures consistency within candidate frames
    3. Combined scoring - balances both metrics
    """

    # Common candidate widths for Intel FPGAs (bytes per row)
    # Based on typical LAB column structures and frame sizes
    DEFAULT_CANDIDATES = [256, 384, 512, 768, 1024, 1536, 2048, 3072]

    # Magic bytes for format detection
    SOF_MAGIC = b'\x00\x09\x0f\xf0'
    RBF_MARKER = b'\xff\xff\xff\xff'  # Common in uncompressed RBF headers

    def __init__(self, bitstream_path: str, verbose: bool = False):
        self.path = Path(bitstream_path)
        self.verbose = verbose
        self.data = None
        self.config_data = None
        self.format_info = {}

    def load_bitstream(self) -> bool:
        """Load and identify bitstream format."""
        try:
            with open(self.path, 'rb') as f:
                self.data = f.read()

            # Detect format
            if self.data[:4] == self.SOF_MAGIC:
                self.format_info['format'] = 'sof'
                self._extract_sof_config()
            elif self.RBF_MARKER in self.data[:1024]:
                self.format_info['format'] = 'rbf'
                self._extract_rbf_config()
            else:
                self.format_info['format'] = 'unknown'
                # Try to use raw data
                self.config_data = self.data

            if self.verbose:
                print(f"Format: {self.format_info['format']}")
                print(f"Total size: {len(self.data)} bytes")
                if self.config_data:
                    print(f"Config data: {len(self.config_data)} bytes")

            return self.config_data is not None and len(self.config_data) > 0

        except Exception as e:
            print(f"Error loading bitstream: {e}", file=sys.stderr)
            return False

    def _extract_sof_config(self):
        """Extract configuration data from SOF file."""
        # SOF structure (simplified):
        # 0x00-0x04: Magic
        # 0x04-0x08: Version
        # 0x08-0x0C: Header length
        # Then sections with length prefixes

        try:
            # Skip to first section (typically at offset 0x100-0x400)
            # Look for large contiguous data block
            offset = 0x100
            max_chunk = 0
            best_offset = offset

            while offset < len(self.data) - 8:
                # Try to read section length (big-endian 32-bit)
                section_len = struct.unpack('>I', self.data[offset:offset+4])[0]

                # Sanity check: reasonable section size
                if 1024 < section_len < len(self.data):
                    if section_len > max_chunk:
                        max_chunk = section_len
                        best_offset = offset + 4
                    offset += section_len + 4
                else:
                    offset += 256  # Skip ahead

            if max_chunk > 0:
                self.config_data = self.data[best_offset:best_offset + max_chunk]
                self.format_info['config_offset'] = best_offset
                self.format_info['config_size'] = max_chunk
            else:
                # Fallback: use everything after header
                self.config_data = self.data[0x400:]

        except Exception as e:
            print(f"Warning: SOF parsing heuristic failed, using fallback: {e}", file=sys.stderr)
            self.config_data = self.data[0x400:]

    def _extract_rbf_config(self):
        """Extract configuration data from RBF file."""
        # For uncompressed RBF, skip header (typically first 1-4KB)
        # Look for high-entropy region (actual config data)

        window_size = 4096
        best_offset = 0
        max_entropy = 0

        for offset in range(0, min(16384, len(self.data) - window_size), 1024):
            entropy = self._compute_entropy(self.data[offset:offset+window_size])
            if entropy > max_entropy:
                max_entropy = entropy
                best_offset = offset

        # If entropy is too low, file might be compressed
        if max_entropy < 6.0:
            self.format_info['likely_compressed'] = True
            print("Warning: Low entropy suggests compression. Use .sof instead.", file=sys.stderr)

        self.config_data = self.data[best_offset:]
        self.format_info['config_offset'] = best_offset

    def detect_width(self, candidates: List[int] = None, top_k: int = 3) -> List[Tuple[int, float]]:
        """
        Detect most likely frame widths.

        Args:
            candidates: List of candidate widths to try (default: common FPGA widths)
            top_k: Number of top candidates to return

        Returns:
            List of (width, score) tuples sorted by score (descending)
        """
        if candidates is None:
            candidates = self.DEFAULT_CANDIDATES

        if self.config_data is None:
            raise ValueError("No configuration data loaded. Call load_bitstream() first.")

        results = []

        for width in candidates:
            if self.verbose:
                print(f"Testing width {width}...")

            score = self._score_width(width)
            results.append((width, score))

            if self.verbose:
                print(f"  Score: {score:.4f}")

        # Sort by score (descending)
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    def _score_width(self, width: int) -> float:
        """
        Compute combined score for a candidate width.

        Score = autocorrelation_strength / (entropy_variance + ε)

        Higher score = more likely correct width
        """
        # Reshape data into 2D array with given width
        data_len = len(self.config_data)
        num_rows = data_len // width

        if num_rows < 10:
            # Not enough rows to analyze
            return 0.0

        # Truncate to fit width
        truncated = self.config_data[:num_rows * width]
        matrix = np.frombuffer(truncated, dtype=np.uint8).reshape(num_rows, width)

        # Compute autocorrelation strength
        autocorr = self._compute_autocorrelation(matrix)

        # Compute entropy variance across rows
        entropy_var = self._compute_entropy_variance(matrix)

        # Combined score (add small epsilon to avoid division by zero)
        epsilon = 0.01
        score = autocorr / (entropy_var + epsilon)

        return score

    def _compute_autocorrelation(self, matrix: np.ndarray) -> float:
        """
        Compute row autocorrelation strength.

        If width aligns with physical frames, adjacent rows should be correlated
        (similar bit patterns in same columns = routing/configuration structure).
        """
        num_rows, width = matrix.shape

        if num_rows < 2:
            return 0.0

        # Compute correlation between adjacent rows
        correlations = []
        for i in range(min(20, num_rows - 1)):  # Sample first 20 pairs
            row1 = matrix[i].astype(np.float32)
            row2 = matrix[i + 1].astype(np.float32)

            # Pearson correlation
            corr = np.corrcoef(row1, row2)[0, 1]
            if not np.isnan(corr):
                correlations.append(abs(corr))

        # Mean absolute correlation
        return np.mean(correlations) if correlations else 0.0

    def _compute_entropy_variance(self, matrix: np.ndarray) -> float:
        """
        Compute variance of row entropies.

        If width is correct, rows should have similar entropy
        (each row = one frame, similar structure).
        High variance = width doesn't align with natural boundaries.
        """
        num_rows = matrix.shape[0]

        entropies = []
        for i in range(min(50, num_rows)):  # Sample first 50 rows
            row_entropy = self._compute_entropy(matrix[i].tobytes())
            entropies.append(row_entropy)

        return np.var(entropies)

    def _compute_entropy(self, data: bytes) -> float:
        """Compute Shannon entropy of byte sequence."""
        if len(data) == 0:
            return 0.0

        # Count byte frequencies
        counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
        probs = counts / len(data)

        # Remove zero probabilities
        probs = probs[probs > 0]

        # Shannon entropy
        return -np.sum(probs * np.log2(probs))

    def visualize_width(self, width: int, rows: int = 100):
        """
        Generate ASCII visualization of bitstream at given width.

        Useful for visually confirming whether detected width shows structure.
        """
        data_len = len(self.config_data)
        num_rows = min(rows, data_len // width)

        truncated = self.config_data[:num_rows * width]
        matrix = np.frombuffer(truncated, dtype=np.uint8).reshape(num_rows, width)

        print(f"\nVisualization (width={width}, first {num_rows} rows):")
        print("=" * 80)

        # Downsample columns to fit terminal width
        display_width = 78
        col_step = max(1, width // display_width)

        for row in matrix:
            # Downsample and convert to grayscale characters
            downsampled = row[::col_step]
            chars = []
            for val in downsampled:
                if val < 64:
                    chars.append(' ')
                elif val < 128:
                    chars.append('.')
                elif val < 192:
                    chars.append('+')
                else:
                    chars.append('#')
            print(''.join(chars))

        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Detect optimal frame width for FPGA bitstream analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 width_detector.py design.sof
  python3 width_detector.py design.rbf --top-k 5
  python3 width_detector.py design.sof --visualize
  python3 width_detector.py design.sof --candidates 512 1024 2048
        """
    )

    parser.add_argument('bitstream', help='Path to .sof or .rbf bitstream file')
    parser.add_argument('--top-k', type=int, default=3,
                       help='Number of top candidates to report (default: 3)')
    parser.add_argument('--candidates', type=int, nargs='+',
                       help='Specific widths to test (default: common FPGA widths)')
    parser.add_argument('--visualize', action='store_true',
                       help='Show ASCII visualization of top candidate')
    parser.add_argument('--json', action='store_true',
                       help='Output results as JSON')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    # Initialize detector
    detector = WidthDetector(args.bitstream, verbose=args.verbose)

    # Load bitstream
    if not detector.load_bitstream():
        print("Failed to load bitstream", file=sys.stderr)
        return 1

    # Detect widths
    try:
        results = detector.detect_width(candidates=args.candidates, top_k=args.top_k)
    except Exception as e:
        print(f"Error during width detection: {e}", file=sys.stderr)
        return 1

    # Output results
    if args.json:
        output = {
            'bitstream': str(detector.path),
            'format': detector.format_info.get('format', 'unknown'),
            'config_size': len(detector.config_data) if detector.config_data else 0,
            'candidates': [
                {'width': w, 'score': float(s)} for w, s in results
            ]
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"\nTop {len(results)} candidate widths:")
        print("-" * 40)
        for i, (width, score) in enumerate(results, 1):
            print(f"{i}. Width {width:4d} bytes - Score: {score:.4f}")

        print("\nRecommendation:")
        best_width, best_score = results[0]
        print(f"  Use width={best_width} for feature extraction")
        print(f"  Confidence: {'High' if best_score > 0.5 else 'Medium' if best_score > 0.2 else 'Low'}")

    # Visualization if requested
    if args.visualize and results:
        detector.visualize_width(results[0][0])

    return 0


if __name__ == '__main__':
    sys.exit(main())
