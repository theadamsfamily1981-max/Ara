#!/usr/bin/env python3
"""
Autocorrelation-Based Width Detection

Implements FFT-based autocorrelation to determine the optimal "frame width"
for reshaping 1D bitstreams into 2D images. This is critical for Arria 10
analysis where the physical chip layout (LAB columns, routing channels) has
periodic structure.

Algorithm:
1. Compute FFT of bit sequence
2. Compute autocorrelation via inverse FFT of power spectrum
3. Find first major peak (excluding DC)
4. Peak position = repeat interval = frame width

Theoretical Basis:
FPGA configuration data has strong spatial periodicity due to regular LAB/DSP
column layout. Autocorrelation detects this periodicity even when exact frame
boundaries are unknown.

Reference: Section 8.2.2 of bitstream forensics paper
"""

import numpy as np
from scipy import signal
from typing import List, Tuple, Optional
import warnings


# Common FPGA frame widths (bits) based on device families
# Arria 10: Typically 256-2048 bits per frame
# Stratix 10: Variable, 512-4096 bits
COMMON_WIDTHS = [256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]


def infer_frame_width(
    bits: np.ndarray,
    max_width: int = 4096,
    candidates: Optional[List[int]] = None,
    use_fft: bool = True
) -> int:
    """
    Infer the optimal frame width for 2D image folding via autocorrelation.

    Args:
        bits: 1D bit array (0/1 values) from bitstream
        max_width: Maximum width to search (default 4096)
        candidates: Optional list of specific widths to test
        use_fft: Use FFT-based autocorrelation (fast) vs direct (slow but exact)

    Returns:
        Detected frame width in bits

    Example:
        >>> from rbf_utils import read_rbf, bytes_to_bits
        >>> rbf = read_rbf('design.rbf')
        >>> bits = bytes_to_bits(rbf)
        >>> width = infer_frame_width(bits)
        >>> print(f"Detected width: {width} bits")
    """
    if candidates is None:
        candidates = [w for w in COMMON_WIDTHS if w <= max_width]

    if len(bits) < max_width * 2:
        warnings.warn(
            f"Bitstream ({len(bits)} bits) too short for reliable width detection. "
            f"Recommend at least {max_width * 2} bits."
        )

    # Compute autocorrelation
    if use_fft:
        autocorr = _fft_autocorrelation(bits)
    else:
        autocorr = _direct_autocorrelation(bits, max_lag=max_width)

    # Score each candidate width
    scores = []
    for width in candidates:
        score = _score_width(autocorr, width)
        scores.append((width, score))

    # Sort by score (descending)
    scores.sort(key=lambda x: x[1], reverse=True)

    # Return best width
    best_width, best_score = scores[0]
    return best_width


def _fft_autocorrelation(bits: np.ndarray) -> np.ndarray:
    """
    Compute autocorrelation via FFT (Wiener-Khinchin theorem).

    Autocorrelation R(τ) = IFFT(|FFT(x)|²)

    This is O(N log N) vs O(N²) for direct computation.
    """
    # Convert to float for FFT
    x = bits.astype(np.float64)

    # Subtract mean (remove DC component)
    x = x - np.mean(x)

    # Compute FFT
    fft_x = np.fft.fft(x)

    # Power spectrum
    power = np.abs(fft_x) ** 2

    # Inverse FFT to get autocorrelation
    autocorr = np.fft.ifft(power).real

    # Normalize
    autocorr = autocorr / autocorr[0]

    return autocorr


def _direct_autocorrelation(bits: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Compute autocorrelation directly (for validation/debugging).

    R(τ) = Σ x[n] * x[n+τ]

    Slow but exact. Use only for small lags.
    """
    x = bits.astype(np.float64) - np.mean(bits)
    autocorr = np.correlate(x, x, mode='full')
    autocorr = autocorr[len(autocorr)//2:]  # Keep only positive lags
    autocorr = autocorr[:max_lag]
    autocorr = autocorr / autocorr[0]  # Normalize
    return autocorr


def _score_width(autocorr: np.ndarray, width: int) -> float:
    """
    Score a candidate width based on autocorrelation peak strength.

    A strong peak at lag=width indicates that rows separated by `width` bits
    are highly correlated (i.e., they represent similar structures like LAB
    columns).

    Score combines:
    1. Peak magnitude at exact width
    2. Average correlation in neighborhood of width (±10%)
    3. Ratio to nearby non-peaks (contrast)
    """
    if width >= len(autocorr):
        return 0.0

    # Peak magnitude at width
    peak_value = autocorr[width]

    # Average in neighborhood (±10%)
    window_size = max(1, int(width * 0.1))
    start = max(1, width - window_size)
    end = min(len(autocorr), width + window_size)
    neighborhood_avg = np.mean(autocorr[start:end])

    # Baseline (average between half-width and width, excluding peak)
    baseline_start = max(1, width // 2)
    baseline_end = max(1, width - window_size)
    if baseline_end > baseline_start:
        baseline = np.mean(autocorr[baseline_start:baseline_end])
    else:
        baseline = 0.1

    # Score: peak strength relative to baseline
    contrast = (neighborhood_avg - baseline) / (baseline + 1e-6)

    # Combined score
    score = peak_value * (1.0 + contrast)

    return score


def get_top_k_widths(
    bits: np.ndarray,
    max_width: int = 4096,
    candidates: Optional[List[int]] = None,
    k: int = 3
) -> List[Tuple[int, float]]:
    """
    Get top-k candidate widths ranked by score.

    Useful for visualizing multiple potential widths to manually verify
    which produces the cleanest 2D image.

    Args:
        bits: 1D bit array
        max_width: Maximum width to search
        candidates: Specific widths to test (or None for defaults)
        k: Number of top candidates to return

    Returns:
        List of (width, score) tuples, sorted by score descending
    """
    if candidates is None:
        candidates = [w for w in COMMON_WIDTHS if w <= max_width]

    # Compute autocorrelation once
    autocorr = _fft_autocorrelation(bits)

    # Score all candidates
    scores = [(width, _score_width(autocorr, width)) for width in candidates]

    # Sort and return top-k
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]


def visualize_autocorrelation(bits: np.ndarray, max_lag: int = 4096, save_path: Optional[str] = None):
    """
    Plot autocorrelation function for visual inspection.

    Useful for debugging and understanding the periodicity of a specific
    bitstream.

    Args:
        bits: 1D bit array
        max_lag: Maximum lag to plot
        save_path: Optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for visualization. Install with: pip install matplotlib")

    autocorr = _fft_autocorrelation(bits)
    autocorr = autocorr[:max_lag]

    # Find peaks
    peaks, properties = signal.find_peaks(autocorr, height=0.1, distance=100)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(autocorr, linewidth=0.5, label='Autocorrelation')
    ax.plot(peaks, autocorr[peaks], 'rx', markersize=10, label='Peaks')

    # Annotate top peaks
    if len(peaks) > 0:
        top_peaks = peaks[np.argsort(autocorr[peaks])[-5:]]  # Top 5
        for peak in top_peaks:
            ax.annotate(
                f'{peak}',
                xy=(peak, autocorr[peak]),
                xytext=(peak, autocorr[peak] + 0.1),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=8
            )

    ax.set_xlabel('Lag (bits)')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('Autocorrelation Function - Frame Width Detection')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved autocorrelation plot to: {save_path}")
    else:
        plt.show()


if __name__ == '__main__':
    # CLI for testing
    import sys
    from rbf_utils import read_rbf, bytes_to_bits

    if len(sys.argv) < 2:
        print("Usage: python autocorr_width.py <bitstream.rbf> [--visualize]")
        sys.exit(1)

    rbf_path = sys.argv[1]
    visualize = '--visualize' in sys.argv

    print(f"Loading bitstream: {rbf_path}")
    rbf_bytes = read_rbf(rbf_path)
    bits = bytes_to_bits(rbf_bytes)

    print(f"Bitstream size: {len(bits):,} bits")
    print("\nDetecting frame width...")

    # Get top 5 candidates
    top_widths = get_top_k_widths(bits, k=5)

    print("\nTop 5 candidate widths:")
    print("-" * 40)
    for i, (width, score) in enumerate(top_widths, 1):
        print(f"{i}. Width {width:4d} bits - Score: {score:.6f}")

    best_width, best_score = top_widths[0]
    print(f"\nRecommended width: {best_width} bits")

    if visualize:
        print("\nGenerating autocorrelation plot...")
        visualize_autocorrelation(bits, max_lag=4096, save_path='autocorr_plot.png')
