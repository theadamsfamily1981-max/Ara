"""
width_detection.py

Autocorrelation-based frame width detection for FPGA bitstreams.

The goal is to find the "natural" width of the bitstream when arranged as a 2D image.
FPGA configuration data often has repeating structures (LUTs, routing, block RAM) that
create periodic patterns. Autocorrelation helps identify these periods.

Algorithm:
1. Compute autocorrelation for different width candidates
2. Find width with maximum autocorrelation peak (strongest periodicity)
3. Optionally use entropy minimization as secondary metric
"""

import numpy as np
from typing import List, Tuple, Optional


def guess_width(
    bits: np.ndarray,
    min_width: int = 800,
    max_width: int = 4096,
    step: int = 8,
    method: str = 'autocorrelation'
) -> int:
    """
    Guess optimal frame width for bitstream-to-image conversion.

    Args:
        bits: 1D array of bits (0 or 1)
        min_width: Minimum width to test (in bits)
        max_width: Maximum width to test
        step: Width increment step
        method: Detection method ('autocorrelation' or 'entropy')

    Returns:
        Optimal width in bits

    Example:
        >>> bits = np.random.randint(0, 2, 1000000)
        >>> width = guess_width(bits, min_width=1000, max_width=2000)
        >>> print(f"Detected width: {width}")
    """
    print(f"🔍 Scanning width range {min_width}-{max_width} (step={step})")
    print(f"📊 Method: {method}")

    if method == 'autocorrelation':
        return _guess_width_autocorr(bits, min_width, max_width, step)
    elif method == 'entropy':
        return _guess_width_entropy(bits, min_width, max_width, step)
    else:
        raise ValueError(f"Unknown method: {method}")


def _guess_width_autocorr(
    bits: np.ndarray,
    min_width: int,
    max_width: int,
    step: int
) -> int:
    """
    Use autocorrelation to find optimal width.

    The intuition: if we reshape the bitstream with the correct width,
    each row will be structurally similar to nearby rows (due to FPGA
    column structure). This creates a strong autocorrelation peak.

    Args:
        bits: 1D bit array
        min_width: Minimum width to test
        max_width: Maximum width to test
        step: Increment step

    Returns:
        Width with maximum autocorrelation score
    """
    candidates = range(min_width, max_width + 1, step)
    scores = []

    print(f"Testing {len(list(candidates))} width candidates...")

    for width in candidates:
        score = autocorrelation_score(bits, width)
        scores.append((width, score))

    # Find maximum
    best_width, best_score = max(scores, key=lambda x: x[1])

    print(f"✅ Best width: {best_width} (score: {best_score:.6f})")

    # Show top 5 candidates
    top5 = sorted(scores, key=lambda x: x[1], reverse=True)[:5]
    print("\n📊 Top 5 candidates:")
    for i, (w, s) in enumerate(top5, 1):
        print(f"   {i}. Width {w:4d}: {s:.6f}")

    return best_width


def _guess_width_entropy(
    bits: np.ndarray,
    min_width: int,
    max_width: int,
    step: int
) -> int:
    """
    Use entropy minimization to find optimal width.

    The intuition: correct width produces an image with more structure
    (lower entropy per row) compared to incorrect width.

    Args:
        bits: 1D bit array
        min_width: Minimum width to test
        max_width: Maximum width to test
        step: Increment step

    Returns:
        Width with minimum average row entropy
    """
    candidates = range(min_width, max_width + 1, step)
    scores = []

    print(f"Testing {len(list(candidates))} width candidates...")

    for width in candidates:
        # Reshape to 2D
        usable_len = (len(bits) // width) * width
        if usable_len < width * 10:  # Need at least 10 rows
            continue

        image = bits[:usable_len].reshape(-1, width)

        # Calculate average row entropy
        avg_entropy = np.mean([_row_entropy(row) for row in image[:100]])
        scores.append((width, avg_entropy))

    # Find minimum entropy
    best_width, best_entropy = min(scores, key=lambda x: x[1])

    print(f"✅ Best width: {best_width} (entropy: {best_entropy:.6f})")

    # Show top 5 candidates
    top5 = sorted(scores, key=lambda x: x[1])[:5]
    print("\n📊 Top 5 candidates:")
    for i, (w, e) in enumerate(top5, 1):
        print(f"   {i}. Width {w:4d}: {e:.6f}")

    return best_width


def autocorrelation_score(bits: np.ndarray, width: int, lag: int = 1) -> float:
    """
    Compute autocorrelation score for a given width.

    Algorithm:
    1. Reshape bits into 2D array with specified width
    2. Compute correlation between each row and row at offset 'lag'
    3. Return average correlation

    Args:
        bits: 1D bit array
        width: Frame width to test
        lag: Row offset for correlation (default: 1 = adjacent rows)

    Returns:
        Average correlation coefficient (0.0-1.0)
    """
    # Truncate to multiple of width
    usable_len = (len(bits) // width) * width
    if usable_len < width * (lag + 2):  # Need enough rows
        return 0.0

    # Reshape to 2D
    image = bits[:usable_len].reshape(-1, width)
    num_rows = image.shape[0]

    if num_rows < lag + 1:
        return 0.0

    # Compute row-to-row correlation
    correlations = []
    for i in range(num_rows - lag):
        row1 = image[i].astype(float)
        row2 = image[i + lag].astype(float)

        # Pearson correlation coefficient
        corr = np.corrcoef(row1, row2)[0, 1]

        # Handle NaN (constant rows)
        if np.isnan(corr):
            corr = 0.0

        correlations.append(corr)

    # Return average correlation
    return float(np.mean(correlations))


def autocorrelation_scan(
    bits: np.ndarray,
    width: int,
    max_lag: int = 50
) -> np.ndarray:
    """
    Scan autocorrelation across multiple lags for analysis.

    Useful for visualizing the periodicity structure.

    Args:
        bits: 1D bit array
        width: Frame width
        max_lag: Maximum lag to test

    Returns:
        Array of correlation scores for lags 0..max_lag
    """
    scores = []
    for lag in range(1, max_lag + 1):
        score = autocorrelation_score(bits, width, lag=lag)
        scores.append(score)

    return np.array(scores)


def _row_entropy(row: np.ndarray) -> float:
    """
    Calculate Shannon entropy of a single row.

    For binary data, this measures randomness (0 = all same, 1 = random).

    Args:
        row: 1D array of bits

    Returns:
        Entropy in bits (0.0-1.0 for binary)
    """
    # Count 0s and 1s
    counts = np.bincount(row.astype(int), minlength=2)
    total = len(row)

    entropy = 0.0
    for count in counts:
        if count > 0:
            p = count / total
            entropy -= p * np.log2(p)

    return entropy


def visualize_width_scan(
    bits: np.ndarray,
    min_width: int = 800,
    max_width: int = 4096,
    step: int = 16,
    save_path: Optional[str] = None
):
    """
    Generate visualization of autocorrelation scores across width range.

    Requires matplotlib. Useful for debugging and presentations.

    Args:
        bits: 1D bit array
        min_width: Minimum width to test
        max_width: Maximum width to test
        step: Increment step
        save_path: Optional path to save plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("❌ matplotlib not available for visualization")
        return

    # Compute scores
    widths = list(range(min_width, max_width + 1, step))
    scores = [autocorrelation_score(bits, w) for w in widths]

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(widths, scores, 'b-', linewidth=2)
    plt.xlabel('Width (bits)', fontsize=12)
    plt.ylabel('Autocorrelation Score', fontsize=12)
    plt.title('Width Detection via Autocorrelation', fontsize=14)
    plt.grid(True, alpha=0.3)

    # Mark maximum
    best_idx = np.argmax(scores)
    best_width = widths[best_idx]
    best_score = scores[best_idx]
    plt.axvline(best_width, color='r', linestyle='--', label=f'Optimal: {best_width}')
    plt.scatter([best_width], [best_score], color='r', s=100, zorder=5)

    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"💾 Saved plot to {save_path}")
    else:
        plt.show()


def detect_multiple_widths(
    bits: np.ndarray,
    min_width: int = 800,
    max_width: int = 4096,
    step: int = 8,
    num_peaks: int = 3
) -> List[Tuple[int, float]]:
    """
    Detect multiple possible widths (useful for multi-region bitstreams).

    Some FPGA bitstreams have multiple configuration regions with different widths.

    Args:
        bits: 1D bit array
        min_width: Minimum width to test
        max_width: Maximum width to test
        step: Increment step
        num_peaks: Number of peaks to return

    Returns:
        List of (width, score) tuples, sorted by score descending
    """
    widths = list(range(min_width, max_width + 1, step))
    scores = [(w, autocorrelation_score(bits, w)) for w in widths]

    # Sort by score descending
    scores.sort(key=lambda x: x[1], reverse=True)

    return scores[:num_peaks]


# Example usage and testing
if __name__ == '__main__':
    print("=" * 60)
    print("A10 Width Detection Test")
    print("=" * 60)
    print()

    # Generate synthetic bitstream with known width
    true_width = 1024
    num_rows = 500

    print(f"🔧 Generating synthetic bitstream (width={true_width}, rows={num_rows})")

    # Create structured data with repeating patterns
    row_template = np.random.randint(0, 2, true_width)
    rows = []
    for i in range(num_rows):
        # Add some noise to each row (90% similar to template)
        noise = np.random.random(true_width) < 0.1
        row = row_template.copy()
        row[noise] = 1 - row[noise]
        rows.append(row)

    bits = np.concatenate(rows)
    print(f"📦 Total bits: {len(bits):,}")

    print("\n" + "=" * 60)
    print("Testing Autocorrelation Method")
    print("=" * 60)
    print()

    # Test width detection
    detected = guess_width(bits, min_width=900, max_width=1200, step=4)

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"True width:     {true_width}")
    print(f"Detected width: {detected}")
    print(f"Error:          {abs(detected - true_width)} bits")
    print()

    if abs(detected - true_width) <= 16:
        print("✅ Detection successful!")
    else:
        print("⚠️  Detection may need tuning")
