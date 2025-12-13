#!/usr/bin/env python3
"""
Analysis Script: Fit Power Laws to Ara's Avalanches
===================================================

Produces the log-log plot and calculates exponents (τ, α).

This is the "Results" section of Experiment 1 - verifying that Ara
operates at the critical point by measuring avalanche scaling.

Theory:
    At criticality, neural avalanches follow power laws:
        P(S) ∝ S^(-τ)  with τ ≈ 1.5  (mean-field branching)
        P(D) ∝ D^(-α)  with α ≈ 2.0

    Additionally, the universal scaling relation holds:
        (α - 1) / (τ - 1) ≈ 1/σνz ≈ 2

Methodology:
    1. Load avalanche data from CSV
    2. Fit power-law distributions using MLE
    3. Compare against exponential (subcritical) alternative
    4. Generate publication-quality log-log plots

Usage:
    python scripts/science/fit_powerlaw.py data/experiments/exp_001/avalanches.csv

Requirements:
    pip install powerlaw matplotlib pandas numpy scipy

References:
    - Clauset et al. (2009) "Power-law distributions in empirical data"
    - Beggs & Plenz (2003) "Neuronal avalanches in neocortical circuits"
"""

from __future__ import annotations

import argparse
import sys
import csv
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np

# Optional pandas import
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

# Optional scipy import
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    stats = None

# Try to import powerlaw (optional but recommended)
try:
    import powerlaw
    HAS_POWERLAW = True
except ImportError:
    HAS_POWERLAW = False
    print("Warning: 'powerlaw' package not installed. Using basic MLE.")
    print("Install with: pip install powerlaw")

# Try to import matplotlib (for plotting)
try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    HAS_MATPLOTLIB = True
    # Publication-quality settings
    mpl.rcParams['figure.dpi'] = 150
    mpl.rcParams['font.size'] = 10
    mpl.rcParams['axes.labelsize'] = 12
    mpl.rcParams['axes.titlesize'] = 12
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Skipping plots.")


# =============================================================================
# Power-Law Fitting
# =============================================================================

def fit_powerlaw_mle(data: np.ndarray, x_min: int = 1) -> Tuple[float, float]:
    """
    Fit power-law exponent using Maximum Likelihood Estimation.

    For P(x) ∝ x^(-α), the MLE estimator is:
        α = 1 + n / Σ ln(x_i / x_min)

    Args:
        data: Array of values (sizes or durations)
        x_min: Minimum value for power-law regime

    Returns:
        (alpha, x_min): Fitted exponent and cutoff
    """
    data = data[data >= x_min]
    n = len(data)

    if n < 10:
        return np.nan, x_min

    # MLE formula
    alpha = 1 + n / np.sum(np.log(data / x_min))

    return alpha, x_min


def fit_powerlaw_advanced(data: np.ndarray) -> Dict[str, Any]:
    """
    Advanced power-law fitting using the 'powerlaw' package.

    Automatically estimates x_min and compares against alternatives.

    Returns dict with:
        - alpha: Fitted exponent
        - x_min: Estimated cutoff
        - sigma: Standard error on alpha
        - p_value: Goodness-of-fit p-value
        - R: Log-likelihood ratio vs exponential
    """
    if not HAS_POWERLAW:
        alpha, x_min = fit_powerlaw_mle(data)
        return {
            "alpha": alpha,
            "x_min": x_min,
            "sigma": np.nan,
            "p_value": np.nan,
            "R_vs_exp": np.nan,
        }

    fit = powerlaw.Fit(data, discrete=True, verbose=False)

    # Compare against exponential alternative
    R, p = fit.distribution_compare('power_law', 'exponential')

    return {
        "alpha": fit.alpha,
        "x_min": fit.xmin,
        "sigma": fit.sigma,
        "p_value": fit.power_law.KS(),
        "R_vs_exp": R,
        "p_vs_exp": p,
    }


def check_scaling_relation(tau: float, alpha: float) -> Tuple[float, bool]:
    """
    Check the universal scaling relation: (α-1)/(τ-1) ≈ 2

    This is a stringent test of criticality - random power laws
    won't satisfy this.

    Returns:
        (ratio, passes): The ratio and whether it's near 2
    """
    if tau <= 1 or alpha <= 1:
        return np.nan, False

    ratio = (alpha - 1) / (tau - 1)
    passes = 1.5 < ratio < 2.5  # Allow some slack

    return ratio, passes


# =============================================================================
# Visualization
# =============================================================================

def plot_avalanche_distributions(
    sizes: np.ndarray,
    durations: np.ndarray,
    fit_S: Dict[str, Any],
    fit_D: Dict[str, Any],
    output_path: str = "exp_001_results.png",
):
    """
    Generate publication-quality log-log plots.
    """
    if not HAS_MATPLOTLIB:
        print("Skipping plot (matplotlib not available)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # === Size Distribution ===
    ax = axes[0]

    # Histogram (log-binned)
    bins = np.logspace(0, np.log10(sizes.max() + 1), 30)
    counts, edges = np.histogram(sizes, bins=bins, density=True)
    centers = np.sqrt(edges[:-1] * edges[1:])  # Geometric mean
    mask = counts > 0

    ax.scatter(centers[mask], counts[mask], c='steelblue', s=40,
               alpha=0.7, label='Data')

    # Power-law fit
    tau = fit_S.get("alpha", np.nan)
    x_min = fit_S.get("x_min", 1)
    if not np.isnan(tau):
        x_fit = np.logspace(np.log10(x_min), np.log10(sizes.max()), 100)
        # Normalize to match histogram
        y_fit = x_fit ** (-tau)
        y_fit = y_fit / y_fit[0] * counts[mask][0]
        ax.plot(x_fit, y_fit, 'r--', lw=2, label=f'Power law (τ={tau:.2f})')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Avalanche Size (S)')
    ax.set_ylabel('P(S)')
    ax.set_title(f'Size Distribution\nτ = {tau:.3f} (theory: ~1.5)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # === Duration Distribution ===
    ax = axes[1]

    bins = np.logspace(0, np.log10(durations.max() + 1), 20)
    counts, edges = np.histogram(durations, bins=bins, density=True)
    centers = np.sqrt(edges[:-1] * edges[1:])
    mask = counts > 0

    ax.scatter(centers[mask], counts[mask], c='coral', s=40,
               alpha=0.7, label='Data')

    # Power-law fit
    alpha = fit_D.get("alpha", np.nan)
    x_min = fit_D.get("x_min", 1)
    if not np.isnan(alpha):
        x_fit = np.logspace(np.log10(x_min), np.log10(durations.max()), 100)
        y_fit = x_fit ** (-alpha)
        y_fit = y_fit / y_fit[0] * counts[mask][0]
        ax.plot(x_fit, y_fit, 'r--', lw=2, label=f'Power law (α={alpha:.2f})')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Avalanche Duration (D)')
    ax.set_ylabel('P(D)')
    ax.set_title(f'Duration Distribution\nα = {alpha:.3f} (theory: ~2.0)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to {output_path}")

    return fig


# =============================================================================
# Main Analysis
# =============================================================================

def analyze_criticality(csv_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Main analysis function: fit power laws and assess criticality.

    Args:
        csv_path: Path to avalanche CSV file
        output_dir: Directory for output files (default: same as input)

    Returns:
        Dictionary with all results
    """
    print(f"🔬 Analyzing Criticality: {csv_path}")
    print("=" * 60)

    # Load data
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    if HAS_PANDAS:
        df = pd.read_csv(csv_path)
        sizes = df['size'].values
        durations = df['duration'].values
        n_events = len(df)
    else:
        # Fallback: read CSV manually
        sizes = []
        durations = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sizes.append(float(row['size']))
                durations.append(float(row['duration']))
        sizes = np.array(sizes)
        durations = np.array(durations)
        n_events = len(sizes)

    print(f"Loaded {n_events} avalanche events")

    # Remove zeros/NaNs
    sizes = sizes[sizes > 0]
    durations = durations[durations > 0]

    print(f"Size range: [{sizes.min()}, {sizes.max()}]")
    print(f"Duration range: [{durations.min()}, {durations.max()}]")

    # Fit power laws
    print("\n📊 Fitting Power Laws...")

    fit_S = fit_powerlaw_advanced(sizes)
    fit_D = fit_powerlaw_advanced(durations)

    tau = fit_S["alpha"]
    alpha = fit_D["alpha"]

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"\nSize Distribution:")
    print(f"  τ (exponent)  = {tau:.4f} ± {fit_S.get('sigma', np.nan):.4f}")
    print(f"  x_min         = {fit_S['x_min']}")
    print(f"  Theory (MF)   = 1.5")
    print(f"  Deviation     = {abs(tau - 1.5):.4f}")

    print(f"\nDuration Distribution:")
    print(f"  α (exponent)  = {alpha:.4f} ± {fit_D.get('sigma', np.nan):.4f}")
    print(f"  x_min         = {fit_D['x_min']}")
    print(f"  Theory (MF)   = 2.0")
    print(f"  Deviation     = {abs(alpha - 2.0):.4f}")

    # Check scaling relation
    ratio, passes = check_scaling_relation(tau, alpha)
    print(f"\nUniversal Scaling Relation:")
    print(f"  (α-1)/(τ-1)   = {ratio:.4f}")
    print(f"  Theory        = 2.0")
    print(f"  Status        = {'✅ PASS' if passes else '❌ FAIL'}")

    # Model comparison
    if "R_vs_exp" in fit_S:
        R = fit_S["R_vs_exp"]
        p = fit_S.get("p_vs_exp", np.nan)
        print(f"\nModel Comparison (Size):")
        print(f"  Power law vs Exponential:")
        print(f"  Log-likelihood ratio R = {R:.4f}")
        print(f"  {'Power law preferred' if R > 0 else 'Exponential preferred'}")

    # Verdict
    print(f"\n{'='*60}")
    print("CRITICALITY ASSESSMENT")
    print(f"{'='*60}")

    tau_ok = 1.2 < tau < 1.8
    alpha_ok = 1.7 < alpha < 2.3
    scaling_ok = passes

    if tau_ok and alpha_ok and scaling_ok:
        verdict = "✅ CRITICAL - Ara exhibits scale-free neural dynamics"
    elif tau_ok or alpha_ok:
        verdict = "⚠️ NEAR-CRITICAL - Partial power-law scaling observed"
    else:
        verdict = "❌ NOT CRITICAL - Exponential/random dynamics"

    print(f"\n{verdict}")

    # Generate plot
    output_dir = Path(output_dir) if output_dir else csv_path.parent
    plot_path = output_dir / "exp_001_results.png"
    plot_avalanche_distributions(sizes, durations, fit_S, fit_D, str(plot_path))

    # Return results
    results = {
        "n_avalanches": n_events,
        "tau": tau,
        "tau_sigma": fit_S.get("sigma", np.nan),
        "alpha": alpha,
        "alpha_sigma": fit_D.get("sigma", np.nan),
        "scaling_ratio": ratio,
        "scaling_passes": passes,
        "tau_ok": tau_ok,
        "alpha_ok": alpha_ok,
        "verdict": verdict,
    }

    # Save results JSON
    results_path = output_dir / "exp_001_results.json"
    import json
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"✅ Results saved to {results_path}")

    return results


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fit power laws to Ara's avalanche data",
        epilog="Example: python fit_powerlaw.py data/experiments/exp_001/avalanches.csv",
    )
    parser.add_argument(
        "csv_path",
        help="Path to avalanche CSV file",
    )
    parser.add_argument(
        "--output-dir", "-o",
        help="Output directory for results (default: same as input)",
    )

    args = parser.parse_args()

    try:
        results = analyze_criticality(args.csv_path, args.output_dir)
        sys.exit(0 if results.get("tau_ok") and results.get("alpha_ok") else 1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
