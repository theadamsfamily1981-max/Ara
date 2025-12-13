#!/usr/bin/env python3
"""
Edge of Autumn: Balanced Regime Finder

Implements the empirical protocol for finding the Edge of Autumn—
the hyperparameter regime where structure, performance, and robustness
are simultaneously acceptable.

Based on the existence theorem:
    Under continuity + boundary conditions + non-triviality,
    the balanced region 𝓑 is guaranteed non-empty.

Usage:
    python -m ara.neuro.arabrain.edge_of_autumn --demo
    python -m ara.neuro.arabrain.edge_of_autumn --beta_min 0.1 --beta_max 10.0 --num_points 20
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class MetricPoint:
    """Single point in the β-sweep."""
    beta: float
    S: float  # Structure metric
    P: float  # Performance metric
    R: float  # Robustness metric

    # Raw components (for analysis)
    mig: Optional[float] = None
    dci: Optional[float] = None
    edi: Optional[float] = None
    accuracy: Optional[float] = None
    recon_loss: Optional[float] = None
    noise_stability: Optional[float] = None

    def __repr__(self):
        return f"MetricPoint(β={self.beta:.2f}, S={self.S:.3f}, P={self.P:.3f}, R={self.R:.3f})"


@dataclass
class BalancedRegime:
    """The Edge of Autumn: where all metrics meet thresholds."""
    beta_star: float           # Optimal β
    beta_range: Tuple[float, float]  # [β_L, β_U] interval
    points: List[MetricPoint]  # All points in the balanced region

    # Thresholds used
    S_star: float
    P_star: float
    R_star: float

    # Quality metrics
    margin: float  # Min distance from threshold
    width: float   # β_U - β_L

    def __repr__(self):
        return (f"BalancedRegime(β*={self.beta_star:.3f}, "
                f"range=[{self.beta_range[0]:.3f}, {self.beta_range[1]:.3f}], "
                f"width={self.width:.3f}, margin={self.margin:.3f})")


@dataclass
class EdgeOfAutumnResult:
    """Complete result of the Edge of Autumn analysis."""
    found: bool
    regime: Optional[BalancedRegime]
    all_points: List[MetricPoint]

    # Deficit analysis
    F_min: float  # Minimum max-deficit
    F_argmin_beta: float  # β where F is minimized

    # Diagnostics
    assumptions_satisfied: Dict[str, bool] = field(default_factory=dict)
    message: str = ""


# =============================================================================
# Metric Computation
# =============================================================================

def compute_structure_metric(
    z: np.ndarray,
    factors: Optional[np.ndarray] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute structure metric S from latent representations.

    Combines MIG, DCI, and EDI into a single score.

    Args:
        z: Latent representations (N, latent_dim)
        factors: Ground truth factors (N, num_factors) if available

    Returns:
        S: Combined structure score in [0, 1]
        components: Individual metric values
    """
    # Simplified metrics when no ground truth factors
    if factors is None:
        # Use intrinsic measures
        # 1. Variance explained (each latent should capture variance)
        var_per_dim = np.var(z, axis=0)
        total_var = np.sum(var_per_dim)
        var_ratio = var_per_dim / (total_var + 1e-8)
        entropy = -np.sum(var_ratio * np.log(var_ratio + 1e-8))
        max_entropy = np.log(z.shape[1])
        uniformity = entropy / max_entropy  # Higher = more uniform use of latents

        # 2. Correlation structure (latents should be uncorrelated)
        corr = np.corrcoef(z.T)
        off_diag = corr[np.triu_indices_from(corr, k=1)]
        decorrelation = 1.0 - np.mean(np.abs(off_diag))

        # 3. Sparsity (each sample should activate few latents strongly)
        z_normalized = (z - z.mean(axis=0)) / (z.std(axis=0) + 1e-8)
        sparsity = np.mean(np.abs(z_normalized) < 1.0)  # Fraction near zero

        S = (uniformity + decorrelation + sparsity) / 3.0

        return S, {
            'uniformity': uniformity,
            'decorrelation': decorrelation,
            'sparsity': sparsity,
            'mig': None,
            'dci': None,
            'edi': None,
        }

    # Full metrics with ground truth factors
    # (Implement MIG, DCI, EDI properly here)
    # For now, placeholder
    return 0.5, {'mig': 0.5, 'dci': 0.5, 'edi': 0.5}


def compute_performance_metric(
    predictions: np.ndarray,
    targets: np.ndarray,
    recon: Optional[np.ndarray] = None,
    original: Optional[np.ndarray] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute performance metric P.

    Combines classification accuracy with reconstruction quality.

    Args:
        predictions: Model predictions (N,) or (N, 1)
        targets: Ground truth labels (N,)
        recon: Reconstructed inputs (optional)
        original: Original inputs (optional)

    Returns:
        P: Combined performance score in [0, 1]
        components: Individual metric values
    """
    predictions = predictions.flatten()
    targets = targets.flatten()

    # Classification accuracy
    if predictions.max() <= 1.0 and predictions.min() >= 0.0:
        # Probabilities
        pred_labels = (predictions > 0.5).astype(float)
    else:
        pred_labels = predictions

    accuracy = np.mean(pred_labels == targets)

    # Reconstruction loss (if available)
    if recon is not None and original is not None:
        mse = np.mean((recon - original) ** 2)
        recon_score = np.exp(-mse)  # Transform to [0, 1], higher = better
    else:
        recon_score = 1.0

    P = 0.7 * accuracy + 0.3 * recon_score

    return P, {
        'accuracy': accuracy,
        'recon_score': recon_score,
    }


def compute_robustness_metric(
    model_fn: Callable,
    x: np.ndarray,
    noise_levels: List[float] = [0.01, 0.05, 0.1],
    rng: Optional[Any] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute robustness metric R.

    Measures stability of predictions under input perturbations.

    Args:
        model_fn: Function that returns predictions given input
        x: Input data (N, ...)
        noise_levels: Noise standard deviations to test
        rng: Random number generator

    Returns:
        R: Robustness score in [0, 1]
        components: Individual stability values
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Get base predictions
    base_pred = model_fn(x)

    stabilities = []
    for noise_std in noise_levels:
        # Add noise
        noise = rng.normal(0, noise_std, x.shape).astype(x.dtype)
        x_noisy = x + noise

        # Get noisy predictions
        noisy_pred = model_fn(x_noisy)

        # Measure stability (1 - normalized difference)
        diff = np.mean(np.abs(noisy_pred - base_pred))
        max_diff = np.abs(base_pred).mean() + 1e-8
        stability = 1.0 - min(1.0, diff / max_diff)
        stabilities.append(stability)

    R = np.mean(stabilities)

    return R, {
        f'stability_{noise_std}': stab
        for noise_std, stab in zip(noise_levels, stabilities)
    }


# =============================================================================
# Deficit Functions
# =============================================================================

def compute_deficits(
    point: MetricPoint,
    S_star: float,
    P_star: float,
    R_star: float,
) -> Tuple[float, float, float, float]:
    """
    Compute deficit functions and max-deficit F.

    Returns:
        f_S, f_P, f_R: Individual deficits
        F: Max deficit
    """
    f_S = S_star - point.S
    f_P = P_star - point.P
    f_R = R_star - point.R
    F = max(f_S, f_P, f_R)
    return f_S, f_P, f_R, F


def is_in_balanced_region(
    point: MetricPoint,
    S_star: float,
    P_star: float,
    R_star: float,
) -> bool:
    """Check if point is in the balanced region 𝓑."""
    return point.S >= S_star and point.P >= P_star and point.R >= R_star


# =============================================================================
# Main Algorithm
# =============================================================================

def find_edge_of_autumn(
    sweep_results: List[MetricPoint],
    S_star: Optional[float] = None,
    P_star: Optional[float] = None,
    R_star: Optional[float] = None,
    threshold_percentile: float = 50.0,
) -> EdgeOfAutumnResult:
    """
    Find the Edge of Autumn (balanced regime) from β-sweep results.

    Args:
        sweep_results: List of MetricPoints from β-sweep
        S_star, P_star, R_star: Thresholds (if None, use percentiles)
        threshold_percentile: Percentile for automatic threshold selection

    Returns:
        EdgeOfAutumnResult with found regime or diagnostic info
    """
    if len(sweep_results) < 3:
        return EdgeOfAutumnResult(
            found=False,
            regime=None,
            all_points=sweep_results,
            F_min=float('inf'),
            F_argmin_beta=0.0,
            message="Insufficient data points (need at least 3)",
        )

    # Sort by beta
    points = sorted(sweep_results, key=lambda p: p.beta)

    # Set thresholds if not provided
    if S_star is None:
        S_star = np.percentile([p.S for p in points], threshold_percentile)
    if P_star is None:
        P_star = np.percentile([p.P for p in points], threshold_percentile)
    if R_star is None:
        R_star = np.percentile([p.R for p in points], threshold_percentile)

    # Check assumptions
    assumptions = {}

    # A2: Boundary behavior
    beta_min_point = points[0]
    beta_max_point = points[-1]

    assumptions['A2_S_low_at_min'] = beta_min_point.S < S_star
    assumptions['A2_R_low_at_min'] = beta_min_point.R < R_star
    assumptions['A2_P_low_at_max'] = beta_max_point.P < P_star

    # A3: No global failure
    all_below = [
        p for p in points
        if p.S < S_star and p.P < P_star and p.R < R_star
    ]
    assumptions['A3_no_global_failure'] = len(all_below) == 0

    # Compute deficits
    F_values = []
    for p in points:
        f_S, f_P, f_R, F = compute_deficits(p, S_star, P_star, R_star)
        F_values.append((p.beta, F))

    # Find minimum F
    min_idx = np.argmin([f[1] for f in F_values])
    F_min = F_values[min_idx][1]
    F_argmin_beta = F_values[min_idx][0]

    # Find balanced region
    balanced_points = [
        p for p in points
        if is_in_balanced_region(p, S_star, P_star, R_star)
    ]

    if not balanced_points:
        # No balanced region found
        return EdgeOfAutumnResult(
            found=False,
            regime=None,
            all_points=points,
            F_min=F_min,
            F_argmin_beta=F_argmin_beta,
            assumptions_satisfied=assumptions,
            message=f"No balanced region found. Min F={F_min:.4f} at β={F_argmin_beta:.3f}",
        )

    # Compute optimal β* (minimize max deficit within balanced region)
    best_point = None
    best_margin = float('-inf')

    for p in balanced_points:
        # Margin = how far above all thresholds
        margin = min(p.S - S_star, p.P - P_star, p.R - R_star)
        if margin > best_margin:
            best_margin = margin
            best_point = p

    # Compute regime bounds
    betas = [p.beta for p in balanced_points]
    beta_L, beta_U = min(betas), max(betas)

    regime = BalancedRegime(
        beta_star=best_point.beta,
        beta_range=(beta_L, beta_U),
        points=balanced_points,
        S_star=S_star,
        P_star=P_star,
        R_star=R_star,
        margin=best_margin,
        width=beta_U - beta_L,
    )

    return EdgeOfAutumnResult(
        found=True,
        regime=regime,
        all_points=points,
        F_min=F_min,
        F_argmin_beta=F_argmin_beta,
        assumptions_satisfied=assumptions,
        message=f"Edge of Autumn found at β*={best_point.beta:.3f}",
    )


# =============================================================================
# β-Sweep Runner
# =============================================================================

def run_beta_sweep(
    model_class,
    train_fn: Callable,
    eval_fn: Callable,
    data: Tuple[np.ndarray, np.ndarray],
    beta_range: Tuple[float, float] = (0.1, 10.0),
    num_points: int = 10,
    **model_kwargs,
) -> List[MetricPoint]:
    """
    Run β-sweep to collect metrics.

    Args:
        model_class: Model class (e.g., EEGAraBrain)
        train_fn: Function to train model
        eval_fn: Function to evaluate model, returns (z, predictions, recon)
        data: (x_train, y_train) tuple
        beta_range: (beta_min, beta_max)
        num_points: Number of β values to test
        **model_kwargs: Additional model arguments

    Returns:
        List of MetricPoints
    """
    betas = np.linspace(beta_range[0], beta_range[1], num_points)
    results = []

    x_train, y_train = data

    for beta in betas:
        print(f"Training with β={beta:.3f}...")

        # Create and train model
        model = model_class(beta=beta, **model_kwargs)
        train_fn(model, x_train, y_train)

        # Evaluate
        z, predictions, recon = eval_fn(model, x_train)

        # Compute metrics
        S, s_components = compute_structure_metric(z)
        P, p_components = compute_performance_metric(predictions, y_train, recon, x_train)

        # Robustness (simplified for sweep)
        R = 0.5 + 0.5 * np.random.random()  # Placeholder

        point = MetricPoint(
            beta=beta,
            S=S,
            P=P,
            R=R,
            mig=s_components.get('mig'),
            dci=s_components.get('dci'),
            edi=s_components.get('edi'),
            accuracy=p_components.get('accuracy'),
            recon_loss=1.0 - p_components.get('recon_score', 1.0),
        )
        results.append(point)
        print(f"  {point}")

    return results


# =============================================================================
# Demo
# =============================================================================

def demo_edge_of_autumn():
    """Demonstrate Edge of Autumn finding with synthetic data."""
    print("\n" + "=" * 70)
    print("EDGE OF AUTUMN DEMONSTRATION")
    print("Finding the balanced representation regime")
    print("=" * 70)

    # Generate synthetic sweep results
    # Simulating typical β-sweep behavior:
    # - Low β: high P, low S, low R
    # - High β: low P, high S, high R
    np.random.seed(42)

    betas = np.linspace(0.5, 10.0, 20)
    points = []

    for beta in betas:
        # S increases with β (structure improves with regularization)
        S = 0.3 + 0.5 * (1 - np.exp(-beta / 3)) + 0.1 * np.random.randn()
        S = np.clip(S, 0, 1)

        # P decreases with β (performance degrades with over-regularization)
        P = 0.9 - 0.4 * (1 - np.exp(-beta / 5)) + 0.1 * np.random.randn()
        P = np.clip(P, 0, 1)

        # R increases with β (robustness improves with regularization)
        R = 0.2 + 0.6 * (1 - np.exp(-beta / 4)) + 0.1 * np.random.randn()
        R = np.clip(R, 0, 1)

        points.append(MetricPoint(beta=beta, S=S, P=P, R=R))

    print("\nSweep Results:")
    print("-" * 70)
    print(f"{'β':>8} {'S':>8} {'P':>8} {'R':>8}")
    print("-" * 70)
    for p in points:
        print(f"{p.beta:>8.2f} {p.S:>8.3f} {p.P:>8.3f} {p.R:>8.3f}")

    # Find Edge of Autumn
    print("\n" + "=" * 70)
    print("Finding Edge of Autumn...")
    print("=" * 70)

    result = find_edge_of_autumn(points, threshold_percentile=50)

    print(f"\nResult: {result.message}")
    print(f"Min max-deficit F = {result.F_min:.4f} at β = {result.F_argmin_beta:.3f}")

    print("\nAssumptions check:")
    for assumption, satisfied in result.assumptions_satisfied.items():
        status = "✓" if satisfied else "✗"
        print(f"  {status} {assumption}")

    if result.found:
        regime = result.regime
        print(f"\n{'=' * 70}")
        print("EDGE OF AUTUMN FOUND!")
        print(f"{'=' * 70}")
        print(f"  Optimal β*:     {regime.beta_star:.3f}")
        print(f"  Balanced range: [{regime.beta_range[0]:.3f}, {regime.beta_range[1]:.3f}]")
        print(f"  Width:          {regime.width:.3f}")
        print(f"  Margin:         {regime.margin:.3f}")
        print(f"\n  Thresholds used:")
        print(f"    S* = {regime.S_star:.3f}")
        print(f"    P* = {regime.P_star:.3f}")
        print(f"    R* = {regime.R_star:.3f}")
        print(f"\n  Points in balanced region: {len(regime.points)}")
        for p in regime.points:
            print(f"    {p}")
    else:
        print("\nNo balanced regime found with current thresholds.")
        print("Consider adjusting thresholds or extending β range.")

    # Visualize (text-based)
    print("\n" + "=" * 70)
    print("Metric Curves (text visualization)")
    print("=" * 70)

    for metric_name, metric_key in [('S (Structure)', 'S'), ('P (Performance)', 'P'), ('R (Robustness)', 'R')]:
        print(f"\n{metric_name}:")
        values = [getattr(p, metric_key) for p in points]
        for i, (p, v) in enumerate(zip(points, values)):
            bar_len = int(v * 40)
            bar = "█" * bar_len + "░" * (40 - bar_len)
            marker = "*" if result.found and regime.beta_range[0] <= p.beta <= regime.beta_range[1] else " "
            print(f"  β={p.beta:5.2f} |{bar}| {v:.2f} {marker}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Find the Edge of Autumn (balanced regime)"
    )
    parser.add_argument('--demo', action='store_true',
                       help='Run demonstration with synthetic data')
    parser.add_argument('--beta_min', type=float, default=0.5)
    parser.add_argument('--beta_max', type=float, default=10.0)
    parser.add_argument('--num_points', type=int, default=20)
    parser.add_argument('--threshold_percentile', type=float, default=50.0)

    args = parser.parse_args()

    if args.demo:
        demo_edge_of_autumn()
    else:
        print("Use --demo to run the demonstration")
        print("For real β-sweep, integrate with EEGAraBrain training")


if __name__ == "__main__":
    main()
