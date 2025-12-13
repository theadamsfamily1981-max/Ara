#!/usr/bin/env python3
"""
RNN Scaling Verification Experiment
====================================

Empirically verifies the Information-Geometric Criticality scaling laws:

    ξ(θ) ~ |E|^(-ν)      (correlation length)
    λ_max(g) ~ |E|^(-γ)  (Fisher information metric)

by tuning a recurrent neural network near its computational phase transition.

Theory Background:
    The spectral radius ρ(W_rec) of the recurrent weight matrix controls
    the dynamical regime:
    - ρ < 1: subcritical (signals decay)
    - ρ = 1: critical (edge of chaos)
    - ρ > 1: supercritical (signals explode)

    The edge function E = ρ - 1 measures distance from criticality.
    At E → 0, both correlation length and Fisher information diverge
    with universal exponents predicted by the GUTC (Grand Unified Theory
    of Criticality).

Target Exponents (2D Ising universality class):
    ν_theory = 1      (correlation length exponent)
    γ_theory = 7/4    (Fisher metric exponent for 2D critical systems)

Usage:
    from ara.cognition.rnn_scaling_experiment import run_experiment

    results = run_experiment(
        rho_range=(0.90, 1.10),
        n_steps=50,
        T_run=10000,
    )

    # Check exponents
    print(f"ν_empirical: {results['nu_empirical']:.3f}")
    print(f"γ_empirical: {results['gamma_empirical']:.3f}")

References:
    - Langton (1990) - Computation at the Edge of Chaos
    - Bertschinger & Natschläger (2004) - Real-Time Computation at Edge of Chaos
    - Amari (1998) - Natural Gradient Works Efficiently in Learning
"""

from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from collections import deque

logger = logging.getLogger("ara.cognition.rnn_scaling")


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ScalingResult:
    """Result from a single spectral radius sweep point."""
    rho: float                    # Spectral radius ρ(W_rec)
    edge_distance: float          # E = ρ - 1
    correlation_length: float     # ξ (fitted from auto-correlation)
    fisher_info: float            # λ_max(g) or EFI proxy
    lyapunov_exponent: float      # Maximum Lyapunov exponent
    trajectory_variance: float    # Variance of internal states


@dataclass
class ExperimentResults:
    """Full experiment results with fitted exponents."""
    # Individual sweep points
    sweep_results: List[ScalingResult]

    # Fitted exponents
    nu_empirical: float           # ξ ~ |E|^(-ν)
    gamma_empirical: float        # g ~ |E|^(-γ)
    nu_stderr: float = 0.0
    gamma_stderr: float = 0.0

    # Theoretical targets
    nu_theory: float = 1.0
    gamma_theory: float = 1.75    # 7/4 for 2D Ising

    # Goodness of fit
    nu_r_squared: float = 0.0
    gamma_r_squared: float = 0.0

    # Metadata
    n_neurons: int = 100
    T_run: int = 10000
    rho_range: Tuple[float, float] = (0.90, 1.10)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nu_empirical": self.nu_empirical,
            "gamma_empirical": self.gamma_empirical,
            "nu_stderr": self.nu_stderr,
            "gamma_stderr": self.gamma_stderr,
            "nu_theory": self.nu_theory,
            "gamma_theory": self.gamma_theory,
            "nu_r_squared": self.nu_r_squared,
            "gamma_r_squared": self.gamma_r_squared,
            "n_points": len(self.sweep_results),
            "n_neurons": self.n_neurons,
            "T_run": self.T_run,
            "rho_range": self.rho_range,
        }


# =============================================================================
# RNN Dynamics
# =============================================================================

def initialize_recurrent_weights(
    n_neurons: int,
    target_rho: float,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Initialize recurrent weight matrix with specified spectral radius.

    Uses orthogonal initialization scaled to achieve target ρ(W).
    """
    rng = np.random.default_rng(seed)

    # Start with random Gaussian
    W = rng.standard_normal((n_neurons, n_neurons)) / np.sqrt(n_neurons)

    # Compute current spectral radius
    eigenvalues = np.linalg.eigvals(W)
    current_rho = np.max(np.abs(eigenvalues))

    # Scale to target
    W = W * (target_rho / current_rho)

    return W


def run_rnn_trajectory(
    W_rec: np.ndarray,
    T: int,
    input_noise_std: float = 0.1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Run RNN dynamics and collect internal state trajectory.

    x_{t+1} = tanh(W_rec @ x_t + input_noise)

    Args:
        W_rec: Recurrent weight matrix (n x n)
        T: Number of timesteps
        input_noise_std: Standard deviation of input noise
        seed: Random seed for reproducibility

    Returns:
        Trajectory array of shape (T, n_neurons)
    """
    rng = np.random.default_rng(seed)
    n = W_rec.shape[0]

    # Initialize state
    x = rng.standard_normal(n) * 0.1
    trajectory = []

    # Warmup (discard transient)
    warmup = min(1000, T // 10)
    for _ in range(warmup):
        u = rng.standard_normal(n) * input_noise_std
        x = np.tanh(W_rec @ x + u)

    # Collect trajectory
    for _ in range(T):
        u = rng.standard_normal(n) * input_noise_std
        x = np.tanh(W_rec @ x + u)
        trajectory.append(x.copy())

    return np.array(trajectory)


# =============================================================================
# Metric Calculations
# =============================================================================

def calculate_autocorrelation(
    trajectory: np.ndarray,
    max_lag: int = 500,
) -> np.ndarray:
    """
    Calculate mean auto-correlation of internal states.

    C(τ) = (1/T) Σ_t <x_t, x_{t+τ}>

    Returns correlation values for lags 0 to max_lag.
    """
    T, n = trajectory.shape
    max_lag = min(max_lag, T // 2)

    correlations = []
    for tau in range(max_lag):
        if tau == 0:
            # C(0) = mean <x_t, x_t>
            c = np.mean(np.sum(trajectory ** 2, axis=1))
        else:
            # C(τ) = mean <x_t, x_{t+τ}>
            x_t = trajectory[:-tau]
            x_tau = trajectory[tau:]
            c = np.mean(np.sum(x_t * x_tau, axis=1))
        correlations.append(c)

    correlations = np.array(correlations)

    # Normalize by C(0)
    if correlations[0] > 0:
        correlations = correlations / correlations[0]

    return correlations


def fit_correlation_length(
    correlations: np.ndarray,
    min_lag: int = 5,
) -> float:
    """
    Fit correlation length ξ from exponential decay of C(τ).

    C(τ) ~ exp(-τ/ξ)
    log C(τ) ~ -τ/ξ

    Returns estimated ξ.
    """
    # Use log-linear fit on positive correlations
    lags = np.arange(len(correlations))
    valid = (correlations > 0.01) & (lags >= min_lag)

    if np.sum(valid) < 3:
        # Not enough points for fit
        return 1.0

    log_c = np.log(correlations[valid])
    tau = lags[valid]

    # Linear fit: log C = -τ/ξ + const
    # Slope = -1/ξ
    try:
        coeffs = np.polyfit(tau, log_c, 1)
        slope = coeffs[0]
        if slope >= 0:
            # Correlation not decaying (supercritical or numerical issue)
            return float(len(correlations))  # Return max
        xi = -1.0 / slope
        return max(1.0, min(float(xi), len(correlations) * 2))
    except Exception:
        return 1.0


def calculate_fisher_proxy(
    W_rec: np.ndarray,
    trajectory: np.ndarray,
    batch_size: int = 1000,
    seed: Optional[int] = None,
) -> float:
    """
    Calculate empirical Fisher information proxy.

    Uses the gradient of one-step prediction loss as a proxy:
    g ≈ E[||∂L/∂W||²]

    where L = ||x_{t+1} - tanh(W @ x_t)||²

    This measures sensitivity of the dynamics to weight perturbations.
    """
    rng = np.random.default_rng(seed)
    T, n = trajectory.shape

    # Sample random time points
    t_indices = rng.choice(T - 1, size=min(batch_size, T - 1), replace=False)

    grad_norms_sq = []
    for t in t_indices:
        x_t = trajectory[t]
        x_next = trajectory[t + 1]

        # Forward pass
        h = W_rec @ x_t
        pred = np.tanh(h)

        # Loss gradient w.r.t. prediction
        dL_dpred = 2 * (pred - x_next)  # Shape: (n,)

        # Gradient of tanh
        dtanh = 1 - pred ** 2  # Shape: (n,)

        # Chain rule: dL/dh = dL/dpred * dtanh
        dL_dh = dL_dpred * dtanh  # Shape: (n,)

        # Gradient w.r.t. W: dL/dW = dL/dh @ x_t.T
        # ||dL/dW||² = ||dL/dh||² * ||x_t||²
        grad_norm_sq = np.sum(dL_dh ** 2) * np.sum(x_t ** 2)
        grad_norms_sq.append(grad_norm_sq)

    # Empirical Fisher = mean of squared gradient norms
    return float(np.mean(grad_norms_sq))


def calculate_lyapunov_exponent(
    W_rec: np.ndarray,
    trajectory: np.ndarray,
    n_samples: int = 1000,
) -> float:
    """
    Estimate maximum Lyapunov exponent from trajectory.

    λ_max ≈ (1/T) Σ_t log||J_t|| where J_t is the Jacobian.

    For tanh RNN: J = diag(1 - x²) @ W
    """
    T = min(n_samples, len(trajectory))
    log_norms = []

    for t in range(T):
        x = trajectory[t]
        # Jacobian: J = diag(1 - tanh²(h)) @ W = diag(1 - x²) @ W
        dtanh = 1 - x ** 2
        J = dtanh[:, None] * W_rec

        # Spectral norm (largest singular value)
        try:
            s = np.linalg.svd(J, compute_uv=False)
            log_norms.append(np.log(s[0] + 1e-10))
        except Exception:
            pass

    if not log_norms:
        return 0.0

    return float(np.mean(log_norms))


# =============================================================================
# Power Law Fitting
# =============================================================================

def fit_power_law(
    E_values: np.ndarray,
    Y_values: np.ndarray,
    use_absolute: bool = True,
) -> Tuple[float, float, float]:
    """
    Fit power law Y ~ |E|^(-exponent).

    log(Y) = -exponent * log|E| + const

    Returns:
        (exponent, stderr, r_squared)
    """
    if use_absolute:
        E_abs = np.abs(E_values)
    else:
        E_abs = E_values

    # Filter valid points (E != 0, Y > 0)
    valid = (E_abs > 1e-6) & (Y_values > 1e-10)
    if np.sum(valid) < 3:
        return 0.0, 0.0, 0.0

    log_E = np.log(E_abs[valid])
    log_Y = np.log(Y_values[valid])

    # Linear regression
    n = len(log_E)
    mean_x = np.mean(log_E)
    mean_y = np.mean(log_Y)

    SS_xy = np.sum((log_E - mean_x) * (log_Y - mean_y))
    SS_xx = np.sum((log_E - mean_x) ** 2)
    SS_yy = np.sum((log_Y - mean_y) ** 2)

    if SS_xx < 1e-10:
        return 0.0, 0.0, 0.0

    slope = SS_xy / SS_xx
    intercept = mean_y - slope * mean_x

    # R-squared
    y_pred = slope * log_E + intercept
    SS_res = np.sum((log_Y - y_pred) ** 2)
    r_squared = 1 - SS_res / SS_yy if SS_yy > 0 else 0.0

    # Standard error of slope
    if n > 2:
        residual_var = SS_res / (n - 2)
        stderr = np.sqrt(residual_var / SS_xx)
    else:
        stderr = 0.0

    # Y ~ |E|^slope, so exponent = -slope
    exponent = -slope

    return float(exponent), float(stderr), float(r_squared)


# =============================================================================
# Main Experiment
# =============================================================================

def run_single_point(
    rho: float,
    n_neurons: int = 100,
    T_run: int = 10000,
    batch_size: int = 1000,
    seed: Optional[int] = None,
) -> ScalingResult:
    """
    Run experiment for a single spectral radius value.
    """
    # Initialize weights
    W_rec = initialize_recurrent_weights(n_neurons, rho, seed)

    # Generate trajectory
    trajectory = run_rnn_trajectory(W_rec, T_run, seed=seed)

    # Calculate metrics
    correlations = calculate_autocorrelation(trajectory)
    xi = fit_correlation_length(correlations)
    fisher = calculate_fisher_proxy(W_rec, trajectory, batch_size, seed)
    lyapunov = calculate_lyapunov_exponent(W_rec, trajectory)
    variance = float(np.var(trajectory))

    return ScalingResult(
        rho=rho,
        edge_distance=rho - 1.0,
        correlation_length=xi,
        fisher_info=fisher,
        lyapunov_exponent=lyapunov,
        trajectory_variance=variance,
    )


def run_experiment(
    rho_range: Tuple[float, float] = (0.90, 1.10),
    n_steps: int = 50,
    n_neurons: int = 100,
    T_run: int = 10000,
    batch_size: int = 1000,
    seed: Optional[int] = 42,
    verbose: bool = True,
) -> ExperimentResults:
    """
    Run full scaling experiment with spectral radius sweep.

    Args:
        rho_range: (min, max) spectral radius values
        n_steps: Number of sweep points
        n_neurons: RNN hidden layer size
        T_run: Trajectory length per point
        batch_size: Batch size for Fisher proxy
        seed: Random seed
        verbose: Print progress

    Returns:
        ExperimentResults with fitted exponents
    """
    if verbose:
        logger.info(f"Starting RNN scaling experiment")
        logger.info(f"  ρ ∈ [{rho_range[0]:.2f}, {rho_range[1]:.2f}], steps={n_steps}")
        logger.info(f"  n_neurons={n_neurons}, T={T_run}")

    rho_values = np.linspace(rho_range[0], rho_range[1], n_steps)
    results = []

    for i, rho in enumerate(rho_values):
        if verbose and i % 10 == 0:
            logger.info(f"  Step {i+1}/{n_steps}: ρ={rho:.4f}")

        point_seed = seed + i if seed is not None else None
        result = run_single_point(
            rho=rho,
            n_neurons=n_neurons,
            T_run=T_run,
            batch_size=batch_size,
            seed=point_seed,
        )
        results.append(result)

    # Extract arrays for fitting
    E_values = np.array([r.edge_distance for r in results])
    xi_values = np.array([r.correlation_length for r in results])
    fisher_values = np.array([r.fisher_info for r in results])

    # Fit power laws (use only points near criticality)
    near_critical = np.abs(E_values) < 0.08

    nu_emp, nu_std, nu_r2 = fit_power_law(
        E_values[near_critical],
        xi_values[near_critical],
    )

    gamma_emp, gamma_std, gamma_r2 = fit_power_law(
        E_values[near_critical],
        fisher_values[near_critical],
    )

    if verbose:
        logger.info(f"\nResults:")
        logger.info(f"  ν_empirical = {nu_emp:.3f} ± {nu_std:.3f} (R²={nu_r2:.3f})")
        logger.info(f"  γ_empirical = {gamma_emp:.3f} ± {gamma_std:.3f} (R²={gamma_r2:.3f})")
        logger.info(f"\nTheoretical targets:")
        logger.info(f"  ν_theory = 1.0 (correlation length)")
        logger.info(f"  γ_theory = 1.75 (7/4, 2D Ising FIM)")

    return ExperimentResults(
        sweep_results=results,
        nu_empirical=nu_emp,
        gamma_empirical=gamma_emp,
        nu_stderr=nu_std,
        gamma_stderr=gamma_std,
        nu_r_squared=nu_r2,
        gamma_r_squared=gamma_r2,
        n_neurons=n_neurons,
        T_run=T_run,
        rho_range=rho_range,
    )


def plot_results(results: ExperimentResults, save_path: Optional[str] = None):
    """
    Plot scaling results (requires matplotlib).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available for plotting")
        return

    E = np.array([r.edge_distance for r in results.sweep_results])
    xi = np.array([r.correlation_length for r in results.sweep_results])
    fisher = np.array([r.fisher_info for r in results.sweep_results])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Correlation length vs |E|
    ax1 = axes[0]
    valid = (np.abs(E) > 1e-4) & (xi > 0)
    ax1.loglog(np.abs(E[valid]), xi[valid], 'o', label='Data')

    # Fit line
    E_fit = np.logspace(-3, -0.5, 100)
    xi_fit = E_fit ** (-results.nu_empirical)
    ax1.loglog(E_fit, xi_fit * np.mean(xi[valid]) / np.mean(E_fit ** (-results.nu_empirical)),
               '--', label=f'ν={results.nu_empirical:.2f}')

    ax1.set_xlabel('|E| = |ρ - 1|')
    ax1.set_ylabel('ξ (correlation length)')
    ax1.set_title('Correlation Length Scaling')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Fisher info vs |E|
    ax2 = axes[1]
    valid = (np.abs(E) > 1e-4) & (fisher > 0)
    ax2.loglog(np.abs(E[valid]), fisher[valid], 'o', label='Data')

    fisher_fit = E_fit ** (-results.gamma_empirical)
    ax2.loglog(E_fit, fisher_fit * np.mean(fisher[valid]) / np.mean(E_fit ** (-results.gamma_empirical)),
               '--', label=f'γ={results.gamma_empirical:.2f}')

    ax2.set_xlabel('|E| = |ρ - 1|')
    ax2.set_ylabel('λ_max(g) (Fisher proxy)')
    ax2.set_title('Fisher Information Scaling')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Raw metrics vs ρ
    ax3 = axes[2]
    ax3.plot([r.rho for r in results.sweep_results],
             [r.correlation_length for r in results.sweep_results],
             'b-o', label='ξ', markersize=3)
    ax3.axvline(x=1.0, color='r', linestyle='--', label='ρ=1 (critical)')
    ax3.set_xlabel('ρ (spectral radius)')
    ax3.set_ylabel('ξ (correlation length)')
    ax3.set_title('Correlation Length vs ρ')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info(f"Plot saved to {save_path}")
    else:
        plt.show()


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demo the RNN scaling experiment."""
    print("=" * 70)
    print("RNN Scaling Verification Experiment")
    print("Verifying Information-Geometric Criticality Scaling Laws")
    print("=" * 70)

    print("\nTheory:")
    print("  ξ(θ) ~ |E|^(-ν)     with ν_theory = 1.0")
    print("  g(θ) ~ |E|^(-γ)     with γ_theory = 7/4 = 1.75")
    print()

    # Run a quick experiment
    results = run_experiment(
        rho_range=(0.92, 1.08),
        n_steps=30,
        n_neurons=50,
        T_run=5000,
        verbose=True,
    )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nCorrelation Length Exponent:")
    print(f"  ν_empirical = {results.nu_empirical:.3f} ± {results.nu_stderr:.3f}")
    print(f"  ν_theory    = {results.nu_theory:.3f}")
    print(f"  R² = {results.nu_r_squared:.3f}")

    print(f"\nFisher Information Exponent:")
    print(f"  γ_empirical = {results.gamma_empirical:.3f} ± {results.gamma_stderr:.3f}")
    print(f"  γ_theory    = {results.gamma_theory:.3f} (7/4 for 2D Ising)")
    print(f"  R² = {results.gamma_r_squared:.3f}")

    # Evaluate agreement
    nu_agree = abs(results.nu_empirical - results.nu_theory) < 0.3
    gamma_agree = abs(results.gamma_empirical - results.gamma_theory) < 0.5

    print(f"\nVerification:")
    print(f"  ν matches theory: {'✓' if nu_agree else '✗'}")
    print(f"  γ matches theory: {'✓' if gamma_agree else '✗'}")

    if nu_agree and gamma_agree:
        print("\n🎉 SCALING LAWS VERIFIED!")
        print("The RNN exhibits universal criticality exponents.")
    else:
        print("\n⚠ Scaling may require larger network or longer runs.")

    print("=" * 70)

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = demo()
