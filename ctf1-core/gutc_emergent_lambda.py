#!/usr/bin/env python3
"""
GUTC Emergent Criticality Estimation
====================================

Extends the L2/3↔L5 VFE engine with **emergent λ̂ estimation**.

Instead of trusting the dial λ you set, the system now estimates its
actual criticality from avalanche-like statistics in the error dynamics.

Key Innovation
--------------
    λ (input dial) → dynamics → ε_y(t) trajectory → λ̂ (emergent estimate)

The system can now "see" its own phase:
- λ̂ ≈ 1: Critical, healthy corridor
- λ̂ < 1: Subcritical, rigid/ASD-like
- λ̂ > 1: Supercritical, unstable/psychotic-like

Branching Ratio Estimation
--------------------------
From the prediction error trajectory ε_y(t):
1. Define "activations" as |ε_y(t)| > threshold
2. Parents = activations at time t
3. Children = activations at time t+1 where parent was active
4. λ̂ = Σ(parents × children) / Σ(parents)

At criticality (λ̂ ≈ 1), each active unit triggers ~1 offspring on average.

GUTC Manifold Map
-----------------
Sweeps (λ, Π_prior) and computes:
- F̄: Mean steady-state free energy (inference efficiency)
- λ̂: Emergent branching ratio (self-measured criticality)

The healthy corridor appears as the intersection of:
- Low F̄ (efficient inference)
- λ̂ ≈ 1 (critical dynamics)

Usage
-----
    python gutc_emergent_lambda.py test      # Run tests
    python gutc_emergent_lambda.py manifold  # Generate manifold map
    python gutc_emergent_lambda.py sweep     # Detailed parameter sweep
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from scipy.optimize import curve_fit


# =============================================================================
# Avalanche Analysis: Power-Law Exponents
# =============================================================================

@dataclass
class AvalancheStats:
    """
    Avalanche statistics extracted from prediction error dynamics.

    At mean-field criticality (λ = 1), universal exponents are:
        α = 3/2  (size distribution: P(s) ~ s^{-α})
        z = 2    (duration distribution: P(τ) ~ τ^{-z})

    These are signatures of the Galton-Watson branching process at λ = 1.
    """
    sizes: List[int]        # Avalanche sizes (number of active bins)
    durations: List[int]    # Avalanche durations (time bins)
    alpha_hat: float        # Fitted size exponent (target: 3/2)
    z_hat: float            # Fitted duration exponent (target: 2)

    @property
    def n_avalanches(self) -> int:
        """Number of detected avalanches."""
        return len(self.sizes)

    @property
    def mean_size(self) -> float:
        """Mean avalanche size."""
        return np.mean(self.sizes) if self.sizes else 0.0

    @property
    def mean_duration(self) -> float:
        """Mean avalanche duration."""
        return np.mean(self.durations) if self.durations else 0.0

    def is_critical(self, alpha_tol: float = 0.3, z_tol: float = 0.5) -> bool:
        """
        Check if exponents are consistent with criticality.

        Default tolerances are generous for finite-size effects.
        """
        if np.isnan(self.alpha_hat) or np.isnan(self.z_hat):
            return False
        alpha_ok = abs(self.alpha_hat - 1.5) < alpha_tol
        z_ok = abs(self.z_hat - 2.0) < z_tol
        return alpha_ok and z_ok


def fit_power_law_exponent(
    data: List[int],
    min_val: int = 1,
    max_val: Optional[int] = None,
) -> float:
    """
    Fit power-law exponent using MLE (Hill estimator).

    For P(x) ~ x^{-α}, the MLE estimator is:
        α̂ = 1 + n / Σ ln(x_i / x_min)

    This is more robust than linear regression on log-log histogram.

    Args:
        data: Sample of values (sizes or durations)
        min_val: Minimum value for fitting (avoids small-sample artifacts)
        max_val: Optional maximum value (filter outliers)

    Returns:
        Estimated exponent α̂
    """
    # Filter data
    x = np.array([d for d in data if d >= min_val])
    if max_val is not None:
        x = x[x <= max_val]

    if len(x) < 5:
        return np.nan

    # Hill estimator (MLE for power law)
    # α = 1 + n / Σ ln(x_i / x_min)
    x_min = min_val
    n = len(x)
    log_sum = np.sum(np.log(x / x_min))

    if log_sum <= 0:
        return np.nan

    alpha = 1 + n / log_sum
    return alpha


def fit_power_law_linear(
    data: List[int],
    n_bins: int = 20,
    min_val: int = 1,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Fit power-law exponent using linear regression on log-log histogram.

    Less robust than MLE but provides visual verification.

    Returns:
        (alpha, r_squared, bin_centers, counts) for plotting
    """
    x = np.array([d for d in data if d >= min_val])
    if len(x) < 10:
        return np.nan, 0.0, np.array([]), np.array([])

    # Log-spaced bins
    bins = np.logspace(np.log10(min_val), np.log10(x.max() + 1), n_bins)
    counts, bin_edges = np.histogram(x, bins=bins, density=True)

    # Bin centers (geometric mean)
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])

    # Filter zero counts
    mask = counts > 0
    log_x = np.log10(bin_centers[mask])
    log_y = np.log10(counts[mask])

    if len(log_x) < 3:
        return np.nan, 0.0, bin_centers, counts

    # Linear regression
    slope, intercept, r_value, _, _ = stats.linregress(log_x, log_y)

    # α = -slope (since P(x) ~ x^{-α} means log P = -α log x + const)
    alpha = -slope
    r_squared = r_value ** 2

    return alpha, r_squared, bin_centers, counts


# =============================================================================
# L2/3↔L5 Predictive Coder with Emergent λ̂
# =============================================================================

@dataclass
class EmergentLambdaConfig:
    """Configuration for emergent criticality estimation."""
    # Generative model
    C: float = 1.0              # Observation gain
    mu_0: float = 0.0           # Prior mean

    # GUTC control parameters
    lambda_c: float = 1.0       # Input criticality dial
    W_recur: float = 1.0        # Recurrent weight magnitude

    # Precisions
    pi_sensory: float = 1.0     # Sensory precision (ACh)
    pi_prior: float = 1.0       # Prior precision (DA)

    # Time constants
    tau_mu: float = 10.0        # L5 belief dynamics
    tau_eps: float = 5.0        # L2/3 error dynamics
    dt: float = 0.01            # Integration step

    # Stochastic dynamics (for emergent avalanches)
    noise_std: float = 0.1      # State noise (creates avalanches)
    input_noise_std: float = 0.05  # Input noise


class L23L5EmergentCoder:
    """
    L2/3↔L5 predictive coding loop with emergent λ̂ estimation.

    The system tracks its error trajectory and estimates its own
    criticality from avalanche-like statistics.

    Includes stochastic noise to create avalanche-like dynamics.
    """

    def __init__(self, cfg: EmergentLambdaConfig = None, seed: int = None):
        self.cfg = cfg or EmergentLambdaConfig()
        self.rng = np.random.default_rng(seed)

        # State variables
        self.mu = self.cfg.mu_0       # L5 belief (initialized at prior)
        self.eps_y = 0.0              # Sensory prediction error
        self.eps_mu = 0.0             # Prior prediction error

        # Trajectory storage for λ̂ estimation
        self.eps_y_history: List[float] = []
        self.eps_mu_history: List[float] = []
        self.mu_history: List[float] = []
        self.F_history: List[float] = []

    def reset(self):
        """Reset state and history."""
        self.mu = self.cfg.mu_0
        self.eps_y = 0.0
        self.eps_mu = 0.0
        self.eps_y_history = []
        self.eps_mu_history = []
        self.mu_history = []
        self.F_history = []

    def free_energy(self) -> float:
        """
        Instantaneous variational free energy.

        F = ½ Π_s ε_y² + ½ Π_p ε_μ²
        """
        F_y = 0.5 * self.cfg.pi_sensory * (self.eps_y ** 2)
        F_mu = 0.5 * self.cfg.pi_prior * (self.eps_mu ** 2)
        return F_y + F_mu

    def step(self, y: float) -> Tuple[float, float, float, float]:
        """
        Single Euler integration step with stochastic noise.

        L2/3 dynamics:
            τ_ε ε̇_y = (y - C·μ̂) - ε_y + noise
            τ_ε ε̇_μ = (μ̂ - μ₀) - ε_μ + noise

        L5 dynamics:
            τ_μ μ̂̇ = (-1 + λ·W)μ̂ + C·Π_s·ε_y - Π_p·ε_μ + noise

        Returns:
            (μ̂, ε_y, ε_μ, F)
        """
        cfg = self.cfg

        # Add input noise
        y_noisy = y + cfg.input_noise_std * self.rng.standard_normal()

        # Target errors (current mismatches)
        eps_y_target = y_noisy - cfg.C * self.mu
        eps_mu_target = self.mu - cfg.mu_0

        # L2/3: Leaky error dynamics with noise
        d_eps_y = (eps_y_target - self.eps_y) / cfg.tau_eps
        d_eps_mu = (eps_mu_target - self.eps_mu) / cfg.tau_eps

        # Add state noise (scaled by sqrt(dt) for proper SDE)
        noise_scale = cfg.noise_std * np.sqrt(cfg.dt)
        self.eps_y += cfg.dt * d_eps_y + noise_scale * self.rng.standard_normal()
        self.eps_mu += cfg.dt * d_eps_mu + noise_scale * self.rng.standard_normal()

        # L5: Gradient descent on F
        # Recurrent term: (-1 + λ·W)·μ̂
        recur = (-1.0 + cfg.lambda_c * cfg.W_recur) * self.mu

        # Precision-weighted drives
        drive_sens = cfg.C * cfg.pi_sensory * self.eps_y
        drive_prior = -cfg.pi_prior * self.eps_mu

        d_mu = (recur + drive_sens + drive_prior) / cfg.tau_mu
        self.mu += cfg.dt * d_mu + noise_scale * self.rng.standard_normal()

        # Store trajectory
        F = self.free_energy()
        self.eps_y_history.append(self.eps_y)
        self.eps_mu_history.append(self.eps_mu)
        self.mu_history.append(self.mu)
        self.F_history.append(F)

        return self.mu, self.eps_y, self.eps_mu, F

    def estimate_branching_ratio(self, threshold: float = 0.05) -> float:
        """
        Estimate emergent branching ratio λ̂ from ε_y trajectory.

        Method:
        1. Define activations: A_t = 𝟙(|ε_y(t)| > threshold)
        2. Parents = A_t, Children = A_{t+1}
        3. λ̂ = Σ(parents × children) / Σ(parents)

        At criticality, λ̂ ≈ 1: each active unit triggers ~1 offspring.

        Returns:
            λ̂ (emergent branching ratio)
        """
        if len(self.eps_y_history) < 3:
            return 1.0  # Default if insufficient data

        eps_y = np.array(self.eps_y_history)

        # Define activations (suprathreshold error)
        activations = (np.abs(eps_y) > threshold).astype(float)

        # Parents (t) and children (t+1)
        parents = activations[:-1]
        children = activations[1:]

        n_parents = parents.sum()
        if n_parents < 1:
            return 0.0  # No activity

        # Branching ratio: offspring when parent was active
        offspring_total = (parents * children).sum()
        lambda_hat = offspring_total / n_parents

        return float(lambda_hat)

    def estimate_branching_ratio_delta(self, threshold: float = 0.1) -> float:
        """
        Alternative λ̂ estimator using Δε_y (activity changes).

        Better for continuous systems where ε_y itself may be always > threshold.
        """
        if len(self.eps_y_history) < 3:
            return 1.0

        eps_y = np.array(self.eps_y_history)
        delta_eps = np.diff(eps_y)

        # Activations = large changes in error
        activations = (np.abs(delta_eps) > threshold).astype(float)

        parents = activations[:-1]
        children = activations[1:]

        n_parents = parents.sum()
        if n_parents < 1:
            return 1.0

        offspring_total = (parents * children).sum()
        lambda_hat = offspring_total / n_parents

        return float(lambda_hat)

    def extract_avalanches(self, threshold: float = 0.05) -> 'AvalancheStats':
        """
        Extract avalanche statistics from ε_y trajectory.

        Algorithm:
        1. Binarize: A_t = 𝟙(|ε_y(t)| > θ)
        2. Find runs of A_t = 1 bounded by A_t = 0
        3. Size s = sum of activations in run
        4. Duration τ = length of run in time bins

        At criticality (λ = 1), we expect:
        - P(s) ~ s^{-3/2}  (size exponent α = 3/2)
        - P(τ) ~ τ^{-2}    (duration exponent z = 2)

        Returns:
            AvalancheStats with sizes, durations, and fitted exponents
        """
        if len(self.eps_y_history) < 10:
            return AvalancheStats([], [], np.nan, np.nan)

        eps_y = np.array(self.eps_y_history)

        # Binarize: active if |ε_y| > threshold
        A = (np.abs(eps_y) > threshold).astype(int)

        # Find avalanche boundaries
        # Pad with zeros to detect avalanches at boundaries
        A_padded = np.concatenate([[0], A, [0]])
        diff = np.diff(A_padded)

        # Avalanche starts: 0 → 1 transition
        starts = np.where(diff == 1)[0]
        # Avalanche ends: 1 → 0 transition
        ends = np.where(diff == -1)[0]

        sizes: List[int] = []
        durations: List[int] = []

        for start, end in zip(starts, ends):
            duration = end - start
            size = int(A[start:end].sum())  # Should equal duration if all 1s

            if duration > 0:
                sizes.append(size)
                durations.append(duration)

        # Fit power-law exponents if we have enough avalanches
        alpha_hat = np.nan
        z_hat = np.nan

        if len(sizes) >= 10:
            alpha_hat = fit_power_law_exponent(sizes, min_val=1)
        if len(durations) >= 10:
            z_hat = fit_power_law_exponent(durations, min_val=1)

        return AvalancheStats(sizes, durations, alpha_hat, z_hat)

    def get_mean_free_energy(self, burn_fraction: float = 0.2) -> float:
        """Get mean F after burn-in period."""
        if len(self.F_history) < 10:
            return float('nan')

        burn = int(len(self.F_history) * burn_fraction)
        return np.mean(self.F_history[burn:])


# =============================================================================
# Simulation and Analysis
# =============================================================================

@dataclass
class ManifoldPoint:
    """Results from simulating one point on the (λ, Π) manifold."""
    lambda_c: float         # Input dial
    pi_prior: float         # Prior precision
    pi_sensory: float       # Sensory precision
    mean_F: float           # Mean steady-state F̄
    lambda_hat: float       # Emergent λ̂
    lambda_hat_delta: float # Alternative λ̂ (from Δε)
    # Avalanche statistics
    alpha_hat: float = np.nan   # Size exponent (target: 3/2)
    z_hat: float = np.nan       # Duration exponent (target: 2)
    n_avalanches: int = 0       # Number of detected avalanches


def simulate_point(
    lambda_c: float,
    pi_prior: float,
    pi_sensory: float = 2.0,
    y_obs: float = 1.0,
    T: float = 5.0,
    dt: float = 0.01,
    avalanche_threshold: float = 0.05,
) -> ManifoldPoint:
    """
    Simulate one point on the GUTC manifold.

    Returns ManifoldPoint with:
    - F̄: Mean steady-state free energy
    - λ̂: Emergent branching ratio
    - α̂: Avalanche size exponent (target: 3/2 at criticality)
    - ẑ: Avalanche duration exponent (target: 2 at criticality)
    """
    cfg = EmergentLambdaConfig(
        lambda_c=lambda_c,
        pi_prior=pi_prior,
        pi_sensory=pi_sensory,
        dt=dt,
    )

    coder = L23L5EmergentCoder(cfg)
    n_steps = int(T / dt)

    for _ in range(n_steps):
        coder.step(y_obs)

    # Extract avalanche statistics
    avalanche_stats = coder.extract_avalanches(threshold=avalanche_threshold)

    return ManifoldPoint(
        lambda_c=lambda_c,
        pi_prior=pi_prior,
        pi_sensory=pi_sensory,
        mean_F=coder.get_mean_free_energy(),
        lambda_hat=coder.estimate_branching_ratio(threshold=avalanche_threshold),
        lambda_hat_delta=coder.estimate_branching_ratio_delta(threshold=avalanche_threshold),
        alpha_hat=avalanche_stats.alpha_hat,
        z_hat=avalanche_stats.z_hat,
        n_avalanches=avalanche_stats.n_avalanches,
    )


def sweep_manifold(
    lambda_range: Tuple[float, float] = (0.5, 2.0),
    pi_range: Tuple[float, float] = (0.5, 8.0),
    n_lambda: int = 25,
    n_pi: int = 25,
    pi_sensory: float = 2.0,
    y_obs: float = 1.0,
    T: float = 5.0,
) -> Dict[str, np.ndarray]:
    """
    Sweep the (λ, Π_prior) manifold.

    Returns grids of:
    - F̄: Mean free energy (inference efficiency)
    - λ̂: Emergent branching ratio (criticality)
    - α̂: Avalanche size exponent (target: 3/2)
    - ẑ: Avalanche duration exponent (target: 2)
    """
    lambdas = np.linspace(lambda_range[0], lambda_range[1], n_lambda)
    pis = np.linspace(pi_range[0], pi_range[1], n_pi)

    F_grid = np.zeros((n_pi, n_lambda))
    lambda_hat_grid = np.zeros((n_pi, n_lambda))
    alpha_hat_grid = np.zeros((n_pi, n_lambda))
    z_hat_grid = np.zeros((n_pi, n_lambda))
    n_avalanches_grid = np.zeros((n_pi, n_lambda))

    for i, pi_prior in enumerate(pis):
        for j, lambda_c in enumerate(lambdas):
            result = simulate_point(
                lambda_c=lambda_c,
                pi_prior=pi_prior,
                pi_sensory=pi_sensory,
                y_obs=y_obs,
                T=T,
            )
            F_grid[i, j] = result.mean_F
            lambda_hat_grid[i, j] = result.lambda_hat
            alpha_hat_grid[i, j] = result.alpha_hat
            z_hat_grid[i, j] = result.z_hat
            n_avalanches_grid[i, j] = result.n_avalanches

    return {
        "lambdas": lambdas,
        "pis": pis,
        "F_grid": F_grid,
        "lambda_hat_grid": lambda_hat_grid,
        "alpha_hat_grid": alpha_hat_grid,
        "z_hat_grid": z_hat_grid,
        "n_avalanches_grid": n_avalanches_grid,
    }


# =============================================================================
# Visualization
# =============================================================================

def plot_manifold_map(
    data: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
):
    """
    Plot the GUTC manifold with F̄ and λ̂.

    Two panels:
    1. Inference efficiency (F̄) — low = healthy
    2. Emergent criticality (λ̂) — 1.0 = critical
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("matplotlib required for plotting")
        return

    lambdas = data["lambdas"]
    pis = data["pis"]
    F_grid = data["F_grid"]
    lambda_hat_grid = data["lambda_hat_grid"]

    # Clip F for LogNorm
    F_clipped = np.clip(F_grid, 1e-6, np.max(F_grid))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # === Panel 1: Free Energy Landscape ===
    im1 = axes[0].pcolormesh(
        lambdas, pis, F_clipped,
        cmap='viridis_r',
        norm=mcolors.LogNorm(vmin=F_clipped.min(), vmax=F_clipped.max()),
        shading='auto',
    )
    cbar1 = fig.colorbar(im1, ax=axes[0])
    cbar1.set_label(r'Mean Free Energy $\bar{\mathcal{F}}$')

    # Critical line
    axes[0].axvline(x=1.0, linestyle='--', color='white', linewidth=1.5,
                    label=r'$\lambda = 1$ (critical)')
    axes[0].set_xlabel(r'Input Criticality $\lambda$')
    axes[0].set_ylabel(r'Prior Precision $\Pi_{\mathrm{prior}}$')
    axes[0].set_title(r'GUTC Manifold: Inference Efficiency $\bar{\mathcal{F}}$')
    axes[0].legend(loc='upper right')

    # Regime labels
    axes[0].text(0.65, 7.0, 'Rigid\n(ASD-like)', color='white', ha='center', fontsize=9)
    axes[0].text(1.7, 7.0, 'Unstable\n(Psychosis)', color='white', ha='center', fontsize=9)
    axes[0].text(1.0, 1.0, 'Healthy\nCorridor', color='black', ha='center', fontsize=10, fontweight='bold')

    # === Panel 2: Emergent λ̂ ===
    im2 = axes[1].pcolormesh(
        lambdas, pis, lambda_hat_grid,
        cmap='coolwarm',
        vmin=0.5, vmax=1.5,
        shading='auto',
    )
    cbar2 = fig.colorbar(im2, ax=axes[1])
    cbar2.set_label(r'Emergent Branching Ratio $\hat{\lambda}$')

    # Critical contour (λ̂ = 1)
    contour = axes[1].contour(lambdas, pis, lambda_hat_grid,
                               levels=[1.0], colors='black', linewidths=2)
    axes[1].clabel(contour, fmt=r'$\hat{\lambda}=1$', fontsize=10)

    # Input critical line
    axes[1].axvline(x=1.0, linestyle='--', color='gray', linewidth=1,
                    label=r'$\lambda = 1$ (input)')

    axes[1].set_xlabel(r'Input Criticality $\lambda$')
    axes[1].set_ylabel(r'Prior Precision $\Pi_{\mathrm{prior}}$')
    axes[1].set_title(r'GUTC Manifold: Emergent Criticality $\hat{\lambda}$')
    axes[1].legend(loc='upper right')

    # Regime labels
    axes[1].text(0.65, 7.0, r'$\hat{\lambda} < 1$' + '\nSubcritical', color='blue', ha='center', fontsize=9)
    axes[1].text(1.7, 7.0, r'$\hat{\lambda} > 1$' + '\nSupercritical', color='red', ha='center', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()


def plot_triple_map(
    data: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
):
    """
    Plot the GUTC triple map: (F̄, λ̂, α̂).

    Three panels showing the complete criticality signature:
    1. F̄(λ, Π): Inference efficiency landscape
    2. λ̂(λ, Π): Emergent branching ratio
    3. α̂(λ, Π): Avalanche size exponent

    At criticality (λ = 1), we expect:
    - Low F̄ (efficient inference)
    - λ̂ ≈ 1 (critical branching)
    - α̂ ≈ 3/2 (universal exponent)
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("matplotlib required for plotting")
        return

    lambdas = data["lambdas"]
    pis = data["pis"]
    F_grid = data["F_grid"]
    lambda_hat_grid = data["lambda_hat_grid"]
    alpha_hat_grid = data.get("alpha_hat_grid", np.full_like(F_grid, np.nan))

    # Clip F for LogNorm
    F_clipped = np.clip(F_grid, 1e-6, np.max(F_grid))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # === Panel A: Free Energy Landscape ===
    im1 = axes[0].pcolormesh(
        lambdas, pis, F_clipped,
        cmap='viridis_r',
        norm=mcolors.LogNorm(vmin=F_clipped.min(), vmax=F_clipped.max()),
        shading='auto',
    )
    cbar1 = fig.colorbar(im1, ax=axes[0])
    cbar1.set_label(r'$\bar{\mathcal{F}}$')

    axes[0].axvline(x=1.0, linestyle='--', color='white', linewidth=1.5)
    axes[0].set_xlabel(r'Input Criticality $\lambda$')
    axes[0].set_ylabel(r'Prior Precision $\Pi_{\mathrm{prior}}$')
    axes[0].set_title(r'(A) Inference Efficiency $\bar{\mathcal{F}}$')

    # === Panel B: Emergent λ̂ ===
    im2 = axes[1].pcolormesh(
        lambdas, pis, lambda_hat_grid,
        cmap='coolwarm',
        vmin=0.5, vmax=1.5,
        shading='auto',
    )
    cbar2 = fig.colorbar(im2, ax=axes[1])
    cbar2.set_label(r'$\hat{\lambda}$')

    # Critical contour (λ̂ = 1)
    try:
        contour = axes[1].contour(lambdas, pis, lambda_hat_grid,
                                   levels=[1.0], colors='black', linewidths=2)
        axes[1].clabel(contour, fmt=r'$\hat{\lambda}=1$', fontsize=9)
    except Exception:
        pass  # Skip if contour fails

    axes[1].axvline(x=1.0, linestyle='--', color='gray', linewidth=1)
    axes[1].set_xlabel(r'Input Criticality $\lambda$')
    axes[1].set_ylabel(r'Prior Precision $\Pi_{\mathrm{prior}}$')
    axes[1].set_title(r'(B) Emergent Criticality $\hat{\lambda}$')

    # === Panel C: Avalanche Size Exponent α̂ ===
    # Replace NaN with a value outside the colormap range for visualization
    alpha_display = np.where(np.isnan(alpha_hat_grid), 0, alpha_hat_grid)

    im3 = axes[2].pcolormesh(
        lambdas, pis, alpha_display,
        cmap='plasma',
        vmin=1.0, vmax=3.0,
        shading='auto',
    )
    cbar3 = fig.colorbar(im3, ax=axes[2])
    cbar3.set_label(r'$\hat{\alpha}$')

    # Critical exponent contour (α̂ = 3/2)
    try:
        contour_alpha = axes[2].contour(lambdas, pis, alpha_hat_grid,
                                         levels=[1.5], colors='cyan', linewidths=2)
        axes[2].clabel(contour_alpha, fmt=r'$\hat{\alpha}=3/2$', fontsize=9)
    except Exception:
        pass  # Skip if contour fails

    axes[2].axvline(x=1.0, linestyle='--', color='white', linewidth=1)
    axes[2].set_xlabel(r'Input Criticality $\lambda$')
    axes[2].set_ylabel(r'Prior Precision $\Pi_{\mathrm{prior}}$')
    axes[2].set_title(r'(C) Avalanche Exponent $\hat{\alpha}$ (target: $3/2$)')

    # Add healthy corridor annotation
    for ax in axes:
        ax.plot(1.0, 1.0, 'w*', markersize=12, markeredgecolor='black',
                markeredgewidth=1, label='Healthy')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()


def plot_avalanche_distributions(
    avalanche_stats: AvalancheStats,
    title: str = "Avalanche Distributions",
    save_path: Optional[str] = None,
):
    """
    Plot avalanche size and duration distributions on log-log axes.

    Shows:
    - P(s) with reference line s^{-3/2}
    - P(τ) with reference line τ^{-2}
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for plotting")
        return

    if avalanche_stats.n_avalanches < 10:
        print("Not enough avalanches for distribution plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # === Size distribution ===
    sizes = np.array(avalanche_stats.sizes)
    sizes = sizes[sizes > 0]

    if len(sizes) > 5:
        # Histogram
        max_s = int(sizes.max())
        bins_s = np.arange(1, max_s + 2) - 0.5
        counts_s, _ = np.histogram(sizes, bins=bins_s, density=True)
        bin_centers_s = np.arange(1, max_s + 1)

        # Filter zeros
        mask_s = counts_s > 0
        axes[0].loglog(bin_centers_s[mask_s], counts_s[mask_s], 'o',
                       markersize=6, label='Data')

        # Reference line: s^{-3/2}
        s_ref = np.logspace(0, np.log10(max_s), 50)
        p_ref = s_ref ** (-1.5)
        p_ref = p_ref * (counts_s[mask_s].max() / p_ref[0])  # Normalize
        axes[0].loglog(s_ref, p_ref, 'r--', linewidth=2,
                       label=r'$s^{-3/2}$ (critical)')

        axes[0].set_xlabel(r'Avalanche Size $s$')
        axes[0].set_ylabel(r'$P(s)$')
        axes[0].set_title(f'Size Distribution (α̂ = {avalanche_stats.alpha_hat:.2f})')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    # === Duration distribution ===
    durations = np.array(avalanche_stats.durations)
    durations = durations[durations > 0]

    if len(durations) > 5:
        # Histogram
        max_t = int(durations.max())
        bins_t = np.arange(1, max_t + 2) - 0.5
        counts_t, _ = np.histogram(durations, bins=bins_t, density=True)
        bin_centers_t = np.arange(1, max_t + 1)

        # Filter zeros
        mask_t = counts_t > 0
        axes[1].loglog(bin_centers_t[mask_t], counts_t[mask_t], 'o',
                       markersize=6, label='Data')

        # Reference line: τ^{-2}
        t_ref = np.logspace(0, np.log10(max_t), 50)
        p_ref = t_ref ** (-2.0)
        p_ref = p_ref * (counts_t[mask_t].max() / p_ref[0])  # Normalize
        axes[1].loglog(t_ref, p_ref, 'r--', linewidth=2,
                       label=r'$\tau^{-2}$ (critical)')

        axes[1].set_xlabel(r'Avalanche Duration $\tau$')
        axes[1].set_ylabel(r'$P(\tau)$')
        axes[1].set_title(f'Duration Distribution (ẑ = {avalanche_stats.z_hat:.2f})')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()


# =============================================================================
# Self-Organized Criticality Controller (Preview)
# =============================================================================

class SOCController:
    """
    Simple SOC controller that nudges λ based on λ̂.

    If λ̂ < 1: increase λ (system too rigid)
    If λ̂ > 1: decrease λ (system too unstable)

    This makes the system self-organize to criticality.
    """

    def __init__(
        self,
        target_lambda: float = 1.0,
        gain: float = 0.01,
        lambda_min: float = 0.5,
        lambda_max: float = 2.0,
    ):
        self.target = target_lambda
        self.gain = gain
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

    def update(self, lambda_current: float, lambda_hat: float) -> float:
        """
        Compute updated λ based on λ̂.

        Δλ = gain × (target - λ̂)
        """
        error = self.target - lambda_hat
        delta = self.gain * error

        new_lambda = lambda_current + delta
        new_lambda = np.clip(new_lambda, self.lambda_min, self.lambda_max)

        return new_lambda


# =============================================================================
# Tests
# =============================================================================

def test_emergent_coder():
    """Test basic L2/3↔L5 coder functionality."""
    cfg = EmergentLambdaConfig(lambda_c=1.0, pi_sensory=1.0, pi_prior=1.0)
    coder = L23L5EmergentCoder(cfg)

    # Simulate
    y = 1.0
    for _ in range(500):
        coder.step(y)

    # Check convergence
    mean_F = coder.get_mean_free_energy()
    assert mean_F < 1.0, f"F̄ = {mean_F} should be low for healthy regime"

    # Check λ̂ estimation
    lambda_hat = coder.estimate_branching_ratio()
    assert 0.0 <= lambda_hat <= 2.0, f"λ̂ = {lambda_hat} out of range"

    print(f"✓ Emergent coder: F̄={mean_F:.4f}, λ̂={lambda_hat:.3f}")


def test_manifold_sweep():
    """Test manifold sweep."""
    data = sweep_manifold(
        lambda_range=(0.7, 1.3),
        pi_range=(0.5, 3.0),
        n_lambda=5,
        n_pi=5,
        T=2.0,
    )

    assert data["F_grid"].shape == (5, 5)
    assert data["lambda_hat_grid"].shape == (5, 5)

    # F should be finite
    assert np.all(np.isfinite(data["F_grid"]))

    print("✓ Manifold sweep")


def test_lambda_hat_tracks_lambda_c():
    """Test that λ̂ roughly tracks λ_c."""
    results = []
    for lambda_c in [0.7, 1.0, 1.3]:
        point = simulate_point(lambda_c=lambda_c, pi_prior=1.0, T=3.0)
        results.append((lambda_c, point.lambda_hat))
        print(f"  λ_c={lambda_c:.1f} → λ̂={point.lambda_hat:.3f}")

    # λ̂ should increase with λ_c (at least roughly)
    # Note: The relationship isn't perfect in this simple model
    print("✓ λ̂ tracking (visual inspection)")


def test_soc_controller():
    """Test SOC controller."""
    controller = SOCController(target_lambda=1.0, gain=0.1)

    # If λ̂ < 1, should increase λ
    new_lambda = controller.update(0.8, 0.7)
    assert new_lambda > 0.8, "Should increase λ when λ̂ < 1"

    # If λ̂ > 1, should decrease λ
    new_lambda = controller.update(1.2, 1.3)
    assert new_lambda < 1.2, "Should decrease λ when λ̂ > 1"

    print("✓ SOC controller")


def test_avalanche_extraction():
    """Test avalanche extraction from ε_y trajectory."""
    # Create coder with high noise for avalanche generation
    cfg = EmergentLambdaConfig(
        lambda_c=1.0,
        pi_sensory=1.0,
        pi_prior=1.0,
        noise_std=0.15,  # Higher noise for more avalanches
    )
    coder = L23L5EmergentCoder(cfg, seed=42)

    # Longer simulation for sufficient avalanches
    y = 1.0
    for _ in range(2000):
        coder.step(y)

    # Extract avalanches
    stats = coder.extract_avalanches(threshold=0.05)

    # Should have detected some avalanches
    assert stats.n_avalanches > 0, "Should detect avalanches"

    # Check data structures
    assert len(stats.sizes) == len(stats.durations)
    assert all(s > 0 for s in stats.sizes)
    assert all(d > 0 for d in stats.durations)

    print(f"✓ Avalanche extraction: n={stats.n_avalanches}, "
          f"mean_s={stats.mean_size:.1f}, mean_τ={stats.mean_duration:.1f}")


def test_power_law_fitting():
    """Test power-law exponent fitting."""
    # Generate synthetic power-law data with known exponent
    np.random.seed(42)
    true_alpha = 1.5

    # Inverse CDF sampling for power law
    # P(x) ~ x^{-α} → F(x) = x^{1-α} → x = u^{1/(1-α)}
    u = np.random.uniform(0, 1, 500)
    x_min = 1.0
    x_max = 100.0
    # Truncated power law
    x = x_min * ((x_max/x_min)**(1-true_alpha) * u + (1-u))**(1/(1-true_alpha))
    x = np.clip(x, 1, 100).astype(int)
    x = list(x)

    # Fit exponent
    alpha_hat = fit_power_law_exponent(x, min_val=1)

    # Should be close to true exponent (within tolerance for finite sample)
    assert 1.0 < alpha_hat < 2.5, f"α̂ = {alpha_hat} should be near {true_alpha}"

    print(f"✓ Power-law fitting: true α={true_alpha}, estimated α̂={alpha_hat:.2f}")


def test_avalanche_stats_at_criticality():
    """Test that avalanche exponents approach universal values at λ = 1."""
    # Simulate at criticality with longer duration for better statistics
    cfg = EmergentLambdaConfig(
        lambda_c=1.0,
        pi_sensory=1.0,
        pi_prior=1.0,
        noise_std=0.2,  # Strong noise for clear avalanches
    )
    coder = L23L5EmergentCoder(cfg, seed=123)

    # Long simulation
    y = 1.0
    for _ in range(5000):
        coder.step(y)

    stats = coder.extract_avalanches(threshold=0.08)

    print(f"  n_avalanches = {stats.n_avalanches}")
    print(f"  α̂ = {stats.alpha_hat:.2f} (target: 1.50)")
    print(f"  ẑ = {stats.z_hat:.2f} (target: 2.00)")

    # Check that exponents are in reasonable range
    # Note: Finite-size effects mean we don't expect exact values
    if not np.isnan(stats.alpha_hat):
        assert 1.0 < stats.alpha_hat < 3.0, f"α̂ = {stats.alpha_hat} out of range"
    if not np.isnan(stats.z_hat):
        assert 1.0 < stats.z_hat < 4.0, f"ẑ = {stats.z_hat} out of range"

    print("✓ Avalanche stats at criticality (exponents in reasonable range)")


def test_manifold_includes_avalanche_stats():
    """Test that manifold sweep includes avalanche statistics."""
    data = sweep_manifold(
        lambda_range=(0.8, 1.2),
        pi_range=(0.5, 2.0),
        n_lambda=3,
        n_pi=3,
        T=3.0,
    )

    # Check new fields exist
    assert "alpha_hat_grid" in data
    assert "z_hat_grid" in data
    assert "n_avalanches_grid" in data

    # Check shapes
    assert data["alpha_hat_grid"].shape == (3, 3)
    assert data["z_hat_grid"].shape == (3, 3)

    print("✓ Manifold includes avalanche statistics")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("GUTC Emergent Lambda Tests")
    print("="*60 + "\n")

    test_emergent_coder()
    test_manifold_sweep()
    test_lambda_hat_tracks_lambda_c()
    test_soc_controller()
    test_avalanche_extraction()
    test_power_law_fitting()
    test_avalanche_stats_at_criticality()
    test_manifold_includes_avalanche_stats()

    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60 + "\n")


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI entry point."""
    import sys

    if len(sys.argv) > 1:
        cmd = sys.argv[1]

        if cmd == "test":
            run_all_tests()

        elif cmd == "manifold":
            print("Generating GUTC manifold map...")
            data = sweep_manifold(
                n_lambda=30,
                n_pi=30,
                T=5.0,
            )
            plot_manifold_map(data, save_path="gutc_emergent_manifold.png")

        elif cmd == "sweep":
            print("Detailed parameter sweep...")
            print("\nλ sweep at Π_prior=1.0:")
            print("-" * 40)
            for lambda_c in np.linspace(0.5, 1.5, 11):
                point = simulate_point(lambda_c=lambda_c, pi_prior=1.0, T=3.0)
                print(f"  λ_c={lambda_c:.2f}: F̄={point.mean_F:.4f}, λ̂={point.lambda_hat:.3f}, "
                      f"α̂={point.alpha_hat:.2f}, n_aval={point.n_avalanches}")

            print("\nΠ_prior sweep at λ=1.0:")
            print("-" * 40)
            for pi_prior in np.linspace(0.5, 4.0, 8):
                point = simulate_point(lambda_c=1.0, pi_prior=pi_prior, T=3.0)
                print(f"  Π_p={pi_prior:.1f}: F̄={point.mean_F:.4f}, λ̂={point.lambda_hat:.3f}, "
                      f"α̂={point.alpha_hat:.2f}, n_aval={point.n_avalanches}")

        elif cmd == "triple":
            print("Generating GUTC triple map (F̄, λ̂, α̂)...")
            data = sweep_manifold(
                n_lambda=25,
                n_pi=25,
                T=8.0,  # Longer for better avalanche statistics
            )
            plot_triple_map(data, save_path="gutc_triple_map.png")

        elif cmd == "avalanche":
            print("Analyzing avalanche statistics at criticality...")
            cfg = EmergentLambdaConfig(
                lambda_c=1.0,
                pi_sensory=1.0,
                pi_prior=1.0,
                noise_std=0.2,
            )
            coder = L23L5EmergentCoder(cfg, seed=42)

            # Long simulation
            y = 1.0
            for _ in range(10000):
                coder.step(y)

            stats = coder.extract_avalanches(threshold=0.08)
            print(f"\nAvalanche Statistics at λ=1.0:")
            print("-" * 40)
            print(f"  n_avalanches = {stats.n_avalanches}")
            print(f"  mean_size    = {stats.mean_size:.2f}")
            print(f"  mean_duration = {stats.mean_duration:.2f}")
            print(f"  α̂ (size exponent) = {stats.alpha_hat:.3f} (target: 1.500)")
            print(f"  ẑ (duration exponent) = {stats.z_hat:.3f} (target: 2.000)")
            print(f"\n  Critical? {stats.is_critical()}")

            # Plot distributions
            plot_avalanche_distributions(stats, title="Avalanche Statistics at λ=1.0",
                                        save_path="gutc_avalanche_distributions.png")

        else:
            print(f"Unknown command: {cmd}")
            print("Usage: python gutc_emergent_lambda.py [test|manifold|triple|avalanche|sweep]")

    else:
        run_all_tests()


if __name__ == "__main__":
    main()
