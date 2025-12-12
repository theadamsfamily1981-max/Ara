#!/usr/bin/env python3
"""
GUTC Figure 2: Comparative Dynamical Regimes
=============================================

Generates the key Figure 2 for the GUTC manuscript, showing:
- Row 1: Prediction error timeseries eps_y(t) for each regime
- Row 2: Avalanche size distributions with power-law fits

Three regimes compared:
1. Healthy (lambda=1.0, balanced precision) - Critical scaling
2. ASD-like (lambda<1, high prior precision) - Subcritical, rigid
3. Psychotic-like (lambda>1, high sensory precision) - Supercritical, unstable

Usage:
    python gutc_figure2_dynamics.py              # Generate figure
    python gutc_figure2_dynamics.py --save       # Save to PNG
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import warnings

# Suppress runtime warnings from log-log fits with limited data
warnings.filterwarnings("ignore", category=RuntimeWarning)


# =============================================================================
# Avalanche Analysis (Reused from gutc_emergent_lambda.py)
# =============================================================================

def extract_avalanche_sizes(activity_signal: np.ndarray) -> np.ndarray:
    """
    Extract avalanche sizes from a 1D boolean activity signal.

    An avalanche is a contiguous run of True values, bounded by False.
    Size = number of active time steps in that run.
    """
    activity_signal = np.asarray(activity_signal, dtype=bool)
    if activity_signal.size == 0:
        return np.array([])

    # Pad with False at both ends to catch start/end boundaries
    padded = np.concatenate([[False], activity_signal, [False]])
    diff = np.diff(padded.astype(int))

    # +1 = start index, -1 = end index
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    sizes = ends - starts
    return sizes[sizes > 0]


def fit_power_law_exponent(
    sizes: np.ndarray,
    min_size_for_fit: int = 3,
    n_bins: int = 15,
    r2_threshold: float = 0.7,
) -> Tuple[float, float]:
    """
    Estimate power-law exponent alpha via linear regression on log-binned data.

    Returns:
        (alpha_hat, r_squared) - fitted exponent and R^2 of fit
    """
    sizes = np.asarray(sizes, dtype=float)
    if len(sizes) < 50:
        return np.nan, 0.0

    sizes = sizes[sizes >= min_size_for_fit]
    if len(sizes) < 30 or sizes.max() < 5:
        return np.nan, 0.0

    # Log-binning
    log_min = np.log10(sizes.min())
    log_max = np.log10(sizes.max())
    log_bins = np.logspace(log_min, log_max, n_bins)

    hist, bin_edges = np.histogram(sizes, bins=log_bins, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    valid = hist > 0
    x_data = bin_centers[valid]
    y_data = hist[valid]

    if len(x_data) < 4:
        return np.nan, 0.0

    # Linear regression in log-log space
    log_x = np.log10(x_data)
    log_y = np.log10(y_data)

    slope, intercept, r_value, p_value, std_err = linregress(log_x, log_y)
    alpha_hat = -slope
    r_squared = r_value ** 2

    # Only return fit if correlation is reasonable
    if r_squared < r2_threshold:
        return np.nan, r_squared

    return alpha_hat, r_squared


# =============================================================================
# L2/3-L5 Predictive Coder (Simplified for Figure)
# =============================================================================

@dataclass
class RegimeConfig:
    """Configuration for a clinical regime."""
    name: str
    lambda_c: float
    pi_sensory: float
    pi_prior: float
    color: str
    description: str


# Clinical regime exemplars
REGIME_CONFIGS = {
    "healthy": RegimeConfig(
        name="Healthy",
        lambda_c=1.0,
        pi_sensory=2.0,
        pi_prior=2.0,
        color='green',
        description=r"$\lambda=1.0$, Balanced $\Pi$"
    ),
    "asd_like": RegimeConfig(
        name="ASD-like (Rigid)",
        lambda_c=0.8,
        pi_sensory=1.0,
        pi_prior=5.0,
        color='blue',
        description=r"$\lambda=0.8$, High $\Pi_{prior}$"
    ),
    "psychotic_like": RegimeConfig(
        name="Psychotic-like (Unstable)",
        lambda_c=1.3,
        pi_sensory=5.0,
        pi_prior=1.0,
        color='red',
        description=r"$\lambda=1.3$, High $\Pi_{sens}$"
    ),
}


class L23L5Coder:
    """
    Minimal 1D predictive-coding loop for figure generation.
    """

    def __init__(
        self,
        C: float = 1.0,
        mu0: float = 0.0,
        lambda_c: float = 1.0,
        W_recur: float = 1.0,
        pi_sensory: float = 1.0,
        pi_prior: float = 1.0,
        tau_mu: float = 10.0,
        tau_eps: float = 5.0,
        dt: float = 0.01,
        noise_std: float = 0.02,
    ):
        self.C = C
        self.mu0 = mu0
        self.lambda_c = lambda_c
        self.W_recur = W_recur
        self.pi_sensory = pi_sensory
        self.pi_prior = pi_prior
        self.tau_mu = tau_mu
        self.tau_eps = tau_eps
        self.dt = dt
        self.noise_std = noise_std

        # State
        self.mu = mu0
        self.eps_y = 0.0
        self.eps_mu = 0.0

        # RNG for noise
        self.rng = np.random.default_rng(42)

    def step(self, y: float) -> Tuple[float, float, float, float]:
        """Single integration step."""
        # Target errors
        eps_y_target = y - self.C * self.mu
        eps_mu_target = self.mu - self.mu0

        # Error dynamics (L2/3)
        d_eps_y = (eps_y_target - self.eps_y) / self.tau_eps
        d_eps_mu = (eps_mu_target - self.eps_mu) / self.tau_eps

        # Add noise for avalanche generation
        noise_scale = self.noise_std * np.sqrt(self.dt)
        self.eps_y += self.dt * d_eps_y + noise_scale * self.rng.standard_normal()
        self.eps_mu += self.dt * d_eps_mu + noise_scale * self.rng.standard_normal()

        # State dynamics (L5)
        recur = (-1.0 + self.lambda_c * self.W_recur) * self.mu
        drive_sens = self.C * self.pi_sensory * self.eps_y
        drive_prior = -self.pi_prior * self.eps_mu

        d_mu = (recur + drive_sens + drive_prior) / self.tau_mu
        self.mu += self.dt * d_mu + noise_scale * self.rng.standard_normal()

        # Free energy
        F = 0.5 * self.pi_sensory * self.eps_y**2 + 0.5 * self.pi_prior * self.eps_mu**2

        return self.mu, self.eps_y, self.eps_mu, F


def simulate_regime(
    config: RegimeConfig,
    T: float = 30.0,
    dt: float = 0.01,
    y_obs: float = 1.0,
) -> Dict[str, np.ndarray]:
    """
    Simulate a single regime and return timeseries data.
    """
    coder = L23L5Coder(
        lambda_c=config.lambda_c,
        pi_sensory=config.pi_sensory,
        pi_prior=config.pi_prior,
        dt=dt,
    )

    n_steps = int(T / dt)
    time = np.linspace(0, T, n_steps)
    eps_y_history = np.zeros(n_steps)
    mu_history = np.zeros(n_steps)
    F_history = np.zeros(n_steps)

    for i in range(n_steps):
        mu, eps_y, eps_mu, F = coder.step(y_obs)
        eps_y_history[i] = eps_y
        mu_history[i] = mu
        F_history[i] = F

    return {
        "time": time,
        "eps_y": eps_y_history,
        "mu": mu_history,
        "F": F_history,
    }


# =============================================================================
# Figure 2: Comparative Dynamics Plot
# =============================================================================

def plot_figure2(
    save_path: Optional[str] = None,
    T: float = 30.0,
    activity_threshold: float = 0.05,
):
    """
    Generate Figure 2: Comparative dynamical regimes.

    Row 1: Prediction error timeseries eps_y(t)
    Row 2: Avalanche size distributions with power-law fits
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    regimes = ["healthy", "asd_like", "psychotic_like"]

    for col_idx, regime_key in enumerate(regimes):
        config = REGIME_CONFIGS[regime_key]

        # === Simulate ===
        data = simulate_regime(config, T=T)
        time = data["time"]
        eps_y = data["eps_y"]

        # === Avalanche extraction ===
        activity_signal = np.abs(eps_y) > activity_threshold
        sizes = extract_avalanche_sizes(activity_signal)
        alpha_hat, r_squared = fit_power_law_exponent(sizes)

        # === Row 1: Timeseries ===
        ax_ts = axes[0, col_idx]
        ax_ts.plot(time, eps_y, color=config.color, linewidth=0.8, alpha=0.8)
        ax_ts.axhline(y=activity_threshold, color='k', linestyle='--', alpha=0.4)
        ax_ts.axhline(y=-activity_threshold, color='k', linestyle='--', alpha=0.4)
        ax_ts.fill_between(time, -activity_threshold, activity_threshold,
                           color='gray', alpha=0.1, label='Quiescent zone')

        ax_ts.set_title(f"{config.name}\n{config.description}", fontsize=11)
        ax_ts.set_xlim(0, T)
        ax_ts.set_ylim(-0.5, 0.5)

        if col_idx == 0:
            ax_ts.set_ylabel(r"Prediction Error $\varepsilon_y(t)$", fontsize=10)
        else:
            ax_ts.set_yticklabels([])

        if col_idx == 1:
            ax_ts.set_xlabel("Time (s)", fontsize=10)

        # === Row 2: Avalanche distribution ===
        ax_hist = axes[1, col_idx]

        if len(sizes) >= 30 and not np.isnan(alpha_hat):
            # Log-binned histogram
            log_min = np.log10(max(1, sizes.min()))
            log_max = np.log10(sizes.max())
            log_bins = np.logspace(log_min, log_max, 15)

            hist, bin_edges = np.histogram(sizes, bins=log_bins, density=True)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            valid = hist > 0

            # Plot data points
            ax_hist.loglog(bin_centers[valid], hist[valid], 'o',
                          color=config.color, markersize=6, alpha=0.7,
                          label='Avalanches')

            # Plot power-law fit
            fit_x = np.logspace(log_min, log_max, 100)
            fit_y = fit_x ** (-alpha_hat)
            # Normalize to match data
            fit_y = fit_y * (hist[valid].max() / fit_y.max()) * 2

            ax_hist.loglog(fit_x, fit_y, '--', color='black', linewidth=2,
                          label=fr'$s^{{-{alpha_hat:.2f}}}$')

            # Reference critical line (alpha = 3/2)
            ref_y = fit_x ** (-1.5)
            ref_y = ref_y * (hist[valid].max() / ref_y.max()) * 2
            ax_hist.loglog(fit_x, ref_y, ':', color='gray', linewidth=1.5,
                          alpha=0.7, label=r'Critical $s^{-3/2}$')

            # Annotation
            ax_hist.text(0.95, 0.95, fr"$\hat{{\alpha}} = {alpha_hat:.2f}$" + f"\n$R^2 = {r_squared:.2f}$",
                        transform=ax_hist.transAxes, fontsize=11,
                        ha='right', va='top',
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

            # Critical status
            if abs(alpha_hat - 1.5) < 0.2:
                status_text = "CRITICAL"
                status_color = 'green'
            elif alpha_hat > 1.5:
                status_text = "SUBCRITICAL"
                status_color = 'blue'
            else:
                status_text = "SUPERCRITICAL"
                status_color = 'red'

            ax_hist.text(0.05, 0.05, status_text,
                        transform=ax_hist.transAxes, fontsize=10,
                        fontweight='bold', color=status_color)

            ax_hist.legend(loc='upper right', fontsize=8)

        else:
            ax_hist.text(0.5, 0.5, f"Insufficient avalanches\n(n={len(sizes)})",
                        ha='center', va='center', transform=ax_hist.transAxes,
                        fontsize=11, color='gray')

        ax_hist.set_xlabel("Avalanche Size $s$", fontsize=10)
        if col_idx == 0:
            ax_hist.set_ylabel(r"$P(s)$ (Density)", fontsize=10)
        else:
            ax_hist.set_yticklabels([])

    # Suptitle
    fig.suptitle(
        r"Figure 2: VFE Engine Dynamics and Emergent Criticality across $(\lambda, \Pi)$ Regimes",
        fontsize=14, fontweight='bold'
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()


# =============================================================================
# Tests
# =============================================================================

def test_avalanche_extraction():
    """Test avalanche extraction."""
    # Simple test signal
    signal = np.array([False, True, True, False, True, False, True, True, True, False])
    sizes = extract_avalanche_sizes(signal)
    assert len(sizes) == 3  # Three avalanches
    assert list(sizes) == [2, 1, 3]  # Sizes 2, 1, 3
    print("Avalanche extraction test passed")


def test_regime_simulation():
    """Test regime simulation."""
    config = REGIME_CONFIGS["healthy"]
    data = simulate_regime(config, T=5.0)
    assert len(data["time"]) == len(data["eps_y"])
    assert np.all(np.isfinite(data["eps_y"]))
    print("Regime simulation test passed")


def run_tests():
    """Run all tests."""
    print("\n" + "="*50)
    print("GUTC Figure 2 Tests")
    print("="*50 + "\n")
    test_avalanche_extraction()
    test_regime_simulation()
    print("\nAll tests passed!")


# =============================================================================
# CLI
# =============================================================================

def main():
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--save":
            plot_figure2(save_path="GUTC_Figure2_Dynamics.png")
        elif sys.argv[1] == "test":
            run_tests()
        else:
            print("Usage: python gutc_figure2_dynamics.py [--save|test]")
    else:
        plot_figure2()


if __name__ == "__main__":
    main()
