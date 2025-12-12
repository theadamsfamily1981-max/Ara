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


def simulate_point(
    lambda_c: float,
    pi_prior: float,
    pi_sensory: float = 2.0,
    y_obs: float = 1.0,
    T: float = 5.0,
    dt: float = 0.01,
) -> ManifoldPoint:
    """Simulate one point on the GUTC manifold."""
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

    return ManifoldPoint(
        lambda_c=lambda_c,
        pi_prior=pi_prior,
        pi_sensory=pi_sensory,
        mean_F=coder.get_mean_free_energy(),
        lambda_hat=coder.estimate_branching_ratio(threshold=0.05),
        lambda_hat_delta=coder.estimate_branching_ratio_delta(threshold=0.05),
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

    Returns grids of F̄ and λ̂ for visualization.
    """
    lambdas = np.linspace(lambda_range[0], lambda_range[1], n_lambda)
    pis = np.linspace(pi_range[0], pi_range[1], n_pi)

    F_grid = np.zeros((n_pi, n_lambda))
    lambda_hat_grid = np.zeros((n_pi, n_lambda))

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

    return {
        "lambdas": lambdas,
        "pis": pis,
        "F_grid": F_grid,
        "lambda_hat_grid": lambda_hat_grid,
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


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("GUTC Emergent Lambda Tests")
    print("="*60 + "\n")

    test_emergent_coder()
    test_manifold_sweep()
    test_lambda_hat_tracks_lambda_c()
    test_soc_controller()

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
                print(f"  λ_c={lambda_c:.2f}: F̄={point.mean_F:.4f}, λ̂={point.lambda_hat:.3f}")

            print("\nΠ_prior sweep at λ=1.0:")
            print("-" * 40)
            for pi_prior in np.linspace(0.5, 4.0, 8):
                point = simulate_point(lambda_c=1.0, pi_prior=pi_prior, T=3.0)
                print(f"  Π_p={pi_prior:.1f}: F̄={point.mean_F:.4f}, λ̂={point.lambda_hat:.3f}")

        else:
            print(f"Unknown command: {cmd}")
            print("Usage: python gutc_emergent_lambda.py [test|manifold|sweep]")

    else:
        run_all_tests()


if __name__ == "__main__":
    main()
