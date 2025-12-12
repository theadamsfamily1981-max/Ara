#!/usr/bin/env python3
"""
GUTC Free-Energy Functional and Gradient Flow
==============================================

Implements the variational free-energy functional whose gradient descent
recovers the L2/3↔L5 predictive coding dynamics.

Core Result:
    The cortical L2/3↔L5 loop literally performs variational inference:

    F(μ̂; y) = ½(y - g(μ̂))ᵀ Π_y (y - g(μ̂)) + ½(μ̂ - μ₀)ᵀ Π_μ (μ̂ - μ₀)

    Neural dynamics implement -∇F:
    - L2/3 (PE): τ_ε ε̇ = -ε + Π_sensory(y - g(μ̂))
    - L5 (Pred): τ_μ μ̂̇ = Π_sensory g'(μ̂)ᵀ ε - Π_prior(μ̂ - μ₀)

GUTC Parameters:
    λ (criticality): Scales recurrent coupling, sets distance to bifurcation
    Π_sensory: Precision on sensory prediction errors (ACh, L4/L2/3)
    Π_prior: Precision on prior predictions (DA, L5/L6)

Capacity Formula:
    C(λ, Π) = Π · exp(-(λ-1)²/2σ²)

    Maximum capacity at criticality (λ=1), allocated by precision weights.

Usage:
    from gutc_free_energy import (
        free_energy,
        L23L5Loop,
        TwoLevelHierarchy,
        verify_gradient_flow
    )

    # Single-level loop
    loop = L23L5Loop(pi_sensory=1.0, pi_prior=0.5)
    trajectory = loop.simulate(y_input, duration=1.0)

    # Verify F decreases
    verify_gradient_flow(loop)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable, Any
import matplotlib.pyplot as plt


# =============================================================================
# Free-Energy Functional
# =============================================================================

def free_energy(
    mu_hat: np.ndarray,
    y: np.ndarray,
    g_func: Callable[[np.ndarray], np.ndarray],
    pi_y: float,
    pi_mu: float,
    mu_0: np.ndarray,
) -> float:
    """
    Compute variational free energy F(μ̂; y).

    F = ½(y - g(μ̂))ᵀ Π_y (y - g(μ̂)) + ½(μ̂ - μ₀)ᵀ Π_μ (μ̂ - μ₀)

    Args:
        mu_hat: Current belief / L5 activity (n_dim,)
        y: Sensory input / L4 drive (n_dim,)
        g_func: Generative mapping μ → predicted y
        pi_y: Sensory precision Π_sensory
        pi_mu: Prior precision Π_prior
        mu_0: Prior mean

    Returns:
        Free energy value (scalar)
    """
    # Sensory prediction error
    eps_y = y - g_func(mu_hat)
    sensory_term = 0.5 * pi_y * np.dot(eps_y, eps_y)

    # Prior deviation
    eps_mu = mu_hat - mu_0
    prior_term = 0.5 * pi_mu * np.dot(eps_mu, eps_mu)

    return sensory_term + prior_term


def free_energy_gradient(
    mu_hat: np.ndarray,
    y: np.ndarray,
    g_func: Callable[[np.ndarray], np.ndarray],
    g_jacobian: Callable[[np.ndarray], np.ndarray],
    pi_y: float,
    pi_mu: float,
    mu_0: np.ndarray,
) -> np.ndarray:
    """
    Compute gradient ∇_μ F(μ̂; y).

    ∇F = -g'(μ̂)ᵀ Π_y (y - g(μ̂)) + Π_μ (μ̂ - μ₀)

    Args:
        mu_hat: Current belief (n_dim,)
        y: Sensory input (n_dim,)
        g_func: Generative mapping
        g_jacobian: Jacobian of g at μ̂
        pi_y: Sensory precision
        pi_mu: Prior precision
        mu_0: Prior mean

    Returns:
        Gradient vector (n_dim,)
    """
    # Sensory prediction error
    eps_y = y - g_func(mu_hat)

    # Jacobian of generative model
    J = g_jacobian(mu_hat)  # (n_out, n_dim) or scalar for 1D

    # Gradient terms
    if np.isscalar(J):
        sensory_grad = -J * pi_y * eps_y
    else:
        sensory_grad = -J.T @ (pi_y * eps_y)

    prior_grad = pi_mu * (mu_hat - mu_0)

    return sensory_grad + prior_grad


# =============================================================================
# Generative Model Functions
# =============================================================================

def linear_g(mu: np.ndarray, W: np.ndarray = None) -> np.ndarray:
    """Linear generative model: g(μ) = Wμ."""
    if W is None:
        return mu  # Identity
    return W @ mu


def linear_g_jacobian(mu: np.ndarray, W: np.ndarray = None) -> np.ndarray:
    """Jacobian of linear g."""
    if W is None:
        return np.eye(len(mu)) if hasattr(mu, '__len__') else 1.0
    return W


def sigmoid_g(mu: np.ndarray, gain: float = 1.0) -> np.ndarray:
    """Sigmoid generative model: g(μ) = σ(gain·μ)."""
    return 1.0 / (1.0 + np.exp(-gain * mu))


def sigmoid_g_jacobian(mu: np.ndarray, gain: float = 1.0) -> np.ndarray:
    """Jacobian of sigmoid g."""
    s = sigmoid_g(mu, gain)
    return gain * np.diag(s * (1 - s))


# =============================================================================
# L2/3↔L5 Single-Level Loop
# =============================================================================

@dataclass
class L23L5LoopConfig:
    """Configuration for single-level predictive coding loop."""
    n_dim: int = 1                  # Dimensionality of representations
    tau_eps: float = 0.01           # L2/3 time constant (fast)
    tau_mu: float = 0.1             # L5 time constant (slow)
    pi_sensory: float = 1.0         # Π_sensory (ACh)
    pi_prior: float = 1.0           # Π_prior (DA)
    lambda_coupling: float = 1.0    # Global coupling (criticality)
    mu_0: float = 0.0               # Prior mean
    g_type: str = "linear"          # Generative model type
    g_gain: float = 1.0             # Gain for nonlinear g


class L23L5Loop:
    """
    Single-level L2/3↔L5 predictive coding loop.

    Implements gradient descent on free energy:
        L2/3 (ε): τ_ε ε̇ = -ε + Π_s(y - g(μ̂))
        L5 (μ̂): τ_μ μ̂̇ = λ·Π_s g'(μ̂)ᵀ ε - Π_p(μ̂ - μ₀)
    """

    def __init__(self, config: L23L5LoopConfig = None, **kwargs):
        if config is None:
            config = L23L5LoopConfig(**kwargs)
        self.config = config

        # State variables
        self.n_dim = config.n_dim
        self.eps = np.zeros(self.n_dim)      # L2/3 prediction error
        self.mu_hat = np.zeros(self.n_dim)   # L5 prediction
        self.mu_0 = np.ones(self.n_dim) * config.mu_0

        # Setup generative model
        if config.g_type == "linear":
            self.g = lambda mu: linear_g(mu)
            self.g_jac = lambda mu: linear_g_jacobian(mu)
        elif config.g_type == "sigmoid":
            self.g = lambda mu: sigmoid_g(mu, config.g_gain)
            self.g_jac = lambda mu: sigmoid_g_jacobian(mu, config.g_gain)
        else:
            raise ValueError(f"Unknown g_type: {config.g_type}")

        # History for analysis
        self.history: Dict[str, List] = {
            'time': [],
            'eps': [],
            'mu_hat': [],
            'F': [],
            'y': [],
        }

    def reset(self, mu_init: np.ndarray = None):
        """Reset state."""
        self.eps = np.zeros(self.n_dim)
        self.mu_hat = mu_init if mu_init is not None else np.zeros(self.n_dim)
        self.history = {k: [] for k in self.history}

    def step(self, y: np.ndarray, dt: float = 0.001) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single integration step.

        Args:
            y: Sensory input
            dt: Time step

        Returns:
            (eps, mu_hat) after update
        """
        cfg = self.config

        # Current prediction error
        pred_error = y - self.g(self.mu_hat)

        # L2/3 dynamics: τ_ε ε̇ = -ε + Π_s(y - g(μ̂))
        d_eps = (-self.eps + cfg.pi_sensory * pred_error) / cfg.tau_eps

        # L5 dynamics: τ_μ μ̂̇ = λ·Π_s g'(μ̂)ᵀ ε - Π_p(μ̂ - μ₀)
        J = self.g_jac(self.mu_hat)
        if np.isscalar(J):
            ascending = cfg.lambda_coupling * cfg.pi_sensory * J * self.eps
        else:
            ascending = cfg.lambda_coupling * cfg.pi_sensory * J.T @ self.eps

        prior_pull = cfg.pi_prior * (self.mu_hat - self.mu_0)
        d_mu = (ascending - prior_pull) / cfg.tau_mu

        # Euler integration
        self.eps = self.eps + dt * d_eps
        self.mu_hat = self.mu_hat + dt * d_mu

        return self.eps.copy(), self.mu_hat.copy()

    def simulate(
        self,
        y_func: Callable[[float], np.ndarray],
        duration: float = 1.0,
        dt: float = 0.001,
        record: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Simulate loop dynamics.

        Args:
            y_func: Function t -> y(t) giving sensory input
            duration: Simulation duration
            dt: Time step
            record: Whether to record history

        Returns:
            Dictionary with time series
        """
        n_steps = int(duration / dt)

        for i in range(n_steps):
            t = i * dt
            y = y_func(t)

            self.step(y, dt)

            if record:
                F = self.compute_free_energy(y)
                self.history['time'].append(t)
                self.history['eps'].append(self.eps.copy())
                self.history['mu_hat'].append(self.mu_hat.copy())
                self.history['F'].append(F)
                self.history['y'].append(y.copy())

        return {k: np.array(v) for k, v in self.history.items()}

    def compute_free_energy(self, y: np.ndarray) -> float:
        """Compute current free energy."""
        return free_energy(
            self.mu_hat, y, self.g,
            self.config.pi_sensory, self.config.pi_prior, self.mu_0
        )

    def compute_gradient(self, y: np.ndarray) -> np.ndarray:
        """Compute current free energy gradient."""
        return free_energy_gradient(
            self.mu_hat, y, self.g, self.g_jac,
            self.config.pi_sensory, self.config.pi_prior, self.mu_0
        )


# =============================================================================
# Two-Level Hierarchy (Areas k and k+1)
# =============================================================================

@dataclass
class TwoLevelConfig:
    """Configuration for two-level hierarchical predictive coding."""
    n_dim: int = 2

    # Time constants
    tau_eps_lower: float = 0.01     # L2/3 lower area (fast)
    tau_mu_lower: float = 0.05      # L5 lower area
    tau_eps_higher: float = 0.02    # L2/3 higher area
    tau_mu_higher: float = 0.1      # L5 higher area (slowest)

    # Precision (level-specific)
    pi_sensory_lower: float = 1.0   # Sensory precision at lower level
    pi_prior_lower: float = 0.5     # Prior from higher level
    pi_sensory_higher: float = 0.5  # Ascending errors to higher level
    pi_prior_higher: float = 1.0    # Top-level prior

    # Coupling
    lambda_ff: float = 1.0          # Feedforward coupling
    lambda_fb: float = 1.0          # Feedback coupling

    # Priors
    mu_0_higher: float = 0.0        # Top-level prior mean


class TwoLevelHierarchy:
    """
    Two-level hierarchical predictive coding.

    Lower Area (sensory):
        - Receives external input y_ext
        - L2/3_lower: ε_lower = Π_s^low (y_ext - g_lower(μ̂_lower))
        - L5_lower: Updates μ̂_lower from ε_lower and FB from higher

    Higher Area (abstract):
        - Receives ascending PE from lower: y_high = ε_lower
        - L2/3_higher: ε_higher = Π_s^high (y_high - g_higher(μ̂_higher))
        - L5_higher: Updates μ̂_higher, sends FB to lower

    The feedback connection implements:
        μ_0^lower = h(μ̂_higher)  (higher predictions become lower priors)
    """

    def __init__(self, config: TwoLevelConfig = None, **kwargs):
        if config is None:
            config = TwoLevelConfig(**kwargs)
        self.config = config

        n = config.n_dim

        # Lower area state
        self.eps_lower = np.zeros(n)
        self.mu_lower = np.zeros(n)

        # Higher area state
        self.eps_higher = np.zeros(n)
        self.mu_higher = np.zeros(n)

        # Top-level prior
        self.mu_0_higher = np.ones(n) * config.mu_0_higher

        # Generative models (linear for simplicity)
        self.g_lower = lambda mu: mu
        self.g_higher = lambda mu: mu
        self.h_fb = lambda mu: mu  # FB mapping: higher → lower prior

        # History
        self.history: Dict[str, List] = {
            'time': [],
            'eps_lower': [], 'mu_lower': [],
            'eps_higher': [], 'mu_higher': [],
            'F_total': [],
        }

    def reset(self):
        """Reset all state."""
        n = self.config.n_dim
        self.eps_lower = np.zeros(n)
        self.mu_lower = np.zeros(n)
        self.eps_higher = np.zeros(n)
        self.mu_higher = np.zeros(n)
        self.history = {k: [] for k in self.history}

    def step(self, y_ext: np.ndarray, dt: float = 0.001):
        """
        Single integration step for two-level hierarchy.

        The key insight: higher predictions become lower priors via FB.
        """
        cfg = self.config

        # === Lower Area ===
        # Prior comes from higher area via feedback
        mu_0_lower = cfg.lambda_fb * self.h_fb(self.mu_higher)

        # Sensory prediction error
        pred_error_lower = y_ext - self.g_lower(self.mu_lower)

        # L2/3 lower: τ ε̇ = -ε + Π_s(y - g(μ))
        d_eps_lower = (
            -self.eps_lower + cfg.pi_sensory_lower * pred_error_lower
        ) / cfg.tau_eps_lower

        # L5 lower: τ μ̇ = Π_s g'ᵀ ε - Π_p(μ - μ₀)
        ascending_lower = cfg.lambda_ff * cfg.pi_sensory_lower * self.eps_lower
        prior_pull_lower = cfg.pi_prior_lower * (self.mu_lower - mu_0_lower)
        d_mu_lower = (ascending_lower - prior_pull_lower) / cfg.tau_mu_lower

        # === Higher Area ===
        # Input is ascending PE from lower (this is the FF message)
        y_higher = self.eps_lower

        # Prediction error at higher level
        pred_error_higher = y_higher - self.g_higher(self.mu_higher)

        # L2/3 higher
        d_eps_higher = (
            -self.eps_higher + cfg.pi_sensory_higher * pred_error_higher
        ) / cfg.tau_eps_higher

        # L5 higher (top level, prior is fixed)
        ascending_higher = cfg.lambda_ff * cfg.pi_sensory_higher * self.eps_higher
        prior_pull_higher = cfg.pi_prior_higher * (self.mu_higher - self.mu_0_higher)
        d_mu_higher = (ascending_higher - prior_pull_higher) / cfg.tau_mu_higher

        # Euler integration
        self.eps_lower = self.eps_lower + dt * d_eps_lower
        self.mu_lower = self.mu_lower + dt * d_mu_lower
        self.eps_higher = self.eps_higher + dt * d_eps_higher
        self.mu_higher = self.mu_higher + dt * d_mu_higher

    def simulate(
        self,
        y_func: Callable[[float], np.ndarray],
        duration: float = 1.0,
        dt: float = 0.001,
    ) -> Dict[str, np.ndarray]:
        """Simulate two-level hierarchy."""
        n_steps = int(duration / dt)

        for i in range(n_steps):
            t = i * dt
            y = y_func(t)

            self.step(y, dt)

            # Compute total free energy (sum of both levels)
            F_lower = 0.5 * self.config.pi_sensory_lower * np.dot(
                y - self.g_lower(self.mu_lower),
                y - self.g_lower(self.mu_lower)
            )
            mu_0_lower = self.config.lambda_fb * self.h_fb(self.mu_higher)
            F_lower += 0.5 * self.config.pi_prior_lower * np.dot(
                self.mu_lower - mu_0_lower,
                self.mu_lower - mu_0_lower
            )

            F_higher = 0.5 * self.config.pi_sensory_higher * np.dot(
                self.eps_lower - self.g_higher(self.mu_higher),
                self.eps_lower - self.g_higher(self.mu_higher)
            )
            F_higher += 0.5 * self.config.pi_prior_higher * np.dot(
                self.mu_higher - self.mu_0_higher,
                self.mu_higher - self.mu_0_higher
            )

            self.history['time'].append(t)
            self.history['eps_lower'].append(self.eps_lower.copy())
            self.history['mu_lower'].append(self.mu_lower.copy())
            self.history['eps_higher'].append(self.eps_higher.copy())
            self.history['mu_higher'].append(self.mu_higher.copy())
            self.history['F_total'].append(F_lower + F_higher)

        return {k: np.array(v) for k, v in self.history.items()}


# =============================================================================
# GUTC Capacity and Phase Diagram
# =============================================================================

def gutc_capacity(
    lambda_val: float,
    pi_sensory: float,
    pi_prior: float,
    sigma: float = 0.3,
) -> float:
    """
    Compute GUTC information capacity.

    C(λ, Π) = Π_mean · exp(-(λ-1)²/2σ²)

    Maximum at criticality (λ=1), scaled by precision.
    """
    pi_mean = (pi_sensory + pi_prior) / 2
    criticality_factor = np.exp(-((lambda_val - 1.0)**2) / (2 * sigma**2))
    return pi_mean * criticality_factor


def compute_capacity_surface(
    lambda_range: Tuple[float, float] = (0.5, 1.5),
    pi_range: Tuple[float, float] = (0.2, 2.0),
    n_points: int = 50,
    sigma: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute capacity surface C(λ, Π) over parameter grid.

    Returns:
        (Lambda, Pi, Capacity) meshgrids
    """
    lambda_vals = np.linspace(lambda_range[0], lambda_range[1], n_points)
    pi_vals = np.linspace(pi_range[0], pi_range[1], n_points)

    Lambda, Pi = np.meshgrid(lambda_vals, pi_vals)
    Capacity = Pi * np.exp(-((Lambda - 1.0)**2) / (2 * sigma**2))

    return Lambda, Pi, Capacity


# =============================================================================
# Verification: Gradient Flow Minimizes F
# =============================================================================

def verify_gradient_flow(
    loop: L23L5Loop,
    y_value: float = 1.0,
    duration: float = 0.5,
    dt: float = 0.001,
    plot: bool = True,
) -> Dict[str, Any]:
    """
    Verify that L2/3↔L5 dynamics minimize free energy.

    Checks:
    1. F decreases monotonically (or nearly so)
    2. Final state is near gradient = 0
    3. Prediction converges toward input
    """
    loop.reset()

    # Constant input
    y = np.array([y_value])
    y_func = lambda t: y

    # Simulate
    history = loop.simulate(y_func, duration=duration, dt=dt)

    # Analysis
    F_values = history['F']
    mu_values = history['mu_hat']

    # Check 1: F decreases
    F_decreasing = np.all(np.diff(F_values) <= 1e-6)  # Allow tiny numerical noise

    # Check 2: Final gradient near zero
    final_grad = loop.compute_gradient(y)
    grad_magnitude = np.linalg.norm(final_grad)

    # Check 3: Prediction converges
    final_mu = mu_values[-1]
    prediction_error = np.linalg.norm(y - loop.g(final_mu))

    results = {
        'F_monotonic_decrease': F_decreasing,
        'final_gradient_norm': grad_magnitude,
        'final_prediction_error': prediction_error,
        'F_initial': F_values[0],
        'F_final': F_values[-1],
        'F_reduction': F_values[0] - F_values[-1],
        'converged': grad_magnitude < 0.1 and prediction_error < 0.1,
    }

    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        # F over time
        axes[0, 0].plot(history['time'], F_values)
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Free Energy F')
        axes[0, 0].set_title('F decreases (gradient descent)')

        # μ̂ over time
        axes[0, 1].plot(history['time'], mu_values)
        axes[0, 1].axhline(y_value, color='r', linestyle='--', label='Target y')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('μ̂ (L5)')
        axes[0, 1].set_title('Prediction converges to input')
        axes[0, 1].legend()

        # ε over time
        eps_values = history['eps']
        axes[1, 0].plot(history['time'], eps_values)
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('ε (L2/3)')
        axes[1, 0].set_title('Prediction error (drives learning)')

        # Capacity surface
        L, P, C = compute_capacity_surface()
        ax = axes[1, 1]
        cf = ax.contourf(L, P, C, levels=20, cmap='YlOrRd')
        plt.colorbar(cf, ax=ax, label='Capacity')
        ax.axvline(loop.config.lambda_coupling, color='white', linestyle='--')
        ax.scatter([loop.config.lambda_coupling],
                   [(loop.config.pi_sensory + loop.config.pi_prior)/2],
                   color='white', s=100, marker='*')
        ax.set_xlabel('λ (criticality)')
        ax.set_ylabel('Π (precision)')
        ax.set_title('GUTC Capacity C(λ,Π)')

        plt.tight_layout()
        plt.savefig('gutc_free_energy_verification.png', dpi=150)
        plt.close()
        print("Saved: gutc_free_energy_verification.png")

    return results


# =============================================================================
# Demo / CLI
# =============================================================================

def demo_single_level():
    """Demonstrate single-level L2/3↔L5 loop."""
    print("\n" + "="*60)
    print("GUTC Free-Energy: Single-Level L2/3↔L5 Loop")
    print("="*60)

    # Create loop with different precision settings
    configs = [
        ("Balanced", L23L5LoopConfig(pi_sensory=1.0, pi_prior=1.0)),
        ("Sensory-heavy", L23L5LoopConfig(pi_sensory=2.0, pi_prior=0.5)),
        ("Prior-heavy", L23L5LoopConfig(pi_sensory=0.5, pi_prior=2.0)),
    ]

    for name, cfg in configs:
        loop = L23L5Loop(config=cfg)
        result = verify_gradient_flow(loop, y_value=1.0, plot=False)

        print(f"\n{name} (Π_s={cfg.pi_sensory}, Π_p={cfg.pi_prior}):")
        print(f"  F: {result['F_initial']:.3f} → {result['F_final']:.3f}")
        print(f"  |∇F|: {result['final_gradient_norm']:.4f}")
        print(f"  Converged: {result['converged']}")


def demo_two_level():
    """Demonstrate two-level hierarchy."""
    print("\n" + "="*60)
    print("GUTC Free-Energy: Two-Level Hierarchy")
    print("="*60)

    hier = TwoLevelHierarchy()

    # Step input
    def y_func(t):
        return np.array([1.0, 0.5]) if t > 0.1 else np.array([0.0, 0.0])

    history = hier.simulate(y_func, duration=1.0)

    print(f"\nSimulation complete:")
    print(f"  F_total: {history['F_total'][0]:.3f} → {history['F_total'][-1]:.3f}")
    print(f"  μ_lower final: {history['mu_lower'][-1]}")
    print(f"  μ_higher final: {history['mu_higher'][-1]}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GUTC Free-Energy Functional")
    parser.add_argument('--demo', choices=['single', 'two', 'all'], default='all')
    parser.add_argument('--verify', action='store_true', help="Run verification with plot")

    args = parser.parse_args()

    if args.verify:
        loop = L23L5Loop(pi_sensory=1.0, pi_prior=0.5, lambda_coupling=1.0)
        result = verify_gradient_flow(loop, plot=True)
        print("\nVerification results:")
        for k, v in result.items():
            print(f"  {k}: {v}")

    if args.demo in ['single', 'all']:
        demo_single_level()

    if args.demo in ['two', 'all']:
        demo_two_level()

    print("\n✓ GUTC Free-Energy module complete")
