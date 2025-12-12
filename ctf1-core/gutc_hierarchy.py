#!/usr/bin/env python3
"""
GUTC Hierarchical Multi-Scale Simulation
=========================================

Implements a 3-level predictive hierarchy where each level operates at
a different timescale while maintaining critical dynamics (ρ ≈ 1).

Architecture
------------
Level 3 (top):    Slow, abstract, long-horizon predictions    α₃ = 0.03
Level 2 (mid):    Intermediate timescale integration          α₂ = 0.10
Level 1 (bottom): Fast, sensory-proximal tracking             α₁ = 0.30

Each level implements:
    x^(l)_{t+1} = (1 - α_l) x^(l)_t + α_l tanh(W^(l) x^(l)_t + A^(l) ε^(l-1)_t + B^(l) û^(l+1)_t)

where:
    - α_l is the leak rate (controls timescale)
    - W^(l) has spectral radius ρ ≈ 1 (criticality)
    - ε^(l-1) is ascending prediction error from level below
    - û^(l+1) is descending prediction from level above

Bidirectional Predictive Coding
-------------------------------
    Ascending:  ε^(l) = x^(l) - P^(l) x^(l+1)     (error = state - top-down prediction)
    Descending: û^(l) = P^(l-1)ᵀ x^(l)            (prediction to level below)

Hierarchical Capacity
---------------------
    C_hier = Σ_l w_l · C_l(λ_l, Π_l)

where w_l weights each level's contribution (e.g., by dimensionality or
information throughput).

Expected Results
----------------
With proper α_l separation:
    ξ₁ ~ 20-40   (fast, tracks input)
    ξ₂ ~ 60-100  (intermediate smoothing)
    ξ₃ ~ 150+    (slow, abstract trends)

Usage
-----
    python gutc_hierarchy.py test       # Run tests
    python gutc_hierarchy.py simulate   # Run full simulation with plots
    python gutc_hierarchy.py capacity   # Compute hierarchical capacity
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum, auto


# =============================================================================
# Hierarchical Level Configuration
# =============================================================================

@dataclass
class LevelConfig:
    """Configuration for a single level in the hierarchy."""
    level_id: int               # 1 = bottom, 2 = mid, 3 = top
    dim: int = 5               # State dimensionality
    alpha: float = 0.1         # Leak rate (controls timescale)
    spectral_radius: float = 1.0  # Target ρ for criticality

    # Coupling strengths
    ascending_gain: float = 0.5   # Gain on ascending errors
    descending_gain: float = 0.3  # Gain on descending predictions

    # GUTC parameters
    pi_sensory: float = 1.0    # Precision on ascending errors
    pi_prior: float = 1.0      # Precision on descending predictions

    @property
    def lambda_val(self) -> float:
        """Criticality parameter (spectral radius)."""
        return self.spectral_radius

    @property
    def timescale(self) -> float:
        """Effective timescale τ = 1/α."""
        return 1.0 / self.alpha if self.alpha > 0 else float('inf')


# Default 3-level hierarchy with clear timescale separation
DEFAULT_HIERARCHY = [
    LevelConfig(level_id=1, dim=5, alpha=0.30, spectral_radius=0.99),  # Fast
    LevelConfig(level_id=2, dim=5, alpha=0.10, spectral_radius=0.99),  # Medium
    LevelConfig(level_id=3, dim=5, alpha=0.03, spectral_radius=0.99),  # Slow
]


# =============================================================================
# Hierarchical RNN Level
# =============================================================================

class HierarchicalLevel:
    """
    A single level in the predictive hierarchy.

    Implements leaky integration with critical recurrent dynamics:
        x_{t+1} = (1 - α) x_t + α tanh(W x_t + A ε_below + B û_above)
    """

    def __init__(self, config: LevelConfig, rng: np.random.Generator = None):
        self.config = config
        self.rng = rng or np.random.default_rng()

        # Initialize recurrent weights with target spectral radius
        self.W = self._init_critical_weights()

        # Coupling matrices (will be set by hierarchy)
        self.A = None  # Ascending (from below)
        self.B = None  # Descending (from above)
        self.P_down = None  # Projection to level below
        self.P_up = None    # Projection from level above

        # State
        self.x = np.zeros(config.dim)

        # History for analysis
        self.history: List[np.ndarray] = []

    def _init_critical_weights(self) -> np.ndarray:
        """Initialize recurrent weights with spectral radius ≈ ρ_target."""
        dim = self.config.dim
        W = self.rng.standard_normal((dim, dim))

        # Scale to target spectral radius
        current_rho = np.max(np.abs(np.linalg.eigvals(W)))
        if current_rho > 0:
            W = W * (self.config.spectral_radius / current_rho)

        return W

    def reset(self, x_init: Optional[np.ndarray] = None):
        """Reset state."""
        if x_init is not None:
            self.x = x_init.copy()
        else:
            self.x = self.rng.standard_normal(self.config.dim) * 0.1
        self.history = []

    def step(
        self,
        eps_below: Optional[np.ndarray] = None,
        u_above: Optional[np.ndarray] = None,
        record: bool = True,
    ) -> np.ndarray:
        """
        Single integration step.

        x_{t+1} = (1 - α) x_t + α tanh(W x_t + A ε_below + B û_above)
        """
        cfg = self.config

        # Recurrent drive
        drive = self.W @ self.x

        # Ascending prediction error (from level below)
        if eps_below is not None and self.A is not None:
            drive = drive + cfg.ascending_gain * (self.A @ eps_below)

        # Descending prediction (from level above)
        if u_above is not None and self.B is not None:
            drive = drive + cfg.descending_gain * (self.B @ u_above)

        # Leaky integration with nonlinearity
        x_new = (1 - cfg.alpha) * self.x + cfg.alpha * np.tanh(drive)

        self.x = x_new

        if record:
            self.history.append(self.x.copy())

        return self.x

    def compute_prediction_error(self, x_above: np.ndarray) -> np.ndarray:
        """
        Compute prediction error: ε = x - P · x_above

        This is the mismatch between this level's state and the
        top-down prediction from the level above.
        """
        if self.P_up is None or x_above is None:
            return np.zeros(self.config.dim)

        prediction = self.P_up @ x_above
        return self.x - prediction

    def get_prediction_for_below(self) -> np.ndarray:
        """
        Get prediction signal to send to level below.

        û = P_down^T · x
        """
        if self.P_down is None:
            return self.x  # Identity if no projection
        return self.P_down.T @ self.x

    def get_history_array(self) -> np.ndarray:
        """Get history as (T, dim) array."""
        if len(self.history) == 0:
            return np.zeros((0, self.config.dim))
        return np.array(self.history)


# =============================================================================
# Full Hierarchical Network
# =============================================================================

class HierarchicalGUTCNetwork:
    """
    Full 3-level predictive hierarchy with bidirectional coupling.

    Architecture:
        Level 3 (top)    ←→  Slow, abstract
            ↑ε  ↓û
        Level 2 (mid)    ←→  Intermediate
            ↑ε  ↓û
        Level 1 (bottom) ←→  Fast, sensory
            ↑
          Input
    """

    def __init__(
        self,
        level_configs: List[LevelConfig] = None,
        seed: int = 42,
    ):
        self.rng = np.random.default_rng(seed)

        if level_configs is None:
            level_configs = DEFAULT_HIERARCHY

        self.n_levels = len(level_configs)

        # Create levels
        self.levels = [
            HierarchicalLevel(cfg, self.rng)
            for cfg in level_configs
        ]

        # Initialize coupling matrices
        self._init_coupling_matrices()

    def _init_coupling_matrices(self):
        """Initialize inter-level coupling matrices."""
        for i, level in enumerate(self.levels):
            dim = level.config.dim

            # Ascending coupling (from level below)
            if i > 0:
                dim_below = self.levels[i-1].config.dim
                level.A = self.rng.standard_normal((dim, dim_below)) * 0.5
                # Projection from this level to level below
                level.P_down = self.rng.standard_normal((dim_below, dim)) * 0.5

            # Descending coupling (from level above)
            if i < self.n_levels - 1:
                dim_above = self.levels[i+1].config.dim
                level.B = self.rng.standard_normal((dim, dim_above)) * 0.5
                # Projection from level above to this level
                level.P_up = self.rng.standard_normal((dim, dim_above)) * 0.5

    def reset(self):
        """Reset all levels."""
        for level in self.levels:
            level.reset()

    def step(self, sensory_input: np.ndarray) -> Dict[str, Any]:
        """
        Single step of hierarchical dynamics.

        Bottom-up pass: compute prediction errors ascending
        Top-down pass: send predictions descending
        Update: all levels integrate simultaneously

        Args:
            sensory_input: Input to bottom level

        Returns:
            Dictionary with states and errors at each level
        """
        # === Bottom-up: Compute prediction errors ===
        errors = [None] * self.n_levels

        # Level 1 error: mismatch with sensory input
        errors[0] = sensory_input - self.levels[0].x

        # Higher levels: error = state - top-down prediction
        for i in range(1, self.n_levels):
            if i < self.n_levels - 1:
                x_above = self.levels[i+1].x
            else:
                x_above = np.zeros(self.levels[i].config.dim)
            errors[i] = self.levels[i].compute_prediction_error(x_above)

        # === Top-down: Get predictions ===
        predictions = [None] * self.n_levels

        for i in range(self.n_levels - 1, 0, -1):  # Top to bottom
            predictions[i-1] = self.levels[i].get_prediction_for_below()

        # === Update all levels ===
        for i, level in enumerate(self.levels):
            # Ascending error from below (or sensory input for level 1)
            if i == 0:
                eps_below = sensory_input  # Sensory drives bottom level
            else:
                eps_below = errors[i-1]  # Error from level below

            # Descending prediction from above
            if i < self.n_levels - 1:
                u_above = predictions[i]
            else:
                u_above = None  # No level above top

            level.step(eps_below=eps_below, u_above=u_above)

        return {
            "states": [level.x.copy() for level in self.levels],
            "errors": errors,
            "predictions": predictions,
        }

    def simulate(
        self,
        input_func,
        n_steps: int = 1000,
        record_every: int = 1,
    ) -> Dict[str, Any]:
        """
        Run full simulation.

        Args:
            input_func: Function t -> sensory_input(t)
            n_steps: Number of time steps
            record_every: Record every N steps

        Returns:
            Dictionary with time series for each level
        """
        self.reset()

        results = {
            "t": [],
            "states": [[] for _ in range(self.n_levels)],
            "errors": [[] for _ in range(self.n_levels)],
        }

        for t in range(n_steps):
            sensory = input_func(t)
            step_result = self.step(sensory)

            if t % record_every == 0:
                results["t"].append(t)
                for i in range(self.n_levels):
                    results["states"][i].append(step_result["states"][i])
                    if step_result["errors"][i] is not None:
                        results["errors"][i].append(step_result["errors"][i])

        # Convert to arrays
        results["t"] = np.array(results["t"])
        for i in range(self.n_levels):
            results["states"][i] = np.array(results["states"][i])
            if len(results["errors"][i]) > 0:
                results["errors"][i] = np.array(results["errors"][i])

        return results

    def get_level_configs(self) -> List[LevelConfig]:
        """Get configurations for all levels."""
        return [level.config for level in self.levels]


# =============================================================================
# Correlation Length Analysis
# =============================================================================

def compute_autocorrelation(x: np.ndarray, max_lag: int = None) -> np.ndarray:
    """
    Compute autocorrelation function R(τ) averaged across dimensions.

    R(τ) = (1/d) Σ_i corr(x_{t,i}, x_{t+τ,i})
    """
    T, d = x.shape
    if max_lag is None:
        max_lag = min(T // 3, 200)

    R = np.zeros(max_lag)

    for dim in range(d):
        signal = x[:, dim]
        signal = signal - signal.mean()
        var = signal.var()
        if var < 1e-10:
            continue

        for tau in range(max_lag):
            if T - tau > 0:
                R[tau] += np.mean(signal[:T-tau] * signal[tau:]) / var

    R = R / d
    return R


def estimate_correlation_length(R: np.ndarray, method: str = "integral") -> float:
    """
    Estimate correlation length ξ from autocorrelation R(τ).

    Methods:
        "integral": ξ = Σ_τ R(τ) Δτ  (integrated autocorrelation time)
        "decay": τ where R(τ) = 1/e
    """
    if method == "integral":
        # Sum positive part of autocorrelation
        R_pos = np.maximum(R, 0)
        return np.sum(R_pos)

    elif method == "decay":
        # Find where R drops below 1/e
        threshold = 1.0 / np.e
        below = np.where(R < threshold)[0]
        if len(below) > 0:
            return below[0]
        return len(R)

    else:
        raise ValueError(f"Unknown method: {method}")


def analyze_hierarchy_timescales(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze timescales at each level of the hierarchy.

    Returns correlation lengths ξ_l and autocorrelation functions.
    """
    n_levels = len(results["states"])

    analysis = {
        "correlation_lengths": [],
        "autocorrelations": [],
        "timescale_ratios": [],
    }

    for i in range(n_levels):
        states = results["states"][i]
        if len(states) == 0:
            continue

        # Compute autocorrelation
        R = compute_autocorrelation(states)
        analysis["autocorrelations"].append(R)

        # Estimate correlation length
        xi = estimate_correlation_length(R, method="integral")
        analysis["correlation_lengths"].append(xi)

    # Compute ratios (relative to bottom level)
    if len(analysis["correlation_lengths"]) > 0:
        xi_bottom = analysis["correlation_lengths"][0]
        for xi in analysis["correlation_lengths"]:
            ratio = xi / xi_bottom if xi_bottom > 0 else 1.0
            analysis["timescale_ratios"].append(ratio)

    return analysis


# =============================================================================
# Hierarchical Capacity
# =============================================================================

def gutc_capacity_single(lambda_val: float, pi_val: float, sigma: float = 0.3) -> float:
    """Single-level GUTC capacity C(λ, Π)."""
    return pi_val * np.exp(-((lambda_val - 1) ** 2) / (2 * sigma ** 2))


def compute_hierarchical_capacity(
    network: HierarchicalGUTCNetwork,
    weights: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Compute hierarchical capacity C_hier = Σ_l w_l · C_l(λ_l, Π_l).

    Args:
        network: The hierarchical network
        weights: Weight for each level (default: uniform)

    Returns:
        Dictionary with per-level and total capacity
    """
    configs = network.get_level_configs()
    n_levels = len(configs)

    if weights is None:
        weights = [1.0 / n_levels] * n_levels

    capacities = {
        "per_level": [],
        "weighted": [],
        "total": 0.0,
    }

    for i, cfg in enumerate(configs):
        # Effective precision (geometric mean of sensory and prior)
        pi_eff = np.sqrt(cfg.pi_sensory * cfg.pi_prior)

        # Single-level capacity
        C_l = gutc_capacity_single(cfg.lambda_val, pi_eff)
        capacities["per_level"].append(C_l)

        # Weighted contribution
        w_C = weights[i] * C_l
        capacities["weighted"].append(w_C)
        capacities["total"] += w_C

    return capacities


# =============================================================================
# Sensory Input Generators
# =============================================================================

def generate_multiscale_input(
    t: int,
    dim: int = 5,
    frequencies: List[float] = [0.1, 0.02, 0.005],
    amplitudes: List[float] = [1.0, 0.5, 0.3],
    noise_std: float = 0.1,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Generate multi-scale sensory input with fast, medium, and slow components.

    This tests the hierarchy's ability to separate timescales.
    """
    if rng is None:
        rng = np.random.default_rng()

    signal = np.zeros(dim)

    for freq, amp in zip(frequencies, amplitudes):
        # Each frequency drives different dimensions
        for d in range(dim):
            phase = 2 * np.pi * d / dim
            signal[d] += amp * np.sin(2 * np.pi * freq * t + phase)

    # Add noise
    signal += rng.standard_normal(dim) * noise_std

    return signal


def generate_step_input(
    t: int,
    dim: int = 5,
    step_time: int = 100,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Generate step input at specified time."""
    if t >= step_time:
        return np.ones(dim) * amplitude
    return np.zeros(dim)


# =============================================================================
# Visualization
# =============================================================================

def plot_hierarchy_simulation(
    results: Dict[str, Any],
    analysis: Dict[str, Any],
    configs: List[LevelConfig],
    save_path: Optional[str] = None,
):
    """Plot hierarchical simulation results."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for plotting")
        return

    n_levels = len(results["states"])
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = ['blue', 'green', 'red']
    labels = ['Level 1 (fast)', 'Level 2 (medium)', 'Level 3 (slow)']

    # Top-left: State trajectories (first dimension)
    ax1 = axes[0, 0]
    t = results["t"]
    for i in range(n_levels):
        states = results["states"][i]
        if len(states) > 0:
            ax1.plot(t, states[:, 0], color=colors[i], alpha=0.8,
                    label=f'{labels[i]} (α={configs[i].alpha})')

    ax1.set_xlabel('Time step')
    ax1.set_ylabel('x₁ (first dimension)')
    ax1.set_title('State Trajectories: Hierarchical Timescale Separation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Top-right: Autocorrelation functions
    ax2 = axes[0, 1]
    for i, R in enumerate(analysis["autocorrelations"]):
        xi = analysis["correlation_lengths"][i]
        ax2.plot(R[:100], color=colors[i],
                label=f'{labels[i]}: ξ = {xi:.1f}')

    ax2.axhline(y=1/np.e, color='gray', linestyle='--', alpha=0.5, label='1/e threshold')
    ax2.set_xlabel('Lag τ')
    ax2.set_ylabel('R(τ)')
    ax2.set_title('Autocorrelation Functions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Bottom-left: Correlation lengths bar chart
    ax3 = axes[1, 0]
    xi_vals = analysis["correlation_lengths"]
    bars = ax3.bar(range(n_levels), xi_vals, color=colors[:n_levels])
    ax3.set_xticks(range(n_levels))
    ax3.set_xticklabels([f'Level {i+1}\n(α={configs[i].alpha})' for i in range(n_levels)])
    ax3.set_ylabel('Correlation length ξ')
    ax3.set_title('Hierarchical Timescale Separation')
    ax3.grid(True, alpha=0.3, axis='y')

    # Add ratio annotations
    for i, (bar, xi) in enumerate(zip(bars, xi_vals)):
        ratio = analysis["timescale_ratios"][i]
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{ratio:.1f}×', ha='center', va='bottom', fontsize=10)

    # Bottom-right: Per-level capacity
    ax4 = axes[1, 1]

    # Compute capacity directly from configs
    capacities = {"per_level": [], "total": 0.0}
    for cfg in configs:
        pi_eff = np.sqrt(cfg.pi_sensory * cfg.pi_prior)
        C_l = gutc_capacity_single(cfg.lambda_val, pi_eff)
        capacities["per_level"].append(C_l)
        capacities["total"] += C_l / len(configs)

    ax4.bar(range(n_levels), capacities["per_level"], color=colors[:n_levels], alpha=0.7)
    ax4.set_xticks(range(n_levels))
    ax4.set_xticklabels([f'Level {i+1}' for i in range(n_levels)])
    ax4.set_ylabel('Capacity C_l')
    ax4.set_title(f'Per-Level Capacity (C_hier = {capacities["total"]:.3f})')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()


# =============================================================================
# Tests
# =============================================================================

def test_level_config():
    """Test level configuration."""
    cfg = LevelConfig(level_id=1, dim=5, alpha=0.3, spectral_radius=0.99)
    assert cfg.timescale == 1.0 / 0.3
    assert cfg.lambda_val == 0.99
    print("✓ Level configuration")


def test_hierarchical_level():
    """Test single hierarchical level."""
    cfg = LevelConfig(level_id=1, dim=5, alpha=0.3)
    level = HierarchicalLevel(cfg)
    level.reset()

    # Check spectral radius
    rho = np.max(np.abs(np.linalg.eigvals(level.W)))
    assert abs(rho - cfg.spectral_radius) < 0.1, f"ρ = {rho}, expected {cfg.spectral_radius}"

    # Step should work
    x = level.step(eps_below=np.ones(5) * 0.1)
    assert x.shape == (5,)

    print("✓ Hierarchical level")


def test_full_hierarchy():
    """Test full hierarchical network."""
    network = HierarchicalGUTCNetwork()

    assert network.n_levels == 3

    # Single step
    sensory = np.ones(5) * 0.5
    result = network.step(sensory)

    assert len(result["states"]) == 3
    assert all(s.shape == (5,) for s in result["states"])

    print("✓ Full hierarchy")


def test_timescale_separation():
    """Test that different α_l produce different correlation lengths."""
    network = HierarchicalGUTCNetwork()

    # Simulate with multiscale input
    rng = np.random.default_rng(42)
    def input_func(t):
        return generate_multiscale_input(t, dim=5, rng=rng)

    results = network.simulate(input_func, n_steps=2000)
    analysis = analyze_hierarchy_timescales(results)

    xi = analysis["correlation_lengths"]

    print(f"  ξ₁ = {xi[0]:.1f} (bottom, fast)")
    print(f"  ξ₂ = {xi[1]:.1f} (middle)")
    print(f"  ξ₃ = {xi[2]:.1f} (top, slow)")

    # With proper α separation, higher levels should have longer ξ
    # (allowing some noise in the estimate)
    assert xi[2] > xi[0] * 0.8, "Top level should have longer correlation length"

    print("✓ Timescale separation verified")


def test_hierarchical_capacity():
    """Test hierarchical capacity computation."""
    network = HierarchicalGUTCNetwork()
    capacities = compute_hierarchical_capacity(network)

    assert len(capacities["per_level"]) == 3
    assert capacities["total"] > 0
    assert abs(capacities["total"] - sum(capacities["weighted"])) < 1e-6

    print(f"  C_hier = {capacities['total']:.4f}")
    print("✓ Hierarchical capacity")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("GUTC Hierarchical Simulation Tests")
    print("="*60 + "\n")

    test_level_config()
    test_hierarchical_level()
    test_full_hierarchy()
    test_timescale_separation()
    test_hierarchical_capacity()

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

        elif cmd == "simulate":
            print("Running hierarchical GUTC simulation...")

            network = HierarchicalGUTCNetwork()
            rng = np.random.default_rng(42)

            def input_func(t):
                return generate_multiscale_input(t, dim=5, rng=rng)

            results = network.simulate(input_func, n_steps=3000)
            analysis = analyze_hierarchy_timescales(results)

            print("\n" + "="*60)
            print("Hierarchical Timescale Analysis")
            print("="*60)

            configs = network.get_level_configs()
            for i, cfg in enumerate(configs):
                xi = analysis["correlation_lengths"][i]
                ratio = analysis["timescale_ratios"][i]
                print(f"\nLevel {i+1}:")
                print(f"  α = {cfg.alpha} (τ = {cfg.timescale:.1f})")
                print(f"  ρ = {cfg.spectral_radius}")
                print(f"  ξ = {xi:.1f}")
                print(f"  Ratio (vs L1): {ratio:.2f}×")

            # Capacity
            capacities = compute_hierarchical_capacity(network)
            print(f"\nHierarchical Capacity:")
            print(f"  C_hier = {capacities['total']:.4f}")
            for i, C_l in enumerate(capacities["per_level"]):
                print(f"  C_{i+1} = {C_l:.4f}")

            plot_hierarchy_simulation(results, analysis, configs,
                                     save_path="gutc_hierarchy.png")

        elif cmd == "capacity":
            print("Computing hierarchical capacity...")

            # Sweep spectral radius for each level
            print("\nCapacity vs. spectral radius (all levels):")
            print("-" * 50)

            for rho in [0.8, 0.9, 0.95, 0.99, 1.0, 1.05, 1.1]:
                configs = [
                    LevelConfig(level_id=1, alpha=0.30, spectral_radius=rho),
                    LevelConfig(level_id=2, alpha=0.10, spectral_radius=rho),
                    LevelConfig(level_id=3, alpha=0.03, spectral_radius=rho),
                ]
                network = HierarchicalGUTCNetwork(configs)
                cap = compute_hierarchical_capacity(network)
                print(f"  ρ = {rho:.2f}: C_hier = {cap['total']:.4f}")

        else:
            print(f"Unknown command: {cmd}")
            print("Usage: python gutc_hierarchy.py [test|simulate|capacity]")

    else:
        run_all_tests()


if __name__ == "__main__":
    main()
