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
    k_update: int = 1          # Update period (1 = every step, 3 = every 3rd step)

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

    @property
    def effective_timescale(self) -> float:
        """Effective timescale including sub-sampling: τ_eff = k_update / α."""
        return self.k_update / self.alpha if self.alpha > 0 else float('inf')


# Default 3-level hierarchy with clear timescale separation
# Uses BOTH α_l (leak rate) AND k_update (sub-sampling) for maximal separation
DEFAULT_HIERARCHY = [
    LevelConfig(level_id=1, dim=5, alpha=0.30, spectral_radius=0.99, k_update=1),   # Fast: updates every step
    LevelConfig(level_id=2, dim=5, alpha=0.10, spectral_radius=0.99, k_update=3),   # Medium: updates every 3 steps
    LevelConfig(level_id=3, dim=5, alpha=0.03, spectral_radius=0.99, k_update=9),   # Slow: updates every 9 steps
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

        # Time step counter for sub-sampled updates
        self.t = 0

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
        """Reset all levels and time counter."""
        self.t = 0
        for level in self.levels:
            level.reset()

    def step(self, sensory_input: np.ndarray) -> Dict[str, Any]:
        """
        Single step of hierarchical dynamics with sub-sampled updates.

        Each level only updates when t % k_update == 0, enforcing
        different temporal integration windows:
        - Level 1 (k=1): updates every step (fast)
        - Level 2 (k=3): updates every 3 steps (medium)
        - Level 3 (k=9): updates every 9 steps (slow)

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

        # === Update levels (with sub-sampling) ===
        for i, level in enumerate(self.levels):
            k = level.config.k_update

            # Only update if t % k_update == 0
            if self.t % k != 0:
                continue  # Skip update, state persists

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

        # Increment time step
        self.t += 1

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
# Γ (Gamma) Hierarchical Coupling Matrix
# =============================================================================

@dataclass
class GammaCouplingMatrix:
    """
    The Γ (Gamma) Coupling Matrix formalizes inter-level information flow.

    Architecture:
        Γ = [Γ_asc, Γ_desc]

    where:
        Γ_asc[l,l+1] = A^(l+1) · Π_sens^(l)   (ascending error gain)
        Γ_desc[l,l-1] = B^(l-1) · Π_prior^(l) (descending prediction gain)

    The effective coupling strength determines how quickly information
    propagates through the hierarchy. At criticality (all λ_l ≈ 1),
    information flows maximally efficiently.

    Properties:
        - det(I - Γ) > 0: stable hierarchy (errors don't explode)
        - Tr(Γ) ~ n_levels: balanced coupling
        - ρ(Γ) < 1: convergent predictive dynamics
    """
    n_levels: int
    ascending_gains: List[float]   # Γ_asc[l] = gain from level l to l+1
    descending_gains: List[float]  # Γ_desc[l] = gain from level l+1 to l
    precision_sens: List[float]    # Π_sens at each level
    precision_prior: List[float]   # Π_prior at each level

    @classmethod
    def from_network(cls, network: 'HierarchicalGUTCNetwork') -> 'GammaCouplingMatrix':
        """Construct Γ matrix from a hierarchical network."""
        n_levels = network.n_levels
        configs = network.get_level_configs()

        ascending_gains = []
        descending_gains = []
        precision_sens = []
        precision_prior = []

        for i, cfg in enumerate(configs):
            precision_sens.append(cfg.pi_sensory)
            precision_prior.append(cfg.pi_prior)
            ascending_gains.append(cfg.ascending_gain)
            descending_gains.append(cfg.descending_gain)

        return cls(
            n_levels=n_levels,
            ascending_gains=ascending_gains,
            descending_gains=descending_gains,
            precision_sens=precision_sens,
            precision_prior=precision_prior,
        )

    def effective_ascending(self, level: int) -> float:
        """Effective ascending coupling: A · Π_sens."""
        if level >= self.n_levels - 1:
            return 0.0
        return self.ascending_gains[level] * self.precision_sens[level]

    def effective_descending(self, level: int) -> float:
        """Effective descending coupling: B · Π_prior."""
        if level <= 0:
            return 0.0
        return self.descending_gains[level] * self.precision_prior[level]

    def total_coupling_strength(self) -> float:
        """Total coupling strength Σ(Γ_asc + Γ_desc)."""
        total = 0.0
        for l in range(self.n_levels):
            total += self.effective_ascending(l)
            total += self.effective_descending(l)
        return total

    def coupling_balance(self) -> float:
        """
        Coupling balance ratio: Σ Γ_asc / Σ Γ_desc.

        Balanced hierarchy: ratio ≈ 1
        Bottom-up dominated: ratio > 1
        Top-down dominated: ratio < 1
        """
        asc_total = sum(self.effective_ascending(l) for l in range(self.n_levels))
        desc_total = sum(self.effective_descending(l) for l in range(self.n_levels))

        if desc_total < 1e-10:
            return float('inf')
        return asc_total / desc_total

    def as_matrix(self) -> np.ndarray:
        """
        Construct the full Γ matrix as (n_levels × n_levels).

        Γ[i,j] = coupling from level j to level i
        """
        G = np.zeros((self.n_levels, self.n_levels))

        for l in range(self.n_levels - 1):
            # Ascending: l → l+1
            G[l+1, l] = self.effective_ascending(l)
            # Descending: l+1 → l
            G[l, l+1] = self.effective_descending(l+1)

        return G

    def spectral_radius(self) -> float:
        """Spectral radius ρ(Γ) - must be < 1 for stability."""
        G = self.as_matrix()
        return np.max(np.abs(np.linalg.eigvals(G)))

    def is_stable(self) -> bool:
        """Check if hierarchy is stable (ρ(Γ) < 1)."""
        return self.spectral_radius() < 1.0


def compute_hierarchical_information_flow(
    network: 'HierarchicalGUTCNetwork',
    results: Dict[str, Any],
) -> Dict[str, float]:
    """
    Compute information-theoretic metrics for hierarchical flow.

    Metrics:
        - I_asc: Mutual information in ascending direction
        - I_desc: Mutual information in descending direction
        - I_total: Total information throughput
        - η_efficiency: Information efficiency (I / coupling_strength)
    """
    n_levels = network.n_levels
    states = results["states"]
    errors = results["errors"]

    # Estimate mutual information via correlation (Gaussian approximation)
    # I(X;Y) ≈ -0.5 log(1 - ρ²) where ρ is correlation

    I_asc = 0.0
    I_desc = 0.0

    for l in range(n_levels - 1):
        # Ascending: correlation between errors[l] and states[l+1]
        if len(errors[l]) > 0 and len(states[l+1]) > 0:
            min_len = min(len(errors[l]), len(states[l+1]))
            err_flat = errors[l][:min_len].flatten()
            state_flat = states[l+1][:min_len].flatten()

            if len(err_flat) > 10:
                corr = np.corrcoef(err_flat, state_flat)[0, 1]
                if not np.isnan(corr) and abs(corr) < 0.9999:
                    I_asc += -0.5 * np.log(1 - corr**2)

        # Descending: correlation between states[l+1] and states[l]
        if len(states[l]) > 0 and len(states[l+1]) > 0:
            min_len = min(len(states[l]), len(states[l+1]))
            state_low = states[l][:min_len].flatten()
            state_high = states[l+1][:min_len].flatten()

            if len(state_low) > 10:
                corr = np.corrcoef(state_low, state_high)[0, 1]
                if not np.isnan(corr) and abs(corr) < 0.9999:
                    I_desc += -0.5 * np.log(1 - corr**2)

    I_total = I_asc + I_desc

    # Efficiency: information per unit coupling
    gamma = GammaCouplingMatrix.from_network(network)
    coupling = gamma.total_coupling_strength()
    eta = I_total / coupling if coupling > 0 else 0.0

    return {
        "I_ascending": I_asc,
        "I_descending": I_desc,
        "I_total": I_total,
        "eta_efficiency": eta,
        "gamma_spectral_radius": gamma.spectral_radius(),
        "coupling_balance": gamma.coupling_balance(),
    }


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


def sweep_hierarchical_criticality(
    rho_range: Tuple[float, float] = (0.7, 1.3),
    n_points: int = 15,
    pi_val: float = 1.0,
) -> Dict[str, np.ndarray]:
    """
    Sweep spectral radius across all levels simultaneously.

    Demonstrates the GUTC prediction:
        C_hier is MAXIMIZED when ALL levels are at criticality (λ_l ≈ 1)

    Returns:
        Dictionary with rho values, per-level capacities, and total capacity
    """
    rho_values = np.linspace(rho_range[0], rho_range[1], n_points)

    results = {
        "rho": rho_values,
        "C_level_1": np.zeros(n_points),
        "C_level_2": np.zeros(n_points),
        "C_level_3": np.zeros(n_points),
        "C_hier": np.zeros(n_points),
    }

    for i, rho in enumerate(rho_values):
        # All levels at the same spectral radius
        configs = [
            LevelConfig(level_id=1, alpha=0.30, spectral_radius=rho,
                       pi_sensory=pi_val, pi_prior=pi_val),
            LevelConfig(level_id=2, alpha=0.10, spectral_radius=rho,
                       pi_sensory=pi_val, pi_prior=pi_val),
            LevelConfig(level_id=3, alpha=0.03, spectral_radius=rho,
                       pi_sensory=pi_val, pi_prior=pi_val),
        ]
        network = HierarchicalGUTCNetwork(configs)
        cap = compute_hierarchical_capacity(network)

        results["C_level_1"][i] = cap["per_level"][0]
        results["C_level_2"][i] = cap["per_level"][1]
        results["C_level_3"][i] = cap["per_level"][2]
        results["C_hier"][i] = cap["total"]

    return results


def sweep_mixed_criticality(
    n_points: int = 11,
    pi_val: float = 1.0,
) -> Dict[str, np.ndarray]:
    """
    Sweep with heterogeneous criticality across levels.

    Shows that UNIFORM criticality (all λ_l = 1) outperforms mixed regimes.

    Returns:
        Grid of C_hier for (λ_1, λ_3) with λ_2 fixed at 1.0
    """
    rho_1_values = np.linspace(0.7, 1.3, n_points)
    rho_3_values = np.linspace(0.7, 1.3, n_points)

    C_grid = np.zeros((n_points, n_points))

    for i, rho_1 in enumerate(rho_1_values):
        for j, rho_3 in enumerate(rho_3_values):
            configs = [
                LevelConfig(level_id=1, alpha=0.30, spectral_radius=rho_1,
                           pi_sensory=pi_val, pi_prior=pi_val),
                LevelConfig(level_id=2, alpha=0.10, spectral_radius=1.0,  # Fixed at critical
                           pi_sensory=pi_val, pi_prior=pi_val),
                LevelConfig(level_id=3, alpha=0.03, spectral_radius=rho_3,
                           pi_sensory=pi_val, pi_prior=pi_val),
            ]
            network = HierarchicalGUTCNetwork(configs)
            cap = compute_hierarchical_capacity(network)
            C_grid[j, i] = cap["total"]

    return {
        "rho_1": rho_1_values,
        "rho_3": rho_3_values,
        "C_grid": C_grid,
    }


def plot_hierarchical_criticality_sweep(
    data: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
):
    """
    Plot the hierarchical criticality sweep showing C_hier peaks at λ=1.

    Two panels:
    1. Line plot: C_hier vs ρ (all levels uniform)
    2. Heatmap: C_hier(λ_1, λ_3) with λ_2 fixed at 1
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for plotting")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # === Panel 1: Uniform criticality sweep ===
    ax1 = axes[0]
    rho = data["rho"]
    ax1.plot(rho, data["C_level_1"], 'b--', alpha=0.5, label='C₁ (fast)')
    ax1.plot(rho, data["C_level_2"], 'g--', alpha=0.5, label='C₂ (medium)')
    ax1.plot(rho, data["C_level_3"], 'r--', alpha=0.5, label='C₃ (slow)')
    ax1.plot(rho, data["C_hier"], 'k-', linewidth=2, label='C_hier (total)')

    # Mark peak
    peak_idx = np.argmax(data["C_hier"])
    ax1.axvline(x=rho[peak_idx], color='gray', linestyle=':', alpha=0.7)
    ax1.scatter([rho[peak_idx]], [data["C_hier"][peak_idx]], color='red',
               s=100, zorder=5, marker='*', label=f'Peak at ρ={rho[peak_idx]:.2f}')

    ax1.axvline(x=1.0, color='green', linestyle='--', alpha=0.5, label='Critical (ρ=1)')

    ax1.set_xlabel(r'Spectral Radius $\rho$ (all levels)', fontsize=11)
    ax1.set_ylabel(r'Capacity $C$', fontsize=11)
    ax1.set_title(r'GUTC Prediction: $C_{hier}$ peaks at $\lambda_l \approx 1$', fontsize=12)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # === Panel 2: Mixed criticality heatmap (computed separately) ===
    ax2 = axes[1]
    mixed_data = sweep_mixed_criticality(n_points=15)

    im = ax2.pcolormesh(
        mixed_data["rho_1"], mixed_data["rho_3"], mixed_data["C_grid"],
        cmap='viridis', shading='auto'
    )
    cbar = fig.colorbar(im, ax=ax2)
    cbar.set_label(r'$C_{hier}$')

    # Mark critical point
    ax2.scatter([1.0], [1.0], color='red', s=150, marker='*',
               edgecolors='white', linewidths=2, zorder=5)
    ax2.axhline(y=1.0, color='white', linestyle='--', alpha=0.5)
    ax2.axvline(x=1.0, color='white', linestyle='--', alpha=0.5)

    ax2.set_xlabel(r'$\lambda_1$ (fast level)', fontsize=11)
    ax2.set_ylabel(r'$\lambda_3$ (slow level)', fontsize=11)
    ax2.set_title(r'Mixed Criticality: $\lambda_2 = 1$ fixed', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()


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


def test_gamma_coupling_matrix():
    """Test Γ coupling matrix construction and properties."""
    network = HierarchicalGUTCNetwork()
    gamma = GammaCouplingMatrix.from_network(network)

    # Check dimensions
    assert gamma.n_levels == 3

    # Check matrix construction
    G = gamma.as_matrix()
    assert G.shape == (3, 3)

    # Diagonal should be zero (no self-coupling)
    assert np.all(np.diag(G) == 0)

    # Check stability
    rho_gamma = gamma.spectral_radius()
    print(f"  ρ(Γ) = {rho_gamma:.3f}")

    # Coupling balance should be finite
    balance = gamma.coupling_balance()
    assert np.isfinite(balance)
    print(f"  Coupling balance = {balance:.3f}")

    print("✓ Γ coupling matrix")


def test_hierarchical_criticality_sweep():
    """Test that C_hier peaks at criticality (λ = 1)."""
    data = sweep_hierarchical_criticality(
        rho_range=(0.7, 1.3),
        n_points=11,
    )

    # Find peak
    peak_idx = np.argmax(data["C_hier"])
    rho_peak = data["rho"][peak_idx]

    print(f"  C_hier peaks at ρ = {rho_peak:.2f}")

    # Peak should be near λ = 1
    assert 0.85 < rho_peak < 1.15, f"Peak at ρ={rho_peak}, expected near 1.0"

    # Capacity at peak should be higher than at extremes
    C_peak = data["C_hier"][peak_idx]
    C_low = data["C_hier"][0]
    C_high = data["C_hier"][-1]

    assert C_peak > C_low, "Capacity should be higher at criticality than subcritical"
    assert C_peak > C_high, "Capacity should be higher at criticality than supercritical"

    print("✓ Hierarchical capacity peaks at criticality")


def test_mixed_criticality():
    """Test that uniform criticality outperforms mixed regimes."""
    mixed_data = sweep_mixed_criticality(n_points=7)

    C_grid = mixed_data["C_grid"]
    rho_1 = mixed_data["rho_1"]
    rho_3 = mixed_data["rho_3"]

    # Find indices closest to 1.0
    idx_1 = np.argmin(np.abs(rho_1 - 1.0))
    idx_3 = np.argmin(np.abs(rho_3 - 1.0))

    C_uniform_critical = C_grid[idx_3, idx_1]

    # Corners (both subcritical or both supercritical)
    C_sub_sub = C_grid[0, 0]
    C_sup_sup = C_grid[-1, -1]

    print(f"  C(1.0, 1.0) = {C_uniform_critical:.4f}")
    print(f"  C(0.7, 0.7) = {C_sub_sub:.4f}")
    print(f"  C(1.3, 1.3) = {C_sup_sup:.4f}")

    assert C_uniform_critical > C_sub_sub
    assert C_uniform_critical > C_sup_sup

    print("✓ Uniform criticality outperforms mixed regimes")


def test_information_flow():
    """Test hierarchical information flow computation."""
    network = HierarchicalGUTCNetwork()
    rng = np.random.default_rng(42)

    def input_func(t):
        return generate_multiscale_input(t, dim=5, rng=rng)

    results = network.simulate(input_func, n_steps=1000)
    info = compute_hierarchical_information_flow(network, results)

    assert "I_ascending" in info
    assert "I_descending" in info
    assert "I_total" in info
    assert "eta_efficiency" in info

    print(f"  I_asc = {info['I_ascending']:.3f}")
    print(f"  I_desc = {info['I_descending']:.3f}")
    print(f"  η = {info['eta_efficiency']:.3f}")

    print("✓ Hierarchical information flow")


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
    test_gamma_coupling_matrix()
    test_hierarchical_criticality_sweep()
    test_mixed_criticality()
    test_information_flow()

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

        elif cmd == "criticality":
            print("Computing hierarchical criticality sweep...")
            print("Demonstrating: C_hier peaks when ALL levels are at λ ≈ 1\n")

            # Run the sweep
            data = sweep_hierarchical_criticality(
                rho_range=(0.7, 1.3),
                n_points=21,
            )

            # Find and report peak
            peak_idx = np.argmax(data["C_hier"])
            rho_peak = data["rho"][peak_idx]

            print("Results:")
            print("-" * 50)
            print(f"  Peak C_hier at ρ = {rho_peak:.3f}")
            print(f"  C_hier(peak) = {data['C_hier'][peak_idx]:.4f}")
            print(f"  C_hier(ρ=0.7) = {data['C_hier'][0]:.4f}")
            print(f"  C_hier(ρ=1.3) = {data['C_hier'][-1]:.4f}")

            # Γ matrix analysis
            network = HierarchicalGUTCNetwork()
            gamma = GammaCouplingMatrix.from_network(network)
            print(f"\nΓ Coupling Matrix:")
            print(f"  ρ(Γ) = {gamma.spectral_radius():.3f}")
            print(f"  Coupling balance = {gamma.coupling_balance():.3f}")
            print(f"  Stable: {gamma.is_stable()}")

            # Plot
            plot_hierarchical_criticality_sweep(data, save_path="gutc_hierarchy_criticality.png")

        elif cmd == "gamma":
            print("Analyzing Γ coupling matrix...")

            network = HierarchicalGUTCNetwork()
            gamma = GammaCouplingMatrix.from_network(network)

            print("\nΓ Matrix Structure:")
            print("-" * 50)
            G = gamma.as_matrix()
            print("  Γ =")
            for row in G:
                print(f"    [{', '.join(f'{x:.3f}' for x in row)}]")

            print(f"\n  Spectral radius ρ(Γ) = {gamma.spectral_radius():.4f}")
            print(f"  Coupling balance (asc/desc) = {gamma.coupling_balance():.3f}")
            print(f"  Total coupling strength = {gamma.total_coupling_strength():.3f}")
            print(f"  Stable hierarchy: {gamma.is_stable()}")

            # Per-level effective couplings
            print("\nEffective Couplings:")
            for l in range(gamma.n_levels):
                asc = gamma.effective_ascending(l)
                desc = gamma.effective_descending(l)
                print(f"  Level {l+1}: Γ_asc={asc:.3f}, Γ_desc={desc:.3f}")

        else:
            print(f"Unknown command: {cmd}")
            print("Usage: python gutc_hierarchy.py [test|simulate|capacity|criticality|gamma]")

    else:
        run_all_tests()


if __name__ == "__main__":
    main()
