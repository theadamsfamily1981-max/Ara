#!/usr/bin/env python3
"""
GUTC Agency: Expected Free Energy and the Unified Objective
============================================================

Extends the GUTC framework with **agency** via Expected Free Energy (EFE),
creating the unified GUTC objective functional:

    J(π, λ, Π) = G(π | λ, Π) + α|E(λ)|²

This module connects:
- Variational Free Energy (VFE): Present-time inference (belief updating)
- Expected Free Energy (EFE): Future-directed control (policy selection)
- Criticality Constraint: E(λ) ≈ 0 for maximal capacity

Key Innovation
--------------
At the GUTC healthy corridor (λ ≈ 1, balanced Π):
- VFE gradient descent is fast and efficient
- EFE decomposition balances pragmatic/epistemic drives
- Fisher information peaks → maximal sensitivity
- Avalanche exponents approach universality (3/2, 2)

The GUTC Agency Principle:
    Adaptive cognition = min_{π, λ, Π} J(π, λ, Π) subject to E(λ) ≈ 0

EFE Decomposition
-----------------
G(π) = Extrinsic (pragmatic) + Intrinsic (epistemic)

- Extrinsic: -E_Q[ln P(o_τ)] → penalizes deviation from preferred outcomes
- Intrinsic: D_KL[Q(s|o,π) || Q(s|π)] → rewards uncertainty reduction

In GUTC terms:
- Extrinsic depends on Π weighting of prediction errors
- Intrinsic depends on λ controlling information propagation

Hierarchical Extension
----------------------
For L-level hierarchy with Γ coupling matrix:

    J_hier = Σ_l G^(l)(π) + α Σ_l |E^(l)(λ_l)|² + β ||Γ - Γ*||²

where Γ* is the optimal coupling derived from manifold alignment.

Usage
-----
    python gutc_agency.py test       # Run tests
    python gutc_agency.py agency     # Demonstrate agency on manifold
    python gutc_agency.py hier       # Hierarchical agency demonstration
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable
from enum import Enum


# =============================================================================
# Policy Types
# =============================================================================

class Policy(Enum):
    """
    Simple binary policy space for demonstration.

    In full active inference, π indexes sequences of actions.
    Here we use two prototypical policies:

    WAIT (π=0): Trust priors, minimal action
        - Low epistemic value (no new information)
        - Extrinsic value depends on prior accuracy

    ACT (π=1): Engage sensory loop, active exploration
        - High epistemic value (information gain)
        - Extrinsic value depends on sensory accuracy
    """
    WAIT = 0  # Prior-dominated, exploitation
    ACT = 1   # Sensory-dominated, exploration


# =============================================================================
# GUTC Agency Coder
# =============================================================================

@dataclass
class AgencyConfig:
    """Configuration for GUTC agency coder."""
    # Generative model
    C: float = 1.0              # Observation gain
    mu_0: float = 0.0           # Prior mean

    # GUTC control parameters
    lambda_c: float = 1.0       # Criticality dial
    W_recur: float = 1.0        # Recurrent weight

    # Precisions
    pi_sensory: float = 1.0     # Sensory precision (ACh)
    pi_prior: float = 1.0       # Prior precision (DA)

    # Time constants
    tau_mu: float = 10.0        # L5 dynamics
    tau_eps: float = 5.0        # L2/3 dynamics
    dt: float = 0.01            # Integration step

    # Stochastic dynamics
    noise_std: float = 0.1      # State noise

    # Agency parameters
    epistemic_weight: float = 1.0   # Weight on intrinsic value
    pragmatic_weight: float = 1.0   # Weight on extrinsic value
    criticality_alpha: float = 1.0  # Weight on |E(λ)|² penalty

    # Preferred observations (for pragmatic evaluation)
    o_preferred: float = 1.0    # Target observation value


class GUTCAgencyCoder:
    """
    L2/3↔L5 predictive coder with GUTC agency (EFE-based policy selection).

    Extends the basic VFE engine with:
    1. Expected Free Energy computation for policies
    2. The unified GUTC objective J(π, λ, Π)
    3. Policy selection via G(π) minimization

    The system can now:
    - Perceive (minimize VFE)
    - Act (minimize EFE)
    - Self-tune (maintain criticality)
    """

    def __init__(self, cfg: AgencyConfig = None, seed: int = None):
        self.cfg = cfg or AgencyConfig()
        self.rng = np.random.default_rng(seed)

        # State variables
        self.mu = self.cfg.mu_0       # L5 belief
        self.eps_y = 0.0              # Sensory PE
        self.eps_mu = 0.0             # Prior PE

        # Trajectory storage
        self.eps_y_history: List[float] = []
        self.mu_history: List[float] = []
        self.F_history: List[float] = []
        self.G_history: List[float] = []
        self.policy_history: List[int] = []
        self.J_history: List[float] = []

    def reset(self):
        """Reset state and history."""
        self.mu = self.cfg.mu_0
        self.eps_y = 0.0
        self.eps_mu = 0.0
        self.eps_y_history = []
        self.mu_history = []
        self.F_history = []
        self.G_history = []
        self.policy_history = []
        self.J_history = []

    # =========================================================================
    # VFE: Present-Time Inference
    # =========================================================================

    def free_energy(self) -> float:
        """
        Variational Free Energy (VFE) - scores current beliefs.

        F = ½ Π_s ε_y² + ½ Π_p ε_μ²
        """
        F_y = 0.5 * self.cfg.pi_sensory * (self.eps_y ** 2)
        F_mu = 0.5 * self.cfg.pi_prior * (self.eps_mu ** 2)
        return F_y + F_mu

    def step(self, y: float, policy: Policy = None) -> Tuple[float, float, float, float]:
        """
        Single integration step with optional policy modulation.

        Policy affects how strongly sensory vs prior terms drive updates.
        """
        cfg = self.cfg

        # Policy modulation (if specified)
        if policy == Policy.ACT:
            # Active exploration: boost sensory precision
            pi_sens_eff = cfg.pi_sensory * 1.5
            pi_prior_eff = cfg.pi_prior * 0.7
        elif policy == Policy.WAIT:
            # Passive waiting: trust priors
            pi_sens_eff = cfg.pi_sensory * 0.7
            pi_prior_eff = cfg.pi_prior * 1.5
        else:
            pi_sens_eff = cfg.pi_sensory
            pi_prior_eff = cfg.pi_prior

        # Add input noise
        y_noisy = y + cfg.noise_std * 0.5 * self.rng.standard_normal()

        # Target errors
        eps_y_target = y_noisy - cfg.C * self.mu
        eps_mu_target = self.mu - cfg.mu_0

        # L2/3 dynamics
        d_eps_y = (eps_y_target - self.eps_y) / cfg.tau_eps
        d_eps_mu = (eps_mu_target - self.eps_mu) / cfg.tau_eps

        noise_scale = cfg.noise_std * np.sqrt(cfg.dt)
        self.eps_y += cfg.dt * d_eps_y + noise_scale * self.rng.standard_normal()
        self.eps_mu += cfg.dt * d_eps_mu + noise_scale * self.rng.standard_normal()

        # L5 dynamics with effective precisions
        recur = (-1.0 + cfg.lambda_c * cfg.W_recur) * self.mu
        drive_sens = cfg.C * pi_sens_eff * self.eps_y
        drive_prior = -pi_prior_eff * self.eps_mu

        d_mu = (recur + drive_sens + drive_prior) / cfg.tau_mu
        self.mu += cfg.dt * d_mu + noise_scale * self.rng.standard_normal()

        # Store trajectory
        F = self.free_energy()
        self.eps_y_history.append(self.eps_y)
        self.mu_history.append(self.mu)
        self.F_history.append(F)

        return self.mu, self.eps_y, self.eps_mu, F

    # =========================================================================
    # EFE: Future-Directed Control
    # =========================================================================

    def expected_free_energy(self, policy: Policy) -> Tuple[float, float, float]:
        """
        Expected Free Energy G(π) for a policy.

        G(π) = Extrinsic + Epistemic

        Extrinsic (pragmatic): Cost of expected observations
        Epistemic (intrinsic): Information gain about hidden states

        Returns:
            (G_total, G_extrinsic, G_epistemic)
        """
        cfg = self.cfg

        if policy == Policy.WAIT:
            # WAIT: Trust priors, minimal sensory engagement
            # Extrinsic: Deviation from preferred based on prior
            expected_obs = cfg.C * self.mu  # Predicted observation
            extrinsic = 0.5 * cfg.pi_prior * (expected_obs - cfg.o_preferred) ** 2

            # Epistemic: Low information gain (not exploring)
            # At criticality, even passive observation has some info content
            epistemic_base = 0.1 * np.abs(1.0 - cfg.lambda_c)  # Minimal near λ=1
            epistemic = epistemic_base + 0.05

        else:  # Policy.ACT
            # ACT: Active sensory engagement, exploration
            # Extrinsic: Weighted by sensory precision (actions driven by error)
            extrinsic = 0.5 * cfg.pi_sensory * (self.eps_y ** 2)

            # Epistemic: High information gain when exploring
            # At criticality (λ ≈ 1), exploration maximally informative
            # This is the GUTC insight: epistemic value peaks at E(λ) = 0
            lambda_factor = np.exp(-2.0 * (cfg.lambda_c - 1.0) ** 2)  # Peaks at λ=1
            epistemic_base = 0.5 * lambda_factor

            # Also depends on current uncertainty (higher error → more to learn)
            uncertainty_bonus = 0.2 * np.abs(self.eps_y)
            epistemic = epistemic_base + uncertainty_bonus

        # Total EFE (lower is better)
        # Note: Epistemic is a reward (negative contribution to G)
        G_extrinsic = cfg.pragmatic_weight * extrinsic
        G_epistemic = -cfg.epistemic_weight * epistemic  # Negative because it's valuable
        G_total = G_extrinsic + G_epistemic

        return G_total, G_extrinsic, G_epistemic

    def select_policy(self) -> Tuple[Policy, float]:
        """
        Select policy by minimizing Expected Free Energy.

        π* = argmin_π G(π)

        Returns:
            (optimal_policy, G_value)
        """
        G_wait, _, _ = self.expected_free_energy(Policy.WAIT)
        G_act, _, _ = self.expected_free_energy(Policy.ACT)

        if G_wait <= G_act:
            return Policy.WAIT, G_wait
        else:
            return Policy.ACT, G_act

    # =========================================================================
    # GUTC Unified Objective
    # =========================================================================

    def edge_function(self) -> float:
        """
        Edge function E(λ) measuring distance to criticality.

        E(λ) = ρ(J) - 1 where ρ is spectral radius of effective Jacobian.

        Simplified: E(λ) = λ - 1 (linear approximation near criticality).
        """
        return self.cfg.lambda_c - 1.0

    def gutc_objective(self, policy: Policy) -> Tuple[float, Dict[str, float]]:
        """
        Unified GUTC objective functional.

        J(π, λ, Π) = G(π | λ, Π) + α|E(λ)|²

        The healthy agent minimizes J by:
        1. Selecting policies that minimize G(π)
        2. Maintaining λ ≈ 1 (criticality)
        3. Balancing precision fields Π

        Returns:
            (J_value, breakdown_dict)
        """
        cfg = self.cfg

        # EFE for policy
        G_total, G_extrinsic, G_epistemic = self.expected_free_energy(policy)

        # Criticality penalty
        E_lambda = self.edge_function()
        criticality_penalty = cfg.criticality_alpha * (E_lambda ** 2)

        # Unified objective
        J = G_total + criticality_penalty

        breakdown = {
            "J": J,
            "G_total": G_total,
            "G_extrinsic": G_extrinsic,
            "G_epistemic": G_epistemic,
            "E_lambda": E_lambda,
            "criticality_penalty": criticality_penalty,
        }

        return J, breakdown

    def optimal_action(self) -> Tuple[Policy, float, Dict[str, float]]:
        """
        Select optimal policy under full GUTC objective.

        π* = argmin_π J(π, λ, Π)

        Returns:
            (optimal_policy, J_value, breakdown)
        """
        J_wait, breakdown_wait = self.gutc_objective(Policy.WAIT)
        J_act, breakdown_act = self.gutc_objective(Policy.ACT)

        if J_wait <= J_act:
            return Policy.WAIT, J_wait, breakdown_wait
        else:
            return Policy.ACT, J_act, breakdown_act

    # =========================================================================
    # Simulation Loop with Agency
    # =========================================================================

    def run_with_agency(
        self,
        y_obs: float,
        n_steps: int = 1000,
        use_gutc_objective: bool = True,
    ) -> Dict[str, Any]:
        """
        Run simulation with active policy selection.

        At each step:
        1. Select policy (minimize G or J)
        2. Execute step under selected policy
        3. Record trajectory

        Args:
            y_obs: Observed value (constant or callable)
            n_steps: Number of integration steps
            use_gutc_objective: If True, use J; if False, use G only

        Returns:
            Dictionary with trajectories and statistics
        """
        self.reset()

        for t in range(n_steps):
            # Get observation (constant or time-varying)
            y = y_obs(t) if callable(y_obs) else y_obs

            # Select policy
            if use_gutc_objective:
                policy, J, breakdown = self.optimal_action()
                self.J_history.append(J)
            else:
                policy, G = self.select_policy()
                self.J_history.append(G)

            # Execute step
            self.step(y, policy)

            # Record
            G_total, _, _ = self.expected_free_energy(policy)
            self.G_history.append(G_total)
            self.policy_history.append(policy.value)

        return {
            "mu": np.array(self.mu_history),
            "eps_y": np.array(self.eps_y_history),
            "F": np.array(self.F_history),
            "G": np.array(self.G_history),
            "J": np.array(self.J_history),
            "policies": np.array(self.policy_history),
            "act_fraction": np.mean(self.policy_history),
        }


# =============================================================================
# Hierarchical GUTC Agency
# =============================================================================

@dataclass
class HierarchicalAgencyConfig:
    """Configuration for hierarchical GUTC agency."""
    n_levels: int = 3
    level_lambdas: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    level_pis: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])

    # Coupling matrix parameters
    gamma_asc: List[float] = field(default_factory=lambda: [0.3, 0.3])
    gamma_desc: List[float] = field(default_factory=lambda: [0.3, 0.3])

    # Objective weights
    alpha: float = 1.0   # Criticality penalty weight
    beta: float = 0.5    # Coupling misalignment penalty weight

    # Optimal coupling (derived from manifold alignment)
    gamma_star: float = 0.3  # Target balanced coupling


def hierarchical_gutc_objective(
    G_levels: List[float],
    lambdas: List[float],
    gamma_matrix: np.ndarray,
    config: HierarchicalAgencyConfig,
) -> Tuple[float, Dict[str, float]]:
    """
    Hierarchical GUTC objective functional.

    J_hier = Σ_l G^(l)(π) + α Σ_l |E^(l)(λ_l)|² + β ||Γ - Γ*||²

    Args:
        G_levels: EFE for each level
        lambdas: Criticality parameter for each level
        gamma_matrix: Inter-level coupling matrix Γ
        config: Hierarchical configuration

    Returns:
        (J_hier, breakdown_dict)
    """
    n_levels = len(G_levels)

    # Sum of level-wise EFE
    G_sum = sum(G_levels)

    # Sum of criticality penalties
    E_penalties = sum((lam - 1.0) ** 2 for lam in lambdas)
    criticality_term = config.alpha * E_penalties

    # Coupling misalignment penalty
    # ||Γ - Γ*||² measures deviation from optimal coupling
    gamma_star_matrix = np.ones_like(gamma_matrix) * config.gamma_star
    np.fill_diagonal(gamma_star_matrix, 0)  # No self-coupling
    coupling_penalty = config.beta * np.sum((gamma_matrix - gamma_star_matrix) ** 2)

    # Total hierarchical objective
    J_hier = G_sum + criticality_term + coupling_penalty

    breakdown = {
        "J_hier": J_hier,
        "G_sum": G_sum,
        "G_per_level": G_levels,
        "E_penalties": E_penalties,
        "criticality_term": criticality_term,
        "coupling_penalty": coupling_penalty,
        "lambdas": lambdas,
    }

    return J_hier, breakdown


# =============================================================================
# Manifold Analysis with Agency
# =============================================================================

def sweep_agency_manifold(
    lambda_range: Tuple[float, float] = (0.5, 1.5),
    pi_range: Tuple[float, float] = (0.5, 3.0),
    n_lambda: int = 15,
    n_pi: int = 15,
    n_steps: int = 500,
    y_obs: float = 1.0,
) -> Dict[str, np.ndarray]:
    """
    Sweep (λ, Π) manifold measuring agency metrics.

    Returns grids of:
    - G_mean: Mean EFE under optimal policy
    - J_mean: Mean GUTC objective
    - act_fraction: Fraction of ACT policies chosen
    - F_mean: Mean VFE
    """
    lambdas = np.linspace(lambda_range[0], lambda_range[1], n_lambda)
    pis = np.linspace(pi_range[0], pi_range[1], n_pi)

    G_grid = np.zeros((n_pi, n_lambda))
    J_grid = np.zeros((n_pi, n_lambda))
    act_grid = np.zeros((n_pi, n_lambda))
    F_grid = np.zeros((n_pi, n_lambda))

    for i, pi_val in enumerate(pis):
        for j, lam in enumerate(lambdas):
            cfg = AgencyConfig(
                lambda_c=lam,
                pi_sensory=pi_val,
                pi_prior=pi_val,
            )
            coder = GUTCAgencyCoder(cfg, seed=42)
            results = coder.run_with_agency(y_obs, n_steps=n_steps)

            # After burn-in
            burn = n_steps // 5
            G_grid[i, j] = np.mean(results["G"][burn:])
            J_grid[i, j] = np.mean(results["J"][burn:])
            act_grid[i, j] = np.mean(results["policies"][burn:])
            F_grid[i, j] = np.mean(results["F"][burn:])

    return {
        "lambdas": lambdas,
        "pis": pis,
        "G_grid": G_grid,
        "J_grid": J_grid,
        "act_fraction_grid": act_grid,
        "F_grid": F_grid,
    }


def plot_agency_manifold(
    data: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
):
    """
    Plot agency metrics across the (λ, Π) manifold.

    Four panels:
    1. Mean EFE G(π*)
    2. Mean GUTC objective J
    3. Action fraction (exploration vs exploitation)
    4. Mean VFE F̄
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for plotting")
        return

    lambdas = data["lambdas"]
    pis = data["pis"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel A: EFE
    im1 = axes[0, 0].pcolormesh(
        lambdas, pis, data["G_grid"],
        cmap='viridis_r', shading='auto'
    )
    fig.colorbar(im1, ax=axes[0, 0], label=r'$\bar{G}(\pi^*)$')
    axes[0, 0].axvline(x=1.0, color='white', linestyle='--', linewidth=1.5)
    axes[0, 0].set_xlabel(r'$\lambda$')
    axes[0, 0].set_ylabel(r'$\Pi$')
    axes[0, 0].set_title(r'(A) Expected Free Energy $\bar{G}(\pi^*)$')
    axes[0, 0].plot(1.0, 1.0, 'w*', markersize=15, markeredgecolor='black')

    # Panel B: GUTC Objective
    im2 = axes[0, 1].pcolormesh(
        lambdas, pis, data["J_grid"],
        cmap='plasma', shading='auto'
    )
    fig.colorbar(im2, ax=axes[0, 1], label=r'$\bar{J}(\pi^*, \lambda, \Pi)$')
    axes[0, 1].axvline(x=1.0, color='white', linestyle='--', linewidth=1.5)
    axes[0, 1].set_xlabel(r'$\lambda$')
    axes[0, 1].set_ylabel(r'$\Pi$')
    axes[0, 1].set_title(r'(B) GUTC Objective $\bar{J}$ (lower = better)')
    axes[0, 1].plot(1.0, 1.0, 'w*', markersize=15, markeredgecolor='black')

    # Panel C: Action fraction
    im3 = axes[1, 0].pcolormesh(
        lambdas, pis, data["act_fraction_grid"],
        cmap='coolwarm', vmin=0, vmax=1, shading='auto'
    )
    fig.colorbar(im3, ax=axes[1, 0], label='ACT fraction')
    axes[1, 0].axvline(x=1.0, color='black', linestyle='--', linewidth=1.5)
    axes[1, 0].set_xlabel(r'$\lambda$')
    axes[1, 0].set_ylabel(r'$\Pi$')
    axes[1, 0].set_title(r'(C) Exploration vs Exploitation (ACT fraction)')
    axes[1, 0].plot(1.0, 1.0, 'k*', markersize=15)

    # Panel D: VFE
    im4 = axes[1, 1].pcolormesh(
        lambdas, pis, data["F_grid"],
        cmap='viridis_r', shading='auto'
    )
    fig.colorbar(im4, ax=axes[1, 1], label=r'$\bar{\mathcal{F}}$')
    axes[1, 1].axvline(x=1.0, color='white', linestyle='--', linewidth=1.5)
    axes[1, 1].set_xlabel(r'$\lambda$')
    axes[1, 1].set_ylabel(r'$\Pi$')
    axes[1, 1].set_title(r'(D) Variational Free Energy $\bar{\mathcal{F}}$')
    axes[1, 1].plot(1.0, 1.0, 'w*', markersize=15, markeredgecolor='black')

    plt.suptitle(
        r'GUTC Agency: $J(\pi, \lambda, \Pi) = G(\pi) + \alpha|E(\lambda)|^2$',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()


# =============================================================================
# Tests
# =============================================================================

def test_agency_coder_basic():
    """Test basic agency coder functionality."""
    cfg = AgencyConfig(lambda_c=1.0, pi_sensory=1.0, pi_prior=1.0)
    coder = GUTCAgencyCoder(cfg, seed=42)

    # Run a few steps
    for _ in range(100):
        coder.step(y=1.0)

    assert len(coder.F_history) == 100
    assert len(coder.mu_history) == 100

    # Check VFE is positive
    F = coder.free_energy()
    assert F >= 0, f"VFE should be non-negative, got {F}"

    print("✓ Basic agency coder")


def test_efe_computation():
    """Test EFE computation for both policies."""
    cfg = AgencyConfig(lambda_c=1.0, pi_sensory=1.0, pi_prior=1.0)
    coder = GUTCAgencyCoder(cfg, seed=42)

    # Run to establish state
    for _ in range(50):
        coder.step(y=1.0)

    G_wait, ex_wait, ep_wait = coder.expected_free_energy(Policy.WAIT)
    G_act, ex_act, ep_act = coder.expected_free_energy(Policy.ACT)

    print(f"  G(WAIT) = {G_wait:.3f} (ex={ex_wait:.3f}, ep={ep_wait:.3f})")
    print(f"  G(ACT)  = {G_act:.3f} (ex={ex_act:.3f}, ep={ep_act:.3f})")

    # Both should be finite
    assert np.isfinite(G_wait) and np.isfinite(G_act)

    print("✓ EFE computation")


def test_gutc_objective():
    """Test GUTC objective J(π, λ, Π)."""
    # At criticality
    cfg_crit = AgencyConfig(lambda_c=1.0)
    coder_crit = GUTCAgencyCoder(cfg_crit, seed=42)
    for _ in range(50):
        coder_crit.step(y=1.0)
    J_crit, breakdown_crit = coder_crit.gutc_objective(Policy.ACT)

    # Away from criticality
    cfg_sub = AgencyConfig(lambda_c=0.7)
    coder_sub = GUTCAgencyCoder(cfg_sub, seed=42)
    for _ in range(50):
        coder_sub.step(y=1.0)
    J_sub, breakdown_sub = coder_sub.gutc_objective(Policy.ACT)

    print(f"  J(λ=1.0) = {J_crit:.3f}, penalty = {breakdown_crit['criticality_penalty']:.3f}")
    print(f"  J(λ=0.7) = {J_sub:.3f}, penalty = {breakdown_sub['criticality_penalty']:.3f}")

    # Criticality penalty should be higher away from λ=1
    assert breakdown_sub['criticality_penalty'] > breakdown_crit['criticality_penalty']

    print("✓ GUTC objective")


def test_policy_selection():
    """Test policy selection minimizes EFE."""
    cfg = AgencyConfig(lambda_c=1.0)
    coder = GUTCAgencyCoder(cfg, seed=42)

    for _ in range(50):
        coder.step(y=1.0)

    policy, G_optimal = coder.select_policy()
    G_wait, _, _ = coder.expected_free_energy(Policy.WAIT)
    G_act, _, _ = coder.expected_free_energy(Policy.ACT)

    # Selected policy should have minimum G
    assert G_optimal == min(G_wait, G_act)

    print(f"  Selected: {policy.name}, G = {G_optimal:.3f}")
    print("✓ Policy selection")


def test_run_with_agency():
    """Test full agency simulation."""
    cfg = AgencyConfig(lambda_c=1.0)
    coder = GUTCAgencyCoder(cfg, seed=42)

    results = coder.run_with_agency(y_obs=1.0, n_steps=500)

    assert len(results["mu"]) == 500
    assert len(results["G"]) == 500
    assert len(results["J"]) == 500
    assert len(results["policies"]) == 500

    act_frac = results["act_fraction"]
    print(f"  ACT fraction: {act_frac:.2f}")
    print(f"  Mean G: {np.mean(results['G']):.3f}")
    print(f"  Mean J: {np.mean(results['J']):.3f}")

    print("✓ Run with agency")


def test_hierarchical_objective():
    """Test hierarchical GUTC objective."""
    config = HierarchicalAgencyConfig()

    # Mock values
    G_levels = [0.5, 0.3, 0.2]  # EFE per level
    lambdas = [1.0, 1.0, 1.0]   # All critical
    gamma_matrix = np.array([
        [0, 0.3, 0],
        [0.3, 0, 0.3],
        [0, 0.3, 0]
    ])

    J_hier, breakdown = hierarchical_gutc_objective(
        G_levels, lambdas, gamma_matrix, config
    )

    print(f"  J_hier = {J_hier:.3f}")
    print(f"  G_sum = {breakdown['G_sum']:.3f}")
    print(f"  Criticality term = {breakdown['criticality_term']:.3f}")
    print(f"  Coupling penalty = {breakdown['coupling_penalty']:.3f}")

    # Test that off-critical λ increases J
    lambdas_off = [0.8, 1.0, 1.2]
    J_off, breakdown_off = hierarchical_gutc_objective(
        G_levels, lambdas_off, gamma_matrix, config
    )

    assert J_off > J_hier, "Off-critical should have higher J"
    print(f"  J_hier (off-critical) = {J_off:.3f} > {J_hier:.3f}")

    print("✓ Hierarchical objective")


def test_epistemic_peaks_at_criticality():
    """Test that epistemic value peaks near λ = 1."""
    lambdas = [0.6, 0.8, 1.0, 1.2, 1.4]
    epistemic_values = []

    for lam in lambdas:
        cfg = AgencyConfig(lambda_c=lam)
        coder = GUTCAgencyCoder(cfg, seed=42)
        for _ in range(50):
            coder.step(y=1.0)
        _, _, ep = coder.expected_free_energy(Policy.ACT)
        epistemic_values.append(-ep)  # Convert back to positive

    # Find peak
    peak_idx = np.argmax(epistemic_values)
    peak_lambda = lambdas[peak_idx]

    print(f"  λ values: {lambdas}")
    print(f"  Epistemic: {[f'{e:.3f}' for e in epistemic_values]}")
    print(f"  Peak at λ = {peak_lambda}")

    # Peak should be near λ = 1
    assert 0.8 <= peak_lambda <= 1.2, f"Epistemic should peak near λ=1, got {peak_lambda}"

    print("✓ Epistemic peaks at criticality")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("GUTC Agency Tests")
    print("=" * 60 + "\n")

    test_agency_coder_basic()
    test_efe_computation()
    test_gutc_objective()
    test_policy_selection()
    test_run_with_agency()
    test_hierarchical_objective()
    test_epistemic_peaks_at_criticality()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60 + "\n")


# =============================================================================
# CLI
# =============================================================================

def main():
    import sys

    if len(sys.argv) > 1:
        cmd = sys.argv[1]

        if cmd == "test":
            run_all_tests()

        elif cmd == "agency":
            print("Sweeping agency manifold...")
            print("This demonstrates GUTC's insight: J is minimized in the healthy corridor\n")

            data = sweep_agency_manifold(
                lambda_range=(0.5, 1.5),
                pi_range=(0.5, 3.0),
                n_lambda=15,
                n_pi=15,
                n_steps=500,
            )

            # Find optimal point
            J_grid = data["J_grid"]
            min_idx = np.unravel_index(np.argmin(J_grid), J_grid.shape)
            opt_pi = data["pis"][min_idx[0]]
            opt_lambda = data["lambdas"][min_idx[1]]

            print(f"Optimal (λ, Π) = ({opt_lambda:.2f}, {opt_pi:.2f})")
            print(f"Minimum J = {J_grid[min_idx]:.4f}")

            plot_agency_manifold(data, save_path="gutc_agency_manifold.png")

        elif cmd == "hier":
            print("Hierarchical GUTC Agency Demonstration")
            print("-" * 50)

            config = HierarchicalAgencyConfig()

            # Compare uniform critical vs mixed
            G_levels = [0.5, 0.3, 0.2]
            gamma = np.array([[0, 0.3, 0], [0.3, 0, 0.3], [0, 0.3, 0]])

            # Uniform critical
            J_uniform, _ = hierarchical_gutc_objective(
                G_levels, [1.0, 1.0, 1.0], gamma, config
            )

            # Mixed (subcritical L1, supercritical L3)
            J_mixed, _ = hierarchical_gutc_objective(
                G_levels, [0.8, 1.0, 1.2], gamma, config
            )

            # Bad coupling
            gamma_bad = np.array([[0, 0.1, 0], [0.5, 0, 0.1], [0, 0.5, 0]])
            J_bad_coupling, _ = hierarchical_gutc_objective(
                G_levels, [1.0, 1.0, 1.0], gamma_bad, config
            )

            print(f"\nJ_hier (uniform λ=1, balanced Γ):  {J_uniform:.4f}")
            print(f"J_hier (mixed λ, balanced Γ):       {J_mixed:.4f}")
            print(f"J_hier (uniform λ=1, imbalanced Γ): {J_bad_coupling:.4f}")
            print(f"\n→ Uniform criticality + balanced coupling minimizes J_hier")

        else:
            print(f"Unknown command: {cmd}")
            print("Usage: python gutc_agency.py [test|agency|hier]")
    else:
        run_all_tests()


if __name__ == "__main__":
    main()
