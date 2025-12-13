#!/usr/bin/env python3
"""
Hierarchical Gaussian Filter (HGF) Implementation

A computational model for learning in volatile environments that implements
hierarchical precision estimation - the core mechanism of predictive processing.

This connects to the Brain Remodulator by providing:
- Concrete implementation of precision-weighted belief updating
- Multi-level volatility estimation (meta-uncertainty)
- Parameters that map to clinical precision aberrations
- Foundation for D_low/D_high estimation from behavioral data

Based on Mathys et al. (2011, 2014) and extended to 5 levels for
modeling "epistemic depth" as in the Beautiful Loop theory.

Usage:
    python -m ara.neuro.remodulator.hgf
    python -m ara.neuro.remodulator.hgf --levels 3 --trials 200
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


def softplus(x: float) -> float:
    """Softplus function: log(1 + exp(x))."""
    if x > 20:
        return x
    return math.log1p(math.exp(x))


# =============================================================================
# HGF Parameters and State
# =============================================================================

@dataclass
class HGFParameters:
    """
    Parameters for the Hierarchical Gaussian Filter.

    Coupling parameters (kappa):
        Control how higher-level volatility affects lower-level learning rate.
        Higher kappa = stronger coupling = more adaptive learning rate.

    Base volatility (omega):
        Default log-volatility at each level, independent of higher levels.
        Higher omega = more volatile beliefs = faster learning.

    Top-level volatility (theta):
        Fixed volatility at the highest level.
        Represents fundamental uncertainty about the world's structure.
    """
    # Coupling parameters
    kappa1: float = 1.0  # Level 2 ↔ Level 3 coupling
    kappa2: float = 1.0  # Level 3 ↔ Level 4 coupling
    kappa3: float = 1.0  # Level 4 ↔ Level 5 coupling

    # Base log-volatility
    omega1: float = 2.0  # Base volatility for Level 2
    omega2: float = 2.0  # Base volatility for Level 3
    omega3: float = 2.0  # Base volatility for Level 4

    # Top-level volatility
    theta: float = 4.0   # Fixed volatility for Level 5

    def to_dict(self) -> Dict[str, float]:
        return {
            "kappa1": self.kappa1, "kappa2": self.kappa2, "kappa3": self.kappa3,
            "omega1": self.omega1, "omega2": self.omega2, "omega3": self.omega3,
            "theta": self.theta,
        }


@dataclass
class HGFState:
    """
    Current state of HGF beliefs at all levels.

    mu: Mean (best estimate)
    pi: Precision (inverse variance, confidence)
    sigma: Standard deviation (uncertainty)
    """
    # Level 2: Logit-probability beliefs
    mu2: float = 0.0
    pi2: float = 1.0

    # Level 3: Volatility beliefs
    mu3: float = 0.0
    pi3: float = 1.0

    # Level 4: Higher volatility beliefs
    mu4: float = 0.0
    pi4: float = 1.0

    # Level 5: Highest volatility beliefs
    mu5: float = 0.0
    pi5: float = 1.0

    @property
    def probability(self) -> float:
        """Inferred probability (Level 1) = sigmoid(mu2)."""
        return float(sigmoid(np.array([self.mu2]))[0])

    @property
    def sigma2(self) -> float:
        """Uncertainty at Level 2."""
        return 1 / math.sqrt(max(self.pi2, 1e-10))

    @property
    def sigma3(self) -> float:
        """Uncertainty at Level 3."""
        return 1 / math.sqrt(max(self.pi3, 1e-10))

    @property
    def sigma4(self) -> float:
        """Uncertainty at Level 4."""
        return 1 / math.sqrt(max(self.pi4, 1e-10))

    @property
    def sigma5(self) -> float:
        """Uncertainty at Level 5."""
        return 1 / math.sqrt(max(self.pi5, 1e-10))

    @property
    def epistemic_depth(self) -> float:
        """
        Epistemic depth: How much the higher levels are resolved.

        Higher values = more confident about meta-uncertainty structure.
        Maps to the "Beautiful Loop" concept of recursive self-knowing.
        """
        # Depth increases as higher-level uncertainty decreases
        depth = (1 / (1 + self.sigma3)) + (1 / (1 + self.sigma4)) + (1 / (1 + self.sigma5))
        return depth / 3  # Normalize to [0, 1]

    def to_dict(self) -> Dict[str, float]:
        return {
            "mu2": self.mu2, "pi2": self.pi2, "sigma2": self.sigma2,
            "mu3": self.mu3, "pi3": self.pi3, "sigma3": self.sigma3,
            "mu4": self.mu4, "pi4": self.pi4, "sigma4": self.sigma4,
            "mu5": self.mu5, "pi5": self.pi5, "sigma5": self.sigma5,
            "probability": self.probability,
            "epistemic_depth": self.epistemic_depth,
        }


# =============================================================================
# Binary HGF (5-Level)
# =============================================================================

class BinaryHGF:
    """
    5-Level Binary Hierarchical Gaussian Filter.

    Hierarchy:
        Level 1: Inferred probability (sigmoid of Level 2)
        Level 2: Logit-probability beliefs
        Level 3: Volatility of Level 2 (learning rate)
        Level 4: Volatility of Level 3 (meta-learning rate)
        Level 5: Highest volatility (epistemic depth)

    Each level:
        - Receives prediction errors from the level below
        - Sends volatility forecasts to the level below
        - Updates beliefs via precision-weighted error minimization

    This implements the core predictive processing computation:
        posterior = prior + (precision_error / precision_total) × error
    """

    def __init__(
        self,
        params: Optional[HGFParameters] = None,
        initial_state: Optional[HGFState] = None,
        n_levels: int = 5,
    ):
        self.params = params or HGFParameters()
        self.state = initial_state or HGFState()
        self.n_levels = n_levels

        # History tracking
        self.history: List[Dict] = []
        self.mu_history: Dict[str, List[float]] = {
            "mu1": [], "mu2": [], "mu3": [], "mu4": [], "mu5": []
        }
        self.pi_history: Dict[str, List[float]] = {
            "pi2": [], "pi3": [], "pi4": [], "pi5": []
        }
        self.prediction_errors: Dict[str, List[float]] = {
            "delta1": [], "delta2": [], "delta3": [], "delta4": []
        }

    def update(self, observation: int) -> HGFState:
        """
        Update beliefs given a binary observation (0 or 1).

        Implements variational Bayesian inference, minimizing free energy
        at each level of the hierarchy.
        """
        u = float(observation)
        p = self.params
        s = self.state

        # === Level 2: Logit-probability beliefs ===
        # Prediction
        hat_mu2 = s.mu2
        sigma_hat2 = math.exp(p.kappa1 * s.mu3 + p.omega1)
        pi_hat2 = 1 / sigma_hat2

        # Level 1 prediction (probability)
        s_hat = sigmoid(np.array([hat_mu2]))[0]

        # Prediction error at Level 1
        delta1 = u - s_hat

        # Precision of prediction error (likelihood precision)
        v1 = s_hat * (1 - s_hat) + 1e-12
        pi1 = 1 / v1

        # Posterior at Level 2
        pi2_post = pi_hat2 + pi1
        mu2_post = hat_mu2 + (pi1 / pi2_post) * delta1

        # === Level 3: Volatility beliefs ===
        hat_mu3 = s.mu3
        sigma_hat3 = math.exp(p.kappa2 * s.mu4 + p.omega2)
        pi_hat3 = 1 / sigma_hat3

        # Prediction error at Level 2
        delta2 = mu2_post - hat_mu2

        # Posterior at Level 3
        pi3_post = pi_hat3 + pi2_post
        mu3_post = hat_mu3 + (pi2_post / pi3_post) * delta2

        # === Level 4: Higher volatility beliefs ===
        hat_mu4 = s.mu4
        sigma_hat4 = math.exp(p.kappa3 * s.mu5 + p.omega3)
        pi_hat4 = 1 / sigma_hat4

        # Prediction error at Level 3
        delta3 = mu3_post - hat_mu3

        # Posterior at Level 4
        pi4_post = pi_hat4 + pi3_post
        mu4_post = hat_mu4 + (pi3_post / pi4_post) * delta3

        # === Level 5: Highest volatility beliefs ===
        hat_mu5 = s.mu5
        sigma_hat5 = math.exp(p.theta)
        pi_hat5 = 1 / sigma_hat5

        # Prediction error at Level 4
        delta4 = mu4_post - hat_mu4

        # Posterior at Level 5
        pi5_post = pi_hat5 + pi4_post
        mu5_post = hat_mu5 + (pi4_post / pi5_post) * delta4

        # === Update state ===
        self.state = HGFState(
            mu2=mu2_post, pi2=pi2_post,
            mu3=mu3_post, pi3=pi3_post,
            mu4=mu4_post, pi4=pi4_post,
            mu5=mu5_post, pi5=pi5_post,
        )

        # === Record history ===
        self.mu_history["mu1"].append(self.state.probability)
        self.mu_history["mu2"].append(mu2_post)
        self.mu_history["mu3"].append(mu3_post)
        self.mu_history["mu4"].append(mu4_post)
        self.mu_history["mu5"].append(mu5_post)

        self.pi_history["pi2"].append(pi2_post)
        self.pi_history["pi3"].append(pi3_post)
        self.pi_history["pi4"].append(pi4_post)
        self.pi_history["pi5"].append(pi5_post)

        self.prediction_errors["delta1"].append(delta1)
        self.prediction_errors["delta2"].append(delta2)
        self.prediction_errors["delta3"].append(delta3)
        self.prediction_errors["delta4"].append(delta4)

        self.history.append({
            "observation": u,
            "state": self.state.to_dict(),
            "errors": {"delta1": delta1, "delta2": delta2, "delta3": delta3, "delta4": delta4},
        })

        return self.state

    def run(self, observations: np.ndarray) -> List[HGFState]:
        """Run HGF on a sequence of observations."""
        states = []
        for obs in observations:
            state = self.update(int(obs))
            states.append(state)
        return states

    def reset(self):
        """Reset to initial state."""
        self.state = HGFState()
        self.history = []
        self.mu_history = {"mu1": [], "mu2": [], "mu3": [], "mu4": [], "mu5": []}
        self.pi_history = {"pi2": [], "pi3": [], "pi4": [], "pi5": []}
        self.prediction_errors = {"delta1": [], "delta2": [], "delta3": [], "delta4": []}


# =============================================================================
# Clinical Parameter Profiles
# =============================================================================

def get_clinical_profile(condition: str) -> HGFParameters:
    """
    Get HGF parameters that model different clinical conditions.

    These are hypothetical mappings based on computational psychiatry theory.
    """
    profiles = {
        # Healthy: Balanced coupling and volatility
        "healthy": HGFParameters(
            kappa1=1.0, kappa2=1.0, kappa3=1.0,
            omega1=2.0, omega2=2.0, omega3=2.0,
            theta=4.0,
        ),

        # Schizophrenia: Reduced learning from sensory evidence
        # Low kappa1 = beliefs don't update well from Level 1 errors
        # High omega at higher levels = unstable meta-beliefs
        "schizophrenia": HGFParameters(
            kappa1=0.3,  # Weak coupling to sensory errors
            kappa2=1.5,  # Overly sensitive to volatility changes
            kappa3=1.0,
            omega1=3.0,  # High baseline volatility (unstable beliefs)
            omega2=3.5,
            omega3=3.0,
            theta=5.0,
        ),

        # ASD: Over-learning from sensory, under-using priors
        # High kappa1 = strong sensory updating
        # Low omega at higher levels = rigid meta-beliefs
        "asd": HGFParameters(
            kappa1=2.0,  # Strong coupling to sensory
            kappa2=0.5,  # Weak higher-level adaptation
            kappa3=0.3,
            omega1=1.0,  # Low baseline volatility (rigid beliefs)
            omega2=1.0,
            omega3=1.0,
            theta=2.0,  # Very stable top-level
        ),

        # Anxiety: High precision on threat-related changes
        # High kappa = sensitive to all changes
        # High omega = everything feels volatile
        "anxiety": HGFParameters(
            kappa1=1.5,
            kappa2=1.5,
            kappa3=1.2,
            omega1=3.0,  # High volatility expectation
            omega2=3.0,
            omega3=2.5,
            theta=4.5,
        ),

        # Depression: Low learning, rigid negative beliefs
        # Low kappa = beliefs don't update from positive evidence
        "depression": HGFParameters(
            kappa1=0.5,  # Weak sensory updating (anhedonia)
            kappa2=0.5,
            kappa3=0.5,
            omega1=1.5,  # Low volatility (rigid beliefs)
            omega2=1.5,
            omega3=1.5,
            theta=3.0,
        ),
    }

    return profiles.get(condition, profiles["healthy"])


# =============================================================================
# Connection to Brain Remodulator
# =============================================================================

def estimate_D_from_hgf(hgf: BinaryHGF) -> Dict[str, float]:
    """
    Estimate Delusion Index metrics from HGF state.

    D_low: Based on higher-level (slow) precision ratios
    D_high: Based on lower-level (fast) precision ratios
    """
    s = hgf.state

    # D_low: Ratio of prior precision to lower-level precision
    # High D_low = rigid higher-level beliefs
    D_low = (s.pi3 + s.pi4 + s.pi5) / (3 * max(s.pi2, 0.01))

    # D_high: Ratio of sensory-level precision to volatility precision
    # High D_high = sensory-dominated
    D_high = s.pi2 / max(s.pi3, 0.01)

    # Delta H: Hierarchical discrepancy
    delta_H = abs(D_low - D_high)

    return {
        "D_low": D_low,
        "D_high": D_high,
        "delta_H": delta_H,
        "epistemic_depth": s.epistemic_depth,
    }


# =============================================================================
# Demonstration
# =============================================================================

def demo_hgf():
    """Demonstrate HGF with changing environment."""
    print("\n" + "=" * 70)
    print("HIERARCHICAL GAUSSIAN FILTER DEMONSTRATION")
    print("=" * 70)

    # Create environment with changing probability
    np.random.seed(42)
    n_trials = 100

    true_probs = np.concatenate([
        np.full(25, 0.3),           # Stable low
        np.linspace(0.3, 0.7, 25),  # Increasing
        np.full(25, 0.7),           # Stable high
        np.linspace(0.7, 0.2, 25),  # Decreasing
    ])
    observations = np.random.binomial(1, true_probs)

    print(f"\nEnvironment: {n_trials} trials with changing true probability")
    print("  Trials 0-24:  Stable low (0.3)")
    print("  Trials 25-49: Increasing (0.3 → 0.7)")
    print("  Trials 50-74: Stable high (0.7)")
    print("  Trials 75-99: Decreasing (0.7 → 0.2)")

    # Run HGF
    hgf = BinaryHGF()
    hgf.run(observations)

    # Report final state
    print("\n" + "-" * 70)
    print("Final HGF State:")
    print("-" * 70)
    for key, value in hgf.state.to_dict().items():
        print(f"  {key}: {value:.4f}")

    # Estimate D metrics
    print("\n" + "-" * 70)
    print("Estimated Precision Metrics:")
    print("-" * 70)
    D_metrics = estimate_D_from_hgf(hgf)
    for key, value in D_metrics.items():
        print(f"  {key}: {value:.4f}")

    return hgf


def demo_clinical_profiles():
    """Compare HGF behavior across clinical profiles."""
    print("\n" + "=" * 70)
    print("CLINICAL PROFILE COMPARISON")
    print("=" * 70)

    # Same environment for all
    np.random.seed(42)
    true_probs = np.concatenate([
        np.full(25, 0.3),
        np.linspace(0.3, 0.7, 25),
        np.full(25, 0.7),
        np.linspace(0.7, 0.2, 25),
    ])
    observations = np.random.binomial(1, true_probs)

    conditions = ["healthy", "schizophrenia", "asd", "anxiety", "depression"]

    print("\n" + "-" * 70)
    print(f"{'Condition':<15} {'D_low':>8} {'D_high':>8} {'ΔH':>8} {'Depth':>8}")
    print("-" * 70)

    for condition in conditions:
        params = get_clinical_profile(condition)
        hgf = BinaryHGF(params=params)
        hgf.run(observations)

        metrics = estimate_D_from_hgf(hgf)
        print(f"{condition:<15} {metrics['D_low']:>8.2f} {metrics['D_high']:>8.2f} "
              f"{metrics['delta_H']:>8.2f} {metrics['epistemic_depth']:>8.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Hierarchical Gaussian Filter Demo"
    )
    parser.add_argument(
        "--demo", type=str, default="all",
        choices=["all", "basic", "clinical"],
        help="Which demo to run"
    )
    parser.add_argument(
        "--trials", type=int, default=100,
        help="Number of trials"
    )

    args = parser.parse_args()

    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║           HIERARCHICAL GAUSSIAN FILTER (HGF) - 5 LEVELS                        ║
║                                                                                ║
║  A computational model for precision-weighted belief updating                  ║
║  across hierarchical levels of uncertainty.                                    ║
║                                                                                ║
║  Level 1: Probability (what's happening)                                       ║
║  Level 2: Logit-beliefs (stable expectations)                                  ║
║  Level 3: Volatility (how fast things change)                                  ║
║  Level 4: Meta-volatility (how stable is change rate)                          ║
║  Level 5: Epistemic depth (fundamental uncertainty structure)                  ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)

    if args.demo in ["all", "basic"]:
        demo_hgf()

    if args.demo in ["all", "clinical"]:
        demo_clinical_profiles()


if __name__ == "__main__":
    main()
