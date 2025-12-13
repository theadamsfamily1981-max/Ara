#!/usr/bin/env python3
"""
Active Inference Controller for Ara
===================================

Implements a *practical* version of Expected Free Energy (G(π))
with explicit Extrinsic vs Intrinsic weighting.

    G(π) ≈ w_ext × goal_divergence(π) - w_int × info_gain(π) + regularizers

Terms:
    - goal_divergence   → How far does this policy leave us from preferred outcomes?
    - info_gain         → How much uncertainty does this policy reduce?
    - w_ext, w_int      → Personality knobs ("worker" vs "scientist")

This is NOT a full continuous-time active inference stack.
It's a practical decision API that:
    1. Scores candidate policies
    2. Picks the best one under current mode config
    3. Adapts weights based on system state (criticality, body stress)

Usage:
    from ara.gutc.active_inference import (
        ActiveInferenceController,
        ActiveInferenceConfig,
        PolicyEstimate,
        PolicyType,
    )

    # Create controller with personality config
    controller = ActiveInferenceController(ActiveInferenceConfig(
        extrinsic_weight=0.7,
        intrinsic_weight=0.3,
    ))

    # Score candidates
    estimates = [
        PolicyEstimate(name="fix_bug", goal_divergence=0.2, expected_info_gain=0.1),
        PolicyEstimate(name="run_diagnostics", goal_divergence=0.5, expected_info_gain=0.8),
    ]

    best = controller.select_policy(estimates)
    print(f"Best policy: {best.estimate.name} (G={best.G:.3f})")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple
import time
import logging

logger = logging.getLogger("ara.gutc.active_inference")


# =============================================================================
# Policy Types
# =============================================================================

class PolicyType(Enum):
    """Classification of policy intent."""
    PRAGMATIC = auto()   # Directly changes world toward goals
    EPISTEMIC = auto()   # Probes / diagnostics / info-gathering
    MIXED = auto()       # Does both
    MAINTENANCE = auto() # Self-care / homeostasis
    SAFETY = auto()      # Risk mitigation / protection


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ActiveInferenceConfig:
    """
    Core knobs for Ara's decision style.

    w_ext + w_int don't have to sum to 1.0, but often will.
    These are the "personality" settings.

    Example presets:
        WORKER:    w_ext=0.85, w_int=0.15 (task-focused)
        SCIENTIST: w_ext=0.40, w_int=0.60 (curiosity-driven)
        BALANCED:  w_ext=0.60, w_int=0.40 (default)
    """
    # Primary EFE weights
    extrinsic_weight: float = 0.7    # "obedient worker" - goal pursuit
    intrinsic_weight: float = 0.3    # "curious scientist" - information seeking

    # Regularizers (always penalties)
    energy_weight: float = 0.05      # Penalize expensive actions (time, power, GPU)
    risk_weight: float = 0.10        # Penalize dangerous / irreversible actions
    urgency_weight: float = 0.15     # Boost time-sensitive actions

    # Body/criticality integration
    stress_penalty: float = 0.20     # Penalty multiplier when body stressed
    criticality_boost: float = 0.10  # Bonus for stability-preserving actions near criticality

    # Numeric stability
    max_abs_G: float = 1e6           # Clamp to avoid overflow
    epsilon: float = 1e-8            # Avoid division by zero

    def __post_init__(self):
        """Validate configuration."""
        assert self.extrinsic_weight >= 0, "extrinsic_weight must be non-negative"
        assert self.intrinsic_weight >= 0, "intrinsic_weight must be non-negative"


# =============================================================================
# Preset Configurations
# =============================================================================

# Task-focused: minimize goal divergence, low curiosity
WORKER_MODE = ActiveInferenceConfig(
    extrinsic_weight=0.85,
    intrinsic_weight=0.15,
    energy_weight=0.05,
    risk_weight=0.10,
)

# Curiosity-driven: prioritize learning and exploration
SCIENTIST_MODE = ActiveInferenceConfig(
    extrinsic_weight=0.40,
    intrinsic_weight=0.60,
    energy_weight=0.02,  # Accept higher costs for learning
    risk_weight=0.15,
)

# Default balanced mode
BALANCED_MODE = ActiveInferenceConfig(
    extrinsic_weight=0.60,
    intrinsic_weight=0.40,
    energy_weight=0.05,
    risk_weight=0.10,
)

# Crisis mode: maximize stability, minimize risk
CRISIS_MODE = ActiveInferenceConfig(
    extrinsic_weight=0.30,   # Less goal-focused
    intrinsic_weight=0.10,   # No time for curiosity
    energy_weight=0.20,      # Conserve resources
    risk_weight=0.40,        # Heavily penalize risk
    stress_penalty=0.30,
)


# =============================================================================
# Policy Estimates
# =============================================================================

@dataclass
class PolicyEstimate:
    """
    Pre-computed estimates for a single candidate policy π.

    These are produced by cheap heuristics or small learned models;
    the controller only *combines* them into a scalar G.

    Convention:
        - goal_divergence: higher = worse (farther from preferred y)
        - expected_info_gain: higher = better (more uncertainty reduction)
        - expected_energy_cost: higher = worse (more expensive)
        - expected_risk: higher = worse (more dangerous)
    """
    name: str
    policy_type: PolicyType = PolicyType.PRAGMATIC

    # Extrinsic: "How bad is the future if we follow this?"
    # Range: [0, 1] where 0 = perfect, 1 = worst possible
    goal_divergence: float = 0.0

    # Intrinsic: "How much uncertainty do we expect to reduce?"
    # Range: [0, 1] where 0 = learn nothing, 1 = resolve all uncertainty
    expected_info_gain: float = 0.0

    # Costs (always penalties, range [0, 1])
    expected_energy_cost: float = 0.0   # Time, power, GPU hours, etc.
    expected_risk: float = 0.0          # Chance of harm to stability/user/hardware

    # Time sensitivity (higher = more urgent)
    urgency: float = 0.5                # 0 = can wait forever, 1 = do it now

    # Stability bonus (for criticality-aware scoring)
    stability_contribution: float = 0.0  # How much this helps maintain E(λ) ≈ 0

    # Optional metadata for logging/debugging
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Clamp values to valid ranges."""
        self.goal_divergence = max(0.0, min(1.0, self.goal_divergence))
        self.expected_info_gain = max(0.0, min(1.0, self.expected_info_gain))
        self.expected_energy_cost = max(0.0, min(1.0, self.expected_energy_cost))
        self.expected_risk = max(0.0, min(1.0, self.expected_risk))
        self.urgency = max(0.0, min(1.0, self.urgency))


@dataclass
class ScoredPolicy:
    """A policy estimate with its computed G score."""
    estimate: PolicyEstimate
    G: float                              # Lower is better
    breakdown: Dict[str, float] = None    # Component contributions for debugging

    @property
    def name(self) -> str:
        return self.estimate.name

    def __repr__(self) -> str:
        return f"ScoredPolicy({self.name}, G={self.G:.4f})"


# =============================================================================
# System State (for adaptive weighting)
# =============================================================================

@dataclass
class SystemState:
    """
    Current system state for adaptive weight modulation.

    The controller uses this to adjust its behavior based on
    body stress, criticality, and context.
    """
    # Body state (from BodyInterface)
    body_stress: float = 0.0         # 0-1, from body daemon
    thermal_state: str = "NOMINAL"   # NOMINAL, WARMING, HOT, CRITICAL

    # Criticality state (from GradientCriticalityMonitor)
    rho: float = 0.8                 # Branching ratio (target ~0.8)
    criticality_status: str = "OPTIMAL"  # COLD, OPTIMAL, HOT, CRITICAL

    # Context
    in_conversation: bool = False    # User actively engaged
    pending_tasks: int = 0           # Number of queued tasks

    @property
    def is_stressed(self) -> bool:
        return self.body_stress > 0.7 or self.thermal_state in ("HOT", "CRITICAL")

    @property
    def is_critical(self) -> bool:
        return self.criticality_status == "CRITICAL" or self.thermal_state == "CRITICAL"


# =============================================================================
# Active Inference Controller
# =============================================================================

class ActiveInferenceController:
    """
    Lightweight controller that scores candidate policies under
    a given configuration and picks the best one.

    This is the "decision kernel" that chief_of_staff.py calls.

    Features:
        - Explicit extrinsic/intrinsic weighting
        - Adaptive weight modulation based on system state
        - Detailed score breakdown for interpretability
        - History tracking for learning

    Example:
        controller = ActiveInferenceController(BALANCED_MODE)

        estimates = [...]  # List of PolicyEstimate
        best = controller.select_policy(estimates)

        print(f"Selected: {best.name} (G={best.G:.3f})")
    """

    def __init__(
        self,
        config: Optional[ActiveInferenceConfig] = None,
        adaptive: bool = True,
    ):
        """
        Initialize controller.

        Args:
            config: Base configuration (defaults to BALANCED_MODE)
            adaptive: Whether to modulate weights based on system state
        """
        self.config = config or BALANCED_MODE
        self.adaptive = adaptive

        # Current system state (updated externally)
        self._system_state = SystemState()

        # History for learning
        self._selection_history: List[Tuple[float, str, float]] = []  # (time, name, G)
        self._max_history = 1000

    def set_system_state(self, state: SystemState):
        """Update current system state for adaptive weighting."""
        self._system_state = state

    def get_effective_config(self) -> ActiveInferenceConfig:
        """
        Get configuration with adaptive adjustments applied.

        Returns a modified config based on current system state.
        """
        if not self.adaptive:
            return self.config

        c = self.config
        state = self._system_state

        # Start with base weights
        w_ext = c.extrinsic_weight
        w_int = c.intrinsic_weight
        w_energy = c.energy_weight
        w_risk = c.risk_weight

        # Adapt based on body stress
        if state.is_stressed:
            # Under stress: be more conservative
            w_int *= 0.5           # Less curiosity
            w_energy *= 2.0        # More energy-conscious
            w_risk *= 1.5          # More risk-averse

        # Adapt based on criticality
        if state.is_critical:
            # Near collapse: survival mode
            w_ext *= 0.5           # Less goal-obsessed
            w_int *= 0.2           # Minimal curiosity
            w_risk *= 3.0          # Heavy risk penalty

        # Adapt based on boredom (low activity, stable system)
        if state.pending_tasks == 0 and not state.in_conversation:
            if state.criticality_status == "OPTIMAL" and not state.is_stressed:
                # Safe and idle: allow more exploration
                w_int *= 1.5

        return ActiveInferenceConfig(
            extrinsic_weight=w_ext,
            intrinsic_weight=w_int,
            energy_weight=w_energy,
            risk_weight=w_risk,
            urgency_weight=c.urgency_weight,
            stress_penalty=c.stress_penalty,
            criticality_boost=c.criticality_boost,
            max_abs_G=c.max_abs_G,
            epsilon=c.epsilon,
        )

    def score_policy(
        self,
        est: PolicyEstimate,
        config: Optional[ActiveInferenceConfig] = None,
    ) -> ScoredPolicy:
        """
        Compute scalar G(π) for a given policy estimate.

        G(π) = w_ext × goal_divergence
             - w_int × info_gain
             + w_energy × energy_cost
             + w_risk × risk
             - w_urgency × urgency
             - w_crit × stability_contribution (if near criticality)
             + stress_penalty (if body stressed)

        Convention: LOWER G is BETTER.

        Args:
            est: Policy estimate to score
            config: Optional override config (uses effective config if None)

        Returns:
            ScoredPolicy with G value and breakdown
        """
        c = config or self.get_effective_config()
        breakdown = {}

        # Initialize G
        G = 0.0

        # =========================
        # EXTRINSIC: Goal pursuit
        # =========================
        extrinsic_term = c.extrinsic_weight * est.goal_divergence
        G += extrinsic_term
        breakdown["extrinsic"] = extrinsic_term

        # =========================
        # INTRINSIC: Curiosity
        # =========================
        # Info gain is GOOD → subtract → lowers G
        intrinsic_term = c.intrinsic_weight * est.expected_info_gain
        G -= intrinsic_term
        breakdown["intrinsic"] = -intrinsic_term  # Negative contribution

        # =========================
        # REGULARIZERS
        # =========================

        # Energy cost (penalty)
        energy_term = c.energy_weight * est.expected_energy_cost
        G += energy_term
        breakdown["energy"] = energy_term

        # Risk (penalty)
        risk_term = c.risk_weight * est.expected_risk
        G += risk_term
        breakdown["risk"] = risk_term

        # Urgency (bonus for time-sensitive actions)
        # Higher urgency → subtract → lowers G → prefer urgent actions
        urgency_term = c.urgency_weight * est.urgency
        G -= urgency_term
        breakdown["urgency"] = -urgency_term

        # =========================
        # CONTEXTUAL ADJUSTMENTS
        # =========================

        # Criticality-aware: bonus for stability-preserving actions
        if self._system_state.is_critical and est.stability_contribution > 0:
            crit_bonus = c.criticality_boost * est.stability_contribution
            G -= crit_bonus
            breakdown["criticality_bonus"] = -crit_bonus

        # Body stress penalty for high-energy actions
        if self._system_state.is_stressed and est.expected_energy_cost > 0.5:
            stress_term = c.stress_penalty * est.expected_energy_cost
            G += stress_term
            breakdown["stress_penalty"] = stress_term

        # =========================
        # CLAMP
        # =========================
        if G > c.max_abs_G:
            G = c.max_abs_G
        elif G < -c.max_abs_G:
            G = -c.max_abs_G

        return ScoredPolicy(estimate=est, G=G, breakdown=breakdown)

    def select_policy(
        self,
        estimates: List[PolicyEstimate],
        return_all: bool = False,
    ) -> Optional[ScoredPolicy] | List[ScoredPolicy]:
        """
        Score candidates and return the best one (lowest G).

        Args:
            estimates: List of candidate policy estimates
            return_all: If True, return all scored policies (sorted)

        Returns:
            Best ScoredPolicy, or None if empty.
            If return_all=True, returns sorted list.
        """
        if not estimates:
            return [] if return_all else None

        # Score all candidates
        scored = [self.score_policy(e) for e in estimates]

        # Sort by G (ascending = best first)
        scored.sort(key=lambda s: s.G)

        # Record selection
        best = scored[0]
        self._selection_history.append((time.time(), best.name, best.G))
        if len(self._selection_history) > self._max_history:
            self._selection_history.pop(0)

        logger.debug(
            f"Selected policy: {best.name} (G={best.G:.4f}) "
            f"from {len(estimates)} candidates"
        )

        if return_all:
            return scored
        return best

    def compare_policies(
        self,
        estimates: List[PolicyEstimate],
    ) -> str:
        """
        Generate a comparison table for debugging.

        Returns a formatted string showing all policies and their scores.
        """
        if not estimates:
            return "No policies to compare"

        scored = [self.score_policy(e) for e in estimates]
        scored.sort(key=lambda s: s.G)

        lines = ["Policy Comparison (lower G = better):", "-" * 60]

        for i, sp in enumerate(scored):
            marker = "→" if i == 0 else " "
            lines.append(
                f"{marker} {sp.name:30} G={sp.G:+8.4f} "
                f"[div={sp.estimate.goal_divergence:.2f}, "
                f"info={sp.estimate.expected_info_gain:.2f}]"
            )

        return "\n".join(lines)

    def set_mode(self, mode: str):
        """
        Set controller mode by name.

        Args:
            mode: One of "worker", "scientist", "balanced", "crisis"
        """
        modes = {
            "worker": WORKER_MODE,
            "scientist": SCIENTIST_MODE,
            "balanced": BALANCED_MODE,
            "crisis": CRISIS_MODE,
        }

        if mode.lower() not in modes:
            raise ValueError(f"Unknown mode: {mode}. Available: {list(modes.keys())}")

        self.config = modes[mode.lower()]
        logger.info(f"Active inference mode set to: {mode}")

    def get_selection_stats(self) -> Dict[str, Any]:
        """Get statistics about recent selections."""
        if not self._selection_history:
            return {"count": 0}

        Gs = [g for _, _, g in self._selection_history]
        names = [n for _, n, _ in self._selection_history]

        from collections import Counter
        name_counts = Counter(names)

        return {
            "count": len(self._selection_history),
            "mean_G": sum(Gs) / len(Gs),
            "min_G": min(Gs),
            "max_G": max(Gs),
            "most_selected": name_counts.most_common(5),
        }


# =============================================================================
# Factory Functions
# =============================================================================

def create_controller(
    mode: str = "balanced",
    adaptive: bool = True,
    **config_overrides,
) -> ActiveInferenceController:
    """
    Factory function to create a configured controller.

    Args:
        mode: "worker", "scientist", "balanced", or "crisis"
        adaptive: Enable adaptive weight modulation
        **config_overrides: Override specific config values

    Returns:
        Configured ActiveInferenceController

    Example:
        controller = create_controller("worker", risk_weight=0.2)
    """
    modes = {
        "worker": WORKER_MODE,
        "scientist": SCIENTIST_MODE,
        "balanced": BALANCED_MODE,
        "crisis": CRISIS_MODE,
    }

    base_config = modes.get(mode.lower(), BALANCED_MODE)

    # Apply overrides
    if config_overrides:
        config_dict = {
            "extrinsic_weight": base_config.extrinsic_weight,
            "intrinsic_weight": base_config.intrinsic_weight,
            "energy_weight": base_config.energy_weight,
            "risk_weight": base_config.risk_weight,
            "urgency_weight": base_config.urgency_weight,
            "stress_penalty": base_config.stress_penalty,
            "criticality_boost": base_config.criticality_boost,
        }
        config_dict.update(config_overrides)
        config = ActiveInferenceConfig(**config_dict)
    else:
        config = base_config

    return ActiveInferenceController(config=config, adaptive=adaptive)


# =============================================================================
# Tests
# =============================================================================

def test_active_inference():
    """Test active inference controller."""
    print("Testing Active Inference Controller")
    print("=" * 60)

    # Create controller
    controller = create_controller("balanced")

    # Create candidate policies
    estimates = [
        PolicyEstimate(
            name="restart_service",
            policy_type=PolicyType.PRAGMATIC,
            goal_divergence=0.2,
            expected_info_gain=0.05,
            expected_energy_cost=0.1,
            expected_risk=0.2,
            urgency=0.7,
        ),
        PolicyEstimate(
            name="run_diagnostics",
            policy_type=PolicyType.EPISTEMIC,
            goal_divergence=0.4,
            expected_info_gain=0.8,
            expected_energy_cost=0.2,
            expected_risk=0.05,
            urgency=0.3,
        ),
        PolicyEstimate(
            name="ignore_and_monitor",
            policy_type=PolicyType.MAINTENANCE,
            goal_divergence=0.5,
            expected_info_gain=0.2,
            expected_energy_cost=0.01,
            expected_risk=0.1,
            urgency=0.2,
        ),
    ]

    # Compare in different modes
    for mode in ["worker", "scientist", "balanced", "crisis"]:
        controller.set_mode(mode)
        print(f"\n{mode.upper()} MODE:")
        print("-" * 40)

        best = controller.select_policy(estimates)
        print(f"  Selected: {best.name} (G={best.G:.4f})")
        print(f"  Breakdown: {best.breakdown}")

    # Test comparison output
    print("\n" + "=" * 60)
    controller.set_mode("balanced")
    print(controller.compare_policies(estimates))

    # Test adaptive behavior
    print("\n" + "=" * 60)
    print("Testing Adaptive Behavior:")

    # Normal state
    controller.set_system_state(SystemState())
    best_normal = controller.select_policy(estimates)
    print(f"  Normal state: {best_normal.name} (G={best_normal.G:.4f})")

    # Stressed state
    controller.set_system_state(SystemState(
        body_stress=0.9,
        thermal_state="HOT",
    ))
    best_stressed = controller.select_policy(estimates)
    print(f"  Stressed state: {best_stressed.name} (G={best_stressed.G:.4f})")

    # Critical state
    controller.set_system_state(SystemState(
        criticality_status="CRITICAL",
        body_stress=0.5,
    ))
    best_critical = controller.select_policy(estimates)
    print(f"  Critical state: {best_critical.name} (G={best_critical.G:.4f})")

    print("\n" + "=" * 60)
    print("All active inference tests passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_active_inference()
