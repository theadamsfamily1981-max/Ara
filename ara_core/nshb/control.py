#!/usr/bin/env python3
"""
NSHB Control Law - Safe Repair Vector Generation
=================================================

Defines the control problem for human state regulation:

    State: z(t) = (λ̂, Π̂_sensory, Π̂_prior)
    Target: Healthy corridor H ⊂ ℝ³ around (λ≈1, moderate Π)
    Control: Δz = -K·∇D(z) where D(z) is distance-to-health

The Repair Vector specifies *direction and magnitude* of desired
state change. Effectors map this to safe, embodied actions.

CRITICAL SAFETY PRINCIPLE:
    The control law only generates *suggestions* for state change.
    All actual interventions go through safe, slow, embodied channels
    (visual cues, haptics, breathing prompts) - never direct neural stim.
"""

from __future__ import annotations

import time
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum, auto
import numpy as np

from .estimators import GUTCState


# =============================================================================
# Healthy Corridor Definition
# =============================================================================

@dataclass
class HealthyCorridor:
    """
    Defines the target region in (λ, Π) space.

    The healthy corridor is where:
    - λ ≈ 1 (near-critical dynamics)
    - Π_sensory and Π_prior are balanced and moderate

    This can be personalized based on individual baselines.
    """
    # Target values
    lambda_target: float = 1.0
    pi_sensory_target: float = 1.0
    pi_prior_target: float = 1.0

    # Tolerance bounds
    lambda_tolerance: float = 0.3
    pi_sensory_tolerance: float = 0.5
    pi_prior_tolerance: float = 0.5

    # Weights for distance calculation
    w_lambda: float = 1.0          # Weight on criticality
    w_pi_sensory: float = 0.5      # Weight on sensory precision
    w_pi_prior: float = 0.5        # Weight on prior precision

    def contains(self, state: GUTCState) -> bool:
        """Check if state is within healthy corridor."""
        lambda_ok = abs(state.lambda_hat - self.lambda_target) < self.lambda_tolerance
        pi_s_ok = abs(state.pi_sensory - self.pi_sensory_target) < self.pi_sensory_tolerance
        pi_p_ok = abs(state.pi_prior - self.pi_prior_target) < self.pi_prior_tolerance
        return lambda_ok and pi_s_ok and pi_p_ok

    def distance(self, state: GUTCState) -> float:
        """
        Compute weighted distance to healthy corridor center.

        D(z) = w_λ(λ̂-λ*)² + w_s(Π̂_s-Π_s*)² + w_p(Π̂_p-Π_p*)²
        """
        d_lambda = (state.lambda_hat - self.lambda_target) ** 2
        d_pi_s = (state.pi_sensory - self.pi_sensory_target) ** 2
        d_pi_p = (state.pi_prior - self.pi_prior_target) ** 2

        return math.sqrt(
            self.w_lambda * d_lambda +
            self.w_pi_sensory * d_pi_s +
            self.w_pi_prior * d_pi_p
        )

    def gradient(self, state: GUTCState) -> np.ndarray:
        """
        Compute gradient of distance function.

        ∇D(z) = [∂D/∂λ, ∂D/∂Π_s, ∂D/∂Π_p]
        """
        d = self.distance(state)
        if d < 1e-6:
            return np.zeros(3)

        # Partial derivatives (chain rule on sqrt)
        grad = np.array([
            self.w_lambda * (state.lambda_hat - self.lambda_target),
            self.w_pi_sensory * (state.pi_sensory - self.pi_sensory_target),
            self.w_pi_prior * (state.pi_prior - self.pi_prior_target),
        ]) / d

        return grad


# =============================================================================
# Repair Vector
# =============================================================================

class InterventionUrgency(Enum):
    """Urgency level for interventions."""
    NONE = auto()          # In healthy corridor, no action needed
    GENTLE = auto()        # Minor drift, subtle nudges
    MODERATE = auto()      # Noticeable drift, active guidance
    URGENT = auto()        # Significant deviation, strong intervention
    CRITICAL = auto()      # Dangerous state, immediate action


@dataclass
class RepairVector:
    """
    Repair vector Δz specifying desired state change.

    This is the output of the control law - a suggestion for
    how to move the human's state back toward healthy.
    """
    timestamp: float

    # Target changes (signed)
    delta_lambda: float            # Desired Δλ
    delta_pi_sensory: float        # Desired ΔΠ_sensory
    delta_pi_prior: float          # Desired ΔΠ_prior

    # Magnitude and urgency
    magnitude: float = 0.0         # ||Δz||
    urgency: InterventionUrgency = InterventionUrgency.NONE

    # Source state
    source_state: Optional[GUTCState] = None

    # Strategy hints (for effectors)
    primary_target: str = "lambda"  # Which parameter to prioritize
    suggested_modality: str = "visual"  # Preferred effector type

    def __post_init__(self):
        self.magnitude = math.sqrt(
            self.delta_lambda**2 +
            self.delta_pi_sensory**2 +
            self.delta_pi_prior**2
        )

    def to_vector(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.delta_lambda, self.delta_pi_sensory, self.delta_pi_prior])

    def scaled(self, factor: float) -> 'RepairVector':
        """Return a scaled copy of this repair vector."""
        return RepairVector(
            timestamp=self.timestamp,
            delta_lambda=self.delta_lambda * factor,
            delta_pi_sensory=self.delta_pi_sensory * factor,
            delta_pi_prior=self.delta_pi_prior * factor,
            urgency=self.urgency,
            source_state=self.source_state,
            primary_target=self.primary_target,
            suggested_modality=self.suggested_modality,
        )


# =============================================================================
# Control Law
# =============================================================================

@dataclass
class ControlConfig:
    """Configuration for the control law."""
    # Gain (how aggressively to nudge)
    gain_k: float = 0.3

    # Maximum allowed repair magnitude per step
    max_delta_lambda: float = 0.2
    max_delta_pi: float = 0.3

    # Urgency thresholds (distance from corridor)
    threshold_gentle: float = 0.1
    threshold_moderate: float = 0.3
    threshold_urgent: float = 0.6
    threshold_critical: float = 1.0

    # Smoothing (low-pass on repair vector)
    smoothing_alpha: float = 0.3   # EMA coefficient

    # Dead zone (no action if very close to target)
    dead_zone_radius: float = 0.05


class NSHBControlLaw:
    """
    NSHB Control Law - Generates repair vectors.

    Implements: Δz = -K·∇D(z)

    Where:
    - D(z) is distance-to-healthy
    - K is the gain
    - Negative gradient points toward health
    """

    def __init__(
        self,
        config: ControlConfig = None,
        corridor: HealthyCorridor = None,
        verbose: bool = True,
    ):
        self.config = config or ControlConfig()
        self.corridor = corridor or HealthyCorridor()
        self.verbose = verbose

        # State
        self.last_repair: Optional[RepairVector] = None
        self.repair_history: List[RepairVector] = []

        # Smoothed repair (for stability)
        self._smoothed_delta = np.zeros(3)

    def compute_repair(self, state: GUTCState) -> RepairVector:
        """
        Compute repair vector for current state.

        Args:
            state: Current GUTC state z(t)

        Returns:
            RepairVector with desired state change
        """
        now = time.time()

        # Check if in healthy corridor (dead zone)
        distance = self.corridor.distance(state)

        if distance < self.config.dead_zone_radius:
            # No repair needed
            repair = RepairVector(
                timestamp=now,
                delta_lambda=0.0,
                delta_pi_sensory=0.0,
                delta_pi_prior=0.0,
                urgency=InterventionUrgency.NONE,
                source_state=state,
            )
            self._record_repair(repair)
            return repair

        # Compute gradient
        gradient = self.corridor.gradient(state)

        # Repair = negative gradient (move toward health)
        raw_delta = -self.config.gain_k * gradient

        # Apply smoothing (low-pass filter for stability)
        alpha = self.config.smoothing_alpha
        self._smoothed_delta = alpha * raw_delta + (1 - alpha) * self._smoothed_delta
        delta = self._smoothed_delta

        # Clamp to max allowed change
        delta[0] = np.clip(delta[0], -self.config.max_delta_lambda, self.config.max_delta_lambda)
        delta[1] = np.clip(delta[1], -self.config.max_delta_pi, self.config.max_delta_pi)
        delta[2] = np.clip(delta[2], -self.config.max_delta_pi, self.config.max_delta_pi)

        # Determine urgency
        urgency = self._classify_urgency(distance)

        # Determine primary target (which parameter needs most correction)
        abs_deltas = np.abs(delta)
        if abs_deltas[0] >= abs_deltas[1] and abs_deltas[0] >= abs_deltas[2]:
            primary = "lambda"
        elif abs_deltas[1] >= abs_deltas[2]:
            primary = "pi_sensory"
        else:
            primary = "pi_prior"

        # Suggest modality based on target
        modality = self._suggest_modality(primary, state)

        repair = RepairVector(
            timestamp=now,
            delta_lambda=float(delta[0]),
            delta_pi_sensory=float(delta[1]),
            delta_pi_prior=float(delta[2]),
            urgency=urgency,
            source_state=state,
            primary_target=primary,
            suggested_modality=modality,
        )

        self._record_repair(repair)

        if self.verbose and urgency != InterventionUrgency.NONE:
            print(f"[Control] Repair: Δλ={delta[0]:+.2f}, ΔΠs={delta[1]:+.2f}, "
                  f"ΔΠp={delta[2]:+.2f} [{urgency.name}]")

        return repair

    def _classify_urgency(self, distance: float) -> InterventionUrgency:
        """Classify intervention urgency based on distance from corridor."""
        if distance < self.config.threshold_gentle:
            return InterventionUrgency.NONE
        elif distance < self.config.threshold_moderate:
            return InterventionUrgency.GENTLE
        elif distance < self.config.threshold_urgent:
            return InterventionUrgency.MODERATE
        elif distance < self.config.threshold_critical:
            return InterventionUrgency.URGENT
        else:
            return InterventionUrgency.CRITICAL

    def _suggest_modality(self, primary: str, state: GUTCState) -> str:
        """Suggest effector modality based on target and state."""
        if primary == "lambda":
            # λ: breathing/meditation for subcritical, grounding for supercritical
            if state.lambda_hat < 1.0:
                return "breathing"  # Increase variability
            else:
                return "haptic"     # Ground/stabilize

        elif primary == "pi_sensory":
            # Π_sensory: visual for reduction, haptic for increase
            if state.pi_sensory > self.corridor.pi_sensory_target:
                return "visual"     # Calm visual feedback
            else:
                return "haptic"     # Increase sensory engagement

        else:  # pi_prior
            # Π_prior: task adaptation, motivational cues
            if state.pi_prior > self.corridor.pi_prior_target:
                return "ui"         # Reduce task demands
            else:
                return "audio"      # Motivational/engagement cues

    def _record_repair(self, repair: RepairVector):
        """Record repair vector in history."""
        self.last_repair = repair
        self.repair_history.append(repair)
        if len(self.repair_history) > 500:
            self.repair_history = self.repair_history[-500:]

    def get_status(self) -> Dict[str, Any]:
        """Get control law status."""
        return {
            "n_repairs": len(self.repair_history),
            "last_urgency": self.last_repair.urgency.name if self.last_repair else None,
            "last_magnitude": self.last_repair.magnitude if self.last_repair else 0,
            "corridor_lambda_target": self.corridor.lambda_target,
            "corridor_pi_s_target": self.corridor.pi_sensory_target,
            "corridor_pi_p_target": self.corridor.pi_prior_target,
            "gain": self.config.gain_k,
        }


# =============================================================================
# Adaptive Control (Personalization)
# =============================================================================

class AdaptiveController:
    """
    Adaptive controller that personalizes the healthy corridor
    and control parameters based on individual history.
    """

    def __init__(
        self,
        base_controller: NSHBControlLaw,
        learning_rate: float = 0.01,
    ):
        self.controller = base_controller
        self.learning_rate = learning_rate

        # Track individual baseline
        self.baseline_lambda: float = 1.0
        self.baseline_pi_s: float = 1.0
        self.baseline_pi_p: float = 1.0

        # Performance history (for learning)
        self.performance_history: List[Tuple[GUTCState, float]] = []

    def update_baseline(self, state: GUTCState, performance: float):
        """
        Update personal baseline based on performance feedback.

        Args:
            state: GUTC state during good performance
            performance: Performance metric (0-1, higher is better)
        """
        self.performance_history.append((state, performance))

        # Only learn from good performance episodes
        if performance > 0.7:
            # EMA update toward this state
            lr = self.learning_rate * performance
            self.baseline_lambda = (1 - lr) * self.baseline_lambda + lr * state.lambda_hat
            self.baseline_pi_s = (1 - lr) * self.baseline_pi_s + lr * state.pi_sensory
            self.baseline_pi_p = (1 - lr) * self.baseline_pi_p + lr * state.pi_prior

            # Update corridor targets
            self.controller.corridor.lambda_target = self.baseline_lambda
            self.controller.corridor.pi_sensory_target = self.baseline_pi_s
            self.controller.corridor.pi_prior_target = self.baseline_pi_p

    def compute_repair(self, state: GUTCState) -> RepairVector:
        """Compute repair with personalized corridor."""
        return self.controller.compute_repair(state)
