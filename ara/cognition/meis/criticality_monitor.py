#!/usr/bin/env python3
"""
MEIS Criticality Monitor
========================

Runtime criticality monitoring for the Memory-Epistemic-Inference System (MEIS).

This module implements GUTC-based criticality monitoring:
- Real-time branching ratio (λ) estimation from spike/activity cascades
- Avalanche statistics collection (size α̂, duration τ̂)
- Edge function E(λ) = ρ(J) - 1 monitoring
- Adaptive policy decisions based on criticality state

The Critical Capacity Principle:
    C_max = k / |E(λ)|    # Diverges at λ = 1 (criticality)

Runtime Policy:
    - λ < 0.9 (subcritical): Increase gain, boost exploration
    - λ > 1.1 (supercritical): Decrease gain, dampen activity
    - 0.9 ≤ λ ≤ 1.1: Maintain current parameters

Usage:
    monitor = CriticalityMonitor(window_size=1000)

    for activity in activity_stream:
        state = monitor.update(activity)
        if state.requires_intervention:
            apply_control(state.suggested_action)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Callable
from enum import Enum
from collections import deque
import warnings


# =============================================================================
# Criticality States and Actions
# =============================================================================

class CriticalityRegime(Enum):
    """
    Classification of dynamical regime based on branching ratio.

    Maps to GUTC clinical quadrants:
    - SUBCRITICAL: λ < 1, damped dynamics (ASD-like over-regularization)
    - CRITICAL: λ ≈ 1, edge of chaos (healthy corridor)
    - SUPERCRITICAL: λ > 1, amplifying dynamics (psychosis-like instability)
    - UNKNOWN: Insufficient data for classification
    """
    UNKNOWN = 0
    SUBCRITICAL = 1      # λ < 0.9
    NEAR_CRITICAL = 2    # 0.9 ≤ λ < 0.95
    CRITICAL = 3         # 0.95 ≤ λ ≤ 1.05
    NEAR_SUPERCRITICAL = 4  # 1.05 < λ ≤ 1.1
    SUPERCRITICAL = 5    # λ > 1.1


class ControlAction(Enum):
    """
    Suggested control actions based on criticality state.

    Following GUTC agency: actions modulate (λ, Π) to restore criticality.
    """
    NONE = 0              # No intervention needed
    INCREASE_GAIN = 1     # Push toward criticality from subcritical
    DECREASE_GAIN = 2     # Pull back from supercritical
    INCREASE_PRECISION = 3   # Sharpen inference
    DECREASE_PRECISION = 4   # Relax inference
    EMERGENCY_DAMPEN = 5     # Severe supercritical → emergency shutdown


@dataclass
class CriticalityState:
    """
    Current criticality state with diagnostic information.

    Provides actionable intelligence for runtime control.
    """
    regime: CriticalityRegime
    lambda_estimate: float
    edge_value: float           # E(λ) = λ - 1
    confidence: float           # 0-1 confidence in estimate

    # Avalanche statistics
    avalanche_exponent_alpha: Optional[float] = None  # Size exponent (~3/2 at criticality)
    avalanche_exponent_tau: Optional[float] = None    # Duration exponent (~2 at criticality)
    branching_ratio_std: float = 0.0

    # Control recommendations
    suggested_action: ControlAction = ControlAction.NONE
    requires_intervention: bool = False
    urgency: float = 0.0        # 0-1 urgency level

    # Historical context
    samples_in_window: int = 0
    time_in_regime_ms: float = 0.0

    def __repr__(self) -> str:
        return (
            f"CriticalityState(regime={self.regime.name}, "
            f"λ={self.lambda_estimate:.3f}±{self.branching_ratio_std:.3f}, "
            f"E(λ)={self.edge_value:.3f}, "
            f"action={self.suggested_action.name})"
        )


@dataclass
class AvalancheEvent:
    """
    Single avalanche event for statistics collection.

    An avalanche is a cascade of activity triggered by initial stimulation.
    At criticality, avalanche statistics follow power laws.
    """
    size: int               # Total activity units in cascade
    duration: int           # Time steps until cascade ends
    peak_activity: float    # Maximum instantaneous activity
    start_time: int         # When cascade began
    trigger_strength: float = 1.0  # Initial perturbation


# =============================================================================
# Branching Ratio Estimator
# =============================================================================

class BranchingRatioEstimator:
    """
    Online estimation of branching ratio λ from activity cascades.

    The branching ratio λ = <A_{t+1}> / <A_t> measures how activity
    propagates across time:

    - λ < 1: Subcritical (activity dies out)
    - λ = 1: Critical (activity marginally sustained)
    - λ > 1: Supercritical (activity grows unbounded)

    Estimation method:
        λ̂ = Σ A_{t+1} / Σ A_t  (regression through origin)

    This is more robust than per-step ratios for sparse activity.
    """

    def __init__(
        self,
        window_size: int = 1000,
        min_samples: int = 50,
        activity_threshold: float = 0.01,
    ):
        self.window_size = window_size
        self.min_samples = min_samples
        self.activity_threshold = activity_threshold

        # Rolling buffers
        self.activity_buffer: deque = deque(maxlen=window_size)
        self.sum_a_t: float = 0.0
        self.sum_a_t_plus_1: float = 0.0
        self.sum_sq_a_t: float = 0.0

        # For variance estimation
        self._pairs: deque = deque(maxlen=window_size)

    def update(self, activity: float) -> Tuple[float, float, int]:
        """
        Update estimator with new activity sample.

        Args:
            activity: Current activity level (non-negative)

        Returns:
            (lambda_estimate, std_estimate, n_samples)
        """
        if len(self.activity_buffer) > 0:
            prev_activity = self.activity_buffer[-1]

            if prev_activity > self.activity_threshold:
                # Update running sums
                self.sum_a_t += prev_activity
                self.sum_a_t_plus_1 += activity
                self.sum_sq_a_t += prev_activity ** 2
                self._pairs.append((prev_activity, activity))

                # Remove oldest if window full
                if len(self._pairs) > self.window_size:
                    old_prev, old_curr = self._pairs[0]
                    self.sum_a_t -= old_prev
                    self.sum_a_t_plus_1 -= old_curr
                    self.sum_sq_a_t -= old_prev ** 2

        self.activity_buffer.append(activity)

        return self.estimate()

    def estimate(self) -> Tuple[float, float, int]:
        """
        Get current branching ratio estimate with uncertainty.

        Returns:
            (lambda_estimate, std_estimate, n_samples)
        """
        n_samples = len(self._pairs)

        if n_samples < self.min_samples or self.sum_a_t < 1e-10:
            return 1.0, float('inf'), n_samples

        # MLE estimate: λ̂ = Σ A_{t+1} / Σ A_t
        lambda_hat = self.sum_a_t_plus_1 / self.sum_a_t

        # Standard error via delta method
        # Var(λ̂) ≈ λ² / n + λ² * Var(A_t) / (E[A_t])²
        mean_a_t = self.sum_a_t / n_samples
        var_a_t = (self.sum_sq_a_t / n_samples) - mean_a_t ** 2

        if mean_a_t > 1e-10:
            se_lambda = np.sqrt(
                (lambda_hat ** 2 / n_samples) +
                (lambda_hat ** 2 * max(0, var_a_t) / (n_samples * mean_a_t ** 2))
            )
        else:
            se_lambda = float('inf')

        return lambda_hat, se_lambda, n_samples

    def reset(self):
        """Reset estimator state."""
        self.activity_buffer.clear()
        self._pairs.clear()
        self.sum_a_t = 0.0
        self.sum_a_t_plus_1 = 0.0
        self.sum_sq_a_t = 0.0


# =============================================================================
# Avalanche Statistics Collector
# =============================================================================

class AvalancheCollector:
    """
    Collect and analyze avalanche statistics.

    At criticality (λ = 1), avalanche statistics follow power laws:
        P(S) ~ S^{-α}  with α ≈ 3/2  (size distribution)
        P(T) ~ T^{-τ}  with τ ≈ 2    (duration distribution)

    Deviations from these exponents indicate off-critical dynamics.
    """

    def __init__(
        self,
        max_avalanches: int = 500,
        activity_threshold: float = 0.1,
        quiescence_threshold: float = 0.01,
    ):
        self.max_avalanches = max_avalanches
        self.activity_threshold = activity_threshold
        self.quiescence_threshold = quiescence_threshold

        self.avalanches: deque = deque(maxlen=max_avalanches)

        # Current cascade tracking
        self._in_cascade = False
        self._cascade_size = 0
        self._cascade_duration = 0
        self._cascade_peak = 0.0
        self._cascade_start = 0
        self._current_time = 0

    def update(self, activity: float) -> Optional[AvalancheEvent]:
        """
        Process new activity sample, detect avalanches.

        Args:
            activity: Current activity level

        Returns:
            Completed AvalancheEvent if one just ended, else None
        """
        self._current_time += 1
        completed = None

        if not self._in_cascade:
            # Check for cascade start
            if activity > self.activity_threshold:
                self._in_cascade = True
                self._cascade_size = int(activity * 100)  # Scale to integer
                self._cascade_duration = 1
                self._cascade_peak = activity
                self._cascade_start = self._current_time
        else:
            # In cascade
            if activity > self.quiescence_threshold:
                # Cascade continues
                self._cascade_size += int(activity * 100)
                self._cascade_duration += 1
                self._cascade_peak = max(self._cascade_peak, activity)
            else:
                # Cascade ended
                completed = AvalancheEvent(
                    size=self._cascade_size,
                    duration=self._cascade_duration,
                    peak_activity=self._cascade_peak,
                    start_time=self._cascade_start,
                )
                self.avalanches.append(completed)
                self._in_cascade = False
                self._cascade_size = 0
                self._cascade_duration = 0

        return completed

    def fit_exponents(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Fit power-law exponents to avalanche statistics.

        Returns:
            (alpha_size, tau_duration) or (None, None) if insufficient data
        """
        if len(self.avalanches) < 30:
            return None, None

        sizes = np.array([a.size for a in self.avalanches if a.size > 0])
        durations = np.array([a.duration for a in self.avalanches if a.duration > 0])

        alpha = self._fit_power_law_exponent(sizes) if len(sizes) > 20 else None
        tau = self._fit_power_law_exponent(durations) if len(durations) > 20 else None

        return alpha, tau

    def _fit_power_law_exponent(self, data: np.ndarray) -> Optional[float]:
        """
        Fit power-law exponent using MLE (Clauset et al. method).

        For P(x) ~ x^{-α}, MLE gives:
            α = 1 + n / Σ ln(x_i / x_min)
        """
        if len(data) < 10:
            return None

        x_min = max(1, np.percentile(data, 10))  # Lower cutoff
        data_above = data[data >= x_min]

        if len(data_above) < 10:
            return None

        n = len(data_above)
        log_sum = np.sum(np.log(data_above / x_min))

        if log_sum < 1e-10:
            return None

        alpha = 1 + n / log_sum

        return alpha

    def reset(self):
        """Reset collector state."""
        self.avalanches.clear()
        self._in_cascade = False
        self._cascade_size = 0
        self._cascade_duration = 0


# =============================================================================
# Main Criticality Monitor
# =============================================================================

class CriticalityMonitor:
    """
    Runtime criticality monitor for MEIS.

    Integrates:
    - Branching ratio estimation (λ̂)
    - Avalanche statistics (α̂, τ̂)
    - Edge function monitoring E(λ)
    - Control action recommendations

    GUTC Integration:
    - Critical capacity C ∝ 1/|E(λ)| diverges at λ = 1
    - Epistemic value peaks at criticality
    - Healthy corridor: 0.95 ≤ λ ≤ 1.05

    Example:
        monitor = CriticalityMonitor()

        for t, activity in enumerate(neural_activity):
            state = monitor.update(activity)

            if state.requires_intervention:
                if state.suggested_action == ControlAction.INCREASE_GAIN:
                    model.gain *= 1.05
                elif state.suggested_action == ControlAction.DECREASE_GAIN:
                    model.gain *= 0.95
    """

    # Regime thresholds (from GUTC theory)
    SUBCRITICAL_THRESHOLD = 0.90
    NEAR_CRITICAL_LOW = 0.95
    NEAR_CRITICAL_HIGH = 1.05
    SUPERCRITICAL_THRESHOLD = 1.10

    # Exponent targets (from universality class)
    TARGET_ALPHA = 1.5  # Size exponent
    TARGET_TAU = 2.0    # Duration exponent

    def __init__(
        self,
        window_size: int = 1000,
        min_confidence_samples: int = 100,
        intervention_threshold: float = 0.15,
        emergency_threshold: float = 0.30,
        update_interval: int = 10,
    ):
        """
        Initialize criticality monitor.

        Args:
            window_size: Samples for rolling estimates
            min_confidence_samples: Minimum samples for confident estimate
            intervention_threshold: |E(λ)| threshold for intervention
            emergency_threshold: |E(λ)| threshold for emergency action
            update_interval: Steps between full state updates
        """
        self.window_size = window_size
        self.min_confidence_samples = min_confidence_samples
        self.intervention_threshold = intervention_threshold
        self.emergency_threshold = emergency_threshold
        self.update_interval = update_interval

        # Component estimators
        self.branching_estimator = BranchingRatioEstimator(
            window_size=window_size,
            min_samples=min_confidence_samples // 2,
        )
        self.avalanche_collector = AvalancheCollector(
            max_avalanches=500,
        )

        # State tracking
        self._step_count = 0
        self._current_regime = CriticalityRegime.UNKNOWN
        self._regime_start_step = 0
        self._last_state: Optional[CriticalityState] = None

        # History for trend analysis
        self._lambda_history: deque = deque(maxlen=100)
        self._regime_history: deque = deque(maxlen=50)

    def update(self, activity: float) -> CriticalityState:
        """
        Update monitor with new activity sample.

        Args:
            activity: Current activity level (non-negative)

        Returns:
            Current CriticalityState with diagnostics and recommendations
        """
        self._step_count += 1

        # Update component estimators
        lambda_hat, lambda_std, n_samples = self.branching_estimator.update(activity)
        avalanche = self.avalanche_collector.update(activity)

        # Full state update at intervals or when needed
        if (self._step_count % self.update_interval == 0 or
            self._last_state is None or
            abs(lambda_hat - 1.0) > self.emergency_threshold):

            state = self._compute_full_state(lambda_hat, lambda_std, n_samples)
            self._last_state = state
        else:
            # Quick update (just lambda)
            state = self._quick_update(lambda_hat, lambda_std, n_samples)

        self._lambda_history.append(lambda_hat)

        return state

    def _compute_full_state(
        self,
        lambda_hat: float,
        lambda_std: float,
        n_samples: int,
    ) -> CriticalityState:
        """Compute comprehensive criticality state."""

        # Classify regime
        regime = self._classify_regime(lambda_hat)

        # Track regime duration
        if regime != self._current_regime:
            self._current_regime = regime
            self._regime_start_step = self._step_count
            self._regime_history.append(regime)

        time_in_regime = self._step_count - self._regime_start_step

        # Edge function
        edge_value = lambda_hat - 1.0

        # Confidence based on sample size and variance
        confidence = self._compute_confidence(n_samples, lambda_std)

        # Avalanche exponents
        alpha_exp, tau_exp = self.avalanche_collector.fit_exponents()

        # Determine control action
        action, requires_intervention, urgency = self._determine_action(
            regime, edge_value, confidence, lambda_hat
        )

        return CriticalityState(
            regime=regime,
            lambda_estimate=lambda_hat,
            edge_value=edge_value,
            confidence=confidence,
            avalanche_exponent_alpha=alpha_exp,
            avalanche_exponent_tau=tau_exp,
            branching_ratio_std=lambda_std,
            suggested_action=action,
            requires_intervention=requires_intervention,
            urgency=urgency,
            samples_in_window=n_samples,
            time_in_regime_ms=time_in_regime,
        )

    def _quick_update(
        self,
        lambda_hat: float,
        lambda_std: float,
        n_samples: int,
    ) -> CriticalityState:
        """Quick state update without full recomputation."""
        if self._last_state is None:
            return self._compute_full_state(lambda_hat, lambda_std, n_samples)

        # Update key fields
        state = CriticalityState(
            regime=self._classify_regime(lambda_hat),
            lambda_estimate=lambda_hat,
            edge_value=lambda_hat - 1.0,
            confidence=self._compute_confidence(n_samples, lambda_std),
            avalanche_exponent_alpha=self._last_state.avalanche_exponent_alpha,
            avalanche_exponent_tau=self._last_state.avalanche_exponent_tau,
            branching_ratio_std=lambda_std,
            suggested_action=self._last_state.suggested_action,
            requires_intervention=self._last_state.requires_intervention,
            urgency=self._last_state.urgency,
            samples_in_window=n_samples,
            time_in_regime_ms=self._step_count - self._regime_start_step,
        )

        return state

    def _classify_regime(self, lambda_hat: float) -> CriticalityRegime:
        """Classify dynamical regime from branching ratio."""
        if lambda_hat < self.SUBCRITICAL_THRESHOLD:
            return CriticalityRegime.SUBCRITICAL
        elif lambda_hat < self.NEAR_CRITICAL_LOW:
            return CriticalityRegime.NEAR_CRITICAL
        elif lambda_hat <= self.NEAR_CRITICAL_HIGH:
            return CriticalityRegime.CRITICAL
        elif lambda_hat <= self.SUPERCRITICAL_THRESHOLD:
            return CriticalityRegime.NEAR_SUPERCRITICAL
        else:
            return CriticalityRegime.SUPERCRITICAL

    def _compute_confidence(self, n_samples: int, lambda_std: float) -> float:
        """Compute confidence in current estimate (0-1)."""
        # Sample-based confidence
        sample_conf = min(1.0, n_samples / self.min_confidence_samples)

        # Variance-based confidence (lower std → higher confidence)
        if lambda_std < float('inf'):
            var_conf = max(0, 1 - lambda_std / 0.5)  # 0.5 std → 0 confidence
        else:
            var_conf = 0.0

        return sample_conf * var_conf

    def _determine_action(
        self,
        regime: CriticalityRegime,
        edge_value: float,
        confidence: float,
        lambda_hat: float,
    ) -> Tuple[ControlAction, bool, float]:
        """
        Determine appropriate control action.

        Returns:
            (action, requires_intervention, urgency)
        """
        abs_edge = abs(edge_value)

        # No action if low confidence
        if confidence < 0.3:
            return ControlAction.NONE, False, 0.0

        # Emergency condition
        if abs_edge > self.emergency_threshold:
            if lambda_hat > 1.0:
                return ControlAction.EMERGENCY_DAMPEN, True, 1.0
            else:
                return ControlAction.INCREASE_GAIN, True, 0.8

        # Standard interventions
        if abs_edge > self.intervention_threshold:
            urgency = min(1.0, abs_edge / self.emergency_threshold)

            if regime in (CriticalityRegime.SUBCRITICAL, CriticalityRegime.NEAR_CRITICAL):
                return ControlAction.INCREASE_GAIN, True, urgency
            elif regime in (CriticalityRegime.SUPERCRITICAL, CriticalityRegime.NEAR_SUPERCRITICAL):
                return ControlAction.DECREASE_GAIN, True, urgency

        # No intervention needed
        return ControlAction.NONE, False, 0.0

    def get_capacity_estimate(self) -> float:
        """
        Estimate current computational capacity.

        Based on GUTC: C ∝ 1/|E(λ)|

        Returns:
            Relative capacity (1.0 at perfect criticality, lower otherwise)
        """
        if self._last_state is None:
            return 0.5  # Unknown

        abs_edge = abs(self._last_state.edge_value)

        if abs_edge < 0.001:
            return 1.0  # At criticality

        # Capacity falls off as 1/(1 + k|E(λ)|)
        k = 5.0  # Steepness parameter
        return 1.0 / (1.0 + k * abs_edge)

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get comprehensive diagnostics dictionary.

        Useful for logging, visualization, and debugging.
        """
        lambda_hat, lambda_std, n_samples = self.branching_estimator.estimate()
        alpha_exp, tau_exp = self.avalanche_collector.fit_exponents()

        # Lambda trend
        if len(self._lambda_history) >= 10:
            recent = list(self._lambda_history)[-10:]
            trend = (recent[-1] - recent[0]) / len(recent)
        else:
            trend = 0.0

        return {
            "lambda_estimate": lambda_hat,
            "lambda_std": lambda_std,
            "lambda_trend": trend,
            "n_samples": n_samples,
            "edge_value": lambda_hat - 1.0,
            "regime": self._current_regime.name,
            "capacity_estimate": self.get_capacity_estimate(),
            "avalanche_alpha": alpha_exp,
            "avalanche_tau": tau_exp,
            "alpha_deviation": abs(alpha_exp - self.TARGET_ALPHA) if alpha_exp else None,
            "tau_deviation": abs(tau_exp - self.TARGET_TAU) if tau_exp else None,
            "n_avalanches": len(self.avalanche_collector.avalanches),
            "total_steps": self._step_count,
        }

    def reset(self):
        """Reset monitor to initial state."""
        self.branching_estimator.reset()
        self.avalanche_collector.reset()
        self._step_count = 0
        self._current_regime = CriticalityRegime.UNKNOWN
        self._regime_start_step = 0
        self._last_state = None
        self._lambda_history.clear()
        self._regime_history.clear()


# =============================================================================
# Adaptive Controller
# =============================================================================

class CriticalityController:
    """
    Closed-loop controller to maintain criticality.

    Uses PID-like control on E(λ) to keep system near λ = 1.

    Control law:
        Δgain = -K_p * E(λ) - K_i * ∫E(λ)dt - K_d * dE/dt
    """

    def __init__(
        self,
        k_p: float = 0.1,
        k_i: float = 0.01,
        k_d: float = 0.05,
        gain_limits: Tuple[float, float] = (0.5, 2.0),
        update_rate: int = 50,
    ):
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.gain_limits = gain_limits
        self.update_rate = update_rate

        # Controller state
        self.current_gain = 1.0
        self._integral = 0.0
        self._prev_error = 0.0
        self._step_count = 0

    def update(self, state: CriticalityState) -> float:
        """
        Compute gain adjustment based on criticality state.

        Args:
            state: Current CriticalityState

        Returns:
            New gain value to apply
        """
        self._step_count += 1

        if self._step_count % self.update_rate != 0:
            return self.current_gain

        if state.confidence < 0.5:
            return self.current_gain

        # PID control on E(λ)
        error = state.edge_value  # Target is E(λ) = 0

        # Proportional
        p_term = -self.k_p * error

        # Integral (with anti-windup)
        self._integral += error
        self._integral = np.clip(self._integral, -10.0, 10.0)
        i_term = -self.k_i * self._integral

        # Derivative
        d_term = -self.k_d * (error - self._prev_error)
        self._prev_error = error

        # Apply control
        delta_gain = p_term + i_term + d_term
        new_gain = self.current_gain * (1.0 + delta_gain)

        # Enforce limits
        new_gain = np.clip(new_gain, self.gain_limits[0], self.gain_limits[1])
        self.current_gain = new_gain

        return new_gain

    def reset(self):
        """Reset controller state."""
        self.current_gain = 1.0
        self._integral = 0.0
        self._prev_error = 0.0
        self._step_count = 0


# =============================================================================
# Tests
# =============================================================================

def test_branching_estimator():
    """Test branching ratio estimation."""
    estimator = BranchingRatioEstimator(window_size=500, min_samples=20)

    # Generate subcritical process (λ = 0.9)
    activity = 1.0
    for _ in range(500):
        activity = max(0, 0.9 * activity + 0.1 * np.random.randn())
        estimator.update(activity)

    lambda_hat, std, n = estimator.estimate()
    print(f"  Subcritical (λ=0.9): λ̂={lambda_hat:.3f}±{std:.3f}, n={n}")

    # Generate critical process (λ = 1.0)
    estimator.reset()
    activity = 1.0
    for _ in range(500):
        activity = max(0, 1.0 * activity + 0.1 * np.random.randn())
        estimator.update(activity)

    lambda_hat, std, n = estimator.estimate()
    print(f"  Critical (λ=1.0): λ̂={lambda_hat:.3f}±{std:.3f}, n={n}")

    print("✓ Branching ratio estimator")


def test_criticality_monitor():
    """Test full criticality monitor."""
    monitor = CriticalityMonitor(window_size=300, min_confidence_samples=50)

    # Run with near-critical dynamics
    activity = 1.0
    for i in range(500):
        # Slight supercritical bias
        activity = max(0, 1.02 * activity + 0.1 * np.random.randn())
        state = monitor.update(activity)

    print(f"  Final state: {state}")
    print(f"  Capacity: {monitor.get_capacity_estimate():.3f}")

    diag = monitor.get_diagnostics()
    print(f"  λ̂ = {diag['lambda_estimate']:.3f}")
    print(f"  Regime: {diag['regime']}")

    print("✓ Criticality monitor")


def test_controller():
    """Test closed-loop controller."""
    monitor = CriticalityMonitor(window_size=200, min_confidence_samples=30)
    controller = CriticalityController(k_p=0.05, update_rate=20)

    # Start supercritical
    activity = 1.0
    true_lambda = 1.15

    lambdas_over_time = []
    gains_over_time = []

    for i in range(1000):
        # Activity with controllable lambda
        effective_lambda = true_lambda * controller.current_gain
        activity = max(0, effective_lambda * activity + 0.1 * np.random.randn())
        activity = min(activity, 10.0)  # Cap for stability

        state = monitor.update(activity)
        gain = controller.update(state)

        if i % 100 == 0:
            lambdas_over_time.append(state.lambda_estimate)
            gains_over_time.append(gain)

    print(f"  Initial λ: {lambdas_over_time[0]:.3f}")
    print(f"  Final λ: {lambdas_over_time[-1]:.3f}")
    print(f"  Final gain: {gains_over_time[-1]:.3f}")

    # Controller should have reduced gain to compensate
    assert controller.current_gain < 1.0, "Controller should reduce gain for supercritical system"

    print("✓ Criticality controller")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("MEIS Criticality Monitor Tests")
    print("=" * 60 + "\n")

    test_branching_estimator()
    test_criticality_monitor()
    test_controller()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60 + "\n")


# =============================================================================
# CLI
# =============================================================================

def main():
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_all_tests()
    else:
        # Demo
        print("MEIS Criticality Monitor Demo")
        print("-" * 40)

        monitor = CriticalityMonitor()

        # Simulate activity with regime transitions
        activity = 1.0
        lambdas = [0.85] * 200 + [1.0] * 200 + [1.15] * 200

        for i, true_lambda in enumerate(lambdas):
            activity = max(0, true_lambda * activity + 0.15 * np.random.randn())
            activity = min(activity, 5.0)

            state = monitor.update(activity)

            if i % 100 == 99:
                print(f"Step {i+1}: {state}")

        print("\nFinal diagnostics:")
        for k, v in monitor.get_diagnostics().items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
