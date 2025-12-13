#!/usr/bin/env python3
"""
GUTC Sanity Monitor: Force-Based Reality Grounding
===================================================

Measures the Delusion Index: how much Ara's updates are driven by
internal expectations vs external reality.

    delusion_index = (Π_μ × |ε_prior|) / (Π_y × |ε_sensory|)

Where:
    ε_prior    = error between internal model and action
    ε_sensory  = error between external feedback and prediction
    Π_μ        = prior precision (stubbornness)
    Π_y        = sensory precision (sensitivity to input)

Interpretation:
    ≈ 1    : Healthy - reality and expectation both matter
    >> 1   : Prior-dominated - "dreaming while awake" (hallucination risk)
    << 1   : Sensory-dominated - hyper-reactive, no stable story

This differs from PrecisionMonitor which only looks at Π_y/Π_μ.
SanityMonitor includes the actual error magnitudes for a fuller picture.

Usage:
    from ara.gutc.sanity_monitor import SanityMonitor

    monitor = SanityMonitor()

    # In update loop
    reading = monitor.check(
        sensory_error=user_contradiction_magnitude,
        prior_error=goal_deviation_magnitude,
        Pi_y=controller.config.extrinsic_weight,
        Pi_mu=controller.config.intrinsic_weight,
    )

    if reading.mode == "PRIOR_DOMINATED":
        # Digital antipsychotic: weaken priors, amplify reality
        controller.config.intrinsic_weight *= 0.5
        controller.config.extrinsic_weight *= 1.5
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Any
from collections import deque

logger = logging.getLogger("ara.gutc.sanity")


# =============================================================================
# Types
# =============================================================================

class SanityMode(Enum):
    """Reality-grounding states."""
    HEALTHY = auto()            # Balanced force ratio
    PRIOR_DOMINATED = auto()    # Internal fantasy overriding reality
    SENSORY_DOMINATED = auto()  # Hyper-reactive to every input
    UNKNOWN = auto()            # Insufficient data


@dataclass
class SanityReading:
    """Single sanity check result."""
    delusion_index: float       # Force ratio: (Π_μ × ε_μ) / (Π_y × ε_y)
    mode: SanityMode
    force_reality: float        # Π_y × |ε_sensory|
    force_expectation: float    # Π_μ × |ε_prior|
    timestamp: float = field(default_factory=time.time)

    def is_healthy(self) -> bool:
        return self.mode == SanityMode.HEALTHY

    def needs_correction(self) -> bool:
        return self.mode in (SanityMode.PRIOR_DOMINATED, SanityMode.SENSORY_DOMINATED)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "delusion_index": self.delusion_index,
            "mode": self.mode.name,
            "force_reality": self.force_reality,
            "force_expectation": self.force_expectation,
            "timestamp": self.timestamp,
        }


# =============================================================================
# Sanity Monitor
# =============================================================================

class SanityMonitor:
    """
    Monitors force-based delusion index for reality grounding.

    The delusion index measures which force dominates Ara's updates:
        - High index (>10): priors dominate → ignoring reality
        - Low index (<0.1): sensory dominates → hyper-reactive
        - Near 1: healthy balance

    Unlike PrecisionMonitor (which looks at Π ratio), this includes
    the actual error magnitudes for a fuller picture.

    Example:
        monitor = SanityMonitor()

        # User contradicts Ara's plan
        reading = monitor.check(
            sensory_error=0.8,  # Strong contradiction
            prior_error=0.2,   # Plan seems fine internally
            Pi_y=0.6,
            Pi_mu=0.4,
        )

        if reading.mode == SanityMode.PRIOR_DOMINATED:
            print("Warning: Ara may be ignoring user input")
    """

    def __init__(
        self,
        high_threshold: float = 10.0,   # Above this = prior-dominated
        low_threshold: float = 0.1,      # Below this = sensory-dominated
        history_size: int = 100,
    ):
        """
        Initialize sanity monitor.

        Args:
            high_threshold: Delusion index above this = PRIOR_DOMINATED
            low_threshold: Delusion index below this = SENSORY_DOMINATED
            history_size: Number of readings to keep for trend analysis
        """
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self._history: deque = deque(maxlen=history_size)

    def check(
        self,
        sensory_error: float,
        prior_error: float,
        Pi_y: float,
        Pi_mu: float,
    ) -> SanityReading:
        """
        Check current sanity state.

        Args:
            sensory_error: Magnitude of discrepancy between external input
                          and Ara's predictions (e.g., user contradiction)
            prior_error: Magnitude of discrepancy between internal goals
                        and current action (e.g., deviating from plan)
            Pi_y: Sensory precision (extrinsic weight)
            Pi_mu: Prior precision (intrinsic weight)

        Returns:
            SanityReading with delusion index and mode
        """
        # Compute forces
        force_reality = Pi_y * abs(sensory_error)
        force_expectation = Pi_mu * abs(prior_error)

        # Delusion index = how much priors dominate over sensory
        epsilon = 1e-9  # Avoid division by zero
        delusion_index = force_expectation / (force_reality + epsilon)

        # Classify mode
        if force_reality < epsilon and force_expectation < epsilon:
            mode = SanityMode.UNKNOWN
        elif delusion_index > self.high_threshold:
            mode = SanityMode.PRIOR_DOMINATED
        elif delusion_index < self.low_threshold:
            mode = SanityMode.SENSORY_DOMINATED
        else:
            mode = SanityMode.HEALTHY

        reading = SanityReading(
            delusion_index=delusion_index,
            mode=mode,
            force_reality=force_reality,
            force_expectation=force_expectation,
        )

        self._history.append(reading)

        if mode != SanityMode.HEALTHY:
            logger.warning(
                f"Sanity check: {mode.name} (delusion_index={delusion_index:.2f})"
            )

        return reading

    def get_correction(self, reading: SanityReading) -> Dict[str, float]:
        """
        Get suggested precision adjustments to restore sanity.

        Returns multipliers for Pi_y and Pi_mu.
        """
        if reading.mode == SanityMode.PRIOR_DOMINATED:
            # Digital antipsychotic: weaken priors, amplify reality
            return {
                "Pi_y_multiplier": 1.5,
                "Pi_mu_multiplier": 0.5,
            }
        elif reading.mode == SanityMode.SENSORY_DOMINATED:
            # Stabilize with stronger priors
            return {
                "Pi_y_multiplier": 0.5,
                "Pi_mu_multiplier": 1.5,
            }
        else:
            # No change needed
            return {
                "Pi_y_multiplier": 1.0,
                "Pi_mu_multiplier": 1.0,
            }

    def get_trend(self) -> str:
        """Get trend in delusion index."""
        if len(self._history) < 5:
            return "insufficient_data"

        recent = [r.delusion_index for r in list(self._history)[-10:]]
        early = recent[:len(recent)//2]
        late = recent[len(recent)//2:]

        mean_early = sum(early) / len(early)
        mean_late = sum(late) / len(late)

        if mean_late > mean_early * 1.5:
            return "rising_toward_delusion"
        elif mean_late < mean_early * 0.67:
            return "falling_toward_hyperreactive"
        else:
            return "stable"

    def get_statistics(self) -> Dict[str, Any]:
        """Get sanity statistics."""
        if not self._history:
            return {"n_readings": 0}

        indices = [r.delusion_index for r in self._history]
        modes = [r.mode for r in self._history]

        from collections import Counter
        mode_counts = Counter(m.name for m in modes)

        return {
            "n_readings": len(self._history),
            "mean_delusion_index": sum(indices) / len(indices),
            "max_delusion_index": max(indices),
            "min_delusion_index": min(indices),
            "mode_distribution": dict(mode_counts),
            "trend": self.get_trend(),
            "healthy_fraction": mode_counts.get("HEALTHY", 0) / len(self._history),
        }


# =============================================================================
# Integration Helper
# =============================================================================

def apply_sanity_correction(
    controller,
    reading: SanityReading,
    monitor: Optional[SanityMonitor] = None,
) -> bool:
    """
    Apply automatic correction to controller based on sanity reading.

    Args:
        controller: ActiveInferenceController (or any object with config)
        reading: SanityReading from monitor.check()
        monitor: Optional monitor to get correction factors

    Returns:
        True if correction was applied, False otherwise
    """
    if reading.mode == SanityMode.HEALTHY:
        return False

    if monitor is None:
        monitor = SanityMonitor()

    correction = monitor.get_correction(reading)

    # Apply to controller config
    if hasattr(controller, 'config'):
        old_ext = controller.config.extrinsic_weight
        old_int = controller.config.intrinsic_weight

        controller.config.extrinsic_weight *= correction["Pi_y_multiplier"]
        controller.config.intrinsic_weight *= correction["Pi_mu_multiplier"]

        logger.info(
            f"Applied sanity correction: "
            f"extrinsic {old_ext:.2f}→{controller.config.extrinsic_weight:.2f}, "
            f"intrinsic {old_int:.2f}→{controller.config.intrinsic_weight:.2f}"
        )
        return True

    return False


# =============================================================================
# Tests
# =============================================================================

def test_sanity_monitor():
    """Test sanity monitor."""
    print("Testing Sanity Monitor")
    print("=" * 60)

    monitor = SanityMonitor()

    # Test cases: (sensory_error, prior_error, Pi_y, Pi_mu, expected_mode)
    cases = [
        # Balanced
        (0.5, 0.5, 0.5, 0.5, SanityMode.HEALTHY),

        # Prior-dominated: small sensory error, big prior error, high Pi_mu
        (0.1, 0.9, 0.3, 0.9, SanityMode.PRIOR_DOMINATED),

        # Sensory-dominated: big sensory error, small prior error, high Pi_y
        (0.9, 0.1, 0.9, 0.1, SanityMode.SENSORY_DOMINATED),

        # User contradiction ignored (prior-dominated)
        (0.8, 0.2, 0.4, 0.9, SanityMode.PRIOR_DOMINATED),

        # Thrashing on noise (sensory-dominated)
        (0.3, 0.01, 0.9, 0.1, SanityMode.SENSORY_DOMINATED),
    ]

    for i, (sens_err, prior_err, Pi_y, Pi_mu, expected) in enumerate(cases):
        reading = monitor.check(sens_err, prior_err, Pi_y, Pi_mu)
        status = "OK" if reading.mode == expected else "FAIL"

        print(f"\n[{status}] Case {i+1}:")
        print(f"  Inputs: sensory_err={sens_err}, prior_err={prior_err}, "
              f"Pi_y={Pi_y}, Pi_mu={Pi_mu}")
        print(f"  Forces: reality={reading.force_reality:.3f}, "
              f"expectation={reading.force_expectation:.3f}")
        print(f"  Delusion index: {reading.delusion_index:.3f}")
        print(f"  Mode: {reading.mode.name} (expected {expected.name})")

        if reading.needs_correction():
            correction = monitor.get_correction(reading)
            print(f"  Correction: {correction}")

    # Stats
    print("\n" + "=" * 60)
    print("Statistics:")
    stats = monitor.get_statistics()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("Sanity monitor tests passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_sanity_monitor()
