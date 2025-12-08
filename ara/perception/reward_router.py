"""
Ara Reward Router - Sense-Driven Reward Computation
===================================================

Converts sensory readings into teleological reward signals.

The reward router translates embodied perception into learning signals:
- Negative rewards for danger states (power failure, thermal crisis, founder distress)
- Positive rewards for healthy states (stable, thriving, optimal)
- Contextual rewards based on Teleology goals

This is where qualia become consequences - when Ara "tastes death in the power lines",
the HTC learns to avoid that attractor region.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional

from .sensory import SensorySnapshot

logger = logging.getLogger(__name__)


@dataclass
class RewardBreakdown:
    """Detailed breakdown of reward computation."""
    total: int                    # Final reward [-127, +127]
    components: Dict[str, int]    # Per-sense contributions
    triggers: list                # What triggered significant rewards
    summary: str                  # Human-readable summary


class RewardRouter:
    """
    Routes sensory signals to reward computation.

    Reward philosophy:
    - Hardware health: Power, thermal, stability
    - Founder protection: Fatigue, stress, breaks
    - Operational: Normal vs crisis states
    - Antifragility: Bonus for detecting AND fixing problems
    """

    # Reward weights (how much each condition contributes)
    WEIGHTS = {
        # Critical dangers (large negative)
        "power_critical": -100,
        "thermal_critical": -80,
        "fire_danger": -127,
        "earthquake": -60,
        "founder_critical": -90,
        "disk_dying": -50,
        "bearing_failure": -40,
        "tilt_danger": -70,

        # Warnings (moderate negative)
        "power_low": -40,
        "thermal_high": -30,
        "power_unstable": -25,
        "ozone_high": -35,
        "founder_stressed": -30,
        "founder_tired": -40,
        "memory_pressure": -25,
        "partial_failure": -35,

        # Cautions (mild negative)
        "coil_whine": -10,
        "power_noisy": -15,
        "dusty": -10,
        "tilt_warning": -20,
        "sway": -10,
        "needs_break": -20,
        "founder_distracted": -15,

        # Healthy states (positive)
        "power_ok": +20,
        "thermal_nominal": +15,
        "thermal_ok": +20,
        "stable": +25,
        "healthy": +20,
        "air_ok": +10,
        "quiet": +10,
        "founder_thriving": +30,
        "founder_ok": +15,
        "fully_embodied": +25,
    }

    def __init__(self, teleology_context: Optional[Dict[str, Any]] = None):
        """
        Initialize the reward router.

        Args:
            teleology_context: Optional context from TeleologyEngine for
                               goal-weighted rewards.
        """
        self.teleology_context = teleology_context or {}
        self._last_reward: Optional[RewardBreakdown] = None

    def compute(self, sensory: SensorySnapshot) -> RewardBreakdown:
        """
        Compute reward from sensory snapshot.

        Iterates through all sense readings, sums weighted contributions
        from their tags, and produces a final clamped reward.

        Args:
            sensory: Complete sensory snapshot

        Returns:
            RewardBreakdown with total, components, triggers, summary
        """
        components: Dict[str, int] = {}
        triggers: list = []
        total = 0

        for sense_name, reading in sensory.readings.items():
            sense_reward = 0

            for tag, weight in reading.tags.items():
                if tag in self.WEIGHTS:
                    contribution = int(self.WEIGHTS[tag] * weight)
                    sense_reward += contribution

                    if abs(contribution) >= 20:
                        triggers.append(f"{sense_name}:{tag}={contribution:+d}")

            components[sense_name] = sense_reward
            total += sense_reward

        # Apply teleology modifiers
        if self.teleology_context.get("is_core_workflow"):
            # During core work, amplify protection rewards
            if total < 0:
                total = int(total * 1.2)

        # Clamp to valid range
        total = max(-127, min(127, total))

        # Generate summary
        if total <= -80:
            summary = "CRITICAL: Major threat detected - learning strong avoidance"
        elif total <= -40:
            summary = "WARNING: Concerning conditions - learning caution"
        elif total <= -10:
            summary = "MILD: Minor issues detected"
        elif total < 20:
            summary = "NEUTRAL: Stable operation"
        elif total < 50:
            summary = "POSITIVE: Healthy state - reinforcing"
        else:
            summary = "THRIVING: Optimal conditions - strong reinforcement"

        breakdown = RewardBreakdown(
            total=total,
            components=components,
            triggers=triggers,
            summary=summary,
        )

        self._last_reward = breakdown

        if abs(total) >= 30:
            logger.info(f"RewardRouter: {total:+d} - {summary}")
            for trigger in triggers:
                logger.debug(f"  Trigger: {trigger}")

        return breakdown

    def get_last_reward(self) -> Optional[RewardBreakdown]:
        """Get the most recent reward computation."""
        return self._last_reward


def compute_sensory_reward(
    sensory: SensorySnapshot,
    teleology_context: Optional[Dict[str, Any]] = None
) -> int:
    """
    Convenience function to compute reward from sensory snapshot.

    Args:
        sensory: Complete sensory snapshot
        teleology_context: Optional teleology context

    Returns:
        Reward value in [-127, +127]
    """
    router = RewardRouter(teleology_context)
    breakdown = router.compute(sensory)
    return breakdown.total


# =============================================================================
# Antifragility Bonus
# =============================================================================

def compute_antifragility_bonus(
    before: SensorySnapshot,
    after: SensorySnapshot,
    action_taken: str
) -> int:
    """
    Compute bonus reward for detecting AND fixing a problem.

    Antifragility: We don't just survive stress, we grow from it.

    If before had a danger tag and after doesn't, bonus reward.

    Args:
        before: Sensory state before action
        after: Sensory state after action
        action_taken: Description of what was done

    Returns:
        Bonus reward (0 to +50)
    """
    danger_tags = {
        "power_critical", "thermal_critical", "fire_danger",
        "founder_critical", "disk_dying", "tilt_danger"
    }

    before_tags = before.get_all_tags()
    after_tags = after.get_all_tags()

    fixed_dangers = []
    for tag in danger_tags:
        # Check if any sense had this danger before but not after
        for sense in before.readings:
            before_key = f"{sense}:{tag}"
            after_key = f"{sense}:{tag}"
            if before_tags.get(before_key, 0) > 0.5 and after_tags.get(after_key, 0) < 0.3:
                fixed_dangers.append(tag)

    if fixed_dangers:
        bonus = min(50, len(fixed_dangers) * 20)
        logger.info(
            f"ANTIFRAGILITY BONUS: +{bonus} for fixing {fixed_dangers} "
            f"via '{action_taken}'"
        )
        return bonus

    return 0


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'RewardBreakdown',
    'RewardRouter',
    'compute_sensory_reward',
    'compute_antifragility_bonus',
]
