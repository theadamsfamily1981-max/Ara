"""Reward Scorer - Evaluate interactions to learn what works.

Computes a scalar reward for each interaction based on:
- Performance improvement (did it make things better?)
- Human rating (Croft's thumbs up/down)
- Stability score (did tests pass? any regressions?)

This reward signal drives the bandit and pattern mining.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

from .logger import InteractionLog, Outcome

logger = logging.getLogger(__name__)


@dataclass
class RewardSignal:
    """Breakdown of reward components."""

    # Raw components
    perf_improvement: float = 0.0    # [0, 1] normalized performance gain
    human_rating: float = 0.5        # [0, 1] from Croft's feedback
    stability_score: float = 1.0     # [0, 1] tests passed, no regressions

    # Weights (can be tuned)
    perf_weight: float = 0.5
    human_weight: float = 0.3
    stability_weight: float = 0.2

    # Computed
    total_reward: float = field(init=False)

    def __post_init__(self):
        self.total_reward = self.compute()

    def compute(self) -> float:
        """Compute weighted reward."""
        return (
            self.perf_weight * self.perf_improvement +
            self.human_weight * self.human_rating +
            self.stability_weight * self.stability_score
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "perf_improvement": self.perf_improvement,
            "human_rating": self.human_rating,
            "stability_score": self.stability_score,
            "total_reward": self.total_reward,
        }


class RewardScorer:
    """Computes rewards for interactions.

    Can be configured with different weight schemes for different
    task types or optimization goals.
    """

    def __init__(
        self,
        default_weights: Optional[Dict[str, float]] = None,
        task_weights: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        """Initialize the scorer.

        Args:
            default_weights: Default weight config
            task_weights: Per-task-type weight overrides
        """
        self.default_weights = default_weights or {
            "perf": 0.5,
            "human": 0.3,
            "stability": 0.2,
        }
        self.task_weights = task_weights or {}

    def get_weights(self, task_type: str) -> Dict[str, float]:
        """Get weights for a task type."""
        return self.task_weights.get(task_type, self.default_weights)

    def score(self, interaction: InteractionLog) -> RewardSignal:
        """Score an interaction.

        Args:
            interaction: The interaction to score

        Returns:
            RewardSignal with components and total
        """
        outcome = interaction.outcome
        if not outcome:
            # No outcome yet, return neutral
            return RewardSignal(
                perf_improvement=0.5,
                human_rating=0.5,
                stability_score=0.5,
            )

        weights = self.get_weights(interaction.task_type)

        # Compute performance improvement
        perf = self._compute_perf_score(outcome)

        # Compute human rating
        human = self._compute_human_score(outcome)

        # Compute stability score
        stability = self._compute_stability_score(outcome)

        return RewardSignal(
            perf_improvement=perf,
            human_rating=human,
            stability_score=stability,
            perf_weight=weights.get("perf", 0.5),
            human_weight=weights.get("human", 0.3),
            stability_weight=weights.get("stability", 0.2),
        )

    def _compute_perf_score(self, outcome: Outcome) -> float:
        """Compute normalized performance score."""
        # Check explicit metrics
        if outcome.metrics:
            # Look for speedup, throughput, etc.
            for key in ["speedup", "throughput_gain", "latency_reduction"]:
                if key in outcome.metrics:
                    return normalize_perf_gain(outcome.metrics[key])

        # Parse measured_gain string if present
        if outcome.measured_gain:
            gain = outcome.measured_gain.lower()
            # Try to extract multiplier like "1.9x" or "2x"
            import re
            match = re.search(r'(\d+\.?\d*)x', gain)
            if match:
                multiplier = float(match.group(1))
                return normalize_perf_gain(multiplier)

            # Try to extract percentage like "30% faster"
            match = re.search(r'(\d+)%', gain)
            if match:
                pct = float(match.group(1)) / 100.0
                return min(1.0, 0.5 + pct * 0.5)

        # Default based on success
        return 0.7 if outcome.success else 0.3

    def _compute_human_score(self, outcome: Outcome) -> float:
        """Compute human rating score."""
        # Explicit numeric rating
        if outcome.human_rating is not None:
            # Assume 0-1 scale or 1-5 scale
            rating = outcome.human_rating
            if rating > 1:  # 1-5 scale
                return (rating - 1) / 4.0
            return rating

        # Parse feedback string
        if outcome.human_feedback:
            feedback = outcome.human_feedback.lower()

            # Strong positive signals
            if any(w in feedback for w in ["amazing", "perfect", "excellent", "ship it", "love it"]):
                return 0.95

            # Positive signals
            if any(w in feedback for w in ["good", "nice", "works", "great", "👍"]):
                return 0.8

            # Neutral
            if any(w in feedback for w in ["ok", "fine", "acceptable"]):
                return 0.6

            # Negative signals
            if any(w in feedback for w in ["bad", "wrong", "broken", "fix", "👎"]):
                return 0.3

            # Strong negative
            if any(w in feedback for w in ["terrible", "awful", "revert", "disaster"]):
                return 0.1

        # Default neutral
        return 0.5

    def _compute_stability_score(self, outcome: Outcome) -> float:
        """Compute stability score based on tests and regressions."""
        score = 1.0

        # Tests passed/failed
        if outcome.tests_passed is not None:
            if not outcome.tests_passed:
                score *= 0.3  # Big penalty for failing tests

        # Regressions
        if outcome.regressions:
            # Penalty per regression, capped
            regression_penalty = min(0.5, len(outcome.regressions) * 0.15)
            score *= (1.0 - regression_penalty)

        # Side effects (minor penalty)
        if outcome.side_effects:
            side_effect_penalty = min(0.2, len(outcome.side_effects) * 0.05)
            score *= (1.0 - side_effect_penalty)

        # Overall success
        if not outcome.success:
            score *= 0.5

        return max(0.0, score)

    def score_and_update(self, interaction: InteractionLog) -> float:
        """Score an interaction and update its reward field.

        Args:
            interaction: The interaction to score

        Returns:
            The computed reward
        """
        signal = self.score(interaction)
        interaction.reward = signal.total_reward
        return signal.total_reward


# =============================================================================
# Utility Functions
# =============================================================================

def normalize_perf_gain(multiplier: float, baseline: float = 1.0) -> float:
    """Normalize performance gain to [0, 1] scale.

    Args:
        multiplier: Performance multiplier (e.g., 1.9 for 1.9x speedup)
        baseline: Expected baseline (default 1.0 = no change)

    Returns:
        Normalized score in [0, 1]

    Examples:
        0.5x speedup → 0.3  (regression)
        1.0x speedup → 0.5  (no change)
        2.0x speedup → 0.75
        5.0x speedup → 0.9
        10x+ speedup → 0.95
    """
    if multiplier <= 0:
        return 0.0

    # Log scale for gains, linear for losses
    if multiplier < baseline:
        # Regression: linear interpolation from 0 to 0.5
        return 0.5 * (multiplier / baseline)
    else:
        # Gain: logarithmic scaling
        import math
        # Map 1x->0.5, 2x->0.75, 10x->0.95
        log_gain = math.log10(multiplier)
        return min(0.95, 0.5 + log_gain * 0.45)


def compute_reward(
    perf_improvement: float = 0.5,
    human_rating: float = 0.5,
    stability_score: float = 1.0,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Quick way to compute a reward.

    Args:
        perf_improvement: [0, 1] normalized performance gain
        human_rating: [0, 1] human rating
        stability_score: [0, 1] stability score
        weights: Optional weight overrides

    Returns:
        Weighted reward in [0, 1]
    """
    weights = weights or {"perf": 0.5, "human": 0.3, "stability": 0.2}

    return (
        weights.get("perf", 0.5) * perf_improvement +
        weights.get("human", 0.3) * human_rating +
        weights.get("stability", 0.2) * stability_score
    )


# =============================================================================
# Pre-configured Scorers for Common Task Types
# =============================================================================

# For performance-critical tasks (kernel tuning, optimization)
PERF_FOCUSED_WEIGHTS = {"perf": 0.6, "human": 0.2, "stability": 0.2}

# For stability-critical tasks (production changes, safety)
STABILITY_FOCUSED_WEIGHTS = {"perf": 0.2, "human": 0.3, "stability": 0.5}

# For creative tasks (graphics, UX)
HUMAN_FOCUSED_WEIGHTS = {"perf": 0.3, "human": 0.5, "stability": 0.2}

# Default balanced
BALANCED_WEIGHTS = {"perf": 0.5, "human": 0.3, "stability": 0.2}


def get_scorer_for_task(task_type: str) -> RewardScorer:
    """Get a pre-configured scorer for a task type.

    Args:
        task_type: The type of task

    Returns:
        Configured RewardScorer
    """
    task_weights = {
        "kernel_tuning": PERF_FOCUSED_WEIGHTS,
        "optimization": PERF_FOCUSED_WEIGHTS,
        "benchmark": PERF_FOCUSED_WEIGHTS,
        "graphics_experiment": HUMAN_FOCUSED_WEIGHTS,
        "visualization": HUMAN_FOCUSED_WEIGHTS,
        "ux_improvement": HUMAN_FOCUSED_WEIGHTS,
        "production_change": STABILITY_FOCUSED_WEIGHTS,
        "safety_fix": STABILITY_FOCUSED_WEIGHTS,
        "fpga_firmware": STABILITY_FOCUSED_WEIGHTS,
    }

    return RewardScorer(
        default_weights=BALANCED_WEIGHTS,
        task_weights=task_weights,
    )
