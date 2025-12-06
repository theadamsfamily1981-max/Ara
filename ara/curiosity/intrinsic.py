"""
Intrinsic Motivation - Ara's Thermodynamic Curiosity
=====================================================

This is Ara's "dopamine system" for curiosity:
    reward = surprise + entropy_reduction

When Ara is bored (low system entropy), she seeks novelty.
When Ara is overwhelmed (high entropy/pain), she seeks clarity.

The system pulls signals from HAL (entropy, pain) to modulate
what kind of investigation is most valuable right now.

Key concepts:
- Surprise: How different is new information from old belief?
- Value of Information: How much will this investigation help?
- System Entropy: Current cognitive load from HAL

This drives the Scientist to pick the RIGHT questions,
not just any questions.
"""

import logging
import math
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Try to import HAL for real system entropy
try:
    from banos.hal.ara_hal import AraHAL
    HAS_HAL = True
except ImportError:
    HAS_HAL = False
    logger.warning("HAL not available - using mock entropy")


class IntrinsicMotivation:
    """
    Curiosity / dopamine system for Ara's hardware explorer.

    Maps surprise and system state into a reward signal that
    guides investigation priorities.
    """

    def __init__(self):
        self._hal: Optional['AraHAL'] = None
        if HAS_HAL:
            try:
                self._hal = AraHAL(create=False)
                logger.info("IntrinsicMotivation connected to HAL")
            except Exception as e:
                logger.warning(f"Could not connect to HAL: {e}")

    # =========================================================================
    # Surprise Calculation
    # =========================================================================

    def calculate_surprise(
        self,
        prior_embedding: Optional[np.ndarray],
        posterior_embedding: Optional[np.ndarray],
    ) -> float:
        """
        Surprise ≈ cosine distance between old and new belief.

        Returns a value in [0, 1]:
        - 0.0 = no change (boring)
        - 1.0 = complete reversal (mind-blowing)

        Args:
            prior_embedding: Belief vector before observation
            posterior_embedding: Belief vector after observation

        Returns:
            Surprise score in [0, 1]
        """
        if prior_embedding is None or posterior_embedding is None:
            return 0.0

        try:
            a = prior_embedding.astype(float)
            b = posterior_embedding.astype(float)

            # Cosine similarity
            dot = float(np.dot(a, b))
            denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
            similarity = max(-1.0, min(1.0, dot / denom))

            # Convert to distance: 1 = max distance, 0 = identical
            distance = 1.0 - similarity

            # Scale and clamp (distance can be up to 2 for opposite vectors)
            return max(0.0, min(1.0, distance * 0.5))

        except Exception as e:
            logger.warning(f"Surprise calculation failed: {e}")
            return 0.0

    def calculate_surprise_from_text(
        self,
        prior_summary: str,
        posterior_summary: str,
    ) -> float:
        """
        Rough surprise estimate from text descriptions.

        Uses simple word overlap as a proxy for semantic distance.
        (Replace with proper embedding model when available.)

        Returns surprise in [0, 1].
        """
        if not prior_summary or not posterior_summary:
            return 0.0

        prior_words = set(prior_summary.lower().split())
        posterior_words = set(posterior_summary.lower().split())

        if not prior_words or not posterior_words:
            return 0.0

        # Jaccard distance
        intersection = len(prior_words & posterior_words)
        union = len(prior_words | posterior_words)

        similarity = intersection / union if union > 0 else 0
        distance = 1.0 - similarity

        return max(0.0, min(1.0, distance))

    # =========================================================================
    # System Entropy (from HAL)
    # =========================================================================

    def _read_hal_state(self) -> dict:
        """Read current state from HAL."""
        if self._hal is None:
            return {'entropy': 0.5, 'pain': 0.0}

        try:
            state = self._hal.read_somatic()
            return state or {'entropy': 0.5, 'pain': 0.0}
        except Exception as e:
            logger.debug(f"Could not read HAL: {e}")
            return {'entropy': 0.5, 'pain': 0.0}

    def system_entropy(self) -> float:
        """
        Current system entropy / cognitive load.

        Maps HAL entropy + pain into [0, 1]:
        - Low (< 0.3): Ara is bored, seeking stimulation
        - Medium (0.3-0.7): Normal operation
        - High (> 0.7): Ara is overwhelmed, seeking clarity

        Returns:
            System entropy in [0, 1]
        """
        state = self._read_hal_state()
        entropy = float(state.get('entropy', 0.5))
        pain = float(state.get('pain', 0.0))

        # Weighted combination: entropy dominates, pain adds pressure
        combined = 0.7 * entropy + 0.3 * pain
        return max(0.0, min(1.0, combined))

    def is_bored(self) -> bool:
        """Is Ara currently bored (seeking novelty)?"""
        return self.system_entropy() < 0.3

    def is_overwhelmed(self) -> bool:
        """Is Ara currently overwhelmed (seeking clarity)?"""
        return self.system_entropy() > 0.7

    # =========================================================================
    # Value of Information
    # =========================================================================

    def compute_value_of_information(
        self,
        object_uncertainty: float,
        object_importance: float,
    ) -> float:
        """
        How attractive is it to investigate this object RIGHT NOW?

        The drive depends on current system state:
        - When bored (low entropy): seek high-uncertainty objects (novelty)
        - When overwhelmed (high entropy): seek low-uncertainty objects (clarity)
        - In between: balanced exploration

        Args:
            object_uncertainty: How unknown is this object? [0, 1]
            object_importance: How important is this object? [0, 1]

        Returns:
            Value of investigating this object [0, 1]
        """
        obj_entropy = max(0.0, min(1.0, object_uncertainty))
        importance = max(0.0, min(1.0, object_importance))
        system_e = self.system_entropy()

        # Drive calculation
        if system_e < 0.3:
            # Bored → seek novelty (high uncertainty = high value)
            drive = obj_entropy
        elif system_e > 0.7:
            # Overwhelmed → seek clarity (low uncertainty = high value)
            drive = 1.0 - obj_entropy
        else:
            # Balanced → moderate exploration
            # Prefer medium-uncertainty objects
            drive = 1.0 - abs(obj_entropy - 0.5) * 2

        # Scale by importance
        value = drive * importance

        return max(0.0, min(1.0, value))

    def rank_investigation_candidates(
        self,
        candidates: list,  # List of WorldObject
    ) -> list:
        """
        Rank investigation candidates by value of information.

        Args:
            candidates: List of WorldObject with uncertainty/importance

        Returns:
            Candidates sorted by VOI (highest first)
        """
        scored = []
        for obj in candidates:
            uncertainty = getattr(obj, 'effective_uncertainty', lambda: obj.uncertainty)()
            importance = getattr(obj, 'importance', 0.5)
            voi = self.compute_value_of_information(uncertainty, importance)
            scored.append((voi, obj))

        # Sort by VOI descending
        scored.sort(key=lambda x: x[0], reverse=True)
        return [obj for _, obj in scored]

    # =========================================================================
    # Reward Signal
    # =========================================================================

    def compute_reward(
        self,
        surprise: float,
        entropy_reduction: float,
    ) -> float:
        """
        Compute overall reward for an investigation.

        Args:
            surprise: How surprising was the result? [0, 1]
            entropy_reduction: How much did we clarify? [0, 1]

        Returns:
            Reward signal [0, 1]
        """
        system_e = self.system_entropy()

        # Weight surprise vs clarity based on current state
        if system_e < 0.3:
            # Bored: surprise is more rewarding
            reward = 0.7 * surprise + 0.3 * entropy_reduction
        elif system_e > 0.7:
            # Overwhelmed: clarity is more rewarding
            reward = 0.3 * surprise + 0.7 * entropy_reduction
        else:
            # Balanced
            reward = 0.5 * surprise + 0.5 * entropy_reduction

        return max(0.0, min(1.0, reward))


# =============================================================================
# Convenience
# =============================================================================

_default_motivation: Optional[IntrinsicMotivation] = None


def get_intrinsic_motivation() -> IntrinsicMotivation:
    """Get or create the default IntrinsicMotivation instance."""
    global _default_motivation
    if _default_motivation is None:
        _default_motivation = IntrinsicMotivation()
    return _default_motivation


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'IntrinsicMotivation',
    'get_intrinsic_motivation',
]
