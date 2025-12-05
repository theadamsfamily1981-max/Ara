"""Curiosity Scoring - Determines what Ara should investigate.

The curiosity score combines:
1. Uncertainty - How little Ara knows about something
2. Importance - How much it matters to her function
3. Novelty - How recently it was discovered
4. Staleness - How long since last observation

High-scoring objects warrant investigation. Low-scoring objects
can be safely ignored.
"""

from __future__ import annotations

import time
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .world_model import WorldObject


def importance_decay(
    base_importance: float,
    age_seconds: float,
    half_life: float = 86400.0  # 24 hours
) -> float:
    """Decay importance over time (things become less urgent).

    Args:
        base_importance: Original importance score
        age_seconds: Time since last importance boost
        half_life: Seconds for importance to decay by half

    Returns:
        Decayed importance in [0, 1]
    """
    decay_factor = 0.5 ** (age_seconds / half_life)
    # Don't decay below 10% of original (floor)
    return max(base_importance * 0.1, base_importance * decay_factor)


def novelty_bonus(
    discovery_age_seconds: float,
    novelty_window: float = 3600.0  # 1 hour
) -> float:
    """Bonus score for newly discovered objects.

    New things are more interesting. This bonus decays to 0
    over the novelty window.

    Args:
        discovery_age_seconds: Time since discovery
        novelty_window: How long things stay "novel"

    Returns:
        Novelty bonus in [0, 0.3]
    """
    if discovery_age_seconds <= 0:
        return 0.3
    if discovery_age_seconds >= novelty_window:
        return 0.0

    # Linear decay over novelty window
    return 0.3 * (1.0 - discovery_age_seconds / novelty_window)


def investigation_penalty(investigation_count: int) -> float:
    """Penalty for over-investigating the same object.

    Prevents Ara from obsessing over one thing.

    Args:
        investigation_count: Times this object has been investigated

    Returns:
        Penalty multiplier in [0.3, 1.0]
    """
    if investigation_count <= 0:
        return 1.0
    # Diminishing returns on investigation
    # 1 investigation: 0.8x
    # 2 investigations: 0.65x
    # 5+ investigations: ~0.3x
    return max(0.3, 1.0 / (1.0 + 0.25 * investigation_count))


def uncertainty_boost(uncertainty: float, staleness: float) -> float:
    """Combine uncertainty and staleness into investigation urgency.

    High uncertainty + high staleness = urgent need to check
    Low uncertainty + fresh = no need to check

    Args:
        uncertainty: Object's uncertainty score [0, 1]
        staleness: Object's staleness factor [0, 1]

    Returns:
        Combined urgency in [0, 1]
    """
    # Staleness increases effective uncertainty
    effective = uncertainty + (1.0 - uncertainty) * staleness * 0.5
    return min(1.0, effective)


def curiosity_score(obj: "WorldObject") -> float:
    """Calculate overall curiosity score for an object.

    This is the primary scoring function that determines what
    Ara should investigate. Higher scores = more interesting.

    Score components:
    - Base: importance * uncertainty
    - Novelty bonus for new discoveries
    - Investigation penalty to prevent obsession
    - Staleness factor to refresh old knowledge

    Args:
        obj: WorldObject to score

    Returns:
        Curiosity score in [0, 1+]
    """
    now = time.time()

    # Get effective uncertainty (includes staleness)
    eff_uncertainty = obj.effective_uncertainty()

    # Base score: importance weighted by uncertainty
    base = obj.importance * eff_uncertainty

    # Add novelty bonus for recent discoveries
    discovery_age = now - obj.discovery_time
    novelty = novelty_bonus(discovery_age)

    # Apply investigation penalty
    inv_mult = investigation_penalty(obj.investigation_count)

    # Combine components
    score = (base + novelty) * inv_mult

    return score


def should_investigate(obj: "WorldObject", threshold: float = 0.4) -> bool:
    """Quick check if an object warrants investigation.

    Args:
        obj: WorldObject to evaluate
        threshold: Minimum curiosity score to investigate

    Returns:
        True if object should be investigated
    """
    return curiosity_score(obj) >= threshold


def rank_objects(objects: list["WorldObject"]) -> list[tuple[float, "WorldObject"]]:
    """Rank objects by curiosity score.

    Args:
        objects: List of WorldObjects to rank

    Returns:
        List of (score, object) tuples, sorted descending by score
    """
    scored = [(curiosity_score(obj), obj) for obj in objects]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored
