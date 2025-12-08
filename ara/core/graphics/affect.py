"""
Ara Affect Computation - Soul State to Expression
=================================================

Computes affect dimensions from HTC resonance and reward history.

Affect Dimensions:
- valence: Emotional tone (-1 negative to +1 positive)
- arousal: Activation level (0 calm to 1 activated)
- certainty: Confidence (0 confused to 1 confident)
- focus: Concentration (0 scattered to 1 focused)

These drive avatar expression and UI theming.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np


@dataclass
class AffectState:
    """Current affect state."""
    valence: float = 0.0    # -1 to +1
    arousal: float = 0.5    # 0 to 1
    certainty: float = 0.5  # 0 to 1
    focus: float = 0.5      # 0 to 1

    def to_dict(self) -> Dict[str, float]:
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "certainty": self.certainty,
            "focus": self.focus,
        }

    @classmethod
    def neutral(cls) -> "AffectState":
        """Return neutral affect state."""
        return cls(valence=0.0, arousal=0.3, certainty=0.5, focus=0.5)


def affect_from_history(
    resonance_hist: List[np.ndarray],
    reward_hist: List[int],
    window: int = 20,
) -> Dict[str, float]:
    """
    Compute affect from resonance and reward history.

    Args:
        resonance_hist: List of resonance vectors from HTC
        reward_hist: List of reward signals
        window: History window for computation

    Returns:
        Dict with valence, arousal, certainty, focus
    """
    if not reward_hist:
        return AffectState.neutral().to_dict()

    # Use recent history
    recent_rewards = reward_hist[-window:]
    recent_resonance = resonance_hist[-window:] if resonance_hist else []

    # Valence: average reward mapped through tanh
    avg_reward = sum(recent_rewards) / len(recent_rewards)
    valence = math.tanh(avg_reward / 50.0)  # Normalize assuming [-127, 127]

    # Arousal: reward variance (high variance = high arousal)
    if len(recent_rewards) > 1:
        mean_r = sum(recent_rewards) / len(recent_rewards)
        variance = sum((r - mean_r) ** 2 for r in recent_rewards) / len(recent_rewards)
        std_r = math.sqrt(variance)
        arousal = math.tanh(std_r / 30.0)
    else:
        arousal = 0.3

    # Certainty: from resonance consistency (if available)
    if recent_resonance and len(recent_resonance) > 1:
        # Compute variance of resonance peaks
        peaks = [float(np.max(r)) if len(r) > 0 else 0.5 for r in recent_resonance]
        mean_peak = sum(peaks) / len(peaks)
        peak_var = sum((p - mean_peak) ** 2 for p in peaks) / len(peaks)
        # Low variance = high certainty
        certainty = 1.0 - math.tanh(math.sqrt(peak_var) * 5)
    else:
        certainty = 0.5

    # Focus: max resonance value (if available)
    if recent_resonance:
        last_res = recent_resonance[-1]
        if len(last_res) > 0:
            focus = float(np.max(last_res))
        else:
            focus = 0.5
    else:
        focus = 0.5

    return {
        "valence": max(-1.0, min(1.0, valence)),
        "arousal": max(0.0, min(1.0, arousal)),
        "certainty": max(0.0, min(1.0, certainty)),
        "focus": max(0.0, min(1.0, focus)),
    }


def affect_from_htc_state(
    resonance_profile: Dict[str, float],
    reward_history: List[float],
    attractor_entropy: float = 0.5,
) -> Dict[str, float]:
    """
    Alternative affect computation from HTC resonance profile.

    Args:
        resonance_profile: Dict mapping attractor names to resonance scores
        reward_history: Recent reward values
        attractor_entropy: Entropy of attractor distribution (0-1)

    Returns:
        Dict with valence, arousal, certainty, focus
    """
    # Valence from reward history
    if reward_history:
        recent = reward_history[-20:]
        avg_reward = sum(recent) / len(recent)
        valence = math.tanh(avg_reward * 2)
    else:
        valence = 0.0

    # Arousal from reward variance
    if len(reward_history) > 1:
        recent = reward_history[-20:]
        mean = sum(recent) / len(recent)
        variance = sum((r - mean) ** 2 for r in recent) / len(recent)
        arousal = math.tanh(math.sqrt(variance) * 3)
    else:
        arousal = 0.3

    # Certainty from entropy (low entropy = high certainty)
    certainty = 1.0 - attractor_entropy

    # Focus from max resonance
    if resonance_profile:
        focus = max(resonance_profile.values())
    else:
        focus = 0.5

    return {
        "valence": max(-1.0, min(1.0, valence)),
        "arousal": max(0.0, min(1.0, arousal)),
        "certainty": max(0.0, min(1.0, certainty)),
        "focus": max(0.0, min(1.0, focus)),
    }


def blend_affect(
    current: Dict[str, float],
    target: Dict[str, float],
    alpha: float = 0.3,
) -> Dict[str, float]:
    """
    Smoothly blend current affect toward target.

    Args:
        current: Current affect state
        target: Target affect state
        alpha: Blend factor (0 = no change, 1 = instant)

    Returns:
        Blended affect state
    """
    return {
        key: current.get(key, 0.5) * (1 - alpha) + target.get(key, 0.5) * alpha
        for key in ["valence", "arousal", "certainty", "focus"]
    }


__all__ = [
    'AffectState',
    'affect_from_history',
    'affect_from_htc_state',
    'blend_affect',
]
