"""
Emotion-based neuromodulation controller.

Modulates learning parameters based on predicted emotional state:
- Arousal ↑ → Temperature ↑ (softer, more exploratory policy)
- Valence ↓ → LR ↓ (conservative updates when negative affect)

All modulations are bounded and can be vetoed by PGU.
"""

import torch
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import warnings

from .head import EmotionPrediction


@dataclass
class ControlModulation:
    """Container for control modulations."""
    lr_multiplier: float = 1.0
    temperature_multiplier: float = 1.0
    weight: float = 1.0  # Overall modulation weight
    reason: str = ""


class EmotionController:
    """
    Emotion-based controller for neuromodulation.

    Safely modulates learning parameters based on emotional state.
    """

    def __init__(
        self,
        arousal_temp_coupling: Tuple[float, float] = (0.8, 1.3),
        valence_lr_coupling: Tuple[float, float] = (0.7, 1.2),
        controller_weight: float = 0.3,
        jerk_threshold: float = 0.1,
        confidence_threshold: float = 0.5,
    ):
        """
        Args:
            arousal_temp_coupling: (min_mult, max_mult) for temperature modulation
            valence_lr_coupling: (min_mult, max_mult) for LR modulation
            controller_weight: Global controller weight (blend factor)
            jerk_threshold: Maximum allowed jerk (||Δ²state||)
            confidence_threshold: Minimum confidence for active control
        """
        self.arousal_temp_coupling = arousal_temp_coupling
        self.valence_lr_coupling = valence_lr_coupling
        self.controller_weight = controller_weight
        self.jerk_threshold = jerk_threshold
        self.confidence_threshold = confidence_threshold

        # State history for jerk computation
        self.prev_valence: Optional[float] = None
        self.prev_arousal: Optional[float] = None
        self.prev_prev_valence: Optional[float] = None
        self.prev_prev_arousal: Optional[float] = None

    def modulate_policy(
        self,
        fdt_metrics: Dict[str, float],
        emotion: EmotionPrediction,
        base_lr: float = 1.0,
        base_temperature: float = 1.0,
    ) -> ControlModulation:
        """
        Compute policy modulations based on emotion and FDT state.

        Args:
            fdt_metrics: FDT metrics (EPR, grad variance, etc.)
            emotion: Predicted emotional state
            base_lr: Base learning rate
            base_temperature: Base temperature

        Returns:
            ControlModulation with multipliers
        """
        # Extract emotion values (aggregate if sequence)
        if emotion.valence is not None:
            if emotion.valence.dim() > 1:
                valence = emotion.valence.mean().item()
            else:
                valence = emotion.valence.mean().item()
        else:
            valence = 0.0

        if emotion.arousal is not None:
            if emotion.arousal.dim() > 1:
                arousal = emotion.arousal.mean().item()
            else:
                arousal = emotion.arousal.mean().item()
        else:
            arousal = 0.5

        if emotion.confidence is not None:
            if emotion.confidence.dim() > 1:
                confidence = emotion.confidence.mean().item()
            else:
                confidence = emotion.confidence.mean().item()
        else:
            confidence = 1.0

        # Check confidence threshold
        if confidence < self.confidence_threshold:
            # Low confidence: reduce controller weight
            warnings.warn(
                f"Low emotion confidence ({confidence:.2f}). "
                f"Reducing controller weight."
            )
            effective_weight = self.controller_weight * 0.5
        else:
            effective_weight = self.controller_weight

        # Compute jerk (second derivative of state)
        jerk = 0.0
        if self.prev_valence is not None and self.prev_prev_valence is not None:
            valence_jerk = abs(
                valence - 2 * self.prev_valence + self.prev_prev_valence
            )
            arousal_jerk = abs(
                arousal - 2 * self.prev_arousal + self.prev_prev_arousal
            )
            jerk = (valence_jerk ** 2 + arousal_jerk ** 2) ** 0.5

        # Check jerk threshold
        if jerk > self.jerk_threshold:
            warnings.warn(
                f"High jerk detected ({jerk:.3f}). Clamping modulation."
            )
            # Reduce modulation strength
            effective_weight *= 0.5

        # Arousal → Temperature coupling
        # High arousal → higher temperature (more exploration)
        # arousal ∈ [0, 1], map to [min_mult, max_mult]
        temp_mult_raw = (
            self.arousal_temp_coupling[0] +
            arousal * (self.arousal_temp_coupling[1] - self.arousal_temp_coupling[0])
        )

        # Valence → LR coupling
        # High valence → higher LR (confident learning)
        # Low valence → lower LR (conservative)
        # valence ∈ [-1, 1], map to [0, 1] first
        valence_normalized = (valence + 1.0) / 2.0
        lr_mult_raw = (
            self.valence_lr_coupling[0] +
            valence_normalized * (self.valence_lr_coupling[1] - self.valence_lr_coupling[0])
        )

        # Blend with baseline (controlled by effective_weight)
        temp_mult = 1.0 + effective_weight * (temp_mult_raw - 1.0)
        lr_mult = 1.0 + effective_weight * (lr_mult_raw - 1.0)

        # Clamp to bounds
        temp_mult = max(self.arousal_temp_coupling[0], min(temp_mult, self.arousal_temp_coupling[1]))
        lr_mult = max(self.valence_lr_coupling[0], min(lr_mult, self.valence_lr_coupling[1]))

        # Update history for jerk computation
        self.prev_prev_valence = self.prev_valence
        self.prev_prev_arousal = self.prev_arousal
        self.prev_valence = valence
        self.prev_arousal = arousal

        modulation = ControlModulation(
            lr_multiplier=lr_mult,
            temperature_multiplier=temp_mult,
            weight=effective_weight,
            reason=f"V={valence:.2f}, A={arousal:.2f}, C={confidence:.2f}, jerk={jerk:.3f}",
        )

        return modulation

    def reset_state(self):
        """Reset internal state history."""
        self.prev_valence = None
        self.prev_arousal = None
        self.prev_prev_valence = None
        self.prev_prev_arousal = None
