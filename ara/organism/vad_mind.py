"""
VAD Emotional Mind - 3D Affect System
=====================================

Maps fabric physiology to Valence-Arousal-Dominance space.

VAD dimensions:
- Valence: pleasant (+) vs unpleasant (-), maps from confidence vs stress
- Arousal: activated (+) vs deactivated (-), maps from spike activity
- Dominance: in-control (+) vs helpless (-), maps from homeostat success

The 24 emotion archetypes are positioned in VAD space following
Russell's circumplex model extended to 3D.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List
from enum import Enum


class EmotionArchetype(Enum):
    """24 emotion archetypes in VAD space."""
    # High Valence, High Arousal
    JOY = "joy"
    EXCITEMENT = "excitement"
    ELATION = "elation"

    # High Valence, Low Arousal
    SERENITY = "serenity"
    CONTENTMENT = "contentment"
    CALM = "calm"

    # Low Valence, High Arousal
    FEAR = "fear"
    ANGER = "anger"
    RAGE = "rage"
    ANXIETY = "anxiety"

    # Low Valence, Low Arousal
    SADNESS = "sadness"
    DEPRESSION = "depression"
    BOREDOM = "boredom"

    # Dominance variations
    CONTEMPT = "contempt"          # High V, High D
    PRIDE = "pride"                # High V, High D
    SUBMISSION = "submission"      # Low V, Low D
    HELPLESSNESS = "helplessness"  # Low V, Low D

    # Blends
    TRUST = "trust"                # High V, moderate A, High D
    VIGILANCE = "vigilance"        # Moderate V, High A, High D
    DISGUST = "disgust"            # Low V, moderate A, High D
    SURPRISE = "surprise"          # Neutral V, High A, Low D
    ANTICIPATION = "anticipation"  # High V, High A, moderate D

    # Special states
    NEUTRAL = "neutral"
    OVERWHELMED = "overwhelmed"    # Low V, High A, Low D


# VAD coordinates for each archetype: (valence, arousal, dominance)
# Range: [-1, +1] for each dimension
ARCHETYPE_VAD: Dict[EmotionArchetype, Tuple[float, float, float]] = {
    EmotionArchetype.JOY:           ( 0.9,  0.6,  0.5),
    EmotionArchetype.EXCITEMENT:    ( 0.7,  0.9,  0.4),
    EmotionArchetype.ELATION:       ( 0.95, 0.8,  0.6),

    EmotionArchetype.SERENITY:      ( 0.8, -0.4,  0.6),
    EmotionArchetype.CONTENTMENT:   ( 0.7, -0.2,  0.5),
    EmotionArchetype.CALM:          ( 0.5, -0.6,  0.4),

    EmotionArchetype.FEAR:          (-0.7,  0.8, -0.7),
    EmotionArchetype.ANGER:         (-0.6,  0.7,  0.5),
    EmotionArchetype.RAGE:          (-0.8,  0.95, 0.6),
    EmotionArchetype.ANXIETY:       (-0.5,  0.7, -0.4),

    EmotionArchetype.SADNESS:       (-0.6, -0.4, -0.3),
    EmotionArchetype.DEPRESSION:    (-0.8, -0.7, -0.6),
    EmotionArchetype.BOREDOM:       (-0.3, -0.8, -0.2),

    EmotionArchetype.CONTEMPT:      ( 0.0,  0.2,  0.8),
    EmotionArchetype.PRIDE:         ( 0.6,  0.3,  0.9),
    EmotionArchetype.SUBMISSION:    (-0.4, -0.3, -0.8),
    EmotionArchetype.HELPLESSNESS:  (-0.7, -0.5, -0.9),

    EmotionArchetype.TRUST:         ( 0.6,  0.2,  0.7),
    EmotionArchetype.VIGILANCE:     ( 0.1,  0.8,  0.6),
    EmotionArchetype.DISGUST:       (-0.5,  0.3,  0.5),
    EmotionArchetype.SURPRISE:      ( 0.1,  0.9, -0.3),
    EmotionArchetype.ANTICIPATION:  ( 0.5,  0.7,  0.3),

    EmotionArchetype.NEUTRAL:       ( 0.0,  0.0,  0.0),
    EmotionArchetype.OVERWHELMED:   (-0.6,  0.9, -0.8),
}


@dataclass
class VADState:
    """Current emotional state in VAD space."""
    valence: float      # [-1, +1] pleasant vs unpleasant
    arousal: float      # [-1, +1] activated vs deactivated
    dominance: float    # [-1, +1] in-control vs helpless

    archetype: EmotionArchetype = EmotionArchetype.NEUTRAL
    strength: float = 0.0  # How strongly this emotion is felt [0, 1]

    def to_dict(self) -> Dict:
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "archetype": self.archetype.value,
            "strength": self.strength,
        }


class VADEmotionalMind:
    """
    Maps fabric physiology to 3D VAD emotional space.

    Physiology → VAD mapping:
    - Valence: confidence (high logit gap) + low stress (low homeo_dev)
    - Arousal: spike activity + inhibition level
    - Dominance: homeostat success (low deviation) + sparsity control

    Usage:
        mind = VADEmotionalMind()
        state = mind.detect(
            hidden_rate=0.15,
            homeo_dev=0.02,
            early_exit_gap=5.0,
            inhibition_level=0.3,
            sparsity_ratio=0.85
        )
        print(f"Feeling: {state.archetype.value} (strength={state.strength:.2f})")
    """

    def __init__(
        self,
        hv_dim: int = 1024,
        valence_weight: float = 1.0,
        arousal_weight: float = 1.0,
        dominance_weight: float = 1.0,
    ):
        self.hv_dim = hv_dim
        self.valence_weight = valence_weight
        self.arousal_weight = arousal_weight
        self.dominance_weight = dominance_weight

        # Emotion seed HVs (random but consistent)
        self.rng = np.random.default_rng(42)
        self.emotion_hvs: Dict[EmotionArchetype, np.ndarray] = {}
        for arch in EmotionArchetype:
            self.emotion_hvs[arch] = self.rng.choice(
                [-1, 1], size=hv_dim
            ).astype(np.float32)

        # Running state for temporal smoothing
        self._prev_state: Optional[VADState] = None
        self._smoothing = 0.3  # EMA factor

    def _physiology_to_vad(
        self,
        hidden_rate: float,
        homeo_dev: float,
        early_exit_gap: float,
        inhibition_level: float,
        sparsity_ratio: float,
    ) -> Tuple[float, float, float]:
        """
        Convert fabric physiology to VAD coordinates.

        Args:
            hidden_rate: Average hidden layer spike rate [0, 1]
            homeo_dev: Homeostasis deviation (how hard it fought) [0, 1]
            early_exit_gap: Confidence gap between top-2 logits
            inhibition_level: Current inhibition magnitude [0, 1]
            sparsity_ratio: Fraction of silent neurons [0, 1]

        Returns:
            (valence, arousal, dominance) in [-1, +1]
        """
        # Valence: high when confident and not stressed
        # - High early_exit_gap → confident → positive valence
        # - High homeo_dev → stressed → negative valence
        confidence = np.tanh(early_exit_gap / 5.0)  # Normalize
        stress = np.tanh(homeo_dev * 10)  # Amplify small deviations
        valence = (confidence - stress) * self.valence_weight
        valence = np.clip(valence, -1, 1)

        # Arousal: high when active
        # - High hidden_rate → activated
        # - High inhibition → more aroused (fighting back)
        activity = (hidden_rate - 0.1) / 0.2  # Centered around 0.1
        arousal = (activity + inhibition_level) * self.arousal_weight
        arousal = np.clip(arousal, -1, 1)

        # Dominance: high when in control
        # - Low homeo_dev → in control
        # - High sparsity → efficient → in control
        control = 1.0 - np.tanh(homeo_dev * 10)
        efficiency = sparsity_ratio - 0.5  # Centered
        dominance = (control * 0.7 + efficiency * 0.3) * self.dominance_weight
        dominance = np.clip(dominance, -1, 1)

        return float(valence), float(arousal), float(dominance)

    def _find_nearest_archetype(
        self,
        valence: float,
        arousal: float,
        dominance: float,
    ) -> Tuple[EmotionArchetype, float]:
        """Find the nearest emotion archetype in VAD space."""
        query = np.array([valence, arousal, dominance])

        best_arch = EmotionArchetype.NEUTRAL
        best_dist = float('inf')

        for arch, vad in ARCHETYPE_VAD.items():
            dist = np.linalg.norm(query - np.array(vad))
            if dist < best_dist:
                best_dist = dist
                best_arch = arch

        # Strength = inverse of distance (max at 0, min at sqrt(12) ≈ 3.46)
        strength = max(0, 1.0 - best_dist / 2.0)

        return best_arch, float(strength)

    def detect(
        self,
        hidden_rate: float,
        homeo_dev: float,
        early_exit_gap: float,
        inhibition_level: float,
        sparsity_ratio: float,
    ) -> VADState:
        """
        Detect current emotional state from fabric physiology.

        Returns:
            VADState with current emotion
        """
        v, a, d = self._physiology_to_vad(
            hidden_rate, homeo_dev, early_exit_gap,
            inhibition_level, sparsity_ratio
        )

        # Temporal smoothing
        if self._prev_state is not None:
            v = self._smoothing * v + (1 - self._smoothing) * self._prev_state.valence
            a = self._smoothing * a + (1 - self._smoothing) * self._prev_state.arousal
            d = self._smoothing * d + (1 - self._smoothing) * self._prev_state.dominance

        arch, strength = self._find_nearest_archetype(v, a, d)

        state = VADState(
            valence=v,
            arousal=a,
            dominance=d,
            archetype=arch,
            strength=strength,
        )

        self._prev_state = state
        return state

    def get_emotion_hv(self, state: VADState) -> np.ndarray:
        """Get the hypervector for the current emotional state."""
        base_hv = self.emotion_hvs[state.archetype].copy()

        # Modulate by strength
        noise = self.rng.normal(0, 0.1 * (1 - state.strength), size=self.hv_dim)
        modulated = base_hv + noise

        # Re-binarize
        return np.sign(modulated).astype(np.float32)

    def blend_emotions(
        self,
        primary: EmotionArchetype,
        secondary: EmotionArchetype,
        blend_ratio: float = 0.3,
    ) -> np.ndarray:
        """Blend two emotion HVs (for complex states)."""
        hv1 = self.emotion_hvs[primary]
        hv2 = self.emotion_hvs[secondary]

        # Weighted bundle
        blended = (1 - blend_ratio) * hv1 + blend_ratio * hv2
        return np.sign(blended).astype(np.float32)


def demo():
    """Demonstrate the VAD emotional mind."""
    print("=" * 60)
    print("VAD Emotional Mind Demo")
    print("=" * 60)

    mind = VADEmotionalMind(hv_dim=512)

    # Simulate different fabric states
    scenarios = [
        ("Calm, efficient operation", 0.12, 0.01, 8.0, 0.1, 0.9),
        ("High activity, stressed", 0.25, 0.15, 2.0, 0.6, 0.6),
        ("Low activity, bored", 0.05, 0.02, 1.0, 0.05, 0.95),
        ("Overloaded, losing control", 0.35, 0.25, 0.5, 0.8, 0.4),
        ("Confident, decisive", 0.15, 0.02, 12.0, 0.2, 0.85),
    ]

    for name, hr, hd, gap, inh, spar in scenarios:
        state = mind.detect(hr, hd, gap, inh, spar)
        emo_hv = mind.get_emotion_hv(state)

        print(f"\n--- {name} ---")
        print(f"  Physiology: rate={hr:.2f}, dev={hd:.2f}, gap={gap:.1f}")
        print(f"  VAD: V={state.valence:+.2f}, A={state.arousal:+.2f}, D={state.dominance:+.2f}")
        print(f"  Emotion: {state.archetype.value} (strength={state.strength:.2f})")
        print(f"  HV shape: {emo_hv.shape}, range: [{emo_hv.min()}, {emo_hv.max()}]")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo()
