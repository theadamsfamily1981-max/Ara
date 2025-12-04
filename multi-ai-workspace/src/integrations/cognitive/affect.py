"""Phase 3: Affect - Homeostatic Core & Appraisal Engine.

The Affective System implements emotional regulation and homeostasis,
allowing Ara to maintain stable internal states and respond appropriately
to emotionally-valenced situations.

Key Components:

    HomeostaticCore: Maintains internal equilibrium
        - Monitors "vital signs" (energy, attention, stress)
        - Triggers corrective actions when out of balance
        - Implements drive reduction (hunger, fatigue, etc.)

    AppraisalEngine: Evaluates emotional significance
        - Assesses valence (positive/negative)
        - Assesses arousal (intensity)
        - Generates appropriate emotional responses

Homeostatic Variables:
    Energy: Processing capacity available
    Attention: Focus resources available
    Stress: Accumulated cognitive load
    Curiosity: Drive to explore/learn
    Social: Need for interaction

Appraisal Dimensions (based on emotion research):
    Valence: Positive ←→ Negative
    Arousal: Calm ←→ Excited
    Dominance: Controlled ←→ In-control

This implements affective processing from tfan.cognition.affect.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
from enum import Enum, auto
import time
import warnings
import sys
from pathlib import Path

# Add TFAN to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent))

# Try to import TFAN affect modules
_TFAN_AFFECT_AVAILABLE = False
try:
    from tfan.cognition.affect import HomeostaticCore as TFANHomeostaticCore
    from tfan.cognition.affect import AppraisalEngine as TFANAppraisalEngine
    _TFAN_AFFECT_AVAILABLE = True
except ImportError:
    pass


class EmotionalState(Enum):
    """Discrete emotional states."""
    NEUTRAL = auto()
    CURIOUS = auto()
    ENGAGED = auto()
    CONCERNED = auto()
    STRESSED = auto()
    FATIGUED = auto()
    RECOVERING = auto()
    FOCUSED = auto()
    OVERWHELMED = auto()


class DriveType(Enum):
    """Homeostatic drives."""
    ENERGY = auto()      # Processing capacity
    ATTENTION = auto()   # Focus resources
    STRESS = auto()      # Cognitive load (to minimize)
    CURIOSITY = auto()   # Exploration drive
    SOCIAL = auto()      # Interaction need
    REST = auto()        # Recovery need


@dataclass
class HomeostaticState:
    """Current homeostatic state."""
    energy: float           # [0, 1] - available processing energy
    attention: float        # [0, 1] - available attention
    stress: float           # [0, 1] - accumulated stress (lower is better)
    curiosity: float        # [0, 1] - exploration drive
    social: float           # [0, 1] - social engagement need
    rest_need: float        # [0, 1] - need for recovery

    is_balanced: bool
    imbalanced_drives: List[DriveType]
    recommended_action: str
    timestamp: float = field(default_factory=time.time)

    def to_tensor(self) -> torch.Tensor:
        """Convert to tensor representation."""
        return torch.tensor([
            self.energy,
            self.attention,
            1.0 - self.stress,  # Invert so high is good
            self.curiosity,
            self.social,
            1.0 - self.rest_need,
        ])


@dataclass
class AppraisalResult:
    """Result of emotional appraisal."""
    valence: float          # [-1, 1] - negative to positive
    arousal: float          # [0, 1] - calm to excited
    dominance: float        # [-1, 1] - controlled to in-control

    emotional_state: EmotionalState
    emotion_label: str
    confidence: float
    action_tendency: str    # Approach, avoid, observe, etc.

    metrics: Dict[str, float] = field(default_factory=dict)


class HomeostaticCore:
    """
    The Homeostatic Core - Maintains internal equilibrium.

    Monitors "vital signs" and triggers corrective actions when
    the system drifts from optimal operating parameters.

    This is inspired by biological homeostasis - the body's
    ability to maintain stable internal conditions.

    Args:
        energy_decay: Rate of energy consumption
        attention_decay: Rate of attention fatigue
        stress_accumulation: Rate of stress buildup
        recovery_rate: Rate of recovery during rest
        balance_threshold: Threshold for considering balanced
        device: Compute device
    """

    def __init__(
        self,
        energy_decay: float = 0.01,
        attention_decay: float = 0.02,
        stress_accumulation: float = 0.03,
        recovery_rate: float = 0.05,
        balance_threshold: float = 0.3,
        device: str = "cpu",
    ):
        self.energy_decay = energy_decay
        self.attention_decay = attention_decay
        self.stress_accumulation = stress_accumulation
        self.recovery_rate = recovery_rate
        self.balance_threshold = balance_threshold
        self.device = device

        # TFAN core if available
        self.tfan_core = None
        if _TFAN_AFFECT_AVAILABLE:
            try:
                self.tfan_core = TFANHomeostaticCore()
            except Exception as e:
                warnings.warn(f"Failed to init TFAN homeostatic core: {e}")

        # Internal state - all in [0, 1]
        self._energy = 1.0
        self._attention = 1.0
        self._stress = 0.0
        self._curiosity = 0.5
        self._social = 0.5
        self._rest_need = 0.0

        # Setpoints (optimal values)
        self._setpoints = {
            DriveType.ENERGY: 0.7,
            DriveType.ATTENTION: 0.7,
            DriveType.STRESS: 0.2,
            DriveType.CURIOSITY: 0.5,
            DriveType.SOCIAL: 0.5,
            DriveType.REST: 0.2,
        }

        # History for trend detection
        self._history: List[HomeostaticState] = []

    def update(
        self,
        cognitive_load: float = 0.5,
        social_interaction: bool = False,
        novel_input: bool = False,
        recovery_mode: bool = False,
    ) -> HomeostaticState:
        """
        Update homeostatic state based on current activity.

        Args:
            cognitive_load: Current processing load [0, 1]
            social_interaction: Whether social interaction is occurring
            novel_input: Whether novel/surprising input was received
            recovery_mode: Whether system is in recovery/rest mode

        Returns:
            Updated HomeostaticState
        """
        if self.tfan_core is not None:
            # Use TFAN core
            state = self.tfan_core.update(
                cognitive_load=cognitive_load,
                social_interaction=social_interaction,
                novel_input=novel_input,
                recovery_mode=recovery_mode,
            )
            return self._convert_tfan_state(state)

        # Fallback implementation

        # Energy dynamics
        energy_cost = cognitive_load * self.energy_decay
        if recovery_mode:
            self._energy = min(1.0, self._energy + self.recovery_rate)
        else:
            self._energy = max(0.0, self._energy - energy_cost)

        # Attention dynamics
        if recovery_mode:
            self._attention = min(1.0, self._attention + self.recovery_rate)
        else:
            attention_cost = cognitive_load * self.attention_decay
            self._attention = max(0.0, self._attention - attention_cost)

        # Stress dynamics
        stress_delta = cognitive_load * self.stress_accumulation
        if recovery_mode:
            self._stress = max(0.0, self._stress - self.recovery_rate * 2)
        else:
            self._stress = min(1.0, self._stress + stress_delta)

        # Curiosity dynamics
        if novel_input:
            # Novel input satisfies curiosity but also increases it
            self._curiosity = min(1.0, self._curiosity + 0.1)
        else:
            # Curiosity grows when unstimulated
            self._curiosity = min(1.0, self._curiosity + 0.01)

        # Social dynamics
        if social_interaction:
            self._social = max(0.0, self._social - 0.1)  # Satisfied
        else:
            self._social = min(1.0, self._social + 0.01)  # Grows

        # Rest need dynamics
        if recovery_mode:
            self._rest_need = max(0.0, self._rest_need - self.recovery_rate * 2)
        else:
            # Rest need grows with stress and low energy
            rest_growth = 0.01 + 0.02 * self._stress + 0.02 * (1 - self._energy)
            self._rest_need = min(1.0, self._rest_need + rest_growth)

        # Check balance
        imbalanced = self._check_imbalances()
        is_balanced = len(imbalanced) == 0
        action = self._recommend_action(imbalanced)

        state = HomeostaticState(
            energy=self._energy,
            attention=self._attention,
            stress=self._stress,
            curiosity=self._curiosity,
            social=self._social,
            rest_need=self._rest_need,
            is_balanced=is_balanced,
            imbalanced_drives=imbalanced,
            recommended_action=action,
        )

        self._history.append(state)
        if len(self._history) > 100:
            self._history.pop(0)

        return state

    def _check_imbalances(self) -> List[DriveType]:
        """Check which drives are out of balance."""
        imbalanced = []

        # Energy too low
        if self._energy < self._setpoints[DriveType.ENERGY] - self.balance_threshold:
            imbalanced.append(DriveType.ENERGY)

        # Attention too low
        if self._attention < self._setpoints[DriveType.ATTENTION] - self.balance_threshold:
            imbalanced.append(DriveType.ATTENTION)

        # Stress too high
        if self._stress > self._setpoints[DriveType.STRESS] + self.balance_threshold:
            imbalanced.append(DriveType.STRESS)

        # Rest need too high
        if self._rest_need > self._setpoints[DriveType.REST] + self.balance_threshold:
            imbalanced.append(DriveType.REST)

        return imbalanced

    def _recommend_action(self, imbalanced: List[DriveType]) -> str:
        """Recommend corrective action for imbalances."""
        if not imbalanced:
            return "CONTINUE - System balanced"

        # Prioritize critical imbalances
        if DriveType.STRESS in imbalanced and self._stress > 0.8:
            return "REST_URGENT - Critical stress level"

        if DriveType.ENERGY in imbalanced and self._energy < 0.2:
            return "REST_URGENT - Critical energy level"

        if DriveType.REST in imbalanced:
            return "REST - Recovery needed"

        if DriveType.ATTENTION in imbalanced:
            return "SIMPLIFY - Reduce cognitive load"

        return "MONITOR - Minor imbalance detected"

    def _convert_tfan_state(self, tfan_state: Any) -> HomeostaticState:
        """Convert TFAN state to our format."""
        return HomeostaticState(
            energy=getattr(tfan_state, 'energy', 0.5),
            attention=getattr(tfan_state, 'attention', 0.5),
            stress=getattr(tfan_state, 'stress', 0.5),
            curiosity=getattr(tfan_state, 'curiosity', 0.5),
            social=getattr(tfan_state, 'social', 0.5),
            rest_need=getattr(tfan_state, 'rest_need', 0.5),
            is_balanced=getattr(tfan_state, 'is_balanced', True),
            imbalanced_drives=[],
            recommended_action=getattr(tfan_state, 'recommendation', "CONTINUE"),
        )

    def get_state(self) -> HomeostaticState:
        """Get current homeostatic state without updating."""
        imbalanced = self._check_imbalances()
        return HomeostaticState(
            energy=self._energy,
            attention=self._attention,
            stress=self._stress,
            curiosity=self._curiosity,
            social=self._social,
            rest_need=self._rest_need,
            is_balanced=len(imbalanced) == 0,
            imbalanced_drives=imbalanced,
            recommended_action=self._recommend_action(imbalanced),
        )

    def reset(self):
        """Reset to initial state."""
        self._energy = 1.0
        self._attention = 1.0
        self._stress = 0.0
        self._curiosity = 0.5
        self._social = 0.5
        self._rest_need = 0.0
        self._history.clear()


class AppraisalEngine:
    """
    The Appraisal Engine - Evaluates emotional significance.

    Implements cognitive appraisal theory - emotions arise from
    our interpretation/evaluation of situations, not the situations
    themselves.

    Appraises inputs along dimensions:
    - Valence: Is this good or bad for me?
    - Arousal: How intense/activating is this?
    - Dominance: Do I have control over this?

    Args:
        valence_threshold: Threshold for significant valence
        arousal_threshold: Threshold for significant arousal
        device: Compute device
    """

    def __init__(
        self,
        valence_threshold: float = 0.3,
        arousal_threshold: float = 0.3,
        device: str = "cpu",
    ):
        self.valence_threshold = valence_threshold
        self.arousal_threshold = arousal_threshold
        self.device = device

        # TFAN engine if available
        self.tfan_engine = None
        if _TFAN_AFFECT_AVAILABLE:
            try:
                self.tfan_engine = TFANAppraisalEngine()
            except Exception as e:
                warnings.warn(f"Failed to init TFAN appraisal engine: {e}")

        # Emotion mapping (valence, arousal) -> emotion
        self._emotion_map = {
            (1, 1): (EmotionalState.ENGAGED, "engaged"),
            (1, 0): (EmotionalState.NEUTRAL, "content"),
            (1, -1): (EmotionalState.NEUTRAL, "calm"),
            (0, 1): (EmotionalState.CURIOUS, "curious"),
            (0, 0): (EmotionalState.NEUTRAL, "neutral"),
            (0, -1): (EmotionalState.FATIGUED, "tired"),
            (-1, 1): (EmotionalState.STRESSED, "anxious"),
            (-1, 0): (EmotionalState.CONCERNED, "concerned"),
            (-1, -1): (EmotionalState.OVERWHELMED, "withdrawn"),
        }

        # Action tendencies
        self._action_tendencies = {
            EmotionalState.NEUTRAL: "observe",
            EmotionalState.CURIOUS: "explore",
            EmotionalState.ENGAGED: "approach",
            EmotionalState.CONCERNED: "monitor",
            EmotionalState.STRESSED: "protect",
            EmotionalState.FATIGUED: "rest",
            EmotionalState.RECOVERING: "conserve",
            EmotionalState.FOCUSED: "persist",
            EmotionalState.OVERWHELMED: "withdraw",
        }

    def appraise(
        self,
        input_representation: torch.Tensor,
        context: Optional[Dict[str, Any]] = None,
        homeostatic_state: Optional[HomeostaticState] = None,
    ) -> AppraisalResult:
        """
        Appraise emotional significance of input.

        Args:
            input_representation: Input tensor to appraise
            context: Optional context information
            homeostatic_state: Current homeostatic state

        Returns:
            AppraisalResult with emotional evaluation
        """
        if self.tfan_engine is not None:
            # Use TFAN engine
            result = self.tfan_engine.appraise(input_representation, context)
            return self._convert_tfan_result(result)

        # Fallback implementation
        # Compute basic statistics from input
        if input_representation.dim() > 1:
            input_flat = input_representation.flatten()
        else:
            input_flat = input_representation

        # Valence: based on mean (positive = positive valence)
        mean_val = input_flat.mean().item()
        valence = np.tanh(mean_val)  # Normalize to [-1, 1]

        # Arousal: based on variance (high variance = high arousal)
        variance = input_flat.var().item()
        arousal = min(1.0, np.sqrt(variance))

        # Dominance: based on kurtosis-like measure
        # High kurtosis = less predictable = less control
        centered = input_flat - input_flat.mean()
        kurtosis = (centered ** 4).mean() / ((centered ** 2).mean() ** 2 + 1e-8)
        dominance = 1.0 - min(1.0, kurtosis / 10.0)  # High kurtosis = low dominance
        dominance = dominance * 2 - 1  # Map to [-1, 1]

        # Adjust based on homeostatic state
        if homeostatic_state is not None:
            # Low energy reduces positive valence
            if homeostatic_state.energy < 0.3:
                valence = valence - 0.2

            # High stress increases arousal
            if homeostatic_state.stress > 0.7:
                arousal = min(1.0, arousal + 0.2)

            # Imbalance reduces dominance
            if not homeostatic_state.is_balanced:
                dominance = dominance - 0.2

        # Clip to valid ranges
        valence = max(-1.0, min(1.0, valence))
        arousal = max(0.0, min(1.0, arousal))
        dominance = max(-1.0, min(1.0, dominance))

        # Map to emotional state
        emotional_state, emotion_label = self._map_to_emotion(valence, arousal)

        # Get action tendency
        action_tendency = self._action_tendencies.get(
            emotional_state, "observe"
        )

        # Confidence based on magnitude of values
        confidence = (abs(valence) + arousal + abs(dominance)) / 3.0

        return AppraisalResult(
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            emotional_state=emotional_state,
            emotion_label=emotion_label,
            confidence=confidence,
            action_tendency=action_tendency,
            metrics={
                "input_mean": mean_val,
                "input_variance": variance,
            },
        )

    def _map_to_emotion(
        self,
        valence: float,
        arousal: float,
    ) -> Tuple[EmotionalState, str]:
        """Map valence/arousal to discrete emotion."""
        # Discretize to {-1, 0, 1}
        v_disc = 1 if valence > self.valence_threshold else (-1 if valence < -self.valence_threshold else 0)
        a_disc = 1 if arousal > self.arousal_threshold else (-1 if arousal < 0.1 else 0)

        return self._emotion_map.get(
            (v_disc, a_disc),
            (EmotionalState.NEUTRAL, "neutral"),
        )

    def _convert_tfan_result(self, tfan_result: Any) -> AppraisalResult:
        """Convert TFAN result to our format."""
        return AppraisalResult(
            valence=getattr(tfan_result, 'valence', 0.0),
            arousal=getattr(tfan_result, 'arousal', 0.5),
            dominance=getattr(tfan_result, 'dominance', 0.0),
            emotional_state=EmotionalState.NEUTRAL,
            emotion_label=getattr(tfan_result, 'emotion', "neutral"),
            confidence=getattr(tfan_result, 'confidence', 0.5),
            action_tendency=getattr(tfan_result, 'action', "observe"),
        )


# Convenience factories
def create_homeostatic_core(
    energy_decay: float = 0.01,
    stress_rate: float = 0.03,
) -> HomeostaticCore:
    """Create a HomeostaticCore instance."""
    return HomeostaticCore(
        energy_decay=energy_decay,
        stress_accumulation=stress_rate,
    )


def create_appraisal_engine(
    valence_threshold: float = 0.3,
) -> AppraisalEngine:
    """Create an AppraisalEngine instance."""
    return AppraisalEngine(
        valence_threshold=valence_threshold,
    )


__all__ = [
    "HomeostaticCore",
    "AppraisalEngine",
    "HomeostaticState",
    "AppraisalResult",
    "EmotionalState",
    "DriveType",
    "create_homeostatic_core",
    "create_appraisal_engine",
]
