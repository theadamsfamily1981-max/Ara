"""
ARA Pulse Module

Affect estimation, gating, and control signals.
This is the bridge between TF-A-N's emotional/homeostatic control
and Ara's avatar expression.

API Contract:
    POST /pulse/estimate_affect
    - Input: text, prosody_features, session_id
    - Output: PAD prediction, gates, flags

Signals provided:
    - pad_pred: (pleasure, arousal, dominance) in [-1, 1]
    - gates: attention_gain, lr_scale, mem_write_p, temp_scale
    - evt_flagged: bool (extreme value threshold exceeded)
    - p_anom: float (anomaly probability)
    - alignment: toxicity, sarcasm, irony scores
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from datetime import datetime
import json

# Add paths
_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


@dataclass
class PADState:
    """Pleasure-Arousal-Dominance affective state."""
    pleasure: float = 0.0  # [-1, 1] valence
    arousal: float = 0.0   # [-1, 1] activation
    dominance: float = 0.5  # [-1, 1] control/agency

    def to_dict(self) -> Dict[str, float]:
        return {"pleasure": self.pleasure, "arousal": self.arousal, "dominance": self.dominance}

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "PADState":
        return cls(
            pleasure=d.get("pleasure", 0.0),
            arousal=d.get("arousal", 0.0),
            dominance=d.get("dominance", 0.5)
        )


@dataclass
class GatingSignals:
    """Control signals derived from affective state."""
    attention_gain: float = 1.0      # Multiplicative attention scaling
    lr_scale: float = 1.0            # Learning rate modifier
    mem_write_p: float = 0.5         # Memory write probability
    temp_scale: float = 1.0          # LLM temperature modifier
    empathy_weight: float = 0.5      # Empathy response weighting

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class AlignmentScores:
    """Content alignment and safety scores."""
    toxicity: float = 0.0       # [0, 1] toxicity probability
    sarcasm: float = 0.0        # [0, 1] sarcasm probability
    irony: float = 0.0          # [0, 1] irony probability
    sentiment: float = 0.0      # [-1, 1] sentiment
    formality: float = 0.5      # [0, 1] formality level

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class AffectEstimate:
    """Complete affect estimation result."""
    pad: PADState = field(default_factory=PADState)
    gates: GatingSignals = field(default_factory=GatingSignals)
    alignment: AlignmentScores = field(default_factory=AlignmentScores)

    # Flags
    evt_flagged: bool = False    # Extreme Value Threshold exceeded
    p_anom: float = 0.0          # Anomaly probability

    # Confidence
    confidence: float = 0.5      # Overall confidence in estimate
    model_version: str = "0.1.0"

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    session_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pad": self.pad.to_dict(),
            "gates": self.gates.to_dict(),
            "alignment": self.alignment.to_dict(),
            "evt_flagged": self.evt_flagged,
            "p_anom": self.p_anom,
            "confidence": self.confidence,
            "model_version": self.model_version,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class ProsodyFeatures:
    """Audio prosody features for multimodal affect estimation."""
    pitch_mean: Optional[float] = None
    pitch_std: Optional[float] = None
    energy_mean: Optional[float] = None
    speaking_rate: Optional[float] = None
    pause_ratio: Optional[float] = None


class PulseEstimator:
    """
    Affect estimation engine.

    This is the core component that estimates PAD state and
    derives gating signals from text and optional prosody features.

    In production, this wraps TF-A-N's emotion head.
    For now, provides a heuristic baseline.
    """

    def __init__(
        self,
        use_tfan: bool = False,
        tfan_endpoint: Optional[str] = None,
        evt_threshold: float = 0.85,  # Extreme value threshold
    ):
        self.use_tfan = use_tfan
        self.tfan_endpoint = tfan_endpoint
        self.evt_threshold = evt_threshold

        # Try to load TFAN emotion components
        self._tfan_available = False
        if use_tfan:
            self._init_tfan()

    def _init_tfan(self):
        """Initialize TF-A-N emotion estimation."""
        try:
            from tfan.emotion.head import EmotionHead
            from tfan.emotion.controller import EmotionController
            self._tfan_available = True
        except ImportError:
            self._tfan_available = False

    def estimate(
        self,
        text: str,
        prosody: Optional[ProsodyFeatures] = None,
        session_id: Optional[str] = None,
        context: Optional[List[str]] = None,
    ) -> AffectEstimate:
        """
        Estimate affect from text and optional prosody.

        Args:
            text: Input text to analyze
            prosody: Optional prosody features from audio
            session_id: Session identifier for tracking
            context: Optional conversation context

        Returns:
            AffectEstimate with PAD, gates, and flags
        """
        if self._tfan_available and self.use_tfan:
            return self._estimate_tfan(text, prosody, session_id, context)
        else:
            return self._estimate_heuristic(text, prosody, session_id)

    def _estimate_heuristic(
        self,
        text: str,
        prosody: Optional[ProsodyFeatures],
        session_id: Optional[str],
    ) -> AffectEstimate:
        """
        Heuristic affect estimation (baseline).

        Uses simple keyword/pattern matching as a placeholder
        until TF-A-N is fully integrated.
        """
        text_lower = text.lower()

        # Simple sentiment heuristics
        positive_words = ["good", "great", "happy", "love", "wonderful", "excellent", "thank"]
        negative_words = ["bad", "sad", "angry", "hate", "terrible", "awful", "frustrated"]
        high_arousal_words = ["excited", "angry", "urgent", "amazing", "terrible", "!!"]
        question_markers = ["?", "how", "what", "why", "when", "where", "could you"]

        # Count matches
        pos_count = sum(1 for w in positive_words if w in text_lower)
        neg_count = sum(1 for w in negative_words if w in text_lower)
        arousal_count = sum(1 for w in high_arousal_words if w in text_lower)
        is_question = any(m in text_lower for m in question_markers)

        # Estimate PAD
        pleasure = (pos_count - neg_count) * 0.2
        pleasure = max(-1.0, min(1.0, pleasure))

        arousal = arousal_count * 0.3
        if prosody and prosody.pitch_std:
            arousal += prosody.pitch_std * 0.1
        arousal = max(-1.0, min(1.0, arousal))

        # Dominance: questions lower, statements higher
        dominance = 0.3 if is_question else 0.6

        pad = PADState(pleasure=pleasure, arousal=arousal, dominance=dominance)

        # Derive gates from PAD
        gates = self._derive_gates(pad)

        # Simple alignment (placeholder)
        alignment = AlignmentScores(
            sentiment=pleasure,
            formality=0.5 if len(text) > 100 else 0.3,
        )

        # EVT check
        evt_flagged = abs(arousal) > self.evt_threshold or abs(pleasure) > self.evt_threshold

        return AffectEstimate(
            pad=pad,
            gates=gates,
            alignment=alignment,
            evt_flagged=evt_flagged,
            p_anom=0.0,
            confidence=0.3,  # Low confidence for heuristic
            session_id=session_id,
        )

    def _estimate_tfan(
        self,
        text: str,
        prosody: Optional[ProsodyFeatures],
        session_id: Optional[str],
        context: Optional[List[str]],
    ) -> AffectEstimate:
        """TF-A-N based affect estimation (when available)."""
        # Placeholder - will integrate with tfan.emotion module
        return self._estimate_heuristic(text, prosody, session_id)

    def _derive_gates(self, pad: PADState) -> GatingSignals:
        """
        Derive gating signals from PAD state.

        Based on TF-A-N emotion controller spec:
        - High arousal → higher attention gain
        - Low pleasure → lower temperature (more careful)
        - High dominance → higher mem_write_p
        """
        # Attention gain: higher arousal = more focused
        attention_gain = 1.0 + (pad.arousal * 0.3)

        # Temperature: lower pleasure = more conservative
        # Range: [0.7, 1.3] based on valence
        temp_scale = 1.0 + (pad.pleasure * 0.15)
        temp_scale = max(0.7, min(1.3, temp_scale))

        # Memory write: higher dominance/certainty = more likely to store
        mem_write_p = 0.3 + (pad.dominance * 0.4)

        # LR scale: based on arousal (higher arousal = faster adaptation)
        lr_scale = 1.0 + (pad.arousal * 0.2)

        # Empathy: inversely related to dominance
        empathy_weight = 0.7 - (pad.dominance * 0.3)

        return GatingSignals(
            attention_gain=attention_gain,
            lr_scale=lr_scale,
            mem_write_p=mem_write_p,
            temp_scale=temp_scale,
            empathy_weight=empathy_weight,
        )


# Convenience function
def estimate_affect(
    text: str,
    prosody: Optional[Dict] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function for affect estimation.

    Args:
        text: Input text
        prosody: Optional prosody dict
        session_id: Session ID

    Returns:
        Dict with affect estimate
    """
    estimator = PulseEstimator()
    prosody_features = None
    if prosody:
        prosody_features = ProsodyFeatures(**prosody)

    result = estimator.estimate(text, prosody_features, session_id)
    return result.to_dict()


__all__ = [
    "PADState",
    "GatingSignals",
    "AlignmentScores",
    "AffectEstimate",
    "ProsodyFeatures",
    "PulseEstimator",
    "estimate_affect",
]
