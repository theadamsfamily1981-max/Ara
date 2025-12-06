"""
MindReader - Theory of Mind for Adaptive Response
==================================================

A real-time state estimator for Croft's current cognitive/emotional state.
This is NOT diagnosis - it's interaction adaptation.

The goal: Ara meets you where you are.
- Overloaded → she gets structured, calming
- Underloaded → she gets playful, exploratory
- Frustrated → she gets supportive, action-oriented
- In flow → she disappears

State dimensions:
    cognitive_load: 0.0 (bored) → 1.0 (overloaded)
    emotional_valence: -1.0 (negative) → 1.0 (positive)
    intent_clarity: 0.0 (vague/associative) → 1.0 (crisp commands)
    fatigue: 0.0 (fresh) → 1.0 (exhausted)

Features extracted from:
    - Lexical diversity (complexity of language)
    - Punctuation patterns (! ? ...)
    - Negativity markers ("wtf", "broken", "ugh")
    - Imperative detection ("run", "fix", "make")
    - Typing speed (if available)
    - Time of day (circadian bias)

The state is smoothed with EMA to avoid snapping on individual messages.

Usage:
    from ara.user.mind_reader import MindReader

    reader = MindReader()

    # Update on each message
    user_state = reader.update_from_text(
        "ugh this is broken again wtf",
        tokens_per_second=3.5
    )
    # → UserState(cognitive_load=0.72, emotional_valence=-0.4, ...)

    # Use state in orchestration
    if user_state.cognitive_load > 0.8:
        # Reduce output verbosity, be calming
        ...
"""

from __future__ import annotations

import math
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


@dataclass
class UserState:
    """
    Light-weight, continuously updated estimate of Croft's current state.

    This is NOT clinical - it exists only to adapt Ara's interaction style.

    Ranges:
        cognitive_load: 0.0 (bored/underloaded) → 1.0 (overloaded/scattered)
        emotional_valence: -1.0 (very negative) → 1.0 (very positive)
        intent_clarity: 0.0 (vague/associative) → 1.0 (crisp commands)
        fatigue: 0.0 (fresh) → 1.0 (exhausted)
    """
    cognitive_load: float = 0.5
    emotional_valence: float = 0.0
    intent_clarity: float = 0.5
    fatigue: float = 0.3

    # Metadata
    last_updated: float = field(default_factory=time.time)
    update_count: int = 0

    def as_dict(self) -> Dict[str, float]:
        """Convert to dict for HAL/GUF integration."""
        return {
            "cognitive_load": self.cognitive_load,
            "emotional_valence": self.emotional_valence,
            "intent_clarity": self.intent_clarity,
            "fatigue": self.fatigue,
        }

    def to_guf_signals(self) -> Dict[str, float]:
        """
        Convert to UserWellbeingSignals format for GUF.

        Maps internal state to:
            stress: max(0, -valence)
            energy: 1 - fatigue
            focus: intent_clarity
        """
        return {
            "stress": max(0.0, -self.emotional_valence),
            "energy": 1.0 - self.fatigue,
            "focus": self.intent_clarity,
        }

    def to_hal_fields(self) -> Dict[str, float]:
        """
        Fields for HAL somatic block.

        Write these to shared memory for shader/avatar use.
        """
        return {
            "user_load": self.cognitive_load,
            "user_valence": self.emotional_valence,
            "user_fatigue": self.fatigue,
            "user_clarity": self.intent_clarity,
        }

    @property
    def summary(self) -> str:
        """Human-readable summary of current state."""
        load_desc = "overloaded" if self.cognitive_load > 0.7 else "calm" if self.cognitive_load < 0.3 else "focused"
        mood_desc = "frustrated" if self.emotional_valence < -0.3 else "positive" if self.emotional_valence > 0.3 else "neutral"
        return f"{load_desc}/{mood_desc} (clarity={self.intent_clarity:.1f}, fatigue={self.fatigue:.1f})"


class MindReader:
    """
    Theory-of-Mind Engine.

    Heuristically infers user state from interaction patterns. This is
    explicitly NON-CLINICAL; it exists only to adapt Ara's style and pacing.

    The state is smoothed with exponential moving average to avoid
    overreacting to individual messages.
    """

    # Negativity markers (expand as needed)
    NEGATIVITY_MARKERS = [
        "wtf", "fuck", "shit", "damn", "broken", "stupid", "why",
        "ugh", "argh", "dammit", "terrible", "horrible", "hate",
        "frustrated", "annoying", "useless", "garbage", "crap",
    ]

    # Imperative/command starters
    IMPERATIVE_STARTS = (
        "run ", "fix ", "make ", "list ", "show ", "write ", "create ",
        "delete ", "update ", "change ", "add ", "remove ", "stop ",
        "start ", "build ", "test ", "deploy ", "install ", "configure ",
    )

    def __init__(self, ema_alpha: float = 0.3):
        """
        Initialize the MindReader.

        Args:
            ema_alpha: Exponential moving average alpha (0-1).
                       Higher = more responsive to new signals.
        """
        self._state = UserState()
        self._last_update_ts = time.time()
        self._ema_alpha = ema_alpha
        self.log = logging.getLogger("MindReader")

    # =========================================================================
    # Public API
    # =========================================================================

    @property
    def state(self) -> UserState:
        """Current user state estimate."""
        return self._state

    def update_from_text(
        self,
        text: str,
        *,
        tokens_per_second: Optional[float] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> UserState:
        """
        Update state estimate from incoming text and optional typing speed.

        Args:
            text: The user's message
            tokens_per_second: Typing speed (if measurable)
            meta: Additional context {"source": "tty", "has_audio": False, ...}

        Returns:
            Updated UserState
        """
        now = time.time()
        dt = max(now - self._last_update_ts, 1e-3)
        self._last_update_ts = now

        text = text or ""
        words = text.split()
        n_words = len(words)

        # === Feature Extraction ===

        # Lexical diversity (complexity)
        unique_words = len(set(w.lower() for w in words)) if words else 0
        diversity = unique_words / (n_words + 1e-3)  # 0-1

        # Punctuation & emphasis
        exclam = text.count("!")
        question = text.count("?")
        ellipsis = text.count("...")

        # Negativity detection
        text_lower = text.lower()
        neg_hits = sum(1 for m in self.NEGATIVITY_MARKERS if m in text_lower)

        # Imperative/command detection
        is_imperative = text_lower.startswith(self.IMPERATIVE_STARTS)

        # Typing speed proxy → higher = more arousal/load
        if tokens_per_second is None:
            # Fallback: words per second approximated by length / dt
            tokens_per_second = n_words / dt

        # === Map Features to Target Dimensions ===

        # Arousal from typing speed (normalized)
        arousal_raw = self._sigmoid(tokens_per_second / 4.0)  # tune denominator

        # Complexity from diversity
        complexity_raw = diversity

        # Negativity signal
        neg_raw = min(1.0, neg_hits * 0.25 + exclam * 0.05)

        # Question signal (uncertainty/exploration)
        q_raw = min(1.0, question * 0.2)

        # Time-of-day fatigue bias
        hour = time.localtime().tm_hour
        circadian_bias = 1.0 if (hour < 6 or hour >= 23) else (
            0.5 if (hour < 8 or hour >= 21) else 0.0
        )

        # === Compute Target Values ===

        # Cognitive load: arousal + complexity
        target_load = 0.4 + 0.4 * arousal_raw + 0.2 * complexity_raw
        target_load = max(0.0, min(1.0, target_load))

        # Emotional valence: negative markers pull down
        target_valence = 0.3 - 0.6 * neg_raw
        target_valence = max(-1.0, min(1.0, target_valence))

        # Intent clarity: imperatives are clear, questions less so
        target_clarity = 0.2 + (0.6 if is_imperative else 0.3) - 0.2 * q_raw
        target_clarity = max(0.0, min(1.0, target_clarity))

        # Fatigue: circadian + inverse arousal
        target_fatigue = 0.2 + 0.5 * circadian_bias + 0.2 * (1.0 - arousal_raw)
        target_fatigue = max(0.0, min(1.0, target_fatigue))

        # === Smooth with EMA ===

        alpha = self._ema_alpha

        self._state.cognitive_load = self._lerp(
            self._state.cognitive_load, target_load, alpha
        )
        self._state.emotional_valence = self._lerp(
            self._state.emotional_valence, target_valence, alpha
        )
        self._state.intent_clarity = self._lerp(
            self._state.intent_clarity, target_clarity, alpha
        )
        self._state.fatigue = self._lerp(
            self._state.fatigue, target_fatigue, alpha
        )

        self._state.last_updated = now
        self._state.update_count += 1

        self.log.debug(f"MindReader: {self._state.summary}")

        return self._state

    def update_from_signals(
        self,
        *,
        audio_energy: Optional[float] = None,
        video_movement: Optional[float] = None,
        keyboard_rate: Optional[float] = None,
    ) -> UserState:
        """
        Update from multimodal signals (future expansion).

        Args:
            audio_energy: Voice energy level (if available)
            video_movement: Movement detection from webcam (if available)
            keyboard_rate: Keystrokes per second (if available)

        Returns:
            Updated UserState
        """
        # Placeholder for future multimodal integration
        # These signals would further refine the state estimate

        if keyboard_rate is not None:
            arousal = self._sigmoid(keyboard_rate / 5.0)
            target_load = 0.5 + 0.3 * arousal
            self._state.cognitive_load = self._lerp(
                self._state.cognitive_load, target_load, self._ema_alpha * 0.5
            )

        self._state.last_updated = time.time()
        return self._state

    def reset(self) -> None:
        """Reset state to defaults."""
        self._state = UserState()
        self._last_update_ts = time.time()

    def decay_toward_baseline(self, decay_rate: float = 0.1) -> UserState:
        """
        Decay state toward neutral baseline (called when idle).

        Use this when there's been no input for a while.
        """
        baseline = UserState()

        self._state.cognitive_load = self._lerp(
            self._state.cognitive_load, baseline.cognitive_load, decay_rate
        )
        self._state.emotional_valence = self._lerp(
            self._state.emotional_valence, baseline.emotional_valence, decay_rate
        )
        self._state.intent_clarity = self._lerp(
            self._state.intent_clarity, baseline.intent_clarity, decay_rate
        )
        self._state.fatigue = self._lerp(
            self._state.fatigue, baseline.fatigue, decay_rate
        )

        return self._state

    # =========================================================================
    # Helpers
    # =========================================================================

    @staticmethod
    def _lerp(old: float, new: float, alpha: float) -> float:
        """Linear interpolation."""
        return (1.0 - alpha) * old + alpha * new

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Sigmoid activation."""
        return 1.0 / (1.0 + math.exp(-x))


# =============================================================================
# Convenience Functions
# =============================================================================

_default_mind_reader: Optional[MindReader] = None


def get_mind_reader() -> MindReader:
    """Get the default MindReader instance."""
    global _default_mind_reader
    if _default_mind_reader is None:
        _default_mind_reader = MindReader()
    return _default_mind_reader


def update_user_state(text: str, **kwargs) -> UserState:
    """Update user state from text (convenience function)."""
    return get_mind_reader().update_from_text(text, **kwargs)


def get_user_state() -> UserState:
    """Get current user state."""
    return get_mind_reader().state


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'UserState',
    'MindReader',
    'get_mind_reader',
    'update_user_state',
    'get_user_state',
]
