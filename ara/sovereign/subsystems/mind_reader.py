"""
MindReader Subsystem: User State Owner

Owns: user.*

Responsibilities:
- Sense user's cognitive mode (deep work, casual, urgent, sleeping)
- Track focus signals
- Detect emotional state
- Maintain presence awareness
"""

from __future__ import annotations

import time
import logging
from typing import Optional, Dict, Any
from enum import Enum

from .ownership import SubsystemBase, Subsystem, GuardedStateWriter

logger = logging.getLogger(__name__)


class CognitiveMode(str, Enum):
    """User's cognitive mode."""
    UNKNOWN = "unknown"
    DEEP_WORK = "deep_work"     # Don't interrupt
    CASUAL = "casual"           # Can chat
    URGENT = "urgent"           # Needs immediate help
    SLEEPING = "sleeping"       # Off hours
    AWAY = "away"               # Not at computer


class MindReaderSubsystem(SubsystemBase):
    """
    MindReader subsystem - senses and updates user state.

    Updates user.* during sense phase.
    """

    subsystem_id = Subsystem.MIND_READER

    def __init__(self, writer: GuardedStateWriter):
        super().__init__(writer)
        self._last_interaction_ts = 0.0
        self._message_history: list[tuple[float, str]] = []  # (timestamp, message)

    def sense(self, last_message: Optional[str] = None) -> Dict[str, Any]:
        """
        Update user state based on signals.

        Called during sense_phase of sovereign tick.
        """
        updates = {}
        now = time.time()

        # Update last message if provided
        if last_message:
            self._last_interaction_ts = now
            self._message_history.append((now, last_message))
            # Keep last 20 messages
            self._message_history = self._message_history[-20:]

            self.write("user.last_message", last_message[:500])
            self.write("user.last_message_ts", now)
            updates["last_message"] = last_message[:50]

        # Detect cognitive mode
        cognitive_mode = self._detect_cognitive_mode(last_message)
        self.write("user.cognitive_mode", cognitive_mode.value)
        updates["cognitive_mode"] = cognitive_mode.value

        # Update presence
        presence = self._detect_presence()
        self.write("user.is_present", presence)
        updates["is_present"] = presence

        # Detect focus level (0.0 = distracted, 1.0 = deep focus)
        focus = self._estimate_focus(last_message)
        self.write("user.focus_level", focus)
        updates["focus_level"] = focus

        # Detect emotional valence (-1.0 = negative, 1.0 = positive)
        valence = self._detect_valence(last_message) if last_message else 0.0
        self.write("user.emotional_valence", valence)
        updates["emotional_valence"] = valence

        return updates

    def _detect_cognitive_mode(self, message: Optional[str] = None) -> CognitiveMode:
        """Detect user's cognitive mode from signals."""
        now = time.time()

        # Check for explicit mode indicators in message
        if message:
            msg_lower = message.lower()

            if any(word in msg_lower for word in ["urgent", "emergency", "asap", "help now"]):
                return CognitiveMode.URGENT

            if any(word in msg_lower for word in ["focusing", "deep work", "don't interrupt", "busy"]):
                return CognitiveMode.DEEP_WORK

        # Check time since last interaction
        time_since = now - self._last_interaction_ts
        if time_since > 3600:  # More than an hour
            # Check if it's sleep hours (11pm - 7am local time)
            hour = time.localtime().tm_hour
            if hour >= 23 or hour < 7:
                return CognitiveMode.SLEEPING
            return CognitiveMode.AWAY

        # Default to casual if recently active
        if time_since < 60:  # Active within last minute
            return CognitiveMode.CASUAL

        return CognitiveMode.UNKNOWN

    def _detect_presence(self) -> bool:
        """Detect if user is present."""
        now = time.time()
        time_since = now - self._last_interaction_ts

        # Present if interaction within last 5 minutes
        return time_since < 300

    def _estimate_focus(self, message: Optional[str] = None) -> float:
        """Estimate user's focus level."""
        if not message:
            return 0.5  # Neutral

        # Long, detailed messages suggest focus
        if len(message) > 200:
            return 0.8

        # Short messages could be distracted
        if len(message) < 20:
            return 0.4

        return 0.6

    def _detect_valence(self, message: str) -> float:
        """Detect emotional valence from message."""
        if not message:
            return 0.0

        msg_lower = message.lower()

        # Simple keyword detection
        positive_words = {"thanks", "great", "awesome", "love", "happy", "good", "nice", "perfect"}
        negative_words = {"frustrated", "angry", "sad", "upset", "hate", "bad", "terrible", "stuck"}

        words = set(msg_lower.split())
        pos_count = len(words & positive_words)
        neg_count = len(words & negative_words)

        if pos_count + neg_count == 0:
            return 0.0

        return (pos_count - neg_count) / (pos_count + neg_count)

    def should_interrupt(self) -> bool:
        """Check if it's okay to interrupt the user."""
        try:
            cognitive_mode = CognitiveMode(self.read("user.cognitive_mode"))
        except (ValueError, AttributeError):
            cognitive_mode = CognitiveMode.UNKNOWN

        # Never interrupt deep work
        if cognitive_mode == CognitiveMode.DEEP_WORK:
            return False

        # Never interrupt sleep
        if cognitive_mode == CognitiveMode.SLEEPING:
            return False

        # Be careful with urgent mode - user is stressed
        if cognitive_mode == CognitiveMode.URGENT:
            return True  # But keep it brief

        return True

    def get_protection_recommendation(self) -> str:
        """Get recommendation for how to interact with user."""
        try:
            cognitive_mode = CognitiveMode(self.read("user.cognitive_mode"))
            valence = self.read("user.emotional_valence")
        except (ValueError, AttributeError):
            return "standard"

        if cognitive_mode == CognitiveMode.DEEP_WORK:
            return "silent_unless_urgent"

        if cognitive_mode == CognitiveMode.URGENT:
            if valence < -0.3:
                return "calm_supportive"
            return "efficient_direct"

        if cognitive_mode == CognitiveMode.SLEEPING:
            return "do_not_disturb"

        return "standard"
