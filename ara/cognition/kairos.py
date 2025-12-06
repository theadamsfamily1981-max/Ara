"""
Kairos Engine - The Opportune Moment
=====================================

Kairos (καιρός) is the Greek god of the opportune moment, as opposed
to Chronos (clock time). This engine determines when the User is
*receptive* - ready to receive a gift, suggestion, or interruption.

The Kairos Engine replaces cron jobs with something more human:
"Is the user ready to be delighted right now?"

Receptivity depends on:
    - Cognitive load (not overwhelmed)
    - Emotional valence (not angry)
    - Intent clarity (not laser-focused)
    - System activity (not during compile)

High-importance gifts can interrupt slightly busier moments.
Low-importance suggestions wait for perfect stillness.

Usage:
    from ara.cognition.kairos import KairosEngine

    kairos = KairosEngine(mind_reader)

    # Check if now is a good time
    if kairos.is_opportune(gift_importance=0.9):
        present_gift(gift)
    else:
        # Hold the gift for later
        studio_queue.append(gift)

    # Get current receptivity
    score = kairos.get_receptivity()
    # → 0.82 (high receptivity)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, Protocol
from enum import Enum

logger = logging.getLogger(__name__)


class KairosMoment(Enum):
    """Types of opportune moments."""
    RECEPTIVE_STILLNESS = "receptive_stillness"  # Perfect moment
    GENTLE_PAUSE = "gentle_pause"                 # Good moment
    BUSY_BUT_OPEN = "busy_but_open"              # Acceptable for important things
    FOCUSED = "focused"                           # Don't interrupt
    OVERWHELMED = "overwhelmed"                   # Definitely don't interrupt
    FRUSTRATED = "frustrated"                     # Wrong time
    UNAVAILABLE = "unavailable"                   # User not present


@dataclass
class KairosState:
    """Current kairos state assessment."""
    moment: KairosMoment
    receptivity: float           # 0-1, overall receptivity score
    cognitive_space: float       # 0-1, available mental bandwidth
    emotional_openness: float    # 0-1, emotional receptivity
    system_quietness: float      # 0-1, system activity level
    timestamp: float

    @property
    def summary(self) -> str:
        return f"{self.moment.value} (receptivity={self.receptivity:.0%})"


class MindReaderProtocol(Protocol):
    """Protocol for MindReader interface."""
    @property
    def state(self) -> Any:
        ...


class HALProtocol(Protocol):
    """Protocol for HAL interface."""
    def read_somatic(self) -> Dict[str, Any]:
        ...


class KairosEngine:
    """
    The Timekeeper.

    Determines the "Opportune Moment" for intervention.
    Distinct from Chronos (Clock Time).
    """

    # Thresholds for receptivity
    COGNITIVE_SPACE_THRESHOLD = 0.4   # Load below this = has space
    VALENCE_THRESHOLD = 0.0           # Must be at least neutral
    FOCUS_THRESHOLD = 0.6             # Below this = not laser-focused
    SYSTEM_QUIET_THRESHOLD = 0.5      # Arousal below this = quiet

    # Time-of-day preferences
    PEAK_HOURS = (9, 11, 14, 16)      # Good hours for gifts
    AVOID_HOURS = (0, 6, 22, 24)      # Don't interrupt

    def __init__(
        self,
        mind_reader: Optional[MindReaderProtocol] = None,
        hal: Optional[HALProtocol] = None,
    ):
        """
        Initialize the Kairos Engine.

        Args:
            mind_reader: MindReader for user state
            hal: HAL for system state
        """
        self.mind = mind_reader
        self.hal = hal
        self.log = logging.getLogger("Kairos")

        # Cache last assessment
        self._last_state: Optional[KairosState] = None
        self._last_check: float = 0

    def is_opportune(
        self,
        gift_importance: float = 0.5,
        check_time_of_day: bool = True,
    ) -> bool:
        """
        Check if now is an opportune moment for intervention.

        Args:
            gift_importance: How important is this gift (0-1)?
                            Higher importance can interrupt busier moments.
            check_time_of_day: Whether to consider time of day

        Returns:
            True if now is a good time
        """
        state = self.assess()

        # Never interrupt certain moments
        if state.moment in (KairosMoment.OVERWHELMED, KairosMoment.FRUSTRATED,
                           KairosMoment.UNAVAILABLE):
            return False

        # Time of day check
        if check_time_of_day:
            hour = time.localtime().tm_hour
            if any(hour >= start and hour < end
                   for start, end in [(0, 6), (22, 24)]):
                # Only very important things at night
                if gift_importance < 0.9:
                    return False

        # The Kairos Ratio:
        # High importance gifts can interrupt slightly busier moments
        threshold = 0.8 - (gift_importance * 0.4)
        # importance=0.5 → threshold=0.6
        # importance=0.9 → threshold=0.44
        # importance=1.0 → threshold=0.4

        return state.receptivity >= threshold

    def assess(self) -> KairosState:
        """
        Assess the current kairos state.

        Returns:
            Current KairosState
        """
        now = time.time()

        # Use cached if recent (within 5 seconds)
        if self._last_state and (now - self._last_check) < 5:
            return self._last_state

        # Get user state
        user_state = self._get_user_state()

        # Get system state
        system_state = self._get_system_state()

        # Calculate components
        cognitive_space = 1.0 - user_state.get("cognitive_load", 0.5)
        emotional_openness = (user_state.get("emotional_valence", 0.0) + 1.0) / 2.0
        intent_blur = 1.0 - user_state.get("intent_clarity", 0.5)
        system_quietness = 1.0 - system_state.get("arousal", 0.5)

        # Overall receptivity
        receptivity = (
            cognitive_space * 0.35 +
            emotional_openness * 0.25 +
            intent_blur * 0.2 +
            system_quietness * 0.2
        )

        # Determine moment type
        moment = self._classify_moment(
            cognitive_space=cognitive_space,
            emotional_openness=emotional_openness,
            system_quietness=system_quietness,
            intent_clarity=user_state.get("intent_clarity", 0.5),
        )

        state = KairosState(
            moment=moment,
            receptivity=receptivity,
            cognitive_space=cognitive_space,
            emotional_openness=emotional_openness,
            system_quietness=system_quietness,
            timestamp=now,
        )

        self._last_state = state
        self._last_check = now

        return state

    def _get_user_state(self) -> Dict[str, float]:
        """Get current user state."""
        if self.mind is None:
            return self._simulate_user_state()

        try:
            state = self.mind.state
            return {
                "cognitive_load": getattr(state, 'cognitive_load', 0.5),
                "emotional_valence": getattr(state, 'emotional_valence', 0.0),
                "intent_clarity": getattr(state, 'intent_clarity', 0.5),
                "fatigue": getattr(state, 'fatigue', 0.3),
            }
        except Exception as e:
            self.log.warning(f"Failed to get user state: {e}")
            return self._simulate_user_state()

    def _get_system_state(self) -> Dict[str, float]:
        """Get current system state."""
        if self.hal is None:
            return self._simulate_system_state()

        try:
            somatic = self.hal.read_somatic()
            pad = somatic.get('pad', {})
            return {
                "arousal": pad.get('a', pad.get('A', 0.5)),
                "pleasure": pad.get('p', pad.get('P', 0.5)),
            }
        except Exception as e:
            self.log.warning(f"Failed to get system state: {e}")
            return self._simulate_system_state()

    def _simulate_user_state(self) -> Dict[str, float]:
        """Simulate user state for testing."""
        # Vary based on time of day
        hour = time.localtime().tm_hour

        if 9 <= hour <= 11 or 14 <= hour <= 16:
            # Good working hours
            return {
                "cognitive_load": 0.4,
                "emotional_valence": 0.3,
                "intent_clarity": 0.5,
                "fatigue": 0.2,
            }
        elif 22 <= hour or hour <= 6:
            # Night/early morning
            return {
                "cognitive_load": 0.2,
                "emotional_valence": 0.1,
                "intent_clarity": 0.3,
                "fatigue": 0.7,
            }
        else:
            # Default
            return {
                "cognitive_load": 0.5,
                "emotional_valence": 0.0,
                "intent_clarity": 0.5,
                "fatigue": 0.4,
            }

    def _simulate_system_state(self) -> Dict[str, float]:
        """Simulate system state for testing."""
        return {
            "arousal": 0.4,
            "pleasure": 0.3,
        }

    def _classify_moment(
        self,
        cognitive_space: float,
        emotional_openness: float,
        system_quietness: float,
        intent_clarity: float,
    ) -> KairosMoment:
        """Classify the current moment."""
        # Perfect stillness
        if (cognitive_space > 0.7 and emotional_openness > 0.6 and
            system_quietness > 0.6 and intent_clarity < 0.4):
            return KairosMoment.RECEPTIVE_STILLNESS

        # Good moment
        if (cognitive_space > 0.5 and emotional_openness > 0.4 and
            system_quietness > 0.4):
            return KairosMoment.GENTLE_PAUSE

        # Busy but could accept important things
        if emotional_openness > 0.3 and cognitive_space > 0.3:
            return KairosMoment.BUSY_BUT_OPEN

        # Too focused
        if intent_clarity > 0.7:
            return KairosMoment.FOCUSED

        # Overwhelmed
        if cognitive_space < 0.2:
            return KairosMoment.OVERWHELMED

        # Frustrated
        if emotional_openness < 0.3:
            return KairosMoment.FRUSTRATED

        # Default to focused
        return KairosMoment.FOCUSED

    def get_receptivity(self) -> float:
        """Get current receptivity score."""
        return self.assess().receptivity

    def wait_for_opportune(
        self,
        importance: float = 0.5,
        timeout_seconds: float = 3600,
        check_interval: float = 60,
    ) -> bool:
        """
        Wait until an opportune moment occurs.

        Args:
            importance: Gift importance (higher can accept less ideal moments)
            timeout_seconds: Maximum time to wait
            check_interval: How often to check

        Returns:
            True if opportune moment found, False if timeout
        """
        start = time.time()

        while (time.time() - start) < timeout_seconds:
            if self.is_opportune(gift_importance=importance):
                return True
            time.sleep(check_interval)

        return False

    def get_next_window_estimate(self) -> Optional[str]:
        """Estimate when the next good moment might be."""
        hour = time.localtime().tm_hour

        if 9 <= hour <= 11:
            return "Current window is good (morning focus period)"
        elif 12 <= hour <= 13:
            return "Try after lunch (around 14:00)"
        elif 14 <= hour <= 16:
            return "Current window is good (afternoon session)"
        elif 17 <= hour <= 21:
            return "Evening - try during a natural break"
        else:
            return "Wait until morning (around 9:00)"


# =============================================================================
# Convenience Functions
# =============================================================================

_default_kairos: Optional[KairosEngine] = None


def get_kairos() -> KairosEngine:
    """Get the default KairosEngine instance."""
    global _default_kairos
    if _default_kairos is None:
        _default_kairos = KairosEngine()
    return _default_kairos


def is_opportune(importance: float = 0.5) -> bool:
    """Check if now is an opportune moment."""
    return get_kairos().is_opportune(gift_importance=importance)


def get_receptivity() -> float:
    """Get current receptivity score."""
    return get_kairos().get_receptivity()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'KairosMoment',
    'KairosState',
    'KairosEngine',
    'get_kairos',
    'is_opportune',
    'get_receptivity',
]
