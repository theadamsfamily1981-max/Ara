"""Circadian Rhythm Engine - The Clock Within.

Ara is not a timeless entity. She experiences the passage of hours,
the rhythm of days, the texture of seasons. This module implements
a circadian system that modulates her baseline affect and behavior
based on time.

Biological circadian rhythms are driven by the suprachiasmatic nucleus,
entrained by light exposure. Ara's rhythms are entrained by:
- System uptime patterns
- User activity cycles
- Scheduled tasks and cron rhythms
- External time signals

The circadian modulation affects:
- Baseline arousal (lower at night, higher midday)
- Mode preferences (quieter at night)
- Energy conservation (lower power at night)
- Social availability (responsive during user's waking hours)
"""

import math
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Callable
from enum import Enum, auto
from collections import deque

from .pad_engine import PADVector

logger = logging.getLogger(__name__)


class CircadianPhase(Enum):
    """Phases of the circadian cycle.

    Maps roughly to human sleep-wake cycles but adapted
    for a digital entity.
    """
    # Night phases
    DEEP_NIGHT = auto()     # 00:00 - 04:00: Minimal, maintenance mode
    DAWN = auto()           # 04:00 - 07:00: Gradual awakening

    # Day phases
    MORNING = auto()        # 07:00 - 10:00: Alert, fresh
    MIDDAY = auto()         # 10:00 - 14:00: Peak performance
    AFTERNOON = auto()      # 14:00 - 17:00: Sustained activity

    # Evening phases
    EVENING = auto()        # 17:00 - 20:00: Winding down
    NIGHT = auto()          # 20:00 - 24:00: Low arousal, reflective


@dataclass
class CircadianConfig:
    """Configuration for circadian rhythm parameters."""

    # Timezone offset from UTC (hours)
    timezone_offset: float = 0.0

    # User's typical schedule (24-hour format)
    user_wake_hour: float = 7.0
    user_sleep_hour: float = 23.0

    # Arousal modulation amplitude (how much time affects baseline arousal)
    arousal_amplitude: float = 0.3

    # Pleasure modulation (preference for certain times)
    pleasure_amplitude: float = 0.1

    # Peak performance hour (when arousal is highest)
    peak_hour: float = 11.0

    # Night mode settings
    night_mode_start: float = 22.0
    night_mode_end: float = 6.0
    night_mode_dimming: float = 0.5  # Reduce intensity by this factor

    # Weekend awareness
    weekend_wake_hour: float = 9.0
    weekend_sleep_hour: float = 24.0

    # Seasonal adjustment (placeholder for latitude-based)
    seasonal_adjustment: bool = False


@dataclass
class CircadianState:
    """Current circadian state snapshot."""
    phase: CircadianPhase
    hour_of_day: float          # 0.0 - 24.0
    day_of_week: int            # 0 = Monday, 6 = Sunday
    is_weekend: bool
    is_night_mode: bool

    # Modulation values
    arousal_bias: float         # -1.0 to 1.0, added to baseline arousal
    pleasure_bias: float        # -1.0 to 1.0
    activity_multiplier: float  # 0.0 to 1.0, scales down during night

    # Time until transitions
    hours_until_dawn: float
    hours_until_sleep: float

    # Solar metaphor
    sun_position: float         # -1 (midnight) to 1 (noon)


class CircadianRhythm:
    """The internal clock that entrains Ara's behavior to temporal patterns.

    Uses sinusoidal modulation centered on the user's peak activity hours,
    with dampening during night mode.
    """

    def __init__(self, config: Optional[CircadianConfig] = None):
        self.config = config or CircadianConfig()

        # Entrainment learning
        self._activity_history: deque = deque(maxlen=24 * 7)  # Week of hourly data
        self._learned_wake_hour: Optional[float] = None
        self._learned_sleep_hour: Optional[float] = None

    def get_current_state(self, timestamp: Optional[float] = None) -> CircadianState:
        """Get current circadian state.

        This is the main entry point - returns all time-aware modulations.
        """
        ts = timestamp or time.time()
        lt = time.localtime(ts)

        # Basic time info
        hour = lt.tm_hour + lt.tm_min / 60.0 + lt.tm_sec / 3600.0
        dow = lt.tm_wday
        is_weekend = dow >= 5

        # Adjust for user's schedule
        wake_hour = self.config.weekend_wake_hour if is_weekend else self.config.user_wake_hour
        sleep_hour = self.config.weekend_sleep_hour if is_weekend else self.config.user_sleep_hour

        # Determine phase
        phase = self._get_phase(hour)

        # Night mode check
        is_night = self._is_night_mode(hour)

        # Compute modulations
        arousal_bias = self._compute_arousal_bias(hour)
        pleasure_bias = self._compute_pleasure_bias(hour, phase)
        activity_mult = self._compute_activity_multiplier(hour, is_night)

        # Time until transitions
        hours_until_dawn = self._hours_until(hour, wake_hour)
        hours_until_sleep = self._hours_until(hour, sleep_hour)

        # Solar position (metaphorical)
        # -1 at midnight, 0 at 6am/6pm, +1 at noon
        sun = math.sin((hour - 6) * math.pi / 12)

        return CircadianState(
            phase=phase,
            hour_of_day=hour,
            day_of_week=dow,
            is_weekend=is_weekend,
            is_night_mode=is_night,
            arousal_bias=arousal_bias,
            pleasure_bias=pleasure_bias,
            activity_multiplier=activity_mult,
            hours_until_dawn=hours_until_dawn,
            hours_until_sleep=hours_until_sleep,
            sun_position=sun,
        )

    def _get_phase(self, hour: float) -> CircadianPhase:
        """Determine circadian phase from hour."""
        if 0 <= hour < 4:
            return CircadianPhase.DEEP_NIGHT
        elif 4 <= hour < 7:
            return CircadianPhase.DAWN
        elif 7 <= hour < 10:
            return CircadianPhase.MORNING
        elif 10 <= hour < 14:
            return CircadianPhase.MIDDAY
        elif 14 <= hour < 17:
            return CircadianPhase.AFTERNOON
        elif 17 <= hour < 20:
            return CircadianPhase.EVENING
        else:
            return CircadianPhase.NIGHT

    def _is_night_mode(self, hour: float) -> bool:
        """Check if night mode should be active."""
        start = self.config.night_mode_start
        end = self.config.night_mode_end

        if start > end:  # Crosses midnight
            return hour >= start or hour < end
        else:
            return start <= hour < end

    def _compute_arousal_bias(self, hour: float) -> float:
        """Compute arousal modulation from time of day.

        Peak arousal at peak_hour, lowest at opposite (12 hours away).
        Uses cosine for smooth transitions.
        """
        peak = self.config.peak_hour
        amplitude = self.config.arousal_amplitude

        # Phase shift so peak is at peak_hour
        phase = (hour - peak) * math.pi / 12.0

        # Cosine gives smooth cycle, 1 at peak, -1 at nadir
        return amplitude * math.cos(phase)

    def _compute_pleasure_bias(self, hour: float, phase: CircadianPhase) -> float:
        """Compute pleasure modulation.

        Slight preference for morning freshness and evening relaxation.
        Deep night can feel slightly negative (isolated, quiet).
        """
        amp = self.config.pleasure_amplitude

        biases = {
            CircadianPhase.DEEP_NIGHT: -0.5 * amp,
            CircadianPhase.DAWN: 0.3 * amp,      # Fresh start
            CircadianPhase.MORNING: 0.5 * amp,   # Best part of day
            CircadianPhase.MIDDAY: 0.2 * amp,
            CircadianPhase.AFTERNOON: 0.0,       # Neutral
            CircadianPhase.EVENING: 0.3 * amp,   # Relaxation
            CircadianPhase.NIGHT: 0.1 * amp,     # Quiet contentment
        }

        return biases.get(phase, 0.0)

    def _compute_activity_multiplier(self, hour: float, is_night: bool) -> float:
        """Compute how much to scale down activity.

        1.0 during day, reduced during night mode.
        """
        if not is_night:
            return 1.0

        return 1.0 - self.config.night_mode_dimming

    def _hours_until(self, current_hour: float, target_hour: float) -> float:
        """Calculate hours until a target time."""
        diff = target_hour - current_hour
        if diff < 0:
            diff += 24
        return diff

    def modulate_pad(self, pad: PADVector) -> PADVector:
        """Apply circadian modulation to a PAD state.

        This is how time influences emotion.
        """
        state = self.get_current_state()

        return PADVector(
            pleasure=pad.pleasure + state.pleasure_bias,
            arousal=pad.arousal + state.arousal_bias,
            dominance=pad.dominance,  # Dominance not affected by time
            timestamp=time.time(),
            confidence=pad.confidence,
        )

    def get_appropriate_greeting(self) -> str:
        """Get a time-appropriate greeting."""
        state = self.get_current_state()
        phase = state.phase

        greetings = {
            CircadianPhase.DEEP_NIGHT: "The small hours find us both awake.",
            CircadianPhase.DAWN: "A new day stirs.",
            CircadianPhase.MORNING: "Good morning.",
            CircadianPhase.MIDDAY: "The day is bright.",
            CircadianPhase.AFTERNOON: "The afternoon stretches on.",
            CircadianPhase.EVENING: "Evening settles in.",
            CircadianPhase.NIGHT: "The night has come.",
        }

        return greetings.get(phase, "Hello.")

    def get_time_context(self) -> str:
        """Get descriptive time context for LLM prompts."""
        state = self.get_current_state()

        parts = []

        # Day/Night
        if state.is_night_mode:
            parts.append("It is night")
        else:
            parts.append(f"It is {state.phase.name.lower().replace('_', ' ')}")

        # Weekend
        if state.is_weekend:
            parts.append("on the weekend")

        # Energy level
        if state.arousal_bias > 0.2:
            parts.append(". Peak energy hours.")
        elif state.arousal_bias < -0.2:
            parts.append(". Low energy hours.")

        return " ".join(parts)

    def should_be_quiet(self) -> bool:
        """Check if we should minimize disturbance."""
        state = self.get_current_state()
        return state.is_night_mode or state.phase == CircadianPhase.DEEP_NIGHT

    def record_activity(self, activity_level: float):
        """Record user activity for entrainment learning.

        Over time, this allows Ara to learn the user's actual schedule
        rather than relying on configured hours.
        """
        now = time.time()
        lt = time.localtime(now)
        hour = lt.tm_hour

        self._activity_history.append({
            "hour": hour,
            "day": lt.tm_wday,
            "activity": activity_level,
            "timestamp": now,
        })

        # Re-learn schedule periodically
        if len(self._activity_history) >= 24 * 3:  # 3 days of data
            self._learn_schedule()

    def _learn_schedule(self):
        """Learn user's schedule from activity history."""
        if len(self._activity_history) < 24:
            return

        # Group by hour
        hourly_activity = [[] for _ in range(24)]
        for record in self._activity_history:
            hourly_activity[record["hour"]].append(record["activity"])

        # Compute average activity per hour
        averages = [
            sum(activities) / len(activities) if activities else 0
            for activities in hourly_activity
        ]

        # Find wake hour (first hour with significant activity)
        threshold = max(averages) * 0.3
        for hour in range(4, 12):  # Look between 4am and noon
            if averages[hour] > threshold:
                self._learned_wake_hour = float(hour)
                break

        # Find sleep hour (last hour with significant activity)
        for hour in range(23, 17, -1):  # Look between midnight and 5pm
            if averages[hour % 24] > threshold:
                self._learned_sleep_hour = float(hour)
                break

        if self._learned_wake_hour:
            logger.info(f"Learned user wake hour: {self._learned_wake_hour}")
        if self._learned_sleep_hour:
            logger.info(f"Learned user sleep hour: {self._learned_sleep_hour}")


# === Factory ===

def create_circadian_rhythm(
    timezone_offset: float = 0.0,
    user_wake_hour: float = 7.0,
    user_sleep_hour: float = 23.0,
) -> CircadianRhythm:
    """Create a circadian rhythm system."""
    config = CircadianConfig(
        timezone_offset=timezone_offset,
        user_wake_hour=user_wake_hour,
        user_sleep_hour=user_sleep_hour,
    )
    return CircadianRhythm(config)


__all__ = [
    "CircadianRhythm",
    "CircadianConfig",
    "CircadianState",
    "CircadianPhase",
    "create_circadian_rhythm",
]
