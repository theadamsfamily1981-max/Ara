# ara/embodied/lizard/wake_protocol.py
"""
Wake Protocol - Criteria and events for waking the Cathedral.

The Wake Protocol defines what constitutes "interesting enough" to
justify the energy cost of waking the full cognitive system.

Wake Event Types:
    SALIENCE: General high-salience detection
    MOTION: Physical movement in the environment
    AUDIO: Significant sound event (speech, alarm, etc.)
    KEYWORD: Wake word detected ("Hey Ara", etc.)
    PROXIMITY: User presence detected
    SCHEDULE: Scheduled wake time
    THERMAL: Thermal event requiring attention
    MANUAL: Explicit wake request

Salience Levels:
    NOISE: Background noise, ignore
    AMBIENT: Environmental activity, log but don't wake
    NOTABLE: Interesting but not urgent
    URGENT: Requires immediate attention
    CRITICAL: Emergency, override all throttling
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Callable


class WakeEventType(Enum):
    """Types of events that can wake the Cathedral."""
    SALIENCE = auto()     # General salience threshold exceeded
    MOTION = auto()       # Motion detected
    AUDIO = auto()        # Significant audio event
    KEYWORD = auto()      # Wake word detected
    PROXIMITY = auto()    # User nearby
    SCHEDULE = auto()     # Scheduled wake
    THERMAL = auto()      # Thermal event
    NETWORK = auto()      # Network activity requiring response
    MANUAL = auto()       # Manual wake request
    DREAM_END = auto()    # Dream cycle completed


class SalienceLevel(Enum):
    """Levels of salience for events."""
    NOISE = 0         # Background noise, ignore
    AMBIENT = 1       # Environmental, log only
    NOTABLE = 2       # Interesting, consider waking
    URGENT = 3        # Requires attention
    CRITICAL = 4      # Emergency, must wake


@dataclass
class WakeEvent:
    """An event that may trigger Cathedral wake."""
    event_type: WakeEventType
    salience: float  # 0-1
    source: str      # Which sensor/subsystem
    reason: str      # Human-readable explanation

    timestamp: datetime = field(default_factory=datetime.now)
    sensor_data: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # Higher = more urgent

    # Processing metadata
    was_processed: bool = False
    wake_triggered: bool = False
    processing_time_ms: float = 0.0

    @property
    def level(self) -> SalienceLevel:
        """Get salience level from numeric value."""
        if self.salience < 0.2:
            return SalienceLevel.NOISE
        elif self.salience < 0.4:
            return SalienceLevel.AMBIENT
        elif self.salience < 0.6:
            return SalienceLevel.NOTABLE
        elif self.salience < 0.8:
            return SalienceLevel.URGENT
        else:
            return SalienceLevel.CRITICAL

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "event_type": self.event_type.name,
            "salience": self.salience,
            "level": self.level.name,
            "source": self.source,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority,
            "wake_triggered": self.wake_triggered,
        }


@dataclass
class WakeCriteria:
    """
    Criteria for determining when to wake the Cathedral.

    These can be adjusted based on:
    - Time of day (higher threshold at night)
    - User preferences
    - Power budget constraints
    - Current attractor basin
    """
    # Salience thresholds by event type
    thresholds: Dict[WakeEventType, float] = field(default_factory=lambda: {
        WakeEventType.SALIENCE: 0.6,
        WakeEventType.MOTION: 0.5,
        WakeEventType.AUDIO: 0.5,
        WakeEventType.KEYWORD: 0.3,      # Low threshold for wake word
        WakeEventType.PROXIMITY: 0.4,
        WakeEventType.SCHEDULE: 0.0,     # Always wake for scheduled
        WakeEventType.THERMAL: 0.7,
        WakeEventType.NETWORK: 0.6,
        WakeEventType.MANUAL: 0.0,       # Always wake for manual
        WakeEventType.DREAM_END: 0.0,
    })

    # Time-based modifiers (multiply threshold)
    night_multiplier: float = 1.5       # Higher threshold at night
    busy_multiplier: float = 1.3        # Higher when user marked busy

    # Power-based modifiers
    low_power_multiplier: float = 1.5   # Higher when battery low
    thermal_multiplier: float = 1.3     # Higher when running hot

    # Cooldowns (seconds between events of same type)
    cooldowns: Dict[WakeEventType, float] = field(default_factory=lambda: {
        WakeEventType.SALIENCE: 30.0,
        WakeEventType.MOTION: 10.0,
        WakeEventType.AUDIO: 5.0,
        WakeEventType.KEYWORD: 2.0,
        WakeEventType.PROXIMITY: 60.0,
        WakeEventType.SCHEDULE: 0.0,
        WakeEventType.THERMAL: 60.0,
        WakeEventType.NETWORK: 10.0,
        WakeEventType.MANUAL: 0.0,
        WakeEventType.DREAM_END: 0.0,
    })

    # Override patterns (regex for audio, keywords, etc.)
    wake_words: List[str] = field(default_factory=lambda: [
        "hey ara",
        "ara",
        "wake up",
        "help",
        "emergency",
    ])

    # Always-wake patterns
    critical_patterns: List[str] = field(default_factory=lambda: [
        "fire",
        "help",
        "emergency",
        "intruder",
    ])

    def get_threshold(
        self,
        event_type: WakeEventType,
        is_night: bool = False,
        is_busy: bool = False,
        low_power: bool = False,
        thermal_warning: bool = False,
    ) -> float:
        """Get adjusted threshold for an event type."""
        base = self.thresholds.get(event_type, 0.6)

        # Apply modifiers
        if is_night:
            base *= self.night_multiplier
        if is_busy:
            base *= self.busy_multiplier
        if low_power:
            base *= self.low_power_multiplier
        if thermal_warning:
            base *= self.thermal_multiplier

        return min(base, 1.0)  # Cap at 1.0

    def should_wake(
        self,
        event: WakeEvent,
        **context
    ) -> bool:
        """Determine if an event should trigger wake."""
        # Critical patterns always wake
        if event.level == SalienceLevel.CRITICAL:
            return True

        # Check threshold
        threshold = self.get_threshold(event.event_type, **context)
        return event.salience >= threshold


class WakeProtocol:
    """
    Manages wake criteria and event processing.

    Tracks event history, applies cooldowns, and determines
    when the Cathedral should be woken.
    """

    def __init__(self, criteria: Optional[WakeCriteria] = None):
        self.criteria = criteria or WakeCriteria()
        self._event_history: List[WakeEvent] = []
        self._last_wake_by_type: Dict[WakeEventType, datetime] = {}

    def evaluate(
        self,
        event: WakeEvent,
        is_night: bool = False,
        is_busy: bool = False,
        low_power: bool = False,
        thermal_warning: bool = False,
    ) -> bool:
        """
        Evaluate whether an event should wake the Cathedral.

        Args:
            event: The wake event to evaluate
            is_night: Whether it's nighttime
            is_busy: Whether user is marked busy
            low_power: Whether in low power mode
            thermal_warning: Whether thermal warning active

        Returns:
            True if Cathedral should wake
        """
        # Record event
        self._event_history.append(event)

        # Trim history (keep last 1000 events)
        if len(self._event_history) > 1000:
            self._event_history = self._event_history[-1000:]

        # Check cooldown
        if not self._check_cooldown(event):
            event.was_processed = True
            event.wake_triggered = False
            return False

        # Evaluate against criteria
        context = {
            "is_night": is_night,
            "is_busy": is_busy,
            "low_power": low_power,
            "thermal_warning": thermal_warning,
        }

        should_wake = self.criteria.should_wake(event, **context)

        event.was_processed = True
        event.wake_triggered = should_wake

        if should_wake:
            self._last_wake_by_type[event.event_type] = datetime.now()

        return should_wake

    def _check_cooldown(self, event: WakeEvent) -> bool:
        """Check if cooldown has elapsed for this event type."""
        last_wake = self._last_wake_by_type.get(event.event_type)

        if last_wake is None:
            return True

        cooldown = self.criteria.cooldowns.get(event.event_type, 30.0)
        elapsed = (datetime.now() - last_wake).total_seconds()

        return elapsed >= cooldown

    def check_wake_word(self, audio_text: str) -> Optional[WakeEvent]:
        """
        Check if audio contains a wake word.

        Args:
            audio_text: Transcribed audio text

        Returns:
            WakeEvent if wake word detected, None otherwise
        """
        text_lower = audio_text.lower()

        # Check critical patterns first
        for pattern in self.criteria.critical_patterns:
            if pattern in text_lower:
                return WakeEvent(
                    event_type=WakeEventType.KEYWORD,
                    salience=1.0,
                    source="audio_transcription",
                    reason=f"Critical pattern detected: {pattern}",
                    priority=100,
                )

        # Check wake words
        for word in self.criteria.wake_words:
            if word in text_lower:
                return WakeEvent(
                    event_type=WakeEventType.KEYWORD,
                    salience=0.8,
                    source="audio_transcription",
                    reason=f"Wake word detected: {word}",
                    priority=50,
                )

        return None

    def get_recent_events(
        self,
        count: int = 10,
        event_type: Optional[WakeEventType] = None,
    ) -> List[WakeEvent]:
        """Get recent wake events."""
        events = self._event_history

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        return events[-count:]

    def get_wake_statistics(self) -> Dict[str, Any]:
        """Get statistics about wake events."""
        if not self._event_history:
            return {"total_events": 0}

        total = len(self._event_history)
        wakes = sum(1 for e in self._event_history if e.wake_triggered)

        by_type = {}
        for event_type in WakeEventType:
            type_events = [e for e in self._event_history if e.event_type == event_type]
            by_type[event_type.name] = {
                "count": len(type_events),
                "wakes": sum(1 for e in type_events if e.wake_triggered),
            }

        return {
            "total_events": total,
            "wake_events": wakes,
            "wake_rate": wakes / total if total > 0 else 0,
            "by_type": by_type,
        }
