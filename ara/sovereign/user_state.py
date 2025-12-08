"""
User State - MindReader Output
==============================

This module tracks Croft's state - the human behind the keyboard.
It's what MindReader produces after analyzing:
- Time of day
- Recent activity patterns
- Emotional signals
- Physical state (from sensors if available)
- Calendar/schedule context

The ChiefOfStaff uses this to implement Founder Protection:
- Don't let him grind at 2am
- Detect burnout before it happens
- Protect flow state when it's precious
- Know when to push vs. when to shield
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any


class CognitiveMode(Enum):
    """Current cognitive mode/state."""
    FLOW = "flow"              # Deep work, don't interrupt
    EXPLORATION = "exploration"  # Curious, open to tangents
    DECOMPRESS = "decompress"  # Recovery, low cognitive load
    ADMIN = "admin"            # Maintenance tasks, interruptible
    EMERGENCY = "emergency"    # Crisis response mode
    CREATIVE = "creative"      # Art, music, synthesis
    DEPLETED = "depleted"      # Low energy, needs protection


class ProtectionLevel(Enum):
    """How much protection the user needs right now."""
    NONE = "none"              # User is fine, full autonomy
    LIGHT = "light"            # Some guardrails
    MODERATE = "moderate"      # Be careful
    HIGH = "high"              # Protect aggressively
    LOCKOUT = "lockout"        # Hard stop, no work allowed


@dataclass
class UserState:
    """
    Current state of the human (Croft).

    This is what MindReader produces after analyzing all available signals.
    The ChiefOfStaff uses this for Founder Protection decisions.
    """

    # Identity
    timestamp: datetime = field(default_factory=datetime.utcnow)
    user_id: str = "croft"  # In case we ever need multi-user

    # Energy levels (0-1)
    fatigue: float = 0.0           # Physical/mental tiredness
    burnout_risk: float = 0.0      # Long-term exhaustion risk
    stress: float = 0.0            # Current stress level
    focus_capacity: float = 1.0    # How much focus available

    # Daily tracking
    flow_hours_used_today: float = 0.0  # Deep work hours spent
    flow_hours_budget: float = 4.0       # Max sustainable flow hours
    total_work_hours_today: float = 0.0
    breaks_taken_today: int = 0

    # Cognitive state
    current_mode: CognitiveMode = CognitiveMode.ADMIN
    mode_started_at: datetime = field(default_factory=datetime.utcnow)
    protection_level: ProtectionLevel = ProtectionLevel.NONE

    # Time context
    is_night: bool = False         # Late night (danger zone)
    night_lockout_start: int = 2   # Hour when lockout starts (2am)
    night_lockout_end: int = 6     # Hour when lockout ends (6am)

    # Recent history
    last_break: Optional[datetime] = None
    last_meal: Optional[datetime] = None
    hours_since_sleep: float = 0.0

    # Emotional state (from affect detection if available)
    emotional_valence: float = 0.0  # -1 (negative) to +1 (positive)
    emotional_arousal: float = 0.5  # 0 (calm) to 1 (activated)

    # Context
    active_projects: List[str] = field(default_factory=list)
    recent_wins: int = 0           # Positive events today
    recent_setbacks: int = 0       # Negative events today

    # =========================================================================
    # Computed Properties
    # =========================================================================

    def focus_intensity_factor(self) -> float:
        """
        How intensely can the user focus right now?

        Returns 0-1 multiplier for cognitive tasks.
        """
        # Base capacity minus fatigue
        base = self.focus_capacity * (1.0 - self.fatigue * 0.7)

        # Burnout reduces capacity significantly
        if self.burnout_risk > 0.5:
            base *= 0.6
        elif self.burnout_risk > 0.3:
            base *= 0.8

        # Stress is a double-edged sword
        # Low stress = no urgency, moderate stress = peak performance,
        # high stress = degraded performance
        if self.stress < 0.2:
            base *= 0.9  # Too relaxed
        elif self.stress < 0.5:
            base *= 1.1  # Optimal stress
        else:
            base *= max(0.5, 1.0 - (self.stress - 0.5))  # Overstressed

        # Night penalty
        if self.is_night:
            base *= 0.6

        return max(0.0, min(1.0, base))

    def is_in_night_lockout_window(self, now: Optional[datetime] = None) -> bool:
        """Check if we're in the night lockout window (e.g., 2am-6am)."""
        now = now or datetime.now()
        hour = now.hour
        # Lockout is active from night_lockout_start to night_lockout_end
        return self.night_lockout_start <= hour < self.night_lockout_end

    def time_until_lockout_ends(self, now: Optional[datetime] = None) -> timedelta:
        """How long until the lockout window ends?"""
        now = now or datetime.now()
        if not self.is_in_night_lockout_window(now):
            return timedelta(0)

        # Calculate time until lockout_end hour
        end_time = now.replace(
            hour=self.night_lockout_end,
            minute=0,
            second=0,
            microsecond=0
        )
        if now.hour >= self.night_lockout_end:
            # Lockout ends tomorrow
            end_time += timedelta(days=1)

        return end_time - now

    def flow_budget_remaining(self) -> float:
        """How many flow hours are left today?"""
        return max(0.0, self.flow_hours_budget - self.flow_hours_used_today)

    def needs_break(self) -> bool:
        """Does the user need a break right now?"""
        if self.last_break is None:
            return self.total_work_hours_today > 2.0

        hours_since_break = (
            datetime.utcnow() - self.last_break
        ).total_seconds() / 3600

        # Need break after 90 minutes of work
        return hours_since_break > 1.5

    def compute_protection_level(self) -> ProtectionLevel:
        """
        Compute the appropriate protection level.

        This is the core Founder Protection logic.
        """
        # Hard lockout during night hours
        if self.is_in_night_lockout_window():
            return ProtectionLevel.LOCKOUT

        # High burnout risk = high protection
        if self.burnout_risk > 0.7:
            return ProtectionLevel.HIGH

        # Very fatigued = moderate protection
        if self.fatigue > 0.8:
            return ProtectionLevel.HIGH
        elif self.fatigue > 0.6:
            return ProtectionLevel.MODERATE

        # Flow budget exhausted = moderate protection
        if self.flow_budget_remaining() <= 0:
            return ProtectionLevel.MODERATE

        # Moderate burnout risk = light protection
        if self.burnout_risk > 0.4:
            return ProtectionLevel.LIGHT

        # Default: no special protection needed
        return ProtectionLevel.NONE

    def can_do_deep_work(self) -> tuple[bool, str]:
        """
        Can the user do deep work right now?

        Returns (can_do, reason).
        """
        protection = self.compute_protection_level()

        if protection == ProtectionLevel.LOCKOUT:
            return False, "Night lockout active - no deep work allowed"

        if protection == ProtectionLevel.HIGH:
            if self.burnout_risk > 0.7:
                return False, "Burnout risk too high - rest required"
            if self.fatigue > 0.8:
                return False, "Too fatigued for deep work"
            return False, "Protection level too high"

        if self.flow_budget_remaining() <= 0:
            return False, "Flow budget exhausted for today"

        if self.current_mode == CognitiveMode.DEPLETED:
            return False, "Currently in depleted state"

        return True, "Clear for deep work"

    def update_from_telemetry(self, telemetry: Dict[str, Any]) -> None:
        """Update state from telemetry sources."""
        self.timestamp = datetime.utcnow()

        # Time-based updates
        now = datetime.now()
        self.is_night = self.is_in_night_lockout_window(now)

        # Telemetry updates (if available)
        if "fatigue" in telemetry:
            self.fatigue = telemetry["fatigue"]
        if "stress" in telemetry:
            self.stress = telemetry["stress"]
        if "emotional_valence" in telemetry:
            self.emotional_valence = telemetry["emotional_valence"]
        if "flow_hours" in telemetry:
            self.flow_hours_used_today = telemetry["flow_hours"]

        # Recompute derived values
        self.protection_level = self.compute_protection_level()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "fatigue": self.fatigue,
            "burnout_risk": self.burnout_risk,
            "stress": self.stress,
            "focus_capacity": self.focus_capacity,
            "flow_hours_used_today": self.flow_hours_used_today,
            "flow_hours_budget": self.flow_hours_budget,
            "current_mode": self.current_mode.value,
            "protection_level": self.protection_level.value,
            "is_night": self.is_night,
            "focus_intensity": self.focus_intensity_factor(),
            "can_deep_work": self.can_do_deep_work()[0],
            "flow_remaining": self.flow_budget_remaining(),
        }


# =============================================================================
# MindReader Stub (will be replaced with real telemetry)
# =============================================================================

class MindReader:
    """
    Reads user state from various telemetry sources.

    For now this is a stub that uses time-based heuristics.
    Later it will integrate with:
    - Calendar
    - Activity monitoring
    - Physiological sensors (if available)
    - Emotional affect detection
    """

    def __init__(self):
        self._state = UserState()
        self._last_update = datetime.utcnow()

    def read(self, telemetry: Optional[Dict[str, Any]] = None) -> UserState:
        """
        Read current user state.

        Args:
            telemetry: Optional telemetry data to incorporate

        Returns:
            Current UserState
        """
        now = datetime.utcnow()
        hours_elapsed = (now - self._last_update).total_seconds() / 3600

        # Time-based fatigue accumulation (stub logic)
        current_hour = datetime.now().hour

        # Night detection
        self._state.is_night = self._state.is_in_night_lockout_window()

        # Fatigue increases through the day
        if 6 <= current_hour < 12:
            base_fatigue = 0.2
        elif 12 <= current_hour < 18:
            base_fatigue = 0.4
        elif 18 <= current_hour < 22:
            base_fatigue = 0.6
        else:
            base_fatigue = 0.8

        self._state.fatigue = base_fatigue

        # Apply external telemetry if provided
        if telemetry:
            self._state.update_from_telemetry(telemetry)

        # Recompute protection level
        self._state.protection_level = self._state.compute_protection_level()
        self._state.timestamp = now
        self._last_update = now

        return self._state

    def report_event(self, event_type: str, data: Optional[Dict[str, Any]] = None) -> None:
        """
        Report an event that affects user state.

        Events: "break_taken", "meal_eaten", "flow_started", "flow_ended",
                "win", "setback", "mode_change"
        """
        now = datetime.utcnow()

        if event_type == "break_taken":
            self._state.last_break = now
            self._state.breaks_taken_today += 1
            self._state.fatigue = max(0, self._state.fatigue - 0.1)

        elif event_type == "meal_eaten":
            self._state.last_meal = now
            self._state.fatigue = max(0, self._state.fatigue - 0.05)

        elif event_type == "flow_started":
            self._state.current_mode = CognitiveMode.FLOW
            self._state.mode_started_at = now

        elif event_type == "flow_ended":
            duration = (now - self._state.mode_started_at).total_seconds() / 3600
            self._state.flow_hours_used_today += duration
            self._state.current_mode = CognitiveMode.ADMIN

        elif event_type == "win":
            self._state.recent_wins += 1
            self._state.emotional_valence = min(1.0, self._state.emotional_valence + 0.1)

        elif event_type == "setback":
            self._state.recent_setbacks += 1
            self._state.emotional_valence = max(-1.0, self._state.emotional_valence - 0.1)
            self._state.stress = min(1.0, self._state.stress + 0.1)

        elif event_type == "mode_change" and data:
            new_mode = data.get("mode")
            if new_mode:
                self._state.current_mode = CognitiveMode(new_mode)
                self._state.mode_started_at = now


# =============================================================================
# Singleton Access
# =============================================================================

_mind_reader: Optional[MindReader] = None


def get_mind_reader() -> MindReader:
    """Get the default MindReader instance."""
    global _mind_reader
    if _mind_reader is None:
        _mind_reader = MindReader()
    return _mind_reader


def get_user_state(telemetry: Optional[Dict[str, Any]] = None) -> UserState:
    """Convenience function to get current user state."""
    return get_mind_reader().read(telemetry)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'CognitiveMode',
    'ProtectionLevel',
    'UserState',
    'MindReader',
    'get_mind_reader',
    'get_user_state',
]
