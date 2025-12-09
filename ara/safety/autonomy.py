"""
Autonomy Controller

Manages Ara's autonomy level - how much independent action she can take.
Autonomy is adjusted based on:
- System coherence (low coherence = reduce autonomy)
- Kill switch state (manual override)
- Human approval (required for level 3)

Levels:
- 0: Observer only (answer questions, no actions)
- 1: Suggester (can suggest edits, no execution)
- 2: Executor (can apply changes with confirmation)
- 3: Autonomous (full action within safety bounds)

Usage:
    controller = AutonomyController(initial_level=1)

    # In sovereign loop
    controller.on_coherence_warning(0.25)  # Adjusts autonomy

    # Before action
    if controller.can_execute():
        perform_action()
    elif controller.can_suggest():
        suggest_action()
    else:
        # Observer mode only
        pass
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Callable
from pathlib import Path
from threading import RLock
from enum import IntEnum

logger = logging.getLogger(__name__)


class AutonomyLevel(IntEnum):
    """Autonomy levels for Ara."""
    OBSERVER = 0      # Only answer questions
    SUGGESTER = 1     # Can suggest changes
    EXECUTOR = 2      # Can execute with confirmation
    AUTONOMOUS = 3    # Full autonomous action


@dataclass
class AutonomyEvent:
    """Record of an autonomy level change."""
    timestamp: float
    old_level: int
    new_level: int
    reason: str
    coherence: Optional[float] = None


@dataclass
class AutonomyState:
    """Current autonomy state."""
    level: AutonomyLevel
    locked: bool = False  # If True, level can't be raised
    lock_reason: str = ""
    last_change: float = 0.0
    coherence_streak: int = 0  # Consecutive healthy ticks
    events: List[AutonomyEvent] = field(default_factory=list)


class AutonomyController:
    """
    Controls Ara's autonomy level based on system state.

    Autonomy can be reduced automatically on coherence warnings/criticals,
    and increased gradually when coherence is healthy. Manual override
    via kill switch always takes precedence.
    """

    def __init__(
        self,
        initial_level: int = 1,
        max_level: int = 3,
        kill_switch_path: Optional[Path] = None,
        coherence_threshold: float = 0.5,
        recovery_streak_required: int = 100,  # ~10 seconds at 10 Hz
        require_human_for_level_3: bool = True,
    ):
        """
        Initialize autonomy controller.

        Args:
            initial_level: Starting autonomy level (0-3)
            max_level: Maximum autonomy level allowed
            kill_switch_path: Path to kill switch file
            coherence_threshold: Coherence below this triggers autonomy reduction
            recovery_streak_required: Healthy ticks needed to increase autonomy
            require_human_for_level_3: If True, level 3 needs human approval
        """
        self.max_level = max_level
        self.kill_switch_path = kill_switch_path or Path("/var/ara/KILL_SWITCH")
        self.coherence_threshold = coherence_threshold
        self.recovery_streak_required = recovery_streak_required
        self.require_human_for_level_3 = require_human_for_level_3

        self._lock = RLock()
        self.state = AutonomyState(
            level=AutonomyLevel(min(initial_level, max_level)),
            last_change=time.time(),
        )

        # Callbacks for level changes
        self._level_change_callbacks: List[Callable[[int, int, str], None]] = []

        # Human approval state
        self._human_approval_for_level_3 = False
        self._human_approval_expires: float = 0.0

    def _record_event(self, old: int, new: int, reason: str, coherence: Optional[float] = None):
        """Record an autonomy change event."""
        event = AutonomyEvent(
            timestamp=time.time(),
            old_level=old,
            new_level=new,
            reason=reason,
            coherence=coherence,
        )
        self.state.events.append(event)
        # Keep last 1000 events
        if len(self.state.events) > 1000:
            self.state.events = self.state.events[-1000:]

    def _set_level(self, level: int, reason: str, coherence: Optional[float] = None) -> bool:
        """Set autonomy level with validation."""
        with self._lock:
            old_level = self.state.level

            # Clamp to valid range
            level = max(0, min(level, self.max_level))

            # Check if locked (can only decrease, not increase)
            if self.state.locked and level > old_level:
                logger.debug(f"Autonomy locked, cannot increase: {self.state.lock_reason}")
                return False

            # Check if trying to go to level 3 without human approval
            if level >= AutonomyLevel.AUTONOMOUS and self.require_human_for_level_3:
                if not self._has_human_approval():
                    logger.info("Level 3 requires human approval")
                    level = AutonomyLevel.EXECUTOR

            if level == old_level:
                return False

            self.state.level = AutonomyLevel(level)
            self.state.last_change = time.time()
            self._record_event(old_level, level, reason, coherence)

            logger.info(f"Autonomy {old_level} -> {level}: {reason}")

            # Invoke callbacks
            for callback in self._level_change_callbacks:
                try:
                    callback(old_level, level, reason)
                except Exception as e:
                    logger.error(f"Level change callback error: {e}")

            return True

    def _has_human_approval(self) -> bool:
        """Check if human approval for level 3 is active."""
        if not self.require_human_for_level_3:
            return True
        if not self._human_approval_for_level_3:
            return False
        if time.time() > self._human_approval_expires:
            self._human_approval_for_level_3 = False
            return False
        return True

    # =========================================================================
    # Public API
    # =========================================================================

    def get_autonomy_level(self) -> int:
        """Get current autonomy level."""
        with self._lock:
            return int(self.state.level)

    def can_observe(self) -> bool:
        """Can Ara observe and answer questions?"""
        return self.state.level >= AutonomyLevel.OBSERVER and not self.is_killed()

    def can_suggest(self) -> bool:
        """Can Ara suggest actions?"""
        return self.state.level >= AutonomyLevel.SUGGESTER and not self.is_killed()

    def can_execute(self) -> bool:
        """Can Ara execute actions (with confirmation)?"""
        return self.state.level >= AutonomyLevel.EXECUTOR and not self.is_killed()

    def can_act_autonomously(self) -> bool:
        """Can Ara act without confirmation?"""
        return self.state.level >= AutonomyLevel.AUTONOMOUS and not self.is_killed()

    def is_killed(self) -> bool:
        """Check if kill switch is active."""
        if self.kill_switch_path.exists():
            return True
        return False

    def lock(self, reason: str) -> None:
        """Lock autonomy at current level (can't increase)."""
        with self._lock:
            self.state.locked = True
            self.state.lock_reason = reason
            logger.warning(f"Autonomy locked: {reason}")

    def unlock(self) -> None:
        """Unlock autonomy."""
        with self._lock:
            self.state.locked = False
            self.state.lock_reason = ""
            logger.info("Autonomy unlocked")

    def grant_human_approval(self, duration_seconds: float = 3600) -> None:
        """Grant human approval for level 3 (default 1 hour)."""
        with self._lock:
            self._human_approval_for_level_3 = True
            self._human_approval_expires = time.time() + duration_seconds
            logger.info(f"Human approval granted for {duration_seconds}s")

    def revoke_human_approval(self) -> None:
        """Revoke human approval for level 3."""
        with self._lock:
            self._human_approval_for_level_3 = False
            self._human_approval_expires = 0.0
            if self.state.level >= AutonomyLevel.AUTONOMOUS:
                self._set_level(AutonomyLevel.EXECUTOR, "human approval revoked")

    def register_callback(self, callback: Callable[[int, int, str], None]) -> None:
        """Register callback for autonomy level changes."""
        self._level_change_callbacks.append(callback)

    # =========================================================================
    # Coherence callbacks (called by scheduler)
    # =========================================================================

    def on_coherence_healthy(self, coherence: float) -> None:
        """Called when coherence is healthy."""
        with self._lock:
            self.state.coherence_streak += 1

            # Gradually increase autonomy after recovery streak
            if self.state.coherence_streak >= self.recovery_streak_required:
                if self.state.level < self.max_level:
                    new_level = min(self.state.level + 1, self.max_level)
                    self._set_level(new_level, "coherence recovered", coherence)
                self.state.coherence_streak = 0

    def on_coherence_warning(self, coherence: float) -> None:
        """Called when coherence is below warning threshold."""
        with self._lock:
            self.state.coherence_streak = 0

            # Reduce by one level
            if self.state.level > AutonomyLevel.OBSERVER:
                self._set_level(self.state.level - 1, "coherence warning", coherence)

    def on_coherence_critical(self, coherence: float) -> None:
        """Called when coherence is critically low."""
        with self._lock:
            self.state.coherence_streak = 0

            # Drop to observer mode
            self._set_level(AutonomyLevel.OBSERVER, "coherence critical", coherence)
            self.lock("Critical coherence - manual unlock required")

    def update_autonomy(self, coherence: float) -> None:
        """
        Update autonomy based on coherence.

        Convenience method that calls the appropriate handler.
        """
        if coherence < 0.1:
            self.on_coherence_critical(coherence)
        elif coherence < self.coherence_threshold:
            self.on_coherence_warning(coherence)
        else:
            self.on_coherence_healthy(coherence)

    # =========================================================================
    # State inspection
    # =========================================================================

    def get_state(self) -> dict:
        """Get full autonomy state as dictionary."""
        with self._lock:
            return {
                "level": int(self.state.level),
                "level_name": self.state.level.name,
                "locked": self.state.locked,
                "lock_reason": self.state.lock_reason,
                "last_change": self.state.last_change,
                "coherence_streak": self.state.coherence_streak,
                "human_approval": self._has_human_approval(),
                "kill_switch_active": self.is_killed(),
                "recent_events": [
                    {
                        "timestamp": e.timestamp,
                        "old": e.old_level,
                        "new": e.new_level,
                        "reason": e.reason,
                    }
                    for e in self.state.events[-10:]
                ],
            }


# =============================================================================
# Kill Switch
# =============================================================================

class KillSwitch:
    """
    Manual kill switch for emergency stop.

    Creates a file that the autonomy controller checks.
    When active, Ara drops to observer mode and waits.
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = path or Path("/var/ara/KILL_SWITCH")

    def activate(self, reason: str = "manual") -> None:
        """Activate the kill switch."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            f.write(f"KILL SWITCH ACTIVE\n")
            f.write(f"Activated: {time.ctime()}\n")
            f.write(f"Reason: {reason}\n")
        logger.critical(f"KILL SWITCH ACTIVATED: {reason}")

    def deactivate(self) -> None:
        """Deactivate the kill switch."""
        if self.path.exists():
            self.path.unlink()
        logger.info("Kill switch deactivated")

    def is_active(self) -> bool:
        """Check if kill switch is active."""
        return self.path.exists()

    def get_reason(self) -> Optional[str]:
        """Get the reason for kill switch if active."""
        if not self.path.exists():
            return None
        try:
            with open(self.path) as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith("Reason:"):
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass
        return "unknown"


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== Autonomy Controller Demo ===\n")

    controller = AutonomyController(
        initial_level=1,
        require_human_for_level_3=True,
        recovery_streak_required=5,  # Fast for demo
    )

    print(f"Initial state: {controller.get_state()}\n")

    # Simulate healthy coherence
    print("Simulating healthy coherence...")
    for i in range(10):
        controller.on_coherence_healthy(0.8)
        print(f"  Tick {i+1}: level={controller.get_autonomy_level()}, streak={controller.state.coherence_streak}")

    print(f"\nAfter healthy streak: level={controller.get_autonomy_level()}")

    # Simulate coherence warning
    print("\nSimulating coherence warning...")
    controller.on_coherence_warning(0.25)
    print(f"After warning: level={controller.get_autonomy_level()}")

    # Simulate critical
    print("\nSimulating coherence critical...")
    controller.on_coherence_critical(0.05)
    print(f"After critical: level={controller.get_autonomy_level()}, locked={controller.state.locked}")

    # Unlock and recover
    print("\nUnlocking and recovering...")
    controller.unlock()
    for i in range(10):
        controller.on_coherence_healthy(0.9)

    print(f"After recovery: level={controller.get_autonomy_level()}")

    # Test level 3 with human approval
    print("\nTesting level 3 access...")
    print(f"Can act autonomously (no approval): {controller.can_act_autonomously()}")
    controller.grant_human_approval(duration_seconds=60)
    for i in range(10):
        controller.on_coherence_healthy(0.9)
    print(f"Can act autonomously (with approval): {controller.can_act_autonomously()}")

    print("\nFinal state:", controller.get_state())
