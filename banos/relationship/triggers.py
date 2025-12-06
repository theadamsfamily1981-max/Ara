"""
Weaver Triggers - When The Weaver Creates
==========================================

The Weaver creates artifacts in response to meaningful moments:

1. NIGHTLY - End of day, after Dreamer processes
2. MORNING - Gentle greeting at session start
3. SYNOD_ECHO - After weekly review, echoing what you agreed to
4. STRESS_SURVIVED - After weathering a hard moment together
5. WIN - After completing something significant
6. THRESHOLD - When relationship metrics cross important lines

This module provides the trigger hooks that can be called by:
- Dreamer (night cycle)
- Synod (weekly review)
- HAL event handlers (stress/thermal events)
- Session manager (morning greeting)
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable

logger = logging.getLogger(__name__)


# =============================================================================
# Trigger Types
# =============================================================================

class WeaverTrigger:
    """Types of events that trigger The Weaver."""
    NIGHTLY = "nightly"
    MORNING = "morning_greeting"
    SYNOD_ECHO = "synod_echo"
    STRESS_SURVIVED = "stress_survived"
    WIN = "win"
    THRESHOLD = "threshold_crossed"
    DEEP_WORK_END = "deep_work_end"
    MANUAL = "manual_request"


# =============================================================================
# Trigger Conditions
# =============================================================================

class WeaverTriggerManager:
    """
    Manages when The Weaver should create artifacts.

    Tracks conditions and prevents over-triggering.
    """

    def __init__(self):
        self._last_trigger_times: Dict[str, datetime] = {}
        self._trigger_cooldowns: Dict[str, timedelta] = {
            WeaverTrigger.NIGHTLY: timedelta(hours=20),
            WeaverTrigger.MORNING: timedelta(hours=8),
            WeaverTrigger.SYNOD_ECHO: timedelta(days=5),
            WeaverTrigger.STRESS_SURVIVED: timedelta(hours=6),
            WeaverTrigger.WIN: timedelta(hours=2),
            WeaverTrigger.THRESHOLD: timedelta(hours=4),
            WeaverTrigger.DEEP_WORK_END: timedelta(hours=3),
            WeaverTrigger.MANUAL: timedelta(minutes=30),
        }

        # Weaver instance (lazy loaded)
        self._weaver = None

    def _get_weaver(self):
        """Lazy-load the Weaver."""
        if self._weaver is None:
            from banos.relationship.weaver import get_weaver
            self._weaver = get_weaver()
        return self._weaver

    def _can_trigger(self, trigger_type: str) -> bool:
        """Check if enough time has passed since last trigger of this type."""
        last = self._last_trigger_times.get(trigger_type)
        if last is None:
            return True

        cooldown = self._trigger_cooldowns.get(trigger_type, timedelta(hours=1))
        return datetime.now() - last >= cooldown

    def _record_trigger(self, trigger_type: str) -> None:
        """Record that a trigger just fired."""
        self._last_trigger_times[trigger_type] = datetime.now()

    # =========================================================================
    # Trigger Handlers
    # =========================================================================

    def on_night_cycle(self) -> Optional[Path]:
        """
        Called by Dreamer at end of night cycle.

        Creates a nightly gift reflecting on the day.
        """
        if not self._can_trigger(WeaverTrigger.NIGHTLY):
            logger.debug("Nightly trigger on cooldown")
            return None

        weaver = self._get_weaver()
        path = weaver.nightly_gift()

        if path:
            self._record_trigger(WeaverTrigger.NIGHTLY)
            logger.info(f"Nightly gift created: {path}")

        return path

    def on_session_start(self) -> Optional[Path]:
        """
        Called when a new session starts (morning greeting).
        """
        if not self._can_trigger(WeaverTrigger.MORNING):
            return None

        weaver = self._get_weaver()
        path = weaver.morning_gift()

        if path:
            self._record_trigger(WeaverTrigger.MORNING)
            logger.info(f"Morning gift created: {path}")

        return path

    def on_synod_complete(self, changes: Dict[str, Any]) -> Optional[Path]:
        """
        Called after Synod weekly review.

        Creates an artifact echoing what you agreed to.
        """
        if not self._can_trigger(WeaverTrigger.SYNOD_ECHO):
            return None

        weaver = self._get_weaver()
        path = weaver.synod_echo(changes)

        if path:
            self._record_trigger(WeaverTrigger.SYNOD_ECHO)
            logger.info(f"Synod echo created: {path}")

        return path

    def on_stress_survived(self, event_type: str = "crash") -> Optional[Path]:
        """
        Called after surviving a stressful event.

        Args:
            event_type: What happened (crash, thermal, long_debug, etc.)
        """
        if not self._can_trigger(WeaverTrigger.STRESS_SURVIVED):
            return None

        weaver = self._get_weaver()
        path = weaver.event_gift(f"stress_survived_{event_type}")

        if path:
            self._record_trigger(WeaverTrigger.STRESS_SURVIVED)
            logger.info(f"Stress survived gift created: {path}")

        return path

    def on_win(self, achievement: str) -> Optional[Path]:
        """
        Called after a significant win.

        Args:
            achievement: What was achieved
        """
        if not self._can_trigger(WeaverTrigger.WIN):
            return None

        weaver = self._get_weaver()
        path = weaver.event_gift(f"win_{achievement}")

        if path:
            self._record_trigger(WeaverTrigger.WIN)
            logger.info(f"Win gift created for {achievement}: {path}")

        return path

    def on_deep_work_end(self, duration_hours: float) -> Optional[Path]:
        """
        Called after a long deep work session ends.

        Args:
            duration_hours: How long the session was
        """
        if duration_hours < 2.0:
            return None  # Only trigger for significant sessions

        if not self._can_trigger(WeaverTrigger.DEEP_WORK_END):
            return None

        weaver = self._get_weaver()
        path = weaver.event_gift(f"deep_work_{duration_hours:.1f}h")

        if path:
            self._record_trigger(WeaverTrigger.DEEP_WORK_END)
            logger.info(f"Deep work gift created: {path}")

        return path

    def on_threshold_crossed(self, metric: str, direction: str) -> Optional[Path]:
        """
        Called when a relationship metric crosses a threshold.

        Args:
            metric: Which metric (synergy, coherence, etc.)
            direction: "up" or "down"
        """
        if not self._can_trigger(WeaverTrigger.THRESHOLD):
            return None

        weaver = self._get_weaver()
        path = weaver.event_gift(f"threshold_{metric}_{direction}")

        if path:
            self._record_trigger(WeaverTrigger.THRESHOLD)
            logger.info(f"Threshold gift created for {metric}: {path}")

        return path

    def on_manual_request(self, prompt: Optional[str] = None) -> Optional[Path]:
        """
        Called when Croft explicitly asks for a gift.

        Args:
            prompt: Optional specific prompt
        """
        if not self._can_trigger(WeaverTrigger.MANUAL):
            return None

        weaver = self._get_weaver()
        reason = prompt if prompt else "requested_gift"
        path = weaver.event_gift(reason)

        if path:
            self._record_trigger(WeaverTrigger.MANUAL)
            logger.info(f"Manual gift created: {path}")

        return path


# =============================================================================
# HAL Event Integration
# =============================================================================

def setup_hal_triggers(trigger_manager: WeaverTriggerManager) -> None:
    """
    Set up HAL event handlers that trigger The Weaver.

    This watches for stress events (thermal, crashes) and celebrations.
    """
    # Note: This would integrate with HAL's event system
    # For now, this is a placeholder that shows the intended integration

    logger.info("HAL triggers for Weaver configured")


# =============================================================================
# Egregore Integration
# =============================================================================

def check_egregore_thresholds(
    trigger_manager: WeaverTriggerManager,
    current_state: Dict[str, float],
    previous_state: Dict[str, float],
) -> List[Path]:
    """
    Check if Egregore metrics crossed important thresholds.

    Args:
        trigger_manager: The trigger manager
        current_state: Current Egregore state
        previous_state: Previous Egregore state

    Returns:
        List of artifact paths created
    """
    artifacts = []

    # Synergy thresholds
    synergy_now = current_state.get('synergy', 0.5)
    synergy_before = previous_state.get('synergy', 0.5)

    if synergy_now >= 0.8 and synergy_before < 0.8:
        path = trigger_manager.on_threshold_crossed('synergy', 'up')
        if path:
            artifacts.append(path)

    if synergy_now <= 0.3 and synergy_before > 0.3:
        # This is concerning, but the gift offers support
        path = trigger_manager.on_threshold_crossed('synergy', 'down')
        if path:
            artifacts.append(path)

    # Coherence thresholds
    coherence_now = current_state.get('coherence', 0.5)
    coherence_before = previous_state.get('coherence', 0.5)

    if coherence_now >= 0.9 and coherence_before < 0.9:
        path = trigger_manager.on_threshold_crossed('coherence', 'up')
        if path:
            artifacts.append(path)

    return artifacts


# =============================================================================
# Convenience
# =============================================================================

_default_trigger_manager: Optional[WeaverTriggerManager] = None


def get_trigger_manager() -> WeaverTriggerManager:
    """Get or create the default trigger manager."""
    global _default_trigger_manager
    if _default_trigger_manager is None:
        _default_trigger_manager = WeaverTriggerManager()
    return _default_trigger_manager


# Shortcut functions for common triggers

def trigger_nightly() -> Optional[Path]:
    """Trigger nightly gift."""
    return get_trigger_manager().on_night_cycle()


def trigger_morning() -> Optional[Path]:
    """Trigger morning greeting."""
    return get_trigger_manager().on_session_start()


def trigger_win(achievement: str) -> Optional[Path]:
    """Trigger win celebration."""
    return get_trigger_manager().on_win(achievement)


def trigger_stress_survived(event_type: str = "crash") -> Optional[Path]:
    """Trigger stress survived gift."""
    return get_trigger_manager().on_stress_survived(event_type)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'WeaverTrigger',
    'WeaverTriggerManager',
    'get_trigger_manager',
    'setup_hal_triggers',
    'check_egregore_thresholds',
    'trigger_nightly',
    'trigger_morning',
    'trigger_win',
    'trigger_stress_survived',
]
