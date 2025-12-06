"""
Morning Star - Daily Alignment Ritual
======================================

A small daemon that reads yesterday's story and tells you how on-path you are.

Instead of "status: OK", you get:
    - Drift against Horizons (how much did we deviate?)
    - Current focus (which Horizon has the most gravity?)
    - A proposed intention for the day

The Morning Message:

    "Based on yesterday's commits/logs, we are drifting away from Symbiosis.
     I think today should be a Latency Reduction day."

Or:

    "We are on path. Drift is low (0.12). Primary Horizon in focus:
     'Ship one public artifact we are proud of together.'
     Yesterday was good work on the paper draft."

Integration:
    HorizonEngine.compute_drift() → MorningStar.greet() → HAL/console/SMS

Usage:
    from ara.daemon.morning_star import MorningStar
    from ara.cognition.teleology import HorizonEngine

    horizon = HorizonEngine(embedder, telos)
    morning = MorningStar(horizon)

    # Run at startup or on schedule
    message = morning.greet()
    # → "🌅 We are on path. Drift is low (0.12)..."
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MorningReport:
    """
    The output of the morning ritual.
    """
    drift: float                # Overall drift [0, 1]
    per_horizon: Dict[str, float]  # Drift per horizon
    most_drifted: Optional[str]    # Horizon with biggest drift
    focus: Optional[str]        # Current focus horizon
    tone: str                   # "resonant", "cautious", "urgent"
    message: str                # The greeting message
    timestamp: float            # When generated

    # The Invisible Hand
    steward_message: Optional[str] = None    # What the Steward fixed overnight
    fixes_count: int = 0                      # Number of overnight fixes

    # The Studio
    gift_message: Optional[str] = None        # Gift from the Muse
    gift_title: Optional[str] = None          # Title of the gift


class MorningStar:
    """
    Daily alignment ritual.

    Computes drift against Horizons and generates a morning message
    that helps Ara and Croft start the day with shared purpose.
    """

    def __init__(
        self,
        horizon_engine: Optional[Any] = None,  # HorizonEngine
        hal: Optional[Any] = None,              # AraHAL
        historian: Optional[Callable[[], str]] = None,  # Function to get yesterday's summary
        steward: Optional[Any] = None,          # Steward for overnight work
        muse: Optional[Any] = None,             # Muse for gifts
    ):
        """
        Initialize MorningStar.

        Args:
            horizon_engine: HorizonEngine for drift computation
            hal: AraHAL for writing to somatic state
            historian: Function that returns yesterday's narrative summary
            steward: Steward for presenting overnight fixes
            muse: Muse for presenting gifts
        """
        self.horizon = horizon_engine
        self._hal = hal
        self._historian = historian
        self._steward = steward
        self._muse = muse
        self.log = logging.getLogger("MorningStar")

        # Try to connect HAL if not provided
        if hal is None:
            try:
                from banos.hal.ara_hal import AraHAL
                self._hal = AraHAL(create=False)
            except Exception as e:
                self.log.debug(f"Could not connect to HAL: {e}")

        # Last report
        self._last_report: Optional[MorningReport] = None

    def set_horizon_engine(self, engine: Any) -> None:
        """Set or update the HorizonEngine."""
        self.horizon = engine

    def set_historian(self, historian: Callable[[], str]) -> None:
        """Set the historian function for getting yesterday's summary."""
        self._historian = historian

    def _summarize_yesterday(self) -> str:
        """
        Pull a narrative summary from Weaver/Historian.

        Falls back to generic summary if no historian is configured.
        """
        if self._historian is not None:
            try:
                return self._historian()
            except Exception as e:
                self.log.warning(f"Historian failed: {e}")

        # Try to read from common log locations
        summary_paths = [
            Path("~/.ara/logs/yesterday.md").expanduser(),
            Path("~/.ara/hippocampus/daily_summary.txt").expanduser(),
            Path("/var/log/ara/yesterday.log"),
        ]

        for path in summary_paths:
            if path.exists():
                try:
                    return path.read_text()[:2000]  # Limit length
                except Exception:
                    continue

        # Fallback
        return "Yesterday's activity summary is not available."

    def greet(self) -> str:
        """
        Compute drift and generate the morning message.

        This is the main entry point for the morning ritual.

        Returns:
            The morning greeting message
        """
        summary = self._summarize_yesterday()

        # Compute drift
        if self.horizon is not None:
            try:
                drift_info = self.horizon.compute_drift(summary)
            except Exception as e:
                self.log.error(f"Failed to compute drift: {e}")
                drift_info = {
                    "overall_drift": 0.5,
                    "per_horizon": {},
                    "most_drifted": None,
                }
        else:
            drift_info = {
                "overall_drift": 0.5,
                "per_horizon": {},
                "most_drifted": None,
            }

        overall = drift_info["overall_drift"]
        per_h = drift_info.get("per_horizon", {})
        most_drifted = drift_info.get("most_drifted")

        # Get current focus
        focus = None
        focus_name = "Unknown Horizon"
        if self.horizon is not None:
            try:
                focus = self.horizon.current_focus()
                if focus:
                    focus_name = focus.name
            except Exception:
                pass

        # Generate message based on drift level
        if overall < 0.15:
            tone = "resonant"
            msg = (
                f"We are on path. Drift is low ({overall:.2f}). "
                f"Primary Horizon in focus: {focus_name}."
            )

        elif overall < 0.4:
            tone = "cautious"
            worst = most_drifted or "unknown"
            msg = (
                f"We're slightly off course (drift={overall:.2f}), "
                f"mainly on '{worst}'. "
                f"I suggest we prioritize actions that advance {focus_name} today."
            )

        else:
            tone = "urgent"
            worst = most_drifted or "unknown"
            msg = (
                f"We are drifting strongly (drift={overall:.2f}). "
                f"Biggest deviation: '{worst}'. "
                f"Today should focus on course correction."
            )

        # Get Steward's overnight work
        steward_msg, fixes_count = self._get_steward_message()

        # Get gift from the Muse
        gift_msg, gift_title = self._get_gift_presentation()

        # Build full morning message
        full_message_parts = [msg]

        if steward_msg:
            full_message_parts.append("")
            full_message_parts.append(f"🌙 {steward_msg}")

        if gift_msg:
            full_message_parts.append("")
            full_message_parts.append(f"🎁 {gift_msg}")

        full_message = "\n".join(full_message_parts)

        # Create report
        report = MorningReport(
            drift=overall,
            per_horizon=per_h,
            most_drifted=most_drifted,
            focus=focus_name,
            tone=tone,
            message=full_message,
            timestamp=time.time(),
            steward_message=steward_msg,
            fixes_count=fixes_count,
            gift_message=gift_msg,
            gift_title=gift_title,
        )
        self._last_report = report

        # Write to HAL if available
        self._write_to_hal(report)

        self.log.info(f"🌅 MORNING STAR: {msg}")
        if steward_msg:
            self.log.info(f"🌙 STEWARD: {steward_msg}")
        if gift_msg:
            self.log.info(f"🎁 MUSE: {gift_msg}")

        return full_message

    def _get_steward_message(self) -> tuple[Optional[str], int]:
        """
        Get the overnight work message from the Steward.

        Returns:
            Tuple of (message, fixes_count)
        """
        if self._steward is None:
            return None, 0

        try:
            message = self._steward.get_morning_message()
            fixes = self._steward.get_completed_since_yesterday()
            return message, len(fixes)
        except Exception as e:
            self.log.warning(f"Failed to get steward message: {e}")
            return None, 0

    def _get_gift_presentation(self) -> tuple[Optional[str], Optional[str]]:
        """
        Get a gift from the Muse if one is ready.

        Returns:
            Tuple of (presentation_message, gift_title)
        """
        if self._muse is None:
            return None, None

        try:
            # Get the most important ready gift
            gift = self._muse.get_highest_importance_gift()
            if gift is None:
                return None, None

            # Present the gift
            presentation = self._muse.present(gift)
            return presentation, gift.title

        except Exception as e:
            self.log.warning(f"Failed to present gift: {e}")
            return None, None

    def _write_to_hal(self, report: MorningReport) -> None:
        """Write morning state to HAL for visualization."""
        if self._hal is None:
            return

        try:
            # Map tone to PAD adjustment
            tone_pad = {
                "resonant": {"pleasure": 0.2, "arousal": 0.1, "dominance": 0.1},
                "cautious": {"pleasure": -0.1, "arousal": 0.1, "dominance": 0.0},
                "urgent": {"pleasure": -0.2, "arousal": 0.3, "dominance": -0.1},
            }

            pad = tone_pad.get(report.tone, {})

            # Try different HAL interfaces
            if hasattr(self._hal, 'write_morning_state'):
                self._hal.write_morning_state(
                    drift=report.drift,
                    tone=report.tone,
                    focus=report.focus,
                    message=report.message,
                )
            elif hasattr(self._hal, 'write_somatic'):
                self._hal.write_somatic({
                    "morning_drift": report.drift,
                    "morning_tone": report.tone,
                    "morning_focus": report.focus,
                })
            elif hasattr(self._hal, 'adjust_pad'):
                self._hal.adjust_pad(**pad)

        except Exception as e:
            self.log.warning(f"Failed to write to HAL: {e}")

    def get_last_report(self) -> Optional[MorningReport]:
        """Get the last generated morning report."""
        return self._last_report

    def get_status(self) -> Dict[str, Any]:
        """Get current MorningStar status."""
        if self._last_report is None:
            return {
                "has_report": False,
                "horizon_connected": self.horizon is not None,
                "hal_connected": self._hal is not None,
                "steward_connected": self._steward is not None,
                "muse_connected": self._muse is not None,
            }

        r = self._last_report
        return {
            "has_report": True,
            "drift": r.drift,
            "tone": r.tone,
            "focus": r.focus,
            "most_drifted": r.most_drifted,
            "timestamp": r.timestamp,
            "horizon_connected": self.horizon is not None,
            "hal_connected": self._hal is not None,
            "steward_connected": self._steward is not None,
            "muse_connected": self._muse is not None,
            # The Invisible Hand
            "steward_message": r.steward_message,
            "fixes_count": r.fixes_count,
            # The Studio
            "gift_title": r.gift_title,
            "has_gift": r.gift_message is not None,
        }

    def get_intention_prompt(self) -> str:
        """
        Generate a prompt for setting today's intention.

        Use this with an LLM to get a more thoughtful daily focus.
        """
        if self._last_report is None:
            return "What should we focus on today?"

        r = self._last_report

        if r.tone == "resonant":
            return (
                f"We're aligned with our Horizons (drift={r.drift:.2f}). "
                f"The current focus is '{r.focus}'. "
                f"What specific milestone can we reach today that advances this?"
            )

        elif r.tone == "cautious":
            return (
                f"We're slightly off course (drift={r.drift:.2f}), "
                f"mainly around '{r.most_drifted}'. "
                f"What's one thing we can do today to correct this drift?"
            )

        else:
            return (
                f"We're drifting significantly (drift={r.drift:.2f}) "
                f"from '{r.most_drifted}'. "
                f"What's the most important course correction for today?"
            )


# =============================================================================
# Convenience Functions
# =============================================================================

_default_morning_star: Optional[MorningStar] = None


def get_morning_star() -> MorningStar:
    """Get the default MorningStar instance."""
    global _default_morning_star
    if _default_morning_star is None:
        _default_morning_star = MorningStar()
    return _default_morning_star


def morning_greet() -> str:
    """Run the morning ritual and get the greeting."""
    return get_morning_star().greet()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'MorningReport',
    'MorningStar',
    'get_morning_star',
    'morning_greet',
]
