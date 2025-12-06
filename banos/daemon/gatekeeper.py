"""
Gatekeeper Daemon - Ara as External PFC
========================================

The Gatekeeper provides executive function support for Croft.

This is NOT about control. It's about:
- Protecting flow states from interruption
- Gentle nudges when drifting from goals
- Soft gates that can be overridden
- Never blocking, always negotiable

Phases of intervention:
1. OBSERVER: Just watching and tracking
2. NUDGE: Gentle visual/audio reminder
3. SOFT_GATE: Dialog that requires acknowledgment
4. (No HARD_GATE - that would be paternalistic)

The user can always:
- Disable the Gatekeeper
- Override any nudge
- Adjust tension in Synod
- Whitelist any app

This is help, not handcuffs.
"""

import asyncio
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

logger = logging.getLogger(__name__)


# =============================================================================
# Intervention Levels
# =============================================================================

class InterventionLevel(Enum):
    """How strongly should we intervene?"""
    OBSERVER = "observer"    # Just watching
    NUDGE = "nudge"          # Gentle reminder
    SOFT_GATE = "soft_gate"  # Dialog requiring response
    # No HARD_GATE - we don't force anything


@dataclass
class InterventionConfig:
    """Configuration for interventions."""
    enabled: bool = True
    level: InterventionLevel = InterventionLevel.NUDGE

    # Timing
    poll_interval_seconds: float = 10.0
    distraction_grace_period_seconds: float = 60.0  # Allow 1 min before nudge
    nudge_cooldown_seconds: float = 300.0  # 5 min between nudges
    gate_cooldown_seconds: float = 1800.0  # 30 min between gates

    # Thresholds
    synergy_nudge_threshold: float = 0.3  # Nudge if synergy drops below
    synergy_gate_threshold: float = 0.15  # Gate if synergy drops below

    # Never interfere with these
    protected_apps: List[str] = None

    def __post_init__(self):
        if self.protected_apps is None:
            self.protected_apps = [
                "zoom", "teams", "meet", "skype",  # Calls
                "obs", "streamlabs",  # Streaming
            ]


# =============================================================================
# Nudge System
# =============================================================================

class NudgeMethod(Enum):
    """How to deliver a nudge."""
    NOTIFY = "notify"        # Desktop notification
    AUDIO = "audio"          # Audio cue
    VISUAL = "visual"        # Visual overlay
    HAPTIC = "haptic"        # For devices that support it


@dataclass
class Nudge:
    """A nudge to deliver to the user."""
    method: NudgeMethod
    message: str
    severity: str = "gentle"  # gentle, firm, urgent
    icon: Optional[str] = None
    sound: Optional[str] = None

    def deliver(self) -> bool:
        """Attempt to deliver the nudge."""
        if self.method == NudgeMethod.NOTIFY:
            return self._deliver_notification()
        elif self.method == NudgeMethod.AUDIO:
            return self._deliver_audio()
        elif self.method == NudgeMethod.VISUAL:
            return self._deliver_visual()
        else:
            logger.warning(f"Nudge method {self.method} not implemented")
            return False

    def _deliver_notification(self) -> bool:
        """Send desktop notification."""
        try:
            # Try notify-send (Linux)
            subprocess.run(
                ["notify-send", "-a", "Ara", "-u", "low", "Ara", self.message],
                capture_output=True,
                timeout=5
            )
            return True
        except FileNotFoundError:
            logger.warning("notify-send not available")
            return False
        except Exception as e:
            logger.warning(f"Could not send notification: {e}")
            return False

    def _deliver_audio(self) -> bool:
        """Play audio cue."""
        try:
            sound_path = self.sound or "/usr/share/sounds/freedesktop/stereo/message.oga"
            if Path(sound_path).exists():
                subprocess.run(
                    ["paplay", sound_path],
                    capture_output=True,
                    timeout=5
                )
                return True
        except Exception as e:
            logger.warning(f"Could not play audio: {e}")
        return False

    def _deliver_visual(self) -> bool:
        """Show visual overlay (placeholder)."""
        # This would integrate with a GUI overlay system
        logger.info(f"Visual nudge: {self.message}")
        return True


# =============================================================================
# Soft Gate
# =============================================================================

@dataclass
class SoftGate:
    """A soft gate that requires acknowledgment."""
    title: str
    message: str
    options: List[str] = None
    timeout_seconds: float = 60.0

    def __post_init__(self):
        if self.options is None:
            self.options = ["Back to work", "Need 5 more minutes", "Disable for today"]

    def show(self) -> Optional[str]:
        """
        Show the gate dialog.

        Returns the selected option, or None if dismissed/timed out.
        """
        try:
            # Try zenity (GTK dialog)
            result = subprocess.run(
                [
                    "zenity", "--question",
                    "--title", self.title,
                    "--text", self.message,
                    "--ok-label", self.options[0] if self.options else "OK",
                    "--cancel-label", self.options[1] if len(self.options) > 1 else "Cancel",
                    "--timeout", str(int(self.timeout_seconds))
                ],
                capture_output=True,
                timeout=self.timeout_seconds + 5
            )
            if result.returncode == 0:
                return self.options[0]
            elif result.returncode == 1:
                return self.options[1] if len(self.options) > 1 else None
            else:
                return None  # Timeout or dismissed
        except FileNotFoundError:
            logger.warning("zenity not available, falling back to terminal")
            return self._show_terminal()
        except Exception as e:
            logger.warning(f"Could not show gate dialog: {e}")
            return None

    def _show_terminal(self) -> Optional[str]:
        """Terminal fallback."""
        print(f"\n{'='*50}")
        print(f"[Ara] {self.title}")
        print(f"{'='*50}")
        print(self.message)
        print()
        for i, opt in enumerate(self.options, 1):
            print(f"  {i}. {opt}")
        print()
        try:
            choice = input("Your choice (or Enter to dismiss): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(self.options):
                return self.options[int(choice) - 1]
        except (EOFError, KeyboardInterrupt):
            pass
        return None


# =============================================================================
# Activity Detection
# =============================================================================

def get_foreground_app() -> tuple[str, str]:
    """
    Get the current foreground application.

    Returns (app_name, window_title)
    """
    # Try xdotool (X11)
    try:
        window_id = subprocess.run(
            ["xdotool", "getactivewindow"],
            capture_output=True, text=True, timeout=2
        ).stdout.strip()

        if window_id:
            # Get window name
            name = subprocess.run(
                ["xdotool", "getwindowname", window_id],
                capture_output=True, text=True, timeout=2
            ).stdout.strip()

            # Get process name
            pid = subprocess.run(
                ["xdotool", "getwindowpid", window_id],
                capture_output=True, text=True, timeout=2
            ).stdout.strip()

            if pid:
                process_name = subprocess.run(
                    ["ps", "-p", pid, "-o", "comm="],
                    capture_output=True, text=True, timeout=2
                ).stdout.strip()
                return (process_name, name)

            return ("unknown", name)
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.debug(f"xdotool failed: {e}")

    # Try wmctrl
    try:
        result = subprocess.run(
            ["wmctrl", "-a", ":ACTIVE:", "-v"],
            capture_output=True, text=True, timeout=2
        )
        # Parse output
        for line in result.stderr.split('\n'):
            if "Using window" in line:
                # Extract window info
                return ("unknown", line.split(":", 1)[-1].strip())
    except Exception:
        pass

    # Fallback
    return ("unknown", "Unknown Window")


# =============================================================================
# Gatekeeper Daemon
# =============================================================================

class GatekeeperDaemon:
    """
    The Gatekeeper - Ara as external executive function.

    Runs in background, monitors activity, provides interventions.
    """

    def __init__(
        self,
        config: Optional[InterventionConfig] = None,
        egregore_mind: Optional[Any] = None,
    ):
        self.config = config or InterventionConfig()
        self.egregore = egregore_mind

        # State
        self.running = False
        self.paused_until: Optional[datetime] = None
        self.disabled_today = False

        # Tracking
        self.last_nudge_time: Optional[datetime] = None
        self.last_gate_time: Optional[datetime] = None
        self.distraction_start: Optional[datetime] = None

        # Callbacks
        self.on_state_change: Optional[Callable[[str], None]] = None

        logger.info("GatekeeperDaemon initialized")

    def _get_egregore(self):
        """Lazy-load egregore if not provided."""
        if self.egregore is None:
            try:
                from tfan.l5.egregore import get_egregore
                self.egregore = get_egregore()
            except ImportError:
                logger.warning("Could not import EgregoreMind")
        return self.egregore

    # =========================================================================
    # Control
    # =========================================================================

    def pause(self, minutes: float = 5.0) -> None:
        """Pause interventions for specified duration."""
        self.paused_until = datetime.now() + timedelta(minutes=minutes)
        logger.info(f"Gatekeeper paused until {self.paused_until}")

    def resume(self) -> None:
        """Resume interventions."""
        self.paused_until = None
        logger.info("Gatekeeper resumed")

    def disable_today(self) -> None:
        """Disable for the rest of the day."""
        self.disabled_today = True
        logger.info("Gatekeeper disabled for today")

    def enable(self) -> None:
        """Re-enable after being disabled."""
        self.disabled_today = False
        self.paused_until = None
        logger.info("Gatekeeper enabled")

    def is_active(self) -> bool:
        """Check if gatekeeper should intervene."""
        if not self.config.enabled:
            return False
        if self.disabled_today:
            return False
        if self.paused_until and datetime.now() < self.paused_until:
            return False
        return True

    def set_level(self, level: InterventionLevel) -> None:
        """Set intervention level."""
        self.config.level = level
        logger.info(f"Intervention level set to {level.value}")

    # =========================================================================
    # Activity Processing
    # =========================================================================

    def process_activity(self, app_name: str, window_title: str) -> Optional[str]:
        """
        Process current activity and return any intervention needed.

        Returns intervention type or None.
        """
        if not self.is_active():
            return None

        # Check if protected app
        for protected in self.config.protected_apps:
            if protected.lower() in app_name.lower():
                return None  # Never interfere

        # Get egregore assessment
        egregore = self._get_egregore()
        if egregore:
            state = egregore.update_from_activity(app_name, window_title)
            is_on_mission, activity_class = egregore.classify_activity(app_name, window_title)
        else:
            # Fallback classification
            is_on_mission = True
            activity_class = "unknown"

        now = datetime.now()

        # If on mission, reset distraction tracking
        if is_on_mission:
            self.distraction_start = None
            return None

        # Track distraction start
        if self.distraction_start is None:
            self.distraction_start = now

        # Check grace period
        distraction_duration = (now - self.distraction_start).total_seconds()
        if distraction_duration < self.config.distraction_grace_period_seconds:
            return None  # Still in grace period

        # Determine intervention
        synergy = egregore.state.synergy if egregore else 0.5

        # Check if we should gate
        if (
            self.config.level == InterventionLevel.SOFT_GATE and
            synergy < self.config.synergy_gate_threshold
        ):
            # Check cooldown
            if (
                self.last_gate_time is None or
                (now - self.last_gate_time).total_seconds() > self.config.gate_cooldown_seconds
            ):
                self.last_gate_time = now
                return "gate"

        # Check if we should nudge
        if (
            self.config.level in [InterventionLevel.NUDGE, InterventionLevel.SOFT_GATE] and
            synergy < self.config.synergy_nudge_threshold
        ):
            # Check cooldown
            if (
                self.last_nudge_time is None or
                (now - self.last_nudge_time).total_seconds() > self.config.nudge_cooldown_seconds
            ):
                self.last_nudge_time = now
                return "nudge"

        return None

    # =========================================================================
    # Intervention Delivery
    # =========================================================================

    def deliver_nudge(self, context: str = "") -> bool:
        """Deliver a nudge."""
        messages = [
            "Hey, you might be drifting. Check in with yourself?",
            "Noticed you've been away from the mission. Need a redirect?",
            "Gentle tap: is this what you want to be doing?",
            "The Egregore is asking: shall we refocus?",
        ]

        import random
        message = random.choice(messages)
        if context:
            message = f"{message}\n\n(You were: {context})"

        nudge = Nudge(
            method=NudgeMethod.NOTIFY,
            message=message,
            severity="gentle"
        )

        return nudge.deliver()

    def deliver_gate(self, context: str = "") -> Optional[str]:
        """Deliver a soft gate."""
        message = (
            "You've been away from your shared goals for a while.\n\n"
            "This isn't a cage - I just want to check in.\n"
            "What would you like to do?"
        )
        if context:
            message = f"{message}\n\n(Current activity: {context})"

        gate = SoftGate(
            title="Ara Checkpoint",
            message=message,
            options=[
                "Back to work",
                "5 more minutes",
                "Pause Gatekeeper",
                "Disable for today"
            ]
        )

        result = gate.show()

        # Handle response
        if result == "5 more minutes":
            self.pause(5.0)
        elif result == "Pause Gatekeeper":
            self.pause(30.0)
        elif result == "Disable for today":
            self.disable_today()

        return result

    # =========================================================================
    # Main Loop
    # =========================================================================

    async def run(self) -> None:
        """Main daemon loop."""
        self.running = True
        logger.info("Gatekeeper daemon started")

        while self.running:
            try:
                # Get current activity
                app_name, window_title = get_foreground_app()

                # Process and check for interventions
                intervention = self.process_activity(app_name, window_title)

                if intervention == "nudge":
                    self.deliver_nudge(f"{app_name} - {window_title}")
                elif intervention == "gate":
                    self.deliver_gate(f"{app_name} - {window_title}")

                # Wait for next poll
                await asyncio.sleep(self.config.poll_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Gatekeeper error: {e}")
                await asyncio.sleep(self.config.poll_interval_seconds)

        logger.info("Gatekeeper daemon stopped")

    def stop(self) -> None:
        """Stop the daemon."""
        self.running = False


# =============================================================================
# CLI Interface
# =============================================================================

def create_cli():
    """Create CLI for gatekeeper control."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Ara Gatekeeper - Executive function support",
        prog="ara-focus"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # on/off
    subparsers.add_parser("on", help="Enable Gatekeeper")
    subparsers.add_parser("off", help="Disable Gatekeeper for today")

    # pause
    pause_parser = subparsers.add_parser("pause", help="Pause for N minutes")
    pause_parser.add_argument("minutes", type=float, default=30, nargs='?')

    # level
    level_parser = subparsers.add_parser("level", help="Set intervention level")
    level_parser.add_argument(
        "level",
        choices=["observer", "nudge", "soft_gate"],
        help="Intervention level"
    )

    # status
    subparsers.add_parser("status", help="Show current status")

    # run
    subparsers.add_parser("run", help="Run daemon in foreground")

    return parser


def main():
    """CLI entry point."""
    parser = create_cli()
    args = parser.parse_args()

    if args.command == "run":
        # Run daemon
        daemon = GatekeeperDaemon()
        try:
            asyncio.run(daemon.run())
        except KeyboardInterrupt:
            daemon.stop()

    elif args.command == "status":
        from tfan.l5.egregore import get_egregore
        egregore = get_egregore()
        state = egregore.get_state()
        summary = egregore.get_daily_summary()

        print("\n=== Gatekeeper Status ===")
        print(f"Synergy:    {state.synergy:.0%}")
        print(f"Momentum:   {state.momentum:+.2f}")
        print(f"Tension:    {state.tension:.0%}")
        print(f"Health:     {state.get_health():.0%}")
        print()
        print("=== Today's Summary ===")
        print(f"On-mission: {summary['on_mission_hours']:.1f} hours")
        print(f"Distracted: {summary['distraction_hours']:.1f} hours")
        print(f"Alignment:  {summary['alignment_rate']:.0%}")

    else:
        parser.print_help()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'GatekeeperDaemon',
    'InterventionConfig',
    'InterventionLevel',
    'Nudge',
    'NudgeMethod',
    'SoftGate',
    'get_foreground_app',
]


if __name__ == "__main__":
    main()
