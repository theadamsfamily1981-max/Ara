"""
Ara Breath-Vision Protocol
===========================

Guided breathing sessions with gentle visual and haptic entrainment.

This is WELLNESS, not medical treatment.
- Time-limited sessions
- Easy exit at any moment
- No claims about curing anything

Philosophy: A gentle breathing exercise with nice visuals.
If it helps you relax, great. If not, stop anytime.
"""

from __future__ import annotations

import asyncio
import math
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import json

from .rails import BreathVisionRails, EmbodimentRails

logger = logging.getLogger(__name__)


# =============================================================================
# Session Types
# =============================================================================

class BreathPhase(Enum):
    """Phase of the breath cycle."""
    INHALE = "inhale"
    HOLD_IN = "hold_in"
    EXHALE = "exhale"
    HOLD_OUT = "hold_out"


class SessionState(Enum):
    """State of a breath-vision session."""
    NOT_STARTED = "not_started"
    CALIBRATING = "calibrating"
    RUNNING = "running"
    PAUSED = "paused"
    ENDING = "ending"
    COMPLETE = "complete"
    STOPPED_EARLY = "stopped_early"


@dataclass
class BreathPattern:
    """A breathing pattern."""
    name: str
    breaths_per_minute: float
    inhale_ratio: float = 1.0
    exhale_ratio: float = 1.0
    hold_in_ratio: float = 0.0
    hold_out_ratio: float = 0.0

    @property
    def cycle_duration_s(self) -> float:
        """Total duration of one breath cycle in seconds."""
        return 60.0 / self.breaths_per_minute

    def phase_duration(self, phase: BreathPhase) -> float:
        """Get duration of a specific phase in seconds."""
        total_ratio = (
            self.inhale_ratio +
            self.exhale_ratio +
            self.hold_in_ratio +
            self.hold_out_ratio
        )
        cycle = self.cycle_duration_s

        if phase == BreathPhase.INHALE:
            return cycle * (self.inhale_ratio / total_ratio)
        elif phase == BreathPhase.EXHALE:
            return cycle * (self.exhale_ratio / total_ratio)
        elif phase == BreathPhase.HOLD_IN:
            return cycle * (self.hold_in_ratio / total_ratio)
        else:
            return cycle * (self.hold_out_ratio / total_ratio)


# Built-in patterns
PATTERNS = {
    "balanced": BreathPattern("balanced", 6, 1.0, 1.0),
    "relaxing": BreathPattern("relaxing", 6, 1.0, 2.0),  # 4-8 breathing
    "energizing": BreathPattern("energizing", 8, 1.0, 1.0, 0.5),
    "box": BreathPattern("box", 4, 1.0, 1.0, 1.0, 1.0),  # Box breathing
    "coherent": BreathPattern("coherent", 5, 1.0, 1.0),  # 5 breaths/min
}


@dataclass
class VisualState:
    """Current visual output state."""
    brightness: float = 0.3
    color: str = "#4080ff"
    fov_expand: float = 0.0  # -1 to 1 (contract to expand)


@dataclass
class HapticState:
    """Current haptic output state."""
    intensity: float = 0.0
    active: bool = False


@dataclass
class SessionSummary:
    """Summary of a completed session."""
    session_id: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    pattern_name: str
    target_breath_rate: float
    breaths_completed: int
    estimated_sync_score: float  # 0-1
    completed_normally: bool
    early_exit_reason: Optional[str] = None


# =============================================================================
# Visual Styles
# =============================================================================

@dataclass
class VisualStyle:
    """A visual style for the session."""
    name: str
    primary_color: str
    secondary_color: str
    motion: str  # "wave", "gentle_sway", "slow_pulse", "none"

    def get_color_for_phase(self, phase: BreathPhase, progress: float) -> str:
        """Get color for current phase and progress (0-1)."""
        # Simple interpolation between primary and secondary
        if phase in (BreathPhase.INHALE, BreathPhase.HOLD_IN):
            # Shift toward secondary (expansion)
            return self._lerp_color(self.primary_color, self.secondary_color, progress)
        else:
            # Shift back to primary (contraction)
            return self._lerp_color(self.secondary_color, self.primary_color, progress)

    def _lerp_color(self, color1: str, color2: str, t: float) -> str:
        """Linear interpolate between two hex colors."""
        r1, g1, b1 = self._hex_to_rgb(color1)
        r2, g2, b2 = self._hex_to_rgb(color2)

        r = int(r1 + (r2 - r1) * t)
        g = int(g1 + (g2 - g1) * t)
        b = int(b1 + (b2 - b1) * t)

        return f"#{r:02x}{g:02x}{b:02x}"

    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


STYLES = {
    "ocean": VisualStyle("ocean", "#1a5f7a", "#57c5b6", "wave"),
    "forest": VisualStyle("forest", "#2d5a27", "#90b77d", "gentle_sway"),
    "night": VisualStyle("night", "#1a1a2e", "#4a4e69", "slow_pulse"),
    "minimal": VisualStyle("minimal", "#2c2c2c", "#4a4a4a", "none"),
    "warm": VisualStyle("warm", "#a64b2a", "#d4a574", "slow_pulse"),
}


# =============================================================================
# Session Logger
# =============================================================================

class SessionLogger:
    """Logs breath-vision sessions."""

    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = log_dir or Path.home() / ".ara" / "breath_vision"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "sessions.jsonl"

    def log(self, summary: SessionSummary):
        """Log a session summary."""
        record = {
            "session_id": summary.session_id,
            "start_time": summary.start_time.isoformat(),
            "end_time": summary.end_time.isoformat(),
            "duration_seconds": summary.duration_seconds,
            "pattern_name": summary.pattern_name,
            "target_breath_rate": summary.target_breath_rate,
            "breaths_completed": summary.breaths_completed,
            "estimated_sync_score": summary.estimated_sync_score,
            "completed_normally": summary.completed_normally,
            "early_exit_reason": summary.early_exit_reason,
        }

        with open(self.log_file, 'a') as f:
            f.write(json.dumps(record) + "\n")


# =============================================================================
# Actuator Interface (Abstract)
# =============================================================================

class BreathVisionActuator:
    """
    Interface for breath-vision output.

    Implement for real hardware.
    """

    async def set_visual(
        self,
        brightness: float,
        color: str,
        fov_expand: float,
        transition_ms: int,
    ):
        """Set visual state."""
        raise NotImplementedError

    async def set_haptic(self, intensity: float, pattern: str):
        """Set haptic state."""
        raise NotImplementedError

    async def haptic_off(self):
        """Turn off haptics."""
        raise NotImplementedError

    async def speak(self, text: str):
        """Speak guidance."""
        raise NotImplementedError

    async def fade_out(self, duration_ms: int):
        """Fade all outputs to neutral."""
        raise NotImplementedError


class MockBreathVisionActuator(BreathVisionActuator):
    """Mock actuator for testing."""

    def __init__(self):
        self.visual_log: List[Dict] = []
        self.haptic_log: List[Dict] = []
        self.speech_log: List[str] = []

    async def set_visual(
        self,
        brightness: float,
        color: str,
        fov_expand: float,
        transition_ms: int,
    ):
        self.visual_log.append({
            "brightness": brightness,
            "color": color,
            "fov_expand": fov_expand,
            "transition_ms": transition_ms,
        })
        logger.debug(f"Visual: {color} @ {brightness}, FOV={fov_expand}")

    async def set_haptic(self, intensity: float, pattern: str):
        self.haptic_log.append({
            "intensity": intensity,
            "pattern": pattern,
        })
        logger.debug(f"Haptic: {pattern} @ {intensity}")

    async def haptic_off(self):
        self.haptic_log.append({"action": "off"})

    async def speak(self, text: str):
        self.speech_log.append(text)
        logger.info(f"Speaking: {text}")

    async def fade_out(self, duration_ms: int):
        await self.set_visual(0.1, "#404040", 0, duration_ms)
        await self.haptic_off()


# =============================================================================
# Breath-Vision Session
# =============================================================================

class BreathVisionSession:
    """
    A single breath-vision session.

    Usage:
        session = BreathVisionSession(duration_minutes=3)
        summary = await session.run()
    """

    def __init__(
        self,
        duration_minutes: float = 3,
        pattern: str = "balanced",
        style: str = "ocean",
        actuator: Optional[BreathVisionActuator] = None,
    ):
        # Safety rails
        self.rails = BreathVisionRails()
        self.embodiment_rails = EmbodimentRails()

        # Configuration
        self.requested_duration = duration_minutes
        self.pattern = PATTERNS.get(pattern, PATTERNS["balanced"])
        self.style = STYLES.get(style, STYLES["ocean"])

        # Hardware
        self.actuator = actuator or MockBreathVisionActuator()

        # State
        self.state = SessionState.NOT_STARTED
        self.session_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.breaths_completed = 0
        self.current_phase = BreathPhase.INHALE
        self.phase_progress = 0.0

        # Logging
        self.logger = SessionLogger()

        # Stop detection
        self._stop_requested = False
        self._stop_reason: Optional[str] = None

    async def run(self) -> SessionSummary:
        """Run the full session."""
        # Check if we can start
        can_start, reason = self.rails.can_start_session()
        if not can_start:
            logger.warning(f"Cannot start session: {reason}")
            return self._create_summary(False, reason)

        # Start the session with clamped duration
        success, actual_duration = self.rails.start_session(self.requested_duration)
        if not success:
            return self._create_summary(False, "Failed to start")

        self.start_time = datetime.utcnow()
        total_seconds = actual_duration * 60

        try:
            # Speak disclaimer
            disclaimer = self.rails.get_disclaimer()
            await self.actuator.speak(disclaimer)
            await asyncio.sleep(3)  # Let them absorb

            # Run the main loop
            self.state = SessionState.RUNNING
            elapsed = 0

            while elapsed < total_seconds and not self._stop_requested:
                # Check for E-stop
                if self.embodiment_rails.is_e_stop_active():
                    self._stop_requested = True
                    self._stop_reason = "emergency_stop"
                    break

                # Update breath phase
                cycle_position = elapsed % self.pattern.cycle_duration_s
                self._update_phase(cycle_position)

                # Update outputs
                await self._update_outputs()

                # Small guidance at phase transitions
                await self._maybe_speak_guidance()

                # Wait
                await asyncio.sleep(0.1)  # 10Hz update
                elapsed = (datetime.utcnow() - self.start_time).total_seconds()

            # Ending sequence
            self.state = SessionState.ENDING
            await self._end_sequence()

        except asyncio.CancelledError:
            self._stop_requested = True
            self._stop_reason = "cancelled"
        except Exception as e:
            logger.error(f"Session error: {e}")
            self._stop_requested = True
            self._stop_reason = f"error: {e}"
        finally:
            self.rails.end_session()

        # Create and log summary
        completed = not self._stop_requested
        summary = self._create_summary(completed, self._stop_reason)
        self.logger.log(summary)

        return summary

    def request_stop(self, reason: str = "user_request"):
        """Request the session to stop."""
        self._stop_requested = True
        self._stop_reason = reason

    def check_utterance(self, text: str) -> bool:
        """Check if utterance is a stop phrase. Returns True if should stop."""
        if self.rails.check_for_stop(text):
            self.request_stop("stop_phrase")
            return True

        # Check for contraindication mentions
        warning = self.rails.check_contraindication(text)
        if warning:
            # Would need to speak warning and ask for confirmation
            # For now, just note it
            logger.warning(f"Contraindication mentioned: {text}")

        return False

    def _update_phase(self, cycle_position: float):
        """Update current breath phase based on cycle position."""
        # Calculate phase boundaries
        inhale_end = self.pattern.phase_duration(BreathPhase.INHALE)
        hold_in_end = inhale_end + self.pattern.phase_duration(BreathPhase.HOLD_IN)
        exhale_end = hold_in_end + self.pattern.phase_duration(BreathPhase.EXHALE)
        # hold_out goes to end of cycle

        old_phase = self.current_phase

        if cycle_position < inhale_end:
            self.current_phase = BreathPhase.INHALE
            self.phase_progress = cycle_position / inhale_end if inhale_end > 0 else 0
        elif cycle_position < hold_in_end:
            self.current_phase = BreathPhase.HOLD_IN
            phase_dur = hold_in_end - inhale_end
            self.phase_progress = (cycle_position - inhale_end) / phase_dur if phase_dur > 0 else 0
        elif cycle_position < exhale_end:
            self.current_phase = BreathPhase.EXHALE
            phase_dur = exhale_end - hold_in_end
            self.phase_progress = (cycle_position - hold_in_end) / phase_dur if phase_dur > 0 else 0
        else:
            self.current_phase = BreathPhase.HOLD_OUT
            phase_dur = self.pattern.cycle_duration_s - exhale_end
            self.phase_progress = (cycle_position - exhale_end) / phase_dur if phase_dur > 0 else 0

        # Count completed breaths
        if old_phase == BreathPhase.HOLD_OUT and self.current_phase == BreathPhase.INHALE:
            self.breaths_completed += 1

    async def _update_outputs(self):
        """Update visual and haptic outputs."""
        # Get visual limits
        limits = self.rails.get_visual_limits()
        max_brightness = limits.get("max_brightness", 0.6)
        max_fov_change = limits.get("fov", {}).get("max_change_percent", 30) / 100

        # Calculate visual state
        base_brightness = 0.3
        brightness_range = max_brightness - base_brightness

        if self.current_phase == BreathPhase.INHALE:
            brightness = base_brightness + brightness_range * self.phase_progress
            fov = max_fov_change * self.phase_progress
        elif self.current_phase == BreathPhase.HOLD_IN:
            brightness = max_brightness
            fov = max_fov_change
        elif self.current_phase == BreathPhase.EXHALE:
            brightness = max_brightness - brightness_range * self.phase_progress
            fov = max_fov_change * (1 - self.phase_progress)
        else:  # HOLD_OUT
            brightness = base_brightness
            fov = 0

        color = self.style.get_color_for_phase(self.current_phase, self.phase_progress)

        await self.actuator.set_visual(
            brightness=brightness,
            color=color,
            fov_expand=fov,
            transition_ms=100,
        )

        # Haptics - gentle pulse on inhale
        haptic_limits = self.rails.get_haptic_limits()
        max_haptic = haptic_limits.get("max_intensity", 0.5)

        if self.current_phase == BreathPhase.INHALE:
            intensity = max_haptic * 0.3 * self.phase_progress
            await self.actuator.set_haptic(intensity, "soft_rise")
        elif self.current_phase == BreathPhase.EXHALE:
            intensity = max_haptic * 0.3 * (1 - self.phase_progress)
            await self.actuator.set_haptic(intensity, "soft_fall")
        else:
            await self.actuator.haptic_off()

    async def _maybe_speak_guidance(self):
        """Maybe speak guidance at phase transitions."""
        # Only speak at very start of phases
        if self.phase_progress > 0.1:
            return

        guidance = {
            BreathPhase.INHALE: "Breathe in...",
            BreathPhase.EXHALE: "Breathe out...",
            BreathPhase.HOLD_IN: "",  # Silent holds
            BreathPhase.HOLD_OUT: "",
        }

        text = guidance.get(self.current_phase, "")
        if text and self.breaths_completed % 3 == 0:  # Don't speak every breath
            await self.actuator.speak(text)

    async def _end_sequence(self):
        """Gentle ending sequence."""
        await self.actuator.speak("Gently returning to normal breathing.")
        await self.actuator.fade_out(2000)
        await asyncio.sleep(2)
        await self.actuator.speak("Session complete. Take your time.")

        self.end_time = datetime.utcnow()
        self.state = SessionState.COMPLETE

    def _create_summary(
        self,
        completed: bool,
        early_reason: Optional[str] = None,
    ) -> SessionSummary:
        """Create a session summary."""
        end_time = self.end_time or datetime.utcnow()
        start_time = self.start_time or end_time

        duration = (end_time - start_time).total_seconds()

        # Estimate sync score based on breaths completed vs expected
        expected_breaths = (duration / 60) * self.pattern.breaths_per_minute
        sync_score = min(1.0, self.breaths_completed / max(1, expected_breaths))

        return SessionSummary(
            session_id=self.session_id,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            pattern_name=self.pattern.name,
            target_breath_rate=self.pattern.breaths_per_minute,
            breaths_completed=self.breaths_completed,
            estimated_sync_score=sync_score,
            completed_normally=completed,
            early_exit_reason=early_reason,
        )


# =============================================================================
# CLI Entry Point
# =============================================================================

async def run_breath_vision_session(
    duration: float = 3,
    pattern: str = "balanced",
    style: str = "ocean",
):
    """Run a breath-vision session from CLI."""
    print("=" * 50)
    print("ARA BREATH-VISION SESSION")
    print("=" * 50)
    print()
    print("This is a breathing exercise for relaxation.")
    print("It is NOT medical treatment.")
    print("Stop any time if you feel uncomfortable.")
    print()
    print(f"Duration: {duration} minutes")
    print(f"Pattern: {pattern}")
    print(f"Style: {style}")
    print()
    print("Say 'stop' or press Ctrl+C to end early.")
    print()

    session = BreathVisionSession(
        duration_minutes=duration,
        pattern=pattern,
        style=style,
    )

    try:
        summary = await session.run()

        print()
        print("=" * 50)
        print("SESSION COMPLETE")
        print("=" * 50)
        print(f"Duration: {summary.duration_seconds:.0f} seconds")
        print(f"Breaths: {summary.breaths_completed}")
        print(f"Sync score: {summary.estimated_sync_score:.0%}")

        if not summary.completed_normally:
            print(f"Early exit: {summary.early_exit_reason}")

    except KeyboardInterrupt:
        print("\n\nSession interrupted. Take your time returning to normal breathing.")
        session.request_stop("keyboard_interrupt")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Ara Breath-Vision Session",
        epilog="""
This is a breathing exercise for relaxation, NOT medical treatment.
Stop any time if you feel uncomfortable.

Patterns:
  balanced   - Equal inhale/exhale (6 breaths/min)
  relaxing   - Longer exhale (4-8 breathing)
  energizing - With brief hold
  box        - Box breathing (4x4)
  coherent   - 5 breaths/min

Styles:
  ocean, forest, night, minimal, warm
        """
    )

    parser.add_argument(
        '--duration', '-d',
        type=float,
        default=3,
        help="Duration in minutes (default: 3, max: 10)"
    )
    parser.add_argument(
        '--pattern', '-p',
        choices=list(PATTERNS.keys()),
        default="balanced",
        help="Breathing pattern (default: balanced)"
    )
    parser.add_argument(
        '--style', '-s',
        choices=list(STYLES.keys()),
        default="ocean",
        help="Visual style (default: ocean)"
    )

    args = parser.parse_args()

    asyncio.run(run_breath_vision_session(
        duration=args.duration,
        pattern=args.pattern,
        style=args.style,
    ))


if __name__ == "__main__":
    main()
