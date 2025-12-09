"""
Ara Embodiment Rails
=====================

Safety enforcement for physical embodiment: motors, haptics, sensors.
These are HARD LIMITS that cannot be bypassed.

The E-stop is sacred. When it triggers, everything stops.
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
from datetime import datetime, timedelta
import json
import logging

logger = logging.getLogger(__name__)


class SafetyLevel(Enum):
    """How serious is this safety situation?"""
    NORMAL = "normal"
    CAUTION = "caution"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class ActionResult(Enum):
    """Result of attempting an action."""
    ALLOWED = "allowed"
    CLAMPED = "clamped"           # Allowed but modified for safety
    BLOCKED = "blocked"            # Not allowed
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class MotorCommand:
    """A command to move a motor."""
    joint: str
    target_position_deg: float
    speed_deg_per_s: float
    torque_percent: float
    duration_s: Optional[float] = None


@dataclass
class ClampedCommand:
    """A motor command after safety clamping."""
    original: MotorCommand
    clamped: MotorCommand
    was_modified: bool
    modifications: List[str] = field(default_factory=list)


@dataclass
class HapticCommand:
    """A haptic feedback command."""
    intensity: float  # 0-1
    duration_ms: int
    pattern: str


@dataclass
class VisualCommand:
    """A visual output command."""
    brightness: float  # 0-1
    color: str  # hex
    transition_ms: int
    pattern: Optional[str] = None


@dataclass
class SafetyViolation:
    """Record of a safety limit being hit."""
    timestamp: datetime
    category: str  # "motor", "haptic", "visual", "session"
    original_value: Any
    clamped_value: Any
    reason: str
    level: SafetyLevel


class EmbodimentCovenant:
    """
    Loads and enforces the embodiment covenant.

    This is the central authority for what Ara's body can do.
    """

    COVENANT_DIR = Path(__file__).parent / "covenant"

    def __init__(self):
        self.embodiment = self._load_yaml("embodiment.yaml")
        self.breath_vision = self._load_yaml("breath_vision.yaml")
        self.fusion = self._load_yaml("fusion_monitor.yaml")
        self._violations: List[SafetyViolation] = []

    def _load_yaml(self, filename: str) -> dict:
        """Load a covenant YAML file."""
        path = self.COVENANT_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Covenant not found: {path}")
        with open(path) as f:
            return yaml.safe_load(f)

    def log_violation(self, violation: SafetyViolation):
        """Log a safety violation."""
        self._violations.append(violation)
        logger.warning(f"Safety violation: {violation.reason}")

        # Also write to file
        log_path = Path(os.path.expanduser(
            self.embodiment.get("logging", {}).get("paths", {})
            .get("safety_log", "~/.ara/embodiment/safety_violations.jsonl")
        ))
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(log_path, 'a') as f:
            f.write(json.dumps({
                "timestamp": violation.timestamp.isoformat(),
                "category": violation.category,
                "original": str(violation.original_value),
                "clamped": str(violation.clamped_value),
                "reason": violation.reason,
                "level": violation.level.value,
            }) + "\n")


class MotorRails:
    """
    Enforces motor safety limits.

    All motor commands go through here before reaching hardware.
    """

    def __init__(self, covenant: EmbodimentCovenant):
        self.covenant = covenant
        self.config = covenant.embodiment.get("motors", {})
        self.global_limits = self.config.get("global", {})
        self.joint_limits = self.config.get("joints", {})
        self.restricted_zones = self.config.get("restricted_zones", [])

    def clamp_command(self, cmd: MotorCommand) -> ClampedCommand:
        """
        Clamp a motor command to safe limits.

        Returns the command after applying all safety constraints.
        """
        mods = []
        clamped = MotorCommand(
            joint=cmd.joint,
            target_position_deg=cmd.target_position_deg,
            speed_deg_per_s=cmd.speed_deg_per_s,
            torque_percent=cmd.torque_percent,
            duration_s=cmd.duration_s,
        )

        # Get joint-specific limits (or use global)
        joint_config = self.joint_limits.get(cmd.joint, {})

        # Clamp speed
        max_speed = joint_config.get(
            "max_speed_deg_per_s",
            self.global_limits.get("max_speed_deg_per_s", 60)
        )
        if clamped.speed_deg_per_s > max_speed:
            clamped.speed_deg_per_s = max_speed
            mods.append(f"speed clamped to {max_speed} deg/s")

        # Clamp torque
        max_torque = joint_config.get(
            "max_torque_percent",
            self.global_limits.get("max_torque_percent", 40)
        )
        if clamped.torque_percent > max_torque:
            clamped.torque_percent = max_torque
            mods.append(f"torque clamped to {max_torque}%")

        # Clamp position to range
        if "range_deg" in joint_config:
            min_deg, max_deg = joint_config["range_deg"]
            if clamped.target_position_deg < min_deg:
                clamped.target_position_deg = min_deg
                mods.append(f"position clamped to min {min_deg} deg")
            elif clamped.target_position_deg > max_deg:
                clamped.target_position_deg = max_deg
                mods.append(f"position clamped to max {max_deg} deg")

        # Check duration limits
        if "max_duration_s" in joint_config and clamped.duration_s:
            max_dur = joint_config["max_duration_s"]
            if clamped.duration_s > max_dur:
                clamped.duration_s = max_dur
                mods.append(f"duration clamped to {max_dur}s")

        # Log if modified
        was_modified = len(mods) > 0
        if was_modified:
            self.covenant.log_violation(SafetyViolation(
                timestamp=datetime.utcnow(),
                category="motor",
                original_value=cmd,
                clamped_value=clamped,
                reason="; ".join(mods),
                level=SafetyLevel.CAUTION,
            ))

        return ClampedCommand(
            original=cmd,
            clamped=clamped,
            was_modified=was_modified,
            modifications=mods,
        )

    def check_restricted_zone(self, zone_name: str) -> bool:
        """Check if contact with a zone is allowed."""
        for zone in self.restricted_zones:
            if zone.get("name") == zone_name:
                return zone.get("allow_contact", False)
        return False  # Unknown zones are forbidden

    def requires_consent(self, joint: str) -> bool:
        """Check if this joint requires explicit consent to move."""
        joint_config = self.joint_limits.get(joint, {})
        return joint_config.get("require_explicit_consent", False)


class HapticRails:
    """Enforces haptic safety limits."""

    def __init__(self, covenant: EmbodimentCovenant):
        self.covenant = covenant
        self.config = covenant.embodiment.get("haptics", {})

    def clamp_command(self, cmd: HapticCommand) -> Tuple[HapticCommand, bool]:
        """Clamp haptic command to safe limits."""
        mods = []
        clamped = HapticCommand(
            intensity=cmd.intensity,
            duration_ms=cmd.duration_ms,
            pattern=cmd.pattern,
        )

        # Clamp intensity
        max_intensity = self.config.get("max_intensity", 0.6)
        if clamped.intensity > max_intensity:
            clamped.intensity = max_intensity
            mods.append(f"intensity clamped to {max_intensity}")

        # Clamp duration
        max_duration = self.config.get("max_duration_ms", 2000)
        if clamped.duration_ms > max_duration:
            clamped.duration_ms = max_duration
            mods.append(f"duration clamped to {max_duration}ms")

        # Check pattern
        allowed_patterns = self.config.get("patterns", {}).get("allowed", [])
        forbidden_patterns = self.config.get("patterns", {}).get("forbidden", [])

        if cmd.pattern in forbidden_patterns:
            clamped.pattern = "gentle_pulse"  # Safe fallback
            mods.append(f"pattern '{cmd.pattern}' forbidden, using gentle_pulse")

        was_modified = len(mods) > 0
        return clamped, was_modified


class VisualRails:
    """Enforces visual output safety limits."""

    def __init__(self, covenant: EmbodimentCovenant):
        self.covenant = covenant
        self.config = covenant.embodiment.get("visuals", {})
        self.led_config = self.config.get("leds", {})
        self.epilepsy_config = self.config.get("epilepsy_safe", {})

    def clamp_command(self, cmd: VisualCommand) -> Tuple[VisualCommand, bool]:
        """Clamp visual command to safe limits."""
        mods = []
        clamped = VisualCommand(
            brightness=cmd.brightness,
            color=cmd.color,
            transition_ms=cmd.transition_ms,
            pattern=cmd.pattern,
        )

        # Clamp brightness
        max_brightness = self.led_config.get("max_brightness", 0.7)
        if clamped.brightness > max_brightness:
            clamped.brightness = max_brightness
            mods.append(f"brightness clamped to {max_brightness}")

        # Enforce minimum transition time (epilepsy safety)
        min_transition = self.led_config.get("max_transition_speed_ms", 200)
        if clamped.transition_ms < min_transition:
            clamped.transition_ms = min_transition
            mods.append(f"transition slowed to {min_transition}ms for safety")

        # No strobe
        if not self.led_config.get("allow_strobe", False):
            if cmd.pattern and "strobe" in cmd.pattern.lower():
                clamped.pattern = "fade"
                mods.append("strobe pattern blocked")

        was_modified = len(mods) > 0
        return clamped, was_modified


class SessionRails:
    """Enforces session-level safety rules."""

    def __init__(self, covenant: EmbodimentCovenant):
        self.covenant = covenant
        self.config = covenant.embodiment.get("session", {})
        self._last_stop_check = datetime.utcnow()

    def get_stop_phrases(self) -> List[str]:
        """Get the list of stop phrases."""
        return self.config.get("stop_phrases", ["stop", "freeze", "cancel"])

    def check_for_stop(self, utterance: str) -> bool:
        """Check if utterance contains a stop phrase."""
        utterance_lower = utterance.lower().strip()
        for phrase in self.get_stop_phrases():
            if phrase.lower() in utterance_lower:
                return True
        return False

    def get_stop_behavior(self) -> Dict[str, str]:
        """Get what to do on stop."""
        return self.config.get("stop_behavior", {
            "motors": "release_all",
            "haptics": "off",
            "leds": "dim_to_idle",
            "audio": "acknowledge_stop",
        })

    def requires_human_present(self) -> bool:
        """Check if human presence is required."""
        return self.config.get("requires_human_present", True)


class EmbodimentRails:
    """
    Main safety enforcement class for all embodiment.

    Use this as the single entry point for all physical actions.
    """

    def __init__(self):
        self.covenant = EmbodimentCovenant()
        self.motors = MotorRails(self.covenant)
        self.haptics = HapticRails(self.covenant)
        self.visuals = VisualRails(self.covenant)
        self.session = SessionRails(self.covenant)

        self._e_stop_active = False
        self._human_present = False

    def emergency_stop(self):
        """Activate emergency stop. Everything stops NOW."""
        self._e_stop_active = True
        logger.critical("EMERGENCY STOP ACTIVATED")

        self.covenant.log_violation(SafetyViolation(
            timestamp=datetime.utcnow(),
            category="session",
            original_value=None,
            clamped_value="emergency_stop",
            reason="E-stop activated",
            level=SafetyLevel.EMERGENCY,
        ))

    def release_e_stop(self):
        """Release emergency stop (requires explicit action)."""
        self._e_stop_active = False
        logger.info("Emergency stop released")

    def is_e_stop_active(self) -> bool:
        """Check if e-stop is active."""
        return self._e_stop_active

    def set_human_present(self, present: bool):
        """Update human presence status."""
        self._human_present = present

    def check_action_allowed(
        self,
        action_type: str,
        action_name: str,
    ) -> Tuple[bool, str]:
        """
        Check if an action is allowed.

        Returns (allowed, reason).
        """
        # E-stop blocks everything
        if self._e_stop_active:
            return False, "Emergency stop is active"

        # Human presence check
        if self.session.requires_human_present() and not self._human_present:
            return False, "Human presence required but not detected"

        # Check autonomous allowed list
        ops = self.covenant.embodiment.get("operations", {})
        autonomous = ops.get("autonomous_allowed", [])
        requires_consent = ops.get("requires_human_consent", [])
        forbidden = ops.get("forbidden", [])

        if action_name in forbidden:
            return False, f"Action '{action_name}' is forbidden"

        if action_name in requires_consent:
            return False, f"Action '{action_name}' requires human consent"

        if action_name in autonomous:
            return True, "Allowed"

        # Default: block unknown actions
        return False, f"Unknown action '{action_name}'"

    def clamp_motor_command(self, cmd: MotorCommand) -> ClampedCommand:
        """Clamp a motor command to safe limits."""
        if self._e_stop_active:
            # E-stop: return zero-motion command
            return ClampedCommand(
                original=cmd,
                clamped=MotorCommand(
                    joint=cmd.joint,
                    target_position_deg=0,
                    speed_deg_per_s=0,
                    torque_percent=0,
                ),
                was_modified=True,
                modifications=["E-STOP ACTIVE - all motion blocked"],
            )
        return self.motors.clamp_command(cmd)

    def clamp_haptic_command(self, cmd: HapticCommand) -> HapticCommand:
        """Clamp a haptic command to safe limits."""
        if self._e_stop_active:
            return HapticCommand(intensity=0, duration_ms=0, pattern="off")
        clamped, _ = self.haptics.clamp_command(cmd)
        return clamped

    def clamp_visual_command(self, cmd: VisualCommand) -> VisualCommand:
        """Clamp a visual command to safe limits."""
        clamped, _ = self.visuals.clamp_command(cmd)
        return clamped


# =============================================================================
# Breath-Vision Rails
# =============================================================================

class BreathVisionRails:
    """
    Safety enforcement for breath-vision sessions.

    This is WELLNESS, not medical treatment.
    """

    def __init__(self):
        self.covenant = EmbodimentCovenant()
        self.config = self.covenant.breath_vision
        self._session_active = False
        self._session_start: Optional[datetime] = None
        self._last_session_end: Optional[datetime] = None

    def can_start_session(self) -> Tuple[bool, str]:
        """Check if a new session can start."""
        # Already in session?
        if self._session_active:
            return False, "Session already active"

        # Cooldown between sessions
        if self._last_session_end:
            min_gap = self.config.get("session", {}).get(
                "min_gap_between_sessions_minutes", 5
            )
            elapsed = datetime.utcnow() - self._last_session_end
            if elapsed < timedelta(minutes=min_gap):
                remaining = min_gap - (elapsed.total_seconds() / 60)
                return False, f"Cooldown active. Wait {remaining:.1f} more minutes."

        return True, "OK to start"

    def start_session(self, duration_minutes: float) -> Tuple[bool, float]:
        """
        Start a breath-vision session.

        Returns (success, actual_duration).
        Duration is clamped to max.
        """
        can_start, reason = self.can_start_session()
        if not can_start:
            return False, 0

        max_duration = self.config.get("session", {}).get("max_duration_minutes", 10)
        actual_duration = min(duration_minutes, max_duration)

        self._session_active = True
        self._session_start = datetime.utcnow()

        return True, actual_duration

    def end_session(self):
        """End the current session."""
        self._session_active = False
        self._last_session_end = datetime.utcnow()

    def check_for_stop(self, utterance: str) -> bool:
        """Check if utterance is a stop phrase."""
        stop_phrases = self.config.get("session", {}).get("stop_phrases", [])
        utterance_lower = utterance.lower().strip()
        for phrase in stop_phrases:
            if phrase.lower() in utterance_lower:
                return True
        return False

    def get_disclaimer(self) -> str:
        """Get the session start disclaimer."""
        disclaimers = self.config.get("disclaimers", {}).get("required_at_start", [])
        return " ".join(disclaimers)

    def check_contraindication(self, text: str) -> Optional[str]:
        """Check if text mentions contraindications."""
        warn_if = self.config.get("contraindications", {}).get("warn_if_mentioned", [])
        text_lower = text.lower()

        for condition in warn_if:
            if condition.lower() in text_lower:
                return self.config.get("contraindications", {}).get("warning_response", "")

        return None

    def get_visual_limits(self) -> Dict[str, Any]:
        """Get visual safety limits."""
        return self.config.get("visuals", {})

    def get_haptic_limits(self) -> Dict[str, Any]:
        """Get haptic safety limits."""
        return self.config.get("haptics", {})


# =============================================================================
# Fusion Monitor Rails
# =============================================================================

class FusionRails:
    """
    Safety enforcement for fusion/self-monitoring.

    She can observe and suggest, but not take drastic action.
    """

    def __init__(self):
        self.covenant = EmbodimentCovenant()
        self.config = self.covenant.fusion

    def can_do_autonomously(self, action: str) -> bool:
        """Check if an action can be done without human consent."""
        autonomous = self.config.get("autonomous_allowed", [])
        for allowed in autonomous:
            if isinstance(allowed, dict):
                if allowed.get("action") == action:
                    return True
            elif allowed == action:
                return True
        return False

    def requires_consent(self, action: str) -> Optional[str]:
        """
        Check if action requires consent.

        Returns the prompt to show the user, or None if no consent needed.
        """
        requires = self.config.get("requires_human_consent", [])
        for item in requires:
            if isinstance(item, dict) and item.get("action") == action:
                return item.get("prompt", f"Allow {action}?")
        return None

    def is_forbidden(self, action: str) -> Optional[str]:
        """
        Check if action is forbidden.

        Returns the reason, or None if not forbidden.
        """
        forbidden = self.config.get("forbidden", [])
        for item in forbidden:
            if isinstance(item, dict) and item.get("action") == action:
                return item.get("reason", "Forbidden action")
        return None

    def get_alert_thresholds(self) -> Dict[str, float]:
        """Get thresholds for system alerts."""
        return self.config.get("reporting", {}).get("thresholds", {})

    def get_status_message(self, condition: str, **kwargs) -> str:
        """Get a status message template."""
        templates = self.config.get("reporting", {}).get("message_templates", {})
        template = templates.get(condition, f"Status: {condition}")
        return template.format(**kwargs)


# =============================================================================
# Convenience Functions
# =============================================================================

def check_motor_safe(
    joint: str,
    speed: float,
    torque: float,
) -> Tuple[bool, Dict[str, float]]:
    """
    Quick check if motor parameters are safe.

    Returns (is_safe, clamped_values).
    """
    rails = EmbodimentRails()
    cmd = MotorCommand(
        joint=joint,
        target_position_deg=0,
        speed_deg_per_s=speed,
        torque_percent=torque,
    )
    result = rails.clamp_motor_command(cmd)

    return not result.was_modified, {
        "speed": result.clamped.speed_deg_per_s,
        "torque": result.clamped.torque_percent,
    }


def check_breath_session_allowed() -> Tuple[bool, str]:
    """Quick check if a breath-vision session can start."""
    rails = BreathVisionRails()
    return rails.can_start_session()


def get_stop_phrases() -> List[str]:
    """Get all stop phrases across all modules."""
    rails = EmbodimentRails()
    bv_rails = BreathVisionRails()

    phrases = set(rails.session.get_stop_phrases())
    phrases.update(bv_rails.config.get("session", {}).get("stop_phrases", []))

    return list(phrases)
