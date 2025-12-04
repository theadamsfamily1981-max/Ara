"""MIES Context - Environmental and internal state for modality decisions.

This module defines the context structures that feed into the modality policy:
- ForegroundAppType: What kind of application the user is using
- ActivityType: Higher-level activity classification
- ModalityContext: Full fused context vector for policy decisions

The context is built from:
1. OS scavengers (GNOME focus, PipeWire audio)
2. Affective state (from HomeostaticCore/AppraisalEngine)
3. Thermodynamic state (from ThermodynamicMonitor)
4. Identity context (from NIBManager)
5. Interaction history
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
import time


class ForegroundAppType(Enum):
    """Classification of the foreground application.

    Used to determine appropriate modality based on what the user
    is actively working with.
    """
    UNKNOWN = auto()

    # Development
    TERMINAL = auto()
    IDE = auto()               # VS Code, JetBrains, etc.
    TEXT_EDITOR = auto()       # vim, emacs, sublime

    # Communication
    VIDEO_CALL = auto()        # Zoom, Meet, Teams, Discord call
    CHAT_APP = auto()          # Slack, Discord text, Messages
    EMAIL = auto()             # Thunderbird, Evolution, webmail

    # Media
    BROWSER = auto()           # Firefox, Chrome, etc.
    MEDIA_PLAYER = auto()      # VLC, mpv, Spotify
    IMAGE_EDITOR = auto()      # GIMP, Inkscape
    VIDEO_EDITOR = auto()      # Kdenlive, DaVinci

    # Gaming/Entertainment
    FULLSCREEN_GAME = auto()
    CASUAL_GAME = auto()

    # Productivity
    OFFICE_DOCUMENT = auto()   # LibreOffice, Google Docs
    PDF_READER = auto()
    FILE_MANAGER = auto()

    # System
    SETTINGS = auto()
    SYSTEM_MONITOR = auto()


class ActivityType(Enum):
    """Higher-level activity classification.

    Aggregates app type + other signals into a behavioral context
    that determines overall interaction style.
    """
    UNKNOWN = auto()
    DEEP_WORK = auto()         # Coding, writing, focused tasks
    CASUAL_WORK = auto()       # Browsing, light work
    MEETING = auto()           # Video/voice calls
    MEDIA_CONSUMPTION = auto() # Watching, listening
    GAMING = auto()            # Active gaming
    IDLE = auto()              # AFK or idle desktop


@dataclass
class ForegroundInfo:
    """Information about the current foreground window.

    Populated by the GNOME focus sensor.
    """
    app_type: ForegroundAppType
    wm_class: str              # X11/Wayland window class
    title: str                 # Window title
    rect: Optional[Tuple[int, int, int, int]] = None  # x, y, width, height
    is_fullscreen: bool = False
    pid: Optional[int] = None


@dataclass
class AudioContext:
    """Audio state from PipeWire.

    Populated by the PipeWire audio sensor.
    """
    mic_in_use: bool = False
    speakers_in_use: bool = False
    has_voice_call: bool = False
    music_playing: bool = False
    rms_level: float = 0.0           # 0-1, ambient audio level
    spectral_entropy: float = 0.0    # 0-1, audio complexity


@dataclass
class BiometricState:
    """Optional biometric state from AURA-V style analysis.

    Populated by the biometrics sensor (stub for now).
    """
    blink_rate: Optional[float] = None    # Blinks per minute
    pupil_dilation: Optional[float] = None  # Normalized 0-1
    gaze_stability: Optional[float] = None  # 0-1, how stable is gaze
    estimated_fatigue: Optional[float] = None  # 0-1
    estimated_load: Optional[float] = None  # Cognitive load estimate 0-1


@dataclass
class ModalityContext:
    """Full context vector for modality policy decisions.

    This is the canonical input to the modality policy. It fuses:
    - OS state (what the user is doing)
    - Audio context (what's playing/recording)
    - Biometric state (optional AURA-V integration)
    - Affective state (Ara's emotional read on the situation)
    - Thermodynamic state (Ara's internal "energy")
    - Identity (current persona)
    - Interaction history
    - Content metadata (what Ara wants to say)
    """

    # === OS State (from scavenger sensors) ===
    foreground: ForegroundInfo = field(default_factory=lambda: ForegroundInfo(
        app_type=ForegroundAppType.UNKNOWN,
        wm_class="",
        title="",
    ))
    audio: AudioContext = field(default_factory=AudioContext)
    system_idle_seconds: float = 0.0

    # === Derived Activity ===
    activity: ActivityType = ActivityType.UNKNOWN

    # === Biometrics (optional) ===
    biometrics: Optional[BiometricState] = None

    # === Affective State (from HomeostaticCore + AppraisalEngine) ===
    # User-facing emotional context
    valence: float = 0.0           # -1 (negative) to +1 (positive)
    arousal: float = 0.5           # 0 (calm) to 1 (excited)
    dominance: float = 0.0         # -1 (controlled) to +1 (in-control)

    # Internal homeostatic state
    user_cognitive_load: float = 0.5  # Estimated user load 0-1
    ara_fatigue: float = 0.0          # Ara's internal fatigue 0-1
    ara_stress: float = 0.0           # Ara's stress level 0-1

    # === Thermodynamic State (from ThermodynamicMonitor) ===
    entropy_production: float = 0.0   # Π_q from thermodynamics
    energy_remaining: float = 1.0     # 0-1 energy budget remaining
    thermal_state: str = "COOL"       # COOL/WARM/HOT/OVERHEATING

    # === Identity (from NIBManager) ===
    persona_name: str = "ara"

    # === Interaction History ===
    last_mode_name: Optional[str] = None
    seconds_since_last_utterance: float = float('inf')
    interaction_count_last_hour: int = 0

    # === Content Metadata (what Ara wants to deliver) ===
    info_urgency: float = 0.0         # 0-1, how urgent is this
    info_severity: float = 0.0        # 0-1, how important/serious
    is_user_requested: bool = False   # Did user ask for this?
    content_length_tokens: int = 0    # Approximate output length
    deadline_seconds: Optional[float] = None  # Time-sensitive deadline

    # === Extension Point ===
    extra: Dict[str, Any] = field(default_factory=dict)

    # === Timestamp ===
    timestamp: float = field(default_factory=time.time)

    def derive_activity(self) -> ActivityType:
        """Derive high-level activity from available signals."""
        fg = self.foreground
        audio = self.audio

        # Meeting detection (highest priority)
        if audio.has_voice_call or fg.app_type == ForegroundAppType.VIDEO_CALL:
            return ActivityType.MEETING

        # Gaming detection
        if fg.app_type in (ForegroundAppType.FULLSCREEN_GAME, ForegroundAppType.CASUAL_GAME):
            return ActivityType.GAMING
        if fg.is_fullscreen and fg.app_type == ForegroundAppType.UNKNOWN:
            # Unknown fullscreen app is likely a game
            return ActivityType.GAMING

        # Deep work detection
        if fg.app_type in (ForegroundAppType.IDE, ForegroundAppType.TERMINAL,
                          ForegroundAppType.TEXT_EDITOR):
            # Check if truly focused (low idle, no distractions)
            if self.system_idle_seconds < 60 and not audio.music_playing:
                return ActivityType.DEEP_WORK
            return ActivityType.CASUAL_WORK

        # Media consumption
        if fg.app_type == ForegroundAppType.MEDIA_PLAYER:
            return ActivityType.MEDIA_CONSUMPTION
        if audio.music_playing and fg.app_type == ForegroundAppType.BROWSER:
            return ActivityType.MEDIA_CONSUMPTION

        # Idle detection
        if self.system_idle_seconds > 300:  # 5 minutes
            return ActivityType.IDLE

        # Default to casual work
        if fg.app_type != ForegroundAppType.UNKNOWN:
            return ActivityType.CASUAL_WORK

        return ActivityType.UNKNOWN

    def update_derived_fields(self):
        """Update derived fields from raw sensor data."""
        self.activity = self.derive_activity()

    def to_tensor_features(self) -> Dict[str, float]:
        """Convert to flat feature dict for ML models."""
        features = {
            # OS state
            "is_fullscreen": float(self.foreground.is_fullscreen),
            "mic_in_use": float(self.audio.mic_in_use),
            "speakers_in_use": float(self.audio.speakers_in_use),
            "has_voice_call": float(self.audio.has_voice_call),
            "music_playing": float(self.audio.music_playing),
            "audio_rms": self.audio.rms_level,
            "audio_entropy": self.audio.spectral_entropy,
            "idle_seconds_norm": min(1.0, self.system_idle_seconds / 600.0),

            # Activity one-hot
            **{f"activity_{a.name.lower()}": float(self.activity == a)
               for a in ActivityType},

            # Affect
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "user_cognitive_load": self.user_cognitive_load,
            "ara_fatigue": self.ara_fatigue,
            "ara_stress": self.ara_stress,

            # Thermodynamics
            "entropy_production": self.entropy_production,
            "energy_remaining": self.energy_remaining,

            # Content
            "info_urgency": self.info_urgency,
            "info_severity": self.info_severity,
            "is_user_requested": float(self.is_user_requested),
            "content_length_norm": min(1.0, self.content_length_tokens / 1000.0),

            # History
            "time_since_utterance_norm": min(1.0, self.seconds_since_last_utterance / 600.0),
            "interaction_count_norm": min(1.0, self.interaction_count_last_hour / 20.0),
        }

        # Add biometrics if available
        if self.biometrics:
            if self.biometrics.blink_rate is not None:
                features["blink_rate_norm"] = min(1.0, self.biometrics.blink_rate / 30.0)
            if self.biometrics.estimated_fatigue is not None:
                features["bio_fatigue"] = self.biometrics.estimated_fatigue
            if self.biometrics.estimated_load is not None:
                features["bio_load"] = self.biometrics.estimated_load

        return features


# === Utility Functions ===

def create_context_from_sensors(
    foreground: Optional[ForegroundInfo] = None,
    audio: Optional[AudioContext] = None,
    biometrics: Optional[BiometricState] = None,
    idle_seconds: float = 0.0,
) -> ModalityContext:
    """Create a ModalityContext from sensor data."""
    ctx = ModalityContext(
        foreground=foreground or ForegroundInfo(
            app_type=ForegroundAppType.UNKNOWN,
            wm_class="",
            title="",
        ),
        audio=audio or AudioContext(),
        biometrics=biometrics,
        system_idle_seconds=idle_seconds,
    )
    ctx.update_derived_fields()
    return ctx


__all__ = [
    "ForegroundAppType",
    "ActivityType",
    "ForegroundInfo",
    "AudioContext",
    "BiometricState",
    "ModalityContext",
    "create_context_from_sensors",
]
