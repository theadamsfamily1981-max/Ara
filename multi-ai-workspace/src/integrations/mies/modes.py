"""MIES Modes - Modality channels and mode definitions.

This module defines:
- ModalityChannel: The output channel (text, audio, avatar)
- ModalityMode: A specific configuration with presence/intrusiveness/cost
- ModalityDecision: The policy output including transition parameters
- DEFAULT_MODES: Pre-configured modes with sensible manifold coordinates

The mode space forms a manifold where transitions can be geodesic
(smooth, continuous) rather than jarring jumps.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import time


class ModalityChannel(Enum):
    """Primary output channel for Ara's response.

    These define the fundamental delivery mechanism.
    """
    SILENT = auto()              # No output
    TEXT_INLINE = auto()         # Text in main conversation
    TEXT_SIDE_PANEL = auto()     # Text in a side widget
    TEXT_NOTIFICATION = auto()   # System notification
    AUDIO_WHISPER = auto()       # Quiet, ambient audio
    AUDIO_FULL = auto()          # Normal speech volume
    AVATAR_OVERLAY_MINI = auto() # Small corner avatar
    AVATAR_OVERLAY_FULL = auto() # Full avatar presence
    SILENT_BACKGROUND = auto()   # Background processing, no output


@dataclass
class ModalityMode:
    """A specific modality configuration.

    Modes exist in a manifold space defined by:
    - presence_intensity: How "present" Ara feels (0 = invisible, 1 = dominant)
    - intrusiveness: How disruptive to user flow (0 = none, 1 = high)
    - energy_cost: Computational/thermodynamic cost (0 = cheap, 1 = expensive)
    - bandwidth_cost: Information bandwidth required (0 = low, 1 = high)

    The policy selects modes by minimizing energy while respecting context.
    """
    name: str
    channel: ModalityChannel

    # Manifold coordinates
    presence_intensity: float = 0.5   # [0, 1] How "present" Ara is
    intrusiveness: float = 0.5        # [0, 1] How disruptive
    energy_cost: float = 0.5          # [0, 1] Thermodynamic cost
    bandwidth_cost: float = 0.5       # [0, 1] Information density

    # Audio parameters (if applicable)
    volume: float = 0.7               # [0, 1] Audio volume
    voice_style: str = "normal"       # normal, whisper, emphatic

    # Avatar parameters (if applicable)
    avatar_size: float = 0.5          # [0, 1] Relative size
    avatar_opacity: float = 1.0       # [0, 1] Transparency
    avatar_position: str = "corner"   # corner, side, center, follow

    # Text parameters (if applicable)
    text_style: str = "normal"        # normal, minimal, detailed

    def distance_to(self, other: 'ModalityMode') -> float:
        """Compute distance in mode manifold (for smooth transitions)."""
        return (
            (self.presence_intensity - other.presence_intensity) ** 2 +
            (self.intrusiveness - other.intrusiveness) ** 2 +
            (self.energy_cost - other.energy_cost) ** 2 +
            (self.bandwidth_cost - other.bandwidth_cost) ** 2
        ) ** 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "channel": self.channel.name,
            "presence_intensity": self.presence_intensity,
            "intrusiveness": self.intrusiveness,
            "energy_cost": self.energy_cost,
            "bandwidth_cost": self.bandwidth_cost,
            "volume": self.volume,
            "voice_style": self.voice_style,
            "avatar_size": self.avatar_size,
            "avatar_opacity": self.avatar_opacity,
            "avatar_position": self.avatar_position,
            "text_style": self.text_style,
        }


@dataclass
class TransitionParams:
    """Parameters for transitioning between modes.

    Supports smooth, geodesic transitions rather than jarring jumps.
    """
    duration_ms: int = 300           # Transition duration
    easing: str = "ease-out"         # CSS-style easing function
    fade_out_first: bool = False     # Fade out old before fade in new
    preserve_context: bool = True    # Keep conversation context visible


@dataclass
class ModalityDecision:
    """The output of the modality policy.

    Includes:
    - The chosen mode
    - Transition parameters for smooth mode changes
    - Permission requirements
    - Rationale for logging/debugging
    """
    mode: ModalityMode
    transition: TransitionParams = field(default_factory=TransitionParams)

    # Permission/etiquette
    should_ask_permission: bool = False
    permission_prompt: Optional[str] = None

    # Policy metadata
    rationale: str = ""
    confidence: float = 1.0
    energy_score: float = 0.0        # Energy function value
    alternatives_considered: List[str] = field(default_factory=list)

    # Timing
    timestamp: float = field(default_factory=time.time)
    valid_for_seconds: float = 30.0  # How long this decision is valid

    def is_valid(self) -> bool:
        """Check if this decision is still valid."""
        return (time.time() - self.timestamp) < self.valid_for_seconds


# === Default Mode Registry ===

# SILENT: No output at all
MODE_SILENT = ModalityMode(
    name="silent",
    channel=ModalityChannel.SILENT,
    presence_intensity=0.0,
    intrusiveness=0.0,
    energy_cost=0.0,
    bandwidth_cost=0.0,
)

# BACKGROUND: Processing silently, might surface later
MODE_BACKGROUND = ModalityMode(
    name="background",
    channel=ModalityChannel.SILENT_BACKGROUND,
    presence_intensity=0.05,
    intrusiveness=0.0,
    energy_cost=0.1,
    bandwidth_cost=0.0,
)

# TEXT_INLINE: Standard text response
MODE_TEXT_INLINE = ModalityMode(
    name="text_inline",
    channel=ModalityChannel.TEXT_INLINE,
    presence_intensity=0.3,
    intrusiveness=0.2,
    energy_cost=0.2,
    bandwidth_cost=0.5,
    text_style="normal",
)

# TEXT_MINIMAL: Brief text response
MODE_TEXT_MINIMAL = ModalityMode(
    name="text_minimal",
    channel=ModalityChannel.TEXT_INLINE,
    presence_intensity=0.2,
    intrusiveness=0.1,
    energy_cost=0.1,
    bandwidth_cost=0.2,
    text_style="minimal",
)

# TEXT_SIDE: Side panel text (less intrusive)
MODE_TEXT_SIDE = ModalityMode(
    name="text_side",
    channel=ModalityChannel.TEXT_SIDE_PANEL,
    presence_intensity=0.25,
    intrusiveness=0.15,
    energy_cost=0.2,
    bandwidth_cost=0.5,
    text_style="normal",
)

# TEXT_NOTIFICATION: System notification
MODE_TEXT_NOTIFICATION = ModalityMode(
    name="text_notification",
    channel=ModalityChannel.TEXT_NOTIFICATION,
    presence_intensity=0.35,
    intrusiveness=0.4,
    energy_cost=0.15,
    bandwidth_cost=0.2,
    text_style="minimal",
)

# AUDIO_WHISPER: Quiet, ambient voice
MODE_AUDIO_WHISPER = ModalityMode(
    name="audio_whisper",
    channel=ModalityChannel.AUDIO_WHISPER,
    presence_intensity=0.4,
    intrusiveness=0.35,
    energy_cost=0.4,
    bandwidth_cost=0.3,
    volume=0.3,
    voice_style="whisper",
)

# AUDIO_NORMAL: Standard voice
MODE_AUDIO_NORMAL = ModalityMode(
    name="audio_normal",
    channel=ModalityChannel.AUDIO_FULL,
    presence_intensity=0.6,
    intrusiveness=0.5,
    energy_cost=0.5,
    bandwidth_cost=0.5,
    volume=0.7,
    voice_style="normal",
)

# AUDIO_EMPHATIC: Louder, more urgent
MODE_AUDIO_EMPHATIC = ModalityMode(
    name="audio_emphatic",
    channel=ModalityChannel.AUDIO_FULL,
    presence_intensity=0.8,
    intrusiveness=0.7,
    energy_cost=0.6,
    bandwidth_cost=0.6,
    volume=0.9,
    voice_style="emphatic",
)

# AVATAR_SUBTLE: Small corner presence
MODE_AVATAR_SUBTLE = ModalityMode(
    name="avatar_subtle",
    channel=ModalityChannel.AVATAR_OVERLAY_MINI,
    presence_intensity=0.35,
    intrusiveness=0.2,
    energy_cost=0.5,
    bandwidth_cost=0.3,
    avatar_size=0.15,
    avatar_opacity=0.7,
    avatar_position="corner",
)

# AVATAR_PRESENT: Medium overlay
MODE_AVATAR_PRESENT = ModalityMode(
    name="avatar_present",
    channel=ModalityChannel.AVATAR_OVERLAY_MINI,
    presence_intensity=0.5,
    intrusiveness=0.4,
    energy_cost=0.6,
    bandwidth_cost=0.5,
    avatar_size=0.25,
    avatar_opacity=0.9,
    avatar_position="side",
)

# AVATAR_FULL: Full avatar with voice
MODE_AVATAR_FULL = ModalityMode(
    name="avatar_full",
    channel=ModalityChannel.AVATAR_OVERLAY_FULL,
    presence_intensity=0.9,
    intrusiveness=0.8,
    energy_cost=0.9,
    bandwidth_cost=0.8,
    volume=0.7,
    voice_style="normal",
    avatar_size=0.4,
    avatar_opacity=1.0,
    avatar_position="side",
)

# AVATAR_FOLLOW: Avatar that follows active window
MODE_AVATAR_FOLLOW = ModalityMode(
    name="avatar_follow",
    channel=ModalityChannel.AVATAR_OVERLAY_MINI,
    presence_intensity=0.45,
    intrusiveness=0.35,
    energy_cost=0.55,
    bandwidth_cost=0.4,
    avatar_size=0.2,
    avatar_opacity=0.85,
    avatar_position="follow",
)


# Registry of all default modes
DEFAULT_MODES: Dict[str, ModalityMode] = {
    "silent": MODE_SILENT,
    "background": MODE_BACKGROUND,
    "text_inline": MODE_TEXT_INLINE,
    "text_minimal": MODE_TEXT_MINIMAL,
    "text_side": MODE_TEXT_SIDE,
    "text_notification": MODE_TEXT_NOTIFICATION,
    "audio_whisper": MODE_AUDIO_WHISPER,
    "audio_normal": MODE_AUDIO_NORMAL,
    "audio_emphatic": MODE_AUDIO_EMPHATIC,
    "avatar_subtle": MODE_AVATAR_SUBTLE,
    "avatar_present": MODE_AVATAR_PRESENT,
    "avatar_full": MODE_AVATAR_FULL,
    "avatar_follow": MODE_AVATAR_FOLLOW,
}


def get_mode_by_channel(channel: ModalityChannel) -> List[ModalityMode]:
    """Get all modes that use a specific channel."""
    return [m for m in DEFAULT_MODES.values() if m.channel == channel]


def get_modes_by_intrusiveness(max_intrusiveness: float) -> List[ModalityMode]:
    """Get all modes below a given intrusiveness threshold."""
    return [m for m in DEFAULT_MODES.values() if m.intrusiveness <= max_intrusiveness]


def interpolate_modes(
    mode_a: ModalityMode,
    mode_b: ModalityMode,
    t: float,
) -> ModalityMode:
    """Interpolate between two modes (for smooth transitions).

    Args:
        mode_a: Starting mode
        mode_b: Ending mode
        t: Interpolation factor [0, 1]

    Returns:
        Interpolated mode (uses mode_b's channel if t > 0.5)
    """
    def lerp(a: float, b: float) -> float:
        return a + t * (b - a)

    return ModalityMode(
        name=f"interp_{mode_a.name}_{mode_b.name}_{t:.2f}",
        channel=mode_b.channel if t > 0.5 else mode_a.channel,
        presence_intensity=lerp(mode_a.presence_intensity, mode_b.presence_intensity),
        intrusiveness=lerp(mode_a.intrusiveness, mode_b.intrusiveness),
        energy_cost=lerp(mode_a.energy_cost, mode_b.energy_cost),
        bandwidth_cost=lerp(mode_a.bandwidth_cost, mode_b.bandwidth_cost),
        volume=lerp(mode_a.volume, mode_b.volume),
        avatar_size=lerp(mode_a.avatar_size, mode_b.avatar_size),
        avatar_opacity=lerp(mode_a.avatar_opacity, mode_b.avatar_opacity),
    )


__all__ = [
    "ModalityChannel",
    "ModalityMode",
    "TransitionParams",
    "ModalityDecision",
    "DEFAULT_MODES",
    "get_mode_by_channel",
    "get_modes_by_intrusiveness",
    "interpolate_modes",
    # Default modes
    "MODE_SILENT",
    "MODE_BACKGROUND",
    "MODE_TEXT_INLINE",
    "MODE_TEXT_MINIMAL",
    "MODE_TEXT_SIDE",
    "MODE_TEXT_NOTIFICATION",
    "MODE_AUDIO_WHISPER",
    "MODE_AUDIO_NORMAL",
    "MODE_AUDIO_EMPHATIC",
    "MODE_AVATAR_SUBTLE",
    "MODE_AVATAR_PRESENT",
    "MODE_AVATAR_FULL",
    "MODE_AVATAR_FOLLOW",
]
