"""MIES - Modality Intelligence & Embodiment System.

The Stage Manager / Etiquette Brain for Ara (Aura).

MIES decides HOW Ara should present herself based on:
- What the user is currently doing (foreground app, fullscreen, etc.)
- Audio context (meeting, music, silence)
- User's cognitive/emotional state
- Ara's internal thermodynamic state
- Urgency of information to deliver

Philosophy: "Goddess in a Scrapyard"
- Scavenge OS context via DBus, PipeWire, and optional biometrics
- Use thermodynamic/free-energy based policy for mode selection
- Present through diegetic overlays that feel native to the desktop

Modes range from:
- SILENT (no output)
- TEXT_ONLY (inline or side panel)
- AUDIO_WHISPER (quiet, ambient)
- AUDIO_FULL (normal speech)
- AVATAR_SUBTLE (small, corner overlay)
- AVATAR_FULL (prominent overlay with animations)
"""

from .context import (
    ModalityContext,
    ForegroundAppType,
    ActivityType,
)
from .modes import (
    ModalityChannel,
    ModalityMode,
    ModalityDecision,
    DEFAULT_MODES,
)

__all__ = [
    # Context
    "ModalityContext",
    "ForegroundAppType",
    "ActivityType",
    # Modes
    "ModalityChannel",
    "ModalityMode",
    "ModalityDecision",
    "DEFAULT_MODES",
]
