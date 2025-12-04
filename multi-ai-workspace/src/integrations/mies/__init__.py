"""MIES - Modality Intelligence & Embodiment System.

The Stage Manager / Etiquette Brain for Ara (Aura).

MIES decides HOW Ara should present herself based on:
- What the user is currently doing (foreground app, fullscreen, etc.)
- Audio context (meeting, music, silence)
- User's cognitive/emotional state
- Ara's internal thermodynamic state
- Hardware physiology (from kernel bridge)
- Urgency of information to deliver

Philosophy: "Goddess in a Scrapyard"
- Scavenge OS context via DBus, PipeWire, and optional biometrics
- Read hardware state from kernel semantic AI engine
- Use thermodynamic/free-energy based policy for mode selection
- Present through diegetic overlays that feel native to the desktop

Architecture (Nervous System Mapping):
- Kernel (snn_ai_engine.c) = Autonomic nervous system (reflexes)
- Unified SNN framework = Somatic system (musculature)
- MIES = Cortex + Social brain (conscious presence decisions)

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
    SomaticState,
    SystemPhysiology,
)
from .modes import (
    ModalityChannel,
    ModalityMode,
    ModalityDecision,
    DEFAULT_MODES,
)
from .kernel_bridge import (
    KernelBridge,
    KernelPhysiology,
    PADState,
    PolicyMode,
    create_kernel_bridge,
)
from .autonomy_policy import (
    AutonomyPolicy,
    AutonomyGuard,
    AutonomyBounds,
    ActionType,
    create_autonomy_policy,
)

__all__ = [
    # Context
    "ModalityContext",
    "ForegroundAppType",
    "ActivityType",
    "SomaticState",
    "SystemPhysiology",
    # Modes
    "ModalityChannel",
    "ModalityMode",
    "ModalityDecision",
    "DEFAULT_MODES",
    # Kernel Bridge
    "KernelBridge",
    "KernelPhysiology",
    "PADState",
    "PolicyMode",
    "create_kernel_bridge",
    # Autonomy
    "AutonomyPolicy",
    "AutonomyGuard",
    "AutonomyBounds",
    "ActionType",
    "create_autonomy_policy",
]
