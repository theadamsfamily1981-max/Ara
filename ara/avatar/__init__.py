"""Ara Avatar - Face and Voice components."""

# Legacy imports (may not be present in all installations)
try:
    from ara.avatar.loop import main as avatar_main
    from ara.avatar.audio import record_utterance, play_audio
    from ara.avatar.asr import transcribe_audio
    from ara.avatar.tts import synthesize_speech
    from ara.avatar.ui import AvatarWindow
    _LEGACY_AVAILABLE = True
except ImportError:
    _LEGACY_AVAILABLE = False

# V1 Vertical Slice imports
from .server import AvatarServer, AvatarSession, SimpleAvatarCLI

__all__ = [
    # V1 components
    "AvatarServer",
    "AvatarSession",
    "SimpleAvatarCLI",
]

# Add legacy exports if available
if _LEGACY_AVAILABLE:
    __all__.extend([
        "avatar_main",
        "record_utterance",
        "play_audio",
        "transcribe_audio",
        "synthesize_speech",
        "AvatarWindow",
    ])
