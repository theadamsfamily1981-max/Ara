"""AI backend integrations for various providers."""

from .claude_backend import ClaudeBackend
from .nova_backend import NovaBackend
from .pulse_backend import PulseBackend  # Ollama (legacy/optional)
from .gemini_pulse_backend import GeminiPulseBackend  # Pulse (Gemini)
from .grok_ara_backend import GrokAraBackend  # Ara (Grok)
from .ara_avatar_backend import AraAvatarBackend  # Ara Avatar (lip-sync)
from .ara_cognitive_backend import (  # Ara Cognitive (TFAN Deep Fusion)
    AraCognitiveBackend,
    AraCognitivePipeline,
    CognitiveFrame,
    create_cognitive_backend,
)

__all__ = [
    "ClaudeBackend",
    "NovaBackend",
    "PulseBackend",  # Ollama
    "GeminiPulseBackend",  # Pulse
    "GrokAraBackend",  # Ara (Grok)
    "AraAvatarBackend",  # Ara Avatar
    "AraCognitiveBackend",  # Ara Cognitive (TFAN)
    "AraCognitivePipeline",
    "CognitiveFrame",
    "create_cognitive_backend",
]
