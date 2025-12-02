"""Ara Avatar - Face and Voice components."""

from ara.avatar.loop import main as avatar_main
from ara.avatar.audio import record_utterance, play_audio
from ara.avatar.asr import transcribe_audio
from ara.avatar.tts import synthesize_speech
from ara.avatar.ui import AvatarWindow

__all__ = [
    "avatar_main",
    "record_utterance",
    "play_audio",
    "transcribe_audio",
    "synthesize_speech",
    "AvatarWindow",
]
