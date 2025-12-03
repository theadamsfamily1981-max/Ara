"""Advanced audio pipeline for realistic voice interaction.

This module provides broadcast-quality TTS, enhanced ASR, and
audio mastering capabilities for the conversational avatar system.

Components:
    - AdvancedTTS: Smart chunking TTS with multi-sample voice cloning
    - AudioMastering: Compressor + reverb + limiter chain
    - EnhancedASR: Whisper with custom vocabulary support
    - ProfileManager: Fast/balanced/deep mode switching

Usage:
    from audio_pipeline import ProfileManager, ProfileMode

    # Quick start with a profile
    pm = ProfileManager()
    profile = pm.set_active_by_mode(ProfileMode.BALANCED)

    # Use individual components
    from audio_pipeline import AdvancedTTS, TTSConfig
    tts = AdvancedTTS(TTSConfig(speed=1.0))
    tts.add_voice_samples(["voice1.wav", "voice2.wav"])
    tts.synthesize("Hello, world!", output_path="output.wav")
"""

from .advanced_tts import AdvancedTTS, TTSConfig, SmartChunker, VoiceSampler
from .audio_mastering import AudioMastering, MasteringConfig
from .enhanced_asr import EnhancedASR, ASRConfig, TranscriptionResult
from .profiles import (
    ProfileManager,
    ProfileMode,
    AudioProfile,
    FAST_PROFILE,
    BALANCED_PROFILE,
    DEEP_PROFILE,
)

__all__ = [
    # Main classes
    'AdvancedTTS',
    'AudioMastering',
    'EnhancedASR',
    'ProfileManager',
    # Configurations
    'TTSConfig',
    'MasteringConfig',
    'ASRConfig',
    # Profiles
    'ProfileMode',
    'AudioProfile',
    'FAST_PROFILE',
    'BALANCED_PROFILE',
    'DEEP_PROFILE',
    # Utilities
    'SmartChunker',
    'VoiceSampler',
    'TranscriptionResult',
]
