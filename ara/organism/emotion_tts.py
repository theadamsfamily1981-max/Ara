"""
Emotion TTS - VAD to Speech Synthesis
=====================================

Maps emotional states to spoken phrases with prosody modulation.

VAD → Prosody mapping:
- Arousal → speech rate (higher = faster)
- Dominance → pitch (higher = lower, more authoritative)
- Valence → subtle rate/pitch adjustments

Pluggable backends:
- espeak (default Linux TTS)
- piper (neural TTS)
- say (macOS)
- mock (print only)
- custom (your RVC/voice pipeline)
"""

from __future__ import annotations
import asyncio
import subprocess
import logging
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, Callable
from enum import Enum

logger = logging.getLogger(__name__)


class TTSBackend(Enum):
    """Available TTS backends."""
    ESPEAK = "espeak"
    PIPER = "piper"
    SAY = "say"  # macOS
    MOCK = "mock"
    CUSTOM = "custom"


@dataclass
class TTSConfig:
    """Configuration for emotion TTS."""
    backend: TTSBackend = TTSBackend.MOCK
    base_rate: int = 175       # Words per minute (espeak)
    base_pitch: int = 50       # Pitch 0-99 (espeak)
    base_volume: float = 0.8   # Volume 0-1
    min_strength: float = 0.3  # Don't speak below this strength
    cooldown: float = 2.0      # Seconds between utterances


# ============================================================================
# Emotion to Phrase Templates
# ============================================================================

EMOTION_PHRASES: Dict[str, str] = {
    # High valence emotions
    "joy": "I feel good about this.",
    "excitement": "This is exciting!",
    "elation": "This is wonderful!",
    "serenity": "All is calm and stable.",
    "contentment": "Things are going well.",
    "calm": "Everything is under control.",
    "trust": "I have confidence in this.",
    "pride": "We've done well here.",
    "anticipation": "Something promising is coming.",

    # Low valence emotions
    "fear": "I sense danger approaching.",
    "anger": "This situation is frustrating.",
    "rage": "This is unacceptable! Escalating now.",
    "anxiety": "I'm concerned about this pattern.",
    "sadness": "This outcome is unfortunate.",
    "depression": "I feel overwhelmed by this.",
    "boredom": "Nothing noteworthy is happening.",
    "disgust": "This pattern is unpleasant.",
    "contempt": "This is beneath acceptable levels.",

    # Dominance variations
    "submission": "I defer to your judgment on this.",
    "helplessness": "I don't know how to resolve this.",
    "vigilance": "Staying alert for changes.",
    "surprise": "That was unexpected.",
    "overwhelmed": "Too many events occurring at once.",

    # Blends (common compound emotions)
    "rage_contempt_blend": "Route conditions enraged me. Taking aggressive action.",
    "fear_submission_blend": "I'm afraid and need help. Requesting assistance.",
    "joy_trust_blend": "Confident and satisfied with current state.",
    "anxiety_vigilance_blend": "Something unstable. Watching closely.",

    # Fallback
    "neutral": "Status unchanged.",
    "unknown": "Processing current state.",
}

# Tags that trigger specific phrases
TAG_PHRASES: Dict[str, str] = {
    "route_flap": "Route flap detected in the network.",
    "route_flap_burst": "Route flap storm! Multiple instabilities.",
    "user_rage": "User intervention may be needed.",
    "fabric_stressed": "Fabric resources are under pressure.",
    "anomaly_burst": "Multiple anomalies occurring.",
    "memory_recall": "I remember seeing this before.",
    "novel_pattern": "This is a new pattern I haven't seen.",
}


def emotion_to_phrase(emotion: str, strength: float, tags: list) -> str:
    """
    Convert emotion + tags to a spoken phrase.

    Args:
        emotion: Emotion name (e.g., "rage", "calm")
        strength: Emotion strength [0, 1]
        tags: List of context tags

    Returns:
        Phrase to speak
    """
    emotion_lower = emotion.lower().replace(" ", "_")

    # Check for tag-specific phrases first
    for tag in tags:
        tag_lower = tag.lower().replace(" ", "_")
        if tag_lower in TAG_PHRASES:
            return TAG_PHRASES[tag_lower]

    # Get base emotion phrase
    phrase = EMOTION_PHRASES.get(emotion_lower, EMOTION_PHRASES["unknown"])

    # Add tag context if available
    if tags and emotion_lower not in ["neutral", "unknown", "calm", "serenity"]:
        tag_str = ", ".join(tags[:2])  # Limit to 2 tags
        phrase = f"{phrase} Related to: {tag_str}."

    # Intensity modifier for high strength
    if strength > 0.85:
        phrase = f"Strongly: {phrase}"

    return phrase


# ============================================================================
# VAD to Prosody Mapping
# ============================================================================

def vad_to_prosody(
    valence: float,
    arousal: float,
    dominance: float,
    base_rate: int = 175,
    base_pitch: int = 50,
) -> Tuple[int, int, float]:
    """
    Map VAD coordinates to prosody parameters.

    Args:
        valence: [-1, +1] pleasant vs unpleasant
        arousal: [-1, +1] activated vs deactivated
        dominance: [-1, +1] in-control vs helpless
        base_rate: Base words per minute
        base_pitch: Base pitch (0-99)

    Returns:
        (rate, pitch, volume)
    """
    # Clamp inputs
    v = max(-1.0, min(1.0, valence))
    a = max(-1.0, min(1.0, arousal))
    d = max(-1.0, min(1.0, dominance))

    # Rate: higher arousal + positive valence = faster
    rate_mult = 1.0 + 0.4 * a + 0.1 * v
    rate = int(base_rate * rate_mult)
    rate = max(100, min(300, rate))

    # Pitch: higher dominance = lower pitch (more authoritative)
    # Higher arousal = slight pitch increase
    pitch = int(base_pitch - 15 * d + 10 * a)
    pitch = max(10, min(90, pitch))

    # Volume: dominance + arousal = louder
    volume = 0.6 + 0.25 * d + 0.15 * abs(a)
    volume = max(0.3, min(1.0, volume))

    return rate, pitch, volume


# ============================================================================
# TTS Backends
# ============================================================================

async def _speak_espeak(text: str, rate: int, pitch: int, volume: float) -> None:
    """Speak using espeak."""
    try:
        amplitude = int(volume * 200)  # espeak amplitude 0-200
        cmd = ["espeak", "-s", str(rate), "-p", str(pitch), "-a", str(amplitude), text]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(proc.wait(), timeout=10.0)
    except FileNotFoundError:
        logger.warning("espeak not found")
    except asyncio.TimeoutError:
        logger.warning("espeak timed out")
    except Exception as e:
        logger.error(f"espeak error: {e}")


async def _speak_piper(text: str, rate: int, pitch: int, volume: float) -> None:
    """Speak using Piper neural TTS."""
    # Piper doesn't support rate/pitch directly - would need SSML or post-processing
    try:
        cmd = ["piper", "--output_file", "-"]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        proc.stdin.write(text.encode())
        await proc.stdin.drain()
        proc.stdin.close()
        await asyncio.wait_for(proc.wait(), timeout=30.0)
    except FileNotFoundError:
        logger.warning("piper not found, falling back to espeak")
        await _speak_espeak(text, rate, pitch, volume)
    except Exception as e:
        logger.error(f"piper error: {e}")


async def _speak_say(text: str, rate: int, pitch: int, volume: float) -> None:
    """Speak using macOS say command."""
    try:
        cmd = ["say", "-r", str(rate), text]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(proc.wait(), timeout=10.0)
    except FileNotFoundError:
        logger.warning("say not found (not macOS?)")
    except Exception as e:
        logger.error(f"say error: {e}")


async def _speak_mock(text: str, rate: int, pitch: int, volume: float) -> None:
    """Mock TTS - just print."""
    print(f"[TTS rate={rate} pitch={pitch} vol={volume:.1f}] {text}")


# ============================================================================
# Main TTS Interface
# ============================================================================

# Global state
_last_speak_time: float = 0.0
_config: TTSConfig = TTSConfig()
_custom_callback: Optional[Callable] = None


def configure_tts(
    backend: TTSBackend = TTSBackend.MOCK,
    base_rate: int = 175,
    base_pitch: int = 50,
    cooldown: float = 2.0,
    custom_callback: Optional[Callable] = None,
) -> None:
    """
    Configure TTS settings.

    Args:
        backend: TTS backend to use
        base_rate: Base speech rate
        base_pitch: Base pitch
        cooldown: Minimum seconds between utterances
        custom_callback: Custom async callback(text, rate, pitch, volume)
    """
    global _config, _custom_callback
    _config = TTSConfig(
        backend=backend,
        base_rate=base_rate,
        base_pitch=base_pitch,
        cooldown=cooldown,
    )
    _custom_callback = custom_callback


async def speak(text: str, rate: int, pitch: int, volume: float) -> None:
    """
    Speak text with specified prosody.

    Args:
        text: Text to speak
        rate: Speech rate (WPM for espeak)
        pitch: Pitch (0-99 for espeak)
        volume: Volume (0-1)
    """
    global _last_speak_time
    import time

    # Cooldown check
    now = time.time()
    if now - _last_speak_time < _config.cooldown:
        return
    _last_speak_time = now

    backend = _config.backend

    if backend == TTSBackend.ESPEAK:
        await _speak_espeak(text, rate, pitch, volume)
    elif backend == TTSBackend.PIPER:
        await _speak_piper(text, rate, pitch, volume)
    elif backend == TTSBackend.SAY:
        await _speak_say(text, rate, pitch, volume)
    elif backend == TTSBackend.CUSTOM and _custom_callback is not None:
        await _custom_callback(text, rate, pitch, volume)
    else:
        await _speak_mock(text, rate, pitch, volume)


async def speak_from_state(state) -> None:
    """
    Speak based on an EmotionState.

    This is the main entry point called by emotion_bridge.

    Args:
        state: EmotionState with emotion, strength, valence, arousal, dominance, tags
    """
    # Filter weak emotions
    if state.strength < _config.min_strength:
        return

    # Also filter if not much happening (low arousal, neutral valence)
    if (abs(state.arousal) < 0.3 and
        abs(state.valence) < 0.2 and
        abs(state.dominance) < 0.2):
        return

    # Generate phrase
    text = emotion_to_phrase(state.emotion, state.strength, state.tags)

    # Compute prosody
    rate, pitch, volume = vad_to_prosody(
        state.valence,
        state.arousal,
        state.dominance,
        _config.base_rate,
        _config.base_pitch,
    )

    # Speak
    await speak(text, rate, pitch, volume)


async def speak_memory_event(event_type: str, index: int, similarity: float) -> None:
    """Speak a memory event (recall/dream)."""
    if event_type == "recall" and similarity > 0.8:
        text = f"Memory surfacing. I've seen this before, index {index}."
        await speak(text, 160, 45, 0.7)
    elif event_type == "dream" and similarity > 0.85:
        text = f"Dreaming of memory {index}."
        await speak(text, 140, 40, 0.5)


# ============================================================================
# Demo
# ============================================================================

def demo():
    """Demonstrate emotion TTS."""
    print("=" * 60)
    print("Emotion TTS Demo")
    print("=" * 60)

    # Test prosody mapping
    test_states = [
        ("calm", 0.6, 0.5, -0.3, 0.5, ["stable"]),
        ("rage", 0.9, -0.8, 0.9, 0.7, ["route_flap_burst"]),
        ("fear", 0.8, -0.7, 0.8, -0.6, ["anomaly_burst"]),
        ("joy", 0.7, 0.8, 0.2, 0.5, ["resolved"]),
        ("anxiety", 0.6, -0.4, 0.5, -0.3, ["unstable"]),
    ]

    print("\n--- Phrase + Prosody Tests ---\n")

    for emotion, strength, valence, arousal, dominance, tags in test_states:
        phrase = emotion_to_phrase(emotion, strength, tags)
        rate, pitch, volume = vad_to_prosody(valence, arousal, dominance)

        print(f"Emotion: {emotion} (s={strength:.1f})")
        print(f"  VAD: V={valence:+.1f}, A={arousal:+.1f}, D={dominance:+.1f}")
        print(f"  Prosody: rate={rate}, pitch={pitch}, vol={volume:.1f}")
        print(f"  Phrase: {phrase}")
        print()

    # Async demo
    print("--- Speaking (mock) ---\n")

    async def run_demo():
        configure_tts(backend=TTSBackend.MOCK, cooldown=0.1)

        @dataclass
        class MockState:
            emotion: str
            strength: float
            valence: float
            arousal: float
            dominance: float
            tags: list

        for emotion, strength, valence, arousal, dominance, tags in test_states[:3]:
            state = MockState(emotion, strength, valence, arousal, dominance, tags)
            await speak_from_state(state)

    asyncio.run(run_demo())

    print("\n" + "=" * 60)


if __name__ == "__main__":
    from dataclasses import dataclass
    demo()
