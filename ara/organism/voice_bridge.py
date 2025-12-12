"""
Voice Bridge - In-Process Connection Between Organism and Voice Daemon
======================================================================

Connects the OrganismRuntime's emotional output to the VoiceDaemon
without requiring actual UART hardware. This enables voice synthesis
driven by the organism's emotional state.

Architecture:
    ┌─────────────────────┐
    │  OrganismRuntime    │
    │  ├─ VADEmotionalMind│──► Emotion State
    │  └─ ReflexiveProbe  │──► Concept Tags
    └─────────────────────┘
              │
              ▼ VoiceBridge (in-process queue)
              │
    ┌─────────────────────┐
    │  VoiceDaemon        │
    │  ├─ TTS Engine      │──► Audio Output
    │  └─ Prosody Engine  │
    └─────────────────────┘

Usage:
    from ara.organism.runtime import OrganismRuntime
    from ara.organism.voice_bridge import VoiceBridge

    runtime = OrganismRuntime(config)
    bridge = VoiceBridge(runtime)

    # Start both
    runtime.start()
    bridge.start()

    # Process telemetry - voice output is automatic
    while True:
        telemetry = get_telemetry()
        state = runtime.step(telemetry)
        # Voice synthesizes emotion automatically

    bridge.stop()
    runtime.stop()
"""

from __future__ import annotations

import logging
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable
from enum import Enum

logger = logging.getLogger("ara.organism.voice_bridge")

# Try to import organism components
try:
    from ara.organism.runtime import OrganismRuntime, OrganismState
    from ara.organism.vad_mind import VADState, EmotionArchetype
    ORGANISM_AVAILABLE = True
except ImportError:
    ORGANISM_AVAILABLE = False
    OrganismRuntime = None
    OrganismState = None
    VADState = None
    EmotionArchetype = None

# Try to import multimodal avatar
try:
    from ara.avatar.multimodal_integration import MultimodalAvatar, get_multimodal_avatar
    MULTIMODAL_AVAILABLE = True
except ImportError:
    MULTIMODAL_AVAILABLE = False
    MultimodalAvatar = None


# =============================================================================
# Emotion to Speech Phrases
# =============================================================================

EMOTION_PHRASES: Dict[str, str] = {
    "joy": "I feel good about this.",
    "excitement": "This is exciting!",
    "elation": "This is wonderful!",
    "serenity": "All is calm.",
    "contentment": "Things are going well.",
    "calm": "Everything is under control.",
    "fear": "I sense danger.",
    "anger": "This is frustrating.",
    "rage": "This is unacceptable!",
    "anxiety": "I'm concerned about this.",
    "sadness": "This is unfortunate.",
    "depression": "I feel overwhelmed.",
    "boredom": "Nothing interesting happening.",
    "contempt": "This is beneath us.",
    "pride": "We've done well.",
    "submission": "I defer to your judgment.",
    "helplessness": "I don't know what to do.",
    "trust": "I have confidence in this.",
    "vigilance": "Staying alert.",
    "disgust": "This is unpleasant.",
    "surprise": "That was unexpected!",
    "anticipation": "Something is coming.",
    "neutral": "Status normal.",
    "overwhelmed": "Too much happening at once.",
}

CLASSIFICATION_PHRASES: Dict[str, str] = {
    "NORMAL": "Operations normal.",
    "ANOMALY": "Anomaly detected.",
}


# =============================================================================
# Voice Message Queue
# =============================================================================

@dataclass
class VoiceMessage:
    """Message to be spoken."""
    text: str
    emotion: str = "neutral"
    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.5
    priority: int = 1  # Higher = more urgent


class VoiceQueue:
    """Thread-safe queue for voice messages with rate limiting."""

    def __init__(self, min_interval: float = 2.0, max_size: int = 10):
        self._queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=max_size)
        self._min_interval = min_interval
        self._last_speak_time = 0.0
        self._lock = threading.Lock()

    def put(self, msg: VoiceMessage) -> bool:
        """Add message to queue (returns False if full)."""
        try:
            # Priority queue uses (priority, item) tuples - negate for max-heap behavior
            self._queue.put_nowait((-msg.priority, time.time(), msg))
            return True
        except queue.Full:
            return False

    def get(self, block: bool = True, timeout: float = 1.0) -> Optional[VoiceMessage]:
        """Get next message, respecting rate limit."""
        with self._lock:
            now = time.time()
            if now - self._last_speak_time < self._min_interval:
                return None

        try:
            _, _, msg = self._queue.get(block=block, timeout=timeout)
            with self._lock:
                self._last_speak_time = time.time()
            return msg
        except queue.Empty:
            return None

    def clear(self) -> None:
        """Clear all pending messages."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break


# =============================================================================
# TTS Backend Adapters
# =============================================================================

class TTSBackend:
    """Abstract TTS backend."""

    def speak(self, text: str, emotion: str, valence: float, arousal: float, dominance: float) -> None:
        raise NotImplementedError


class EspeakBackend(TTSBackend):
    """espeak/espeak-ng backend."""

    def __init__(self):
        # Check for espeak
        self.cmd = None
        for cmd in ["espeak-ng", "espeak"]:
            try:
                result = subprocess.run([cmd, "--version"], capture_output=True)
                if result.returncode == 0:
                    self.cmd = cmd
                    break
            except FileNotFoundError:
                continue

        if not self.cmd:
            logger.warning("espeak not found")

    def _compute_prosody(self, valence: float, arousal: float, dominance: float) -> tuple:
        """Compute rate and pitch from VAD."""
        base_rate = 175
        base_pitch = 50

        # Rate: higher arousal → faster
        rate = int(base_rate * (1 + arousal * 0.3))
        rate = max(100, min(300, rate))

        # Pitch: higher dominance → lower
        pitch = int(base_pitch - dominance * 20 + arousal * 10)
        pitch = max(10, min(90, pitch))

        return rate, pitch

    def speak(self, text: str, emotion: str, valence: float, arousal: float, dominance: float) -> None:
        if not self.cmd:
            print(f"[VOICE] {text}")
            return

        rate, pitch = self._compute_prosody(valence, arousal, dominance)

        try:
            subprocess.run(
                [self.cmd, "-s", str(rate), "-p", str(pitch), text],
                capture_output=True,
                timeout=10
            )
        except Exception as e:
            logger.error(f"espeak error: {e}")


class MultimodalBackend(TTSBackend):
    """Use MultimodalAvatar for TTS."""

    def __init__(self, avatar: MultimodalAvatar):
        self.avatar = avatar

    def speak(self, text: str, emotion: str, valence: float, arousal: float, dominance: float) -> None:
        self.avatar.speak(
            text,
            emotion=emotion,
            valence=valence,
            arousal=arousal,
            dominance=dominance
        )


class MockBackend(TTSBackend):
    """Mock backend - just prints."""

    def speak(self, text: str, emotion: str, valence: float, arousal: float, dominance: float) -> None:
        print(f"[VOICE {emotion}] {text}")


# =============================================================================
# Voice Bridge
# =============================================================================

@dataclass
class VoiceBridgeConfig:
    """Configuration for VoiceBridge."""

    # Rate limiting
    min_speak_interval: float = 2.0

    # What to speak
    speak_emotions: bool = True
    speak_classifications: bool = True
    speak_novel_concepts: bool = True

    # Thresholds
    min_emotion_strength: float = 0.5
    emotion_change_threshold: float = 0.3

    # Backend
    use_multimodal: bool = True  # Use MultimodalAvatar if available
    use_espeak: bool = True      # Fall back to espeak


class VoiceBridge:
    """
    Bridge between OrganismRuntime and voice synthesis.

    Monitors organism state and speaks emotional changes.
    """

    def __init__(
        self,
        organism: Optional[OrganismRuntime] = None,
        config: Optional[VoiceBridgeConfig] = None,
        avatar: Optional[MultimodalAvatar] = None,
    ):
        self.organism = organism
        self.config = config or VoiceBridgeConfig()

        # Voice queue
        self._queue = VoiceQueue(min_interval=self.config.min_speak_interval)

        # TTS backend
        self._backend: Optional[TTSBackend] = None
        self._select_backend(avatar)

        # State tracking
        self._last_emotion: Optional[str] = None
        self._last_classification: Optional[str] = None

        # Control
        self._running = False
        self._poll_thread: Optional[threading.Thread] = None
        self._speak_thread: Optional[threading.Thread] = None

    def _select_backend(self, avatar: Optional[MultimodalAvatar]) -> None:
        """Select best available TTS backend."""
        if avatar and self.config.use_multimodal:
            self._backend = MultimodalBackend(avatar)
            logger.info("Using MultimodalAvatar for voice")
        elif MULTIMODAL_AVAILABLE and self.config.use_multimodal:
            try:
                self._backend = MultimodalBackend(get_multimodal_avatar())
                logger.info("Using MultimodalAvatar for voice")
            except Exception:
                pass

        if not self._backend and self.config.use_espeak:
            backend = EspeakBackend()
            if backend.cmd:
                self._backend = backend
                logger.info(f"Using {backend.cmd} for voice")

        if not self._backend:
            self._backend = MockBackend()
            logger.warning("Using mock voice backend (print only)")

    def attach_organism(self, organism: OrganismRuntime) -> None:
        """Attach to an organism runtime."""
        self.organism = organism

    def start(self) -> None:
        """Start the voice bridge."""
        if self._running:
            return

        self._running = True

        # Start poll thread (monitors organism)
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()

        # Start speak thread (processes queue)
        self._speak_thread = threading.Thread(target=self._speak_loop, daemon=True)
        self._speak_thread.start()

        logger.info("VoiceBridge started")

    def stop(self) -> None:
        """Stop the voice bridge."""
        self._running = False

        if self._poll_thread:
            self._poll_thread.join(timeout=1.0)
        if self._speak_thread:
            self._speak_thread.join(timeout=1.0)

        self._queue.clear()
        logger.info("VoiceBridge stopped")

    def _poll_loop(self) -> None:
        """Poll organism state and queue voice messages."""
        poll_hz = 10.0
        period = 1.0 / poll_hz

        while self._running:
            if self.organism:
                try:
                    state = self.organism.state
                    self._process_state(state)
                except Exception as e:
                    logger.debug(f"Poll error: {e}")

            time.sleep(period)

    def _process_state(self, state: OrganismState) -> None:
        """Process organism state and generate voice messages."""
        if state is None:
            return

        # Emotion changes
        if self.config.speak_emotions and state.emotion:
            emotion_name = state.emotion.archetype.value
            strength = state.emotion.strength

            # Check if emotion changed significantly
            if emotion_name != self._last_emotion:
                if strength >= self.config.min_emotion_strength:
                    phrase = EMOTION_PHRASES.get(emotion_name, f"Feeling {emotion_name}.")

                    # Amplify phrase at high strength
                    if strength > 0.8:
                        phrase = phrase.upper()

                    msg = VoiceMessage(
                        text=phrase,
                        emotion=emotion_name,
                        valence=state.emotion.valence,
                        arousal=state.emotion.arousal,
                        dominance=state.emotion.dominance,
                        priority=int(strength * 10)
                    )
                    self._queue.put(msg)
                    self._last_emotion = emotion_name

        # Classification changes
        if self.config.speak_classifications:
            if state.classification != self._last_classification:
                # Only speak anomalies or state changes
                if state.classification == "ANOMALY":
                    phrase = CLASSIFICATION_PHRASES.get("ANOMALY", "Anomaly detected.")
                    msg = VoiceMessage(
                        text=phrase,
                        emotion="anxiety",
                        arousal=0.7,
                        priority=8
                    )
                    self._queue.put(msg)
                elif self._last_classification == "ANOMALY":
                    # Returning to normal
                    phrase = CLASSIFICATION_PHRASES.get("NORMAL", "Normal operations.")
                    msg = VoiceMessage(
                        text=phrase,
                        emotion="calm",
                        arousal=-0.3,
                        priority=5
                    )
                    self._queue.put(msg)

                self._last_classification = state.classification

        # Novel concepts
        if self.config.speak_novel_concepts and state.is_novel:
            tag = state.concept_tag.replace("_", " ")
            phrase = f"New pattern detected: {tag}."
            msg = VoiceMessage(
                text=phrase,
                emotion="surprise",
                arousal=0.5,
                priority=6
            )
            self._queue.put(msg)

    def _speak_loop(self) -> None:
        """Process voice queue and speak."""
        while self._running:
            msg = self._queue.get(timeout=0.5)
            if msg:
                try:
                    self._backend.speak(
                        msg.text,
                        msg.emotion,
                        msg.valence,
                        msg.arousal,
                        msg.dominance
                    )
                except Exception as e:
                    logger.error(f"Speak error: {e}")

    def speak_now(
        self,
        text: str,
        emotion: str = "neutral",
        valence: float = 0.0,
        arousal: float = 0.0,
        dominance: float = 0.5,
    ) -> None:
        """Immediately speak (bypasses organism state)."""
        msg = VoiceMessage(
            text=text,
            emotion=emotion,
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            priority=10  # High priority
        )
        self._queue.put(msg)


# =============================================================================
# Singleton accessor
# =============================================================================

_bridge_instance: Optional[VoiceBridge] = None


def get_voice_bridge() -> VoiceBridge:
    """Get or create the singleton VoiceBridge."""
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = VoiceBridge()
    return _bridge_instance


def start_voice_bridge(organism: Optional[OrganismRuntime] = None) -> VoiceBridge:
    """Start the voice bridge (creates if needed)."""
    bridge = get_voice_bridge()
    if organism:
        bridge.attach_organism(organism)
    bridge.start()
    return bridge


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demo the voice bridge with mock organism."""
    print("=" * 60)
    print("Voice Bridge Demo")
    print("=" * 60)

    # Create bridge with mock backend
    config = VoiceBridgeConfig(
        min_speak_interval=1.0,
        use_multimodal=False,
        use_espeak=True,
    )
    bridge = VoiceBridge(config=config)
    bridge.start()

    # Simulate emotion changes
    print("\nSimulating emotion transitions...\n")

    test_messages = [
        ("Status normal.", "neutral", 0.0, -0.3, 0.5),
        ("I'm feeling calm.", "calm", 0.3, -0.4, 0.6),
        ("Something's happening!", "anxiety", -0.3, 0.6, -0.2),
        ("ANOMALY DETECTED!", "fear", -0.7, 0.8, -0.5),
        ("Returning to normal.", "calm", 0.2, -0.3, 0.5),
    ]

    for text, emotion, val, aro, dom in test_messages:
        bridge.speak_now(text, emotion, val, aro, dom)
        time.sleep(2.5)

    bridge.stop()
    print("\nDemo complete!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo()
