"""
Multimodal Integration Layer for Ara Avatar
=============================================

Wires together the disparate multimodal components:
- TTS synthesis (Coqui XTTS-v2 / pyttsx3 / espeak)
- Speech recognition (Google Speech API / sounddevice)
- Vision (Ollama llava for webcam analysis)
- Voice Daemon (emotional prosody from organism)

This creates a unified MultimodalAvatar that can:
- Listen to user speech
- See through webcam
- Speak with emotional prosody
- Route emotional state from organism runtime

Usage:
    from ara.avatar.multimodal_integration import MultimodalAvatar

    avatar = MultimodalAvatar()
    await avatar.initialize()

    # Voice input
    text = avatar.listen()

    # Voice output with emotion
    avatar.speak("Hello!", emotion="joy", arousal=0.7)

    # Vision query
    description = avatar.see("What do you see?")
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable, Tuple
from enum import Enum

import numpy as np

logger = logging.getLogger("ara.avatar.multimodal")

# =============================================================================
# Optional Imports
# =============================================================================

# TTS Backends
COQUI_AVAILABLE = False
PYTTSX3_AVAILABLE = False

try:
    from TTS.api import TTS as CoquiTTS
    import torch
    import sounddevice as sd
    COQUI_AVAILABLE = True
    logger.info("Coqui TTS available")
except ImportError:
    pass

if not COQUI_AVAILABLE:
    try:
        import pyttsx3
        PYTTSX3_AVAILABLE = True
        logger.info("pyttsx3 available (fallback)")
    except ImportError:
        pass

# Speech Recognition
SR_AVAILABLE = False
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
    logger.info("Speech recognition available")
except ImportError:
    pass

# Vision
VISION_AVAILABLE = False
try:
    import cv2
    from PIL import Image
    import ollama
    VISION_AVAILABLE = True
    logger.info("Vision (ollama + opencv) available")
except ImportError:
    pass

# Sounddevice for audio
AUDIO_AVAILABLE = False
try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    pass


# =============================================================================
# Emotion to Prosody Mapping
# =============================================================================

@dataclass
class VoiceProsody:
    """Voice prosody parameters."""
    rate: int = 175      # Words per minute
    pitch: int = 50      # 0-99 scale
    volume: float = 0.9  # 0.0-1.0


# Map emotional archetypes to prosody adjustments
EMOTION_PROSODY: Dict[str, VoiceProsody] = {
    "joy": VoiceProsody(rate=190, pitch=60, volume=0.95),
    "excitement": VoiceProsody(rate=210, pitch=65, volume=1.0),
    "serenity": VoiceProsody(rate=150, pitch=45, volume=0.8),
    "calm": VoiceProsody(rate=160, pitch=50, volume=0.85),
    "fear": VoiceProsody(rate=200, pitch=55, volume=0.85),
    "anger": VoiceProsody(rate=180, pitch=35, volume=1.0),
    "sadness": VoiceProsody(rate=140, pitch=40, volume=0.7),
    "anxiety": VoiceProsody(rate=195, pitch=58, volume=0.9),
    "contempt": VoiceProsody(rate=155, pitch=30, volume=0.85),
    "neutral": VoiceProsody(rate=175, pitch=50, volume=0.9),
}


def compute_prosody(
    emotion: str = "neutral",
    valence: float = 0.0,
    arousal: float = 0.0,
    dominance: float = 0.5,
) -> VoiceProsody:
    """
    Compute voice prosody from emotion and VAD coordinates.

    Args:
        emotion: Emotional archetype name
        valence: -1.0 (negative) to 1.0 (positive)
        arousal: -1.0 (calm) to 1.0 (excited)
        dominance: -1.0 (submissive) to 1.0 (dominant)

    Returns:
        VoiceProsody with rate, pitch, volume
    """
    # Start with emotion-based prosody
    base = EMOTION_PROSODY.get(emotion, EMOTION_PROSODY["neutral"])

    # Modulate by VAD coordinates
    # Higher arousal → faster rate
    rate = int(base.rate * (1 + arousal * 0.2))
    rate = max(100, min(250, rate))

    # Higher dominance → lower pitch (more authoritative)
    pitch = int(base.pitch - dominance * 15)
    pitch = max(20, min(80, pitch))

    # Valence affects volume slightly
    volume = base.volume + valence * 0.1
    volume = max(0.5, min(1.0, volume))

    return VoiceProsody(rate=rate, pitch=pitch, volume=volume)


# =============================================================================
# TTS Engine Interface
# =============================================================================

class TTSEngine:
    """Abstract TTS engine interface."""

    def speak(self, text: str, prosody: VoiceProsody) -> None:
        """Speak text with given prosody."""
        raise NotImplementedError


class CoquiEngine(TTSEngine):
    """High-quality neural TTS using Coqui XTTS-v2."""

    def __init__(self, voice_reference: Optional[str] = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Coqui XTTS-v2 on {self.device}...")

        # Accept TOS
        os.environ['COQUI_TOS_AGREED'] = '1'

        self.tts = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
        self.voice_reference = voice_reference
        self.sample_rate = 24000

        if voice_reference and os.path.exists(voice_reference):
            logger.info(f"Using voice reference: {voice_reference}")
        else:
            logger.info("No voice reference - using default voice")

    def speak(self, text: str, prosody: VoiceProsody) -> None:
        """Synthesize and play speech."""
        try:
            if self.voice_reference and os.path.exists(self.voice_reference):
                audio = self.tts.tts(
                    text=text,
                    speaker_wav=self.voice_reference,
                    language="en"
                )
            else:
                audio = self.tts.tts(text=text, language="en")

            # Apply rate adjustment by resampling
            # (XTTS doesn't support rate directly, so we speed up/slow down playback)
            rate_factor = prosody.rate / 175.0
            adjusted_rate = int(self.sample_rate * rate_factor)

            audio_np = np.array(audio, dtype=np.float32) * prosody.volume
            sd.play(audio_np, adjusted_rate)
            sd.wait()

        except Exception as e:
            logger.error(f"Coqui TTS error: {e}")


class Pyttsx3Engine(TTSEngine):
    """Fallback TTS using pyttsx3."""

    def __init__(self):
        self.engine = pyttsx3.init()
        self._setup_voice()

    def _setup_voice(self) -> None:
        """Configure for best available voice."""
        voices = self.engine.getProperty('voices')
        preferred = ['samantha', 'zira', 'female', 'eva', 'victoria']

        for pref in preferred:
            for voice in voices:
                if pref in voice.name.lower():
                    self.engine.setProperty('voice', voice.id)
                    logger.info(f"Using voice: {voice.name}")
                    return

    def speak(self, text: str, prosody: VoiceProsody) -> None:
        """Speak text with prosody."""
        self.engine.setProperty('rate', prosody.rate)
        self.engine.setProperty('volume', prosody.volume)
        self.engine.say(text)
        self.engine.runAndWait()


class MockTTSEngine(TTSEngine):
    """Mock TTS for testing (just prints)."""

    def speak(self, text: str, prosody: VoiceProsody) -> None:
        print(f"[VOICE r={prosody.rate} p={prosody.pitch}] {text}")


# =============================================================================
# Speech Recognition
# =============================================================================

class SpeechRecognizer:
    """Voice input using Google Speech API."""

    def __init__(self):
        if not SR_AVAILABLE:
            raise RuntimeError("speech_recognition not installed")

        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        self.mic = sr.Microphone()

        # Calibrate
        logger.info("Calibrating microphone...")
        with self.mic as src:
            self.recognizer.adjust_for_ambient_noise(src, duration=2)
        logger.info("Microphone ready")

    def listen(self, timeout: float = 7.0, phrase_time_limit: float = 12.0) -> Optional[str]:
        """
        Listen for speech and transcribe.

        Returns:
            Transcribed text or None if nothing heard
        """
        try:
            with self.mic as src:
                logger.debug("Listening...")
                audio = self.recognizer.listen(
                    src,
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit
                )

            logger.debug("Processing...")
            text = self.recognizer.recognize_google(audio)
            return text

        except sr.WaitTimeoutError:
            logger.debug("No speech detected")
            return None
        except sr.UnknownValueError:
            logger.debug("Could not understand audio")
            return None
        except sr.RequestError as e:
            logger.error(f"Speech API error: {e}")
            return None


# =============================================================================
# Vision System
# =============================================================================

class VisionSystem:
    """Webcam vision using Ollama llava model."""

    def __init__(self, model: str = "llava"):
        if not VISION_AVAILABLE:
            raise RuntimeError("opencv/ollama not installed")

        self.model = model
        self.camera: Optional[cv2.VideoCapture] = None
        self.frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self, device: int = 0, fps: int = 8) -> bool:
        """Start webcam capture."""
        if self._running:
            return True

        self.camera = cv2.VideoCapture(device)
        if not self.camera.isOpened():
            logger.error("Could not open camera")
            return False

        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, fps)

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, args=(fps,), daemon=True)
        self._thread.start()

        logger.info("Vision started")
        return True

    def stop(self) -> None:
        """Stop webcam capture."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self.camera:
            self.camera.release()
            self.camera = None
        logger.info("Vision stopped")

    def _capture_loop(self, fps: int) -> None:
        """Background frame capture."""
        delay = 1.0 / fps
        while self._running and self.camera and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                with self.frame_lock:
                    self.frame = frame.copy()
            time.sleep(delay)

    def get_frame_base64(self) -> Optional[str]:
        """Get current frame as base64 JPEG."""
        with self.frame_lock:
            if self.frame is None:
                return None
            frame = self.frame.copy()

        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        # Compress
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=70)
        return base64.b64encode(buf.getvalue()).decode()

    def see(self, prompt: str = "What do you see?") -> Optional[str]:
        """
        Query vision model about current view.

        Args:
            prompt: Question to ask about the image

        Returns:
            Model's description or None if no frame
        """
        img_b64 = self.get_frame_base64()
        if not img_b64:
            return None

        try:
            response = ollama.chat(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": prompt,
                    "images": [img_b64]
                }],
                options={"num_predict": 150}
            )
            return response["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Vision error: {e}")
            return None


# =============================================================================
# Unified Multimodal Avatar
# =============================================================================

@dataclass
class MultimodalConfig:
    """Configuration for MultimodalAvatar."""

    # TTS
    voice_reference: Optional[str] = None
    prefer_neural_tts: bool = True

    # Speech recognition
    listen_timeout: float = 7.0
    phrase_time_limit: float = 12.0

    # Vision
    vision_model: str = "llava"
    camera_device: int = 0
    camera_fps: int = 8

    # Emotion defaults
    default_emotion: str = "neutral"


class MultimodalAvatar:
    """
    Unified multimodal avatar interface.

    Combines:
    - TTS with emotional prosody
    - Speech recognition
    - Vision (webcam + LLM)
    - Emotion state from organism

    Usage:
        avatar = MultimodalAvatar()
        await avatar.initialize()

        # Voice loop
        while True:
            text = avatar.listen()
            if text:
                response = process(text)
                avatar.speak(response, emotion="joy")
    """

    def __init__(self, config: Optional[MultimodalConfig] = None):
        self.config = config or MultimodalConfig()

        self.tts: Optional[TTSEngine] = None
        self.recognizer: Optional[SpeechRecognizer] = None
        self.vision: Optional[VisionSystem] = None

        # Emotion state (can be set from organism runtime)
        self._emotion = "neutral"
        self._valence = 0.0
        self._arousal = 0.0
        self._dominance = 0.5

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all multimodal components."""
        logger.info("Initializing MultimodalAvatar...")

        # TTS
        if self.config.prefer_neural_tts and COQUI_AVAILABLE:
            logger.info("Using Coqui XTTS-v2 (neural)")
            self.tts = CoquiEngine(self.config.voice_reference)
        elif PYTTSX3_AVAILABLE:
            logger.info("Using pyttsx3 (fallback)")
            self.tts = Pyttsx3Engine()
        else:
            logger.warning("No TTS available - using mock")
            self.tts = MockTTSEngine()

        # Speech recognition
        if SR_AVAILABLE:
            try:
                self.recognizer = SpeechRecognizer()
            except Exception as e:
                logger.warning(f"Speech recognition init failed: {e}")

        # Vision
        if VISION_AVAILABLE:
            self.vision = VisionSystem(self.config.vision_model)

        self._initialized = True
        logger.info("MultimodalAvatar initialized")

    def set_emotion(
        self,
        emotion: str = "neutral",
        valence: float = 0.0,
        arousal: float = 0.0,
        dominance: float = 0.5,
    ) -> None:
        """
        Set current emotional state (called from organism runtime).

        This affects voice prosody on next speak() call.
        """
        self._emotion = emotion
        self._valence = valence
        self._arousal = arousal
        self._dominance = dominance

    def speak(
        self,
        text: str,
        emotion: Optional[str] = None,
        valence: Optional[float] = None,
        arousal: Optional[float] = None,
        dominance: Optional[float] = None,
    ) -> None:
        """
        Speak text with emotional prosody.

        Args:
            text: Text to speak
            emotion: Override emotion (or use current)
            valence/arousal/dominance: Override VAD (or use current)
        """
        if not self.tts:
            print(f"[Ara]: {text}")
            return

        # Use provided or current emotion
        emo = emotion or self._emotion
        val = valence if valence is not None else self._valence
        aro = arousal if arousal is not None else self._arousal
        dom = dominance if dominance is not None else self._dominance

        # Compute prosody
        prosody = compute_prosody(emo, val, aro, dom)

        logger.debug(f"Speaking with {emo}: {prosody}")
        print(f"\nAra: {text}\n")

        self.tts.speak(text, prosody)

    def listen(
        self,
        timeout: Optional[float] = None,
        phrase_time_limit: Optional[float] = None,
    ) -> Optional[str]:
        """
        Listen for speech and transcribe.

        Returns:
            Transcribed text or None
        """
        if not self.recognizer:
            logger.warning("Speech recognition not available")
            return None

        return self.recognizer.listen(
            timeout=timeout or self.config.listen_timeout,
            phrase_time_limit=phrase_time_limit or self.config.phrase_time_limit,
        )

    def start_vision(self) -> bool:
        """Start webcam capture."""
        if not self.vision:
            logger.warning("Vision not available")
            return False
        return self.vision.start(
            device=self.config.camera_device,
            fps=self.config.camera_fps,
        )

    def stop_vision(self) -> None:
        """Stop webcam capture."""
        if self.vision:
            self.vision.stop()

    def see(self, prompt: str = "What do you see?") -> Optional[str]:
        """
        Query vision about current view.

        Auto-starts vision if not running.
        """
        if not self.vision:
            logger.warning("Vision not available")
            return None

        if not self.vision._running:
            if not self.start_vision():
                return None
            time.sleep(0.5)  # Let camera warm up

        return self.vision.see(prompt)

    def needs_vision(self, text: str) -> bool:
        """Check if query needs vision capabilities."""
        keywords = [
            'look', 'see', 'what is', "what's", 'show', 'camera',
            'in front', 'looking at', 'describe', 'identify', 'watch',
            'visible', 'observe', 'view'
        ]
        text_lower = text.lower()
        return any(kw in text_lower for kw in keywords)

    def close(self) -> None:
        """Clean up resources."""
        self.stop_vision()
        logger.info("MultimodalAvatar closed")


# =============================================================================
# Integration with Organism Runtime
# =============================================================================

class OrganismVoiceBridge:
    """
    Bridge between OrganismRuntime and MultimodalAvatar.

    Receives emotion state from organism and updates avatar prosody.

    Usage:
        from ara.organism.runtime import OrganismRuntime

        runtime = OrganismRuntime()
        avatar = MultimodalAvatar()
        bridge = OrganismVoiceBridge(runtime, avatar)

        # Start bridge (runs in background)
        bridge.start()
    """

    def __init__(
        self,
        organism,  # OrganismRuntime
        avatar: MultimodalAvatar,
        poll_hz: float = 10.0,
    ):
        self.organism = organism
        self.avatar = avatar
        self.poll_hz = poll_hz
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the bridge (background thread)."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info("OrganismVoiceBridge started")

    def stop(self) -> None:
        """Stop the bridge."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        logger.info("OrganismVoiceBridge stopped")

    def _poll_loop(self) -> None:
        """Poll organism state and update avatar."""
        period = 1.0 / self.poll_hz

        while self._running:
            try:
                # Get organism state
                state = self.organism.state

                if state and state.emotion:
                    # Update avatar emotion
                    self.avatar.set_emotion(
                        emotion=state.emotion.archetype.value,
                        valence=state.emotion.valence,
                        arousal=state.emotion.arousal,
                        dominance=state.emotion.dominance,
                    )
            except Exception as e:
                logger.debug(f"Bridge poll error: {e}")

            time.sleep(period)


# =============================================================================
# Singleton accessor
# =============================================================================

_avatar_instance: Optional[MultimodalAvatar] = None


def get_multimodal_avatar() -> MultimodalAvatar:
    """Get or create the singleton MultimodalAvatar."""
    global _avatar_instance
    if _avatar_instance is None:
        _avatar_instance = MultimodalAvatar()
    return _avatar_instance


async def initialize_multimodal() -> MultimodalAvatar:
    """Initialize and return the multimodal avatar."""
    avatar = get_multimodal_avatar()
    if not avatar._initialized:
        await avatar.initialize()
    return avatar


# =============================================================================
# Test / Demo
# =============================================================================

async def demo():
    """Demo the multimodal avatar."""
    print("=" * 60)
    print("Multimodal Avatar Demo")
    print("=" * 60)

    avatar = await initialize_multimodal()

    # Speak with different emotions
    emotions = ["neutral", "joy", "sadness", "excitement", "calm"]
    for emotion in emotions:
        avatar.speak(f"This is how I sound when feeling {emotion}.", emotion=emotion)
        await asyncio.sleep(0.5)

    # Vision test (if available)
    if avatar.vision:
        print("\nStarting vision...")
        avatar.start_vision()
        await asyncio.sleep(1)

        description = avatar.see("Describe what you see briefly.")
        if description:
            print(f"I see: {description}")
            avatar.speak(f"I see: {description[:100]}", emotion="interest")

        avatar.stop_vision()

    # Listening test
    if avatar.recognizer:
        print("\nSay something (7 second timeout)...")
        avatar.speak("I'm listening. Say something.")

        text = avatar.listen()
        if text:
            print(f"You said: {text}")
            avatar.speak(f"I heard you say: {text}", emotion="acknowledgment")
        else:
            avatar.speak("I didn't hear anything.", emotion="neutral")

    avatar.close()
    print("\nDemo complete!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(demo())
