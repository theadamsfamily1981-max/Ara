#!/usr/bin/env python3
"""
Emotional Voice Daemon - UART to Voice Synthesis
=================================================

Listens to UART emissions from the organism runtime and
converts emotional states to speech via TTS.

Protocol (from organism_runtime):
    EMO:<archetype>:<strength>:<dominance>:<valence>:<arousal>
    TAG:<concept>:<similarity>:<novel>
    CLS:<label>:<confidence>

Voice modulation:
- Higher arousal → faster speech rate
- Higher dominance → lower pitch, more assertive
- Lower valence → slower, softer voice

Usage:
    python voice_daemon.py --port /dev/ttyUSB0 --baud 115200

Or programmatically:
    daemon = VoiceDaemon(port="/dev/ttyUSB0")
    daemon.run()
"""

from __future__ import annotations
import subprocess
import logging
import time
import re
from dataclasses import dataclass
from typing import Optional, Dict, Callable
from enum import Enum

logger = logging.getLogger(__name__)

# Optional imports
try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False
    serial = None


class TTSBackend(Enum):
    """Available TTS backends."""
    ESPEAK = "espeak"
    PIPER = "piper"
    SAY = "say"  # macOS
    MOCK = "mock"


@dataclass
class VoiceConfig:
    """Voice configuration."""
    port: str = "/dev/ttyUSB0"
    baud: int = 115200
    tts_backend: TTSBackend = TTSBackend.ESPEAK
    base_rate: int = 175       # Words per minute
    base_pitch: int = 50       # Pitch (0-99)
    speak_emotions: bool = True
    speak_concepts: bool = False
    speak_classifications: bool = True
    min_speak_interval: float = 2.0  # Seconds between utterances
    verbose: bool = True


# Emotion to voice mapping
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


class VoiceDaemon:
    """
    Daemon that converts UART emotion packets to speech.

    The daemon:
    1. Listens to UART for EMO/TAG/CLS packets
    2. Maps emotions to prosody (rate, pitch)
    3. Calls TTS backend to speak
    """

    def __init__(self, config: Optional[VoiceConfig] = None):
        self.cfg = config or VoiceConfig()
        self.ser: Optional[any] = None
        self.last_speak_time: float = 0.0
        self._running = False

        # Track last state to avoid repetition
        self._last_emotion: Optional[str] = None
        self._last_classification: Optional[str] = None

    def open_uart(self) -> bool:
        """Open UART connection."""
        if not HAS_SERIAL:
            logger.error("pyserial not installed")
            return False

        try:
            self.ser = serial.Serial(
                self.cfg.port,
                baudrate=self.cfg.baud,
                timeout=0.5
            )
            logger.info(f"UART opened: {self.cfg.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to open UART: {e}")
            return False

    def close_uart(self) -> None:
        """Close UART connection."""
        if self.ser is not None:
            self.ser.close()
            self.ser = None

    def _compute_prosody(
        self,
        valence: float,
        arousal: float,
        dominance: float,
    ) -> tuple[int, int]:
        """
        Compute speech rate and pitch from VAD coordinates.

        Returns:
            (rate, pitch) for TTS
        """
        # Rate: higher arousal → faster
        rate = int(self.cfg.base_rate * (1 + arousal * 0.3))
        rate = max(100, min(300, rate))

        # Pitch: higher dominance → lower, more authoritative
        # Higher arousal → higher pitch variation
        pitch = int(self.cfg.base_pitch - dominance * 20 + arousal * 10)
        pitch = max(10, min(90, pitch))

        return rate, pitch

    def _speak_espeak(self, text: str, rate: int, pitch: int) -> None:
        """Speak using espeak."""
        try:
            subprocess.run(
                ["espeak", "-s", str(rate), "-p", str(pitch), text],
                capture_output=True,
                timeout=10
            )
        except FileNotFoundError:
            logger.warning("espeak not found, falling back to print")
            print(f"[VOICE] {text}")
        except subprocess.TimeoutExpired:
            logger.warning("espeak timed out")

    def _speak_piper(self, text: str, rate: int, pitch: int) -> None:
        """Speak using Piper TTS."""
        # Piper doesn't support rate/pitch directly via CLI
        # Would need to use the Python API
        try:
            subprocess.run(
                ["piper", "--output-file", "-", text],
                capture_output=True,
                timeout=10
            )
        except FileNotFoundError:
            logger.warning("piper not found, falling back to espeak")
            self._speak_espeak(text, rate, pitch)

    def _speak_say(self, text: str, rate: int, pitch: int) -> None:
        """Speak using macOS say command."""
        try:
            subprocess.run(
                ["say", "-r", str(rate), text],
                capture_output=True,
                timeout=10
            )
        except FileNotFoundError:
            logger.warning("say not found (not macOS?)")
            print(f"[VOICE] {text}")

    def _speak_mock(self, text: str, rate: int, pitch: int) -> None:
        """Mock TTS - just print."""
        print(f"[VOICE r={rate} p={pitch}] {text}")

    def speak(self, text: str, rate: int, pitch: int) -> None:
        """Speak text with prosody."""
        now = time.time()
        if now - self.last_speak_time < self.cfg.min_speak_interval:
            return

        backend = self.cfg.tts_backend
        if backend == TTSBackend.ESPEAK:
            self._speak_espeak(text, rate, pitch)
        elif backend == TTSBackend.PIPER:
            self._speak_piper(text, rate, pitch)
        elif backend == TTSBackend.SAY:
            self._speak_say(text, rate, pitch)
        else:
            self._speak_mock(text, rate, pitch)

        self.last_speak_time = now

    def handle_emotion(
        self,
        archetype: str,
        strength: float,
        dominance: float,
        valence: float,
        arousal: float,
    ) -> None:
        """Handle EMO packet."""
        if not self.cfg.speak_emotions:
            return

        # Avoid repetition
        if archetype == self._last_emotion and strength < 0.7:
            return
        self._last_emotion = archetype

        # Get phrase
        phrase = EMOTION_PHRASES.get(archetype, f"Feeling {archetype}.")

        # Modulate by strength
        if strength > 0.8:
            phrase = phrase.upper()

        # Compute prosody
        rate, pitch = self._compute_prosody(valence, arousal, dominance)

        if self.cfg.verbose:
            logger.info(f"Emotion: {archetype} -> '{phrase}' (r={rate}, p={pitch})")

        self.speak(phrase, rate, pitch)

    def handle_concept(
        self,
        tag: str,
        similarity: float,
        is_novel: bool,
    ) -> None:
        """Handle TAG packet."""
        if not self.cfg.speak_concepts:
            return

        if is_novel:
            phrase = f"New pattern detected: {tag.replace('_', ' ')}."
            self.speak(phrase, self.cfg.base_rate, self.cfg.base_pitch)

    def handle_classification(
        self,
        label: str,
        confidence: float,
    ) -> None:
        """Handle CLS packet."""
        if not self.cfg.speak_classifications:
            return

        # Only speak anomalies or on state change
        if label == self._last_classification and label == "NORMAL":
            return
        self._last_classification = label

        phrase = CLASSIFICATION_PHRASES.get(label, f"Classification: {label}.")

        if label == "ANOMALY":
            # Speak with urgency
            self.speak(phrase, self.cfg.base_rate + 30, self.cfg.base_pitch + 10)
        else:
            self.speak(phrase, self.cfg.base_rate, self.cfg.base_pitch)

    def parse_line(self, line: str) -> None:
        """Parse and handle a UART line."""
        line = line.strip()
        if not line:
            return

        if line.startswith("EMO:"):
            # EMO:<archetype>:<strength>:<dominance>:<valence>:<arousal>
            parts = line[4:].split(":")
            if len(parts) >= 5:
                archetype = parts[0]
                strength = float(parts[1])
                dominance = float(parts[2])
                valence = float(parts[3])
                arousal = float(parts[4])
                self.handle_emotion(archetype, strength, dominance, valence, arousal)

        elif line.startswith("TAG:"):
            # TAG:<tag>:<similarity>:<novel>
            parts = line[4:].split(":")
            if len(parts) >= 3:
                tag = parts[0]
                similarity = float(parts[1])
                is_novel = parts[2] == "1"
                self.handle_concept(tag, similarity, is_novel)

        elif line.startswith("CLS:"):
            # CLS:<label>:<confidence>
            parts = line[4:].split(":")
            if len(parts) >= 2:
                label = parts[0]
                confidence = float(parts[1])
                self.handle_classification(label, confidence)

    def run(self) -> None:
        """Run the daemon loop."""
        if not self.open_uart():
            logger.error("Cannot start: UART failed")
            return

        self._running = True
        logger.info("Voice daemon started")

        try:
            while self._running:
                if self.ser is None:
                    break

                line = self.ser.readline().decode("utf-8", errors="ignore")
                if line:
                    self.parse_line(line)

        except KeyboardInterrupt:
            logger.info("Interrupted")

        finally:
            self.close_uart()
            logger.info("Voice daemon stopped")

    def stop(self) -> None:
        """Stop the daemon."""
        self._running = False


class MockVoiceDaemon(VoiceDaemon):
    """Voice daemon that reads from stdin instead of UART (for testing)."""

    def run(self) -> None:
        """Run with stdin input."""
        self._running = True
        print("Mock Voice Daemon - enter UART lines (Ctrl+C to exit)")
        print("Format: EMO:joy:0.8:0.5:0.7:0.6")
        print()

        try:
            while self._running:
                line = input("> ")
                self.parse_line(line)

        except (KeyboardInterrupt, EOFError):
            pass

        print("\nDaemon stopped")


def demo():
    """Run a demo of the voice daemon."""
    print("=" * 60)
    print("Emotional Voice Daemon Demo")
    print("=" * 60)

    cfg = VoiceConfig(tts_backend=TTSBackend.MOCK)
    daemon = VoiceDaemon(cfg)

    # Simulate some UART packets
    test_lines = [
        "EMO:calm:0.6:0.5:0.5:-0.4",
        "CLS:NORMAL:0.85",
        "EMO:anxiety:0.7:-0.4:-0.5:0.7",
        "TAG:emergent_1:0.35:1",
        "CLS:ANOMALY:0.92",
        "EMO:fear:0.8:-0.7:-0.7:0.8",
        "EMO:calm:0.5:0.4:0.6:-0.3",
        "CLS:NORMAL:0.78",
    ]

    print("\n--- Simulating UART packets ---\n")

    for line in test_lines:
        print(f"UART: {line}")
        daemon.parse_line(line)
        time.sleep(0.5)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Emotional Voice Daemon")
    parser.add_argument("--port", default="/dev/ttyUSB0", help="UART port")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate")
    parser.add_argument("--backend", choices=["espeak", "piper", "say", "mock"],
                        default="mock", help="TTS backend")
    parser.add_argument("--demo", action="store_true", help="Run demo mode")
    parser.add_argument("--stdin", action="store_true", help="Read from stdin")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.demo:
        demo()
    elif args.stdin:
        cfg = VoiceConfig(
            tts_backend=TTSBackend[args.backend.upper()]
        )
        daemon = MockVoiceDaemon(cfg)
        daemon.run()
    else:
        cfg = VoiceConfig(
            port=args.port,
            baud=args.baud,
            tts_backend=TTSBackend[args.backend.upper()]
        )
        daemon = VoiceDaemon(cfg)
        daemon.run()
