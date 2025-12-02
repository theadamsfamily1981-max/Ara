"""
Audio Input/Output for Ara Avatar

Handles microphone recording and audio playback.
Uses sounddevice for cross-platform audio.

Install:
    pip install sounddevice numpy scipy
"""

import logging
import time
import threading
from typing import Optional
from pathlib import Path

logger = logging.getLogger("ara.avatar.audio")

# Try to import audio libraries
AUDIO_AVAILABLE = False
try:
    import numpy as np
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    logger.warning("sounddevice not available. Install with: pip install sounddevice numpy")
    np = None
    sd = None


# Audio settings
SAMPLE_RATE = 16000  # 16kHz for ASR compatibility
CHANNELS = 1         # Mono
DTYPE = 'int16'


class AudioRecorder:
    """
    Voice Activity Detection (VAD) based audio recorder.

    Records audio from microphone until silence is detected.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        silence_threshold: float = 500.0,
        silence_duration: float = 1.5,
        max_duration: float = 30.0
    ):
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.max_duration = max_duration
        self._recording = False
        self._audio_data = []

    def record(self) -> Optional[bytes]:
        """
        Record audio until silence or max duration.

        Returns:
            Raw audio bytes (16-bit PCM) or None if recording failed
        """
        if not AUDIO_AVAILABLE:
            logger.error("Audio not available")
            return None

        self._audio_data = []
        self._recording = True
        silence_start = None
        recording_start = time.time()

        logger.info("Recording started... (speak now)")

        def callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"Audio status: {status}")

            # Store audio
            self._audio_data.append(indata.copy())

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=CHANNELS,
                dtype=DTYPE,
                callback=callback,
                blocksize=int(self.sample_rate * 0.1)  # 100ms blocks
            ):
                while self._recording:
                    time.sleep(0.1)
                    elapsed = time.time() - recording_start

                    # Check max duration
                    if elapsed > self.max_duration:
                        logger.info("Max duration reached")
                        break

                    # Check for silence (VAD)
                    if self._audio_data:
                        recent = np.concatenate(self._audio_data[-5:])  # Last 500ms
                        rms = np.sqrt(np.mean(recent.astype(np.float32) ** 2))

                        if rms < self.silence_threshold:
                            if silence_start is None:
                                silence_start = time.time()
                            elif time.time() - silence_start > self.silence_duration:
                                logger.info("Silence detected, stopping")
                                break
                        else:
                            silence_start = None

        except Exception as e:
            logger.error(f"Recording error: {e}")
            return None

        self._recording = False

        if not self._audio_data:
            return None

        # Concatenate all audio
        audio = np.concatenate(self._audio_data)
        logger.info(f"Recorded {len(audio) / self.sample_rate:.1f}s of audio")

        return audio.tobytes()

    def stop(self):
        """Stop recording."""
        self._recording = False


def record_utterance(
    silence_threshold: float = 500.0,
    silence_duration: float = 1.5,
    max_duration: float = 30.0
) -> Optional[bytes]:
    """
    Record a single utterance from the microphone.

    Uses Voice Activity Detection to stop when user stops speaking.

    Args:
        silence_threshold: RMS threshold for silence detection
        silence_duration: How long silence before stopping (seconds)
        max_duration: Maximum recording duration (seconds)

    Returns:
        Raw audio bytes (16-bit PCM, 16kHz mono) or None
    """
    recorder = AudioRecorder(
        silence_threshold=silence_threshold,
        silence_duration=silence_duration,
        max_duration=max_duration
    )
    return recorder.record()


def play_audio(audio_data: bytes, sample_rate: int = SAMPLE_RATE) -> bool:
    """
    Play audio through speakers.

    Args:
        audio_data: Raw audio bytes (16-bit PCM)
        sample_rate: Sample rate of the audio

    Returns:
        True if playback succeeded
    """
    if not AUDIO_AVAILABLE:
        logger.error("Audio not available")
        return False

    try:
        # Convert bytes to numpy array
        audio = np.frombuffer(audio_data, dtype=np.int16)

        logger.info(f"Playing {len(audio) / sample_rate:.1f}s of audio")
        sd.play(audio, samplerate=sample_rate)
        sd.wait()  # Block until done

        return True

    except Exception as e:
        logger.error(f"Playback error: {e}")
        return False


def save_audio(audio_data: bytes, path: str, sample_rate: int = SAMPLE_RATE) -> bool:
    """
    Save audio to WAV file.

    Args:
        audio_data: Raw audio bytes
        path: Output file path
        sample_rate: Sample rate

    Returns:
        True if save succeeded
    """
    try:
        from scipy.io import wavfile
        audio = np.frombuffer(audio_data, dtype=np.int16)
        wavfile.write(path, sample_rate, audio)
        logger.info(f"Saved audio to {path}")
        return True
    except Exception as e:
        logger.error(f"Save error: {e}")
        return False


def load_audio(path: str) -> Optional[tuple]:
    """
    Load audio from WAV file.

    Returns:
        Tuple of (audio_bytes, sample_rate) or None
    """
    try:
        from scipy.io import wavfile
        sample_rate, audio = wavfile.read(path)
        return audio.tobytes(), sample_rate
    except Exception as e:
        logger.error(f"Load error: {e}")
        return None


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing audio recording...")
    print("Speak something, then pause for 1.5 seconds to stop.")

    audio = record_utterance()

    if audio:
        print(f"Recorded {len(audio)} bytes")
        print("Playing back...")
        play_audio(audio)
        print("Done!")
    else:
        print("Recording failed")
