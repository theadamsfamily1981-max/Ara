"""
Automatic Speech Recognition for Ara Avatar

Converts audio to text. Supports multiple backends:
- Whisper (local, via faster-whisper or openai-whisper)
- Vosk (local, lightweight)
- Google Speech (cloud)

Install (pick one):
    pip install faster-whisper  # Recommended
    pip install vosk
    pip install SpeechRecognition  # For Google
"""

import logging
from typing import Optional
from pathlib import Path
import tempfile

logger = logging.getLogger("ara.avatar.asr")

# Try to import ASR backends
WHISPER_AVAILABLE = False
VOSK_AVAILABLE = False
SR_AVAILABLE = False

try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
    logger.info("faster-whisper available")
except ImportError:
    pass

try:
    import vosk
    VOSK_AVAILABLE = True
    logger.info("vosk available")
except ImportError:
    pass

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
    logger.info("SpeechRecognition available")
except ImportError:
    pass


# Global model instance (lazy loaded)
_whisper_model = None


def _get_whisper_model():
    """Get or create Whisper model."""
    global _whisper_model
    if _whisper_model is None and WHISPER_AVAILABLE:
        logger.info("Loading Whisper model (base)...")
        # Use 'base' model - good balance of speed/accuracy
        # Options: tiny, base, small, medium, large-v2
        _whisper_model = WhisperModel(
            "base",
            device="cuda",  # or "cpu"
            compute_type="float16"  # or "int8" for CPU
        )
        logger.info("Whisper model loaded")
    return _whisper_model


def transcribe_audio(
    audio_data: bytes,
    sample_rate: int = 16000,
    language: str = "en"
) -> str:
    """
    Transcribe audio to text.

    Tries backends in order: Whisper > Vosk > Google

    Args:
        audio_data: Raw audio bytes (16-bit PCM)
        sample_rate: Sample rate of audio
        language: Language code (e.g., "en", "es")

    Returns:
        Transcribed text or empty string
    """
    # Try Whisper first (best quality)
    if WHISPER_AVAILABLE:
        return _transcribe_whisper(audio_data, sample_rate, language)

    # Try Vosk (good offline option)
    if VOSK_AVAILABLE:
        return _transcribe_vosk(audio_data, sample_rate)

    # Try Google (requires internet)
    if SR_AVAILABLE:
        return _transcribe_google(audio_data, sample_rate, language)

    logger.error(
        "No ASR backend available. Install one of:\n"
        "  pip install faster-whisper\n"
        "  pip install vosk\n"
        "  pip install SpeechRecognition"
    )
    return ""


def _transcribe_whisper(
    audio_data: bytes,
    sample_rate: int,
    language: str
) -> str:
    """Transcribe using Whisper."""
    import numpy as np

    model = _get_whisper_model()
    if model is None:
        return ""

    try:
        # Convert to float32 normalized audio
        audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        # Transcribe
        segments, info = model.transcribe(
            audio,
            language=language,
            vad_filter=True,  # Filter out non-speech
            beam_size=5
        )

        # Combine segments
        text = " ".join(segment.text for segment in segments)
        logger.info(f"Whisper transcribed: {text[:100]}...")
        return text.strip()

    except Exception as e:
        logger.error(f"Whisper transcription error: {e}")
        return ""


def _transcribe_vosk(audio_data: bytes, sample_rate: int) -> str:
    """Transcribe using Vosk."""
    import json

    try:
        # Vosk needs a model - check if downloaded
        model_path = Path.home() / ".ara" / "models" / "vosk-model-small-en-us"

        if not model_path.exists():
            logger.warning(
                f"Vosk model not found at {model_path}. "
                "Download from https://alphacephei.com/vosk/models"
            )
            return ""

        model = vosk.Model(str(model_path))
        rec = vosk.KaldiRecognizer(model, sample_rate)

        # Process audio
        rec.AcceptWaveform(audio_data)
        result = json.loads(rec.FinalResult())

        text = result.get("text", "")
        logger.info(f"Vosk transcribed: {text[:100]}...")
        return text.strip()

    except Exception as e:
        logger.error(f"Vosk transcription error: {e}")
        return ""


def _transcribe_google(
    audio_data: bytes,
    sample_rate: int,
    language: str
) -> str:
    """Transcribe using Google Speech Recognition."""
    try:
        import numpy as np
        from scipy.io import wavfile
        import io

        # SpeechRecognition needs a WAV file
        recognizer = sr.Recognizer()

        # Convert to WAV in memory
        audio_np = np.frombuffer(audio_data, dtype=np.int16)

        wav_buffer = io.BytesIO()
        wavfile.write(wav_buffer, sample_rate, audio_np)
        wav_buffer.seek(0)

        # Recognize
        with sr.AudioFile(wav_buffer) as source:
            audio = recognizer.record(source)

        text = recognizer.recognize_google(audio, language=language)
        logger.info(f"Google transcribed: {text[:100]}...")
        return text.strip()

    except sr.UnknownValueError:
        logger.warning("Google could not understand audio")
        return ""
    except sr.RequestError as e:
        logger.error(f"Google Speech Recognition error: {e}")
        return ""
    except Exception as e:
        logger.error(f"Google transcription error: {e}")
        return ""


# ============================================================
# Fallback: Keyboard input
# ============================================================

def get_text_input(prompt: str = "You: ") -> str:
    """
    Fallback: Get text input from keyboard.

    Use this when no ASR is available.
    """
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        return ""


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("ASR backends available:")
    print(f"  Whisper: {WHISPER_AVAILABLE}")
    print(f"  Vosk: {VOSK_AVAILABLE}")
    print(f"  Google: {SR_AVAILABLE}")

    if not any([WHISPER_AVAILABLE, VOSK_AVAILABLE, SR_AVAILABLE]):
        print("\nNo ASR backend available. Using keyboard input.")
        text = get_text_input()
        print(f"You typed: {text}")
    else:
        print("\nRecording for test transcription...")
        from ara.avatar.audio import record_utterance
        audio = record_utterance()
        if audio:
            text = transcribe_audio(audio)
            print(f"Transcribed: {text}")
