"""
Text-to-Speech for Ara Avatar

Converts text to audio. Supports multiple backends:
- Piper (local, fast, good quality)
- Coqui TTS (local, many voices)
- espeak (local, lightweight)
- gTTS (cloud, Google)

Install (pick one):
    pip install piper-tts  # Recommended
    pip install TTS        # Coqui
    pip install gTTS       # Google (requires internet)
    # espeak is usually pre-installed on Linux
"""

import logging
import subprocess
import tempfile
from typing import Optional
from pathlib import Path

logger = logging.getLogger("ara.avatar.tts")

# Try to import TTS backends
PIPER_AVAILABLE = False
COQUI_AVAILABLE = False
GTTS_AVAILABLE = False
ESPEAK_AVAILABLE = False

try:
    import piper
    PIPER_AVAILABLE = True
    logger.info("piper-tts available")
except ImportError:
    pass

try:
    from TTS.api import TTS
    COQUI_AVAILABLE = True
    logger.info("Coqui TTS available")
except ImportError:
    pass

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
    logger.info("gTTS available")
except ImportError:
    pass

# Check for espeak / espeak-ng
ESPEAK_CMD = None
try:
    result = subprocess.run(["espeak-ng", "--version"], capture_output=True)
    if result.returncode == 0:
        ESPEAK_AVAILABLE = True
        ESPEAK_CMD = "espeak-ng"
        logger.info("espeak-ng available")
except FileNotFoundError:
    pass

if not ESPEAK_AVAILABLE:
    try:
        result = subprocess.run(["espeak", "--version"], capture_output=True)
        if result.returncode == 0:
            ESPEAK_AVAILABLE = True
            ESPEAK_CMD = "espeak"
            logger.info("espeak available")
    except FileNotFoundError:
        pass


# Global TTS instance (lazy loaded)
_piper_voice = None
_coqui_tts = None


def _get_piper_voice():
    """Get or create Piper voice."""
    global _piper_voice
    if _piper_voice is None and PIPER_AVAILABLE:
        # Check for downloaded voice model
        voice_path = Path.home() / ".ara" / "models" / "piper" / "en_US-lessac-medium.onnx"

        if not voice_path.exists():
            logger.warning(
                f"Piper voice not found at {voice_path}. "
                "Download from https://github.com/rhasspy/piper/releases"
            )
            return None

        logger.info("Loading Piper voice...")
        _piper_voice = piper.PiperVoice.load(str(voice_path))
        logger.info("Piper voice loaded")

    return _piper_voice


def _get_coqui_tts():
    """Get or create Coqui TTS."""
    global _coqui_tts
    if _coqui_tts is None and COQUI_AVAILABLE:
        logger.info("Loading Coqui TTS...")
        # Use a good English voice
        _coqui_tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
        logger.info("Coqui TTS loaded")
    return _coqui_tts


def synthesize_speech(
    text: str,
    voice: str = "default",
    speed: float = 1.0
) -> Optional[bytes]:
    """
    Synthesize speech from text.

    Tries backends in order: Piper > Coqui > gTTS > espeak

    Args:
        text: Text to speak
        voice: Voice name (backend-specific)
        speed: Speech speed multiplier

    Returns:
        Raw audio bytes (16-bit PCM, 16kHz) or None
    """
    if not text.strip():
        return None

    # Try Piper first (best local quality/speed)
    if PIPER_AVAILABLE:
        audio = _synthesize_piper(text, speed)
        if audio:
            return audio

    # Try Coqui (high quality but slower)
    if COQUI_AVAILABLE:
        audio = _synthesize_coqui(text, speed)
        if audio:
            return audio

    # Try gTTS (requires internet)
    if GTTS_AVAILABLE:
        audio = _synthesize_gtts(text)
        if audio:
            return audio

    # Try espeak (lowest quality but always available)
    if ESPEAK_AVAILABLE:
        audio = _synthesize_espeak(text, speed)
        if audio:
            return audio

    logger.error(
        "No TTS backend available. Install one of:\n"
        "  pip install piper-tts\n"
        "  pip install TTS\n"
        "  pip install gTTS\n"
        "  sudo apt install espeak"
    )
    return None


def _synthesize_piper(text: str, speed: float) -> Optional[bytes]:
    """Synthesize using Piper."""
    import wave
    import io

    voice = _get_piper_voice()
    if voice is None:
        return None

    try:
        # Piper synthesize() needs a wave.Wave_write object
        # Create a BytesIO buffer and wrap it with wave module
        wav_buffer = io.BytesIO()

        with wave.open(wav_buffer, 'wb') as wav_file:
            voice.synthesize(text, wav_file)

        # Read the complete WAV data
        wav_buffer.seek(0)
        wav_data = wav_buffer.read()

        if len(wav_data) <= 44:  # Just header, no audio
            logger.warning("Piper produced no audio data")
            return None

        # Skip WAV header (44 bytes) to get raw PCM
        audio = wav_data[44:]

        logger.info(f"Piper synthesized {len(audio)} bytes")
        return audio

    except Exception as e:
        logger.error(f"Piper synthesis error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def _synthesize_coqui(text: str, speed: float) -> Optional[bytes]:
    """Synthesize using Coqui TTS."""
    import numpy as np

    tts = _get_coqui_tts()
    if tts is None:
        return None

    try:
        # Synthesize to numpy array
        wav = tts.tts(text)

        # Convert to 16-bit PCM
        audio = (np.array(wav) * 32767).astype(np.int16)
        logger.info(f"Coqui synthesized {len(audio)} samples")
        return audio.tobytes()

    except Exception as e:
        logger.error(f"Coqui synthesis error: {e}")
        return None


def _synthesize_gtts(text: str) -> Optional[bytes]:
    """Synthesize using Google TTS."""
    import numpy as np
    from scipy.io import wavfile
    import io

    try:
        # Generate speech
        tts = gTTS(text=text, lang='en')

        # Save to temporary MP3
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tts.save(f.name)
            mp3_path = f.name

        # Convert MP3 to WAV using ffmpeg
        wav_path = mp3_path.replace(".mp3", ".wav")
        subprocess.run([
            "ffmpeg", "-y", "-i", mp3_path,
            "-ar", "16000", "-ac", "1", "-f", "wav", wav_path
        ], capture_output=True)

        # Load WAV
        sample_rate, audio = wavfile.read(wav_path)

        # Cleanup
        Path(mp3_path).unlink(missing_ok=True)
        Path(wav_path).unlink(missing_ok=True)

        logger.info(f"gTTS synthesized {len(audio)} samples")
        return audio.tobytes()

    except Exception as e:
        logger.error(f"gTTS synthesis error: {e}")
        return None


def _synthesize_espeak(text: str, speed: float) -> Optional[bytes]:
    """Synthesize using espeak."""
    import numpy as np
    from scipy.io import wavfile

    try:
        # Generate WAV with espeak
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name

        # Calculate words per minute (default 175)
        wpm = int(175 * speed)

        subprocess.run([
            ESPEAK_CMD,
            "-w", wav_path,
            "-s", str(wpm),
            "-v", "en",  # English voice
            text
        ], capture_output=True)

        # Load and convert to 16kHz
        sample_rate, audio = wavfile.read(wav_path)

        # Resample if needed
        if sample_rate != 16000:
            from scipy import signal
            num_samples = int(len(audio) * 16000 / sample_rate)
            audio = signal.resample(audio, num_samples).astype(np.int16)

        # Cleanup
        Path(wav_path).unlink(missing_ok=True)

        logger.info(f"espeak synthesized {len(audio)} samples")
        return audio.tobytes()

    except Exception as e:
        logger.error(f"espeak synthesis error: {e}")
        return None


# ============================================================
# Fallback: Print text
# ============================================================

def print_speech(text: str):
    """
    Fallback: Just print the text.

    Use this when no TTS is available.
    """
    print(f"\n[Ara]: {text}\n")


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("TTS backends available:")
    print(f"  Piper: {PIPER_AVAILABLE}")
    print(f"  Coqui: {COQUI_AVAILABLE}")
    print(f"  gTTS: {GTTS_AVAILABLE}")
    print(f"  espeak: {ESPEAK_AVAILABLE}")

    test_text = "Hello! I am Ara, your cognitive AI companion. How are you today?"

    if not any([PIPER_AVAILABLE, COQUI_AVAILABLE, GTTS_AVAILABLE, ESPEAK_AVAILABLE]):
        print("\nNo TTS backend available. Using text output.")
        print_speech(test_text)
    else:
        print(f"\nSynthesizing: {test_text}")
        audio = synthesize_speech(test_text)
        if audio:
            print(f"Generated {len(audio)} bytes of audio")
            from ara.avatar.audio import play_audio
            play_audio(audio)
