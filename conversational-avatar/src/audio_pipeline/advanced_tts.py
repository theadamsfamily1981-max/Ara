"""Advanced TTS with smart chunking and multi-sample voice cloning."""

import re
import numpy as np
from pathlib import Path
from typing import Optional, List, Union
from dataclasses import dataclass
import tempfile
import wave
import struct

try:
    import torch
    from TTS.api import TTS
    HAS_TTS = True
except ImportError:
    HAS_TTS = False

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TTSConfig:
    """Configuration for advanced TTS."""
    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    device: str = "cuda"
    language: str = "en"
    speed: float = 1.0
    # Chunking settings
    max_chunk_chars: int = 220
    min_chunk_chars: int = 40
    # Voice settings
    voice_samples: List[str] = None
    # Quality settings
    sample_rate: int = 24000
    enable_denoising: bool = True


class SmartChunker:
    """Prosody-aware text chunking for natural TTS output."""

    # Sentence endings with strong pauses
    STRONG_BREAKS = re.compile(r'(?<=[\.\!\?])\s+')
    # Medium breaks (semicolons, colons)
    MEDIUM_BREAKS = re.compile(r'(?<=[\;\:])\s+')
    # Weak breaks (commas)
    WEAK_BREAKS = re.compile(r'(?<=[\,])\s+')
    # Parenthetical breaks
    PAREN_BREAKS = re.compile(r'(?<=[\)\]\}])\s+')

    def __init__(self, max_chars: int = 220, min_chars: int = 40):
        self.max_chars = max_chars
        self.min_chars = min_chars

    def split_for_tts(self, text: str) -> List[str]:
        """Split text into prosody-friendly chunks.

        Prioritizes natural break points:
        1. Sentence endings (. ! ?)
        2. Semicolons and colons
        3. Commas
        4. Parenthetical endings
        5. Word boundaries as fallback

        Args:
            text: Input text to chunk

        Returns:
            List of text chunks
        """
        if not text or len(text.strip()) == 0:
            return []

        text = text.strip()

        # If text is short enough, return as-is
        if len(text) <= self.max_chars:
            return [text]

        # First pass: split on sentence boundaries
        sentences = self.STRONG_BREAKS.split(text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # If adding this sentence would exceed limit
            if len(current_chunk) + len(sentence) + 1 > self.max_chars:
                # Save current chunk if it's substantial
                if len(current_chunk) >= self.min_chars:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                # If single sentence is too long, split further
                if len(sentence) > self.max_chars:
                    sub_chunks = self._split_long_sentence(sentence)
                    chunks.extend(sub_chunks[:-1])
                    current_chunk = sub_chunks[-1] if sub_chunks else ""
                else:
                    current_chunk = sentence
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence

        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _split_long_sentence(self, sentence: str) -> List[str]:
        """Split a long sentence on secondary break points."""
        # Try medium breaks first
        parts = self.MEDIUM_BREAKS.split(sentence)
        if len(parts) > 1 and all(len(p) <= self.max_chars for p in parts):
            return [p.strip() for p in parts if p.strip()]

        # Try weak breaks (commas)
        parts = self.WEAK_BREAKS.split(sentence)
        if len(parts) > 1:
            return self._merge_small_parts(parts)

        # Fallback: split on word boundaries
        words = sentence.split()
        chunks = []
        current = ""

        for word in words:
            if len(current) + len(word) + 1 > self.max_chars:
                if current:
                    chunks.append(current.strip())
                current = word
            else:
                current = (current + " " + word) if current else word

        if current:
            chunks.append(current.strip())

        return chunks

    def _merge_small_parts(self, parts: List[str]) -> List[str]:
        """Merge small parts to avoid tiny chunks."""
        chunks = []
        current = ""

        for part in parts:
            part = part.strip()
            if not part:
                continue

            if len(current) + len(part) + 1 <= self.max_chars:
                current = (current + ", " + part) if current else part
            else:
                if current:
                    chunks.append(current)
                current = part

        if current:
            chunks.append(current)

        return chunks


class VoiceSampler:
    """Manage multiple voice samples for consistent cloning."""

    def __init__(self, sample_paths: List[str] = None):
        self.samples = []
        self.combined_path = None

        if sample_paths:
            self.add_samples(sample_paths)

    def add_samples(self, paths: List[str]) -> None:
        """Add voice sample paths."""
        for path in paths:
            p = Path(path)
            if p.exists() and p.suffix.lower() in ['.wav', '.mp3', '.flac', '.ogg']:
                self.samples.append(str(p))
                logger.info(f"Added voice sample: {p.name}")
            else:
                logger.warning(f"Voice sample not found or invalid format: {path}")

    def get_combined_sample(self) -> Optional[str]:
        """Get path to combined voice sample.

        Concatenates multiple samples for more robust voice cloning.
        XTTS-v2 works best with 6-30 seconds of combined samples.
        """
        if not self.samples:
            return None

        if len(self.samples) == 1:
            return self.samples[0]

        # Create combined sample if not already done
        if self.combined_path is None:
            self.combined_path = self._combine_samples()

        return self.combined_path

    def _combine_samples(self) -> str:
        """Combine multiple WAV samples into one."""
        try:
            import soundfile as sf

            all_audio = []
            target_sr = None

            for sample_path in self.samples:
                audio, sr = sf.read(sample_path)
                if target_sr is None:
                    target_sr = sr
                elif sr != target_sr:
                    # Resample if needed
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

                all_audio.append(audio)
                # Add small silence between samples
                all_audio.append(np.zeros(int(target_sr * 0.3)))

            combined = np.concatenate(all_audio)

            # Save to temp file
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            sf.write(temp_file.name, combined, target_sr)

            logger.info(f"Combined {len(self.samples)} voice samples -> {len(combined)/target_sr:.1f}s")
            return temp_file.name

        except ImportError:
            logger.warning("soundfile/librosa not available, using first sample only")
            return self.samples[0]
        except Exception as e:
            logger.error(f"Failed to combine voice samples: {e}")
            return self.samples[0]

    def get_random_sample(self) -> Optional[str]:
        """Get a random sample (for variation)."""
        if not self.samples:
            return None
        import random
        return random.choice(self.samples)


class AdvancedTTS:
    """Advanced TTS with smart chunking and multi-sample voice cloning."""

    def __init__(self, config: TTSConfig = None):
        """Initialize advanced TTS.

        Args:
            config: TTS configuration
        """
        self.config = config or TTSConfig()
        self.chunker = SmartChunker(
            max_chars=self.config.max_chunk_chars,
            min_chars=self.config.min_chunk_chars
        )
        self.voice_sampler = VoiceSampler(self.config.voice_samples)
        self.tts = None
        self._loaded = False

    def load(self) -> None:
        """Load the TTS model."""
        if self._loaded:
            return

        if not HAS_TTS:
            raise ImportError("TTS library not installed. Run: pip install TTS")

        device = self.config.device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
            logger.warning("CUDA not available, falling back to CPU")

        logger.info(f"Loading TTS model '{self.config.model_name}' on {device}...")

        try:
            self.tts = TTS(self.config.model_name).to(device)
            self._loaded = True
            logger.info("TTS model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load TTS model: {e}")
            raise

    def synthesize(
        self,
        text: str,
        output_path: Optional[Path] = None,
        speaker_wav: Optional[str] = None,
        use_chunking: bool = True,
        progress_callback: callable = None
    ) -> Path:
        """Synthesize speech with smart chunking.

        Args:
            text: Text to synthesize
            output_path: Output audio path
            speaker_wav: Override voice sample
            use_chunking: Whether to use smart chunking
            progress_callback: Callback for progress updates (chunk_idx, total_chunks)

        Returns:
            Path to generated audio file
        """
        if not self._loaded:
            self.load()

        if not text or text.strip() == "":
            logger.warning("Empty text provided for synthesis")
            return None

        # Get voice sample
        voice = speaker_wav or self.voice_sampler.get_combined_sample()

        # Generate output path if not provided
        if output_path is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            output_path = Path(f"outputs/audio/tts_{timestamp}.wav")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Chunk text if needed
        if use_chunking:
            chunks = self.chunker.split_for_tts(text)
        else:
            chunks = [text]

        logger.info(f"Synthesizing {len(chunks)} chunk(s)...")

        if len(chunks) == 1:
            # Single chunk - direct synthesis
            return self._synthesize_single(chunks[0], output_path, voice)

        # Multiple chunks - synthesize and concatenate
        chunk_audios = []

        for i, chunk in enumerate(chunks):
            if progress_callback:
                progress_callback(i, len(chunks))

            try:
                audio = self._synthesize_to_array(chunk, voice)
                chunk_audios.append(audio)

                # Add small pause between chunks for natural pacing
                pause_samples = int(self.config.sample_rate * 0.15)  # 150ms pause
                chunk_audios.append(np.zeros(pause_samples))

            except Exception as e:
                logger.error(f"Failed to synthesize chunk {i}: {e}")
                continue

        if not chunk_audios:
            raise RuntimeError("Failed to synthesize any chunks")

        # Concatenate all chunks
        full_audio = np.concatenate(chunk_audios)

        # Save to file
        self._save_audio(full_audio, output_path)

        if progress_callback:
            progress_callback(len(chunks), len(chunks))

        logger.info(f"Synthesized {len(full_audio)/self.config.sample_rate:.1f}s audio -> {output_path}")
        return output_path

    def synthesize_streaming(
        self,
        text: str,
        speaker_wav: Optional[str] = None
    ):
        """Generator that yields audio chunks as they're synthesized.

        Useful for real-time playback during synthesis.

        Yields:
            numpy array of audio samples
        """
        if not self._loaded:
            self.load()

        voice = speaker_wav or self.voice_sampler.get_combined_sample()
        chunks = self.chunker.split_for_tts(text)

        for chunk in chunks:
            try:
                audio = self._synthesize_to_array(chunk, voice)
                yield audio

                # Yield pause between chunks
                pause = np.zeros(int(self.config.sample_rate * 0.15))
                yield pause

            except Exception as e:
                logger.error(f"Streaming synthesis error: {e}")
                continue

    def _synthesize_single(self, text: str, output_path: Path, voice: str) -> Path:
        """Synthesize a single chunk directly to file."""
        try:
            if "xtts" in self.config.model_name.lower():
                self.tts.tts_to_file(
                    text=text,
                    file_path=str(output_path),
                    speaker_wav=voice,
                    language=self.config.language,
                    speed=self.config.speed
                )
            else:
                self.tts.tts_to_file(
                    text=text,
                    file_path=str(output_path),
                    speaker=voice,
                    language=self.config.language
                )
            return output_path

        except Exception as e:
            logger.error(f"Single chunk synthesis failed: {e}")
            raise

    def _synthesize_to_array(self, text: str, voice: str) -> np.ndarray:
        """Synthesize text to numpy array."""
        try:
            if "xtts" in self.config.model_name.lower():
                audio = self.tts.tts(
                    text=text,
                    speaker_wav=voice,
                    language=self.config.language,
                    speed=self.config.speed
                )
            else:
                audio = self.tts.tts(
                    text=text,
                    speaker=voice,
                    language=self.config.language
                )
            return np.array(audio, dtype=np.float32)

        except Exception as e:
            logger.error(f"Array synthesis failed: {e}")
            raise

    def _save_audio(self, audio: np.ndarray, path: Path) -> None:
        """Save audio array to WAV file."""
        try:
            import soundfile as sf
            sf.write(str(path), audio, self.config.sample_rate)
        except ImportError:
            # Fallback to wave module
            with wave.open(str(path), 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.config.sample_rate)
                # Convert float to int16
                audio_int = (audio * 32767).astype(np.int16)
                wf.writeframes(audio_int.tobytes())

    def add_voice_samples(self, paths: List[str]) -> None:
        """Add voice samples for cloning."""
        self.voice_sampler.add_samples(paths)

    def set_speed(self, speed: float) -> None:
        """Set speech speed (0.5-2.0)."""
        self.config.speed = max(0.5, min(2.0, speed))

    def set_language(self, language: str) -> None:
        """Set synthesis language."""
        self.config.language = language
