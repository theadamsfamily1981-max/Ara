"""Enhanced ASR with Whisper optimization and custom vocabulary."""

import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
import time
import threading
import queue

from ..utils.logger import get_logger

logger = get_logger(__name__)

# Try to import whisper
try:
    import whisper
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class ASRConfig:
    """Configuration for enhanced ASR."""
    # Model settings
    model_size: str = "base"  # tiny, base, small, medium, large
    device: str = "cuda"
    compute_type: str = "float16"  # float16, float32, int8

    # Language settings
    language: str = "en"
    detect_language: bool = False

    # Transcription settings
    beam_size: int = 5
    best_of: int = 5
    temperature: float = 0.0

    # Custom vocabulary for better recognition
    custom_vocabulary: List[str] = field(default_factory=list)
    initial_prompt: str = ""

    # VAD settings
    use_vad: bool = True
    vad_threshold: float = 0.5
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 500

    # Performance
    condition_on_previous_text: bool = True
    compression_ratio_threshold: float = 2.4
    logprob_threshold: float = -1.0
    no_speech_threshold: float = 0.6


@dataclass
class TranscriptionResult:
    """Result of a transcription."""
    text: str
    language: str
    confidence: float
    segments: List[Dict[str, Any]]
    duration: float
    processing_time: float

    @property
    def words_per_second(self) -> float:
        """Calculate transcription speed."""
        word_count = len(self.text.split())
        return word_count / self.processing_time if self.processing_time > 0 else 0


class CustomVocabularyManager:
    """Manage custom vocabulary for better recognition."""

    def __init__(self):
        self.terms: Dict[str, str] = {}  # term -> phonetic hint
        self.contexts: List[str] = []

    def add_term(self, term: str, phonetic_hint: str = None) -> None:
        """Add a custom term.

        Args:
            term: The term to recognize
            phonetic_hint: Optional phonetic spelling hint
        """
        self.terms[term.lower()] = phonetic_hint or term

    def add_terms(self, terms: List[str]) -> None:
        """Add multiple terms."""
        for term in terms:
            self.add_term(term)

    def add_context(self, context: str) -> None:
        """Add a context sentence to help recognition."""
        self.contexts.append(context)

    def build_prompt(self) -> str:
        """Build an initial prompt for Whisper.

        The initial prompt helps Whisper recognize domain-specific terms.
        """
        parts = []

        # Add context sentences
        if self.contexts:
            parts.extend(self.contexts)

        # Add terms as a natural sentence
        if self.terms:
            terms_list = list(self.terms.keys())
            if len(terms_list) <= 5:
                parts.append(f"Keywords: {', '.join(terms_list)}.")
            else:
                # Sample for longer lists
                import random
                sampled = random.sample(terms_list, min(10, len(terms_list)))
                parts.append(f"Keywords include: {', '.join(sampled)}.")

        return " ".join(parts)

    def post_process(self, text: str) -> str:
        """Post-process transcription to fix known terms.

        Corrects common misrecognitions of custom vocabulary.
        """
        result = text

        for term, hint in self.terms.items():
            # Simple fuzzy matching for common misrecognitions
            # This could be enhanced with proper fuzzy matching
            variations = self._generate_variations(term)
            for variation in variations:
                if variation.lower() in result.lower():
                    # Case-preserving replacement
                    import re
                    result = re.sub(
                        re.escape(variation),
                        term,
                        result,
                        flags=re.IGNORECASE
                    )

        return result

    def _generate_variations(self, term: str) -> List[str]:
        """Generate common misrecognition variations of a term."""
        variations = [term]

        # Add without special characters
        cleaned = ''.join(c for c in term if c.isalnum() or c.isspace())
        if cleaned != term:
            variations.append(cleaned)

        # Add common phonetic variations
        # This is simplified - could be enhanced with phonetic algorithms
        replacements = {
            'ai': ['ay', 'ey'],
            'tion': ['shun', 'shen'],
            'ph': ['f'],
            'ck': ['k', 'c'],
        }

        for orig, repls in replacements.items():
            if orig in term.lower():
                for repl in repls:
                    variations.append(term.lower().replace(orig, repl))

        return variations


class EnhancedASR:
    """Enhanced ASR with Whisper optimization."""

    def __init__(self, config: ASRConfig = None):
        """Initialize enhanced ASR.

        Args:
            config: ASR configuration
        """
        self.config = config or ASRConfig()
        self.model = None
        self._loaded = False

        # Custom vocabulary
        self.vocabulary = CustomVocabularyManager()
        if self.config.custom_vocabulary:
            self.vocabulary.add_terms(self.config.custom_vocabulary)

        # Transcription cache
        self._cache: Dict[int, TranscriptionResult] = {}
        self._cache_lock = threading.Lock()

    def load(self) -> None:
        """Load the Whisper model."""
        if self._loaded:
            return

        if not HAS_WHISPER:
            raise ImportError("Whisper not installed. Run: pip install openai-whisper")

        device = self.config.device
        if device == "cuda" and HAS_TORCH and not torch.cuda.is_available():
            device = "cpu"
            logger.warning("CUDA not available, falling back to CPU")

        logger.info(f"Loading Whisper {self.config.model_size} on {device}...")

        try:
            self.model = whisper.load_model(
                self.config.model_size,
                device=device
            )
            self._loaded = True
            logger.info("Whisper model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        use_cache: bool = True
    ) -> TranscriptionResult:
        """Transcribe audio to text.

        Args:
            audio: Audio as numpy array
            sample_rate: Audio sample rate
            use_cache: Whether to use transcription cache

        Returns:
            TranscriptionResult with transcription details
        """
        if not self._loaded:
            self.load()

        start_time = time.time()

        # Check cache
        if use_cache:
            cache_key = hash(audio.tobytes())
            with self._cache_lock:
                if cache_key in self._cache:
                    logger.debug("Using cached transcription")
                    return self._cache[cache_key]

        # Ensure correct format
        audio = self._prepare_audio(audio, sample_rate)

        # Build initial prompt
        initial_prompt = self.config.initial_prompt
        if self.vocabulary.terms:
            vocab_prompt = self.vocabulary.build_prompt()
            initial_prompt = f"{initial_prompt} {vocab_prompt}".strip()

        # Transcribe
        try:
            result = self.model.transcribe(
                audio,
                language=None if self.config.detect_language else self.config.language,
                task="transcribe",
                beam_size=self.config.beam_size,
                best_of=self.config.best_of,
                temperature=self.config.temperature,
                initial_prompt=initial_prompt if initial_prompt else None,
                condition_on_previous_text=self.config.condition_on_previous_text,
                compression_ratio_threshold=self.config.compression_ratio_threshold,
                logprob_threshold=self.config.logprob_threshold,
                no_speech_threshold=self.config.no_speech_threshold,
                fp16=(self.config.compute_type == "float16" and self.config.device == "cuda")
            )

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise

        processing_time = time.time() - start_time

        # Extract text and apply post-processing
        text = result["text"].strip()
        if self.vocabulary.terms:
            text = self.vocabulary.post_process(text)

        # Calculate confidence from segment probabilities
        segments = result.get("segments", [])
        if segments:
            avg_logprob = np.mean([s.get("avg_logprob", -1) for s in segments])
            confidence = np.exp(avg_logprob)  # Convert log prob to probability
        else:
            confidence = 0.0

        # Calculate audio duration
        duration = len(audio) / 16000  # Whisper uses 16kHz

        transcription = TranscriptionResult(
            text=text,
            language=result.get("language", self.config.language),
            confidence=float(confidence),
            segments=[
                {
                    "start": s["start"],
                    "end": s["end"],
                    "text": s["text"],
                    "confidence": np.exp(s.get("avg_logprob", -1))
                }
                for s in segments
            ],
            duration=duration,
            processing_time=processing_time
        )

        # Cache result
        if use_cache:
            with self._cache_lock:
                # Limit cache size
                if len(self._cache) > 100:
                    # Remove oldest entries
                    oldest_keys = list(self._cache.keys())[:50]
                    for k in oldest_keys:
                        del self._cache[k]
                self._cache[cache_key] = transcription

        logger.info(
            f"Transcribed {duration:.1f}s audio in {processing_time:.2f}s "
            f"({transcription.words_per_second:.1f} words/s)"
        )

        return transcription

    def transcribe_file(self, audio_path: Path) -> TranscriptionResult:
        """Transcribe an audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            TranscriptionResult
        """
        if not self._loaded:
            self.load()

        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            import soundfile as sf
            audio, sample_rate = sf.read(str(audio_path))
        except ImportError:
            # Fallback to whisper's built-in loading
            audio = whisper.load_audio(str(audio_path))
            sample_rate = 16000

        return self.transcribe(audio, sample_rate)

    def transcribe_streaming(
        self,
        audio_queue: queue.Queue,
        result_callback: callable,
        stop_event: threading.Event
    ) -> None:
        """Transcribe audio in streaming mode.

        Continuously reads from audio_queue and calls result_callback
        with transcription results.

        Args:
            audio_queue: Queue containing audio chunks (numpy arrays)
            result_callback: Called with TranscriptionResult for each chunk
            stop_event: Set this to stop streaming
        """
        if not self._loaded:
            self.load()

        audio_buffer = []
        buffer_duration = 0.0
        min_buffer_duration = 1.0  # Minimum audio to process
        max_buffer_duration = 30.0  # Maximum before forced processing

        logger.info("Starting streaming transcription")

        while not stop_event.is_set():
            try:
                # Get audio chunk with timeout
                chunk = audio_queue.get(timeout=0.1)
                audio_buffer.append(chunk)
                buffer_duration += len(chunk) / 16000

                # Process when we have enough audio
                if buffer_duration >= min_buffer_duration:
                    # Check for speech activity (simple energy-based VAD)
                    combined = np.concatenate(audio_buffer)
                    energy = np.mean(combined ** 2)

                    # Only transcribe if there's likely speech
                    if energy > 0.001 or buffer_duration >= max_buffer_duration:
                        try:
                            result = self.transcribe(combined, use_cache=False)
                            if result.text.strip():
                                result_callback(result)
                        except Exception as e:
                            logger.error(f"Streaming transcription error: {e}")

                        # Reset buffer
                        audio_buffer = []
                        buffer_duration = 0.0

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                break

        # Process remaining audio
        if audio_buffer:
            combined = np.concatenate(audio_buffer)
            try:
                result = self.transcribe(combined, use_cache=False)
                if result.text.strip():
                    result_callback(result)
            except Exception as e:
                logger.error(f"Final transcription error: {e}")

        logger.info("Streaming transcription stopped")

    def _prepare_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Prepare audio for Whisper (16kHz, mono, float32)."""
        # Convert to float32
        if audio.dtype != np.float32:
            if np.issubdtype(audio.dtype, np.integer):
                max_val = np.iinfo(audio.dtype).max
                audio = audio.astype(np.float32) / max_val
            else:
                audio = audio.astype(np.float32)

        # Convert stereo to mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            try:
                import librosa
                audio = librosa.resample(
                    audio,
                    orig_sr=sample_rate,
                    target_sr=16000
                )
            except ImportError:
                # Simple linear interpolation fallback
                ratio = 16000 / sample_rate
                new_length = int(len(audio) * ratio)
                audio = np.interp(
                    np.linspace(0, len(audio) - 1, new_length),
                    np.arange(len(audio)),
                    audio
                )

        return audio

    def add_vocabulary(self, terms: List[str]) -> None:
        """Add custom vocabulary terms."""
        self.vocabulary.add_terms(terms)

    def add_context(self, context: str) -> None:
        """Add context to help recognition."""
        self.vocabulary.add_context(context)

    def set_initial_prompt(self, prompt: str) -> None:
        """Set the initial prompt for transcription."""
        self.config.initial_prompt = prompt

    def clear_cache(self) -> None:
        """Clear the transcription cache."""
        with self._cache_lock:
            self._cache.clear()

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self._loaded:
            return {"loaded": False}

        return {
            "loaded": True,
            "model_size": self.config.model_size,
            "device": str(next(self.model.parameters()).device),
            "languages": whisper.tokenizer.LANGUAGES if HAS_WHISPER else [],
            "vocabulary_terms": len(self.vocabulary.terms)
        }
