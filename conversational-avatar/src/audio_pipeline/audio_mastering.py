"""Audio mastering chain for broadcast-quality TTS output."""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from pathlib import Path

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MasteringConfig:
    """Configuration for audio mastering chain."""
    # Normalization
    target_loudness_db: float = -16.0  # Target LUFS
    normalize: bool = True

    # Compressor settings
    enable_compressor: bool = True
    comp_threshold_db: float = -18.0
    comp_ratio: float = 3.0
    comp_attack_ms: float = 5.0
    comp_release_ms: float = 50.0
    comp_knee_db: float = 6.0
    comp_makeup_gain_db: float = 2.0

    # Limiter settings
    enable_limiter: bool = True
    limiter_threshold_db: float = -1.0
    limiter_release_ms: float = 50.0

    # Reverb settings
    enable_reverb: bool = True
    reverb_room_size: float = 0.15  # Small room (0.0-1.0)
    reverb_damping: float = 0.7
    reverb_wet_level: float = 0.08  # Subtle
    reverb_dry_level: float = 0.92

    # De-esser (reduces sibilance)
    enable_deesser: bool = True
    deesser_threshold_db: float = -20.0
    deesser_frequency: float = 6000.0  # Hz
    deesser_reduction_db: float = 6.0

    # High-pass filter (removes rumble)
    enable_highpass: bool = True
    highpass_frequency: float = 80.0  # Hz

    # Sample rate
    sample_rate: int = 24000


class Compressor:
    """Dynamic range compressor with soft knee."""

    def __init__(
        self,
        threshold_db: float,
        ratio: float,
        attack_ms: float,
        release_ms: float,
        knee_db: float,
        makeup_gain_db: float,
        sample_rate: int
    ):
        self.threshold = self._db_to_linear(threshold_db)
        self.threshold_db = threshold_db
        self.ratio = ratio
        self.knee_db = knee_db
        self.makeup_gain = self._db_to_linear(makeup_gain_db)

        # Time constants
        self.attack_coeff = np.exp(-1.0 / (attack_ms * sample_rate / 1000))
        self.release_coeff = np.exp(-1.0 / (release_ms * sample_rate / 1000))

        self.envelope = 0.0

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Apply compression to audio."""
        output = np.zeros_like(audio)

        for i, sample in enumerate(audio):
            # Get absolute level
            level = abs(sample)

            # Envelope follower
            if level > self.envelope:
                self.envelope = self.attack_coeff * self.envelope + (1 - self.attack_coeff) * level
            else:
                self.envelope = self.release_coeff * self.envelope + (1 - self.release_coeff) * level

            # Calculate gain reduction with soft knee
            level_db = self._linear_to_db(self.envelope + 1e-10)

            if level_db < self.threshold_db - self.knee_db / 2:
                # Below knee - no compression
                gain_db = 0.0
            elif level_db > self.threshold_db + self.knee_db / 2:
                # Above knee - full compression
                gain_db = self.threshold_db + (level_db - self.threshold_db) / self.ratio - level_db
            else:
                # In knee - smooth transition
                knee_factor = (level_db - self.threshold_db + self.knee_db / 2) / self.knee_db
                gain_db = knee_factor * knee_factor * (1 / self.ratio - 1) * self.knee_db / 2

            # Apply gain
            gain = self._db_to_linear(gain_db) * self.makeup_gain
            output[i] = sample * gain

        return output

    @staticmethod
    def _db_to_linear(db: float) -> float:
        return 10 ** (db / 20)

    @staticmethod
    def _linear_to_db(linear: float) -> float:
        return 20 * np.log10(max(linear, 1e-10))


class Limiter:
    """Brick-wall limiter to prevent clipping."""

    def __init__(self, threshold_db: float, release_ms: float, sample_rate: int):
        self.threshold = 10 ** (threshold_db / 20)
        self.release_coeff = np.exp(-1.0 / (release_ms * sample_rate / 1000))
        self.gain = 1.0

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Apply limiting to audio."""
        output = np.zeros_like(audio)

        for i, sample in enumerate(audio):
            # Calculate required gain
            level = abs(sample)
            if level > self.threshold:
                target_gain = self.threshold / level
            else:
                target_gain = 1.0

            # Smooth gain changes
            if target_gain < self.gain:
                self.gain = target_gain  # Instant attack
            else:
                self.gain = self.release_coeff * self.gain + (1 - self.release_coeff) * target_gain

            output[i] = sample * self.gain

        return output


class SimpleReverb:
    """Simple Schroeder reverb for subtle room ambience."""

    def __init__(
        self,
        room_size: float,
        damping: float,
        wet_level: float,
        dry_level: float,
        sample_rate: int
    ):
        self.wet = wet_level
        self.dry = dry_level
        self.sample_rate = sample_rate

        # Comb filter delays (in samples)
        base_delays = [1557, 1617, 1491, 1422, 1277, 1356, 1188, 1116]
        self.comb_delays = [int(d * room_size * sample_rate / 44100) for d in base_delays]
        self.comb_buffers = [np.zeros(d) for d in self.comb_delays]
        self.comb_indices = [0] * len(self.comb_delays)
        self.comb_feedback = [0.84 - 0.1 * damping] * len(self.comb_delays)
        self.comb_filters = [0.0] * len(self.comb_delays)
        self.comb_damping = damping

        # All-pass filter delays
        allpass_delays = [225, 556, 441, 341]
        self.allpass_delays = [int(d * sample_rate / 44100) for d in allpass_delays]
        self.allpass_buffers = [np.zeros(d) for d in self.allpass_delays]
        self.allpass_indices = [0] * len(self.allpass_delays)
        self.allpass_feedback = 0.5

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Apply reverb to audio."""
        output = np.zeros_like(audio)

        for i, sample in enumerate(audio):
            # Parallel comb filters
            comb_sum = 0.0
            for j in range(len(self.comb_delays)):
                buf = self.comb_buffers[j]
                idx = self.comb_indices[j]
                delay = self.comb_delays[j]

                # Read from buffer
                delayed = buf[idx]

                # Low-pass filter in feedback
                self.comb_filters[j] = delayed * (1 - self.comb_damping) + self.comb_filters[j] * self.comb_damping

                # Write to buffer
                buf[idx] = sample + self.comb_filters[j] * self.comb_feedback[j]

                # Update index
                self.comb_indices[j] = (idx + 1) % delay

                comb_sum += delayed

            comb_sum /= len(self.comb_delays)

            # Series all-pass filters
            allpass_out = comb_sum
            for j in range(len(self.allpass_delays)):
                buf = self.allpass_buffers[j]
                idx = self.allpass_indices[j]
                delay = self.allpass_delays[j]

                delayed = buf[idx]
                buf[idx] = allpass_out + delayed * self.allpass_feedback
                allpass_out = delayed - allpass_out * self.allpass_feedback

                self.allpass_indices[j] = (idx + 1) % delay

            # Mix dry and wet
            output[i] = sample * self.dry + allpass_out * self.wet

        return output


class HighPassFilter:
    """Simple high-pass filter to remove rumble."""

    def __init__(self, cutoff_hz: float, sample_rate: int):
        # RC filter coefficient
        rc = 1.0 / (2 * np.pi * cutoff_hz)
        dt = 1.0 / sample_rate
        self.alpha = rc / (rc + dt)
        self.prev_input = 0.0
        self.prev_output = 0.0

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Apply high-pass filter to audio."""
        output = np.zeros_like(audio)

        for i, sample in enumerate(audio):
            output[i] = self.alpha * (self.prev_output + sample - self.prev_input)
            self.prev_input = sample
            self.prev_output = output[i]

        return output


class DeEsser:
    """De-esser to reduce sibilance."""

    def __init__(
        self,
        threshold_db: float,
        center_freq: float,
        reduction_db: float,
        sample_rate: int
    ):
        self.threshold = 10 ** (threshold_db / 20)
        self.reduction = 10 ** (-reduction_db / 20)

        # Band-pass filter coefficients for sibilance detection
        # Simple 2nd order bandpass around center_freq
        w0 = 2 * np.pi * center_freq / sample_rate
        Q = 2.0
        alpha = np.sin(w0) / (2 * Q)

        b0 = alpha
        b1 = 0
        b2 = -alpha
        a0 = 1 + alpha
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha

        self.b = np.array([b0/a0, b1/a0, b2/a0])
        self.a = np.array([1, a1/a0, a2/a0])

        self.x_hist = [0.0, 0.0]
        self.y_hist = [0.0, 0.0]
        self.envelope = 0.0

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Apply de-essing to audio."""
        output = np.zeros_like(audio)

        for i, sample in enumerate(audio):
            # Band-pass filter for detection
            bp = (self.b[0] * sample +
                  self.b[1] * self.x_hist[0] +
                  self.b[2] * self.x_hist[1] -
                  self.a[1] * self.y_hist[0] -
                  self.a[2] * self.y_hist[1])

            # Update history
            self.x_hist[1] = self.x_hist[0]
            self.x_hist[0] = sample
            self.y_hist[1] = self.y_hist[0]
            self.y_hist[0] = bp

            # Envelope follower
            self.envelope = 0.99 * self.envelope + 0.01 * abs(bp)

            # Apply reduction if above threshold
            if self.envelope > self.threshold:
                gain = self.reduction
            else:
                gain = 1.0

            output[i] = sample * gain

        return output


class AudioMastering:
    """Complete audio mastering chain for TTS output."""

    def __init__(self, config: MasteringConfig = None):
        """Initialize mastering chain.

        Args:
            config: Mastering configuration
        """
        self.config = config or MasteringConfig()
        self._init_processors()

    def _init_processors(self) -> None:
        """Initialize audio processors."""
        sr = self.config.sample_rate

        # High-pass filter
        if self.config.enable_highpass:
            self.highpass = HighPassFilter(
                cutoff_hz=self.config.highpass_frequency,
                sample_rate=sr
            )
        else:
            self.highpass = None

        # De-esser
        if self.config.enable_deesser:
            self.deesser = DeEsser(
                threshold_db=self.config.deesser_threshold_db,
                center_freq=self.config.deesser_frequency,
                reduction_db=self.config.deesser_reduction_db,
                sample_rate=sr
            )
        else:
            self.deesser = None

        # Compressor
        if self.config.enable_compressor:
            self.compressor = Compressor(
                threshold_db=self.config.comp_threshold_db,
                ratio=self.config.comp_ratio,
                attack_ms=self.config.comp_attack_ms,
                release_ms=self.config.comp_release_ms,
                knee_db=self.config.comp_knee_db,
                makeup_gain_db=self.config.comp_makeup_gain_db,
                sample_rate=sr
            )
        else:
            self.compressor = None

        # Reverb
        if self.config.enable_reverb:
            self.reverb = SimpleReverb(
                room_size=self.config.reverb_room_size,
                damping=self.config.reverb_damping,
                wet_level=self.config.reverb_wet_level,
                dry_level=self.config.reverb_dry_level,
                sample_rate=sr
            )
        else:
            self.reverb = None

        # Limiter
        if self.config.enable_limiter:
            self.limiter = Limiter(
                threshold_db=self.config.limiter_threshold_db,
                release_ms=self.config.limiter_release_ms,
                sample_rate=sr
            )
        else:
            self.limiter = None

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Apply full mastering chain to audio.

        Signal flow:
        Input -> High-pass -> De-esser -> Compressor -> Reverb -> Limiter -> Normalize -> Output

        Args:
            audio: Input audio as numpy array

        Returns:
            Mastered audio as numpy array
        """
        if audio is None or len(audio) == 0:
            return audio

        # Ensure float32
        audio = audio.astype(np.float32)

        logger.debug(f"Mastering {len(audio)/self.config.sample_rate:.2f}s audio")

        # Processing chain
        processed = audio.copy()

        # 1. High-pass filter (remove rumble)
        if self.highpass:
            processed = self.highpass.process(processed)
            logger.debug("Applied high-pass filter")

        # 2. De-esser (reduce sibilance)
        if self.deesser:
            processed = self.deesser.process(processed)
            logger.debug("Applied de-esser")

        # 3. Compressor (dynamic range control)
        if self.compressor:
            processed = self.compressor.process(processed)
            logger.debug("Applied compressor")

        # 4. Reverb (subtle room ambience)
        if self.reverb:
            processed = self.reverb.process(processed)
            logger.debug("Applied reverb")

        # 5. Limiter (prevent clipping)
        if self.limiter:
            processed = self.limiter.process(processed)
            logger.debug("Applied limiter")

        # 6. Normalization
        if self.config.normalize:
            processed = self._normalize(processed)
            logger.debug("Applied normalization")

        return processed

    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to target loudness."""
        # Simple peak normalization for now
        # TODO: Implement proper LUFS normalization

        peak = np.max(np.abs(audio))
        if peak > 0:
            target_peak = 10 ** (self.config.target_loudness_db / 20)
            audio = audio * (target_peak / peak)

        return audio

    def process_file(self, input_path: Path, output_path: Path = None) -> Path:
        """Process an audio file through the mastering chain.

        Args:
            input_path: Input audio file path
            output_path: Output path (defaults to input with _mastered suffix)

        Returns:
            Path to mastered audio file
        """
        input_path = Path(input_path)

        if output_path is None:
            output_path = input_path.with_stem(input_path.stem + "_mastered")
        output_path = Path(output_path)

        try:
            import soundfile as sf

            # Load audio
            audio, sr = sf.read(str(input_path))

            # Update sample rate if different
            if sr != self.config.sample_rate:
                logger.warning(f"Sample rate mismatch: {sr} vs {self.config.sample_rate}")

            # Process
            processed = self.process(audio)

            # Save
            sf.write(str(output_path), processed, sr)

            logger.info(f"Mastered audio saved to {output_path}")
            return output_path

        except ImportError:
            logger.error("soundfile not installed. Run: pip install soundfile")
            raise
        except Exception as e:
            logger.error(f"Failed to process audio file: {e}")
            raise

    def reset(self) -> None:
        """Reset all processor states (for processing multiple files)."""
        self._init_processors()
