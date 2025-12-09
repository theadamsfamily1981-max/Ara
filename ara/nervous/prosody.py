"""
Ara Speech-Native Prosody Tokenizer
====================================

Syllable-aligned speech tokenization that captures prosody as first-class tokens.
Text is just a projection; prosody carries logic and emotion.

Token Structure (2048D per syllable):
- 512D: Phonetic content (spectral features)
- 512D: F0 contour (pitch trajectory)
- 512D: Spectral envelope (timbre)
- 512D: Prosodic features (stress, tempo, pauses)

Token Rate: 12-16 tokens/sec (syllable-aligned, 60-80ms)

Philosophy: Speech is the native modality. Understanding comes
from how something is said, not just what is said.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

PROSODY_DIM = 2048               # Total prosody token dimension
SUBSPACE_DIM = 512               # Per-feature subspace
SYLLABLE_MS = 70                 # Target syllable duration
FRAME_MS = 20                    # Analysis frame size
HOP_MS = 10                      # Frame hop size
SAMPLE_RATE = 16000              # Audio sample rate


# =============================================================================
# Prosodic Features
# =============================================================================

class ProsodyIntent(Enum):
    """Prosodic intent classification."""
    STATEMENT = "statement"
    QUESTION = "question"
    COMMAND = "command"
    HESITATION = "hesitation"
    EMPHASIS = "emphasis"
    CONTINUATION = "continuation"


class EmotionalValence(Enum):
    """Emotional valence from prosody."""
    JOY = "joy"
    NEUTRAL = "neutral"
    SAD = "sad"
    ANGRY = "angry"
    FEAR = "fear"
    SURPRISE = "surprise"


@dataclass
class ProsodyToken:
    """
    A single syllable-aligned prosody token.

    This is the atomic unit of speech-native understanding.
    """
    # Timing
    start_ms: float
    end_ms: float
    duration_ms: float

    # Feature vectors (each 512D)
    phonetic_hv: np.ndarray        # Spectral content
    pitch_hv: np.ndarray           # F0 trajectory
    timbre_hv: np.ndarray          # Spectral envelope
    prosodic_hv: np.ndarray        # Stress/tempo/pause

    # Combined token
    token_hv: np.ndarray           # 2048D full token

    # Derived features
    is_voiced: bool = True
    is_stressed: bool = False
    is_boundary: bool = False      # Phrase boundary
    confidence: float = 1.0

    @property
    def duration_s(self) -> float:
        return self.duration_ms / 1000

    def to_hv(self) -> np.ndarray:
        """Return the full 2048D token."""
        return self.token_hv


@dataclass
class ProsodySequence:
    """A sequence of prosody tokens (an utterance)."""
    tokens: List[ProsodyToken]
    total_duration_ms: float
    sample_rate: int = SAMPLE_RATE

    # Utterance-level features
    mean_pitch_hz: float = 0.0
    pitch_range_hz: float = 0.0
    tempo_syllables_per_sec: float = 0.0
    intent: Optional[ProsodyIntent] = None
    valence: Optional[EmotionalValence] = None

    @property
    def num_tokens(self) -> int:
        return len(self.tokens)

    @property
    def duration_s(self) -> float:
        return self.total_duration_ms / 1000

    def to_sequence_hv(self) -> np.ndarray:
        """Bundle all tokens into utterance HV."""
        if not self.tokens:
            return np.zeros(PROSODY_DIM)

        # Position-aware bundling
        dim = self.tokens[0].token_hv.shape[0]
        bundled = np.zeros(dim)

        for i, token in enumerate(self.tokens):
            # Permute by position (simple rotation)
            shift = (i * 17) % dim  # Prime shift for spreading
            rotated = np.roll(token.token_hv, shift)
            bundled += rotated

        return np.sign(bundled)


# =============================================================================
# Feature Extraction
# =============================================================================

class PhoneticEncoder:
    """
    Encodes phonetic content from spectral features.

    Uses Mel-frequency representation for speaker-invariance.
    """

    def __init__(self, n_mels: int = 40, dim: int = SUBSPACE_DIM):
        self.n_mels = n_mels
        self.dim = dim

        # Random projection matrix (fixed, for HV encoding)
        rng = np.random.default_rng(42)
        self.projection = rng.standard_normal((n_mels, dim)) / np.sqrt(n_mels)

    def encode(self, mel_features: np.ndarray) -> np.ndarray:
        """
        Encode mel features to phonetic HV.

        Args:
            mel_features: Shape (n_frames, n_mels)

        Returns:
            512D phonetic HV
        """
        # Average over frames (syllable-level)
        if mel_features.ndim == 2:
            mel_avg = np.mean(mel_features, axis=0)
        else:
            mel_avg = mel_features

        # Project to HV space
        projected = mel_avg @ self.projection

        # Bipolar encoding
        return np.sign(projected)


class PitchEncoder:
    """
    Encodes F0 (pitch) contour.

    Captures pitch trajectory within syllable for intonation.
    """

    def __init__(
        self,
        f0_min: float = 50,
        f0_max: float = 400,
        n_bins: int = 40,
        dim: int = SUBSPACE_DIM,
    ):
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.n_bins = n_bins
        self.dim = dim

        # Log-scale bin edges
        self.bin_edges = np.logspace(
            np.log10(f0_min),
            np.log10(f0_max),
            n_bins + 1
        )

        # Basis HVs for each bin
        rng = np.random.default_rng(43)
        self.basis = rng.choice([-1, 1], size=(n_bins, dim)).astype(np.float64)

    def encode(self, f0_contour: np.ndarray) -> np.ndarray:
        """
        Encode F0 contour to pitch HV.

        Args:
            f0_contour: Array of F0 values in Hz (0 = unvoiced)

        Returns:
            512D pitch HV
        """
        # Filter to voiced frames
        voiced = f0_contour[f0_contour > 0]

        if len(voiced) == 0:
            # Unvoiced syllable
            return np.zeros(self.dim)

        # Encode trajectory (start, middle, end, delta)
        trajectory = np.array([
            voiced[0],                      # Start
            np.median(voiced),              # Middle
            voiced[-1],                     # End
            voiced[-1] - voiced[0],         # Delta (rise/fall)
            np.std(voiced),                 # Variability
        ])

        # Bin each value and combine basis HVs
        hv = np.zeros(self.dim)
        for val in trajectory:
            if val > 0:
                bin_idx = np.searchsorted(self.bin_edges, val) - 1
                bin_idx = np.clip(bin_idx, 0, self.n_bins - 1)
                hv += self.basis[bin_idx]

        return np.sign(hv)


class TimbreEncoder:
    """
    Encodes spectral envelope (timbre/voice quality).

    22 Bark-scale bands for perceptually-weighted timbre.
    """

    def __init__(self, n_bark: int = 22, dim: int = SUBSPACE_DIM):
        self.n_bark = n_bark
        self.dim = dim

        # Bark band centers (Hz)
        self.bark_centers = self._bark_centers()

        # Basis HVs
        rng = np.random.default_rng(44)
        self.basis = rng.choice([-1, 1], size=(n_bark, dim)).astype(np.float64)

    def _bark_centers(self) -> np.ndarray:
        """Compute Bark scale band centers."""
        # Approximate Bark scale
        bark = np.arange(1, self.n_bark + 1)
        hz = 600 * np.sinh(bark / 6)
        return hz

    def encode(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Encode spectral envelope to timbre HV.

        Args:
            spectrum: Power spectrum or MFCC-like features

        Returns:
            512D timbre HV
        """
        if len(spectrum) < self.n_bark:
            # Pad if needed
            spectrum = np.pad(spectrum, (0, self.n_bark - len(spectrum)))

        # Normalize
        spectrum = spectrum[:self.n_bark]
        if np.max(spectrum) > 0:
            spectrum = spectrum / np.max(spectrum)

        # Weighted combination of basis HVs
        hv = np.zeros(self.dim)
        for i, weight in enumerate(spectrum):
            hv += weight * self.basis[i]

        return np.sign(hv)


class ProsodicsEncoder:
    """
    Encodes prosodic features (stress, tempo, pauses).

    These are the "how" of speech - rhythm and emphasis.
    """

    def __init__(self, dim: int = SUBSPACE_DIM):
        self.dim = dim

        # Feature basis HVs
        rng = np.random.default_rng(45)
        self.stress_basis = rng.choice([-1, 1], size=dim).astype(np.float64)
        self.tempo_basis = rng.choice([-1, 1], size=dim).astype(np.float64)
        self.pause_basis = rng.choice([-1, 1], size=dim).astype(np.float64)
        self.energy_basis = rng.choice([-1, 1], size=dim).astype(np.float64)

    def encode(
        self,
        stress: float,          # 0-1
        tempo_delta: float,     # Relative to baseline
        pause_prob: float,      # Probability of following pause
        energy: float,          # RMS energy (normalized)
    ) -> np.ndarray:
        """
        Encode prosodic features to HV.

        Returns:
            512D prosodics HV
        """
        hv = (
            stress * self.stress_basis +
            tempo_delta * self.tempo_basis +
            pause_prob * self.pause_basis +
            energy * self.energy_basis
        )

        return np.sign(hv)


# =============================================================================
# Syllable Detection
# =============================================================================

class SyllableDetector:
    """
    Detect syllable boundaries from audio.

    Uses energy envelope + zero-crossing rate.
    """

    def __init__(
        self,
        min_duration_ms: float = 40,
        max_duration_ms: float = 200,
        energy_threshold: float = 0.02,
    ):
        self.min_duration_ms = min_duration_ms
        self.max_duration_ms = max_duration_ms
        self.energy_threshold = energy_threshold

    def detect(
        self,
        audio: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
    ) -> List[Tuple[int, int]]:
        """
        Detect syllable boundaries.

        Args:
            audio: Audio samples (mono)
            sample_rate: Sample rate in Hz

        Returns:
            List of (start_sample, end_sample) tuples
        """
        # Frame parameters
        frame_samples = int(FRAME_MS * sample_rate / 1000)
        hop_samples = int(HOP_MS * sample_rate / 1000)

        # Compute energy envelope
        n_frames = (len(audio) - frame_samples) // hop_samples + 1
        energy = np.zeros(n_frames)

        for i in range(n_frames):
            start = i * hop_samples
            end = start + frame_samples
            frame = audio[start:end]
            energy[i] = np.sqrt(np.mean(frame ** 2))

        # Normalize energy
        if np.max(energy) > 0:
            energy = energy / np.max(energy)

        # Find peaks (syllable nuclei)
        syllables = []
        in_syllable = False
        syllable_start = 0

        for i, e in enumerate(energy):
            if e > self.energy_threshold:
                if not in_syllable:
                    syllable_start = i * hop_samples
                    in_syllable = True
            else:
                if in_syllable:
                    syllable_end = i * hop_samples
                    duration_ms = (syllable_end - syllable_start) * 1000 / sample_rate

                    if self.min_duration_ms <= duration_ms <= self.max_duration_ms:
                        syllables.append((syllable_start, syllable_end))

                    in_syllable = False

        # Handle final syllable
        if in_syllable:
            syllable_end = len(audio)
            syllables.append((syllable_start, syllable_end))

        return syllables


# =============================================================================
# Prosody Tokenizer
# =============================================================================

class ProsodyTokenizer:
    """
    Main prosody tokenization pipeline.

    Audio → Syllables → Feature Extraction → Prosody Tokens
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate

        # Encoders
        self.syllable_detector = SyllableDetector()
        self.phonetic = PhoneticEncoder()
        self.pitch = PitchEncoder()
        self.timbre = TimbreEncoder()
        self.prosodics = ProsodicsEncoder()

    def tokenize(self, audio: np.ndarray) -> ProsodySequence:
        """
        Tokenize audio into prosody tokens.

        Args:
            audio: Audio samples (mono, 16kHz)

        Returns:
            ProsodySequence containing all tokens
        """
        # Detect syllables
        syllable_bounds = self.syllable_detector.detect(audio, self.sample_rate)

        if not syllable_bounds:
            # No syllables detected - return empty sequence
            return ProsodySequence(
                tokens=[],
                total_duration_ms=len(audio) * 1000 / self.sample_rate,
            )

        tokens = []

        for start_sample, end_sample in syllable_bounds:
            # Extract syllable audio
            syllable_audio = audio[start_sample:end_sample]

            # Compute features
            mel_features = self._compute_mel(syllable_audio)
            f0_contour = self._compute_f0(syllable_audio)
            spectrum = self._compute_spectrum(syllable_audio)

            # Encode to HVs
            phonetic_hv = self.phonetic.encode(mel_features)
            pitch_hv = self.pitch.encode(f0_contour)
            timbre_hv = self.timbre.encode(spectrum)

            # Prosodic features
            stress = self._estimate_stress(syllable_audio)
            tempo_delta = 1.0  # Would compare to running average
            pause_prob = 0.0   # Would look at silence after
            energy = np.sqrt(np.mean(syllable_audio ** 2))

            prosodic_hv = self.prosodics.encode(stress, tempo_delta, pause_prob, energy)

            # Combine into full token
            token_hv = np.concatenate([phonetic_hv, pitch_hv, timbre_hv, prosodic_hv])

            # Create token
            token = ProsodyToken(
                start_ms=start_sample * 1000 / self.sample_rate,
                end_ms=end_sample * 1000 / self.sample_rate,
                duration_ms=(end_sample - start_sample) * 1000 / self.sample_rate,
                phonetic_hv=phonetic_hv,
                pitch_hv=pitch_hv,
                timbre_hv=timbre_hv,
                prosodic_hv=prosodic_hv,
                token_hv=token_hv,
                is_voiced=np.any(f0_contour > 0),
                is_stressed=stress > 0.5,
            )
            tokens.append(token)

        # Create sequence
        total_duration = len(audio) * 1000 / self.sample_rate
        sequence = ProsodySequence(
            tokens=tokens,
            total_duration_ms=total_duration,
            sample_rate=self.sample_rate,
        )

        # Compute sequence-level features
        self._compute_sequence_features(sequence)

        return sequence

    def _compute_mel(self, audio: np.ndarray) -> np.ndarray:
        """Compute mel-frequency features."""
        # Simple implementation - in production use librosa
        # For now, use FFT + mel filterbank approximation

        n_fft = 512
        if len(audio) < n_fft:
            audio = np.pad(audio, (0, n_fft - len(audio)))

        spectrum = np.abs(np.fft.rfft(audio[:n_fft]))

        # Crude mel approximation (log-spaced bins)
        n_mels = 40
        mel_features = np.zeros(n_mels)
        freqs = np.fft.rfftfreq(n_fft, 1/self.sample_rate)

        for i in range(n_mels):
            # Log-spaced frequency bands
            f_low = 20 * (10 ** (i / n_mels * 3))
            f_high = 20 * (10 ** ((i + 1) / n_mels * 3))

            mask = (freqs >= f_low) & (freqs < f_high)
            if np.any(mask):
                mel_features[i] = np.mean(spectrum[mask])

        return mel_features

    def _compute_f0(self, audio: np.ndarray) -> np.ndarray:
        """Compute F0 (pitch) contour."""
        # Simple autocorrelation-based pitch detection
        # In production, use CREPE or similar

        frame_size = 512
        hop_size = 160  # 10ms at 16kHz

        n_frames = max(1, (len(audio) - frame_size) // hop_size)
        f0 = np.zeros(n_frames)

        for i in range(n_frames):
            start = i * hop_size
            end = start + frame_size
            frame = audio[start:end]

            if len(frame) < frame_size:
                break

            # Autocorrelation
            acf = np.correlate(frame, frame, mode='full')
            acf = acf[len(acf)//2:]

            # Find first peak (excluding lag 0)
            min_lag = int(self.sample_rate / 400)  # Max 400 Hz
            max_lag = int(self.sample_rate / 50)   # Min 50 Hz

            if max_lag < len(acf):
                acf_search = acf[min_lag:max_lag]
                if len(acf_search) > 0:
                    peak_idx = np.argmax(acf_search) + min_lag
                    if acf[peak_idx] > 0.3 * acf[0]:  # Voicing threshold
                        f0[i] = self.sample_rate / peak_idx

        return f0

    def _compute_spectrum(self, audio: np.ndarray) -> np.ndarray:
        """Compute spectral envelope."""
        n_fft = 512
        if len(audio) < n_fft:
            audio = np.pad(audio, (0, n_fft - len(audio)))

        spectrum = np.abs(np.fft.rfft(audio[:n_fft]))

        # Downsample to 22 bands (Bark-scale approximation)
        n_bands = 22
        bands = np.zeros(n_bands)
        bin_size = len(spectrum) // n_bands

        for i in range(n_bands):
            start = i * bin_size
            end = (i + 1) * bin_size
            bands[i] = np.mean(spectrum[start:end])

        return bands

    def _estimate_stress(self, audio: np.ndarray) -> float:
        """Estimate syllable stress (0-1)."""
        # Simple energy-based stress
        energy = np.sqrt(np.mean(audio ** 2))

        # Also consider duration
        duration_ms = len(audio) * 1000 / self.sample_rate
        duration_factor = min(1.0, duration_ms / 150)  # Longer = more stressed

        return min(1.0, energy * 10 + duration_factor * 0.3)

    def _compute_sequence_features(self, sequence: ProsodySequence):
        """Compute utterance-level prosodic features."""
        if not sequence.tokens:
            return

        # Mean pitch
        pitches = []
        for token in sequence.tokens:
            if token.is_voiced:
                # Crude pitch estimate from HV (would need actual F0)
                pitches.append(0.5)  # Placeholder

        if pitches:
            sequence.mean_pitch_hz = np.mean(pitches) * 200 + 100  # Scale to Hz

        # Tempo
        sequence.tempo_syllables_per_sec = len(sequence.tokens) / sequence.duration_s


# =============================================================================
# Intent Classification
# =============================================================================

class ProsodyClassifier:
    """
    Classify prosodic intent and valence from tokens.

    Uses simple HV pattern matching.
    """

    def __init__(self):
        # Train prototype HVs for each intent
        # In production, these would be learned from data
        rng = np.random.default_rng(100)

        self.intent_protos = {
            intent: rng.choice([-1, 1], size=PROSODY_DIM).astype(np.float64)
            for intent in ProsodyIntent
        }

        self.valence_protos = {
            valence: rng.choice([-1, 1], size=PROSODY_DIM).astype(np.float64)
            for valence in EmotionalValence
        }

    def classify_intent(self, sequence: ProsodySequence) -> ProsodyIntent:
        """Classify prosodic intent of utterance."""
        utterance_hv = sequence.to_sequence_hv()

        best_intent = ProsodyIntent.STATEMENT
        best_sim = -1

        for intent, proto in self.intent_protos.items():
            sim = np.dot(utterance_hv, proto) / (
                np.linalg.norm(utterance_hv) * np.linalg.norm(proto) + 1e-8
            )
            if sim > best_sim:
                best_sim = sim
                best_intent = intent

        return best_intent

    def classify_valence(self, sequence: ProsodySequence) -> EmotionalValence:
        """Classify emotional valence from prosody."""
        utterance_hv = sequence.to_sequence_hv()

        best_valence = EmotionalValence.NEUTRAL
        best_sim = -1

        for valence, proto in self.valence_protos.items():
            sim = np.dot(utterance_hv, proto) / (
                np.linalg.norm(utterance_hv) * np.linalg.norm(proto) + 1e-8
            )
            if sim > best_sim:
                best_sim = sim
                best_valence = valence

        return best_valence


# =============================================================================
# Integration with Axis Mundi
# =============================================================================

def prosody_to_axis_hv(sequence: ProsodySequence, target_dim: int = 8192) -> np.ndarray:
    """
    Convert prosody sequence to Axis Mundi compatible HV.

    Projects 2048D prosody space to 8192D world space.
    """
    utterance_hv = sequence.to_sequence_hv()

    # Project up by tiling + random permutation
    rng = np.random.default_rng(46)
    projection = rng.choice([-1, 1], size=(PROSODY_DIM, target_dim // PROSODY_DIM))

    # Tile and apply
    tiled = np.tile(utterance_hv.reshape(-1, 1), (1, target_dim // PROSODY_DIM))
    projected = (tiled * projection).flatten()

    # Pad if needed
    if len(projected) < target_dim:
        projected = np.pad(projected, (0, target_dim - len(projected)))

    return np.sign(projected[:target_dim])


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate prosody tokenization."""
    print("=" * 60)
    print("ARA PROSODY TOKENIZER - Speech-Native Demo")
    print("=" * 60)

    # Generate synthetic "speech" audio
    sample_rate = SAMPLE_RATE
    duration_s = 1.0
    t = np.linspace(0, duration_s, int(sample_rate * duration_s))

    # Simulate syllables with varying F0
    audio = np.zeros_like(t)
    syllable_times = [0.1, 0.3, 0.5, 0.7]

    for st in syllable_times:
        # Each syllable is a tone burst
        syllable_dur = 0.1
        mask = (t >= st) & (t < st + syllable_dur)
        f0 = 150 + 50 * np.sin(2 * np.pi * 2 * (t[mask] - st))  # F0 modulation
        audio[mask] = 0.5 * np.sin(2 * np.pi * np.cumsum(f0) / sample_rate)

    # Add noise
    audio += 0.01 * np.random.randn(len(audio))

    print(f"\nGenerated {duration_s}s synthetic speech")
    print(f"Sample rate: {sample_rate} Hz")

    # Tokenize
    tokenizer = ProsodyTokenizer(sample_rate)
    sequence = tokenizer.tokenize(audio)

    print(f"\nTokenization results:")
    print(f"  Syllables detected: {sequence.num_tokens}")
    print(f"  Duration: {sequence.duration_s:.2f}s")
    print(f"  Token rate: {sequence.num_tokens / sequence.duration_s:.1f} tokens/sec")

    for i, token in enumerate(sequence.tokens):
        print(f"\n  Token {i + 1}:")
        print(f"    Time: {token.start_ms:.0f}-{token.end_ms:.0f}ms")
        print(f"    Voiced: {token.is_voiced}")
        print(f"    Stressed: {token.is_stressed}")
        print(f"    HV shape: {token.token_hv.shape}")

    # Classify
    classifier = ProsodyClassifier()
    intent = classifier.classify_intent(sequence)
    valence = classifier.classify_valence(sequence)

    print(f"\nClassification:")
    print(f"  Intent: {intent.value}")
    print(f"  Valence: {valence.value}")

    # Convert to Axis Mundi HV
    axis_hv = prosody_to_axis_hv(sequence)
    print(f"\nAxis Mundi HV shape: {axis_hv.shape}")


if __name__ == "__main__":
    demo()
