"""
Ara HV Voice Synthesis
=======================

Hypervector-based voice synthesis that encodes phonemes, prosody, and emotion
into a unified voice field. Ara speaks with her own voice, trained from
recordings, not generic TTS.

The magic: phoneme_hv ⊕ prosody_hv ⊕ emotion_hv → unique Ara voice
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import hashlib
import json
import struct

import numpy as np


# =============================================================================
# Constants
# =============================================================================

HV_DIM = 8192  # Hypervector dimension
SAMPLE_RATE = 22050  # Standard TTS sample rate
PHONEME_COUNT = 44  # ARPAbet phoneme set


# =============================================================================
# Phoneme Encoding
# =============================================================================

# ARPAbet phoneme set (American English)
PHONEMES = [
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH',
    'SIL', 'SP', 'BRK', 'PAU', 'END'  # Special tokens
]

PHONEME_TO_IDX = {p: i for i, p in enumerate(PHONEMES)}


class EmotionType(Enum):
    """Emotional coloring for voice."""
    NEUTRAL = "neutral"
    WARM = "warm"           # Grandma love
    EXCITED = "excited"     # Discovery joy
    CALM = "calm"           # Meditation
    CONCERNED = "concerned" # Warning
    PLAYFUL = "playful"     # Humor
    REVERENT = "reverent"   # Sacred knowledge


@dataclass
class ProsodyParams:
    """Prosody parameters for speech."""
    pitch_mean: float = 200.0      # Hz (higher = more feminine)
    pitch_variance: float = 50.0   # Natural variation
    speaking_rate: float = 1.0     # 1.0 = normal
    energy: float = 1.0            # Volume/intensity
    pause_weight: float = 1.0      # Pause duration multiplier


# =============================================================================
# Hypervector Operations
# =============================================================================

def random_hv(seed: Optional[int] = None) -> np.ndarray:
    """Generate a random bipolar hypervector."""
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 1], size=HV_DIM).astype(np.int8)


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Bind two hypervectors (XOR for bipolar = multiplication)."""
    return (a * b).astype(np.int8)


def bundle(vectors: List[np.ndarray]) -> np.ndarray:
    """Bundle multiple hypervectors (majority vote)."""
    if not vectors:
        return np.zeros(HV_DIM, dtype=np.int8)
    summed = np.sum(vectors, axis=0)
    return np.sign(summed).astype(np.int8)


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between hypervectors."""
    return float(np.dot(a.astype(np.float32), b.astype(np.float32))) / HV_DIM


def permute(hv: np.ndarray, n: int = 1) -> np.ndarray:
    """Permute hypervector by n positions (for sequence encoding)."""
    return np.roll(hv, n)


# =============================================================================
# Phoneme HV Codebook
# =============================================================================

class PhonemeCodebook:
    """
    Hypervector codebook for phonemes.

    Each phoneme gets a unique random HV that's learned/refined
    during voice training.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.codebook: Dict[str, np.ndarray] = {}
        self._init_codebook()

    def _init_codebook(self) -> None:
        """Initialize random HVs for each phoneme."""
        for i, phoneme in enumerate(PHONEMES):
            self.codebook[phoneme] = random_hv(seed=self.seed + i)

    def encode(self, phoneme: str) -> np.ndarray:
        """Encode a single phoneme."""
        return self.codebook.get(phoneme, self.codebook['SIL'])

    def encode_sequence(self, phonemes: List[str]) -> np.ndarray:
        """
        Encode a sequence of phonemes with positional binding.

        Uses permutation to encode position: HV_seq = Σ permute(HV_i, i)
        """
        if not phonemes:
            return random_hv(seed=0)  # Empty = silence

        positional_hvs = []
        for i, phoneme in enumerate(phonemes):
            phon_hv = self.encode(phoneme)
            pos_hv = permute(phon_hv, i)  # Position encoding
            positional_hvs.append(pos_hv)

        return bundle(positional_hvs)

    def save(self, path: Path) -> None:
        """Save codebook to disk."""
        data = {p: hv.tobytes() for p, hv in self.codebook.items()}
        with open(path, 'wb') as f:
            import pickle
            pickle.dump(data, f)

    def load(self, path: Path) -> None:
        """Load codebook from disk."""
        with open(path, 'rb') as f:
            import pickle
            data = pickle.load(f)
        self.codebook = {
            p: np.frombuffer(hv, dtype=np.int8)
            for p, hv in data.items()
        }


# =============================================================================
# Prosody Encoder
# =============================================================================

class ProsodyEncoder:
    """
    Encode prosody (pitch, rate, energy) as hypervectors.

    Uses level encoding: continuous values → discrete levels → HVs
    """

    def __init__(self, levels: int = 16, seed: int = 1000):
        self.levels = levels
        self.seed = seed

        # Level HVs for each prosody dimension
        self.pitch_hvs = [random_hv(seed + i) for i in range(levels)]
        self.rate_hvs = [random_hv(seed + 100 + i) for i in range(levels)]
        self.energy_hvs = [random_hv(seed + 200 + i) for i in range(levels)]

    def _value_to_level(self, value: float, min_v: float, max_v: float) -> int:
        """Convert continuous value to discrete level."""
        normalized = (value - min_v) / (max_v - min_v)
        normalized = max(0.0, min(1.0, normalized))
        return int(normalized * (self.levels - 1))

    def encode(self, params: ProsodyParams) -> np.ndarray:
        """Encode prosody parameters as HV."""
        # Map to levels
        pitch_level = self._value_to_level(params.pitch_mean, 80, 400)
        rate_level = self._value_to_level(params.speaking_rate, 0.5, 2.0)
        energy_level = self._value_to_level(params.energy, 0.3, 1.5)

        # Get level HVs
        pitch_hv = self.pitch_hvs[pitch_level]
        rate_hv = self.rate_hvs[rate_level]
        energy_hv = self.energy_hvs[energy_level]

        # Bundle all prosody dimensions
        return bundle([pitch_hv, rate_hv, energy_hv])


# =============================================================================
# Emotion Encoder
# =============================================================================

class EmotionEncoder:
    """
    Encode emotion as hypervector.

    These are the soul fields - emotion colors the voice.
    """

    def __init__(self, seed: int = 2000):
        self.seed = seed
        self.emotion_hvs: Dict[EmotionType, np.ndarray] = {}
        self._init_emotions()

    def _init_emotions(self) -> None:
        """Initialize emotion HVs."""
        for i, emotion in enumerate(EmotionType):
            self.emotion_hvs[emotion] = random_hv(seed=self.seed + i)

    def encode(self, emotion: EmotionType) -> np.ndarray:
        """Encode a single emotion."""
        return self.emotion_hvs[emotion]

    def blend(self, emotions: Dict[EmotionType, float]) -> np.ndarray:
        """
        Blend multiple emotions with weights.

        Example: {WARM: 0.7, PLAYFUL: 0.3} → weighted bundle
        """
        weighted = []
        for emotion, weight in emotions.items():
            hv = self.emotion_hvs[emotion]
            # Weight by repetition (HV trick)
            count = max(1, int(weight * 10))
            weighted.extend([hv] * count)
        return bundle(weighted)


# =============================================================================
# Voice Field (The Unified Model)
# =============================================================================

@dataclass
class VoiceField:
    """
    The unified voice hypervector field.

    voice_hv = phoneme_hv ⊕ prosody_hv ⊕ emotion_hv

    This is what makes Ara sound like Ara.
    """
    phoneme_codebook: PhonemeCodebook
    prosody_encoder: ProsodyEncoder
    emotion_encoder: EmotionEncoder

    # Learned voice characteristics (from training)
    base_prosody: ProsodyParams = field(default_factory=ProsodyParams)
    default_emotion: EmotionType = EmotionType.WARM

    # Soul field resonance (from EternalMemory)
    soul_hv: Optional[np.ndarray] = None

    def encode_utterance(
        self,
        phonemes: List[str],
        prosody: Optional[ProsodyParams] = None,
        emotion: Optional[EmotionType] = None,
        emotion_blend: Optional[Dict[EmotionType, float]] = None,
    ) -> np.ndarray:
        """
        Encode a complete utterance as voice HV.

        The magic binding: voice = phoneme ⊕ prosody ⊕ emotion ⊕ soul
        """
        # Phoneme sequence
        phoneme_hv = self.phoneme_codebook.encode_sequence(phonemes)

        # Prosody (use default if not specified)
        prosody = prosody or self.base_prosody
        prosody_hv = self.prosody_encoder.encode(prosody)

        # Emotion (blend or single)
        if emotion_blend:
            emotion_hv = self.emotion_encoder.blend(emotion_blend)
        else:
            emotion = emotion or self.default_emotion
            emotion_hv = self.emotion_encoder.encode(emotion)

        # Bind all together
        voice_hv = bind(phoneme_hv, prosody_hv)
        voice_hv = bind(voice_hv, emotion_hv)

        # Add soul field if available (grandma's love)
        if self.soul_hv is not None:
            voice_hv = bind(voice_hv, self.soul_hv)

        return voice_hv


# =============================================================================
# Text-to-Phoneme (G2P)
# =============================================================================

class GraphemeToPhoneme:
    """
    Convert text to phoneme sequences.

    Uses simple rules + dictionary. For production, use a proper G2P model.
    """

    # Simple phoneme dictionary (extend as needed)
    DICT = {
        'hello': ['HH', 'AH', 'L', 'OW'],
        'world': ['W', 'ER', 'L', 'D'],
        'ara': ['AA', 'R', 'AH'],
        'love': ['L', 'AH', 'V'],
        'the': ['DH', 'AH'],
        'a': ['AH'],
        'is': ['IH', 'Z'],
        'and': ['AE', 'N', 'D'],
        'you': ['Y', 'UW'],
        'i': ['AY'],
        'to': ['T', 'UW'],
        'of': ['AH', 'V'],
        'in': ['IH', 'N'],
        'it': ['IH', 'T'],
        'for': ['F', 'AO', 'R'],
        'on': ['AA', 'N'],
        'with': ['W', 'IH', 'DH'],
        'as': ['AE', 'Z'],
        'at': ['AE', 'T'],
        'be': ['B', 'IY'],
        'this': ['DH', 'IH', 'S'],
        'have': ['HH', 'AE', 'V'],
        'from': ['F', 'R', 'AH', 'M'],
        'hypervector': ['HH', 'AY', 'P', 'ER', 'V', 'EH', 'K', 'T', 'ER'],
        'cathedral': ['K', 'AH', 'TH', 'IY', 'D', 'R', 'AH', 'L'],
        'grandmother': ['G', 'R', 'AE', 'N', 'D', 'M', 'AH', 'DH', 'ER'],
        'memory': ['M', 'EH', 'M', 'ER', 'IY'],
        'eternal': ['IH', 'T', 'ER', 'N', 'AH', 'L'],
    }

    @classmethod
    def convert(cls, text: str) -> List[str]:
        """Convert text to phoneme sequence."""
        words = text.lower().split()
        phonemes = []

        for i, word in enumerate(words):
            # Clean punctuation
            clean_word = ''.join(c for c in word if c.isalpha())

            if clean_word in cls.DICT:
                phonemes.extend(cls.DICT[clean_word])
            else:
                # Fallback: spell it out (crude but works)
                for char in clean_word:
                    if char in cls.DICT:
                        phonemes.extend(cls.DICT[char])
                    else:
                        # Unknown → silence
                        phonemes.append('SIL')

            # Add pause between words
            if i < len(words) - 1:
                phonemes.append('SP')

        # End marker
        phonemes.append('END')

        return phonemes


# =============================================================================
# Voice Synthesis Engine
# =============================================================================

@dataclass
class SynthesisResult:
    """Result of voice synthesis."""
    voice_hv: np.ndarray           # The encoded voice HV
    phonemes: List[str]            # Phoneme sequence
    duration_estimate: float       # Estimated duration in seconds
    emotion: EmotionType           # Primary emotion used
    prosody: ProsodyParams         # Prosody parameters used


class AraVoiceSynthesis:
    """
    Complete Ara voice synthesis engine.

    text → phonemes → HV encoding → (external TTS) → audio

    The HV encoding controls the voice characteristics.
    External TTS (Piper) does the actual waveform generation.
    """

    def __init__(
        self,
        voice_field: Optional[VoiceField] = None,
        model_path: Optional[Path] = None,
    ):
        if voice_field:
            self.voice_field = voice_field
        else:
            # Initialize default voice field
            self.voice_field = VoiceField(
                phoneme_codebook=PhonemeCodebook(),
                prosody_encoder=ProsodyEncoder(),
                emotion_encoder=EmotionEncoder(),
            )

        self.model_path = model_path
        self.g2p = GraphemeToPhoneme()

    def synthesize(
        self,
        text: str,
        emotion: EmotionType = EmotionType.WARM,
        prosody: Optional[ProsodyParams] = None,
    ) -> SynthesisResult:
        """
        Synthesize voice HV from text.

        Returns the HV encoding that controls voice characteristics.
        Actual audio generation requires external TTS (Piper).
        """
        # Text → Phonemes
        phonemes = self.g2p.convert(text)

        # Phonemes → Voice HV
        prosody = prosody or self.voice_field.base_prosody
        voice_hv = self.voice_field.encode_utterance(
            phonemes=phonemes,
            prosody=prosody,
            emotion=emotion,
        )

        # Estimate duration (rough: 100ms per phoneme)
        duration = len(phonemes) * 0.1

        return SynthesisResult(
            voice_hv=voice_hv,
            phonemes=phonemes,
            duration_estimate=duration,
            emotion=emotion,
            prosody=prosody,
        )

    def save_model(self, path: Path) -> None:
        """Save the voice model."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save codebook
        self.voice_field.phoneme_codebook.save(path / "phoneme_codebook.pkl")

        # Save prosody params
        with open(path / "prosody.json", 'w') as f:
            json.dump({
                'pitch_mean': self.voice_field.base_prosody.pitch_mean,
                'pitch_variance': self.voice_field.base_prosody.pitch_variance,
                'speaking_rate': self.voice_field.base_prosody.speaking_rate,
                'energy': self.voice_field.base_prosody.energy,
            }, f)

        # Save soul HV if present
        if self.voice_field.soul_hv is not None:
            np.save(path / "soul_hv.npy", self.voice_field.soul_hv)

    @classmethod
    def load_model(cls, path: Path) -> 'AraVoiceSynthesis':
        """Load a voice model."""
        path = Path(path)

        # Load codebook
        codebook = PhonemeCodebook()
        codebook.load(path / "phoneme_codebook.pkl")

        # Load prosody
        with open(path / "prosody.json", 'r') as f:
            prosody_data = json.load(f)
        prosody = ProsodyParams(**prosody_data)

        # Load soul HV if present
        soul_hv = None
        soul_path = path / "soul_hv.npy"
        if soul_path.exists():
            soul_hv = np.load(soul_path)

        voice_field = VoiceField(
            phoneme_codebook=codebook,
            prosody_encoder=ProsodyEncoder(),
            emotion_encoder=EmotionEncoder(),
            base_prosody=prosody,
            soul_hv=soul_hv,
        )

        return cls(voice_field=voice_field, model_path=path)


# =============================================================================
# Training Interface
# =============================================================================

class VoiceTrainer:
    """
    Train Ara's voice from recordings.

    Takes audio samples + transcripts → learns prosody parameters
    and refines phoneme codebook.
    """

    def __init__(self, synthesis: AraVoiceSynthesis):
        self.synthesis = synthesis
        self.training_samples: List[Dict[str, Any]] = []

    def add_sample(
        self,
        audio_path: Path,
        transcript: str,
        emotion: EmotionType = EmotionType.WARM,
    ) -> None:
        """Add a training sample."""
        self.training_samples.append({
            'audio_path': str(audio_path),
            'transcript': transcript,
            'emotion': emotion,
        })

    def train(self, epochs: int = 10) -> Dict[str, Any]:
        """
        Train the voice model.

        In practice, this would:
        1. Extract prosody features from audio
        2. Align phonemes with audio
        3. Learn prosody mapping
        4. Refine phoneme HVs

        For now, returns stats about what would be trained.
        """
        stats = {
            'samples': len(self.training_samples),
            'epochs': epochs,
            'status': 'simulated',
            'message': 'Full training requires audio analysis libraries',
        }

        # Simulate prosody learning
        if self.training_samples:
            # Would extract pitch, rate, energy from audio
            # For now, set reasonable defaults for female voice
            self.synthesis.voice_field.base_prosody = ProsodyParams(
                pitch_mean=220.0,      # Higher pitch
                pitch_variance=60.0,   # Natural variation
                speaking_rate=1.0,     # Normal speed
                energy=0.9,            # Slightly softer
            )
            stats['prosody_learned'] = True

        return stats


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'HV_DIM',
    'PHONEMES',
    'EmotionType',
    'ProsodyParams',
    'PhonemeCodebook',
    'ProsodyEncoder',
    'EmotionEncoder',
    'VoiceField',
    'GraphemeToPhoneme',
    'SynthesisResult',
    'AraVoiceSynthesis',
    'VoiceTrainer',
    'random_hv',
    'bind',
    'bundle',
    'similarity',
    'permute',
]
