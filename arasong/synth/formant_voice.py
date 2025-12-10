#!/usr/bin/env python3
"""
AraSong Synthesizer - Formant-Based Vocal Synthesis
====================================================

Production-grade vocal synthesis using formant filter banks.
Uses IIR biquad filters for vowel shaping without scipy dependency.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .numpy_osc import SineOsc, PulseOsc, NoiseOsc, note_to_freq
from .dynamics import EmotionParams, get_emotion_params, generate_vibrato, generate_pitch_drift


# =============================================================================
# Formant Data
# =============================================================================

@dataclass
class FormantSet:
    """Formant frequencies and bandwidths for a vowel."""
    f1: float  # First formant (Hz)
    f2: float  # Second formant (Hz)
    f3: float  # Third formant (Hz)
    bw1: float = 80.0   # Bandwidth of F1
    bw2: float = 100.0  # Bandwidth of F2
    bw3: float = 120.0  # Bandwidth of F3
    gain1: float = 1.0  # Relative gain of F1
    gain2: float = 0.5  # Relative gain of F2
    gain3: float = 0.25 # Relative gain of F3


# Standard vowel formants (based on average adult values)
VOWEL_FORMANTS: Dict[str, FormantSet] = {
    # Open vowels
    'a': FormantSet(f1=800, f2=1200, f3=2500, gain1=1.0, gain2=0.6, gain3=0.3),   # "father"
    'ah': FormantSet(f1=750, f2=1100, f3=2400, gain1=1.0, gain2=0.55, gain3=0.28),

    # Front vowels
    'e': FormantSet(f1=400, f2=2200, f3=2800, gain1=0.9, gain2=0.5, gain3=0.25),   # "hey"
    'eh': FormantSet(f1=550, f2=1800, f3=2600, gain1=0.95, gain2=0.5, gain3=0.25), # "bed"
    'i': FormantSet(f1=300, f2=2300, f3=3000, gain1=0.8, gain2=0.45, gain3=0.2),   # "see"
    'ih': FormantSet(f1=400, f2=2000, f3=2800, gain1=0.85, gain2=0.45, gain3=0.22),# "bit"

    # Back vowels
    'o': FormantSet(f1=500, f2=800, f3=2500, gain1=1.0, gain2=0.55, gain3=0.25),   # "go"
    'oh': FormantSet(f1=600, f2=900, f3=2400, gain1=0.95, gain2=0.5, gain3=0.25),  # "law"
    'u': FormantSet(f1=350, f2=700, f3=2500, gain1=0.9, gain2=0.5, gain3=0.22),    # "you"
    'oo': FormantSet(f1=300, f2=850, f3=2400, gain1=0.85, gain2=0.45, gain3=0.2),  # "book"

    # Neutral
    'uh': FormantSet(f1=600, f2=1200, f3=2500, gain1=0.9, gain2=0.5, gain3=0.25),  # "but"
    'er': FormantSet(f1=500, f2=1400, f3=1600, gain1=0.85, gain2=0.55, gain3=0.35),# "bird"

    # Nasal approximations
    'mm': FormantSet(f1=300, f2=1000, f3=2200, bw1=120, bw2=150, bw3=150, gain1=0.6, gain2=0.3, gain3=0.15),
    'nn': FormantSet(f1=350, f2=1200, f3=2500, bw1=100, bw2=140, bw3=140, gain1=0.65, gain2=0.35, gain3=0.18),
}


def guess_vowel(syllable: str) -> str:
    """Guess the primary vowel sound in a syllable."""
    syllable = syllable.lower().strip('.,!?"\'-()[]')

    # Direct vowel patterns
    vowel_patterns = [
        ('ee', 'i'), ('ea', 'i'), ('ie', 'i'),
        ('oo', 'u'), ('ou', 'u'), ('ew', 'u'),
        ('ay', 'e'), ('ai', 'e'), ('ey', 'e'),
        ('ow', 'o'), ('oa', 'o'),
        ('er', 'er'), ('ir', 'er'), ('ur', 'er'),
        ('ar', 'ah'),
    ]

    for pattern, vowel in vowel_patterns:
        if pattern in syllable:
            return vowel

    # Find last vowel (usually the sung one)
    vowel_map = {
        'a': 'a', 'e': 'eh', 'i': 'ih', 'o': 'oh', 'u': 'uh',
    }

    for char in reversed(syllable):
        if char in vowel_map:
            return vowel_map[char]

    return 'a'  # Default


def get_formants(vowel: str) -> FormantSet:
    """Get formant set for a vowel, defaulting to 'a'."""
    return VOWEL_FORMANTS.get(vowel, VOWEL_FORMANTS['a'])


# =============================================================================
# Biquad Filter (IIR)
# =============================================================================

@dataclass
class BiquadCoeffs:
    """Biquad filter coefficients."""
    b0: float
    b1: float
    b2: float
    a1: float
    a2: float


def design_bandpass(center_freq: float, bandwidth: float, sample_rate: float) -> BiquadCoeffs:
    """
    Design a 2nd-order bandpass filter (biquad).

    Uses cookbook formulas for BPF (constant skirt gain, peak gain = Q).
    """
    w0 = 2.0 * np.pi * center_freq / sample_rate
    Q = center_freq / max(bandwidth, 1.0)

    alpha = np.sin(w0) / (2.0 * Q)

    b0 = alpha
    b1 = 0.0
    b2 = -alpha
    a0 = 1.0 + alpha
    a1 = -2.0 * np.cos(w0)
    a2 = 1.0 - alpha

    # Normalize
    return BiquadCoeffs(
        b0=b0 / a0,
        b1=b1 / a0,
        b2=b2 / a0,
        a1=a1 / a0,
        a2=a2 / a0
    )


def apply_biquad(samples: np.ndarray, coeffs: BiquadCoeffs) -> np.ndarray:
    """
    Apply biquad filter to samples (vectorized).

    Uses direct form II transposed for numerical stability.
    """
    n = len(samples)
    output = np.zeros(n, dtype=np.float32)

    # State variables
    z1 = 0.0
    z2 = 0.0

    # Process sample by sample (necessary for IIR)
    # This is the one place we can't fully vectorize without scipy
    for i in range(n):
        x = samples[i]
        y = coeffs.b0 * x + z1
        z1 = coeffs.b1 * x - coeffs.a1 * y + z2
        z2 = coeffs.b2 * x - coeffs.a2 * y
        output[i] = y

    return output


def apply_biquad_vectorized(samples: np.ndarray, coeffs: BiquadCoeffs) -> np.ndarray:
    """
    Apply biquad filter using numpy's vectorized operations where possible.

    This is a compromise - still needs iteration for IIR but uses numpy internally.
    """
    n = len(samples)
    output = np.zeros(n, dtype=np.float32)

    # Process in chunks for better cache utilization
    chunk_size = 1024
    z1, z2 = 0.0, 0.0

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = samples[start:end]

        for i, x in enumerate(chunk):
            y = coeffs.b0 * x + z1
            z1 = coeffs.b1 * x - coeffs.a1 * y + z2
            z2 = coeffs.b2 * x - coeffs.a2 * y
            output[start + i] = y

    return output


# =============================================================================
# Glottal Source
# =============================================================================

def generate_glottal_pulse(freq: float, num_samples: int, sample_rate: int,
                           pulse_width: float = 0.4) -> np.ndarray:
    """
    Generate glottal pulse train (voice source).

    More realistic than simple sawtooth - models glottal opening/closing.
    """
    t = np.arange(num_samples, dtype=np.float32) / sample_rate
    phase = (freq * t) % 1.0

    # Glottal pulse shape:
    # - Rising phase (glottis opening): 0 to pulse_width
    # - Peak/closing phase: pulse_width to 1.0

    output = np.zeros(num_samples, dtype=np.float32)

    # Rising phase: quadratic rise
    mask_rise = phase < pulse_width
    output[mask_rise] = (phase[mask_rise] / pulse_width) ** 1.5

    # Falling phase: sharp drop then small negative excursion
    mask_fall = ~mask_rise
    fall_phase = (phase[mask_fall] - pulse_width) / (1.0 - pulse_width)
    output[mask_fall] = (1.0 - fall_phase) ** 2 * (1.0 - 0.3 * fall_phase)

    # Normalize to [-1, 1] range
    output = 2.0 * output - 1.0

    return output


def generate_glottal_with_vibrato(freq: float, num_samples: int, sample_rate: int,
                                   vibrato_mult: np.ndarray,
                                   pulse_width: float = 0.4) -> np.ndarray:
    """Generate glottal pulse with time-varying pitch (vibrato)."""
    # Integrate instantaneous frequency to get phase
    inst_freq = freq * vibrato_mult
    phase_inc = inst_freq / sample_rate
    phase = np.cumsum(phase_inc) % 1.0

    output = np.zeros(num_samples, dtype=np.float32)

    mask_rise = phase < pulse_width
    output[mask_rise] = (phase[mask_rise] / pulse_width) ** 1.5

    mask_fall = ~mask_rise
    fall_phase = (phase[mask_fall] - pulse_width) / (1.0 - pulse_width)
    output[mask_fall] = (1.0 - fall_phase) ** 2 * (1.0 - 0.3 * fall_phase)

    output = 2.0 * output - 1.0
    return output


# =============================================================================
# Formant Vocal Synthesizer
# =============================================================================

@dataclass
class VoiceConfig:
    """Configuration for vocal synthesis."""
    sample_rate: int = 48000
    breathiness: float = 0.1      # Noise mix (0-1)
    brightness: float = 0.5       # High-frequency emphasis
    warmth: float = 0.5           # Low-frequency emphasis
    vibrato_depth: float = 10.0   # Cents
    vibrato_rate: float = 5.0     # Hz
    humanize: float = 0.1         # Pitch drift amount


class FormantVoiceSynth:
    """
    Production-grade vocal synthesizer using formant filtering.

    Pipeline:
    1. Glottal pulse train (pitched source)
    2. + Noise (breathiness)
    3. Formant filter bank (vowel shaping)
    4. + Vibrato and humanization
    5. ADSR envelope
    """

    def __init__(self, config: Optional[VoiceConfig] = None):
        self.config = config or VoiceConfig()
        self.sr = self.config.sample_rate

        # Cache filter coefficients for common formants
        self._filter_cache: Dict[Tuple[float, float], BiquadCoeffs] = {}

    def _get_filter(self, center: float, bandwidth: float) -> BiquadCoeffs:
        """Get or create cached filter coefficients."""
        key = (round(center, 1), round(bandwidth, 1))
        if key not in self._filter_cache:
            self._filter_cache[key] = design_bandpass(center, bandwidth, self.sr)
        return self._filter_cache[key]

    def render_note(self, note: str, octave: int, duration: float,
                    syllable: str, amplitude: float = 0.5,
                    emotion_params: Optional[EmotionParams] = None) -> np.ndarray:
        """
        Render a single sung note.

        Args:
            note: Note name (C, D, E, etc.)
            octave: Octave number
            duration: Duration in seconds
            syllable: Text being sung (for vowel detection)
            amplitude: Volume (0-1)
            emotion_params: Optional emotion parameters for expression

        Returns:
            Audio samples as numpy array
        """
        freq = note_to_freq(note, octave)
        num_samples = int(duration * self.sr)

        # Get emotion params (use defaults if not provided)
        if emotion_params is None:
            emotion_params = get_emotion_params("neutral")

        # Generate vibrato
        vibrato = generate_vibrato(
            num_samples, self.sr,
            emotion_params.vibrato_depth_cents,
            emotion_params.vibrato_rate_hz,
            delay_ratio=0.15  # Vibrato starts after initial attack
        )

        # Add humanization (pitch drift)
        drift = generate_pitch_drift(
            num_samples, self.sr,
            max_drift_cents=3.0 * self.config.humanize,
            rate_hz=0.25
        )

        pitch_mod = vibrato * drift

        # Generate glottal source with pitch modulation
        source = generate_glottal_with_vibrato(freq, num_samples, self.sr, pitch_mod)

        # Add breathiness (noise)
        breathiness = emotion_params.breathiness
        if breathiness > 0:
            noise = np.random.randn(num_samples).astype(np.float32) * breathiness
            source = source * (1.0 - breathiness * 0.5) + noise

        # Apply formant filtering
        vowel = guess_vowel(syllable)
        formants = get_formants(vowel)
        voiced = self._apply_formants(source, formants, emotion_params)

        # Generate envelope
        envelope = self._generate_envelope(num_samples, emotion_params)

        # Apply envelope and amplitude
        output = voiced * envelope * amplitude * emotion_params.vel_scale

        return output.astype(np.float32)

    def _apply_formants(self, source: np.ndarray, formants: FormantSet,
                        params: EmotionParams) -> np.ndarray:
        """Apply formant filter bank to source signal."""
        # Adjust formants based on brightness/warmth
        brightness_scale = 0.8 + 0.4 * params.brightness
        warmth_scale = 0.9 + 0.2 * params.warmth

        # Filter for each formant
        f1_coeffs = self._get_filter(formants.f1 * warmth_scale, formants.bw1)
        f2_coeffs = self._get_filter(formants.f2, formants.bw2)
        f3_coeffs = self._get_filter(formants.f3 * brightness_scale, formants.bw3)

        # Apply filters and mix
        f1_out = apply_biquad_vectorized(source, f1_coeffs) * formants.gain1
        f2_out = apply_biquad_vectorized(source, f2_coeffs) * formants.gain2
        f3_out = apply_biquad_vectorized(source, f3_coeffs) * formants.gain3

        # Mix formants with original for body
        output = f1_out + f2_out + f3_out + source * 0.1

        # Normalize
        max_val = np.max(np.abs(output))
        if max_val > 0:
            output = output / max_val

        return output

    def _generate_envelope(self, num_samples: int, params: EmotionParams) -> np.ndarray:
        """Generate ADSR envelope based on emotion parameters."""
        attack_samples = int(params.attack_ms * self.sr / 1000)
        decay_samples = int(params.decay_ms * self.sr / 1000)
        release_samples = int(params.release_ms * self.sr / 1000)
        sustain_samples = max(0, num_samples - attack_samples - decay_samples - release_samples)

        envelope = np.ones(num_samples, dtype=np.float32) * params.sustain_level

        # Attack
        if attack_samples > 0:
            t = np.linspace(0, 1, attack_samples, dtype=np.float32)
            envelope[:attack_samples] = np.power(t, 1.0 / max(params.vel_curve, 0.1))

        # Decay
        decay_start = attack_samples
        decay_end = decay_start + decay_samples
        if decay_samples > 0:
            t = np.linspace(0, 1, decay_samples, dtype=np.float32)
            envelope[decay_start:decay_end] = 1.0 - (1.0 - params.sustain_level) * t

        # Release
        if release_samples > 0 and num_samples > release_samples:
            t = np.linspace(1, 0, release_samples, dtype=np.float32)
            envelope[-release_samples:] = params.sustain_level * np.power(t, params.vel_curve)

        return envelope

    def render_melody(self, melody_data: List[Dict], bpm: float,
                      emotion: str = "neutral", energy: float = 0.5,
                      amplitude: float = 0.6) -> np.ndarray:
        """
        Render a complete melody section.

        Args:
            melody_data: List of bar data with notes
            bpm: Beats per minute
            emotion: Emotion name for expression
            energy: Energy level (0-1)
            amplitude: Overall volume

        Returns:
            Audio samples for entire melody
        """
        beats_per_second = bpm / 60.0
        seconds_per_beat = 1.0 / beats_per_second
        beats_per_bar = 4  # Assume 4/4

        # Get emotion params
        base_params = get_emotion_params(emotion)
        # Scale by energy
        from .dynamics import scale_params_by_energy
        params = scale_params_by_energy(base_params, energy)

        # Calculate total duration
        max_bar = max(m['bar'] for m in melody_data) if melody_data else 1
        total_bars = max_bar + 2
        total_duration = total_bars * beats_per_bar * seconds_per_beat
        total_samples = int(total_duration * self.sr)

        output = np.zeros(total_samples, dtype=np.float32)

        for bar_data in melody_data:
            bar_num = bar_data['bar']
            notes = bar_data['notes']

            bar_start_beat = (bar_num - 1) * beats_per_bar
            current_beat = bar_start_beat

            for note_info in notes:
                note_name, octave, duration_beats, syllable = note_info

                start_time = current_beat * seconds_per_beat
                duration_sec = duration_beats * seconds_per_beat

                # Render note with emotion
                note_samples = self.render_note(
                    note_name, octave, duration_sec, syllable,
                    amplitude, params
                )

                # Mix into output
                start_sample = int(start_time * self.sr)
                end_sample = min(start_sample + len(note_samples), total_samples)
                actual_len = end_sample - start_sample

                if actual_len > 0:
                    output[start_sample:end_sample] += note_samples[:actual_len]

                current_beat += duration_beats

        return output
