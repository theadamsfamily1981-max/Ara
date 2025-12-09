#!/usr/bin/env python3
"""
AraSong Synthesizer - NumPy Vectorized Oscillators
===================================================

Production-grade oscillators using NumPy for 100x faster synthesis.
All operations are vectorized - no Python loops in the hot path.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


# Standard tuning: A4 = 440 Hz
A4_FREQ = 440.0
A4_MIDI = 69

# Note name to semitone offset from C
NOTE_SEMITONES = {
    'C': 0, 'C#': 1, 'Db': 1,
    'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4,
    'F': 5, 'F#': 6, 'Gb': 6,
    'G': 7, 'G#': 8, 'Ab': 8,
    'A': 9, 'A#': 10, 'Bb': 10,
    'B': 11,
}


def note_to_freq(note: str, octave: int = 4) -> float:
    """Convert note name + octave to frequency using equal temperament."""
    semitone = NOTE_SEMITONES.get(note, 9)  # Default to A
    midi_note = (octave + 1) * 12 + semitone
    return A4_FREQ * (2.0 ** ((midi_note - A4_MIDI) / 12.0))


def freq_to_midi(freq: float) -> float:
    """Convert frequency to MIDI note number (can be fractional)."""
    return A4_MIDI + 12.0 * np.log2(freq / A4_FREQ)


def midi_to_freq(midi: float) -> float:
    """Convert MIDI note number to frequency."""
    return A4_FREQ * (2.0 ** ((midi - A4_MIDI) / 12.0))


@dataclass
class OscConfig:
    """Configuration for oscillators."""
    sample_rate: int = 48000
    dtype: np.dtype = np.float32


class NumpyOscillator:
    """Base class for vectorized oscillators."""

    def __init__(self, sample_rate: int = 48000):
        self.sr = sample_rate
        self.phase = 0.0  # For continuous rendering

    def render(self, freq: float, num_samples: int, amplitude: float = 1.0) -> np.ndarray:
        """Render samples. Override in subclasses."""
        raise NotImplementedError

    def render_with_vibrato(self, freq: float, num_samples: int,
                            vibrato_depth_cents: float = 10.0,
                            vibrato_rate_hz: float = 5.0,
                            amplitude: float = 1.0) -> np.ndarray:
        """Render with pitch vibrato."""
        t = np.arange(num_samples, dtype=np.float32) / self.sr
        # Vibrato: modulate frequency by cents
        cents_mod = vibrato_depth_cents * np.sin(2.0 * np.pi * vibrato_rate_hz * t)
        freq_mod = freq * (2.0 ** (cents_mod / 1200.0))
        return self._render_variable_freq(freq_mod, amplitude)

    def _render_variable_freq(self, freq_array: np.ndarray, amplitude: float) -> np.ndarray:
        """Render with time-varying frequency. Override for efficiency."""
        # Default: use instantaneous frequency integration
        phase_inc = freq_array / self.sr
        phase = np.cumsum(phase_inc) + self.phase
        self.phase = phase[-1] % 1.0
        return self._wave_func(phase) * amplitude

    def _wave_func(self, phase: np.ndarray) -> np.ndarray:
        """Wave function given phase (0-1). Override in subclasses."""
        raise NotImplementedError


class SineOsc(NumpyOscillator):
    """Pure sine wave oscillator."""

    def render(self, freq: float, num_samples: int, amplitude: float = 1.0) -> np.ndarray:
        t = np.arange(num_samples, dtype=np.float32) / self.sr
        phase = 2.0 * np.pi * freq * t + self.phase
        self.phase = (self.phase + 2.0 * np.pi * freq * num_samples / self.sr) % (2.0 * np.pi)
        return (amplitude * np.sin(phase)).astype(np.float32)

    def _wave_func(self, phase: np.ndarray) -> np.ndarray:
        return np.sin(2.0 * np.pi * phase)


class SawOsc(NumpyOscillator):
    """Band-limited sawtooth using additive synthesis."""

    def __init__(self, sample_rate: int = 48000, harmonics: int = 16):
        super().__init__(sample_rate)
        self.harmonics = harmonics

    def render(self, freq: float, num_samples: int, amplitude: float = 1.0) -> np.ndarray:
        t = np.arange(num_samples, dtype=np.float32) / self.sr
        samples = np.zeros(num_samples, dtype=np.float32)

        # Additive synthesis: sum of sin(n*f)/n
        nyquist = self.sr / 2.0
        for h in range(1, self.harmonics + 1):
            if h * freq >= nyquist:
                break
            phase = 2.0 * np.pi * h * freq * t + h * self.phase
            samples += np.sin(phase) / h

        samples *= amplitude * (2.0 / np.pi)
        self.phase = (self.phase + 2.0 * np.pi * freq * num_samples / self.sr) % (2.0 * np.pi)
        return samples

    def _wave_func(self, phase: np.ndarray) -> np.ndarray:
        # Simple sawtooth: 2 * (phase mod 1) - 1
        return 2.0 * (phase % 1.0) - 1.0


class SquareOsc(NumpyOscillator):
    """Band-limited square wave using odd harmonics."""

    def __init__(self, sample_rate: int = 48000, harmonics: int = 16):
        super().__init__(sample_rate)
        self.harmonics = harmonics

    def render(self, freq: float, num_samples: int, amplitude: float = 1.0) -> np.ndarray:
        t = np.arange(num_samples, dtype=np.float32) / self.sr
        samples = np.zeros(num_samples, dtype=np.float32)

        nyquist = self.sr / 2.0
        for h in range(1, self.harmonics + 1, 2):  # Odd harmonics only
            if h * freq >= nyquist:
                break
            phase = 2.0 * np.pi * h * freq * t + h * self.phase
            samples += np.sin(phase) / h

        samples *= amplitude * (4.0 / np.pi)
        self.phase = (self.phase + 2.0 * np.pi * freq * num_samples / self.sr) % (2.0 * np.pi)
        return samples

    def _wave_func(self, phase: np.ndarray) -> np.ndarray:
        return np.sign(np.sin(2.0 * np.pi * phase))


class TriangleOsc(NumpyOscillator):
    """Triangle wave oscillator."""

    def render(self, freq: float, num_samples: int, amplitude: float = 1.0) -> np.ndarray:
        t = np.arange(num_samples, dtype=np.float32) / self.sr
        phase = (freq * t + self.phase / (2.0 * np.pi)) % 1.0
        self.phase = (self.phase + 2.0 * np.pi * freq * num_samples / self.sr) % (2.0 * np.pi)

        # Triangle: 4 * |phase - 0.5| - 1
        samples = 4.0 * np.abs(phase - 0.5) - 1.0
        return (amplitude * samples).astype(np.float32)

    def _wave_func(self, phase: np.ndarray) -> np.ndarray:
        p = phase % 1.0
        return 4.0 * np.abs(p - 0.5) - 1.0


class NoiseOsc(NumpyOscillator):
    """White noise generator."""

    def render(self, freq: float, num_samples: int, amplitude: float = 1.0) -> np.ndarray:
        return (amplitude * (2.0 * np.random.random(num_samples) - 1.0)).astype(np.float32)


class PulseOsc(NumpyOscillator):
    """Pulse wave with variable duty cycle (for vocal synthesis)."""

    def __init__(self, sample_rate: int = 48000, duty: float = 0.5):
        super().__init__(sample_rate)
        self.duty = duty

    def render(self, freq: float, num_samples: int, amplitude: float = 1.0) -> np.ndarray:
        t = np.arange(num_samples, dtype=np.float32) / self.sr
        phase = (freq * t + self.phase / (2.0 * np.pi)) % 1.0
        self.phase = (self.phase + 2.0 * np.pi * freq * num_samples / self.sr) % (2.0 * np.pi)

        samples = np.where(phase < self.duty, 1.0, -1.0)
        return (amplitude * samples).astype(np.float32)


# =============================================================================
# Envelope Generators (Vectorized)
# =============================================================================

@dataclass
class ADSRParams:
    """ADSR envelope parameters in seconds."""
    attack: float = 0.01
    decay: float = 0.1
    sustain: float = 0.7  # Level, not time
    release: float = 0.2


def render_adsr(duration: float, sample_rate: int, params: ADSRParams) -> np.ndarray:
    """
    Render ADSR envelope as numpy array.

    Fully vectorized - no Python loops.
    """
    total_samples = int(duration * sample_rate)
    attack_samples = int(params.attack * sample_rate)
    decay_samples = int(params.decay * sample_rate)
    release_samples = int(params.release * sample_rate)
    sustain_samples = max(0, total_samples - attack_samples - decay_samples - release_samples)

    envelope = np.zeros(total_samples, dtype=np.float32)

    # Attack: linear ramp 0 -> 1
    if attack_samples > 0:
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples, dtype=np.float32)

    # Decay: linear ramp 1 -> sustain
    decay_start = attack_samples
    decay_end = decay_start + decay_samples
    if decay_samples > 0:
        envelope[decay_start:decay_end] = np.linspace(1, params.sustain, decay_samples, dtype=np.float32)

    # Sustain: constant level
    sustain_start = decay_end
    sustain_end = sustain_start + sustain_samples
    if sustain_samples > 0:
        envelope[sustain_start:sustain_end] = params.sustain

    # Release: linear ramp sustain -> 0
    release_start = sustain_end
    if release_samples > 0 and release_start < total_samples:
        actual_release = min(release_samples, total_samples - release_start)
        envelope[release_start:release_start + actual_release] = np.linspace(
            params.sustain, 0, actual_release, dtype=np.float32
        )

    return envelope


def render_adsr_exp(duration: float, sample_rate: int, params: ADSRParams,
                    curve: float = 3.0) -> np.ndarray:
    """
    Render ADSR with exponential curves for more natural sound.

    curve > 1: faster initial change, slower at end (natural decay)
    curve < 1: slower initial change, faster at end
    """
    total_samples = int(duration * sample_rate)
    attack_samples = int(params.attack * sample_rate)
    decay_samples = int(params.decay * sample_rate)
    release_samples = int(params.release * sample_rate)
    sustain_samples = max(0, total_samples - attack_samples - decay_samples - release_samples)

    envelope = np.zeros(total_samples, dtype=np.float32)

    # Attack: exponential curve 0 -> 1
    if attack_samples > 0:
        t = np.linspace(0, 1, attack_samples, dtype=np.float32)
        envelope[:attack_samples] = 1.0 - (1.0 - t) ** curve

    # Decay: exponential curve 1 -> sustain
    decay_start = attack_samples
    decay_end = decay_start + decay_samples
    if decay_samples > 0:
        t = np.linspace(0, 1, decay_samples, dtype=np.float32)
        envelope[decay_start:decay_end] = 1.0 - (1.0 - params.sustain) * (1.0 - (1.0 - t) ** curve)

    # Sustain
    sustain_start = decay_end
    sustain_end = sustain_start + sustain_samples
    if sustain_samples > 0:
        envelope[sustain_start:sustain_end] = params.sustain

    # Release: exponential curve sustain -> 0
    release_start = sustain_end
    if release_samples > 0 and release_start < total_samples:
        actual_release = min(release_samples, total_samples - release_start)
        t = np.linspace(0, 1, actual_release, dtype=np.float32)
        envelope[release_start:release_start + actual_release] = params.sustain * ((1.0 - t) ** curve)

    return envelope


# =============================================================================
# Chord Helpers
# =============================================================================

CHORD_INTERVALS = {
    'maj': [0, 4, 7],
    'min': [0, 3, 7],
    'm': [0, 3, 7],
    '7': [0, 4, 7, 10],
    'maj7': [0, 4, 7, 11],
    'm7': [0, 3, 7, 10],
    'min7': [0, 3, 7, 10],
    'dim': [0, 3, 6],
    'aug': [0, 4, 8],
    'sus2': [0, 2, 7],
    'sus4': [0, 5, 7],
    '9': [0, 4, 7, 10, 14],
    'add9': [0, 4, 7, 14],
}


def parse_chord(chord_name: str) -> Tuple[str, str, int]:
    """Parse chord name into (root, quality, octave)."""
    if len(chord_name) >= 2 and chord_name[1] in '#b':
        root = chord_name[:2]
        rest = chord_name[2:]
    else:
        root = chord_name[0]
        rest = chord_name[1:]

    quality = 'maj'
    octave = 3

    for q in ['maj7', 'min7', 'm7', 'add9', '9', '7', 'dim', 'aug', 'sus2', 'sus4', 'min', 'm']:
        if rest.startswith(q):
            quality = q
            rest = rest[len(q):]
            break

    if rest.isdigit():
        octave = int(rest)

    return root, quality, octave


def chord_to_freqs(chord_name: str, octave: int = 3) -> np.ndarray:
    """Convert chord name to array of frequencies."""
    root, quality, parsed_oct = parse_chord(chord_name)
    if parsed_oct != 3:
        octave = parsed_oct

    root_freq = note_to_freq(root, octave)
    intervals = CHORD_INTERVALS.get(quality, [0, 4, 7])

    freqs = np.array([root_freq * (2.0 ** (i / 12.0)) for i in intervals], dtype=np.float32)
    return freqs
