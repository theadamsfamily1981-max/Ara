#!/usr/bin/env python3
"""
AraSong Synthesizer - Basic Oscillators
========================================

Pure Python oscillators for AraSong.
These feed into the aravoice_rt audio kernel.
"""

import math
from typing import List, Tuple
from dataclasses import dataclass

# Note frequencies (A4 = 440Hz)
NOTE_FREQS = {
    'C': 261.63, 'C#': 277.18, 'Db': 277.18,
    'D': 293.66, 'D#': 311.13, 'Eb': 311.13,
    'E': 329.63,
    'F': 349.23, 'F#': 369.99, 'Gb': 369.99,
    'G': 392.00, 'G#': 415.30, 'Ab': 415.30,
    'A': 440.00, 'A#': 466.16, 'Bb': 466.16,
    'B': 493.88,
}


def note_to_freq(note: str, octave: int = 4) -> float:
    """Convert note name + octave to frequency."""
    base = NOTE_FREQS.get(note.replace(str(octave), ''), 440.0)
    return base * (2 ** (octave - 4))


@dataclass
class OscState:
    """Oscillator state for continuous rendering."""
    phase: float = 0.0
    freq: float = 440.0
    sample_rate: float = 48000.0


class Oscillator:
    """Base oscillator class."""

    def __init__(self, sample_rate: float = 48000.0):
        self.sample_rate = sample_rate
        self.phase = 0.0

    def render(self, freq: float, num_samples: int, amplitude: float = 1.0) -> List[float]:
        """Render samples. Override in subclasses."""
        raise NotImplementedError


class SineOsc(Oscillator):
    """Pure sine wave oscillator."""

    def render(self, freq: float, num_samples: int, amplitude: float = 1.0) -> List[float]:
        samples = []
        phase_inc = 2.0 * math.pi * freq / self.sample_rate

        for _ in range(num_samples):
            sample = amplitude * math.sin(self.phase)
            samples.append(sample)
            self.phase += phase_inc
            if self.phase >= 2.0 * math.pi:
                self.phase -= 2.0 * math.pi

        return samples


class SawOsc(Oscillator):
    """Sawtooth wave oscillator (band-limited approximation)."""

    def __init__(self, sample_rate: float = 48000.0, harmonics: int = 16):
        super().__init__(sample_rate)
        self.harmonics = harmonics

    def render(self, freq: float, num_samples: int, amplitude: float = 1.0) -> List[float]:
        samples = []
        phase_inc = 2.0 * math.pi * freq / self.sample_rate

        for _ in range(num_samples):
            sample = 0.0
            # Additive synthesis for band-limited saw
            for h in range(1, self.harmonics + 1):
                if h * freq < self.sample_rate / 2:  # Nyquist limit
                    sample += math.sin(h * self.phase) / h
            sample *= amplitude * (2.0 / math.pi)
            samples.append(sample)

            self.phase += phase_inc
            if self.phase >= 2.0 * math.pi:
                self.phase -= 2.0 * math.pi

        return samples


class SquareOsc(Oscillator):
    """Square wave oscillator (band-limited)."""

    def __init__(self, sample_rate: float = 48000.0, harmonics: int = 16):
        super().__init__(sample_rate)
        self.harmonics = harmonics

    def render(self, freq: float, num_samples: int, amplitude: float = 1.0) -> List[float]:
        samples = []
        phase_inc = 2.0 * math.pi * freq / self.sample_rate

        for _ in range(num_samples):
            sample = 0.0
            # Odd harmonics only for square
            for h in range(1, self.harmonics + 1, 2):
                if h * freq < self.sample_rate / 2:
                    sample += math.sin(h * self.phase) / h
            sample *= amplitude * (4.0 / math.pi)
            samples.append(sample)

            self.phase += phase_inc
            if self.phase >= 2.0 * math.pi:
                self.phase -= 2.0 * math.pi

        return samples


class TriangleOsc(Oscillator):
    """Triangle wave oscillator."""

    def render(self, freq: float, num_samples: int, amplitude: float = 1.0) -> List[float]:
        samples = []
        phase_inc = freq / self.sample_rate

        for _ in range(num_samples):
            # Triangle from phase
            t = self.phase
            if t < 0.25:
                sample = 4.0 * t
            elif t < 0.75:
                sample = 2.0 - 4.0 * t
            else:
                sample = 4.0 * t - 4.0

            samples.append(amplitude * sample)

            self.phase += phase_inc
            if self.phase >= 1.0:
                self.phase -= 1.0

        return samples


class NoiseOsc(Oscillator):
    """White noise generator."""

    def render(self, freq: float, num_samples: int, amplitude: float = 1.0) -> List[float]:
        import random
        return [amplitude * (2.0 * random.random() - 1.0) for _ in range(num_samples)]


# =============================================================================
# Envelope Generator
# =============================================================================

@dataclass
class ADSREnvelope:
    """ADSR envelope generator."""
    attack: float = 0.01   # seconds
    decay: float = 0.1     # seconds
    sustain: float = 0.7   # level (0-1)
    release: float = 0.2   # seconds

    def render(self, duration: float, sample_rate: float) -> List[float]:
        """Generate envelope for given duration."""
        total_samples = int(duration * sample_rate)
        attack_samples = int(self.attack * sample_rate)
        decay_samples = int(self.decay * sample_rate)
        release_samples = int(self.release * sample_rate)
        sustain_samples = max(0, total_samples - attack_samples - decay_samples - release_samples)

        envelope = []

        # Attack
        for i in range(attack_samples):
            envelope.append(i / max(attack_samples, 1))

        # Decay
        for i in range(decay_samples):
            level = 1.0 - (1.0 - self.sustain) * (i / max(decay_samples, 1))
            envelope.append(level)

        # Sustain
        for _ in range(sustain_samples):
            envelope.append(self.sustain)

        # Release
        for i in range(release_samples):
            level = self.sustain * (1.0 - i / max(release_samples, 1))
            envelope.append(level)

        # Pad or trim to exact length
        while len(envelope) < total_samples:
            envelope.append(0.0)
        envelope = envelope[:total_samples]

        return envelope


# =============================================================================
# Chord/Note Helpers
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
}


def parse_chord(chord_name: str) -> Tuple[str, str, int]:
    """Parse chord name into (root, quality, octave)."""
    # Handle accidentals
    if len(chord_name) >= 2 and chord_name[1] in '#b':
        root = chord_name[:2]
        rest = chord_name[2:]
    else:
        root = chord_name[0]
        rest = chord_name[1:]

    # Default quality and octave
    quality = 'maj'
    octave = 3

    # Parse quality
    for q in ['maj7', 'min7', 'm7', '7', 'dim', 'aug', 'sus2', 'sus4', 'min', 'm']:
        if rest.startswith(q):
            quality = q
            rest = rest[len(q):]
            break

    # Parse octave if present
    if rest.isdigit():
        octave = int(rest)

    return root, quality, octave


def chord_to_freqs(chord_name: str, octave: int = 3) -> List[float]:
    """Convert chord name to list of frequencies."""
    root, quality, oct = parse_chord(chord_name)
    if oct != 3:
        octave = oct

    root_freq = note_to_freq(root, octave)
    intervals = CHORD_INTERVALS.get(quality, [0, 4, 7])

    freqs = []
    for interval in intervals:
        freq = root_freq * (2 ** (interval / 12))
        freqs.append(freq)

    return freqs
