"""AraSong Synthesizer components."""

from .oscillators import (
    SineOsc,
    SawOsc,
    SquareOsc,
    TriangleOsc,
    NoiseOsc,
    ADSREnvelope,
    note_to_freq,
    chord_to_freqs,
    parse_chord,
)

__all__ = [
    "SineOsc",
    "SawOsc",
    "SquareOsc",
    "TriangleOsc",
    "NoiseOsc",
    "ADSREnvelope",
    "note_to_freq",
    "chord_to_freqs",
    "parse_chord",
]
