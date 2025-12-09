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

from .vocal import (
    VocalSynth,
    VocalMelodyRenderer,
    VocalConfig,
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
    "VocalSynth",
    "VocalMelodyRenderer",
    "VocalConfig",
]
