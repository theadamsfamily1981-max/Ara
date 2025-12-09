#!/usr/bin/env python3
"""
AraSong Synthesizer - Vocal Synthesizer
========================================

Synthesizes sung vocals from melody data.
Uses formant synthesis to approximate human voice.
"""

import math
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .oscillators import note_to_freq, ADSREnvelope


# Formant frequencies for basic vowels (F1, F2, F3 in Hz)
# These shape the "color" of each vowel sound
VOWEL_FORMANTS = {
    'a': (800, 1200, 2500),   # as in "father"
    'e': (400, 2200, 2800),   # as in "hey"
    'i': (300, 2300, 3000),   # as in "see"
    'o': (500, 800, 2500),    # as in "go"
    'u': (350, 700, 2500),    # as in "you"
    'default': (500, 1500, 2500),
}


@dataclass
class VocalConfig:
    """Configuration for vocal synthesis."""
    vibrato_rate: float = 5.0      # Hz
    vibrato_depth: float = 0.015   # Semitone fraction
    breathiness: float = 0.1       # Noise mix
    warmth: float = 0.8            # Lower formant emphasis


class VocalSynth:
    """
    Vocal synthesizer using formant synthesis.

    Creates a voice-like sound by:
    1. Generating a pulse/saw-like source (glottal pulse)
    2. Filtering through formant resonators
    3. Adding vibrato and breathiness
    """

    def __init__(self, sample_rate: float = 48000.0, config: Optional[VocalConfig] = None):
        self.sample_rate = sample_rate
        self.config = config or VocalConfig()
        self.phase = 0.0
        self.vibrato_phase = 0.0

    def _guess_vowel(self, syllable: str) -> str:
        """Guess the primary vowel in a syllable."""
        syllable = syllable.lower().strip('.,!?"\'-')

        # Look for vowels
        vowels_found = []
        for i, c in enumerate(syllable):
            if c in 'aeiou':
                vowels_found.append((i, c))

        if not vowels_found:
            return 'a'  # Default

        # For now, use the last vowel (typically the sung one)
        return vowels_found[-1][1]

    def _render_glottal_source(self, freq: float, num_samples: int, vibrato: bool = True) -> List[float]:
        """Generate glottal pulse train (voice source)."""
        samples = []
        phase_inc = freq / self.sample_rate
        vib_rate = self.config.vibrato_rate
        vib_depth = self.config.vibrato_depth

        for i in range(num_samples):
            # Apply vibrato
            if vibrato:
                vib = math.sin(2.0 * math.pi * vib_rate * i / self.sample_rate)
                freq_mod = freq * (1.0 + vib_depth * vib)
                phase_inc = freq_mod / self.sample_rate

            # Glottal pulse approximation (soft saw with rounded edges)
            # More natural than pure saw
            t = self.phase
            if t < 0.4:
                # Rising phase (glottis opening)
                sample = 2.5 * t
            elif t < 0.6:
                # Peak
                sample = 1.0 - 5.0 * (t - 0.4) ** 2
            else:
                # Falling phase (glottis closing)
                sample = -0.5 * (t - 0.6) / 0.4

            samples.append(sample)

            self.phase += phase_inc
            if self.phase >= 1.0:
                self.phase -= 1.0

        return samples

    def _apply_formants(self, samples: List[float], vowel: str) -> List[float]:
        """Apply formant filtering to shape the vowel sound."""
        f1, f2, f3 = VOWEL_FORMANTS.get(vowel, VOWEL_FORMANTS['default'])

        # Simple resonant filter approximation
        # Using multiple band-pass filters
        output = [0.0] * len(samples)

        # State variables for each formant filter
        states = [[0.0, 0.0] for _ in range(3)]
        formants = [(f1, 0.6), (f2, 0.3), (f3, 0.1)]  # (freq, amplitude)

        for i, s in enumerate(samples):
            total = 0.0

            for j, (f, amp) in enumerate(formants):
                # Simple 2-pole resonant filter
                # Q controls resonance bandwidth
                Q = 10.0
                w0 = 2.0 * math.pi * f / self.sample_rate
                alpha = math.sin(w0) / (2.0 * Q)

                # Bandpass coefficients (simplified)
                b0 = alpha
                a1 = -2.0 * math.cos(w0)
                a2 = 1.0 - alpha

                # Apply filter
                y = b0 * s - a1 * states[j][0] - a2 * states[j][1]
                y = y / (1.0 + alpha)  # Normalize

                states[j][1] = states[j][0]
                states[j][0] = y

                total += y * amp

            output[i] = total * self.config.warmth + s * (1.0 - self.config.warmth) * 0.3

        return output

    def _add_breathiness(self, samples: List[float]) -> List[float]:
        """Add breath noise for realism."""
        import random

        noise_level = self.config.breathiness
        return [s + noise_level * (random.random() * 2.0 - 1.0) * 0.3 for s in samples]

    def render_note(self, note: str, octave: int, duration: float,
                    syllable: str, amplitude: float = 0.5) -> List[float]:
        """
        Render a single sung note.

        Args:
            note: Note name (C, D, E, etc.)
            octave: Octave number
            duration: Duration in seconds
            syllable: Text syllable being sung
            amplitude: Volume (0-1)

        Returns:
            Audio samples
        """
        freq = note_to_freq(note, octave)
        num_samples = int(duration * self.sample_rate)

        # Generate source
        source = self._render_glottal_source(freq, num_samples)

        # Apply formants based on vowel
        vowel = self._guess_vowel(syllable)
        voiced = self._apply_formants(source, vowel)

        # Add breathiness
        voiced = self._add_breathiness(voiced)

        # Apply envelope
        env = ADSREnvelope(
            attack=0.03,   # Quick attack for clarity
            decay=0.05,
            sustain=0.85,
            release=0.08
        )
        envelope = env.render(duration, self.sample_rate)

        # Scale and apply envelope
        return [s * e * amplitude for s, e in zip(voiced, envelope)]

    def render_melody(self, melody_data: List[Dict], bpm: float,
                      section_start_bar: int = 0,
                      amplitude: float = 0.6) -> List[float]:
        """
        Render a complete melody section.

        Args:
            melody_data: List of {"bar": int, "notes": [[note, octave, duration, syllable], ...]}
            bpm: Beats per minute
            section_start_bar: Starting bar number of this section
            amplitude: Overall volume

        Returns:
            Audio samples for the entire melody
        """
        beats_per_second = bpm / 60.0
        seconds_per_beat = 1.0 / beats_per_second
        beats_per_bar = 4  # Assume 4/4

        # Calculate total duration (find the last bar)
        max_bar = max(m['bar'] for m in melody_data) if melody_data else 1
        total_bars = max_bar + 2  # Add buffer
        total_duration = total_bars * beats_per_bar * seconds_per_beat
        total_samples = int(total_duration * self.sample_rate)

        output = [0.0] * total_samples

        for bar_data in melody_data:
            bar_num = bar_data['bar']
            notes = bar_data['notes']

            # Calculate bar start time
            bar_start_beat = (bar_num - 1) * beats_per_bar
            current_beat = bar_start_beat

            for note_info in notes:
                note_name, octave, duration_beats, syllable = note_info

                # Calculate timing
                start_time = current_beat * seconds_per_beat
                duration_sec = duration_beats * seconds_per_beat

                # Render note
                note_samples = self.render_note(
                    note_name, octave, duration_sec, syllable, amplitude
                )

                # Mix into output
                start_sample = int(start_time * self.sample_rate)
                for i, s in enumerate(note_samples):
                    if start_sample + i < total_samples:
                        output[start_sample + i] += s

                current_beat += duration_beats

        return output


class VocalMelodyRenderer:
    """
    High-level renderer for vocal melodies from song JSON.
    """

    def __init__(self, sample_rate: float = 48000.0):
        self.sample_rate = sample_rate
        self.synth = VocalSynth(sample_rate)

    def render_section_vocals(self, song_data: Dict, section_name: str,
                               section_bars: int, amplitude: float = 0.6) -> List[float]:
        """
        Render vocals for a specific section.

        Args:
            song_data: Complete song JSON data
            section_name: Name of section (e.g., "verse1")
            section_bars: Number of bars in this section
            amplitude: Volume

        Returns:
            Audio samples, or empty list if no melody data
        """
        bpm = song_data.get('tempo', {}).get('bpm', 92)
        beats_per_bar = song_data.get('tempo', {}).get('time_signature', [4, 4])[0]

        # Get melody data for this section
        vocal_melody = song_data.get('vocal_melody', {})
        melody_data = vocal_melody.get(section_name, [])

        if not melody_data:
            # No melody for this section - return silence
            seconds_per_beat = 60.0 / bpm
            total_duration = section_bars * beats_per_bar * seconds_per_beat
            return [0.0] * int(total_duration * self.sample_rate)

        # Render the melody
        samples = self.synth.render_melody(melody_data, bpm, amplitude=amplitude)

        # Ensure correct length for section
        seconds_per_beat = 60.0 / bpm
        expected_duration = section_bars * beats_per_bar * seconds_per_beat
        expected_samples = int(expected_duration * self.sample_rate)

        # Trim or pad
        if len(samples) > expected_samples:
            samples = samples[:expected_samples]
        while len(samples) < expected_samples:
            samples.append(0.0)

        return samples
