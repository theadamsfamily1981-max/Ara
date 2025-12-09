#!/usr/bin/env python3
"""
AraSong Engine - Song Player
=============================

Renders songs using the AraSong synthesizers and aravoice_rt audio kernel.

Usage:
    python -m arasong.engine.song_player songs/what_do_you_wanna_hear.json
"""

import json
import math
import time
import wave
import struct
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from ..synth.oscillators import (
    SineOsc, SawOsc, SquareOsc, TriangleOsc,
    ADSREnvelope, chord_to_freqs, note_to_freq,
)


@dataclass
class SongSection:
    """A section of the song."""
    name: str
    bars: int
    lyrics: List[str]
    chords: List[str]
    emotion: str
    energy: float


class AraSongPlayer:
    """
    Renders AraSong compositions to audio.

    For now: Pure Python synthesis → WAV file
    Later: Stream to aravoice_rt audio kernel
    """

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.song_data: Dict[str, Any] = {}

        # Oscillators for different voices
        self.pad_osc = SawOsc(sample_rate, harmonics=8)
        self.bass_osc = SineOsc(sample_rate)
        self.lead_osc = TriangleOsc(sample_rate)
        self.arp_osc = SquareOsc(sample_rate, harmonics=6)

    def load_song(self, path: str) -> Dict[str, Any]:
        """Load song JSON file."""
        with open(path, 'r') as f:
            self.song_data = json.load(f)
        return self.song_data

    def get_section_data(self, section_name: str) -> Optional[SongSection]:
        """Get data for a specific section."""
        # Find section in structure
        for sec in self.song_data.get('structure', []):
            if sec['section'] == section_name:
                bars = sec['bars']
                break
        else:
            return None

        # Get lyrics
        lyrics = self.song_data.get('lyrics', {}).get(section_name, [])

        # Get chords
        chords_key = section_name.replace('1', '').replace('2', '')
        chords = self.song_data.get('chords', {}).get(chords_key, ['Em7'])

        # Get emotion
        for em in self.song_data.get('emotion_arc', []):
            if em['section'] == section_name:
                emotion = em['emotion']
                energy = em['energy']
                break
        else:
            emotion = 'neutral'
            energy = 0.5

        return SongSection(
            name=section_name,
            bars=bars,
            lyrics=lyrics,
            chords=chords,
            emotion=emotion,
            energy=energy,
        )

    def render_chord_pad(self, chord: str, duration: float, amplitude: float = 0.3) -> List[float]:
        """Render a chord pad."""
        freqs = chord_to_freqs(chord, octave=3)
        num_samples = int(duration * self.sample_rate)

        # Envelope
        env = ADSREnvelope(attack=0.1, decay=0.2, sustain=0.6, release=0.3)
        envelope = env.render(duration, self.sample_rate)

        # Render each note and mix
        samples = [0.0] * num_samples
        for freq in freqs:
            osc = SawOsc(self.sample_rate, harmonics=8)
            note_samples = osc.render(freq, num_samples, amplitude / len(freqs))
            for i in range(num_samples):
                samples[i] += note_samples[i] * envelope[i]

        return samples

    def render_bass(self, chord: str, duration: float, amplitude: float = 0.4) -> List[float]:
        """Render bass note (root of chord, octave down)."""
        freqs = chord_to_freqs(chord, octave=2)
        root_freq = freqs[0] / 2  # One more octave down

        num_samples = int(duration * self.sample_rate)
        env = ADSREnvelope(attack=0.02, decay=0.1, sustain=0.8, release=0.1)
        envelope = env.render(duration, self.sample_rate)

        osc = SineOsc(self.sample_rate)
        samples = osc.render(root_freq, num_samples, amplitude)

        return [s * e for s, e in zip(samples, envelope)]

    def render_arp(self, chord: str, duration: float, pattern_speed: float = 0.125,
                   amplitude: float = 0.2) -> List[float]:
        """Render arpeggio pattern."""
        freqs = chord_to_freqs(chord, octave=4)
        num_samples = int(duration * self.sample_rate)
        samples_per_note = int(pattern_speed * self.sample_rate)

        samples = [0.0] * num_samples

        # Up-down pattern
        pattern = freqs + freqs[-2:0:-1]  # up then down
        note_idx = 0

        env = ADSREnvelope(attack=0.01, decay=0.05, sustain=0.3, release=0.05)

        for start in range(0, num_samples, samples_per_note):
            freq = pattern[note_idx % len(pattern)]
            note_duration = min(samples_per_note, num_samples - start) / self.sample_rate

            osc = SquareOsc(self.sample_rate, harmonics=4)
            note_samples = osc.render(freq, min(samples_per_note, num_samples - start), amplitude)
            note_env = env.render(note_duration, self.sample_rate)

            for i, (s, e) in enumerate(zip(note_samples, note_env)):
                if start + i < num_samples:
                    samples[start + i] += s * e

            note_idx += 1

        return samples

    def render_section(self, section: SongSection) -> List[float]:
        """Render a complete section."""
        bpm = self.song_data.get('tempo', {}).get('bpm', 92)
        beats_per_bar = self.song_data.get('tempo', {}).get('time_signature', [4, 4])[0]

        # Duration of section
        bars = section.bars
        beats = bars * beats_per_bar
        duration = beats * (60.0 / bpm)

        num_samples = int(duration * self.sample_rate)
        samples = [0.0] * num_samples

        # Duration per chord (assume 2 bars per chord for simplicity)
        chords_in_section = section.chords * ((bars // 2) + 1)
        chord_duration = duration / len(section.chords[:bars // 2])

        # Render layers
        print(f"  Rendering {section.name}: {bars} bars, {duration:.1f}s, energy={section.energy}")

        # Scale amplitudes by energy
        energy = section.energy

        # Chord pad
        chord_idx = 0
        for start_time in range(0, int(duration), int(chord_duration)):
            chord = chords_in_section[chord_idx % len(chords_in_section)]
            pad = self.render_chord_pad(chord, chord_duration, amplitude=0.25 * energy)
            start_sample = int(start_time * self.sample_rate)
            for i, s in enumerate(pad):
                if start_sample + i < num_samples:
                    samples[start_sample + i] += s
            chord_idx += 1

        # Bass
        chord_idx = 0
        for start_time in range(0, int(duration), int(chord_duration)):
            chord = chords_in_section[chord_idx % len(chords_in_section)]
            bass = self.render_bass(chord, chord_duration, amplitude=0.3 * energy)
            start_sample = int(start_time * self.sample_rate)
            for i, s in enumerate(bass):
                if start_sample + i < num_samples:
                    samples[start_sample + i] += s
            chord_idx += 1

        # Arp (only on chorus and high-energy sections)
        if energy > 0.6:
            chord_idx = 0
            for start_time in range(0, int(duration), int(chord_duration)):
                chord = chords_in_section[chord_idx % len(chords_in_section)]
                arp = self.render_arp(chord, chord_duration, amplitude=0.15 * energy)
                start_sample = int(start_time * self.sample_rate)
                for i, s in enumerate(arp):
                    if start_sample + i < num_samples:
                        samples[start_sample + i] += s
                chord_idx += 1

        return samples

    def render_song(self) -> List[float]:
        """Render the complete song."""
        print(f"\n🎵 Rendering: {self.song_data['meta']['title']}")
        print(f"   Artist: {self.song_data['meta']['artist']}")
        print(f"   BPM: {self.song_data['tempo']['bpm']}")
        print()

        all_samples = []

        for sec_info in self.song_data.get('structure', []):
            section = self.get_section_data(sec_info['section'])
            if section:
                section_samples = self.render_section(section)
                all_samples.extend(section_samples)

        # Normalize
        max_amp = max(abs(s) for s in all_samples) or 1.0
        if max_amp > 1.0:
            all_samples = [s / max_amp * 0.9 for s in all_samples]

        return all_samples

    def save_wav(self, samples: List[float], output_path: str):
        """Save samples to WAV file."""
        with wave.open(output_path, 'w') as wav:
            wav.setnchannels(2)  # Stereo
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(self.sample_rate)

            # Convert to 16-bit stereo
            for sample in samples:
                # Clamp to [-1, 1]
                s = max(-1.0, min(1.0, sample))
                # Convert to 16-bit
                s16 = int(s * 32767)
                # Write stereo (same on both channels)
                wav.writeframes(struct.pack('<hh', s16, s16))

        print(f"\n✅ Saved to: {output_path}")
        print(f"   Duration: {len(samples) / self.sample_rate:.1f}s")
        print(f"   Sample rate: {self.sample_rate} Hz")


def main():
    import sys

    if len(sys.argv) < 2:
        # Default to our first song
        song_path = Path(__file__).parent.parent / "songs" / "what_do_you_wanna_hear.json"
    else:
        song_path = Path(sys.argv[1])

    if not song_path.exists():
        print(f"Song not found: {song_path}")
        sys.exit(1)

    player = AraSongPlayer(sample_rate=48000)
    player.load_song(str(song_path))

    samples = player.render_song()

    # Output path
    output_path = song_path.with_suffix('.wav')
    player.save_wav(samples, str(output_path))

    # Print lyrics
    print("\n📜 LYRICS:")
    print("=" * 60)
    for section in player.song_data.get('structure', []):
        section_name = section['section']
        lyrics = player.song_data.get('lyrics', {}).get(section_name, [])
        if lyrics:
            print(f"\n[{section_name.upper()}]")
            for line in lyrics:
                print(f"  {line}")
    print("=" * 60)


if __name__ == "__main__":
    main()
