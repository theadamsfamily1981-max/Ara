#!/usr/bin/env python3
"""
AraSong Engine - Production Song Player (v2)
=============================================

Numpy-vectorized, emotion-driven song rendering with proper mixing.
This is the production-grade replacement for song_player.py.

Features:
- Numpy vectorized synthesis (100x faster)
- Emotion-driven dynamics
- Formant-based vocal synthesis
- Automatic ducking and compression
- Professional mixdown
"""

import json
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from ..synth.numpy_osc import (
    SineOsc, SawOsc, SquareOsc, TriangleOsc, NoiseOsc,
    note_to_freq, chord_to_freqs, render_adsr, ADSRParams,
)
from ..synth.dynamics import (
    EmotionParams, get_emotion_params, scale_params_by_energy,
    build_emotion_arc_from_song, generate_vibrato,
)
from ..synth.formant_voice import FormantVoiceSynth, VoiceConfig
from .mixer import Mixer, save_wav_numpy, CompressorConfig


@dataclass
class SongSection:
    """A section of the song."""
    name: str
    bars: int
    lyrics: List[str]
    chords: List[str]
    emotion: str
    energy: float


class AraSongPlayerV2:
    """
    Production-grade song renderer.

    Uses numpy for all synthesis, with proper mixing pipeline.
    """

    def __init__(self, sample_rate: int = 48000):
        self.sr = sample_rate
        self.song_data: Dict[str, Any] = {}

        # Numpy-based oscillators
        self.pad_osc = SawOsc(sample_rate, harmonics=8)
        self.bass_osc = SineOsc(sample_rate)
        self.arp_osc = SquareOsc(sample_rate, harmonics=6)

        # Formant voice synth
        voice_config = VoiceConfig(sample_rate=sample_rate)
        self.voice_synth = FormantVoiceSynth(voice_config)

        # Mixer
        self.mixer = Mixer(sample_rate)

    def load_song(self, path: str) -> Dict[str, Any]:
        """Load song JSON file."""
        with open(path, 'r') as f:
            self.song_data = json.load(f)
        return self.song_data

    def get_section_data(self, section_name: str) -> Optional[SongSection]:
        """Get data for a specific section."""
        for sec in self.song_data.get('structure', []):
            if sec['section'] == section_name:
                bars = sec['bars']
                break
        else:
            return None

        lyrics = self.song_data.get('lyrics', {}).get(section_name, [])
        chords_key = section_name.replace('1', '').replace('2', '')
        chords = self.song_data.get('chords', {}).get(chords_key, ['Em7'])

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

    def render_chord_pad(self, chord: str, duration: float,
                         emotion_params: EmotionParams) -> np.ndarray:
        """Render a chord pad with emotion-driven dynamics."""
        freqs = chord_to_freqs(chord, octave=3)
        num_samples = int(duration * self.sr)

        # ADSR based on emotion
        adsr = ADSRParams(
            attack=emotion_params.attack_ms / 1000,
            decay=emotion_params.decay_ms / 1000,
            sustain=emotion_params.sustain_level,
            release=emotion_params.release_ms / 1000,
        )
        envelope = render_adsr(duration, self.sr, adsr)

        # Render each note
        samples = np.zeros(num_samples, dtype=np.float32)
        amp_per_note = emotion_params.vel_scale * 0.25 / len(freqs)

        for freq in freqs:
            osc = SawOsc(self.sr, harmonics=8)
            note_samples = osc.render(freq, num_samples, amp_per_note)
            samples += note_samples

        # Apply envelope
        samples = samples * envelope[:len(samples)]

        return samples

    def render_bass(self, chord: str, duration: float,
                    emotion_params: EmotionParams) -> np.ndarray:
        """Render bass note."""
        freqs = chord_to_freqs(chord, octave=2)
        root_freq = freqs[0] / 2  # One more octave down

        num_samples = int(duration * self.sr)

        adsr = ADSRParams(
            attack=0.02,
            decay=0.1,
            sustain=0.8,
            release=0.1,
        )
        envelope = render_adsr(duration, self.sr, adsr)

        osc = SineOsc(self.sr)
        samples = osc.render(root_freq, num_samples, emotion_params.vel_scale * 0.35)

        return (samples * envelope[:len(samples)]).astype(np.float32)

    def render_arp(self, chord: str, duration: float, pattern_speed: float = 0.125,
                   emotion_params: EmotionParams = None) -> np.ndarray:
        """Render arpeggio pattern."""
        if emotion_params is None:
            emotion_params = get_emotion_params("neutral")

        freqs = chord_to_freqs(chord, octave=4)
        num_samples = int(duration * self.sr)
        samples_per_note = int(pattern_speed * self.sr)

        samples = np.zeros(num_samples, dtype=np.float32)

        # Up-down pattern
        pattern = list(freqs) + list(freqs[-2:0:-1])
        note_idx = 0

        adsr = ADSRParams(attack=0.01, decay=0.05, sustain=0.3, release=0.05)

        for start in range(0, num_samples, samples_per_note):
            freq = pattern[note_idx % len(pattern)]
            note_len = min(samples_per_note, num_samples - start)
            note_duration = note_len / self.sr

            osc = SquareOsc(self.sr, harmonics=4)
            note_samples = osc.render(freq, note_len, emotion_params.vel_scale * 0.15)
            note_env = render_adsr(note_duration, self.sr, adsr)

            samples[start:start + note_len] += note_samples[:note_len] * note_env[:note_len]
            note_idx += 1

        return samples

    def render_section_instruments(self, section: SongSection) -> np.ndarray:
        """Render instrumental tracks for a section."""
        bpm = self.song_data.get('tempo', {}).get('bpm', 92)
        beats_per_bar = self.song_data.get('tempo', {}).get('time_signature', [4, 4])[0]

        duration = section.bars * beats_per_bar * (60.0 / bpm)
        num_samples = int(duration * self.sr)

        # Get emotion params
        base_params = get_emotion_params(section.emotion)
        params = scale_params_by_energy(base_params, section.energy)

        # Duration per chord
        chords_per_section = max(1, section.bars // 2)
        chord_duration = duration / chords_per_section

        samples = np.zeros(num_samples, dtype=np.float32)

        # Render each chord
        for i in range(chords_per_section):
            chord = section.chords[i % len(section.chords)]
            start_sample = int(i * chord_duration * self.sr)
            end_sample = min(start_sample + int(chord_duration * self.sr), num_samples)
            chunk_len = end_sample - start_sample
            actual_duration = chunk_len / self.sr

            # Pad
            pad = self.render_chord_pad(chord, actual_duration, params)
            pad_len = min(len(pad), chunk_len)
            samples[start_sample:start_sample + pad_len] += pad[:pad_len]

            # Bass
            bass = self.render_bass(chord, actual_duration, params)
            bass_len = min(len(bass), chunk_len)
            samples[start_sample:start_sample + bass_len] += bass[:bass_len]

            # Arp (high energy only)
            if section.energy > 0.6:
                arp = self.render_arp(chord, actual_duration, emotion_params=params)
                arp_len = min(len(arp), chunk_len)
                samples[start_sample:start_sample + arp_len] += arp[:arp_len]

        return samples

    def render_section_vocals(self, section: SongSection) -> np.ndarray:
        """Render vocal track for a section."""
        melody_data = self.song_data.get('vocal_melody', {}).get(section.name, [])

        if not melody_data:
            # No vocals for this section
            bpm = self.song_data.get('tempo', {}).get('bpm', 92)
            beats_per_bar = self.song_data.get('tempo', {}).get('time_signature', [4, 4])[0]
            duration = section.bars * beats_per_bar * (60.0 / bpm)
            return np.zeros(int(duration * self.sr), dtype=np.float32)

        bpm = self.song_data.get('tempo', {}).get('bpm', 92)

        # Use formant synth
        samples = self.voice_synth.render_melody(
            melody_data, bpm,
            emotion=section.emotion,
            energy=section.energy,
            amplitude=0.7
        )

        # Trim to section length
        beats_per_bar = self.song_data.get('tempo', {}).get('time_signature', [4, 4])[0]
        duration = section.bars * beats_per_bar * (60.0 / bpm)
        expected_samples = int(duration * self.sr)

        if len(samples) > expected_samples:
            samples = samples[:expected_samples]
        elif len(samples) < expected_samples:
            samples = np.pad(samples, (0, expected_samples - len(samples)))

        return samples

    def render_song(self) -> np.ndarray:
        """Render the complete song with proper mixing."""
        print(f"\n🎵 Rendering: {self.song_data['meta']['title']}")
        print(f"   Artist: {self.song_data['meta']['artist']}")
        print(f"   BPM: {self.song_data['tempo']['bpm']}")
        print()

        start_time = time.time()

        all_instruments = []
        all_vocals = []

        for sec_info in self.song_data.get('structure', []):
            section = self.get_section_data(sec_info['section'])
            if not section:
                continue

            print(f"  Rendering {section.name}: {section.bars} bars, "
                  f"emotion={section.emotion}, energy={section.energy:.1f}")

            # Render instruments
            instruments = self.render_section_instruments(section)
            all_instruments.append(instruments)

            # Render vocals
            vocals = self.render_section_vocals(section)
            if np.max(np.abs(vocals)) > 0.01:
                print(f"    + Adding vocals for {section.name}")
            all_vocals.append(vocals)

        # Concatenate
        instrument_track = np.concatenate(all_instruments) if all_instruments else np.array([])
        vocal_track = np.concatenate(all_vocals) if all_vocals else np.array([])

        # Use mixer for professional output
        self.mixer.clear()
        self.mixer.add_music(instrument_track)
        self.mixer.add_vocals(vocal_track)

        # Final mixdown with ducking and compression
        final_mix = self.mixer.mixdown()

        render_time = time.time() - start_time
        duration = len(final_mix) / self.sr

        print(f"\n✅ Rendered {duration:.1f}s in {render_time:.1f}s "
              f"({duration/render_time:.1f}x realtime)")

        return final_mix

    def save_wav(self, samples: np.ndarray, output_path: str):
        """Save samples to WAV file."""
        save_wav_numpy(samples, output_path, self.sr, channels=2)
        print(f"   Saved to: {output_path}")


def main():
    """CLI entry point."""
    import sys

    if len(sys.argv) < 2:
        song_path = Path(__file__).parent.parent / "songs" / "what_do_you_wanna_hear.json"
    else:
        song_path = Path(sys.argv[1])

    if not song_path.exists():
        print(f"Song not found: {song_path}")
        sys.exit(1)

    player = AraSongPlayerV2(sample_rate=48000)
    player.load_song(str(song_path))

    samples = player.render_song()

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
                if line:
                    print(f"  {line}")
    print("=" * 60)


if __name__ == "__main__":
    main()
