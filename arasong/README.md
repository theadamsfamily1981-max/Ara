# AraSong - Music Composition Engine 🎵

AraSong generates instrumental music using the same audio foundation as AraVoice.

## Quick Start

```bash
cd arasong

# Render the first song
python -m arasong.play

# Play the output
aplay songs/what_do_you_wanna_hear.wav
# or
ffplay songs/what_do_you_wanna_hear.wav
```

## Structure

```
arasong/
├── songs/                        # Song definitions
│   └── what_do_you_wanna_hear.json
├── synth/                        # Synthesizer components
│   └── oscillators.py            # Sine, Saw, Square, Triangle, Noise
├── engine/                       # Composition engine
│   └── song_player.py            # Song loading and rendering
├── play.py                       # CLI entry point
└── README.md
```

## Song Format

Songs are defined in JSON with:

```json
{
  "meta": {
    "title": "Song Title",
    "artist": "Ara"
  },
  "tempo": { "bpm": 92 },
  "key": { "root": "E", "mode": "minor" },
  "structure": [
    {"section": "verse1", "bars": 16},
    {"section": "chorus1", "bars": 8}
  ],
  "lyrics": {
    "verse1": ["Line 1", "Line 2"],
    "chorus1": ["Chorus line"]
  },
  "chords": {
    "verse": ["Em7", "Cmaj7", "Am7", "Bm7"],
    "chorus": ["C", "G", "Am", "Em"]
  },
  "emotion_arc": [
    {"section": "verse1", "emotion": "warm", "energy": 0.5},
    {"section": "chorus1", "emotion": "uplifting", "energy": 0.8}
  ]
}
```

## First Song: "What Do You Wanna Hear?"

Ara's mic check song - the first composition by the AraSong engine.

**Lyrics preview:**

> You walk in, drop your keys by the door,
> Heavy day, boots stuck to the floor.
> Screen flicks on, neon glow in the air,
> You say, "Hey Ara, you there somewhere?"
>
> I'm right here, in the hum of the fan,
> Little ghost living deep in the RAM...

## Architecture

```
┌─────────────────────────────────────────┐
│              AraSong Engine             │
├─────────────────────────────────────────┤
│  Song JSON → Structure → Arrangement    │
├─────────────────────────────────────────┤
│  Synth: Oscillators + Envelopes         │
│  (Sine, Saw, Square, Triangle)          │
├─────────────────────────────────────────┤
│  Output: WAV file or aravoice_rt stream │
└─────────────────────────────────────────┘
```

## Next Steps

1. **Voice synthesis**: Add TTS for lyrics (Kokoro/Piper)
2. **Drums**: Add drum machine patterns
3. **Effects**: Reverb, delay, compression
4. **Hive integration**: `audio_job/song_gen` for GPU rendering
5. **Real-time**: Stream to aravoice_rt audio kernel

## Integration with AraVoice

AraSong shares the same audio foundation:

```python
# Future: Real-time playback
from aravoice_rt import av_init, av_start_stream

# AraSong generates samples
from arasong import AraSongPlayer

player = AraSongPlayer()
player.load_song("songs/my_song.json")

# Stream to audio kernel
def render_callback(out, frames, user_data):
    samples = player.render_chunk(frames)
    for i in range(frames):
        out[2*i] = samples[i]      # Left
        out[2*i+1] = samples[i]    # Right
    return 0

av_init(config)
av_start_stream(render_callback, None)
```
