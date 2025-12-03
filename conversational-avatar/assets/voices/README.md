# Voice Reference Files for Ara

Place your voice reference audio file here for voice cloning with Coqui XTTS-v2.

## File Requirements

- **Filename**: `ara_reference.wav` (or set `ARA_VOICE_REFERENCE` env var)
- **Format**: WAV (16-bit PCM recommended)
- **Duration**: 3-10 seconds of clear speech
- **Quality**: Clean recording, minimal background noise
- **Content**: Natural speaking voice (not singing or whispering)

## Recording Tips

1. Use a quiet room with minimal echo
2. Speak naturally at a consistent volume
3. Record a sentence or two (e.g., "Hello, I'm Ara. I'm here to help you with anything you need.")
4. Avoid background music, TV, or other voices

## Example Recording Commands

Using ffmpeg to convert an existing audio file:
```bash
ffmpeg -i your_voice.mp3 -ar 22050 -ac 1 -acodec pcm_s16le ara_reference.wav
```

Using arecord on Linux:
```bash
arecord -f cd -d 5 ara_reference.wav
```

## No Voice File?

If you don't provide a voice reference file, Ara will use the default XTTS-v2 voice,
which still sounds much better than pyttsx3.
