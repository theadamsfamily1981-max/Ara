# AraVoice Audio Kernel

Layer 0 of the AraVoice stack: low-latency audio output.

```
┌─────────────────────────────────────────────────────┐
│                   AraVoice Stack                    │
├─────────────────────────────────────────────────────┤
│  Layer 4: Memory & Context (what to say)            │
│  Layer 3: Emotion Engine (how to say it)            │
│  Layer 2: Voice Profiles (who is speaking)          │
│  Layer 1: TTS Runtime (text → audio)                │
│  Layer 0: Audio Kernel ← YOU ARE HERE               │
└─────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Install PortAudio
make install-deps

# Build
make

# Hear the machine sing
./build/sine_test
```

## Files

```
aravoice_rt/
├── include/
│   └── audio.h           # Public API
├── src/
│   └── audio_portaudio.c # PortAudio backend
├── examples/
│   ├── sine_test.c       # Basic "hello audio"
│   └── ringbuf_test.c    # Async producer/consumer
├── Makefile
└── README.md
```

## API

```c
#include "audio.h"

// Initialize
AvDeviceConfig cfg = AV_DEFAULT_CONFIG;
av_init(&cfg);

// Render callback (called from audio thread)
int my_render(float *out, int frames, void *user_data) {
    // Fill 'out' with interleaved float32 samples [-1, 1]
    // frames * channels total samples
    return 0;
}

// Start playback
av_start_stream(my_render, &my_state);

// ... audio plays ...

// Cleanup
av_stop_stream();
av_shutdown();
```

## Ring Buffer (for TTS/async sources)

```c
// Create buffer (~500ms of audio)
AvRingBuffer *rb = av_ringbuf_create(24000, 2);

// Producer thread (TTS):
float samples[1024];
generate_tts_audio(samples, 512);
av_ringbuf_write(rb, samples, 512);

// Audio callback (consumer):
int render(float *out, int frames, void *user_data) {
    av_ringbuf_read(rb, out, frames);
    return 0;
}
```

## Next Steps

1. **sine_test works** → Audio kernel is alive
2. **ringbuf_test works** → Ready for async TTS
3. **Wrap in Python** → `ctypes` or `cffi` bindings
4. **Add TTS model** → Replace sine with Kokoro/Piper output
5. **Emotion engine** → Control prosody via tags

## Platform Support

| Platform | Backend | Status |
|----------|---------|--------|
| Linux    | PortAudio/ALSA | ✅ Working |
| macOS    | PortAudio/CoreAudio | Should work |
| Windows  | PortAudio/WASAPI | Should work |
| Pi       | PortAudio/ALSA | Should work |

## Performance

- Target latency: <50ms
- Actual latency: depends on `frames_per_buffer`
  - 256 frames @ 48kHz = ~5ms buffer
  - Total ~15-30ms with default device settings
