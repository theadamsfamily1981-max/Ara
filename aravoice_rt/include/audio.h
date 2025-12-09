/*
 * AraVoice Audio Kernel - Public API
 * ===================================
 * Layer 0: Device I/O, lock-free buffers, minimal latency.
 *
 * This is the foundation that TTS, emotion engine, and AraSong
 * all build upon.
 */

#ifndef ARAVOICE_AUDIO_H
#define ARAVOICE_AUDIO_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/* ------------------------------------------------------------------------- */
/* Configuration                                                             */
/* ------------------------------------------------------------------------- */

typedef struct {
    int sample_rate;        /* e.g., 44100, 48000 */
    int channels;           /* 1 = mono, 2 = stereo */
    int frames_per_buffer;  /* e.g., 256, 512, 1024 */
} AvDeviceConfig;

/* Default config: 48kHz stereo, 256 frame buffer (~5ms latency) */
#define AV_DEFAULT_CONFIG { 48000, 2, 256 }

/* ------------------------------------------------------------------------- */
/* Render Callback                                                           */
/* ------------------------------------------------------------------------- */

/*
 * Called by the audio thread to request samples.
 *
 * Parameters:
 *   out       - Buffer to fill with interleaved float32 samples [-1, 1]
 *   frames    - Number of frames to generate (total samples = frames * channels)
 *   user_data - Pointer passed to av_start_stream()
 *
 * Returns:
 *   0 on success, non-zero to signal error (stream continues regardless)
 *
 * IMPORTANT: This runs in the audio thread. Keep it fast, no allocations,
 * no blocking calls, no printf. Lock-free only.
 */
typedef int (*AvRenderCallback)(float *out, int frames, void *user_data);

/* ------------------------------------------------------------------------- */
/* Lifecycle                                                                 */
/* ------------------------------------------------------------------------- */

/*
 * Initialize the audio subsystem with given config.
 * Returns 0 on success, -1 on error.
 */
int av_init(const AvDeviceConfig *cfg);

/*
 * Start the audio stream with a render callback.
 * The callback will be invoked from the audio thread.
 * Returns 0 on success, -1 on error.
 */
int av_start_stream(AvRenderCallback cb, void *user_data);

/*
 * Stop the audio stream (callback stops being called).
 */
void av_stop_stream(void);

/*
 * Shutdown the audio subsystem and release resources.
 */
void av_shutdown(void);

/* ------------------------------------------------------------------------- */
/* Utilities                                                                 */
/* ------------------------------------------------------------------------- */

/*
 * Get current output latency in milliseconds.
 * Returns -1 if stream not active.
 */
float av_get_latency_ms(void);

/*
 * Get actual sample rate (may differ from requested).
 */
int av_get_actual_sample_rate(void);

/* ------------------------------------------------------------------------- */
/* Future: Ring buffer for producer-consumer pattern                         */
/* ------------------------------------------------------------------------- */

/*
 * Lock-free ring buffer for feeding audio from another thread.
 * Use this when your audio source (TTS, song gen) runs asynchronously.
 */

typedef struct AvRingBuffer AvRingBuffer;

AvRingBuffer *av_ringbuf_create(int capacity_frames, int channels);
void av_ringbuf_destroy(AvRingBuffer *rb);

/* Producer side (your TTS thread) */
int av_ringbuf_write(AvRingBuffer *rb, const float *data, int frames);
int av_ringbuf_write_available(AvRingBuffer *rb);

/* Consumer side (audio callback) */
int av_ringbuf_read(AvRingBuffer *rb, float *out, int frames);
int av_ringbuf_read_available(AvRingBuffer *rb);

#ifdef __cplusplus
}
#endif

#endif /* ARAVOICE_AUDIO_H */
