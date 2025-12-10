/*
 * AraVoice Audio Kernel - PortAudio Backend
 * ==========================================
 * Cross-platform audio output via PortAudio.
 *
 * Build: gcc -c audio_portaudio.c -I../include -lportaudio
 */

#include "audio.h"
#include <portaudio.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>

/* ------------------------------------------------------------------------- */
/* Global State                                                              */
/* ------------------------------------------------------------------------- */

static PaStream *g_stream = NULL;
static AvRenderCallback g_cb = NULL;
static void *g_cb_user = NULL;
static AvDeviceConfig g_cfg = {0};
static atomic_int g_initialized = 0;

/* ------------------------------------------------------------------------- */
/* PortAudio Callback                                                        */
/* ------------------------------------------------------------------------- */

static int pa_callback(
    const void *inputBuffer,
    void *outputBuffer,
    unsigned long framesPerBuffer,
    const PaStreamCallbackTimeInfo *timeInfo,
    PaStreamCallbackFlags statusFlags,
    void *userData
) {
    (void)inputBuffer;
    (void)timeInfo;
    (void)statusFlags;
    (void)userData;

    float *out = (float *)outputBuffer;
    int frames = (int)framesPerBuffer;

    if (!g_cb) {
        /* No callback registered, output silence */
        memset(out, 0, frames * g_cfg.channels * sizeof(float));
        return paContinue;
    }

    /* Call user's render callback */
    int rc = g_cb(out, frames, g_cb_user);
    (void)rc; /* For now, ignore return value */

    return paContinue;
}

/* ------------------------------------------------------------------------- */
/* Public API                                                                */
/* ------------------------------------------------------------------------- */

int av_init(const AvDeviceConfig *cfg) {
    if (atomic_load(&g_initialized)) {
        fprintf(stderr, "[aravoice] Already initialized\n");
        return -1;
    }

    PaError err = Pa_Initialize();
    if (err != paNoError) {
        fprintf(stderr, "[aravoice] PortAudio init error: %s\n",
                Pa_GetErrorText(err));
        return -1;
    }

    g_cfg = *cfg;

    /* Get default output device */
    PaDeviceIndex devIndex = Pa_GetDefaultOutputDevice();
    if (devIndex == paNoDevice) {
        fprintf(stderr, "[aravoice] No default output device found\n");
        Pa_Terminate();
        return -1;
    }

    const PaDeviceInfo *devInfo = Pa_GetDeviceInfo(devIndex);
    printf("[aravoice] Using device: %s\n", devInfo->name);
    printf("[aravoice] Config: %d Hz, %d ch, %d frames/buf\n",
           g_cfg.sample_rate, g_cfg.channels, g_cfg.frames_per_buffer);

    /* Configure output stream */
    PaStreamParameters outParams;
    memset(&outParams, 0, sizeof(outParams));
    outParams.device = devIndex;
    outParams.channelCount = g_cfg.channels;
    outParams.sampleFormat = paFloat32;
    outParams.suggestedLatency = devInfo->defaultLowOutputLatency;
    outParams.hostApiSpecificStreamInfo = NULL;

    /* Open stream */
    err = Pa_OpenStream(
        &g_stream,
        NULL,                       /* no input */
        &outParams,
        g_cfg.sample_rate,
        g_cfg.frames_per_buffer,
        paNoFlag,
        pa_callback,
        NULL
    );

    if (err != paNoError) {
        fprintf(stderr, "[aravoice] Failed to open stream: %s\n",
                Pa_GetErrorText(err));
        Pa_Terminate();
        g_stream = NULL;
        return -1;
    }

    atomic_store(&g_initialized, 1);
    printf("[aravoice] Audio kernel initialized\n");
    return 0;
}

int av_start_stream(AvRenderCallback cb, void *user_data) {
    if (!g_stream) {
        fprintf(stderr, "[aravoice] Stream not initialized\n");
        return -1;
    }

    g_cb = cb;
    g_cb_user = user_data;

    PaError err = Pa_StartStream(g_stream);
    if (err != paNoError) {
        fprintf(stderr, "[aravoice] Failed to start stream: %s\n",
                Pa_GetErrorText(err));
        return -1;
    }

    printf("[aravoice] Stream started\n");
    return 0;
}

void av_stop_stream(void) {
    if (g_stream && Pa_IsStreamActive(g_stream)) {
        Pa_StopStream(g_stream);
        printf("[aravoice] Stream stopped\n");
    }
    g_cb = NULL;
    g_cb_user = NULL;
}

void av_shutdown(void) {
    if (g_stream) {
        if (Pa_IsStreamActive(g_stream)) {
            Pa_StopStream(g_stream);
        }
        Pa_CloseStream(g_stream);
        g_stream = NULL;
    }

    if (atomic_load(&g_initialized)) {
        Pa_Terminate();
        atomic_store(&g_initialized, 0);
        printf("[aravoice] Audio kernel shutdown\n");
    }
}

float av_get_latency_ms(void) {
    if (!g_stream) return -1.0f;

    const PaStreamInfo *info = Pa_GetStreamInfo(g_stream);
    if (!info) return -1.0f;

    return (float)(info->outputLatency * 1000.0);
}

int av_get_actual_sample_rate(void) {
    if (!g_stream) return -1;

    const PaStreamInfo *info = Pa_GetStreamInfo(g_stream);
    if (!info) return -1;

    return (int)info->sampleRate;
}

/* ------------------------------------------------------------------------- */
/* Ring Buffer Implementation                                                */
/* ------------------------------------------------------------------------- */

struct AvRingBuffer {
    float *buffer;
    int capacity;       /* in samples (frames * channels) */
    int channels;
    atomic_int read_pos;
    atomic_int write_pos;
};

AvRingBuffer *av_ringbuf_create(int capacity_frames, int channels) {
    AvRingBuffer *rb = (AvRingBuffer *)malloc(sizeof(AvRingBuffer));
    if (!rb) return NULL;

    int capacity_samples = capacity_frames * channels;
    rb->buffer = (float *)calloc(capacity_samples, sizeof(float));
    if (!rb->buffer) {
        free(rb);
        return NULL;
    }

    rb->capacity = capacity_samples;
    rb->channels = channels;
    atomic_store(&rb->read_pos, 0);
    atomic_store(&rb->write_pos, 0);

    return rb;
}

void av_ringbuf_destroy(AvRingBuffer *rb) {
    if (rb) {
        free(rb->buffer);
        free(rb);
    }
}

int av_ringbuf_write_available(AvRingBuffer *rb) {
    int r = atomic_load(&rb->read_pos);
    int w = atomic_load(&rb->write_pos);
    int used = (w - r + rb->capacity) % rb->capacity;
    return (rb->capacity - 1 - used) / rb->channels;
}

int av_ringbuf_write(AvRingBuffer *rb, const float *data, int frames) {
    int samples = frames * rb->channels;
    int w = atomic_load(&rb->write_pos);

    for (int i = 0; i < samples; i++) {
        int next_w = (w + 1) % rb->capacity;
        int r = atomic_load(&rb->read_pos);
        if (next_w == r) {
            /* Buffer full */
            return i / rb->channels;
        }
        rb->buffer[w] = data[i];
        w = next_w;
    }

    atomic_store(&rb->write_pos, w);
    return frames;
}

int av_ringbuf_read_available(AvRingBuffer *rb) {
    int r = atomic_load(&rb->read_pos);
    int w = atomic_load(&rb->write_pos);
    int used = (w - r + rb->capacity) % rb->capacity;
    return used / rb->channels;
}

int av_ringbuf_read(AvRingBuffer *rb, float *out, int frames) {
    int samples = frames * rb->channels;
    int r = atomic_load(&rb->read_pos);

    for (int i = 0; i < samples; i++) {
        int w = atomic_load(&rb->write_pos);
        if (r == w) {
            /* Buffer empty, zero-fill remaining */
            memset(out + i, 0, (samples - i) * sizeof(float));
            return i / rb->channels;
        }
        out[i] = rb->buffer[r];
        r = (r + 1) % rb->capacity;
    }

    atomic_store(&rb->read_pos, r);
    return frames;
}
