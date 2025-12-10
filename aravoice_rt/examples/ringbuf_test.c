/*
 * AraVoice Ring Buffer Test
 * ==========================
 * Demonstrates async audio production with the lock-free ring buffer.
 * One thread produces samples, audio callback consumes them.
 *
 * This pattern is how TTS will feed audio to the kernel.
 */

#include "audio.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <stdatomic.h>
#include <pthread.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ------------------------------------------------------------------------- */
/* Globals                                                                   */
/* ------------------------------------------------------------------------- */

static atomic_int g_running = 1;
static AvRingBuffer *g_ringbuf = NULL;

static void handle_signal(int sig) {
    (void)sig;
    atomic_store(&g_running, 0);
}

/* ------------------------------------------------------------------------- */
/* Audio Callback - Consumer                                                 */
/* ------------------------------------------------------------------------- */

static int render_from_ringbuf(float *out, int frames, void *user_data) {
    (void)user_data;

    int read = av_ringbuf_read(g_ringbuf, out, frames);
    if (read < frames) {
        /* Underrun - ringbuf already zero-filled the rest */
    }

    return 0;
}

/* ------------------------------------------------------------------------- */
/* Producer Thread - Simulates TTS generating audio                          */
/* ------------------------------------------------------------------------- */

typedef struct {
    float freq;
    float sample_rate;
    int channels;
} ProducerArgs;

static void *producer_thread(void *arg) {
    ProducerArgs *args = (ProducerArgs *)arg;

    float phase = 0.0f;
    const float two_pi = 2.0f * (float)M_PI;
    const float phase_inc = two_pi * args->freq / args->sample_rate;
    const float amplitude = 0.5f;

    /* Produce audio in chunks, simulating TTS output */
    const int chunk_frames = 1024;
    float *chunk = (float *)malloc(chunk_frames * args->channels * sizeof(float));

    while (atomic_load(&g_running)) {
        /* Generate a chunk of sine wave */
        for (int i = 0; i < chunk_frames; i++) {
            float sample = amplitude * sinf(phase);
            chunk[2 * i + 0] = sample;
            chunk[2 * i + 1] = sample;
            phase += phase_inc;
            if (phase >= two_pi) phase -= two_pi;
        }

        /* Wait for space in ring buffer */
        while (atomic_load(&g_running) &&
               av_ringbuf_write_available(g_ringbuf) < chunk_frames) {
            struct timespec ts = {0, 1000000}; /* 1ms */
            nanosleep(&ts, NULL);
        }

        /* Write to ring buffer */
        av_ringbuf_write(g_ringbuf, chunk, chunk_frames);
    }

    free(chunk);
    return NULL;
}

/* ------------------------------------------------------------------------- */
/* Main                                                                      */
/* ------------------------------------------------------------------------- */

int main(void) {
    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);

    /* Initialize audio */
    AvDeviceConfig cfg = AV_DEFAULT_CONFIG;
    if (av_init(&cfg) != 0) {
        fprintf(stderr, "Failed to initialize audio\n");
        return 1;
    }

    /* Create ring buffer (~500ms worth of audio) */
    int ringbuf_frames = cfg.sample_rate / 2;
    g_ringbuf = av_ringbuf_create(ringbuf_frames, cfg.channels);
    if (!g_ringbuf) {
        fprintf(stderr, "Failed to create ring buffer\n");
        av_shutdown();
        return 1;
    }

    /* Start producer thread */
    ProducerArgs args = {
        .freq = 440.0f,
        .sample_rate = (float)cfg.sample_rate,
        .channels = cfg.channels
    };

    pthread_t producer;
    if (pthread_create(&producer, NULL, producer_thread, &args) != 0) {
        fprintf(stderr, "Failed to create producer thread\n");
        av_ringbuf_destroy(g_ringbuf);
        av_shutdown();
        return 1;
    }

    /* Let producer fill buffer a bit */
    struct timespec ts = {0, 100000000}; /* 100ms */
    nanosleep(&ts, NULL);

    /* Start audio playback */
    if (av_start_stream(render_from_ringbuf, NULL) != 0) {
        fprintf(stderr, "Failed to start audio stream\n");
        atomic_store(&g_running, 0);
        pthread_join(producer, NULL);
        av_ringbuf_destroy(g_ringbuf);
        av_shutdown();
        return 1;
    }

    printf("\n");
    printf("  ╔═══════════════════════════════════════╗\n");
    printf("  ║     AraVoice Ring Buffer Test         ║\n");
    printf("  ╠═══════════════════════════════════════╣\n");
    printf("  ║  Producer thread → Ring Buffer        ║\n");
    printf("  ║  Ring Buffer → Audio Callback         ║\n");
    printf("  ║                                       ║\n");
    printf("  ║  This is how TTS will feed audio.     ║\n");
    printf("  ║                                       ║\n");
    printf("  ║  Press Ctrl+C to stop                 ║\n");
    printf("  ╚═══════════════════════════════════════╝\n");
    printf("\n");

    /* Wait for signal */
    while (atomic_load(&g_running)) {
        nanosleep(&ts, NULL);
    }

    /* Cleanup */
    printf("\nStopping...\n");
    pthread_join(producer, NULL);
    av_stop_stream();
    av_ringbuf_destroy(g_ringbuf);
    av_shutdown();
    printf("Done.\n");

    return 0;
}
