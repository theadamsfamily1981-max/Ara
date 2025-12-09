/*
 * AraVoice Sine Test - "Hello Audio World"
 * =========================================
 * Plays a 440Hz sine wave to verify the audio kernel works.
 *
 * Build:
 *   gcc -o sine_test sine_test.c ../src/audio_portaudio.c \
 *       -I../include -lportaudio -lm
 *
 * Run:
 *   ./sine_test          # Default 440Hz
 *   ./sine_test 880      # Custom frequency
 */

#include "audio.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <stdatomic.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ------------------------------------------------------------------------- */
/* Signal Handler                                                            */
/* ------------------------------------------------------------------------- */

static atomic_int g_running = 1;

static void handle_signal(int sig) {
    (void)sig;
    atomic_store(&g_running, 0);
}

/* ------------------------------------------------------------------------- */
/* Sine Wave Generator                                                       */
/* ------------------------------------------------------------------------- */

typedef struct {
    float phase;
    float freq;
    float sample_rate;
    float amplitude;
} SineState;

static int render_sine(float *out, int frames, void *user_data) {
    SineState *state = (SineState *)user_data;
    const float two_pi = 2.0f * (float)M_PI;
    const float phase_inc = two_pi * state->freq / state->sample_rate;

    for (int i = 0; i < frames; i++) {
        float sample = state->amplitude * sinf(state->phase);

        /* Stereo output */
        out[2 * i + 0] = sample;  /* Left */
        out[2 * i + 1] = sample;  /* Right */

        state->phase += phase_inc;
        if (state->phase >= two_pi) {
            state->phase -= two_pi;
        }
    }

    return 0;
}

/* ------------------------------------------------------------------------- */
/* Main                                                                      */
/* ------------------------------------------------------------------------- */

int main(int argc, char *argv[]) {
    /* Parse frequency argument */
    float freq = 440.0f;
    if (argc > 1) {
        freq = (float)atof(argv[1]);
        if (freq < 20.0f || freq > 20000.0f) {
            fprintf(stderr, "Frequency must be between 20 and 20000 Hz\n");
            return 1;
        }
    }

    /* Setup signal handler for clean exit */
    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);

    /* Initialize audio */
    AvDeviceConfig cfg = AV_DEFAULT_CONFIG;
    if (av_init(&cfg) != 0) {
        fprintf(stderr, "Failed to initialize audio\n");
        return 1;
    }

    /* Setup sine generator */
    SineState state = {
        .phase = 0.0f,
        .freq = freq,
        .sample_rate = (float)cfg.sample_rate,
        .amplitude = 0.5f  /* -6dB to avoid clipping */
    };

    /* Start playback */
    if (av_start_stream(render_sine, &state) != 0) {
        fprintf(stderr, "Failed to start audio stream\n");
        av_shutdown();
        return 1;
    }

    printf("\n");
    printf("  ╔═══════════════════════════════════════╗\n");
    printf("  ║       AraVoice Audio Kernel Test      ║\n");
    printf("  ╠═══════════════════════════════════════╣\n");
    printf("  ║  Frequency: %7.1f Hz                ║\n", freq);
    printf("  ║  Latency:   %7.2f ms                ║\n", av_get_latency_ms());
    printf("  ║                                       ║\n");
    printf("  ║  Press Ctrl+C to stop                 ║\n");
    printf("  ╚═══════════════════════════════════════╝\n");
    printf("\n");

    /* Wait for signal */
    while (atomic_load(&g_running)) {
        struct timespec ts = {0, 100000000}; /* 100ms */
        nanosleep(&ts, NULL);
    }

    /* Cleanup */
    printf("\nStopping...\n");
    av_stop_stream();
    av_shutdown();
    printf("Done.\n");

    return 0;
}
