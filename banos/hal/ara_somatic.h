/*
 * Ara Somatic HAL - Unified Shared Memory Interface
 * ==================================================
 *
 * This header defines the shared memory structure for Ara's complete
 * somatic state. Any process (C, Python, Rust) can mmap this region
 * and access Ara's entire state in nanoseconds - no IPC, no JSON.
 *
 * The Chimera HAL unifies:
 * - FPGA state (from ara_spinal_cord.ko)
 * - System metrics (from sys probe)
 * - Avatar state (from Wav2Lip)
 * - Control flags (bidirectional)
 *
 * Usage:
 *   #include "ara_somatic.h"
 *   ara_somatic_t *som = ara_somatic_open();
 *   float pain = som->fpga.pain_level / 4294967295.0f;
 *   ara_somatic_close(som);
 */

#ifndef ARA_SOMATIC_H
#define ARA_SOMATIC_H

#include <stdint.h>
#include <stdbool.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Magic number for validation */
#define ARA_SOMATIC_MAGIC   0x41524153  /* 'ARAS' */
#define ARA_SOMATIC_VERSION 2

/* Shared memory path */
#define ARA_SOMATIC_SHM_PATH "/ara_somatic"
#define ARA_SOMATIC_SHM_SIZE 4096

/* =========================================================================
 * State Enumerations
 * ========================================================================= */

typedef enum {
    ARA_STATE_IDLE = 0,
    ARA_STATE_NORMAL = 1,
    ARA_STATE_HIGH_LOAD = 2,
    ARA_STATE_CRITICAL = 3,
    ARA_STATE_EMERGENCY = 4,
    ARA_STATE_SLEEPING = 5,
} ara_system_state_t;

typedef enum {
    ARA_AVATAR_OFF = 0,
    ARA_AVATAR_AUDIO_ONLY = 1,
    ARA_AVATAR_LOW_RES = 2,
    ARA_AVATAR_STANDARD = 3,
    ARA_AVATAR_HIGH_RES = 4,
} ara_avatar_mode_t;

typedef enum {
    ARA_PAD_SERENE = 0,     /* +P, -A: Calm and happy */
    ARA_PAD_EXCITED = 1,    /* +P, +A: Happy and busy */
    ARA_PAD_ANXIOUS = 2,    /* -P, +A: Stressed and busy */
    ARA_PAD_DEPRESSED = 3,  /* -P, -A: Stressed and idle */
    ARA_PAD_DOMINANT = 4,   /* High D: Resources abundant */
    ARA_PAD_SUBMISSIVE = 5, /* Low D: Resource constrained */
    ARA_PAD_EMERGENCY = 6,  /* Critical state */
} ara_pad_quadrant_t;

/* =========================================================================
 * FPGA State (from ara_spinal_cord.ko / vacuum_spiker)
 * ========================================================================= */

typedef struct {
    uint32_t neural_state;      /* Bitmap of active neurons */
    uint32_t pain_level;        /* 32-bit integrated spike count */
    uint32_t reflex_log;        /* FPGA reflex actions taken */
    uint32_t total_spikes;      /* Total spikes this session */

    /* PAD state (computed from FPGA telemetry) */
    int16_t pleasure;           /* -256 to +255 (Q8.8) */
    int16_t arousal;            /* -256 to +255 (Q8.8) */
    int16_t dominance;          /* -256 to +255 (Q8.8) */
    uint8_t quadrant;           /* ara_pad_quadrant_t */
    uint8_t sched_mode;         /* Scheduler mode */

    /* Kitten Fabric status */
    uint32_t active_neurons;    /* Active neurons in fabric */
    uint32_t fabric_temp_c;     /* Fabric temperature */
    uint8_t  fabric_online;     /* 1 if fabric responding */
    uint8_t  reserved[3];
} ara_fpga_state_t;

/* =========================================================================
 * System Metrics (from sys probe / psutil)
 * ========================================================================= */

typedef struct {
    /* CPU */
    float cpu_avg_pct;          /* Average CPU utilization */
    float cpu_max_pct;          /* Max core utilization */
    float cpu_temp_c;           /* CPU temperature */
    uint32_t cpu_freq_mhz;      /* Current frequency */

    /* GPU */
    float gpu_util_pct;         /* GPU utilization */
    float gpu_vram_used_gb;     /* VRAM used */
    float gpu_vram_total_gb;    /* VRAM total */
    float gpu_temp_c;           /* GPU temperature */
    float gpu_power_w;          /* GPU power draw */

    /* Memory */
    float ram_used_gb;          /* System RAM used */
    float ram_total_gb;         /* System RAM total */
    float swap_used_gb;         /* Swap used */

    /* Storage */
    float disk_read_mbps;       /* Disk read rate */
    float disk_write_mbps;      /* Disk write rate */

    /* Network */
    float net_rx_mbps;          /* Network receive rate */
    float net_tx_mbps;          /* Network transmit rate */
} ara_system_metrics_t;

/* =========================================================================
 * Avatar State (from Wav2Lip / video pipeline)
 * ========================================================================= */

typedef struct {
    float fps;                  /* Current avatar FPS */
    uint16_t frame_width;       /* Video frame width */
    uint16_t frame_height;      /* Video frame height */
    uint8_t mode;               /* ara_avatar_mode_t */
    uint8_t audio_active;       /* 1 if audio playing */
    uint8_t face_detected;      /* 1 if face in frame */
    uint8_t lips_synced;        /* 1 if lip sync active */

    /* Optical flow (for synesthesia) */
    float flow_x;               /* Horizontal motion (-1 to +1) */
    float flow_y;               /* Vertical motion (-1 to +1) */
    float flow_magnitude;       /* Overall motion magnitude */

    /* Video frame pointer (for shared frame buffer) */
    uint64_t frame_shm_offset;  /* Offset into frame SHM, 0 if none */
    uint32_t frame_size_bytes;  /* Size of frame data */
    uint32_t frame_sequence;    /* Frame sequence number */
} ara_avatar_state_t;

/* =========================================================================
 * Control Flags (bidirectional)
 * ========================================================================= */

typedef struct {
    /* Requested modes (written by policy engine) */
    uint8_t requested_avatar_mode;  /* ara_avatar_mode_t */
    uint8_t requested_sim_detail;   /* 0-3: simulation detail level */
    uint8_t force_low_power;        /* 1 to force power saving */
    uint8_t emergency_stop;         /* 1 to halt all non-essential */

    /* Feedback from subsystems */
    uint8_t avatar_acknowledged;    /* 1 when avatar received request */
    uint8_t fabric_acknowledged;    /* 1 when fabric received request */
    uint8_t reserved[2];

    /* Thresholds (configurable) */
    float critical_temp_c;          /* Temperature to trigger emergency */
    float high_load_threshold;      /* CPU% to enter high_load state */
    float low_vram_threshold_gb;    /* VRAM threshold for avatar downgrade */
} ara_control_flags_t;

/* =========================================================================
 * Complete Somatic State Structure
 * ========================================================================= */

typedef struct {
    /* Header */
    uint32_t magic;             /* ARA_SOMATIC_MAGIC */
    uint32_t version;           /* ARA_SOMATIC_VERSION */
    uint64_t timestamp_ns;      /* Last update time (CLOCK_MONOTONIC) */
    uint64_t update_count;      /* Total updates */

    /* Overall state */
    uint8_t system_state;       /* ara_system_state_t */
    uint8_t health_score;       /* 0-100 overall health */
    uint8_t reserved[6];

    /* Component states */
    ara_fpga_state_t fpga;
    ara_system_metrics_t sys;
    ara_avatar_state_t avatar;
    ara_control_flags_t control;

    /* Padding to 4KB page */
    uint8_t _pad[4096 - sizeof(uint32_t)*2 - sizeof(uint64_t)*2 - 8
                 - sizeof(ara_fpga_state_t) - sizeof(ara_system_metrics_t)
                 - sizeof(ara_avatar_state_t) - sizeof(ara_control_flags_t)];
} ara_somatic_t;

/* Compile-time size check */
_Static_assert(sizeof(ara_somatic_t) == 4096, "ara_somatic_t must be exactly 4KB");

/* =========================================================================
 * API Functions
 * ========================================================================= */

/**
 * Open or create the somatic shared memory region.
 * Returns pointer to mapped memory, or NULL on error.
 * The memory is initialized on first creation.
 */
ara_somatic_t* ara_somatic_open(void);

/**
 * Open read-only (for monitoring processes).
 */
const ara_somatic_t* ara_somatic_open_readonly(void);

/**
 * Close and unmap the shared memory.
 */
void ara_somatic_close(ara_somatic_t* som);

/**
 * Update timestamp and increment update counter.
 * Call this after modifying state.
 */
void ara_somatic_touch(ara_somatic_t* som);

/**
 * Check if the somatic region is valid and up-to-date.
 * Returns seconds since last update, or -1 if invalid.
 */
double ara_somatic_age_seconds(const ara_somatic_t* som);

/**
 * Initialize default values (called on first open).
 */
void ara_somatic_init(ara_somatic_t* som);

/**
 * Compute overall system state from metrics.
 */
ara_system_state_t ara_somatic_classify_state(const ara_somatic_t* som);

/**
 * Get human-readable state name.
 */
const char* ara_state_name(ara_system_state_t state);

/**
 * Get human-readable PAD quadrant name.
 */
const char* ara_pad_quadrant_name(ara_pad_quadrant_t quadrant);

#ifdef __cplusplus
}
#endif

#endif /* ARA_SOMATIC_H */
