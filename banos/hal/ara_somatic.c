/*
 * Ara Somatic HAL - Implementation
 * =================================
 *
 * Provides unified shared memory access for Ara's somatic state.
 * See ara_somatic.h for full documentation.
 */

#define _GNU_SOURCE
#include "ara_somatic.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>

/* =========================================================================
 * Internal Helpers
 * ========================================================================= */

static uint64_t get_timestamp_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* =========================================================================
 * API Implementation
 * ========================================================================= */

ara_somatic_t* ara_somatic_open(void)
{
    int fd;
    ara_somatic_t* som;
    int created = 0;

    /* Try to open existing */
    fd = shm_open(ARA_SOMATIC_SHM_PATH, O_RDWR, 0666);
    if (fd < 0) {
        /* Create new */
        fd = shm_open(ARA_SOMATIC_SHM_PATH, O_RDWR | O_CREAT | O_EXCL, 0666);
        if (fd < 0) {
            /* Race condition: try open again */
            fd = shm_open(ARA_SOMATIC_SHM_PATH, O_RDWR, 0666);
            if (fd < 0) {
                perror("ara_somatic_open: shm_open failed");
                return NULL;
            }
        } else {
            created = 1;
            if (ftruncate(fd, ARA_SOMATIC_SHM_SIZE) < 0) {
                perror("ara_somatic_open: ftruncate failed");
                close(fd);
                shm_unlink(ARA_SOMATIC_SHM_PATH);
                return NULL;
            }
        }
    }

    /* Map the memory */
    som = (ara_somatic_t*)mmap(NULL, ARA_SOMATIC_SHM_SIZE,
                               PROT_READ | PROT_WRITE,
                               MAP_SHARED, fd, 0);
    close(fd);

    if (som == MAP_FAILED) {
        perror("ara_somatic_open: mmap failed");
        return NULL;
    }

    /* Initialize if newly created or invalid */
    if (created || som->magic != ARA_SOMATIC_MAGIC) {
        ara_somatic_init(som);
    }

    return som;
}

const ara_somatic_t* ara_somatic_open_readonly(void)
{
    int fd;
    const ara_somatic_t* som;

    fd = shm_open(ARA_SOMATIC_SHM_PATH, O_RDONLY, 0666);
    if (fd < 0) {
        perror("ara_somatic_open_readonly: shm_open failed");
        return NULL;
    }

    som = (const ara_somatic_t*)mmap(NULL, ARA_SOMATIC_SHM_SIZE,
                                     PROT_READ, MAP_SHARED, fd, 0);
    close(fd);

    if (som == MAP_FAILED) {
        perror("ara_somatic_open_readonly: mmap failed");
        return NULL;
    }

    if (som->magic != ARA_SOMATIC_MAGIC) {
        fprintf(stderr, "ara_somatic_open_readonly: invalid magic\n");
        munmap((void*)som, ARA_SOMATIC_SHM_SIZE);
        return NULL;
    }

    return som;
}

void ara_somatic_close(ara_somatic_t* som)
{
    if (som) {
        munmap(som, ARA_SOMATIC_SHM_SIZE);
    }
}

void ara_somatic_touch(ara_somatic_t* som)
{
    if (som) {
        som->timestamp_ns = get_timestamp_ns();
        som->update_count++;
    }
}

double ara_somatic_age_seconds(const ara_somatic_t* som)
{
    if (!som || som->magic != ARA_SOMATIC_MAGIC) {
        return -1.0;
    }

    uint64_t now = get_timestamp_ns();
    uint64_t age_ns = now - som->timestamp_ns;
    return (double)age_ns / 1e9;
}

void ara_somatic_init(ara_somatic_t* som)
{
    if (!som) return;

    memset(som, 0, sizeof(*som));

    /* Header */
    som->magic = ARA_SOMATIC_MAGIC;
    som->version = ARA_SOMATIC_VERSION;
    som->timestamp_ns = get_timestamp_ns();
    som->update_count = 1;

    /* Default state */
    som->system_state = ARA_STATE_IDLE;
    som->health_score = 100;

    /* FPGA defaults */
    som->fpga.quadrant = ARA_PAD_SERENE;

    /* Control defaults */
    som->control.requested_avatar_mode = ARA_AVATAR_STANDARD;
    som->control.critical_temp_c = 85.0f;
    som->control.high_load_threshold = 80.0f;
    som->control.low_vram_threshold_gb = 2.0f;
}

ara_system_state_t ara_somatic_classify_state(const ara_somatic_t* som)
{
    if (!som) return ARA_STATE_IDLE;

    /* Emergency conditions */
    if (som->control.emergency_stop) {
        return ARA_STATE_EMERGENCY;
    }

    if (som->sys.cpu_temp_c >= som->control.critical_temp_c ||
        som->sys.gpu_temp_c >= som->control.critical_temp_c) {
        return ARA_STATE_CRITICAL;
    }

    /* FPGA-based classification (if online) */
    if (som->fpga.fabric_online) {
        if (som->fpga.quadrant == ARA_PAD_EMERGENCY) {
            return ARA_STATE_EMERGENCY;
        }
        if (som->fpga.quadrant == ARA_PAD_ANXIOUS) {
            return ARA_STATE_HIGH_LOAD;
        }
    }

    /* CPU-based classification */
    if (som->sys.cpu_avg_pct >= som->control.high_load_threshold) {
        return ARA_STATE_HIGH_LOAD;
    }

    /* VRAM pressure */
    float vram_free = som->sys.gpu_vram_total_gb - som->sys.gpu_vram_used_gb;
    if (vram_free < som->control.low_vram_threshold_gb && vram_free > 0) {
        return ARA_STATE_HIGH_LOAD;
    }

    /* Low activity */
    if (som->sys.cpu_avg_pct < 10.0f && !som->avatar.audio_active) {
        return ARA_STATE_SLEEPING;
    }

    return ARA_STATE_NORMAL;
}

const char* ara_state_name(ara_system_state_t state)
{
    switch (state) {
        case ARA_STATE_IDLE:      return "idle";
        case ARA_STATE_NORMAL:    return "normal";
        case ARA_STATE_HIGH_LOAD: return "high_load";
        case ARA_STATE_CRITICAL:  return "critical";
        case ARA_STATE_EMERGENCY: return "emergency";
        case ARA_STATE_SLEEPING:  return "sleeping";
        default:                  return "unknown";
    }
}

const char* ara_pad_quadrant_name(ara_pad_quadrant_t quadrant)
{
    switch (quadrant) {
        case ARA_PAD_SERENE:     return "serene";
        case ARA_PAD_EXCITED:    return "excited";
        case ARA_PAD_ANXIOUS:    return "anxious";
        case ARA_PAD_DEPRESSED:  return "depressed";
        case ARA_PAD_DOMINANT:   return "dominant";
        case ARA_PAD_SUBMISSIVE: return "submissive";
        case ARA_PAD_EMERGENCY:  return "emergency";
        default:                 return "unknown";
    }
}
