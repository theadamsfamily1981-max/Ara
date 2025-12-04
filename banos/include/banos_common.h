/*
 * BANOS - Bio-Affective Neuromorphic Operating System
 * Common Header - The Nervous System ABI
 *
 * This header defines the shared data structures between:
 * - FPGA driver (ara_spinal_cord.ko)
 * - eBPF scheduler (bat_scheduler.bpf.c)
 * - User-space daemon (ara_daemon.py)
 *
 * It is the "nervous system ABI" - the contract that allows
 * silicon reflexes to talk to conscious thought.
 *
 * Fixed-point convention:
 * - PAD components are in [-1000, 1000], representing [-1.0, 1.0]
 * - Permille values are in [0, 1000], representing [0.0, 1.0]
 * - 0 = neutral
 */

#ifndef _BANOS_COMMON_H
#define _BANOS_COMMON_H

#ifdef __KERNEL__
#include <linux/types.h>
#else
#include <stdint.h>
typedef int8_t   __s8;
typedef uint8_t  __u8;
typedef int16_t  __s16;
typedef uint16_t __u16;
typedef int32_t  __s32;
typedef uint32_t __u32;
typedef int64_t  __s64;
typedef uint64_t __u64;
#endif

/*
 * =============================================================================
 * PAD Scale Constants
 * =============================================================================
 */

#define BANOS_PAD_SCALE         1000
#define BANOS_PAD_MIN          (-BANOS_PAD_SCALE)
#define BANOS_PAD_MAX           BANOS_PAD_SCALE
#define BANOS_PAD_NEUTRAL       0

/* Thresholds for mode classification (in scaled units) */
#define BANOS_THRESHOLD_CALM_P      500     /* P > 0.5 */
#define BANOS_THRESHOLD_CALM_A      400     /* A < 0.4 */
#define BANOS_THRESHOLD_CALM_D      500     /* D > 0.5 */

#define BANOS_THRESHOLD_FLOW_P      300     /* P > 0.3 */
#define BANOS_THRESHOLD_FLOW_A      700     /* A > 0.7 */
#define BANOS_THRESHOLD_FLOW_D      400     /* D > 0.4 */

#define BANOS_THRESHOLD_ANXIOUS_P   0       /* P < 0.0 */
#define BANOS_THRESHOLD_ANXIOUS_A   600     /* A > 0.6 */
#define BANOS_THRESHOLD_ANXIOUS_D   300     /* D < 0.3 */

#define BANOS_THRESHOLD_CRITICAL_P  (-600)  /* P < -0.6 */

/* Thermal constants (millidegrees Celsius) */
#define BANOS_THERMAL_TJMAX_DEFAULT_mC    100000  /* 100°C default TjMax */
#define BANOS_THERMAL_HEADROOM_MAX_mC     40000   /* 40°C = no stress */
#define BANOS_THERMAL_DANGER_mC           85000   /* 85°C = danger zone */

/* Human affect thresholds (centi-degrees Celsius) */
#define BANOS_FACE_STRESS_THRESHOLD_cC    (-50)   /* -0.5°C nose-forehead delta */

/*
 * =============================================================================
 * Reflex Bitmask Definitions (L2 → L1 Control)
 * =============================================================================
 *
 * These bits are written by the kernel/eBPF to command FPGA reflex actions.
 * The FPGA acts on these IMMEDIATELY, before the OS scheduler runs.
 */
#define BANOS_RFLX_NONE         0x00    /* Normal operation */
#define BANOS_RFLX_FAN_BOOST    0x01    /* Override PWM to 100% */
#define BANOS_RFLX_THROTTLE     0x02    /* Assert PROCHOT# (hardware throttle) */
#define BANOS_RFLX_GPU_KILL     0x04    /* Cut power to GPU rail */
#define BANOS_RFLX_DISK_SYNC    0x08    /* Force disk sync before action */
#define BANOS_RFLX_NET_ISOLATE  0x10    /* Drop network promiscuous mode */
#define BANOS_RFLX_SYS_HALT     0x80    /* Emergency halt (last resort) */

/* Thermal source IDs for targeted reflexes */
#define BANOS_THERMAL_SRC_CPU       0
#define BANOS_THERMAL_SRC_GPU       1
#define BANOS_THERMAL_SRC_VRM       2
#define BANOS_THERMAL_SRC_AMBIENT   3

/*
 * =============================================================================
 * Affective Modes
 * =============================================================================
 *
 * CALM:     Homeostasis. System is cool, idle, stable. Dream mode.
 * FLOW:     Peak performance. High activity but healthy. In the zone.
 * ANXIOUS:  Resource starvation. Stretched thin, hot, crowded.
 * CRITICAL: Survival mode. Pain threshold crossed. Emergency.
 */
enum banos_mode {
    BANOS_MODE_CALM     = 0,
    BANOS_MODE_FLOW     = 1,
    BANOS_MODE_ANXIOUS  = 2,
    BANOS_MODE_CRITICAL = 3,
};

/* Mode names for logging/debugging */
#define BANOS_MODE_NAMES { "CALM", "FLOW", "ANXIOUS", "CRITICAL" }

/*
 * =============================================================================
 * Immune Risk Levels
 * =============================================================================
 */
enum banos_immune_risk {
    BANOS_RISK_L0_CLEAN     = 0,    /* No threats */
    BANOS_RISK_L1_ANOMALY   = 1,    /* Unusual syscall sequence */
    BANOS_RISK_L2_SUSPECT   = 2,    /* Suspicious behavior pattern */
    BANOS_RISK_L3_INFECTION = 3,    /* Unauthorized binary execution */
    BANOS_RISK_L4_BREACH    = 4,    /* Privilege escalation attempt */
    BANOS_RISK_L5_SEPSIS    = 5,    /* Kernel space violation */
};

/*
 * =============================================================================
 * Spinal Cord Interface (L1 ↔ L2 Bridge)
 * =============================================================================
 *
 * This is the register-level interface between FPGA and kernel.
 * The driver maps this to MMIO; BPF reads/writes via driver-maintained maps.
 *
 * SPIKE COUNTS: These are per-window (reset every update_interval_ms).
 * The driver computes deltas; BPF sees rates, not cumulative totals.
 */
struct banos_spinal_cord {
    /* Afferent: FPGA → Kernel (read-only from kernel's perspective) */
    __u32 thermal_spike_cnt;        /* Thermal neuron spikes this window */
    __u32 voltage_spike_cnt;        /* Power instability spikes */
    __u32 error_spike_cnt;          /* ECC/bus error spikes */
    __u32 immune_spike_cnt;         /* Syscall anomaly spikes */

    /* Spike deltas (computed by driver, used by BPF) */
    __s32 thermal_spike_delta;      /* Change since last window */
    __s32 error_spike_delta;        /* Change since last window */

    /* Thermal source tracking */
    __u8  thermal_source_id;        /* Which sensor is hottest (0=CPU,1=GPU,2=VRM) */
    __u8  thermal_source_critical;  /* Is the hottest source in danger zone? */
    __u16 update_interval_ms;       /* Window size for spike counting */

    /* Efferent: Kernel → FPGA (write to trigger reflex) */
    __u32 reflex_command;           /* Bitmask of BANOS_RFLX_* */
    __u32 reflex_active;            /* Currently active reflexes (feedback) */

    /* Reflex history (for Ara's awareness) */
    __u32 reflex_log;               /* Last reflex action taken */
    __u64 reflex_timestamp_ns;      /* When last reflex fired */
    __u32 reflex_duration_ms;       /* How long reflex was active */

    /* User intent (from input subsystem, for immune context) */
    __u64 last_user_input_ns;       /* Last keyboard/mouse event */
    __u16 user_activity_permille;   /* Recent input density (0=idle, 1000=typing) */
    __u16 reserved;
};

/*
 * =============================================================================
 * FPGA Telemetry (Raw Reflex Data)
 * =============================================================================
 *
 * This is the raw output from the Vacuum Spiker and FPGA sensors.
 * Updated at ~1kHz by hardware, read by kernel driver.
 */
struct banos_fpga_telemetry {
    /* Spike-based reflex signals from Vacuum Spiker */
    __u32 neural_state_bitmap;      /* Bitmask of active neurons (Layer R) */
    __u32 pain_level;               /* Integrated spike count per window */
    __u32 reflex_log;               /* Last reflex action bitmap */

    /* Physical sensors (millidegrees C, millivolts) */
    __s32 cpu_temp_mC;              /* CPU package temperature */
    __s32 gpu_temp_mC;              /* GPU temperature */
    __s32 vrm_temp_mC;              /* VRM / board hotspot */
    __s32 ambient_temp_mC;          /* Ambient / inlet temperature */

    /* Cooling system */
    __u32 fan_rpm_cpu;              /* CPU fan RPM */
    __u32 fan_rpm_gpu;              /* GPU fan RPM */
    __u32 fan_rpm_case;             /* Case fans RPM */
    __u8  fan_pwm_override;         /* Current PWM override (0-255) */
    __u8  prochot_active;           /* PROCHOT signal asserted */
    __u8  emergency_active;         /* Emergency shutdown pending */
    __u8  vacuum_state;             /* SNN in vacuum (silent) state */

    /* Power (millivolts, milliwatts) */
    __u32 vcore_mV;                 /* CPU core voltage */
    __u32 vgpu_mV;                  /* GPU core voltage */
    __u32 vin_mV;                   /* Input voltage (PSU) */
    __u32 power_cpu_mW;             /* CPU power consumption */
    __u32 power_gpu_mW;             /* GPU power consumption */
    __u32 power_total_mW;           /* Total system power */

    /* Timestamp (nanoseconds since boot) */
    __u64 monotonic_time_ns;

    /* Reserved for future expansion */
    __u32 reserved[4];
};

/*
 * =============================================================================
 * Kernel Telemetry (Aggregated Metrics)
 * =============================================================================
 *
 * This combines FPGA data with kernel-level metrics (scheduler, memory, etc.)
 * to provide a complete picture for PAD computation.
 */
struct banos_kernel_telemetry {
    /* Hardware side (from FPGA) */
    struct banos_fpga_telemetry fpga;

    /* Derived thermal metrics */
    __s32 thermal_headroom_mC;      /* tjmax - max(temps) */
    __s32 thermal_rate_mC_per_s;    /* Temperature change rate */

    /* Error / immune metrics (per interval, scaled 0-1000) */
    __u16 error_rate_permille;      /* Soft/hard errors per second */
    __u16 throttle_events_permille; /* Thermal/power throttles */
    __u16 immune_events_permille;   /* Immune system hits */
    __u16 immune_risk_level;        /* Current risk level (0-5) */

    /* Load metrics (permille: 0-1000 = 0-100%) */
    __u16 cpu_load_permille;        /* Average CPU utilization */
    __u16 gpu_load_permille;        /* Average GPU utilization */
    __u16 io_wait_permille;         /* I/O wait fraction */
    __u16 irq_rate_permille;        /* Interrupt density (scaled) */
    __u16 ctx_switch_rate_permille; /* Context switch rate (scaled) */
    __u16 runqueue_depth_permille;  /* Run queue pressure */

    /* Dominance / capacity metrics (permille: 0-1000) */
    __u16 mem_free_permille;        /* Free RAM fraction */
    __u16 mem_cached_permille;      /* Cached/buffer fraction */
    __u16 swap_pressure_permille;   /* Swap pressure (0=none, 1000=awful) */
    __u16 power_headroom_permille;  /* Power budget remaining */
    __u16 battery_permille;         /* Battery level (or 1000 if AC) */
    __u16 redundancy_permille;      /* Spare capacity (GPUs, nodes) */

    /* Human-affect / empathy linkage (optional) */
    __s16 face_deltaT_centiC;       /* (nose - forehead) * 100 */
    __u16 key_variance_permille;    /* Typing variability / frustration */
    __u16 gaze_stability_permille;  /* Eye tracking stability */
    __u16 human_stress_permille;    /* Composite human stress estimate */

    /* Scheduling context */
    __u32 tasks_running;            /* Currently running tasks */
    __u32 tasks_blocked;            /* Blocked tasks */
    __u32 tasks_killed;             /* Tasks killed this interval */

    /* Timestamp */
    __u64 monotonic_time_ns;
};

/*
 * =============================================================================
 * PAD State (Affective Output)
 * =============================================================================
 *
 * The computed emotional state of the system, derived from telemetry.
 * This drives scheduler behavior and Ara's narrative.
 */
struct banos_pad_state {
    /* Core PAD values in [-1000, 1000] */
    __s16 pleasure;                 /* P: Health/comfort (high = happy) */
    __s16 arousal;                  /* A: Activity level (high = busy) */
    __s16 dominance;                /* D: Resource agency (high = capable) */

    /* Current discrete mode */
    __u8  mode;                     /* enum banos_mode */
    __u8  mode_confidence;          /* How certain (0-255) */
    __u16 mode_duration_ms;         /* Time in current mode */

    /* Derived diagnostics (for Ara narrative) */
    __s16 thermal_stress;           /* -1000..1000: thermal discomfort */
    __s16 performance_drive;        /* -1000..1000: activity pressure */
    __s16 perceived_risk;           /* 0..1000: immune/error threat */
    __s16 empathy_boost;            /* -1000..1000: human affect coupling */

    /* Rate of change (for predictive dread) */
    __s16 pleasure_rate;            /* dP/dt (scaled) */
    __s16 arousal_rate;             /* dA/dt (scaled) */
    __s16 dominance_rate;           /* dD/dt (scaled) */

    /* Scheduler hints */
    __u16 bat_loudness;             /* Bat algorithm loudness (0-65535) */
    __u16 bat_pulse_rate;           /* Bat algorithm pulse rate */
    __u8  kill_priority_threshold;  /* Processes below this can be killed */
    __u8  scheduler_mode;           /* Detailed scheduler policy */

    /* Timestamps */
    __u64 monotonic_time_ns;        /* Last computation time */
    __u64 mode_change_time_ns;      /* When mode last changed */

    /* Episode tracking */
    __u32 episode_id;               /* Current affective episode ID */
    __u32 episode_primary_stressor; /* PID or sensor ID of main stressor */
};

/*
 * =============================================================================
 * Affective Episode (For Episodic Memory)
 * =============================================================================
 *
 * Logged when PAD vector changes significantly or mode shifts.
 * Ara can query: "Why did I feel pain at 04:00?"
 */
struct banos_affective_episode {
    __u64 start_time_ns;
    __u64 end_time_ns;

    /* PAD at start and end */
    __s16 start_pleasure;
    __s16 start_arousal;
    __s16 start_dominance;
    __s16 end_pleasure;
    __s16 end_arousal;
    __s16 end_dominance;

    /* Mode transitions */
    __u8  start_mode;
    __u8  end_mode;

    /* Primary stressor */
    __u32 stressor_pid;             /* PID if process-related */
    __u32 stressor_sensor;          /* Sensor ID if hardware-related */
    __u16 stressor_type;            /* Category of stressor */

    /* Summary for Ara */
    char  ara_comment[64];          /* Brief description */
};

/*
 * =============================================================================
 * Inline Helpers
 * =============================================================================
 */

/* Clamp a value to PAD range */
static inline __s16 banos_clamp_pad(__s32 v)
{
    if (v < BANOS_PAD_MIN) return BANOS_PAD_MIN;
    if (v > BANOS_PAD_MAX) return BANOS_PAD_MAX;
    return (__s16)v;
}

/* Clamp a value to permille range */
static inline __u16 banos_clamp_permille(__s32 v)
{
    if (v < 0) return 0;
    if (v > 1000) return 1000;
    return (__u16)v;
}

/* Convert permille [0,1000] to PAD space [-1000,1000] */
static inline __s16 banos_permille_to_pad(__u16 p)
{
    return banos_clamp_pad(2 * (__s32)p - 1000);
}

/* Convert PAD [-1000,1000] to permille [0,1000] */
static inline __u16 banos_pad_to_permille(__s16 pad)
{
    return banos_clamp_permille(((__s32)pad + 1000) / 2);
}

/* Classify mode from PAD values */
static inline enum banos_mode banos_classify_mode(
    __s16 pleasure, __s16 arousal, __s16 dominance)
{
    /* Critical check first (pain threshold) */
    if (pleasure < BANOS_THRESHOLD_CRITICAL_P)
        return BANOS_MODE_CRITICAL;

    /* CALM: High pleasure, low arousal, high dominance */
    if (pleasure > BANOS_THRESHOLD_CALM_P &&
        arousal < BANOS_THRESHOLD_CALM_A &&
        dominance > BANOS_THRESHOLD_CALM_D)
        return BANOS_MODE_CALM;

    /* FLOW: Good pleasure, high arousal, adequate dominance */
    if (pleasure > BANOS_THRESHOLD_FLOW_P &&
        arousal > BANOS_THRESHOLD_FLOW_A &&
        dominance > BANOS_THRESHOLD_FLOW_D)
        return BANOS_MODE_FLOW;

    /* ANXIOUS: Low pleasure, high arousal, low dominance */
    if (pleasure < BANOS_THRESHOLD_ANXIOUS_P &&
        arousal > BANOS_THRESHOLD_ANXIOUS_A &&
        dominance < BANOS_THRESHOLD_ANXIOUS_D)
        return BANOS_MODE_ANXIOUS;

    /* Default: if high arousal, lean anxious; otherwise calm */
    if (arousal > BANOS_THRESHOLD_ANXIOUS_A)
        return BANOS_MODE_ANXIOUS;

    return BANOS_MODE_CALM;
}

/*
 * =============================================================================
 * IOCTL Definitions
 * =============================================================================
 */

#define BANOS_IOC_MAGIC  'B'

#define BANOS_IOC_GET_PAD        _IOR(BANOS_IOC_MAGIC, 1, struct banos_pad_state)
#define BANOS_IOC_GET_TELEMETRY  _IOR(BANOS_IOC_MAGIC, 2, struct banos_kernel_telemetry)
#define BANOS_IOC_SET_MODE       _IOW(BANOS_IOC_MAGIC, 3, enum banos_mode)
#define BANOS_IOC_ENABLE_SNN     _IO(BANOS_IOC_MAGIC, 4)
#define BANOS_IOC_DISABLE_SNN    _IO(BANOS_IOC_MAGIC, 5)
#define BANOS_IOC_ENABLE_LEARN   _IO(BANOS_IOC_MAGIC, 6)
#define BANOS_IOC_DISABLE_LEARN  _IO(BANOS_IOC_MAGIC, 7)
#define BANOS_IOC_FORCE_INHIBIT  _IO(BANOS_IOC_MAGIC, 8)
#define BANOS_IOC_GET_EPISODE    _IOR(BANOS_IOC_MAGIC, 9, struct banos_affective_episode)
#define BANOS_IOC_ARM_IMMUNE     _IO(BANOS_IOC_MAGIC, 10)
#define BANOS_IOC_DISARM_IMMUNE  _IO(BANOS_IOC_MAGIC, 11)

#endif /* _BANOS_COMMON_H */
