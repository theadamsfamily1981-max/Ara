/*
 * BANOS - Bio-Affective Neuromorphic Operating System
 * PAD State Header - Pleasure-Arousal-Dominance Computation
 *
 * This header defines the affective state computation used by the
 * Bat Algorithm scheduler. Instead of "nice" values, processes have
 * metabolic cost, and scheduling decisions are driven by the system's
 * emotional state.
 *
 * The PAD model maps hardware telemetry to three dimensions:
 *   P (Pleasure)  = f(thermal_stress, error_rate) - Higher = calm/happy
 *   A (Arousal)   = f(cpu_load, io_rate)          - Higher = busy/alert
 *   D (Dominance) = f(memory_avail, battery)      - Higher = resourceful
 */

#ifndef __BANOS_PAD_STATE_H
#define __BANOS_PAD_STATE_H

#include <linux/types.h>

/* Fixed-point arithmetic: Q8.8 format (16-bit) */
#define PAD_FRAC_BITS   8
#define PAD_SCALE       (1 << PAD_FRAC_BITS)
#define PAD_ONE         PAD_SCALE
#define PAD_HALF        (PAD_SCALE >> 1)

/* Convert float to fixed-point (compile-time only) */
#define PAD_FLOAT(x)    ((s16)((x) * PAD_SCALE))

/* Fixed-point multiplication */
static inline s16 pad_mul(s16 a, s16 b)
{
    return (s16)(((s32)a * (s32)b) >> PAD_FRAC_BITS);
}

/* Fixed-point division */
static inline s16 pad_div(s16 a, s16 b)
{
    if (b == 0) return PAD_ONE;  /* Avoid div by zero */
    return (s16)(((s32)a << PAD_FRAC_BITS) / b);
}

/*
 * PAD State Structure
 * Values range from -256 to +255 (Q8.8: -1.0 to ~1.0)
 */
struct pad_state {
    s16 pleasure;       /* Inverse of stress: higher = happy */
    s16 arousal;        /* Activity level: higher = busy */
    s16 dominance;      /* Resource agency: higher = capable */
    u64 timestamp_ns;   /* When this was computed */
};

/*
 * System Telemetry for PAD Computation
 */
struct banos_telemetry {
    /* Thermal (from FPGA or hwmon) */
    u16 cpu_temp_raw;       /* 0-65535, mapped to temp range */
    u16 gpu_temp_raw;
    u32 pain_level;         /* From Vacuum Spiker (32-bit to match FPGA ABI) */

    /* Load metrics */
    u32 cpu_load_pct;       /* 0-100 * 100 (2 decimal precision) */
    u32 io_util_pct;        /* I/O utilization */
    u32 irq_rate;           /* Interrupts per second */

    /* Resources */
    u64 mem_total_kb;
    u64 mem_avail_kb;
    u32 battery_pct;        /* 0-100, or 100 if desktop */

    /* Error tracking */
    u32 error_count;        /* Recent errors (resets periodically) */
    u32 oom_events;         /* OOM killer invocations */
};

/*
 * PAD Quadrant Classification
 * Maps continuous PAD to discrete behavioral modes
 */
enum pad_quadrant {
    PAD_SERENE = 0,     /* +P, -A: Calm and happy (normal ops) */
    PAD_EXCITED,        /* +P, +A: Happy and busy (flow state) */
    PAD_ANXIOUS,        /* -P, +A: Stressed and busy (danger!) */
    PAD_DEPRESSED,      /* -P, -A: Stressed and idle (recovery) */
    PAD_DOMINANT,       /* High D: Has resources to spare */
    PAD_SUBMISSIVE,     /* Low D: Resource constrained */
    PAD_EMERGENCY,      /* Critical: Emergency mode */
};

/*
 * Compute PAD from telemetry
 *
 * Pleasure = 1/ThermalStress + 1/ErrorRate (clamped to [-1, 1])
 * Arousal  = CPULoad + IORate (clamped)
 * Dominance = MemAvail/MemTotal + Battery/100 (clamped)
 */
static inline void compute_pad(const struct banos_telemetry *tel,
                               struct pad_state *pad)
{
    s32 p, a, d;

    /* Pleasure: inverse of thermal stress and errors */
    /* Scale pain_level (0-2^32) to thermal_stress (0-256) */
    s16 thermal_stress = (s16)(tel->pain_level >> 24);  /* 32-bit >> 24 = 8-bit */
    if (thermal_stress < 1) thermal_stress = 1;

    s16 error_factor = (s16)(tel->error_count > 256 ? 256 : tel->error_count);
    if (error_factor < 1) error_factor = 1;

    /* P = (256/thermal) + (256/errors) - 256, scaled */
    p = pad_div(PAD_ONE, thermal_stress) + pad_div(PAD_ONE, error_factor);
    p = p - PAD_ONE;  /* Center around 0 */

    /* Arousal: CPU load + I/O utilization */
    /* Scale percentage to fixed-point */
    a = ((s32)tel->cpu_load_pct * PAD_SCALE) / 10000;  /* /100 for pct, /100 for precision */
    a += ((s32)tel->io_util_pct * PAD_SCALE) / 20000;  /* IO weighted less */

    /* Dominance: memory availability + battery */
    if (tel->mem_total_kb > 0) {
        d = ((s32)tel->mem_avail_kb * PAD_SCALE) / (s32)tel->mem_total_kb;
    } else {
        d = PAD_HALF;
    }
    d += ((s32)tel->battery_pct * PAD_SCALE) / 200;  /* Battery weighted less */
    d = d - PAD_HALF;  /* Center around 0 */

    /* Clamp to [-256, 255] (Q8.8 range) */
    if (p > 255) p = 255;
    if (p < -256) p = -256;
    if (a > 255) a = 255;
    if (a < -256) a = -256;
    if (d > 255) d = 255;
    if (d < -256) d = -256;

    pad->pleasure = (s16)p;
    pad->arousal = (s16)a;
    pad->dominance = (s16)d;
}

/*
 * Classify PAD state into behavioral quadrant
 */
static inline enum pad_quadrant classify_pad(const struct pad_state *pad)
{
    /* Emergency check first */
    if (pad->pleasure < PAD_FLOAT(-0.7))
        return PAD_EMERGENCY;

    /* Dominance check */
    if (pad->dominance > PAD_FLOAT(0.5))
        return PAD_DOMINANT;
    if (pad->dominance < PAD_FLOAT(-0.5))
        return PAD_SUBMISSIVE;

    /* Quadrant classification */
    if (pad->pleasure >= 0) {
        if (pad->arousal >= 0)
            return PAD_EXCITED;
        else
            return PAD_SERENE;
    } else {
        if (pad->arousal >= 0)
            return PAD_ANXIOUS;
        else
            return PAD_DEPRESSED;
    }
}

/*
 * Scheduler mode based on PAD quadrant
 */
enum sched_mode {
    SCHED_MODE_NORMAL = 0,      /* Standard time-slicing */
    SCHED_MODE_THROUGHPUT,      /* Maximize throughput (flow) */
    SCHED_MODE_INTERACTIVE,     /* Prioritize responsiveness */
    SCHED_MODE_POWERSAVE,       /* Minimize energy */
    SCHED_MODE_DEADLINE,        /* Strict deadlines (anxious) */
    SCHED_MODE_EMERGENCY,       /* Kill non-essential */
};

static inline enum sched_mode pad_to_sched_mode(enum pad_quadrant q)
{
    switch (q) {
    case PAD_SERENE:
        return SCHED_MODE_NORMAL;
    case PAD_EXCITED:
        return SCHED_MODE_THROUGHPUT;
    case PAD_ANXIOUS:
        return SCHED_MODE_DEADLINE;
    case PAD_DEPRESSED:
        return SCHED_MODE_POWERSAVE;
    case PAD_DOMINANT:
        return SCHED_MODE_THROUGHPUT;
    case PAD_SUBMISSIVE:
        return SCHED_MODE_POWERSAVE;
    case PAD_EMERGENCY:
        return SCHED_MODE_EMERGENCY;
    default:
        return SCHED_MODE_NORMAL;
    }
}

/*
 * Time slice adjustment based on PAD
 * Returns multiplier in Q8.8 format
 */
static inline s16 pad_timeslice_factor(const struct pad_state *pad)
{
    /* High arousal = shorter slices (more responsive) */
    /* Low pleasure = shorter slices (race to idle) */
    s16 factor = PAD_ONE;

    if (pad->arousal > PAD_FLOAT(0.5)) {
        factor = pad_mul(factor, PAD_FLOAT(0.75));
    } else if (pad->arousal < PAD_FLOAT(-0.3)) {
        factor = pad_mul(factor, PAD_FLOAT(1.25));
    }

    if (pad->pleasure < PAD_FLOAT(-0.3)) {
        factor = pad_mul(factor, PAD_FLOAT(0.5));
    }

    return factor;
}

/*
 * Process kill priority threshold based on PAD
 * Returns minimum priority that should survive (0-20 scale)
 */
static inline int pad_kill_threshold(const struct pad_state *pad)
{
    if (pad->pleasure < PAD_FLOAT(-0.8))
        return 15;  /* Only critical survives */
    if (pad->pleasure < PAD_FLOAT(-0.5))
        return 10;  /* Kill low priority */
    if (pad->pleasure < PAD_FLOAT(-0.2))
        return 5;   /* Kill very low priority */
    return 0;       /* No killing needed */
}

#endif /* __BANOS_PAD_STATE_H */
