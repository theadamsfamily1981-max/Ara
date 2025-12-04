/*
 * BANOS - Bio-Affective Neuromorphic Operating System
 * Affective Layer BPF - PAD Computation with Sigmoidal Stress Model
 *
 * This eBPF program implements the "synthetic neurochemistry" of BANOS:
 * - Reads telemetry from shared map (updated by kernel driver)
 * - Computes PAD using hybrid sigmoidal model
 * - Classifies mode (CALM/FLOW/ANXIOUS/CRITICAL)
 * - Exposes state to scheduler and Ara daemon
 *
 * PAD Formulation (Hybrid Sigmoidal with Derivative Damping):
 *
 * Pleasure (Health):
 *   P = 1 - tanh(α·ThermalStress + β·ErrorRate + γ·ImmuneEvents)
 *
 * Arousal (Activity):
 *   A = clamp(avg(CPU_load, GPU_load, IO_wait), 0, 1)
 *
 * Dominance (Control):
 *   D = (FreeRAM + PowerHeadroom)/2 × (1 - SwapPressure)
 *
 * Human Affect Coupling (Empathy):
 *   If user stressed (cold nose), boost system Dominance to compensate
 */

#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>

/* Include common definitions */
#include "../include/banos_common.h"

char LICENSE[] SEC("license") = "GPL";

/*
 * =============================================================================
 * BPF Maps
 * =============================================================================
 */

/* Telemetry input: updated by kernel driver or userspace daemon */
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct banos_kernel_telemetry);
} banos_telemetry_map SEC(".maps");

/* PAD output: consumed by scheduler and Ara */
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct banos_pad_state);
} banos_pad_map SEC(".maps");

/* History buffer for derivative computation (ring buffer of last 8 states) */
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 8);
    __type(key, __u32);
    __type(value, struct banos_pad_state);
} banos_pad_history SEC(".maps");

/* History index */
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, __u32);
} banos_history_idx SEC(".maps");

/* Configuration parameters (tunable from userspace) */
struct banos_affective_config {
    /* Sigmoid slope parameters */
    __u16 thermal_alpha;        /* Thermal stress weight (default: 400) */
    __u16 error_beta;           /* Error rate weight (default: 300) */
    __u16 immune_gamma;         /* Immune events weight (default: 300) */

    /* Derivative damping (λ for dP/dt term) */
    __u16 derivative_lambda;    /* Default: 200 */

    /* Empathy coupling strength */
    __u16 empathy_strength;     /* Default: 150 */

    /* Adaptation rate (for homeostatic normalization) */
    __u16 adaptation_rate;      /* Default: 10 (slow adaptation) */

    /* Reserved */
    __u16 reserved[4];
};

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct banos_affective_config);
} banos_config_map SEC(".maps");

/*
 * =============================================================================
 * Integer Math Helpers (No FP in BPF)
 * =============================================================================
 */

/*
 * Integer approximation of tanh(x) for x in [-8, 8] (scaled by 1000)
 * Uses piecewise linear approximation:
 *   tanh(x) ≈ x for |x| < 1
 *   tanh(x) ≈ sign(x) for |x| > 3
 *   Linear interpolation between
 */
static __always_inline __s32 int_tanh(__s32 x_scaled)
{
    /* x_scaled is x * 1000, output is tanh(x) * 1000 */
    __s32 abs_x = x_scaled < 0 ? -x_scaled : x_scaled;
    __s32 sign = x_scaled < 0 ? -1 : 1;

    if (abs_x < 500) {
        /* Linear region: tanh(x) ≈ x for small x */
        return x_scaled;
    } else if (abs_x < 1000) {
        /* Transition: 0.5 to 1.0 */
        return sign * (500 + (abs_x - 500) * 462 / 500);
    } else if (abs_x < 2000) {
        /* Approaching saturation: 1.0 to 2.0 */
        return sign * (962 + (abs_x - 1000) * 30 / 1000);
    } else {
        /* Saturated */
        return sign * 1000;
    }
}

/*
 * Safe division avoiding div-by-zero
 */
static __always_inline __s32 safe_div(__s32 num, __s32 denom)
{
    if (denom == 0) return num > 0 ? 1000 : -1000;
    return num / denom;
}

/*
 * Exponential moving average (for smoothing)
 * new_avg = (1 - alpha) * old_avg + alpha * new_val
 * alpha_permille is alpha * 1000
 */
static __always_inline __s32 ema(__s32 old_val, __s32 new_val, __u32 alpha_permille)
{
    return ((1000 - alpha_permille) * old_val + alpha_permille * new_val) / 1000;
}

/*
 * =============================================================================
 * PAD Computation Core
 * =============================================================================
 */

static __always_inline void compute_pleasure(
    const struct banos_kernel_telemetry *kt,
    const struct banos_affective_config *cfg,
    struct banos_pad_state *pad)
{
    /*
     * Pleasure = 1 - tanh(α·ThermalStress + β·ErrorRate + γ·ImmuneEvents)
     *
     * ThermalStress: Based on thermal headroom
     *   - 40°C headroom = 0 stress
     *   - 0°C headroom = 1000 stress
     */

    /* Thermal stress from headroom */
    __s32 thermal_stress;
    {
        __s32 headroom = kt->thermal_headroom_mC;
        if (headroom <= 0) {
            thermal_stress = 1000;
        } else if (headroom >= BANOS_THERMAL_HEADROOM_MAX_mC) {
            thermal_stress = 0;
        } else {
            thermal_stress = 1000 - (headroom * 1000 / BANOS_THERMAL_HEADROOM_MAX_mC);
        }
    }

    /* Get weights from config */
    __u16 alpha = cfg ? cfg->thermal_alpha : 400;
    __u16 beta = cfg ? cfg->error_beta : 300;
    __u16 gamma = cfg ? cfg->immune_gamma : 300;

    /* Weighted stress sum (each component 0-1000, weights sum to ~1000) */
    __s32 weighted_stress =
        (alpha * thermal_stress +
         beta * kt->error_rate_permille +
         gamma * kt->immune_events_permille) / 1000;

    /* Apply sigmoid: P = 1000 - tanh(stress) * 1000 */
    /* Scale stress to tanh input range */
    __s32 tanh_input = weighted_stress * 3 / 1000;  /* Map to ~[-3, 3] */
    __s32 tanh_out = int_tanh(tanh_input * 1000);

    /* P = 1 - tanh(stress) in scaled form */
    __s16 pleasure = banos_clamp_pad(1000 - tanh_out);

    /* Store diagnostics */
    pad->thermal_stress = banos_permille_to_pad(thermal_stress);
    pad->pleasure = pleasure;
}

static __always_inline void compute_arousal(
    const struct banos_kernel_telemetry *kt,
    struct banos_pad_state *pad)
{
    /*
     * Arousal = avg(CPU_load, GPU_load, IO_wait)
     * Simple average, clamped to [0, 1000]
     */

    __s32 cpu = kt->cpu_load_permille;
    __s32 gpu = kt->gpu_load_permille;
    __s32 io = kt->io_wait_permille;

    __s32 avg_load = (cpu + gpu + io) / 3;
    __u16 arousal_permille = banos_clamp_permille(avg_load);

    pad->arousal = banos_permille_to_pad(arousal_permille);
    pad->performance_drive = banos_permille_to_pad((cpu + gpu) / 2);
}

static __always_inline void compute_dominance(
    const struct banos_kernel_telemetry *kt,
    const struct banos_affective_config *cfg,
    struct banos_pad_state *pad)
{
    /*
     * Dominance = (FreeRAM + PowerHeadroom)/2 × (1 - SwapPressure)
     */

    __s32 mem_free = kt->mem_free_permille;
    __s32 power_h = kt->power_headroom_permille;
    __s32 swap_p = kt->swap_pressure_permille;

    /* Base dominance from resources */
    __s32 base_dom = (mem_free + power_h) / 2;

    /* Reduce by swap pressure */
    __s32 dom_permille = base_dom * (1000 - swap_p) / 1000;

    /* Empathy coupling: if user is stressed, boost system dominance */
    if (kt->face_deltaT_centiC < BANOS_FACE_STRESS_THRESHOLD_cC) {
        /* User stressed (cold nose) - system should be more stable */
        __u16 empathy = cfg ? cfg->empathy_strength : 150;
        dom_permille += empathy;
    }

    dom_permille = banos_clamp_permille(dom_permille);
    pad->dominance = banos_permille_to_pad(dom_permille);

    /* Track empathy effect */
    if (kt->face_deltaT_centiC < BANOS_FACE_STRESS_THRESHOLD_cC) {
        pad->empathy_boost = cfg ? cfg->empathy_strength : 150;
    } else {
        pad->empathy_boost = 0;
    }
}

static __always_inline void compute_derivatives(
    const struct banos_pad_state *prev,
    struct banos_pad_state *curr,
    __u64 dt_ns)
{
    /*
     * Compute rate of change for predictive dread
     * Rate is scaled: change per second * 1000
     */

    if (dt_ns == 0 || !prev) {
        curr->pleasure_rate = 0;
        curr->arousal_rate = 0;
        curr->dominance_rate = 0;
        return;
    }

    /* Convert dt to milliseconds for reasonable scaling */
    __u64 dt_ms = dt_ns / 1000000;
    if (dt_ms == 0) dt_ms = 1;

    /* dP/dt in units per second */
    __s32 dp = curr->pleasure - prev->pleasure;
    __s32 da = curr->arousal - prev->arousal;
    __s32 dd = curr->dominance - prev->dominance;

    curr->pleasure_rate = banos_clamp_pad((dp * 1000) / (__s32)dt_ms);
    curr->arousal_rate = banos_clamp_pad((da * 1000) / (__s32)dt_ms);
    curr->dominance_rate = banos_clamp_pad((dd * 1000) / (__s32)dt_ms);
}

static __always_inline void compute_scheduler_hints(
    struct banos_pad_state *pad)
{
    /*
     * Bat Algorithm parameters based on PAD
     *
     * High P (Pleasure) -> High Loudness (exploration, try new things)
     * Low P (Pain) -> High Pulse Rate (exploitation, safe defaults)
     */

    /* Loudness: inversely related to stress */
    /* P = 1000 (happy) -> loudness = 65535 (explore) */
    /* P = -1000 (pain) -> loudness = 0 (exploit) */
    __s32 p_normalized = (pad->pleasure + 1000) / 2;  /* 0..1000 */
    pad->bat_loudness = (__u16)(p_normalized * 65535 / 1000);

    /* Pulse rate: related to arousal */
    __s32 a_normalized = (pad->arousal + 1000) / 2;  /* 0..1000 */
    pad->bat_pulse_rate = (__u16)(a_normalized * 1000 / 1000);

    /* Kill threshold based on pain level */
    if (pad->pleasure < -800) {
        pad->kill_priority_threshold = 15;  /* Only critical survives */
    } else if (pad->pleasure < -500) {
        pad->kill_priority_threshold = 10;
    } else if (pad->pleasure < -200) {
        pad->kill_priority_threshold = 5;
    } else {
        pad->kill_priority_threshold = 0;   /* No killing */
    }
}

/*
 * =============================================================================
 * Main PAD Update Function
 * =============================================================================
 */

static __always_inline void banos_update_pad(void)
{
    __u32 key = 0;

    /* Get telemetry */
    struct banos_kernel_telemetry *kt;
    kt = bpf_map_lookup_elem(&banos_telemetry_map, &key);
    if (!kt)
        return;

    /* Get config */
    struct banos_affective_config *cfg;
    cfg = bpf_map_lookup_elem(&banos_config_map, &key);

    /* Get current PAD state */
    struct banos_pad_state *pad;
    pad = bpf_map_lookup_elem(&banos_pad_map, &key);
    if (!pad)
        return;

    /* Save previous state for derivatives */
    struct banos_pad_state prev = *pad;
    __u64 prev_time = pad->monotonic_time_ns;

    /* Compute PAD components */
    compute_pleasure(kt, cfg, pad);
    compute_arousal(kt, pad);
    compute_dominance(kt, cfg, pad);

    /* Update timestamp */
    pad->monotonic_time_ns = bpf_ktime_get_ns();

    /* Compute derivatives (for predictive dread) */
    __u64 dt = pad->monotonic_time_ns - prev_time;
    compute_derivatives(&prev, pad, dt);

    /* Classify mode */
    enum banos_mode new_mode = banos_classify_mode(
        pad->pleasure, pad->arousal, pad->dominance);

    /* Track mode changes */
    if (new_mode != pad->mode) {
        pad->mode_change_time_ns = pad->monotonic_time_ns;
        pad->mode_duration_ms = 0;
        pad->episode_id++;
    } else {
        pad->mode_duration_ms += dt / 1000000;
    }
    pad->mode = new_mode;

    /* Compute scheduler hints */
    compute_scheduler_hints(pad);

    /* Calculate perceived risk from immune events */
    pad->perceived_risk = banos_clamp_permille(
        kt->immune_events_permille + kt->error_rate_permille);
}

/*
 * =============================================================================
 * BPF Programs
 * =============================================================================
 */

/*
 * Periodic PAD update - attached to scheduler tick
 */
SEC("tracepoint/sched/sched_switch")
int banos_pad_tick(struct trace_event_raw_sched_switch *ctx)
{
    banos_update_pad();
    return 0;
}

/*
 * Alternative: Timer-based update for more control
 */
SEC("tp_btf/irq_handler_exit")
int banos_pad_irq_tick(void *ctx)
{
    /* Rate limit: only update every ~10ms */
    static __u64 last_update = 0;
    __u64 now = bpf_ktime_get_ns();

    if (now - last_update < 10000000)  /* 10ms */
        return 0;

    last_update = now;
    banos_update_pad();
    return 0;
}

/*
 * Thermal event trigger - immediate PAD update on thermal changes
 */
SEC("tracepoint/thermal/thermal_zone_trip")
int banos_thermal_event(void *ctx)
{
    banos_update_pad();
    return 0;
}

/*
 * OOM event trigger - immediate PAD update on memory pressure
 */
SEC("tracepoint/oom/oom_score_adj_update")
int banos_oom_event(void *ctx)
{
    banos_update_pad();
    return 0;
}
