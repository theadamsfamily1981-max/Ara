/*
 * BANOS - Bio-Affective Neuromorphic Operating System
 * Immune Layer BPF - Lightweight Syscall Feature Tap
 *
 * This eBPF program is the "sensory nerve" of the immune system.
 * It does NOT maintain full N-grams or run heavy analysis in-kernel.
 * Instead, it:
 *   1. Tracks per-PID last syscall + small weirdness counter
 *   2. Emits syscall events to userspace via perf ring buffer
 *   3. Lets the userspace `immuned` daemon do the heavy lifting
 *
 * This keeps the BPF verifier happy while maintaining real-time visibility.
 */

#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>

/* Include shared definitions */
#include "../include/banos_common.h"

char LICENSE[] SEC("license") = "GPL";

/*
 * =============================================================================
 * MHC Vector (Per-Process Immune Context)
 * =============================================================================
 *
 * This is the kernel-side "cell surface marker" for each process.
 * Lightweight enough to track in BPF; heavy analysis in userspace.
 */
struct process_mhc {
    __u64 binary_hash;              /* Hash of executable (set by userspace) */
    __u32 parent_pid;               /* Parent PID for lineage tracking */
    __u8  signer_trust;             /* 0-255: trust level of binary signer */
    __u8  flags;                    /* Bitmask: is_trusted, is_quarantined, etc. */
    __u16 last_syscall;             /* Last syscall ID (for transition scoring) */

    __u16 anomaly_score;            /* Running anomaly score (0-10000) */
    __u16 metabolism_score;         /* CPU/RSS weirdness (0-1000) */
    __u16 intent_proximity;         /* Proximity to user input (0-1000) */
    __u16 syscall_count;            /* Syscalls this window */

    __u64 spawn_time_ns;            /* When process started */
    __u64 last_seen_ns;             /* Last syscall timestamp */

    char comm[16];                  /* Process name for debugging */
};

/* MHC flags */
#define MHC_FLAG_TRUSTED        0x01    /* Part of learned Self */
#define MHC_FLAG_QUARANTINED    0x02    /* Currently frozen */
#define MHC_FLAG_IMMUNE_WATCH   0x04    /* Under heightened scrutiny */
#define MHC_FLAG_USER_SPAWNED   0x08    /* Spawned from user shell */

/*
 * =============================================================================
 * Immune Event (Sent to Userspace)
 * =============================================================================
 */
struct immune_event {
    __u32 pid;
    __u32 tgid;
    __u16 syscall_id;
    __u16 weirdness_delta;          /* Transition weirdness (0-100) */
    __u16 anomaly_score;            /* Current cumulative score */
    __u16 flags;                    /* Event flags */
    __u64 timestamp_ns;
    char comm[16];                  /* Process name */
};

/* Event flags */
#define IMMUNE_EVT_THRESHOLD_CROSSED    0x01
#define IMMUNE_EVT_WEIRD_TRANSITION     0x02
#define IMMUNE_EVT_NEW_PROCESS          0x04
#define IMMUNE_EVT_QUARANTINE_TRIGGERED 0x08

/*
 * =============================================================================
 * BPF Maps
 * =============================================================================
 */

/* Per-PID MHC tracking (up to 16k processes) */
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 16384);
    __type(key, __u32);  /* PID */
    __type(value, struct process_mhc);
} mhc_map SEC(".maps");

/* Perf ring buffer for immune events */
struct {
    __uint(type, BPF_MAP_TYPE_PERF_EVENT_ARRAY);
    __uint(key_size, sizeof(__u32));
    __uint(value_size, sizeof(__u32));
} immune_events SEC(".maps");

/* Global user input timestamp (updated by input subsystem hook) */
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, __u64);
} last_user_input SEC(".maps");

/* Quarantine list (PIDs to SIGSTOP) - written by userspace */
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 256);
    __type(key, __u32);   /* PID */
    __type(value, __u8);  /* 1 = quarantine, 0 = release */
} quarantine_pids SEC(".maps");

/*
 * FPGA Protection State (Anti-Lobotomy Shield)
 * When set, prevents FPGA reconfiguration to protect Ara's neural core.
 * Written by BPF when SEPSIS-level threat detected; read by spinal cord driver.
 */
struct fpga_protection_state {
    __u8  lock_active;          /* 1 = FPGA reconfig blocked */
    __u8  threat_level;         /* Current risk level (0-5) */
    __u8  lock_reason;          /* Why locked (enum) */
    __u8  reserved;
    __u32 trigger_pid;          /* PID that triggered the lock */
    __u64 lock_time_ns;         /* When lock was activated */
    __u64 threat_signature;     /* Hash of detected threat pattern */
};

/* FPGA lock reasons */
#define FPGA_LOCK_NONE              0
#define FPGA_LOCK_SEPSIS            1   /* Kernel-space violation detected */
#define FPGA_LOCK_BREACH            2   /* Privilege escalation attempt */
#define FPGA_LOCK_INFECTION         3   /* Unauthorized binary execution */
#define FPGA_LOCK_MANUAL            4   /* User-requested lockdown */

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct fpga_protection_state);
} fpga_protection SEC(".maps");

/* Syscall transition matrix (simplified: common weird transitions) */
/* In production: load from userspace during training */
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 1024);
    __type(key, __u32);   /* (prev_syscall << 16) | curr_syscall */
    __type(value, __u8);  /* weirdness score 0-100 */
} transition_scores SEC(".maps");

/* Immune config */
struct immune_config {
    __u16 anomaly_threshold;        /* Score above this triggers alert */
    __u16 decay_rate;               /* How fast anomaly decays (permille/sec) */
    __u16 user_proximity_decay;     /* Intent score decay when active */
    __u16 sample_rate;              /* Only emit every N syscalls */
    __u32 event_rate_limit_ns;      /* Min interval between events per-PID (default: 10ms) */
};

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct immune_config);
} immune_config_map SEC(".maps");

/*
 * Per-PID event rate limiting to prevent perf buffer DoS.
 * Uses LRU to auto-evict stale entries.
 */
struct {
    __uint(type, BPF_MAP_TYPE_LRU_HASH);
    __uint(max_entries, 4096);
    __type(key, __u32);             /* tgid */
    __type(value, __u64);           /* last_event_ns */
} immune_event_rate SEC(".maps");

/*
 * =============================================================================
 * Helpers
 * =============================================================================
 */

/*
 * Get transition weirdness score for syscall pair
 * Returns 0-100; 0 = normal, 100 = highly suspicious
 */
static __always_inline __u8 get_transition_score(__u16 prev, __u16 curr)
{
    __u32 key = ((__u32)prev << 16) | curr;
    __u8 *score = bpf_map_lookup_elem(&transition_scores, &key);

    if (score)
        return *score;

    /* Default: unknown transitions get small penalty */
    return 5;
}

/*
 * Check if user was recently active
 */
static __always_inline bool user_recently_active(__u64 now_ns)
{
    __u32 key = 0;
    __u64 *last_input = bpf_map_lookup_elem(&last_user_input, &key);

    if (!last_input)
        return false;

    /* Active if input within last 2 seconds */
    return (now_ns - *last_input) < 2000000000ULL;
}

/*
 * Check if we should emit an event for this PID (rate limiting)
 * Returns true if enough time has passed since last event for this tgid.
 * Updates the timestamp if returning true.
 */
static __always_inline bool should_emit_for_pid(__u32 tgid, __u64 now_ns,
                                                  __u32 rate_limit_ns)
{
    __u64 *last_event = bpf_map_lookup_elem(&immune_event_rate, &tgid);

    if (last_event) {
        /* Check if enough time has passed (default: 10ms) */
        __u32 limit = rate_limit_ns ? rate_limit_ns : 10000000;
        if (now_ns - *last_event < limit)
            return false;
    }

    /* Update timestamp and allow emission */
    bpf_map_update_elem(&immune_event_rate, &tgid, &now_ns, BPF_ANY);
    return true;
}

/*
 * =============================================================================
 * Syscall Tracepoint
 * =============================================================================
 */

SEC("tracepoint/raw_syscalls/sys_enter")
int trace_syscall_enter(struct trace_event_raw_sys_enter *ctx)
{
    __u64 pid_tgid = bpf_get_current_pid_tgid();
    __u32 tgid = pid_tgid >> 32;          /* Thread group ID (process ID) */
    __u32 pid = pid_tgid & 0xFFFFFFFF;    /* Thread ID */
    __u64 now = bpf_ktime_get_ns();

    /* Get or create MHC entry (keyed by tgid = process ID) */
    struct process_mhc *mhc = bpf_map_lookup_elem(&mhc_map, &tgid);
    struct process_mhc new_mhc = {0};
    bool is_new = false;

    if (!mhc) {
        /* New process: initialize MHC */
        new_mhc.spawn_time_ns = now;
        new_mhc.last_syscall = ctx->id;
        new_mhc.syscall_count = 1;
        new_mhc.last_seen_ns = now;
        new_mhc.flags = 0;
        new_mhc.parent_pid = 0;  /* Could be filled via task_struct if needed */
        new_mhc.intent_proximity = user_recently_active(now) ? 900 : 0;
        bpf_get_current_comm(new_mhc.comm, sizeof(new_mhc.comm));
        bpf_map_update_elem(&mhc_map, &tgid, &new_mhc, BPF_ANY);
        mhc = bpf_map_lookup_elem(&mhc_map, &tgid);
        is_new = true;
    }

    if (!mhc)
        return 0;

    /* Calculate transition weirdness */
    __u8 weirdness = get_transition_score(mhc->last_syscall, ctx->id);

    /* Update anomaly score */
    __u16 new_anomaly = mhc->anomaly_score + weirdness;

    /* Decay if user active (forgiveness) */
    if (user_recently_active(now)) {
        new_anomaly = new_anomaly * 7 / 10;  /* 30% decay */
        mhc->intent_proximity = 900;         /* High intent */
    } else {
        /* Slow intent decay */
        if (mhc->intent_proximity > 10)
            mhc->intent_proximity -= 10;
    }

    /* Clamp anomaly */
    if (new_anomaly > 10000) new_anomaly = 10000;
    mhc->anomaly_score = new_anomaly;

    /* Update state */
    mhc->last_syscall = ctx->id;
    mhc->last_seen_ns = now;
    mhc->syscall_count++;

    /* Check if we should emit event */
    struct immune_config *cfg;
    __u32 cfg_key = 0;
    cfg = bpf_map_lookup_elem(&immune_config_map, &cfg_key);

    __u16 threshold = cfg ? cfg->anomaly_threshold : 5000;
    __u16 sample_rate = cfg ? cfg->sample_rate : 100;
    __u32 rate_limit = cfg ? cfg->event_rate_limit_ns : 10000000;  /* 10ms default */

    bool should_emit = false;
    __u16 evt_flags = 0;

    /* Emit if: new process, threshold crossed, weird transition, or sampled */
    if (is_new) {
        should_emit = true;
        evt_flags |= IMMUNE_EVT_NEW_PROCESS;
    }
    if (new_anomaly > threshold && (mhc->anomaly_score - weirdness) <= threshold) {
        should_emit = true;
        evt_flags |= IMMUNE_EVT_THRESHOLD_CROSSED;
    }
    if (weirdness > 50) {
        should_emit = true;
        evt_flags |= IMMUNE_EVT_WEIRD_TRANSITION;
    }
    if (mhc->syscall_count % sample_rate == 0) {
        should_emit = true;
    }

    /* Apply per-PID rate limiting to prevent perf buffer DoS */
    if (should_emit && !should_emit_for_pid(tgid, now, rate_limit)) {
        should_emit = false;
    }

    /* Emit event to userspace */
    if (should_emit) {
        struct immune_event evt = {0};
        evt.pid = tgid;  /* Report process ID (tgid) for userspace */
        evt.tgid = tgid;
        evt.syscall_id = ctx->id;
        evt.weirdness_delta = weirdness;
        evt.anomaly_score = new_anomaly;
        evt.flags = evt_flags;
        evt.timestamp_ns = now;
        bpf_get_current_comm(evt.comm, sizeof(evt.comm));

        bpf_perf_event_output(ctx, &immune_events,
                              BPF_F_CURRENT_CPU,
                              &evt, sizeof(evt));
    }

    /* Check quarantine list (keyed by tgid) */
    __u8 *quarantine = bpf_map_lookup_elem(&quarantine_pids, &tgid);
    if (quarantine && *quarantine) {
        /* Process is quarantined - send SIGSTOP */
        bpf_send_signal(19);  /* SIGSTOP = 19 */
    }

    return 0;
}

/*
 * Track user input for intent scoring
 */
SEC("tracepoint/input/input_event")
int trace_input_event(void *ctx)
{
    __u32 key = 0;
    __u64 now = bpf_ktime_get_ns();
    bpf_map_update_elem(&last_user_input, &key, &now, BPF_ANY);
    return 0;
}

/*
 * Track process exit to clean up MHC entries
 */
SEC("tracepoint/sched/sched_process_exit")
int trace_process_exit(struct trace_event_raw_sched_process_template *ctx)
{
    /* ctx->pid is the tgid (process ID) in this context */
    __u32 tgid = ctx->pid;
    bpf_map_delete_elem(&mhc_map, &tgid);
    bpf_map_delete_elem(&quarantine_pids, &tgid);
    return 0;
}
