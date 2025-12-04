/*
 * BANOS - Bio-Affective Neuromorphic Operating System
 * Bat Algorithm Scheduler (sched_ext BPF)
 *
 * This scheduler replaces traditional round-robin with the Bat Algorithm:
 * - Processes are "prey"
 * - CPU cores are "bats" using echolocation
 * - Bat "loudness" increases with system arousal
 * - Under stress, bats become aggressive: consolidate to race-to-sleep
 *
 * The scheduler reads PAD state from shared memory (updated by kernel module)
 * and adjusts behavior accordingly:
 * - SERENE:  Normal time-slicing
 * - EXCITED: Throughput mode, favor long-running tasks
 * - ANXIOUS: Deadline mode, short slices, kill low priority
 * - DEPRESSED: Power-save, consolidate to few cores
 */

#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>

/* Include our PAD definitions (simplified for BPF) */
#define PAD_FRAC_BITS   8
#define PAD_SCALE       (1 << PAD_FRAC_BITS)
#define PAD_ONE         PAD_SCALE

char _license[] SEC("license") = "GPL";

/*
 * Shared state with user-space daemon
 */
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, u32);
    __type(value, struct banos_shared_state);
} banos_state SEC(".maps");

struct banos_shared_state {
    /* PAD state (updated by kernel module or user daemon) */
    s16 pleasure;
    s16 arousal;
    s16 dominance;
    u8  quadrant;      /* enum pad_quadrant */
    u8  sched_mode;    /* enum sched_mode */

    /* Bat algorithm parameters */
    u16 loudness;      /* Bat loudness (0-65535) */
    u16 pulse_rate;    /* Bat pulse rate */
    u32 frequency;     /* Echolocation frequency */

    /* Kill threshold */
    u8  kill_priority_threshold;
    u8  reserved[3];

    /* Statistics */
    u64 tasks_scheduled;
    u64 tasks_killed;
    u64 mode_switches;
};

/*
 * Per-task scheduling data
 */
struct task_ctx {
    u64 last_run_ns;        /* Last time task ran */
    u32 slice_ns;           /* Current time slice */
    u32 metabolic_cost;     /* Energy cost of running this task */
    u16 priority;           /* Effective priority (0-20) */
    u8  is_interactive;     /* Task yields frequently */
    u8  should_kill;        /* Marked for termination */
};

struct {
    __uint(type, BPF_MAP_TYPE_TASK_STORAGE);
    __uint(map_flags, BPF_F_NO_PREALLOC);
    __type(key, int);
    __type(value, struct task_ctx);
} task_ctx_stor SEC(".maps");

/*
 * Dispatch queues
 * DSQ 0: Global queue
 * DSQ 1-N: Per-CPU local queues
 */
#define SHARED_DSQ      0
#define LOCAL_DSQ_BASE  1

/*
 * Scheduling parameters
 */
const volatile u32 default_slice_ns = 5000000;   /* 5ms default */
const volatile u32 min_slice_ns = 1000000;       /* 1ms minimum */
const volatile u32 max_slice_ns = 20000000;      /* 20ms maximum */

/*
 * sched_ext operations structure
 */
UEI_DEFINE(uei);

/*
 * Get or create task context
 */
static struct task_ctx *get_task_ctx(struct task_struct *p)
{
    struct task_ctx *ctx;

    ctx = bpf_task_storage_get(&task_ctx_stor, p, NULL, 0);
    if (!ctx) {
        struct task_ctx new_ctx = {
            .last_run_ns = 0,
            .slice_ns = default_slice_ns,
            .metabolic_cost = 100,
            .priority = 10,
            .is_interactive = 0,
            .should_kill = 0,
        };
        ctx = bpf_task_storage_get(&task_ctx_stor, p, &new_ctx,
                                   BPF_LOCAL_STORAGE_GET_F_CREATE);
    }
    return ctx;
}

/*
 * Get current shared state
 */
static struct banos_shared_state *get_shared_state(void)
{
    u32 key = 0;
    return bpf_map_lookup_elem(&banos_state, &key);
}

/*
 * Compute time slice based on PAD state and task properties
 * Implements the "echolocation" part of Bat Algorithm
 */
static u32 compute_slice(struct task_ctx *ctx, struct banos_shared_state *state)
{
    u32 slice = default_slice_ns;

    if (!state)
        return slice;

    /* Bat loudness affects slice duration */
    /* Higher loudness (stressed) = shorter slices */
    if (state->loudness > 32768) {
        slice = slice >> 1;  /* Halve slice when loud */
    } else if (state->loudness < 8192) {
        slice = slice + (slice >> 1);  /* 1.5x when quiet */
    }

    /* Mode-specific adjustments */
    switch (state->sched_mode) {
    case 0:  /* NORMAL */
        break;
    case 1:  /* THROUGHPUT */
        slice = slice << 1;  /* Double slice for throughput */
        break;
    case 2:  /* INTERACTIVE */
        slice = slice >> 1;  /* Halve for responsiveness */
        break;
    case 3:  /* POWERSAVE */
        slice = slice << 2;  /* 4x slice, let tasks complete */
        break;
    case 4:  /* DEADLINE */
        slice = min_slice_ns;  /* Minimum slice */
        break;
    case 5:  /* EMERGENCY */
        slice = min_slice_ns;
        break;
    }

    /* Interactive tasks get shorter slices always */
    if (ctx->is_interactive)
        slice = slice >> 1;

    /* Clamp to valid range */
    if (slice < min_slice_ns)
        slice = min_slice_ns;
    if (slice > max_slice_ns)
        slice = max_slice_ns;

    return slice;
}

/*
 * select_cpu: Choose which CPU should run a waking task
 * Implements spatial aspect of Bat Algorithm
 */
s32 BPF_STRUCT_OPS(bat_select_cpu, struct task_struct *p, s32 prev_cpu, u64 wake_flags)
{
    struct banos_shared_state *state = get_shared_state();
    s32 cpu;

    /* In power-save mode, try to consolidate to fewer CPUs */
    if (state && state->sched_mode == 3) {  /* POWERSAVE */
        /* Prefer lower-numbered CPUs to let others idle */
        cpu = bpf_get_smp_processor_id();
        if (cpu > 0)
            cpu = 0;  /* Force to CPU 0 if possible */
        return cpu;
    }

    /* In emergency, also consolidate */
    if (state && state->sched_mode == 5) {  /* EMERGENCY */
        return 0;  /* Everything on CPU 0 */
    }

    /* Default: let task run where it was */
    return prev_cpu;
}

/*
 * enqueue: Place a runnable task in a dispatch queue
 */
void BPF_STRUCT_OPS(bat_enqueue, struct task_struct *p, u64 enq_flags)
{
    struct task_ctx *ctx = get_task_ctx(p);
    struct banos_shared_state *state = get_shared_state();
    u32 slice;
    u64 vtime;

    if (!ctx)
        return;

    /* Check if task should be killed */
    if (state && ctx->priority < state->kill_priority_threshold) {
        /* Mark for SIGKILL - will be handled by user-space daemon */
        ctx->should_kill = 1;
        /* Still enqueue to let it receive the signal */
    }

    /* Compute time slice */
    slice = compute_slice(ctx, state);
    ctx->slice_ns = slice;

    /* Calculate virtual time for fair scheduling */
    vtime = p->scx.dsq_vtime;

    /* Higher priority = lower vtime = scheduled sooner */
    /* Adjust vtime based on metabolic cost */
    if (ctx->metabolic_cost > 100) {
        vtime += (ctx->metabolic_cost - 100) * 1000;  /* Penalize heavy tasks */
    }

    /* In anxious mode, penalize non-interactive tasks heavily */
    if (state && state->sched_mode == 4 && !ctx->is_interactive) {
        vtime += 10000000;  /* +10ms penalty */
    }

    /* Dispatch to shared queue with computed vtime */
    scx_bpf_dispatch_vtime(p, SHARED_DSQ, slice, vtime, enq_flags);
}

/*
 * dispatch: Pull tasks from DSQs to run
 */
void BPF_STRUCT_OPS(bat_dispatch, s32 cpu, struct task_struct *prev)
{
    /* Consume from shared DSQ */
    scx_bpf_consume(SHARED_DSQ);
}

/*
 * running: Called when a task starts running
 */
void BPF_STRUCT_OPS(bat_running, struct task_struct *p)
{
    struct task_ctx *ctx = get_task_ctx(p);
    struct banos_shared_state *state = get_shared_state();

    if (ctx) {
        ctx->last_run_ns = bpf_ktime_get_ns();
    }

    if (state) {
        state->tasks_scheduled++;
    }
}

/*
 * stopping: Called when a task stops running
 */
void BPF_STRUCT_OPS(bat_stopping, struct task_struct *p, bool runnable)
{
    struct task_ctx *ctx = get_task_ctx(p);
    u64 now, delta;

    if (!ctx)
        return;

    now = bpf_ktime_get_ns();
    delta = now - ctx->last_run_ns;

    /* Detect interactive tasks: they yield before slice expires */
    if (delta < (ctx->slice_ns >> 1)) {
        ctx->is_interactive = 1;
    } else if (delta > ctx->slice_ns) {
        ctx->is_interactive = 0;
    }

    /* Update metabolic cost estimate */
    /* CPU-bound tasks have higher cost */
    if (!runnable) {
        /* Task blocked (I/O) - reduce cost */
        if (ctx->metabolic_cost > 50)
            ctx->metabolic_cost -= 10;
    } else {
        /* Task was preempted (CPU-bound) - increase cost */
        if (ctx->metabolic_cost < 200)
            ctx->metabolic_cost += 5;
    }

    /* Update vtime for fairness */
    p->scx.dsq_vtime += delta;
}

/*
 * quiescent: Called when CPU goes idle
 * Good time for power management in POWERSAVE mode
 */
void BPF_STRUCT_OPS(bat_quiescent, struct task_struct *p, u64 deq_flags)
{
    /* Could trigger CPU frequency scaling here */
}

/*
 * enable: Called when task becomes schedulable by this scheduler
 */
void BPF_STRUCT_OPS(bat_enable, struct task_struct *p)
{
    /* Initialize task context */
    get_task_ctx(p);
}

/*
 * init_task: Initialize scheduling data for a new task
 */
s32 BPF_STRUCT_OPS(bat_init_task, struct task_struct *p,
                   struct scx_init_task_args *args)
{
    struct task_ctx *ctx;

    ctx = bpf_task_storage_get(&task_ctx_stor, p, NULL,
                               BPF_LOCAL_STORAGE_GET_F_CREATE);
    if (!ctx)
        return -ENOMEM;

    /* Set initial priority based on nice value */
    /* nice -20 -> priority 20, nice +19 -> priority 1 */
    s32 nice = TASK_NICE(p);
    ctx->priority = 20 - (nice + 20) / 2;

    /* Set initial metabolic cost based on scheduling policy */
    if (p->policy == SCHED_IDLE) {
        ctx->metabolic_cost = 10;
        ctx->priority = 1;
    } else if (p->policy == SCHED_BATCH) {
        ctx->metabolic_cost = 150;
    } else {
        ctx->metabolic_cost = 100;
    }

    return 0;
}

/*
 * init: Initialize the scheduler
 */
s32 BPF_STRUCT_OPS_SLEEPABLE(bat_init)
{
    /* Create shared DSQ */
    return scx_bpf_create_dsq(SHARED_DSQ, -1);
}

/*
 * exit: Cleanup when scheduler is unloaded
 */
void BPF_STRUCT_OPS(bat_exit, struct scx_exit_info *ei)
{
    UEI_RECORD(uei, ei);
}

/*
 * Register sched_ext operations
 */
SCX_OPS_DEFINE(bat_ops,
    .select_cpu     = (void *)bat_select_cpu,
    .enqueue        = (void *)bat_enqueue,
    .dispatch       = (void *)bat_dispatch,
    .running        = (void *)bat_running,
    .stopping       = (void *)bat_stopping,
    .quiescent      = (void *)bat_quiescent,
    .enable         = (void *)bat_enable,
    .init_task      = (void *)bat_init_task,
    .init           = (void *)bat_init,
    .exit           = (void *)bat_exit,
    .name           = "bat_scheduler",
);
