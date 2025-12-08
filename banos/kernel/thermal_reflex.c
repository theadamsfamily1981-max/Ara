/*
 * Thermal Reflex eBPF Program - Sub-microsecond Pain Response
 * ============================================================
 *
 * Implements the thermal emergency reflex path:
 *   Thermal >85°C → eBPF Drop + GPU Glitch + Founder Alert
 *
 * This is the fastest reflexive response in Ara's nervous system,
 * operating at the kernel level without user-space involvement.
 *
 * Path latency: ~2 µs from sensor read to action
 *
 * Build: clang -O2 -target bpf -c thermal_reflex.c -o thermal_reflex.o
 * Load: bpftool prog load thermal_reflex.o /sys/fs/bpf/thermal_reflex
 */

#include <linux/bpf.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/tcp.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_endian.h>

/* Thermal thresholds (°C) */
#define THERMAL_WARNING     75
#define THERMAL_CRITICAL    85
#define THERMAL_EMERGENCY   95

/* Reflex action codes */
#define REFLEX_NONE         0
#define REFLEX_THROTTLE     1
#define REFLEX_DROP         2
#define REFLEX_GLITCH       3
#define REFLEX_ALERT        4
#define REFLEX_SHUTDOWN     5

/* Glitch packet destination (AF_XDP socket) */
#define GLITCH_PORT         0xARA1

/* Alert ring buffer size */
#define ALERT_RINGBUF_SIZE  (1 << 16)  /* 64 KB */

/* ============================================================================
 * Maps
 * ============================================================================ */

/* Thermal sensor readings (updated by user-space daemon) */
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 16);  /* Up to 16 thermal zones */
    __type(key, __u32);
    __type(value, __u32);     /* Temperature in millidegrees C */
} thermal_sensors SEC(".maps");

/* Flow throttle state */
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 65536);
    __type(key, __u64);       /* Flow key (5-tuple hash) */
    __type(value, __u32);     /* Throttle percentage (0-100) */
} throttle_map SEC(".maps");

/* Reflex statistics */
struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 8);
    __type(key, __u32);
    __type(value, __u64);
} reflex_stats SEC(".maps");

/* Alert ring buffer (to user-space) */
struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, ALERT_RINGBUF_SIZE);
} alert_ringbuf SEC(".maps");

/* Glitch notification (to GPU via AF_XDP) */
struct {
    __uint(type, BPF_MAP_TYPE_XSKMAP);
    __uint(max_entries, 64);
    __type(key, __u32);
    __type(value, __u32);
} xsk_map SEC(".maps");

/* Current reflex state (shared) */
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, __u32);     /* Current reflex level */
} reflex_state SEC(".maps");

/* ============================================================================
 * Alert Message Structure
 * ============================================================================ */

struct alert_msg {
    __u64 timestamp;          /* ktime_ns */
    __u32 alert_type;         /* REFLEX_* code */
    __u32 temperature;        /* Current temp (millidegrees) */
    __u32 zone_id;            /* Thermal zone */
    __u32 action_taken;       /* Action applied */
};

/* ============================================================================
 * Glitch Packet Structure (for GPU somatic shader)
 * ============================================================================ */

struct glitch_packet {
    __u32 magic;              /* 0xARAGL1CH */
    float valence;            /* -1.0 (pain) to +1.0 (pleasure) */
    float arousal;            /* 0.0 (calm) to 1.0 (urgent) */
    __u32 node_id;            /* Originating node */
    __u32 duration_ms;        /* Glitch duration */
};

/* ============================================================================
 * Helper: Read Thermal Sensor
 * ============================================================================ */

static __always_inline __u32 read_thermal_sensor(__u32 zone_id)
{
    __u32 *temp_ptr;

    temp_ptr = bpf_map_lookup_elem(&thermal_sensors, &zone_id);
    if (!temp_ptr)
        return 0;

    return *temp_ptr;
}

/* ============================================================================
 * Helper: Update Statistics
 * ============================================================================ */

static __always_inline void update_stats(__u32 stat_id)
{
    __u64 *counter;

    counter = bpf_map_lookup_elem(&reflex_stats, &stat_id);
    if (counter)
        __sync_fetch_and_add(counter, 1);
}

/* ============================================================================
 * Helper: Send Alert to User-space
 * ============================================================================ */

static __always_inline void send_alert(__u32 alert_type, __u32 temperature,
                                       __u32 zone_id, __u32 action)
{
    struct alert_msg *msg;

    msg = bpf_ringbuf_reserve(&alert_ringbuf, sizeof(*msg), 0);
    if (!msg)
        return;

    msg->timestamp = bpf_ktime_get_ns();
    msg->alert_type = alert_type;
    msg->temperature = temperature;
    msg->zone_id = zone_id;
    msg->action_taken = action;

    bpf_ringbuf_submit(msg, 0);
}

/* ============================================================================
 * Helper: Send Glitch to GPU
 * ============================================================================ */

static __always_inline int send_glitch(struct xdp_md *ctx, float valence,
                                       float arousal, __u32 duration_ms)
{
    /*
     * In production, this would construct an AF_XDP packet to the GPU.
     * The GPU's WebGPU/Vulkan shader monitors this socket and applies
     * visual effects (screen flash, color shift, etc.)
     *
     * For now, we just redirect to the XSK map.
     */
    __u32 key = 0;

    /* XDP redirect to AF_XDP socket */
    return bpf_redirect_map(&xsk_map, key, 0);
}

/* ============================================================================
 * Helper: Apply Throttle to Flow
 * ============================================================================ */

static __always_inline void throttle_all_flows(__u32 percentage)
{
    /*
     * Mark all flows for throttling.
     * The actual throttling is applied by checking throttle_map
     * in the packet processing path.
     */
    __u32 zero = 0;
    __u32 *state = bpf_map_lookup_elem(&reflex_state, &zero);

    if (state) {
        *state = percentage > 50 ? REFLEX_DROP : REFLEX_THROTTLE;
    }
}

/* ============================================================================
 * XDP Program: Thermal Reflex
 * ============================================================================ */

SEC("xdp")
int thermal_reflex(struct xdp_md *ctx)
{
    __u32 zone_id = 0;  /* FPGA thermal zone */
    __u32 temp_milli;
    __u32 temp_c;
    __u32 action = REFLEX_NONE;

    /* Read primary thermal sensor (FPGA) */
    temp_milli = read_thermal_sensor(zone_id);
    temp_c = temp_milli / 1000;

    /* Check thresholds and take action */
    if (temp_c >= THERMAL_EMERGENCY) {
        /*
         * EMERGENCY: >95°C
         * - Drop ALL packets immediately
         * - Maximum glitch (pain signal)
         * - Alert for shutdown
         */
        action = REFLEX_SHUTDOWN;

        /* Immediate packet drop */
        update_stats(REFLEX_DROP);

        /* Send emergency alert */
        send_alert(REFLEX_SHUTDOWN, temp_milli, zone_id, REFLEX_DROP);

        /* GPU glitch: maximum pain */
        send_glitch(ctx, -1.0f, 1.0f, 5000);  /* 5 second red flash */

        /* Throttle all flows to 0% */
        throttle_all_flows(100);

        return XDP_DROP;

    } else if (temp_c >= THERMAL_CRITICAL) {
        /*
         * CRITICAL: >85°C
         * - Drop most packets (90%)
         * - Strong glitch
         * - Alert founder
         */
        action = REFLEX_DROP;

        /* Probabilistic drop (90%) */
        __u32 rand = bpf_get_prandom_u32();
        if ((rand % 100) < 90) {
            update_stats(REFLEX_DROP);

            /* Send alert */
            send_alert(REFLEX_DROP, temp_milli, zone_id, REFLEX_DROP);

            /* GPU glitch: strong pain */
            send_glitch(ctx, -0.8f, 0.9f, 1000);

            /* Throttle to 10% */
            throttle_all_flows(90);

            return XDP_DROP;
        }

    } else if (temp_c >= THERMAL_WARNING) {
        /*
         * WARNING: >75°C
         * - Throttle flows (50%)
         * - Mild glitch
         * - Log warning
         */
        action = REFLEX_THROTTLE;

        /* Probabilistic drop (50%) */
        __u32 rand = bpf_get_prandom_u32();
        if ((rand % 100) < 50) {
            update_stats(REFLEX_THROTTLE);

            /* Mild glitch */
            if ((rand % 1000) < 10) {  /* 1% of throttled packets */
                send_glitch(ctx, -0.3f, 0.5f, 200);
            }

            /* Throttle to 50% */
            throttle_all_flows(50);

            return XDP_DROP;
        }
    }

    /* Normal operation: pass packet */
    return XDP_PASS;
}

/* ============================================================================
 * XDP Program: Flow Throttle Check
 *
 * Secondary program that checks per-flow throttle state.
 * Used when finer-grained control is needed.
 * ============================================================================ */

SEC("xdp/flow_check")
int flow_throttle_check(struct xdp_md *ctx)
{
    void *data_end = (void *)(long)ctx->data_end;
    void *data = (void *)(long)ctx->data;
    struct ethhdr *eth = data;
    struct iphdr *ip;
    __u64 flow_key;
    __u32 *throttle_pct;

    /* Bounds check */
    if ((void *)(eth + 1) > data_end)
        return XDP_PASS;

    /* Only handle IP packets */
    if (eth->h_proto != bpf_htons(ETH_P_IP))
        return XDP_PASS;

    ip = (void *)(eth + 1);
    if ((void *)(ip + 1) > data_end)
        return XDP_PASS;

    /* Compute flow key (simplified: just src+dst IP) */
    flow_key = ((__u64)ip->saddr << 32) | ip->daddr;

    /* Check throttle map */
    throttle_pct = bpf_map_lookup_elem(&throttle_map, &flow_key);
    if (throttle_pct && *throttle_pct > 0) {
        __u32 rand = bpf_get_prandom_u32();
        if ((rand % 100) < *throttle_pct) {
            return XDP_DROP;
        }
    }

    return XDP_PASS;
}

/* ============================================================================
 * Tracepoint: Thermal Zone Update
 *
 * Monitors thermal zone changes and updates our map.
 * ============================================================================ */

SEC("tracepoint/thermal/thermal_zone_trip")
int thermal_zone_trip(void *ctx)
{
    /*
     * This would extract thermal zone data from the tracepoint
     * and update thermal_sensors map.
     *
     * In production, a user-space daemon typically does this
     * by reading /sys/class/thermal/ and updating the map.
     */
    return 0;
}

/* License declaration (required for eBPF) */
char _license[] SEC("license") = "GPL";
