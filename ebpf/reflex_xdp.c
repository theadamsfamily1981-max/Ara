/**
 * Ara Reflex XDP Program - Spinal Reflex Arc
 * ==========================================
 *
 * Fast packet classification and reflexive actions at the NIC level.
 *
 * This eBPF/XDP program implements Ara's "spinal reflexes":
 * - Parse teleological headers on incoming packets
 * - Apply reflexive actions (drop, rate-limit, mark)
 * - Emit events to userspace for learning
 *
 * Timing: Decisions in ~10μs, before kernel networking stack
 *
 * Teleological Header Format (after UDP payload start):
 *   4 bytes: magic ("ALWY")
 *   1 byte:  version
 *   1 byte:  priority (0-255, teleology-aligned)
 *   1 byte:  affect_valence (quantized -1..1 to 0..255)
 *   1 byte:  affect_arousal (quantized 0..1 to 0..255)
 *   2 bytes: reserved
 *   16 bytes: hv_hash (truncated hash of H_flow or H_moment)
 *
 * Build:
 *   clang -O2 -target bpf -c reflex_xdp.c -o reflex_xdp.o
 *
 * Load:
 *   ip link set dev eth0 xdp obj reflex_xdp.o sec xdp
 */

#include <linux/bpf.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/udp.h>
#include <linux/tcp.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_endian.h>

/* Teleological header magic bytes */
#define SOULMESH_MAGIC_0 'A'
#define SOULMESH_MAGIC_1 'L'
#define SOULMESH_MAGIC_2 'W'
#define SOULMESH_MAGIC_3 'Y'
#define SOULMESH_VERSION 1

/* Priority thresholds */
#define PRIORITY_PAIN_THRESHOLD 200    /* Pain packet threshold */
#define PRIORITY_SUSPICIOUS_THRESHOLD 150

/* Rate limiting parameters */
#define RATE_LIMIT_WINDOW_NS 1000000000ULL  /* 1 second */
#define RATE_LIMIT_MAX_PACKETS 1000

/* ========================================================================= */
/* Data Structures                                                           */
/* ========================================================================= */

/**
 * Teleological header structure
 */
struct teleology_hdr {
    __u8  magic[4];      /* "ALWY" */
    __u8  version;
    __u8  priority;      /* 0-255 (teleology-aligned) */
    __u8  affect_val;    /* quantized valence */
    __u8  affect_ar;     /* quantized arousal */
    __u16 reserved;
    __u8  hv_hash[16];   /* truncated hash of H_flow or H_moment */
} __attribute__((packed));

/**
 * Reflex event for perf buffer
 */
struct reflex_event {
    __u64 timestamp;
    __u32 src_ip;
    __u32 dst_ip;
    __u16 src_port;
    __u16 dst_port;
    __u8  protocol;
    __u8  priority;
    __u8  affect_val;
    __u8  affect_ar;
    __u8  action;        /* 0=pass, 1=drop, 2=mark */
    __u8  reason;        /* why this action was taken */
    __u8  hv_hash[16];
} __attribute__((packed));

/**
 * Rate limit state per source IP
 */
struct rate_limit_state {
    __u64 window_start;
    __u32 packet_count;
};

/* ========================================================================= */
/* BPF Maps                                                                  */
/* ========================================================================= */

/* Perf event buffer for reflex events */
struct {
    __uint(type, BPF_MAP_TYPE_PERF_EVENT_ARRAY);
    __uint(key_size, sizeof(__u32));
    __uint(value_size, sizeof(__u32));
} reflex_events SEC(".maps");

/* Rate limiting state per source IP */
struct {
    __uint(type, BPF_MAP_TYPE_LRU_HASH);
    __uint(max_entries, 65536);
    __type(key, __u32);             /* source IP */
    __type(value, struct rate_limit_state);
} rate_limit_map SEC(".maps");

/* Configuration map */
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, __u32);           /* flags: bit 0 = enabled, bit 1 = learning mode */
} config_map SEC(".maps");

/* Statistics counters */
struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 8);         /* 0=total, 1=passed, 2=dropped, 3=marked, 4=pain, 5=suspicious, 6=teleology, 7=rate_limited */
    __type(key, __u32);
    __type(value, __u64);
} stats_map SEC(".maps");

/* ========================================================================= */
/* Helper Functions                                                          */
/* ========================================================================= */

static __always_inline void increment_stat(__u32 stat_id)
{
    __u64 *counter = bpf_map_lookup_elem(&stats_map, &stat_id);
    if (counter)
        __sync_fetch_and_add(counter, 1);
}

static __always_inline int is_soulmesh_packet(struct teleology_hdr *th)
{
    return (th->magic[0] == SOULMESH_MAGIC_0 &&
            th->magic[1] == SOULMESH_MAGIC_1 &&
            th->magic[2] == SOULMESH_MAGIC_2 &&
            th->magic[3] == SOULMESH_MAGIC_3 &&
            th->version == SOULMESH_VERSION);
}

static __always_inline int check_rate_limit(__u32 src_ip)
{
    struct rate_limit_state *state;
    struct rate_limit_state new_state = {0};
    __u64 now = bpf_ktime_get_ns();

    state = bpf_map_lookup_elem(&rate_limit_map, &src_ip);
    if (!state) {
        /* First packet from this IP */
        new_state.window_start = now;
        new_state.packet_count = 1;
        bpf_map_update_elem(&rate_limit_map, &src_ip, &new_state, BPF_ANY);
        return 0; /* Allow */
    }

    /* Check if we're in a new window */
    if (now - state->window_start > RATE_LIMIT_WINDOW_NS) {
        state->window_start = now;
        state->packet_count = 1;
        return 0; /* Allow */
    }

    /* Increment counter */
    state->packet_count++;

    /* Check if over limit */
    if (state->packet_count > RATE_LIMIT_MAX_PACKETS) {
        return 1; /* Rate limited */
    }

    return 0; /* Allow */
}

static __always_inline void emit_reflex_event(
    struct xdp_md *ctx,
    __u32 src_ip, __u32 dst_ip,
    __u16 src_port, __u16 dst_port,
    __u8 protocol,
    struct teleology_hdr *th,
    __u8 action, __u8 reason)
{
    struct reflex_event event = {0};

    event.timestamp = bpf_ktime_get_ns();
    event.src_ip = src_ip;
    event.dst_ip = dst_ip;
    event.src_port = src_port;
    event.dst_port = dst_port;
    event.protocol = protocol;
    event.action = action;
    event.reason = reason;

    if (th) {
        event.priority = th->priority;
        event.affect_val = th->affect_val;
        event.affect_ar = th->affect_ar;
        __builtin_memcpy(event.hv_hash, th->hv_hash, 16);
    }

    bpf_perf_event_output(ctx, &reflex_events, BPF_F_CURRENT_CPU,
                          &event, sizeof(event));
}

/* ========================================================================= */
/* Main XDP Program                                                          */
/* ========================================================================= */

SEC("xdp")
int xdp_reflex(struct xdp_md *ctx)
{
    void *data_end = (void *)(long)ctx->data_end;
    void *data = (void *)(long)ctx->data;

    /* Statistics */
    increment_stat(0); /* total packets */

    /* Parse Ethernet header */
    struct ethhdr *eth = data;
    if ((void *)(eth + 1) > data_end)
        return XDP_PASS;

    /* Only handle IPv4 */
    if (eth->h_proto != bpf_htons(ETH_P_IP))
        return XDP_PASS;

    /* Parse IP header */
    struct iphdr *ip = (void *)(eth + 1);
    if ((void *)(ip + 1) > data_end)
        return XDP_PASS;

    __u32 src_ip = ip->saddr;
    __u32 dst_ip = ip->daddr;
    __u8 protocol = ip->protocol;
    __u16 src_port = 0;
    __u16 dst_port = 0;

    /* Calculate IP header length */
    __u32 ip_hdr_len = ip->ihl * 4;
    if (ip_hdr_len < sizeof(*ip))
        return XDP_PASS;

    void *transport = (void *)ip + ip_hdr_len;
    struct teleology_hdr *th = NULL;

    /* Parse transport header */
    if (protocol == IPPROTO_UDP) {
        struct udphdr *udp = transport;
        if ((void *)(udp + 1) > data_end)
            return XDP_PASS;

        src_port = bpf_ntohs(udp->source);
        dst_port = bpf_ntohs(udp->dest);

        /* Look for teleological header in UDP payload */
        th = (struct teleology_hdr *)(udp + 1);
        if ((void *)(th + 1) > data_end)
            th = NULL;

    } else if (protocol == IPPROTO_TCP) {
        struct tcphdr *tcp = transport;
        if ((void *)(tcp + 1) > data_end)
            return XDP_PASS;

        src_port = bpf_ntohs(tcp->source);
        dst_port = bpf_ntohs(tcp->dest);
    }

    /* Check for SoulMesh teleological header */
    if (th && is_soulmesh_packet(th)) {
        increment_stat(6); /* teleology packets */

        /* Pain packet detection - very high priority = danger */
        if (th->priority >= PRIORITY_PAIN_THRESHOLD) {
            increment_stat(4); /* pain packets */

            /* Emit event for userspace learning */
            emit_reflex_event(ctx, src_ip, dst_ip, src_port, dst_port,
                             protocol, th, 1 /* drop */, 1 /* pain */);

            /* Apply rate limiting to pain packets */
            if (check_rate_limit(src_ip)) {
                increment_stat(7); /* rate limited */
                increment_stat(2); /* dropped */
                return XDP_DROP;
            }

            /* Mark packet for further inspection but allow through */
            /* (In production, might redirect to analysis queue) */
        }

        /* Suspicious packet detection */
        if (th->priority >= PRIORITY_SUSPICIOUS_THRESHOLD &&
            th->priority < PRIORITY_PAIN_THRESHOLD) {
            increment_stat(5); /* suspicious packets */

            /* Emit event for learning */
            emit_reflex_event(ctx, src_ip, dst_ip, src_port, dst_port,
                             protocol, th, 2 /* mark */, 2 /* suspicious */);

            increment_stat(3); /* marked */
        }
    }

    /* Default: pass packet through */
    increment_stat(1); /* passed */
    return XDP_PASS;
}

/* License declaration required for BPF */
char _license[] SEC("license") = "GPL";
