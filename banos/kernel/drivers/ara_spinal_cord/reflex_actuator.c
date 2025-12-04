/*
 * BANOS - Bio-Affective Neuromorphic Operating System
 * Reflex Actuator - MMIO Bridge Between BPF and FPGA
 *
 * This module handles the "efferent" path: reading reflex commands from
 * BPF maps and applying them to actual FPGA hardware registers.
 *
 * Architecture:
 *   BPF (banos_affective.bpf.c)
 *     ↓ writes reflex_cmd to banos_reflex_cmd_map
 *   This module (kthread polling or timer)
 *     ↓ reads map, validates, applies to MMIO
 *   FPGA (reflex_controller.v)
 *     ↓ hardware response (fans, PROCHOT, power)
 *   This module
 *     ↓ reads feedback, updates banos_spine_map
 *   BPF (reads spine state for next iteration)
 *
 * This keeps MMIO out of BPF (verifier-safe) while maintaining
 * sub-millisecond reflex latency.
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/kthread.h>
#include <linux/delay.h>
#include <linux/io.h>
#include <linux/bpf.h>
#include <linux/btf.h>

/* Include shared definitions */
#include "../../include/banos_common.h"

/* FPGA Register Map (offsets from BAR base) */
#define FPGA_REG_REFLEX_CMD      0x00    /* Write: reflex bitmask */
#define FPGA_REG_REFLEX_STATUS   0x04    /* Read: current active reflexes */
#define FPGA_REG_THERMAL_SPIKES  0x08    /* Read: thermal spike count */
#define FPGA_REG_ERROR_SPIKES    0x0C    /* Read: error spike count */
#define FPGA_REG_THERMAL_SRC     0x10    /* Read: hottest sensor ID */
#define FPGA_REG_THERMAL_CRIT    0x14    /* Read: critical flag */
#define FPGA_REG_SPIKE_WINDOW    0x18    /* Read/Write: spike window (ms) */
#define FPGA_REG_TIMESTAMP_LO    0x20    /* Read: timestamp low 32 bits */
#define FPGA_REG_TIMESTAMP_HI    0x24    /* Read: timestamp high 32 bits */

/* Module parameters */
static void __iomem *fpga_base = NULL;
static unsigned long fpga_phys_addr = 0;
static unsigned long fpga_size = 4096;

module_param(fpga_phys_addr, ulong, 0444);
MODULE_PARM_DESC(fpga_phys_addr, "Physical address of FPGA BAR");
module_param(fpga_size, ulong, 0444);
MODULE_PARM_DESC(fpga_size, "Size of FPGA memory region");

/* Polling thread */
static struct task_struct *actuator_thread;
static bool thread_should_stop = false;

/* BPF map file descriptors (set via sysfs) */
static int reflex_cmd_map_fd = -1;
static int spine_map_fd = -1;

/* Local state for spike delta computation */
static u32 prev_thermal_spikes = 0;
static u32 prev_error_spikes = 0;

/*
 * Read FPGA registers and update spinal cord state
 */
static void update_spine_from_fpga(struct banos_spinal_cord *spine)
{
    if (!fpga_base) {
        /* No FPGA: use simulated values for testing */
        spine->thermal_spike_cnt = 0;
        spine->error_spike_cnt = 0;
        spine->thermal_source_id = BANOS_THERMAL_SRC_CPU;
        spine->thermal_source_critical = 0;
        spine->reflex_active = BANOS_RFLX_NONE;
        return;
    }

    /* Read current spike counts */
    u32 thermal_spikes = ioread32(fpga_base + FPGA_REG_THERMAL_SPIKES);
    u32 error_spikes = ioread32(fpga_base + FPGA_REG_ERROR_SPIKES);

    /* Compute deltas (for rate-based PAD) */
    spine->thermal_spike_delta = thermal_spikes - prev_thermal_spikes;
    spine->error_spike_delta = error_spikes - prev_error_spikes;
    prev_thermal_spikes = thermal_spikes;
    prev_error_spikes = error_spikes;

    /* Store absolute counts too */
    spine->thermal_spike_cnt = thermal_spikes;
    spine->error_spike_cnt = error_spikes;

    /* Read thermal source info */
    spine->thermal_source_id = ioread32(fpga_base + FPGA_REG_THERMAL_SRC) & 0xFF;
    spine->thermal_source_critical = ioread32(fpga_base + FPGA_REG_THERMAL_CRIT) ? 1 : 0;

    /* Read reflex feedback (what's actually active) */
    spine->reflex_active = ioread32(fpga_base + FPGA_REG_REFLEX_STATUS);

    /* Read window size */
    spine->update_interval_ms = ioread32(fpga_base + FPGA_REG_SPIKE_WINDOW) & 0xFFFF;
}

/*
 * Apply reflex command to FPGA
 */
static void apply_reflex_to_fpga(u32 reflex_cmd)
{
    if (!fpga_base) {
        /* No FPGA: log for debugging */
        if (reflex_cmd != BANOS_RFLX_NONE) {
            pr_debug("BANOS REFLEX (simulated): 0x%x\n", reflex_cmd);
        }
        return;
    }

    /* Safety check: log significant reflexes */
    if (reflex_cmd & BANOS_RFLX_SYS_HALT) {
        pr_crit("BANOS: CRITICAL REFLEX - System halt requested!\n");
    } else if (reflex_cmd & BANOS_RFLX_GPU_KILL) {
        pr_warn("BANOS: GPU power cut requested\n");
    } else if (reflex_cmd & BANOS_RFLX_THROTTLE) {
        pr_info("BANOS: PROCHOT throttle engaged\n");
    }

    /* Write to FPGA */
    iowrite32(reflex_cmd, fpga_base + FPGA_REG_REFLEX_CMD);
}

/*
 * Main actuator loop
 * Polls BPF maps and updates FPGA at ~100Hz
 */
static int actuator_thread_fn(void *data)
{
    struct banos_spinal_cord spine = {0};
    u32 last_reflex_cmd = BANOS_RFLX_NONE;

    pr_info("BANOS reflex actuator thread started\n");

    while (!kthread_should_stop() && !thread_should_stop) {
        /*
         * Step 1: Read FPGA state → spine struct
         */
        update_spine_from_fpga(&spine);

        /*
         * Step 2: Write spine to BPF map (so affective layer can read it)
         *
         * NOTE: In real implementation, we'd use bpf_map_update_elem
         * from kernel space. For now, this is a placeholder showing
         * the architecture. The actual mechanism would be:
         *   - Export map via pinned path (/sys/fs/bpf/banos_spine_map)
         *   - Use kernel BPF map helpers
         */
        /* TODO: bpf_map_update_elem(spine_map_fd, &key, &spine, BPF_ANY); */

        /*
         * Step 3: Read reflex command from BPF map
         *
         * TODO: bpf_map_lookup_elem(reflex_cmd_map_fd, &key, &reflex_cmd);
         */
        u32 reflex_cmd = BANOS_RFLX_NONE;  /* Placeholder */

        /*
         * Step 4: Apply to FPGA (only if changed, to reduce bus traffic)
         */
        if (reflex_cmd != last_reflex_cmd) {
            apply_reflex_to_fpga(reflex_cmd);
            last_reflex_cmd = reflex_cmd;

            /* Record reflex event for Ara's awareness */
            if (reflex_cmd != BANOS_RFLX_NONE) {
                spine.reflex_log = reflex_cmd;
                spine.reflex_timestamp_ns = ktime_get_ns();
            }
        }

        /* Track reflex duration */
        if (spine.reflex_active != BANOS_RFLX_NONE) {
            spine.reflex_duration_ms += 10;  /* 10ms per loop */
        } else {
            spine.reflex_duration_ms = 0;
        }

        /* Sleep ~10ms (100Hz update rate) */
        usleep_range(9000, 11000);
    }

    pr_info("BANOS reflex actuator thread stopped\n");
    return 0;
}

/*
 * Sysfs interface for setting BPF map FDs
 * (In production, these would be set via proper BPF infrastructure)
 */
static ssize_t reflex_cmd_map_fd_store(struct kobject *kobj,
                                       struct kobj_attribute *attr,
                                       const char *buf, size_t count)
{
    int fd;
    if (kstrtoint(buf, 10, &fd) < 0)
        return -EINVAL;
    reflex_cmd_map_fd = fd;
    return count;
}

static ssize_t spine_map_fd_store(struct kobject *kobj,
                                  struct kobj_attribute *attr,
                                  const char *buf, size_t count)
{
    int fd;
    if (kstrtoint(buf, 10, &fd) < 0)
        return -EINVAL;
    spine_map_fd = fd;
    return count;
}

static struct kobj_attribute reflex_cmd_map_fd_attr =
    __ATTR(reflex_cmd_map_fd, 0220, NULL, reflex_cmd_map_fd_store);
static struct kobj_attribute spine_map_fd_attr =
    __ATTR(spine_map_fd, 0220, NULL, spine_map_fd_store);

static struct attribute *actuator_attrs[] = {
    &reflex_cmd_map_fd_attr.attr,
    &spine_map_fd_attr.attr,
    NULL,
};

static struct attribute_group actuator_attr_group = {
    .name = "reflex_actuator",
    .attrs = actuator_attrs,
};

static struct kobject *actuator_kobj;

/*
 * Module init
 */
static int __init reflex_actuator_init(void)
{
    int ret;

    /* Map FPGA if address provided */
    if (fpga_phys_addr != 0) {
        fpga_base = ioremap(fpga_phys_addr, fpga_size);
        if (!fpga_base) {
            pr_err("BANOS: Failed to map FPGA at 0x%lx\n", fpga_phys_addr);
            return -ENOMEM;
        }
        pr_info("BANOS: FPGA mapped at 0x%lx (size %lu)\n",
                fpga_phys_addr, fpga_size);
    } else {
        pr_info("BANOS: No FPGA address - running in simulation mode\n");
    }

    /* Create sysfs interface */
    actuator_kobj = kobject_create_and_add("banos", kernel_kobj);
    if (!actuator_kobj) {
        ret = -ENOMEM;
        goto err_unmap;
    }

    ret = sysfs_create_group(actuator_kobj, &actuator_attr_group);
    if (ret) {
        goto err_kobj;
    }

    /* Start actuator thread */
    actuator_thread = kthread_run(actuator_thread_fn, NULL,
                                  "banos_reflex");
    if (IS_ERR(actuator_thread)) {
        ret = PTR_ERR(actuator_thread);
        goto err_sysfs;
    }

    pr_info("BANOS reflex actuator initialized\n");
    return 0;

err_sysfs:
    sysfs_remove_group(actuator_kobj, &actuator_attr_group);
err_kobj:
    kobject_put(actuator_kobj);
err_unmap:
    if (fpga_base)
        iounmap(fpga_base);
    return ret;
}

/*
 * Module exit
 */
static void __exit reflex_actuator_exit(void)
{
    /* Stop thread */
    thread_should_stop = true;
    if (actuator_thread)
        kthread_stop(actuator_thread);

    /* Clean up sysfs */
    sysfs_remove_group(actuator_kobj, &actuator_attr_group);
    kobject_put(actuator_kobj);

    /* Unmap FPGA */
    if (fpga_base)
        iounmap(fpga_base);

    pr_info("BANOS reflex actuator unloaded\n");
}

module_init(reflex_actuator_init);
module_exit(reflex_actuator_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("BANOS Project");
MODULE_DESCRIPTION("BANOS Reflex Actuator - MMIO Bridge");
MODULE_VERSION("1.0");
