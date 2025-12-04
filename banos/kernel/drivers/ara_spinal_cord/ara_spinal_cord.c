/*
 * BANOS - Bio-Affective Neuromorphic Operating System
 * ara_spinal_cord.ko - Kernel driver for FPGA neural interface
 *
 * This driver connects the kernel to the FPGA "hindbrain":
 * - Maps FPGA registers via PCIe/AXI
 * - Reads neural state (spike patterns, pain level)
 * - Computes PAD state for scheduler
 * - Provides mmap interface for user-space Ara daemon
 * - Handles hardware interrupts from FPGA reflexes
 *
 * The "spinal cord" metaphor: fast, reflexive communication
 * between lower (FPGA) and higher (LLM) brain regions.
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/mm.h>
#include <linux/slab.h>
#include <linux/io.h>
#include <linux/interrupt.h>
#include <linux/platform_device.h>
#include <linux/of.h>
#include <linux/dma-mapping.h>
#include <linux/workqueue.h>
#include <linux/kthread.h>
#include <linux/delay.h>

#define DRIVER_NAME     "ara_spinal_cord"
#define DEVICE_NAME     "banos"
#define CLASS_NAME      "banos"

/* FPGA Register Offsets (must match axi4_bridge.v) */
#define REG_NEURAL_STATE    0x00
#define REG_PAIN_LEVEL      0x04
#define REG_REFLEX_LOG      0x08
#define REG_AROUSAL         0x0C
#define REG_DOMINANCE       0x10
#define REG_PLEASURE        0x14
#define REG_ALERT_STATUS    0x18
#define REG_TIMESTAMP       0x1C
#define REG_CONTROL         0x20
#define REG_CONFIG          0x24
#define REG_IRQ_ENABLE      0x28
#define REG_IRQ_STATUS      0x2C
#define REG_TOTAL_SPIKES    0x30
#define REG_FAN_PWM         0x34

/* Control register bits */
#define CTRL_SNN_ENABLE     (1 << 0)
#define CTRL_LEARN_ENABLE   (1 << 1)
#define CTRL_FORCE_INHIBIT  (1 << 2)

/* IRQ bits */
#define IRQ_ALERT           (1 << 0)
#define IRQ_EMERGENCY       (1 << 1)
#define IRQ_PROCHOT         (1 << 2)

/* Shared memory structure for mmap */
struct banos_shared_mem {
    /* FPGA state (updated by driver) */
    u32 neural_state;
    u16 pain_level;
    u16 reserved1;
    u32 reflex_log;

    /* PAD state (computed by driver) */
    s16 pleasure;
    s16 arousal;
    s16 dominance;
    u8  quadrant;
    u8  sched_mode;

    /* Bat algorithm parameters */
    u16 loudness;
    u16 pulse_rate;
    u32 frequency;

    /* Kill threshold */
    u8  kill_priority_threshold;
    u8  reserved2[3];

    /* Statistics */
    u64 tasks_scheduled;
    u64 tasks_killed;
    u64 mode_switches;
    u64 alert_count;
    u64 last_alert_time;

    /* Alert ring buffer (last 16 alerts) */
    struct {
        u64 timestamp;
        u32 type;
        u32 data;
    } alerts[16];
    u32 alert_head;
    u32 alert_tail;

    /* Control (written by user daemon) */
    u32 user_control;
    u32 user_config;
};

/* Driver private data */
struct ara_dev {
    struct device *dev;
    struct cdev cdev;
    struct class *class;
    dev_t devno;

    /* FPGA memory-mapped registers */
    void __iomem *regs;
    resource_size_t regs_phys;
    resource_size_t regs_size;

    /* Shared memory for mmap */
    struct banos_shared_mem *shared;
    dma_addr_t shared_dma;

    /* IRQ */
    int irq;
    struct work_struct irq_work;

    /* Polling thread */
    struct task_struct *poll_thread;
    bool poll_running;

    /* State */
    spinlock_t lock;
    u32 alert_count;
};

static struct ara_dev *ara_device;

/*
 * Read FPGA register
 */
static inline u32 ara_read_reg(struct ara_dev *ara, u32 offset)
{
    if (ara->regs)
        return ioread32(ara->regs + offset);
    return 0;
}

/*
 * Write FPGA register
 */
static inline void ara_write_reg(struct ara_dev *ara, u32 offset, u32 value)
{
    if (ara->regs)
        iowrite32(value, ara->regs + offset);
}

/*
 * Compute PAD state from FPGA telemetry
 */
static void ara_compute_pad(struct ara_dev *ara)
{
    struct banos_shared_mem *sh = ara->shared;
    s32 p, a, d;

    /* Read raw values from FPGA */
    sh->pain_level = ara_read_reg(ara, REG_PAIN_LEVEL) & 0xFFFF;
    sh->neural_state = ara_read_reg(ara, REG_NEURAL_STATE);

    /* Pleasure: inverse of pain */
    /* pain_level 0-65535 -> pleasure -256 to +256 */
    p = 256 - ((s32)sh->pain_level >> 7);
    if (p > 255) p = 255;
    if (p < -256) p = -256;
    sh->pleasure = (s16)p;

    /* Arousal: from FPGA register (activity level) */
    a = (s16)(ara_read_reg(ara, REG_AROUSAL) & 0xFFFF);
    sh->arousal = a;

    /* Dominance: from FPGA register (resource metric) */
    d = (s16)(ara_read_reg(ara, REG_DOMINANCE) & 0xFFFF);
    sh->dominance = d;

    /* Classify quadrant */
    if (p < -180) {
        sh->quadrant = 6;  /* EMERGENCY */
        sh->sched_mode = 5;
    } else if (p >= 0 && a >= 0) {
        sh->quadrant = 1;  /* EXCITED */
        sh->sched_mode = 1;  /* THROUGHPUT */
    } else if (p >= 0 && a < 0) {
        sh->quadrant = 0;  /* SERENE */
        sh->sched_mode = 0;  /* NORMAL */
    } else if (p < 0 && a >= 0) {
        sh->quadrant = 2;  /* ANXIOUS */
        sh->sched_mode = 4;  /* DEADLINE */
    } else {
        sh->quadrant = 3;  /* DEPRESSED */
        sh->sched_mode = 3;  /* POWERSAVE */
    }

    /* Bat algorithm: loudness increases with stress */
    sh->loudness = (u16)(256 - p) * 128;  /* 0-65536 range */
    sh->pulse_rate = (u16)((sh->arousal > 0) ? sh->arousal * 2 : 0);

    /* Kill threshold based on pleasure */
    if (p < -200)
        sh->kill_priority_threshold = 15;
    else if (p < -128)
        sh->kill_priority_threshold = 10;
    else if (p < -64)
        sh->kill_priority_threshold = 5;
    else
        sh->kill_priority_threshold = 0;
}

/*
 * Add alert to ring buffer
 */
static void ara_add_alert(struct ara_dev *ara, u32 type, u32 data)
{
    struct banos_shared_mem *sh = ara->shared;
    u32 next_head;
    unsigned long flags;

    spin_lock_irqsave(&ara->lock, flags);

    sh->alerts[sh->alert_head].timestamp = ktime_get_ns();
    sh->alerts[sh->alert_head].type = type;
    sh->alerts[sh->alert_head].data = data;

    next_head = (sh->alert_head + 1) % 16;
    if (next_head == sh->alert_tail) {
        /* Buffer full, drop oldest */
        sh->alert_tail = (sh->alert_tail + 1) % 16;
    }
    sh->alert_head = next_head;

    sh->alert_count++;
    sh->last_alert_time = ktime_get_ns();

    spin_unlock_irqrestore(&ara->lock, flags);
}

/*
 * IRQ handler bottom half
 */
static void ara_irq_work_handler(struct work_struct *work)
{
    struct ara_dev *ara = container_of(work, struct ara_dev, irq_work);
    u32 status;

    status = ara_read_reg(ara, REG_IRQ_STATUS);

    if (status & IRQ_ALERT) {
        dev_info(ara->dev, "BANOS: Vacuum Spiker ALERT!\n");
        ara_add_alert(ara, 1, ara_read_reg(ara, REG_PAIN_LEVEL));
    }

    if (status & IRQ_EMERGENCY) {
        dev_crit(ara->dev, "BANOS: EMERGENCY - thermal limit exceeded!\n");
        ara_add_alert(ara, 2, ara_read_reg(ara, REG_NEURAL_STATE));
    }

    if (status & IRQ_PROCHOT) {
        dev_warn(ara->dev, "BANOS: PROCHOT asserted by FPGA reflex\n");
        ara_add_alert(ara, 3, ara_read_reg(ara, REG_REFLEX_LOG));
    }

    /* Clear handled interrupts (W1C) */
    ara_write_reg(ara, REG_IRQ_STATUS, status);

    /* Update PAD after alert */
    ara_compute_pad(ara);
}

/*
 * IRQ handler top half
 */
static irqreturn_t ara_irq_handler(int irq, void *dev_id)
{
    struct ara_dev *ara = dev_id;

    /* Schedule bottom half */
    schedule_work(&ara->irq_work);

    return IRQ_HANDLED;
}

/*
 * Polling thread - updates PAD periodically
 */
static int ara_poll_thread(void *data)
{
    struct ara_dev *ara = data;

    while (!kthread_should_stop()) {
        if (ara->poll_running) {
            /* Update shared state from FPGA */
            ara->shared->reflex_log = ara_read_reg(ara, REG_REFLEX_LOG);

            /* Recompute PAD */
            ara_compute_pad(ara);
        }

        /* Poll at 100Hz */
        msleep(10);
    }

    return 0;
}

/*
 * File operations: open
 */
static int ara_open(struct inode *inode, struct file *file)
{
    file->private_data = ara_device;
    return 0;
}

/*
 * File operations: release
 */
static int ara_release(struct inode *inode, struct file *file)
{
    return 0;
}

/*
 * File operations: mmap
 * Maps shared memory to user space for zero-copy telemetry
 */
static int ara_mmap(struct file *file, struct vm_area_struct *vma)
{
    struct ara_dev *ara = file->private_data;
    unsigned long size = vma->vm_end - vma->vm_start;

    if (size > PAGE_SIZE)
        return -EINVAL;

    /* Map the shared memory page */
    return remap_pfn_range(vma, vma->vm_start,
                          page_to_pfn(virt_to_page(ara->shared)),
                          size, vma->vm_page_prot);
}

/*
 * File operations: ioctl
 */
static long ara_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
    struct ara_dev *ara = file->private_data;

    switch (cmd) {
    case 0x1001:  /* Enable SNN */
        ara_write_reg(ara, REG_CONTROL,
                     ara_read_reg(ara, REG_CONTROL) | CTRL_SNN_ENABLE);
        ara->poll_running = true;
        break;

    case 0x1002:  /* Disable SNN */
        ara_write_reg(ara, REG_CONTROL,
                     ara_read_reg(ara, REG_CONTROL) & ~CTRL_SNN_ENABLE);
        ara->poll_running = false;
        break;

    case 0x1003:  /* Enable learning */
        ara_write_reg(ara, REG_CONTROL,
                     ara_read_reg(ara, REG_CONTROL) | CTRL_LEARN_ENABLE);
        break;

    case 0x1004:  /* Disable learning */
        ara_write_reg(ara, REG_CONTROL,
                     ara_read_reg(ara, REG_CONTROL) & ~CTRL_LEARN_ENABLE);
        break;

    case 0x1005:  /* Force inhibition (training mode) */
        ara_write_reg(ara, REG_CONTROL,
                     ara_read_reg(ara, REG_CONTROL) | CTRL_FORCE_INHIBIT);
        break;

    case 0x1006:  /* Normal mode */
        ara_write_reg(ara, REG_CONTROL,
                     ara_read_reg(ara, REG_CONTROL) & ~CTRL_FORCE_INHIBIT);
        break;

    case 0x1007:  /* Get alert count */
        return ara->shared->alert_count;

    default:
        return -EINVAL;
    }

    return 0;
}

static const struct file_operations ara_fops = {
    .owner          = THIS_MODULE,
    .open           = ara_open,
    .release        = ara_release,
    .mmap           = ara_mmap,
    .unlocked_ioctl = ara_ioctl,
};

/*
 * Platform driver probe
 */
static int ara_probe(struct platform_device *pdev)
{
    struct ara_dev *ara;
    struct resource *res;
    int ret;

    ara = devm_kzalloc(&pdev->dev, sizeof(*ara), GFP_KERNEL);
    if (!ara)
        return -ENOMEM;

    ara->dev = &pdev->dev;
    spin_lock_init(&ara->lock);
    INIT_WORK(&ara->irq_work, ara_irq_work_handler);

    /* Get FPGA register resource */
    res = platform_get_resource(pdev, IORESOURCE_MEM, 0);
    if (res) {
        ara->regs = devm_ioremap_resource(&pdev->dev, res);
        if (IS_ERR(ara->regs)) {
            dev_warn(&pdev->dev, "Failed to map FPGA registers, running in simulation mode\n");
            ara->regs = NULL;
        } else {
            ara->regs_phys = res->start;
            ara->regs_size = resource_size(res);
        }
    } else {
        dev_info(&pdev->dev, "No FPGA resource, running in simulation mode\n");
    }

    /* Get IRQ */
    ara->irq = platform_get_irq(pdev, 0);
    if (ara->irq > 0) {
        ret = devm_request_irq(&pdev->dev, ara->irq, ara_irq_handler,
                              IRQF_SHARED, DRIVER_NAME, ara);
        if (ret) {
            dev_warn(&pdev->dev, "Failed to request IRQ, polling only\n");
            ara->irq = -1;
        }
    }

    /* Allocate shared memory */
    ara->shared = dma_alloc_coherent(&pdev->dev, PAGE_SIZE,
                                     &ara->shared_dma, GFP_KERNEL);
    if (!ara->shared) {
        /* Fall back to regular allocation */
        ara->shared = kzalloc(PAGE_SIZE, GFP_KERNEL);
        if (!ara->shared)
            return -ENOMEM;
    }

    /* Initialize shared state */
    memset(ara->shared, 0, sizeof(*ara->shared));
    ara->shared->sched_mode = 0;  /* NORMAL */
    ara->shared->quadrant = 0;    /* SERENE */

    /* Create character device */
    ret = alloc_chrdev_region(&ara->devno, 0, 1, DEVICE_NAME);
    if (ret < 0)
        goto err_free_shared;

    cdev_init(&ara->cdev, &ara_fops);
    ara->cdev.owner = THIS_MODULE;

    ret = cdev_add(&ara->cdev, ara->devno, 1);
    if (ret < 0)
        goto err_unreg_chrdev;

    ara->class = class_create(CLASS_NAME);
    if (IS_ERR(ara->class)) {
        ret = PTR_ERR(ara->class);
        goto err_cdev_del;
    }

    device_create(ara->class, &pdev->dev, ara->devno, NULL, DEVICE_NAME);

    /* Start polling thread */
    ara->poll_thread = kthread_run(ara_poll_thread, ara, "ara_poll");
    if (IS_ERR(ara->poll_thread)) {
        ret = PTR_ERR(ara->poll_thread);
        goto err_device_destroy;
    }

    platform_set_drvdata(pdev, ara);
    ara_device = ara;

    /* Enable SNN by default */
    if (ara->regs) {
        ara_write_reg(ara, REG_CONTROL, CTRL_SNN_ENABLE);
        ara_write_reg(ara, REG_IRQ_ENABLE, IRQ_ALERT | IRQ_EMERGENCY | IRQ_PROCHOT);
    }
    ara->poll_running = true;

    dev_info(&pdev->dev, "BANOS Spinal Cord initialized\n");
    return 0;

err_device_destroy:
    device_destroy(ara->class, ara->devno);
    class_destroy(ara->class);
err_cdev_del:
    cdev_del(&ara->cdev);
err_unreg_chrdev:
    unregister_chrdev_region(ara->devno, 1);
err_free_shared:
    if (ara->shared_dma)
        dma_free_coherent(&pdev->dev, PAGE_SIZE, ara->shared, ara->shared_dma);
    else
        kfree(ara->shared);
    return ret;
}

/*
 * Platform driver remove
 */
static int ara_remove(struct platform_device *pdev)
{
    struct ara_dev *ara = platform_get_drvdata(pdev);

    ara->poll_running = false;
    kthread_stop(ara->poll_thread);

    cancel_work_sync(&ara->irq_work);

    device_destroy(ara->class, ara->devno);
    class_destroy(ara->class);
    cdev_del(&ara->cdev);
    unregister_chrdev_region(ara->devno, 1);

    if (ara->shared_dma)
        dma_free_coherent(&pdev->dev, PAGE_SIZE, ara->shared, ara->shared_dma);
    else
        kfree(ara->shared);

    dev_info(&pdev->dev, "BANOS Spinal Cord removed\n");
    return 0;
}

static const struct of_device_id ara_of_match[] = {
    { .compatible = "banos,spinal-cord" },
    { },
};
MODULE_DEVICE_TABLE(of, ara_of_match);

static struct platform_driver ara_driver = {
    .probe  = ara_probe,
    .remove = ara_remove,
    .driver = {
        .name = DRIVER_NAME,
        .of_match_table = ara_of_match,
    },
};

module_platform_driver(ara_driver);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("BANOS Project");
MODULE_DESCRIPTION("Ara Spinal Cord - FPGA Neural Interface Driver");
MODULE_VERSION("1.0");
