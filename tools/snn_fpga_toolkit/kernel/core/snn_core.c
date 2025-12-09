/*
 * SNN Kernel Core Module
 *
 * Main entry point for the SNN-optimized kernel system
 * Coordinates GPU-FPGA integration, memory management, and real-time scheduling
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/fs.h>
#include <linux/device.h>
#include <linux/cdev.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <linux/mutex.h>
#include <linux/version.h>

#include "snn_core.h"

MODULE_LICENSE("GPL");
MODULE_AUTHOR("SNN Kernel Development Team");
MODULE_DESCRIPTION("SNN-Optimized Kernel for FPGA-GPU Integration");
MODULE_VERSION("1.0.0");

/* Module parameters */
static int debug_level = 0;
module_param(debug_level, int, 0644);
MODULE_PARM_DESC(debug_level, "Debug logging level (0-5)");

static int max_devices = 4;
module_param(max_devices, int, 0444);
MODULE_PARM_DESC(max_devices, "Maximum number of devices to manage");

/* Global state */
static struct snn_core_state {
    struct class *class;
    struct device *device;
    struct cdev cdev;
    dev_t devt;

    struct mutex lock;
    bool initialized;

    /* Subsystem pointers */
    struct snn_pcie_manager *pcie_mgr;
    struct snn_memory_manager *mem_mgr;
    struct snn_rt_scheduler *rt_sched;
    struct snn_cuda_bridge *cuda_bridge;
    struct snn_nvme_manager *nvme_mgr;
    struct snn_pipeline *pipeline;

    /* Statistics */
    atomic64_t total_operations;
    atomic64_t errors;

} *core_state;

/* Debug logging */
#define snn_debug(level, fmt, ...) \
    do { \
        if (debug_level >= level) \
            pr_info("SNN[%d]: " fmt, level, ##__VA_ARGS__); \
    } while (0)

#define snn_err(fmt, ...) pr_err("SNN_ERR: " fmt, ##__VA_ARGS__)
#define snn_warn(fmt, ...) pr_warn("SNN_WARN: " fmt, ##__VA_ARGS__)
#define snn_info(fmt, ...) pr_info("SNN: " fmt, ##__VA_ARGS__)

/*
 * Character device operations
 */

static int snn_open(struct inode *inode, struct file *filp)
{
    snn_debug(2, "Device opened\n");

    if (!core_state->initialized) {
        snn_err("Core not initialized\n");
        return -EAGAIN;
    }

    filp->private_data = core_state;
    atomic64_inc(&core_state->total_operations);

    return 0;
}

static int snn_release(struct inode *inode, struct file *filp)
{
    snn_debug(2, "Device closed\n");
    return 0;
}

static long snn_ioctl_init(struct file *filp, unsigned long arg)
{
    snn_kernel_init_t config;
    int ret;

    if (copy_from_user(&config, (void __user *)arg, sizeof(config)))
        return -EFAULT;

    snn_debug(1, "Initializing: GPU=%u, FPGA=%u, mem=%llu\n",
              config.gpu_id, config.fpga_id, config.pinned_mem_size);

    mutex_lock(&core_state->lock);

    /* Initialize PCIe subsystem */
    ret = snn_pcie_init(core_state->pcie_mgr, &config);
    if (ret) {
        snn_err("PCIe init failed: %d\n", ret);
        goto out;
    }

    /* Initialize memory subsystem */
    ret = snn_memory_init(core_state->mem_mgr, &config);
    if (ret) {
        snn_err("Memory init failed: %d\n", ret);
        goto out;
    }

    /* Initialize RT scheduler */
    ret = snn_rt_sched_init(core_state->rt_sched, &config);
    if (ret) {
        snn_err("RT scheduler init failed: %d\n", ret);
        goto out;
    }

    /* Initialize CUDA bridge */
    ret = snn_cuda_bridge_init(core_state->cuda_bridge, &config);
    if (ret) {
        snn_err("CUDA bridge init failed: %d\n", ret);
        goto out;
    }

    /* Initialize NVMe subsystem */
    ret = snn_nvme_init(core_state->nvme_mgr, &config);
    if (ret) {
        snn_err("NVMe init failed: %d\n", ret);
        goto out;
    }

    /* Initialize SNN pipeline */
    ret = snn_pipeline_init(core_state->pipeline, &config);
    if (ret) {
        snn_err("Pipeline init failed: %d\n", ret);
        goto out;
    }

    core_state->initialized = true;
    snn_info("Initialization complete\n");

out:
    mutex_unlock(&core_state->lock);
    return ret;
}

static long snn_ioctl_get_device_info(struct file *filp, unsigned long arg)
{
    snn_device_info_t info;
    int ret;

    if (copy_from_user(&info, (void __user *)arg, sizeof(info)))
        return -EFAULT;

    ret = snn_pcie_get_device_info(core_state->pcie_mgr, &info);
    if (ret)
        return ret;

    if (copy_to_user((void __user *)arg, &info, sizeof(info)))
        return -EFAULT;

    return 0;
}

static long snn_ioctl_alloc_mem(struct file *filp, unsigned long arg)
{
    snn_mem_alloc_t req;
    int ret;

    if (copy_from_user(&req, (void __user *)arg, sizeof(req)))
        return -EFAULT;

    snn_debug(2, "Allocating %llu bytes with flags 0x%x\n", req.size, req.flags);

    ret = snn_memory_alloc(core_state->mem_mgr, &req);
    if (ret)
        return ret;

    if (copy_to_user((void __user *)arg, &req, sizeof(req)))
        return -EFAULT;

    return 0;
}

static long snn_ioctl_free_mem(struct file *filp, unsigned long arg)
{
    __u32 mem_id;

    if (get_user(mem_id, (__u32 __user *)arg))
        return -EFAULT;

    return snn_memory_free(core_state->mem_mgr, mem_id);
}

static long snn_ioctl_p2p_transfer(struct file *filp, unsigned long arg)
{
    snn_p2p_transfer_t transfer;
    int ret;

    if (copy_from_user(&transfer, (void __user *)arg, sizeof(transfer)))
        return -EFAULT;

    snn_debug(2, "P2P transfer: %d->%d, size=%llu\n",
              transfer.src_dev, transfer.dst_dev, transfer.size);

    ret = snn_pcie_p2p_transfer(core_state->pcie_mgr, &transfer);
    if (ret)
        return ret;

    if (copy_to_user((void __user *)arg, &transfer, sizeof(transfer)))
        return -EFAULT;

    return 0;
}

static long snn_ioctl_set_rt_params(struct file *filp, unsigned long arg)
{
    snn_rt_sched_params_t params;

    if (copy_from_user(&params, (void __user *)arg, sizeof(params)))
        return -EFAULT;

    return snn_rt_sched_set_params(core_state->rt_sched, current, &params);
}

static long snn_ioctl_snn_compute(struct file *filp, unsigned long arg)
{
    snn_compute_params_t params;

    if (copy_from_user(&params, (void __user *)arg, sizeof(params)))
        return -EFAULT;

    return snn_pipeline_execute(core_state->pipeline, &params);
}

static long snn_ioctl_nvme_io(struct file *filp, unsigned long arg)
{
    snn_nvme_io_t io;
    int ret;

    if (copy_from_user(&io, (void __user *)arg, sizeof(io)))
        return -EFAULT;

    ret = snn_nvme_submit_io(core_state->nvme_mgr, &io);
    if (ret)
        return ret;

    if (copy_to_user((void __user *)arg, &io, sizeof(io)))
        return -EFAULT;

    return 0;
}

static long snn_ioctl_get_stats(struct file *filp, unsigned long arg)
{
    snn_perf_stats_t stats = {0};

    /* Gather statistics from subsystems */
    snn_pcie_get_stats(core_state->pcie_mgr, &stats);
    snn_memory_get_stats(core_state->mem_mgr, &stats);
    snn_rt_sched_get_stats(core_state->rt_sched, &stats);
    snn_nvme_get_stats(core_state->nvme_mgr, &stats);

    if (copy_to_user((void __user *)arg, &stats, sizeof(stats)))
        return -EFAULT;

    return 0;
}

static long snn_ioctl_reset_stats(struct file *filp, unsigned long arg)
{
    snn_pcie_reset_stats(core_state->pcie_mgr);
    snn_memory_reset_stats(core_state->mem_mgr);
    snn_rt_sched_reset_stats(core_state->rt_sched);
    snn_nvme_reset_stats(core_state->nvme_mgr);

    atomic64_set(&core_state->total_operations, 0);
    atomic64_set(&core_state->errors, 0);

    return 0;
}

static long snn_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
    long ret = 0;

    snn_debug(3, "IOCTL cmd=0x%x\n", cmd);

    switch (cmd) {
    case SNN_IOC_INIT:
        ret = snn_ioctl_init(filp, arg);
        break;
    case SNN_IOC_GET_DEVICE_INFO:
        ret = snn_ioctl_get_device_info(filp, arg);
        break;
    case SNN_IOC_ALLOC_MEM:
        ret = snn_ioctl_alloc_mem(filp, arg);
        break;
    case SNN_IOC_FREE_MEM:
        ret = snn_ioctl_free_mem(filp, arg);
        break;
    case SNN_IOC_P2P_TRANSFER:
        ret = snn_ioctl_p2p_transfer(filp, arg);
        break;
    case SNN_IOC_SET_RT_PARAMS:
        ret = snn_ioctl_set_rt_params(filp, arg);
        break;
    case SNN_IOC_SNN_COMPUTE:
        ret = snn_ioctl_snn_compute(filp, arg);
        break;
    case SNN_IOC_NVME_IO:
        ret = snn_ioctl_nvme_io(filp, arg);
        break;
    case SNN_IOC_GET_STATS:
        ret = snn_ioctl_get_stats(filp, arg);
        break;
    case SNN_IOC_RESET_STATS:
        ret = snn_ioctl_reset_stats(filp, arg);
        break;
    default:
        snn_warn("Unknown IOCTL: 0x%x\n", cmd);
        ret = -ENOTTY;
    }

    if (ret < 0)
        atomic64_inc(&core_state->errors);

    return ret;
}

static const struct file_operations snn_fops = {
    .owner = THIS_MODULE,
    .open = snn_open,
    .release = snn_release,
    .unlocked_ioctl = snn_ioctl,
    .compat_ioctl = snn_ioctl,
};

/*
 * Module initialization
 */

static int __init snn_core_init(void)
{
    int ret;

    snn_info("Initializing SNN Kernel v%d.%d.%d\n",
             SNN_KERNEL_VERSION_MAJOR,
             SNN_KERNEL_VERSION_MINOR,
             SNN_KERNEL_VERSION_PATCH);

    /* Allocate core state */
    core_state = kzalloc(sizeof(*core_state), GFP_KERNEL);
    if (!core_state)
        return -ENOMEM;

    mutex_init(&core_state->lock);
    atomic64_set(&core_state->total_operations, 0);
    atomic64_set(&core_state->errors, 0);

    /* Allocate subsystem structures */
    core_state->pcie_mgr = kzalloc(sizeof(*core_state->pcie_mgr), GFP_KERNEL);
    core_state->mem_mgr = kzalloc(sizeof(*core_state->mem_mgr), GFP_KERNEL);
    core_state->rt_sched = kzalloc(sizeof(*core_state->rt_sched), GFP_KERNEL);
    core_state->cuda_bridge = kzalloc(sizeof(*core_state->cuda_bridge), GFP_KERNEL);
    core_state->nvme_mgr = kzalloc(sizeof(*core_state->nvme_mgr), GFP_KERNEL);
    core_state->pipeline = kzalloc(sizeof(*core_state->pipeline), GFP_KERNEL);

    if (!core_state->pcie_mgr || !core_state->mem_mgr ||
        !core_state->rt_sched || !core_state->cuda_bridge ||
        !core_state->nvme_mgr || !core_state->pipeline) {
        ret = -ENOMEM;
        goto fail;
    }

    /* Register character device */
    ret = alloc_chrdev_region(&core_state->devt, 0, 1, "snn");
    if (ret < 0) {
        snn_err("Failed to allocate char device region: %d\n", ret);
        goto fail;
    }

    cdev_init(&core_state->cdev, &snn_fops);
    core_state->cdev.owner = THIS_MODULE;

    ret = cdev_add(&core_state->cdev, core_state->devt, 1);
    if (ret < 0) {
        snn_err("Failed to add char device: %d\n", ret);
        goto fail_chrdev;
    }

    /* Create device class */
#if LINUX_VERSION_CODE >= KERNEL_VERSION(6, 4, 0)
    core_state->class = class_create("snn");
#else
    core_state->class = class_create(THIS_MODULE, "snn");
#endif
    if (IS_ERR(core_state->class)) {
        ret = PTR_ERR(core_state->class);
        snn_err("Failed to create device class: %d\n", ret);
        goto fail_cdev;
    }

    /* Create device */
    core_state->device = device_create(core_state->class, NULL,
                                       core_state->devt, NULL, "snn");
    if (IS_ERR(core_state->device)) {
        ret = PTR_ERR(core_state->device);
        snn_err("Failed to create device: %d\n", ret);
        goto fail_class;
    }

    snn_info("SNN Kernel initialized successfully (device major=%d)\n",
             MAJOR(core_state->devt));

    return 0;

fail_class:
    class_destroy(core_state->class);
fail_cdev:
    cdev_del(&core_state->cdev);
fail_chrdev:
    unregister_chrdev_region(core_state->devt, 1);
fail:
    kfree(core_state->pipeline);
    kfree(core_state->nvme_mgr);
    kfree(core_state->cuda_bridge);
    kfree(core_state->rt_sched);
    kfree(core_state->mem_mgr);
    kfree(core_state->pcie_mgr);
    kfree(core_state);
    return ret;
}

static void __exit snn_core_exit(void)
{
    snn_info("Shutting down SNN Kernel\n");

    if (!core_state)
        return;

    /* Cleanup subsystems if initialized */
    if (core_state->initialized) {
        snn_pipeline_cleanup(core_state->pipeline);
        snn_nvme_cleanup(core_state->nvme_mgr);
        snn_cuda_bridge_cleanup(core_state->cuda_bridge);
        snn_rt_sched_cleanup(core_state->rt_sched);
        snn_memory_cleanup(core_state->mem_mgr);
        snn_pcie_cleanup(core_state->pcie_mgr);
    }

    /* Cleanup device */
    if (core_state->device)
        device_destroy(core_state->class, core_state->devt);

    if (core_state->class)
        class_destroy(core_state->class);

    cdev_del(&core_state->cdev);
    unregister_chrdev_region(core_state->devt, 1);

    /* Free subsystem structures */
    kfree(core_state->pipeline);
    kfree(core_state->nvme_mgr);
    kfree(core_state->cuda_bridge);
    kfree(core_state->rt_sched);
    kfree(core_state->mem_mgr);
    kfree(core_state->pcie_mgr);
    kfree(core_state);

    snn_info("SNN Kernel shutdown complete\n");
}

module_init(snn_core_init);
module_exit(snn_core_exit);
