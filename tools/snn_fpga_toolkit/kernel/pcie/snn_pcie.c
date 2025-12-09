/*
 * SNN PCIe 5.0 Driver
 *
 * Manages PCIe 5.0 devices and implements peer-to-peer data transfers
 * between GPU and FPGA for high-bandwidth, low-latency communication
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/pci.h>
#include <linux/dma-mapping.h>
#include <linux/delay.h>
#include <linux/interrupt.h>

#include "../core/snn_core.h"

#define PCI_VENDOR_ID_NVIDIA    0x10DE
#define PCI_VENDOR_ID_XILINX    0x10EE
#define PCI_VENDOR_ID_INTEL_FPGA 0x8086

/* PCIe capabilities */
#define SNN_PCIE_CAP_P2P        (1 << 0)
#define SNN_PCIE_CAP_DMA        (1 << 1)
#define SNN_PCIE_CAP_ATS        (1 << 2)

static const struct pci_device_id snn_pci_ids[] = {
    /* NVIDIA GPUs - Generic match for demonstration */
    { PCI_DEVICE(PCI_VENDOR_ID_NVIDIA, PCI_ANY_ID) },
    /* Xilinx FPGAs - Alveo series */
    { PCI_DEVICE(PCI_VENDOR_ID_XILINX, PCI_ANY_ID) },
    /* Intel FPGAs - Stratix/Agilex */
    { PCI_DEVICE(PCI_VENDOR_ID_INTEL_FPGA, PCI_ANY_ID) },
    { 0, }
};
MODULE_DEVICE_TABLE(pci, snn_pci_ids);

/*
 * Detect device type based on vendor and device ID
 */
static snn_device_type_t detect_device_type(struct pci_dev *pdev)
{
    if (pdev->vendor == PCI_VENDOR_ID_NVIDIA) {
        /* NVIDIA GPU */
        return SNN_DEV_GPU;
    } else if (pdev->vendor == PCI_VENDOR_ID_XILINX ||
               (pdev->vendor == PCI_VENDOR_ID_INTEL_FPGA &&
                pdev->class == PCI_CLASS_ACCELERATOR_PROCESSING << 8)) {
        /* FPGA accelerator */
        return SNN_DEV_FPGA;
    }

    return SNN_DEV_CPU; /* Unknown */
}

/*
 * Check if device supports peer-to-peer DMA
 */
static bool check_p2p_capability(struct pci_dev *pdev)
{
    u16 pcie_flags;
    int pos;

    /* Check for PCIe capability */
    pos = pci_find_capability(pdev, PCI_CAP_ID_EXP);
    if (!pos)
        return false;

    /* Check PCIe device capabilities */
    pci_read_config_word(pdev, pos + PCI_EXP_DEVCTL, &pcie_flags);

    /* Check for relaxed ordering and no-snoop (required for P2P) */
    if (!(pcie_flags & PCI_EXP_DEVCTL_RELAX_EN))
        pr_info("SNN: Device %s: Relaxed ordering not enabled\n",
                pci_name(pdev));

    /* Check if device is behind IOMMU (may affect P2P) */
    if (pdev->dev.iommu) {
        pr_info("SNN: Device %s: Behind IOMMU, P2P may be restricted\n",
                pci_name(pdev));
    }

    /* Basic P2P capability check - device has BAR memory */
    return pci_resource_len(pdev, 0) > 0;
}

/*
 * Initialize PCIe device
 */
static int snn_pci_probe(struct pci_dev *pdev, const struct pci_device_id *id)
{
    struct snn_pcie_device *snn_dev;
    int ret;
    snn_device_type_t dev_type;

    dev_type = detect_device_type(pdev);
    pr_info("SNN: Probing device %s (type=%d)\n", pci_name(pdev), dev_type);

    /* Enable device */
    ret = pci_enable_device(pdev);
    if (ret) {
        dev_err(&pdev->dev, "Failed to enable device: %d\n", ret);
        return ret;
    }

    /* Set DMA mask for 64-bit addressing */
    ret = dma_set_mask_and_coherent(&pdev->dev, DMA_BIT_MASK(64));
    if (ret) {
        dev_warn(&pdev->dev, "Failed to set 64-bit DMA mask, trying 32-bit\n");
        ret = dma_set_mask_and_coherent(&pdev->dev, DMA_BIT_MASK(32));
        if (ret) {
            dev_err(&pdev->dev, "Failed to set DMA mask\n");
            goto fail_disable;
        }
    }

    /* Request memory regions */
    ret = pci_request_regions(pdev, "snn");
    if (ret) {
        dev_err(&pdev->dev, "Failed to request regions: %d\n", ret);
        goto fail_disable;
    }

    /* Enable bus mastering for DMA */
    pci_set_master(pdev);

    /* Allocate device structure */
    snn_dev = kzalloc(sizeof(*snn_dev), GFP_KERNEL);
    if (!snn_dev) {
        ret = -ENOMEM;
        goto fail_release;
    }

    snn_dev->pdev = pdev;
    snn_dev->type = dev_type;
    snn_dev->device_id = 0; /* Will be assigned by manager */

    /* Map BAR0 */
    snn_dev->bar0_size = pci_resource_len(pdev, 0);
    if (snn_dev->bar0_size > 0) {
        snn_dev->bar0 = pci_iomap(pdev, 0, snn_dev->bar0_size);
        if (!snn_dev->bar0) {
            dev_err(&pdev->dev, "Failed to map BAR0\n");
            ret = -ENOMEM;
            goto fail_free;
        }
        dev_info(&pdev->dev, "Mapped BAR0: size=%llu\n", snn_dev->bar0_size);
    }

    /* Check P2P capability */
    snn_dev->p2p_capable = check_p2p_capability(pdev);
    dev_info(&pdev->dev, "P2P capable: %d\n", snn_dev->p2p_capable);

    /* Store device in PCI driver data */
    pci_set_drvdata(pdev, snn_dev);

    pr_info("SNN: Device %s initialized successfully\n", pci_name(pdev));
    return 0;

fail_free:
    kfree(snn_dev);
fail_release:
    pci_release_regions(pdev);
fail_disable:
    pci_disable_device(pdev);
    return ret;
}

/*
 * Cleanup PCIe device
 */
static void snn_pci_remove(struct pci_dev *pdev)
{
    struct snn_pcie_device *snn_dev = pci_get_drvdata(pdev);

    pr_info("SNN: Removing device %s\n", pci_name(pdev));

    if (!snn_dev)
        return;

    if (snn_dev->bar0)
        pci_iounmap(pdev, snn_dev->bar0);

    kfree(snn_dev);

    pci_release_regions(pdev);
    pci_disable_device(pdev);
}

static struct pci_driver snn_pci_driver = {
    .name = "snn_pcie",
    .id_table = snn_pci_ids,
    .probe = snn_pci_probe,
    .remove = snn_pci_remove,
};

/*
 * Initialize PCIe subsystem
 */
int snn_pcie_init(struct snn_pcie_manager *mgr, const snn_kernel_init_t *config)
{
    int ret;

    if (!mgr)
        return -EINVAL;

    pr_info("SNN: Initializing PCIe subsystem\n");

    spin_lock_init(&mgr->lock);
    INIT_LIST_HEAD(&mgr->devices);
    mgr->num_devices = 0;
    mgr->max_p2p_streams = config->max_p2p_streams;

    /* Allocate P2P completion structures */
    mgr->p2p_completions = kcalloc(mgr->max_p2p_streams,
                                   sizeof(struct completion),
                                   GFP_KERNEL);
    if (!mgr->p2p_completions)
        return -ENOMEM;

    /* Initialize completions */
    for (u32 i = 0; i < mgr->max_p2p_streams; i++)
        init_completion(&mgr->p2p_completions[i]);

    atomic64_set(&mgr->p2p_transfers, 0);
    atomic64_set(&mgr->p2p_bytes, 0);

    /* Register PCI driver */
    ret = pci_register_driver(&snn_pci_driver);
    if (ret) {
        pr_err("SNN: Failed to register PCI driver: %d\n", ret);
        kfree(mgr->p2p_completions);
        return ret;
    }

    pr_info("SNN: PCIe subsystem initialized\n");
    return 0;
}

/*
 * Cleanup PCIe subsystem
 */
void snn_pcie_cleanup(struct snn_pcie_manager *mgr)
{
    struct snn_pcie_device *dev, *tmp;

    if (!mgr)
        return;

    pr_info("SNN: Cleaning up PCIe subsystem\n");

    pci_unregister_driver(&snn_pci_driver);

    spin_lock(&mgr->lock);
    list_for_each_entry_safe(dev, tmp, &mgr->devices, list) {
        list_del(&dev->list);
        kfree(dev);
    }
    spin_unlock(&mgr->lock);

    kfree(mgr->p2p_completions);
}

/*
 * Get device information
 */
int snn_pcie_get_device_info(struct snn_pcie_manager *mgr,
                             snn_device_info_t *info)
{
    struct snn_pcie_device *dev;
    bool found = false;

    if (!mgr || !info)
        return -EINVAL;

    spin_lock(&mgr->lock);
    list_for_each_entry(dev, &mgr->devices, list) {
        if (dev->type == info->type && dev->device_id == info->device_id) {
            struct pci_dev *pdev = dev->pdev;
            int pos;

            /* Fill device info */
            snprintf(info->name, sizeof(info->name), "%s", pci_name(pdev));
            info->vendor_id = pdev->vendor;
            info->device_specific_id = pdev->device;
            info->online = 1;

            /* Get PCIe capabilities */
            pos = pci_find_capability(pdev, PCI_CAP_ID_EXP);
            if (pos) {
                u16 lnksta;
                pci_read_config_word(pdev, pos + PCI_EXP_LNKSTA, &lnksta);

                info->caps.pcie_gen = (lnksta & PCI_EXP_LNKSTA_CLS) >> 0;
                info->caps.pcie_lanes = (lnksta & PCI_EXP_LNKSTA_NLW) >> 4;

                /* Calculate bandwidth: Gen5 = 32 GT/s * lanes * encoding efficiency (128/130) */
                if (info->caps.pcie_gen == 5) {
                    info->caps.max_bandwidth_mbps =
                        (u64)info->caps.pcie_lanes * 32 * 1000 / 8 * 128 / 130;
                } else if (info->caps.pcie_gen == 4) {
                    info->caps.max_bandwidth_mbps =
                        (u64)info->caps.pcie_lanes * 16 * 1000 / 8 * 128 / 130;
                }
            }

            info->caps.memory_size = dev->bar0_size;
            info->caps.supports_p2p = dev->p2p_capable;
            info->caps.supports_pinned_mem = 1;
            info->caps.dma_channels = 1; /* Simplified */

            found = true;
            break;
        }
    }
    spin_unlock(&mgr->lock);

    return found ? 0 : -ENODEV;
}

/*
 * Perform P2P transfer between devices
 */
int snn_pcie_p2p_transfer(struct snn_pcie_manager *mgr,
                          snn_p2p_transfer_t *transfer)
{
    struct snn_pcie_device *src_dev = NULL, *dst_dev = NULL;
    struct snn_pcie_device *dev;
    int ret = 0;

    if (!mgr || !transfer)
        return -EINVAL;

    pr_debug("SNN: P2P transfer: %d->%d, %llu bytes\n",
             transfer->src_dev, transfer->dst_dev, transfer->size);

    /* Find source and destination devices */
    spin_lock(&mgr->lock);
    list_for_each_entry(dev, &mgr->devices, list) {
        if (dev->type == transfer->src_dev)
            src_dev = dev;
        if (dev->type == transfer->dst_dev)
            dst_dev = dev;
    }
    spin_unlock(&mgr->lock);

    if (!src_dev || !dst_dev) {
        pr_err("SNN: P2P: Device not found\n");
        return -ENODEV;
    }

    if (!src_dev->p2p_capable || !dst_dev->p2p_capable) {
        pr_err("SNN: P2P: Devices not P2P capable\n");
        return -EOPNOTSUPP;
    }

    /*
     * NOTE: Real P2P transfer implementation would use:
     * 1. NVIDIA GPUDirect RDMA for GPU P2P
     * 2. PCIe peer-to-peer BAR mapping
     * 3. DMA engines on FPGA
     * 4. NTB (Non-Transparent Bridge) for complex topologies
     *
     * This is a simplified version showing the structure.
     * Production code would require vendor-specific APIs.
     */

    /* For demonstration, we simulate the transfer */
    if (transfer->flags & SNN_TRANSFER_ASYNC) {
        /* Async transfer - return immediately with completion handle */
        u32 stream_id = transfer->stream_id % mgr->max_p2p_streams;
        transfer->completion_handle = (u64)stream_id;

        /* In real implementation, initiate DMA here */
        complete(&mgr->p2p_completions[stream_id]);
    } else {
        /* Synchronous transfer - wait for completion */
        /* In real implementation, perform blocking DMA */
        udelay(10); /* Simulate transfer time */
    }

    /* Update statistics */
    atomic64_inc(&mgr->p2p_transfers);
    atomic64_add(transfer->size, &mgr->p2p_bytes);

    return ret;
}

/*
 * Get statistics
 */
void snn_pcie_get_stats(struct snn_pcie_manager *mgr, snn_perf_stats_t *stats)
{
    if (!mgr || !stats)
        return;

    stats->p2p_transfers = atomic64_read(&mgr->p2p_transfers);
    stats->p2p_bytes_transferred = atomic64_read(&mgr->p2p_bytes);

    /* Calculate average latency (simplified) */
    if (stats->p2p_transfers > 0)
        stats->avg_p2p_latency_ns = 1000; /* Placeholder */
}

/*
 * Reset statistics
 */
void snn_pcie_reset_stats(struct snn_pcie_manager *mgr)
{
    if (!mgr)
        return;

    atomic64_set(&mgr->p2p_transfers, 0);
    atomic64_set(&mgr->p2p_bytes, 0);
}

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("SNN PCIe 5.0 Driver for GPU-FPGA P2P Communication");
