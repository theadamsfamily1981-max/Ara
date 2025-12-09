/*
 * A10PED Neuromorphic AI Tile Driver
 *
 * Linux PCIe kernel driver for BittWare A10PED dual Arria 10 FPGA accelerator.
 * Provides character device interface for AI tile v0 CSR access and DMA.
 *
 * Features:
 *   - Maps PCIe BAR0 for CSR register access
 *   - Maps PCIe BAR2 for DMA memory window
 *   - Provides ioctl interface for memcopy operations
 *   - Supports multiple FPGAs (dual-tile operation)
 *
 * Usage:
 *   insmod a10ped_driver.ko
 *   # Creates /dev/a10ped0 and /dev/a10ped1
 *
 * Author: A10PED Neuromorphic Project
 * License: BSD-3-Clause (compatible with GPL for kernel modules)
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/pci.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/uaccess.h>
#include <linux/dma-mapping.h>
#include <linux/interrupt.h>
#include <linux/delay.h>

#define DRV_NAME "a10ped"
#define DRV_VERSION "0.1.0"

// Intel Arria 10 PCIe IDs
#define PCI_VENDOR_ID_INTEL     0x8086
#define PCI_DEVICE_ID_A10_GX    0x09C4      // Arria 10 GX (verify for A10PED)

// BAR sizes
#define BAR0_SIZE               (1 * 1024 * 1024)   // 1MB for CSR
#define BAR2_SIZE               (256 * 1024 * 1024) // 256MB for DMA

// CSR register offsets (from ai_tile_registers.yaml)
#define AI_TILE_CTRL_OFFSET     0x00
#define AI_TILE_STATUS_OFFSET   0x04
#define AI_TILE_CMD_SRC_LO_OFFSET   0x08
#define AI_TILE_CMD_SRC_HI_OFFSET   0x0C
#define AI_TILE_CMD_DST_LO_OFFSET   0x10
#define AI_TILE_CMD_DST_HI_OFFSET   0x14
#define AI_TILE_CMD_LEN_OFFSET      0x18
#define AI_TILE_CMD_CFG_OFFSET      0x1C

// Control bits
#define AI_TILE_CTRL_START      (1 << 0)
#define AI_TILE_CTRL_RESET      (1 << 1)

// Status bits
#define AI_TILE_STATUS_BUSY     (1 << 0)
#define AI_TILE_STATUS_DONE     (1 << 1)
#define AI_TILE_STATUS_ERROR    (1 << 2)

// ioctl commands
#define A10PED_IOC_MAGIC        'A'
#define A10PED_IOCTL_MEMCOPY    _IOW(A10PED_IOC_MAGIC, 1, struct a10ped_memcopy_cmd)
#define A10PED_IOCTL_RESET      _IO(A10PED_IOC_MAGIC, 2)
#define A10PED_IOCTL_GET_STATUS _IOR(A10PED_IOC_MAGIC, 3, uint32_t)

// Memcopy command structure
struct a10ped_memcopy_cmd {
    uint64_t src_addr;      // Source address in FPGA memory
    uint64_t dst_addr;      // Destination address in FPGA memory
    uint32_t length;        // Transfer length in bytes
    uint32_t mode;          // Command mode (0x0 = memcopy)
};

// Device structure
struct a10ped_dev {
    struct pci_dev *pdev;
    struct cdev cdev;
    dev_t devno;

    void __iomem *bar0;     // CSR mapped memory
    void __iomem *bar2;     // DMA mapped memory

    resource_size_t bar0_phys;
    resource_size_t bar2_phys;
    resource_size_t bar0_len;
    resource_size_t bar2_len;

    int tile_id;            // 0 or 1 for dual-tile boards
    spinlock_t lock;
};

static struct class *a10ped_class;
static int major_num;
static struct a10ped_dev *devices[2] = {NULL, NULL};  // Support up to 2 FPGAs

/*
 * Read 32-bit CSR register
 */
static inline uint32_t csr_read32(struct a10ped_dev *dev, uint32_t offset)
{
    return ioread32(dev->bar0 + offset);
}

/*
 * Write 32-bit CSR register
 */
static inline void csr_write32(struct a10ped_dev *dev, uint32_t offset, uint32_t value)
{
    iowrite32(value, dev->bar0 + offset);
}

/*
 * Wait for tile to become idle
 * Returns 0 on success, -ETIMEDOUT on timeout
 */
static int wait_for_idle(struct a10ped_dev *dev, unsigned int timeout_ms)
{
    unsigned int elapsed = 0;
    uint32_t status;

    while (elapsed < timeout_ms) {
        status = csr_read32(dev, AI_TILE_STATUS_OFFSET);
        if (!(status & AI_TILE_STATUS_BUSY))
            return 0;

        msleep(10);
        elapsed += 10;
    }

    return -ETIMEDOUT;
}

/*
 * Execute memcopy command
 */
static int execute_memcopy(struct a10ped_dev *dev, struct a10ped_memcopy_cmd *cmd)
{
    int ret;
    uint32_t status;
    unsigned long flags;

    // Validate parameters
    if (cmd->length == 0 || cmd->length > (16 * 1024 * 1024)) {
        pr_err("%s: Invalid transfer length: %u\n", DRV_NAME, cmd->length);
        return -EINVAL;
    }

    if ((cmd->src_addr & 0x3F) || (cmd->dst_addr & 0x3F) || (cmd->length & 0x3F)) {
        pr_err("%s: Addresses/length must be 64-byte aligned\n", DRV_NAME);
        return -EINVAL;
    }

    spin_lock_irqsave(&dev->lock, flags);

    // Wait for any previous command to complete
    ret = wait_for_idle(dev, 5000);  // 5 second timeout
    if (ret) {
        pr_err("%s: Tile busy timeout\n", DRV_NAME);
        spin_unlock_irqrestore(&dev->lock, flags);
        return ret;
    }

    // Write command parameters
    csr_write32(dev, AI_TILE_CMD_SRC_LO_OFFSET, (uint32_t)(cmd->src_addr & 0xFFFFFFFF));
    csr_write32(dev, AI_TILE_CMD_SRC_HI_OFFSET, (uint32_t)(cmd->src_addr >> 32));
    csr_write32(dev, AI_TILE_CMD_DST_LO_OFFSET, (uint32_t)(cmd->dst_addr & 0xFFFFFFFF));
    csr_write32(dev, AI_TILE_CMD_DST_HI_OFFSET, (uint32_t)(cmd->dst_addr >> 32));
    csr_write32(dev, AI_TILE_CMD_LEN_OFFSET, cmd->length);
    csr_write32(dev, AI_TILE_CMD_CFG_OFFSET, cmd->mode);  // 0x0 = memcopy

    // Start command
    csr_write32(dev, AI_TILE_CTRL_OFFSET, AI_TILE_CTRL_START);

    spin_unlock_irqrestore(&dev->lock, flags);

    // Wait for completion (polling mode for now)
    ret = wait_for_idle(dev, 10000);  // 10 second timeout
    if (ret) {
        pr_err("%s: Command timeout\n", DRV_NAME);
        return ret;
    }

    // Check status
    status = csr_read32(dev, AI_TILE_STATUS_OFFSET);
    if (status & AI_TILE_STATUS_ERROR) {
        pr_err("%s: Command error, status=0x%08x\n", DRV_NAME, status);
        return -EIO;
    }

    if (status & AI_TILE_STATUS_DONE) {
        return 0;  // Success
    }

    return -EIO;  // Unexpected status
}

/*
 * File operations: open
 */
static int a10ped_open(struct inode *inode, struct file *filp)
{
    struct a10ped_dev *dev = container_of(inode->i_cdev, struct a10ped_dev, cdev);
    filp->private_data = dev;
    return 0;
}

/*
 * File operations: release
 */
static int a10ped_release(struct inode *inode, struct file *filp)
{
    return 0;
}

/*
 * File operations: ioctl
 */
static long a10ped_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
    struct a10ped_dev *dev = filp->private_data;
    struct a10ped_memcopy_cmd memcopy_cmd;
    uint32_t status;
    int ret;

    switch (cmd) {
    case A10PED_IOCTL_MEMCOPY:
        if (copy_from_user(&memcopy_cmd, (void __user *)arg, sizeof(memcopy_cmd)))
            return -EFAULT;

        ret = execute_memcopy(dev, &memcopy_cmd);
        return ret;

    case A10PED_IOCTL_RESET:
        csr_write32(dev, AI_TILE_CTRL_OFFSET, AI_TILE_CTRL_RESET);
        msleep(10);
        return 0;

    case A10PED_IOCTL_GET_STATUS:
        status = csr_read32(dev, AI_TILE_STATUS_OFFSET);
        if (copy_to_user((void __user *)arg, &status, sizeof(status)))
            return -EFAULT;
        return 0;

    default:
        return -ENOTTY;
    }
}

/*
 * File operations: mmap (for direct BAR access)
 */
static int a10ped_mmap(struct file *filp, struct vm_area_struct *vma)
{
    struct a10ped_dev *dev = filp->private_data;
    unsigned long size = vma->vm_end - vma->vm_start;
    unsigned long offset = vma->vm_pgoff << PAGE_SHIFT;

    // Only allow mapping BAR0 (CSR) for now
    if (offset == 0 && size <= dev->bar0_len) {
        return io_remap_pfn_range(vma, vma->vm_start,
                                   dev->bar0_phys >> PAGE_SHIFT,
                                   size, vma->vm_page_prot);
    }

    return -EINVAL;
}

static struct file_operations a10ped_fops = {
    .owner = THIS_MODULE,
    .open = a10ped_open,
    .release = a10ped_release,
    .unlocked_ioctl = a10ped_ioctl,
    .mmap = a10ped_mmap,
};

/*
 * PCI probe function
 */
static int a10ped_probe(struct pci_dev *pdev, const struct pci_device_id *id)
{
    struct a10ped_dev *dev;
    int ret, tile_id;

    pr_info("%s: Probing device %04x:%04x\n", DRV_NAME,
            pdev->vendor, pdev->device);

    // Allocate device structure
    dev = kzalloc(sizeof(*dev), GFP_KERNEL);
    if (!dev)
        return -ENOMEM;

    dev->pdev = pdev;
    spin_lock_init(&dev->lock);

    // Enable PCI device
    ret = pci_enable_device(pdev);
    if (ret) {
        pr_err("%s: Failed to enable PCI device\n", DRV_NAME);
        goto err_free_dev;
    }

    // Request memory regions
    ret = pci_request_regions(pdev, DRV_NAME);
    if (ret) {
        pr_err("%s: Failed to request PCI regions\n", DRV_NAME);
        goto err_disable_device;
    }

    // Map BAR0 (CSR)
    dev->bar0_phys = pci_resource_start(pdev, 0);
    dev->bar0_len = pci_resource_len(pdev, 0);
    dev->bar0 = pci_iomap(pdev, 0, dev->bar0_len);
    if (!dev->bar0) {
        pr_err("%s: Failed to map BAR0\n", DRV_NAME);
        ret = -ENOMEM;
        goto err_release_regions;
    }

    pr_info("%s: BAR0 mapped at 0x%llx (len=%llu)\n", DRV_NAME,
            (unsigned long long)dev->bar0_phys,
            (unsigned long long)dev->bar0_len);

    // Map BAR2 (DMA) if present
    if (pci_resource_len(pdev, 2) > 0) {
        dev->bar2_phys = pci_resource_start(pdev, 2);
        dev->bar2_len = pci_resource_len(pdev, 2);
        dev->bar2 = pci_iomap(pdev, 2, dev->bar2_len);
        if (!dev->bar2) {
            pr_warn("%s: Failed to map BAR2 (DMA disabled)\n", DRV_NAME);
        } else {
            pr_info("%s: BAR2 mapped at 0x%llx (len=%llu)\n", DRV_NAME,
                    (unsigned long long)dev->bar2_phys,
                    (unsigned long long)dev->bar2_len);
        }
    }

    // Set DMA mask
    ret = dma_set_mask_and_coherent(&pdev->dev, DMA_BIT_MASK(64));
    if (ret) {
        pr_warn("%s: Failed to set 64-bit DMA mask, trying 32-bit\n", DRV_NAME);
        ret = dma_set_mask_and_coherent(&pdev->dev, DMA_BIT_MASK(32));
        if (ret) {
            pr_err("%s: Failed to set DMA mask\n", DRV_NAME);
            goto err_unmap;
        }
    }

    // Assign tile ID (0 or 1 based on which slot is free)
    for (tile_id = 0; tile_id < 2; tile_id++) {
        if (devices[tile_id] == NULL) {
            dev->tile_id = tile_id;
            devices[tile_id] = dev;
            break;
        }
    }

    if (tile_id == 2) {
        pr_err("%s: Too many devices (max 2 supported)\n", DRV_NAME);
        ret = -ENODEV;
        goto err_unmap;
    }

    // Create character device
    dev->devno = MKDEV(major_num, tile_id);
    cdev_init(&dev->cdev, &a10ped_fops);
    dev->cdev.owner = THIS_MODULE;

    ret = cdev_add(&dev->cdev, dev->devno, 1);
    if (ret) {
        pr_err("%s: Failed to add character device\n", DRV_NAME);
        goto err_clear_slot;
    }

    // Create device node
    device_create(a10ped_class, &pdev->dev, dev->devno, NULL, "a10ped%d", tile_id);

    pci_set_drvdata(pdev, dev);

    pr_info("%s: Device a10ped%d registered successfully\n", DRV_NAME, tile_id);
    return 0;

err_clear_slot:
    devices[tile_id] = NULL;
err_unmap:
    if (dev->bar2)
        pci_iounmap(pdev, dev->bar2);
    pci_iounmap(pdev, dev->bar0);
err_release_regions:
    pci_release_regions(pdev);
err_disable_device:
    pci_disable_device(pdev);
err_free_dev:
    kfree(dev);
    return ret;
}

/*
 * PCI remove function
 */
static void a10ped_remove(struct pci_dev *pdev)
{
    struct a10ped_dev *dev = pci_get_drvdata(pdev);

    pr_info("%s: Removing device a10ped%d\n", DRV_NAME, dev->tile_id);

    device_destroy(a10ped_class, dev->devno);
    cdev_del(&dev->cdev);

    devices[dev->tile_id] = NULL;

    if (dev->bar2)
        pci_iounmap(pdev, dev->bar2);
    pci_iounmap(pdev, dev->bar0);

    pci_release_regions(pdev);
    pci_disable_device(pdev);

    kfree(dev);
}

static struct pci_device_id a10ped_id_table[] = {
    { PCI_DEVICE(PCI_VENDOR_ID_INTEL, PCI_DEVICE_ID_A10_GX) },
    { 0, }
};
MODULE_DEVICE_TABLE(pci, a10ped_id_table);

static struct pci_driver a10ped_driver = {
    .name = DRV_NAME,
    .id_table = a10ped_id_table,
    .probe = a10ped_probe,
    .remove = a10ped_remove,
};

/*
 * Module init
 */
static int __init a10ped_init(void)
{
    int ret;
    dev_t devno;

    pr_info("%s: Loading driver version %s\n", DRV_NAME, DRV_VERSION);

    // Allocate major number
    ret = alloc_chrdev_region(&devno, 0, 2, DRV_NAME);
    if (ret < 0) {
        pr_err("%s: Failed to allocate char device region\n", DRV_NAME);
        return ret;
    }
    major_num = MAJOR(devno);

    // Create device class
    a10ped_class = class_create(THIS_MODULE, DRV_NAME);
    if (IS_ERR(a10ped_class)) {
        pr_err("%s: Failed to create device class\n", DRV_NAME);
        unregister_chrdev_region(devno, 2);
        return PTR_ERR(a10ped_class);
    }

    // Register PCI driver
    ret = pci_register_driver(&a10ped_driver);
    if (ret) {
        pr_err("%s: Failed to register PCI driver\n", DRV_NAME);
        class_destroy(a10ped_class);
        unregister_chrdev_region(devno, 2);
        return ret;
    }

    pr_info("%s: Driver loaded successfully\n", DRV_NAME);
    return 0;
}

/*
 * Module exit
 */
static void __exit a10ped_exit(void)
{
    pr_info("%s: Unloading driver\n", DRV_NAME);

    pci_unregister_driver(&a10ped_driver);
    class_destroy(a10ped_class);
    unregister_chrdev_region(MKDEV(major_num, 0), 2);

    pr_info("%s: Driver unloaded\n", DRV_NAME);
}

module_init(a10ped_init);
module_exit(a10ped_exit);

MODULE_LICENSE("Dual BSD/GPL");
MODULE_AUTHOR("A10PED Neuromorphic Project");
MODULE_DESCRIPTION("Driver for BittWare A10PED neuromorphic AI tile");
MODULE_VERSION(DRV_VERSION);
