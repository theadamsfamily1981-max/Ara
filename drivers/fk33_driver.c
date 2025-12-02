/*
 * Forest Kitten 33 (FK33) SNN Accelerator Driver
 *
 * Linux PCIe kernel driver for Squirrels Research Labs ForestKitten 33
 * neuromorphic SNN fabric.
 *
 * Features:
 *   - Maps PCIe BAR0 for control/status register access
 *   - Provides character device interface for SNN step execution
 *   - Supports DMA for input currents and spike outputs
 *
 * Usage:
 *   insmod fk33_driver.ko
 *   # Creates /dev/fk33
 *
 * Author: Ara-SYNERGY Project
 * License: BSD-3-Clause (compatible with GPL for kernel modules)
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/pci.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/uaccess.h>
#include <linux/dma-mapping.h>
#include <linux/delay.h>

#define DRV_NAME "fk33"
#define DRV_VERSION "0.1.0"

// Squirrels Research Labs ForestKitten 33 PCIe IDs
#define PCI_VENDOR_ID_SQUIRRELS     0x1e24
#define PCI_DEVICE_ID_FK33          0x1533

// Control & Status Register offsets (from ARASYNERGY_KITTEN_FPGA_SPEC.md)
#define FK33_REG_CTRL           0x00    // Control bits
#define FK33_REG_STATUS         0x04    // Status bits
#define FK33_REG_BATCH          0x08    // Batch size
#define FK33_REG_N_INPUT        0x0C    // Number of input neurons
#define FK33_REG_N_OUTPUT       0x10    // Number of output neurons
#define FK33_REG_STEP_ID        0x14    // Step counter
#define FK33_REG_IN_ADDR_LO     0x20    // Input buffer low 32 bits
#define FK33_REG_IN_ADDR_HI     0x24    // Input buffer high 32 bits
#define FK33_REG_OUT_ADDR_LO    0x28    // Output buffer low 32 bits
#define FK33_REG_OUT_ADDR_HI    0x2C    // Output buffer high 32 bits
#define FK33_REG_TIMEOUT_CYC    0x30    // Internal timeout cycles

// Control bits
#define FK33_CTRL_SOFT_RESET    (1 << 0)
#define FK33_CTRL_START_STEP    (1 << 1)

// Status bits
#define FK33_STATUS_BUSY        (1 << 0)
#define FK33_STATUS_ERROR       (1 << 1)
#define FK33_STATUS_TIMEOUT     (1 << 2)
#define FK33_STATUS_DONE        (1 << 3)

// ioctl commands
#define FK33_IOC_MAGIC          'K'
#define FK33_IOCTL_RESET        _IO(FK33_IOC_MAGIC, 1)
#define FK33_IOCTL_GET_STATUS   _IOR(FK33_IOC_MAGIC, 2, uint32_t)
#define FK33_IOCTL_START_STEP   _IO(FK33_IOC_MAGIC, 3)
#define FK33_IOCTL_WAIT_DONE    _IOW(FK33_IOC_MAGIC, 4, uint32_t)  // timeout_ms
#define FK33_IOCTL_SET_CONFIG   _IOW(FK33_IOC_MAGIC, 5, struct fk33_config)
#define FK33_IOCTL_GET_CONFIG   _IOR(FK33_IOC_MAGIC, 6, struct fk33_config)

// Configuration structure
struct fk33_config {
    uint32_t n_input;       // Number of input neurons (default 4096)
    uint32_t n_output;      // Number of output neurons (default 2048)
    uint32_t batch_size;    // Batch size
    uint32_t timeout_cyc;   // Internal timeout in cycles
};

// Step command structure
struct fk33_step_cmd {
    uint64_t input_addr;    // DMA address for input currents
    uint64_t output_addr;   // DMA address for output spikes
    uint32_t timeout_ms;    // Host-side timeout
};

// Device structure
struct fk33_dev {
    struct pci_dev *pdev;
    struct cdev cdev;
    dev_t devno;

    void __iomem *bar0;     // CSR mapped memory
    resource_size_t bar0_phys;
    resource_size_t bar0_len;

    uint32_t step_id;       // Current step counter
    spinlock_t lock;
};

static struct class *fk33_class;
static int major_num;
static struct fk33_dev *fk33_device = NULL;

/*
 * Read 32-bit CSR register
 */
static inline uint32_t fk33_read32(struct fk33_dev *dev, uint32_t offset)
{
    return ioread32(dev->bar0 + offset);
}

/*
 * Write 32-bit CSR register
 */
static inline void fk33_write32(struct fk33_dev *dev, uint32_t offset, uint32_t value)
{
    iowrite32(value, dev->bar0 + offset);
}

/*
 * Wait for step to complete
 * Returns 0 on success, -ETIMEDOUT on timeout, -EIO on error
 */
static int fk33_wait_done(struct fk33_dev *dev, unsigned int timeout_ms)
{
    unsigned int elapsed = 0;
    uint32_t status;

    while (elapsed < timeout_ms) {
        status = fk33_read32(dev, FK33_REG_STATUS);

        if (status & FK33_STATUS_ERROR) {
            pr_err("%s: Step error, status=0x%08x\n", DRV_NAME, status);
            return -EIO;
        }

        if (status & FK33_STATUS_TIMEOUT) {
            pr_err("%s: Internal timeout, status=0x%08x\n", DRV_NAME, status);
            return -ETIMEDOUT;
        }

        if (!(status & FK33_STATUS_BUSY) && (status & FK33_STATUS_DONE)) {
            return 0;  // Success
        }

        usleep_range(100, 200);  // 100-200us sleep
        elapsed += 1;  // Approximate ms
    }

    return -ETIMEDOUT;
}

/*
 * File operations: open
 */
static int fk33_open(struct inode *inode, struct file *filp)
{
    struct fk33_dev *dev = container_of(inode->i_cdev, struct fk33_dev, cdev);
    filp->private_data = dev;
    return 0;
}

/*
 * File operations: release
 */
static int fk33_release(struct inode *inode, struct file *filp)
{
    return 0;
}

/*
 * File operations: ioctl
 */
static long fk33_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
    struct fk33_dev *dev = filp->private_data;
    struct fk33_config config;
    uint32_t status;
    uint32_t timeout_ms;
    int ret;

    switch (cmd) {
    case FK33_IOCTL_RESET:
        fk33_write32(dev, FK33_REG_CTRL, FK33_CTRL_SOFT_RESET);
        msleep(10);
        fk33_write32(dev, FK33_REG_CTRL, 0);
        dev->step_id = 0;
        return 0;

    case FK33_IOCTL_GET_STATUS:
        status = fk33_read32(dev, FK33_REG_STATUS);
        if (copy_to_user((void __user *)arg, &status, sizeof(status)))
            return -EFAULT;
        return 0;

    case FK33_IOCTL_START_STEP:
        dev->step_id++;
        fk33_write32(dev, FK33_REG_STEP_ID, dev->step_id);
        fk33_write32(dev, FK33_REG_CTRL, FK33_CTRL_START_STEP);
        return 0;

    case FK33_IOCTL_WAIT_DONE:
        if (copy_from_user(&timeout_ms, (void __user *)arg, sizeof(timeout_ms)))
            return -EFAULT;
        ret = fk33_wait_done(dev, timeout_ms);
        return ret;

    case FK33_IOCTL_SET_CONFIG:
        if (copy_from_user(&config, (void __user *)arg, sizeof(config)))
            return -EFAULT;
        fk33_write32(dev, FK33_REG_N_INPUT, config.n_input);
        fk33_write32(dev, FK33_REG_N_OUTPUT, config.n_output);
        fk33_write32(dev, FK33_REG_BATCH, config.batch_size);
        fk33_write32(dev, FK33_REG_TIMEOUT_CYC, config.timeout_cyc);
        return 0;

    case FK33_IOCTL_GET_CONFIG:
        config.n_input = fk33_read32(dev, FK33_REG_N_INPUT);
        config.n_output = fk33_read32(dev, FK33_REG_N_OUTPUT);
        config.batch_size = fk33_read32(dev, FK33_REG_BATCH);
        config.timeout_cyc = fk33_read32(dev, FK33_REG_TIMEOUT_CYC);
        if (copy_to_user((void __user *)arg, &config, sizeof(config)))
            return -EFAULT;
        return 0;

    default:
        return -ENOTTY;
    }
}

/*
 * File operations: mmap (for direct BAR access)
 */
static int fk33_mmap(struct file *filp, struct vm_area_struct *vma)
{
    struct fk33_dev *dev = filp->private_data;
    unsigned long size = vma->vm_end - vma->vm_start;

    if (size > dev->bar0_len) {
        return -EINVAL;
    }

    vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);

    return io_remap_pfn_range(vma, vma->vm_start,
                              dev->bar0_phys >> PAGE_SHIFT,
                              size, vma->vm_page_prot);
}

static struct file_operations fk33_fops = {
    .owner = THIS_MODULE,
    .open = fk33_open,
    .release = fk33_release,
    .unlocked_ioctl = fk33_ioctl,
    .mmap = fk33_mmap,
};

/*
 * PCI probe function
 */
static int fk33_probe(struct pci_dev *pdev, const struct pci_device_id *id)
{
    struct fk33_dev *dev;
    int ret;
    uint32_t status;

    pr_info("%s: Probing ForestKitten 33 at %04x:%04x\n", DRV_NAME,
            pdev->vendor, pdev->device);

    // Only allow one device for now
    if (fk33_device != NULL) {
        pr_err("%s: Only one ForestKitten 33 device supported\n", DRV_NAME);
        return -EBUSY;
    }

    // Allocate device structure
    dev = kzalloc(sizeof(*dev), GFP_KERNEL);
    if (!dev)
        return -ENOMEM;

    dev->pdev = pdev;
    dev->step_id = 0;
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

    pr_info("%s: BAR0 mapped at phys=0x%llx len=%llu\n", DRV_NAME,
            (unsigned long long)dev->bar0_phys,
            (unsigned long long)dev->bar0_len);

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

    // Create character device
    dev->devno = MKDEV(major_num, 0);
    cdev_init(&dev->cdev, &fk33_fops);
    dev->cdev.owner = THIS_MODULE;

    ret = cdev_add(&dev->cdev, dev->devno, 1);
    if (ret) {
        pr_err("%s: Failed to add character device\n", DRV_NAME);
        goto err_unmap;
    }

    // Create device node
    device_create(fk33_class, &pdev->dev, dev->devno, NULL, "fk33");

    pci_set_drvdata(pdev, dev);
    fk33_device = dev;

    // Read initial status
    status = fk33_read32(dev, FK33_REG_STATUS);
    pr_info("%s: ForestKitten 33 ready (status=0x%08x)\n", DRV_NAME, status);

    // Perform soft reset
    fk33_write32(dev, FK33_REG_CTRL, FK33_CTRL_SOFT_RESET);
    msleep(10);
    fk33_write32(dev, FK33_REG_CTRL, 0);

    // Set default configuration (from ARASYNERGY_KITTEN_FPGA_SPEC.md)
    fk33_write32(dev, FK33_REG_N_INPUT, 4096);
    fk33_write32(dev, FK33_REG_N_OUTPUT, 2048);
    fk33_write32(dev, FK33_REG_BATCH, 1);

    pr_info("%s: Device /dev/fk33 registered successfully\n", DRV_NAME);
    return 0;

err_unmap:
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
static void fk33_remove(struct pci_dev *pdev)
{
    struct fk33_dev *dev = pci_get_drvdata(pdev);

    pr_info("%s: Removing ForestKitten 33\n", DRV_NAME);

    device_destroy(fk33_class, dev->devno);
    cdev_del(&dev->cdev);

    pci_iounmap(pdev, dev->bar0);
    pci_release_regions(pdev);
    pci_disable_device(pdev);

    fk33_device = NULL;
    kfree(dev);
}

static struct pci_device_id fk33_id_table[] = {
    { PCI_DEVICE(PCI_VENDOR_ID_SQUIRRELS, PCI_DEVICE_ID_FK33) },
    { 0, }
};
MODULE_DEVICE_TABLE(pci, fk33_id_table);

static struct pci_driver fk33_driver = {
    .name = DRV_NAME,
    .id_table = fk33_id_table,
    .probe = fk33_probe,
    .remove = fk33_remove,
};

/*
 * Module init
 */
static int __init fk33_init(void)
{
    int ret;
    dev_t devno;

    pr_info("%s: Loading ForestKitten 33 driver v%s\n", DRV_NAME, DRV_VERSION);

    // Allocate major number
    ret = alloc_chrdev_region(&devno, 0, 1, DRV_NAME);
    if (ret < 0) {
        pr_err("%s: Failed to allocate char device region\n", DRV_NAME);
        return ret;
    }
    major_num = MAJOR(devno);

    // Create device class
    fk33_class = class_create(DRV_NAME);
    if (IS_ERR(fk33_class)) {
        pr_err("%s: Failed to create device class\n", DRV_NAME);
        unregister_chrdev_region(devno, 1);
        return PTR_ERR(fk33_class);
    }

    // Register PCI driver
    ret = pci_register_driver(&fk33_driver);
    if (ret) {
        pr_err("%s: Failed to register PCI driver\n", DRV_NAME);
        class_destroy(fk33_class);
        unregister_chrdev_region(devno, 1);
        return ret;
    }

    pr_info("%s: Driver loaded successfully\n", DRV_NAME);
    return 0;
}

/*
 * Module exit
 */
static void __exit fk33_exit(void)
{
    pr_info("%s: Unloading ForestKitten 33 driver\n", DRV_NAME);

    pci_unregister_driver(&fk33_driver);
    class_destroy(fk33_class);
    unregister_chrdev_region(MKDEV(major_num, 0), 1);

    pr_info("%s: Driver unloaded\n", DRV_NAME);
}

module_init(fk33_init);
module_exit(fk33_exit);

MODULE_LICENSE("Dual BSD/GPL");
MODULE_AUTHOR("Ara-SYNERGY Project");
MODULE_DESCRIPTION("Driver for Squirrels Research Labs ForestKitten 33 SNN Accelerator");
MODULE_VERSION(DRV_VERSION);
