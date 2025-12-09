/*
 * SNN NVMe Direct I/O Manager
 *
 * Provides high-speed direct I/O to NVMe storage for SNN training data
 * Bypasses page cache for maximum throughput and minimum latency
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/blkdev.h>
#include <linux/bio.h>
#include <linux/fs.h>

#include "../core/snn_core.h"

/*
 * NVMe request completion callback
 */
static void nvme_io_complete(struct bio *bio)
{
    struct snn_nvme_request *req = bio->bi_private;

    pr_debug("SNN_NVME: I/O complete, status=%d\n", bio->bi_status);

    /* Update status */
    req->status = blk_status_to_errno(bio->bi_status);

    /* Signal completion */
    complete(&req->done);

    /* Cleanup bio */
    bio_put(bio);
}

/*
 * Initialize NVMe manager
 */
int snn_nvme_init(struct snn_nvme_manager *mgr,
                  const snn_kernel_init_t *config)
{
    if (!mgr)
        return -EINVAL;

    pr_info("SNN_NVME: Initializing NVMe manager\n");

    spin_lock_init(&mgr->lock);
    INIT_LIST_HEAD(&mgr->pending_requests);
    mgr->ctrl = NULL;
    mgr->queue_depth = config->nvme_queue_depth;

    if (mgr->queue_depth == 0)
        mgr->queue_depth = 128; /* Default queue depth */

    atomic64_set(&mgr->reads, 0);
    atomic64_set(&mgr->writes, 0);
    atomic64_set(&mgr->bytes_transferred, 0);

    pr_info("SNN_NVME: NVMe manager initialized (queue_depth=%u)\n",
            mgr->queue_depth);

    return 0;
}

/*
 * Cleanup NVMe manager
 */
void snn_nvme_cleanup(struct snn_nvme_manager *mgr)
{
    struct snn_nvme_request *req, *tmp;

    if (!mgr)
        return;

    pr_info("SNN_NVME: Cleaning up NVMe manager\n");

    /* Wait for and cleanup pending requests */
    spin_lock(&mgr->lock);
    list_for_each_entry_safe(req, tmp, &mgr->pending_requests, list) {
        pr_warn("SNN_NVME: Canceling pending request\n");
        list_del(&req->list);
        kfree(req);
    }
    spin_unlock(&mgr->lock);

    pr_info("SNN_NVME: NVMe manager cleanup complete\n");
}

/*
 * Submit direct I/O operation
 */
static int submit_direct_io(struct snn_nvme_manager *mgr,
                           snn_nvme_io_t *io, int op)
{
    struct block_device *bdev;
    struct bio *bio;
    struct snn_nvme_request *req;
    sector_t sector;
    unsigned int nr_pages;
    int ret = 0;

    /*
     * Real implementation would:
     * 1. Get NVMe block device from file descriptor or path
     * 2. Validate buffer is in pinned memory
     * 3. Build bio with physical pages
     * 4. Submit to NVMe queue
     *
     * This is a simplified version showing the structure.
     */

    /* Allocate request structure */
    req = kzalloc(sizeof(*req), GFP_KERNEL);
    if (!req)
        return -ENOMEM;

    req->offset = io->offset;
    req->size = io->size;
    req->buffer = (void *)io->buffer_addr;
    req->status = 0;
    init_completion(&req->done);

    /* Calculate sector and number of pages */
    sector = io->offset >> 9; /* Convert to 512-byte sectors */
    nr_pages = (io->size + PAGE_SIZE - 1) >> PAGE_SHIFT;

    /*
     * In production, we would:
     * 1. Open NVMe device: bdev = blkdev_get_by_path(path, ...)
     * 2. Create bio for direct I/O
     * 3. Add pages from pinned memory to bio
     * 4. Submit bio to block layer
     *
     * For now, simulate the operation
     */

    pr_debug("SNN_NVME: Simulating %s: offset=%llu, size=%llu, pages=%u\n",
             op == REQ_OP_READ ? "read" : "write",
             io->offset, io->size, nr_pages);

    /* Track request */
    spin_lock(&mgr->lock);
    list_add_tail(&req->list, &mgr->pending_requests);
    spin_unlock(&mgr->lock);

    /* Simulate async I/O */
    io->completion_handle = (u64)req;

    /* Update statistics */
    if (op == REQ_OP_READ)
        atomic64_inc(&mgr->reads);
    else
        atomic64_inc(&mgr->writes);

    atomic64_add(io->size, &mgr->bytes_transferred);

    /* For synchronous operations, wait for completion */
    if (!(io->flags & SNN_TRANSFER_ASYNC)) {
        /* Simulate I/O delay */
        msleep(1);

        spin_lock(&mgr->lock);
        list_del(&req->list);
        spin_unlock(&mgr->lock);

        ret = req->status;
        kfree(req);
    }

    return ret;
}

/*
 * Submit NVMe read operation
 */
int snn_nvme_read(struct snn_nvme_manager *mgr, snn_nvme_io_t *io)
{
    if (!mgr || !io)
        return -EINVAL;

    pr_debug("SNN_NVME: Read request: offset=%llu, size=%llu\n",
             io->offset, io->size);

    return submit_direct_io(mgr, io, REQ_OP_READ);
}

/*
 * Submit NVMe write operation
 */
int snn_nvme_write(struct snn_nvme_manager *mgr, snn_nvme_io_t *io)
{
    if (!mgr || !io)
        return -EINVAL;

    pr_debug("SNN_NVME: Write request: offset=%llu, size=%llu\n",
             io->offset, io->size);

    return submit_direct_io(mgr, io, REQ_OP_WRITE);
}

/*
 * Submit generic NVMe I/O based on flags
 */
int snn_nvme_submit_io(struct snn_nvme_manager *mgr, snn_nvme_io_t *io)
{
    if (!mgr || !io)
        return -EINVAL;

    /* Determine operation from flags */
    if (io->flags & SNN_TRANSFER_ASYNC) {
        /* Async I/O */
        return submit_direct_io(mgr, io, REQ_OP_READ);
    } else {
        /* Sync I/O - default to read, caller should use read/write directly */
        return submit_direct_io(mgr, io, REQ_OP_READ);
    }
}

/*
 * Wait for I/O completion
 */
int snn_nvme_wait(struct snn_nvme_manager *mgr, u64 completion_handle,
                  u64 timeout_ns)
{
    struct snn_nvme_request *req = (struct snn_nvme_request *)completion_handle;
    unsigned long timeout_jiffies;
    int ret;

    if (!mgr || !req)
        return -EINVAL;

    /* Convert timeout to jiffies */
    if (timeout_ns == 0) {
        timeout_jiffies = MAX_SCHEDULE_TIMEOUT;
    } else {
        timeout_jiffies = nsecs_to_jiffies(timeout_ns);
    }

    /* Wait for completion */
    ret = wait_for_completion_timeout(&req->done, timeout_jiffies);
    if (ret == 0) {
        pr_err("SNN_NVME: I/O timeout\n");
        return -ETIMEDOUT;
    }

    /* Get status and cleanup */
    ret = req->status;

    spin_lock(&mgr->lock);
    list_del(&req->list);
    spin_unlock(&mgr->lock);

    kfree(req);

    return ret;
}

/*
 * Get statistics
 */
void snn_nvme_get_stats(struct snn_nvme_manager *mgr, snn_perf_stats_t *stats)
{
    if (!mgr || !stats)
        return;

    stats->nvme_reads = atomic64_read(&mgr->reads);
    stats->nvme_writes = atomic64_read(&mgr->writes);
    stats->nvme_bytes_transferred = atomic64_read(&mgr->bytes_transferred);
}

/*
 * Reset statistics
 */
void snn_nvme_reset_stats(struct snn_nvme_manager *mgr)
{
    if (!mgr)
        return;

    atomic64_set(&mgr->reads, 0);
    atomic64_set(&mgr->writes, 0);
    atomic64_set(&mgr->bytes_transferred, 0);
}

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("SNN NVMe Direct I/O Manager");
