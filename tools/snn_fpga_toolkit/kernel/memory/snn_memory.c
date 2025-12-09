/*
 * SNN Memory Management
 *
 * Implements pinned memory allocation and management for GPU-FPGA access
 * Ensures deterministic, swap-free memory for real-time processing
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/mm.h>
#include <linux/dma-mapping.h>
#include <linux/vmalloc.h>
#include <linux/highmem.h>

#include "../core/snn_core.h"

/*
 * Allocate physically contiguous pinned memory
 * This memory is locked in RAM and will not be swapped out
 */
static void *alloc_pinned_memory(size_t size, dma_addr_t *phys_addr,
                                 struct device *dev)
{
    void *virt_addr;
    struct page **pages;
    unsigned long num_pages;
    unsigned long i;
    gfp_t gfp_flags = GFP_KERNEL | __GFP_NOWARN | __GFP_NORETRY;

    num_pages = (size + PAGE_SIZE - 1) >> PAGE_SHIFT;

    /* Allocate array of page pointers */
    pages = kvmalloc_array(num_pages, sizeof(struct page *), GFP_KERNEL);
    if (!pages)
        return NULL;

    /* Allocate and pin pages */
    for (i = 0; i < num_pages; i++) {
        pages[i] = alloc_page(gfp_flags);
        if (!pages[i]) {
            pr_err("SNN_MEM: Failed to allocate page %lu/%lu\n",
                   i, num_pages);
            goto fail_pages;
        }

        /* Lock page in memory (prevent swapping) */
        SetPageReserved(pages[i]);
        lock_page(pages[i]);
    }

    /* Map pages into contiguous virtual memory */
    virt_addr = vmap(pages, num_pages, VM_MAP, PAGE_KERNEL);
    if (!virt_addr) {
        pr_err("SNN_MEM: Failed to map pages\n");
        goto fail_pages;
    }

    /* Get physical address of first page */
    *phys_addr = page_to_phys(pages[0]);

    pr_debug("SNN_MEM: Allocated %zu bytes at virt=%p, phys=%llx\n",
             size, virt_addr, (u64)*phys_addr);

    kvfree(pages);
    return virt_addr;

fail_pages:
    for (i = 0; i < num_pages; i++) {
        if (pages[i]) {
            unlock_page(pages[i]);
            ClearPageReserved(pages[i]);
            __free_page(pages[i]);
        }
    }
    kvfree(pages);
    return NULL;
}

/*
 * Free pinned memory
 */
static void free_pinned_memory(void *virt_addr, size_t size)
{
    unsigned long addr = (unsigned long)virt_addr;
    unsigned long end = addr + size;
    struct page *page;

    if (!virt_addr)
        return;

    /* Unlock and free all pages */
    while (addr < end) {
        page = vmalloc_to_page((void *)addr);
        if (page) {
            unlock_page(page);
            ClearPageReserved(page);
            __free_page(page);
        }
        addr += PAGE_SIZE;
    }

    vunmap(virt_addr);
}

/*
 * Initialize memory management subsystem
 */
int snn_memory_init(struct snn_memory_manager *mgr,
                    const snn_kernel_init_t *config)
{
    if (!mgr)
        return -EINVAL;

    pr_info("SNN_MEM: Initializing memory manager\n");

    spin_lock_init(&mgr->lock);
    INIT_LIST_HEAD(&mgr->regions);
    mgr->next_mem_id = 1;
    mgr->pool_size = config->pinned_mem_size;
    mgr->pool_used = 0;

    atomic64_set(&mgr->allocated, 0);
    atomic64_set(&mgr->peak_allocated, 0);

    /* Preallocate pinned memory pool if requested */
    if (config->pinned_mem_size > 0) {
        dma_addr_t phys_addr;

        pr_info("SNN_MEM: Preallocating %llu bytes of pinned memory\n",
                config->pinned_mem_size);

        mgr->pinned_pool = alloc_pinned_memory(config->pinned_mem_size,
                                               &phys_addr, NULL);
        if (!mgr->pinned_pool) {
            pr_err("SNN_MEM: Failed to allocate pinned pool\n");
            /* Continue without pool - allocate on demand */
            mgr->pool_size = 0;
        } else {
            pr_info("SNN_MEM: Pinned pool allocated at %p\n",
                    mgr->pinned_pool);
        }
    }

    pr_info("SNN_MEM: Memory manager initialized\n");
    return 0;
}

/*
 * Cleanup memory manager
 */
void snn_memory_cleanup(struct snn_memory_manager *mgr)
{
    struct snn_mem_region *region, *tmp;

    if (!mgr)
        return;

    pr_info("SNN_MEM: Cleaning up memory manager\n");

    /* Free all allocated regions */
    spin_lock(&mgr->lock);
    list_for_each_entry_safe(region, tmp, &mgr->regions, list) {
        pr_warn("SNN_MEM: Freeing leaked region %u (%llu bytes)\n",
                region->mem_id, region->size);

        if (region->virtual_addr && !(region->flags & SNN_MEM_PINNED)) {
            free_pinned_memory(region->virtual_addr, region->size);
        }

        list_del(&region->list);
        kfree(region);
    }
    spin_unlock(&mgr->lock);

    /* Free pinned pool */
    if (mgr->pinned_pool) {
        free_pinned_memory(mgr->pinned_pool, mgr->pool_size);
        mgr->pinned_pool = NULL;
    }

    pr_info("SNN_MEM: Memory manager cleanup complete\n");
}

/*
 * Allocate pinned memory region
 */
int snn_memory_alloc(struct snn_memory_manager *mgr, snn_mem_alloc_t *req)
{
    struct snn_mem_region *region;
    dma_addr_t phys_addr = 0;
    void *virt_addr = NULL;
    u64 current_allocated;

    if (!mgr || !req)
        return -EINVAL;

    if (req->size == 0)
        return -EINVAL;

    /* Validate alignment */
    if (req->alignment == 0)
        req->alignment = PAGE_SIZE;

    if (!IS_ALIGNED(req->size, PAGE_SIZE))
        req->size = ALIGN(req->size, PAGE_SIZE);

    pr_debug("SNN_MEM: Allocating %llu bytes, flags=0x%x\n",
             req->size, req->flags);

    /* Allocate memory region structure */
    region = kzalloc(sizeof(*region), GFP_KERNEL);
    if (!region)
        return -ENOMEM;

    /* Allocate memory based on flags */
    if (req->flags & SNN_MEM_PINNED) {
        /* Try to allocate from pool first */
        spin_lock(&mgr->lock);
        if (mgr->pinned_pool && mgr->pool_used + req->size <= mgr->pool_size) {
            virt_addr = mgr->pinned_pool + mgr->pool_used;
            phys_addr = virt_to_phys(virt_addr);
            mgr->pool_used += req->size;
            spin_unlock(&mgr->lock);
        } else {
            spin_unlock(&mgr->lock);

            /* Allocate new pinned memory */
            virt_addr = alloc_pinned_memory(req->size, &phys_addr, NULL);
            if (!virt_addr) {
                pr_err("SNN_MEM: Failed to allocate pinned memory\n");
                kfree(region);
                return -ENOMEM;
            }
        }
    } else {
        /* Regular kernel memory allocation */
        virt_addr = vmalloc(req->size);
        if (!virt_addr) {
            kfree(region);
            return -ENOMEM;
        }
        phys_addr = virt_to_phys(virt_addr);
    }

    /* Initialize memory if requested */
    if (req->flags & SNN_MEM_ZERO)
        memset(virt_addr, 0, req->size);

    /* Fill region structure */
    spin_lock(&mgr->lock);
    region->mem_id = mgr->next_mem_id++;
    region->size = req->size;
    region->virtual_addr = virt_addr;
    region->physical_addr = phys_addr;
    region->flags = req->flags;
    region->device_mask = req->device_mask;
    atomic_set(&region->refcount, 1);

    list_add_tail(&region->list, &mgr->regions);
    spin_unlock(&mgr->lock);

    /* Update statistics */
    current_allocated = atomic64_add_return(req->size, &mgr->allocated);
    if (current_allocated > atomic64_read(&mgr->peak_allocated))
        atomic64_set(&mgr->peak_allocated, current_allocated);

    /* Fill request outputs */
    req->mem_id = region->mem_id;
    req->virtual_addr = (u64)virt_addr;
    req->physical_addr = phys_addr;

    pr_debug("SNN_MEM: Allocated region %u: virt=%llx, phys=%llx\n",
             region->mem_id, req->virtual_addr, req->physical_addr);

    return 0;
}

/*
 * Free memory region
 */
int snn_memory_free(struct snn_memory_manager *mgr, u32 mem_id)
{
    struct snn_mem_region *region, *tmp;
    bool found = false;

    if (!mgr)
        return -EINVAL;

    pr_debug("SNN_MEM: Freeing region %u\n", mem_id);

    spin_lock(&mgr->lock);
    list_for_each_entry_safe(region, tmp, &mgr->regions, list) {
        if (region->mem_id == mem_id) {
            /* Decrement refcount */
            if (atomic_dec_and_test(&region->refcount)) {
                /* Free memory if not from pool */
                bool from_pool = (region->virtual_addr >= mgr->pinned_pool &&
                                  region->virtual_addr < mgr->pinned_pool + mgr->pool_size);

                if (!from_pool && (region->flags & SNN_MEM_PINNED)) {
                    free_pinned_memory(region->virtual_addr, region->size);
                } else if (!from_pool) {
                    vfree(region->virtual_addr);
                }

                /* Update statistics */
                atomic64_sub(region->size, &mgr->allocated);

                list_del(&region->list);
                kfree(region);
            }

            found = true;
            break;
        }
    }
    spin_unlock(&mgr->lock);

    return found ? 0 : -ENOENT;
}

/*
 * Get memory statistics
 */
void snn_memory_get_stats(struct snn_memory_manager *mgr,
                          snn_perf_stats_t *stats)
{
    if (!mgr || !stats)
        return;

    stats->pinned_mem_allocated = atomic64_read(&mgr->allocated);
    stats->pinned_mem_peak = atomic64_read(&mgr->peak_allocated);
}

/*
 * Reset statistics
 */
void snn_memory_reset_stats(struct snn_memory_manager *mgr)
{
    if (!mgr)
        return;

    /* Don't reset current allocated, only peak */
    atomic64_set(&mgr->peak_allocated, atomic64_read(&mgr->allocated));
}

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("SNN Pinned Memory Management");
