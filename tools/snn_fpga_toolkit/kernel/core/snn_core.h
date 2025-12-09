/*
 * SNN Kernel Core - Internal Header
 *
 * Internal structures and function prototypes for kernel subsystems
 */

#ifndef _SNN_CORE_H
#define _SNN_CORE_H

#include <linux/types.h>
#include <linux/list.h>
#include <linux/spinlock.h>
#include <linux/atomic.h>
#include <linux/completion.h>
#include <linux/pci.h>

#include "../../include/snn_kernel/snn_types.h"

/* Forward declarations */
struct snn_pcie_manager;
struct snn_memory_manager;
struct snn_rt_scheduler;
struct snn_cuda_bridge;
struct snn_nvme_manager;
struct snn_pipeline;
struct snn_ai_engine;

/*
 * PCIe Subsystem
 */

struct snn_pcie_device {
    struct pci_dev *pdev;
    snn_device_type_t type;
    void __iomem *bar0;
    u64 bar0_size;
    u32 device_id;
    bool p2p_capable;
    struct list_head list;
};

struct snn_pcie_manager {
    spinlock_t lock;
    struct list_head devices;
    u32 num_devices;

    /* P2P transfer state */
    struct completion *p2p_completions;
    u32 max_p2p_streams;
    atomic64_t p2p_transfers;
    atomic64_t p2p_bytes;
};

int snn_pcie_init(struct snn_pcie_manager *mgr, const snn_kernel_init_t *config);
void snn_pcie_cleanup(struct snn_pcie_manager *mgr);
int snn_pcie_get_device_info(struct snn_pcie_manager *mgr, snn_device_info_t *info);
int snn_pcie_p2p_transfer(struct snn_pcie_manager *mgr, snn_p2p_transfer_t *transfer);
void snn_pcie_get_stats(struct snn_pcie_manager *mgr, snn_perf_stats_t *stats);
void snn_pcie_reset_stats(struct snn_pcie_manager *mgr);

/*
 * Memory Management Subsystem
 */

struct snn_mem_region {
    u32 mem_id;
    u64 size;
    void *virtual_addr;
    dma_addr_t physical_addr;
    u32 flags;
    u32 device_mask;
    atomic_t refcount;
    struct list_head list;
};

struct snn_memory_manager {
    spinlock_t lock;
    struct list_head regions;
    u32 next_mem_id;

    /* Memory pool */
    void *pinned_pool;
    u64 pool_size;
    u64 pool_used;

    /* Statistics */
    atomic64_t allocated;
    atomic64_t peak_allocated;
};

int snn_memory_init(struct snn_memory_manager *mgr, const snn_kernel_init_t *config);
void snn_memory_cleanup(struct snn_memory_manager *mgr);
int snn_memory_alloc(struct snn_memory_manager *mgr, snn_mem_alloc_t *req);
int snn_memory_free(struct snn_memory_manager *mgr, u32 mem_id);
void snn_memory_get_stats(struct snn_memory_manager *mgr, snn_perf_stats_t *stats);
void snn_memory_reset_stats(struct snn_memory_manager *mgr);

/*
 * Real-Time Scheduler
 */

struct snn_rt_task {
    struct task_struct *task;
    snn_rt_sched_params_t params;
    u64 last_execution;
    u64 deadline_misses;
    struct list_head list;
};

struct snn_rt_scheduler {
    spinlock_t lock;
    struct list_head tasks;
    u32 num_tasks;

    /* Statistics */
    atomic64_t tasks_executed;
    atomic64_t deadline_misses;
};

int snn_rt_sched_init(struct snn_rt_scheduler *sched, const snn_kernel_init_t *config);
void snn_rt_sched_cleanup(struct snn_rt_scheduler *sched);
int snn_rt_sched_set_params(struct snn_rt_scheduler *sched,
                            struct task_struct *task,
                            const snn_rt_sched_params_t *params);
void snn_rt_sched_get_stats(struct snn_rt_scheduler *sched, snn_perf_stats_t *stats);
void snn_rt_sched_reset_stats(struct snn_rt_scheduler *sched);

/*
 * CUDA Bridge
 */

struct snn_cuda_bridge {
    void *cuda_context;
    u32 gpu_id;
    bool initialized;

    /* CUDA-specific state */
    void *cuda_module;
    void *cuda_streams[16];
    u32 num_streams;
};

int snn_cuda_bridge_init(struct snn_cuda_bridge *bridge, const snn_kernel_init_t *config);
void snn_cuda_bridge_cleanup(struct snn_cuda_bridge *bridge);

/*
 * NVMe Manager
 */

struct snn_nvme_request {
    u64 offset;
    u64 size;
    void *buffer;
    struct completion done;
    int status;
    struct list_head list;
};

struct snn_nvme_manager {
    spinlock_t lock;
    struct nvme_ctrl *ctrl;
    struct list_head pending_requests;
    u32 queue_depth;

    /* Statistics */
    atomic64_t reads;
    atomic64_t writes;
    atomic64_t bytes_transferred;
};

int snn_nvme_init(struct snn_nvme_manager *mgr, const snn_kernel_init_t *config);
void snn_nvme_cleanup(struct snn_nvme_manager *mgr);
int snn_nvme_submit_io(struct snn_nvme_manager *mgr, snn_nvme_io_t *io);
void snn_nvme_get_stats(struct snn_nvme_manager *mgr, snn_perf_stats_t *stats);
void snn_nvme_reset_stats(struct snn_nvme_manager *mgr);

/*
 * SNN Pipeline
 */

struct snn_pipeline {
    spinlock_t lock;
    bool initialized;

    /* Pipeline configuration */
    u32 use_gpu;
    u32 use_fpga;

    /* Computation resources */
    struct snn_pcie_manager *pcie_mgr;
    struct snn_memory_manager *mem_mgr;
    struct snn_cuda_bridge *cuda_bridge;

    /* AI engine */
    struct snn_ai_engine *ai_engine;
};

int snn_pipeline_init(struct snn_pipeline *pipeline, const snn_kernel_init_t *config);
void snn_pipeline_cleanup(struct snn_pipeline *pipeline);
int snn_pipeline_execute(struct snn_pipeline *pipeline, const snn_compute_params_t *params);

/*
 * Semantic AI Engine
 */
int snn_ai_init(struct snn_ai_engine **engine, const struct snn_ai_config *config);
void snn_ai_cleanup(struct snn_ai_engine *engine);

#endif /* _SNN_CORE_H */
