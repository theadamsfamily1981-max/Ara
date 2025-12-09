/*
 * SNN Kernel - Type Definitions
 *
 * Core type definitions for the SNN-optimized kernel system
 * supporting GPU-FPGA integration via PCIe 5.0
 */

#ifndef _SNN_TYPES_H
#define _SNN_TYPES_H

#include <linux/types.h>
#include <linux/ioctl.h>

/* Version information */
#define SNN_KERNEL_VERSION_MAJOR 1
#define SNN_KERNEL_VERSION_MINOR 0
#define SNN_KERNEL_VERSION_PATCH 0

/* Device types */
typedef enum {
    SNN_DEV_CPU = 0,
    SNN_DEV_GPU = 1,
    SNN_DEV_FPGA = 2,
    SNN_DEV_NVME = 3,
    SNN_DEV_MAX
} snn_device_type_t;

/* Memory flags */
#define SNN_MEM_CPU     (1 << 0)
#define SNN_MEM_GPU     (1 << 1)
#define SNN_MEM_FPGA    (1 << 2)
#define SNN_MEM_PINNED  (1 << 3)
#define SNN_MEM_DMA     (1 << 4)
#define SNN_MEM_ZERO    (1 << 5)

/* Transfer flags */
#define SNN_TRANSFER_ASYNC      (1 << 0)
#define SNN_TRANSFER_BLOCKING   (1 << 1)
#define SNN_TRANSFER_P2P        (1 << 2)
#define SNN_TRANSFER_DMA        (1 << 3)

/* Real-time priority levels */
#define SNN_RT_PRIO_MAX     99
#define SNN_RT_PRIO_HIGH    90
#define SNN_RT_PRIO_MEDIUM  50
#define SNN_RT_PRIO_LOW     10
#define SNN_RT_PRIO_MIN     1

/* Device capabilities */
typedef struct {
    __u32 pcie_gen;              /* PCIe generation (5 for PCIe 5.0) */
    __u32 pcie_lanes;            /* Number of PCIe lanes */
    __u64 max_bandwidth_mbps;    /* Max bandwidth in MB/s */
    __u64 memory_size;           /* Device memory size */
    __u32 supports_p2p;          /* P2P transfer support */
    __u32 supports_pinned_mem;   /* Pinned memory support */
    __u32 dma_channels;          /* Number of DMA channels */
    __u32 compute_units;         /* Number of compute units/SMs */
} snn_device_caps_t;

/* Device information */
typedef struct {
    __u32 device_id;
    snn_device_type_t type;
    char name[64];
    __u32 vendor_id;
    __u32 device_specific_id;
    snn_device_caps_t caps;
    __u32 online;                /* Device online status */
} snn_device_info_t;

/* Memory allocation request */
typedef struct {
    __u64 size;
    __u32 flags;                 /* SNN_MEM_* flags */
    __u32 alignment;
    __u32 device_mask;           /* Bitmask of devices that need access */
    __u64 physical_addr;         /* OUT: Physical address */
    __u64 virtual_addr;          /* OUT: Virtual address */
    __u32 mem_id;                /* OUT: Memory region ID */
} snn_mem_alloc_t;

/* P2P transfer descriptor */
typedef struct {
    snn_device_type_t src_dev;
    snn_device_type_t dst_dev;
    __u64 src_addr;
    __u64 dst_addr;
    __u64 size;
    __u32 flags;                 /* SNN_TRANSFER_* flags */
    __u32 stream_id;             /* For async transfers */
    __u64 completion_handle;     /* OUT: Handle for checking completion */
} snn_p2p_transfer_t;

/* Real-time scheduling parameters */
typedef struct {
    __u32 priority;              /* RT priority (1-99) */
    __u32 cpu_affinity;          /* CPU core affinity mask */
    __u64 deadline_ns;           /* Deadline in nanoseconds */
    __u64 period_ns;             /* Period for periodic tasks */
    __u32 preemptible;           /* Can be preempted */
} snn_rt_sched_params_t;

/* SNN computation parameters */
typedef struct {
    __u32 num_neurons;
    __u32 num_synapses;
    __u32 timesteps;
    __u32 batch_size;
    __u32 use_gpu;
    __u32 use_fpga;
    __u64 weight_matrix_addr;
    __u64 spike_train_addr;
    __u64 output_addr;
} snn_compute_params_t;

/* NVMe direct I/O request */
typedef struct {
    __u64 offset;                /* File/block offset */
    __u64 size;                  /* Transfer size */
    __u64 buffer_addr;           /* Buffer address (pinned memory) */
    __u32 flags;                 /* I/O flags */
    __u32 queue_id;              /* NVMe queue ID */
    __u64 completion_handle;     /* OUT: Completion handle */
} snn_nvme_io_t;

/* Performance statistics */
typedef struct {
    __u64 p2p_transfers;
    __u64 p2p_bytes_transferred;
    __u64 avg_p2p_latency_ns;
    __u64 pinned_mem_allocated;
    __u64 pinned_mem_peak;
    __u64 rt_tasks_executed;
    __u64 rt_deadline_misses;
    __u64 nvme_reads;
    __u64 nvme_writes;
    __u64 nvme_bytes_transferred;
} snn_perf_stats_t;

/* Initialization configuration */
typedef struct {
    __u32 gpu_id;
    __u32 fpga_id;
    __u64 pinned_mem_size;       /* Preallocated pinned memory pool */
    __u32 rt_priority;           /* Default RT priority */
    __u32 enable_monitoring;     /* Enable performance monitoring */
    __u32 enable_debugging;      /* Enable debug logging */
    __u32 max_p2p_streams;       /* Max concurrent P2P streams */
    __u32 nvme_queue_depth;      /* NVMe queue depth */
} snn_kernel_init_t;

/* IOCTL commands */
#define SNN_IOC_MAGIC 'S'

#define SNN_IOC_INIT            _IOW(SNN_IOC_MAGIC, 1, snn_kernel_init_t)
#define SNN_IOC_GET_DEVICE_INFO _IOWR(SNN_IOC_MAGIC, 2, snn_device_info_t)
#define SNN_IOC_ALLOC_MEM       _IOWR(SNN_IOC_MAGIC, 3, snn_mem_alloc_t)
#define SNN_IOC_FREE_MEM        _IOW(SNN_IOC_MAGIC, 4, __u32)
#define SNN_IOC_P2P_TRANSFER    _IOWR(SNN_IOC_MAGIC, 5, snn_p2p_transfer_t)
#define SNN_IOC_SET_RT_PARAMS   _IOW(SNN_IOC_MAGIC, 6, snn_rt_sched_params_t)
#define SNN_IOC_SNN_COMPUTE     _IOW(SNN_IOC_MAGIC, 7, snn_compute_params_t)
#define SNN_IOC_NVME_IO         _IOWR(SNN_IOC_MAGIC, 8, snn_nvme_io_t)
#define SNN_IOC_GET_STATS       _IOR(SNN_IOC_MAGIC, 9, snn_perf_stats_t)
#define SNN_IOC_RESET_STATS     _IO(SNN_IOC_MAGIC, 10)

/* Error codes */
#define SNN_SUCCESS             0
#define SNN_ERR_INVALID_PARAM   -1
#define SNN_ERR_NO_MEMORY       -2
#define SNN_ERR_DEVICE_BUSY     -3
#define SNN_ERR_NOT_SUPPORTED   -4
#define SNN_ERR_TIMEOUT         -5
#define SNN_ERR_DMA_FAILED      -6
#define SNN_ERR_RT_SCHED        -7
#define SNN_ERR_DEVICE_ERROR    -8

#endif /* _SNN_TYPES_H */
