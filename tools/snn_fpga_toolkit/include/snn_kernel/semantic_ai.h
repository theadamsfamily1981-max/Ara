/*
 * SNN Semantic AI - Type Definitions
 *
 * AI-powered decision making for intelligent resource allocation,
 * workload characterization, and adaptive optimization
 */

#ifndef _SNN_SEMANTIC_AI_H
#define _SNN_SEMANTIC_AI_H

#include "snn_types.h"

/* Workload characteristics */
typedef enum {
    SNN_WORKLOAD_DENSE = 0,          /* Dense matrix operations */
    SNN_WORKLOAD_SPARSE = 1,         /* Sparse, event-driven */
    SNN_WORKLOAD_MIXED = 2,          /* Mixed computation */
    SNN_WORKLOAD_IO_BOUND = 3,       /* I/O intensive */
    SNN_WORKLOAD_COMPUTE_BOUND = 4,  /* Computation intensive */
    SNN_WORKLOAD_UNKNOWN = 5         /* Not yet characterized */
} snn_workload_type_t;

/* AI decision confidence */
typedef enum {
    SNN_CONFIDENCE_LOW = 0,
    SNN_CONFIDENCE_MEDIUM = 1,
    SNN_CONFIDENCE_HIGH = 2,
    SNN_CONFIDENCE_VERY_HIGH = 3
} snn_ai_confidence_t;

/* Resource allocation recommendation */
typedef struct {
    __u32 use_gpu;                   /* Recommended GPU usage (0-100%) */
    __u32 use_fpga;                  /* Recommended FPGA usage (0-100%) */
    __u32 use_cpu;                   /* Recommended CPU usage (0-100%) */
    __u32 gpu_neurons;               /* Neurons to assign to GPU */
    __u32 fpga_neurons;              /* Neurons to assign to FPGA */
    __u32 cpu_neurons;               /* Neurons to assign to CPU */
    __u32 batch_size;                /* Recommended batch size */
    __u32 memory_prefetch;           /* Enable memory prefetching */
    snn_ai_confidence_t confidence;  /* Decision confidence */
    __u64 expected_latency_ns;       /* Expected execution time */
} snn_ai_allocation_t;

/* Workload features for AI analysis */
typedef struct {
    __u32 num_neurons;
    __u32 num_synapses;
    __u32 timesteps;
    __u32 batch_size;
    float sparsity;                  /* Sparsity ratio (0.0-1.0) */
    float computation_intensity;     /* FLOPs per byte */
    __u64 data_size;                 /* Total data size */
    __u32 memory_bandwidth_req;      /* Required bandwidth (MB/s) */
    __u32 is_sequential;             /* Sequential or parallel */
} snn_workload_features_t;

/* System state for AI reasoning */
typedef struct {
    __u32 gpu_utilization;           /* GPU usage (0-100) */
    __u32 fpga_utilization;          /* FPGA usage (0-100) */
    __u32 cpu_utilization;           /* CPU usage (0-100) */
    __u64 gpu_memory_free;           /* Free GPU memory */
    __u64 fpga_memory_free;          /* Free FPGA memory */
    __u64 cpu_memory_free;           /* Free CPU memory */
    __u32 pcie_bandwidth_used;       /* PCIe bandwidth usage (0-100) */
    __u32 nvme_queue_depth_used;     /* NVMe queue usage (0-100) */
    __u32 rt_deadline_miss_rate;     /* Deadline miss rate (0-100) */
    __u64 temperature_gpu;           /* GPU temperature (mC) */
    __u64 temperature_fpga;          /* FPGA temperature (mC) */
    __u32 power_budget_remaining;    /* Remaining power budget (W) */
} snn_system_state_t;

/* AI learning feedback */
typedef struct {
    __u64 actual_latency_ns;         /* Actual execution time */
    __u64 expected_latency_ns;       /* Expected execution time */
    __u32 deadline_met;              /* Was deadline met? */
    __u32 resource_utilization;      /* Average resource usage (0-100) */
    __u32 p2p_efficiency;            /* P2P transfer efficiency (0-100) */
    float reward;                    /* Reinforcement learning reward */
} snn_ai_feedback_t;

/* AI model state */
typedef struct {
    __u64 total_decisions;           /* Total decisions made */
    __u64 successful_decisions;      /* Successful predictions */
    __u64 learning_iterations;       /* Training iterations */
    float average_reward;            /* Average RL reward */
    __u32 model_version;             /* Current model version */
    __u32 confidence_threshold;      /* Min confidence for autonomous action */
} snn_ai_stats_t;

/* Knowledge graph node types */
typedef enum {
    SNN_KG_NODE_DEVICE = 0,
    SNN_KG_NODE_WORKLOAD = 1,
    SNN_KG_NODE_PATTERN = 2,
    SNN_KG_NODE_OPTIMIZATION = 3
} snn_kg_node_type_t;

/* Knowledge graph relationship types */
typedef enum {
    SNN_KG_REL_PERFORMS_WELL = 0,
    SNN_KG_REL_PERFORMS_POORLY = 1,
    SNN_KG_REL_REQUIRES = 2,
    SNN_KG_REL_CONFLICTS_WITH = 3,
    SNN_KG_REL_SIMILAR_TO = 4
} snn_kg_rel_type_t;

/* AI control flags */
#define SNN_AI_ENABLE_LEARNING      (1 << 0)
#define SNN_AI_ENABLE_AUTONOMOUS    (1 << 1)
#define SNN_AI_ENABLE_ADAPTATION    (1 << 2)
#define SNN_AI_ENABLE_PREFETCH      (1 << 3)
#define SNN_AI_ENABLE_POWER_MGMT    (1 << 4)

/* AI configuration */
typedef struct {
    __u32 flags;                     /* AI control flags */
    __u32 learning_rate;             /* Learning rate (fixed-point: x/1000) */
    __u32 exploration_rate;          /* Epsilon for exploration (0-100) */
    __u32 history_size;              /* Number of past decisions to track */
    __u32 model_update_interval;     /* Update model every N decisions */
    __u32 confidence_threshold;      /* Min confidence for autonomous action */
} snn_ai_config_t;

/* IOCTL extensions for AI */
#define SNN_IOC_AI_CONFIG       _IOW(SNN_IOC_MAGIC, 20, snn_ai_config_t)
#define SNN_IOC_AI_RECOMMEND    _IOWR(SNN_IOC_MAGIC, 21, snn_ai_allocation_t)
#define SNN_IOC_AI_FEEDBACK     _IOW(SNN_IOC_MAGIC, 22, snn_ai_feedback_t)
#define SNN_IOC_AI_GET_STATS    _IOR(SNN_IOC_MAGIC, 23, snn_ai_stats_t)
#define SNN_IOC_AI_RESET_MODEL  _IO(SNN_IOC_MAGIC, 24)

#endif /* _SNN_SEMANTIC_AI_H */
