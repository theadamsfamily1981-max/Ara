/*
 * SNN Semantic AI - Internal Structures
 *
 * Internal data structures for AI engine
 */

#ifndef _SNN_AI_INTERNAL_H
#define _SNN_AI_INTERNAL_H

#include "../../include/snn_kernel/semantic_ai.h"

/* Q-learning configuration */
#define SNN_AI_STATE_SPACE_SIZE   512    /* Number of discrete states */
#define SNN_AI_ACTION_SPACE_SIZE  128    /* Number of possible actions */
#define SNN_AI_HISTORY_SIZE       1000   /* Workload history size */

/* Q-table for reinforcement learning */
struct snn_q_table {
    s64 values[SNN_AI_STATE_SPACE_SIZE][SNN_AI_ACTION_SPACE_SIZE];
};

/* Workload history entry */
struct snn_workload_entry {
    snn_workload_features_t features;
    snn_ai_allocation_t allocation;
    snn_ai_feedback_t feedback;
    u64 timestamp;
};

/* Workload history */
struct snn_workload_history {
    struct snn_workload_entry entries[SNN_AI_HISTORY_SIZE];
};

/* Knowledge graph node */
struct snn_kg_node {
    u32 id;
    snn_kg_node_type_t type;
    char name[64];
    void *data;  /* Type-specific data */
    struct list_head edges;  /* List of edges */
    struct list_head list;
};

/* Knowledge graph edge */
struct snn_kg_edge {
    snn_kg_rel_type_t type;
    struct snn_kg_node *from;
    struct snn_kg_node *to;
    float weight;  /* Relationship strength (0.0-1.0) */
    struct list_head list;
};

/* Knowledge graph */
struct snn_knowledge_graph {
    spinlock_t lock;
    struct list_head nodes;
    u32 node_count;
    u32 edge_count;
};

/* Forward declarations */
struct snn_ai_engine;

/* AI engine functions */
int snn_ai_engine_init(struct snn_ai_engine **engine,
                       const snn_ai_config_t *config);
void snn_ai_engine_cleanup(struct snn_ai_engine *engine);

int snn_ai_recommend(struct snn_ai_engine *engine,
                    const snn_compute_params_t *params,
                    const snn_system_state_t *sys_state,
                    snn_ai_allocation_t *allocation);

int snn_ai_feedback(struct snn_ai_engine *engine,
                   const snn_compute_params_t *params,
                   const snn_system_state_t *sys_state,
                   const snn_ai_feedback_t *feedback);

void snn_ai_get_stats(struct snn_ai_engine *engine, snn_ai_stats_t *stats);

/* Knowledge graph functions */
int snn_kg_init(struct snn_knowledge_graph **kg);
void snn_kg_cleanup(struct snn_knowledge_graph *kg);

struct snn_kg_node *snn_kg_add_node(struct snn_knowledge_graph *kg,
                                    snn_kg_node_type_t type,
                                    const char *name,
                                    void *data);

int snn_kg_add_edge(struct snn_knowledge_graph *kg,
                    struct snn_kg_node *from,
                    struct snn_kg_node *to,
                    snn_kg_rel_type_t type,
                    float weight);

int snn_kg_query(struct snn_knowledge_graph *kg,
                struct snn_kg_node *node,
                snn_kg_rel_type_t rel_type,
                struct list_head *results);

#endif /* _SNN_AI_INTERNAL_H */
