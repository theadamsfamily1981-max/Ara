/*
 * CSR++ Dynamic Graph Structure
 *
 * Compressed Sparse Row (CSR++) format optimized for:
 * - Concurrent read/write access (RCU-based)
 * - Cache-efficient graph traversal
 * - Dynamic edge insertion/deletion
 * - Fast neighbor lookups
 *
 * CSR++ extends traditional CSR with:
 * - In-place edge updates without full reconstruction
 * - Read-Copy-Update (RCU) for lock-free reads
 * - Tombstone-based deletion
 * - Periodic compaction
 */

#ifndef _SNN_CSR_GRAPH_H
#define _SNN_CSR_GRAPH_H

#include <linux/types.h>
#include <linux/spinlock.h>
#include <linux/rcupdate.h>
#include "snn_fixed_point.h"

/*
 * Graph dimensions
 */
#define SNN_GRAPH_MAX_NODES    1024   /* Max nodes in graph */
#define SNN_GRAPH_MAX_EDGES    8192   /* Max edges in graph */
#define SNN_GRAPH_EDGE_TYPES   16     /* Max edge types */

/*
 * Edge representation in CSR format
 *
 * Memory layout optimized for cache efficiency:
 * - 16 bytes total (fits 4 edges per cache line)
 * - Aligned for SIMD operations
 */
struct snn_csr_edge {
	u32 dst_node;        /* Destination node ID */
	u16 edge_type;       /* Edge type/relation */
	u16 flags;           /* Flags (TOMBSTONE, etc.) */
	fp_t weight;         /* Edge weight (fixed-point) */
} __attribute__((aligned(16)));

#define SNN_EDGE_FLAG_TOMBSTONE  (1 << 0)  /* Deleted edge */
#define SNN_EDGE_FLAG_TEMPORARY  (1 << 1)  /* Temporary edge */

/*
 * CSR++ Graph Structure
 *
 * Memory layout:
 * - row_ptr: Offset array (node_count + 1 entries)
 * - edges: Flat edge array (edge_count entries)
 *
 * Example for 4 nodes with edges:
 * 0 -> 1, 2
 * 1 -> 3
 * 2 -> (none)
 * 3 -> 0, 1, 2
 *
 * row_ptr: [0, 2, 3, 3, 6]
 * edges:   [(1,w1), (2,w2), (3,w3), (0,w4), (1,w5), (2,w6)]
 *
 * Node 0 edges: edges[row_ptr[0]..row_ptr[1]) = edges[0..2)
 * Node 1 edges: edges[row_ptr[1]..row_ptr[2]) = edges[2..3)
 * Node 2 edges: edges[row_ptr[2]..row_ptr[3]) = edges[3..3) (empty)
 * Node 3 edges: edges[row_ptr[3]..row_ptr[4]) = edges[3..6)
 */
struct snn_csr_graph {
	/* Graph topology (CSR format) */
	u32 *row_ptr;                    /* Row pointer array [node_count + 1] */
	struct snn_csr_edge *edges;      /* Edge array [edge_count] */

	/* Graph metadata */
	u32 node_count;                  /* Number of nodes */
	u32 edge_count;                  /* Number of edges (including tombstones) */
	u32 live_edge_count;             /* Number of live edges */
	u32 capacity;                    /* Edge array capacity */

	/* Node features (for GNN) */
	fp_t *node_features;             /* Node feature matrix [node_count × feature_dim] */
	u32 feature_dim;                 /* Feature dimension per node */

	/* Edge type weights (for typed graphs) */
	fp_t edge_type_weights[SNN_GRAPH_EDGE_TYPES];

	/* Concurrency control */
	spinlock_t write_lock;           /* Protects writes */
	struct rcu_head rcu;             /* RCU callback for safe reclaim */

	/* Statistics */
	atomic64_t traversals;           /* Number of graph traversals */
	atomic64_t updates;              /* Number of updates */
	u32 compaction_count;            /* Number of compactions */
	u32 tombstone_count;             /* Number of tombstones */
} __attribute__((aligned(64)));

/*
 * Graph iterator for traversal
 */
struct snn_graph_iterator {
	struct snn_csr_graph *graph;
	u32 node_id;
	u32 edge_idx;
	u32 end_idx;
};

/*
 * Initialize CSR++ graph
 */
int snn_csr_graph_init(struct snn_csr_graph **graph,
                       u32 max_nodes,
                       u32 max_edges,
                       u32 feature_dim);

/*
 * Cleanup CSR++ graph
 */
void snn_csr_graph_cleanup(struct snn_csr_graph *graph);

/*
 * Add node to graph
 * Returns node ID
 */
u32 snn_csr_graph_add_node(struct snn_csr_graph *graph,
                           const fp_t *features);

/*
 * Add edge to graph (CSR++ dynamic insertion)
 *
 * Uses append strategy with periodic compaction
 */
int snn_csr_graph_add_edge(struct snn_csr_graph *graph,
                           u32 src_node,
                           u32 dst_node,
                           u16 edge_type,
                           fp_t weight);

/*
 * Remove edge (tombstone-based)
 */
int snn_csr_graph_remove_edge(struct snn_csr_graph *graph,
                              u32 src_node,
                              u32 dst_node);

/*
 * Update edge weight
 */
int snn_csr_graph_update_weight(struct snn_csr_graph *graph,
                                u32 src_node,
                                u32 dst_node,
                                fp_t new_weight);

/*
 * Get neighbors of node (lock-free read via RCU)
 *
 * Returns number of neighbors copied
 */
u32 snn_csr_graph_get_neighbors(struct snn_csr_graph *graph,
                                u32 node_id,
                                u32 *neighbors,
                                u32 max_neighbors);

/*
 * Begin iterator for node's edges
 */
void snn_graph_iter_begin(struct snn_csr_graph *graph,
                          u32 node_id,
                          struct snn_graph_iterator *iter);

/*
 * Get next edge (returns false when done)
 */
bool snn_graph_iter_next(struct snn_graph_iterator *iter,
                         u32 *dst_node,
                         u16 *edge_type,
                         fp_t *weight);

/*
 * Compact graph (remove tombstones, rebuild CSR)
 *
 * This is an expensive operation - only call when tombstone ratio is high
 */
int snn_csr_graph_compact(struct snn_csr_graph *graph);

/*
 * Get node feature vector
 */
const fp_t *snn_csr_graph_get_features(struct snn_csr_graph *graph,
                                       u32 node_id);

/*
 * Update node features
 */
int snn_csr_graph_set_features(struct snn_csr_graph *graph,
                               u32 node_id,
                               const fp_t *features);

/*
 * Get graph statistics
 */
void snn_csr_graph_stats(struct snn_csr_graph *graph,
                         u64 *traversals,
                         u64 *updates,
                         u32 *tombstones,
                         float *fill_ratio);

/*
 * Helper: Check if compaction is needed
 */
static inline bool snn_csr_graph_needs_compaction(struct snn_csr_graph *graph)
{
	/* Compact when >25% edges are tombstones */
	return (graph->tombstone_count > graph->live_edge_count / 4);
}

/*
 * Helper: Get degree of node (number of outgoing edges)
 */
static inline u32 snn_csr_graph_get_degree(struct snn_csr_graph *graph,
                                           u32 node_id)
{
	if (node_id >= graph->node_count)
		return 0;

	return graph->row_ptr[node_id + 1] - graph->row_ptr[node_id];
}

/*
 * Helper: Check if edge exists
 */
bool snn_csr_graph_has_edge(struct snn_csr_graph *graph,
                            u32 src_node,
                            u32 dst_node);

#endif /* _SNN_CSR_GRAPH_H */
