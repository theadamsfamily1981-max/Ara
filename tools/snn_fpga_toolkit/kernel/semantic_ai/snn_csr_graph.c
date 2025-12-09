/*
 * CSR++ Dynamic Graph Implementation
 *
 * High-performance concurrent graph structure for kernel-space GNN
 */

#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/sort.h>
#include <linux/rcupdate.h>
#include "snn_csr_graph.h"

/*
 * Initialize CSR++ graph
 */
int snn_csr_graph_init(struct snn_csr_graph **graph_ptr,
                       u32 max_nodes,
                       u32 max_edges,
                       u32 feature_dim)
{
	struct snn_csr_graph *graph;

	graph = kzalloc(sizeof(*graph), GFP_KERNEL);
	if (!graph)
		return -ENOMEM;

	/* Allocate row pointer array */
	graph->row_ptr = kzalloc((max_nodes + 1) * sizeof(u32), GFP_KERNEL);
	if (!graph->row_ptr) {
		kfree(graph);
		return -ENOMEM;
	}

	/* Allocate edge array */
	graph->edges = kzalloc(max_edges * sizeof(struct snn_csr_edge), GFP_KERNEL);
	if (!graph->edges) {
		kfree(graph->row_ptr);
		kfree(graph);
		return -ENOMEM;
	}

	/* Allocate node features */
	graph->node_features = kzalloc(max_nodes * feature_dim * sizeof(fp_t), GFP_KERNEL);
	if (!graph->node_features) {
		kfree(graph->edges);
		kfree(graph->row_ptr);
		kfree(graph);
		return -ENOMEM;
	}

	graph->node_count = 0;
	graph->edge_count = 0;
	graph->live_edge_count = 0;
	graph->capacity = max_edges;
	graph->feature_dim = feature_dim;
	graph->tombstone_count = 0;
	graph->compaction_count = 0;

	/* Initialize edge type weights to 1.0 */
	for (u32 i = 0; i < SNN_GRAPH_EDGE_TYPES; i++)
		graph->edge_type_weights[i] = FP_ONE;

	spin_lock_init(&graph->write_lock);
	atomic64_set(&graph->traversals, 0);
	atomic64_set(&graph->updates, 0);

	*graph_ptr = graph;

	pr_info("SNN_CSR: Graph initialized (nodes=%u, edges=%u, features=%u)\n",
	        max_nodes, max_edges, feature_dim);

	return 0;
}

/*
 * Cleanup CSR++ graph
 */
void snn_csr_graph_cleanup(struct snn_csr_graph *graph)
{
	if (!graph)
		return;

	/* Wait for any RCU readers */
	synchronize_rcu();

	kfree(graph->node_features);
	kfree(graph->edges);
	kfree(graph->row_ptr);
	kfree(graph);
}

/*
 * Add node to graph
 */
u32 snn_csr_graph_add_node(struct snn_csr_graph *graph,
                           const fp_t *features)
{
	u32 node_id;

	spin_lock(&graph->write_lock);

	node_id = graph->node_count;

	/* Update row_ptr for new node */
	graph->row_ptr[node_id + 1] = graph->edge_count;

	/* Copy features */
	if (features) {
		memcpy(&graph->node_features[node_id * graph->feature_dim],
		       features,
		       graph->feature_dim * sizeof(fp_t));
	}

	graph->node_count++;

	spin_unlock(&graph->write_lock);

	return node_id;
}

/*
 * Add edge to graph (CSR++ dynamic insertion)
 */
int snn_csr_graph_add_edge(struct snn_csr_graph *graph,
                           u32 src_node,
                           u32 dst_node,
                           u16 edge_type,
                           fp_t weight)
{
	struct snn_csr_edge *edge;
	u32 insert_pos;

	if (src_node >= graph->node_count || dst_node >= graph->node_count)
		return -EINVAL;

	if (graph->edge_count >= graph->capacity)
		return -ENOSPC;

	spin_lock(&graph->write_lock);

	/* Find insertion position (end of src_node's edges) */
	insert_pos = graph->row_ptr[src_node + 1];

	/* Shift subsequent edges if not at end */
	if (insert_pos < graph->edge_count) {
		memmove(&graph->edges[insert_pos + 1],
		        &graph->edges[insert_pos],
		        (graph->edge_count - insert_pos) * sizeof(struct snn_csr_edge));

		/* Update row pointers for shifted nodes */
		for (u32 i = src_node + 1; i <= graph->node_count; i++)
			graph->row_ptr[i]++;
	}

	/* Insert new edge */
	edge = &graph->edges[insert_pos];
	edge->dst_node = dst_node;
	edge->edge_type = edge_type;
	edge->flags = 0;
	edge->weight = weight;

	graph->edge_count++;
	graph->live_edge_count++;

	atomic64_inc(&graph->updates);

	spin_unlock(&graph->write_lock);

	return 0;
}

/*
 * Remove edge (tombstone-based)
 */
int snn_csr_graph_remove_edge(struct snn_csr_graph *graph,
                              u32 src_node,
                              u32 dst_node)
{
	u32 start, end;
	struct snn_csr_edge *edge;
	bool found = false;

	if (src_node >= graph->node_count)
		return -EINVAL;

	spin_lock(&graph->write_lock);

	start = graph->row_ptr[src_node];
	end = graph->row_ptr[src_node + 1];

	/* Find and mark edge as tombstone */
	for (u32 i = start; i < end; i++) {
		edge = &graph->edges[i];
		if (edge->dst_node == dst_node && !(edge->flags & SNN_EDGE_FLAG_TOMBSTONE)) {
			edge->flags |= SNN_EDGE_FLAG_TOMBSTONE;
			graph->tombstone_count++;
			graph->live_edge_count--;
			found = true;
			break;
		}
	}

	spin_unlock(&graph->write_lock);

	/* Check if compaction is needed */
	if (found && snn_csr_graph_needs_compaction(graph)) {
		pr_debug("SNN_CSR: Compaction triggered (tombstones=%u/%u)\n",
		         graph->tombstone_count, graph->edge_count);
		snn_csr_graph_compact(graph);
	}

	return found ? 0 : -ENOENT;
}

/*
 * Update edge weight
 */
int snn_csr_graph_update_weight(struct snn_csr_graph *graph,
                                u32 src_node,
                                u32 dst_node,
                                fp_t new_weight)
{
	u32 start, end;
	struct snn_csr_edge *edge;
	bool found = false;

	if (src_node >= graph->node_count)
		return -EINVAL;

	spin_lock(&graph->write_lock);

	start = graph->row_ptr[src_node];
	end = graph->row_ptr[src_node + 1];

	/* Find and update edge */
	for (u32 i = start; i < end; i++) {
		edge = &graph->edges[i];
		if (edge->dst_node == dst_node && !(edge->flags & SNN_EDGE_FLAG_TOMBSTONE)) {
			edge->weight = new_weight;
			found = true;
			break;
		}
	}

	atomic64_inc(&graph->updates);

	spin_unlock(&graph->write_lock);

	return found ? 0 : -ENOENT;
}

/*
 * Get neighbors of node (lock-free read via RCU)
 */
u32 snn_csr_graph_get_neighbors(struct snn_csr_graph *graph,
                                u32 node_id,
                                u32 *neighbors,
                                u32 max_neighbors)
{
	u32 start, end, count = 0;
	struct snn_csr_edge *edge;

	if (node_id >= graph->node_count)
		return 0;

	rcu_read_lock();

	start = graph->row_ptr[node_id];
	end = graph->row_ptr[node_id + 1];

	for (u32 i = start; i < end && count < max_neighbors; i++) {
		edge = &graph->edges[i];
		if (!(edge->flags & SNN_EDGE_FLAG_TOMBSTONE)) {
			neighbors[count++] = edge->dst_node;
		}
	}

	atomic64_inc(&graph->traversals);

	rcu_read_unlock();

	return count;
}

/*
 * Begin iterator for node's edges
 */
void snn_graph_iter_begin(struct snn_csr_graph *graph,
                          u32 node_id,
                          struct snn_graph_iterator *iter)
{
	iter->graph = graph;
	iter->node_id = node_id;

	if (node_id < graph->node_count) {
		iter->edge_idx = graph->row_ptr[node_id];
		iter->end_idx = graph->row_ptr[node_id + 1];
	} else {
		iter->edge_idx = 0;
		iter->end_idx = 0;
	}
}

/*
 * Get next edge
 */
bool snn_graph_iter_next(struct snn_graph_iterator *iter,
                         u32 *dst_node,
                         u16 *edge_type,
                         fp_t *weight)
{
	struct snn_csr_edge *edge;

	/* Skip tombstones */
	while (iter->edge_idx < iter->end_idx) {
		edge = &iter->graph->edges[iter->edge_idx++];

		if (!(edge->flags & SNN_EDGE_FLAG_TOMBSTONE)) {
			*dst_node = edge->dst_node;
			*edge_type = edge->edge_type;
			*weight = edge->weight;
			return true;
		}
	}

	return false;
}

/*
 * Compact graph (remove tombstones, rebuild CSR)
 */
int snn_csr_graph_compact(struct snn_csr_graph *graph)
{
	struct snn_csr_edge *new_edges;
	u32 new_count = 0;
	u32 node_idx = 0;

	if (graph->tombstone_count == 0)
		return 0;

	/* Allocate new edge array */
	new_edges = kzalloc(graph->capacity * sizeof(struct snn_csr_edge), GFP_KERNEL);
	if (!new_edges)
		return -ENOMEM;

	spin_lock(&graph->write_lock);

	/* Rebuild CSR without tombstones */
	graph->row_ptr[0] = 0;

	for (u32 node = 0; node < graph->node_count; node++) {
		u32 start = graph->row_ptr[node];
		u32 end = graph->row_ptr[node + 1];

		for (u32 i = start; i < end; i++) {
			if (!(graph->edges[i].flags & SNN_EDGE_FLAG_TOMBSTONE)) {
				new_edges[new_count++] = graph->edges[i];
			}
		}

		graph->row_ptr[node + 1] = new_count;
	}

	/* Swap edge arrays */
	kfree(graph->edges);
	graph->edges = new_edges;
	graph->edge_count = new_count;
	graph->live_edge_count = new_count;
	graph->tombstone_count = 0;
	graph->compaction_count++;

	spin_unlock(&graph->write_lock);

	pr_debug("SNN_CSR: Compaction complete (edges: %u -> %u)\n",
	         graph->edge_count + graph->tombstone_count, new_count);

	return 0;
}

/*
 * Get node feature vector
 */
const fp_t *snn_csr_graph_get_features(struct snn_csr_graph *graph,
                                       u32 node_id)
{
	if (node_id >= graph->node_count)
		return NULL;

	return &graph->node_features[node_id * graph->feature_dim];
}

/*
 * Update node features
 */
int snn_csr_graph_set_features(struct snn_csr_graph *graph,
                               u32 node_id,
                               const fp_t *features)
{
	if (node_id >= graph->node_count)
		return -EINVAL;

	memcpy(&graph->node_features[node_id * graph->feature_dim],
	       features,
	       graph->feature_dim * sizeof(fp_t));

	return 0;
}

/*
 * Get graph statistics
 */
void snn_csr_graph_stats(struct snn_csr_graph *graph,
                         u64 *traversals,
                         u64 *updates,
                         u32 *tombstones,
                         float *fill_ratio)
{
	*traversals = atomic64_read(&graph->traversals);
	*updates = atomic64_read(&graph->updates);
	*tombstones = graph->tombstone_count;
	*fill_ratio = (float)graph->edge_count / (float)graph->capacity;
}

/*
 * Check if edge exists
 */
bool snn_csr_graph_has_edge(struct snn_csr_graph *graph,
                            u32 src_node,
                            u32 dst_node)
{
	u32 start, end;
	struct snn_csr_edge *edge;

	if (src_node >= graph->node_count)
		return false;

	start = graph->row_ptr[src_node];
	end = graph->row_ptr[src_node + 1];

	for (u32 i = start; i < end; i++) {
		edge = &graph->edges[i];
		if (edge->dst_node == dst_node && !(edge->flags & SNN_EDGE_FLAG_TOMBSTONE))
			return true;
	}

	return false;
}

MODULE_LICENSE("GPL");
MODULE_AUTHOR("SNN Kernel Team");
MODULE_DESCRIPTION("CSR++ Dynamic Graph Structure");
