/*
 * Graph Neural Network (GNN) for State Embedding
 *
 * Implements lightweight GNN for kernel-space graph convolution:
 * - Graph Convolutional Network (GCN) layers
 * - Message passing via neighbor aggregation
 * - Fixed-point arithmetic throughout
 * - Multi-hop reasoning (k-hop neighborhoods)
 *
 * State Embedding Pipeline:
 * 1. Initialize node features from system state
 * 2. Apply k layers of graph convolution
 * 3. Aggregate to global graph embedding
 * 4. Use embedding as part of AI decision state
 */

#ifndef _SNN_GNN_H
#define _SNN_GNN_H

#include <linux/types.h>
#include "snn_fixed_point.h"
#include "snn_csr_graph.h"

/*
 * GNN hyperparameters
 */
#define SNN_GNN_MAX_LAYERS      3      /* Max GCN layers */
#define SNN_GNN_HIDDEN_DIM      32     /* Hidden layer dimension */
#define SNN_GNN_OUTPUT_DIM      16     /* Output embedding dimension */

/*
 * Aggregation functions for graph convolution
 */
enum snn_gnn_aggregation {
	SNN_GNN_AGG_MEAN = 0,    /* Mean aggregation */
	SNN_GNN_AGG_SUM,         /* Sum aggregation */
	SNN_GNN_AGG_MAX,         /* Max aggregation */
};

/*
 * GCN Layer weights
 *
 * For layer l: h^(l) = σ(W^(l) · AGG(h^(l-1)))
 * where AGG aggregates neighbor features
 */
struct snn_gcn_layer {
	fp_t *weights;           /* Weight matrix [in_dim × out_dim] */
	fp_t *bias;              /* Bias vector [out_dim] */
	u32 input_dim;
	u32 output_dim;
	enum snn_gnn_aggregation agg_type;
} __attribute__((aligned(64)));

/*
 * GNN Model
 */
struct snn_gnn_model {
	/* Layers */
	struct snn_gcn_layer layers[SNN_GNN_MAX_LAYERS];
	u32 num_layers;

	/* Graph reference */
	struct snn_csr_graph *graph;

	/* Working memory for forward pass */
	fp_t *layer_outputs[SNN_GNN_MAX_LAYERS + 1];  /* Layer activations */
	u32 max_nodes;

	/* Global pooling */
	enum snn_gnn_aggregation pool_type;

	/* Statistics */
	atomic64_t forward_passes;
	atomic64_t total_latency_ns;
} __attribute__((aligned(64)));

/*
 * Initialize GNN model
 */
int snn_gnn_init(struct snn_gnn_model **model,
                 struct snn_csr_graph *graph,
                 u32 input_dim,
                 u32 hidden_dim,
                 u32 output_dim,
                 u32 num_layers);

/*
 * Cleanup GNN model
 */
void snn_gnn_cleanup(struct snn_gnn_model *model);

/*
 * Forward pass: Compute graph embedding
 *
 * Input:  Node features from graph
 * Output: Global graph embedding (output_dim dimensional vector)
 *
 * Returns: 0 on success
 */
int snn_gnn_forward(struct snn_gnn_model *model,
                    fp_t *output_embedding);

/*
 * Get node embedding after forward pass
 *
 * This returns the learned representation of a specific node
 * after GNN message passing.
 */
int snn_gnn_get_node_embedding(struct snn_gnn_model *model,
                               u32 node_id,
                               fp_t *embedding);

/*
 * Update GNN weights (simple gradient-free update)
 *
 * For kernel use, we use simple weight perturbation instead of backprop
 */
int snn_gnn_update_weights(struct snn_gnn_model *model,
                           fp_t learning_rate);

/*
 * Get GNN statistics
 */
void snn_gnn_get_stats(struct snn_gnn_model *model,
                       u64 *forward_passes,
                       u64 *avg_latency_ns);

/*
 * Graph Convolution Layer Operations
 */

/*
 * Apply single GCN layer
 *
 * For each node v:
 *   h_v^(l) = σ(W^(l) · (1/(deg(v)+1)) · Σ_{u ∈ N(v) ∪ {v}} h_u^(l-1))
 *
 * Uses mean aggregation with self-loops
 */
int snn_gcn_layer_forward(struct snn_gcn_layer *layer,
                          struct snn_csr_graph *graph,
                          const fp_t *input_features,
                          fp_t *output_features,
                          u32 num_nodes);

/*
 * Aggregate neighbor features
 */
void snn_gcn_aggregate(struct snn_csr_graph *graph,
                       u32 node_id,
                       const fp_t *input_features,
                       u32 feature_dim,
                       enum snn_gnn_aggregation agg_type,
                       fp_t *aggregated);

/*
 * Global pooling (graph-level aggregation)
 *
 * Aggregates all node embeddings into single graph embedding
 */
void snn_gnn_global_pool(const fp_t *node_features,
                         u32 num_nodes,
                         u32 feature_dim,
                         enum snn_gnn_aggregation pool_type,
                         fp_t *graph_embedding);

/*
 * Activation functions (fixed-point)
 */

/* ReLU: max(0, x) */
static inline fp_t fp_relu(fp_t x)
{
	return (x > 0) ? x : 0;
}

/* Leaky ReLU: max(0.1x, x) */
static inline fp_t fp_leaky_relu(fp_t x)
{
	if (x > 0)
		return x;
	/* 0.1x ≈ x >> 3 (actually 0.125, close enough) */
	return x >> 3;
}

/*
 * Matrix-vector multiplication (fixed-point)
 *
 * out = W × in + bias
 * W: [out_dim × in_dim]
 * in: [in_dim]
 * out: [out_dim]
 */
static inline void fp_matvec(const fp_t *W,
                             const fp_t *in,
                             const fp_t *bias,
                             fp_t *out,
                             u32 out_dim,
                             u32 in_dim)
{
	for (u32 i = 0; i < out_dim; i++) {
		fp_t sum = 0;
		for (u32 j = 0; j < in_dim; j++) {
			sum = fp_add(sum, fp_mul(W[i * in_dim + j], in[j]));
		}
		out[i] = fp_add(sum, bias ? bias[i] : 0);
	}
}

/*
 * Helper: Initialize random weights (Xavier/Glorot initialization)
 *
 * For kernel, use simple random initialization scaled by sqrt(1/fan_in)
 */
void snn_gnn_init_weights(fp_t *weights, u32 in_dim, u32 out_dim);

#endif /* _SNN_GNN_H */
