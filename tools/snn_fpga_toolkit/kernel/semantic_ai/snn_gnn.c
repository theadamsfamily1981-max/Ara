/*
 * Graph Neural Network Implementation
 *
 * Lightweight GNN for kernel-space graph convolution
 */

#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/random.h>
#include "snn_gnn.h"

/*
 * Initialize GNN model
 */
int snn_gnn_init(struct snn_gnn_model **model_ptr,
                 struct snn_csr_graph *graph,
                 u32 input_dim,
                 u32 hidden_dim,
                 u32 output_dim,
                 u32 num_layers)
{
	struct snn_gnn_model *model;
	int ret;

	if (num_layers > SNN_GNN_MAX_LAYERS)
		return -EINVAL;

	model = kzalloc(sizeof(*model), GFP_KERNEL);
	if (!model)
		return -ENOMEM;

	model->graph = graph;
	model->num_layers = num_layers;
	model->max_nodes = graph->node_count;
	model->pool_type = SNN_GNN_AGG_MEAN;

	/* Initialize layers */
	for (u32 i = 0; i < num_layers; i++) {
		struct snn_gcn_layer *layer = &model->layers[i];
		u32 in_dim, out_dim;

		/* Determine layer dimensions */
		if (i == 0) {
			in_dim = input_dim;
		} else {
			in_dim = hidden_dim;
		}

		if (i == num_layers - 1) {
			out_dim = output_dim;
		} else {
			out_dim = hidden_dim;
		}

		layer->input_dim = in_dim;
		layer->output_dim = out_dim;
		layer->agg_type = SNN_GNN_AGG_MEAN;

		/* Allocate weights */
		layer->weights = kzalloc(in_dim * out_dim * sizeof(fp_t), GFP_KERNEL);
		if (!layer->weights) {
			ret = -ENOMEM;
			goto cleanup;
		}

		layer->bias = kzalloc(out_dim * sizeof(fp_t), GFP_KERNEL);
		if (!layer->bias) {
			kfree(layer->weights);
			ret = -ENOMEM;
			goto cleanup;
		}

		/* Initialize weights */
		snn_gnn_init_weights(layer->weights, in_dim, out_dim);

		/* Initialize bias to zero */
		memset(layer->bias, 0, out_dim * sizeof(fp_t));
	}

	/* Allocate working memory for layer outputs */
	for (u32 i = 0; i <= num_layers; i++) {
		u32 dim = (i == 0) ? input_dim :
		          (i == num_layers) ? output_dim : hidden_dim;

		model->layer_outputs[i] = kzalloc(model->max_nodes * dim * sizeof(fp_t),
		                                  GFP_KERNEL);
		if (!model->layer_outputs[i]) {
			ret = -ENOMEM;
			goto cleanup;
		}
	}

	atomic64_set(&model->forward_passes, 0);
	atomic64_set(&model->total_latency_ns, 0);

	*model_ptr = model;

	pr_info("SNN_GNN: Model initialized (%u layers, dim: %u->%u->%u)\n",
	        num_layers, input_dim, hidden_dim, output_dim);

	return 0;

cleanup:
	snn_gnn_cleanup(model);
	return ret;
}

/*
 * Cleanup GNN model
 */
void snn_gnn_cleanup(struct snn_gnn_model *model)
{
	if (!model)
		return;

	for (u32 i = 0; i < model->num_layers; i++) {
		kfree(model->layers[i].weights);
		kfree(model->layers[i].bias);
	}

	for (u32 i = 0; i <= model->num_layers; i++) {
		kfree(model->layer_outputs[i]);
	}

	kfree(model);
}

/*
 * Aggregate neighbor features
 */
void snn_gcn_aggregate(struct snn_csr_graph *graph,
                       u32 node_id,
                       const fp_t *input_features,
                       u32 feature_dim,
                       enum snn_gnn_aggregation agg_type,
                       fp_t *aggregated)
{
	struct snn_graph_iterator iter;
	u32 neighbor;
	u16 edge_type;
	fp_t weight;
	u32 count = 0;

	/* Initialize aggregated features */
	memset(aggregated, 0, feature_dim * sizeof(fp_t));

	/* Add self-loop (important for GCN!) */
	const fp_t *self_features = &input_features[node_id * feature_dim];
	memcpy(aggregated, self_features, feature_dim * sizeof(fp_t));
	count = 1;

	/* Aggregate neighbor features */
	snn_graph_iter_begin(graph, node_id, &iter);

	while (snn_graph_iter_next(&iter, &neighbor, &edge_type, &weight)) {
		const fp_t *neighbor_features = &input_features[neighbor * feature_dim];

		switch (agg_type) {
		case SNN_GNN_AGG_MEAN:
		case SNN_GNN_AGG_SUM:
			/* Accumulate features */
			for (u32 i = 0; i < feature_dim; i++) {
				aggregated[i] = fp_add(aggregated[i],
				                      fp_mul(neighbor_features[i], weight));
			}
			break;

		case SNN_GNN_AGG_MAX:
			/* Element-wise max */
			for (u32 i = 0; i < feature_dim; i++) {
				if (neighbor_features[i] > aggregated[i])
					aggregated[i] = neighbor_features[i];
			}
			break;
		}

		count++;
	}

	/* For mean aggregation, divide by count */
	if (agg_type == SNN_GNN_AGG_MEAN && count > 0) {
		fp_t divisor = FP_FROM_INT(count);
		for (u32 i = 0; i < feature_dim; i++) {
			aggregated[i] = fp_div(aggregated[i], divisor);
		}
	}
}

/*
 * Apply single GCN layer
 */
int snn_gcn_layer_forward(struct snn_gcn_layer *layer,
                          struct snn_csr_graph *graph,
                          const fp_t *input_features,
                          fp_t *output_features,
                          u32 num_nodes)
{
	fp_t *aggregated;

	/* Allocate temporary buffer for aggregation */
	aggregated = kzalloc(layer->input_dim * sizeof(fp_t), GFP_KERNEL);
	if (!aggregated)
		return -ENOMEM;

	/* Process each node */
	for (u32 node = 0; node < num_nodes; node++) {
		/* Step 1: Aggregate neighbor features */
		snn_gcn_aggregate(graph, node, input_features, layer->input_dim,
		                 layer->agg_type, aggregated);

		/* Step 2: Apply linear transformation: W × aggregated + bias */
		fp_t *out = &output_features[node * layer->output_dim];
		fp_matvec(layer->weights, aggregated, layer->bias,
		         out, layer->output_dim, layer->input_dim);

		/* Step 3: Apply activation (ReLU) */
		for (u32 i = 0; i < layer->output_dim; i++) {
			out[i] = fp_relu(out[i]);
		}
	}

	kfree(aggregated);
	return 0;
}

/*
 * Global pooling
 */
void snn_gnn_global_pool(const fp_t *node_features,
                         u32 num_nodes,
                         u32 feature_dim,
                         enum snn_gnn_aggregation pool_type,
                         fp_t *graph_embedding)
{
	memset(graph_embedding, 0, feature_dim * sizeof(fp_t));

	switch (pool_type) {
	case SNN_GNN_AGG_MEAN:
	case SNN_GNN_AGG_SUM:
		/* Sum all node features */
		for (u32 node = 0; node < num_nodes; node++) {
			const fp_t *features = &node_features[node * feature_dim];
			for (u32 i = 0; i < feature_dim; i++) {
				graph_embedding[i] = fp_add(graph_embedding[i], features[i]);
			}
		}

		/* For mean, divide by count */
		if (pool_type == SNN_GNN_AGG_MEAN && num_nodes > 0) {
			fp_t divisor = FP_FROM_INT(num_nodes);
			for (u32 i = 0; i < feature_dim; i++) {
				graph_embedding[i] = fp_div(graph_embedding[i], divisor);
			}
		}
		break;

	case SNN_GNN_AGG_MAX:
		/* Element-wise max across all nodes */
		if (num_nodes > 0) {
			memcpy(graph_embedding, node_features, feature_dim * sizeof(fp_t));

			for (u32 node = 1; node < num_nodes; node++) {
				const fp_t *features = &node_features[node * feature_dim];
				for (u32 i = 0; i < feature_dim; i++) {
					if (features[i] > graph_embedding[i])
						graph_embedding[i] = features[i];
				}
			}
		}
		break;
	}
}

/*
 * Forward pass: Compute graph embedding
 */
int snn_gnn_forward(struct snn_gnn_model *model,
                    fp_t *output_embedding)
{
	u64 start_ns, end_ns;
	int ret;

	start_ns = ktime_get_ns();

	/* Layer 0 input: Copy node features from graph */
	for (u32 node = 0; node < model->graph->node_count; node++) {
		const fp_t *features = snn_csr_graph_get_features(model->graph, node);
		if (features) {
			memcpy(&model->layer_outputs[0][node * model->layers[0].input_dim],
			       features,
			       model->layers[0].input_dim * sizeof(fp_t));
		}
	}

	/* Apply GCN layers */
	for (u32 i = 0; i < model->num_layers; i++) {
		ret = snn_gcn_layer_forward(&model->layers[i],
		                            model->graph,
		                            model->layer_outputs[i],
		                            model->layer_outputs[i + 1],
		                            model->graph->node_count);
		if (ret < 0)
			return ret;

		pr_debug("SNN_GNN: Layer %u forward complete\n", i);
	}

	/* Global pooling: Aggregate all node embeddings */
	u32 output_dim = model->layers[model->num_layers - 1].output_dim;
	snn_gnn_global_pool(model->layer_outputs[model->num_layers],
	                   model->graph->node_count,
	                   output_dim,
	                   model->pool_type,
	                   output_embedding);

	end_ns = ktime_get_ns();

	atomic64_inc(&model->forward_passes);
	atomic64_add(end_ns - start_ns, &model->total_latency_ns);

	pr_debug("SNN_GNN: Forward pass complete (%llu ns)\n", end_ns - start_ns);

	return 0;
}

/*
 * Get node embedding after forward pass
 */
int snn_gnn_get_node_embedding(struct snn_gnn_model *model,
                               u32 node_id,
                               fp_t *embedding)
{
	u32 output_dim;

	if (node_id >= model->graph->node_count)
		return -EINVAL;

	output_dim = model->layers[model->num_layers - 1].output_dim;

	memcpy(embedding,
	       &model->layer_outputs[model->num_layers][node_id * output_dim],
	       output_dim * sizeof(fp_t));

	return 0;
}

/*
 * Get GNN statistics
 */
void snn_gnn_get_stats(struct snn_gnn_model *model,
                       u64 *forward_passes,
                       u64 *avg_latency_ns)
{
	u64 passes = atomic64_read(&model->forward_passes);
	u64 total_latency = atomic64_read(&model->total_latency_ns);

	*forward_passes = passes;
	*avg_latency_ns = (passes > 0) ? (total_latency / passes) : 0;
}

/*
 * Initialize random weights (Xavier initialization)
 */
void snn_gnn_init_weights(fp_t *weights, u32 in_dim, u32 out_dim)
{
	u32 count = in_dim * out_dim;

	/* Xavier: scale = sqrt(2 / (in_dim + out_dim))
	 * For simplicity, use scale ≈ 1/sqrt(in_dim)
	 */
	fp_t scale = fp_div(FP_ONE, fp_sqrt(FP_FROM_INT(in_dim)));

	for (u32 i = 0; i < count; i++) {
		/* Generate random value in [-1, 1] */
		u32 random_val;
		get_random_bytes(&random_val, sizeof(random_val));

		s32 signed_val = (s32)(random_val % 512) - 256;  /* [-256, 256] */
		fp_t random_fp = FP_FROM_INT(signed_val) >> 8;    /* [-1, 1] */

		weights[i] = fp_mul(random_fp, scale);
	}

	pr_debug("SNN_GNN: Initialized weights [%u × %u] with scale=%d.%03d\n",
	         in_dim, out_dim,
	         FP_TO_INT(scale),
	         FP_TO_FRAC(scale, 1000));
}

MODULE_LICENSE("GPL");
MODULE_AUTHOR("SNN Kernel Team");
MODULE_DESCRIPTION("Graph Neural Network for State Embedding");
