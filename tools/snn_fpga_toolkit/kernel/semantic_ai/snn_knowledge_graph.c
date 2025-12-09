/*
 * SNN Knowledge Graph
 *
 * Semantic knowledge representation and reasoning for
 * system optimization and decision making
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/string.h>

#include "snn_ai_internal.h"

/*
 * Initialize knowledge graph
 */
int snn_kg_init(struct snn_knowledge_graph **kg_ptr)
{
    struct snn_knowledge_graph *kg;

    kg = kzalloc(sizeof(*kg), GFP_KERNEL);
    if (!kg)
        return -ENOMEM;

    spin_lock_init(&kg->lock);
    INIT_LIST_HEAD(&kg->nodes);
    kg->node_count = 0;
    kg->edge_count = 0;

    *kg_ptr = kg;

    pr_info("SNN_KG: Knowledge graph initialized\n");
    return 0;
}

/*
 * Cleanup knowledge graph
 */
void snn_kg_cleanup(struct snn_knowledge_graph *kg)
{
    struct snn_kg_node *node, *node_tmp;
    struct snn_kg_edge *edge, *edge_tmp;

    if (!kg)
        return;

    pr_info("SNN_KG: Cleaning up knowledge graph\n");

    spin_lock(&kg->lock);

    /* Free all nodes and their edges */
    list_for_each_entry_safe(node, node_tmp, &kg->nodes, list) {
        /* Free edges for this node */
        list_for_each_entry_safe(edge, edge_tmp, &node->edges, list) {
            list_del(&edge->list);
            kfree(edge);
        }

        /* Free node data and node itself */
        if (node->data)
            kfree(node->data);
        list_del(&node->list);
        kfree(node);
    }

    spin_unlock(&kg->lock);
    kfree(kg);
}

/*
 * Add node to knowledge graph
 */
struct snn_kg_node *snn_kg_add_node(struct snn_knowledge_graph *kg,
                                    snn_kg_node_type_t type,
                                    const char *name,
                                    void *data)
{
    struct snn_kg_node *node;
    static u32 next_id = 1;

    if (!kg || !name)
        return NULL;

    node = kzalloc(sizeof(*node), GFP_KERNEL);
    if (!node)
        return NULL;

    node->id = next_id++;
    node->type = type;
    strncpy(node->name, name, sizeof(node->name) - 1);
    node->data = data;
    INIT_LIST_HEAD(&node->edges);

    spin_lock(&kg->lock);
    list_add_tail(&node->list, &kg->nodes);
    kg->node_count++;
    spin_unlock(&kg->lock);

    pr_debug("SNN_KG: Added node: id=%u, type=%d, name=%s\n",
             node->id, type, name);

    return node;
}

/*
 * Find node by name
 */
static struct snn_kg_node *find_node_by_name(struct snn_knowledge_graph *kg,
                                             const char *name)
{
    struct snn_kg_node *node;

    list_for_each_entry(node, &kg->nodes, list) {
        if (strcmp(node->name, name) == 0)
            return node;
    }

    return NULL;
}

/*
 * Add edge between nodes
 */
int snn_kg_add_edge(struct snn_knowledge_graph *kg,
                    struct snn_kg_node *from,
                    struct snn_kg_node *to,
                    snn_kg_rel_type_t type,
                    float weight)
{
    struct snn_kg_edge *edge;

    if (!kg || !from || !to)
        return -EINVAL;

    edge = kzalloc(sizeof(*edge), GFP_KERNEL);
    if (!edge)
        return -ENOMEM;

    edge->type = type;
    edge->from = from;
    edge->to = to;
    edge->weight = weight;

    spin_lock(&kg->lock);
    list_add_tail(&edge->list, &from->edges);
    kg->edge_count++;
    spin_unlock(&kg->lock);

    pr_debug("SNN_KG: Added edge: %s -[%d]-> %s (weight=%.2f)\n",
             from->name, type, to->name, weight);

    return 0;
}

/*
 * Query knowledge graph for related nodes
 */
int snn_kg_query(struct snn_knowledge_graph *kg,
                struct snn_kg_node *node,
                snn_kg_rel_type_t rel_type,
                struct list_head *results)
{
    struct snn_kg_edge *edge;
    int count = 0;

    if (!kg || !node || !results)
        return -EINVAL;

    spin_lock(&kg->lock);

    /* Find all edges of specified type from this node */
    list_for_each_entry(edge, &node->edges, list) {
        if (edge->type == rel_type) {
            /* In a real implementation, we'd add to results list */
            /* For now, just count */
            count++;
        }
    }

    spin_unlock(&kg->lock);

    return count;
}

/*
 * Build initial knowledge base with domain knowledge
 */
int snn_kg_build_initial_kb(struct snn_knowledge_graph *kg)
{
    struct snn_kg_node *gpu_node, *fpga_node;
    struct snn_kg_node *dense_wl, *sparse_wl;
    struct snn_kg_node *opt_batch, *opt_p2p;

    if (!kg)
        return -EINVAL;

    /* Create device nodes */
    gpu_node = snn_kg_add_node(kg, SNN_KG_NODE_DEVICE, "GPU", NULL);
    fpga_node = snn_kg_add_node(kg, SNN_KG_NODE_DEVICE, "FPGA", NULL);

    /* Create workload pattern nodes */
    dense_wl = snn_kg_add_node(kg, SNN_KG_NODE_WORKLOAD, "Dense_Workload", NULL);
    sparse_wl = snn_kg_add_node(kg, SNN_KG_NODE_WORKLOAD, "Sparse_Workload", NULL);

    /* Create optimization nodes */
    opt_batch = snn_kg_add_node(kg, SNN_KG_NODE_OPTIMIZATION, "Batch_Processing", NULL);
    opt_p2p = snn_kg_add_node(kg, SNN_KG_NODE_OPTIMIZATION, "P2P_Transfer", NULL);

    if (!gpu_node || !fpga_node || !dense_wl || !sparse_wl || !opt_batch || !opt_p2p)
        return -ENOMEM;

    /* Encode domain knowledge as relationships */

    /* GPU performs well on dense workloads */
    snn_kg_add_edge(kg, gpu_node, dense_wl, SNN_KG_REL_PERFORMS_WELL, 0.9f);

    /* FPGA performs well on sparse workloads */
    snn_kg_add_edge(kg, fpga_node, sparse_wl, SNN_KG_REL_PERFORMS_WELL, 0.85f);

    /* GPU performs poorly on sparse workloads (relatively) */
    snn_kg_add_edge(kg, gpu_node, sparse_wl, SNN_KG_REL_PERFORMS_POORLY, 0.6f);

    /* Dense workloads require batch processing */
    snn_kg_add_edge(kg, dense_wl, opt_batch, SNN_KG_REL_REQUIRES, 0.8f);

    /* Both devices benefit from P2P transfers */
    snn_kg_add_edge(kg, gpu_node, opt_p2p, SNN_KG_REL_REQUIRES, 0.9f);
    snn_kg_add_edge(kg, fpga_node, opt_p2p, SNN_KG_REL_REQUIRES, 0.9f);

    pr_info("SNN_KG: Built initial knowledge base (%u nodes, %u edges)\n",
            kg->node_count, kg->edge_count);

    return 0;
}

/*
 * Query knowledge graph for device recommendation
 */
int snn_kg_recommend_device(struct snn_knowledge_graph *kg,
                           snn_workload_type_t workload_type,
                           float *gpu_score,
                           float *fpga_score)
{
    struct snn_kg_node *workload_node = NULL;
    struct snn_kg_node *gpu_node, *fpga_node;
    struct snn_kg_edge *edge;
    const char *workload_names[] = {
        "Dense_Workload",
        "Sparse_Workload",
        "Mixed_Workload",
        "IO_Bound_Workload",
        "Compute_Bound_Workload"
    };

    if (!kg || !gpu_score || !fpga_score)
        return -EINVAL;

    *gpu_score = 0.5f;  /* Default: neutral */
    *fpga_score = 0.5f;

    if (workload_type >= ARRAY_SIZE(workload_names))
        return 0;

    spin_lock(&kg->lock);

    /* Find workload node */
    workload_node = find_node_by_name(kg, workload_names[workload_type]);
    if (!workload_node) {
        spin_unlock(&kg->lock);
        return 0;
    }

    /* Find device nodes */
    gpu_node = find_node_by_name(kg, "GPU");
    fpga_node = find_node_by_name(kg, "FPGA");

    if (!gpu_node || !fpga_node) {
        spin_unlock(&kg->lock);
        return 0;
    }

    /* Analyze edges from devices to workload */
    list_for_each_entry(edge, &gpu_node->edges, list) {
        if (edge->to == workload_node) {
            if (edge->type == SNN_KG_REL_PERFORMS_WELL)
                *gpu_score = edge->weight;
            else if (edge->type == SNN_KG_REL_PERFORMS_POORLY)
                *gpu_score = 1.0f - edge->weight;
        }
    }

    list_for_each_entry(edge, &fpga_node->edges, list) {
        if (edge->to == workload_node) {
            if (edge->type == SNN_KG_REL_PERFORMS_WELL)
                *fpga_score = edge->weight;
            else if (edge->type == SNN_KG_REL_PERFORMS_POORLY)
                *fpga_score = 1.0f - edge->weight;
        }
    }

    spin_unlock(&kg->lock);

    pr_debug("SNN_KG: Recommendation for workload %d: GPU=%.2f, FPGA=%.2f\n",
             workload_type, *gpu_score, *fpga_score);

    return 0;
}

/*
 * Update knowledge graph based on performance feedback
 */
int snn_kg_update_from_feedback(struct snn_knowledge_graph *kg,
                               snn_device_type_t device,
                               snn_workload_type_t workload,
                               bool success,
                               float performance)
{
    struct snn_kg_node *device_node = NULL, *workload_node = NULL;
    struct snn_kg_edge *edge;
    bool edge_found = false;
    const char *device_names[] = {"CPU", "GPU", "FPGA"};
    const char *workload_names[] = {
        "Dense_Workload",
        "Sparse_Workload",
        "Mixed_Workload",
        "IO_Bound_Workload",
        "Compute_Bound_Workload"
    };

    if (!kg)
        return -EINVAL;

    spin_lock(&kg->lock);

    /* Find nodes */
    if (device < ARRAY_SIZE(device_names))
        device_node = find_node_by_name(kg, device_names[device]);

    if (workload < ARRAY_SIZE(workload_names))
        workload_node = find_node_by_name(kg, workload_names[workload]);

    if (!device_node || !workload_node) {
        spin_unlock(&kg->lock);
        return -ENOENT;
    }

    /* Find or create edge */
    list_for_each_entry(edge, &device_node->edges, list) {
        if (edge->to == workload_node) {
            /* Update existing edge weight based on performance */
            float alpha = 0.1f; /* Learning rate */
            float target = success ? performance : (1.0f - performance);

            edge->weight = edge->weight * (1.0f - alpha) + target * alpha;
            edge_found = true;

            pr_debug("SNN_KG: Updated edge %s->%s: weight=%.2f\n",
                     device_names[device], workload_names[workload],
                     edge->weight);
            break;
        }
    }

    if (!edge_found) {
        /* Create new edge */
        spin_unlock(&kg->lock);
        snn_kg_add_edge(kg, device_node, workload_node,
                       success ? SNN_KG_REL_PERFORMS_WELL : SNN_KG_REL_PERFORMS_POORLY,
                       performance);
        return 0;
    }

    spin_unlock(&kg->lock);
    return 0;
}

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("SNN Knowledge Graph");
