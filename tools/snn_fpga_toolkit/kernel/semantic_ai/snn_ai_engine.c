/*
 * SNN Semantic AI Engine
 *
 * Intelligent decision-making engine for adaptive resource allocation,
 * workload characterization, and system optimization
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/spinlock.h>
#include <linux/random.h>
#include <linux/math64.h>

#include "../core/snn_core.h"
#include "snn_ai_internal.h"

/* AI engine state */
struct snn_ai_engine {
    spinlock_t lock;
    bool initialized;
    bool learning_enabled;
    bool autonomous_mode;

    /* Configuration */
    snn_ai_config_t config;

    /* Q-learning state */
    struct snn_q_table *q_table;
    u32 current_state;
    u32 last_action;

    /* Workload history */
    struct snn_workload_history *history;
    u32 history_count;

    /* Knowledge graph */
    struct snn_knowledge_graph *kg;

    /* Statistics */
    atomic64_t total_decisions;
    atomic64_t successful_decisions;
    atomic64_t learning_iterations;
    s64 cumulative_reward;  /* Fixed-point: scaled by 1000 */

    /* Performance tracking */
    u64 avg_latency_ns;
    u32 avg_utilization;
};

/*
 * Workload characterization using feature extraction
 */
static snn_workload_type_t characterize_workload(
    const snn_compute_params_t *params,
    snn_workload_features_t *features)
{
    u64 synapses_per_neuron;
    float sparsity_estimate;

    if (!params || !features)
        return SNN_WORKLOAD_UNKNOWN;

    memset(features, 0, sizeof(*features));

    features->num_neurons = params->num_neurons;
    features->num_synapses = params->num_synapses;
    features->timesteps = params->timesteps;
    features->batch_size = params->batch_size;

    /* Calculate sparsity (synapses / (neurons * neurons)) */
    if (params->num_neurons > 0) {
        u64 max_synapses = (u64)params->num_neurons * params->num_neurons;
        sparsity_estimate = 1.0f - ((float)params->num_synapses / max_synapses);
        features->sparsity = sparsity_estimate;
    }

    /* Estimate computation intensity (FLOPs per byte) */
    synapses_per_neuron = params->num_synapses / max(params->num_neurons, 1u);
    features->computation_intensity = (float)synapses_per_neuron * 2.0f; /* MAC ops */

    /* Total data size */
    features->data_size = (u64)params->num_synapses * sizeof(float) +
                         (u64)params->num_neurons * params->timesteps;

    /* Memory bandwidth requirement (bytes/ms) */
    features->memory_bandwidth_req = (u32)(features->data_size / 10); /* Estimate */

    /* Determine workload type */
    if (features->sparsity > 0.7f) {
        return SNN_WORKLOAD_SPARSE;
    } else if (features->sparsity < 0.3f) {
        return SNN_WORKLOAD_DENSE;
    } else if (features->computation_intensity > 100.0f) {
        return SNN_WORKLOAD_COMPUTE_BOUND;
    } else if (features->memory_bandwidth_req > 10000) {
        return SNN_WORKLOAD_IO_BOUND;
    } else {
        return SNN_WORKLOAD_MIXED;
    }
}

/*
 * Discretize system state for Q-learning
 */
static u32 discretize_state(const snn_system_state_t *state,
                           const snn_workload_features_t *features)
{
    u32 state_id = 0;

    /* Encode GPU utilization (0-3) */
    state_id |= (state->gpu_utilization / 25) << 0;

    /* Encode FPGA utilization (0-3) */
    state_id |= (state->fpga_utilization / 25) << 2;

    /* Encode sparsity category (0-3) */
    state_id |= ((u32)(features->sparsity * 4.0f)) << 4;

    /* Encode workload size (0-3) */
    u32 size_category = (features->num_neurons > 100000) ? 3 :
                       (features->num_neurons > 10000) ? 2 :
                       (features->num_neurons > 1000) ? 1 : 0;
    state_id |= size_category << 6;

    /* Encode deadline miss rate (0-1) */
    state_id |= (state->rt_deadline_miss_rate > 5 ? 1 : 0) << 8;

    return state_id % SNN_AI_STATE_SPACE_SIZE;
}

/*
 * Select action using epsilon-greedy policy
 */
static u32 select_action(struct snn_ai_engine *engine, u32 state)
{
    u32 random_val;
    u32 best_action = 0;
    s64 best_q_value = S64_MIN;
    u32 action;

    /* Epsilon-greedy exploration */
    get_random_bytes(&random_val, sizeof(random_val));
    random_val = random_val % 100;

    if (random_val < engine->config.exploration_rate) {
        /* Explore: random action */
        get_random_bytes(&action, sizeof(action));
        return action % SNN_AI_ACTION_SPACE_SIZE;
    }

    /* Exploit: choose best action */
    for (action = 0; action < SNN_AI_ACTION_SPACE_SIZE; action++) {
        s64 q_value = engine->q_table->values[state][action];
        if (q_value > best_q_value) {
            best_q_value = q_value;
            best_action = action;
        }
    }

    return best_action;
}

/*
 * Decode action into allocation recommendation
 */
static void decode_action(u32 action, snn_ai_allocation_t *allocation)
{
    /* Action encoding:
     * bits 0-1: GPU allocation (0=0%, 1=33%, 2=67%, 3=100%)
     * bits 2-3: FPGA allocation (0=0%, 1=33%, 2=67%, 3=100%)
     * bits 4-5: batch size multiplier (1x, 2x, 4x, 8x)
     * bit 6: prefetching enabled
     */

    u32 gpu_level = (action >> 0) & 0x3;
    u32 fpga_level = (action >> 2) & 0x3;
    u32 batch_mult = (action >> 4) & 0x3;
    u32 prefetch = (action >> 6) & 0x1;

    allocation->use_gpu = gpu_level * 33;
    allocation->use_fpga = fpga_level * 33;
    allocation->use_cpu = 100 - allocation->use_gpu - allocation->use_fpga;

    /* Clamp to 100% */
    if (allocation->use_gpu > 100) allocation->use_gpu = 100;
    if (allocation->use_fpga > 100) allocation->use_fpga = 100;

    allocation->batch_size = 1 << batch_mult; /* 1, 2, 4, 8 */
    allocation->memory_prefetch = prefetch;
    allocation->confidence = SNN_CONFIDENCE_MEDIUM;
}

/*
 * Calculate reward for reinforcement learning
 */
static s64 calculate_reward(const snn_ai_feedback_t *feedback)
{
    s64 reward = 0;

    /* Reward for meeting deadline (most important) */
    if (feedback->deadline_met) {
        reward += 1000;
    } else {
        reward -= 2000; /* Strong penalty for missing deadline */
    }

    /* Reward for latency (closer to expected is better) */
    s64 latency_error = (s64)feedback->actual_latency_ns -
                       (s64)feedback->expected_latency_ns;
    latency_error = abs64(latency_error);

    if (latency_error < 1000000) { /* < 1ms error */
        reward += 500;
    } else if (latency_error < 10000000) { /* < 10ms error */
        reward += 100;
    }

    /* Reward for resource efficiency */
    if (feedback->resource_utilization > 80 && feedback->resource_utilization < 95) {
        reward += 200; /* Good utilization */
    } else if (feedback->resource_utilization < 50) {
        reward -= 100; /* Underutilization */
    }

    /* Reward for P2P efficiency */
    if (feedback->p2p_efficiency > 85) {
        reward += 100;
    }

    return reward;
}

/*
 * Update Q-table using Q-learning algorithm
 */
static void update_q_table(struct snn_ai_engine *engine,
                          u32 state, u32 action,
                          s64 reward, u32 next_state)
{
    s64 old_q, max_next_q, new_q;
    s64 learning_rate = engine->config.learning_rate; /* 0-1000 */
    s64 discount_factor = 900; /* 0.9 in fixed-point */
    u32 i;

    /* Find max Q-value for next state */
    max_next_q = S64_MIN;
    for (i = 0; i < SNN_AI_ACTION_SPACE_SIZE; i++) {
        s64 q = engine->q_table->values[next_state][i];
        if (q > max_next_q)
            max_next_q = q;
    }

    if (max_next_q == S64_MIN)
        max_next_q = 0;

    /* Q-learning update:
     * Q(s,a) = Q(s,a) + α * (reward + γ * max(Q(s',a')) - Q(s,a))
     */
    old_q = engine->q_table->values[state][action];

    new_q = old_q + (learning_rate * (reward + (discount_factor * max_next_q / 1000) - old_q)) / 1000;

    engine->q_table->values[state][action] = new_q;

    pr_debug("SNN_AI: Q-update: state=%u, action=%u, reward=%lld, old_q=%lld, new_q=%lld\n",
             state, action, reward, old_q, new_q);
}

/*
 * Add workload to history for pattern recognition
 */
static void add_to_history(struct snn_ai_engine *engine,
                          const snn_workload_features_t *features,
                          const snn_ai_allocation_t *allocation,
                          const snn_ai_feedback_t *feedback)
{
    struct snn_workload_entry *entry;
    u32 idx;

    if (!engine->history)
        return;

    spin_lock(&engine->lock);

    idx = engine->history_count % SNN_AI_HISTORY_SIZE;
    entry = &engine->history->entries[idx];

    entry->features = *features;
    entry->allocation = *allocation;
    entry->feedback = *feedback;
    entry->timestamp = ktime_get_ns();

    engine->history_count++;

    spin_unlock(&engine->lock);
}

/*
 * Find similar workloads in history
 */
static bool find_similar_workload(struct snn_ai_engine *engine,
                                 const snn_workload_features_t *features,
                                 snn_ai_allocation_t *allocation)
{
    u32 i, count;
    float min_distance = FLT_MAX;
    u32 best_match = 0;
    bool found = false;

    if (!engine->history || engine->history_count == 0)
        return false;

    count = min(engine->history_count, SNN_AI_HISTORY_SIZE);

    /* Find most similar workload using simple distance metric */
    for (i = 0; i < count; i++) {
        struct snn_workload_entry *entry = &engine->history->entries[i];
        float distance;

        /* Simple Euclidean distance in feature space */
        distance = abs((s32)features->num_neurons - (s32)entry->features.num_neurons) +
                  abs((s32)features->num_synapses - (s32)entry->features.num_synapses) +
                  abs((s32)features->sparsity * 1000 - (s32)entry->features.sparsity * 1000);

        if (distance < min_distance && entry->feedback.deadline_met) {
            min_distance = distance;
            best_match = i;
            found = true;
        }
    }

    if (found && min_distance < 1000.0f) { /* Similarity threshold */
        *allocation = engine->history->entries[best_match].allocation;
        allocation->confidence = SNN_CONFIDENCE_HIGH;
        pr_debug("SNN_AI: Found similar workload, distance=%.2f\n", min_distance);
        return true;
    }

    return false;
}

/*
 * Initialize AI engine
 */
int snn_ai_engine_init(struct snn_ai_engine **engine_ptr,
                       const snn_ai_config_t *config)
{
    struct snn_ai_engine *engine;

    engine = kzalloc(sizeof(*engine), GFP_KERNEL);
    if (!engine)
        return -ENOMEM;

    /* Allocate Q-table */
    engine->q_table = kzalloc(sizeof(*engine->q_table), GFP_KERNEL);
    if (!engine->q_table) {
        kfree(engine);
        return -ENOMEM;
    }

    /* Allocate history */
    engine->history = kzalloc(sizeof(*engine->history), GFP_KERNEL);
    if (!engine->history) {
        kfree(engine->q_table);
        kfree(engine);
        return -ENOMEM;
    }

    /* Initialize */
    spin_lock_init(&engine->lock);
    engine->config = *config;
    engine->learning_enabled = config->flags & SNN_AI_ENABLE_LEARNING;
    engine->autonomous_mode = config->flags & SNN_AI_ENABLE_AUTONOMOUS;

    atomic64_set(&engine->total_decisions, 0);
    atomic64_set(&engine->successful_decisions, 0);
    atomic64_set(&engine->learning_iterations, 0);
    engine->cumulative_reward = 0;

    engine->initialized = true;
    *engine_ptr = engine;

    pr_info("SNN_AI: Engine initialized (learning=%d, autonomous=%d)\n",
            engine->learning_enabled, engine->autonomous_mode);

    return 0;
}

/*
 * Cleanup AI engine
 */
void snn_ai_engine_cleanup(struct snn_ai_engine *engine)
{
    if (!engine)
        return;

    pr_info("SNN_AI: Cleaning up engine\n");

    kfree(engine->history);
    kfree(engine->q_table);
    kfree(engine);
}

/*
 * Get allocation recommendation from AI
 */
int snn_ai_recommend(struct snn_ai_engine *engine,
                    const snn_compute_params_t *params,
                    const snn_system_state_t *sys_state,
                    snn_ai_allocation_t *allocation)
{
    snn_workload_features_t features;
    snn_workload_type_t workload_type;
    u32 state, action;

    if (!engine || !engine->initialized || !params || !allocation)
        return -EINVAL;

    memset(allocation, 0, sizeof(*allocation));

    /* Characterize workload */
    workload_type = characterize_workload(params, &features);

    pr_debug("SNN_AI: Workload type=%d, sparsity=%.2f, comp_int=%.2f\n",
             workload_type, features.sparsity, features.computation_intensity);

    /* Try to find similar workload in history */
    if (find_similar_workload(engine, &features, allocation)) {
        pr_debug("SNN_AI: Using historical match\n");
        return 0;
    }

    /* Use RL to select action */
    state = discretize_state(sys_state, &features);
    action = select_action(engine, state);

    /* Decode action into allocation */
    decode_action(action, allocation);

    /* Calculate expected neurons per device */
    allocation->gpu_neurons = (params->num_neurons * allocation->use_gpu) / 100;
    allocation->fpga_neurons = (params->num_neurons * allocation->use_fpga) / 100;
    allocation->cpu_neurons = params->num_neurons - allocation->gpu_neurons -
                             allocation->fpga_neurons;

    /* Store state and action for learning */
    spin_lock(&engine->lock);
    engine->current_state = state;
    engine->last_action = action;
    spin_unlock(&engine->lock);

    atomic64_inc(&engine->total_decisions);

    pr_debug("SNN_AI: Recommendation: GPU=%u%%, FPGA=%u%%, batch=%u\n",
             allocation->use_gpu, allocation->use_fpga, allocation->batch_size);

    return 0;
}

/*
 * Provide feedback for learning
 */
int snn_ai_feedback(struct snn_ai_engine *engine,
                   const snn_compute_params_t *params,
                   const snn_system_state_t *sys_state,
                   const snn_ai_feedback_t *feedback)
{
    snn_workload_features_t features;
    s64 reward;
    u32 next_state;
    snn_ai_allocation_t dummy_alloc;

    if (!engine || !engine->initialized || !feedback)
        return -EINVAL;

    if (!engine->learning_enabled)
        return 0;

    /* Calculate reward */
    reward = calculate_reward(feedback);

    /* Update cumulative reward */
    spin_lock(&engine->lock);
    engine->cumulative_reward += reward;
    spin_unlock(&engine->lock);

    /* Get next state */
    characterize_workload(params, &features);
    next_state = discretize_state(sys_state, &features);

    /* Update Q-table */
    update_q_table(engine, engine->current_state, engine->last_action,
                  reward, next_state);

    /* Add to history */
    decode_action(engine->last_action, &dummy_alloc);
    add_to_history(engine, &features, &dummy_alloc, feedback);

    if (feedback->deadline_met)
        atomic64_inc(&engine->successful_decisions);

    atomic64_inc(&engine->learning_iterations);

    pr_debug("SNN_AI: Feedback processed: reward=%lld, deadline_met=%u\n",
             reward, feedback->deadline_met);

    return 0;
}

/*
 * Get AI statistics
 */
void snn_ai_get_stats(struct snn_ai_engine *engine, snn_ai_stats_t *stats)
{
    u64 total, successful;

    if (!engine || !stats)
        return;

    total = atomic64_read(&engine->total_decisions);
    successful = atomic64_read(&engine->successful_decisions);

    stats->total_decisions = total;
    stats->successful_decisions = successful;
    stats->learning_iterations = atomic64_read(&engine->learning_iterations);

    if (total > 0) {
        stats->average_reward = (float)engine->cumulative_reward / total / 1000.0f;
    } else {
        stats->average_reward = 0.0f;
    }

    stats->model_version = 1;
    stats->confidence_threshold = engine->config.confidence_threshold;
}

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("SNN Semantic AI Engine");
