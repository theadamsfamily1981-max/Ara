/*
 * Quantization for Q-Learning
 *
 * INT8 quantization for Q-tables provides:
 * - 8x memory reduction (s64 -> s8)
 * - Faster cache performance
 * - Lower memory bandwidth
 * - Nanosecond-level lookups
 *
 * Quantization formula:
 * q = clamp(round(x / scale) + zero_point, -128, 127)
 *
 * Dequantization:
 * x = (q - zero_point) * scale
 */

#ifndef _SNN_QUANTIZATION_H
#define _SNN_QUANTIZATION_H

#include <linux/types.h>
#include "snn_fixed_point.h"
#include "snn_ai_internal.h"

/*
 * Quantized Q-table using INT8
 * Memory: 512 states * 128 actions * 1 byte = 64 KB
 * vs. Original: 512 * 128 * 8 bytes = 512 KB (8x reduction!)
 */
struct snn_q_table_quantized {
    /* Quantized values: -128 to 127 */
    s8 values[SNN_AI_STATE_SPACE_SIZE][SNN_AI_ACTION_SPACE_SIZE];

    /* Quantization parameters (fixed-point) */
    fp_t scale;           /* Scaling factor */
    s8 zero_point;        /* Zero point (typically 0 for symmetric) */

    /* Statistics for adaptive quantization */
    fp_t min_q_value;     /* Minimum Q-value seen */
    fp_t max_q_value;     /* Maximum Q-value seen */
    u64 updates;          /* Number of updates */

    /* Calibration flag */
    bool calibrated;
} __attribute__((aligned(64)));  /* Cache-line aligned */

/*
 * Initialize quantized Q-table
 */
static inline void snn_qtable_init(struct snn_q_table_quantized *qtable)
{
    memset(qtable, 0, sizeof(*qtable));

    /* Initial calibration values (will be refined) */
    qtable->scale = FP_FROM_INT(10);  /* Initial scale: 10.0 */
    qtable->zero_point = 0;            /* Symmetric quantization */
    qtable->min_q_value = 0;
    qtable->max_q_value = 0;
    qtable->updates = 0;
    qtable->calibrated = false;
}

/*
 * Calibrate quantization parameters based on Q-value distribution
 * Should be called periodically during learning
 */
static inline void snn_qtable_calibrate(struct snn_q_table_quantized *qtable,
                                       fp_t *full_precision_values,
                                       u32 num_values)
{
    fp_t min_val = full_precision_values[0];
    fp_t max_val = full_precision_values[0];
    u32 i;

    /* Find min and max */
    for (i = 1; i < num_values; i++) {
        if (full_precision_values[i] < min_val)
            min_val = full_precision_values[i];
        if (full_precision_values[i] > max_val)
            max_val = full_precision_values[i];
    }

    qtable->min_q_value = min_val;
    qtable->max_q_value = max_val;

    /* Calculate scale: (max - min) / 255
     * We use 255 instead of 256 to leave room at boundaries
     */
    fp_t range = max_val - min_val;
    if (range == 0)
        range = FP_ONE;  /* Avoid division by zero */

    qtable->scale = fp_div(range, FP_FROM_INT(255));

    /* For symmetric quantization, zero_point = 0
     * For asymmetric, zero_point = round(-min / scale)
     */
    qtable->zero_point = 0;

    qtable->calibrated = true;

    pr_debug("SNN_QUANT: Calibrated - min=%d.%02d, max=%d.%02d, scale=%d.%02d\n",
             FP_TO_INT(min_val), (fp_abs(min_val & (FP_ONE-1)) * 100) >> FP_SHIFT,
             FP_TO_INT(max_val), (fp_abs(max_val & (FP_ONE-1)) * 100) >> FP_SHIFT,
             FP_TO_INT(qtable->scale), (fp_abs(qtable->scale & (FP_ONE-1)) * 100) >> FP_SHIFT);
}

/*
 * Set quantized Q-value (quantize on write)
 */
static inline void snn_qtable_set(struct snn_q_table_quantized *qtable,
                                 u32 state, u32 action, fp_t value)
{
    s8 q_value;

    /* Update min/max for adaptive quantization */
    if (!qtable->calibrated || value < qtable->min_q_value)
        qtable->min_q_value = value;
    if (!qtable->calibrated || value > qtable->max_q_value)
        qtable->max_q_value = value;

    /* Quantize: q = round(value / scale) + zero_point */
    q_value = fp_quantize_s8(value, qtable->scale, qtable->zero_point);

    qtable->values[state][action] = q_value;
    qtable->updates++;

    /* Recalibrate every 1000 updates for adaptive quantization */
    if (qtable->updates % 1000 == 0 && qtable->updates > 0) {
        /* Collect samples for recalibration */
        /* (In full implementation, maintain running statistics) */
        fp_t range = qtable->max_q_value - qtable->min_q_value;
        if (range > 0)
            qtable->scale = fp_div(range, FP_FROM_INT(255));
    }
}

/*
 * Get Q-value (dequantize on read)
 * Ultra-fast: single load + multiply
 */
static inline fp_t snn_qtable_get(const struct snn_q_table_quantized *qtable,
                                 u32 state, u32 action)
{
    s8 q_value = qtable->values[state][action];
    return fp_dequantize_s8(q_value, qtable->scale, qtable->zero_point);
}

/*
 * Get all Q-values for a state (for softmax)
 * Returns array of dequantized values
 */
static inline void snn_qtable_get_state(const struct snn_q_table_quantized *qtable,
                                       u32 state, fp_t *q_values, u32 num_actions)
{
    u32 i;

    for (i = 0; i < num_actions; i++) {
        s8 q_val = qtable->values[state][i];
        q_values[i] = fp_dequantize_s8(q_val, qtable->scale, qtable->zero_point);
    }
}

/*
 * Find best action (argmax) without full dequantization
 * Since scale and zero_point are constant, we can compare quantized values directly!
 */
static inline u32 snn_qtable_argmax(const struct snn_q_table_quantized *qtable,
                                   u32 state)
{
    u32 best_action = 0;
    s8 best_q = qtable->values[state][0];
    u32 i;

    for (i = 1; i < SNN_AI_ACTION_SPACE_SIZE; i++) {
        if (qtable->values[state][i] > best_q) {
            best_q = qtable->values[state][i];
            best_action = i;
        }
    }

    return best_action;
}

/*
 * Update Q-value with Q-learning rule (in fixed-point)
 * Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
 *
 * This operates in fixed-point space for maximum efficiency
 */
static inline void snn_qtable_update(struct snn_q_table_quantized *qtable,
                                    u32 state, u32 action,
                                    fp_t reward, u32 next_state,
                                    fp_t alpha, fp_t gamma)
{
    /* Get current Q-value */
    fp_t old_q = snn_qtable_get(qtable, state, action);

    /* Find max Q-value for next state */
    fp_t max_next_q = FP_FROM_INT(-10000);  /* Very negative initial value */
    u32 i;

    for (i = 0; i < SNN_AI_ACTION_SPACE_SIZE; i++) {
        fp_t q = snn_qtable_get(qtable, next_state, i);
        if (q > max_next_q)
            max_next_q = q;
    }

    /* Q-learning update: Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)] */
    fp_t target = reward + fp_mul(gamma, max_next_q);
    fp_t td_error = target - old_q;
    fp_t delta = fp_mul(alpha, td_error);
    fp_t new_q = old_q + delta;

    /* Write back quantized value */
    snn_qtable_set(qtable, state, action, new_q);

    pr_debug("SNN_QUANT: Update s=%u a=%u: old=%d.%02d new=%d.%02d reward=%d.%02d\n",
             state, action,
             FP_TO_INT(old_q), (fp_abs(old_q & (FP_ONE-1)) * 100) >> FP_SHIFT,
             FP_TO_INT(new_q), (fp_abs(new_q & (FP_ONE-1)) * 100) >> FP_SHIFT,
             FP_TO_INT(reward), (fp_abs(reward & (FP_ONE-1)) * 100) >> FP_SHIFT);
}

/*
 * Compressed LUT for sparse Q-tables (optional optimization)
 *
 * For very sparse state spaces, we can store only the most probable
 * state-action pairs, achieving 100x+ memory reduction
 */

#define CLUT_CAPACITY 8192  /* Store top 8K entries (8 KB for values + overhead) */

struct clut_entry {
    u32 state : 20;   /* Up to 1M states */
    u32 action : 8;   /* Up to 256 actions */
    s8 q_value;       /* Quantized Q-value */
    u32 access_count; /* For LRU eviction */
    u32 hash;         /* For fast lookup */
} __attribute__((packed));

struct compressed_lut {
    struct clut_entry *entries;
    u32 capacity;
    u32 used;
    s8 default_q_value;  /* For unpopulated states */

    /* Hash table for O(1) lookup */
    u32 *hash_table;
    u32 hash_table_size;

    /* Statistics */
    atomic64_t hits;
    atomic64_t misses;

    spinlock_t lock;
};

/*
 * Initialize compressed LUT
 */
static inline int snn_clut_init(struct compressed_lut **clut_ptr)
{
    struct compressed_lut *clut;

    clut = kzalloc(sizeof(*clut), GFP_KERNEL);
    if (!clut)
        return -ENOMEM;

    clut->entries = kzalloc(CLUT_CAPACITY * sizeof(struct clut_entry), GFP_KERNEL);
    if (!clut->entries) {
        kfree(clut);
        return -ENOMEM;
    }

    clut->hash_table_size = CLUT_CAPACITY * 2;  /* 2x for lower collision */
    clut->hash_table = kzalloc(clut->hash_table_size * sizeof(u32), GFP_KERNEL);
    if (!clut->hash_table) {
        kfree(clut->entries);
        kfree(clut);
        return -ENOMEM;
    }

    clut->capacity = CLUT_CAPACITY;
    clut->used = 0;
    clut->default_q_value = 0;

    atomic64_set(&clut->hits, 0);
    atomic64_set(&clut->misses, 0);

    spin_lock_init(&clut->lock);

    *clut_ptr = clut;

    pr_info("SNN_CLUT: Initialized with capacity %u (memory: %zu KB)\n",
            CLUT_CAPACITY,
            (CLUT_CAPACITY * sizeof(struct clut_entry)) / 1024);

    return 0;
}

/*
 * Cleanup compressed LUT
 */
static inline void snn_clut_cleanup(struct compressed_lut *clut)
{
    if (!clut)
        return;

    kfree(clut->hash_table);
    kfree(clut->entries);
    kfree(clut);
}

/*
 * Hash function for state-action pair
 */
static inline u32 snn_clut_hash(u32 state, u32 action)
{
    /* Simple multiplicative hash */
    u32 combined = (state << 8) | action;
    return hash_32(combined, 16);
}

/*
 * Lookup Q-value in compressed LUT
 * Returns default value if not found
 * Latency: O(1) average, ~10-20 ns
 */
static inline s8 snn_clut_lookup(struct compressed_lut *clut,
                                u32 state, u32 action)
{
    u32 hash = snn_clut_hash(state, action);
    u32 idx = hash % clut->hash_table_size;
    u32 probe_count = 0;

    /* Linear probing (max 16 probes) */
    while (probe_count < 16) {
        u32 entry_idx = clut->hash_table[idx];

        if (entry_idx == 0) {
            /* Empty slot - not found */
            atomic64_inc(&clut->misses);
            return clut->default_q_value;
        }

        entry_idx--;  /* Adjust for 1-based indexing */

        if (clut->entries[entry_idx].state == state &&
            clut->entries[entry_idx].action == action) {
            /* Found! */
            clut->entries[entry_idx].access_count++;
            atomic64_inc(&clut->hits);
            return clut->entries[entry_idx].q_value;
        }

        idx = (idx + 1) % clut->hash_table_size;
        probe_count++;
    }

    atomic64_inc(&clut->misses);
    return clut->default_q_value;
}

/*
 * Insert/update entry in compressed LUT
 */
static inline int snn_clut_insert(struct compressed_lut *clut,
                                 u32 state, u32 action, s8 q_value)
{
    u32 hash = snn_clut_hash(state, action);
    u32 idx = hash % clut->hash_table_size;
    u32 entry_idx;

    spin_lock(&clut->lock);

    /* Find slot using linear probing */
    for (u32 probe = 0; probe < 16; probe++) {
        entry_idx = clut->hash_table[idx];

        if (entry_idx == 0) {
            /* Empty slot - insert new entry */
            if (clut->used >= clut->capacity) {
                spin_unlock(&clut->lock);
                return -ENOMEM;  /* Table full - need LRU eviction */
            }

            entry_idx = clut->used + 1;  /* 1-based indexing */
            clut->entries[clut->used].state = state;
            clut->entries[clut->used].action = action;
            clut->entries[clut->used].q_value = q_value;
            clut->entries[clut->used].access_count = 1;
            clut->entries[clut->used].hash = hash;

            clut->hash_table[idx] = entry_idx;
            clut->used++;

            spin_unlock(&clut->lock);
            return 0;
        }

        entry_idx--;  /* Adjust for 1-based */

        if (clut->entries[entry_idx].state == state &&
            clut->entries[entry_idx].action == action) {
            /* Update existing entry */
            clut->entries[entry_idx].q_value = q_value;
            clut->entries[entry_idx].access_count++;
            spin_unlock(&clut->lock);
            return 0;
        }

        idx = (idx + 1) % clut->hash_table_size;
    }

    spin_unlock(&clut->lock);
    return -ENOSPC;  /* Collision chain too long */
}

#endif /* _SNN_QUANTIZATION_H */
