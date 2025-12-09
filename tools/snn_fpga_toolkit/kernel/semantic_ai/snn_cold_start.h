/*
 * Cold-Start Safety Mechanisms
 *
 * Ensures safe operation during AI engine warm-up period when:
 * - Q-table is not yet trained
 * - System state is uncertain
 * - No historical data available
 *
 * Strategy:
 * 1. Conservative fallback policies
 * 2. Gradual transition from heuristics to learned policy
 * 3. Safety constraints on action selection
 * 4. Confidence-based decision making
 */

#ifndef _SNN_COLD_START_H
#define _SNN_COLD_START_H

#include <linux/types.h>
#include "snn_fixed_point.h"
#include "../../include/snn_kernel/semantic_ai.h"

/*
 * Cold-start phase tracking
 */
enum snn_cold_start_phase {
	SNN_COLD_BOOTSTRAP = 0,    /* Pure heuristics (0-100 decisions) */
	SNN_COLD_WARMUP,           /* Mix heuristics + learned (100-1000) */
	SNN_COLD_TRANSITION,       /* Mostly learned (1000-5000) */
	SNN_COLD_TRAINED,          /* Fully learned (>5000) */
};

/*
 * Fallback policy types
 */
enum snn_fallback_policy {
	SNN_FALLBACK_BALANCED = 0,  /* 50/50 GPU/FPGA split */
	SNN_FALLBACK_CONSERVATIVE,  /* Prefer CPU (safest) */
	SNN_FALLBACK_HEURISTIC,     /* Simple workload-based rules */
	SNN_FALLBACK_RANDOM,        /* Random exploration */
};

/*
 * Cold-start state
 */
struct snn_cold_start {
	/* Phase tracking */
	enum snn_cold_start_phase phase;
	u64 total_decisions;
	u64 phase_threshold[4];

	/* Fallback policy */
	enum snn_fallback_policy fallback_type;

	/* Confidence tracking */
	fp_t policy_confidence;     /* 0.0 = no confidence, 1.0 = full confidence */
	fp_t min_confidence;        /* Minimum confidence for learned policy */

	/* Safety constraints */
	bool enforce_constraints;
	u32 max_gpu_allocation;     /* Max % GPU allocation during bootstrap */
	u32 max_fpga_allocation;    /* Max % FPGA allocation during bootstrap */

	/* Statistics */
	atomic64_t fallback_decisions;
	atomic64_t learned_decisions;
	atomic64_t constraint_violations;
};

/*
 * Initialize cold-start system
 */
void snn_cold_start_init(struct snn_cold_start *cs);

/*
 * Get current cold-start phase
 */
enum snn_cold_start_phase snn_cold_start_get_phase(struct snn_cold_start *cs);

/*
 * Update cold-start state after decision
 */
void snn_cold_start_update(struct snn_cold_start *cs, fp_t q_value_variance);

/*
 * Check if should use fallback policy
 *
 * Returns true if system should use fallback instead of learned policy
 */
bool snn_cold_start_should_use_fallback(struct snn_cold_start *cs);

/*
 * Generate fallback allocation
 *
 * Returns conservative allocation when Q-table not trained
 */
void snn_cold_start_fallback_allocation(struct snn_cold_start *cs,
                                       const snn_compute_params_t *params,
                                       const snn_workload_features_t *features,
                                       snn_ai_allocation_t *allocation);

/*
 * Apply safety constraints to allocation
 *
 * Ensures allocation doesn't violate safety bounds during cold-start
 */
void snn_cold_start_apply_constraints(struct snn_cold_start *cs,
                                     snn_ai_allocation_t *allocation);

/*
 * Calculate policy confidence
 *
 * Based on Q-value variance and training iterations
 */
fp_t snn_cold_start_calc_confidence(u64 training_iterations,
                                   fp_t q_value_variance);

/*
 * Get cold-start statistics
 */
void snn_cold_start_get_stats(struct snn_cold_start *cs,
                              u64 *fallback_decisions,
                              u64 *learned_decisions,
                              u64 *constraint_violations);

/*
 * Heuristic allocation policies
 */

/* Simple workload-based heuristic */
static inline void heuristic_workload_based(const snn_workload_features_t *features,
                                           snn_ai_allocation_t *allocation)
{
	/* Dense workloads → GPU */
	if (features->sparsity < 0.3f) {
		allocation->use_gpu = 80;
		allocation->use_fpga = 10;
		allocation->batch_size = 64;
	}
	/* Sparse workloads → FPGA */
	else if (features->sparsity > 0.7f) {
		allocation->use_gpu = 20;
		allocation->use_fpga = 70;
		allocation->batch_size = 32;
	}
	/* Medium sparsity → Balanced */
	else {
		allocation->use_gpu = 50;
		allocation->use_fpga = 40;
		allocation->batch_size = 48;
	}

	allocation->cpu_neurons = 0;  /* Calculated later */
	allocation->confidence = SNN_AI_CONFIDENCE_LOW;
}

/* Balanced 50/50 allocation */
static inline void heuristic_balanced(snn_ai_allocation_t *allocation)
{
	allocation->use_gpu = 50;
	allocation->use_fpga = 50;
	allocation->batch_size = 32;
	allocation->cpu_neurons = 0;
	allocation->confidence = SNN_AI_CONFIDENCE_LOW;
}

/* Conservative CPU-heavy allocation */
static inline void heuristic_conservative(snn_ai_allocation_t *allocation)
{
	allocation->use_gpu = 20;
	allocation->use_fpga = 20;
	allocation->batch_size = 16;
	allocation->cpu_neurons = 0;  /* Will get remaining neurons */
	allocation->confidence = SNN_AI_CONFIDENCE_LOW;
}

#endif /* _SNN_COLD_START_H */
