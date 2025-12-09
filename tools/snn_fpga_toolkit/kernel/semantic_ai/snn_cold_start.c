/*
 * Cold-Start Safety Implementation
 *
 * Provides safe operation during AI engine warm-up
 */

#include <linux/kernel.h>
#include <linux/random.h>
#include "snn_cold_start.h"

/*
 * Initialize cold-start system
 */
void snn_cold_start_init(struct snn_cold_start *cs)
{
	cs->phase = SNN_COLD_BOOTSTRAP;
	cs->total_decisions = 0;

	/* Phase thresholds */
	cs->phase_threshold[SNN_COLD_BOOTSTRAP] = 100;      /* 0-100 decisions */
	cs->phase_threshold[SNN_COLD_WARMUP] = 1000;        /* 100-1000 */
	cs->phase_threshold[SNN_COLD_TRANSITION] = 5000;    /* 1000-5000 */
	cs->phase_threshold[SNN_COLD_TRAINED] = UINT64_MAX; /* >5000 */

	/* Use heuristic fallback by default */
	cs->fallback_type = SNN_FALLBACK_HEURISTIC;

	/* Start with zero confidence */
	cs->policy_confidence = 0;
	cs->min_confidence = FP_FROM_FLOAT(0.5f);  /* Need 50% confidence to use learned policy */

	/* Enable safety constraints during bootstrap */
	cs->enforce_constraints = true;
	cs->max_gpu_allocation = 60;   /* Max 60% to GPU during bootstrap */
	cs->max_fpga_allocation = 60;  /* Max 60% to FPGA during bootstrap */

	atomic64_set(&cs->fallback_decisions, 0);
	atomic64_set(&cs->learned_decisions, 0);
	atomic64_set(&cs->constraint_violations, 0);

	pr_info("SNN_COLD_START: Initialized (phase=BOOTSTRAP, fallback=HEURISTIC)\n");
}

/*
 * Get current cold-start phase
 */
enum snn_cold_start_phase snn_cold_start_get_phase(struct snn_cold_start *cs)
{
	u64 decisions = cs->total_decisions;

	if (decisions < cs->phase_threshold[SNN_COLD_BOOTSTRAP])
		return SNN_COLD_BOOTSTRAP;
	else if (decisions < cs->phase_threshold[SNN_COLD_WARMUP])
		return SNN_COLD_WARMUP;
	else if (decisions < cs->phase_threshold[SNN_COLD_TRANSITION])
		return SNN_COLD_TRANSITION;
	else
		return SNN_COLD_TRAINED;
}

/*
 * Calculate policy confidence
 */
fp_t snn_cold_start_calc_confidence(u64 training_iterations,
                                   fp_t q_value_variance)
{
	fp_t confidence;

	/* Confidence increases with training iterations */
	fp_t iteration_factor;
	if (training_iterations < 100) {
		iteration_factor = fp_div(FP_FROM_INT(training_iterations), FP_FROM_INT(100));
	} else if (training_iterations < 1000) {
		iteration_factor = FP_FROM_FLOAT(0.5f) +
		                  fp_div(FP_FROM_INT(training_iterations - 100), FP_FROM_INT(1800));
	} else {
		iteration_factor = FP_FROM_FLOAT(0.9f);
	}

	/* Confidence decreases with high Q-value variance (unstable learning) */
	fp_t variance_factor;
	if (q_value_variance < FP_FROM_FLOAT(0.01f)) {
		variance_factor = FP_ONE;  /* Low variance = high confidence */
	} else if (q_value_variance < FP_FROM_FLOAT(0.1f)) {
		variance_factor = FP_FROM_FLOAT(0.7f);
	} else {
		variance_factor = FP_FROM_FLOAT(0.3f);  /* High variance = low confidence */
	}

	/* Combine factors */
	confidence = fp_mul(iteration_factor, variance_factor);

	/* Clamp to [0, 1] */
	if (confidence < 0)
		confidence = 0;
	if (confidence > FP_ONE)
		confidence = FP_ONE;

	return confidence;
}

/*
 * Update cold-start state after decision
 */
void snn_cold_start_update(struct snn_cold_start *cs, fp_t q_value_variance)
{
	enum snn_cold_start_phase old_phase, new_phase;

	cs->total_decisions++;

	old_phase = cs->phase;
	new_phase = snn_cold_start_get_phase(cs);

	/* Update phase if changed */
	if (new_phase != old_phase) {
		cs->phase = new_phase;
		pr_info("SNN_COLD_START: Phase transition %u -> %u (decisions=%llu)\n",
		        old_phase, new_phase, cs->total_decisions);

		/* Adjust constraints as we progress */
		switch (new_phase) {
		case SNN_COLD_WARMUP:
			cs->max_gpu_allocation = 80;
			cs->max_fpga_allocation = 80;
			break;
		case SNN_COLD_TRANSITION:
			cs->max_gpu_allocation = 100;
			cs->max_fpga_allocation = 100;
			cs->enforce_constraints = false;  /* Disable constraints */
			break;
		case SNN_COLD_TRAINED:
			/* Fully trained - no restrictions */
			break;
		default:
			break;
		}
	}

	/* Update confidence (based on training iterations and variance) */
	cs->policy_confidence = snn_cold_start_calc_confidence(cs->total_decisions,
	                                                      q_value_variance);
}

/*
 * Check if should use fallback policy
 */
bool snn_cold_start_should_use_fallback(struct snn_cold_start *cs)
{
	/* Always use fallback during bootstrap */
	if (cs->phase == SNN_COLD_BOOTSTRAP)
		return true;

	/* Use fallback if confidence too low */
	if (cs->policy_confidence < cs->min_confidence)
		return true;

	/* During warmup, use fallback 50% of the time */
	if (cs->phase == SNN_COLD_WARMUP) {
		u32 random_val;
		get_random_bytes(&random_val, sizeof(random_val));
		return (random_val % 100) < 50;
	}

	/* During transition, use fallback 20% of the time */
	if (cs->phase == SNN_COLD_TRANSITION) {
		u32 random_val;
		get_random_bytes(&random_val, sizeof(random_val));
		return (random_val % 100) < 20;
	}

	/* Fully trained - don't use fallback */
	return false;
}

/*
 * Generate fallback allocation
 */
void snn_cold_start_fallback_allocation(struct snn_cold_start *cs,
                                       const snn_compute_params_t *params,
                                       const snn_workload_features_t *features,
                                       snn_ai_allocation_t *allocation)
{
	switch (cs->fallback_type) {
	case SNN_FALLBACK_HEURISTIC:
		heuristic_workload_based(features, allocation);
		break;

	case SNN_FALLBACK_BALANCED:
		heuristic_balanced(allocation);
		break;

	case SNN_FALLBACK_CONSERVATIVE:
		heuristic_conservative(allocation);
		break;

	case SNN_FALLBACK_RANDOM:
		/* Random allocation for exploration */
		{
			u32 random_val;
			get_random_bytes(&random_val, sizeof(random_val));
			allocation->use_gpu = random_val % 100;

			get_random_bytes(&random_val, sizeof(random_val));
			allocation->use_fpga = random_val % (100 - allocation->use_gpu);

			allocation->batch_size = 32;
			allocation->confidence = SNN_AI_CONFIDENCE_LOW;
		}
		break;
	}

	/* Apply constraints */
	snn_cold_start_apply_constraints(cs, allocation);

	atomic64_inc(&cs->fallback_decisions);

	pr_debug("SNN_COLD_START: Fallback allocation (type=%u, GPU=%u%%, FPGA=%u%%)\n",
	         cs->fallback_type, allocation->use_gpu, allocation->use_fpga);
}

/*
 * Apply safety constraints to allocation
 */
void snn_cold_start_apply_constraints(struct snn_cold_start *cs,
                                     snn_ai_allocation_t *allocation)
{
	bool violated = false;

	if (!cs->enforce_constraints)
		return;

	/* Constrain GPU allocation */
	if (allocation->use_gpu > cs->max_gpu_allocation) {
		allocation->use_gpu = cs->max_gpu_allocation;
		violated = true;
	}

	/* Constrain FPGA allocation */
	if (allocation->use_fpga > cs->max_fpga_allocation) {
		allocation->use_fpga = cs->max_fpga_allocation;
		violated = true;
	}

	/* Ensure total doesn't exceed 100% */
	if (allocation->use_gpu + allocation->use_fpga > 100) {
		/* Scale proportionally */
		u32 total = allocation->use_gpu + allocation->use_fpga;
		allocation->use_gpu = (allocation->use_gpu * 100) / total;
		allocation->use_fpga = (allocation->use_fpga * 100) / total;
		violated = true;
	}

	if (violated) {
		atomic64_inc(&cs->constraint_violations);
		pr_debug("SNN_COLD_START: Applied constraints (GPU=%u%%, FPGA=%u%%)\n",
		         allocation->use_gpu, allocation->use_fpga);
	}
}

/*
 * Get cold-start statistics
 */
void snn_cold_start_get_stats(struct snn_cold_start *cs,
                              u64 *fallback_decisions,
                              u64 *learned_decisions,
                              u64 *constraint_violations)
{
	*fallback_decisions = atomic64_read(&cs->fallback_decisions);
	*learned_decisions = atomic64_read(&cs->learned_decisions);
	*constraint_violations = atomic64_read(&cs->constraint_violations);
}

MODULE_LICENSE("GPL");
MODULE_AUTHOR("SNN Kernel Team");
MODULE_DESCRIPTION("Cold-Start Safety Mechanisms");
