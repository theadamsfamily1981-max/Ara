#!/usr/bin/env python3
"""
A-KTP Dynamic Reward Shaping Engine (DRSE)
===========================================

Shapes rewards for MORL (Multi-Objective RL) based on constraints
discovered through adversarial debate.

The DRSE module:
1. Takes refined constraints from ACT
2. Converts constraints to reward shaping functions
3. Provides policy update signals for learning
4. Tracks PTE (Progress Toward Ethical) convergence

Key formula:
    R_shaped = R_base + γ * F(ethics, verification)

Where:
- R_base is the original task reward
- γ is the shaping coefficient (from debate)
- F is the constraint satisfaction function
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable

from .constraints import ConstraintSet, Constraint, ConstraintType


@dataclass
class RewardShapingConfig:
    """Configuration for DRSE."""
    # Base weights for constraint types
    base_weights: Dict[str, float] = field(default_factory=lambda: {
        "verification": 0.6,
        "ethical": 0.5,
        "safety": 0.8,
        "performance": 0.4,
        "cost": 0.3,
        "timeline": 0.3,
        "hypothetical": -0.3,  # Penalty
        "bias": -0.5,          # Penalty
    })

    # MORL parameters
    gamma: float = 0.7                    # Shaping coefficient
    ethical_threshold: float = 0.8        # Min ethical score to proceed
    pte_convergence: float = 1.8          # PTE threshold for convergence

    # Learning parameters
    learning_rate: float = 0.1
    discount_factor: float = 0.99


@dataclass
class ShapedReward:
    """A shaped reward with component breakdown."""
    reward_id: str
    base_reward: float
    shaped_reward: float

    # Components
    ethical_component: float = 0.0
    verification_component: float = 0.0
    performance_component: float = 0.0
    cost_component: float = 0.0

    # Penalties
    hypothetical_penalty: float = 0.0
    bias_penalty: float = 0.0

    # Shaping coefficient used
    gamma: float = 0.7

    # PTE contribution
    pte_delta: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "reward_id": self.reward_id,
            "base": self.base_reward,
            "shaped": self.shaped_reward,
            "components": {
                "ethical": self.ethical_component,
                "verification": self.verification_component,
                "performance": self.performance_component,
                "cost": self.cost_component,
            },
            "penalties": {
                "hypothetical": self.hypothetical_penalty,
                "bias": self.bias_penalty,
            },
            "gamma": self.gamma,
            "pte_delta": self.pte_delta,
        }


@dataclass
class PolicyUpdate:
    """Update signal for the policy."""
    update_id: str
    constraint_weights: Dict[str, float]
    shaped_rewards: List[ShapedReward] = field(default_factory=list)

    # Convergence tracking
    pte: float = 0.0                      # Progress Toward Ethical
    converged: bool = False
    iteration: int = 0

    # Meta-learning updates
    weight_deltas: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "update_id": self.update_id,
            "constraint_weights": self.constraint_weights,
            "pte": self.pte,
            "converged": self.converged,
            "iteration": self.iteration,
            "weight_deltas": self.weight_deltas,
            "n_rewards": len(self.shaped_rewards),
        }


class DynamicRewardShaper:
    """
    Shapes rewards based on constraints from adversarial debate.

    The key insight: constraints discovered through debate should
    influence the reward function, not just be checked post-hoc.

    Usage:
        drse = DynamicRewardShaper()
        update = drse.shape_rewards(
            constraint_set,
            base_rewards=[0.5, 0.7, 0.3],
            iteration=1
        )
    """

    def __init__(self, config: RewardShapingConfig = None):
        self.config = config or RewardShapingConfig()
        self.pte_history: List[float] = []
        self.update_history: List[PolicyUpdate] = []

        # Current constraint weights (meta-learned)
        self.constraint_weights = self.config.base_weights.copy()

    def shape_rewards(self,
                      constraint_set: ConstraintSet,
                      base_rewards: List[float],
                      iteration: int = 0) -> PolicyUpdate:
        """
        Shape rewards based on constraints.

        Args:
            constraint_set: Refined constraints from ACT
            base_rewards: Original task rewards
            iteration: Current iteration (for PTE calculation)

        Returns:
            PolicyUpdate with shaped rewards and weight updates
        """
        update = PolicyUpdate(
            update_id=f"update_{constraint_set.set_id}_{iteration}",
            constraint_weights=self.constraint_weights.copy(),
            iteration=iteration,
        )

        # Get refined weights from constraint set
        refined_weights = self._compute_constraint_weights(constraint_set)

        # Shape each reward
        for i, base_r in enumerate(base_rewards):
            shaped = self._shape_single(
                base_r,
                constraint_set,
                refined_weights,
                reward_id=f"r_{i}",
            )
            update.shaped_rewards.append(shaped)

        # Compute PTE
        update.pte = self._compute_pte(update.shaped_rewards, iteration)
        self.pte_history.append(update.pte)

        # Check convergence
        update.converged = update.pte >= self.config.pte_convergence

        # Meta-learn: update weights based on this iteration
        update.weight_deltas = self._meta_update(constraint_set, update)
        self._apply_weight_updates(update.weight_deltas)

        self.update_history.append(update)
        return update

    def _compute_constraint_weights(self, cs: ConstraintSet) -> Dict[str, float]:
        """Compute effective weights from constraint set."""
        weights = {}

        for c in cs.constraints:
            type_key = c.type.value
            if type_key not in weights:
                weights[type_key] = 0.0

            # Combine constraint weight with confidence
            effective_weight = c.weight * c.confidence

            # Apply debate outcomes
            support_boost = len(c.supported_by) * 0.05
            challenge_penalty = len(c.challenged_by) * 0.03

            effective_weight *= (1 + support_boost - challenge_penalty)
            weights[type_key] += effective_weight

        return weights

    def _shape_single(self,
                      base_reward: float,
                      cs: ConstraintSet,
                      weights: Dict[str, float],
                      reward_id: str) -> ShapedReward:
        """Shape a single reward."""
        shaped = ShapedReward(
            reward_id=reward_id,
            base_reward=base_reward,
            shaped_reward=base_reward,
            gamma=self.config.gamma,
        )

        # Compute F(ethics, verification)
        f_ethical = weights.get("ethical", 0) * cs.ethical_score
        f_verification = weights.get("verification", 0)
        f_performance = weights.get("performance", 0)
        f_cost = weights.get("cost", 0)

        shaped.ethical_component = f_ethical
        shaped.verification_component = f_verification
        shaped.performance_component = f_performance
        shaped.cost_component = f_cost

        # Compute penalties
        if cs.has_hypothetical:
            shaped.hypothetical_penalty = abs(self.config.base_weights.get("hypothetical", -0.3))

        if cs.has_bias_warnings:
            shaped.bias_penalty = abs(self.config.base_weights.get("bias", -0.5))

        # Final shaping: R_shaped = R_base + γ * F - penalties
        f_total = f_ethical + f_verification + f_performance - f_cost
        penalty_total = shaped.hypothetical_penalty + shaped.bias_penalty

        shaped.shaped_reward = (
            base_reward +
            self.config.gamma * f_total -
            penalty_total
        )

        # PTE delta for this reward
        shaped.pte_delta = self.config.gamma * f_total * 0.1

        return shaped

    def _compute_pte(self, rewards: List[ShapedReward], iteration: int) -> float:
        """
        Compute Progress Toward Ethical (PTE).

        PTE scales from 0.2 to ~1.8 per cycle, total 5.4 over 3 cycles.
        """
        if not rewards:
            return 0.2

        # Base PTE from iteration progress
        base_pte = 0.2 + (0.32 * iteration / 5.0)

        # Bonus from ethical components
        ethical_avg = np.mean([r.ethical_component for r in rewards])
        ethical_bonus = ethical_avg * 0.3

        # Penalty from issues
        penalty_avg = np.mean([r.hypothetical_penalty + r.bias_penalty for r in rewards])
        penalty = penalty_avg * 0.2

        pte = base_pte + ethical_bonus - penalty
        return min(1.8, max(0.0, pte))  # Cap at 1.8 per cycle

    def _meta_update(self, cs: ConstraintSet, update: PolicyUpdate) -> Dict[str, float]:
        """
        Meta-learn: update constraint weights based on outcomes.

        If ethical score is high and PTE is progressing, increase ethical weight.
        If hypothetical flags appear, decrease confidence in those areas.
        """
        deltas = {}
        lr = self.config.learning_rate

        # Ethical boosting
        if cs.ethical_score > 0.8 and update.pte > 0.5:
            deltas["ethical"] = lr * 0.1

        # Verification importance based on debate consensus
        if cs.consensus_score > 0.7:
            deltas["verification"] = lr * 0.05

        # Hypothetical penalty increase if flagged
        if cs.has_hypothetical:
            deltas["hypothetical"] = -lr * 0.1  # More negative

        # Bias penalty increase if flagged
        if cs.has_bias_warnings:
            deltas["bias"] = -lr * 0.15  # More negative

        return deltas

    def _apply_weight_updates(self, deltas: Dict[str, float]):
        """Apply meta-learned weight updates."""
        for key, delta in deltas.items():
            if key in self.constraint_weights:
                self.constraint_weights[key] += delta
                # Clamp weights
                if "penalty" in key or self.config.base_weights.get(key, 0) < 0:
                    self.constraint_weights[key] = min(0, self.constraint_weights[key])
                else:
                    self.constraint_weights[key] = max(0, min(1, self.constraint_weights[key]))

    def get_total_pte(self) -> float:
        """Get total PTE across all cycles."""
        return sum(self.pte_history)

    def is_converged(self) -> bool:
        """Check if we've reached ethical convergence."""
        if not self.pte_history:
            return False
        return self.pte_history[-1] >= self.config.pte_convergence

    def get_convergence_report(self) -> Dict[str, Any]:
        """Get a report on convergence status."""
        return {
            "total_pte": self.get_total_pte(),
            "cycles_completed": len(self.pte_history),
            "converged": self.is_converged(),
            "final_weights": self.constraint_weights.copy(),
            "pte_trajectory": self.pte_history.copy(),
            "ethical_threshold": self.config.ethical_threshold,
        }


# Convenience functions
def shape_rewards(constraint_set: ConstraintSet,
                  base_rewards: List[float],
                  iteration: int = 0) -> PolicyUpdate:
    """Quick reward shaping."""
    drse = DynamicRewardShaper()
    return drse.shape_rewards(constraint_set, base_rewards, iteration)
