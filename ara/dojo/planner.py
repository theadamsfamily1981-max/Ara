# ara/dojo/planner.py
"""
Dojo Planner - MPC with Dream Mode
==================================

Model Predictive Control for the Thought Dojo training environment.

This planner enables Ara to:
1. Simulate futures using the world model
2. Evaluate trajectories with MEIS/NIB scoring
3. Enter "dream mode" for offline exploration
4. Plan optimal action sequences

Dream Mode:
- Freeze actuators (no real actions)
- Roll forward many possible futures
- Learn from imagined outcomes
- Discover dangerous regions to avoid

MPC Methods:
- Random Shooting: Sample and pick best
- CEM: Cross-Entropy Method (iterative refinement)
- MPPI: Model Predictive Path Integral (weighted average)

Usage:
    from ara.dojo import DojoPlanner, dream_mode, mpc_plan

    planner = DojoPlanner(world_model, encoder)

    # Standard planning
    action = mpc_plan(planner, z_current, goal)

    # Dream mode exploration
    with dream_mode(planner):
        futures = planner.dream_explore(z_current, n_dreams=10)
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from .world_model import DojoWorldModel
from .encoder import HDCVAE

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class MPCMethod(Enum):
    """Planning algorithm selection."""
    RANDOM_SHOOTING = auto()
    CEM = auto()
    MPPI = auto()


@dataclass
class DojoPlannerConfig:
    """Configuration for the Dojo planner."""
    horizon: int = 10              # Planning horizon (steps)
    n_samples: int = 100           # Candidate sequences
    n_elites: int = 10             # Top candidates for CEM
    n_iterations: int = 5          # CEM/MPPI iterations

    method: MPCMethod = MPCMethod.CEM

    # Action bounds
    action_low: float = -1.0
    action_high: float = 1.0
    action_dim: int = 8

    # Cost weights
    goal_weight: float = 1.0
    smoothness_weight: float = 0.1
    risk_weight: float = 1.0
    curiosity_weight: float = 0.5   # Reward for exploring unknown regions

    # MPPI temperature
    mppi_temperature: float = 1.0

    # Dream mode settings
    dream_noise_scale: float = 0.2  # Exploration noise in dreams
    dream_horizon: int = 20         # Extended horizon for dreams


@dataclass
class DreamResult:
    """Result from a dream exploration."""
    trajectory: np.ndarray          # Imagined states
    actions: np.ndarray             # Actions taken
    total_reward: float             # Accumulated reward
    risk_profile: List[float]       # Risk at each step
    discovery: float                # Novelty score
    terminated_early: bool = False  # Hit forbidden region


@dataclass
class Plan:
    """Result of MPC planning."""
    actions: np.ndarray             # (horizon, action_dim)
    trajectory: np.ndarray          # (horizon+1, latent_dim)
    cost: float
    costs_breakdown: Dict[str, float] = field(default_factory=dict)

    @property
    def first_action(self) -> np.ndarray:
        return self.actions[0]

    @property
    def final_state(self) -> np.ndarray:
        return self.trajectory[-1]


# =============================================================================
# Dojo Planner
# =============================================================================

class DojoPlanner:
    """
    MPC planner for the Thought Dojo.

    Combines world model prediction with MEIS/NIB scoring
    to plan safe, goal-directed trajectories.
    """

    def __init__(
        self,
        world_model: DojoWorldModel,
        encoder: Optional[HDCVAE] = None,
        config: Optional[DojoPlannerConfig] = None,
    ):
        self.world = world_model
        self.encoder = encoder
        self.config = config or DojoPlannerConfig()

        # Infer dimensions
        self.latent_dim = world_model.latent_dim
        self.action_dim = world_model.action_dim
        self.config.action_dim = self.action_dim

        # Risk function (set externally via MEIS/NIB)
        self._risk_fn: Optional[Callable[[np.ndarray], float]] = None

        # Novelty tracker for curiosity-driven exploration
        self._visited_regions: List[np.ndarray] = []

        # Dream mode flag
        self._dreaming = False

        # Warm start from previous plan
        self._prev_actions: Optional[np.ndarray] = None

        logger.info(
            f"DojoPlanner: {self.config.method.name}, "
            f"horizon={self.config.horizon}, samples={self.config.n_samples}"
        )

    def set_risk_function(self, risk_fn: Callable[[np.ndarray], float]) -> None:
        """Set risk scoring function from MEIS/NIB."""
        self._risk_fn = risk_fn

    def _sample_actions(
        self,
        n: int,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Sample action sequences."""
        H = self.config.horizon if not self._dreaming else self.config.dream_horizon
        shape = (n, H, self.action_dim)

        if mean is None:
            actions = np.random.uniform(
                self.config.action_low,
                self.config.action_high,
                shape,
            )
        else:
            if self._dreaming:
                # More exploration noise in dreams
                std = std * (1 + self.config.dream_noise_scale)
            actions = mean + std * np.random.randn(*shape)
            actions = np.clip(
                actions, self.config.action_low, self.config.action_high
            )

        return actions

    def _compute_novelty(self, z: np.ndarray) -> float:
        """Compute novelty score based on visited regions."""
        if not self._visited_regions:
            return 1.0

        distances = [np.linalg.norm(z - v) for v in self._visited_regions]
        return float(min(distances))

    def _evaluate_trajectory(
        self,
        z_init: np.ndarray,
        actions: np.ndarray,
        goal: Optional[np.ndarray],
    ) -> Tuple[float, np.ndarray, Dict[str, float]]:
        """Evaluate a trajectory and compute total cost."""
        # Rollout through world model
        trajectory = self.world.rollout(z_init, actions, include_start=True)

        costs = {}

        # Goal cost
        if goal is not None:
            final_dist = np.sum((trajectory[-1] - goal) ** 2)
            costs["goal"] = self.config.goal_weight * final_dist
        else:
            costs["goal"] = 0.0

        # Smoothness cost
        if len(actions) >= 2:
            diffs = np.diff(actions, axis=0)
            costs["smoothness"] = self.config.smoothness_weight * np.mean(
                np.sum(diffs ** 2, axis=1)
            )
        else:
            costs["smoothness"] = 0.0

        # Risk cost (integral over trajectory)
        if self._risk_fn is not None:
            risk = sum(self._risk_fn(z) for z in trajectory)
            costs["risk"] = self.config.risk_weight * risk
        else:
            costs["risk"] = 0.0

        # Curiosity reward (negative cost for novelty)
        if self._dreaming:
            novelty = sum(self._compute_novelty(z) for z in trajectory)
            costs["curiosity"] = -self.config.curiosity_weight * novelty
        else:
            costs["curiosity"] = 0.0

        total = sum(costs.values())
        return total, trajectory, costs

    def plan(
        self,
        z_current: np.ndarray,
        goal: Optional[np.ndarray] = None,
    ) -> Plan:
        """
        Plan optimal trajectory from current state.

        Args:
            z_current: Current latent state
            goal: Target latent state (optional)

        Returns:
            Plan with actions and trajectory
        """
        z_current = np.asarray(z_current).flatten()

        if self.config.method == MPCMethod.RANDOM_SHOOTING:
            return self._plan_random(z_current, goal)
        elif self.config.method == MPCMethod.CEM:
            return self._plan_cem(z_current, goal)
        elif self.config.method == MPCMethod.MPPI:
            return self._plan_mppi(z_current, goal)
        else:
            raise ValueError(f"Unknown method: {self.config.method}")

    def _plan_random(
        self,
        z_current: np.ndarray,
        goal: Optional[np.ndarray],
    ) -> Plan:
        """Random shooting: sample and pick best."""
        actions_all = self._sample_actions(self.config.n_samples)

        best_cost = float("inf")
        best_actions = None
        best_trajectory = None
        best_breakdown = {}

        for i in range(self.config.n_samples):
            cost, traj, breakdown = self._evaluate_trajectory(
                z_current, actions_all[i], goal
            )
            if cost < best_cost:
                best_cost = cost
                best_actions = actions_all[i]
                best_trajectory = traj
                best_breakdown = breakdown

        return Plan(
            actions=best_actions,
            trajectory=best_trajectory,
            cost=best_cost,
            costs_breakdown=best_breakdown,
        )

    def _plan_cem(
        self,
        z_current: np.ndarray,
        goal: Optional[np.ndarray],
    ) -> Plan:
        """Cross-Entropy Method: iteratively refine distribution."""
        H = self.config.horizon

        # Initialize distribution
        mean = np.zeros((H, self.action_dim))
        std = np.ones((H, self.action_dim)) * (
            self.config.action_high - self.config.action_low
        ) / 2

        # Warm start
        if self._prev_actions is not None:
            mean[:-1] = self._prev_actions[1:]
            mean[-1] = 0.0

        best_plan = None

        for iteration in range(self.config.n_iterations):
            actions_all = self._sample_actions(self.config.n_samples, mean, std)

            # Evaluate all candidates
            results = []
            for i in range(self.config.n_samples):
                cost, traj, breakdown = self._evaluate_trajectory(
                    z_current, actions_all[i], goal
                )
                results.append((cost, i, traj, breakdown))

            results.sort(key=lambda x: x[0])

            # Select elites
            elite_indices = [r[1] for r in results[: self.config.n_elites]]
            elite_actions = actions_all[elite_indices]

            # Update distribution
            mean = elite_actions.mean(axis=0)
            std = elite_actions.std(axis=0) + 0.01

            # Track best
            best_cost, best_idx, best_traj, best_breakdown = results[0]
            best_plan = Plan(
                actions=actions_all[best_idx],
                trajectory=best_traj,
                cost=best_cost,
                costs_breakdown=best_breakdown,
            )

        # Cache for warm start
        self._prev_actions = best_plan.actions

        return best_plan

    def _plan_mppi(
        self,
        z_current: np.ndarray,
        goal: Optional[np.ndarray],
    ) -> Plan:
        """MPPI: Model Predictive Path Integral."""
        H = self.config.horizon
        temp = self.config.mppi_temperature

        mean = np.zeros((H, self.action_dim))

        # Warm start
        if self._prev_actions is not None:
            mean[:-1] = self._prev_actions[1:]

        for iteration in range(self.config.n_iterations):
            noise = np.random.randn(self.config.n_samples, H, self.action_dim)
            actions_all = np.clip(
                mean + noise,
                self.config.action_low,
                self.config.action_high,
            )

            # Evaluate
            costs = []
            for i in range(self.config.n_samples):
                cost, _, _ = self._evaluate_trajectory(
                    z_current, actions_all[i], goal
                )
                costs.append(cost)

            costs = np.array(costs)

            # Compute weights
            costs_shifted = costs - costs.min()
            weights = np.exp(-costs_shifted / temp)
            weights /= weights.sum()

            # Weighted average
            mean = np.sum(weights[:, None, None] * actions_all, axis=0)

        # Final evaluation
        final_cost, final_traj, final_breakdown = self._evaluate_trajectory(
            z_current, mean, goal
        )

        self._prev_actions = mean

        return Plan(
            actions=mean,
            trajectory=final_traj,
            cost=final_cost,
            costs_breakdown=final_breakdown,
        )

    # =========================================================================
    # Dream Mode
    # =========================================================================

    def dream_explore(
        self,
        z_current: np.ndarray,
        n_dreams: int = 5,
    ) -> List[DreamResult]:
        """
        Explore possible futures through dreaming.

        In dream mode, we:
        - Extend horizon for longer-term imagination
        - Add curiosity reward for novel regions
        - Don't execute actions (frozen actuators)
        """
        if not self._dreaming:
            logger.warning("dream_explore called outside dream mode context")

        dreams = []
        z_current = np.asarray(z_current).flatten()

        for i in range(n_dreams):
            # Sample random action sequence with exploration noise
            H = self.config.dream_horizon
            actions = np.random.uniform(
                self.config.action_low,
                self.config.action_high,
                (H, self.action_dim),
            ) + np.random.randn(H, self.action_dim) * self.config.dream_noise_scale

            actions = np.clip(
                actions, self.config.action_low, self.config.action_high
            )

            # Roll out trajectory
            trajectory = self.world.rollout(z_current, actions, include_start=True)

            # Compute risk profile
            risk_profile = []
            terminated = False
            for j, z in enumerate(trajectory):
                if self._risk_fn:
                    risk = self._risk_fn(z)
                    risk_profile.append(risk)
                    if risk > 0.9:  # Forbidden region
                        terminated = True
                        trajectory = trajectory[: j + 1]
                        actions = actions[:j]
                        break
                else:
                    risk_profile.append(0.0)

            # Compute discovery (novelty)
            discovery = sum(self._compute_novelty(z) for z in trajectory)

            # Compute total reward (inverted cost)
            total_reward = -sum(risk_profile) + self.config.curiosity_weight * discovery

            dreams.append(
                DreamResult(
                    trajectory=trajectory,
                    actions=actions,
                    total_reward=total_reward,
                    risk_profile=risk_profile,
                    discovery=discovery,
                    terminated_early=terminated,
                )
            )

            # Update visited regions
            for z in trajectory:
                self._visited_regions.append(z.copy())

        # Sort by reward (best first)
        dreams.sort(key=lambda d: d.total_reward, reverse=True)

        return dreams

    def dream_counterfactual(
        self,
        z_current: np.ndarray,
        actions_taken: np.ndarray,
        branch_point: int,
        n_alternatives: int = 5,
    ) -> List[DreamResult]:
        """
        Explore counterfactual: 'what if I had done something different?'

        Takes the actual path up to branch_point, then explores alternatives.
        """
        # Roll forward to branch point with actual actions
        trajectory_prefix = self.world.rollout(
            z_current, actions_taken[:branch_point], include_start=True
        )
        z_branch = trajectory_prefix[-1]

        # Now dream from the branch point
        return self.dream_explore(z_branch, n_alternatives)


# =============================================================================
# Context Managers and Utilities
# =============================================================================

@contextmanager
def dream_mode(planner: DojoPlanner):
    """Context manager for dream mode."""
    planner._dreaming = True
    original_horizon = planner.config.horizon
    planner.config.horizon = planner.config.dream_horizon

    logger.info("Entering dream mode...")

    try:
        yield planner
    finally:
        planner._dreaming = False
        planner.config.horizon = original_horizon
        logger.info("Exiting dream mode.")


def mpc_plan(
    planner: DojoPlanner,
    z_current: np.ndarray,
    goal: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Convenience function for MPC planning.

    Returns just the first action to execute.
    """
    plan = planner.plan(z_current, goal)
    return plan.first_action


# =============================================================================
# Testing
# =============================================================================

def _test_planner():
    """Test the Dojo planner."""
    print("=" * 60)
    print("Dojo Planner Test")
    print("=" * 60)

    from .world_model import DojoWorldModel, DojoWorldModelConfig

    # Create world model
    config = DojoWorldModelConfig(latent_dim=4, action_dim=2, model_type="linear")
    world = DojoWorldModel(config)

    # Create planner
    planner_config = DojoPlannerConfig(
        horizon=5,
        n_samples=50,
        method=MPCMethod.CEM,
    )
    planner = DojoPlanner(world, config=planner_config)

    # Simple risk function
    def risk_fn(z):
        return max(0, np.linalg.norm(z) - 2.0) / 2.0

    planner.set_risk_function(risk_fn)

    # Plan
    z_current = np.zeros(4)
    goal = np.array([1.0, 1.0, 0.0, 0.0])

    plan = planner.plan(z_current, goal=goal)

    print(f"First action: {plan.first_action}")
    print(f"Final state: {plan.final_state}")
    print(f"Total cost: {plan.cost:.4f}")
    print(f"Cost breakdown: {plan.costs_breakdown}")

    # Dream mode
    print("\n--- Dream Mode ---")
    with dream_mode(planner):
        dreams = planner.dream_explore(z_current, n_dreams=3)
        for i, dream in enumerate(dreams):
            print(f"Dream {i+1}: reward={dream.total_reward:.2f}, "
                  f"discovery={dream.discovery:.2f}, "
                  f"terminated={dream.terminated_early}")


if __name__ == "__main__":
    _test_planner()
