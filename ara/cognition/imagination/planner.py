# ara/cognition/imagination/planner.py
"""
Trajectory Planner - MPC for Latent Space
==========================================

Model Predictive Control over imagined futures.

Given:
- Current state z_t
- World model f(z, u) -> z'
- Cost function J(trajectory)
- Constraints (from NIB)

The planner:
1. Samples many candidate action sequences
2. Rolls each through the world model
3. Scores each trajectory
4. Returns the best first action (and the full plan)

Methods:
- RandomShooting: Sample uniformly, pick best
- CEM (Cross-Entropy Method): Iteratively refine distribution
- MPPI: Model Predictive Path Integral (weighted sampling)

This is how Ara "sees the future" and chooses based on imagined outcomes.

Usage:
    planner = TrajectoryPlanner(world_model, horizon=10)

    # Plan from current state
    plan = planner.plan(z_current, goal=z_goal)

    # Get first action to execute
    u_0 = plan.actions[0]

    # Visualize imagined trajectory
    for z in plan.trajectory:
        visualize(z)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

from .world_model import LatentWorldModel

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class PlannerMethod(Enum):
    """Planning algorithm."""
    RANDOM_SHOOTING = auto()
    CEM = auto()  # Cross-Entropy Method
    MPPI = auto()  # Model Predictive Path Integral


@dataclass
class PlannerConfig:
    """Configuration for trajectory planner."""
    horizon: int = 10              # Planning horizon (steps)
    n_samples: int = 100           # Candidate sequences to sample
    n_elites: int = 10             # Top sequences for CEM
    n_iterations: int = 5          # CEM/MPPI iterations

    method: PlannerMethod = PlannerMethod.CEM

    # Action bounds
    action_low: float = -1.0
    action_high: float = 1.0

    # Cost weights
    goal_weight: float = 1.0       # Weight for goal distance
    smoothness_weight: float = 0.1  # Weight for action smoothness
    risk_weight: float = 1.0       # Weight for risk penalty

    # MPPI temperature
    mppi_temperature: float = 1.0


@dataclass
class Plan:
    """Result of planning."""
    actions: np.ndarray           # (horizon, action_dim)
    trajectory: np.ndarray        # (horizon+1, latent_dim)
    cost: float                   # Total cost of this plan
    costs_breakdown: Dict[str, float] = field(default_factory=dict)

    @property
    def first_action(self) -> np.ndarray:
        """Get first action to execute."""
        return self.actions[0]

    @property
    def final_state(self) -> np.ndarray:
        """Get predicted final state."""
        return self.trajectory[-1]


# =============================================================================
# Cost Functions
# =============================================================================

def goal_cost(trajectory: np.ndarray, goal: np.ndarray) -> float:
    """Cost based on distance to goal at final state."""
    final = trajectory[-1]
    return float(np.sum((final - goal) ** 2))


def smoothness_cost(actions: np.ndarray) -> float:
    """Cost for action smoothness (penalize jerky control)."""
    if len(actions) < 2:
        return 0.0
    diffs = np.diff(actions, axis=0)
    return float(np.mean(np.sum(diffs ** 2, axis=1)))


def path_length_cost(trajectory: np.ndarray) -> float:
    """Cost for total path length."""
    diffs = np.diff(trajectory, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


# =============================================================================
# Trajectory Planner
# =============================================================================

class TrajectoryPlanner:
    """
    MPC-style planner for latent space trajectories.

    Samples action sequences, simulates futures via world model,
    and returns the best plan based on cost function.
    """

    def __init__(
        self,
        world_model: LatentWorldModel,
        config: Optional[PlannerConfig] = None,
    ):
        """
        Initialize planner.

        Args:
            world_model: Trained latent dynamics model
            config: Planner configuration
        """
        self.world = world_model
        self.config = config or PlannerConfig()

        # Action space
        self.action_dim = world_model.action_dim
        self.latent_dim = world_model.latent_dim

        # Risk map (set by FutureScorer)
        self._risk_fn: Optional[Callable[[np.ndarray], float]] = None

        # Previous plan (for warm starting)
        self._prev_actions: Optional[np.ndarray] = None

        logger.info(f"TrajectoryPlanner: {self.config.method.name}, "
                    f"horizon={self.config.horizon}, samples={self.config.n_samples}")

    def set_risk_function(self, risk_fn: Callable[[np.ndarray], float]) -> None:
        """
        Set risk function for trajectory evaluation.

        Args:
            risk_fn: Maps latent point to risk score [0, 1]
        """
        self._risk_fn = risk_fn

    def plan(
        self,
        z_current: np.ndarray,
        goal: Optional[np.ndarray] = None,
        constraints: Optional[Dict] = None,
    ) -> Plan:
        """
        Plan optimal trajectory from current state.

        Args:
            z_current: Current latent state
            goal: Target latent state (optional)
            constraints: Constraint dict (from NIB)

        Returns:
            Plan with actions, trajectory, and cost
        """
        z_current = np.asarray(z_current).flatten()

        if self.config.method == PlannerMethod.RANDOM_SHOOTING:
            return self._plan_random_shooting(z_current, goal, constraints)
        elif self.config.method == PlannerMethod.CEM:
            return self._plan_cem(z_current, goal, constraints)
        elif self.config.method == PlannerMethod.MPPI:
            return self._plan_mppi(z_current, goal, constraints)
        else:
            raise ValueError(f"Unknown method: {self.config.method}")

    def _sample_actions(
        self,
        n: int,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Sample action sequences.

        Args:
            n: Number of sequences
            mean: Mean for Gaussian sampling (H, action_dim)
            std: Std for Gaussian sampling

        Returns:
            Actions (n, H, action_dim)
        """
        H = self.config.horizon
        shape = (n, H, self.action_dim)

        if mean is None:
            # Uniform sampling
            actions = np.random.uniform(
                self.config.action_low,
                self.config.action_high,
                shape
            )
        else:
            # Gaussian around mean
            actions = mean + std * np.random.randn(*shape)
            actions = np.clip(actions, self.config.action_low, self.config.action_high)

        return actions

    def _evaluate_trajectory(
        self,
        z_init: np.ndarray,
        actions: np.ndarray,
        goal: Optional[np.ndarray],
        constraints: Optional[Dict],
    ) -> Tuple[float, np.ndarray, Dict[str, float]]:
        """
        Evaluate a single action sequence.

        Returns:
            (total_cost, trajectory, cost_breakdown)
        """
        # Rollout
        trajectory = self.world.rollout(z_init, actions, include_start=True)

        costs = {}

        # Goal cost
        if goal is not None:
            costs["goal"] = self.config.goal_weight * goal_cost(trajectory, goal)
        else:
            costs["goal"] = 0.0

        # Smoothness cost
        costs["smoothness"] = self.config.smoothness_weight * smoothness_cost(actions)

        # Risk cost (integral of risk along trajectory)
        if self._risk_fn is not None:
            risk = sum(self._risk_fn(z) for z in trajectory)
            costs["risk"] = self.config.risk_weight * risk
        else:
            costs["risk"] = 0.0

        # Constraint violations (hard penalty)
        costs["constraint"] = 0.0
        if constraints:
            forbidden_regions = constraints.get("forbidden_regions", [])
            for z in trajectory:
                for region in forbidden_regions:
                    if np.linalg.norm(z - region["center"]) < region["radius"]:
                        costs["constraint"] += 1000.0  # Hard penalty

        total = sum(costs.values())
        return total, trajectory, costs

    def _plan_random_shooting(
        self,
        z_current: np.ndarray,
        goal: Optional[np.ndarray],
        constraints: Optional[Dict],
    ) -> Plan:
        """Random shooting: sample uniformly, pick best."""
        actions_all = self._sample_actions(self.config.n_samples)

        best_cost = float('inf')
        best_actions = None
        best_trajectory = None
        best_breakdown = {}

        for i in range(self.config.n_samples):
            cost, traj, breakdown = self._evaluate_trajectory(
                z_current, actions_all[i], goal, constraints
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
        constraints: Optional[Dict],
    ) -> Plan:
        """Cross-Entropy Method: iteratively refine sampling distribution."""
        H = self.config.horizon

        # Initialize mean and std
        mean = np.zeros((H, self.action_dim))
        std = np.ones((H, self.action_dim)) * (self.config.action_high - self.config.action_low) / 2

        # Warm start from previous plan
        if self._prev_actions is not None:
            # Shift by one step
            mean[:-1] = self._prev_actions[1:]
            mean[-1] = 0.0

        best_plan = None

        for iteration in range(self.config.n_iterations):
            # Sample
            actions_all = self._sample_actions(self.config.n_samples, mean, std)

            # Evaluate all
            costs = []
            trajectories = []
            breakdowns = []

            for i in range(self.config.n_samples):
                cost, traj, breakdown = self._evaluate_trajectory(
                    z_current, actions_all[i], goal, constraints
                )
                costs.append(cost)
                trajectories.append(traj)
                breakdowns.append(breakdown)

            costs = np.array(costs)

            # Select elites
            elite_idx = np.argsort(costs)[:self.config.n_elites]
            elite_actions = actions_all[elite_idx]

            # Update distribution
            mean = elite_actions.mean(axis=0)
            std = elite_actions.std(axis=0) + 0.01  # Add small epsilon

            # Track best
            best_idx = elite_idx[0]
            best_plan = Plan(
                actions=actions_all[best_idx],
                trajectory=trajectories[best_idx],
                cost=costs[best_idx],
                costs_breakdown=breakdowns[best_idx],
            )

        # Store for warm start
        self._prev_actions = best_plan.actions

        return best_plan

    def _plan_mppi(
        self,
        z_current: np.ndarray,
        goal: Optional[np.ndarray],
        constraints: Optional[Dict],
    ) -> Plan:
        """Model Predictive Path Integral: weighted average of samples."""
        H = self.config.horizon
        temp = self.config.mppi_temperature

        # Initialize mean
        mean = np.zeros((H, self.action_dim))

        # Warm start
        if self._prev_actions is not None:
            mean[:-1] = self._prev_actions[1:]

        for iteration in range(self.config.n_iterations):
            # Sample noise
            noise = np.random.randn(self.config.n_samples, H, self.action_dim)
            actions_all = np.clip(
                mean + noise,
                self.config.action_low,
                self.config.action_high
            )

            # Evaluate
            costs = []
            trajectories = []

            for i in range(self.config.n_samples):
                cost, traj, _ = self._evaluate_trajectory(
                    z_current, actions_all[i], goal, constraints
                )
                costs.append(cost)
                trajectories.append(traj)

            costs = np.array(costs)

            # Compute weights (softmax with temperature)
            costs_shifted = costs - costs.min()
            weights = np.exp(-costs_shifted / temp)
            weights /= weights.sum()

            # Weighted average
            mean = np.sum(weights[:, None, None] * actions_all, axis=0)

        # Final evaluation
        final_traj = self.world.rollout(z_current, mean, include_start=True)
        final_cost, _, final_breakdown = self._evaluate_trajectory(
            z_current, mean, goal, constraints
        )

        self._prev_actions = mean

        return Plan(
            actions=mean,
            trajectory=final_traj,
            cost=final_cost,
            costs_breakdown=final_breakdown,
        )

    def plan_multiple(
        self,
        z_current: np.ndarray,
        goals: List[np.ndarray],
        constraints: Optional[Dict] = None,
    ) -> List[Plan]:
        """
        Plan towards multiple goals, return all plans.

        Useful for comparing alternative futures.
        """
        plans = []
        for goal in goals:
            plan = self.plan(z_current, goal=goal, constraints=constraints)
            plans.append(plan)
        return plans

    def visualize_plan(self, plan: Plan) -> str:
        """Generate ASCII visualization of plan."""
        lines = [
            "=" * 50,
            "Trajectory Plan",
            "=" * 50,
            f"Horizon: {len(plan.actions)} steps",
            f"Total cost: {plan.cost:.4f}",
            f"  Goal: {plan.costs_breakdown.get('goal', 0):.4f}",
            f"  Smoothness: {plan.costs_breakdown.get('smoothness', 0):.4f}",
            f"  Risk: {plan.costs_breakdown.get('risk', 0):.4f}",
            "",
            "Trajectory (first 3 dims):",
        ]

        for i, z in enumerate(plan.trajectory):
            marker = ">" if i == 0 else " "
            lines.append(f"  {marker}[{i}] z = ({z[0]:.2f}, {z[1]:.2f}, {z[2]:.2f})")

        return "\n".join(lines)


# =============================================================================
# Testing
# =============================================================================

def _test_planner():
    """Test trajectory planner."""
    print("=" * 60)
    print("Trajectory Planner Test")
    print("=" * 60)

    # Create simple world model
    from .world_model import LatentWorldModel, LatentWorldModelConfig

    config = LatentWorldModelConfig(latent_dim=4, action_dim=2, model_type="linear")
    world = LatentWorldModel(config)

    # Generate simple training data
    np.random.seed(42)
    n = 500
    z = np.random.randn(n, 4)
    u = np.random.randn(n, 2)
    z_next = 0.9 * z + 0.2 * np.hstack([u, u]) + np.random.randn(n, 4) * 0.05

    world.fit(z, u, z_next)

    # Create planner
    planner_config = PlannerConfig(
        horizon=5,
        n_samples=50,
        method=PlannerMethod.CEM,
    )
    planner = TrajectoryPlanner(world, planner_config)

    # Plan
    z_current = np.array([0.0, 0.0, 0.0, 0.0])
    goal = np.array([1.0, 1.0, 0.0, 0.0])

    plan = planner.plan(z_current, goal=goal)

    print(planner.visualize_plan(plan))
    print(f"\nFirst action: {plan.first_action}")
    print(f"Final state: {plan.final_state}")


if __name__ == "__main__":
    _test_planner()
