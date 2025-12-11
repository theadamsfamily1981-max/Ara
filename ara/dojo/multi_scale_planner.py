#!/usr/bin/env python3
# ara/dojo/multi_scale_planner.py
"""
Multi-Scale Planner for Ara
===========================

Combines multiple planners with different horizons (short/medium/long)
and enforces that chosen actions are good across timescales.

This is where "don't be locally optimal but globally stupid" gets baked in.

Modes:
- Fusion: Weighted average of per-scale actions
- Consensus: Only act if all scales broadly agree
- Hierarchical: Long-term sets goals, short-term executes

Usage:
    from ara.dojo import MultiScalePlanner, DojoWorldModel, DojoPlanner

    world_model = load_world_model("models/world_model_best.pt")
    planner = MultiScalePlanner(world_model, action_dim=8)

    action, debug = planner.plan(z_current, goal=goal)
    # action is good at 10, 50, and 200 steps simultaneously
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


# =============================================================================
# Configuration
# =============================================================================

class FusionMode(Enum):
    """How to combine multi-scale plans."""
    FUSION = auto()       # Weighted average
    CONSENSUS = auto()    # Only act if scales agree
    HIERARCHICAL = auto() # Long sets goals, short executes


@dataclass
class ScaleConfig:
    """Configuration for a single planning scale."""
    name: str
    horizon: int           # Planning horizon (steps)
    num_samples: int       # MPC samples
    weight: float          # Importance in fusion
    iterations: int = 3    # CEM/MPPI iterations


@dataclass
class MultiScaleConfig:
    """Configuration for multi-scale planner."""
    mode: FusionMode = FusionMode.FUSION
    agreement_tolerance: float = 0.15  # Max L2 diff for consensus

    # Default scales: reactive / tactical / strategic
    scales: List[ScaleConfig] = field(default_factory=lambda: [
        ScaleConfig(name="reactive", horizon=5, num_samples=64, weight=0.4),
        ScaleConfig(name="tactical", horizon=25, num_samples=48, weight=0.35),
        ScaleConfig(name="strategic", horizon=100, num_samples=32, weight=0.25),
    ])

    # Fallback action when scales disagree
    fallback_to_cautious: bool = True
    cautious_scale: float = 0.1  # Scale down actions when uncertain


@dataclass
class MultiScalePlan:
    """Result of multi-scale planning."""
    action: np.ndarray                        # Final fused action
    per_scale_actions: Dict[str, np.ndarray]  # Action from each scale
    per_scale_costs: Dict[str, float]         # Cost from each scale
    agreement_score: float                    # How much scales agree (0-1)
    mode_used: str                            # Which fusion mode was applied
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# Multi-Scale Planner
# =============================================================================

class MultiScalePlanner:
    """
    Multi-scale MPC planner for Ara.

    Runs several planners at different horizons and combines their
    recommendations to ensure actions are good across timescales.
    """

    def __init__(
        self,
        world_model: Any,
        action_dim: int = 8,
        latent_dim: int = 10,
        config: Optional[MultiScaleConfig] = None,
        cost_fn: Optional[Callable] = None,
        risk_fn: Optional[Callable] = None,
        device: str = "cpu",
    ):
        """
        Initialize multi-scale planner.

        Args:
            world_model: Trained world model for dynamics
            action_dim: Action space dimension
            latent_dim: Latent space dimension
            config: Multi-scale configuration
            cost_fn: Custom cost function (z, u) -> float
            risk_fn: Risk scoring function for MEIS/NIB
            device: "cpu" or "cuda"
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for MultiScalePlanner")

        self.world_model = world_model
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.config = config or MultiScaleConfig()
        self.cost_fn = cost_fn
        self.risk_fn = risk_fn
        self.device = torch.device(device)

        # Action bounds
        self.action_low = -1.0
        self.action_high = 1.0

        # Build per-scale planners
        self._planners: Dict[str, ScaleConfig] = {}
        for scale in self.config.scales:
            self._planners[scale.name] = scale

        logger.info(
            f"MultiScalePlanner: {len(self._planners)} scales, "
            f"mode={self.config.mode.name}"
        )

    def plan(
        self,
        z_current: np.ndarray,
        goal: Optional[np.ndarray] = None,
    ) -> MultiScalePlan:
        """
        Compute multi-scale action.

        Args:
            z_current: Current latent state
            goal: Optional goal state in latent space

        Returns:
            MultiScalePlan with fused action and debug info
        """
        z_current = np.asarray(z_current).flatten()
        if goal is not None:
            goal = np.asarray(goal).flatten()

        # Run each scale
        per_scale_actions = {}
        per_scale_costs = {}
        per_scale_trajs = {}

        for name, scale_config in self._planners.items():
            action, cost, traj = self._plan_single_scale(
                z_current, goal, scale_config
            )
            per_scale_actions[name] = action
            per_scale_costs[name] = cost
            per_scale_trajs[name] = traj

        # Compute agreement score
        agreement = self._compute_agreement(per_scale_actions)

        # Fuse actions based on mode
        warnings = []

        if self.config.mode == FusionMode.CONSENSUS:
            if agreement < (1.0 - self.config.agreement_tolerance):
                if self.config.fallback_to_cautious:
                    # Scales disagree - fall back to cautious action
                    action = self._cautious_action(per_scale_actions)
                    warnings.append(
                        f"Scales disagreed (agreement={agreement:.2f}), "
                        f"using cautious action"
                    )
                else:
                    # Use reactive scale as tiebreaker
                    action = per_scale_actions.get(
                        "reactive",
                        list(per_scale_actions.values())[0]
                    )
                    warnings.append("Scales disagreed, using reactive fallback")
            else:
                # Consensus reached - use weighted fusion
                action = self._fusion_action(per_scale_actions)

        elif self.config.mode == FusionMode.HIERARCHICAL:
            action = self._hierarchical_action(
                per_scale_actions, per_scale_trajs, z_current, goal
            )

        else:  # FUSION
            action = self._fusion_action(per_scale_actions)

        return MultiScalePlan(
            action=action,
            per_scale_actions=per_scale_actions,
            per_scale_costs=per_scale_costs,
            agreement_score=agreement,
            mode_used=self.config.mode.name,
            warnings=warnings,
        )

    def _plan_single_scale(
        self,
        z_current: np.ndarray,
        goal: Optional[np.ndarray],
        scale: ScaleConfig,
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """Run CEM planning at a single scale."""
        H = scale.horizon
        N = scale.num_samples

        # Initialize action distribution
        mean = np.zeros((H, self.action_dim))
        std = np.ones((H, self.action_dim)) * 0.5

        best_actions = mean.copy()
        best_cost = float("inf")
        best_traj = None

        for iteration in range(scale.iterations):
            # Sample action sequences
            noise = np.random.randn(N, H, self.action_dim)
            actions = mean + std * noise
            actions = np.clip(actions, self.action_low, self.action_high)

            # Evaluate each sequence
            costs = []
            trajs = []

            for i in range(N):
                cost, traj = self._evaluate_trajectory(
                    z_current, actions[i], goal
                )
                costs.append(cost)
                trajs.append(traj)

            costs = np.array(costs)

            # Select elites
            n_elite = max(1, N // 5)
            elite_idx = np.argsort(costs)[:n_elite]
            elite_actions = actions[elite_idx]

            # Update distribution
            mean = elite_actions.mean(axis=0)
            std = elite_actions.std(axis=0) + 0.01

            # Track best
            if costs[elite_idx[0]] < best_cost:
                best_cost = costs[elite_idx[0]]
                best_actions = actions[elite_idx[0]]
                best_traj = trajs[elite_idx[0]]

        return best_actions[0], best_cost, best_traj

    def _evaluate_trajectory(
        self,
        z_init: np.ndarray,
        actions: np.ndarray,
        goal: Optional[np.ndarray],
    ) -> Tuple[float, np.ndarray]:
        """Evaluate a trajectory through world model."""
        z = z_init.copy()
        trajectory = [z.copy()]
        total_cost = 0.0

        for u in actions:
            # Predict next state
            z_next = self._predict_next(z, u)
            trajectory.append(z_next.copy())

            # Accumulate cost
            if self.cost_fn is not None:
                total_cost += self.cost_fn(z, u)
            elif goal is not None:
                # Default: distance to goal
                total_cost += np.sum((z - goal) ** 2)
            else:
                # Default: small action penalty
                total_cost += 0.01 * np.sum(u ** 2)

            # Risk penalty
            if self.risk_fn is not None:
                risk = self.risk_fn(z_next)
                total_cost += risk * 10.0

            z = z_next

        # Terminal cost
        if goal is not None:
            total_cost += np.sum((z - goal) ** 2) * 5.0

        return total_cost, np.array(trajectory)

    def _predict_next(self, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Predict next state using world model."""
        # Handle different world model interfaces
        if hasattr(self.world_model, 'predict'):
            return self.world_model.predict(z, u)
        elif hasattr(self.world_model, 'forward'):
            with torch.no_grad():
                z_t = torch.from_numpy(z.astype(np.float32)).unsqueeze(0)
                u_t = torch.from_numpy(u.astype(np.float32)).unsqueeze(0)
                z_next = self.world_model(z_t, u_t).squeeze(0).numpy()
            return z_next
        else:
            # Fallback: identity dynamics
            return z + 0.1 * u[:len(z)] if len(u) >= len(z) else z

    def _compute_agreement(
        self,
        per_scale_actions: Dict[str, np.ndarray],
    ) -> float:
        """Compute how much scales agree (0 = total disagreement, 1 = perfect)."""
        actions = list(per_scale_actions.values())
        if len(actions) < 2:
            return 1.0

        # Pairwise L2 distances
        dists = []
        for i in range(len(actions)):
            for j in range(i + 1, len(actions)):
                dist = np.linalg.norm(actions[i] - actions[j])
                dists.append(dist)

        avg_dist = np.mean(dists)

        # Normalize by action space diameter
        diameter = np.sqrt(self.action_dim) * (self.action_high - self.action_low)
        agreement = 1.0 - min(avg_dist / diameter, 1.0)

        return float(agreement)

    def _fusion_action(
        self,
        per_scale_actions: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Weighted combination of per-scale actions."""
        weighted_sum = np.zeros(self.action_dim)
        total_weight = 0.0

        for name, action in per_scale_actions.items():
            scale = self._planners.get(name)
            if scale is None:
                continue
            weighted_sum += scale.weight * action
            total_weight += scale.weight

        if total_weight <= 0:
            return np.zeros(self.action_dim)

        return weighted_sum / total_weight

    def _cautious_action(
        self,
        per_scale_actions: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Conservative action when scales disagree."""
        # Average and scale down
        avg = self._fusion_action(per_scale_actions)
        return avg * self.config.cautious_scale

    def _hierarchical_action(
        self,
        per_scale_actions: Dict[str, np.ndarray],
        per_scale_trajs: Dict[str, np.ndarray],
        z_current: np.ndarray,
        goal: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Hierarchical planning: strategic sets waypoint, reactive executes.

        Long-horizon plan sets intermediate goals, short-horizon executes
        the first step toward the nearest waypoint.
        """
        # Get strategic trajectory for waypoints
        strategic_traj = per_scale_trajs.get("strategic")
        if strategic_traj is None or len(strategic_traj) < 2:
            return self._fusion_action(per_scale_actions)

        # Use a point ~10 steps ahead as intermediate goal
        waypoint_idx = min(10, len(strategic_traj) - 1)
        waypoint = strategic_traj[waypoint_idx]

        # Re-plan reactive scale toward waypoint
        reactive_scale = self._planners.get("reactive")
        if reactive_scale is None:
            return self._fusion_action(per_scale_actions)

        action, _, _ = self._plan_single_scale(z_current, waypoint, reactive_scale)
        return action


# =============================================================================
# Testing
# =============================================================================

def _test_multi_scale():
    """Test multi-scale planner."""
    if not TORCH_AVAILABLE:
        print("PyTorch not available, skipping test")
        return

    print("=" * 60)
    print("Multi-Scale Planner Test")
    print("=" * 60)

    # Simple linear world model for testing
    class SimpleWorldModel(nn.Module):
        def __init__(self, latent_dim=10, action_dim=4):
            super().__init__()
            self.A = nn.Linear(latent_dim, latent_dim, bias=False)
            self.B = nn.Linear(action_dim, latent_dim, bias=False)

        def forward(self, z, u):
            return self.A(z) + self.B(u)

        def predict(self, z, u):
            with torch.no_grad():
                z_t = torch.from_numpy(z.astype(np.float32)).unsqueeze(0)
                u_t = torch.from_numpy(u.astype(np.float32)).unsqueeze(0)
                return self.forward(z_t, u_t).squeeze(0).numpy()

    world_model = SimpleWorldModel(latent_dim=4, action_dim=2)

    # Create planner
    config = MultiScaleConfig(
        mode=FusionMode.FUSION,
        scales=[
            ScaleConfig(name="reactive", horizon=3, num_samples=20, weight=0.5),
            ScaleConfig(name="tactical", horizon=10, num_samples=15, weight=0.3),
            ScaleConfig(name="strategic", horizon=20, num_samples=10, weight=0.2),
        ],
    )

    planner = MultiScalePlanner(
        world_model=world_model,
        action_dim=2,
        latent_dim=4,
        config=config,
    )

    # Plan
    z_current = np.zeros(4)
    goal = np.array([1.0, 1.0, 0.0, 0.0])

    result = planner.plan(z_current, goal=goal)

    print(f"\nFused action: {result.action}")
    print(f"Agreement score: {result.agreement_score:.2f}")
    print(f"Mode used: {result.mode_used}")
    print(f"\nPer-scale actions:")
    for name, action in result.per_scale_actions.items():
        cost = result.per_scale_costs[name]
        print(f"  {name}: {action} (cost={cost:.2f})")

    if result.warnings:
        print(f"\nWarnings: {result.warnings}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _test_multi_scale()
