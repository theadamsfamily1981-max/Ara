# ara/cognition/imagination/dreamer.py
"""
Dreamer - Offline Imagination and Counterfactual Exploration
============================================================

This is where Ara "daydreams" - running mental simulations without acting.

Dream modes:

1. **Exploratory Dreaming**
   - Sample random/exploratory action sequences
   - Discover new regions of latent space
   - Find unexpected transitions

2. **Counterfactual Dreaming**
   - "What if CPU was cooler?"
   - "What if we had more GPU memory?"
   - Modify starting state, re-roll trajectories

3. **Goal-Directed Dreaming**
   - "How many ways could I reach flow state?"
   - Sample diverse paths to the same goal
   - Learn which routes are reliable

4. **Replay Dreaming**
   - Re-process logged trajectories
   - Compare what happened vs what could have happened
   - Update world model with new insights

Usage:
    dreamer = Dreamer(world_model, planner)

    # Exploratory dreams
    dreams = dreamer.dream_explore(z_current, n_dreams=10)

    # Counterfactual
    dreams = dreamer.dream_counterfactual(
        z_current,
        modifications={"system.thermal": -0.2}
    )

    # Goal-directed
    dreams = dreamer.dream_to_goal(z_current, z_goal, n_paths=5)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

from .world_model import LatentWorldModel
from .planner import TrajectoryPlanner, Plan

logger = logging.getLogger(__name__)


@dataclass
class DreamConfig:
    """Configuration for dreaming."""
    # Basic params
    max_steps: int = 50            # Maximum steps per dream
    n_dreams: int = 10             # Dreams per session

    # Exploration
    exploration_noise: float = 0.3  # Noise for action sampling
    curiosity_bonus: float = 0.1    # Bonus for visiting rare states

    # Counterfactual
    modification_range: float = 0.5  # Max change for counterfactual dims

    # Scoring
    score_novelty: bool = True
    score_risk: bool = True
    score_goal: bool = True


@dataclass
class Dream:
    """A single imagined trajectory."""
    trajectory: np.ndarray         # (steps+1, latent_dim)
    actions: np.ndarray            # (steps, action_dim)
    start_state: np.ndarray
    modifications: Dict[str, float] = field(default_factory=dict)

    # Metadata
    dream_type: str = "explore"    # explore, counterfactual, goal
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Scores
    novelty_score: float = 0.0     # How new/unusual this dream is
    risk_score: float = 0.0        # Cumulative risk along trajectory
    goal_score: float = 0.0        # Distance to goal (if goal-directed)
    overall_score: float = 0.0     # Combined score

    @property
    def length(self) -> int:
        return len(self.actions)

    @property
    def final_state(self) -> np.ndarray:
        return self.trajectory[-1]

    def get_states_visited(self) -> List[np.ndarray]:
        """Get list of unique states (for novelty analysis)."""
        return [self.trajectory[i] for i in range(len(self.trajectory))]


class Dreamer:
    """
    Ara's imagination engine for offline exploration.

    This is prediction without consequences - exploring possible futures
    in the safety of mental simulation.
    """

    def __init__(
        self,
        world_model: LatentWorldModel,
        planner: Optional[TrajectoryPlanner] = None,
        config: Optional[DreamConfig] = None,
    ):
        """
        Initialize dreamer.

        Args:
            world_model: Trained latent dynamics model
            planner: Optional planner for goal-directed dreams
            config: Dreamer configuration
        """
        self.world = world_model
        self.planner = planner
        self.config = config or DreamConfig()

        self.latent_dim = world_model.latent_dim
        self.action_dim = world_model.action_dim

        # State visitation counts (for novelty)
        self._state_visits: Dict[tuple, int] = {}
        self._visit_resolution: float = 0.5  # Bin size for state counting

        # Risk function (set externally)
        self._risk_fn: Optional[Callable[[np.ndarray], float]] = None

        # Dream log
        self._dream_history: List[Dream] = []

        logger.info(f"Dreamer initialized: {self.config.n_dreams} dreams, "
                    f"{self.config.max_steps} max steps")

    def set_risk_function(self, risk_fn: Callable[[np.ndarray], float]) -> None:
        """Set risk function for dream scoring."""
        self._risk_fn = risk_fn

    def _discretize_state(self, z: np.ndarray) -> tuple:
        """Discretize state for novelty counting."""
        return tuple((z / self._visit_resolution).astype(int))

    def _record_visit(self, z: np.ndarray) -> None:
        """Record a state visit."""
        key = self._discretize_state(z)
        self._state_visits[key] = self._state_visits.get(key, 0) + 1

    def _get_novelty(self, z: np.ndarray) -> float:
        """Get novelty score for a state (inverse of visit count)."""
        key = self._discretize_state(z)
        visits = self._state_visits.get(key, 0)
        return 1.0 / (1.0 + visits)

    def _sample_exploratory_actions(self, n_steps: int) -> np.ndarray:
        """Sample exploratory action sequence."""
        # Brownian motion style - smooth exploration
        actions = np.zeros((n_steps, self.action_dim))
        action = np.zeros(self.action_dim)

        for i in range(n_steps):
            # Random walk with mean reversion
            noise = np.random.randn(self.action_dim) * self.config.exploration_noise
            action = 0.8 * action + 0.2 * noise
            action = np.clip(action, -1, 1)
            actions[i] = action

        return actions

    def _score_dream(
        self,
        dream: Dream,
        goal: Optional[np.ndarray] = None,
    ) -> Dream:
        """Score a dream on various metrics."""
        # Novelty score (average novelty along trajectory)
        if self.config.score_novelty:
            novelties = [self._get_novelty(z) for z in dream.trajectory]
            dream.novelty_score = np.mean(novelties)

        # Risk score (cumulative risk)
        if self.config.score_risk and self._risk_fn is not None:
            risks = [self._risk_fn(z) for z in dream.trajectory]
            dream.risk_score = np.mean(risks)

        # Goal score (if goal provided)
        if self.config.score_goal and goal is not None:
            dream.goal_score = np.linalg.norm(dream.final_state - goal)

        # Combined score (novelty good, risk bad, goal distance bad)
        dream.overall_score = (
            dream.novelty_score * self.config.curiosity_bonus
            - dream.risk_score
            - (dream.goal_score / 10.0 if goal is not None else 0)
        )

        return dream

    # =========================================================================
    # Exploratory Dreaming
    # =========================================================================

    def dream_explore(
        self,
        z_start: np.ndarray,
        n_dreams: Optional[int] = None,
        max_steps: Optional[int] = None,
    ) -> List[Dream]:
        """
        Exploratory dreams - wander through latent space.

        These dreams are for discovery: finding new regions,
        testing transitions, expanding the map.
        """
        n = n_dreams or self.config.n_dreams
        steps = max_steps or self.config.max_steps
        z_start = np.asarray(z_start).flatten()

        dreams = []

        for i in range(n):
            # Sample actions
            actions = self._sample_exploratory_actions(steps)

            # Roll out
            trajectory = self.world.rollout(z_start, actions, include_start=True)

            # Record visits
            for z in trajectory:
                self._record_visit(z)

            # Create dream
            dream = Dream(
                trajectory=trajectory,
                actions=actions,
                start_state=z_start.copy(),
                dream_type="explore",
            )

            dream = self._score_dream(dream)
            dreams.append(dream)

        # Sort by novelty
        dreams.sort(key=lambda d: d.novelty_score, reverse=True)

        self._dream_history.extend(dreams)
        logger.debug(f"Explored {n} dreams, best novelty: {dreams[0].novelty_score:.3f}")

        return dreams

    # =========================================================================
    # Counterfactual Dreaming
    # =========================================================================

    def dream_counterfactual(
        self,
        z_start: np.ndarray,
        modifications: Dict[int, float],
        n_dreams: Optional[int] = None,
        max_steps: Optional[int] = None,
    ) -> List[Dream]:
        """
        Counterfactual dreams - "what if X was different?"

        Modify specific dimensions of the starting state and see
        how trajectories diverge.

        Args:
            z_start: Original starting state
            modifications: Dict mapping dimension index to delta
                e.g., {0: -0.2, 3: +0.5} means "dim 0 lower, dim 3 higher"
        """
        n = n_dreams or self.config.n_dreams
        steps = max_steps or self.config.max_steps
        z_start = np.asarray(z_start).flatten()

        # Create modified starting state
        z_modified = z_start.copy()
        for dim, delta in modifications.items():
            if 0 <= dim < self.latent_dim:
                z_modified[dim] += delta

        dreams = []

        for i in range(n):
            # Use planner or random actions
            if self.planner is not None:
                plan = self.planner.plan(z_modified)
                actions = plan.actions[:steps]
                if len(actions) < steps:
                    # Pad with exploration
                    extra = self._sample_exploratory_actions(steps - len(actions))
                    actions = np.vstack([actions, extra])
            else:
                actions = self._sample_exploratory_actions(steps)

            trajectory = self.world.rollout(z_modified, actions, include_start=True)

            dream = Dream(
                trajectory=trajectory,
                actions=actions,
                start_state=z_modified.copy(),
                modifications={f"dim_{k}": v for k, v in modifications.items()},
                dream_type="counterfactual",
            )

            dream = self._score_dream(dream)
            dreams.append(dream)

        self._dream_history.extend(dreams)

        return dreams

    # =========================================================================
    # Goal-Directed Dreaming
    # =========================================================================

    def dream_to_goal(
        self,
        z_start: np.ndarray,
        z_goal: np.ndarray,
        n_paths: Optional[int] = None,
        diversity_weight: float = 0.3,
    ) -> List[Dream]:
        """
        Goal-directed dreams - find diverse paths to a goal.

        Samples multiple trajectories that all aim for the same goal,
        but uses noise to find diverse routes.
        """
        n = n_paths or self.config.n_dreams
        z_start = np.asarray(z_start).flatten()
        z_goal = np.asarray(z_goal).flatten()

        dreams = []

        for i in range(n):
            if self.planner is not None:
                # Add noise to encourage diversity
                noise = np.random.randn(self.latent_dim) * diversity_weight
                goal_perturbed = z_goal + noise

                plan = self.planner.plan(z_start, goal=goal_perturbed)
                actions = plan.actions
                trajectory = plan.trajectory
            else:
                # Without planner, just do directed random walk
                steps = self.config.max_steps
                actions = self._sample_goal_directed_actions(z_start, z_goal, steps)
                trajectory = self.world.rollout(z_start, actions, include_start=True)

            dream = Dream(
                trajectory=trajectory,
                actions=actions,
                start_state=z_start.copy(),
                dream_type="goal",
            )

            dream = self._score_dream(dream, goal=z_goal)
            dreams.append(dream)

        # Sort by goal score (lower is better)
        dreams.sort(key=lambda d: d.goal_score)

        self._dream_history.extend(dreams)
        logger.debug(f"Found {n} paths to goal, best distance: {dreams[0].goal_score:.3f}")

        return dreams

    def _sample_goal_directed_actions(
        self,
        z_start: np.ndarray,
        z_goal: np.ndarray,
        n_steps: int,
    ) -> np.ndarray:
        """Simple goal-directed action sampling (gradient toward goal)."""
        actions = np.zeros((n_steps, self.action_dim))
        z = z_start.copy()

        for i in range(n_steps):
            # Direction toward goal
            direction = z_goal - z
            direction = direction / (np.linalg.norm(direction) + 1e-8)

            # Project to action space (simple: use first action_dim dims of direction)
            action = direction[:self.action_dim] if len(direction) >= self.action_dim else \
                     np.concatenate([direction, np.zeros(self.action_dim - len(direction))])

            # Add noise
            action += np.random.randn(self.action_dim) * 0.2
            action = np.clip(action, -1, 1)

            actions[i] = action

            # Update z estimate
            z = self.world.predict(z, action)

        return actions

    # =========================================================================
    # Replay Dreaming
    # =========================================================================

    def dream_replay(
        self,
        logged_trajectory: np.ndarray,
        logged_actions: np.ndarray,
        alternative_starts: Optional[List[np.ndarray]] = None,
    ) -> List[Dream]:
        """
        Replay dreams - what could have happened differently?

        Takes a real logged trajectory and explores alternatives:
        - What if we had started from a slightly different state?
        - What if we had taken different actions?
        """
        dreams = []

        # Original trajectory as reference
        original = Dream(
            trajectory=logged_trajectory,
            actions=logged_actions,
            start_state=logged_trajectory[0].copy(),
            dream_type="replay_original",
        )
        dreams.append(original)

        # Alternative starts
        if alternative_starts is not None:
            for alt_start in alternative_starts:
                # Replay same actions from different start
                traj = self.world.rollout(alt_start, logged_actions, include_start=True)
                dream = Dream(
                    trajectory=traj,
                    actions=logged_actions,
                    start_state=alt_start.copy(),
                    dream_type="replay_alt_start",
                )
                dream = self._score_dream(dream)
                dreams.append(dream)

        # Alternative actions (randomized)
        for _ in range(3):
            # Keep half the actions, randomize the rest
            alt_actions = logged_actions.copy()
            mask = np.random.rand(len(alt_actions)) > 0.5
            alt_actions[mask] += np.random.randn(mask.sum(), self.action_dim) * 0.3
            alt_actions = np.clip(alt_actions, -1, 1)

            traj = self.world.rollout(logged_trajectory[0], alt_actions, include_start=True)
            dream = Dream(
                trajectory=traj,
                actions=alt_actions,
                start_state=logged_trajectory[0].copy(),
                dream_type="replay_alt_actions",
            )
            dream = self._score_dream(dream)
            dreams.append(dream)

        self._dream_history.extend(dreams)

        return dreams

    # =========================================================================
    # Branching Futures
    # =========================================================================

    def dream_branches(
        self,
        z_start: np.ndarray,
        branch_point: int = 5,
        n_branches: int = 5,
        steps_after_branch: int = 10,
    ) -> Dict[str, List[Dream]]:
        """
        Branch dreams - shared start, then diverge.

        Creates a "tree" of futures that share a common prefix
        but diverge at a branch point.

        Returns:
            Dict with "trunk" (shared prefix) and "branches" (diverging futures)
        """
        z_start = np.asarray(z_start).flatten()

        # Trunk: shared prefix
        trunk_actions = self._sample_exploratory_actions(branch_point)
        trunk_traj = self.world.rollout(z_start, trunk_actions, include_start=True)

        trunk = Dream(
            trajectory=trunk_traj,
            actions=trunk_actions,
            start_state=z_start.copy(),
            dream_type="branch_trunk",
        )

        # Branch point state
        z_branch = trunk_traj[-1]

        # Create branches
        branches = []
        for i in range(n_branches):
            branch_actions = self._sample_exploratory_actions(steps_after_branch)
            branch_traj = self.world.rollout(z_branch, branch_actions, include_start=True)

            # Full trajectory includes trunk
            full_traj = np.vstack([trunk_traj[:-1], branch_traj])
            full_actions = np.vstack([trunk_actions, branch_actions])

            branch = Dream(
                trajectory=full_traj,
                actions=full_actions,
                start_state=z_start.copy(),
                dream_type=f"branch_{i}",
            )
            branch = self._score_dream(branch)
            branches.append(branch)

        self._dream_history.append(trunk)
        self._dream_history.extend(branches)

        return {
            "trunk": [trunk],
            "branches": branches,
        }

    # =========================================================================
    # Analysis
    # =========================================================================

    def get_dream_statistics(self) -> Dict[str, float]:
        """Get statistics about dream history."""
        if not self._dream_history:
            return {}

        novelties = [d.novelty_score for d in self._dream_history]
        risks = [d.risk_score for d in self._dream_history]

        return {
            "total_dreams": len(self._dream_history),
            "unique_states_visited": len(self._state_visits),
            "avg_novelty": np.mean(novelties),
            "max_novelty": np.max(novelties),
            "avg_risk": np.mean(risks),
            "max_risk": np.max(risks),
        }

    def get_most_novel_dream(self) -> Optional[Dream]:
        """Get the most novel dream from history."""
        if not self._dream_history:
            return None
        return max(self._dream_history, key=lambda d: d.novelty_score)

    def get_safest_dream(self) -> Optional[Dream]:
        """Get the safest (lowest risk) dream from history."""
        if not self._dream_history:
            return None
        return min(self._dream_history, key=lambda d: d.risk_score)


# =============================================================================
# Testing
# =============================================================================

def _test_dreamer():
    """Test dreamer."""
    print("=" * 60)
    print("Dreamer Test")
    print("=" * 60)

    from .world_model import LatentWorldModel, LatentWorldModelConfig

    # Create world model
    config = LatentWorldModelConfig(latent_dim=4, action_dim=2, model_type="linear")
    world = LatentWorldModel(config)

    # Quick training
    np.random.seed(42)
    n = 500
    z = np.random.randn(n, 4)
    u = np.random.randn(n, 2)
    z_next = 0.9 * z + 0.2 * np.hstack([u, u]) + np.random.randn(n, 4) * 0.05
    world.fit(z, u, z_next)

    # Create dreamer
    dreamer = Dreamer(world)

    # Test exploration
    z_start = np.array([0.0, 0.0, 0.0, 0.0])
    explore_dreams = dreamer.dream_explore(z_start, n_dreams=5, max_steps=20)
    print(f"\nExploration: {len(explore_dreams)} dreams")
    print(f"Best novelty: {explore_dreams[0].novelty_score:.3f}")

    # Test counterfactual
    counter_dreams = dreamer.dream_counterfactual(
        z_start,
        modifications={0: 1.0, 1: -0.5},
        n_dreams=3,
    )
    print(f"\nCounterfactual: {len(counter_dreams)} dreams")
    print(f"Start modified: {counter_dreams[0].start_state}")

    # Test goal-directed
    z_goal = np.array([2.0, 2.0, 0.0, 0.0])
    goal_dreams = dreamer.dream_to_goal(z_start, z_goal, n_paths=3)
    print(f"\nGoal-directed: {len(goal_dreams)} paths")
    print(f"Best final distance: {goal_dreams[0].goal_score:.3f}")

    # Test branching
    branches = dreamer.dream_branches(z_start, branch_point=3, n_branches=3)
    print(f"\nBranches: trunk + {len(branches['branches'])} branches")

    # Stats
    stats = dreamer.get_dream_statistics()
    print(f"\nDream stats: {stats}")


if __name__ == "__main__":
    _test_dreamer()
