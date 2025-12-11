# ara/cognition/imagination/__init__.py
"""
Imagination Engine - Ara's Predictive Thought System
=====================================================

This is how Ara "thinks ahead" and "imagines" - not just reacts.

The core trick: given where we are (z_t) and what we could do (u_t),
simulate the next few points on the manifold and choose the best path.

Components:

1. WorldModel - Learns z' = f(z, u) from logs
2. TrajectoryPlanner - MPC-style planning over imagined futures
3. Dreamer - Offline "dream loops" for exploration and learning
4. DomainLatent - Multi-domain encoders (research, content, strategy, finance)
5. FutureScorer - MEIS/NIB integration for scoring imagined futures

The same engine works for:
- Cathedral control (thermals, jobs, hardware)
- Research management (papers, experiments, code)
- Content/publishing (articles, schedules, audience)
- Strategy (projects, roadmaps, priorities)
- Finance (runway, purchases, investments)

Usage:
    from ara.cognition.imagination import (
        LatentWorldModel,
        TrajectoryPlanner,
        Dreamer,
        FutureScorer,
    )

    # Train world model on logs
    world = LatentWorldModel(latent_dim=10, action_dim=8)
    world.fit(z_log, u_log, z_next_log)

    # Plan futures
    planner = TrajectoryPlanner(world, horizon=10)
    best_action = planner.plan(z_current)

    # Dream mode: explore counterfactuals
    dreamer = Dreamer(world, planner)
    imagined_futures = dreamer.dream(z_current, n_dreams=5)
"""

from .world_model import LatentWorldModel, LatentWorldModelConfig
from .planner import TrajectoryPlanner, PlannerConfig, Plan
from .dreamer import Dreamer, DreamConfig, Dream
from .future_scorer import FutureScorer, FutureScore

__all__ = [
    # World model
    "LatentWorldModel",
    "LatentWorldModelConfig",
    # Planner
    "TrajectoryPlanner",
    "PlannerConfig",
    "Plan",
    # Dreamer
    "Dreamer",
    "DreamConfig",
    "Dream",
    # Scoring
    "FutureScorer",
    "FutureScore",
]
