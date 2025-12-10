"""
Ara Swarm Intelligence Layer
============================

Layered intelligence hierarchy with automatic specialization:

    Layer 0: Reflex/worker - tiny context, no tools, safe steps
    Layer 1: Specialist - domain tools, bounded reasoning
    Layer 2: Planner - decomposes goals, routes to specialists
    Layer 3: Governor - monitors, enforces safety, final approvals

Usage:
    from ara_core.swarm import Orchestrator, run_job, get_stats

    # Run a job through the hierarchy
    result = run_job("code_refactor", risk="medium")

    # Get layer statistics
    stats = get_stats()
"""

from .schema import (
    AgentLayer,
    RiskLevel,
    JobOutcome,
    AgentRun,
    JobFix,
    JobRecord,
)

from .orchestrator import (
    Orchestrator,
    run_job,
    get_orchestrator,
)

from .stats import (
    LayerStats,
    PatternStats,
    compute_layer_stats,
    compute_pattern_stats,
    get_optimization_suggestions,
)

from .patterns import (
    Pattern,
    PatternRegistry,
    select_pattern,
)

__all__ = [
    # Schema
    "AgentLayer",
    "RiskLevel",
    "JobOutcome",
    "AgentRun",
    "JobFix",
    "JobRecord",
    # Orchestrator
    "Orchestrator",
    "run_job",
    "get_orchestrator",
    # Stats
    "LayerStats",
    "PatternStats",
    "compute_layer_stats",
    "compute_pattern_stats",
    "get_optimization_suggestions",
    # Patterns
    "Pattern",
    "PatternRegistry",
    "select_pattern",
]
