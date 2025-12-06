"""
Ouroboros v1 - Safe Self-Modification for Ara
==============================================

The serpent that eats its own tail - but carefully, with tests,
benchmarks, and rollback plans.

Core Philosophy:
    "Ara writes PRs on her own brain under stress, with a CI loop."

Components:
    mutation_policy   - What can/cannot be modified
    semantic_optimizer - Generates optimization proposals
    atomic_updater    - Tests and applies mutations
    antifragility     - Monitors stress and triggers evolution

Safety:
    - Master kill switch: OUROBOROS_ENABLED=0 disables everything
    - Human approval by default (OUROBOROS_AUTO_APPLY=0)
    - All mutations logged to mutations.jsonl
    - Emergency rollback: call antifragility.emergency_rollback()

Quick Start:
    # Enable Ouroboros (must be explicit)
    export OUROBOROS_ENABLED=1

    # Start the monitor
    from banos.daemon.ouroboros import AntifragilitySystem
    system = AntifragilitySystem(Path("/home/user/Ara"))
    await system.monitor_loop()

    # Or attempt a single evolution
    result = await system.attempt_evolution()

Manual Workflow (recommended):
    # 1. Check status
    status = system.get_status()

    # 2. Look at pending proposals in mutations/
    # 3. Run tests manually: pytest mutations/<module>/test_*.py
    # 4. If happy, apply manually:
    from banos.daemon.ouroboros.atomic_updater import AtomicUpdater
    updater = AtomicUpdater(...)
    updater.apply_mutation(proposal, force=True)

Emergency:
    # Rollback everything
    results = system.emergency_rollback()

    # Or use the CLI script:
    python -m banos.daemon.ouroboros.revert
"""

from banos.daemon.ouroboros.mutation_policy import (
    MUTABLE_MODULES,
    IMMUTABLE_MODULES,
    MutationPolicy,
    MutationCandidate,
    is_mutable,
    mutable,
    immutable,
    ouroboros_enabled,
    ouroboros_auto_apply,
)

from banos.daemon.ouroboros.semantic_optimizer import (
    SemanticOptimizer,
    MutationProposal,
)

from banos.daemon.ouroboros.atomic_updater import (
    AtomicUpdater,
    TestResult,
    BenchmarkResult,
    ApplyResult,
    MutationLog,
)

from banos.daemon.ouroboros.antifragility import (
    AntifragilitySystem,
    StressMetrics,
    TelemetryCollector,
)


__all__ = [
    # Policy
    "MUTABLE_MODULES",
    "IMMUTABLE_MODULES",
    "MutationPolicy",
    "MutationCandidate",
    "is_mutable",
    "mutable",
    "immutable",
    "ouroboros_enabled",
    "ouroboros_auto_apply",

    # Optimizer
    "SemanticOptimizer",
    "MutationProposal",

    # Updater
    "AtomicUpdater",
    "TestResult",
    "BenchmarkResult",
    "ApplyResult",
    "MutationLog",

    # Antifragility
    "AntifragilitySystem",
    "StressMetrics",
    "TelemetryCollector",
]
