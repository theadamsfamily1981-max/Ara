"""
CADD - Collective Association Drift Detection
==============================================

Entropy-based bias detection mapping to T_s_bias drift.
Detects "emergent sociological bias" when high T_s + label shift occurs.

Cathedral Immune System:
    - H_influence < 1.2 bits → Kill monoculture agents
    - Association matrix A_t^i(c,s) → T_s monitoring
    - MEIS governance auto-corrects swarm topology

Detection Metrics:
    - Association Entropy: H(A) per agent
    - Collective Drift: ΔH across swarm
    - Monoculture Index: max contribution dominance
    - Bias Geometry: Distribution of associations

Interventions:
    - INJECT_DIVERSITY: Spawn 3x morons with orthogonal priors
    - PAUSE_CONSOLIDATION: Stop learning during drift
    - MANUAL_REVIEW: Human-in-the-loop for critical drift

Usage:
    from ara_core.cadd import (
        CADDSentinel, AssociationMatrix, DriftAlert,
        sentinel_tick, get_sentinel, sentinel_status,
        DrugSynthesisOrchestrator, CodeSnippetConfig, JobResult
    )

    # Initialize sentinel
    sentinel = get_sentinel(n_agents=1000)

    # Update associations
    sentinel.update_association(agent_id, concept, signal, strength)

    # Check for drift
    alerts = sentinel_tick()
    if alerts:
        for alert in alerts:
            handle_alert(alert)

    # Use orchestrator for gated execution
    orch = DrugSynthesisOrchestrator(sentinel, run_code_fn=sandbox.run)
    result = orch.run_synthesis_job("job_1", "code")
"""

from .sentinel import (
    DriftType,
    DriftAlert,
    AssociationMatrix,
    AgentProfile,
    CADDSentinel,
    SentinelConfig,
    get_sentinel,
    sentinel_tick,
    sentinel_status,
    sentinel_alerts,
)

from .orchestrator import (
    DrugSynthesisOrchestrator,
    CodeSnippetConfig,
    JobResult,
)

__all__ = [
    # Sentinel
    "DriftType",
    "DriftAlert",
    "AssociationMatrix",
    "AgentProfile",
    "CADDSentinel",
    "SentinelConfig",
    "get_sentinel",
    "sentinel_tick",
    "sentinel_status",
    "sentinel_alerts",
    # Orchestrator
    "DrugSynthesisOrchestrator",
    "CodeSnippetConfig",
    "JobResult",
]
