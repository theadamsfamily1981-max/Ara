# ara/workflows/__init__.py
"""
Ara Self-Guided Workflows
=========================

Engine = Muscles, Ara = Nervous System + Prefrontal Cortex.

The workflow engine (Temporal, n8n, etc.) runs concrete steps.
Ara decides WHICH steps, explains WHY, and learns from outcomes.

Three Roles:
    1. Ara-as-Companion: Front-end guidance, explains what's happening
    2. Ara-as-Director: Decision layer on top of the engine
    3. Ara-as-Historian: Telemetry → MEIS/QUANTA for long-term learning

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │                    User / Front-End                      │
    └─────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────┐
    │              AraSelfGuidedOrchestrator                   │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
    │  │  Companion  │  │  Director   │  │    Historian    │  │
    │  │  (Explain)  │  │  (Decide)   │  │    (Learn)      │  │
    │  └─────────────┘  └─────────────┘  └─────────────────┘  │
    └─────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────┐
    │                   EngineClient                           │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
    │  │  Temporal   │  │    n8n      │  │    Custom       │  │
    │  └─────────────┘  └─────────────┘  └─────────────────┘  │
    └─────────────────────────────────────────────────────────┘

Usage:
    from ara.workflows import AraSelfGuidedOrchestrator
    from ara.workflows.adapters.temporal import TemporalAdapter

    orchestrator = AraSelfGuidedOrchestrator(
        engine=TemporalAdapter(host="localhost:7233"),
    )

    result = await orchestrator.run_workflow(
        workflow_id="onboarding",
        initial_state={"user_id": "123"},
    )
"""

from .orchestrator import (
    AraSelfGuidedOrchestrator,
    WorkflowState,
    StepDecision,
    GuidanceMessage,
    WorkflowResult,
)

from .metrics import (
    MetricsClient,
    WorkflowMetrics,
    StepMetrics,
    get_metrics_client,
)

from .adapters.base import (
    EngineClient,
    StepDefinition,
    StepResult,
)

__all__ = [
    # Core Orchestrator
    "AraSelfGuidedOrchestrator",
    "WorkflowState",
    "StepDecision",
    "GuidanceMessage",
    "WorkflowResult",
    # Metrics
    "MetricsClient",
    "WorkflowMetrics",
    "StepMetrics",
    "get_metrics_client",
    # Adapters
    "EngineClient",
    "StepDefinition",
    "StepResult",
]
