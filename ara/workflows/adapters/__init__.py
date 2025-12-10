# ara/workflows/adapters/__init__.py
"""
Workflow Engine Adapters
========================

Thin wrappers around workflow engines (Temporal, n8n, etc.)
that expose a common interface for Ara to control.

The engine does the heavy lifting (retries, scheduling, persistence).
Ara makes the decisions (which step, what to tell user, what to learn).
"""

from .base import (
    EngineClient,
    StepDefinition,
    StepResult,
    WorkflowDefinition,
)

__all__ = [
    "EngineClient",
    "StepDefinition",
    "StepResult",
    "WorkflowDefinition",
]
