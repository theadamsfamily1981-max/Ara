# ara_hive/src/__init__.py
"""
HiveHD Orchestration Layer
==========================

This layer sits above the Artificial Bee Colony (ABC) implementation,
providing a higher-level interface that Ara can command.

Lower Layer (existing):
    - WaggleBoard: Shared state for distributed coordination
    - BeeAgent: ABC-based workers that pick sites and execute tasks
    - HiveNode: Multi-agent runner per machine

This Layer (new):
    - QueenOrchestrator: High-level task router (Ara's interface)
    - Workflow: Multi-step task definitions
    - Tool Registry: Available capabilities
    - Pattern Library: Reusable execution strategies

Ara commands the Queen.
The Queen dispatches to Bees.
Bees execute tools.
"""

from .queen import (
    QueenOrchestrator,
    TaskRequest,
    TaskResult,
    RoutingStrategy,
    TaskKind,
)

from .workflow import (
    Workflow,
    WorkflowStep,
    StepStatus,
    WorkflowExecutor,
)

from .registry import (
    ToolRegistry,
    Tool,
    ToolResult,
    get_tool_registry,
)

__all__ = [
    # Queen
    "QueenOrchestrator",
    "TaskRequest",
    "TaskResult",
    "RoutingStrategy",
    "TaskKind",
    # Workflows
    "Workflow",
    "WorkflowStep",
    "StepStatus",
    "WorkflowExecutor",
    # Registry
    "ToolRegistry",
    "Tool",
    "ToolResult",
    "get_tool_registry",
]
