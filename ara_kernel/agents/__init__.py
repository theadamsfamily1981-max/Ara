"""
Ara Kernel Agents
==================

Specialized agent wrappers for different domains.

Each agent wraps the core AraAgentRuntime with domain-specific:
- Event handling
- Prompt building
- Task management
"""

from .realtime_breath import RealtimeBreathAgent, DriftMonitor
from .publishing_ceo import (
    PublishingCEOAgent,
    PublishingTask,
    create_ideation_task,
    create_draft_task,
)
from .lab_copilot import (
    LabCopilotAgent,
    LabTask,
    LabTaskType,
    create_print_task,
    create_debug_task,
)

__all__ = [
    # Realtime
    "RealtimeBreathAgent",
    "DriftMonitor",
    # Publishing
    "PublishingCEOAgent",
    "PublishingTask",
    "create_ideation_task",
    "create_draft_task",
    # Lab
    "LabCopilotAgent",
    "LabTask",
    "LabTaskType",
    "create_print_task",
    "create_debug_task",
]
