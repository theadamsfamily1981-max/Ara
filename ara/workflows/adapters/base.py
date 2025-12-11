# ara/workflows/adapters/base.py
"""
Base Protocol for Workflow Engine Adapters
==========================================

Defines the interface that Ara uses to control workflow engines.
Engines implement this protocol; Ara calls these methods.

Protocol Methods:
    - list_steps(workflow_id) -> List[StepDefinition]
    - run_step(step_id, state) -> StepResult
    - get_workflow_status(workflow_id) -> WorkflowStatus
    - cancel_workflow(workflow_id) -> bool
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

log = logging.getLogger("Ara.Workflows.Adapters")


# =============================================================================
# Data Structures
# =============================================================================

class StepType(str, Enum):
    """Types of workflow steps."""
    ACTIVITY = "activity"      # Compute/IO task
    DECISION = "decision"      # Branch point
    HUMAN = "human"            # Requires human input
    SUBWORKFLOW = "subworkflow"  # Nested workflow
    SIGNAL = "signal"          # Wait for external signal
    TIMER = "timer"            # Wait for time


class StepStatus(str, Enum):
    """Status of a step execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class WorkflowStatus(str, Enum):
    """Status of a workflow execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class StepDefinition:
    """
    Definition of a workflow step.

    The engine provides these; Ara decides which to run.
    """
    id: str
    name: str
    type: StepType

    # Metadata for Ara's decision-making
    description: str = ""
    requires: List[str] = field(default_factory=list)  # Dependency step IDs
    timeout_seconds: int = 300

    # For human steps
    prompt_template: Optional[str] = None
    input_schema: Optional[Dict[str, Any]] = None

    # For decision steps
    branches: List[str] = field(default_factory=list)
    condition: Optional[str] = None  # Expression to evaluate

    # Metadata
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "description": self.description,
            "requires": self.requires,
            "timeout_seconds": self.timeout_seconds,
            "prompt_template": self.prompt_template,
            "branches": self.branches,
        }


@dataclass
class StepResult:
    """
    Result of executing a workflow step.

    The engine returns these; Ara learns from them.
    """
    step_id: str
    status: StepStatus

    # Output
    output: Any = None
    error: Optional[str] = None

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # For human steps
    human_input: Optional[Dict[str, Any]] = None

    # For decision steps
    branch_taken: Optional[str] = None

    # Metadata
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status == StepStatus.COMPLETED

    @property
    def duration_seconds(self) -> float:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "status": self.status.value,
            "success": self.success,
            "output": str(self.output)[:500] if self.output else None,
            "error": self.error,
            "duration_seconds": self.duration_seconds,
            "branch_taken": self.branch_taken,
        }


@dataclass
class WorkflowDefinition:
    """
    Definition of a complete workflow.

    Contains all steps and their relationships.
    """
    id: str
    name: str
    description: str = ""

    # Steps
    steps: List[StepDefinition] = field(default_factory=list)

    # Entry/exit
    entry_step: Optional[str] = None
    exit_steps: List[str] = field(default_factory=list)

    # Metadata
    version: str = "1.0.0"
    meta: Dict[str, Any] = field(default_factory=dict)

    def get_step(self, step_id: str) -> Optional[StepDefinition]:
        """Get step by ID."""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None

    def get_ready_steps(self, completed: List[str]) -> List[StepDefinition]:
        """Get steps whose dependencies are all satisfied."""
        ready = []
        for step in self.steps:
            if step.id in completed:
                continue
            if all(req in completed for req in step.requires):
                ready.append(step)
        return ready


@dataclass
class WorkflowExecution:
    """
    State of a running workflow.
    """
    workflow_id: str
    execution_id: str
    status: WorkflowStatus

    # Current state
    state: Dict[str, Any] = field(default_factory=dict)
    completed_steps: List[str] = field(default_factory=list)
    current_step: Optional[str] = None

    # History
    step_results: List[StepResult] = field(default_factory=list)

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Error info
    error: Optional[str] = None


# =============================================================================
# Engine Client Protocol
# =============================================================================

@runtime_checkable
class EngineClient(Protocol):
    """
    Protocol for workflow engine adapters.

    Implement this to connect Ara to your workflow engine.
    The engine handles: retries, scheduling, persistence.
    Ara handles: decisions, guidance, learning.
    """

    @abstractmethod
    async def list_workflows(self) -> List[WorkflowDefinition]:
        """
        List all available workflow definitions.

        Returns:
            List of workflow definitions the engine knows about
        """
        ...

    @abstractmethod
    async def get_workflow(self, workflow_id: str) -> Optional[WorkflowDefinition]:
        """
        Get a specific workflow definition.

        Args:
            workflow_id: ID of the workflow

        Returns:
            Workflow definition or None if not found
        """
        ...

    @abstractmethod
    async def list_steps(self, workflow_id: str) -> List[StepDefinition]:
        """
        List all steps in a workflow.

        Args:
            workflow_id: ID of the workflow

        Returns:
            List of step definitions
        """
        ...

    @abstractmethod
    async def start_workflow(
        self,
        workflow_id: str,
        initial_state: Dict[str, Any],
    ) -> WorkflowExecution:
        """
        Start a new workflow execution.

        Args:
            workflow_id: ID of the workflow to start
            initial_state: Initial state/inputs

        Returns:
            Execution handle
        """
        ...

    @abstractmethod
    async def run_step(
        self,
        execution_id: str,
        step_id: str,
        state: Dict[str, Any],
    ) -> StepResult:
        """
        Execute a single step in a workflow.

        This is the core method that runs actual work.
        The engine handles retries, timeouts, etc.

        Args:
            execution_id: ID of the workflow execution
            step_id: ID of the step to run
            state: Current workflow state

        Returns:
            Result of the step execution
        """
        ...

    @abstractmethod
    async def get_execution_status(
        self,
        execution_id: str,
    ) -> WorkflowExecution:
        """
        Get current status of a workflow execution.

        Args:
            execution_id: ID of the execution

        Returns:
            Current execution state
        """
        ...

    @abstractmethod
    async def cancel_execution(
        self,
        execution_id: str,
        reason: str = "",
    ) -> bool:
        """
        Cancel a running workflow execution.

        Args:
            execution_id: ID of the execution to cancel
            reason: Why it's being cancelled

        Returns:
            True if cancelled successfully
        """
        ...

    @abstractmethod
    async def send_signal(
        self,
        execution_id: str,
        signal_name: str,
        payload: Any = None,
    ) -> bool:
        """
        Send a signal to a running workflow.

        Used for human input, external events, etc.

        Args:
            execution_id: ID of the execution
            signal_name: Name of the signal
            payload: Signal data

        Returns:
            True if signal was delivered
        """
        ...


# =============================================================================
# Base Adapter (for subclassing)
# =============================================================================

class BaseEngineAdapter(ABC):
    """
    Base class for engine adapters.

    Provides common functionality; subclasses implement engine-specific logic.
    """

    def __init__(self, name: str = "base"):
        self.name = name
        self._workflows: Dict[str, WorkflowDefinition] = {}
        self._executions: Dict[str, WorkflowExecution] = {}
        log.info("Engine adapter initialized: %s", name)

    def register_workflow(self, workflow: WorkflowDefinition) -> None:
        """Register a workflow definition."""
        self._workflows[workflow.id] = workflow
        log.info("Registered workflow: %s", workflow.id)

    async def list_workflows(self) -> List[WorkflowDefinition]:
        return list(self._workflows.values())

    async def get_workflow(self, workflow_id: str) -> Optional[WorkflowDefinition]:
        return self._workflows.get(workflow_id)

    async def list_steps(self, workflow_id: str) -> List[StepDefinition]:
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            return []
        return workflow.steps

    async def get_execution_status(
        self,
        execution_id: str,
    ) -> WorkflowExecution:
        execution = self._executions.get(execution_id)
        if not execution:
            raise ValueError(f"Execution not found: {execution_id}")
        return execution

    # Subclasses must implement:
    # - start_workflow
    # - run_step
    # - cancel_execution
    # - send_signal


__all__ = [
    # Enums
    "StepType",
    "StepStatus",
    "WorkflowStatus",
    # Data classes
    "StepDefinition",
    "StepResult",
    "WorkflowDefinition",
    "WorkflowExecution",
    # Protocol
    "EngineClient",
    # Base class
    "BaseEngineAdapter",
]
