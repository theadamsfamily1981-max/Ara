# ara/workflows/adapters/temporal.py
"""
Temporal Workflow Engine Adapter
================================

Connects Ara to Temporal (https://temporal.io).

Temporal provides:
    - Durable execution
    - Automatic retries
    - Workflow versioning
    - Activity heartbeating
    - Signal/query support

Ara provides:
    - Decision-making (which activity next?)
    - User guidance (what to tell them?)
    - Learning (outcomes → MEIS/QUANTA)

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │                 AraSelfGuidedOrchestrator                │
    │                        │                                 │
    │                        ▼                                 │
    │                 TemporalAdapter                          │
    │                        │                                 │
    │          ┌─────────────┴─────────────┐                  │
    │          ▼                           ▼                  │
    │   WorkflowClient              ActivityClient            │
    │   (start, signal, query)     (execute tasks)            │
    └─────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────┐
    │                   Temporal Server                        │
    │   (schedules, persists, retries, distributes)           │
    └─────────────────────────────────────────────────────────┘

Usage:
    from ara.workflows.adapters.temporal import TemporalAdapter

    adapter = TemporalAdapter(
        host="localhost:7233",
        namespace="ara",
        task_queue="ara-tasks",
    )

    # Register workflow
    adapter.register_workflow(my_workflow_def)

    # Ara orchestrator uses this adapter
    orchestrator = AraSelfGuidedOrchestrator(engine=adapter)
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

from .base import (
    BaseEngineAdapter,
    StepDefinition,
    StepResult,
    StepStatus,
    StepType,
    WorkflowDefinition,
    WorkflowExecution,
    WorkflowStatus,
)

log = logging.getLogger("Ara.Workflows.Temporal")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TemporalConfig:
    """Configuration for Temporal connection."""
    host: str = "localhost:7233"
    namespace: str = "default"
    task_queue: str = "ara-tasks"

    # TLS
    tls_enabled: bool = False
    tls_cert_path: Optional[str] = None
    tls_key_path: Optional[str] = None

    # Timeouts
    default_workflow_timeout: int = 3600  # 1 hour
    default_activity_timeout: int = 300   # 5 minutes
    default_heartbeat_timeout: int = 60   # 1 minute

    # Retries
    max_retries: int = 3
    initial_interval_seconds: int = 1
    backoff_coefficient: float = 2.0
    max_interval_seconds: int = 60


# =============================================================================
# Activity Registry
# =============================================================================

@dataclass
class ActivityDefinition:
    """Definition of a Temporal activity."""
    name: str
    handler: Callable
    timeout_seconds: int = 300
    heartbeat_timeout_seconds: int = 60
    max_retries: int = 3
    description: str = ""


class ActivityRegistry:
    """
    Registry of activities that Ara can invoke via Temporal.

    Activities are the "muscles" - they do actual work.
    Ara decides which activities to run and when.
    """

    def __init__(self):
        self._activities: Dict[str, ActivityDefinition] = {}

    def register(
        self,
        name: str,
        handler: Callable,
        timeout_seconds: int = 300,
        description: str = "",
    ) -> None:
        """Register an activity handler."""
        self._activities[name] = ActivityDefinition(
            name=name,
            handler=handler,
            timeout_seconds=timeout_seconds,
            description=description,
        )
        log.info("Registered Temporal activity: %s", name)

    def get(self, name: str) -> Optional[ActivityDefinition]:
        return self._activities.get(name)

    def list(self) -> List[ActivityDefinition]:
        return list(self._activities.values())


# Global activity registry
_activity_registry = ActivityRegistry()


def activity(
    name: Optional[str] = None,
    timeout_seconds: int = 300,
    description: str = "",
):
    """
    Decorator to register a function as a Temporal activity.

    Usage:
        @activity(name="fetch_user", timeout_seconds=60)
        async def fetch_user(user_id: str) -> dict:
            ...
    """
    def decorator(func: Callable) -> Callable:
        activity_name = name or func.__name__
        _activity_registry.register(
            name=activity_name,
            handler=func,
            timeout_seconds=timeout_seconds,
            description=description or func.__doc__ or "",
        )
        return func
    return decorator


# =============================================================================
# Temporal Adapter
# =============================================================================

class TemporalAdapter(BaseEngineAdapter):
    """
    Adapter that connects Ara to Temporal workflow engine.

    This is a simulation/mock for development.
    In production, replace _execute_activity with actual Temporal calls.
    """

    def __init__(
        self,
        config: Optional[TemporalConfig] = None,
    ):
        super().__init__(name="temporal")
        self.config = config or TemporalConfig()
        self.activity_registry = _activity_registry

        # Simulation state
        self._pending_signals: Dict[str, List[tuple]] = {}

        log.info(
            "TemporalAdapter initialized: host=%s, namespace=%s, queue=%s",
            self.config.host,
            self.config.namespace,
            self.config.task_queue,
        )

    # =========================================================================
    # EngineClient Implementation
    # =========================================================================

    async def start_workflow(
        self,
        workflow_id: str,
        initial_state: Dict[str, Any],
    ) -> WorkflowExecution:
        """
        Start a new workflow execution.

        In production: calls temporal_client.start_workflow()
        """
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow not found: {workflow_id}")

        execution_id = f"{workflow_id}-{uuid.uuid4().hex[:8]}"

        execution = WorkflowExecution(
            workflow_id=workflow_id,
            execution_id=execution_id,
            status=WorkflowStatus.RUNNING,
            state=initial_state.copy(),
            started_at=datetime.utcnow(),
        )

        self._executions[execution_id] = execution

        log.info(
            "Started Temporal workflow: %s (execution=%s)",
            workflow_id, execution_id,
        )

        return execution

    async def run_step(
        self,
        execution_id: str,
        step_id: str,
        state: Dict[str, Any],
    ) -> StepResult:
        """
        Execute a single step (activity) in the workflow.

        In production: this would be handled by Temporal's activity worker.
        Here we simulate direct execution.
        """
        execution = self._executions.get(execution_id)
        if not execution:
            raise ValueError(f"Execution not found: {execution_id}")

        workflow = self._workflows.get(execution.workflow_id)
        if not workflow:
            raise ValueError(f"Workflow not found: {execution.workflow_id}")

        step = workflow.get_step(step_id)
        if not step:
            raise ValueError(f"Step not found: {step_id}")

        result = StepResult(
            step_id=step_id,
            status=StepStatus.RUNNING,
            started_at=datetime.utcnow(),
        )

        execution.current_step = step_id

        try:
            if step.type == StepType.ACTIVITY:
                output = await self._execute_activity(step, state)
                result.output = output
                result.status = StepStatus.COMPLETED

            elif step.type == StepType.DECISION:
                branch = await self._evaluate_decision(step, state)
                result.branch_taken = branch
                result.status = StepStatus.COMPLETED

            elif step.type == StepType.HUMAN:
                # Wait for signal
                result = await self._wait_for_human_input(
                    execution_id, step, state
                )

            elif step.type == StepType.SIGNAL:
                result = await self._wait_for_signal(
                    execution_id, step.meta.get("signal_name", step_id)
                )

            elif step.type == StepType.TIMER:
                await self._wait_timer(step)
                result.status = StepStatus.COMPLETED

            elif step.type == StepType.SUBWORKFLOW:
                output = await self._run_subworkflow(step, state)
                result.output = output
                result.status = StepStatus.COMPLETED

            else:
                # Unknown step type
                result.status = StepStatus.COMPLETED

        except Exception as e:
            log.error("Step %s failed: %s", step_id, e)
            result.status = StepStatus.FAILED
            result.error = str(e)

        result.completed_at = datetime.utcnow()

        # Update execution state
        if result.success:
            execution.completed_steps.append(step_id)
            if result.output:
                # Merge output into state
                if isinstance(result.output, dict):
                    execution.state.update(result.output)
                else:
                    execution.state[step_id] = result.output

        execution.step_results.append(result)
        execution.current_step = None

        log.info(
            "Step %s completed: status=%s, duration=%.2fs",
            step_id, result.status.value, result.duration_seconds,
        )

        return result

    async def cancel_execution(
        self,
        execution_id: str,
        reason: str = "",
    ) -> bool:
        """Cancel a running workflow."""
        execution = self._executions.get(execution_id)
        if not execution:
            return False

        execution.status = WorkflowStatus.CANCELLED
        execution.completed_at = datetime.utcnow()
        execution.error = reason or "Cancelled by user"

        log.info("Cancelled execution: %s (%s)", execution_id, reason)
        return True

    async def send_signal(
        self,
        execution_id: str,
        signal_name: str,
        payload: Any = None,
    ) -> bool:
        """Send a signal to a workflow (for human input, etc.)."""
        if execution_id not in self._pending_signals:
            self._pending_signals[execution_id] = []

        self._pending_signals[execution_id].append((signal_name, payload))
        log.info("Signal sent: %s -> %s", signal_name, execution_id)
        return True

    # =========================================================================
    # Activity Execution (Simulation)
    # =========================================================================

    async def _execute_activity(
        self,
        step: StepDefinition,
        state: Dict[str, Any],
    ) -> Any:
        """
        Execute an activity step.

        In production: Temporal worker handles this.
        Here we call the registered handler directly.
        """
        activity_name = step.meta.get("activity") or step.id

        activity_def = self.activity_registry.get(activity_name)
        if activity_def:
            # Call registered handler
            handler = activity_def.handler
            if asyncio.iscoroutinefunction(handler):
                return await handler(state)
            else:
                return handler(state)

        # No handler registered - simulate success
        log.warning("No handler for activity: %s (simulating success)", activity_name)
        return {"simulated": True, "activity": activity_name}

    async def _evaluate_decision(
        self,
        step: StepDefinition,
        state: Dict[str, Any],
    ) -> str:
        """Evaluate a decision step and return the branch to take."""
        if step.condition:
            # Simple expression evaluation
            try:
                result = eval(step.condition, {"state": state})
                return str(result)
            except Exception as e:
                log.warning("Decision condition failed: %s", e)

        # Default to first branch
        return step.branches[0] if step.branches else "default"

    async def _wait_for_human_input(
        self,
        execution_id: str,
        step: StepDefinition,
        state: Dict[str, Any],
    ) -> StepResult:
        """Wait for human input via signal."""
        result = StepResult(
            step_id=step.id,
            status=StepStatus.RUNNING,
            started_at=datetime.utcnow(),
        )

        # In production: workflow would pause here.
        # For simulation, check for pending signals.
        timeout = step.timeout_seconds
        start = datetime.utcnow()

        while (datetime.utcnow() - start).total_seconds() < timeout:
            signals = self._pending_signals.get(execution_id, [])
            for signal_name, payload in signals:
                if signal_name == step.id or signal_name == "human_input":
                    result.human_input = payload
                    result.status = StepStatus.COMPLETED
                    signals.remove((signal_name, payload))
                    return result

            await asyncio.sleep(0.1)

        # Timeout
        result.status = StepStatus.FAILED
        result.error = "Human input timeout"
        return result

    async def _wait_for_signal(
        self,
        execution_id: str,
        signal_name: str,
    ) -> StepResult:
        """Wait for a named signal."""
        result = StepResult(
            step_id=signal_name,
            status=StepStatus.RUNNING,
            started_at=datetime.utcnow(),
        )

        # Check pending signals
        signals = self._pending_signals.get(execution_id, [])
        for name, payload in signals:
            if name == signal_name:
                result.output = payload
                result.status = StepStatus.COMPLETED
                signals.remove((name, payload))
                return result

        # For simulation, just complete
        result.status = StepStatus.COMPLETED
        return result

    async def _wait_timer(self, step: StepDefinition) -> None:
        """Wait for a timer duration."""
        duration = step.meta.get("duration_seconds", 1)
        await asyncio.sleep(min(duration, 1))  # Cap at 1s for simulation

    async def _run_subworkflow(
        self,
        step: StepDefinition,
        state: Dict[str, Any],
    ) -> Any:
        """Run a sub-workflow."""
        subworkflow_id = step.meta.get("workflow_id")
        if not subworkflow_id:
            return {"error": "No sub-workflow specified"}

        # Recursive execution
        execution = await self.start_workflow(subworkflow_id, state)

        # Run all ready steps
        workflow = self._workflows.get(subworkflow_id)
        if workflow:
            while execution.status == WorkflowStatus.RUNNING:
                ready = workflow.get_ready_steps(execution.completed_steps)
                if not ready:
                    break

                for ready_step in ready:
                    await self.run_step(
                        execution.execution_id,
                        ready_step.id,
                        execution.state,
                    )

        return execution.state


# =============================================================================
# Built-in Activities (for testing)
# =============================================================================

@activity(name="echo", timeout_seconds=10, description="Echo input back")
async def echo_activity(state: Dict[str, Any]) -> Dict[str, Any]:
    """Simple echo activity for testing."""
    return {"echo": state}


@activity(name="delay", timeout_seconds=60, description="Delay execution")
async def delay_activity(state: Dict[str, Any]) -> Dict[str, Any]:
    """Delay for specified seconds."""
    seconds = state.get("delay_seconds", 1)
    await asyncio.sleep(min(seconds, 5))  # Cap at 5s for safety
    return {"delayed": seconds}


@activity(name="validate", timeout_seconds=30, description="Validate input")
async def validate_activity(state: Dict[str, Any]) -> Dict[str, Any]:
    """Validate required fields are present."""
    required = state.get("required_fields", [])
    missing = [f for f in required if f not in state]

    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    return {"valid": True, "checked": required}


# =============================================================================
# Convenience
# =============================================================================

_default_adapter: Optional[TemporalAdapter] = None


def get_temporal_adapter(config: Optional[TemporalConfig] = None) -> TemporalAdapter:
    """Get the default Temporal adapter."""
    global _default_adapter
    if _default_adapter is None:
        _default_adapter = TemporalAdapter(config)
    return _default_adapter


__all__ = [
    "TemporalConfig",
    "TemporalAdapter",
    "ActivityRegistry",
    "ActivityDefinition",
    "activity",
    "get_temporal_adapter",
]
