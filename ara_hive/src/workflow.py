# ara_hive/src/workflow.py
"""
Workflow Primitives for the Hive
================================

Workflows are multi-step task definitions that the Hive executes.
They support dependencies, branching, and parallel execution.

This is distinct from ara/workflows/ which provides Ara's
self-guided orchestration on TOP of workflow engines.
Here, we define the workflow structure that RUNS inside the Hive.

Workflow Structure:
    Workflow
    └── WorkflowStep (many)
        ├── name: Step identifier
        ├── tool: Tool to execute
        ├── params: Tool parameters
        ├── depends_on: Dependencies
        └── on_failure: Failure handling

Execution Modes:
    - SEQUENTIAL: Steps run in order
    - PARALLEL: Independent steps run concurrently
    - CONDITIONAL: Steps run based on conditions
    - RETRY: Steps retry on failure

Usage:
    from ara_hive.src.workflow import Workflow, WorkflowStep

    workflow = Workflow(
        name="data_pipeline",
        steps=[
            WorkflowStep(name="fetch", tool="web_fetch", params={"url": "..."}),
            WorkflowStep(name="parse", tool="json_parse", depends_on=["fetch"]),
            WorkflowStep(name="store", tool="db_insert", depends_on=["parse"]),
        ],
    )

    executor = WorkflowExecutor(queen)
    result = await executor.run(workflow, initial_state={})
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .queen import QueenOrchestrator, TaskRequest, TaskResult

log = logging.getLogger("Hive.Workflow")


# =============================================================================
# Types
# =============================================================================

class StepStatus(str, Enum):
    """Status of a workflow step."""
    PENDING = "pending"
    WAITING = "waiting"      # Waiting for dependencies
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class FailureAction(str, Enum):
    """What to do when a step fails."""
    ABORT = "abort"          # Stop the workflow
    CONTINUE = "continue"    # Continue with next steps
    RETRY = "retry"          # Retry the step
    FALLBACK = "fallback"    # Use fallback step


@dataclass
class WorkflowStep:
    """
    A single step in a workflow.

    Executes a tool with parameters.
    Can depend on other steps completing first.
    """
    name: str
    tool: str
    params: Dict[str, Any] = field(default_factory=dict)

    # Dependencies
    depends_on: List[str] = field(default_factory=list)

    # Failure handling
    on_failure: FailureAction = FailureAction.ABORT
    max_retries: int = 0
    retry_delay_seconds: float = 1.0
    fallback_step: Optional[str] = None

    # Conditional execution
    condition: Optional[str] = None  # Python expression

    # Timing
    timeout_seconds: int = 60

    # Runtime state
    status: StepStatus = StepStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    attempts: int = 0

    @property
    def duration_ms(self) -> float:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return 0.0

    def is_ready(self, completed_steps: set[str]) -> bool:
        """Check if all dependencies are satisfied."""
        return all(dep in completed_steps for dep in self.depends_on)

    def should_run(self, state: Dict[str, Any]) -> bool:
        """Check if step should run based on condition."""
        if not self.condition:
            return True
        try:
            return eval(self.condition, {"state": state})
        except Exception as e:
            log.warning(f"Condition evaluation failed for {self.name}: {e}")
            return False

    def get_params(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Get parameters with state interpolation."""
        params = {}
        for key, value in self.params.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                # Interpolate from state
                path = value[2:-1]
                params[key] = self._get_from_state(state, path)
            else:
                params[key] = value
        return params

    def _get_from_state(self, state: Dict[str, Any], path: str) -> Any:
        """Get value from state by dotted path."""
        parts = path.split(".")
        current = state
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        return current

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "tool": self.tool,
            "status": self.status.value,
            "attempts": self.attempts,
            "duration_ms": self.duration_ms,
            "error": self.error,
        }


@dataclass
class Workflow:
    """
    A multi-step workflow definition.

    Workflows are executed by the WorkflowExecutor.
    """
    name: str
    steps: List[WorkflowStep] = field(default_factory=list)
    description: str = ""

    # Metadata
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)

    # Global settings
    timeout_seconds: int = 3600  # 1 hour
    max_parallel: int = 10

    # Callbacks
    on_start: Optional[Callable] = None
    on_complete: Optional[Callable] = None
    on_step_complete: Optional[Callable] = None

    def get_step(self, name: str) -> Optional[WorkflowStep]:
        """Get step by name."""
        for step in self.steps:
            if step.name == name:
                return step
        return None

    def get_ready_steps(self, completed: set[str]) -> List[WorkflowStep]:
        """Get steps that are ready to run."""
        ready = []
        for step in self.steps:
            if step.status == StepStatus.PENDING and step.is_ready(completed):
                ready.append(step)
        return ready

    def get_execution_order(self) -> List[List[str]]:
        """
        Get steps grouped by execution level.

        Returns list of lists: [[level0_steps], [level1_steps], ...]
        Steps in the same level can run in parallel.
        """
        levels = []
        remaining = {s.name for s in self.steps}
        completed = set()

        while remaining:
            # Find steps with all dependencies satisfied
            level = []
            for step in self.steps:
                if step.name in remaining and step.is_ready(completed):
                    level.append(step.name)

            if not level:
                # Circular dependency or invalid workflow
                log.error(f"Cannot resolve dependencies. Remaining: {remaining}")
                break

            levels.append(level)
            remaining -= set(level)
            completed.update(level)

        return levels

    def validate(self) -> List[str]:
        """Validate workflow definition. Returns list of errors."""
        errors = []

        # Check for duplicate names
        names = [s.name for s in self.steps]
        duplicates = [n for n in names if names.count(n) > 1]
        if duplicates:
            errors.append(f"Duplicate step names: {set(duplicates)}")

        # Check dependencies exist
        name_set = set(names)
        for step in self.steps:
            for dep in step.depends_on:
                if dep not in name_set:
                    errors.append(f"Step '{step.name}' depends on unknown step '{dep}'")

        # Check for circular dependencies
        def has_cycle(step_name: str, visited: set, stack: set) -> bool:
            visited.add(step_name)
            stack.add(step_name)

            step = self.get_step(step_name)
            if step:
                for dep in step.depends_on:
                    if dep not in visited:
                        if has_cycle(dep, visited, stack):
                            return True
                    elif dep in stack:
                        return True

            stack.remove(step_name)
            return False

        for step in self.steps:
            if has_cycle(step.name, set(), set()):
                errors.append(f"Circular dependency detected involving '{step.name}'")
                break

        return errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "steps": [s.to_dict() for s in self.steps],
        }


@dataclass
class WorkflowResult:
    """Result of a workflow execution."""
    workflow_name: str
    execution_id: str
    success: bool

    # Step results
    steps_completed: int = 0
    steps_failed: int = 0
    steps_skipped: int = 0

    # State
    final_state: Dict[str, Any] = field(default_factory=dict)
    step_results: Dict[str, Any] = field(default_factory=dict)

    # Errors
    errors: List[str] = field(default_factory=list)

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def duration_seconds(self) -> float:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_name": self.workflow_name,
            "execution_id": self.execution_id,
            "success": self.success,
            "steps_completed": self.steps_completed,
            "steps_failed": self.steps_failed,
            "duration_seconds": self.duration_seconds,
            "errors": self.errors,
        }


# =============================================================================
# Workflow Executor
# =============================================================================

class WorkflowExecutor:
    """
    Executes workflows step by step.

    Uses the Queen to dispatch individual steps.
    Handles dependencies, parallelism, and failure.
    """

    def __init__(
        self,
        queen: QueenOrchestrator,
        max_parallel: int = 10,
    ):
        """
        Initialize executor.

        Args:
            queen: QueenOrchestrator for task dispatch
            max_parallel: Maximum parallel steps
        """
        self.queen = queen
        self.max_parallel = max_parallel
        self._running: Dict[str, asyncio.Task] = {}

        log.info("WorkflowExecutor initialized")

    async def run(
        self,
        workflow: Workflow,
        initial_state: Optional[Dict[str, Any]] = None,
    ) -> WorkflowResult:
        """
        Execute a workflow.

        Args:
            workflow: Workflow to execute
            initial_state: Initial state/inputs

        Returns:
            WorkflowResult with outcomes
        """
        execution_id = str(uuid.uuid4())[:8]

        result = WorkflowResult(
            workflow_name=workflow.name,
            execution_id=execution_id,
            success=False,
            started_at=datetime.utcnow(),
            final_state=initial_state.copy() if initial_state else {},
        )

        # Validate
        errors = workflow.validate()
        if errors:
            result.errors = errors
            result.completed_at = datetime.utcnow()
            return result

        log.info(f"Starting workflow '{workflow.name}' (id={execution_id})")

        # Callback
        if workflow.on_start:
            await self._call_hook(workflow.on_start, workflow, result.final_state)

        # Reset step states
        for step in workflow.steps:
            step.status = StepStatus.PENDING
            step.result = None
            step.error = None
            step.attempts = 0

        # Execute
        completed_steps: set[str] = set()
        failed = False

        try:
            while True:
                # Get ready steps
                ready = workflow.get_ready_steps(completed_steps)

                if not ready:
                    # Check if we're done or stuck
                    pending = [s for s in workflow.steps if s.status == StepStatus.PENDING]
                    if not pending:
                        break  # All done
                    if not self._running:
                        # Stuck - no ready steps and nothing running
                        result.errors.append("Workflow stuck: no ready steps")
                        failed = True
                        break

                # Check conditions and filter
                runnable = []
                for step in ready:
                    if step.should_run(result.final_state):
                        runnable.append(step)
                    else:
                        step.status = StepStatus.SKIPPED
                        result.steps_skipped += 1
                        completed_steps.add(step.name)
                        log.info(f"Skipped step '{step.name}' (condition not met)")

                # Limit parallelism
                slots = min(
                    self.max_parallel - len(self._running),
                    workflow.max_parallel,
                    len(runnable),
                )

                # Start steps
                for step in runnable[:slots]:
                    task = asyncio.create_task(
                        self._run_step(step, result.final_state, workflow)
                    )
                    self._running[step.name] = task

                if not self._running:
                    continue

                # Wait for at least one to complete
                done, _ = await asyncio.wait(
                    self._running.values(),
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # Process completed
                for task in done:
                    step_name = next(
                        name for name, t in self._running.items() if t is task
                    )
                    del self._running[step_name]

                    try:
                        step_result = task.result()
                        step = workflow.get_step(step_name)

                        if step_result["success"]:
                            step.status = StepStatus.COMPLETED
                            step.result = step_result["output"]
                            result.steps_completed += 1
                            completed_steps.add(step_name)

                            # Store in state
                            result.final_state[step_name] = step_result["output"]
                            result.step_results[step_name] = step_result

                            # Callback
                            if workflow.on_step_complete:
                                await self._call_hook(
                                    workflow.on_step_complete,
                                    step, step_result,
                                )

                        else:
                            step.status = StepStatus.FAILED
                            step.error = step_result.get("error")
                            result.steps_failed += 1
                            result.errors.append(
                                f"Step '{step_name}' failed: {step.error}"
                            )

                            # Handle failure
                            if step.on_failure == FailureAction.ABORT:
                                failed = True
                            elif step.on_failure == FailureAction.CONTINUE:
                                completed_steps.add(step_name)
                            elif step.on_failure == FailureAction.FALLBACK:
                                if step.fallback_step:
                                    # Queue fallback
                                    pass

                    except Exception as e:
                        log.exception(f"Step {step_name} exception")
                        step = workflow.get_step(step_name)
                        step.status = StepStatus.FAILED
                        step.error = str(e)
                        result.steps_failed += 1
                        result.errors.append(f"Step '{step_name}' exception: {e}")
                        failed = True

                if failed:
                    break

        except asyncio.TimeoutError:
            result.errors.append("Workflow timed out")
        except Exception as e:
            log.exception("Workflow execution failed")
            result.errors.append(str(e))

        # Cancel any still running
        for name, task in self._running.items():
            task.cancel()
        self._running.clear()

        result.success = not failed and result.steps_failed == 0
        result.completed_at = datetime.utcnow()

        # Callback
        if workflow.on_complete:
            await self._call_hook(workflow.on_complete, workflow, result)

        log.info(
            f"Workflow '{workflow.name}' {result.success and 'completed' or 'failed'}: "
            f"completed={result.steps_completed}, failed={result.steps_failed}, "
            f"duration={result.duration_seconds:.1f}s"
        )

        return result

    async def _run_step(
        self,
        step: WorkflowStep,
        state: Dict[str, Any],
        workflow: Workflow,
    ) -> Dict[str, Any]:
        """Execute a single step."""
        step.status = StepStatus.RUNNING
        step.started_at = datetime.utcnow()
        step.attempts += 1

        log.info(f"Running step '{step.name}' (attempt {step.attempts})")

        # Get interpolated params
        params = step.get_params(state)

        # Create task request
        from .queen import TaskRequest

        request = TaskRequest(
            instruction=f"Execute step '{step.name}' of workflow '{workflow.name}'",
            tool=step.tool,
            params=params,
            timeout_seconds=step.timeout_seconds,
            max_retries=step.max_retries,
        )

        # Dispatch to Queen
        result = await self.queen.dispatch(request)

        step.completed_at = datetime.utcnow()

        return {
            "success": result.success,
            "output": result.output,
            "error": result.error,
            "duration_ms": step.duration_ms,
        }

    async def _call_hook(self, hook: Callable, *args) -> None:
        """Call a workflow hook safely."""
        try:
            if asyncio.iscoroutinefunction(hook):
                await hook(*args)
            else:
                hook(*args)
        except Exception as e:
            log.warning(f"Workflow hook failed: {e}")


# =============================================================================
# Convenience Builders
# =============================================================================

def sequential_workflow(
    name: str,
    steps: List[tuple[str, str, Dict[str, Any]]],
) -> Workflow:
    """
    Create a simple sequential workflow.

    Args:
        name: Workflow name
        steps: List of (step_name, tool_name, params)

    Returns:
        Workflow with steps in sequence
    """
    workflow_steps = []
    prev_name = None

    for step_name, tool_name, params in steps:
        step = WorkflowStep(
            name=step_name,
            tool=tool_name,
            params=params,
            depends_on=[prev_name] if prev_name else [],
        )
        workflow_steps.append(step)
        prev_name = step_name

    return Workflow(name=name, steps=workflow_steps)


def parallel_workflow(
    name: str,
    steps: List[tuple[str, str, Dict[str, Any]]],
) -> Workflow:
    """
    Create a workflow where all steps run in parallel.

    Args:
        name: Workflow name
        steps: List of (step_name, tool_name, params)

    Returns:
        Workflow with no dependencies (all parallel)
    """
    workflow_steps = [
        WorkflowStep(name=step_name, tool=tool_name, params=params)
        for step_name, tool_name, params in steps
    ]
    return Workflow(name=name, steps=workflow_steps)


__all__ = [
    # Types
    "StepStatus",
    "FailureAction",
    "WorkflowStep",
    "Workflow",
    "WorkflowResult",
    # Executor
    "WorkflowExecutor",
    # Builders
    "sequential_workflow",
    "parallel_workflow",
]
