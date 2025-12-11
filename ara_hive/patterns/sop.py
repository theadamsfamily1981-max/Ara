# ara_hive/patterns/sop.py
"""
Standard Operating Procedures (SOPs)
====================================

SOPs are structured execution plans that define how roles accomplish tasks.
They are the "brain" of each role - encoding expert knowledge into
repeatable procedures.

MetaGPT Mapping:
    _rc.react()         -> SOPExecutor.execute()
    _rc._actions        -> SOP.steps (sequence of operations)
    _rc._watch          -> SOP.triggers (what activates this SOP)
    _rc.todo            -> SOPExecutor._current_step

Key Concepts:
    - SOP: Complete procedure with phases and steps
    - SOPPhase: Logical grouping of steps (Planning, Execution, Validation)
    - SOPStep: Single atomic operation within a phase
    - SOPExecutor: Runs SOPs step-by-step

Usage:
    from ara_hive.patterns.sop import SOP, SOPStep, SOPPhase, SOPExecutor

    # Define an SOP
    sop = SOP(
        name="feature_development",
        phases=[
            SOPPhase(name="planning", steps=[
                SOPStep(name="gather_requirements", action="analyze"),
                SOPStep(name="write_prd", action="generate"),
            ]),
            SOPPhase(name="implementation", steps=[
                SOPStep(name="write_code", action="generate"),
                SOPStep(name="write_tests", action="generate"),
            ]),
        ],
    )

    # Execute
    executor = SOPExecutor(queen)
    result = await executor.execute(sop, context)
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..src.queen import QueenOrchestrator, TaskRequest
    from .roles import Role, RoleContext, RoleOutput

log = logging.getLogger("Hive.Patterns.SOP")


# =============================================================================
# Types
# =============================================================================

class SOPStepType(str, Enum):
    """Types of SOP steps."""
    ANALYZE = "analyze"        # Analyze input/situation
    GENERATE = "generate"      # Generate content/code
    VALIDATE = "validate"      # Validate output
    TRANSFORM = "transform"    # Transform data
    COMMUNICATE = "communicate"  # Send message/signal
    DECIDE = "decide"          # Make a decision
    WAIT = "wait"              # Wait for input/event
    LOOP = "loop"              # Iterate over items
    BRANCH = "branch"          # Conditional branch
    DELEGATE = "delegate"      # Delegate to another role


class SOPStepStatus(str, Enum):
    """Status of a step execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    WAITING = "waiting"


class SOPPhaseType(str, Enum):
    """Standard phase types."""
    INITIALIZATION = "initialization"
    PLANNING = "planning"
    EXECUTION = "execution"
    VALIDATION = "validation"
    COMPLETION = "completion"


# =============================================================================
# SOP Step
# =============================================================================

@dataclass
class SOPStep:
    """
    A single step in an SOP.

    Steps are atomic operations that:
        - Use a tool or invoke a capability
        - Transform state
        - Make decisions
        - Communicate with other roles
    """
    name: str
    action: SOPStepType
    description: str = ""

    # What tool/capability to use
    tool: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)

    # Input/output mapping
    input_keys: List[str] = field(default_factory=list)  # Keys from context
    output_key: Optional[str] = None  # Key to store result

    # Control flow
    condition: Optional[str] = None  # Python expression for conditional execution
    loop_over: Optional[str] = None  # Key containing items to iterate
    branch_on: Optional[str] = None  # Key for branching decision
    branches: Dict[str, str] = field(default_factory=dict)  # value -> step name

    # Delegation
    delegate_to: Optional[str] = None  # Role name to delegate to
    delegate_instruction: Optional[str] = None

    # Error handling
    on_error: str = "fail"  # fail, skip, retry, fallback
    max_retries: int = 1
    fallback_step: Optional[str] = None

    # Timing
    timeout_seconds: int = 60

    # Runtime state (set during execution)
    status: SOPStepStatus = SOPStepStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    attempts: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def duration_ms(self) -> float:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return 0.0

    def get_params(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get parameters with context interpolation."""
        resolved = {}
        for key, value in self.params.items():
            resolved[key] = self._resolve_value(value, context)
        return resolved

    def _resolve_value(self, value: Any, context: Dict[str, Any]) -> Any:
        """Resolve a value, interpolating from context if needed."""
        if isinstance(value, str):
            if value.startswith("${") and value.endswith("}"):
                path = value[2:-1]
                return self._get_from_context(context, path)
            elif "${" in value:
                # Template string
                import re
                def replace(m):
                    path = m.group(1)
                    val = self._get_from_context(context, path)
                    return str(val) if val is not None else ""
                return re.sub(r'\$\{([^}]+)\}', replace, value)
        return value

    def _get_from_context(self, context: Dict[str, Any], path: str) -> Any:
        """Get value from context by dotted path."""
        parts = path.split(".")
        current = context
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            elif hasattr(current, part):
                current = getattr(current, part)
            else:
                return None
        return current

    def should_execute(self, context: Dict[str, Any]) -> bool:
        """Check if step should execute based on condition."""
        if not self.condition:
            return True
        try:
            return bool(eval(self.condition, {"context": context, "state": context}))
        except Exception as e:
            log.warning(f"Step {self.name} condition failed: {e}")
            return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "action": self.action.value,
            "status": self.status.value,
            "tool": self.tool,
            "attempts": self.attempts,
            "duration_ms": self.duration_ms,
            "error": self.error,
        }


# =============================================================================
# SOP Phase
# =============================================================================

@dataclass
class SOPPhase:
    """
    A phase groups related steps.

    Phases provide:
        - Logical organization (Planning, Execution, Validation)
        - Checkpoints between phases
        - Phase-level error handling
    """
    name: str
    phase_type: SOPPhaseType = SOPPhaseType.EXECUTION
    steps: List[SOPStep] = field(default_factory=list)
    description: str = ""

    # Phase control
    required: bool = True  # If False, phase can be skipped
    parallel_steps: bool = False  # If True, run steps in parallel
    checkpoint: bool = True  # If True, checkpoint state after phase

    # Runtime state
    status: SOPStepStatus = SOPStepStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def duration_ms(self) -> float:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return 0.0

    def get_step(self, name: str) -> Optional[SOPStep]:
        """Get step by name."""
        for step in self.steps:
            if step.name == name:
                return step
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.phase_type.value,
            "status": self.status.value,
            "steps": [s.to_dict() for s in self.steps],
            "duration_ms": self.duration_ms,
        }


# =============================================================================
# SOP
# =============================================================================

@dataclass
class SOP:
    """
    Standard Operating Procedure - a complete execution plan.

    An SOP encodes expert knowledge for a specific task type.
    It defines:
        - What phases to go through
        - What steps to execute in each phase
        - How to handle errors and edge cases
        - What outputs to produce
    """
    name: str
    phases: List[SOPPhase] = field(default_factory=list)
    description: str = ""
    version: str = "1.0.0"

    # What activates this SOP
    triggers: List[str] = field(default_factory=list)  # Keywords or patterns

    # Expected inputs/outputs
    required_inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)

    # For which role
    role: Optional[str] = None

    # Metadata
    author: str = ""
    tags: List[str] = field(default_factory=list)

    def get_phase(self, name: str) -> Optional[SOPPhase]:
        """Get phase by name."""
        for phase in self.phases:
            if phase.name == name:
                return phase
        return None

    def get_step(self, phase_name: str, step_name: str) -> Optional[SOPStep]:
        """Get step by phase and step name."""
        phase = self.get_phase(phase_name)
        if phase:
            return phase.get_step(step_name)
        return None

    def get_all_steps(self) -> List[SOPStep]:
        """Get all steps across all phases."""
        steps = []
        for phase in self.phases:
            steps.extend(phase.steps)
        return steps

    def validate(self) -> List[str]:
        """Validate SOP definition. Returns list of errors."""
        errors = []

        # Check for empty SOP
        if not self.phases:
            errors.append("SOP has no phases")

        # Check for empty phases
        for phase in self.phases:
            if not phase.steps:
                errors.append(f"Phase '{phase.name}' has no steps")

        # Check step names are unique within phases
        for phase in self.phases:
            names = [s.name for s in phase.steps]
            dupes = [n for n in names if names.count(n) > 1]
            if dupes:
                errors.append(f"Phase '{phase.name}' has duplicate steps: {set(dupes)}")

        # Check branch targets exist
        all_step_names = {s.name for s in self.get_all_steps()}
        for step in self.get_all_steps():
            for target in step.branches.values():
                if target not in all_step_names:
                    errors.append(
                        f"Step '{step.name}' branches to unknown step '{target}'"
                    )

        return errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "role": self.role,
            "phases": [p.to_dict() for p in self.phases],
            "triggers": self.triggers,
            "tags": self.tags,
        }


# =============================================================================
# SOP Result
# =============================================================================

@dataclass
class SOPResult:
    """Result of executing an SOP."""
    sop_name: str
    execution_id: str
    success: bool

    # Outputs
    outputs: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)

    # Phase/step details
    phases_completed: int = 0
    phases_failed: int = 0
    steps_completed: int = 0
    steps_failed: int = 0

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
            "sop_name": self.sop_name,
            "execution_id": self.execution_id,
            "success": self.success,
            "phases_completed": self.phases_completed,
            "steps_completed": self.steps_completed,
            "duration_seconds": self.duration_seconds,
            "errors": self.errors,
        }


# =============================================================================
# SOP Executor
# =============================================================================

class SOPExecutor:
    """
    Executes SOPs step by step.

    The executor:
        - Maintains execution context
        - Runs steps sequentially or in parallel
        - Handles errors and retries
        - Tracks state between steps
    """

    def __init__(
        self,
        queen: Optional[QueenOrchestrator] = None,
        role_factory: Optional[Callable[[str], Role]] = None,
    ):
        """
        Initialize executor.

        Args:
            queen: QueenOrchestrator for tool dispatch
            role_factory: Factory to create roles for delegation
        """
        self.queen = queen
        self.role_factory = role_factory
        self._current_step: Optional[SOPStep] = None
        self._execution_context: Dict[str, Any] = {}

    async def execute(
        self,
        sop: SOP,
        context: Dict[str, Any],
        role: Optional[Role] = None,
    ) -> SOPResult:
        """
        Execute an SOP.

        Args:
            sop: SOP to execute
            context: Initial context/inputs
            role: Role executing this SOP (optional)

        Returns:
            SOPResult with outcomes
        """
        execution_id = str(uuid.uuid4())[:8]

        result = SOPResult(
            sop_name=sop.name,
            execution_id=execution_id,
            success=False,
            started_at=datetime.utcnow(),
        )

        # Validate SOP
        errors = sop.validate()
        if errors:
            result.errors = errors
            result.completed_at = datetime.utcnow()
            return result

        # Check required inputs
        for key in sop.required_inputs:
            if key not in context:
                result.errors.append(f"Missing required input: {key}")
                result.completed_at = datetime.utcnow()
                return result

        # Initialize execution context
        self._execution_context = {
            "inputs": context.copy(),
            "outputs": {},
            "state": {},
            "role": role.name if role else None,
        }

        log.info(f"Starting SOP '{sop.name}' (id={execution_id})")

        # Execute phases
        failed = False
        for phase in sop.phases:
            if failed and phase.required:
                result.phases_failed += 1
                continue

            phase_result = await self._execute_phase(phase, sop)

            if phase_result["success"]:
                result.phases_completed += 1
                result.steps_completed += phase_result["steps_completed"]
            else:
                result.phases_failed += 1
                result.steps_failed += phase_result["steps_failed"]
                result.steps_completed += phase_result["steps_completed"]
                result.errors.extend(phase_result["errors"])
                if phase.required:
                    failed = True

        # Collect outputs
        result.outputs = self._execution_context.get("outputs", {})
        result.artifacts = self._execution_context.get("artifacts", {})
        result.success = not failed
        result.completed_at = datetime.utcnow()

        log.info(
            f"SOP '{sop.name}' {'completed' if result.success else 'failed'}: "
            f"phases={result.phases_completed}/{len(sop.phases)}, "
            f"steps={result.steps_completed}, "
            f"duration={result.duration_seconds:.1f}s"
        )

        return result

    async def _execute_phase(
        self,
        phase: SOPPhase,
        sop: SOP,
    ) -> Dict[str, Any]:
        """Execute a single phase."""
        phase.status = SOPStepStatus.RUNNING
        phase.started_at = datetime.utcnow()

        result = {
            "success": True,
            "steps_completed": 0,
            "steps_failed": 0,
            "errors": [],
        }

        log.info(f"Starting phase '{phase.name}' ({phase.phase_type.value})")

        if phase.parallel_steps:
            # Run all steps in parallel
            step_results = await asyncio.gather(
                *[self._execute_step(step) for step in phase.steps],
                return_exceptions=True,
            )

            for step, step_result in zip(phase.steps, step_results):
                if isinstance(step_result, Exception):
                    result["steps_failed"] += 1
                    result["errors"].append(f"Step '{step.name}': {step_result}")
                    result["success"] = False
                elif step_result.get("success"):
                    result["steps_completed"] += 1
                else:
                    result["steps_failed"] += 1
                    if step_result.get("error"):
                        result["errors"].append(
                            f"Step '{step.name}': {step_result['error']}"
                        )
                    result["success"] = False
        else:
            # Run steps sequentially
            for step in phase.steps:
                if not step.should_execute(self._execution_context):
                    step.status = SOPStepStatus.SKIPPED
                    continue

                step_result = await self._execute_step(step)

                if step_result.get("success"):
                    result["steps_completed"] += 1
                else:
                    result["steps_failed"] += 1
                    if step_result.get("error"):
                        result["errors"].append(
                            f"Step '{step.name}': {step_result['error']}"
                        )

                    if step.on_error == "fail":
                        result["success"] = False
                        break
                    elif step.on_error == "skip":
                        continue

        phase.status = (
            SOPStepStatus.COMPLETED if result["success"] else SOPStepStatus.FAILED
        )
        phase.completed_at = datetime.utcnow()

        return result

    async def _execute_step(self, step: SOPStep) -> Dict[str, Any]:
        """Execute a single step."""
        step.status = SOPStepStatus.RUNNING
        step.started_at = datetime.utcnow()
        step.attempts += 1

        self._current_step = step

        log.debug(f"Executing step '{step.name}' ({step.action.value})")

        try:
            # Get resolved parameters
            params = step.get_params(self._execution_context)

            # Handle different action types
            if step.action == SOPStepType.DELEGATE:
                result = await self._execute_delegate(step, params)
            elif step.action == SOPStepType.LOOP:
                result = await self._execute_loop(step, params)
            elif step.action == SOPStepType.BRANCH:
                result = await self._execute_branch(step, params)
            elif step.action == SOPStepType.WAIT:
                result = await self._execute_wait(step, params)
            elif step.tool:
                result = await self._execute_tool(step, params)
            else:
                # Built-in action handlers
                result = await self._execute_builtin(step, params)

            # Store output
            if step.output_key and result.get("output"):
                self._execution_context["outputs"][step.output_key] = result["output"]
                self._execution_context["state"][step.output_key] = result["output"]

            step.result = result.get("output")
            step.status = SOPStepStatus.COMPLETED
            step.completed_at = datetime.utcnow()

            return {"success": True, "output": result.get("output")}

        except asyncio.TimeoutError:
            error = f"Step timed out after {step.timeout_seconds}s"
            step.error = error
            step.status = SOPStepStatus.FAILED
            step.completed_at = datetime.utcnow()
            return {"success": False, "error": error}

        except Exception as e:
            log.exception(f"Step {step.name} failed")
            step.error = str(e)
            step.status = SOPStepStatus.FAILED
            step.completed_at = datetime.utcnow()
            return {"success": False, "error": str(e)}

    async def _execute_tool(
        self,
        step: SOPStep,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a tool step."""
        if not self.queen:
            raise RuntimeError("No Queen bound for tool execution")

        from ..src.queen import TaskRequest

        request = TaskRequest(
            instruction=step.description or f"Execute {step.name}",
            tool=step.tool,
            params=params,
            timeout_seconds=step.timeout_seconds,
        )

        result = await self.queen.dispatch(request)

        return {
            "success": result.success,
            "output": result.output,
            "error": result.error,
        }

    async def _execute_delegate(
        self,
        step: SOPStep,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Delegate to another role."""
        if not step.delegate_to:
            return {"success": False, "error": "No delegation target"}

        if not self.role_factory:
            return {"success": False, "error": "No role factory for delegation"}

        # Create role
        role = self.role_factory(step.delegate_to)
        if not role:
            return {"success": False, "error": f"Unknown role: {step.delegate_to}"}

        # Bind queen
        if self.queen:
            role.bind_queen(self.queen)

        # Create context
        from .roles import RoleContext

        instruction = step.delegate_instruction or step.description
        context = RoleContext(
            task_id=str(uuid.uuid4())[:8],
            instruction=instruction,
            inputs=params,
            memory=self._execution_context.get("state", {}),
        )

        # Execute role
        output = await role.execute(context)

        return {
            "success": output.success,
            "output": output.content,
            "error": output.error,
        }

    async def _execute_loop(
        self,
        step: SOPStep,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a loop step."""
        if not step.loop_over:
            return {"success": False, "error": "No loop_over specified"}

        items = self._execution_context.get("state", {}).get(step.loop_over, [])
        if not isinstance(items, (list, tuple)):
            items = [items]

        results = []
        for item in items:
            # Create sub-context with loop item
            sub_context = {**self._execution_context, "item": item}
            sub_params = step.get_params(sub_context)

            if step.tool:
                result = await self._execute_tool(step, sub_params)
                results.append(result.get("output"))

        return {"success": True, "output": results}

    async def _execute_branch(
        self,
        step: SOPStep,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a branch step."""
        if not step.branch_on:
            return {"success": False, "error": "No branch_on specified"}

        value = self._execution_context.get("state", {}).get(step.branch_on)
        value_str = str(value) if value is not None else "default"

        next_step = step.branches.get(value_str, step.branches.get("default"))

        return {
            "success": True,
            "output": {"branch_taken": value_str, "next_step": next_step},
        }

    async def _execute_wait(
        self,
        step: SOPStep,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a wait step."""
        wait_time = params.get("seconds", 1.0)
        await asyncio.sleep(wait_time)
        return {"success": True, "output": {"waited_seconds": wait_time}}

    async def _execute_builtin(
        self,
        step: SOPStep,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute built-in action types without tools."""
        if step.action == SOPStepType.ANALYZE:
            # Placeholder analysis - in production use LLM
            return {
                "success": True,
                "output": {"analysis": f"Analysis of: {params}"},
            }

        elif step.action == SOPStepType.GENERATE:
            # Placeholder generation
            return {
                "success": True,
                "output": {"generated": f"Generated from: {params}"},
            }

        elif step.action == SOPStepType.VALIDATE:
            # Placeholder validation
            return {
                "success": True,
                "output": {"valid": True, "errors": []},
            }

        elif step.action == SOPStepType.TRANSFORM:
            # Pass through params as transform result
            return {"success": True, "output": params}

        elif step.action == SOPStepType.COMMUNICATE:
            # Log the communication
            message = params.get("message", "")
            log.info(f"SOP Communication: {message}")
            return {"success": True, "output": {"sent": message}}

        elif step.action == SOPStepType.DECIDE:
            # Placeholder decision
            return {
                "success": True,
                "output": {"decision": "proceed", "confidence": 0.8},
            }

        return {"success": False, "error": f"Unhandled action: {step.action}"}


# =============================================================================
# SOP Builders
# =============================================================================

def create_role_sop(
    role_name: str,
    task_type: str = "general",
) -> SOP:
    """
    Create a standard SOP for a role.

    Args:
        role_name: Name of the role
        task_type: Type of task (general, review, implement, etc.)

    Returns:
        SOP configured for the role
    """
    # Standard phases for most roles
    phases = [
        SOPPhase(
            name="understand",
            phase_type=SOPPhaseType.PLANNING,
            steps=[
                SOPStep(
                    name="analyze_input",
                    action=SOPStepType.ANALYZE,
                    description="Analyze the task requirements",
                    output_key="analysis",
                ),
            ],
        ),
        SOPPhase(
            name="execute",
            phase_type=SOPPhaseType.EXECUTION,
            steps=[
                SOPStep(
                    name="generate_output",
                    action=SOPStepType.GENERATE,
                    description="Generate the main output",
                    output_key="result",
                ),
            ],
        ),
        SOPPhase(
            name="validate",
            phase_type=SOPPhaseType.VALIDATION,
            steps=[
                SOPStep(
                    name="validate_output",
                    action=SOPStepType.VALIDATE,
                    description="Validate the output",
                    output_key="validation",
                ),
            ],
        ),
    ]

    return SOP(
        name=f"{role_name}_{task_type}",
        description=f"Standard {task_type} SOP for {role_name}",
        role=role_name,
        phases=phases,
        triggers=[task_type],
    )


def create_sequential_sop(
    name: str,
    steps: List[tuple[str, SOPStepType, str]],
) -> SOP:
    """
    Create a simple sequential SOP.

    Args:
        name: SOP name
        steps: List of (step_name, action_type, description)

    Returns:
        SOP with single phase containing all steps
    """
    sop_steps = [
        SOPStep(name=step_name, action=action, description=desc)
        for step_name, action, desc in steps
    ]

    return SOP(
        name=name,
        phases=[
            SOPPhase(
                name="main",
                phase_type=SOPPhaseType.EXECUTION,
                steps=sop_steps,
            ),
        ],
    )


__all__ = [
    # Types
    "SOPStepType",
    "SOPStepStatus",
    "SOPPhaseType",
    # Core
    "SOPStep",
    "SOPPhase",
    "SOP",
    "SOPResult",
    # Executor
    "SOPExecutor",
    # Builders
    "create_role_sop",
    "create_sequential_sop",
]
