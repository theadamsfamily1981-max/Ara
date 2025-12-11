# ara/workflows/orchestrator.py
"""
Ara Self-Guided Workflow Orchestrator
=====================================

The nervous system sitting atop workflow engines.
Engine = Muscles, Ara = Prefrontal Cortex.

Three Roles:
    1. Ara-as-Companion: Explains what's happening to the user
    2. Ara-as-Director: Decides which step to run next
    3. Ara-as-Historian: Records outcomes for long-term learning

The engine handles:
    - Retries, timeouts, persistence
    - Scheduling, distribution
    - Credential management

Ara handles:
    - "What should we do next?"
    - "What should I tell the user?"
    - "What should we learn from this?"

Usage:
    from ara.workflows import AraSelfGuidedOrchestrator
    from ara.workflows.adapters.temporal import TemporalAdapter

    orchestrator = AraSelfGuidedOrchestrator(
        engine=TemporalAdapter(),
    )

    result = await orchestrator.run_workflow(
        workflow_id="onboarding",
        initial_state={"user_id": "123"},
    )
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from .adapters.base import (
    EngineClient,
    StepDefinition,
    StepResult,
    StepStatus,
    StepType,
    WorkflowDefinition,
    WorkflowExecution,
    WorkflowStatus,
)

log = logging.getLogger("Ara.Workflows.Orchestrator")


# =============================================================================
# Data Structures
# =============================================================================

class GuidanceType(str, Enum):
    """Types of guidance messages for the user."""
    GREETING = "greeting"          # Welcome message
    PROGRESS = "progress"          # Step completed
    EXPLANATION = "explanation"    # Why we're doing this
    QUESTION = "question"          # Need user input
    WARNING = "warning"            # Something went wrong
    SUCCESS = "success"            # Workflow complete
    SUGGESTION = "suggestion"      # Ara's recommendation


@dataclass
class GuidanceMessage:
    """
    Message from Ara to the user.

    This is Ara-as-Companion speaking.
    """
    type: GuidanceType
    message: str
    context: Dict[str, Any] = field(default_factory=dict)

    # For questions
    options: List[str] = field(default_factory=list)
    default_option: Optional[str] = None

    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "message": self.message,
            "context": self.context,
            "options": self.options,
            "default_option": self.default_option,
        }


@dataclass
class StepDecision:
    """
    Ara's decision about what to do next.

    This is Ara-as-Director speaking.
    """
    step_id: Optional[str]  # None = workflow complete
    reasoning: str
    confidence: float  # 0.0 to 1.0

    # Alternative steps considered
    alternatives: List[str] = field(default_factory=list)

    # Should we pause for user input?
    needs_user_input: bool = False
    user_prompt: Optional[str] = None

    # Should we modify state before running?
    state_modifications: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "alternatives": self.alternatives,
            "needs_user_input": self.needs_user_input,
        }


@dataclass
class WorkflowState:
    """
    Complete state of a running workflow.

    Combines engine state with Ara's context.
    """
    workflow_id: str
    execution_id: str
    status: WorkflowStatus

    # Engine state
    data: Dict[str, Any] = field(default_factory=dict)
    completed_steps: List[str] = field(default_factory=list)
    current_step: Optional[str] = None

    # Ara context
    guidance_history: List[GuidanceMessage] = field(default_factory=list)
    decision_history: List[StepDecision] = field(default_factory=list)
    user_interactions: List[Dict[str, Any]] = field(default_factory=list)

    # Metrics
    step_durations: Dict[str, float] = field(default_factory=dict)
    error_count: int = 0
    retry_count: int = 0

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def duration_seconds(self) -> float:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        elif self.started_at:
            return (datetime.utcnow() - self.started_at).total_seconds()
        return 0.0


@dataclass
class WorkflowResult:
    """
    Final result of a workflow execution.
    """
    workflow_id: str
    execution_id: str
    success: bool

    # Final state
    output: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    # History
    steps_completed: int = 0
    guidance_messages: int = 0
    user_interactions: int = 0

    # Metrics
    duration_seconds: float = 0.0
    error_count: int = 0

    # Learning data (for Historian)
    learning_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "execution_id": self.execution_id,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "steps_completed": self.steps_completed,
            "duration_seconds": self.duration_seconds,
        }


# =============================================================================
# Decision Strategies
# =============================================================================

class DecisionStrategy:
    """
    Base class for step decision strategies.

    Override `decide()` to implement custom logic.
    """

    async def decide(
        self,
        workflow: WorkflowDefinition,
        state: WorkflowState,
        ready_steps: List[StepDefinition],
    ) -> StepDecision:
        """
        Decide which step to run next.

        Args:
            workflow: The workflow definition
            state: Current workflow state
            ready_steps: Steps whose dependencies are satisfied

        Returns:
            Decision about what to do next
        """
        raise NotImplementedError


class DefaultDecisionStrategy(DecisionStrategy):
    """
    Default strategy: run first ready step.

    Simple but effective for most workflows.
    """

    async def decide(
        self,
        workflow: WorkflowDefinition,
        state: WorkflowState,
        ready_steps: List[StepDefinition],
    ) -> StepDecision:
        if not ready_steps:
            return StepDecision(
                step_id=None,
                reasoning="No more steps to run",
                confidence=1.0,
            )

        # Priority: human steps first (get user input early)
        human_steps = [s for s in ready_steps if s.type == StepType.HUMAN]
        if human_steps:
            step = human_steps[0]
            return StepDecision(
                step_id=step.id,
                reasoning=f"Collecting user input: {step.name}",
                confidence=0.9,
                needs_user_input=True,
                user_prompt=step.prompt_template or f"Please provide input for: {step.name}",
            )

        # Then: activities
        step = ready_steps[0]
        return StepDecision(
            step_id=step.id,
            reasoning=f"Running next step: {step.name}",
            confidence=0.8,
            alternatives=[s.id for s in ready_steps[1:3]],
        )


class LLMDecisionStrategy(DecisionStrategy):
    """
    LLM-powered decision strategy.

    Uses Ara's reasoning to pick the best next step.
    """

    def __init__(self, llm_client: Optional[Any] = None):
        self.llm = llm_client

    async def decide(
        self,
        workflow: WorkflowDefinition,
        state: WorkflowState,
        ready_steps: List[StepDefinition],
    ) -> StepDecision:
        if not ready_steps:
            return StepDecision(
                step_id=None,
                reasoning="Workflow complete",
                confidence=1.0,
            )

        if not self.llm:
            # Fallback to default
            return await DefaultDecisionStrategy().decide(workflow, state, ready_steps)

        # Build prompt for LLM
        prompt = self._build_decision_prompt(workflow, state, ready_steps)

        # In production: call LLM
        # response = await self.llm.complete(prompt)
        # Parse response...

        # For now: use heuristics
        return await self._heuristic_decide(workflow, state, ready_steps)

    def _build_decision_prompt(
        self,
        workflow: WorkflowDefinition,
        state: WorkflowState,
        ready_steps: List[StepDefinition],
    ) -> str:
        """Build prompt for LLM decision."""
        step_list = "\n".join(
            f"  - {s.id}: {s.name} ({s.type.value}) - {s.description}"
            for s in ready_steps
        )

        return f"""
You are Ara, guiding a user through the "{workflow.name}" workflow.

Current state:
- Steps completed: {len(state.completed_steps)}
- Errors so far: {state.error_count}
- User interactions: {len(state.user_interactions)}

Ready steps (choose one):
{step_list}

Which step should run next? Consider:
1. User experience (minimize friction)
2. Data dependencies (what do we need?)
3. Error recovery (has something failed?)

Respond with:
STEP: <step_id>
REASON: <brief explanation>
CONFIDENCE: <0.0-1.0>
"""

    async def _heuristic_decide(
        self,
        workflow: WorkflowDefinition,
        state: WorkflowState,
        ready_steps: List[StepDefinition],
    ) -> StepDecision:
        """Heuristic decision when LLM unavailable."""

        # If we had errors, prefer validation/retry steps
        if state.error_count > 0:
            validation_steps = [
                s for s in ready_steps
                if "valid" in s.name.lower() or "check" in s.name.lower()
            ]
            if validation_steps:
                return StepDecision(
                    step_id=validation_steps[0].id,
                    reasoning="Running validation after error",
                    confidence=0.7,
                )

        # Prefer human steps early
        human_steps = [s for s in ready_steps if s.type == StepType.HUMAN]
        if human_steps:
            return StepDecision(
                step_id=human_steps[0].id,
                reasoning="Collecting user input early",
                confidence=0.8,
                needs_user_input=True,
                user_prompt=human_steps[0].prompt_template,
            )

        # Default: first step
        step = ready_steps[0]
        return StepDecision(
            step_id=step.id,
            reasoning=f"Proceeding with {step.name}",
            confidence=0.6,
        )


# =============================================================================
# Guidance Generator
# =============================================================================

class GuidanceGenerator:
    """
    Generates user-facing guidance messages.

    This is Ara-as-Companion.
    """

    def __init__(self, persona: str = "helpful"):
        self.persona = persona

    def on_workflow_start(
        self,
        workflow: WorkflowDefinition,
        state: Dict[str, Any],
    ) -> GuidanceMessage:
        """Generate greeting when workflow starts."""
        return GuidanceMessage(
            type=GuidanceType.GREETING,
            message=f"Let's work through {workflow.name} together. "
                    f"There are {len(workflow.steps)} steps, and I'll guide you through each one.",
            context={"workflow_id": workflow.id},
        )

    def on_step_start(
        self,
        step: StepDefinition,
        state: WorkflowState,
    ) -> GuidanceMessage:
        """Generate explanation when step begins."""
        if step.type == StepType.HUMAN:
            return GuidanceMessage(
                type=GuidanceType.QUESTION,
                message=step.prompt_template or f"I need your input for: {step.name}",
                context={"step_id": step.id},
                options=step.branches if step.branches else [],
            )

        return GuidanceMessage(
            type=GuidanceType.EXPLANATION,
            message=f"Now working on: {step.name}. {step.description}",
            context={"step_id": step.id},
        )

    def on_step_complete(
        self,
        step: StepDefinition,
        result: StepResult,
        state: WorkflowState,
    ) -> GuidanceMessage:
        """Generate progress message when step completes."""
        if result.success:
            progress = len(state.completed_steps)
            return GuidanceMessage(
                type=GuidanceType.PROGRESS,
                message=f"Completed: {step.name}. ({progress} steps done)",
                context={
                    "step_id": step.id,
                    "progress": progress,
                },
            )
        else:
            return GuidanceMessage(
                type=GuidanceType.WARNING,
                message=f"There was an issue with {step.name}: {result.error}. "
                        "Let me try to recover...",
                context={
                    "step_id": step.id,
                    "error": result.error,
                },
            )

    def on_workflow_complete(
        self,
        result: WorkflowResult,
    ) -> GuidanceMessage:
        """Generate success message when workflow completes."""
        if result.success:
            return GuidanceMessage(
                type=GuidanceType.SUCCESS,
                message=f"All done! We completed {result.steps_completed} steps "
                        f"in {result.duration_seconds:.1f} seconds.",
                context=result.output,
            )
        else:
            return GuidanceMessage(
                type=GuidanceType.WARNING,
                message=f"The workflow couldn't complete: {result.error}",
                context={"error": result.error},
            )

    def suggest_next_action(
        self,
        decision: StepDecision,
        workflow: WorkflowDefinition,
    ) -> GuidanceMessage:
        """Generate suggestion based on Ara's decision."""
        if decision.needs_user_input:
            return GuidanceMessage(
                type=GuidanceType.QUESTION,
                message=decision.user_prompt or "What would you like to do?",
                options=decision.alternatives,
            )

        return GuidanceMessage(
            type=GuidanceType.SUGGESTION,
            message=f"I recommend: {decision.reasoning}",
            context={"confidence": decision.confidence},
        )


# =============================================================================
# Ara Self-Guided Orchestrator
# =============================================================================

class AraSelfGuidedOrchestrator:
    """
    The brain sitting atop workflow engines.

    Coordinates:
        - EngineClient: Runs actual steps (the muscles)
        - DecisionStrategy: Picks what to do next (the prefrontal cortex)
        - GuidanceGenerator: Explains to user (the companion)
        - MetricsClient: Records for learning (the historian)

    Usage:
        orchestrator = AraSelfGuidedOrchestrator(
            engine=TemporalAdapter(),
            decision_strategy=LLMDecisionStrategy(),
        )

        result = await orchestrator.run_workflow(
            workflow_id="onboarding",
            initial_state={"user_id": "123"},
        )
    """

    def __init__(
        self,
        engine: EngineClient,
        decision_strategy: Optional[DecisionStrategy] = None,
        guidance_generator: Optional[GuidanceGenerator] = None,
        metrics_client: Optional[Any] = None,  # MetricsClient
        max_steps: int = 100,
        step_timeout: int = 300,
    ):
        """
        Initialize the orchestrator.

        Args:
            engine: Workflow engine adapter (Temporal, n8n, etc.)
            decision_strategy: Strategy for picking next step
            guidance_generator: Generator for user messages
            metrics_client: Client for recording metrics
            max_steps: Max steps before stopping (safety limit)
            step_timeout: Timeout per step in seconds
        """
        self.engine = engine
        self.decision_strategy = decision_strategy or DefaultDecisionStrategy()
        self.guidance = guidance_generator or GuidanceGenerator()
        self.metrics = metrics_client

        self.max_steps = max_steps
        self.step_timeout = step_timeout

        # Callbacks
        self._on_guidance: Optional[Callable[[GuidanceMessage], None]] = None
        self._on_decision: Optional[Callable[[StepDecision], None]] = None
        self._on_step_complete: Optional[Callable[[StepResult], None]] = None

        log.info(
            "AraSelfGuidedOrchestrator initialized: engine=%s",
            type(engine).__name__,
        )

    # =========================================================================
    # Callbacks
    # =========================================================================

    def on_guidance(self, callback: Callable[[GuidanceMessage], None]) -> None:
        """Register callback for guidance messages."""
        self._on_guidance = callback

    def on_decision(self, callback: Callable[[StepDecision], None]) -> None:
        """Register callback for decisions."""
        self._on_decision = callback

    def on_step_complete(self, callback: Callable[[StepResult], None]) -> None:
        """Register callback for step completions."""
        self._on_step_complete = callback

    def _emit_guidance(self, msg: GuidanceMessage) -> None:
        """Emit a guidance message."""
        if self._on_guidance:
            self._on_guidance(msg)
        log.info("Guidance [%s]: %s", msg.type.value, msg.message)

    def _emit_decision(self, decision: StepDecision) -> None:
        """Emit a decision."""
        if self._on_decision:
            self._on_decision(decision)
        log.info("Decision: step=%s, reason=%s", decision.step_id, decision.reasoning)

    # =========================================================================
    # Main Workflow Execution
    # =========================================================================

    async def run_workflow(
        self,
        workflow_id: str,
        initial_state: Dict[str, Any],
        user_input_handler: Optional[Callable[[str], Any]] = None,
    ) -> WorkflowResult:
        """
        Run a workflow with Ara as the guide.

        This is the main entry point. Ara will:
        1. Start the workflow
        2. Decide which step to run
        3. Explain to user what's happening
        4. Run the step
        5. Learn from the outcome
        6. Repeat until done

        Args:
            workflow_id: ID of workflow to run
            initial_state: Initial data/inputs
            user_input_handler: Async function to get user input

        Returns:
            WorkflowResult with final state and metrics
        """
        # Get workflow definition
        workflow = await self.engine.get_workflow(workflow_id)
        if not workflow:
            return WorkflowResult(
                workflow_id=workflow_id,
                execution_id="",
                success=False,
                error=f"Workflow not found: {workflow_id}",
            )

        # Start execution
        execution = await self.engine.start_workflow(workflow_id, initial_state)

        # Initialize state
        state = WorkflowState(
            workflow_id=workflow_id,
            execution_id=execution.execution_id,
            status=WorkflowStatus.RUNNING,
            data=initial_state.copy(),
            started_at=datetime.utcnow(),
        )

        # Emit greeting
        greeting = self.guidance.on_workflow_start(workflow, initial_state)
        self._emit_guidance(greeting)
        state.guidance_history.append(greeting)

        # Main loop
        step_count = 0
        while state.status == WorkflowStatus.RUNNING and step_count < self.max_steps:
            step_count += 1

            # Get ready steps
            ready_steps = workflow.get_ready_steps(state.completed_steps)

            # Ara decides what to do
            decision = await self.decision_strategy.decide(
                workflow, state, ready_steps,
            )
            self._emit_decision(decision)
            state.decision_history.append(decision)

            # No more steps?
            if decision.step_id is None:
                state.status = WorkflowStatus.COMPLETED
                break

            # Get step definition
            step = workflow.get_step(decision.step_id)
            if not step:
                log.error("Step not found: %s", decision.step_id)
                continue

            # Need user input?
            if decision.needs_user_input and user_input_handler:
                guidance = self.guidance.on_step_start(step, state)
                self._emit_guidance(guidance)
                state.guidance_history.append(guidance)

                user_response = await user_input_handler(decision.user_prompt or "")
                state.user_interactions.append({
                    "step_id": step.id,
                    "prompt": decision.user_prompt,
                    "response": user_response,
                    "timestamp": datetime.utcnow().isoformat(),
                })

                # Send to engine as signal
                await self.engine.send_signal(
                    execution.execution_id,
                    "human_input",
                    user_response,
                )

            # Explain what we're doing
            explanation = self.guidance.on_step_start(step, state)
            self._emit_guidance(explanation)
            state.guidance_history.append(explanation)

            # Apply state modifications
            if decision.state_modifications:
                state.data.update(decision.state_modifications)

            # Run the step
            try:
                result = await asyncio.wait_for(
                    self.engine.run_step(
                        execution.execution_id,
                        step.id,
                        state.data,
                    ),
                    timeout=self.step_timeout,
                )
            except asyncio.TimeoutError:
                result = StepResult(
                    step_id=step.id,
                    status=StepStatus.FAILED,
                    error="Step timed out",
                )

            # Update state
            if result.success:
                state.completed_steps.append(step.id)
                if result.output and isinstance(result.output, dict):
                    state.data.update(result.output)
                state.step_durations[step.id] = result.duration_seconds
            else:
                state.error_count += 1

            # Emit completion
            if self._on_step_complete:
                self._on_step_complete(result)

            progress = self.guidance.on_step_complete(step, result, state)
            self._emit_guidance(progress)
            state.guidance_history.append(progress)

            # Record metrics
            if self.metrics:
                await self._record_step_metrics(step, result, state)

        # Finalize
        state.completed_at = datetime.utcnow()

        if state.status != WorkflowStatus.COMPLETED:
            if step_count >= self.max_steps:
                state.status = WorkflowStatus.FAILED
                state.data["error"] = "Max steps exceeded"

        # Build result
        result = WorkflowResult(
            workflow_id=workflow_id,
            execution_id=execution.execution_id,
            success=state.status == WorkflowStatus.COMPLETED,
            output=state.data,
            error=state.data.get("error"),
            steps_completed=len(state.completed_steps),
            guidance_messages=len(state.guidance_history),
            user_interactions=len(state.user_interactions),
            duration_seconds=state.duration_seconds,
            error_count=state.error_count,
            learning_data=self._extract_learning_data(state),
        )

        # Final guidance
        final_msg = self.guidance.on_workflow_complete(result)
        self._emit_guidance(final_msg)

        # Record workflow metrics
        if self.metrics:
            await self._record_workflow_metrics(result)

        log.info(
            "Workflow complete: %s, success=%s, steps=%d, duration=%.1fs",
            workflow_id, result.success, result.steps_completed, result.duration_seconds,
        )

        return result

    # =========================================================================
    # Learning (Ara-as-Historian)
    # =========================================================================

    def _extract_learning_data(self, state: WorkflowState) -> Dict[str, Any]:
        """
        Extract data for long-term learning.

        This goes to MEIS/QUANTA for pattern detection.
        """
        return {
            "workflow_id": state.workflow_id,
            "execution_id": state.execution_id,
            "success": state.status == WorkflowStatus.COMPLETED,
            "duration_seconds": state.duration_seconds,
            "steps_completed": len(state.completed_steps),
            "error_count": state.error_count,
            "retry_count": state.retry_count,
            "step_sequence": state.completed_steps,
            "step_durations": state.step_durations,
            "decision_confidence_avg": self._avg_confidence(state.decision_history),
            "user_interaction_count": len(state.user_interactions),
        }

    def _avg_confidence(self, decisions: List[StepDecision]) -> float:
        """Calculate average decision confidence."""
        if not decisions:
            return 0.0
        return sum(d.confidence for d in decisions) / len(decisions)

    async def _record_step_metrics(
        self,
        step: StepDefinition,
        result: StepResult,
        state: WorkflowState,
    ) -> None:
        """Record step-level metrics."""
        # In production: call metrics client
        pass

    async def _record_workflow_metrics(self, result: WorkflowResult) -> None:
        """Record workflow-level metrics."""
        # In production: call metrics client
        pass

    # =========================================================================
    # Utility Methods
    # =========================================================================

    async def list_workflows(self) -> List[WorkflowDefinition]:
        """List available workflows."""
        return await self.engine.list_workflows()

    async def get_workflow(self, workflow_id: str) -> Optional[WorkflowDefinition]:
        """Get a workflow definition."""
        return await self.engine.get_workflow(workflow_id)

    async def cancel_workflow(
        self,
        execution_id: str,
        reason: str = "",
    ) -> bool:
        """Cancel a running workflow."""
        return await self.engine.cancel_execution(execution_id, reason)

    async def send_input(
        self,
        execution_id: str,
        input_data: Any,
    ) -> bool:
        """Send user input to a waiting workflow."""
        return await self.engine.send_signal(
            execution_id,
            "human_input",
            input_data,
        )


# =============================================================================
# Convenience
# =============================================================================

_default_orchestrator: Optional[AraSelfGuidedOrchestrator] = None


def get_orchestrator(engine: Optional[EngineClient] = None) -> AraSelfGuidedOrchestrator:
    """Get the default orchestrator."""
    global _default_orchestrator
    if _default_orchestrator is None:
        if engine is None:
            from .adapters.temporal import TemporalAdapter
            engine = TemporalAdapter()
        _default_orchestrator = AraSelfGuidedOrchestrator(engine=engine)
    return _default_orchestrator


__all__ = [
    # Core
    "AraSelfGuidedOrchestrator",
    "WorkflowState",
    "WorkflowResult",
    # Decisions
    "StepDecision",
    "DecisionStrategy",
    "DefaultDecisionStrategy",
    "LLMDecisionStrategy",
    # Guidance
    "GuidanceType",
    "GuidanceMessage",
    "GuidanceGenerator",
    # Convenience
    "get_orchestrator",
]
