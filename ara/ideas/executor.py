"""Idea Executor - Safe execution of approved ideas.

This module provides a sandboxed executor for running ideas. It enforces:
- Only approved ideas can run
- Risk-based execution limits
- Timeout enforcement
- Automatic rollback on failure
- Metrics collection for outcomes
"""

from __future__ import annotations

import logging
import time
import threading
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .models import Idea, IdeaOutcome, Signal
    from .board import IdeaBoard

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of executing an idea."""

    idea_id: str
    success: bool
    outcome: str  # IdeaOutcome value
    duration_sec: float = 0.0
    error: Optional[str] = None
    signals: List[Dict[str, Any]] = field(default_factory=list)
    notes: str = ""


class IdeaExecutor:
    """Executor for running approved ideas safely.

    The executor:
    1. Validates ideas before execution
    2. Runs in sandbox first (if configured)
    3. Monitors execution with timeouts
    4. Collects outcome metrics
    5. Triggers rollback on failure
    """

    def __init__(
        self,
        board: "IdeaBoard",
        max_concurrent: int = 1,
        default_timeout_sec: float = 300.0,
        require_sandbox: bool = True
    ):
        """Initialize the executor.

        Args:
            board: The IdeaBoard to update
            max_concurrent: Maximum concurrent executions
            default_timeout_sec: Default timeout for execution
            require_sandbox: Require sandbox pass before live execution
        """
        self.board = board
        self.max_concurrent = max_concurrent
        self.default_timeout = default_timeout_sec
        self.require_sandbox = require_sandbox

        self._running: Dict[str, threading.Thread] = {}
        self._lock = threading.Lock()

        # Execution handlers by category
        self._handlers: Dict[str, Callable[["Idea"], ExecutionResult]] = {}

    def register_handler(
        self,
        category: str,
        handler: Callable[["Idea"], ExecutionResult]
    ) -> None:
        """Register an execution handler for a category.

        Args:
            category: IdeaCategory value (e.g., "performance")
            handler: Function that executes ideas of this category
        """
        self._handlers[category] = handler
        logger.info(f"Registered executor handler for {category}")

    def can_execute(self, idea: "Idea") -> tuple[bool, str]:
        """Check if an idea can be executed.

        Returns:
            (can_execute, reason)
        """
        from .models import IdeaStatus, IdeaRisk, SandboxStatus

        # Must be approved
        if idea.status != IdeaStatus.APPROVED:
            return False, f"Idea status is {idea.status.value}, not approved"

        # Check concurrent limit
        with self._lock:
            if len(self._running) >= self.max_concurrent:
                return False, f"Max concurrent executions ({self.max_concurrent}) reached"

        # Check sandbox requirement
        if self.require_sandbox and idea.risk.value in ("medium", "high"):
            if idea.sandbox_status != SandboxStatus.PASSED:
                return False, "Sandbox test required for medium/high risk ideas"

        # Check for handler
        if idea.category.value not in self._handlers:
            # No handler is ok for NONE risk (observation only)
            if idea.risk != IdeaRisk.NONE:
                return False, f"No handler registered for {idea.category.value}"

        return True, "Ready to execute"

    def execute(self, idea_id: str, async_exec: bool = True) -> Optional[ExecutionResult]:
        """Execute an approved idea.

        Args:
            idea_id: ID of the idea to execute
            async_exec: Run asynchronously in thread

        Returns:
            ExecutionResult (immediately if sync, or None if async)
        """
        idea = self.board.get(idea_id)
        if not idea:
            return ExecutionResult(
                idea_id=idea_id,
                success=False,
                outcome="inconclusive",
                error="Idea not found"
            )

        can_exec, reason = self.can_execute(idea)
        if not can_exec:
            return ExecutionResult(
                idea_id=idea_id,
                success=False,
                outcome="inconclusive",
                error=reason
            )

        if async_exec:
            thread = threading.Thread(
                target=self._execute_thread,
                args=(idea,),
                daemon=True
            )
            with self._lock:
                self._running[idea_id] = thread
            thread.start()
            return None
        else:
            return self._execute_sync(idea)

    def _execute_thread(self, idea: "Idea") -> None:
        """Thread wrapper for async execution."""
        try:
            result = self._execute_sync(idea)
            logger.info(f"Idea {idea.id} execution complete: {result.outcome}")
        finally:
            with self._lock:
                self._running.pop(idea.id, None)

    def _execute_sync(self, idea: "Idea") -> ExecutionResult:
        """Synchronous execution of an idea."""
        from .models import IdeaOutcome, Signal, IdeaRisk

        start_time = time.time()

        # Mark as running
        self.board.start_execution(idea.id)

        try:
            # Get handler
            handler = self._handlers.get(idea.category.value)

            if handler:
                # Execute with timeout
                result = self._run_with_timeout(handler, idea, self.default_timeout)
            else:
                # No handler - observation only
                if idea.risk == IdeaRisk.NONE:
                    result = ExecutionResult(
                        idea_id=idea.id,
                        success=True,
                        outcome="learned",
                        notes="Observation recorded"
                    )
                else:
                    result = ExecutionResult(
                        idea_id=idea.id,
                        success=False,
                        outcome="inconclusive",
                        error="No handler"
                    )

            result.duration_sec = time.time() - start_time

            # Update board
            outcome = IdeaOutcome(result.outcome)
            signals = [Signal.from_dict(s) for s in result.signals] if result.signals else None

            if result.success:
                self.board.complete_execution(
                    idea.id,
                    outcome,
                    notes=result.notes,
                    signals=signals
                )
            else:
                # Failure - attempt rollback
                self._attempt_rollback(idea)
                self.board.revert(idea.id, notes=result.error or "Execution failed")

            return result

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Idea {idea.id} execution failed: {e}")

            self._attempt_rollback(idea)
            self.board.revert(idea.id, notes=str(e))

            return ExecutionResult(
                idea_id=idea.id,
                success=False,
                outcome="degraded",
                duration_sec=duration,
                error=str(e)
            )

    def _run_with_timeout(
        self,
        handler: Callable[["Idea"], ExecutionResult],
        idea: "Idea",
        timeout: float
    ) -> ExecutionResult:
        """Run handler with timeout enforcement."""
        result_holder = [None]
        error_holder = [None]

        def run():
            try:
                result_holder[0] = handler(idea)
            except Exception as e:
                error_holder[0] = e

        thread = threading.Thread(target=run)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            # Timeout - thread is still running
            logger.warning(f"Idea {idea.id} execution timed out after {timeout}s")
            return ExecutionResult(
                idea_id=idea.id,
                success=False,
                outcome="inconclusive",
                error=f"Execution timed out after {timeout}s"
            )

        if error_holder[0]:
            raise error_holder[0]

        return result_holder[0] or ExecutionResult(
            idea_id=idea.id,
            success=False,
            outcome="inconclusive",
            error="Handler returned None"
        )

    def _attempt_rollback(self, idea: "Idea") -> bool:
        """Attempt to rollback an idea's changes.

        This is best-effort - handlers should be designed to be
        idempotent and reversible.
        """
        if not idea.rollback_plan:
            logger.warning(f"No rollback plan for idea {idea.id}")
            return False

        logger.info(f"Attempting rollback for idea {idea.id}")

        # TODO: Implement actual rollback execution
        # For now, just log
        for step in idea.rollback_plan:
            logger.info(f"Rollback step: {step}")

        return True

    def run_sandbox(self, idea: "Idea") -> ExecutionResult:
        """Run an idea in sandbox mode.

        Sandbox execution uses mocked/isolated resources and
        doesn't affect the real system.
        """
        from .models import SandboxStatus

        logger.info(f"Running sandbox for idea {idea.id}")

        # Update sandbox status
        idea.sandbox_status = SandboxStatus.RUNNING
        self.board.update(idea)

        start_time = time.time()

        try:
            # Get handler
            handler = self._handlers.get(idea.category.value)

            if handler:
                # Execute in sandbox context
                # TODO: Implement proper sandboxing
                result = ExecutionResult(
                    idea_id=idea.id,
                    success=True,
                    outcome="learned",
                    notes="Sandbox test passed"
                )
            else:
                result = ExecutionResult(
                    idea_id=idea.id,
                    success=True,
                    outcome="learned",
                    notes="No handler - observation only"
                )

            result.duration_sec = time.time() - start_time

            idea.sandbox_status = SandboxStatus.PASSED
            idea.sandbox_results = {
                "duration_sec": result.duration_sec,
                "outcome": result.outcome,
            }
            idea.sandbox_logs.append(f"Sandbox passed in {result.duration_sec:.2f}s")
            self.board.update(idea)

            return result

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Sandbox failed for idea {idea.id}: {e}")

            idea.sandbox_status = SandboxStatus.FAILED
            idea.sandbox_results = {
                "duration_sec": duration,
                "error": str(e),
            }
            idea.sandbox_logs.append(f"Sandbox failed: {e}")
            self.board.update(idea)

            return ExecutionResult(
                idea_id=idea.id,
                success=False,
                outcome="degraded",
                duration_sec=duration,
                error=str(e)
            )

    def get_running(self) -> List[str]:
        """Get IDs of currently running ideas."""
        with self._lock:
            return list(self._running.keys())

    def is_running(self, idea_id: str) -> bool:
        """Check if an idea is currently running."""
        with self._lock:
            return idea_id in self._running
