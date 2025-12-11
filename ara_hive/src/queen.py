# ara_hive/src/queen.py
"""
Queen Orchestrator - The Hive Mind
==================================

The Queen is Ara's interface to the Hive.
She routes tasks from Ara (The Face) to Bees (The Workers).

Responsibilities:
    1. Task routing: Match tasks to appropriate Bees/Tools
    2. Pattern selection: Choose execution strategy
    3. Load balancing: Distribute work across the hive
    4. Failure handling: Retry, fallback, escalate

Architecture:
    Ara → QueenOrchestrator → Routing Strategy → BeeAgent/Tool

Routing Strategies:
    - DIRECT: Route to specific tool by name
    - DOMAIN: Route to any tool in a domain
    - CAPABILITY: Route based on required capabilities
    - LLM: Let LLM agent choose the tool
    - PATTERN: Use a predefined execution pattern

Usage:
    from ara_hive.src.queen import QueenOrchestrator, TaskRequest

    queen = QueenOrchestrator()

    # Direct execution
    result = await queen.dispatch(TaskRequest(
        instruction="Fetch the webpage",
        tool="web_fetch",
        params={"url": "https://example.com"},
    ))

    # Domain-based routing
    result = await queen.dispatch(TaskRequest(
        instruction="Search for AI papers",
        kind="research",
        params={"query": "transformer architectures"},
    ))
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

from .registry import ToolRegistry, ToolResult, ToolDomain, get_tool_registry

if TYPE_CHECKING:
    from ..bee_agent import BeeAgent
    from ..waggle_board import WaggleBoard

log = logging.getLogger("Hive.Queen")


# =============================================================================
# Types
# =============================================================================

class TaskKind(str, Enum):
    """High-level task categories."""
    RESEARCH = "research"      # Web search, document reading
    CODE = "code"              # Code generation, execution
    DATA = "data"              # Data processing, transformation
    HARDWARE = "hardware"      # Hardware control, GPIO
    SYSTEM = "system"          # System operations
    LLM = "llm"                # LLM operations
    CUSTOM = "custom"          # User-defined


class RoutingStrategy(str, Enum):
    """How to route a task."""
    DIRECT = "direct"          # Use specified tool
    DOMAIN = "domain"          # Route to domain
    CAPABILITY = "capability"  # Match capabilities
    LLM = "llm"                # LLM decides
    PATTERN = "pattern"        # Use execution pattern


class TaskStatus(str, Enum):
    """Status of a task in the queue."""
    PENDING = "pending"
    ROUTING = "routing"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskRequest:
    """
    A task request from Ara to the Hive.

    Either specify a tool directly, or provide enough
    context for the Queen to route appropriately.
    """
    instruction: str  # Natural language description

    # Routing hints
    tool: Optional[str] = None           # Direct tool name
    kind: Optional[TaskKind] = None      # High-level category
    domain: Optional[ToolDomain] = None  # Tool domain
    capabilities: List[str] = field(default_factory=list)

    # Parameters for the tool
    params: Dict[str, Any] = field(default_factory=dict)

    # Execution settings
    timeout_seconds: int = 60
    max_retries: int = 1
    strategy: RoutingStrategy = RoutingStrategy.DIRECT

    # Metadata
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    priority: int = 5  # 1=highest, 10=lowest
    context: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Infer strategy if not set
        if self.strategy == RoutingStrategy.DIRECT and not self.tool:
            if self.kind:
                self.strategy = RoutingStrategy.DOMAIN
            elif self.capabilities:
                self.strategy = RoutingStrategy.CAPABILITY
            else:
                self.strategy = RoutingStrategy.LLM


@dataclass
class TaskResult:
    """
    Result of a task execution.

    Returned to Ara after the Hive processes a task.
    """
    request_id: str
    status: TaskStatus
    success: bool

    # Output
    output: Any = None
    error: Optional[str] = None

    # Execution details
    tool_used: Optional[str] = None
    strategy_used: RoutingStrategy = RoutingStrategy.DIRECT
    attempts: int = 1

    # Timing
    queued_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def duration_ms(self) -> float:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return 0.0

    @property
    def queue_time_ms(self) -> float:
        if self.queued_at and self.started_at:
            return (self.started_at - self.queued_at).total_seconds() * 1000
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "status": self.status.value,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "tool_used": self.tool_used,
            "strategy_used": self.strategy_used.value,
            "attempts": self.attempts,
            "duration_ms": self.duration_ms,
        }


# =============================================================================
# Queen Orchestrator
# =============================================================================

class QueenOrchestrator:
    """
    The Hive Mind - routes tasks from Ara to Bees.

    The Queen is stateless and can run anywhere.
    All state is in the WaggleBoard or ToolRegistry.
    """

    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        board: Optional[WaggleBoard] = None,
        llm_router: Optional[Callable] = None,
    ):
        """
        Initialize the Queen.

        Args:
            registry: Tool registry (uses global if not provided)
            board: WaggleBoard for distributed execution (optional)
            llm_router: LLM function for smart routing (optional)
        """
        self.registry = registry or get_tool_registry()
        self.board = board
        self.llm_router = llm_router

        # Named bees for direct dispatch
        self.bees: Dict[str, BeeAgent] = {}

        # Routing statistics
        self._stats = {
            "tasks_dispatched": 0,
            "tasks_succeeded": 0,
            "tasks_failed": 0,
            "by_strategy": {s.value: 0 for s in RoutingStrategy},
            "by_tool": {},
        }

        log.info("QueenOrchestrator initialized")

    # =========================================================================
    # Bee Management
    # =========================================================================

    def register_bee(self, name: str, bee: BeeAgent) -> None:
        """Register a named bee for direct dispatch."""
        self.bees[name] = bee
        log.info(f"Registered bee: {name}")

    def unregister_bee(self, name: str) -> None:
        """Unregister a bee."""
        self.bees.pop(name, None)

    def list_bees(self) -> List[str]:
        """List registered bees."""
        return list(self.bees.keys())

    # =========================================================================
    # Main Dispatch
    # =========================================================================

    async def dispatch(self, request: TaskRequest) -> TaskResult:
        """
        Dispatch a task request.

        This is the main entry point from Ara.
        Routes the task based on strategy and executes it.
        """
        result = TaskResult(
            request_id=request.request_id,
            status=TaskStatus.PENDING,
            success=False,
            queued_at=datetime.utcnow(),
        )

        log.info(
            f"Dispatching task {request.request_id}: "
            f"'{request.instruction[:50]}...' (strategy={request.strategy.value})"
        )

        # Route the task
        result.status = TaskStatus.ROUTING
        tool_name = await self._route_task(request)

        if not tool_name:
            result.status = TaskStatus.FAILED
            result.error = "No tool found for task"
            return result

        result.tool_used = tool_name
        result.strategy_used = request.strategy

        # Execute with retries
        result.status = TaskStatus.EXECUTING
        result.started_at = datetime.utcnow()

        for attempt in range(request.max_retries + 1):
            result.attempts = attempt + 1

            tool_result = await self._execute_tool(
                tool_name,
                request.params,
                request.timeout_seconds,
            )

            if tool_result.success:
                result.status = TaskStatus.COMPLETED
                result.success = True
                result.output = tool_result.output
                break
            else:
                result.error = tool_result.error
                if attempt < request.max_retries:
                    log.warning(
                        f"Task {request.request_id} attempt {attempt + 1} failed, retrying..."
                    )
                    await asyncio.sleep(0.5 * (attempt + 1))  # Backoff

        if not result.success:
            result.status = TaskStatus.FAILED

        result.completed_at = datetime.utcnow()

        # Update stats
        self._update_stats(result)

        log.info(
            f"Task {request.request_id} {result.status.value}: "
            f"tool={tool_name}, attempts={result.attempts}, "
            f"duration={result.duration_ms:.0f}ms"
        )

        return result

    async def dispatch_batch(
        self,
        requests: List[TaskRequest],
        parallel: bool = True,
    ) -> List[TaskResult]:
        """
        Dispatch multiple tasks.

        Args:
            requests: List of task requests
            parallel: If True, execute in parallel

        Returns:
            List of results in same order as requests
        """
        if parallel:
            tasks = [self.dispatch(req) for req in requests]
            return await asyncio.gather(*tasks)
        else:
            results = []
            for req in requests:
                result = await self.dispatch(req)
                results.append(result)
            return results

    # =========================================================================
    # Routing
    # =========================================================================

    async def _route_task(self, request: TaskRequest) -> Optional[str]:
        """
        Route a task to an appropriate tool.

        Returns the tool name to use.
        """
        strategy = request.strategy

        if strategy == RoutingStrategy.DIRECT:
            return request.tool

        elif strategy == RoutingStrategy.DOMAIN:
            return await self._route_by_domain(request)

        elif strategy == RoutingStrategy.CAPABILITY:
            return await self._route_by_capability(request)

        elif strategy == RoutingStrategy.LLM:
            return await self._route_by_llm(request)

        elif strategy == RoutingStrategy.PATTERN:
            # Patterns are handled separately
            return request.tool

        return None

    async def _route_by_domain(self, request: TaskRequest) -> Optional[str]:
        """Route based on task kind or domain."""
        # Map kind to domain
        domain = request.domain
        if not domain and request.kind:
            kind_to_domain = {
                TaskKind.RESEARCH: ToolDomain.WEB,
                TaskKind.CODE: ToolDomain.CODE,
                TaskKind.DATA: ToolDomain.DATA,
                TaskKind.HARDWARE: ToolDomain.HARDWARE,
                TaskKind.SYSTEM: ToolDomain.SYSTEM,
                TaskKind.LLM: ToolDomain.LLM,
            }
            domain = kind_to_domain.get(request.kind)

        if not domain:
            return None

        # Find best tool in domain
        tool = self.registry.find_tool_for_task(
            request.instruction,
            domain_hint=domain.value,
        )

        return tool.name if tool else None

    async def _route_by_capability(self, request: TaskRequest) -> Optional[str]:
        """Route based on required capabilities."""
        for cap in request.capabilities:
            tools = self.registry.get_by_tag(cap)
            for tool in tools:
                if tool.can_execute():
                    return tool.name

        return None

    async def _route_by_llm(self, request: TaskRequest) -> Optional[str]:
        """Let LLM choose the tool."""
        if not self.llm_router:
            # Fallback to keyword matching
            tool = self.registry.find_tool_for_task(request.instruction)
            return tool.name if tool else None

        # Build tool list for LLM
        tool_descriptions = self.registry.get_tool_descriptions()

        # Ask LLM
        prompt = f"""
Given the task: "{request.instruction}"

And these available tools:
{tool_descriptions}

Which tool should be used? Reply with just the tool name.
"""
        try:
            tool_name = await self.llm_router(prompt)
            tool_name = tool_name.strip()

            # Verify tool exists
            if self.registry.get(tool_name):
                return tool_name
        except Exception as e:
            log.warning(f"LLM routing failed: {e}")

        return None

    # =========================================================================
    # Execution
    # =========================================================================

    async def _execute_tool(
        self,
        tool_name: str,
        params: Dict[str, Any],
        timeout: int,
    ) -> ToolResult:
        """Execute a tool by name."""
        # Check for named bee first
        bee_name = params.pop("_bee", None)
        if bee_name and bee_name in self.bees:
            return await self._execute_via_bee(bee_name, tool_name, params)

        # Check for distributed execution via WaggleBoard
        if self.board and params.pop("_distributed", False):
            return await self._execute_distributed(tool_name, params)

        # Direct tool execution
        return await self.registry.execute(tool_name, params)

    async def _execute_via_bee(
        self,
        bee_name: str,
        tool_name: str,
        params: Dict[str, Any],
    ) -> ToolResult:
        """Execute via a named bee."""
        bee = self.bees.get(bee_name)
        if not bee:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"Bee not found: {bee_name}",
            )

        # Create task for bee
        task = {
            "tool": tool_name,
            "params": params,
        }

        try:
            result = await bee.execute(task)
            return ToolResult(
                tool_name=tool_name,
                success=result.get("status") == "success",
                output=result.get("output"),
                error=result.get("error"),
            )
        except Exception as e:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=str(e),
            )

    async def _execute_distributed(
        self,
        tool_name: str,
        params: Dict[str, Any],
    ) -> ToolResult:
        """Execute via WaggleBoard (distributed)."""
        if not self.board:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error="WaggleBoard not configured",
            )

        # Submit task to board
        task_id = self.board.submit_task(
            task_type=tool_name,
            payload=params,
        )

        # Wait for completion (with timeout)
        # In production, would use proper async polling
        await asyncio.sleep(1)

        # Check result
        task = self.board.get_task(task_id)
        if task and task.status == "completed":
            return ToolResult(
                tool_name=tool_name,
                success=True,
                output=task.result,
            )
        elif task and task.error:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=task.error,
            )
        else:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error="Task timeout",
            )

    # =========================================================================
    # Statistics
    # =========================================================================

    def _update_stats(self, result: TaskResult) -> None:
        """Update routing statistics."""
        self._stats["tasks_dispatched"] += 1

        if result.success:
            self._stats["tasks_succeeded"] += 1
        else:
            self._stats["tasks_failed"] += 1

        self._stats["by_strategy"][result.strategy_used.value] += 1

        if result.tool_used:
            if result.tool_used not in self._stats["by_tool"]:
                self._stats["by_tool"][result.tool_used] = {"success": 0, "failed": 0}
            if result.success:
                self._stats["by_tool"][result.tool_used]["success"] += 1
            else:
                self._stats["by_tool"][result.tool_used]["failed"] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        return {
            **self._stats,
            "success_rate": (
                self._stats["tasks_succeeded"] / max(1, self._stats["tasks_dispatched"])
            ),
            "registered_bees": len(self.bees),
            "available_tools": len(self.registry.get_available()),
        }


# =============================================================================
# Convenience
# =============================================================================

_default_queen: Optional[QueenOrchestrator] = None


def get_queen() -> QueenOrchestrator:
    """Get the global Queen instance."""
    global _default_queen
    if _default_queen is None:
        _default_queen = QueenOrchestrator()
    return _default_queen


__all__ = [
    # Types
    "TaskKind",
    "RoutingStrategy",
    "TaskStatus",
    "TaskRequest",
    "TaskResult",
    # Queen
    "QueenOrchestrator",
    "get_queen",
]
