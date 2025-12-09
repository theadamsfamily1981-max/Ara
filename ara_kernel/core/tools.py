"""
Tools Registry
===============

Simple tool registry for v0.1.
Tools are callables: fn(args: dict) -> dict

The quantum optimizer and other specialized tools plug in here.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Callable, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    """Tool categories for organization and safety."""
    UTIL = "util"
    PUBLISHING = "publishing"
    GITHUB = "github"
    HARDWARE = "hardware"
    QUANTUM = "quantum"
    LAB = "lab"
    MEMORY = "memory"


@dataclass
class ToolSpec:
    """Specification for a registered tool."""
    name: str
    fn: Callable
    category: ToolCategory
    description: str = ""
    requires_approval: bool = False
    is_async: bool = False
    timeout_sec: float = 60.0


@dataclass
class ToolResult:
    """Result from a single tool execution."""
    tool: str
    ok: bool
    output: Any = None
    error: Optional[str] = None
    duration_ms: float = 0


class ToolsRegistry:
    """
    Tool registry for the Ara kernel.

    Tools are callables: fn(args: dict) -> dict
    Supports both sync and async tools.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(
        self,
        name: str,
        fn: Callable[[Dict[str, Any]], Union[Dict[str, Any], Any]],
        category: ToolCategory = ToolCategory.UTIL,
        description: str = "",
        requires_approval: bool = False,
        timeout_sec: float = 60.0,
    ) -> None:
        """Register a tool."""
        is_async = asyncio.iscoroutinefunction(fn)
        spec = ToolSpec(
            name=name,
            fn=fn,
            category=category,
            description=description,
            requires_approval=requires_approval,
            is_async=is_async,
            timeout_sec=timeout_sec,
        )
        logger.info(f"Registering tool: {name} (category={category.value}, async={is_async})")
        self._tools[name] = spec

    def get(self, name: str) -> Optional[ToolSpec]:
        """Get a tool spec by name."""
        return self._tools.get(name)

    def list_tools(self, category: Optional[ToolCategory] = None) -> List[str]:
        """List all registered tools, optionally filtered by category."""
        if category:
            return [name for name, spec in self._tools.items() if spec.category == category]
        return list(self._tools.keys())

    def execute_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a plan synchronously.

        Plan format:
        {
            "thought": "...",
            "actions": [{"tool": "name", "args": {...}}, ...]
        }
        """
        actions: List[Dict[str, Any]] = plan.get("actions", [])
        results: List[ToolResult] = []

        for action in actions:
            tool_name = action.get("tool", "")
            args = action.get("args", {}) or {}

            result = self._execute_single(tool_name, args)
            results.append(result)

        return {
            "actions_executed": len([r for r in results if r.ok]),
            "actions_failed": len([r for r in results if not r.ok]),
            "results": [
                {
                    "tool": r.tool,
                    "ok": r.ok,
                    "output": r.output,
                    "error": r.error,
                    "duration_ms": r.duration_ms,
                }
                for r in results
            ],
        }

    async def execute_plan_async(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a plan asynchronously."""
        actions: List[Dict[str, Any]] = plan.get("actions", [])
        results: List[ToolResult] = []

        for action in actions:
            tool_name = action.get("tool", "")
            args = action.get("args", {}) or {}

            result = await self._execute_single_async(tool_name, args)
            results.append(result)

        return {
            "actions_executed": len([r for r in results if r.ok]),
            "actions_failed": len([r for r in results if not r.ok]),
            "results": [
                {
                    "tool": r.tool,
                    "ok": r.ok,
                    "output": r.output,
                    "error": r.error,
                    "duration_ms": r.duration_ms,
                }
                for r in results
            ],
        }

    def _execute_single(self, tool_name: str, args: Dict[str, Any]) -> ToolResult:
        """Execute a single tool synchronously."""
        import time

        spec = self._tools.get(tool_name)
        if not spec:
            logger.warning(f"Unknown tool: {tool_name}")
            return ToolResult(tool=tool_name, ok=False, error="unknown_tool")

        if spec.requires_approval:
            logger.warning(f"Tool '{tool_name}' requires approval - skipping")
            return ToolResult(tool=tool_name, ok=False, error="requires_approval")

        try:
            start = time.time()
            logger.info(f"Executing tool '{tool_name}' with args: {args}")

            if spec.is_async:
                # Run async tool in event loop
                output = asyncio.get_event_loop().run_until_complete(spec.fn(args))
            else:
                output = spec.fn(args)

            duration_ms = (time.time() - start) * 1000
            return ToolResult(tool=tool_name, ok=True, output=output, duration_ms=duration_ms)

        except Exception as e:
            logger.exception(f"Tool '{tool_name}' failed")
            return ToolResult(tool=tool_name, ok=False, error=str(e))

    async def _execute_single_async(self, tool_name: str, args: Dict[str, Any]) -> ToolResult:
        """Execute a single tool asynchronously."""
        import time

        spec = self._tools.get(tool_name)
        if not spec:
            logger.warning(f"Unknown tool: {tool_name}")
            return ToolResult(tool=tool_name, ok=False, error="unknown_tool")

        if spec.requires_approval:
            logger.warning(f"Tool '{tool_name}' requires approval - skipping")
            return ToolResult(tool=tool_name, ok=False, error="requires_approval")

        try:
            start = time.time()
            logger.info(f"Executing tool '{tool_name}' with args: {args}")

            if spec.is_async:
                output = await asyncio.wait_for(
                    spec.fn(args),
                    timeout=spec.timeout_sec,
                )
            else:
                output = spec.fn(args)

            duration_ms = (time.time() - start) * 1000
            return ToolResult(tool=tool_name, ok=True, output=output, duration_ms=duration_ms)

        except asyncio.TimeoutError:
            return ToolResult(tool=tool_name, ok=False, error="timeout")
        except Exception as e:
            logger.exception(f"Tool '{tool_name}' failed")
            return ToolResult(tool=tool_name, ok=False, error=str(e))


# =============================================================================
# Built-in Tool Implementations
# =============================================================================

def echo_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Simple echo tool for testing."""
    return {"echo": args}


def memory_search_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Search memory (stub)."""
    query = args.get("query", "")
    return {
        "status": "stub",
        "query": query,
        "results": [],
        "message": "Memory search not yet connected.",
    }


def quantum_optimizer_stub(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Hook where your quantum simulator lives later.
    For now, just echo & pretend.

    When ready, this will:
    - Parse problem spec (portfolio, scheduler, Hamiltonian, etc.)
    - Use Qiskit/PennyLane/etc. + desktop simulator
    - Return optimized suggestion
    """
    problem = args.get("problem", "unknown")
    problem_type = args.get("type", "generic")

    return {
        "status": "stub",
        "problem": problem,
        "type": problem_type,
        "message": "Quantum optimizer not yet implemented. Ready for Qiskit/PennyLane integration.",
        "suggested_approach": _suggest_quantum_approach(problem_type),
    }


def _suggest_quantum_approach(problem_type: str) -> str:
    """Suggest a quantum approach based on problem type."""
    approaches = {
        "portfolio": "QAOA for portfolio optimization",
        "scheduler": "VQE for scheduling constraints",
        "search": "Grover's algorithm for unstructured search",
        "ml": "Quantum kernel methods or variational classifier",
        "simulation": "VQE/QPE for Hamiltonian simulation",
    }
    return approaches.get(problem_type, "QAOA or VQE depending on problem structure")


def draft_content_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Draft content (stub for publishing pipeline)."""
    content_type = args.get("type", "generic")
    topic = args.get("topic", "")

    return {
        "status": "stub",
        "content_type": content_type,
        "topic": topic,
        "message": "Content drafting requires model integration.",
    }


def github_search_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Search GitHub (stub)."""
    query = args.get("query", "")
    repo = args.get("repo", "")

    return {
        "status": "stub",
        "query": query,
        "repo": repo,
        "message": "GitHub search requires API integration.",
    }


# =============================================================================
# Registry Builder
# =============================================================================

def build_default_registry() -> ToolsRegistry:
    """Build the default tool registry with all built-in tools."""
    reg = ToolsRegistry()

    # Utility tools
    reg.register(
        "util.echo",
        echo_tool,
        category=ToolCategory.UTIL,
        description="Echo input for testing",
    )

    # Memory tools
    reg.register(
        "memory.search",
        memory_search_tool,
        category=ToolCategory.MEMORY,
        description="Search episodic and semantic memory",
    )

    # Quantum tools
    reg.register(
        "quantum.optimize",
        quantum_optimizer_stub,
        category=ToolCategory.QUANTUM,
        description="Quantum optimization (stub - ready for Qiskit/PennyLane)",
    )

    # Publishing tools
    reg.register(
        "publishing.draft",
        draft_content_tool,
        category=ToolCategory.PUBLISHING,
        description="Draft content for publishing pipeline",
    )

    # GitHub tools
    reg.register(
        "github.search",
        github_search_tool,
        category=ToolCategory.GITHUB,
        description="Search GitHub repos and issues",
    )

    return reg
