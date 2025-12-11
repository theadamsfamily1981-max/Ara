# ara_hive/src/registry.py
"""
Tool Registry - Capabilities Available to the Hive
===================================================

The Tool Registry is the catalog of everything Bees can do.

Tools are organized by domain:
    - core: HDC, Reflexes, Teleology (Ara's native capabilities)
    - web: Fetch, Search, Scrape
    - code: Execute, Analyze, Generate
    - file: Read, Write, Search
    - hardware: GPIO, Sensors, Actuators
    - llm: Generate, Embed, Summarize

Each tool has:
    - name: Unique identifier
    - domain: Category for routing
    - func: Actual callable
    - description: For LLM agent selection
    - schema: Input/output specification

Usage:
    from ara_hive.src.registry import get_tool_registry, Tool

    registry = get_tool_registry()

    # Register a tool
    registry.register(Tool(
        name="web_fetch",
        domain="web",
        func=my_fetch_function,
        description="Fetch content from a URL",
    ))

    # Execute a tool
    result = await registry.execute("web_fetch", {"url": "https://..."})
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

log = logging.getLogger("Hive.Registry")


# =============================================================================
# Types
# =============================================================================

class ToolDomain(str, Enum):
    """Domains of tools available to the Hive."""
    CORE = "core"          # Ara's native capabilities (HDC, Reflexes, etc.)
    WEB = "web"            # Web operations (fetch, search, scrape)
    CODE = "code"          # Code operations (execute, analyze, generate)
    FILE = "file"          # File operations (read, write, search)
    HARDWARE = "hardware"  # Hardware control (GPIO, sensors, actuators)
    LLM = "llm"            # LLM operations (generate, embed, summarize)
    SYSTEM = "system"      # System operations (shell, process, network)
    DATA = "data"          # Data operations (parse, transform, validate)
    CUSTOM = "custom"      # User-defined tools


class ToolStatus(str, Enum):
    """Status of a tool."""
    AVAILABLE = "available"
    BUSY = "busy"
    DISABLED = "disabled"
    ERROR = "error"


@dataclass
class ToolResult:
    """Result of executing a tool."""
    tool_name: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }


@dataclass
class Tool:
    """
    A tool that Bees can execute.

    Tools are the atomic units of work in the Hive.
    """
    name: str
    domain: ToolDomain
    func: Callable
    description: str = ""

    # Schema for input validation
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None

    # Execution constraints
    timeout_seconds: int = 60
    max_concurrent: int = 10
    requires_auth: bool = False

    # Metadata
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)

    # Runtime state
    status: ToolStatus = ToolStatus.AVAILABLE
    _current_executions: int = 0

    def can_execute(self) -> bool:
        """Check if tool can accept more executions."""
        return (
            self.status == ToolStatus.AVAILABLE and
            self._current_executions < self.max_concurrent
        )

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        """Execute the tool with given parameters."""
        start = time.time()
        self._current_executions += 1

        try:
            # Call the function
            if asyncio.iscoroutinefunction(self.func):
                output = await asyncio.wait_for(
                    self.func(**params),
                    timeout=self.timeout_seconds,
                )
            else:
                output = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.func(**params),
                )

            return ToolResult(
                tool_name=self.name,
                success=True,
                output=output,
                duration_ms=(time.time() - start) * 1000,
            )

        except asyncio.TimeoutError:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"Tool timed out after {self.timeout_seconds}s",
                duration_ms=(time.time() - start) * 1000,
            )

        except Exception as e:
            log.exception(f"Tool {self.name} failed: {e}")
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

        finally:
            self._current_executions -= 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "domain": self.domain.value,
            "description": self.description,
            "version": self.version,
            "tags": self.tags,
            "status": self.status.value,
        }

    def to_llm_description(self) -> str:
        """Format for LLM tool selection."""
        return f"{self.name}: {self.description}"


# =============================================================================
# Tool Registry
# =============================================================================

class ToolRegistry:
    """
    Central registry of all tools available to the Hive.

    The Queen consults this to route tasks to appropriate tools.
    Bees use this to execute their assigned work.
    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._by_domain: Dict[ToolDomain, List[str]] = {d: [] for d in ToolDomain}
        self._by_tag: Dict[str, List[str]] = {}

        log.info("ToolRegistry initialized")

    # =========================================================================
    # Registration
    # =========================================================================

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        if tool.name in self._tools:
            log.warning(f"Overwriting existing tool: {tool.name}")

        self._tools[tool.name] = tool

        # Index by domain
        if tool.name not in self._by_domain[tool.domain]:
            self._by_domain[tool.domain].append(tool.name)

        # Index by tags
        for tag in tool.tags:
            if tag not in self._by_tag:
                self._by_tag[tag] = []
            if tool.name not in self._by_tag[tag]:
                self._by_tag[tag].append(tool.name)

        log.info(f"Registered tool: {tool.name} ({tool.domain.value})")

    def register_function(
        self,
        name: str,
        func: Callable,
        domain: Union[str, ToolDomain] = ToolDomain.CUSTOM,
        description: str = "",
        **kwargs,
    ) -> Tool:
        """Convenience method to register a function as a tool."""
        if isinstance(domain, str):
            domain = ToolDomain(domain)

        tool = Tool(
            name=name,
            domain=domain,
            func=func,
            description=description or func.__doc__ or "",
            **kwargs,
        )
        self.register(tool)
        return tool

    def unregister(self, name: str) -> bool:
        """Unregister a tool."""
        if name not in self._tools:
            return False

        tool = self._tools.pop(name)

        # Remove from indices
        if name in self._by_domain[tool.domain]:
            self._by_domain[tool.domain].remove(name)
        for tag in tool.tags:
            if tag in self._by_tag and name in self._by_tag[tag]:
                self._by_tag[tag].remove(name)

        log.info(f"Unregistered tool: {name}")
        return True

    # =========================================================================
    # Lookup
    # =========================================================================

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_by_domain(self, domain: Union[str, ToolDomain]) -> List[Tool]:
        """Get all tools in a domain."""
        if isinstance(domain, str):
            domain = ToolDomain(domain)
        return [self._tools[n] for n in self._by_domain.get(domain, [])]

    def get_by_tag(self, tag: str) -> List[Tool]:
        """Get all tools with a tag."""
        return [self._tools[n] for n in self._by_tag.get(tag, [])]

    def get_available(self) -> List[Tool]:
        """Get all available tools."""
        return [t for t in self._tools.values() if t.can_execute()]

    def list_all(self) -> List[str]:
        """List all tool names."""
        return list(self._tools.keys())

    def list_domains(self) -> Dict[str, int]:
        """List domains with tool counts."""
        return {d.value: len(names) for d, names in self._by_domain.items() if names}

    # =========================================================================
    # Execution
    # =========================================================================

    async def execute(
        self,
        tool_name: str,
        params: Dict[str, Any],
    ) -> ToolResult:
        """Execute a tool by name."""
        tool = self._tools.get(tool_name)
        if not tool:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"Tool not found: {tool_name}",
            )

        if not tool.can_execute():
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"Tool {tool_name} is not available (status={tool.status.value})",
            )

        return await tool.execute(params)

    async def execute_batch(
        self,
        calls: List[tuple[str, Dict[str, Any]]],
    ) -> List[ToolResult]:
        """Execute multiple tools in parallel."""
        tasks = [self.execute(name, params) for name, params in calls]
        return await asyncio.gather(*tasks)

    # =========================================================================
    # For LLM Agents
    # =========================================================================

    def get_tool_descriptions(
        self,
        domain: Optional[ToolDomain] = None,
        limit: int = 50,
    ) -> str:
        """
        Get tool descriptions formatted for LLM.

        Used by Bees to understand what tools they can use.
        """
        if domain:
            tools = self.get_by_domain(domain)
        else:
            tools = list(self._tools.values())

        tools = tools[:limit]

        lines = [f"Available tools ({len(tools)}):\n"]
        for tool in tools:
            lines.append(f"  - {tool.to_llm_description()}")

        return "\n".join(lines)

    def find_tool_for_task(
        self,
        task_description: str,
        domain_hint: Optional[str] = None,
    ) -> Optional[Tool]:
        """
        Find the best tool for a task description.

        Simple keyword matching; in production use embedding similarity.
        """
        candidates = []

        if domain_hint:
            try:
                domain = ToolDomain(domain_hint)
                candidates = self.get_by_domain(domain)
            except ValueError:
                pass

        if not candidates:
            candidates = list(self._tools.values())

        task_lower = task_description.lower()

        # Score by keyword match
        best_tool = None
        best_score = 0

        for tool in candidates:
            if not tool.can_execute():
                continue

            score = 0
            desc_lower = (tool.description + " " + tool.name).lower()

            # Check for keyword matches
            for word in task_lower.split():
                if len(word) > 3 and word in desc_lower:
                    score += 1

            # Exact name match
            if tool.name.lower() in task_lower:
                score += 5

            if score > best_score:
                best_score = score
                best_tool = tool

        return best_tool

    # =========================================================================
    # Stats
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            "total_tools": len(self._tools),
            "by_domain": self.list_domains(),
            "available": len(self.get_available()),
            "tags": list(self._by_tag.keys()),
        }


# =============================================================================
# Built-in Tools
# =============================================================================

def _register_builtin_tools(registry: ToolRegistry) -> None:
    """Register built-in tools."""

    # Echo tool for testing
    registry.register_function(
        name="echo",
        func=lambda message="": {"echo": message},
        domain=ToolDomain.CORE,
        description="Echo back the input message",
        tags=["test", "debug"],
    )

    # Delay tool for testing
    async def delay_tool(seconds: float = 1.0) -> Dict[str, Any]:
        await asyncio.sleep(seconds)
        return {"delayed": seconds}

    registry.register_function(
        name="delay",
        func=delay_tool,
        domain=ToolDomain.CORE,
        description="Wait for specified seconds",
        tags=["test", "debug"],
    )

    # Fail tool for testing error handling
    def fail_tool(message: str = "Test failure") -> None:
        raise RuntimeError(message)

    registry.register_function(
        name="fail",
        func=fail_tool,
        domain=ToolDomain.CORE,
        description="Always fails (for testing)",
        tags=["test", "debug"],
    )


# =============================================================================
# Decorator
# =============================================================================

def tool(
    name: Optional[str] = None,
    domain: Union[str, ToolDomain] = ToolDomain.CUSTOM,
    description: str = "",
    **kwargs,
):
    """
    Decorator to register a function as a tool.

    Usage:
        @tool(name="my_tool", domain="web", description="Does something")
        async def my_tool(url: str) -> dict:
            ...
    """
    def decorator(func: Callable) -> Callable:
        tool_name = name or func.__name__

        # Defer registration until registry is available
        func._tool_metadata = {
            "name": tool_name,
            "domain": domain if isinstance(domain, ToolDomain) else ToolDomain(domain),
            "description": description or func.__doc__ or "",
            **kwargs,
        }

        return func

    return decorator


def register_decorated_tools(registry: ToolRegistry, module: Any) -> int:
    """
    Register all decorated tools from a module.

    Usage:
        import my_tools
        register_decorated_tools(registry, my_tools)
    """
    count = 0
    for name in dir(module):
        obj = getattr(module, name)
        if callable(obj) and hasattr(obj, "_tool_metadata"):
            metadata = obj._tool_metadata
            registry.register(Tool(
                func=obj,
                **metadata,
            ))
            count += 1
    return count


# =============================================================================
# Singleton
# =============================================================================

_default_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = ToolRegistry()
        _register_builtin_tools(_default_registry)
    return _default_registry


__all__ = [
    # Types
    "ToolDomain",
    "ToolStatus",
    "ToolResult",
    "Tool",
    # Registry
    "ToolRegistry",
    "get_tool_registry",
    # Decorators
    "tool",
    "register_decorated_tools",
]
