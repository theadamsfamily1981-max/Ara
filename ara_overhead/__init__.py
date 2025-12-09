"""
API Overhead Engine
====================

Pre-compiled workflows and API cheat sheets for zero-LLM execution.

The core insight: once a pattern is known, we don't ask the LLM again.
We just look it up and execute.

Components:
- api_cheats/*.json: Service schemas, auth patterns, endpoints
- errors.jsonl: Error → rule mappings for automatic handling
- exec_graph.json: Multi-step workflows as executable graphs
- rate_limits.json: Per-service rate limiting configs

Usage:
    from ara_overhead import APIOverheadEngine

    engine = APIOverheadEngine()
    result = engine.call_service("github", "create_file", payload)
    # Automatically handles auth, rate limits, errors
"""

from .runtime import APIOverheadEngine
from .schemas import ServiceSchema, ErrorRule, ExecGraph

__all__ = [
    "APIOverheadEngine",
    "ServiceSchema",
    "ErrorRule",
    "ExecGraph",
]
