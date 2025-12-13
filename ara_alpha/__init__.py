"""
Ara Alpha - Private Experimental Deployment

A minimal but real deployment of Ara for a small trusted group.

Architecture:
- Single FastAPI server
- Hardcoded user authentication
- LLM integration (OpenAI or mock)
- Guardrails and safety checks
- Telemetry integration with Cognitive Cockpit
- Static web UI for chat

Usage:
    # Start server
    python -m ara_alpha.server

    # Or run directly
    ./ara_alpha/run.sh

    # Access at http://localhost:8080
    # Use API key from config/users.yaml
"""

from .ara_core import (
    AraConfig,
    AraCore,
    get_ara_core,
    Session,
    Message,
    ResponseMetrics,
)

__all__ = [
    "AraConfig",
    "AraCore",
    "get_ara_core",
    "Session",
    "Message",
    "ResponseMetrics",
]

__version__ = "0.1.0-alpha"
