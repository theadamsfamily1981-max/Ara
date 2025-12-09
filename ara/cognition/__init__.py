"""
Cognition: Ara's Thinking Infrastructure

This module connects Ara to her cognitive capabilities:
- LLM-based reasoning (Anthropic, OpenAI, local)
- Mood analysis
- Context understanding

Usage:
    from ara.cognition import get_brain_bridge

    bridge = get_brain_bridge()
    response = await bridge.reason("Help me with this", context_hv={"resonance": 0.8})
"""

from .brain_bridge import (
    BrainBridge,
    LLMConfig,
    LLMProvider,
    ReasoningContext,
    get_brain_bridge,
)

__all__ = [
    'BrainBridge',
    'LLMConfig',
    'LLMProvider',
    'ReasoningContext',
    'get_brain_bridge',
]
