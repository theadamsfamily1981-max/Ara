"""
Protocol Module - Card ↔ LLM Communication
==========================================

Defines the STATE_HPV_QUERY and NEW_POLICY_HDC protocol for
communication between the neuromorphic subcortex and GPU cortex.

The protocol enables:
- Card → LLM: Compressed state + summary for rare escalations
- LLM → Card: New policies as HPVs + optional weight updates
"""

from ara.protocol.ipc import (
    StateHPVQuery,
    NewPolicyHDC,
    NeuroSymbiosisProtocol,
)

__all__ = ["StateHPVQuery", "NewPolicyHDC", "NeuroSymbiosisProtocol"]
