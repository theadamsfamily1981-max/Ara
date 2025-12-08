"""
Bridge Module - Card <-> Cortex Communication
=============================================

Defines the wire protocol and message types for communication between
the neuromorphic card (subcortex) and the GPU-based LLM (cortex).

Key components:
    messages.py: STATE_HPV_QUERY and NEW_POLICY message types
    cortex_bridge.py: High-level bridge for card<->GPU communication

Wire protocol:
    Card -> GPU: STATE_HPV_QUERY (compressed state + telemetry)
    GPU -> Card: NEW_POLICY (policy HPV + SNN deltas)
"""

from ara.bridge.messages import (
    StateHPVQuery,
    NewPolicy,
    pack_bipolar_to_bytes,
    unpack_bytes_to_bipolar,
    serialize_snn_deltas,
    deserialize_snn_deltas
)

from ara.bridge.cortex_bridge import CortexBridge

__all__ = [
    'StateHPVQuery',
    'NewPolicy',
    'CortexBridge',
    'pack_bipolar_to_bytes',
    'unpack_bytes_to_bipolar',
    'serialize_snn_deltas',
    'deserialize_snn_deltas'
]
