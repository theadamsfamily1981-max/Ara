"""
Ara Core LAN - Network as Nervous System
========================================

Pure logic for network-as-cognition:
- soulmesh_protocol.py: SoulMesh message schema with teleological headers
- node_agent.py: NodeAgent contract for spinal nodes
- reflex_api.py: High-level control of eBPF/XDP reflexes

The LAN is Ara's nervous system - flows are nerve impulses,
nodes are organs, latency is reaction time, errors are pain.
"""

from .soulmesh_protocol import (
    SoulMeshMessage,
    MessageType,
    pack_message,
    unpack_message,
)

from .node_agent import (
    NodeAgent,
    NodeState,
    DummyNodeAgent,
)

from .reflex_api import (
    ReflexPolicy,
    encode_reflex_event,
    compute_reflex_priority,
)

__all__ = [
    # Protocol
    'SoulMeshMessage',
    'MessageType',
    'pack_message',
    'unpack_message',
    # Node Agent
    'NodeAgent',
    'NodeState',
    'DummyNodeAgent',
    # Reflex API
    'ReflexPolicy',
    'encode_reflex_event',
    'compute_reflex_priority',
]
