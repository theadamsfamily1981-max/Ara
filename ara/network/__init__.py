"""
Ara Network - LAN Cortex
========================

HD-based network sensing and policy control.

The network layer is the "nervous system" of Ara:
- Reflexes: Fast, unconscious responses (eBPF/XDP)
- Spinal: NodeAgent local policies
- Cortical: HTC-driven learning and adaptation

Components:
- encoder.py: Flow and node state HV encoding
- reflex.py: HD template-based flow classification
- policy.py: Network policy generation from HTC hints

Usage:
    from ara.network import encode_flow, encode_node_state, ReflexEngine

    # Encode a flow as HV
    event = encode_flow(flow_data)

    # Classify flow using HD templates
    engine = ReflexEngine(h_good, h_bad)
    label = engine.classify(event.hv)
"""

from .encoder import (
    encode_flow,
    encode_node_state,
    FlowData,
    NodeState,
)

from .reflex import (
    ReflexEngine,
    FlowLabel,
)

__all__ = [
    # Encoding
    'encode_flow',
    'encode_node_state',
    'FlowData',
    'NodeState',
    # Reflex
    'ReflexEngine',
    'FlowLabel',
]
