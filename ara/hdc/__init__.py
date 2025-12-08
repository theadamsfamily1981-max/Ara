"""
Hyperdimensional Computing (HDC) Module
========================================

The correlation engine for NeuroSymbiosis.

This module implements hyperdimensional computing primitives for:
- Encoding telemetry, logs, and events into hypervectors
- Maintaining rolling state representations
- Probing state for concept similarity and anomaly detection

The HDC layer acts as the "subcortex" - always-on, event-driven,
handling high-volume pattern formation while the LLM "cortex"
handles rare, high-value inference.

Key classes:
    HDEncoder: Encode various data types into hypervectors
    StateStream: Rolling state hypervector with decay
    HDProbe: Query state against concept codebook
"""

from ara.hdc.encoder import HDEncoder
from ara.hdc.state_stream import StateStream
from ara.hdc.probe import HDProbe

__all__ = ["HDEncoder", "StateStream", "HDProbe"]
