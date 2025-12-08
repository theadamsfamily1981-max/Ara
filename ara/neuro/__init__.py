"""
Neuromorphic Processing Module
==============================

On-chip spiking and Hebbian learning primitives.

This module implements the "protocol collapse" from NeuroSymbiosis v1 to v2:
- v1: Card sends STATE_HPV_QUERY → GPU LLM → NEW_POLICY_HDC back
- v2: On-chip Hebbian learning: pre × post × ρ → Δw

Key classes:
    HebbianPolicyLearner: Three-factor learning rule
    SpikeEncoder: HPV → spike stream conversion
    UnifiedHead: Reflex head with shared learnable weights
    ReflexUnit: Complete integrated unit

The goal: eliminate GPU round-trip by doing policy adaptation on-chip.
"""

from ara.neuro.hebbian import HebbianPolicyLearner, HebbianConfig
from ara.neuro.spike_encoder import SpikeEncoder, SpikeEncoderConfig
from ara.neuro.unified_head import UnifiedHead, UnifiedHeadConfig, ReflexUnit

__all__ = [
    "HebbianPolicyLearner",
    "HebbianConfig",
    "SpikeEncoder",
    "SpikeEncoderConfig",
    "UnifiedHead",
    "UnifiedHeadConfig",
    "ReflexUnit",
]
