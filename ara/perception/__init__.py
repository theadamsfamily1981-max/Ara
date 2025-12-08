"""
Ara Perception - The Sensory Nervous System
============================================

This package implements Ara's embodied perception:
- 7 hardware-rooted senses (vision, hearing, touch, smell, taste, vestibular, proprioception)
- 1 founder-focused sense (interoception)
- Structured readings with numeric values, semantic tags, and poetic qualia
- HV encoding for the Holographic Teleoplastic Core
- Reward routing for Teleology

Usage:
    from ara.perception import SensorySystem, HVEncoder

    senses = SensorySystem()
    snapshot = senses.read_all()

    encoder = HVEncoder()
    context_hv = encoder.encode_sensory_snapshot(snapshot)
"""

from .sensory import (
    SenseReading,
    SensorySnapshot,
    SensorySystem,
    get_sensory_system,
)

from .hv_encoder import (
    HVEncoder,
    get_hv_encoder,
)

from .reward_router import (
    RewardRouter,
    compute_sensory_reward,
)

__all__ = [
    'SenseReading',
    'SensorySnapshot',
    'SensorySystem',
    'get_sensory_system',
    'HVEncoder',
    'get_hv_encoder',
    'RewardRouter',
    'compute_sensory_reward',
]
