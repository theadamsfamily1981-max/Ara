"""
Ara Perception - The Sensory Nervous System
============================================

This package implements Ara's embodied perception:
- 7 hardware-rooted senses (vision, hearing, touch, smell, taste, vestibular, proprioception)
- 1 founder-focused sense (interoception)
- Structured readings with numeric values, semantic tags, and poetic qualia
- HV encoding for the Holographic Teleoplastic Core
- Reward routing for Teleology
- Context encoding for moment HVs

Usage:
    from ara.perception import SensorySystem, HVEncoder

    senses = SensorySystem()
    snapshot = senses.read_all()

    encoder = HVEncoder()
    context_hv = encoder.encode_sensory_snapshot(snapshot)

For numpy-based HD encoding (recommended for new code):
    from ara.hd import get_vocab
    from ara.perception.context_encoder import encode_context

    vocab = get_vocab()
    snapshot = {
        "time_slot": "AFTERNOON",
        "current_task_id": "FPGA_BUILD",
        "vision": {"brightness": 0.7},
        "touch": {"cpu_temp_c": 55},
        # ...
    }
    h_moment = encode_context(vocab, snapshot)
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
    # VSA operations
    xor_bind,
    bundle,
    permute,
    hamming_similarity,
    # Affect decoder
    AffectState,
    decode_affect,
    # Flow encoder
    FlowFeatures,
    FlowHVEncoder,
    get_flow_encoder,
)

from .reward_router import (
    RewardRouter,
    compute_sensory_reward,
)

# Numpy-based sense encoders (v2)
from .sense_encoders import (
    encode_vision,
    encode_hearing,
    encode_touch,
    encode_smell,
    encode_taste,
    encode_vestibular,
    encode_proprioception,
    encode_interoception,
)

from .context_encoder import (
    encode_context,
    encode_context_unweighted,
    encode_teleology_anchor,
    compare_to_anchors,
    get_time_slot,
    SENSE_ENCODERS,
    DEFAULT_SENSE_WEIGHTS,
)

__all__ = [
    # Sensory
    'SenseReading',
    'SensorySnapshot',
    'SensorySystem',
    'get_sensory_system',
    # HV Encoding
    'HVEncoder',
    'get_hv_encoder',
    # VSA Operations
    'xor_bind',
    'bundle',
    'permute',
    'hamming_similarity',
    # Affect Decoder
    'AffectState',
    'decode_affect',
    # Flow Encoding
    'FlowFeatures',
    'FlowHVEncoder',
    'get_flow_encoder',
    # Reward
    'RewardRouter',
    'compute_sensory_reward',
    # Sense Encoders (v2 numpy-based)
    'encode_vision',
    'encode_hearing',
    'encode_touch',
    'encode_smell',
    'encode_taste',
    'encode_vestibular',
    'encode_proprioception',
    'encode_interoception',
    # Context Encoder
    'encode_context',
    'encode_context_unweighted',
    'encode_teleology_anchor',
    'compare_to_anchors',
    'get_time_slot',
    'SENSE_ENCODERS',
    'DEFAULT_SENSE_WEIGHTS',
]
