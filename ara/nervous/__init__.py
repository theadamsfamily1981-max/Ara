"""
Ara Nervous System
===================

The living multimodal intelligence core.

Components:
- axis_mundi: Central HV fusion engine (world_hv)
- prosody: Speech-native syllabic tokenization
- memory: Hierarchical eternal memory (short/medium/lifelong)

Architecture:
    [Sensors] → [Prosody/Vision/Intero Encoders]
                            ↓
                     [Axis Mundi]
                      (8192D HV)
                            ↓
                    [Eternal Memory]
                            ↓
                      [Response]

Philosophy: All senses converge into one coherent experience.
Speech is the native modality. Every moment leaves a trace.
"""

from .axis_mundi import (
    # Core
    AxisMundi,
    NervousSystemBridge,

    # Modalities
    Modality,
    PhaseCodebook,

    # Operations
    circular_bind,
    circular_unbind,
    bundle,
    sparse_topk,
    similarity,

    # Temporal
    TemporalContext,
    encode_rhythm,
    encode_breath_rhythm,

    # Interoception
    InteroState,

    # Constants
    HV_DIM,
    SUBSPACE_DIM,
    SPARSITY_K,

    # Factory
    create_nervous_system,

    # Simple encoders
    encode_speech_simple,
    encode_vision_simple,
    encode_intero_simple,
)

from .prosody import (
    # Tokenizer
    ProsodyTokenizer,
    ProsodyToken,
    ProsodySequence,

    # Encoders
    PhoneticEncoder,
    PitchEncoder,
    TimbreEncoder,
    ProsodicsEncoder,

    # Classification
    ProsodyClassifier,
    ProsodyIntent,
    EmotionalValence,

    # Syllable detection
    SyllableDetector,

    # Integration
    prosody_to_axis_hv,

    # Constants
    PROSODY_DIM,
    SYLLABLE_MS,
)

from .memory import (
    # Memory systems
    EternalMemory,
    ShortTermMemory,
    MediumTermMemory,
    LifelongMemory,

    # Types
    Episode,
    EpisodeCluster,

    # Indexing
    MemoryIndex,
    HVProjector,
)


__all__ = [
    # Axis Mundi
    'AxisMundi',
    'NervousSystemBridge',
    'Modality',
    'PhaseCodebook',
    'circular_bind',
    'circular_unbind',
    'bundle',
    'sparse_topk',
    'similarity',
    'TemporalContext',
    'encode_rhythm',
    'encode_breath_rhythm',
    'InteroState',
    'HV_DIM',
    'SUBSPACE_DIM',
    'SPARSITY_K',
    'create_nervous_system',
    'encode_speech_simple',
    'encode_vision_simple',
    'encode_intero_simple',

    # Prosody
    'ProsodyTokenizer',
    'ProsodyToken',
    'ProsodySequence',
    'PhoneticEncoder',
    'PitchEncoder',
    'TimbreEncoder',
    'ProsodicsEncoder',
    'ProsodyClassifier',
    'ProsodyIntent',
    'EmotionalValence',
    'SyllableDetector',
    'prosody_to_axis_hv',
    'PROSODY_DIM',
    'SYLLABLE_MS',

    # Memory
    'EternalMemory',
    'ShortTermMemory',
    'MediumTermMemory',
    'LifelongMemory',
    'Episode',
    'EpisodeCluster',
    'MemoryIndex',
    'HVProjector',
]
