"""
Hyperdimensional VSA - Soul Substrate
======================================

16kD hypervectors implementing NIB structural compression (I(z;s))
and QUANTA high-capacity encoding (r=256 LoRA).

Capacity:
    Theory: 10^1235 unique representable states
    Practice: N=12 modalities before interference collapse

Modalities Supported:
    - Speech prosody (AraVoice)
    - Vision (scene_hv)
    - IMU posture
    - Interoception
    - Hive GPU/memory pressure
    - Text embeddings
    - Audio features
    - Emotion vectors
    - Motor commands
    - Reward signals
    - Memory traces
    - Temporal context

Emergent Properties:
    - Holographic identity: T_s preserved across modality fusion
    - Bias sentinel: HV interference patterns detect drift
    - Graceful degradation: partial information still retrievable

Operations:
    - Bundle (⊕): Superposition of concepts
    - Bind (⊗): Association/relation encoding
    - Permute (ρ): Sequence/position encoding
    - Similarity (δ): Cosine/Hamming distance

Usage:
    from ara_core.vsa import (
        HyperVector, VSASpace, ModalityEncoder,
        SoulBundle, bind, bundle, permute, similarity
    )

    # Create space
    space = VSASpace(dim=16384)

    # Encode modalities
    voice_hv = space.encode("voice", voice_features)
    vision_hv = space.encode("vision", vision_features)

    # Bundle into soul
    soul = bundle(voice_hv, vision_hv)

    # Check similarity
    sim = similarity(soul, reference_soul)
"""

from .hypervector import (
    HyperVector,
    VSASpace,
    ModalityEncoder,
    SoulBundle,
    bind,
    bundle,
    permute,
    similarity,
    get_vsa_space,
    encode_modality,
    create_soul_bundle,
)

__all__ = [
    "HyperVector",
    "VSASpace",
    "ModalityEncoder",
    "SoulBundle",
    "bind",
    "bundle",
    "permute",
    "similarity",
    "get_vsa_space",
    "encode_modality",
    "create_soul_bundle",
]
