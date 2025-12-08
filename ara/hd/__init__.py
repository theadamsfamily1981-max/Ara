"""
Ara HD (Hyperdimensional) Computing Module
==========================================

Core VSA/HD operations and vocabulary for Ara's sensorium.

This module provides:
- Binary hypervector operations (XOR binding, majority bundling)
- Vocabulary management for roles, features, bins, tags
- Cosine similarity for attractor matching

Canonical parameters:
- Dimension: D = 16,384 bits
- Representation: {0, 1} binary (internally converted to {-1, +1} for math)
- Binding: XOR (self-inverse, associative)
- Bundling: Majority vote (sum + sign)

References:
- Kanerva (2009): Hyperdimensional Computing
- Gayler (2003): Vector Symbolic Architectures
- Karunaratne (2020): In-memory HDC

Usage:
    from ara.hd import HDVocab, bind, bundle, cosine, random_hv

    vocab = HDVocab()
    h_role = vocab.role("VISION")
    h_feat = vocab.feature("BRIGHTNESS")
    h_val = vocab.bin("HIGH")

    h_attr = bind(h_role, bind(h_feat, h_val))
    h_context = bundle([h_attr1, h_attr2, h_attr3])
"""

from .ops import (
    DIM,
    random_hv,
    bind,
    bundle,
    cosine,
    hamming_distance,
    permute,
)

from .vocab import HDVocab, get_vocab

__all__ = [
    # Constants
    'DIM',
    # Operations
    'random_hv',
    'bind',
    'bundle',
    'cosine',
    'hamming_distance',
    'permute',
    # Vocabulary
    'HDVocab',
    'get_vocab',
]
