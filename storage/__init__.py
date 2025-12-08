"""
Ara Storage Module
==================

Storage systems for the Teleoplastic Cybernetic Organism.

Submodules:
    heim_optimized: 100× compression with 0% recall loss
    oversample_rerank: Two-stage retrieval pipeline
"""

from .heim_optimized import (
    HEIM_CONFIG,
    get_heim_config,
    heim_compress,
    heim_decompress,
    ClusterIndex,
    get_cluster_index,
    validate_geometry,
    validate_bundling,
)

from .oversample_rerank import (
    OVERSAMPLE_CONFIG,
    oversample_rerank,
    OversampleTuner,
    get_tuner,
    get_metrics,
)

__all__ = [
    # Heim
    'HEIM_CONFIG',
    'get_heim_config',
    'heim_compress',
    'heim_decompress',
    'ClusterIndex',
    'get_cluster_index',
    'validate_geometry',
    'validate_bundling',
    # Oversample
    'OVERSAMPLE_CONFIG',
    'oversample_rerank',
    'OversampleTuner',
    'get_tuner',
    'get_metrics',
]
