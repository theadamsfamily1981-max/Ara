"""
Ara HD (Hyperdimensional) Computing Module
==========================================

Core VSA/HD operations and vocabulary for Ara's sensorium.

This module provides:
- Binary hypervector operations (XOR binding, majority bundling)
- Vocabulary management for roles, features, bins, tags
- Cosine similarity for attractor matching
- Capacity/interference diagnostics
- Runtime health monitoring

Canonical parameters:
- Dimension: D = 16,384 bits
- Representation: {0, 1} binary (internally converted to {-1, +1} for math)
- Binding: XOR (self-inverse, associative)
- Bundling: Majority vote (sum + sign)

Health Contract (HTC-16k):
- Codebook geometry: mean |cos| < 0.02, tail fraction < 1%
- Attractor diversity: mean |cos| < 0.15, no >10% in tight clusters
- Bundling capacity: ≤50 features/moment, signal ≥3σ above noise

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

Health checks:
    from ara.hd import run_full_health_check, get_health_monitor

    report = run_full_health_check()
    assert report.is_healthy, report.summary
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

from .diagnostics import (
    HealthThresholds,
    DEFAULT_THRESHOLDS,
    SoulHealthReport,
    check_codebook_geometry,
    stress_test_bundling,
    check_attractor_diversity,
    run_full_health_check,
)

from .health import (
    SoulHealthMonitor,
    get_health_monitor,
    HealthAlert,
    AlertManager,
    get_alert_manager,
)

from .hv_types import (
    DenseHV,
    SparseHV,
    dense_to_sparse,
    sparse_to_dense,
    sparsify,
    sparse_cosine,
    sparse_bind,
    sparse_bundle,
)

from .projection import (
    HDProjection,
    ProjectionRegistry,
    ORGAN_DIMENSIONS,
    get_projection_registry,
    project_down,
    project_up,
)

from .shards import (
    ShardRole,
    ShardConfig,
    HTCShard,
    SoftwareHTCShard,
    ShardRegistry,
    get_shard_registry,
)

from .fpga_search import (
    SearchResult,
    FPGADevice,
    SimDevice,
    HTCSearchFPGA,
    get_fpga_search,
    query_resonance,
)

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
    # Diagnostics
    'HealthThresholds',
    'DEFAULT_THRESHOLDS',
    'SoulHealthReport',
    'check_codebook_geometry',
    'stress_test_bundling',
    'check_attractor_diversity',
    'run_full_health_check',
    # Health Monitoring
    'SoulHealthMonitor',
    'get_health_monitor',
    'HealthAlert',
    'AlertManager',
    'get_alert_manager',
    # HV Types (Sparse/Dense)
    'DenseHV',
    'SparseHV',
    'dense_to_sparse',
    'sparse_to_dense',
    'sparsify',
    'sparse_cosine',
    'sparse_bind',
    'sparse_bundle',
    # Projections (MicroHD)
    'HDProjection',
    'ProjectionRegistry',
    'ORGAN_DIMENSIONS',
    'get_projection_registry',
    'project_down',
    'project_up',
    # Shards
    'ShardRole',
    'ShardConfig',
    'HTCShard',
    'SoftwareHTCShard',
    'ShardRegistry',
    'get_shard_registry',
    # FPGA Search (Soul CAM)
    'SearchResult',
    'FPGADevice',
    'SimDevice',
    'HTCSearchFPGA',
    'get_fpga_search',
    'query_resonance',
]
