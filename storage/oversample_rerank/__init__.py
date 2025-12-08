"""
Ara Oversample + Rerank - Two-Stage Retrieval Pipeline
======================================================

This module implements the two-stage retrieval that achieves
100× compression with 0% recall loss:

    Stage 1: Coarse retrieval (D=173, 4× oversample) → ~32 candidates
    Stage 2: Full precision rerank (D=16k) → Top-K with perfect recall

Architecture:
    Query HV (16k) → Heim compress (173b) → L2 Index → N×Candidates
                                                           ↓
                         Top-K ← Teleology Rerank ← GPU/FPGA Rescore

Key Parameters:
    - oversample_factor: 2-8× (adaptive)
    - teleology_weight: 0.1-0.5 (adaptive)
    - latency_budget: 500 µs (Stage 2)
"""

from .config import OVERSAMPLE_CONFIG, get_oversample_config
from .retrieval import (
    oversample_rerank,
    RetrievalResult,
    RetrievalCandidate,
)
from .tuner import (
    OversampleTuner,
    get_tuner,
)
from .telemetry import (
    RetrievalMetrics,
    record_retrieval,
    get_metrics,
)

__all__ = [
    # Config
    'OVERSAMPLE_CONFIG',
    'get_oversample_config',
    # Retrieval
    'oversample_rerank',
    'RetrievalResult',
    'RetrievalCandidate',
    # Tuner
    'OversampleTuner',
    'get_tuner',
    # Telemetry
    'RetrievalMetrics',
    'record_retrieval',
    'get_metrics',
]
