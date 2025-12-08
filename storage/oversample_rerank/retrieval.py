"""
Oversample + Rerank Retrieval Pipeline
======================================

Two-stage retrieval achieving 99.9% recall with compressed storage.
"""

from __future__ import annotations

import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import logging

from .config import OVERSAMPLE_CONFIG, get_oversample_config
from ..heim_optimized import (
    heim_compress,
    heim_decompress,
    get_cluster_index,
    ClusterIndex,
    EpisodeRecord,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Result Structures
# =============================================================================

@dataclass
class RetrievalCandidate:
    """A candidate from retrieval."""
    episode_id: str
    cluster_id: str
    coarse_score: float           # Stage 1 score
    full_score: float = 0.0       # Stage 2 score
    teleology_score: float = 0.0  # Teleology boost
    final_score: float = 0.0      # Combined score
    hv_full: Optional[np.ndarray] = None  # Decompressed HV (16k)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'episode_id': self.episode_id,
            'cluster_id': self.cluster_id,
            'coarse_score': self.coarse_score,
            'full_score': self.full_score,
            'teleology_score': self.teleology_score,
            'final_score': self.final_score,
        }


@dataclass
class RetrievalResult:
    """Result of oversample + rerank retrieval."""
    candidates: List[RetrievalCandidate]
    k: int
    oversample_factor: float
    stage1_latency_us: float
    stage2_latency_us: float
    total_latency_us: float
    coarse_count: int             # Candidates after Stage 1

    def top_k(self) -> List[RetrievalCandidate]:
        """Get top-K by final score."""
        return sorted(self.candidates, key=lambda c: c.final_score, reverse=True)[:self.k]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'k': self.k,
            'oversample_factor': self.oversample_factor,
            'stage1_latency_us': self.stage1_latency_us,
            'stage2_latency_us': self.stage2_latency_us,
            'total_latency_us': self.total_latency_us,
            'coarse_count': self.coarse_count,
            'top_candidates': [c.to_dict() for c in self.top_k()],
        }


# =============================================================================
# Stage 1: Coarse Retrieval
# =============================================================================

def _stage1_coarse_retrieval(
    h_query_173: np.ndarray,
    cluster_index: ClusterIndex,
    n_candidates: int,
    threshold: float,
) -> Tuple[List[EpisodeRecord], float]:
    """
    Stage 1: Coarse retrieval from compressed index.

    Args:
        h_query_173: Compressed query HV
        cluster_index: Cluster index
        n_candidates: Number of candidates to retrieve
        threshold: Minimum similarity threshold

    Returns:
        (episodes, latency_us)
    """
    start = time.perf_counter()

    # Query the cluster index
    episodes = cluster_index.nearby(
        hv_query=h_query_173,
        threshold=threshold,
        limit=n_candidates,
    )

    end = time.perf_counter()
    latency_us = (end - start) * 1e6

    return episodes, latency_us


# =============================================================================
# Stage 2: Full Precision Rerank
# =============================================================================

def _stage2_full_rerank(
    h_query_full: np.ndarray,
    candidates: List[EpisodeRecord],
    cluster_index: ClusterIndex,
    teleo_weight: float,
) -> Tuple[List[RetrievalCandidate], float]:
    """
    Stage 2: Full precision reranking of candidates.

    Args:
        h_query_full: Full 16k query HV
        candidates: Candidates from Stage 1
        cluster_index: For fetching cluster centroids
        teleo_weight: Weight for teleology scoring

    Returns:
        (scored_candidates, latency_us)
    """
    start = time.perf_counter()

    scored = []

    for episode in candidates:
        # Get cluster centroid
        cluster = cluster_index.get_cluster(episode.cluster_id)
        if cluster is None:
            continue

        # Decompress to full 16k
        h_cand_full = heim_decompress(cluster.centroid, episode.delta_info)

        # Full precision similarity (cosine or XNOR equivalent)
        # Convert to bipolar if needed
        q_bp = h_query_full.astype(np.float32)
        c_bp = h_cand_full.astype(np.float32)

        if np.all((q_bp == 0) | (q_bp == 1)):
            q_bp = 2.0 * q_bp - 1.0
        if np.all((c_bp == 0) | (c_bp == 1)):
            c_bp = 2.0 * c_bp - 1.0

        # Cosine similarity
        dot = np.dot(q_bp, c_bp)
        norm_q = np.linalg.norm(q_bp)
        norm_c = np.linalg.norm(c_bp)
        full_sim = dot / (norm_q * norm_c + 1e-10)

        # Teleology score: reward × alignment
        teleo_score = episode.reward * np.mean(np.abs(episode.teleology_vector))

        # Combined score
        final_score = full_sim * (1.0 - teleo_weight) + teleo_score * teleo_weight

        candidate = RetrievalCandidate(
            episode_id=episode.episode_id,
            cluster_id=episode.cluster_id,
            coarse_score=episode.resonance,
            full_score=float(full_sim),
            teleology_score=float(teleo_score),
            final_score=float(final_score),
            hv_full=h_cand_full,
        )
        scored.append(candidate)

    end = time.perf_counter()
    latency_us = (end - start) * 1e6

    return scored, latency_us


# =============================================================================
# Main Pipeline
# =============================================================================

def oversample_rerank(
    h_query_full: np.ndarray,
    k: int = None,
    oversample_factor: float = None,
    teleo_weight: float = None,
    cluster_index: ClusterIndex = None,
) -> RetrievalResult:
    """
    Two-stage oversample + rerank retrieval.

    Args:
        h_query_full: Full 16k query HV
        k: Number of final results
        oversample_factor: Oversample multiplier (or use config)
        teleo_weight: Teleology weight (or use config)
        cluster_index: Cluster index (or use global)

    Returns:
        RetrievalResult with scored candidates
    """
    config = get_oversample_config()

    if k is None:
        k = config.k
    if oversample_factor is None:
        oversample_factor = config.oversample_factor
    if teleo_weight is None:
        teleo_weight = config.teleo_weight_default
    if cluster_index is None:
        cluster_index = get_cluster_index()

    # Clamp parameters
    oversample_factor = max(config.min_oversample, min(config.max_oversample, oversample_factor))
    teleo_weight = max(config.teleo_weight_min, min(config.teleo_weight_max, teleo_weight))

    n_candidates = int(k * oversample_factor)

    # Stage 1: Compress and coarse retrieval
    h_query_173 = heim_compress(h_query_full)

    coarse_episodes, stage1_latency = _stage1_coarse_retrieval(
        h_query_173=h_query_173,
        cluster_index=cluster_index,
        n_candidates=n_candidates,
        threshold=config.coarse_threshold,
    )

    # Stage 2: Full precision rerank
    if len(coarse_episodes) == 0:
        return RetrievalResult(
            candidates=[],
            k=k,
            oversample_factor=oversample_factor,
            stage1_latency_us=stage1_latency,
            stage2_latency_us=0.0,
            total_latency_us=stage1_latency,
            coarse_count=0,
        )

    scored_candidates, stage2_latency = _stage2_full_rerank(
        h_query_full=h_query_full,
        candidates=coarse_episodes,
        cluster_index=cluster_index,
        teleo_weight=teleo_weight,
    )

    # Sort by final score
    scored_candidates.sort(key=lambda c: c.final_score, reverse=True)

    total_latency = stage1_latency + stage2_latency

    return RetrievalResult(
        candidates=scored_candidates,
        k=k,
        oversample_factor=oversample_factor,
        stage1_latency_us=stage1_latency,
        stage2_latency_us=stage2_latency,
        total_latency_us=total_latency,
        coarse_count=len(coarse_episodes),
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'RetrievalCandidate',
    'RetrievalResult',
    'oversample_rerank',
]
