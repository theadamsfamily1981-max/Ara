"""
Ara HTC Retrieval - Unified Search API
======================================

Single entry point for all HTC-based retrieval operations.

The Sovereign Loop calls this module; internally it routes to:
    - Fast path: Direct HTC resonance (for simple attractor lookup)
    - Deep path: Oversample + Rerank (for episodic/long-range recall)
    - Anneal path: Neuromorphic annealing (for constraint satisfaction)

Architecture:
                            htc_retrieval.retrieve(h_moment, mode)
                                        │
                ┌───────────────────────┼───────────────────────┐
                │                       │                       │
                ▼                       ▼                       ▼
           FAST PATH              DEEP PATH              ANNEAL PATH
         (< 1 µs HTC)         (Oversample+Rerank)     (QuantumBridge)
                │                       │                       │
                ▼                       ▼                       ▼
        Top-K attractors      Top-K episodes + teleology    Solution HV
"""

from __future__ import annotations

import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Union
from enum import Enum, auto
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Retrieval Modes
# =============================================================================

class RetrievalMode(Enum):
    """Retrieval mode selection."""
    FAST = auto()       # Direct HTC resonance (~1 µs)
    DEEP = auto()       # Oversample + Rerank (~500 µs)
    ANNEAL = auto()     # Neuromorphic annealing (~ms)
    AUTO = auto()       # Automatic selection based on context


# =============================================================================
# Result Structures
# =============================================================================

@dataclass
class AttractorMatch:
    """A matched attractor from HTC."""
    attractor_id: int
    similarity: float
    teleology_boost: float = 0.0
    final_score: float = 0.0
    label: str = ""

    def __post_init__(self):
        if self.final_score == 0.0:
            self.final_score = self.similarity + self.teleology_boost


@dataclass
class EpisodeMatch:
    """A matched episode from deep retrieval."""
    episode_id: str
    cluster_id: str
    similarity: float
    teleology_score: float
    final_score: float
    reward: float = 0.0
    timestamp: float = 0.0


@dataclass
class RetrievalResult:
    """Unified retrieval result."""
    mode: RetrievalMode
    latency_us: float

    # Fast path results
    attractors: List[AttractorMatch] = field(default_factory=list)

    # Deep path results
    episodes: List[EpisodeMatch] = field(default_factory=list)

    # Anneal path results
    solution_hv: Optional[np.ndarray] = None
    solution_energy: float = 0.0

    # Metadata
    k: int = 8
    oversample_factor: float = 1.0
    details: Dict[str, Any] = field(default_factory=dict)

    def top_attractors(self, n: int = None) -> List[AttractorMatch]:
        """Get top-N attractors by score."""
        if n is None:
            n = self.k
        return sorted(self.attractors, key=lambda a: a.final_score, reverse=True)[:n]

    def top_episodes(self, n: int = None) -> List[EpisodeMatch]:
        """Get top-N episodes by score."""
        if n is None:
            n = self.k
        return sorted(self.episodes, key=lambda e: e.final_score, reverse=True)[:n]

    def best_attractor(self) -> Optional[AttractorMatch]:
        """Get single best attractor."""
        top = self.top_attractors(1)
        return top[0] if top else None

    def best_episode(self) -> Optional[EpisodeMatch]:
        """Get single best episode."""
        top = self.top_episodes(1)
        return top[0] if top else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'mode': self.mode.name,
            'latency_us': self.latency_us,
            'k': self.k,
            'oversample_factor': self.oversample_factor,
            'n_attractors': len(self.attractors),
            'n_episodes': len(self.episodes),
            'top_attractor': self.best_attractor().__dict__ if self.best_attractor() else None,
            'top_episode': self.best_episode().__dict__ if self.best_episode() else None,
        }


# =============================================================================
# HTC Interface (Software Fallback)
# =============================================================================

class SoftwareHTC:
    """
    Software HTC implementation for when FPGA is unavailable.

    Uses numpy for XNOR-popcount simulation.
    """

    def __init__(self, D: int = 16384, R: int = 2048):
        self.D = D
        self.R = R

        # Weight matrix (attractors): R × D bipolar
        self._weights: Optional[np.ndarray] = None
        self._labels: List[str] = []
        self._initialized = False

        # Statistics
        self._queries = 0
        self._total_latency = 0.0

    def initialize(self, weights: np.ndarray = None, labels: List[str] = None) -> None:
        """Initialize with attractor weights."""
        if weights is not None:
            self._weights = weights.astype(np.float32)
            self.R = weights.shape[0]
        else:
            # Random initialization for testing
            rng = np.random.default_rng(42)
            self._weights = rng.choice([-1, 1], size=(self.R, self.D)).astype(np.float32)

        if labels is not None:
            self._labels = labels
        else:
            self._labels = [f"attractor_{i}" for i in range(self.R)]

        self._initialized = True

    def query(
        self,
        h_query: np.ndarray,
        k: int = 8,
    ) -> Tuple[List[int], List[float], float]:
        """
        Query HTC for top-K attractors.

        Args:
            h_query: Query HV (D,)
            k: Number of results

        Returns:
            (top_ids, top_scores, latency_us)
        """
        if not self._initialized:
            self.initialize()

        start = time.perf_counter()
        self._queries += 1

        # Ensure query is correct shape
        h = h_query.astype(np.float32).flatten()
        if len(h) != self.D:
            # Pad or truncate
            if len(h) < self.D:
                h = np.pad(h, (0, self.D - len(h)))
            else:
                h = h[:self.D]

        # Convert to bipolar if binary
        if np.all((h == 0) | (h == 1)):
            h = 2.0 * h - 1.0

        # Compute similarities (dot product for bipolar = XNOR-popcount equivalent)
        similarities = self._weights @ h / self.D  # Normalized to [-1, 1]

        # Get top-K
        top_indices = np.argpartition(-similarities, k)[:k]
        top_indices = top_indices[np.argsort(-similarities[top_indices])]
        top_scores = similarities[top_indices].tolist()

        end = time.perf_counter()
        latency_us = (end - start) * 1e6
        self._total_latency += latency_us

        return top_indices.tolist(), top_scores, latency_us

    def get_label(self, attractor_id: int) -> str:
        """Get label for an attractor."""
        if 0 <= attractor_id < len(self._labels):
            return self._labels[attractor_id]
        return f"unknown_{attractor_id}"

    def get_stats(self) -> Dict[str, Any]:
        return {
            'D': self.D,
            'R': self.R,
            'queries': self._queries,
            'avg_latency_us': self._total_latency / max(self._queries, 1),
            'initialized': self._initialized,
        }


# =============================================================================
# Retrieval Engine
# =============================================================================

class HTCRetrieval:
    """
    Unified HTC retrieval engine.

    Provides a single interface for all retrieval operations.
    """

    def __init__(self):
        # HTC backend
        self._htc: Optional[SoftwareHTC] = None
        self._fpga_htc = None  # Will be set if FPGA available

        # Deep retrieval (lazy import to avoid circular deps)
        self._oversample_rerank = None
        self._cluster_index = None

        # Anneal path
        self._quantum_bridge = None

        # Statistics
        self._fast_queries = 0
        self._deep_queries = 0
        self._anneal_queries = 0

    def _ensure_htc(self) -> SoftwareHTC:
        """Ensure HTC is initialized."""
        if self._htc is None:
            self._htc = SoftwareHTC()
            self._htc.initialize()
        return self._htc

    def _get_oversample_rerank(self):
        """Lazy import oversample_rerank."""
        if self._oversample_rerank is None:
            try:
                from storage.oversample_rerank import oversample_rerank
                self._oversample_rerank = oversample_rerank
            except ImportError:
                pass
        return self._oversample_rerank

    def _get_quantum_bridge(self):
        """Lazy import quantum_bridge."""
        if self._quantum_bridge is None:
            try:
                from .quantum_bridge import QuantumBridgeSolver
                self._quantum_bridge = QuantumBridgeSolver()
            except ImportError:
                pass
        return self._quantum_bridge

    # =========================================================================
    # Main API
    # =========================================================================

    def retrieve(
        self,
        h_query: np.ndarray,
        mode: RetrievalMode = RetrievalMode.AUTO,
        k: int = 8,
        oversample_factor: float = None,
        teleology_context: Optional[Dict] = None,
    ) -> RetrievalResult:
        """
        Unified retrieval entry point.

        Args:
            h_query: Query HV (any dimension, will be adapted)
            mode: Retrieval mode (FAST, DEEP, ANNEAL, AUTO)
            k: Number of results
            oversample_factor: For DEEP mode
            teleology_context: Context for teleology scoring

        Returns:
            RetrievalResult with matched attractors/episodes
        """
        # Auto-select mode based on context
        if mode == RetrievalMode.AUTO:
            mode = self._auto_select_mode(h_query, teleology_context)

        # Dispatch to appropriate path
        if mode == RetrievalMode.FAST:
            return self._fast_retrieve(h_query, k, teleology_context)
        elif mode == RetrievalMode.DEEP:
            return self._deep_retrieve(h_query, k, oversample_factor, teleology_context)
        elif mode == RetrievalMode.ANNEAL:
            return self._anneal_retrieve(h_query, teleology_context)
        else:
            return self._fast_retrieve(h_query, k, teleology_context)

    def _auto_select_mode(
        self,
        h_query: np.ndarray,
        context: Optional[Dict],
    ) -> RetrievalMode:
        """
        Auto-select retrieval mode based on context.

        Heuristics:
            - If context requests "deep recall" → DEEP
            - If context has "constraint" field → ANNEAL
            - Default → FAST
        """
        if context is None:
            return RetrievalMode.FAST

        if context.get('deep_recall', False):
            return RetrievalMode.DEEP

        if context.get('constraint', False):
            return RetrievalMode.ANNEAL

        return RetrievalMode.FAST

    # =========================================================================
    # Fast Path
    # =========================================================================

    def _fast_retrieve(
        self,
        h_query: np.ndarray,
        k: int,
        teleology_context: Optional[Dict],
    ) -> RetrievalResult:
        """
        Fast path: Direct HTC resonance search.

        ~1 µs latency (software), ~100 ns (FPGA)
        """
        start = time.perf_counter()
        self._fast_queries += 1

        htc = self._ensure_htc()
        top_ids, top_scores, htc_latency = htc.query(h_query, k=k)

        # Build attractor matches
        attractors = []
        for idx, (attractor_id, score) in enumerate(zip(top_ids, top_scores)):
            # Teleology boost (if context provided)
            teleo_boost = 0.0
            if teleology_context and 'reward_weights' in teleology_context:
                # Simple boost based on attractor index (placeholder)
                teleo_boost = teleology_context['reward_weights'].get(attractor_id, 0.0) * 0.1

            match = AttractorMatch(
                attractor_id=attractor_id,
                similarity=score,
                teleology_boost=teleo_boost,
                label=htc.get_label(attractor_id),
            )
            attractors.append(match)

        end = time.perf_counter()
        total_latency = (end - start) * 1e6

        return RetrievalResult(
            mode=RetrievalMode.FAST,
            latency_us=total_latency,
            attractors=attractors,
            k=k,
            details={'htc_latency_us': htc_latency},
        )

    # =========================================================================
    # Deep Path
    # =========================================================================

    def _deep_retrieve(
        self,
        h_query: np.ndarray,
        k: int,
        oversample_factor: Optional[float],
        teleology_context: Optional[Dict],
    ) -> RetrievalResult:
        """
        Deep path: Oversample + Rerank for episodic recall.

        ~500 µs latency with 99.9% recall
        """
        start = time.perf_counter()
        self._deep_queries += 1

        oversample_func = self._get_oversample_rerank()

        if oversample_func is None:
            # Fallback to fast path
            logger.warning("Deep retrieval unavailable, falling back to fast path")
            return self._fast_retrieve(h_query, k, teleology_context)

        # Call oversample + rerank
        if oversample_factor is None:
            oversample_factor = 4.0

        teleo_weight = 0.2
        if teleology_context and 'teleo_weight' in teleology_context:
            teleo_weight = teleology_context['teleo_weight']

        result = oversample_func(
            h_query_full=h_query,
            k=k,
            oversample_factor=oversample_factor,
            teleo_weight=teleo_weight,
        )

        # Convert to EpisodeMatch
        episodes = []
        for cand in result.candidates[:k]:
            match = EpisodeMatch(
                episode_id=cand.episode_id,
                cluster_id=cand.cluster_id,
                similarity=cand.full_score,
                teleology_score=cand.teleology_score,
                final_score=cand.final_score,
            )
            episodes.append(match)

        end = time.perf_counter()
        total_latency = (end - start) * 1e6

        return RetrievalResult(
            mode=RetrievalMode.DEEP,
            latency_us=total_latency,
            episodes=episodes,
            k=k,
            oversample_factor=oversample_factor,
            details={
                'stage1_latency_us': result.stage1_latency_us,
                'stage2_latency_us': result.stage2_latency_us,
                'coarse_count': result.coarse_count,
            },
        )

    # =========================================================================
    # Anneal Path
    # =========================================================================

    def _anneal_retrieve(
        self,
        h_query: np.ndarray,
        teleology_context: Optional[Dict],
    ) -> RetrievalResult:
        """
        Anneal path: Neuromorphic annealing for constraint satisfaction.

        Used when the query encodes a constraint problem.
        """
        start = time.perf_counter()
        self._anneal_queries += 1

        quantum_bridge = self._get_quantum_bridge()

        if quantum_bridge is None:
            logger.warning("Anneal path unavailable")
            return RetrievalResult(
                mode=RetrievalMode.ANNEAL,
                latency_us=(time.perf_counter() - start) * 1e6,
                details={'error': 'quantum_bridge_unavailable'},
            )

        # Extract constraint from context
        constraint = teleology_context.get('constraint', {}) if teleology_context else {}

        # Run annealing (simplified)
        try:
            solution = quantum_bridge.solve_from_hv(h_query, constraint)
            solution_hv = solution.get('hv', None)
            energy = solution.get('energy', 0.0)
        except Exception as e:
            logger.error(f"Annealing failed: {e}")
            solution_hv = None
            energy = float('inf')

        end = time.perf_counter()
        total_latency = (end - start) * 1e6

        return RetrievalResult(
            mode=RetrievalMode.ANNEAL,
            latency_us=total_latency,
            solution_hv=solution_hv,
            solution_energy=energy,
            details={'constraint': constraint},
        )

    # =========================================================================
    # Configuration
    # =========================================================================

    def set_fpga_htc(self, fpga_htc) -> None:
        """Set FPGA HTC backend."""
        self._fpga_htc = fpga_htc

    def set_htc_weights(self, weights: np.ndarray, labels: List[str] = None) -> None:
        """Set HTC attractor weights."""
        htc = self._ensure_htc()
        htc.initialize(weights, labels)

    def get_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics."""
        stats = {
            'fast_queries': self._fast_queries,
            'deep_queries': self._deep_queries,
            'anneal_queries': self._anneal_queries,
            'total_queries': self._fast_queries + self._deep_queries + self._anneal_queries,
        }

        if self._htc:
            stats['htc'] = self._htc.get_stats()

        return stats


# =============================================================================
# Singleton
# =============================================================================

_htc_retrieval: Optional[HTCRetrieval] = None


def get_htc_retrieval() -> HTCRetrieval:
    """Get the global HTC retrieval instance."""
    global _htc_retrieval
    if _htc_retrieval is None:
        _htc_retrieval = HTCRetrieval()
    return _htc_retrieval


def retrieve(
    h_query: np.ndarray,
    mode: RetrievalMode = RetrievalMode.AUTO,
    k: int = 8,
    **kwargs,
) -> RetrievalResult:
    """
    Convenience function for retrieval.

    This is the main entry point for Sovereign Loop.
    """
    return get_htc_retrieval().retrieve(h_query, mode, k, **kwargs)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'RetrievalMode',
    'AttractorMatch',
    'EpisodeMatch',
    'RetrievalResult',
    'HTCRetrieval',
    'SoftwareHTC',
    'get_htc_retrieval',
    'retrieve',
]
