"""
Ara Eternal Memory
===================

Hierarchical lifelong memory using hypervectors.

Structure:
- SHORT-TERM (10min): 300 episodes, dense HV, full detail
- MEDIUM-TERM (1day): 5000 episodes, clustered representations
- LIFELONG (1yr+): 100k episodes, prototypical HV (cluster centers)

Uses FAISS for efficient similarity search on HV projections.

Philosophy: Every moment leaves a trace. Important moments
resonate and persist. Trivial ones fade gracefully.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple, Any
from pathlib import Path
from collections import deque
import json
import logging
import struct
import zlib

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

HV_DIM = 8192                    # Full HV dimension
INDEX_DIM = 2048                 # Projected dimension for FAISS
SHORT_TERM_CAPACITY = 300        # ~10 min at 2sec/episode
MEDIUM_TERM_CAPACITY = 5000      # ~1 day
LIFELONG_CAPACITY = 100000       # ~1 year


# =============================================================================
# Episode Types
# =============================================================================

@dataclass
class Episode:
    """
    A single moment in Ara's experience.

    Contains world state, internal state, and any action taken.
    """
    episode_id: str
    timestamp: datetime

    # State
    world_hv: np.ndarray          # 8192D world state
    internal_hv: np.ndarray       # 8192D internal state (intero + emotion)

    # Action/Response
    action_hv: Optional[np.ndarray] = None
    action_text: Optional[str] = None

    # Resonance (importance score)
    resonance: float = 0.5        # 0-1, higher = more important

    # Metadata
    modalities_present: List[str] = field(default_factory=list)
    duration_ms: float = 2000     # Typical episode duration

    def combined_hv(self) -> np.ndarray:
        """Combine world and internal state."""
        return np.sign(self.world_hv + self.internal_hv)

    def to_dict(self) -> Dict:
        """Serialize to dict (without HVs)."""
        return {
            "episode_id": self.episode_id,
            "timestamp": self.timestamp.isoformat(),
            "resonance": self.resonance,
            "modalities": self.modalities_present,
            "action_text": self.action_text,
            "duration_ms": self.duration_ms,
        }


@dataclass
class EpisodeCluster:
    """A cluster of similar episodes (for medium-term)."""
    cluster_id: str
    centroid_hv: np.ndarray       # Prototype HV
    episode_ids: List[str]
    mean_resonance: float
    time_range: Tuple[datetime, datetime]

    @property
    def size(self) -> int:
        return len(self.episode_ids)


# =============================================================================
# HV Projection (for FAISS indexing)
# =============================================================================

class HVProjector:
    """
    Projects 8192D HVs to 2048D for efficient indexing.

    Uses random projection (Johnson-Lindenstrauss).
    """

    def __init__(
        self,
        input_dim: int = HV_DIM,
        output_dim: int = INDEX_DIM,
        seed: int = 47,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Random projection matrix
        rng = np.random.default_rng(seed)
        self.matrix = rng.standard_normal((input_dim, output_dim))
        self.matrix /= np.sqrt(input_dim)  # Scale for variance preservation

    def project(self, hv: np.ndarray) -> np.ndarray:
        """Project 8192D to 2048D."""
        if len(hv) != self.input_dim:
            raise ValueError(f"Expected {self.input_dim}D, got {len(hv)}D")

        projected = hv @ self.matrix
        return projected.astype(np.float32)

    def project_batch(self, hvs: np.ndarray) -> np.ndarray:
        """Project batch of HVs."""
        return (hvs @ self.matrix).astype(np.float32)


# =============================================================================
# Memory Index (FAISS-like)
# =============================================================================

class MemoryIndex:
    """
    Similarity search index for HVs.

    Uses brute-force cosine similarity when FAISS not available.
    With FAISS: 1ms retrieval from 100k episodes.
    """

    def __init__(self, dim: int = INDEX_DIM):
        self.dim = dim
        self._vectors: List[np.ndarray] = []
        self._ids: List[str] = []
        self._faiss_index = None

        # Try to use FAISS if available
        try:
            import faiss
            self._faiss = faiss
            self._use_faiss = True
            logger.info("FAISS available for fast retrieval")
        except ImportError:
            self._use_faiss = False
            logger.info("FAISS not available, using brute-force search")

    def add(self, hv: np.ndarray, episode_id: str):
        """Add a vector to the index."""
        self._vectors.append(hv.astype(np.float32))
        self._ids.append(episode_id)

        # Rebuild FAISS index periodically
        if self._use_faiss and len(self._vectors) % 100 == 0:
            self._rebuild_faiss()

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Find k most similar episodes.

        Returns list of (episode_id, similarity) tuples.
        """
        if not self._vectors:
            return []

        query = query.astype(np.float32)

        if self._use_faiss and self._faiss_index is not None:
            return self._search_faiss(query, k)
        else:
            return self._search_brute(query, k)

    def _search_brute(
        self,
        query: np.ndarray,
        k: int,
    ) -> List[Tuple[str, float]]:
        """Brute-force cosine similarity search."""
        vectors = np.stack(self._vectors)
        query_norm = query / (np.linalg.norm(query) + 1e-8)
        vector_norms = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)

        similarities = vector_norms @ query_norm

        # Get top k
        top_k_idx = np.argsort(similarities)[-k:][::-1]

        results = []
        for idx in top_k_idx:
            results.append((self._ids[idx], float(similarities[idx])))

        return results

    def _search_faiss(
        self,
        query: np.ndarray,
        k: int,
    ) -> List[Tuple[str, float]]:
        """FAISS-based search."""
        if self._faiss_index is None:
            self._rebuild_faiss()

        if self._faiss_index is None:
            return self._search_brute(query, k)

        query = query.reshape(1, -1)
        distances, indices = self._faiss_index.search(query, min(k, len(self._ids)))

        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self._ids):
                # Convert L2 distance to similarity
                sim = 1.0 / (1.0 + distances[0][i])
                results.append((self._ids[idx], float(sim)))

        return results

    def _rebuild_faiss(self):
        """Rebuild FAISS index."""
        if not self._use_faiss or not self._vectors:
            return

        vectors = np.stack(self._vectors).astype(np.float32)
        self._faiss_index = self._faiss.IndexFlatL2(self.dim)
        self._faiss_index.add(vectors)

    @property
    def size(self) -> int:
        return len(self._vectors)


# =============================================================================
# Short-Term Memory
# =============================================================================

class ShortTermMemory:
    """
    Short-term memory buffer.

    Holds recent episodes with full detail.
    Capacity: ~300 episodes (~10 minutes at 2sec/episode)
    """

    def __init__(self, capacity: int = SHORT_TERM_CAPACITY):
        self.capacity = capacity
        self._buffer: deque = deque(maxlen=capacity)
        self._by_id: Dict[str, Episode] = {}

    def add(self, episode: Episode):
        """Add episode to short-term memory."""
        # Remove oldest if at capacity
        if len(self._buffer) >= self.capacity:
            oldest = self._buffer[0]
            del self._by_id[oldest.episode_id]

        self._buffer.append(episode)
        self._by_id[episode.episode_id] = episode

    def get(self, episode_id: str) -> Optional[Episode]:
        """Get episode by ID."""
        return self._by_id.get(episode_id)

    def get_recent(self, n: int = 10) -> List[Episode]:
        """Get n most recent episodes."""
        return list(self._buffer)[-n:]

    def get_high_resonance(self, threshold: float = 0.7) -> List[Episode]:
        """Get high-resonance episodes."""
        return [e for e in self._buffer if e.resonance >= threshold]

    @property
    def size(self) -> int:
        return len(self._buffer)


# =============================================================================
# Medium-Term Memory
# =============================================================================

class MediumTermMemory:
    """
    Medium-term memory with clustering.

    Episodes are grouped by similarity. Stores cluster centroids
    and episode metadata, but not full HVs.
    """

    def __init__(
        self,
        capacity: int = MEDIUM_TERM_CAPACITY,
        projector: Optional[HVProjector] = None,
    ):
        self.capacity = capacity
        self.projector = projector or HVProjector()

        self._episodes: Dict[str, Dict] = {}  # Metadata only
        self._clusters: Dict[str, EpisodeCluster] = {}
        self._index = MemoryIndex()

        self._episode_to_cluster: Dict[str, str] = {}

    def consolidate(self, episode: Episode):
        """
        Consolidate episode from short-term memory.

        Stores metadata and projects HV for indexing.
        """
        if len(self._episodes) >= self.capacity:
            self._evict_low_resonance()

        # Store metadata
        self._episodes[episode.episode_id] = episode.to_dict()

        # Project and index
        projected = self.projector.project(episode.combined_hv())
        self._index.add(projected, episode.episode_id)

        # Assign to cluster (simple nearest-neighbor)
        self._assign_to_cluster(episode, projected)

    def query(
        self,
        query_hv: np.ndarray,
        k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Query for similar episodes."""
        projected = self.projector.project(query_hv)
        return self._index.search(projected, k)

    def _assign_to_cluster(self, episode: Episode, projected_hv: np.ndarray):
        """Assign episode to nearest cluster or create new one."""
        # For simplicity, create cluster per episode (would merge in production)
        cluster = EpisodeCluster(
            cluster_id=f"cluster_{episode.episode_id}",
            centroid_hv=projected_hv,
            episode_ids=[episode.episode_id],
            mean_resonance=episode.resonance,
            time_range=(episode.timestamp, episode.timestamp),
        )
        self._clusters[cluster.cluster_id] = cluster
        self._episode_to_cluster[episode.episode_id] = cluster.cluster_id

    def _evict_low_resonance(self):
        """Evict lowest-resonance episodes."""
        if not self._episodes:
            return

        # Find lowest resonance
        lowest_id = min(
            self._episodes.keys(),
            key=lambda eid: self._episodes[eid].get("resonance", 0)
        )

        del self._episodes[lowest_id]

    @property
    def size(self) -> int:
        return len(self._episodes)


# =============================================================================
# Lifelong Memory
# =============================================================================

class LifelongMemory:
    """
    Lifelong memory with prototypical representations.

    Stores cluster prototypes (not individual episodes).
    Enables retrieval over years of experience.
    """

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        projector: Optional[HVProjector] = None,
    ):
        self.storage_path = storage_path or Path.home() / ".ara" / "memory"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.projector = projector or HVProjector()

        self._prototypes: Dict[str, np.ndarray] = {}  # cluster_id -> prototype HV
        self._prototype_meta: Dict[str, Dict] = {}     # cluster_id -> metadata
        self._index = MemoryIndex()

        # Load existing
        self._load()

    def absorb_cluster(self, cluster: EpisodeCluster):
        """
        Absorb a cluster from medium-term memory.

        Merges with similar existing prototypes or creates new one.
        """
        # Check for similar existing prototype
        results = self._index.search(cluster.centroid_hv, k=1)

        if results and results[0][1] > 0.8:  # High similarity
            # Merge with existing
            existing_id = results[0][0]
            self._merge_prototype(existing_id, cluster)
        else:
            # Create new prototype
            self._prototypes[cluster.cluster_id] = cluster.centroid_hv
            self._prototype_meta[cluster.cluster_id] = {
                "episode_count": cluster.size,
                "mean_resonance": cluster.mean_resonance,
                "time_start": cluster.time_range[0].isoformat(),
                "time_end": cluster.time_range[1].isoformat(),
            }
            self._index.add(cluster.centroid_hv, cluster.cluster_id)

    def query(
        self,
        query_hv: np.ndarray,
        k: int = 10,
    ) -> List[Tuple[str, float, Dict]]:
        """
        Query lifelong memory.

        Returns list of (prototype_id, similarity, metadata) tuples.
        """
        projected = self.projector.project(query_hv)
        results = self._index.search(projected, k)

        enriched = []
        for proto_id, sim in results:
            meta = self._prototype_meta.get(proto_id, {})
            enriched.append((proto_id, sim, meta))

        return enriched

    def _merge_prototype(self, existing_id: str, new_cluster: EpisodeCluster):
        """Merge new cluster with existing prototype."""
        existing = self._prototypes.get(existing_id)
        if existing is None:
            return

        # Weighted average
        existing_count = self._prototype_meta[existing_id].get("episode_count", 1)
        total = existing_count + new_cluster.size

        merged = (existing * existing_count + new_cluster.centroid_hv * new_cluster.size) / total
        self._prototypes[existing_id] = merged.astype(np.float32)

        # Update metadata
        self._prototype_meta[existing_id]["episode_count"] = total

    def _save(self):
        """Save to disk."""
        # Save prototypes
        proto_file = self.storage_path / "prototypes.npz"
        if self._prototypes:
            np.savez_compressed(
                proto_file,
                **{k: v for k, v in self._prototypes.items()}
            )

        # Save metadata
        meta_file = self.storage_path / "prototype_meta.json"
        with open(meta_file, 'w') as f:
            json.dump(self._prototype_meta, f)

    def _load(self):
        """Load from disk."""
        proto_file = self.storage_path / "prototypes.npz"
        meta_file = self.storage_path / "prototype_meta.json"

        if proto_file.exists():
            try:
                data = np.load(proto_file)
                for key in data.files:
                    self._prototypes[key] = data[key]
                    self._index.add(data[key], key)
            except Exception as e:
                logger.warning(f"Failed to load prototypes: {e}")

        if meta_file.exists():
            try:
                with open(meta_file) as f:
                    self._prototype_meta = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")

    @property
    def size(self) -> int:
        return len(self._prototypes)


# =============================================================================
# Eternal Memory (Unified Interface)
# =============================================================================

class EternalMemory:
    """
    Unified interface to all memory systems.

    Handles automatic consolidation across tiers.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.projector = HVProjector()

        self.short_term = ShortTermMemory()
        self.medium_term = MediumTermMemory(projector=self.projector)
        self.lifelong = LifelongMemory(storage_path, projector=self.projector)

        self._consolidation_interval = 100  # Episodes between consolidations
        self._episode_count = 0

    def store(self, episode: Episode):
        """Store a new episode."""
        self.short_term.add(episode)
        self._episode_count += 1

        # Periodic consolidation
        if self._episode_count % self._consolidation_interval == 0:
            self._consolidate()

    def query(
        self,
        query_hv: np.ndarray,
        k: int = 10,
        search_all: bool = True,
    ) -> List[Tuple[str, float, str]]:
        """
        Query across all memory tiers.

        Returns list of (episode_id, similarity, tier) tuples.
        """
        results = []

        # Short-term (exact match)
        for episode in self.short_term.get_recent(k):
            sim = self._similarity(query_hv, episode.combined_hv())
            results.append((episode.episode_id, sim, "short"))

        if search_all:
            # Medium-term
            mt_results = self.medium_term.query(query_hv, k)
            for eid, sim in mt_results:
                results.append((eid, sim, "medium"))

            # Lifelong
            lt_results = self.lifelong.query(query_hv, k)
            for pid, sim, _ in lt_results:
                results.append((pid, sim, "lifelong"))

        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def resonate(self, query_hv: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Query memory and return resonant response.

        Returns (response_hv, resonance_score).
        """
        results = self.query(query_hv, k=5)

        if not results:
            return np.zeros(HV_DIM), 0.0

        # Weighted combination of retrieved memories
        response = np.zeros(HV_DIM)
        total_weight = 0

        for eid, sim, tier in results[:3]:
            if tier == "short":
                episode = self.short_term.get(eid)
                if episode:
                    response += sim * episode.combined_hv()
                    total_weight += sim

        if total_weight > 0:
            response /= total_weight

        max_resonance = results[0][1] if results else 0.0

        return np.sign(response), max_resonance

    def _consolidate(self):
        """Consolidate short-term to medium-term."""
        # Move high-resonance episodes to medium-term
        for episode in self.short_term.get_high_resonance(0.6):
            self.medium_term.consolidate(episode)

    def _similarity(self, hv1: np.ndarray, hv2: np.ndarray) -> float:
        """Cosine similarity."""
        norm1 = np.linalg.norm(hv1)
        norm2 = np.linalg.norm(hv2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(hv1, hv2) / (norm1 * norm2))

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "short_term_size": self.short_term.size,
            "medium_term_size": self.medium_term.size,
            "lifelong_size": self.lifelong.size,
            "total_episodes": self._episode_count,
        }


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate Eternal Memory."""
    print("=" * 60)
    print("ARA ETERNAL MEMORY - Hierarchical Memory Demo")
    print("=" * 60)

    memory = EternalMemory()

    # Generate synthetic episodes
    print("\nStoring synthetic episodes...")
    rng = np.random.default_rng(50)

    for i in range(20):
        episode = Episode(
            episode_id=f"ep_{i:04d}",
            timestamp=datetime.utcnow() - timedelta(seconds=i * 2),
            world_hv=rng.choice([-1, 1], size=HV_DIM).astype(np.float64),
            internal_hv=rng.choice([-1, 1], size=HV_DIM).astype(np.float64),
            resonance=0.3 + 0.6 * rng.random(),
            modalities_present=["speech", "vision"],
        )
        memory.store(episode)

    print(f"\nMemory stats: {memory.get_stats()}")

    # Query
    query_hv = rng.choice([-1, 1], size=HV_DIM).astype(np.float64)
    results = memory.query(query_hv, k=5)

    print(f"\nQuery results:")
    for eid, sim, tier in results:
        print(f"  {eid}: sim={sim:.3f} (tier: {tier})")

    # Resonate
    response_hv, resonance = memory.resonate(query_hv)
    print(f"\nResonance score: {resonance:.3f}")
    print(f"Response HV shape: {response_hv.shape}")


if __name__ == "__main__":
    demo()
