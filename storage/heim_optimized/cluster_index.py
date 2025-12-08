"""
Heim Cluster Index - Semantic Deduplication & Delta Storage
===========================================================

Episodes are stored as:
    - Cluster centroid (173 bits, shared)
    - Delta link (tiny correction, per-episode)
    - Metadata (reward, teleology, timestamp)

This achieves ~100× compression through semantic clustering.
"""

from __future__ import annotations

import numpy as np
import time
import uuid
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path
import logging

from .config import HEIM_CONFIG
from .encoder import hv_hamming_sim, hv_xnor_popcount


logger = logging.getLogger(__name__)


# =============================================================================
# Cluster Data Structures
# =============================================================================

@dataclass
class Cluster:
    """A cluster centroid with metadata."""
    id: str
    centroid: np.ndarray          # D_compressed binary
    count: int = 1                 # Episodes in this cluster
    created: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'centroid': self.centroid.tolist(),
            'count': self.count,
            'created': self.created,
            'last_updated': self.last_updated,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Cluster':
        return cls(
            id=d['id'],
            centroid=np.array(d['centroid'], dtype=np.uint8),
            count=d.get('count', 1),
            created=d.get('created', time.time()),
            last_updated=d.get('last_updated', time.time()),
        )


@dataclass
class EpisodeRecord:
    """Record linking an episode to its cluster."""
    episode_id: str
    cluster_id: str
    delta_info: np.ndarray        # Sparse corrections
    reward: float
    teleology_vector: np.ndarray  # Teleology alignment
    timestamp: float = field(default_factory=time.time)
    resonance: float = 0.0        # Last resonance score

    def to_dict(self) -> Dict[str, Any]:
        return {
            'episode_id': self.episode_id,
            'cluster_id': self.cluster_id,
            'delta_info': self.delta_info.tolist(),
            'reward': self.reward,
            'teleology_vector': self.teleology_vector.tolist(),
            'timestamp': self.timestamp,
            'resonance': self.resonance,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'EpisodeRecord':
        return cls(
            episode_id=d['episode_id'],
            cluster_id=d['cluster_id'],
            delta_info=np.array(d.get('delta_info', []), dtype=np.float32),
            reward=d.get('reward', 0.0),
            teleology_vector=np.array(d.get('teleology_vector', []), dtype=np.float32),
            timestamp=d.get('timestamp', time.time()),
            resonance=d.get('resonance', 0.0),
        )


# =============================================================================
# Cluster Index
# =============================================================================

class ClusterIndex:
    """
    Semantic cluster index for Heim-compressed episodes.

    Provides:
        - O(n) nearest centroid lookup (or O(log n) with tree)
        - Cluster assignment with automatic centroid updates
        - Near-duplicate detection
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        merge_threshold: float = None,
        duplicate_threshold: float = None,
    ):
        """
        Initialize cluster index.

        Args:
            db_path: Path to persistent storage (None = in-memory only)
            merge_threshold: Similarity for cluster merge
            duplicate_threshold: Similarity for near-duplicate
        """
        self.db_path = db_path
        self.merge_threshold = merge_threshold or HEIM_CONFIG.cluster_merge_threshold
        self.duplicate_threshold = duplicate_threshold or HEIM_CONFIG.duplicate_threshold

        # In-memory index
        self.clusters: Dict[str, Cluster] = {}
        self.episodes: Dict[str, EpisodeRecord] = {}

        # Statistics
        self._total_episodes = 0
        self._duplicates_detected = 0
        self._clusters_created = 0
        self._clusters_merged = 0

        # Load from disk if exists
        if db_path and db_path.exists():
            self._load()

    def _load(self) -> None:
        """Load index from disk."""
        try:
            with open(self.db_path, 'r') as f:
                data = json.load(f)

            for c_data in data.get('clusters', []):
                c = Cluster.from_dict(c_data)
                self.clusters[c.id] = c

            for e_data in data.get('episodes', []):
                e = EpisodeRecord.from_dict(e_data)
                self.episodes[e.episode_id] = e

            logger.info(f"Loaded {len(self.clusters)} clusters, {len(self.episodes)} episodes")

        except Exception as e:
            logger.warning(f"Failed to load cluster index: {e}")

    def _save(self) -> None:
        """Save index to disk."""
        if not self.db_path:
            return

        try:
            data = {
                'clusters': [c.to_dict() for c in self.clusters.values()],
                'episodes': [e.to_dict() for e in self.episodes.values()],
            }

            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.db_path, 'w') as f:
                json.dump(data, f)

        except Exception as e:
            logger.warning(f"Failed to save cluster index: {e}")

    # =========================================================================
    # Core Operations
    # =========================================================================

    def find_nearest_cluster(
        self,
        hv_compressed: np.ndarray,
    ) -> Tuple[Optional[Cluster], float]:
        """
        Find the nearest cluster centroid.

        Args:
            hv_compressed: D_compressed binary HV

        Returns:
            (best_cluster, similarity) or (None, 0) if empty
        """
        if not self.clusters:
            return None, 0.0

        best_cluster = None
        best_sim = -1.0

        for cluster in self.clusters.values():
            sim = hv_hamming_sim(hv_compressed, cluster.centroid)
            if sim > best_sim:
                best_sim = sim
                best_cluster = cluster

        return best_cluster, best_sim

    def assign(
        self,
        hv_compressed: np.ndarray,
        reward: float = 0.0,
        teleology_vector: Optional[np.ndarray] = None,
    ) -> Tuple[str, str, bool]:
        """
        Assign an episode to a cluster.

        Args:
            hv_compressed: D_compressed binary HV
            reward: Episode reward signal
            teleology_vector: Teleology alignment vector

        Returns:
            (episode_id, cluster_id, is_new_cluster)
        """
        self._total_episodes += 1

        # Default teleology vector
        if teleology_vector is None:
            teleology_vector = np.zeros(4, dtype=np.float32)

        # Find nearest cluster
        best_cluster, best_sim = self.find_nearest_cluster(hv_compressed)

        episode_id = str(uuid.uuid4())[:16]

        if best_cluster and best_sim >= self.duplicate_threshold:
            # Near-duplicate: link to cluster, minimal delta
            self._duplicates_detected += 1
            delta_info = np.array([], dtype=np.float32)

            episode = EpisodeRecord(
                episode_id=episode_id,
                cluster_id=best_cluster.id,
                delta_info=delta_info,
                reward=reward,
                teleology_vector=teleology_vector,
            )
            self.episodes[episode_id] = episode

            best_cluster.count += 1
            best_cluster.last_updated = time.time()

            return episode_id, best_cluster.id, False

        elif best_cluster and best_sim >= self.merge_threshold:
            # Join cluster: compute delta
            delta_info = self._compute_delta(hv_compressed, best_cluster.centroid)

            episode = EpisodeRecord(
                episode_id=episode_id,
                cluster_id=best_cluster.id,
                delta_info=delta_info,
                reward=reward,
                teleology_vector=teleology_vector,
            )
            self.episodes[episode_id] = episode

            # Update centroid (online mean in binary space)
            self._update_centroid(best_cluster, hv_compressed)

            return episode_id, best_cluster.id, False

        else:
            # Create new cluster
            cluster_id = str(uuid.uuid4())[:16]
            new_cluster = Cluster(
                id=cluster_id,
                centroid=hv_compressed.copy(),
                count=1,
            )
            self.clusters[cluster_id] = new_cluster
            self._clusters_created += 1

            episode = EpisodeRecord(
                episode_id=episode_id,
                cluster_id=cluster_id,
                delta_info=np.array([], dtype=np.float32),
                reward=reward,
                teleology_vector=teleology_vector,
            )
            self.episodes[episode_id] = episode

            return episode_id, cluster_id, True

    def _compute_delta(
        self,
        hv: np.ndarray,
        centroid: np.ndarray,
    ) -> np.ndarray:
        """
        Compute sparse delta between HV and centroid.

        Returns packed array of (index, sign) pairs for differing bits.
        Max 10 corrections to keep delta small.
        """
        diff_indices = np.where(hv != centroid)[0]

        # Limit to top 10 by importance (arbitrary for now)
        if len(diff_indices) > 10:
            diff_indices = diff_indices[:10]

        if len(diff_indices) == 0:
            return np.array([], dtype=np.float32)

        # Pack as [idx0, sign0, idx1, sign1, ...]
        delta = np.zeros(len(diff_indices) * 2, dtype=np.float32)
        for i, idx in enumerate(diff_indices):
            delta[2*i] = idx
            delta[2*i + 1] = 1.0 if hv[idx] else -1.0

        return delta

    def _update_centroid(self, cluster: Cluster, hv_new: np.ndarray) -> None:
        """
        Update cluster centroid with new member (online majority vote).
        """
        # Simple running majority: keep existing bit if count is high
        # This is approximate but fast
        cluster.count += 1
        cluster.last_updated = time.time()

        # For now, keep centroid stable if cluster is large
        # (proper implementation would track bit-wise sums)
        if cluster.count < 10:
            # Early cluster: do a majority vote update
            weight_old = (cluster.count - 1) / cluster.count
            weight_new = 1.0 / cluster.count

            combined = weight_old * cluster.centroid.astype(np.float32) + weight_new * hv_new.astype(np.float32)
            cluster.centroid = (combined > 0.5).astype(np.uint8)

    # =========================================================================
    # Retrieval
    # =========================================================================

    def nearby(
        self,
        hv_query: np.ndarray,
        threshold: float = 0.0,
        limit: int = 32,
    ) -> List[EpisodeRecord]:
        """
        Find episodes near a query HV.

        Args:
            hv_query: D_compressed binary query
            threshold: Minimum similarity
            limit: Maximum results

        Returns:
            List of EpisodeRecords sorted by similarity (descending)
        """
        # Find candidate clusters
        cluster_scores = []
        for cluster in self.clusters.values():
            sim = hv_hamming_sim(hv_query, cluster.centroid)
            if sim >= threshold:
                cluster_scores.append((cluster, sim))

        cluster_scores.sort(key=lambda x: x[1], reverse=True)

        # Gather episodes from top clusters
        candidates = []
        for cluster, cluster_sim in cluster_scores[:limit]:
            for episode in self.episodes.values():
                if episode.cluster_id == cluster.id:
                    candidates.append((episode, cluster_sim))

            if len(candidates) >= limit:
                break

        candidates.sort(key=lambda x: x[1], reverse=True)
        return [e for e, _ in candidates[:limit]]

    def get_episode(self, episode_id: str) -> Optional[EpisodeRecord]:
        """Get episode by ID."""
        return self.episodes.get(episode_id)

    def get_cluster(self, cluster_id: str) -> Optional[Cluster]:
        """Get cluster by ID."""
        return self.clusters.get(cluster_id)

    # =========================================================================
    # Eviction
    # =========================================================================

    def score_for_eviction(self, episode: EpisodeRecord, now: float) -> float:
        """
        Compute eviction score (lower = more likely to evict).

        Formula: resonance * 0.6 - age * 0.3 + teleology * 0.1
        """
        age_days = (now - episode.timestamp) / 86400.0
        resonance = episode.resonance
        teleology = episode.reward * np.mean(np.abs(episode.teleology_vector))

        return resonance * 0.6 - age_days * 0.3 + teleology * 0.1

    def get_eviction_candidates(self, count: int) -> List[EpisodeRecord]:
        """Get episodes most likely to be evicted."""
        now = time.time()
        scored = [(e, self.score_for_eviction(e, now)) for e in self.episodes.values()]
        scored.sort(key=lambda x: x[1])
        return [e for e, _ in scored[:count]]

    def evict(self, episode_id: str) -> bool:
        """Evict an episode."""
        if episode_id not in self.episodes:
            return False

        episode = self.episodes.pop(episode_id)

        # Update cluster count
        if episode.cluster_id in self.clusters:
            cluster = self.clusters[episode.cluster_id]
            cluster.count -= 1

            # Remove empty clusters
            if cluster.count <= 0:
                del self.clusters[episode.cluster_id]

        return True

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            'total_clusters': len(self.clusters),
            'total_episodes': len(self.episodes),
            'episodes_ingested': self._total_episodes,
            'duplicates_detected': self._duplicates_detected,
            'clusters_created': self._clusters_created,
            'dedup_ratio': self._duplicates_detected / max(self._total_episodes, 1),
            'avg_cluster_size': len(self.episodes) / max(len(self.clusters), 1),
        }

    def sync(self) -> None:
        """Sync to disk."""
        self._save()


# =============================================================================
# Singleton
# =============================================================================

_cluster_index: Optional[ClusterIndex] = None


def get_cluster_index(db_path: Optional[Path] = None) -> ClusterIndex:
    """Get the global cluster index instance."""
    global _cluster_index
    if _cluster_index is None:
        _cluster_index = ClusterIndex(db_path=db_path)
    return _cluster_index


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'Cluster',
    'EpisodeRecord',
    'ClusterIndex',
    'get_cluster_index',
]
