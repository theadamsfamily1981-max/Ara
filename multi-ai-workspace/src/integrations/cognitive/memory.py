"""CXL Infinite Memory - The Cortex Extension.

This module implements EpisodicMemory with CXL (Compute Express Link) paging,
effectively giving Ara infinite recall by treating disk/NVMe/remote memory
as a single addressable space.

Key Concepts:

    EpisodicMemory: Long-term memory for experiences/interactions
        - Stores episodes (interactions, events, learnings)
        - Enables recall via embedding similarity search
        - Supports consolidation and forgetting

    CXLPager: Virtual memory abstraction over heterogeneous storage
        - Treats RAM, NVMe, disk, network as unified address space
        - Automatic paging based on access patterns
        - LRU eviction with importance weighting

    Episode: A structured memory unit
        - Content (text, embeddings, metadata)
        - Timestamp and context
        - Importance score for retention priority
        - Embedding for similarity search

Memory Tiers:
    Tier 0: Hot (GPU/HBM) - Currently active
    Tier 1: Warm (RAM) - Recently accessed
    Tier 2: Cool (NVMe/SSD) - Infrequently accessed
    Tier 3: Cold (Disk/Network) - Archive

This implements infinite memory from tfan/memory/cxl_pager.py.
"""

import torch
import numpy as np
import pickle
import hashlib
import time
import warnings
import json
import mmap
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import OrderedDict
import sys

# Add TFAN to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent))

# Try to import TFAN memory modules
_TFAN_MEMORY_AVAILABLE = False
try:
    from tfan.memory.cxl_pager import CXLPager as TFANCXLPager
    from tfan.memory.episodic import EpisodicMemory as TFANEpisodicMemory
    _TFAN_MEMORY_AVAILABLE = True
except ImportError:
    pass


class MemoryTier(Enum):
    """Memory storage tiers."""
    HOT = auto()    # GPU/HBM - Currently active
    WARM = auto()   # RAM - Recently accessed
    COOL = auto()   # NVMe/SSD - Infrequently accessed
    COLD = auto()   # Disk/Network - Archive


@dataclass
class Episode:
    """A structured memory unit."""
    id: str                          # Unique identifier
    content: str                     # Text content
    embedding: Optional[np.ndarray]  # Vector embedding
    timestamp: float                 # Creation time
    importance: float                # Retention priority [0, 1]
    access_count: int = 0            # Times accessed
    last_access: float = 0.0         # Last access time
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_bytes(self) -> bytes:
        """Serialize episode to bytes."""
        data = {
            "id": self.id,
            "content": self.content,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "timestamp": self.timestamp,
            "importance": self.importance,
            "access_count": self.access_count,
            "last_access": self.last_access,
            "context": self.context,
            "metadata": self.metadata,
        }
        return pickle.dumps(data)

    @classmethod
    def from_bytes(cls, data: bytes) -> "Episode":
        """Deserialize episode from bytes."""
        d = pickle.loads(data)
        return cls(
            id=d["id"],
            content=d["content"],
            embedding=np.array(d["embedding"]) if d["embedding"] else None,
            timestamp=d["timestamp"],
            importance=d["importance"],
            access_count=d.get("access_count", 0),
            last_access=d.get("last_access", 0.0),
            context=d.get("context", {}),
            metadata=d.get("metadata", {}),
        )


@dataclass
class MemoryAddress:
    """Address in the virtual memory space."""
    tier: MemoryTier
    offset: int
    size: int
    episode_id: str


class CXLPager:
    """
    CXL Pager - Virtual memory over heterogeneous storage.

    Provides a unified address space over RAM, disk, and potentially
    network storage. Enables "infinite" memory by paging.

    Args:
        capacity_gb: Total virtual capacity in GB
        ram_budget_mb: RAM budget for hot/warm data in MB
        storage_path: Path for cool/cold storage
        enable_mmap: Use memory-mapped files for cool tier
    """

    def __init__(
        self,
        capacity_gb: float = 1024.0,  # 1TB virtual
        ram_budget_mb: float = 512.0,
        storage_path: Optional[str] = None,
        enable_mmap: bool = True,
    ):
        self.capacity_bytes = int(capacity_gb * 1024 * 1024 * 1024)
        self.ram_budget_bytes = int(ram_budget_mb * 1024 * 1024)
        self.enable_mmap = enable_mmap

        # Storage path
        if storage_path is None:
            storage_path = str(Path.home() / ".ara" / "memory")
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # TFAN pager if available
        self.tfan_pager = None
        if _TFAN_MEMORY_AVAILABLE:
            try:
                self.tfan_pager = TFANCXLPager(capacity_gb=capacity_gb)
            except Exception as e:
                warnings.warn(f"Failed to init TFAN CXL pager: {e}")

        # Memory tiers (fallback implementation)
        self._hot: Dict[str, bytes] = {}       # In-memory, most accessed
        self._warm: OrderedDict[str, bytes] = OrderedDict()  # In-memory, LRU
        self._cool_index: Dict[str, str] = {}  # Maps ID to file path

        # Address mapping
        self._addresses: Dict[str, MemoryAddress] = {}

        # Statistics
        self._total_allocated = 0
        self._hits = {"hot": 0, "warm": 0, "cool": 0, "cold": 0}
        self._misses = 0

    def write(self, data: bytes, episode_id: Optional[str] = None) -> str:
        """
        Write data to virtual memory.

        Args:
            data: Data to write
            episode_id: Optional ID (generated if not provided)

        Returns:
            Address/ID for retrieval
        """
        if self.tfan_pager is not None:
            return self.tfan_pager.write(data)

        # Generate ID if needed
        if episode_id is None:
            episode_id = hashlib.sha256(
                f"{time.time()}{len(data)}".encode()
            ).hexdigest()[:16]

        size = len(data)
        self._total_allocated += size

        # Determine tier based on size and budget
        if self._get_ram_usage() + size < self.ram_budget_bytes:
            # Fits in RAM
            tier = MemoryTier.WARM
            self._warm[episode_id] = data
            self._warm.move_to_end(episode_id)
        else:
            # Page to disk
            tier = MemoryTier.COOL
            self._write_to_disk(episode_id, data)

        # Create address
        self._addresses[episode_id] = MemoryAddress(
            tier=tier,
            offset=0,  # Not used in this implementation
            size=size,
            episode_id=episode_id,
        )

        # Enforce RAM budget
        self._enforce_ram_budget()

        return episode_id

    def read(self, episode_id: str) -> Optional[bytes]:
        """
        Read data from virtual memory.

        Args:
            episode_id: Address/ID to read

        Returns:
            Data bytes or None if not found
        """
        if self.tfan_pager is not None:
            return self.tfan_pager.read(episode_id)

        # Check hot tier
        if episode_id in self._hot:
            self._hits["hot"] += 1
            return self._hot[episode_id]

        # Check warm tier
        if episode_id in self._warm:
            self._hits["warm"] += 1
            # Move to end (most recently used)
            self._warm.move_to_end(episode_id)
            # Promote to hot if frequently accessed
            return self._warm[episode_id]

        # Check cool tier (disk)
        if episode_id in self._cool_index:
            self._hits["cool"] += 1
            data = self._read_from_disk(episode_id)
            if data:
                # Page into warm
                self._warm[episode_id] = data
                self._addresses[episode_id].tier = MemoryTier.WARM
                self._enforce_ram_budget()
            return data

        self._misses += 1
        return None

    def _write_to_disk(self, episode_id: str, data: bytes):
        """Write data to disk storage."""
        file_path = self.storage_path / f"{episode_id}.mem"
        with open(file_path, "wb") as f:
            f.write(data)
        self._cool_index[episode_id] = str(file_path)

    def _read_from_disk(self, episode_id: str) -> Optional[bytes]:
        """Read data from disk storage."""
        if episode_id not in self._cool_index:
            return None

        file_path = Path(self._cool_index[episode_id])
        if not file_path.exists():
            return None

        with open(file_path, "rb") as f:
            return f.read()

    def _get_ram_usage(self) -> int:
        """Get current RAM usage."""
        hot_size = sum(len(d) for d in self._hot.values())
        warm_size = sum(len(d) for d in self._warm.values())
        return hot_size + warm_size

    def _enforce_ram_budget(self):
        """Evict data to stay within RAM budget."""
        while self._get_ram_usage() > self.ram_budget_bytes and self._warm:
            # Evict oldest from warm
            episode_id, data = self._warm.popitem(last=False)
            # Page to disk
            self._write_to_disk(episode_id, data)
            self._addresses[episode_id].tier = MemoryTier.COOL

    def delete(self, episode_id: str) -> bool:
        """Delete data from virtual memory."""
        # Remove from all tiers
        if episode_id in self._hot:
            del self._hot[episode_id]
        if episode_id in self._warm:
            del self._warm[episode_id]
        if episode_id in self._cool_index:
            file_path = Path(self._cool_index[episode_id])
            if file_path.exists():
                file_path.unlink()
            del self._cool_index[episode_id]
        if episode_id in self._addresses:
            del self._addresses[episode_id]
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get pager statistics."""
        return {
            "total_allocated_bytes": self._total_allocated,
            "ram_usage_bytes": self._get_ram_usage(),
            "ram_budget_bytes": self.ram_budget_bytes,
            "hot_count": len(self._hot),
            "warm_count": len(self._warm),
            "cool_count": len(self._cool_index),
            "hits": self._hits.copy(),
            "misses": self._misses,
            "hit_rate": (
                sum(self._hits.values()) /
                (sum(self._hits.values()) + self._misses + 1)
            ),
        }


class VectorIndex:
    """Simple vector similarity index for memory search."""

    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self._vectors: Dict[str, np.ndarray] = {}

    def add(self, episode_id: str, embedding: np.ndarray):
        """Add embedding to index."""
        self._vectors[episode_id] = embedding.flatten()

    def search(
        self,
        query: np.ndarray,
        k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Search for similar embeddings."""
        if not self._vectors:
            return []

        query_flat = query.flatten()

        # Compute similarities
        similarities = []
        for episode_id, embedding in self._vectors.items():
            # Cosine similarity
            sim = np.dot(query_flat, embedding) / (
                np.linalg.norm(query_flat) * np.linalg.norm(embedding) + 1e-8
            )
            similarities.append((episode_id, float(sim)))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

    def remove(self, episode_id: str):
        """Remove from index."""
        if episode_id in self._vectors:
            del self._vectors[episode_id]

    def __len__(self):
        return len(self._vectors)


class EpisodicMemory:
    """
    Episodic Memory - Long-term memory with CXL backing.

    Stores experiences/interactions for later recall via
    embedding similarity search.

    Args:
        use_cxl: Enable CXL paging for infinite capacity
        capacity_gb: Virtual capacity in GB (if CXL enabled)
        ram_budget_mb: RAM budget in MB
        embedding_dim: Dimension of embeddings
        storage_path: Path for persistent storage
    """

    def __init__(
        self,
        use_cxl: bool = True,
        capacity_gb: float = 1024.0,
        ram_budget_mb: float = 512.0,
        embedding_dim: int = 768,
        storage_path: Optional[str] = None,
    ):
        self.use_cxl = use_cxl
        self.embedding_dim = embedding_dim

        # TFAN memory if available
        self.tfan_memory = None
        if _TFAN_MEMORY_AVAILABLE:
            try:
                self.tfan_memory = TFANEpisodicMemory(use_cxl=use_cxl)
            except Exception as e:
                warnings.warn(f"Failed to init TFAN episodic memory: {e}")

        # Storage backend
        if use_cxl and self.tfan_memory is None:
            self.storage = CXLPager(
                capacity_gb=capacity_gb,
                ram_budget_mb=ram_budget_mb,
                storage_path=storage_path,
            )
        else:
            self.storage = {}  # Simple dict for non-CXL mode

        # Vector index for similarity search
        self.index = VectorIndex(dimension=embedding_dim)

        # Episode metadata cache (always in RAM for fast lookup)
        self._metadata_cache: Dict[str, Dict[str, Any]] = {}

        # Statistics
        self.total_episodes = 0
        self.total_recalls = 0

    def store_episode(
        self,
        content: str,
        embedding: Optional[np.ndarray] = None,
        importance: float = 0.5,
        context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store an episode in memory.

        Args:
            content: Text content of the episode
            embedding: Vector embedding for similarity search
            importance: Retention priority [0, 1]
            context: Contextual information
            metadata: Additional metadata

        Returns:
            Episode ID
        """
        if self.tfan_memory is not None:
            return self.tfan_memory.store_episode(
                content=content,
                embedding=embedding,
                importance=importance,
            )

        # Generate ID
        episode_id = hashlib.sha256(
            f"{time.time()}{content[:50]}".encode()
        ).hexdigest()[:16]

        # Create episode
        episode = Episode(
            id=episode_id,
            content=content,
            embedding=embedding,
            timestamp=time.time(),
            importance=importance,
            context=context or {},
            metadata=metadata or {},
        )

        # Store in backend
        if self.use_cxl and isinstance(self.storage, CXLPager):
            self.storage.write(episode.to_bytes(), episode_id)
        else:
            self.storage[episode_id] = episode

        # Add to vector index
        if embedding is not None:
            self.index.add(episode_id, embedding)

        # Cache metadata
        self._metadata_cache[episode_id] = {
            "timestamp": episode.timestamp,
            "importance": episode.importance,
            "content_preview": content[:100],
        }

        self.total_episodes += 1
        return episode_id

    def recall(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        min_importance: float = 0.0,
    ) -> List[Episode]:
        """
        Recall episodes similar to query.

        Args:
            query_embedding: Query vector
            k: Number of results
            min_importance: Minimum importance threshold

        Returns:
            List of similar episodes
        """
        if self.tfan_memory is not None:
            return self.tfan_memory.recall(query_embedding)

        self.total_recalls += 1

        # Search index
        results = self.index.search(query_embedding, k=k * 2)  # Over-fetch for filtering

        episodes = []
        for episode_id, similarity in results:
            episode = self._load_episode(episode_id)
            if episode and episode.importance >= min_importance:
                # Update access stats
                episode.access_count += 1
                episode.last_access = time.time()
                episodes.append(episode)

                if len(episodes) >= k:
                    break

        return episodes

    def recall_by_id(self, episode_id: str) -> Optional[Episode]:
        """Recall specific episode by ID."""
        return self._load_episode(episode_id)

    def recall_recent(self, n: int = 10) -> List[Episode]:
        """Recall most recent episodes."""
        # Sort by timestamp
        sorted_ids = sorted(
            self._metadata_cache.keys(),
            key=lambda x: self._metadata_cache[x]["timestamp"],
            reverse=True,
        )

        episodes = []
        for episode_id in sorted_ids[:n]:
            episode = self._load_episode(episode_id)
            if episode:
                episodes.append(episode)

        return episodes

    def _load_episode(self, episode_id: str) -> Optional[Episode]:
        """Load episode from storage."""
        if self.use_cxl and isinstance(self.storage, CXLPager):
            data = self.storage.read(episode_id)
            if data:
                return Episode.from_bytes(data)
            return None
        else:
            return self.storage.get(episode_id)

    def forget(self, episode_id: str) -> bool:
        """Remove episode from memory."""
        # Remove from index
        self.index.remove(episode_id)

        # Remove from cache
        if episode_id in self._metadata_cache:
            del self._metadata_cache[episode_id]

        # Remove from storage
        if self.use_cxl and isinstance(self.storage, CXLPager):
            return self.storage.delete(episode_id)
        else:
            if episode_id in self.storage:
                del self.storage[episode_id]
                return True
            return False

    def consolidate(self, min_importance: float = 0.3) -> int:
        """
        Consolidate memory - remove low-importance episodes.

        Returns:
            Number of episodes removed
        """
        removed = 0
        to_remove = []

        for episode_id, meta in self._metadata_cache.items():
            if meta["importance"] < min_importance:
                to_remove.append(episode_id)

        for episode_id in to_remove:
            if self.forget(episode_id):
                removed += 1

        return removed

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        stats = {
            "total_episodes": self.total_episodes,
            "cached_episodes": len(self._metadata_cache),
            "indexed_episodes": len(self.index),
            "total_recalls": self.total_recalls,
            "use_cxl": self.use_cxl,
        }

        if self.use_cxl and isinstance(self.storage, CXLPager):
            stats["pager_stats"] = self.storage.get_stats()

        return stats


# Convenience factories
def create_episodic_memory(
    use_cxl: bool = True,
    capacity_gb: float = 1024.0,
) -> EpisodicMemory:
    """Create an EpisodicMemory instance."""
    return EpisodicMemory(
        use_cxl=use_cxl,
        capacity_gb=capacity_gb,
    )


def create_cxl_pager(
    capacity_gb: float = 1024.0,
    ram_budget_mb: float = 512.0,
) -> CXLPager:
    """Create a CXLPager instance."""
    return CXLPager(
        capacity_gb=capacity_gb,
        ram_budget_mb=ram_budget_mb,
    )


__all__ = [
    "EpisodicMemory",
    "CXLPager",
    "Episode",
    "MemoryAddress",
    "MemoryTier",
    "VectorIndex",
    "create_episodic_memory",
    "create_cxl_pager",
]
