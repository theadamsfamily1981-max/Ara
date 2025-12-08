"""
Binary Associative Memory
==========================

Content-addressable memory using binary codes and Hamming distance.

This is the "fast pattern recall" component:
    - Store binary codes with labels/metadata
    - Query with partial/noisy code
    - Retrieve nearest neighbors via Hamming distance

Use cases in Ara:
    1. "Have I seen this pattern before?" queries
    2. Mood code matching (recognize frustration signatures)
    3. Hippocampus-style episodic memory indexing
    4. Fast pre-filter before expensive similarity search

Implementation:
    - CPU: numpy bit operations
    - GPU: CUDA with __popcll intrinsic (via PyTorch)
    - FPGA: Can offload to neuromorphic co-processor

Usage:
    from ara.neuro.binary.memory import BinaryMemory

    memory = BinaryMemory(code_dim=512, capacity=100000)

    # Store codes
    memory.store(code, label="interesting_state_42", metadata={"flow": 0.8})

    # Query
    matches = memory.query(query_code, k=5)
    for match in matches:
        print(f"Label: {match.label}, Distance: {match.distance}")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

log = logging.getLogger(__name__)

# Try torch import
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


@dataclass
class MemoryEntry:
    """A single entry in binary memory."""
    code: np.ndarray          # Binary code (packed or unpacked)
    label: str = ""           # Human-readable label
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0     # For LRU-style management


@dataclass
class QueryResult:
    """Result from memory query."""
    entry: MemoryEntry
    distance: int             # Hamming distance
    similarity: float         # 1 - (distance / code_dim)
    rank: int                 # 0 = best match


class BinaryMemory:
    """
    Binary associative memory with Hamming distance search.

    Stores binary codes and retrieves nearest neighbors.
    Supports both exact and approximate search.
    """

    def __init__(
        self,
        code_dim: int = 512,
        capacity: int = 100000,
        packed: bool = True,
        use_gpu: bool = False,
    ):
        """
        Initialize binary memory.

        Args:
            code_dim: Dimension of binary codes
            capacity: Maximum number of entries
            packed: Store codes as packed uint64 (more memory efficient)
            use_gpu: Use GPU for search (requires PyTorch)
        """
        self.code_dim = code_dim
        self.capacity = capacity
        self.packed = packed
        self.use_gpu = use_gpu and TORCH_AVAILABLE

        # Packed dimension
        self.n_words = (code_dim + 63) // 64

        # Storage
        self.entries: List[MemoryEntry] = []

        # Code matrix for fast batch search
        # Shape: (n_entries, n_words) if packed, else (n_entries, code_dim)
        if packed:
            self._codes = np.zeros((capacity, self.n_words), dtype=np.uint64)
        else:
            self._codes = np.zeros((capacity, code_dim), dtype=np.int8)

        self._size = 0

        # GPU tensors (lazy init)
        self._codes_gpu = None

        log.info(
            f"BinaryMemory initialized: dim={code_dim}, capacity={capacity}, "
            f"packed={packed}, gpu={self.use_gpu}"
        )

    @property
    def size(self) -> int:
        """Current number of entries."""
        return self._size

    def store(
        self,
        code: Union[np.ndarray, "torch.Tensor"],
        label: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Store a binary code in memory.

        Args:
            code: Binary code to store
            label: Human-readable label
            metadata: Additional metadata

        Returns:
            Index of stored entry
        """
        # Convert to numpy
        if TORCH_AVAILABLE and isinstance(code, torch.Tensor):
            code = code.detach().cpu().numpy()

        code = np.asarray(code).flatten()

        # Binarize if needed
        if code.dtype in (np.float32, np.float64):
            code = (code >= 0).astype(np.int8) * 2 - 1  # {-1, +1}

        # Pack if needed
        if self.packed:
            code_stored = self._pack_code(code)
        else:
            code_stored = code.astype(np.int8)

        # Check capacity
        if self._size >= self.capacity:
            # Simple FIFO eviction
            log.warning("BinaryMemory: Capacity reached, evicting oldest entry")
            self._evict_oldest()

        # Store
        idx = self._size
        self._codes[idx] = code_stored

        entry = MemoryEntry(
            code=code_stored,
            label=label,
            metadata=metadata or {},
        )
        self.entries.append(entry)
        self._size += 1

        # Invalidate GPU cache
        self._codes_gpu = None

        return idx

    def store_batch(
        self,
        codes: Union[np.ndarray, "torch.Tensor"],
        labels: Optional[List[str]] = None,
    ) -> List[int]:
        """
        Store multiple codes at once.

        Args:
            codes: Array of shape (n, code_dim)
            labels: Optional list of labels

        Returns:
            List of indices
        """
        if TORCH_AVAILABLE and isinstance(codes, torch.Tensor):
            codes = codes.detach().cpu().numpy()

        n = codes.shape[0]
        labels = labels or [""] * n

        indices = []
        for i in range(n):
            idx = self.store(codes[i], label=labels[i])
            indices.append(idx)

        return indices

    def query(
        self,
        code: Union[np.ndarray, "torch.Tensor"],
        k: int = 5,
        max_distance: Optional[int] = None,
    ) -> List[QueryResult]:
        """
        Find k nearest neighbors by Hamming distance.

        Args:
            code: Query code
            k: Number of neighbors to return
            max_distance: Maximum Hamming distance to consider

        Returns:
            List of QueryResult sorted by distance
        """
        if self._size == 0:
            return []

        # Convert query to appropriate format
        if TORCH_AVAILABLE and isinstance(code, torch.Tensor):
            code = code.detach().cpu().numpy()

        code = np.asarray(code).flatten()

        if code.dtype in (np.float32, np.float64):
            code = (code >= 0).astype(np.int8) * 2 - 1

        if self.packed:
            query_packed = self._pack_code(code)
        else:
            query_packed = code.astype(np.int8)

        # Compute distances
        if self.use_gpu and TORCH_AVAILABLE:
            distances = self._hamming_gpu(query_packed)
        else:
            distances = self._hamming_cpu(query_packed)

        # Filter by max distance
        if max_distance is not None:
            valid_mask = distances <= max_distance
            valid_indices = np.where(valid_mask)[0]
            if len(valid_indices) == 0:
                return []
            distances = distances[valid_indices]
        else:
            valid_indices = np.arange(self._size)

        # Get top-k
        k = min(k, len(distances))
        top_k_idx = np.argpartition(distances, k - 1)[:k]
        top_k_idx = top_k_idx[np.argsort(distances[top_k_idx])]

        # Build results
        results = []
        for rank, idx in enumerate(top_k_idx):
            entry_idx = valid_indices[idx]
            entry = self.entries[entry_idx]
            entry.access_count += 1

            dist = int(distances[idx])
            sim = 1.0 - (dist / self.code_dim)

            results.append(QueryResult(
                entry=entry,
                distance=dist,
                similarity=sim,
                rank=rank,
            ))

        return results

    def query_batch(
        self,
        codes: Union[np.ndarray, "torch.Tensor"],
        k: int = 5,
    ) -> List[List[QueryResult]]:
        """
        Batch query for multiple codes.

        Args:
            codes: Array of shape (n, code_dim)
            k: Number of neighbors per query

        Returns:
            List of result lists
        """
        if TORCH_AVAILABLE and isinstance(codes, torch.Tensor):
            codes = codes.detach().cpu().numpy()

        return [self.query(codes[i], k=k) for i in range(codes.shape[0])]

    def contains_similar(
        self,
        code: Union[np.ndarray, "torch.Tensor"],
        threshold: float = 0.9,
    ) -> bool:
        """
        Check if memory contains a similar code.

        Args:
            code: Query code
            threshold: Minimum similarity (0-1)

        Returns:
            True if similar code exists
        """
        max_distance = int((1 - threshold) * self.code_dim)
        results = self.query(code, k=1, max_distance=max_distance)
        return len(results) > 0

    def get_all_codes(self, packed: bool = True) -> np.ndarray:
        """Get all stored codes as array."""
        if packed:
            return self._codes[:self._size].copy()
        else:
            # Unpack if stored packed
            if self.packed:
                return np.array([
                    self._unpack_code(self._codes[i])
                    for i in range(self._size)
                ])
            else:
                return self._codes[:self._size].copy()

    def clear(self) -> None:
        """Clear all entries."""
        self.entries.clear()
        self._size = 0
        self._codes_gpu = None
        log.info("BinaryMemory: Cleared")

    # =========================================================================
    # INTERNAL: Hamming Distance
    # =========================================================================

    def _hamming_cpu(self, query: np.ndarray) -> np.ndarray:
        """
        Compute Hamming distances on CPU.

        Args:
            query: Query code (packed or unpacked)

        Returns:
            Array of distances to all entries
        """
        if self.packed:
            # XNOR + popcount
            codes = self._codes[:self._size]  # (n, n_words)
            query_exp = query[np.newaxis, :]  # (1, n_words)

            # XOR to find differing bits
            xor = codes ^ query_exp

            # Popcount each word
            distances = np.zeros(self._size, dtype=np.int32)
            for word_idx in range(self.n_words):
                word_xor = xor[:, word_idx]
                # Popcount via parallel algorithm
                distances += self._popcount64(word_xor)

            return distances
        else:
            # Simple element-wise comparison
            codes = self._codes[:self._size]  # (n, code_dim)
            diff = (codes != query).astype(np.int32)
            return diff.sum(axis=1)

    def _hamming_gpu(self, query: np.ndarray) -> np.ndarray:
        """
        Compute Hamming distances on GPU.

        Args:
            query: Query code

        Returns:
            Array of distances
        """
        if self._codes_gpu is None:
            # Transfer to GPU
            self._codes_gpu = torch.from_numpy(
                self._codes[:self._size].astype(np.int64)
            ).cuda()

        query_gpu = torch.from_numpy(query.astype(np.int64)).cuda()
        query_gpu = query_gpu.unsqueeze(0)  # (1, n_words)

        # XOR
        xor = self._codes_gpu ^ query_gpu

        # Popcount (parallel algorithm in PyTorch)
        def popcount_torch(v):
            v = v - ((v >> 1) & 0x5555555555555555)
            v = (v & 0x3333333333333333) + ((v >> 2) & 0x3333333333333333)
            v = (v + (v >> 4)) & 0x0F0F0F0F0F0F0F0F
            v = (v * 0x0101010101010101) >> 56
            return v

        bit_counts = popcount_torch(xor)
        distances = bit_counts.sum(dim=1)

        return distances.cpu().numpy().astype(np.int32)

    def _popcount64(self, arr: np.ndarray) -> np.ndarray:
        """Fast popcount for uint64 array."""
        v = arr.astype(np.uint64)
        v = v - ((v >> 1) & 0x5555555555555555)
        v = (v & 0x3333333333333333) + ((v >> 2) & 0x3333333333333333)
        v = (v + (v >> 4)) & 0x0F0F0F0F0F0F0F0F
        v = (v * 0x0101010101010101) >> 56
        return v.astype(np.int32)

    # =========================================================================
    # INTERNAL: Packing
    # =========================================================================

    def _pack_code(self, code: np.ndarray) -> np.ndarray:
        """Pack binary code into uint64 words."""
        # Binarize to {0, 1}
        binary = (code >= 0).astype(np.uint64)

        # Pad to multiple of 64
        pad_to = self.n_words * 64
        if len(binary) < pad_to:
            binary = np.pad(binary, (0, pad_to - len(binary)))

        # Reshape and pack
        binary = binary.reshape(self.n_words, 64)
        powers = (1 << np.arange(64, dtype=np.uint64))
        packed = (binary * powers).sum(axis=1)

        return packed

    def _unpack_code(self, packed: np.ndarray) -> np.ndarray:
        """Unpack uint64 words to binary code."""
        unpacked = np.zeros(self.n_words * 64, dtype=np.int8)

        for word_idx in range(self.n_words):
            word = packed[word_idx]
            for bit_idx in range(64):
                if (word >> bit_idx) & 1:
                    unpacked[word_idx * 64 + bit_idx] = 1
                else:
                    unpacked[word_idx * 64 + bit_idx] = -1

        return unpacked[:self.code_dim]

    def _evict_oldest(self) -> None:
        """Evict oldest entry (FIFO)."""
        if self._size > 0:
            self.entries.pop(0)
            self._codes[:-1] = self._codes[1:]
            self._size -= 1


# =============================================================================
# SPECIALIZED MEMORIES
# =============================================================================

class MoodCodeMemory(BinaryMemory):
    """
    Specialized memory for mood/state codes.

    Stores binary signatures of internal states and recognizes
    patterns that correlate with flow, frustration, etc.
    """

    def __init__(self, code_dim: int = 256, capacity: int = 10000):
        super().__init__(code_dim=code_dim, capacity=capacity)

        # Mood labels
        self.mood_categories = [
            "flow", "frustration", "curiosity", "fatigue",
            "excitement", "confusion", "satisfaction", "neutral"
        ]

    def store_mood(
        self,
        code: np.ndarray,
        mood: str,
        intensity: float = 1.0,
        context: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Store a mood code.

        Args:
            code: Binary code
            mood: Mood category
            intensity: Mood intensity (0-1)
            context: Additional context

        Returns:
            Entry index
        """
        metadata = {
            "mood": mood,
            "intensity": intensity,
            "context": context or {},
        }

        return self.store(code, label=f"mood:{mood}", metadata=metadata)

    def recognize_mood(
        self,
        code: np.ndarray,
        threshold: float = 0.8,
    ) -> Optional[Tuple[str, float]]:
        """
        Recognize mood from code.

        Args:
            code: Query code
            threshold: Minimum similarity

        Returns:
            (mood, confidence) or None
        """
        results = self.query(code, k=5)

        if not results:
            return None

        best = results[0]
        if best.similarity < threshold:
            return None

        mood = best.entry.metadata.get("mood", "unknown")
        return (mood, best.similarity)

    def get_mood_statistics(self) -> Dict[str, int]:
        """Get count of each mood category."""
        counts = {mood: 0 for mood in self.mood_categories}

        for entry in self.entries:
            mood = entry.metadata.get("mood", "neutral")
            if mood in counts:
                counts[mood] += 1

        return counts


class EpisodicIndex(BinaryMemory):
    """
    Binary index for episodic memory.

    Uses binary codes as fast lookup keys for full episodes
    stored elsewhere (e.g., in Hippocampus).
    """

    def __init__(self, code_dim: int = 512, capacity: int = 50000):
        super().__init__(code_dim=code_dim, capacity=capacity)

    def index_episode(
        self,
        code: np.ndarray,
        episode_id: str,
        summary: str = "",
        importance: float = 0.5,
    ) -> int:
        """
        Index an episode with its binary code.

        Args:
            code: Binary code summarizing episode
            episode_id: Unique episode identifier
            summary: Brief text summary
            importance: Importance score (0-1)

        Returns:
            Entry index
        """
        metadata = {
            "episode_id": episode_id,
            "summary": summary,
            "importance": importance,
        }

        return self.store(code, label=episode_id, metadata=metadata)

    def recall_episodes(
        self,
        query_code: np.ndarray,
        k: int = 10,
        min_importance: float = 0.0,
    ) -> List[str]:
        """
        Recall episode IDs similar to query.

        Args:
            query_code: Query code
            k: Number of episodes
            min_importance: Minimum importance filter

        Returns:
            List of episode IDs
        """
        results = self.query(query_code, k=k * 2)  # Get more to filter

        episode_ids = []
        for r in results:
            if r.entry.metadata.get("importance", 0) >= min_importance:
                episode_ids.append(r.entry.metadata.get("episode_id", ""))
                if len(episode_ids) >= k:
                    break

        return episode_ids


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'MemoryEntry',
    'QueryResult',
    'BinaryMemory',
    'MoodCodeMemory',
    'EpisodicIndex',
]
