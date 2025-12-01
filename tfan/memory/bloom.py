#!/usr/bin/env python
"""
Bloom Filter Prefetcher for KV Cache

Uses Bloom filters to track access patterns and predict future cache accesses.
Enables intelligent prefetching to improve cache hit rates in CXL memory tiering.

Architecture:
1. Access Tracking: Record (layer, block) → (layer, block) transitions
2. Bloom Filters: Multiple filters for different lookahead distances
3. Prediction: Query filters to predict next likely accesses
4. Adaptation: Periodically reset filters to adapt to changing patterns

Performance targets:
- Prefetch accuracy ≥80%
- Space overhead <1% of cache size
- Prediction latency <100μs

Bloom Filter Properties:
- Probabilistic data structure
- No false negatives (if pattern seen, will find it)
- Possible false positives (may predict non-existent pattern)
- Space-efficient (~10 bits per element)

Usage:
    prefetcher = BloomPrefetcher(
        capacity=10000,
        error_rate=0.01,
        lookahead=4
    )

    # Record access
    prefetcher.record_access(layer_idx, block_idx)

    # Predict next accesses
    predictions = prefetcher.predict_next(layer_idx, block_idx)
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import hashlib
import time


@dataclass
class BloomConfig:
    """Configuration for Bloom filter prefetcher."""
    capacity: int = 10000  # Expected number of patterns
    error_rate: float = 0.01  # False positive rate
    lookahead: int = 4  # Number of blocks to lookahead
    num_hash_functions: Optional[int] = None  # Auto-calculated if None
    reset_interval: float = 300.0  # Reset filters every 5 min to adapt


class BloomFilter:
    """
    Space-efficient Bloom filter.

    Uses multiple hash functions to set/check bits in a bit array.
    """

    def __init__(self, capacity: int, error_rate: float):
        """
        Initialize Bloom filter.

        Args:
            capacity: Expected number of elements
            error_rate: Desired false positive rate
        """
        self.capacity = capacity
        self.error_rate = error_rate

        # Calculate optimal bit array size
        # m = -(n * ln(p)) / (ln(2)^2)
        self.size = self._calculate_size(capacity, error_rate)

        # Calculate optimal number of hash functions
        # k = (m/n) * ln(2)
        self.num_hashes = self._calculate_num_hashes(self.size, capacity)

        # Bit array
        self.bits = np.zeros(self.size, dtype=bool)

        # Statistics
        self.num_elements = 0

    @staticmethod
    def _calculate_size(capacity: int, error_rate: float) -> int:
        """Calculate optimal bit array size."""
        import math
        m = -(capacity * math.log(error_rate)) / (math.log(2) ** 2)
        return int(m)

    @staticmethod
    def _calculate_num_hashes(size: int, capacity: int) -> int:
        """Calculate optimal number of hash functions."""
        import math
        k = (size / capacity) * math.log(2)
        return max(1, int(k))

    def _hash(self, item: bytes, seed: int) -> int:
        """Generate hash value for item with seed."""
        h = hashlib.sha256(item + str(seed).encode()).digest()
        return int.from_bytes(h[:4], 'big') % self.size

    def add(self, item: Tuple[int, int]):
        """
        Add item to Bloom filter.

        Args:
            item: (layer_idx, block_idx) tuple
        """
        item_bytes = f"{item[0]}:{item[1]}".encode()

        for i in range(self.num_hashes):
            idx = self._hash(item_bytes, i)
            self.bits[idx] = True

        self.num_elements += 1

    def contains(self, item: Tuple[int, int]) -> bool:
        """
        Check if item might be in filter.

        Args:
            item: (layer_idx, block_idx) tuple

        Returns:
            True if item might be present (or false positive)
            False if item definitely not present
        """
        item_bytes = f"{item[0]}:{item[1]}".encode()

        for i in range(self.num_hashes):
            idx = self._hash(item_bytes, i)
            if not self.bits[idx]:
                return False

        return True

    def reset(self):
        """Clear all bits."""
        self.bits.fill(False)
        self.num_elements = 0


class BloomPrefetcher:
    """
    Bloom filter-based cache prefetcher.

    Tracks access patterns and predicts future accesses using Bloom filters.
    """

    def __init__(self, config: Optional[BloomConfig] = None):
        """
        Initialize Bloom prefetcher.

        Args:
            config: Bloom filter configuration
        """
        self.config = config or BloomConfig()

        # Create Bloom filters for different lookahead distances
        # filters[i] tracks (current_block → block at distance i+1)
        self.filters: List[BloomFilter] = []

        for _ in range(self.config.lookahead):
            bloom = BloomFilter(
                capacity=self.config.capacity,
                error_rate=self.config.error_rate
            )
            self.filters.append(bloom)

        # Access history (for learning patterns)
        self.history: List[Tuple[int, int]] = []
        self.max_history = 1000  # Keep recent history for pattern learning

        # Last reset time
        self.last_reset = time.time()

        # Statistics
        self.num_predictions = 0
        self.num_correct_predictions = 0
        self.num_accesses = 0

        print(f"✓ Bloom Prefetcher initialized")
        print(f"  Capacity: {self.config.capacity:,}")
        print(f"  Error rate: {self.config.error_rate:.2%}")
        print(f"  Lookahead: {self.config.lookahead}")
        print(f"  Bloom size: {self.filters[0].size:,} bits ({self.filters[0].size / 8 / 1024:.1f} KB)")
        print(f"  Hash functions: {self.filters[0].num_hashes}")

    def record_access(self, layer_idx: int, block_idx: int):
        """
        Record cache access and update Bloom filters.

        Args:
            layer_idx: Layer index
            block_idx: Block index
        """
        self.num_accesses += 1

        current = (layer_idx, block_idx)

        # Update Bloom filters based on recent history
        for i, prev_access in enumerate(reversed(self.history)):
            if i >= self.config.lookahead:
                break

            # Record transition: prev_access → current at distance (i+1)
            # This means: after seeing prev_access, we saw current after i+1 steps
            pattern = (prev_access, current)
            self.filters[i].add(pattern)

        # Update history
        self.history.append(current)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        # Periodic reset to adapt to changing patterns
        if time.time() - self.last_reset > self.config.reset_interval:
            self._periodic_reset()

    def predict_next(
        self,
        layer_idx: int,
        block_idx: int
    ) -> List[Tuple[int, int]]:
        """
        Predict next likely cache accesses.

        Args:
            layer_idx: Current layer index
            block_idx: Current block index

        Returns:
            List of predicted (layer_idx, block_idx) tuples
        """
        self.num_predictions += 1

        current = (layer_idx, block_idx)
        predictions = []

        # Strategy 1: Sequential pattern (most common)
        # Predict next few blocks in same layer
        for offset in range(1, self.config.lookahead + 1):
            candidate = (layer_idx, block_idx + offset)

            # Check if this pattern exists in any Bloom filter
            for i in range(min(offset, len(self.filters))):
                pattern = (current, candidate)
                if self.filters[i].contains(pattern):
                    predictions.append(candidate)
                    break

        # Strategy 2: Same block in next layer (cross-layer pattern)
        # Common when processing same sequence position across layers
        next_layer_candidate = (layer_idx + 1, block_idx)
        for bloom in self.filters:
            pattern = (current, next_layer_candidate)
            if bloom.contains(pattern):
                if next_layer_candidate not in predictions:
                    predictions.append(next_layer_candidate)
                break

        # Limit predictions to avoid over-prefetching
        return predictions[:self.config.lookahead]

    def verify_prediction(self, layer_idx: int, block_idx: int, was_hit: bool):
        """
        Verify if prediction was correct (for statistics).

        Args:
            layer_idx: Layer index that was accessed
            block_idx: Block index that was accessed
            was_hit: Whether it was a cache hit
        """
        if was_hit:
            self.num_correct_predictions += 1

    def get_stats(self) -> dict:
        """Get prefetcher statistics."""
        accuracy = (
            self.num_correct_predictions / self.num_predictions
            if self.num_predictions > 0 else 0.0
        )

        total_bits = sum(bloom.size for bloom in self.filters)
        total_size_kb = total_bits / 8 / 1024

        return {
            'accuracy': accuracy,
            'num_predictions': self.num_predictions,
            'num_correct': self.num_correct_predictions,
            'num_accesses': self.num_accesses,
            'bloom_size_kb': total_size_kb,
            'num_filters': len(self.filters),
            'history_size': len(self.history)
        }

    def _periodic_reset(self):
        """Reset filters periodically to adapt to changing patterns."""
        print(f"🔄 Bloom filter periodic reset (adapting to new patterns)")

        # Keep some recent patterns by replaying recent history
        recent_history = self.history[-100:] if len(self.history) > 100 else self.history

        # Reset all filters
        for bloom in self.filters:
            bloom.reset()

        # Replay recent history
        replay_history = []
        for access in recent_history:
            layer_idx, block_idx = access

            # Update filters
            for i, prev in enumerate(reversed(replay_history)):
                if i >= self.config.lookahead:
                    break
                pattern = (prev, access)
                self.filters[i].add(pattern)

            replay_history.append(access)

        self.last_reset = time.time()

    def reset(self):
        """Fully reset prefetcher state."""
        for bloom in self.filters:
            bloom.reset()

        self.history.clear()
        self.num_predictions = 0
        self.num_correct_predictions = 0
        self.num_accesses = 0
        self.last_reset = time.time()

        print("✓ Bloom prefetcher reset")


class AdaptiveBloomPrefetcher(BloomPrefetcher):
    """
    Adaptive Bloom prefetcher with dynamic lookahead adjustment.

    Automatically adjusts lookahead distance based on prediction accuracy.
    """

    def __init__(self, config: Optional[BloomConfig] = None):
        super().__init__(config)

        # Adaptive parameters
        self.target_accuracy = 0.80  # Target 80% accuracy
        self.adjustment_interval = 1000  # Adjust every N predictions
        self.min_lookahead = 2
        self.max_lookahead = 8

    def record_access(self, layer_idx: int, block_idx: int):
        """Record access and potentially adjust lookahead."""
        super().record_access(layer_idx, block_idx)

        # Periodically adjust lookahead based on accuracy
        if self.num_predictions > 0 and self.num_predictions % self.adjustment_interval == 0:
            self._adjust_lookahead()

    def _adjust_lookahead(self):
        """Adjust lookahead distance based on accuracy."""
        current_accuracy = (
            self.num_correct_predictions / self.num_predictions
            if self.num_predictions > 0 else 0.0
        )

        old_lookahead = self.config.lookahead

        if current_accuracy < self.target_accuracy - 0.1:
            # Accuracy too low, reduce lookahead (be more conservative)
            self.config.lookahead = max(
                self.min_lookahead,
                self.config.lookahead - 1
            )

            # Remove excess Bloom filters
            while len(self.filters) > self.config.lookahead:
                self.filters.pop()

        elif current_accuracy > self.target_accuracy + 0.1:
            # Accuracy very high, can increase lookahead (be more aggressive)
            self.config.lookahead = min(
                self.max_lookahead,
                self.config.lookahead + 1
            )

            # Add new Bloom filter
            while len(self.filters) < self.config.lookahead:
                bloom = BloomFilter(
                    capacity=self.config.capacity,
                    error_rate=self.config.error_rate
                )
                self.filters.append(bloom)

        if old_lookahead != self.config.lookahead:
            print(f"🔧 Adjusted lookahead: {old_lookahead} → {self.config.lookahead} (accuracy: {current_accuracy:.2%})")
