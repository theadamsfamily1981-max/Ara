#!/usr/bin/env python
"""
KV Pager - File-Backed KV Cache Management

Implements memory-efficient KV cache with file-backed paging for long contexts.
Enables 128k+ context inference by storing cold cache blocks on disk/CXL memory.

Architecture:
1. Tiered Cache: Hot (GPU VRAM) → Warm (CPU RAM) → Cold (Disk/CXL)
2. LRU Eviction: Least-recently-used blocks evicted to lower tiers
3. Prefetching: Anticipate access patterns and prefetch cold blocks
4. Compression: Optional zstd compression for cold storage

Performance targets:
- 128k context without OOM on 24GB GPU
- Cache hit-rate ≥90%
- Prefetch accuracy ≥80%
- ≤8% throughput penalty vs in-memory cache

Usage:
    pager = KVPager(
        max_gpu_blocks=1024,
        max_cpu_blocks=4096,
        block_size=16,
        prefetch_strategy='sequential'
    )

    # Store KV cache
    pager.store_block(layer_idx, block_idx, key, value)

    # Retrieve KV cache
    key, value = pager.load_block(layer_idx, block_idx)
"""

import torch
import numpy as np
from typing import Optional, Tuple, Dict, List, Literal
from dataclasses import dataclass
from collections import OrderedDict
import tempfile
import mmap
import struct
import zstd
import time
from pathlib import Path


@dataclass
class KVPageConfig:
    """Configuration for KV cache paging."""
    max_gpu_blocks: int = 1024  # Max blocks in GPU VRAM
    max_cpu_blocks: int = 4096  # Max blocks in CPU RAM
    block_size: int = 16  # Tokens per block
    prefetch_strategy: Literal['none', 'sequential', 'landmark'] = 'sequential'
    use_compression: bool = True  # Compress cold storage
    compression_level: int = 3  # zstd compression level (1-22)
    profile_stats: bool = False  # Enable statistics tracking


@dataclass
class KVCacheStats:
    """Statistics for KV cache performance."""
    gpu_hits: int = 0
    cpu_hits: int = 0
    disk_hits: int = 0
    misses: int = 0
    evictions: int = 0
    prefetches: int = 0
    prefetch_hits: int = 0
    total_load_time_ms: float = 0.0
    total_store_time_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        total = self.gpu_hits + self.cpu_hits + self.disk_hits + self.misses
        if total == 0:
            return 0.0
        return (self.gpu_hits + self.cpu_hits + self.disk_hits) / total

    @property
    def prefetch_accuracy(self) -> float:
        if self.prefetches == 0:
            return 0.0
        return self.prefetch_hits / self.prefetches


class KVBlock:
    """Single KV cache block."""

    def __init__(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
        block_idx: int
    ):
        """
        Initialize KV block.

        Args:
            key: [num_heads, block_size, head_dim]
            value: [num_heads, block_size, head_dim]
            layer_idx: Transformer layer index
            block_idx: Block index within sequence
        """
        self.key = key
        self.value = value
        self.layer_idx = layer_idx
        self.block_idx = block_idx
        self.last_access = time.time()

    def to_bytes(self, compress: bool = False) -> bytes:
        """Serialize to bytes."""
        # Convert to numpy and serialize
        key_np = self.key.cpu().numpy()
        value_np = self.value.cpu().numpy()

        # Pack: layer_idx, block_idx, key_shape, value_shape, key_data, value_data
        header = struct.pack(
            'II',  # layer_idx, block_idx
            self.layer_idx,
            self.block_idx
        )

        key_bytes = key_np.tobytes()
        value_bytes = value_np.tobytes()

        # Shape metadata
        key_shape = struct.pack('III', *key_np.shape)
        value_shape = struct.pack('III', *value_np.shape)

        data = header + key_shape + value_shape + key_bytes + value_bytes

        if compress:
            data = zstd.compress(data)

        return data

    @classmethod
    def from_bytes(cls, data: bytes, compress: bool = False, device: str = 'cuda'):
        """Deserialize from bytes."""
        if compress:
            data = zstd.decompress(data)

        # Unpack header
        layer_idx, block_idx = struct.unpack('II', data[:8])
        offset = 8

        # Unpack shapes
        key_shape = struct.unpack('III', data[offset:offset+12])
        offset += 12
        value_shape = struct.unpack('III', data[offset:offset+12])
        offset += 12

        # Unpack arrays
        key_size = int(np.prod(key_shape)) * 4  # float32
        value_size = int(np.prod(value_shape)) * 4

        key_np = np.frombuffer(data[offset:offset+key_size], dtype=np.float32).reshape(key_shape)
        offset += key_size
        value_np = np.frombuffer(data[offset:offset+value_size], dtype=np.float32).reshape(value_shape)

        key = torch.from_numpy(key_np).to(device)
        value = torch.from_numpy(value_np).to(device)

        return cls(key, value, layer_idx, block_idx)


class KVPager:
    """
    File-backed KV cache manager with tiered storage.

    Implements LRU eviction and prefetching for efficient long-context inference.
    """

    def __init__(
        self,
        config: Optional[KVPageConfig] = None,
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize KV pager.

        Args:
            config: Paging configuration
            cache_dir: Directory for cold storage (default: temp dir)
        """
        self.config = config or KVPageConfig()

        # Tiered caches (LRU using OrderedDict)
        self.gpu_cache: OrderedDict[Tuple[int, int], KVBlock] = OrderedDict()
        self.cpu_cache: OrderedDict[Tuple[int, int], KVBlock] = OrderedDict()

        # Cold storage directory
        self.cache_dir = cache_dir or Path(tempfile.mkdtemp(prefix='kv_cache_'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Statistics
        self.stats = KVCacheStats()

        # Prefetch queue
        self.prefetch_queue: List[Tuple[int, int]] = []

        print(f"✓ KV Pager initialized")
        print(f"  GPU blocks: {self.config.max_gpu_blocks}")
        print(f"  CPU blocks: {self.config.max_cpu_blocks}")
        print(f"  Block size: {self.config.block_size} tokens")
        print(f"  Prefetch: {self.config.prefetch_strategy}")
        print(f"  Cache dir: {self.cache_dir}")

    def _get_cache_path(self, layer_idx: int, block_idx: int) -> Path:
        """Get file path for cold storage."""
        return self.cache_dir / f"l{layer_idx}_b{block_idx}.kv"

    def store_block(
        self,
        layer_idx: int,
        block_idx: int,
        key: torch.Tensor,
        value: torch.Tensor
    ):
        """
        Store KV block in cache.

        Args:
            layer_idx: Transformer layer index
            block_idx: Block index
            key: [num_heads, block_size, head_dim]
            value: [num_heads, block_size, head_dim]
        """
        start_time = time.perf_counter()

        block = KVBlock(key, value, layer_idx, block_idx)
        cache_key = (layer_idx, block_idx)

        # Store in GPU cache
        self.gpu_cache[cache_key] = block
        self.gpu_cache.move_to_end(cache_key)  # Mark as most recent

        # Evict if over capacity
        if len(self.gpu_cache) > self.config.max_gpu_blocks:
            self._evict_gpu_block()

        if self.config.profile_stats:
            self.stats.total_store_time_ms += (time.perf_counter() - start_time) * 1000

    def load_block(
        self,
        layer_idx: int,
        block_idx: int,
        device: str = 'cuda'
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Load KV block from cache.

        Args:
            layer_idx: Transformer layer index
            block_idx: Block index
            device: Target device

        Returns:
            (key, value) if found, None otherwise
        """
        start_time = time.perf_counter()

        cache_key = (layer_idx, block_idx)

        # Check GPU cache (hot)
        if cache_key in self.gpu_cache:
            block = self.gpu_cache[cache_key]
            self.gpu_cache.move_to_end(cache_key)  # LRU update
            self.stats.gpu_hits += 1

            if self.config.profile_stats:
                self.stats.total_load_time_ms += (time.perf_counter() - start_time) * 1000

            return block.key.to(device), block.value.to(device)

        # Check CPU cache (warm)
        if cache_key in self.cpu_cache:
            block = self.cpu_cache[cache_key]
            self.cpu_cache.move_to_end(cache_key)
            self.stats.cpu_hits += 1

            # Promote to GPU cache
            self.gpu_cache[cache_key] = block
            self.gpu_cache.move_to_end(cache_key)
            if len(self.gpu_cache) > self.config.max_gpu_blocks:
                self._evict_gpu_block()

            if self.config.profile_stats:
                self.stats.total_load_time_ms += (time.perf_counter() - start_time) * 1000

            return block.key.to(device), block.value.to(device)

        # Check disk/cold storage
        cache_path = self._get_cache_path(layer_idx, block_idx)
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                data = f.read()

            block = KVBlock.from_bytes(
                data,
                compress=self.config.use_compression,
                device=device
            )
            self.stats.disk_hits += 1

            # Promote to GPU cache
            self.gpu_cache[cache_key] = block
            self.gpu_cache.move_to_end(cache_key)
            if len(self.gpu_cache) > self.config.max_gpu_blocks:
                self._evict_gpu_block()

            if self.config.profile_stats:
                self.stats.total_load_time_ms += (time.perf_counter() - start_time) * 1000

            return block.key, block.value

        # Miss
        self.stats.misses += 1

        if self.config.profile_stats:
            self.stats.total_load_time_ms += (time.perf_counter() - start_time) * 1000

        return None

    def _evict_gpu_block(self):
        """Evict least-recently-used block from GPU to CPU cache."""
        if not self.gpu_cache:
            return

        # Pop LRU (first item in OrderedDict)
        cache_key, block = self.gpu_cache.popitem(last=False)

        # Move to CPU cache
        self.cpu_cache[cache_key] = block
        self.cpu_cache.move_to_end(cache_key)
        self.stats.evictions += 1

        # Evict from CPU if over capacity
        if len(self.cpu_cache) > self.config.max_cpu_blocks:
            self._evict_cpu_block()

    def _evict_cpu_block(self):
        """Evict least-recently-used block from CPU to disk."""
        if not self.cpu_cache:
            return

        # Pop LRU
        cache_key, block = self.cpu_cache.popitem(last=False)

        # Write to disk
        cache_path = self._get_cache_path(*cache_key)
        data = block.to_bytes(compress=self.config.use_compression)

        with open(cache_path, 'wb') as f:
            f.write(data)

    def prefetch(self, layer_idx: int, block_idx: int):
        """
        Prefetch block into cache.

        Args:
            layer_idx: Layer index
            block_idx: Block index
        """
        # Trigger async prefetch
        self.prefetch_queue.append((layer_idx, block_idx))
        self.stats.prefetches += 1

        # Actually load (in production, this would be async)
        result = self.load_block(layer_idx, block_idx)
        if result is not None:
            self.stats.prefetch_hits += 1

    def predict_prefetch(self, layer_idx: int, block_idx: int) -> List[Tuple[int, int]]:
        """
        Predict next blocks to prefetch.

        Args:
            layer_idx: Current layer
            block_idx: Current block

        Returns:
            List of (layer_idx, block_idx) to prefetch
        """
        if self.config.prefetch_strategy == 'none':
            return []

        elif self.config.prefetch_strategy == 'sequential':
            # Prefetch next 2 blocks in sequence
            return [
                (layer_idx, block_idx + 1),
                (layer_idx, block_idx + 2)
            ]

        elif self.config.prefetch_strategy == 'landmark':
            # Prefetch landmark blocks (would need landmark indices)
            # For now, prefetch next block
            return [(layer_idx, block_idx + 1)]

        return []

    def clear(self):
        """Clear all caches."""
        self.gpu_cache.clear()
        self.cpu_cache.clear()

        # Remove disk cache files
        for cache_file in self.cache_dir.glob('*.kv'):
            cache_file.unlink()

        print("✓ KV cache cleared")

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            'hit_rate': self.stats.hit_rate,
            'gpu_hits': self.stats.gpu_hits,
            'cpu_hits': self.stats.cpu_hits,
            'disk_hits': self.stats.disk_hits,
            'misses': self.stats.misses,
            'evictions': self.stats.evictions,
            'prefetch_accuracy': self.stats.prefetch_accuracy,
            'avg_load_time_ms': (
                self.stats.total_load_time_ms /
                max(1, self.stats.gpu_hits + self.stats.cpu_hits + self.stats.disk_hits)
            ) if self.config.profile_stats else 0.0
        }

    def __del__(self):
        """Cleanup on deletion."""
        # Clean up temp directory if we created it
        if hasattr(self, 'cache_dir'):
            try:
                for cache_file in self.cache_dir.glob('*.kv'):
                    cache_file.unlink()
                self.cache_dir.rmdir()
            except:
                pass
