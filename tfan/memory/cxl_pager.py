#!/usr/bin/env python
"""
CXL Pager - CXL/UMA Memory Tiering for KV Cache

Extends KV cache paging with CXL memory and NVMe as cold storage tiers.
Enables 128k+ context inference without OOM by intelligently tiering cache
across: GPU VRAM → CPU RAM → CXL Memory → NVMe SSD.

Architecture:
1. Multi-tier Cache: GPU → CPU → CXL → NVMe
2. Smart Eviction: LRU with access pattern awareness
3. Bloom Prefetching: Predict access patterns using Bloom filters
4. DMA Optimization: Direct memory access for CXL transfers
5. Compression: zstd for cold tiers to reduce bandwidth

Performance targets:
- 128k context without OOM (24GB GPU)
- ≤8% tokens/s penalty vs in-memory
- Cache hit-rate ≥90%
- Prefetch accuracy ≥80%

CXL Benefits:
- Load-store semantics (no PCIe overhead)
- Higher bandwidth than NVMe (~64 GB/s)
- Lower latency than disk (~300ns vs ~100μs)
- Transparent to application

Usage:
    pager = CXLPager(
        max_gpu_blocks=1024,
        max_cpu_blocks=4096,
        max_cxl_blocks=16384,  # CXL memory
        block_size=16,
        enable_bloom_prefetch=True
    )

    # Store/load KV blocks
    pager.store_block(layer_idx, block_idx, key, value)
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

from .bloom import BloomPrefetcher, BloomConfig


@dataclass
class CXLPageConfig:
    """Configuration for CXL-aware paging."""
    # Tier capacities
    max_gpu_blocks: int = 1024  # GPU VRAM
    max_cpu_blocks: int = 4096  # CPU RAM
    max_cxl_blocks: int = 16384  # CXL memory (if available)
    max_nvme_blocks: int = 65536  # NVMe SSD

    # Block configuration
    block_size: int = 16  # Tokens per block

    # Prefetching
    enable_bloom_prefetch: bool = True
    prefetch_lookahead: int = 4  # Blocks to prefetch ahead

    # Compression
    use_compression_cxl: bool = False  # CXL is fast, compression optional
    use_compression_nvme: bool = True  # NVMe slower, compress to save bandwidth
    compression_level: int = 3

    # CXL-specific
    cxl_device_path: Optional[str] = None  # Path to CXL device
    use_dma: bool = True  # Use DMA for CXL transfers

    # Statistics
    profile_stats: bool = False


@dataclass
class CXLCacheStats:
    """Statistics for CXL cache performance."""
    # Hit counts by tier
    gpu_hits: int = 0
    cpu_hits: int = 0
    cxl_hits: int = 0
    nvme_hits: int = 0
    misses: int = 0

    # Eviction counts
    gpu_evictions: int = 0
    cpu_evictions: int = 0
    cxl_evictions: int = 0

    # Prefetch stats
    prefetches: int = 0
    prefetch_hits: int = 0

    # Timing
    total_load_time_ms: float = 0.0
    total_store_time_ms: float = 0.0
    gpu_load_time_ms: float = 0.0
    cpu_load_time_ms: float = 0.0
    cxl_load_time_ms: float = 0.0
    nvme_load_time_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        total = self.gpu_hits + self.cpu_hits + self.cxl_hits + self.nvme_hits + self.misses
        if total == 0:
            return 0.0
        return (self.gpu_hits + self.cpu_hits + self.cxl_hits + self.nvme_hits) / total

    @property
    def prefetch_accuracy(self) -> float:
        if self.prefetches == 0:
            return 0.0
        return self.prefetch_hits / self.prefetches

    @property
    def tier_distribution(self) -> Dict[str, float]:
        """Distribution of hits across tiers."""
        total_hits = self.gpu_hits + self.cpu_hits + self.cxl_hits + self.nvme_hits
        if total_hits == 0:
            return {}

        return {
            'gpu': self.gpu_hits / total_hits,
            'cpu': self.cpu_hits / total_hits,
            'cxl': self.cxl_hits / total_hits,
            'nvme': self.nvme_hits / total_hits
        }


class CXLBlock:
    """KV cache block for CXL storage."""

    def __init__(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
        block_idx: int
    ):
        self.key = key
        self.value = value
        self.layer_idx = layer_idx
        self.block_idx = block_idx
        self.last_access = time.time()
        self.access_count = 0

    def to_bytes(self, compress: bool = False) -> bytes:
        """Serialize to bytes."""
        key_np = self.key.cpu().numpy()
        value_np = self.value.cpu().numpy()

        header = struct.pack('IIII',
            self.layer_idx,
            self.block_idx,
            self.access_count,
            int(self.last_access)
        )

        key_shape = struct.pack('III', *key_np.shape)
        value_shape = struct.pack('III', *value_np.shape)

        key_bytes = key_np.tobytes()
        value_bytes = value_np.tobytes()

        data = header + key_shape + value_shape + key_bytes + value_bytes

        if compress:
            data = zstd.compress(data, self.compression_level if hasattr(self, 'compression_level') else 3)

        return data

    @classmethod
    def from_bytes(cls, data: bytes, compress: bool = False, device: str = 'cuda'):
        """Deserialize from bytes."""
        if compress:
            data = zstd.decompress(data)

        layer_idx, block_idx, access_count, last_access_ts = struct.unpack('IIII', data[:16])
        offset = 16

        key_shape = struct.unpack('III', data[offset:offset+12])
        offset += 12
        value_shape = struct.unpack('III', data[offset:offset+12])
        offset += 12

        key_size = int(np.prod(key_shape)) * 4
        value_size = int(np.prod(value_shape)) * 4

        key_np = np.frombuffer(data[offset:offset+key_size], dtype=np.float32).reshape(key_shape)
        offset += key_size
        value_np = np.frombuffer(data[offset:offset+value_size], dtype=np.float32).reshape(value_shape)

        key = torch.from_numpy(key_np).to(device)
        value = torch.from_numpy(value_np).to(device)

        block = cls(key, value, layer_idx, block_idx)
        block.access_count = access_count
        block.last_access = float(last_access_ts)

        return block


class CXLPager:
    """
    CXL-aware KV cache pager with multi-tier storage.

    Tiers (in order of speed):
    1. GPU VRAM: Fastest, limited capacity (~24 GB)
    2. CPU RAM: Fast, larger capacity (~128 GB)
    3. CXL Memory: Medium, very large capacity (~512 GB)
    4. NVMe SSD: Slowest, huge capacity (~2 TB)
    """

    def __init__(
        self,
        config: Optional[CXLPageConfig] = None,
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize CXL pager.

        Args:
            config: CXL paging configuration
            cache_dir: Directory for NVMe cache
        """
        self.config = config or CXLPageConfig()

        # Tiered caches (LRU)
        self.gpu_cache: OrderedDict[Tuple[int, int], CXLBlock] = OrderedDict()
        self.cpu_cache: OrderedDict[Tuple[int, int], CXLBlock] = OrderedDict()
        self.cxl_cache: OrderedDict[Tuple[int, int], CXLBlock] = OrderedDict()

        # NVMe storage directory
        self.cache_dir = cache_dir or Path(tempfile.mkdtemp(prefix='cxl_cache_'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # CXL memory-mapped region (simulated with mmap for now)
        # In production, this would map to actual CXL device
        self.cxl_mmap = None
        if self.config.cxl_device_path:
            self._init_cxl_device()
        else:
            # Simulate CXL with memory-mapped file
            self._init_cxl_simulation()

        # Statistics
        self.stats = CXLCacheStats()

        # Bloom filter prefetcher
        self.bloom_prefetcher = None
        if self.config.enable_bloom_prefetch:
            bloom_config = BloomConfig(
                capacity=10000,
                error_rate=0.01,
                lookahead=self.config.prefetch_lookahead
            )
            self.bloom_prefetcher = BloomPrefetcher(bloom_config)

        print(f"✓ CXL Pager initialized")
        print(f"  GPU blocks: {self.config.max_gpu_blocks}")
        print(f"  CPU blocks: {self.config.max_cpu_blocks}")
        print(f"  CXL blocks: {self.config.max_cxl_blocks}")
        print(f"  NVMe blocks: {self.config.max_nvme_blocks}")
        print(f"  Block size: {self.config.block_size} tokens")
        print(f"  Bloom prefetch: {self.config.enable_bloom_prefetch}")
        print(f"  Cache dir: {self.cache_dir}")

    def _init_cxl_device(self):
        """Initialize real CXL device."""
        # In production, open CXL device and create memory mapping
        # For now, this is a placeholder
        print(f"⚠ CXL device support not implemented, using simulation")
        self._init_cxl_simulation()

    def _init_cxl_simulation(self):
        """Initialize CXL simulation using memory-mapped file."""
        # Create large memory-mapped file to simulate CXL memory
        cxl_file = self.cache_dir / 'cxl_memory.bin'

        # Size: max_cxl_blocks * estimated block size
        # Estimate ~256 KB per block (8 heads * 16 tokens * 64 dim * 4 bytes * 2 (K+V))
        cxl_size = self.config.max_cxl_blocks * 256 * 1024

        with open(cxl_file, 'wb') as f:
            f.write(b'\x00' * cxl_size)

        # Memory map the file
        self.cxl_file = open(cxl_file, 'r+b')
        self.cxl_mmap = mmap.mmap(self.cxl_file.fileno(), 0)

        print(f"  CXL simulation: {cxl_size / (1024**3):.2f} GB")

    def _get_nvme_path(self, layer_idx: int, block_idx: int) -> Path:
        """Get NVMe file path."""
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

        block = CXLBlock(key, value, layer_idx, block_idx)
        cache_key = (layer_idx, block_idx)

        # Store in GPU cache (hot tier)
        self.gpu_cache[cache_key] = block
        self.gpu_cache.move_to_end(cache_key)

        # Evict if over capacity
        if len(self.gpu_cache) > self.config.max_gpu_blocks:
            self._evict_gpu_block()

        # Update Bloom filter
        if self.bloom_prefetcher:
            self.bloom_prefetcher.record_access(layer_idx, block_idx)

        if self.config.profile_stats:
            self.stats.total_store_time_ms += (time.perf_counter() - start_time) * 1000

    def load_block(
        self,
        layer_idx: int,
        block_idx: int,
        device: str = 'cuda'
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Load KV block from cache with multi-tier fallback.

        Args:
            layer_idx: Transformer layer index
            block_idx: Block index
            device: Target device

        Returns:
            (key, value) if found, None otherwise
        """
        start_time = time.perf_counter()
        cache_key = (layer_idx, block_idx)

        # Tier 1: GPU cache (fastest)
        if cache_key in self.gpu_cache:
            tier_start = time.perf_counter()
            block = self.gpu_cache[cache_key]
            block.access_count += 1
            block.last_access = time.time()
            self.gpu_cache.move_to_end(cache_key)
            self.stats.gpu_hits += 1

            if self.config.profile_stats:
                self.stats.gpu_load_time_ms += (time.perf_counter() - tier_start) * 1000
                self.stats.total_load_time_ms += (time.perf_counter() - start_time) * 1000

            # Trigger prefetch if enabled
            if self.bloom_prefetcher:
                self._trigger_prefetch(layer_idx, block_idx)

            return block.key.to(device), block.value.to(device)

        # Tier 2: CPU cache
        if cache_key in self.cpu_cache:
            tier_start = time.perf_counter()
            block = self.cpu_cache[cache_key]
            block.access_count += 1
            block.last_access = time.time()
            self.cpu_cache.move_to_end(cache_key)
            self.stats.cpu_hits += 1

            # Promote to GPU
            self._promote_to_gpu(cache_key, block)

            if self.config.profile_stats:
                self.stats.cpu_load_time_ms += (time.perf_counter() - tier_start) * 1000
                self.stats.total_load_time_ms += (time.perf_counter() - start_time) * 1000

            if self.bloom_prefetcher:
                self._trigger_prefetch(layer_idx, block_idx)

            return block.key.to(device), block.value.to(device)

        # Tier 3: CXL cache
        if cache_key in self.cxl_cache:
            tier_start = time.perf_counter()
            block = self._load_from_cxl(cache_key, device)
            if block:
                self.stats.cxl_hits += 1

                # Promote to GPU
                self._promote_to_gpu(cache_key, block)

                if self.config.profile_stats:
                    self.stats.cxl_load_time_ms += (time.perf_counter() - tier_start) * 1000
                    self.stats.total_load_time_ms += (time.perf_counter() - start_time) * 1000

                if self.bloom_prefetcher:
                    self._trigger_prefetch(layer_idx, block_idx)

                return block.key, block.value

        # Tier 4: NVMe storage (slowest)
        nvme_path = self._get_nvme_path(layer_idx, block_idx)
        if nvme_path.exists():
            tier_start = time.perf_counter()

            with open(nvme_path, 'rb') as f:
                data = f.read()

            block = CXLBlock.from_bytes(
                data,
                compress=self.config.use_compression_nvme,
                device=device
            )
            self.stats.nvme_hits += 1

            # Promote to GPU
            self._promote_to_gpu(cache_key, block)

            if self.config.profile_stats:
                self.stats.nvme_load_time_ms += (time.perf_counter() - tier_start) * 1000
                self.stats.total_load_time_ms += (time.perf_counter() - start_time) * 1000

            if self.bloom_prefetcher:
                self._trigger_prefetch(layer_idx, block_idx)

            return block.key, block.value

        # Miss
        self.stats.misses += 1

        if self.config.profile_stats:
            self.stats.total_load_time_ms += (time.perf_counter() - start_time) * 1000

        return None

    def _promote_to_gpu(self, cache_key: Tuple[int, int], block: CXLBlock):
        """Promote block to GPU cache."""
        self.gpu_cache[cache_key] = block
        self.gpu_cache.move_to_end(cache_key)

        if len(self.gpu_cache) > self.config.max_gpu_blocks:
            self._evict_gpu_block()

    def _evict_gpu_block(self):
        """Evict LRU block from GPU to CPU."""
        if not self.gpu_cache:
            return

        cache_key, block = self.gpu_cache.popitem(last=False)
        self.stats.gpu_evictions += 1

        # Move to CPU
        self.cpu_cache[cache_key] = block
        self.cpu_cache.move_to_end(cache_key)

        if len(self.cpu_cache) > self.config.max_cpu_blocks:
            self._evict_cpu_block()

    def _evict_cpu_block(self):
        """Evict LRU block from CPU to CXL."""
        if not self.cpu_cache:
            return

        cache_key, block = self.cpu_cache.popitem(last=False)
        self.stats.cpu_evictions += 1

        # Move to CXL
        self._store_to_cxl(cache_key, block)

        if len(self.cxl_cache) > self.config.max_cxl_blocks:
            self._evict_cxl_block()

    def _evict_cxl_block(self):
        """Evict LRU block from CXL to NVMe."""
        if not self.cxl_cache:
            return

        cache_key, _ = self.cxl_cache.popitem(last=False)
        self.stats.cxl_evictions += 1

        # Load from CXL and write to NVMe
        block = self._load_from_cxl(cache_key, device='cpu')
        if block:
            nvme_path = self._get_nvme_path(*cache_key)
            data = block.to_bytes(compress=self.config.use_compression_nvme)

            with open(nvme_path, 'wb') as f:
                f.write(data)

    def _store_to_cxl(self, cache_key: Tuple[int, int], block: CXLBlock):
        """Store block to CXL memory."""
        # For simulation, just track in cxl_cache dict
        # In production, would write to CXL memory-mapped region
        self.cxl_cache[cache_key] = block
        self.cxl_cache.move_to_end(cache_key)

    def _load_from_cxl(self, cache_key: Tuple[int, int], device: str = 'cuda') -> Optional[CXLBlock]:
        """Load block from CXL memory."""
        # For simulation, just retrieve from cxl_cache dict
        # In production, would read from CXL memory-mapped region
        if cache_key in self.cxl_cache:
            block = self.cxl_cache[cache_key]
            block.access_count += 1
            block.last_access = time.time()
            self.cxl_cache.move_to_end(cache_key)
            return block
        return None

    def _trigger_prefetch(self, layer_idx: int, block_idx: int):
        """Trigger prefetch based on Bloom filter predictions."""
        if not self.bloom_prefetcher:
            return

        # Get predicted next blocks
        predictions = self.bloom_prefetcher.predict_next(layer_idx, block_idx)

        for pred_layer, pred_block in predictions:
            cache_key = (pred_layer, pred_block)

            # Only prefetch if not already in hot tiers
            if cache_key not in self.gpu_cache and cache_key not in self.cpu_cache:
                self.stats.prefetches += 1

                # Async prefetch (in production, this would be truly async)
                result = self.load_block(pred_layer, pred_block)
                if result is not None:
                    self.stats.prefetch_hits += 1

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        stats_dict = {
            'hit_rate': self.stats.hit_rate,
            'tier_distribution': self.stats.tier_distribution,
            'gpu_hits': self.stats.gpu_hits,
            'cpu_hits': self.stats.cpu_hits,
            'cxl_hits': self.stats.cxl_hits,
            'nvme_hits': self.stats.nvme_hits,
            'misses': self.stats.misses,
            'prefetch_accuracy': self.stats.prefetch_accuracy,
        }

        if self.config.profile_stats:
            stats_dict.update({
                'avg_load_time_ms': self.stats.total_load_time_ms / max(1,
                    self.stats.gpu_hits + self.stats.cpu_hits +
                    self.stats.cxl_hits + self.stats.nvme_hits),
                'gpu_avg_load_ms': self.stats.gpu_load_time_ms / max(1, self.stats.gpu_hits),
                'cpu_avg_load_ms': self.stats.cpu_load_time_ms / max(1, self.stats.cpu_hits),
                'cxl_avg_load_ms': self.stats.cxl_load_time_ms / max(1, self.stats.cxl_hits),
                'nvme_avg_load_ms': self.stats.nvme_load_time_ms / max(1, self.stats.nvme_hits),
            })

        return stats_dict

    def clear(self):
        """Clear all caches."""
        self.gpu_cache.clear()
        self.cpu_cache.clear()
        self.cxl_cache.clear()

        # Clear NVMe files
        for nvme_file in self.cache_dir.glob('*.kv'):
            nvme_file.unlink()

        # Clear Bloom filter
        if self.bloom_prefetcher:
            self.bloom_prefetcher.reset()

        print("✓ CXL cache cleared")

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, 'cxl_mmap') and self.cxl_mmap:
            self.cxl_mmap.close()

        if hasattr(self, 'cxl_file') and self.cxl_file:
            self.cxl_file.close()

        # Clean up temp directory
        if hasattr(self, 'cache_dir'):
            try:
                for cache_file in self.cache_dir.glob('*'):
                    cache_file.unlink()
                self.cache_dir.rmdir()
            except:
                pass
