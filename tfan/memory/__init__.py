"""
TF-A-N CXL/UMA Memory Tiering

This module provides CXL-aware memory tiering for extreme long-context inference:
- CXLPager: Multi-tier KV cache with NVMe/CXL as cold storage
- BloomPrefetcher: Bloom filter-based access pattern prediction
- UMAOptimizer: Unified memory architecture optimizations

Hard gates:
- 128k context without OOM on 24GB GPU
- ≤8% tokens/s penalty vs in-memory cache
- Cache hit-rate ≥90%
- Prefetch accuracy ≥80%
"""

__all__ = ['CXLPager', 'CXLPageConfig', 'BloomPrefetcher', 'BloomConfig', 'BloomFilter']


def __getattr__(name):
    """Lazy import to avoid requiring torch for bloom-only imports."""
    if name in ('CXLPager', 'CXLPageConfig'):
        from .cxl_pager import CXLPager, CXLPageConfig
        if name == 'CXLPager':
            return CXLPager
        return CXLPageConfig
    elif name in ('BloomPrefetcher', 'BloomConfig', 'BloomFilter'):
        from .bloom import BloomPrefetcher, BloomConfig, BloomFilter
        if name == 'BloomPrefetcher':
            return BloomPrefetcher
        elif name == 'BloomConfig':
            return BloomConfig
        return BloomFilter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
