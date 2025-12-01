"""
TF-A-N Production Serving Infrastructure

This module provides production-ready serving components for TF-A-N models:
- SSARunner: O(N log N) selective sparse attention for long contexts
- KVPager: File-backed KV cache management
- TTWHook: Tripwire hook for VFE-based alignment triggering

Hard gates:
- 128k prefill ≥3× faster than dense baseline (RTX 3090)
- p99 latency under SLO (streaming tokens/s)
- KV cache hit-rate ≥90%
"""

from .ssa_runner import SSARunner
from .kv_pager import KVPager
from .ttw_hook import TTWHook

__all__ = ['SSARunner', 'KVPager', 'TTWHook']
