"""
TF-A-N PGU (Proof Generation Unit) with TurboCache

Substitution-aware caching for SMT solver results with:
- Alpha-rename normalization for variable-order independence
- Unsat-core reuse across similar queries
- Blake2b hashing for fast cache lookup

Hard gates:
- p95 ≤120 ms on replayed updates
- Cache hit-rate ≥60% on PGU corpus
- Correctness parity with cold Z3 run (0 mismatches)
"""

from .normalizer import alpha_rename, normalize_formula
from .cache import TurboCache, CacheStats
from .corpus_replay import CorpusReplayer, replay_corpus, generate_synthetic_corpus

__all__ = [
    'alpha_rename',
    'normalize_formula',
    'TurboCache',
    'CacheStats',
    'CorpusReplayer',
    'replay_corpus',
    'generate_synthetic_corpus'
]
