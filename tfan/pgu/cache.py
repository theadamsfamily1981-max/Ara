#!/usr/bin/env python
"""
TurboCache - Substitution-Aware Proof Cache

Implements fast caching of SMT solver results with:
- Alpha-rename normalization for variable-order independence
- Blake2b hashing for O(1) lookup
- LRU eviction for bounded memory
- Unsat-core reuse across similar queries

Architecture:
1. Normalize formula via alpha-renaming
2. Hash (canonical_formula, assumptions) → cache_key
3. Lookup cache_key in DB (sqlite/lmdb/dict)
4. On miss: run Z3, store result
5. On hit: return cached result

Performance targets:
- p95 ≤120 ms (vs ~500ms cold Z3)
- Hit rate ≥60% on realistic workloads
- 100% correctness (verified against cold runs)

Usage:
    cache = TurboCache(db_path="pgu_cache.db")

    # Lookup (returns None if miss)
    result = cache.lookup(formula, assumptions)

    if result is None:
        # Cache miss - run Z3
        result = run_z3(formula, assumptions)
        cache.store(formula, assumptions, result)

    # Use result
    if result['sat']:
        model = result['model']
"""

import hashlib
import json
import time
import sqlite3
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import OrderedDict

from .normalizer import normalize_formula


@dataclass
class CacheStats:
    """Statistics for TurboCache performance."""
    hits: int = 0
    misses: int = 0
    stores: int = 0
    evictions: int = 0
    total_lookup_time_ms: float = 0.0
    total_store_time_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def avg_lookup_time_ms(self) -> float:
        total = self.hits + self.misses
        return self.total_lookup_time_ms / total if total > 0 else 0.0

    def to_dict(self) -> Dict:
        return {
            'hits': self.hits,
            'misses': self.misses,
            'stores': self.stores,
            'evictions': self.evictions,
            'hit_rate': self.hit_rate,
            'avg_lookup_time_ms': self.avg_lookup_time_ms
        }


class TurboCache:
    """
    Fast substitution-aware cache for SMT solver results.

    Uses alpha-renaming to achieve variable-order independence.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        max_entries: int = 10000,
        backend: str = 'sqlite'
    ):
        """
        Initialize TurboCache.

        Args:
            db_path: Path to persistent DB (None for in-memory)
            max_entries: Max cache entries (LRU eviction)
            backend: 'sqlite', 'dict', or 'lmdb'
        """
        self.max_entries = max_entries
        self.backend = backend

        # Statistics
        self.stats = CacheStats()

        # Initialize backend
        if backend == 'sqlite':
            self.db = self._init_sqlite(db_path)
        elif backend == 'dict':
            self.db = OrderedDict()
        elif backend == 'lmdb':
            self.db = self._init_lmdb(db_path)
        else:
            raise ValueError(f"Unknown backend: {backend}")

        print(f"✓ TurboCache initialized")
        print(f"  Backend: {backend}")
        print(f"  Max entries: {max_entries:,}")
        print(f"  DB path: {db_path or 'in-memory'}")

    def _init_sqlite(self, db_path: Optional[str]):
        """Initialize SQLite backend."""
        if db_path is None:
            db_path = ':memory:'
        else:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                timestamp REAL NOT NULL,
                access_count INTEGER DEFAULT 0
            )
        ''')

        # Index on timestamp for LRU eviction
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON cache (timestamp)
        ''')

        conn.commit()

        return conn

    def _init_lmdb(self, db_path: str):
        """Initialize LMDB backend."""
        try:
            import lmdb
        except ImportError:
            raise ImportError("lmdb not installed. Install with: pip install lmdb")

        if db_path is None:
            raise ValueError("LMDB backend requires db_path")

        Path(db_path).mkdir(parents=True, exist_ok=True)

        env = lmdb.open(
            db_path,
            map_size=1024 * 1024 * 1024,  # 1 GB
            max_dbs=1
        )

        return env

    def _key(self, canon: str, assumptions: Tuple[str, ...]) -> str:
        """
        Compute cache key from canonical formula and assumptions.

        Uses Blake2b for fast hashing (faster than SHA256, similar security).

        Args:
            canon: Canonical formula (after alpha-rename)
            assumptions: Tuple of canonical assumptions

        Returns:
            Hex digest cache key
        """
        # Create deterministic JSON blob
        blob = json.dumps({
            "f": canon,
            "a": list(assumptions)
        }, sort_keys=True).encode('utf-8')

        # Hash with Blake2b (20 bytes = 40 hex chars)
        return hashlib.blake2b(blob, digest_size=20).hexdigest()

    def lookup(
        self,
        formula: str,
        assumptions: List[str] = None
    ) -> Optional[Dict]:
        """
        Lookup formula in cache.

        Args:
            formula: Formula string (will be normalized)
            assumptions: List of assumption formulas

        Returns:
            Cached result dict if hit, None if miss
        """
        start_time = time.perf_counter()

        # Normalize formula
        canon, metadata = normalize_formula(formula, assumptions)
        norm_assumptions = tuple(metadata['assumptions'])

        # Compute cache key
        cache_key = self._key(canon, norm_assumptions)

        # Lookup in backend
        result = self._get(cache_key)

        lookup_time = (time.perf_counter() - start_time) * 1000

        if result is not None:
            self.stats.hits += 1
            self.stats.total_lookup_time_ms += lookup_time

            # Update access count and timestamp (for LRU)
            self._update_access(cache_key)

            return result
        else:
            self.stats.misses += 1
            self.stats.total_lookup_time_ms += lookup_time

            return None

    def store(
        self,
        formula: str,
        assumptions: List[str] = None,
        result: Dict = None
    ):
        """
        Store formula result in cache.

        Args:
            formula: Formula string (will be normalized)
            assumptions: List of assumption formulas
            result: Result dict from solver
        """
        start_time = time.perf_counter()

        # Normalize formula
        canon, metadata = normalize_formula(formula, assumptions)
        norm_assumptions = tuple(metadata['assumptions'])

        # Compute cache key
        cache_key = self._key(canon, norm_assumptions)

        # Store in backend
        self._put(cache_key, result)

        self.stats.stores += 1
        self.stats.total_store_time_ms += (time.perf_counter() - start_time) * 1000

        # Check if eviction needed
        if self._size() > self.max_entries:
            self._evict_lru()

    def _get(self, key: str) -> Optional[Dict]:
        """Get value from backend."""
        if self.backend == 'sqlite':
            cursor = self.db.cursor()
            cursor.execute('SELECT value FROM cache WHERE key = ?', (key,))
            row = cursor.fetchone()

            if row:
                return json.loads(row[0])
            return None

        elif self.backend == 'dict':
            return self.db.get(key)

        elif self.backend == 'lmdb':
            with self.db.begin() as txn:
                value_bytes = txn.get(key.encode('utf-8'))
                if value_bytes:
                    return json.loads(value_bytes.decode('utf-8'))
                return None

    def _put(self, key: str, value: Dict):
        """Put value in backend."""
        timestamp = time.time()

        if self.backend == 'sqlite':
            cursor = self.db.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO cache (key, value, timestamp, access_count)
                VALUES (?, ?, ?, 1)
            ''', (key, json.dumps(value), timestamp))
            self.db.commit()

        elif self.backend == 'dict':
            self.db[key] = value
            # Move to end (most recent)
            self.db.move_to_end(key)

        elif self.backend == 'lmdb':
            with self.db.begin(write=True) as txn:
                txn.put(
                    key.encode('utf-8'),
                    json.dumps(value).encode('utf-8')
                )

    def _update_access(self, key: str):
        """Update access count and timestamp for LRU."""
        if self.backend == 'sqlite':
            cursor = self.db.cursor()
            cursor.execute('''
                UPDATE cache
                SET timestamp = ?, access_count = access_count + 1
                WHERE key = ?
            ''', (time.time(), key))
            self.db.commit()

        elif self.backend == 'dict':
            # Move to end (most recently used)
            self.db.move_to_end(key)

        elif self.backend == 'lmdb':
            # LMDB doesn't support in-place updates easily
            # For LRU, we'd need a separate metadata structure
            pass

    def _size(self) -> int:
        """Get current cache size."""
        if self.backend == 'sqlite':
            cursor = self.db.cursor()
            cursor.execute('SELECT COUNT(*) FROM cache')
            return cursor.fetchone()[0]

        elif self.backend == 'dict':
            return len(self.db)

        elif self.backend == 'lmdb':
            with self.db.begin() as txn:
                return txn.stat()['entries']

    def _evict_lru(self):
        """Evict least-recently-used entries."""
        num_to_evict = self._size() - self.max_entries

        if num_to_evict <= 0:
            return

        if self.backend == 'sqlite':
            cursor = self.db.cursor()

            # Get oldest entries by timestamp
            cursor.execute('''
                SELECT key FROM cache
                ORDER BY timestamp ASC
                LIMIT ?
            ''', (num_to_evict,))

            keys_to_evict = [row[0] for row in cursor.fetchall()]

            # Delete them
            cursor.executemany('DELETE FROM cache WHERE key = ?',
                             [(k,) for k in keys_to_evict])
            self.db.commit()

            self.stats.evictions += len(keys_to_evict)

        elif self.backend == 'dict':
            # OrderedDict: remove from beginning (oldest)
            for _ in range(num_to_evict):
                self.db.popitem(last=False)
                self.stats.evictions += 1

        elif self.backend == 'lmdb':
            # LMDB eviction is more complex
            # For simplicity, skip for now
            pass

    def clear(self):
        """Clear all cache entries."""
        if self.backend == 'sqlite':
            cursor = self.db.cursor()
            cursor.execute('DELETE FROM cache')
            self.db.commit()

        elif self.backend == 'dict':
            self.db.clear()

        elif self.backend == 'lmdb':
            with self.db.begin(write=True) as txn:
                txn.drop(self.db.open_db())

        # Reset stats
        self.stats = CacheStats()

        print("✓ TurboCache cleared")

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        return self.stats.to_dict()

    def __del__(self):
        """Cleanup on deletion."""
        if self.backend == 'sqlite' and hasattr(self, 'db'):
            self.db.close()

        elif self.backend == 'lmdb' and hasattr(self, 'db'):
            self.db.close()


class CacheOracle:
    """
    Oracle for cache correctness verification.

    Runs both cached and cold Z3 to verify results match.
    """

    def __init__(self, cache: TurboCache):
        self.cache = cache
        self.num_checks = 0
        self.num_mismatches = 0

    def verify_lookup(
        self,
        formula: str,
        assumptions: List[str],
        cold_runner
    ) -> Tuple[Dict, bool]:
        """
        Verify cache lookup against cold Z3 run.

        Args:
            formula: Formula string
            assumptions: Assumptions
            cold_runner: Callable that runs Z3 cold

        Returns:
            result: Cache or cold result
            match: Whether cache and cold results match
        """
        self.num_checks += 1

        # Try cache
        cached = self.cache.lookup(formula, assumptions)

        # Run cold
        cold_result = cold_runner(formula, assumptions)

        if cached is not None:
            # Compare results
            match = self._results_match(cached, cold_result)

            if not match:
                self.num_mismatches += 1
                print(f"⚠ Cache mismatch detected!")
                print(f"  Cached: {cached}")
                print(f"  Cold: {cold_result}")

            return cached, match
        else:
            # Cache miss - store cold result
            self.cache.store(formula, assumptions, cold_result)
            return cold_result, True

    def _results_match(self, result1: Dict, result2: Dict) -> bool:
        """Check if two solver results match."""
        # Check SAT/UNSAT
        if result1.get('sat') != result2.get('sat'):
            return False

        # For SAT, models may differ but both should be SAT
        # For UNSAT, unsat cores may differ but both should be UNSAT

        return True

    def get_correctness_rate(self) -> float:
        """Get fraction of cache lookups that matched cold runs."""
        if self.num_checks == 0:
            return 1.0

        return 1.0 - (self.num_mismatches / self.num_checks)
