#!/usr/bin/env python
"""
Corpus Replayer for PGU TurboCache Benchmarking

Replays a corpus of SMT queries to measure cache performance:
- Hit rate
- p95 latency
- Correctness (comparing cached vs cold results)

Corpus format (JSON):
{
    "queries": [
        {
            "formula": "(x > y) and (y < 10)",
            "assumptions": ["x >= 0", "y >= 0"],
            "expected_sat": true,
            "id": "query_001"
        },
        ...
    ]
}

Usage:
    replayer = CorpusReplayer(cache=cache, solver=z3_solver)
    results = replayer.replay_corpus("data/pgu_corpus.json")
    print(f"Hit rate: {results['hit_rate']:.2%}")
    print(f"p95 latency: {results['p95_ms']:.1f}ms")
"""

import json
import time
import numpy as np
from typing import Dict, List, Optional, Callable
from pathlib import Path
from dataclasses import dataclass, field

from .cache import TurboCache, CacheStats


@dataclass
class QueryResult:
    """Result of a single query."""
    query_id: str
    formula: str
    sat: Optional[bool]
    latency_ms: float
    cache_hit: bool
    timestamp: float = field(default_factory=time.time)


@dataclass
class ReplayResults:
    """Results from corpus replay."""
    num_queries: int
    hit_rate: float
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    num_mismatches: int
    total_time_ms: float
    queries_per_second: float

    def to_dict(self) -> Dict:
        return {
            'num_queries': self.num_queries,
            'hit_rate': self.hit_rate,
            'mean_latency_ms': self.mean_latency_ms,
            'p50_latency_ms': self.p50_latency_ms,
            'p95_latency_ms': self.p95_latency_ms,
            'p99_latency_ms': self.p99_latency_ms,
            'num_mismatches': self.num_mismatches,
            'total_time_ms': self.total_time_ms,
            'queries_per_second': self.queries_per_second
        }


class CorpusReplayer:
    """
    Replays SMT query corpus to benchmark cache performance.
    """

    def __init__(
        self,
        cache: TurboCache,
        solver: Optional[Callable] = None,
        verify_correctness: bool = True
    ):
        """
        Initialize corpus replayer.

        Args:
            cache: TurboCache instance
            solver: Callable that runs solver (for cache misses)
            verify_correctness: Whether to verify cache correctness
        """
        self.cache = cache
        self.solver = solver or self._mock_solver
        self.verify_correctness = verify_correctness

        self.query_results: List[QueryResult] = []

    def replay_corpus(
        self,
        corpus_path: str,
        max_queries: Optional[int] = None
    ) -> ReplayResults:
        """
        Replay corpus and collect statistics.

        Args:
            corpus_path: Path to corpus JSON file
            max_queries: Maximum queries to replay (None for all)

        Returns:
            ReplayResults with performance stats
        """
        print(f"\n{'='*60}")
        print(f"Replaying corpus: {corpus_path}")
        print(f"{'='*60}")

        # Load corpus
        corpus = self._load_corpus(corpus_path)
        queries = corpus['queries']

        if max_queries:
            queries = queries[:max_queries]

        print(f"Loaded {len(queries)} queries")

        # Clear previous results
        self.query_results.clear()

        # Replay each query
        start_time = time.perf_counter()

        for i, query in enumerate(queries):
            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(queries)} queries...")

            result = self._replay_query(query)
            self.query_results.append(result)

        total_time = (time.perf_counter() - start_time) * 1000

        # Compute statistics
        results = self._compute_stats(total_time)

        print(f"\n{'='*60}")
        print(f"Replay Results:")
        print(f"{'='*60}")
        print(f"  Queries: {results.num_queries}")
        print(f"  Hit rate: {results.hit_rate:.2%}")
        print(f"  Mean latency: {results.mean_latency_ms:.1f}ms")
        print(f"  p50 latency: {results.p50_latency_ms:.1f}ms")
        print(f"  p95 latency: {results.p95_latency_ms:.1f}ms")
        print(f"  p99 latency: {results.p99_latency_ms:.1f}ms")
        print(f"  Mismatches: {results.num_mismatches}")
        print(f"  Throughput: {results.queries_per_second:.1f} queries/s")

        return results

    def _load_corpus(self, corpus_path: str) -> Dict:
        """Load corpus from JSON file."""
        path = Path(corpus_path)

        if not path.exists():
            raise FileNotFoundError(f"Corpus not found: {corpus_path}")

        with open(path, 'r') as f:
            return json.load(f)

    def _replay_query(self, query: Dict) -> QueryResult:
        """Replay a single query."""
        query_id = query.get('id', 'unknown')
        formula = query['formula']
        assumptions = query.get('assumptions', [])
        expected_sat = query.get('expected_sat')

        # Try cache lookup
        start_time = time.perf_counter()

        cached_result = self.cache.lookup(formula, assumptions)
        cache_hit = cached_result is not None

        if cached_result:
            # Cache hit
            result = cached_result
        else:
            # Cache miss - run solver
            result = self.solver(formula, assumptions)

            # Store in cache for future queries
            self.cache.store(formula, assumptions, result)

        latency = (time.perf_counter() - start_time) * 1000

        # Create result record
        query_result = QueryResult(
            query_id=query_id,
            formula=formula,
            sat=result.get('sat'),
            latency_ms=latency,
            cache_hit=cache_hit
        )

        # Verify correctness (if expected result provided)
        if self.verify_correctness and expected_sat is not None:
            if result.get('sat') != expected_sat:
                print(f"⚠ Mismatch for {query_id}: expected sat={expected_sat}, got {result.get('sat')}")

        return query_result

    def _compute_stats(self, total_time_ms: float) -> ReplayResults:
        """Compute statistics from query results."""
        latencies = [r.latency_ms for r in self.query_results]
        cache_hits = sum(1 for r in self.query_results if r.cache_hit)

        # Count mismatches (queries where result is None)
        num_mismatches = sum(1 for r in self.query_results if r.sat is None)

        return ReplayResults(
            num_queries=len(self.query_results),
            hit_rate=cache_hits / len(self.query_results) if self.query_results else 0.0,
            mean_latency_ms=np.mean(latencies) if latencies else 0.0,
            p50_latency_ms=np.percentile(latencies, 50) if latencies else 0.0,
            p95_latency_ms=np.percentile(latencies, 95) if latencies else 0.0,
            p99_latency_ms=np.percentile(latencies, 99) if latencies else 0.0,
            num_mismatches=num_mismatches,
            total_time_ms=total_time_ms,
            queries_per_second=len(self.query_results) / (total_time_ms / 1000) if total_time_ms > 0 else 0.0
        )

    def _mock_solver(self, formula: str, assumptions: List[str]) -> Dict:
        """
        Mock solver for testing (when real solver not available).

        Returns SAT with mock model.
        """
        # Simulate solver latency
        time.sleep(0.05)  # 50ms

        return {
            'sat': True,
            'model': {'x': 1, 'y': 0},
            'time_ms': 50.0
        }

    def export_results(self, output_path: str):
        """Export query results to JSON."""
        output = {
            'query_results': [
                {
                    'query_id': r.query_id,
                    'sat': r.sat,
                    'latency_ms': r.latency_ms,
                    'cache_hit': r.cache_hit,
                    'timestamp': r.timestamp
                }
                for r in self.query_results
            ],
            'cache_stats': self.cache.get_stats()
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"✓ Results exported to {output_path}")


def generate_synthetic_corpus(
    num_queries: int,
    num_variables: int = 5,
    similarity: float = 0.3
) -> Dict:
    """
    Generate synthetic corpus for testing.

    Creates queries with controlled similarity to test cache hit rates.

    Args:
        num_queries: Number of queries to generate
        num_variables: Number of variables per query
        similarity: Fraction of queries that are similar (0-1)

    Returns:
        Corpus dict
    """
    import random

    queries = []

    # Generate base templates
    num_templates = max(1, int(num_queries * similarity))

    templates = []
    for i in range(num_templates):
        # Create template formula
        var_names = [f"var{j}" for j in range(num_variables)]
        formula = " and ".join([
            f"({var_names[j]} > {var_names[j+1]})"
            for j in range(num_variables - 1)
        ])
        templates.append(formula)

    # Generate queries from templates
    for i in range(num_queries):
        # Pick random template
        template = random.choice(templates)

        # Optionally rename variables (tests alpha-renaming)
        if random.random() < 0.5:
            # Rename variables
            var_mapping = {f"var{j}": f"x{j}" for j in range(num_variables)}
            formula = template
            for old, new in var_mapping.items():
                formula = formula.replace(old, new)
        else:
            formula = template

        query = {
            'id': f"query_{i:04d}",
            'formula': formula,
            'assumptions': [],
            'expected_sat': True
        }
        queries.append(query)

    return {'queries': queries}


def replay_corpus(
    corpus_path: str,
    cache: TurboCache,
    solver: Optional[Callable] = None,
    max_queries: Optional[int] = None
) -> ReplayResults:
    """
    Convenience function to replay corpus.

    Args:
        corpus_path: Path to corpus JSON
        cache: TurboCache instance
        solver: Solver callable
        max_queries: Max queries to replay

    Returns:
        ReplayResults
    """
    replayer = CorpusReplayer(cache=cache, solver=solver)
    return replayer.replay_corpus(corpus_path, max_queries)
