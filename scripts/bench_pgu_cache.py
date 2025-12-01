#!/usr/bin/env python
"""
Benchmark PGU TurboCache

Tests TurboCache against hard gates:
- p95 ≤120 ms on replayed updates
- Cache hit-rate ≥60% on PGU corpus
- Correctness parity with cold Z3 run (0 mismatches)

Usage:
    # Benchmark with synthetic corpus
    python scripts/bench_pgu_cache.py \
        --num-queries 1000 \
        --similarity 0.7

    # Benchmark with real corpus
    python scripts/bench_pgu_cache.py \
        --corpus data/pgu_corpus.json \
        --timeout-ms 120

    # Compare with baseline (no cache)
    python scripts/bench_pgu_cache.py \
        --corpus data/pgu_corpus.json \
        --include-baseline
"""

import argparse
import json
import time
import sys
from pathlib import Path

# Add tfan to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tfan.pgu import TurboCache, CorpusReplayer, generate_synthetic_corpus


def mock_z3_solver(formula: str, assumptions: list) -> dict:
    """
    Mock Z3 solver for testing.

    In production, this would call actual Z3.
    For benchmarking, we simulate latency.
    """
    # Simulate Z3 latency (200-500ms)
    import random
    latency_ms = random.uniform(200, 500)
    time.sleep(latency_ms / 1000)

    # Return mock SAT result
    return {
        'sat': True,
        'model': {'x1': 1, 'x2': 0},
        'time_ms': latency_ms,
        'solver': 'mock_z3'
    }


def benchmark_with_cache(
    corpus_path: str,
    cache: TurboCache,
    solver,
    max_queries: int = None
) -> dict:
    """
    Benchmark with TurboCache enabled.

    Args:
        corpus_path: Path to corpus
        cache: TurboCache instance
        solver: Solver callable
        max_queries: Max queries to process

    Returns:
        Results dict
    """
    print(f"\n{'='*60}")
    print("Benchmarking WITH TurboCache")
    print(f"{'='*60}")

    replayer = CorpusReplayer(cache=cache, solver=solver)
    results = replayer.replay_corpus(corpus_path, max_queries)

    return results.to_dict()


def benchmark_baseline(
    corpus_path: str,
    solver,
    max_queries: int = None
) -> dict:
    """
    Benchmark baseline (no cache).

    Args:
        corpus_path: Path to corpus
        solver: Solver callable
        max_queries: Max queries to process

    Returns:
        Results dict
    """
    print(f"\n{'='*60}")
    print("Benchmarking BASELINE (no cache)")
    print(f"{'='*60}")

    # Load corpus
    with open(corpus_path, 'r') as f:
        corpus = json.load(f)

    queries = corpus['queries']
    if max_queries:
        queries = queries[:max_queries]

    print(f"Loaded {len(queries)} queries")

    latencies = []
    start_time = time.perf_counter()

    for i, query in enumerate(queries):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(queries)} queries...")

        formula = query['formula']
        assumptions = query.get('assumptions', [])

        # Run solver (no cache)
        query_start = time.perf_counter()
        result = solver(formula, assumptions)
        latency = (time.perf_counter() - query_start) * 1000

        latencies.append(latency)

    total_time = (time.perf_counter() - start_time) * 1000

    # Compute stats
    import numpy as np

    return {
        'num_queries': len(queries),
        'mean_latency_ms': np.mean(latencies),
        'p95_latency_ms': np.percentile(latencies, 95),
        'p99_latency_ms': np.percentile(latencies, 99),
        'total_time_ms': total_time,
        'queries_per_second': len(queries) / (total_time / 1000)
    }


def check_gates(cached_results: dict, baseline_results: dict = None) -> dict:
    """
    Check if hard gates are met.

    Gates:
    - p95 ≤120 ms
    - Cache hit-rate ≥60%
    - Correctness: 0 mismatches

    Args:
        cached_results: Results with cache
        baseline_results: Results without cache (optional)

    Returns:
        Gate results dict
    """
    gates = {}

    # Gate 1: p95 latency
    p95 = float(cached_results['p95_latency_ms'])
    gates['p95_latency'] = {
        'value': p95,
        'threshold': 120.0,
        'pass': bool(p95 <= 120.0)
    }

    status = '✓' if p95 <= 120.0 else '✗'
    print(f"\n{status} p95 latency: {p95:.1f}ms (target: ≤120ms)")

    # Gate 2: Hit rate
    hit_rate = float(cached_results['hit_rate'])
    gates['hit_rate'] = {
        'value': hit_rate,
        'threshold': 0.60,
        'pass': bool(hit_rate >= 0.60)
    }

    status = '✓' if hit_rate >= 0.60 else '✗'
    print(f"{status} Hit rate: {hit_rate:.2%} (target: ≥60%)")

    # Gate 3: Correctness
    mismatches = int(cached_results.get('num_mismatches', 0))
    gates['correctness'] = {
        'value': mismatches,
        'threshold': 0,
        'pass': bool(mismatches == 0)
    }

    status = '✓' if mismatches == 0 else '✗'
    print(f"{status} Mismatches: {mismatches} (target: 0)")

    # Speedup vs baseline (if available)
    if baseline_results:
        baseline_qps = float(baseline_results['queries_per_second'])
        cached_qps = float(cached_results['queries_per_second'])
        speedup = cached_qps / baseline_qps

        gates['speedup'] = {
            'value': float(speedup),
            'threshold': 2.0,  # Expect at least 2× speedup
            'pass': bool(speedup >= 2.0)
        }

        status = '✓' if speedup >= 2.0 else '✗'
        print(f"{status} Speedup: {speedup:.2f}× (baseline: {baseline_qps:.1f} q/s, cached: {cached_qps:.1f} q/s)")

    # Overall pass
    all_pass = all(g['pass'] for g in gates.values())
    gates['overall'] = {'pass': bool(all_pass)}

    return gates


def main():
    parser = argparse.ArgumentParser(description="Benchmark PGU TurboCache")
    parser.add_argument('--corpus', type=str, help="Path to corpus JSON")
    parser.add_argument('--num-queries', type=int, default=1000, help="Number of synthetic queries")
    parser.add_argument('--similarity', type=float, default=0.7, help="Query similarity (0-1)")
    parser.add_argument('--max-queries', type=int, help="Max queries to process")
    parser.add_argument('--timeout-ms', type=int, default=120, help="p95 timeout threshold")
    parser.add_argument('--include-baseline', action='store_true', help="Include baseline benchmark")
    parser.add_argument('--backend', type=str, default='dict', choices=['dict', 'sqlite', 'lmdb'],
                        help="Cache backend")
    parser.add_argument('--max-entries', type=int, default=10000, help="Max cache entries")
    parser.add_argument('--output', type=str, default='pgu_benchmark_results.json', help="Output file")

    args = parser.parse_args()

    # Prepare corpus
    if args.corpus:
        corpus_path = args.corpus

        if not Path(corpus_path).exists():
            print(f"✗ Corpus not found: {corpus_path}")
            print(f"Generating synthetic corpus instead...")
            corpus = generate_synthetic_corpus(
                num_queries=args.num_queries,
                similarity=args.similarity
            )

            # Save synthetic corpus
            corpus_path = 'data/pgu_corpus_synthetic.json'
            Path(corpus_path).parent.mkdir(parents=True, exist_ok=True)

            with open(corpus_path, 'w') as f:
                json.dump(corpus, f, indent=2)

            print(f"✓ Synthetic corpus saved to {corpus_path}")
    else:
        # Generate synthetic corpus
        print(f"Generating synthetic corpus...")
        corpus = generate_synthetic_corpus(
            num_queries=args.num_queries,
            similarity=args.similarity
        )

        corpus_path = 'data/pgu_corpus_synthetic.json'
        Path(corpus_path).parent.mkdir(parents=True, exist_ok=True)

        with open(corpus_path, 'w') as f:
            json.dump(corpus, f, indent=2)

        print(f"✓ Synthetic corpus saved to {corpus_path}")

    # Initialize cache
    db_path = None if args.backend == 'dict' else f'pgu_cache.{args.backend}'

    cache = TurboCache(
        db_path=db_path,
        max_entries=args.max_entries,
        backend=args.backend
    )

    # Benchmark with cache
    cached_results = benchmark_with_cache(
        corpus_path=corpus_path,
        cache=cache,
        solver=mock_z3_solver,
        max_queries=args.max_queries
    )

    # Benchmark baseline (if requested)
    baseline_results = None
    if args.include_baseline:
        baseline_results = benchmark_baseline(
            corpus_path=corpus_path,
            solver=mock_z3_solver,
            max_queries=args.max_queries
        )

    # Check gates
    gates = check_gates(cached_results, baseline_results)
    cached_results['gates'] = gates

    # Save results
    output = {
        'cached': cached_results,
        'baseline': baseline_results,
        'config': {
            'backend': args.backend,
            'max_entries': args.max_entries,
            'corpus_path': corpus_path,
            'max_queries': args.max_queries
        }
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✓ Benchmark complete. Results saved to {output_path}")
    print(f"{'='*60}")

    # Exit with appropriate code
    if gates['overall']['pass']:
        print("\n✓ All gates PASSED")
        sys.exit(0)
    else:
        print("\n✗ Some gates FAILED")
        sys.exit(1)


if __name__ == '__main__':
    main()
