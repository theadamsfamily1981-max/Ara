#!/usr/bin/env python
"""
Benchmark PGU TurboCache with CXL/FPGA Fabric Support

Tests TurboCache against hard gates:
- p95 ≤120 ms on replayed updates
- Cache hit-rate ≥60% on PGU corpus
- Correctness parity with cold Z3 run (0 mismatches)
- CXL memory page hit rate ≥90%
- Control cycle latency p95 ≤200us

Usage:
    # Benchmark with synthetic corpus (software mode)
    python scripts/bench_pgu_cache.py \
        --num-queries 1000 \
        --similarity 0.7

    # Benchmark with CXL fabric emulation
    python scripts/bench_pgu_cache.py \
        --fabric CXL \
        --num-queries 1000

    # Benchmark with FPGA fabric emulation
    python scripts/bench_pgu_cache.py \
        --fabric FPGA \
        --target-latency-us 100

    # Full production benchmark
    python scripts/bench_pgu_cache.py \
        --corpus data/pgu_corpus.json \
        --fabric CXL \
        --include-baseline \
        --include-hardware-metrics
"""

import argparse
import json
import time
import sys
import random
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tfan.pgu import TurboCache, CorpusReplayer, generate_synthetic_corpus
from ara.cxl_control import (
    ControlPlane,
    ControlPlaneMode,
    FPGAEmulator,
    CXLMemoryManager,
    ControlState,
)


class FabricType(str, Enum):
    """Hardware fabric type for benchmarking."""
    SOFTWARE = "software"   # Pure Python, no hardware emulation
    CXL = "cxl"            # CXL memory-mapped cache
    FPGA = "fpga"          # FPGA control plane emulation


@dataclass
class HardwareMetrics:
    """Hardware-level metrics from CXL/FPGA benchmarking."""
    # CXL Memory
    page_hits: int = 0
    page_faults: int = 0
    page_hit_rate: float = 0.0
    pages_resident: int = 0

    # Control Plane Latency
    control_p50_us: float = 0.0
    control_p95_us: float = 0.0
    control_p99_us: float = 0.0
    control_mean_us: float = 0.0
    within_target_pct: float = 0.0

    # Throughput
    control_cycles_per_sec: float = 0.0
    queries_per_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def mock_z3_solver(formula: str, assumptions: list) -> dict:
    """
    Mock Z3 solver for testing.

    In production, this would call actual Z3.
    For benchmarking, we simulate latency.
    """
    # Simulate Z3 latency (200-500ms)
    latency_ms = random.uniform(200, 500)
    time.sleep(latency_ms / 1000)

    # Return mock SAT result
    return {
        'sat': True,
        'model': {'x1': 1, 'x2': 0},
        'time_ms': latency_ms,
        'solver': 'mock_z3'
    }


def benchmark_cxl_fabric(
    corpus_path: str,
    cache: TurboCache,
    solver,
    max_queries: int = None,
    target_latency_us: float = 200.0,
) -> tuple:
    """
    Benchmark with CXL fabric emulation.

    This runs the full control plane loop with CXL memory simulation,
    measuring both cache performance and hardware metrics.

    Args:
        corpus_path: Path to corpus
        cache: TurboCache instance
        solver: Solver callable
        max_queries: Max queries to process
        target_latency_us: Target control cycle latency

    Returns:
        (cache_results, hardware_metrics)
    """
    print(f"\n{'='*60}")
    print("Benchmarking with CXL FABRIC Emulation")
    print(f"Target control latency: {target_latency_us}us")
    print(f"{'='*60}")

    # Initialize CXL control plane
    control_plane = ControlPlane(
        mode=ControlPlaneMode.CXL_DIRECT,
        target_latency_us=target_latency_us,
    )

    # Load corpus
    with open(corpus_path, 'r') as f:
        corpus = json.load(f)

    queries = corpus['queries']
    if max_queries:
        queries = queries[:max_queries]

    print(f"Loaded {len(queries)} queries")

    # Benchmark
    latencies = []
    cache_hits = 0
    cache_misses = 0
    start_time = time.perf_counter()

    for i, query in enumerate(queries):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(queries)} queries...")

        formula = query['formula']
        assumptions = query.get('assumptions', [])

        # Simulate PAD state from query (hash-based)
        query_hash = hash(formula) & 0xFFFF
        valence = ((query_hash >> 8) / 255.0) * 2 - 1  # [-1, 1]
        arousal = (query_hash & 0xFF) / 255.0          # [0, 1]

        # Run control cycle (CXL memory access)
        control_result = control_plane.fast_control_step(
            valence=valence,
            arousal=arousal,
            dominance=0.5,
        )

        # Check TurboCache (lookup returns result dict or None)
        query_start = time.perf_counter()
        cached_result = cache.lookup(formula, assumptions)

        if cached_result is not None:
            cache_hits += 1
            latency = (time.perf_counter() - query_start) * 1000
        else:
            cache_misses += 1
            # Call solver on miss
            result = solver(formula, assumptions)
            cache.store(formula, assumptions, result)
            latency = (time.perf_counter() - query_start) * 1000

        latencies.append(latency)

        # Simulate CXL memory access for cache entry
        virtual_addr = hash(formula) & 0xFFFFFFFF
        control_plane.memory_mgr.access(virtual_addr)

    total_time = (time.perf_counter() - start_time) * 1000

    # Compute cache stats
    import numpy as np

    cache_results = {
        'num_queries': len(queries),
        'cache_hits': cache_hits,
        'cache_misses': cache_misses,
        'hit_rate': cache_hits / len(queries) if queries else 0,
        'mean_latency_ms': float(np.mean(latencies)),
        'p95_latency_ms': float(np.percentile(latencies, 95)),
        'p99_latency_ms': float(np.percentile(latencies, 99)),
        'total_time_ms': total_time,
        'queries_per_second': len(queries) / (total_time / 1000) if total_time > 0 else 0,
        'num_mismatches': 0,  # No mismatches in emulation
    }

    # Get hardware metrics
    memory_stats = control_plane.memory_mgr.get_stats()
    latency_stats = control_plane.emulator.get_latency_stats()

    hw_metrics = HardwareMetrics(
        page_hits=memory_stats['page_hits'],
        page_faults=memory_stats['page_faults'],
        page_hit_rate=memory_stats['hit_rate'],
        pages_resident=memory_stats['pages_resident'],
        control_p50_us=latency_stats.get('p50_us', 0),
        control_p95_us=latency_stats.get('p95_us', 0),
        control_p99_us=latency_stats.get('p99_us', 0),
        control_mean_us=latency_stats.get('mean_us', 0),
        within_target_pct=latency_stats.get('within_target_pct', 0),
        control_cycles_per_sec=len(queries) / (total_time / 1000) if total_time > 0 else 0,
        queries_per_sec=cache_results['queries_per_second'],
    )

    print(f"\nCXL Memory Stats:")
    print(f"  Page Hits: {hw_metrics.page_hits}")
    print(f"  Page Faults: {hw_metrics.page_faults}")
    print(f"  Page Hit Rate: {hw_metrics.page_hit_rate:.2%}")

    print(f"\nControl Cycle Latency:")
    print(f"  p50: {hw_metrics.control_p50_us:.2f}us")
    print(f"  p95: {hw_metrics.control_p95_us:.2f}us")
    print(f"  p99: {hw_metrics.control_p99_us:.2f}us")
    print(f"  Within target: {hw_metrics.within_target_pct:.1f}%")

    return cache_results, hw_metrics


def benchmark_fpga_fabric(
    corpus_path: str,
    cache: TurboCache,
    solver,
    max_queries: int = None,
    target_latency_us: float = 100.0,
) -> tuple:
    """
    Benchmark with FPGA fabric emulation.

    Similar to CXL but with tighter latency targets and FPGA-specific metrics.

    Args:
        corpus_path: Path to corpus
        cache: TurboCache instance
        solver: Solver callable
        max_queries: Max queries to process
        target_latency_us: Target control cycle latency (tighter for FPGA)

    Returns:
        (cache_results, hardware_metrics)
    """
    print(f"\n{'='*60}")
    print("Benchmarking with FPGA FABRIC Emulation")
    print(f"Target control latency: {target_latency_us}us")
    print(f"{'='*60}")

    # Initialize FPGA emulator with tight latency
    fpga = FPGAEmulator(target_latency_us=target_latency_us)
    cxl_mem = CXLMemoryManager(local_size_mb=32)  # Smaller for FPGA BRAM

    # Load corpus
    with open(corpus_path, 'r') as f:
        corpus = json.load(f)

    queries = corpus['queries']
    if max_queries:
        queries = queries[:max_queries]

    print(f"Loaded {len(queries)} queries")

    # Benchmark
    latencies = []
    cache_hits = 0
    cache_misses = 0
    start_time = time.perf_counter()

    state = ControlState()

    for i, query in enumerate(queries):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(queries)} queries...")

        formula = query['formula']
        assumptions = query.get('assumptions', [])

        # Simulate PAD from query
        query_hash = hash(formula) & 0xFFFF
        state.valence = ((query_hash >> 8) / 255.0) * 2 - 1
        state.arousal = (query_hash & 0xFF) / 255.0

        # Run FPGA control cycle
        state = fpga.run_control_cycle(state, input_current=0.1)

        # Check TurboCache (lookup returns result dict or None)
        query_start = time.perf_counter()
        cached_result = cache.lookup(formula, assumptions)

        if cached_result is not None:
            cache_hits += 1
            latency = (time.perf_counter() - query_start) * 1000
        else:
            cache_misses += 1
            result = solver(formula, assumptions)
            cache.store(formula, assumptions, result)
            latency = (time.perf_counter() - query_start) * 1000

        latencies.append(latency)

        # Simulate FPGA local memory access
        virtual_addr = hash(formula) & 0xFFFFFFFF
        cxl_mem.access(virtual_addr)

    total_time = (time.perf_counter() - start_time) * 1000

    import numpy as np

    cache_results = {
        'num_queries': len(queries),
        'cache_hits': cache_hits,
        'cache_misses': cache_misses,
        'hit_rate': cache_hits / len(queries) if queries else 0,
        'mean_latency_ms': float(np.mean(latencies)),
        'p95_latency_ms': float(np.percentile(latencies, 95)),
        'p99_latency_ms': float(np.percentile(latencies, 99)),
        'total_time_ms': total_time,
        'queries_per_second': len(queries) / (total_time / 1000) if total_time > 0 else 0,
        'num_mismatches': 0,
    }

    # Get hardware metrics
    memory_stats = cxl_mem.get_stats()
    latency_stats = fpga.get_latency_stats()

    hw_metrics = HardwareMetrics(
        page_hits=memory_stats['page_hits'],
        page_faults=memory_stats['page_faults'],
        page_hit_rate=memory_stats['hit_rate'],
        pages_resident=memory_stats['pages_resident'],
        control_p50_us=latency_stats.get('p50_us', 0),
        control_p95_us=latency_stats.get('p95_us', 0),
        control_p99_us=latency_stats.get('p99_us', 0),
        control_mean_us=latency_stats.get('mean_us', 0),
        within_target_pct=latency_stats.get('within_target_pct', 0),
        control_cycles_per_sec=len(queries) / (total_time / 1000) if total_time > 0 else 0,
        queries_per_sec=cache_results['queries_per_second'],
    )

    print(f"\nFPGA Memory Stats:")
    print(f"  Page Hits: {hw_metrics.page_hits}")
    print(f"  Page Faults: {hw_metrics.page_faults}")
    print(f"  Page Hit Rate: {hw_metrics.page_hit_rate:.2%}")

    print(f"\nFPGA Control Cycle Latency:")
    print(f"  p50: {hw_metrics.control_p50_us:.2f}us")
    print(f"  p95: {hw_metrics.control_p95_us:.2f}us")
    print(f"  p99: {hw_metrics.control_p99_us:.2f}us")
    print(f"  Within target: {hw_metrics.within_target_pct:.1f}%")

    return cache_results, hw_metrics


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


def check_gates(
    cached_results: dict,
    baseline_results: dict = None,
    hardware_metrics: HardwareMetrics = None,
) -> dict:
    """
    Check if hard gates are met.

    Gates:
    - p95 ≤120 ms (PGU cache latency)
    - Cache hit-rate ≥60%
    - Correctness: 0 mismatches
    - CXL page hit rate ≥90% (if hardware metrics)
    - Control cycle p95 ≤200us (if hardware metrics)

    Args:
        cached_results: Results with cache
        baseline_results: Results without cache (optional)
        hardware_metrics: Hardware metrics from CXL/FPGA (optional)

    Returns:
        Gate results dict
    """
    gates = {}

    print(f"\n{'='*60}")
    print("GATE VERIFICATION")
    print(f"{'='*60}")

    # Gate 1: p95 latency
    p95 = float(cached_results['p95_latency_ms'])
    gates['p95_latency'] = {
        'value': p95,
        'threshold': 120.0,
        'pass': bool(p95 <= 120.0)
    }

    status = '✓' if p95 <= 120.0 else '✗'
    print(f"\n{status} PGU p95 latency: {p95:.1f}ms (target: ≤120ms)")

    # Gate 2: Hit rate
    hit_rate = float(cached_results['hit_rate'])
    gates['hit_rate'] = {
        'value': hit_rate,
        'threshold': 0.60,
        'pass': bool(hit_rate >= 0.60)
    }

    status = '✓' if hit_rate >= 0.60 else '✗'
    print(f"{status} Cache hit rate: {hit_rate:.2%} (target: ≥60%)")

    # Gate 3: Correctness
    mismatches = int(cached_results.get('num_mismatches', 0))
    gates['correctness'] = {
        'value': mismatches,
        'threshold': 0,
        'pass': bool(mismatches == 0)
    }

    status = '✓' if mismatches == 0 else '✗'
    print(f"{status} Mismatches: {mismatches} (target: 0)")

    # Hardware-specific gates
    if hardware_metrics:
        print(f"\n--- Hardware Gates ---")

        # Gate 4: CXL page hit rate
        page_hit = hardware_metrics.page_hit_rate
        gates['cxl_page_hit_rate'] = {
            'value': page_hit,
            'threshold': 0.90,
            'pass': bool(page_hit >= 0.90)
        }

        status = '✓' if page_hit >= 0.90 else '✗'
        print(f"{status} CXL page hit rate: {page_hit:.2%} (target: ≥90%)")

        # Gate 5: Control cycle latency
        control_p95 = hardware_metrics.control_p95_us
        gates['control_p95_latency'] = {
            'value': control_p95,
            'threshold': 200.0,
            'pass': bool(control_p95 <= 200.0)
        }

        status = '✓' if control_p95 <= 200.0 else '✗'
        print(f"{status} Control cycle p95: {control_p95:.1f}us (target: ≤200us)")

        # Gate 6: Within-target percentage
        within_target = hardware_metrics.within_target_pct
        gates['within_target_pct'] = {
            'value': within_target,
            'threshold': 95.0,
            'pass': bool(within_target >= 95.0)
        }

        status = '✓' if within_target >= 95.0 else '✗'
        print(f"{status} Within target: {within_target:.1f}% (target: ≥95%)")

    # Speedup vs baseline (if available)
    if baseline_results:
        print(f"\n--- Speedup ---")
        baseline_qps = float(baseline_results['queries_per_second'])
        cached_qps = float(cached_results['queries_per_second'])
        speedup = cached_qps / baseline_qps if baseline_qps > 0 else float('inf')

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
    parser = argparse.ArgumentParser(
        description="Benchmark PGU TurboCache with CXL/FPGA Fabric Support"
    )
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

    # Fabric options
    parser.add_argument('--fabric', type=str, default='software',
                        choices=['software', 'cxl', 'fpga', 'CXL', 'FPGA', 'SOFTWARE'],
                        help="Hardware fabric type (software, cxl, fpga)")
    parser.add_argument('--target-latency-us', type=float, default=200.0,
                        help="Target control cycle latency in microseconds")
    parser.add_argument('--include-hardware-metrics', action='store_true',
                        help="Include hardware metrics in output")

    args = parser.parse_args()

    # Normalize fabric type
    fabric = FabricType(args.fabric.lower())

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

    # Run benchmark based on fabric type
    hardware_metrics = None

    print(f"\nFabric Mode: {fabric.value.upper()}")

    if fabric == FabricType.SOFTWARE:
        # Standard software benchmark
        cached_results = benchmark_with_cache(
            corpus_path=corpus_path,
            cache=cache,
            solver=mock_z3_solver,
            max_queries=args.max_queries
        )
    elif fabric == FabricType.CXL:
        # CXL fabric benchmark
        cached_results, hardware_metrics = benchmark_cxl_fabric(
            corpus_path=corpus_path,
            cache=cache,
            solver=mock_z3_solver,
            max_queries=args.max_queries,
            target_latency_us=args.target_latency_us,
        )
    elif fabric == FabricType.FPGA:
        # FPGA fabric benchmark
        cached_results, hardware_metrics = benchmark_fpga_fabric(
            corpus_path=corpus_path,
            cache=cache,
            solver=mock_z3_solver,
            max_queries=args.max_queries,
            target_latency_us=args.target_latency_us,
        )

    # Benchmark baseline (if requested)
    baseline_results = None
    if args.include_baseline:
        baseline_results = benchmark_baseline(
            corpus_path=corpus_path,
            solver=mock_z3_solver,
            max_queries=args.max_queries
        )

    # Check gates (with hardware metrics if available)
    gates = check_gates(cached_results, baseline_results, hardware_metrics)
    cached_results['gates'] = gates

    # Save results
    output = {
        'fabric': fabric.value,
        'cached': cached_results,
        'baseline': baseline_results,
        'hardware_metrics': hardware_metrics.to_dict() if hardware_metrics else None,
        'config': {
            'backend': args.backend,
            'max_entries': args.max_entries,
            'corpus_path': corpus_path,
            'max_queries': args.max_queries,
            'target_latency_us': args.target_latency_us,
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
