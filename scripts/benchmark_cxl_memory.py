#!/usr/bin/env python
"""
Benchmark CXL Memory Tiering

Tests CXL memory tiering against hard gates:
- 128k context without OOM on 24GB GPU
- ≤8% tokens/s penalty vs in-memory cache
- Cache hit-rate ≥90%
- Prefetch accuracy ≥80%

Comparison:
1. Baseline: In-memory KV cache (no paging)
2. CXL Tiering: Multi-tier cache with CXL
3. Bloom Prefetch: CXL + Bloom filter prefetching

Usage:
    # Quick benchmark
    python scripts/benchmark_cxl_memory.py \
        --seq-lengths 4096 8192 16384 32768 65536 131072 \
        --num-trials 5

    # Compare with baseline
    python scripts/benchmark_cxl_memory.py \
        --seq-lengths 131072 \
        --include-baseline \
        --num-trials 10

    # Profile detailed stats
    python scripts/benchmark_cxl_memory.py \
        --seq-lengths 131072 \
        --profile \
        --output cxl_benchmark_results.json
"""

import argparse
import torch
import numpy as np
import time
import json
from pathlib import Path
from typing import Dict, List, Optional
import sys

# Add tfan to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tfan.memory import CXLPager, CXLPageConfig, BloomPrefetcher, BloomConfig


def create_dummy_kv_block(
    num_heads: int = 8,
    block_size: int = 16,
    head_dim: int = 64,
    device: str = 'cuda'
) -> tuple:
    """Create dummy KV block for benchmarking."""
    key = torch.randn(num_heads, block_size, head_dim, device=device)
    value = torch.randn(num_heads, block_size, head_dim, device=device)
    return key, value


def benchmark_cxl_tiering(
    seq_length: int,
    num_layers: int = 12,
    block_size: int = 16,
    num_trials: int = 10,
    enable_bloom: bool = True,
    profile: bool = False
) -> Dict:
    """
    Benchmark CXL memory tiering.

    Args:
        seq_length: Sequence length
        num_layers: Number of transformer layers
        block_size: Tokens per block
        num_trials: Number of trials
        enable_bloom: Enable Bloom filter prefetching
        profile: Enable detailed profiling

    Returns:
        stats: Benchmark statistics
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking CXL Tiering: seq_length={seq_length:,}")
    print(f"{'='*60}")

    num_blocks = seq_length // block_size

    # Configure CXL pager
    config = CXLPageConfig(
        max_gpu_blocks=64,  # Small GPU cache to force tiering
        max_cpu_blocks=256,
        max_cxl_blocks=1024,
        block_size=block_size,
        enable_bloom_prefetch=enable_bloom,
        prefetch_lookahead=4,
        profile_stats=profile
    )

    pager = CXLPager(config=config)

    # Trial results
    write_times = []
    read_times = []
    total_times = []

    for trial in range(num_trials):
        print(f"\n  Trial {trial+1}/{num_trials}")

        # Clear cache between trials
        pager.clear()
        pager.stats = pager.stats.__class__()  # Reset stats

        # Phase 1: Write all blocks (simulates prefill)
        write_start = time.perf_counter()

        for layer_idx in range(num_layers):
            for block_idx in range(num_blocks):
                key, value = create_dummy_kv_block(device='cuda')
                pager.store_block(layer_idx, block_idx, key, value)

        write_time = (time.perf_counter() - write_start) * 1000

        # Phase 2: Read blocks in sequence (simulates generation)
        read_start = time.perf_counter()

        for layer_idx in range(num_layers):
            for block_idx in range(num_blocks):
                result = pager.load_block(layer_idx, block_idx, device='cuda')
                assert result is not None, f"Block not found: L{layer_idx} B{block_idx}"

        read_time = (time.perf_counter() - read_start) * 1000
        total_time = write_time + read_time

        write_times.append(write_time)
        read_times.append(read_time)
        total_times.append(total_time)

        # Get stats
        cache_stats = pager.get_stats()

        print(f"    Write: {write_time:.1f}ms")
        print(f"    Read: {read_time:.1f}ms")
        print(f"    Hit rate: {cache_stats['hit_rate']:.2%}")

        if enable_bloom:
            print(f"    Prefetch accuracy: {cache_stats['prefetch_accuracy']:.2%}")

    # Aggregate stats
    final_stats = pager.get_stats()

    results = {
        'seq_length': seq_length,
        'num_layers': num_layers,
        'num_blocks_per_layer': num_blocks,
        'block_size': block_size,
        'num_trials': num_trials,
        'enable_bloom': enable_bloom,
        'write_time_mean_ms': np.mean(write_times),
        'write_time_p99_ms': np.percentile(write_times, 99),
        'read_time_mean_ms': np.mean(read_times),
        'read_time_p99_ms': np.percentile(read_times, 99),
        'total_time_mean_ms': np.mean(total_times),
        'total_time_p99_ms': np.percentile(total_times, 99),
        'tokens_per_second': seq_length / (np.mean(total_times) / 1000),
        'cache_hit_rate': final_stats['hit_rate'],
        'tier_distribution': final_stats['tier_distribution'],
        'prefetch_accuracy': final_stats.get('prefetch_accuracy', 0.0) if enable_bloom else None
    }

    if profile:
        results.update({
            'gpu_avg_load_ms': final_stats.get('gpu_avg_load_ms', 0),
            'cpu_avg_load_ms': final_stats.get('cpu_avg_load_ms', 0),
            'cxl_avg_load_ms': final_stats.get('cxl_avg_load_ms', 0),
            'nvme_avg_load_ms': final_stats.get('nvme_avg_load_ms', 0),
        })

    # Cleanup
    pager.clear()
    del pager

    return results


def benchmark_baseline(
    seq_length: int,
    num_layers: int = 12,
    block_size: int = 16,
    num_trials: int = 10
) -> Dict:
    """
    Benchmark baseline in-memory KV cache (no tiering).

    Args:
        seq_length: Sequence length
        num_layers: Number of layers
        block_size: Tokens per block
        num_trials: Number of trials

    Returns:
        stats: Benchmark statistics
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking BASELINE (in-memory): seq_length={seq_length:,}")
    print(f"{'='*60}")

    num_blocks = seq_length // block_size

    # Store all blocks in memory (dict)
    total_times = []

    for trial in range(num_trials):
        cache = {}

        # Write phase
        write_start = time.perf_counter()

        for layer_idx in range(num_layers):
            for block_idx in range(num_blocks):
                key, value = create_dummy_kv_block(device='cuda')
                cache[(layer_idx, block_idx)] = (key, value)

        # Read phase
        for layer_idx in range(num_layers):
            for block_idx in range(num_blocks):
                key, value = cache[(layer_idx, block_idx)]
                # Simulate some work
                _ = key.mean()

        total_time = (time.perf_counter() - write_start) * 1000
        total_times.append(total_time)

        print(f"  Trial {trial+1}/{num_trials}: {total_time:.1f}ms")

        # Cleanup
        del cache

    return {
        'seq_length': seq_length,
        'total_time_mean_ms': np.mean(total_times),
        'total_time_p99_ms': np.percentile(total_times, 99),
        'tokens_per_second': seq_length / (np.mean(total_times) / 1000)
    }


def check_gates(cxl_stats: Dict, baseline_stats: Optional[Dict] = None) -> Dict:
    """
    Check if hard gates are met.

    Gates:
    - 128k context without OOM
    - ≤8% tokens/s penalty vs baseline
    - Cache hit-rate ≥90%
    - Prefetch accuracy ≥80% (if enabled)

    Args:
        cxl_stats: CXL benchmark stats
        baseline_stats: Optional baseline stats

    Returns:
        gate_results: Dict with pass/fail for each gate
    """
    results = {}

    # Gate 1: No OOM (implicit - if we got here, no OOM)
    results['no_oom'] = {
        'value': True,
        'pass': True
    }
    print(f"\n✓ No OOM for {cxl_stats['seq_length']:,} tokens")

    # Gate 2: Performance penalty ≤8%
    if baseline_stats:
        baseline_throughput = baseline_stats['tokens_per_second']
        cxl_throughput = cxl_stats['tokens_per_second']
        penalty = (baseline_throughput - cxl_throughput) / baseline_throughput

        results['performance_penalty'] = {
            'value': penalty,
            'threshold': 0.08,
            'pass': penalty <= 0.08
        }

        status = '✓' if penalty <= 0.08 else '✗'
        print(f"{status} Performance penalty: {penalty:.2%} (target: ≤8%)")
        print(f"    Baseline: {baseline_throughput:.1f} tok/s")
        print(f"    CXL: {cxl_throughput:.1f} tok/s")

    # Gate 3: Hit rate ≥90%
    hit_rate = cxl_stats['cache_hit_rate']
    results['hit_rate'] = {
        'value': hit_rate,
        'threshold': 0.90,
        'pass': hit_rate >= 0.90
    }

    status = '✓' if hit_rate >= 0.90 else '✗'
    print(f"{status} Cache hit-rate: {hit_rate:.2%} (target: ≥90%)")

    # Gate 4: Prefetch accuracy ≥80% (if Bloom enabled)
    if cxl_stats['enable_bloom'] and cxl_stats['prefetch_accuracy'] is not None:
        prefetch_acc = cxl_stats['prefetch_accuracy']
        results['prefetch_accuracy'] = {
            'value': prefetch_acc,
            'threshold': 0.80,
            'pass': prefetch_acc >= 0.80
        }

        status = '✓' if prefetch_acc >= 0.80 else '✗'
        print(f"{status} Prefetch accuracy: {prefetch_acc:.2%} (target: ≥80%)")

    # Overall pass
    all_pass = all(r['pass'] for r in results.values())
    results['overall'] = {'pass': all_pass}

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark CXL memory tiering")
    parser.add_argument('--seq-lengths', type=int, nargs='+',
                        default=[4096, 8192, 16384, 32768, 65536, 131072],
                        help="Sequence lengths to benchmark")
    parser.add_argument('--num-layers', type=int, default=12, help="Number of layers")
    parser.add_argument('--block-size', type=int, default=16, help="Block size")
    parser.add_argument('--num-trials', type=int, default=10, help="Trials per sequence length")
    parser.add_argument('--include-baseline', action='store_true', help="Include baseline benchmark")
    parser.add_argument('--no-bloom', action='store_true', help="Disable Bloom prefetching")
    parser.add_argument('--profile', action='store_true', help="Enable detailed profiling")
    parser.add_argument('--output', type=str, default='cxl_benchmark_results.json', help="Output file")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("⚠ CUDA not available, using CPU (benchmarks will be slower)")

    # Run benchmarks
    all_results = []

    for seq_length in args.seq_lengths:
        # Benchmark CXL tiering
        cxl_stats = benchmark_cxl_tiering(
            seq_length=seq_length,
            num_layers=args.num_layers,
            block_size=args.block_size,
            num_trials=args.num_trials,
            enable_bloom=not args.no_bloom,
            profile=args.profile
        )

        # Benchmark baseline (if requested)
        baseline_stats = None
        if args.include_baseline:
            baseline_stats = benchmark_baseline(
                seq_length=seq_length,
                num_layers=args.num_layers,
                block_size=args.block_size,
                num_trials=args.num_trials
            )

        # Check gates
        gate_results = check_gates(cxl_stats, baseline_stats)
        cxl_stats['gates'] = gate_results

        # Combine results
        result = {
            'seq_length': seq_length,
            'cxl': cxl_stats,
            'baseline': baseline_stats
        }
        all_results.append(result)

        print(f"\n{'-'*60}")
        print(f"Results for seq_length={seq_length:,}:")
        print(json.dumps(cxl_stats, indent=2))

    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✓ Benchmark complete. Results saved to {output_path}")
    print(f"{'='*60}")

    # Summary
    print("\nSummary:")
    for result in all_results:
        seq_len = result['seq_length']
        cxl = result['cxl']

        throughput = cxl['tokens_per_second']
        hit_rate = cxl['cache_hit_rate']

        gate_pass = cxl.get('gates', {}).get('overall', {}).get('pass', False)
        status = '✓' if gate_pass else '✗'

        print(f"  {status} {seq_len:>6,} tokens: {throughput:>8.1f} tok/s, hit-rate={hit_rate:>5.1%}")


if __name__ == '__main__':
    main()
