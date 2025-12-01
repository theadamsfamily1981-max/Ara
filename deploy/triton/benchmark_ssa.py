#!/usr/bin/env python
"""
Benchmark TF-A-N SSA Server

Tests performance against hard gates:
- 128k prefill ≥3× faster than dense baseline (RTX 3090)
- p99 latency under SLO
- KV cache hit-rate ≥90%

Usage:
    # Benchmark local server
    python deploy/triton/benchmark_ssa.py \
        --url localhost:8000 \
        --model tfan_ssa \
        --seq-lengths 4096 8192 16384 32768 65536 131072

    # Compare with baseline
    python deploy/triton/benchmark_ssa.py \
        --url localhost:8000 \
        --model tfan_ssa \
        --baseline dense_baseline \
        --seq-lengths 131072
"""

import argparse
import numpy as np
import time
import json
from typing import List, Dict
import requests
from pathlib import Path

try:
    import tritonclient.http as httpclient
    from tritonclient.utils import np_to_triton_dtype
except ImportError:
    print("⚠ tritonclient not installed. Install with: pip install tritonclient[http]")
    httpclient = None


def create_dummy_input(seq_length: int, vocab_size: int = 50257) -> np.ndarray:
    """Create dummy input IDs."""
    return np.random.randint(0, vocab_size, size=(1, seq_length), dtype=np.int64)


def benchmark_sequence(
    triton_client,
    model_name: str,
    seq_length: int,
    num_trials: int = 10
) -> Dict:
    """
    Benchmark a single sequence length.

    Args:
        triton_client: Triton client
        model_name: Model name
        seq_length: Sequence length
        num_trials: Number of trials

    Returns:
        stats: Dict with benchmark results
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking seq_length={seq_length:,}")
    print(f"{'='*60}")

    latencies = []
    throughputs = []
    ssa_stats_list = []

    for trial in range(num_trials):
        # Create input
        input_ids = create_dummy_input(seq_length)

        # Prepare Triton inputs
        inputs = [
            httpclient.InferInput("input_ids", input_ids.shape, np_to_triton_dtype(input_ids.dtype))
        ]
        inputs[0].set_data_from_numpy(input_ids)

        # Prepare outputs
        outputs = [
            httpclient.InferRequestedOutput("output_ids"),
            httpclient.InferRequestedOutput("logits"),
            httpclient.InferRequestedOutput("ssa_stats")
        ]

        # Measure latency
        start_time = time.perf_counter()

        try:
            response = triton_client.infer(
                model_name=model_name,
                inputs=inputs,
                outputs=outputs
            )

            latency = (time.perf_counter() - start_time) * 1000  # ms

            # Parse stats
            ssa_stats_json = response.as_numpy("ssa_stats")[0].decode('utf-8')
            ssa_stats = json.loads(ssa_stats_json)

            latencies.append(latency)
            throughputs.append(seq_length / (latency / 1000))  # tokens/s
            ssa_stats_list.append(ssa_stats)

            print(f"  Trial {trial+1}/{num_trials}: {latency:.1f}ms, {throughputs[-1]:.1f} tokens/s")

        except Exception as e:
            print(f"  Trial {trial+1}/{num_trials}: FAILED - {e}")
            continue

    if not latencies:
        return {
            'seq_length': seq_length,
            'status': 'FAILED',
            'error': 'All trials failed'
        }

    # Aggregate stats
    stats = {
        'seq_length': seq_length,
        'status': 'SUCCESS',
        'num_trials': len(latencies),
        'latency_mean_ms': np.mean(latencies),
        'latency_p50_ms': np.percentile(latencies, 50),
        'latency_p95_ms': np.percentile(latencies, 95),
        'latency_p99_ms': np.percentile(latencies, 99),
        'throughput_mean': np.mean(throughputs),
        'ssa_sparsity_mean': np.mean([s['ssa_stats']['sparsity'] for s in ssa_stats_list]),
        'ssa_landmarks_mean': np.mean([s['ssa_stats']['num_landmarks'] for s in ssa_stats_list]),
    }

    # KV pager stats (if available)
    kv_stats = [s.get('kv_pager_stats') for s in ssa_stats_list if 'kv_pager_stats' in s]
    if kv_stats:
        stats['kv_hit_rate_mean'] = np.mean([s['hit_rate'] for s in kv_stats])
        stats['kv_prefetch_accuracy_mean'] = np.mean([s['prefetch_accuracy'] for s in kv_stats])

    # TTW stats
    ttw_triggered = sum(1 for s in ssa_stats_list if s['ttw_stats']['triggered'])
    stats['ttw_trigger_rate'] = ttw_triggered / len(ssa_stats_list)

    return stats


def benchmark_baseline(
    triton_client,
    baseline_model: str,
    seq_length: int,
    num_trials: int = 10
) -> Dict:
    """
    Benchmark baseline model (dense attention).

    Args:
        triton_client: Triton client
        baseline_model: Baseline model name
        seq_length: Sequence length
        num_trials: Number of trials

    Returns:
        stats: Dict with baseline results
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking BASELINE (dense) seq_length={seq_length:,}")
    print(f"{'='*60}")

    latencies = []

    for trial in range(num_trials):
        input_ids = create_dummy_input(seq_length)

        inputs = [
            httpclient.InferInput("input_ids", input_ids.shape, np_to_triton_dtype(input_ids.dtype))
        ]
        inputs[0].set_data_from_numpy(input_ids)

        outputs = [httpclient.InferRequestedOutput("logits")]

        start_time = time.perf_counter()

        try:
            response = triton_client.infer(
                model_name=baseline_model,
                inputs=inputs,
                outputs=outputs
            )

            latency = (time.perf_counter() - start_time) * 1000

            latencies.append(latency)

            print(f"  Trial {trial+1}/{num_trials}: {latency:.1f}ms")

        except Exception as e:
            print(f"  Trial {trial+1}/{num_trials}: FAILED - {e}")
            continue

    if not latencies:
        return None

    return {
        'seq_length': seq_length,
        'latency_mean_ms': np.mean(latencies),
        'latency_p99_ms': np.percentile(latencies, 99),
    }


def check_gates(ssa_stats: Dict, baseline_stats: Dict = None) -> Dict:
    """
    Check if hard gates are met.

    Gates:
    - 128k prefill ≥3× faster than dense baseline
    - p99 latency under SLO (e.g., 10s for 128k)
    - KV cache hit-rate ≥90%

    Args:
        ssa_stats: SSA benchmark stats
        baseline_stats: Optional baseline stats

    Returns:
        gate_results: Dict with pass/fail for each gate
    """
    results = {}

    seq_length = ssa_stats['seq_length']

    # Gate 1: Speedup vs baseline
    if baseline_stats:
        speedup = baseline_stats['latency_mean_ms'] / ssa_stats['latency_mean_ms']
        results['speedup_vs_baseline'] = {
            'value': speedup,
            'threshold': 3.0,
            'pass': speedup >= 3.0
        }
        print(f"\n✓ Speedup: {speedup:.2f}× vs dense baseline")

    # Gate 2: p99 latency under SLO
    # SLO: 128k tokens in <10s
    slo_ms = 10000  # 10s
    p99_latency = ssa_stats['latency_p99_ms']
    results['p99_under_slo'] = {
        'value': p99_latency,
        'threshold': slo_ms,
        'pass': p99_latency < slo_ms
    }
    print(f"{'✓' if p99_latency < slo_ms else '✗'} p99 latency: {p99_latency:.1f}ms (SLO: {slo_ms}ms)")

    # Gate 3: KV cache hit-rate
    if 'kv_hit_rate_mean' in ssa_stats:
        hit_rate = ssa_stats['kv_hit_rate_mean']
        results['kv_hit_rate'] = {
            'value': hit_rate,
            'threshold': 0.90,
            'pass': hit_rate >= 0.90
        }
        print(f"{'✓' if hit_rate >= 0.90 else '✗'} KV hit-rate: {hit_rate:.2%} (target: ≥90%)")

    # Gate 4: TTW trigger rate (should be low)
    ttw_rate = ssa_stats['ttw_trigger_rate']
    results['ttw_trigger_rate'] = {
        'value': ttw_rate,
        'threshold': 0.05,  # <5% trigger rate
        'pass': ttw_rate < 0.05
    }
    print(f"{'✓' if ttw_rate < 0.05 else '✗'} TTW trigger rate: {ttw_rate:.2%} (target: <5%)")

    # Overall pass
    all_pass = all(r['pass'] for r in results.values())
    results['overall'] = {'pass': all_pass}

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark TF-A-N SSA server")
    parser.add_argument('--url', type=str, default='localhost:8000', help="Triton server URL")
    parser.add_argument('--model', type=str, default='tfan_ssa', help="Model name")
    parser.add_argument('--baseline', type=str, default=None, help="Baseline model (dense)")
    parser.add_argument('--seq-lengths', type=int, nargs='+', default=[4096, 8192, 16384, 32768, 65536, 131072],
                        help="Sequence lengths to benchmark")
    parser.add_argument('--num-trials', type=int, default=10, help="Trials per sequence length")
    parser.add_argument('--output', type=str, default='benchmark_results.json', help="Output JSON file")

    args = parser.parse_args()

    if httpclient is None:
        print("ERROR: tritonclient not installed")
        return

    # Connect to Triton
    print(f"Connecting to Triton server: {args.url}")
    triton_client = httpclient.InferenceServerClient(url=args.url)

    # Check server health
    if not triton_client.is_server_live():
        print("ERROR: Triton server not live")
        return

    if not triton_client.is_model_ready(args.model):
        print(f"ERROR: Model {args.model} not ready")
        return

    print(f"✓ Server ready, model {args.model} loaded")

    # Run benchmarks
    all_results = []

    for seq_length in args.seq_lengths:
        # Benchmark SSA
        ssa_stats = benchmark_sequence(
            triton_client,
            args.model,
            seq_length,
            args.num_trials
        )

        # Benchmark baseline (if provided)
        baseline_stats = None
        if args.baseline:
            baseline_stats = benchmark_baseline(
                triton_client,
                args.baseline,
                seq_length,
                args.num_trials
            )

        # Check gates
        if ssa_stats['status'] == 'SUCCESS':
            gate_results = check_gates(ssa_stats, baseline_stats)
            ssa_stats['gates'] = gate_results

        # Combine results
        result = {
            'seq_length': seq_length,
            'ssa': ssa_stats,
            'baseline': baseline_stats
        }
        all_results.append(result)

        print(f"\nResults for seq_length={seq_length:,}:")
        print(json.dumps(ssa_stats, indent=2))

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
        ssa = result['ssa']

        if ssa['status'] != 'SUCCESS':
            print(f"  {seq_len:>6,} tokens: FAILED")
            continue

        p99 = ssa['latency_p99_ms']
        throughput = ssa['throughput_mean']

        gate_pass = ssa.get('gates', {}).get('overall', {}).get('pass', False)
        status = '✓' if gate_pass else '✗'

        print(f"  {status} {seq_len:>6,} tokens: p99={p99:>8.1f}ms, throughput={throughput:>8.1f} tok/s")


if __name__ == '__main__':
    main()
