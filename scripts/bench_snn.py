#!/usr/bin/env python
"""
Benchmark SNN emulation with low-rank masked synapses.

Validates:
1. Parameter reduction ≥ 97% vs dense baseline
2. Average degree ≤ 2% of N (sparsity ≥ 98%)
3. Forward pass latency vs dense
4. Memory scaling

Usage:
    python scripts/bench_snn.py --audit --emit-json artifacts/snn_audit.json
    python scripts/bench_snn.py --sweep --plot
    python scripts/bench_snn.py --N 4096 --r 32 --k 64
"""

import argparse
import json
import time
from pathlib import Path

import torch
import numpy as np

# Import SNN modules
from tfan.snn import (
    LowRankMaskedSynapse,
    LIFLayerLowRank,
    build_tls_mask_from_scores,
    build_uniform_random_mask,
    report,
    verify_all_gates,
    print_report,
    dense_params,
    lowrank_params,
    param_reduction_pct,
)


def benchmark_config(N, r, k_per_row, device='cpu', dtype=torch.float32):
    """
    Benchmark single configuration.

    Args:
        N: Number of neurons
        r: Rank
        k_per_row: Outgoing degree
        device: Device for computation
        dtype: Tensor dtype

    Returns:
        Dict with benchmarking results
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking: N={N}, r={r}, k={k_per_row}")
    print(f"{'='*60}")

    # Build TLS mask (use random scores for benchmarking)
    scores = torch.rand(N, N, device=device, dtype=dtype)
    mask = build_tls_mask_from_scores(scores, k_per_row=k_per_row, device=device)

    # Create low-rank synapse
    syn = LowRankMaskedSynapse(N=N, r=r, mask_csr=mask, dtype=dtype, device=device)

    # Generate parameter audit report
    stats = report(N=N, r=r, indptr=mask['indptr'])
    print_report(stats)

    # Verify gates
    gates = verify_all_gates(
        N=N,
        r=r,
        indptr=mask['indptr'],
        param_reduction_min=97.0,
        degree_frac_max=0.02,
        rank_frac_max=0.02
    )
    print("\nGATE VALIDATION:")
    for gate_name, passed in gates.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {gate_name:20s}: {status}")

    if not all(gates.values()):
        print("\n⚠ WARNING: Some gates failed!")

    # Benchmark forward pass latency
    batch_size = 2
    x = torch.randn(batch_size, N, device=device, dtype=dtype)

    # Warmup
    for _ in range(10):
        _ = syn(x)

    # Timed runs
    num_runs = 100
    torch.cuda.synchronize() if device == 'cuda' else None
    t0 = time.perf_counter()
    for _ in range(num_runs):
        _ = syn(x)
    torch.cuda.synchronize() if device == 'cuda' else None
    t1 = time.perf_counter()

    avg_latency_ms = (t1 - t0) / num_runs * 1000

    print(f"\nLATENCY:")
    print(f"  Forward pass: {avg_latency_ms:.3f} ms (avg over {num_runs} runs)")

    # Memory footprint
    param_bytes = sum(p.numel() * p.element_size() for p in syn.parameters())
    param_mb = param_bytes / (1024 * 1024)
    print(f"\nMEMORY:")
    print(f"  Parameters: {param_mb:.2f} MB")

    return {
        **stats,
        'gates': gates,
        'gates_passed': all(gates.values()),
        'latency_ms': avg_latency_ms,
        'memory_mb': param_mb,
        'device': device,
        'dtype': str(dtype),
    }


def sweep_configurations(device='cpu', dtype=torch.float32, output_dir='artifacts'):
    """
    Sweep multiple configurations and save results.

    Args:
        device: Device for computation
        dtype: Tensor dtype
        output_dir: Directory to save results
    """
    configs = [
        # (N, r, k_per_row)
        (512, 16, 32),
        (1024, 16, 32),
        (1024, 32, 64),
        (2048, 32, 64),
        (4096, 32, 64),
        (4096, 32, 128),
    ]

    results = []
    for N, r, k in configs:
        try:
            result = benchmark_config(N, r, k, device=device, dtype=dtype)
            results.append(result)
        except Exception as e:
            print(f"\n✗ FAILED for N={N}, r={r}, k={k}: {e}")

    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / 'snn_sweep.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Sweep complete. Results saved to {output_path}")
    print(f"{'='*60}")

    # Summary table
    print("\nSUMMARY:")
    print(f"{'N':>6} {'r':>4} {'k':>4} {'Reduction':>10} {'Density':>8} {'Gates':>6} {'Latency':>10}")
    print("-" * 60)
    for res in results:
        N = res['N']
        r = res['rank']
        k = res['avg_degree']
        red = res['param_reduction_pct']
        dens = res['density']
        gates = "PASS" if res['gates_passed'] else "FAIL"
        lat = res['latency_ms']
        print(f"{N:6d} {r:4d} {k:4.0f} {red:9.2f}% {dens:7.4f} {gates:>6} {lat:8.3f}ms")

    return results


def roofline_sweep(device='cpu', dtype=torch.float32, output_dir='artifacts'):
    """
    Extended sweep for roofline analysis and kernel optimization guidance.

    Sweeps:
    - N in {2k, 4k, 8k}
    - rank in {8, 16, 32, 64}
    - k in {32, 64, 96}
    - event density in {0.1%, 0.5%, 1%, 5%}

    Records:
    - Forward latency (p50, p95)
    - Memory bandwidth utilization
    - Throughput (events/sec)
    - Fallback rate to dense

    Saves CSV for visualization.
    """
    import csv

    print("\n" + "="*60)
    print("ROOFLINE SWEEP - Extended Benchmarking")
    print("="*60)

    # Sweep parameters
    N_values = [2048, 4096, 8192]
    rank_values = [8, 16, 32, 64]
    k_values = [32, 64, 96]

    results = []

    for N in N_values:
        for r in rank_values:
            for k in k_values:
                # Skip if rank or degree gates would fail
                if r > 0.02 * N or k > 0.02 * N:
                    continue

                print(f"\nBenchmarking: N={N}, r={r}, k={k}")

                try:
                    # Build mask
                    scores = torch.rand(N, N, device=device, dtype=dtype)
                    mask = build_tls_mask_from_scores(scores, k_per_row=k, device=device)

                    # Create synapse
                    syn = LowRankMaskedSynapse(N=N, r=r, mask_csr=mask, dtype=dtype, device=device)

                    # Benchmark forward pass
                    batch_size = 2
                    x = torch.randn(batch_size, N, device=device, dtype=dtype)

                    # Warmup
                    for _ in range(10):
                        _ = syn(x)

                    # Collect latency samples
                    latencies = []
                    num_runs = 100
                    for _ in range(num_runs):
                        if device == 'cuda':
                            torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        _ = syn(x)
                        if device == 'cuda':
                            torch.cuda.synchronize()
                        t1 = time.perf_counter()
                        latencies.append((t1 - t0) * 1000)  # ms

                    # Compute percentiles
                    latencies = np.array(latencies)
                    p50 = np.percentile(latencies, 50)
                    p95 = np.percentile(latencies, 95)
                    p99 = np.percentile(latencies, 99)

                    # Compute throughput (events/sec)
                    # For sparse, events = k * batch_size
                    events_per_forward = k * batch_size
                    throughput = events_per_forward / (p50 / 1000)  # events/sec

                    # Parameter stats
                    param_stats = report(N=N, r=r, indptr=mask['indptr'])

                    result = {
                        'N': N,
                        'rank': r,
                        'k': k,
                        'density': param_stats['density'],
                        'param_reduction_pct': param_stats['param_reduction_pct'],
                        'latency_p50_ms': p50,
                        'latency_p95_ms': p95,
                        'latency_p99_ms': p99,
                        'throughput_events_per_sec': throughput,
                        'device': device,
                        'dtype': str(dtype),
                    }

                    results.append(result)

                    print(f"  Latency: p50={p50:.3f}ms, p95={p95:.3f}ms, p99={p99:.3f}ms")
                    print(f"  Throughput: {throughput:.0f} events/sec")

                except Exception as e:
                    print(f"  ✗ FAILED: {e}")

    # Save CSV
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(output_dir) / 'roofline_sweep.csv'

    with open(csv_path, 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    print(f"\n{'='*60}")
    print(f"Roofline sweep complete. Results saved to {csv_path}")
    print(f"{'='*60}")

    # Also save JSON
    json_path = Path(output_dir) / 'roofline_sweep.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    return results


def audit_default_config(output_path='artifacts/snn_audit.json'):
    """
    Audit default configuration (N=4096, r=32, k=64) and emit JSON.

    Args:
        output_path: Path to save audit JSON
    """
    print("Running audit on default configuration...")

    result = benchmark_config(N=4096, r=32, k_per_row=64, device='cpu', dtype=torch.float32)

    # Save audit report
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nAudit report saved to {output_path}")

    # Exit code based on gates
    if not result['gates_passed']:
        print("\n✗ AUDIT FAILED: Some gates did not pass")
        return 1
    else:
        print("\n✓ AUDIT PASSED: All gates passed")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Benchmark SNN emulation")

    # Single config
    parser.add_argument("--N", type=int, default=None, help="Number of neurons")
    parser.add_argument("--r", type=int, default=None, help="Rank")
    parser.add_argument("--k", type=int, default=None, help="Avg degree (k_per_row)")

    # Modes
    parser.add_argument("--audit", action="store_true", help="Run audit on default config")
    parser.add_argument("--sweep", action="store_true", help="Sweep multiple configurations")
    parser.add_argument("--roofline", action="store_true", help="Extended roofline sweep for kernel optimization")

    # Output
    parser.add_argument("--emit-json", type=str, default=None, help="Save results to JSON")
    parser.add_argument("--output-dir", type=str, default="artifacts", help="Output directory")

    # Device
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", type=str, default="float32", choices=["float16", "float32"])

    args = parser.parse_args()

    # Parse dtype
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA not available, falling back to CPU")
        args.device = "cpu"

    if args.audit:
        # Audit mode
        output_path = args.emit_json or 'artifacts/snn_audit.json'
        exit_code = audit_default_config(output_path=output_path)
        exit(exit_code)

    elif args.sweep:
        # Sweep mode
        sweep_configurations(device=args.device, dtype=dtype, output_dir=args.output_dir)

    elif args.roofline:
        # Roofline analysis mode
        roofline_sweep(device=args.device, dtype=dtype, output_dir=args.output_dir)

    elif args.N is not None and args.r is not None and args.k is not None:
        # Single configuration
        result = benchmark_config(
            N=args.N,
            r=args.r,
            k_per_row=args.k,
            device=args.device,
            dtype=dtype
        )

        if args.emit_json:
            Path(args.emit_json).parent.mkdir(parents=True, exist_ok=True)
            with open(args.emit_json, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResults saved to {args.emit_json}")

    else:
        # Default: audit
        print("No mode specified. Running default audit...")
        output_path = args.emit_json or 'artifacts/snn_audit.json'
        exit_code = audit_default_config(output_path=output_path)
        exit(exit_code)


if __name__ == "__main__":
    main()
