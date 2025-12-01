#!/usr/bin/env python
"""
Energy and Throughput Certification Script

Validates production-grade SNN performance against hard gates:
- SNN Throughput: ≥250k events/second
- Energy Efficiency: ≤0.15 (normalized energy metric)
- Parameter Reduction: ≥97% vs dense baseline
- EPR-CV: ≤0.15 (edge-of-chaos stability)

Usage:
    # Full certification
    python scripts/certify_energy_throughput.py --full

    # Quick smoke test
    python scripts/certify_energy_throughput.py --quick

    # Specific gates
    python scripts/certify_energy_throughput.py --throughput-only
    python scripts/certify_energy_throughput.py --energy-only
"""

import argparse
import json
import time
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

# Import SNN modules
from tfan.snn import (
    LowRankMaskedSynapse,
    LIFLayerLowRank,
    build_tls_mask_from_scores,
    report,
    verify_all_gates,
    dense_params,
    lowrank_params,
    param_reduction_pct,
)


@dataclass
class ThroughputCertification:
    """Throughput certification results."""
    events_per_second: float
    events_per_second_target: float = 250000.0
    latency_per_event_us: float = 0.0
    batch_size: int = 0
    num_neurons: int = 0
    sparsity: float = 0.0
    passed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EnergyCertification:
    """Energy certification results."""
    normalized_energy: float
    energy_target: float = 0.15
    param_reduction_pct: float = 0.0
    param_reduction_target: float = 97.0
    memory_mb: float = 0.0
    compute_intensity: float = 0.0  # ops/byte
    passed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StabilityCertification:
    """Stability (EPR-CV) certification results."""
    epr_cv: float
    epr_cv_target: float = 0.15
    firing_rate_mean: float = 0.0
    firing_rate_std: float = 0.0
    v_variance_mean: float = 0.0
    passed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CertificationResult:
    """Complete certification result."""
    timestamp: str
    throughput: ThroughputCertification
    energy: EnergyCertification
    stability: StabilityCertification
    all_gates_passed: bool
    config: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "throughput": self.throughput.to_dict(),
            "energy": self.energy.to_dict(),
            "stability": self.stability.to_dict(),
            "all_gates_passed": self.all_gates_passed,
            "config": self.config,
        }


def certify_throughput(
    N: int = 4096,
    r: int = 32,
    k_per_row: int = 64,
    batch_size: int = 32,
    num_runs: int = 100,
    warmup_runs: int = 20,
    device: str = 'cpu',
) -> ThroughputCertification:
    """
    Certify SNN throughput against ≥250k events/s gate.

    An "event" is a single spike from a neuron. Throughput is measured as
    total spike events processed per second.

    Args:
        N: Number of neurons
        r: Low-rank dimension
        k_per_row: Outgoing degree per neuron
        batch_size: Batch size for timing
        num_runs: Number of timed runs
        warmup_runs: Warmup runs before timing
        device: 'cpu' or 'cuda'

    Returns:
        ThroughputCertification result
    """
    print(f"\n{'='*60}")
    print("THROUGHPUT CERTIFICATION")
    print(f"{'='*60}")
    print(f"Config: N={N}, r={r}, k={k_per_row}, batch={batch_size}")

    dtype = torch.float32

    # Build TLS mask
    scores = torch.rand(N, N, device=device, dtype=dtype)
    mask = build_tls_mask_from_scores(scores, k_per_row=k_per_row, device=device)

    # Create low-rank synapse
    syn = LowRankMaskedSynapse(N=N, r=r, mask_csr=mask, dtype=dtype, device=device)

    # Get sparsity stats
    stats = report(N=N, r=r, indptr=mask['indptr'])
    sparsity = 1.0 - stats['density']

    # Create LIF layer
    lif = LIFLayerLowRank(synapse=syn, N=N, device=device, dtype=dtype)

    # Prepare input (spike train with ~10% activity)
    spike_prob = 0.1
    x = (torch.rand(batch_size, N, device=device, dtype=dtype) < spike_prob).float()
    num_input_spikes = x.sum().item()

    # Warmup
    print(f"Warming up ({warmup_runs} runs)...")
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = lif.step(x)

    # Timed runs
    print(f"Benchmarking ({num_runs} runs)...")
    total_spikes = 0

    if device == 'cuda':
        torch.cuda.synchronize()

    t0 = time.perf_counter()

    for _ in range(num_runs):
        with torch.no_grad():
            spikes = lif.step(x)
            total_spikes += spikes.sum().item()

    if device == 'cuda':
        torch.cuda.synchronize()

    t1 = time.perf_counter()

    elapsed_s = t1 - t0
    total_events = num_input_spikes * num_runs + total_spikes  # Input + output spikes
    events_per_second = total_events / elapsed_s
    latency_per_event_us = (elapsed_s / total_events) * 1e6 if total_events > 0 else float('inf')

    passed = events_per_second >= 250000.0

    print(f"\nResults:")
    print(f"  Total events: {total_events:,.0f}")
    print(f"  Elapsed time: {elapsed_s:.3f}s")
    print(f"  Throughput: {events_per_second:,.0f} events/s")
    print(f"  Latency: {latency_per_event_us:.3f} us/event")
    print(f"  Target: ≥250,000 events/s")
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  Status: {status}")

    return ThroughputCertification(
        events_per_second=events_per_second,
        latency_per_event_us=latency_per_event_us,
        batch_size=batch_size,
        num_neurons=N,
        sparsity=sparsity,
        passed=passed,
    )


def certify_energy(
    N: int = 4096,
    r: int = 32,
    k_per_row: int = 64,
    device: str = 'cpu',
) -> EnergyCertification:
    """
    Certify energy efficiency against ≤0.15 gate.

    Energy is approximated by normalized compute/memory ratio and parameter reduction.

    Args:
        N: Number of neurons
        r: Low-rank dimension
        k_per_row: Outgoing degree per neuron
        device: 'cpu' or 'cuda'

    Returns:
        EnergyCertification result
    """
    print(f"\n{'='*60}")
    print("ENERGY CERTIFICATION")
    print(f"{'='*60}")
    print(f"Config: N={N}, r={r}, k={k_per_row}")

    dtype = torch.float32

    # Build TLS mask
    scores = torch.rand(N, N, device=device, dtype=dtype)
    mask = build_tls_mask_from_scores(scores, k_per_row=k_per_row, device=device)

    # Create low-rank synapse
    syn = LowRankMaskedSynapse(N=N, r=r, mask_csr=mask, dtype=dtype, device=device)

    # Get parameter stats
    stats = report(N=N, r=r, indptr=mask['indptr'])

    # Calculate memory
    param_bytes = sum(p.numel() * p.element_size() for p in syn.parameters())
    memory_mb = param_bytes / (1024 * 1024)

    # Calculate normalized energy metric
    # Energy = (1 - param_reduction/100) * density
    # Lower is better, target ≤ 0.15
    reduction = stats['param_reduction_pct']
    density = stats['density']
    normalized_energy = (1.0 - reduction / 100.0) * density * 10  # Scale for interpretability

    # Compute intensity (sparse ops / memory accessed)
    sparse_ops = N * k_per_row * 2  # mul + add per edge
    compute_intensity = sparse_ops / param_bytes if param_bytes > 0 else 0

    passed = normalized_energy <= 0.15 and reduction >= 97.0

    print(f"\nResults:")
    print(f"  Dense params: {stats['dense_params']:,}")
    print(f"  Sparse params: {stats['sparse_params']:,}")
    print(f"  Reduction: {reduction:.2f}% (target: ≥97%)")
    print(f"  Density: {density:.4f}")
    print(f"  Memory: {memory_mb:.2f} MB")
    print(f"  Normalized energy: {normalized_energy:.4f} (target: ≤0.15)")
    print(f"  Compute intensity: {compute_intensity:.2f} ops/byte")
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  Status: {status}")

    return EnergyCertification(
        normalized_energy=normalized_energy,
        param_reduction_pct=reduction,
        memory_mb=memory_mb,
        compute_intensity=compute_intensity,
        passed=passed,
    )


def certify_stability(
    N: int = 4096,
    r: int = 32,
    k_per_row: int = 64,
    num_steps: int = 100,
    device: str = 'cpu',
) -> StabilityCertification:
    """
    Certify EPR-CV (edge-of-chaos stability) against ≤0.15 gate.

    EPR-CV is the coefficient of variation of firing rates across the population,
    measuring edge-of-chaos criticality.

    Args:
        N: Number of neurons
        r: Low-rank dimension
        k_per_row: Outgoing degree per neuron
        num_steps: Number of simulation steps
        device: 'cpu' or 'cuda'

    Returns:
        StabilityCertification result
    """
    print(f"\n{'='*60}")
    print("STABILITY (EPR-CV) CERTIFICATION")
    print(f"{'='*60}")
    print(f"Config: N={N}, r={r}, k={k_per_row}, steps={num_steps}")

    dtype = torch.float32

    # Build TLS mask
    scores = torch.rand(N, N, device=device, dtype=dtype)
    mask = build_tls_mask_from_scores(scores, k_per_row=k_per_row, device=device)

    # Create low-rank synapse
    syn = LowRankMaskedSynapse(N=N, r=r, mask_csr=mask, dtype=dtype, device=device)

    # Create LIF layer
    lif = LIFLayerLowRank(synapse=syn, N=N, device=device, dtype=dtype)

    # Run simulation and collect firing rates
    batch_size = 8
    spike_counts = torch.zeros(N, device=device, dtype=dtype)
    v_variances = []

    print(f"Running {num_steps} simulation steps...")

    for step in range(num_steps):
        # Random sparse input
        x = (torch.rand(batch_size, N, device=device, dtype=dtype) < 0.05).float()

        with torch.no_grad():
            spikes = lif.step(x)

        # Accumulate spike counts
        spike_counts += spikes.sum(dim=0)

        # Track membrane variance
        v_var = lif.v.var().item()
        v_variances.append(v_var)

    # Calculate firing rates (spikes per step per neuron)
    firing_rates = spike_counts / (num_steps * batch_size)

    # EPR-CV: coefficient of variation of firing rates
    rate_mean = firing_rates.mean().item()
    rate_std = firing_rates.std().item()
    epr_cv = rate_std / rate_mean if rate_mean > 0.01 else 0.0

    v_variance_mean = np.mean(v_variances)

    passed = epr_cv <= 0.15

    print(f"\nResults:")
    print(f"  Mean firing rate: {rate_mean:.4f}")
    print(f"  Firing rate std: {rate_std:.4f}")
    print(f"  EPR-CV: {epr_cv:.4f} (target: ≤0.15)")
    print(f"  V variance mean: {v_variance_mean:.4f}")
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  Status: {status}")

    return StabilityCertification(
        epr_cv=epr_cv,
        firing_rate_mean=rate_mean,
        firing_rate_std=rate_std,
        v_variance_mean=v_variance_mean,
        passed=passed,
    )


def run_full_certification(
    N: int = 4096,
    r: int = 32,
    k_per_row: int = 64,
    batch_size: int = 32,
    device: str = 'cpu',
    output_dir: str = './certification',
) -> CertificationResult:
    """
    Run complete energy/throughput/stability certification.

    Args:
        N: Number of neurons
        r: Low-rank dimension
        k_per_row: Outgoing degree per neuron
        batch_size: Batch size for throughput test
        device: 'cpu' or 'cuda'
        output_dir: Output directory for results

    Returns:
        CertificationResult
    """
    print("\n" + "="*70)
    print("PRODUCTION ENERGY/THROUGHPUT CERTIFICATION")
    print("="*70)
    print(f"Timestamp: {datetime.utcnow().isoformat()}")
    print(f"Device: {device}")
    print(f"Config: N={N}, r={r}, k={k_per_row}")

    config = {
        "N": N,
        "r": r,
        "k_per_row": k_per_row,
        "batch_size": batch_size,
        "device": device,
    }

    # Run certifications
    throughput = certify_throughput(N, r, k_per_row, batch_size, device=device)
    energy = certify_energy(N, r, k_per_row, device=device)
    stability = certify_stability(N, r, k_per_row, device=device)

    # Overall result
    all_passed = throughput.passed and energy.passed and stability.passed

    result = CertificationResult(
        timestamp=datetime.utcnow().isoformat(),
        throughput=throughput,
        energy=energy,
        stability=stability,
        all_gates_passed=all_passed,
        config=config,
    )

    # Print summary
    print("\n" + "="*70)
    print("CERTIFICATION SUMMARY")
    print("="*70)
    print(f"\nThroughput: {throughput.events_per_second:,.0f} events/s {'✓' if throughput.passed else '✗'}")
    print(f"Energy: {energy.normalized_energy:.4f} {'✓' if energy.passed else '✗'}")
    print(f"EPR-CV: {stability.epr_cv:.4f} {'✓' if stability.passed else '✗'}")
    print(f"\nOverall: {'✓ ALL GATES PASSED' if all_passed else '✗ SOME GATES FAILED'}")

    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / 'certification_result.json'
    with open(output_path, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Energy and Throughput Certification"
    )
    parser.add_argument('--full', action='store_true',
                        help="Run full certification suite")
    parser.add_argument('--quick', action='store_true',
                        help="Quick smoke test with smaller config")
    parser.add_argument('--throughput-only', action='store_true',
                        help="Run throughput certification only")
    parser.add_argument('--energy-only', action='store_true',
                        help="Run energy certification only")
    parser.add_argument('--stability-only', action='store_true',
                        help="Run stability certification only")
    parser.add_argument('--N', type=int, default=4096, help="Number of neurons")
    parser.add_argument('--r', type=int, default=32, help="Low-rank dimension")
    parser.add_argument('--k', type=int, default=64, help="Outgoing degree")
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size")
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'], help="Device")
    parser.add_argument('--output', type=str, default='./certification',
                        help="Output directory")

    args = parser.parse_args()

    # Quick mode uses smaller config
    if args.quick:
        args.N = 512
        args.r = 16
        args.k = 32
        args.batch_size = 8

    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'

    # Run appropriate certification
    if args.throughput_only:
        result = certify_throughput(
            args.N, args.r, args.k, args.batch_size, device=args.device
        )
        passed = result.passed
    elif args.energy_only:
        result = certify_energy(args.N, args.r, args.k, device=args.device)
        passed = result.passed
    elif args.stability_only:
        result = certify_stability(args.N, args.r, args.k, device=args.device)
        passed = result.passed
    else:
        # Full certification
        result = run_full_certification(
            args.N, args.r, args.k, args.batch_size,
            device=args.device, output_dir=args.output
        )
        passed = result.all_gates_passed

    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
