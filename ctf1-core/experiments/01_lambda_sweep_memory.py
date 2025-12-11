#!/usr/bin/env python3
"""
Experiment 01: Lambda Sweep - Memory Capacity

Question: Does memory capacity M_W peak at λ = 1 (criticality)?

Protocol:
1. For λ ∈ [0.5, 1.5], create CriticalCore
2. Drive with random input, collect states
3. Train linear readouts to reconstruct past inputs (lags 1..K)
4. Sum R²_k to get M_W(λ)
5. Plot M_W vs λ

Expected result: M_W peaks near λ = 1.0
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core import CriticalCore, compute_memory_capacity, lambda_sweep_memory


def main():
    print("=" * 60)
    print("Experiment 01: Lambda Sweep - Memory Capacity")
    print("=" * 60)

    # Parameters
    lambdas = np.linspace(0.5, 1.5, 21)
    n_dims = 100
    T = 2000
    max_lag = 50
    seed = 42

    print(f"\nParameters:")
    print(f"  λ range: [{lambdas[0]:.2f}, {lambdas[-1]:.2f}]")
    print(f"  Reservoir dim: {n_dims}")
    print(f"  Sequence length: {T}")
    print(f"  Max lag: {max_lag}")

    # Run sweep
    print("\nRunning λ sweep...")
    results = lambda_sweep_memory(lambdas, n_dims=n_dims, T=T, max_lag=max_lag, seed=seed)

    # Find peak
    peak_idx = np.argmax(results['M_W'])
    peak_lambda = results['lambdas'][peak_idx]
    peak_M_W = results['M_W'][peak_idx]

    print(f"\nResults:")
    print("-" * 40)
    print(f"{'λ':>8} {'M_W':>10} {'E(λ)':>10}")
    print("-" * 40)

    for i in range(0, len(lambdas), 2):  # Print every other
        lam = results['lambdas'][i]
        M = results['M_W'][i]
        E = results['E'][i]
        marker = " <-- PEAK" if i == peak_idx else ""
        print(f"{lam:>8.2f} {M:>10.2f} {E:>+10.3f}{marker}")

    print("-" * 40)
    print(f"\nPeak: λ = {peak_lambda:.3f}, M_W = {peak_M_W:.2f}")

    # Verify prediction
    if 0.9 <= peak_lambda <= 1.1:
        print("\n✓ CONFIRMED: Memory capacity peaks near criticality (λ ≈ 1)")
    else:
        print(f"\n✗ Unexpected: Peak at λ = {peak_lambda:.2f}, expected near 1.0")

    # Save results
    output_file = os.path.join(os.path.dirname(__file__), '..', 'plots', 'memory_vs_lambda.npz')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.savez(output_file, **results)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
