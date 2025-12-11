"""
CTF-1: Memory Metrics

Two key measurements:

1. Memory Capacity M_W(λ):
   - Jaeger-style: Sum of R² for reconstructing past inputs
   - M_W = Σ_k R²_k where k is the lag
   - Should peak at λ ≈ 1

2. Avalanche Statistics:
   - Size distribution P(s) ~ s^{-τ}
   - At criticality: τ = 3/2 (Beggs & Plenz)
   - Sub/supercritical: exponential cutoff

These are the empirical signatures of edge-of-chaos.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from .critical_core import CriticalCore


# =============================================================================
# Memory Capacity: M_W(λ)
# =============================================================================

def compute_memory_capacity(
    core: CriticalCore,
    T: int = 2000,
    max_lag: int = 50,
    washout: int = 100,
) -> Tuple[float, List[float]]:
    """
    Compute memory capacity via Jaeger's method.

    Drive reservoir with random input, then train linear readouts
    to reconstruct input at various lags. Sum of R² = memory capacity.

    Args:
        core: CriticalCore instance
        T: Length of input sequence
        max_lag: Maximum lag to test
        washout: Initial transient to discard

    Returns:
        (total_M_W, list of R²_k per lag)
    """
    core.reset()

    # Generate random input
    inputs = np.random.randn(T)

    # Collect reservoir states
    states = []
    for t, u in enumerate(inputs):
        core.step(np.array([u]))
        if t >= washout:
            states.append(core.x.copy())

    X = np.array(states)  # (T - washout, n)
    U = inputs[washout:]  # (T - washout,)

    # Compute R² for each lag
    R2_per_lag = []
    total_capacity = 0.0

    for k in range(1, min(max_lag + 1, len(U))):
        # Target: input at lag k
        target = U[:-k]
        X_k = X[k:]

        if len(target) < 10:
            break

        # Linear regression: target = X_k @ w
        try:
            w, residuals, rank, s = np.linalg.lstsq(X_k, target, rcond=None)
            pred = X_k @ w

            ss_res = np.sum((target - pred) ** 2)
            ss_tot = np.sum((target - target.mean()) ** 2)

            if ss_tot > 1e-10:
                R2 = 1.0 - ss_res / ss_tot
                R2 = max(0.0, R2)  # Clamp negative R²
            else:
                R2 = 0.0
        except:
            R2 = 0.0

        R2_per_lag.append(R2)
        total_capacity += R2

    return total_capacity, R2_per_lag


def lambda_sweep_memory(
    lambdas: np.ndarray,
    n_dims: int = 100,
    T: int = 2000,
    max_lag: int = 50,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Sweep λ and compute M_W at each point.

    This produces the classic "memory peaks at criticality" curve.

    Args:
        lambdas: Array of λ values to test
        n_dims: Reservoir dimension
        T: Sequence length
        max_lag: Max lag for memory test
        seed: Random seed

    Returns:
        Dictionary with 'lambdas', 'M_W', 'E' arrays
    """
    M_W_values = []
    E_values = []

    for lam in lambdas:
        core = CriticalCore(n_dims=n_dims, lambda_init=lam, seed=seed)
        M_W, _ = compute_memory_capacity(core, T=T, max_lag=max_lag)
        E = core.E_spectral()

        M_W_values.append(M_W)
        E_values.append(E)

    return {
        'lambdas': np.array(lambdas),
        'M_W': np.array(M_W_values),
        'E': np.array(E_values),
    }


# =============================================================================
# Avalanche Statistics
# =============================================================================

def measure_avalanches(
    core: CriticalCore,
    n_trials: int = 1000,
    threshold: float = 0.5,
    max_steps: int = 100,
    perturbation_size: float = 0.1,
) -> Dict[str, List]:
    """
    Measure avalanche size/duration distribution.

    Protocol:
    1. Reset to quiescent state
    2. Apply small perturbation
    3. Count active units until quiescence
    4. Record avalanche size and duration

    Args:
        core: CriticalCore instance
        n_trials: Number of avalanches to trigger
        threshold: Activity threshold
        max_steps: Max steps per avalanche
        perturbation_size: Size of initial kick

    Returns:
        Dictionary with 'sizes', 'durations' lists
    """
    sizes = []
    durations = []

    for _ in range(n_trials):
        # Reset to small random state
        core.x = np.random.randn(core.n) * 0.01

        # Apply perturbation to random subset
        n_perturb = max(1, core.n // 20)
        perturb_idx = np.random.choice(core.n, n_perturb, replace=False)
        core.x[perturb_idx] += perturbation_size

        # Track avalanche
        size = 0
        duration = 0

        for step in range(max_steps):
            # Count active units
            active = np.sum(np.abs(core.x) > threshold)
            size += active
            duration += 1

            if active == 0:
                break

            # Step dynamics (no external input)
            core.step()

        if size > 0:
            sizes.append(size)
            durations.append(duration)

    return {
        'sizes': sizes,
        'durations': durations,
    }


def estimate_power_law_exponent(sizes: List[int], s_min: int = 5) -> float:
    """
    Estimate power-law exponent τ from avalanche sizes.

    Uses maximum likelihood estimation:
    τ = 1 + n / Σ ln(s_i / s_min)

    At criticality, expect τ ≈ 1.5 (3/2).

    Args:
        sizes: List of avalanche sizes
        s_min: Minimum size cutoff

    Returns:
        Estimated exponent τ
    """
    sizes_arr = np.array([s for s in sizes if s >= s_min])

    if len(sizes_arr) < 10:
        return 0.0

    n = len(sizes_arr)
    tau = 1.0 + n / np.sum(np.log(sizes_arr / s_min))

    return float(np.clip(tau, 0.5, 4.0))


def avalanche_analysis(
    core: CriticalCore,
    n_trials: int = 500,
) -> Dict[str, float]:
    """
    Full avalanche analysis.

    Returns:
        Dictionary with 'tau', 'mean_size', 'mean_duration', 'cv_size'
    """
    data = measure_avalanches(core, n_trials=n_trials)

    sizes = data['sizes']
    durations = data['durations']

    if not sizes:
        return {
            'tau': 0.0,
            'mean_size': 0.0,
            'mean_duration': 0.0,
            'cv_size': 0.0,
        }

    tau = estimate_power_law_exponent(sizes)
    mean_size = np.mean(sizes)
    mean_duration = np.mean(durations)
    cv_size = np.std(sizes) / (mean_size + 1e-10)  # Coefficient of variation

    return {
        'tau': tau,
        'mean_size': mean_size,
        'mean_duration': mean_duration,
        'cv_size': cv_size,
    }


def lambda_sweep_avalanches(
    lambdas: np.ndarray,
    n_dims: int = 100,
    n_trials: int = 300,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Sweep λ and measure avalanche statistics at each point.

    Args:
        lambdas: Array of λ values
        n_dims: Reservoir dimension
        n_trials: Avalanches per λ
        seed: Random seed

    Returns:
        Dictionary with arrays: 'lambdas', 'tau', 'mean_size', 'cv_size'
    """
    tau_values = []
    mean_size_values = []
    cv_values = []

    for lam in lambdas:
        core = CriticalCore(n_dims=n_dims, lambda_init=lam, seed=seed)
        stats = avalanche_analysis(core, n_trials=n_trials)

        tau_values.append(stats['tau'])
        mean_size_values.append(stats['mean_size'])
        cv_values.append(stats['cv_size'])

    return {
        'lambdas': np.array(lambdas),
        'tau': np.array(tau_values),
        'mean_size': np.array(mean_size_values),
        'cv_size': np.array(cv_values),
    }


# =============================================================================
# Quick test
# =============================================================================

if __name__ == "__main__":
    print("CTF-1: Memory Metrics Test")
    print("=" * 50)

    # Test memory capacity at different λ
    print("\n1. Memory Capacity vs λ:")
    for lam in [0.7, 0.9, 1.0, 1.1, 1.3]:
        core = CriticalCore(n_dims=100, lambda_init=lam, seed=42)
        M_W, _ = compute_memory_capacity(core, T=1000, max_lag=30)
        E = core.E_spectral()
        print(f"  λ={lam:.1f}: M_W={M_W:.2f}, E={E:+.3f}")

    # Test avalanche statistics
    print("\n2. Avalanche Statistics vs λ:")
    for lam in [0.7, 0.9, 1.0, 1.1, 1.3]:
        core = CriticalCore(n_dims=100, lambda_init=lam, seed=42)
        stats = avalanche_analysis(core, n_trials=200)
        print(f"  λ={lam:.1f}: τ={stats['tau']:.2f}, "
              f"<s>={stats['mean_size']:.1f}, CV={stats['cv_size']:.2f}")

    print("\nExpected: M_W peaks at λ≈1, τ≈1.5 at criticality")
