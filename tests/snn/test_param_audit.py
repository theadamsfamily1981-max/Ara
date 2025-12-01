"""
Unit tests for parameter auditing and gate validation.
"""

import pytest
import torch

from tfan.snn.auditors import (
    dense_params,
    lowrank_params,
    param_reduction_pct,
    assert_param_gate,
    assert_degree_gate,
    assert_rank_gate,
    verify_all_gates,
)
from tfan.snn.mask_tls import build_uniform_random_mask


def test_param_counts():
    """Test parameter counting functions."""
    N = 4096
    r = 32

    # Dense
    dense = dense_params(N)
    assert dense == N * N
    assert dense == 16_777_216

    # Low-rank
    lowrank = lowrank_params(N, r)
    assert lowrank == 2 * N * r
    assert lowrank == 262_144


def test_param_reduction_pct():
    """Test parameter reduction percentage calculation."""
    # N=4096, r=32 should give 98.4% reduction
    reduction = param_reduction_pct(N=4096, r=32)
    assert reduction > 98.0
    assert reduction < 99.0
    assert abs(reduction - 98.4375) < 0.01  # Expected value


def test_param_gate_pass():
    """Test parameter gate with passing configuration."""
    N, r = 4096, 32
    # Should pass with 97% requirement (actual: 98.4%)
    assert_param_gate(N, r, pct_required=97.0)


def test_param_gate_fail():
    """Test parameter gate with failing configuration."""
    N, r = 4096, 1000  # Too large rank
    # Should fail with 97% requirement
    with pytest.raises(AssertionError):
        assert_param_gate(N, r, pct_required=97.0)


def test_degree_gate_pass():
    """Test degree gate with sparse mask."""
    N = 4096
    k = 64  # 1.56% density
    mask = build_uniform_random_mask(N=N, k_per_row=k, seed=42)

    # Should pass with 2% requirement (actual: 1.56%)
    assert_degree_gate(mask['indptr'], N=N, max_frac=0.02)


def test_degree_gate_fail():
    """Test degree gate with dense mask."""
    N = 100
    k = 10  # 10% density - too dense
    mask = build_uniform_random_mask(N=N, k_per_row=k, seed=42)

    # Should fail with 2% requirement
    with pytest.raises(AssertionError):
        assert_degree_gate(mask['indptr'], N=N, max_frac=0.02)


def test_rank_gate_pass():
    """Test rank gate with low rank."""
    N = 4096
    r = 32  # 0.78% of N
    # Should pass with 2% requirement
    assert_rank_gate(N, r, max_frac=0.02)


def test_rank_gate_fail():
    """Test rank gate with high rank."""
    N = 1000
    r = 100  # 10% of N
    # Should fail with 2% requirement
    with pytest.raises(AssertionError):
        assert_rank_gate(N, r, max_frac=0.02)


def test_verify_all_gates_pass():
    """Test complete gate verification with passing config."""
    N = 4096
    r = 32
    k = 64
    mask = build_uniform_random_mask(N=N, k_per_row=k, seed=42)

    gates = verify_all_gates(
        N=N,
        r=r,
        indptr=mask['indptr'],
        param_reduction_min=97.0,
        degree_frac_max=0.02,
        rank_frac_max=0.02
    )

    assert gates['param_reduction'] is True
    assert gates['degree'] is True
    assert gates['rank'] is True
    assert all(gates.values())


def test_verify_all_gates_mixed():
    """Test gate verification with some failures."""
    N = 1000
    r = 500  # Too high rank (50% of N)
    k = 50  # Too high degree (5% of N)
    mask = build_uniform_random_mask(N=N, k_per_row=k, seed=42)

    gates = verify_all_gates(
        N=N,
        r=r,
        indptr=mask['indptr'],
        param_reduction_min=97.0,
        degree_frac_max=0.02,
        rank_frac_max=0.02
    )

    # Param reduction should fail (low reduction with high rank)
    assert gates['param_reduction'] is False
    # Degree should fail (5% > 2%)
    assert gates['degree'] is False
    # Rank should fail (50% > 2%)
    assert gates['rank'] is False


def test_default_config_gates():
    """Test gates for default configuration (from config file)."""
    # Default: N=4096, r=32, k=64
    N = 4096
    r = 32
    k = 64
    mask = build_uniform_random_mask(N=N, k_per_row=k, seed=42)

    # All gates should pass
    gates = verify_all_gates(
        N=N,
        r=r,
        indptr=mask['indptr'],
        param_reduction_min=97.0,
        degree_frac_max=0.02,
        rank_frac_max=0.02
    )

    assert all(gates.values()), f"Default config failed gates: {gates}"

    # Verify specific values
    reduction = param_reduction_pct(N, r)
    assert reduction >= 98.0, f"Reduction {reduction:.2f}% < 98%"

    avg_degree = float((mask['indptr'][1:] - mask['indptr'][:-1]).float().mean())
    assert avg_degree == k, f"Avg degree {avg_degree} != {k}"

    degree_frac = avg_degree / N
    assert degree_frac <= 0.02, f"Degree fraction {degree_frac:.4f} > 0.02"
