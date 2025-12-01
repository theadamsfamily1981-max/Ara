"""
Unit tests for forward pass correctness.
"""

import pytest
import torch

from tfan.snn import (
    LowRankMaskedSynapse,
    LIFLayerLowRank,
    build_uniform_random_mask,
    TemporalBasis,
)


def test_lowrank_synapse_shapes():
    """Test that low-rank synapse produces correct output shapes."""
    N = 512
    r = 16
    batch_size = 2

    syn = LowRankMaskedSynapse(N=N, r=r, mask_csr=None, dtype=torch.float32)

    x = torch.randn(batch_size, N)
    y = syn(x)

    assert y.shape == (batch_size, N)


def test_lowrank_synapse_with_mask():
    """Test low-rank synapse with sparse mask."""
    N = 256
    r = 16
    k = 32
    batch_size = 4

    # Build mask
    mask = build_uniform_random_mask(N=N, k_per_row=k, seed=42)

    # Create synapse
    syn = LowRankMaskedSynapse(
        N=N, r=r, mask_csr=mask, dtype=torch.float32, dense_fallback=True
    )

    x = torch.randn(batch_size, N)
    y = syn(x)

    assert y.shape == (batch_size, N)
    # Output should have some zeros due to masking
    assert (y == 0).any()


def test_lif_layer_forward():
    """Test LIF layer forward pass."""
    N = 256
    r = 16
    k = 32
    batch_size = 2

    # Build mask
    mask = build_uniform_random_mask(N=N, k_per_row=k, seed=42)

    # Create LIF layer
    lif = LIFLayerLowRank(
        N=N,
        r=r,
        synapse_cls=LowRankMaskedSynapse,
        mask_csr=mask,
        v_th=1.0,
        alpha=0.95,
        dtype=torch.float32
    )

    # Initialize state
    v, s = lif.init_state(batch=batch_size)

    assert v.shape == (batch_size, N)
    assert s.shape == (batch_size, N)
    assert (v == 0).all()
    assert (s == 0).all()

    # Forward step
    v_next, s_next = lif(v, s)

    assert v_next.shape == (batch_size, N)
    assert s_next.shape == (batch_size, N)


def test_lif_layer_spiking():
    """Test that LIF layer can produce spikes."""
    N = 128
    r = 16
    batch_size = 1

    lif = LIFLayerLowRank(
        N=N,
        r=r,
        synapse_cls=LowRankMaskedSynapse,
        mask_csr=None,
        v_th=1.0,
        alpha=0.95,
        dtype=torch.float32
    )

    v, s = lif.init_state(batch=batch_size)

    # Run for several timesteps
    for t in range(100):
        v, s = lif(v, s)

    # Should have produced some spikes
    assert s.max() > 0, "No spikes produced after 100 timesteps"


def test_temporal_basis():
    """Test temporal basis kernel updates."""
    B = 4
    N = 128
    batch_size = 2
    heads = 8

    basis = TemporalBasis(B=B, taus=(2., 4., 8., 16.), dtype=torch.float32)

    # Initialize state
    state = basis.init_state(batch=batch_size, N=N, heads=heads, dtype=torch.float32)

    assert state.shape == (heads, batch_size, N, B)
    assert (state == 0).all()

    # Generate sparse spikes
    spikes = (torch.rand(batch_size, N) > 0.95).float()

    # Update
    state_next = basis.step(state, spikes, dt=1.0)

    assert state_next.shape == (heads, batch_size, N, B)
    # State should have updated where spikes occurred
    if spikes.sum() > 0:
        assert not (state_next == 0).all()


def test_gradient_flow():
    """Test that gradients flow through LIF layer."""
    N = 64
    r = 8
    batch_size = 2

    lif = LIFLayerLowRank(
        N=N,
        r=r,
        synapse_cls=LowRankMaskedSynapse,
        mask_csr=None,
        v_th=1.0,
        alpha=0.95,
        dtype=torch.float32
    )

    v, s = lif.init_state(batch=batch_size)

    # Ensure gradients are tracked
    v.requires_grad_()

    # Forward steps
    v1, s1 = lif(v, s)
    v2, s2 = lif(v1, s1)

    # Create loss (maximize firing)
    loss = -s2.sum()

    # Backward
    loss.backward()

    # Check gradients exist
    assert v.grad is not None
    assert not (v.grad == 0).all(), "All gradients are zero"

    # Check synapse parameters have gradients
    for param in lif.syn.parameters():
        assert param.grad is not None


def test_deterministic_forward():
    """Test that forward pass is deterministic with same seed."""
    N = 128
    r = 16
    batch_size = 2

    torch.manual_seed(42)
    mask1 = build_uniform_random_mask(N=N, k_per_row=32, seed=42)
    lif1 = LIFLayerLowRank(N=N, r=r, mask_csr=mask1, dtype=torch.float32)
    v1, s1 = lif1.init_state(batch=batch_size)
    v1_next, s1_next = lif1(v1, s1)

    torch.manual_seed(42)
    mask2 = build_uniform_random_mask(N=N, k_per_row=32, seed=42)
    lif2 = LIFLayerLowRank(N=N, r=r, mask_csr=mask2, dtype=torch.float32)
    v2, s2 = lif2.init_state(batch=batch_size)
    v2_next, s2_next = lif2(v2, s2)

    # Should be identical
    torch.testing.assert_close(v1_next, v2_next)
    torch.testing.assert_close(s1_next, s2_next)


@pytest.mark.parametrize("N,r,k", [
    (256, 16, 32),
    (512, 16, 64),
    (1024, 32, 64),
])
def test_various_sizes(N, r, k):
    """Test various network sizes."""
    batch_size = 2

    mask = build_uniform_random_mask(N=N, k_per_row=k, seed=42)
    lif = LIFLayerLowRank(N=N, r=r, mask_csr=mask, dtype=torch.float32)

    v, s = lif.init_state(batch=batch_size)
    v_next, s_next = lif(v, s)

    assert v_next.shape == (batch_size, N)
    assert s_next.shape == (batch_size, N)
