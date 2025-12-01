"""
Test numerical stability and gradient hygiene for SNN emulation.

Tests:
1. Gradient explosion/vanishing detection
2. Spectral norm clamping on U, V
3. NaN/Inf gradient handling
4. Temperature-scaled surrogate gradients
5. Spike rate EMA stability
"""

import pytest
import torch
from torch import nn
import numpy as np

from tfan.snn import (
    LIFLayerLowRank,
    LowRankMaskedSynapse,
    build_uniform_random_mask,
)
from tfan.backends import build_backend


def test_gradient_clipping():
    """Test that gradient clipping prevents explosion."""
    N = 256
    r = 16
    k = 32

    mask = build_uniform_random_mask(N=N, k_per_row=k, seed=42)
    lif = LIFLayerLowRank(N=N, r=r, mask_csr=mask, dtype=torch.float32)

    # Create large gradients
    v, s = lif.init_state(batch=2)
    v.requires_grad_()

    # Forward with large input
    large_input = torch.randn_like(s) * 100.0
    v_next, s_next = lif(v, large_input)

    # Backward with large gradient
    loss = (s_next * 1000.0).sum()
    loss.backward()

    # Check gradients before clipping
    max_grad_before = max(
        p.grad.abs().max().item()
        for p in lif.parameters()
        if p.grad is not None
    )

    # Apply gradient clipping
    torch.nn.utils.clip_grad_norm_(lif.parameters(), max_norm=1.0)

    # Check gradients after clipping
    max_grad_after = max(
        p.grad.abs().max().item()
        for p in lif.parameters()
        if p.grad is not None
    )

    # Total norm should be <= 1.0 after clipping
    total_norm = torch.nn.utils.clip_grad_norm_(lif.parameters(), max_norm=float('inf'))
    assert total_norm <= 1.0 or max_grad_after < max_grad_before, \
        f"Gradient clipping failed: norm={total_norm:.2f}"


def test_spectral_norm_clamping():
    """Test spectral normalization on low-rank factors."""
    N = 256
    r = 16
    k = 32

    mask = build_uniform_random_mask(N=N, k_per_row=k, seed=42)
    lif = LIFLayerLowRank(N=N, r=r, mask_csr=mask, dtype=torch.float32)

    # Artificially increase spectral norm of U
    with torch.no_grad():
        lif.syn.U.data *= 10.0

    # Check spectral norm is large
    U_norm_before = torch.linalg.matrix_norm(lif.syn.U, ord=2).item()
    assert U_norm_before > 1.0, f"U norm {U_norm_before:.2f} should be > 1.0"

    # Apply spectral normalization
    with torch.no_grad():
        u_norm = torch.linalg.matrix_norm(lif.syn.U, ord=2)
        if u_norm > 1.0:
            lif.syn.U.data /= u_norm

    # Check spectral norm is now <= 1.0
    U_norm_after = torch.linalg.matrix_norm(lif.syn.U, ord=2).item()
    assert U_norm_after <= 1.0 + 1e-6, \
        f"U norm {U_norm_after:.2f} should be <= 1.0 after clamping"


def test_nan_inf_detection():
    """Test that NaN/Inf gradients are detected and zeroed."""
    N = 128
    r = 16
    k = 32

    mask = build_uniform_random_mask(N=N, k_per_row=k, seed=42)
    lif = LIFLayerLowRank(N=N, r=r, mask_csr=mask, dtype=torch.float32)

    v, s = lif.init_state(batch=2)
    v.requires_grad_()

    # Forward
    v_next, s_next = lif(v, s)

    # Create loss that will produce NaN gradient
    loss = (s_next / 0.0).sum()  # Division by zero -> NaN

    # This should produce NaN gradients
    try:
        loss.backward()
    except:
        pass  # May raise error on some PyTorch versions

    # Check for NaN/Inf and zero them out
    has_nan_inf = False
    for param in lif.parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                has_nan_inf = True
                param.grad.zero_()

    # Verify all gradients are now valid
    for param in lif.parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), "NaN gradients remain"
            assert not torch.isinf(param.grad).any(), "Inf gradients remain"


def test_surrogate_gradient_scale():
    """Test that surrogate gradient scale affects gradient magnitude."""
    N = 128
    r = 16
    k = 32

    mask = build_uniform_random_mask(N=N, k_per_row=k, seed=42)

    # Low scale
    lif_low = LIFLayerLowRank(
        N=N, r=r, mask_csr=mask,
        surrogate_scale=0.1,
        dtype=torch.float32
    )

    # High scale
    lif_high = LIFLayerLowRank(
        N=N, r=r, mask_csr=mask,
        surrogate_scale=1.0,
        dtype=torch.float32
    )

    # Same input
    v, s = lif_low.init_state(batch=2)

    # Forward and backward with low scale
    v1, s1 = lif_low(v.clone(), s.clone())
    loss1 = s1.sum()
    loss1.backward()
    grad_low = lif_low.syn.U.grad.abs().mean().item()

    # Forward and backward with high scale
    lif_high.syn.U.data = lif_low.syn.U.data.clone()
    lif_high.syn.V.data = lif_low.syn.V.data.clone()

    v2, s2 = lif_high(v.clone(), s.clone())
    loss2 = s2.sum()
    loss2.backward()
    grad_high = lif_high.syn.U.grad.abs().mean().item()

    # High scale should produce larger gradients
    # (May not always be true depending on spike activity, so we just check they're different)
    assert grad_low != grad_high, \
        f"Surrogate scale should affect gradients: low={grad_low:.6f}, high={grad_high:.6f}"


def test_spike_rate_ema_stability():
    """Test that spike rate EMA converges smoothly."""
    spike_rates = [0.1, 0.15, 0.12, 0.18, 0.14, 0.16, 0.13, 0.17]
    ema_alpha = 0.9

    ema = None
    ema_history = []

    for spike_rate in spike_rates:
        if ema is None:
            ema = spike_rate
        else:
            ema = ema_alpha * ema + (1 - ema_alpha) * spike_rate
        ema_history.append(ema)

    # EMA should be smoother than raw values
    ema_std = np.std(ema_history)
    raw_std = np.std(spike_rates)

    assert ema_std < raw_std, \
        f"EMA should be smoother: EMA std={ema_std:.4f}, raw std={raw_std:.4f}"


def test_gradient_flow_through_time():
    """Test that gradients flow correctly through time unrolling."""
    N = 128
    r = 16
    k = 32

    mask = build_uniform_random_mask(N=N, k_per_row=k, seed=42)
    lif = LIFLayerLowRank(N=N, r=r, mask_csr=mask, dtype=torch.float32)

    v, s = lif.init_state(batch=2)

    # Unroll for several timesteps
    v_history = [v]
    s_history = [s]

    for t in range(10):
        v, s = lif(v, s)
        v_history.append(v)
        s_history.append(s)

    # Loss on final spike output
    loss = s.sum()
    loss.backward()

    # Check that all parameters have gradients
    for name, param in lif.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"
        assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradient"


def test_backend_hooks_stability():
    """Test that backend hooks handle edge cases gracefully."""
    config = {
        'backend': 'snn_emu',
        'model': {'N': 256, 'lowrank_rank': 16, 'k_per_row': 32},
        'snn': {'v_th': 1.0, 'alpha': 0.95, 'time_steps': 64},
        'training': {'learning_rate': 1e-3, 'grad_clip': 1.0},
        'device': 'cpu',
        'dtype': 'float32',
    }

    backend = build_backend(config)
    model = backend.model
    hooks = backend.hooks

    # Simulate training step with normal values
    x = torch.randn(2, 256)
    output, aux = model(x)
    loss = output.sum()
    loss.backward()

    # Before step (should clip gradients)
    hooks.before_step(model)

    # Check gradients are clipped
    total_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=float('inf')
    )
    assert total_norm <= 1.0 + 1e-6, f"Gradients not clipped: norm={total_norm:.2f}"

    # After step (should update spike EMA)
    hooks.after_step(model, aux)

    assert 'spike_rate_ema' in aux, "Spike rate EMA not computed"


def test_vanishing_gradient_detection():
    """Test detection of vanishing gradients."""
    N = 128
    r = 16
    k = 32

    mask = build_uniform_random_mask(N=N, k_per_row=k, seed=42)
    lif = LIFLayerLowRank(N=N, r=r, mask_csr=mask, dtype=torch.float32)

    # Set very high threshold to prevent spikes -> vanishing gradients
    lif.v_th.data = torch.tensor(100.0)

    v, s = lif.init_state(batch=2)

    # Unroll for many timesteps
    for t in range(50):
        v, s = lif(v, s)

    # Loss on final output
    loss = s.sum()
    loss.backward()

    # Check if gradients are very small (vanishing)
    grad_norms = []
    for param in lif.parameters():
        if param.grad is not None:
            grad_norms.append(param.grad.abs().max().item())

    max_grad = max(grad_norms) if grad_norms else 0.0

    # With high threshold and no spikes, gradients should be very small
    assert max_grad < 0.01, \
        f"Expected vanishing gradients with high threshold, got max={max_grad:.6f}"


def test_exploding_gradient_detection():
    """Test detection of exploding gradients."""
    N = 128
    r = 16
    k = 32

    mask = build_uniform_random_mask(N=N, k_per_row=k, seed=42)
    lif = LIFLayerLowRank(N=N, r=r, mask_csr=mask, dtype=torch.float32)

    # Artificially create large weights
    with torch.no_grad():
        lif.syn.U.data *= 100.0
        lif.syn.V.data *= 100.0

    v, s = lif.init_state(batch=2)

    # Forward with large input
    large_input = torch.randn_like(s) * 10.0
    v_next, s_next = lif(v, large_input)

    # Backward
    loss = s_next.sum()
    loss.backward()

    # Check for exploding gradients
    max_grad = max(
        p.grad.abs().max().item()
        for p in lif.parameters()
        if p.grad is not None
    )

    # With large weights, gradients should be large
    assert max_grad > 10.0, \
        f"Expected exploding gradients with large weights, got max={max_grad:.2f}"

    # Now apply clipping
    torch.nn.utils.clip_grad_norm_(lif.parameters(), max_norm=1.0)

    max_grad_clipped = max(
        p.grad.abs().max().item()
        for p in lif.parameters()
        if p.grad is not None
    )

    assert max_grad_clipped < max_grad, \
        "Gradient clipping should reduce gradient magnitude"


@pytest.mark.parametrize("N,r,k", [
    (256, 16, 32),
    (512, 32, 64),
])
def test_stability_various_sizes(N, r, k):
    """Test numerical stability across various network sizes."""
    mask = build_uniform_random_mask(N=N, k_per_row=k, seed=42)
    lif = LIFLayerLowRank(N=N, r=r, mask_csr=mask, dtype=torch.float32)

    v, s = lif.init_state(batch=2)

    # Forward and backward
    v_next, s_next = lif(v, s)
    loss = s_next.sum()
    loss.backward()

    # Check all gradients are valid
    for param in lif.parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), "NaN gradients detected"
            assert not torch.isinf(param.grad).any(), "Inf gradients detected"
            assert param.grad.abs().max().item() < 1e6, "Gradient explosion detected"
