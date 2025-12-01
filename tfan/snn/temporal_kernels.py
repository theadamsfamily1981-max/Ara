# tfan/snn/temporal_kernels.py
"""
Temporal basis kernels for synaptic response modeling.

Implements shared synaptic kernel dictionaries to eliminate per-edge
temporal parameters, achieving major parameter reduction while maintaining
biologically plausible dynamics.
"""

import torch
from torch import nn


def exp_kernel_state(dt, tau, z_prev, spike_in):
    """
    Update exponential kernel state.

    Args:
        dt: Time step (ms)
        tau: Time constant (ms)
        z_prev: Previous kernel state
        spike_in: Input spike

    Returns:
        Updated kernel state: z(t) = exp(-dt/tau)*z(t-1) + spike_in
    """
    decay = torch.exp(-dt / tau)
    return decay * z_prev + spike_in


class TemporalBasis(nn.Module):
    """
    Small dictionary of synaptic kernels for temporal response modeling.

    Instead of learning per-edge FIR filters, we share a small set of
    basis kernels (typically B=4) with learnable per-head coefficients.
    This reduces temporal parameters from O(edges × filter_length) to
    O(heads × B), a massive reduction.

    Basis kernels use exponential decay with different time constants:
    - Fast (2ms): AMPA-like rapid excitation
    - Medium (4-8ms): Mixed dynamics
    - Slow (16ms): NMDA-like slow modulation

    Args:
        B: Number of basis kernels (default: 4)
        taus: Time constants in ms for each basis
        dtype: Tensor dtype
        device: Tensor device

    Example:
        >>> basis = TemporalBasis(B=4, taus=(2., 4., 8., 16.))
        >>> state = basis.init_state(batch=2, N=512, heads=8)
        >>> spikes = torch.rand(2, 512) > 0.95  # Sparse spikes
        >>> state = basis.step(state, spikes, dt=1.0)
    """

    def __init__(
        self,
        B: int = 4,
        taus: tuple = (2., 4., 8., 16.),
        dtype: torch.dtype = torch.float16,
        device = None
    ):
        super().__init__()
        taus = torch.tensor(list(taus)[:B], dtype=dtype, device=device)
        self.register_buffer('taus', taus)
        self.B = len(taus)

    def init_state(
        self,
        batch: int,
        N: int,
        heads: int = 1,
        dtype: torch.dtype = torch.float16,
        device = None
    ):
        """
        Initialize basis kernel states.

        Args:
            batch: Batch size
            N: Number of neurons
            heads: Number of attention heads
            dtype: Tensor dtype
            device: Tensor device

        Returns:
            state: [heads, batch, N, B] zero tensor
        """
        return torch.zeros(heads, batch, N, self.B, dtype=dtype, device=device)

    def step(self, z, spikes_t, dt: float = 1.0):
        """
        Update all basis kernel states given input spikes.

        Args:
            z: Current state [heads, batch, N, B]
            spikes_t: Input spikes [batch, N]
            dt: Time step (ms)

        Returns:
            Updated state [heads, batch, N, B]
        """
        # z: [H, Batch, N, B], spikes_t: [Batch, N] (broadcast over heads/B)
        H, B = z.shape[0], z.shape[-1]
        assert B == self.B, f"State has {B} basis kernels, expected {self.B}"

        # Broadcast spikes to [H, Batch, N, B]
        spikes = spikes_t.unsqueeze(0).unsqueeze(-1).expand(H, -1, -1, B)

        # Compute decay factors for each basis
        taus = self.taus.view(1, 1, 1, B)
        decay = torch.exp(torch.tensor(-dt, dtype=z.dtype, device=z.device) / taus)

        # Update: z(t) = decay * z(t-1) + spikes
        return decay * z + spikes

    def get_response(self, z, coeffs):
        """
        Compute weighted sum of basis kernels.

        Args:
            z: Basis states [heads, batch, N, B]
            coeffs: Per-head coefficients [heads, B]

        Returns:
            Combined response [heads, batch, N]
        """
        # coeffs: [H, B] -> [H, 1, 1, B]
        c = coeffs.view(coeffs.shape[0], 1, 1, -1)
        # z: [H, Batch, N, B], c: [H, 1, 1, B]
        return (z * c).sum(dim=-1)  # [H, Batch, N]


class AlphaKernel(nn.Module):
    """
    Alpha function kernel: k(t) = (t/tau) * exp(-t/tau)

    Provides more realistic synaptic waveform with rise and fall time.
    Used for higher-fidelity modeling when needed.
    """

    def __init__(self, tau: float = 5.0, dtype: torch.dtype = torch.float16):
        super().__init__()
        self.register_buffer('tau', torch.tensor(tau, dtype=dtype))

    def response(self, t):
        """
        Compute alpha function response at time t.

        Args:
            t: Time since spike (ms)

        Returns:
            Alpha function value
        """
        return (t / self.tau) * torch.exp(-t / self.tau)


__all__ = ['TemporalBasis', 'AlphaKernel', 'exp_kernel_state']
