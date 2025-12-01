# tfan/snn/lif_lowrank.py
"""
Leaky Integrate-and-Fire neurons with low-rank masked synapses.

Integrates surrogate gradient training with parameter-efficient synaptic
connectivity for GPU-emulated SNNs.
"""

import torch
from torch import nn
from typing import Optional, Dict


class SurrogateSpikeFn(torch.autograd.Function):
    """
    Heaviside function with triangular surrogate gradient.

    Forward: H(v - v_th) = 1 if v > v_th else 0
    Backward: Triangular slope in window [-1, 1] around threshold

    This enables gradient-based training of discrete spiking neurons.
    """

    @staticmethod
    def forward(ctx, v_minus_th, scale=0.3):
        """
        Args:
            v_minus_th: Membrane potential minus threshold
            scale: Surrogate gradient scale factor

        Returns:
            Binary spike: 1 if v > v_th else 0
        """
        out = (v_minus_th > 0).to(v_minus_th.dtype)
        ctx.save_for_backward(v_minus_th)
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        Triangular surrogate: slope = max(0, 1 - |x|)

        Provides smooth gradients in [-1, 1] window around threshold.
        """
        (x,) = ctx.saved_tensors
        # Triangular surrogate slope in [-1, 1]
        slope = torch.clamp(1.0 - x.abs(), min=0.0)
        return grad_output * slope * ctx.scale, None


def spike_surrogate(v_minus_th, scale=0.3):
    """
    Apply spike function with surrogate gradient.

    Args:
        v_minus_th: Membrane potential minus threshold
        scale: Surrogate gradient scale

    Returns:
        Binary spikes with differentiable backward pass
    """
    return SurrogateSpikeFn.apply(v_minus_th, scale)


class LIFLayerLowRank(nn.Module):
    """
    LIF neuron layer with low-rank masked synapses.

    Dynamics:
        I_t = syn(s_{t-1})
        v_t = α·v_{t-1} + I_t - v_th·s_{t-1}
        s_t = H(v_t - v_th)

    where:
        - syn() is a LowRankMaskedSynapse (97-99% param reduction)
        - α is leak decay (learnable)
        - v_th is spike threshold (learnable)
        - H() is Heaviside with surrogate gradient

    Args:
        N: Number of neurons
        r: Low-rank dimension
        synapse_cls: Synapse class (default: LowRankMaskedSynapse)
        mask_csr: Sparse connectivity mask
        v_th: Spike threshold
        alpha: Membrane leak decay
        dtype: Parameter dtype
        device: Parameter device

    Example:
        >>> from tfan.snn.lowrank_synapse import LowRankMaskedSynapse
        >>> from tfan.snn.mask_tls import build_tls_mask_from_scores
        >>>
        >>> # Build topological mask
        >>> scores = torch.rand(512, 512)
        >>> mask = build_tls_mask_from_scores(scores, k_per_row=32)
        >>>
        >>> # Create LIF layer
        >>> lif = LIFLayerLowRank(
        ...     N=512, r=16, synapse_cls=LowRankMaskedSynapse,
        ...     mask_csr=mask, v_th=1.0, alpha=0.95
        ... )
        >>>
        >>> # Forward step
        >>> v = torch.zeros(2, 512)
        >>> s = torch.zeros(2, 512)
        >>> v_next, s_next = lif(v, s)
    """

    def __init__(
        self,
        N: int,
        r: int = 32,
        synapse_cls = None,
        mask_csr: Optional[Dict[str, torch.Tensor]] = None,
        v_th: float = 1.0,
        alpha: float = 0.95,
        surrogate_scale: float = 0.3,
        dtype: torch.dtype = torch.float16,
        device = None
    ):
        super().__init__()

        # Import here to avoid circular dependency
        if synapse_cls is None:
            from tfan.snn.lowrank_synapse import LowRankMaskedSynapse
            synapse_cls = LowRankMaskedSynapse

        # Synaptic connectivity
        self.syn = synapse_cls(
            N=N, r=r, mask_csr=mask_csr, dtype=dtype, device=device
        )

        # Neuron parameters (learnable)
        self.v_th = nn.Parameter(torch.tensor(v_th, dtype=dtype, device=device))
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=dtype, device=device))
        self.surrogate_scale = surrogate_scale

    def forward(self, v, s_prev, head_id: int = 0):
        """
        Single time step of LIF dynamics.

        Args:
            v: Membrane potential [batch, N]
            s_prev: Previous spikes [batch, N]
            head_id: Attention head index (for multi-head models)

        Returns:
            v_next: Updated membrane potential [batch, N]
            s: Output spikes [batch, N]
        """
        # Synaptic input from previous spikes
        I = self.syn(s_prev)  # [batch, N]

        # LIF update: v_t = α·v_{t-1} + I_t - v_th·s_{t-1}
        v_next = self.alpha * v + I - self.v_th * s_prev

        # Spike generation with surrogate gradient
        s = spike_surrogate(v_next - self.v_th, scale=self.surrogate_scale)

        return v_next, s

    def init_state(self, batch: int, device = None):
        """
        Initialize neuron state (membrane potential and spikes).

        Args:
            batch: Batch size
            device: Tensor device

        Returns:
            v: Zero membrane potential [batch, N]
            s: Zero spikes [batch, N]
        """
        N = self.syn.N
        dtype = self.v_th.dtype
        device = device or self.v_th.device
        v = torch.zeros(batch, N, dtype=dtype, device=device)
        s = torch.zeros(batch, N, dtype=dtype, device=device)
        return v, s

    def summary(self):
        """
        Print layer summary.
        """
        syn_summary = self.syn.summary()
        return {
            **syn_summary,
            'v_th': self.v_th.item(),
            'alpha': self.alpha.item(),
            'surrogate_scale': self.surrogate_scale,
        }


__all__ = ['LIFLayerLowRank', 'SurrogateSpikeFn', 'spike_surrogate']
