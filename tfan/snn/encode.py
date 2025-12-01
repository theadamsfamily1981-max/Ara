"""
Spike encoding schemes for converting continuous inputs to spike trains.

Implements:
- Rate coding (Poisson)
- Latency coding
- Delta/temporal coding
"""

import torch
import torch.nn as nn
from typing import Literal


class RateEncoder(nn.Module):
    """
    Rate (Poisson) encoding: spike probability ∝ input intensity.
    
    s_t ~ Bernoulli(x * λ)
    
    Args:
        lambda_scale: Scaling factor for spike rate
    """
    
    def __init__(self, lambda_scale: float = 1.0):
        super().__init__()
        self.lambda_scale = lambda_scale
    
    def forward(self, x: torch.Tensor, T: int) -> torch.Tensor:
        """
        Encode input as Poisson spike trains.
        
        Args:
            x: Input [batch, ...] in range [0, 1]
            T: Number of time steps
            
        Returns:
            spikes: [batch, T, ...] binary spike trains
        """
        batch_shape = x.shape
        
        # Clamp to [0, 1] and scale
        rate = torch.clamp(x * self.lambda_scale, 0, 1)
        
        # Generate Poisson spikes
        spikes = torch.rand(batch_shape[0], T, *batch_shape[1:], device=x.device)
        spikes = (spikes < rate.unsqueeze(1)).float()
        
        return spikes


class LatencyEncoder(nn.Module):
    """
    Latency coding: higher intensities spike earlier.
    
    t_spike = T * (1 - x)
    
    Args:
        T_max: Maximum latency (time steps)
    """
    
    def __init__(self, T_max: int = 256):
        super().__init__()
        self.T_max = T_max
    
    def forward(self, x: torch.Tensor, T: int = None) -> torch.Tensor:
        """
        Encode input as latency-coded spikes.
        
        Args:
            x: Input [batch, ...] in range [0, 1]
            T: Number of time steps (default: T_max)
            
        Returns:
            spikes: [batch, T, ...] with single spike per neuron
        """
        if T is None:
            T = self.T_max
        
        batch_shape = x.shape
        
        # Compute spike times: t = T * (1 - x)
        # x=1 -> t=0 (early), x=0 -> t=T-1 (late)
        spike_times = (T * (1 - torch.clamp(x, 0, 1))).long()
        spike_times = torch.clamp(spike_times, 0, T - 1)
        
        # Create spike trains
        spikes = torch.zeros(batch_shape[0], T, *batch_shape[1:], device=x.device)
        
        # Set spikes at computed times (vectorized)
        # This is a bit tricky - use scatter along time dimension
        for b in range(batch_shape[0]):
            flat_times = spike_times[b].flatten()
            flat_indices = torch.arange(flat_times.numel(), device=x.device)
            
            # Create time indices for each feature
            time_indices = flat_times.view(-1)
            feature_indices = flat_indices.view(-1)
            
            # Set spikes
            spikes[b].view(T, -1)[time_indices, feature_indices] = 1.0
        
        return spikes


class DeltaEncoder(nn.Module):
    """
    Delta/temporal coding: spike when input changes beyond threshold.
    
    s_t = 1 if |x_t - x_{t-1}| > ε else 0
    
    Args:
        threshold: Change threshold for spiking
    """
    
    def __init__(self, threshold: float = 0.1):
        super().__init__()
        self.threshold = threshold
    
    def forward(self, x: torch.Tensor, T: int = None) -> torch.Tensor:
        """
        Encode input as delta-modulated spikes.
        
        Args:
            x: Input [batch, T, ...] temporal sequence
            T: (ignored, uses x.shape[1])
            
        Returns:
            spikes: [batch, T, ...] spikes on changes
        """
        # Compute differences
        diff = torch.cat([
            torch.zeros_like(x[:, :1, ...]),
            x[:, 1:, ...] - x[:, :-1, ...]
        ], dim=1)
        
        # Spike on threshold crossing
        spikes = (torch.abs(diff) > self.threshold).float()
        
        return spikes


def create_encoder(encoder_type: Literal["rate", "latency", "delta"], **kwargs) -> nn.Module:
    """Factory function for creating encoders."""
    if encoder_type == "rate":
        return RateEncoder(**kwargs)
    elif encoder_type == "latency":
        return LatencyEncoder(**kwargs)
    elif encoder_type == "delta":
        return DeltaEncoder(**kwargs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


__all__ = ["RateEncoder", "LatencyEncoder", "DeltaEncoder", "create_encoder"]
