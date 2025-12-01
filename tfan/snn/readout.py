"""
Readout mechanisms for converting spike trains to predictions.

Implements:
- Spike count readout (sum over time)
- Membrane potential readout
- Optional CTC for sequence labeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpikeCountReadout(nn.Module):
    """
    Spike count readout: sum spikes over time, then linear classification.
    
    y = Linear(Σ_t s_t)
    
    Args:
        in_features: Input feature dimension
        out_features: Output classes
        bias: Whether to use bias in linear layer
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
    
    def forward(self, spike_traces: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spike_traces: [batch, T, in_features] or [batch, T, C, H, W]
            
        Returns:
            logits: [batch, out_features]
        """
        # Sum over time
        spike_counts = spike_traces.sum(dim=1)
        
        # Flatten if needed (for conv outputs)
        if spike_counts.dim() > 2:
            spike_counts = spike_counts.flatten(1)
        
        # Linear classification
        logits = self.linear(spike_counts)
        
        return logits


class MembraneReadout(nn.Module):
    """
    Membrane potential readout: average membrane voltage over time.
    
    Requires tracking membrane potentials during forward pass.
    
    Args:
        in_features: Input dimension
        out_features: Output classes
    """
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, membrane_traces: torch.Tensor) -> torch.Tensor:
        """
        Args:
            membrane_traces: [batch, T, in_features] membrane potentials
            
        Returns:
            logits: [batch, out_features]
        """
        # Average over time
        avg_membrane = membrane_traces.mean(dim=1)
        
        # Classification
        logits = self.linear(avg_membrane)
        
        return logits


class CTCReadout(nn.Module):
    """
    CTC (Connectionist Temporal Classification) readout for sequence tasks.
    
    Useful for audio/speech with temporal alignment.
    
    Args:
        in_features: Input feature dimension
        num_classes: Number of output classes (including blank)
    """
    
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)
        self.num_classes = num_classes
    
    def forward(self, spike_traces: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spike_traces: [batch, T, in_features]
            
        Returns:
            log_probs: [T, batch, num_classes] for CTC loss
        """
        # Linear projection at each timestep
        logits = self.linear(spike_traces)  # [batch, T, num_classes]
        
        # Log softmax
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Transpose for CTC: [T, batch, num_classes]
        log_probs = log_probs.transpose(0, 1)
        
        return log_probs


__all__ = ["SpikeCountReadout", "MembraneReadout", "CTCReadout"]
