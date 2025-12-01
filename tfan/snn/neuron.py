"""
Spiking neuron models with surrogate gradients for GPU-emulated SNN.

Implements:
- LIF (Leaky Integrate-and-Fire)
- PLIF (Parametric LIF with learnable tau)
- Izhikevich (simplified)
- Surrogate gradient functions for backpropagation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Literal
import math


# ============================================================================
# Surrogate Gradient Functions
# ============================================================================

class PiecewiseLinearSurrogate(torch.autograd.Function):
    """
    Piecewise linear surrogate gradient for Heaviside step function.

    Forward: H(x) = 1 if x > 0 else 0
    Backward: σ'(x) = clamp(1 - |x|/k, 0, 1) / k
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, k: float = 1.0):
        ctx.save_for_backward(x)
        ctx.k = k
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, = ctx.saved_tensors
        k = ctx.k
        # Piecewise linear: σ'(x) = max(0, 1 - |x|/k) / k
        grad_input = torch.clamp(1 - torch.abs(x) / k, min=0) / k
        return grad_output * grad_input, None


class FastSigmoidSurrogate(torch.autograd.Function):
    """
    Fast sigmoid surrogate gradient.

    Forward: H(x) = 1 if x > 0 else 0
    Backward: σ'(x) = 1 / (1 + |x|)^2
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, k: float = 1.0):
        ctx.save_for_backward(x)
        ctx.k = k
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, = ctx.saved_tensors
        k = ctx.k
        # Fast sigmoid: σ'(x) = k / (k + |x|)^2
        grad_input = k / ((k + torch.abs(x)) ** 2)
        return grad_output * grad_input, None


class ATanSurrogate(torch.autograd.Function):
    """
    Arctan surrogate gradient (smooth).

    Forward: H(x) = 1 if x > 0 else 0
    Backward: σ'(x) = k / (k^2 + x^2)
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, k: float = 1.0):
        ctx.save_for_backward(x)
        ctx.k = k
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, = ctx.saved_tensors
        k = ctx.k
        # Arctan derivative: σ'(x) = k / (k^2 + x^2)
        grad_input = k / (k**2 + x**2)
        return grad_output * grad_input, None


def get_surrogate_fn(surrogate_type: str = "plinear", k: float = 1.0):
    """Factory function to get surrogate gradient function."""
    if surrogate_type == "plinear":
        return lambda x: PiecewiseLinearSurrogate.apply(x, k)
    elif surrogate_type == "fastsigmoid":
        return lambda x: FastSigmoidSurrogate.apply(x, k)
    elif surrogate_type == "atan":
        return lambda x: ATanSurrogate.apply(x, k)
    else:
        raise ValueError(f"Unknown surrogate type: {surrogate_type}")


# ============================================================================
# Neuron State Management
# ============================================================================

class NeuronState:
    """Container for neuron state variables."""

    def __init__(self, v: torch.Tensor, s: Optional[torch.Tensor] = None):
        self.v = v  # Membrane potential
        self.s = s  # Spikes (previous timestep)

    def reset(self, mode: Literal["zeros", "carry"] = "zeros"):
        """Reset neuron state."""
        if mode == "zeros":
            self.v = torch.zeros_like(self.v)
            if self.s is not None:
                self.s = torch.zeros_like(self.s)
        # "carry" mode keeps current state

    def detach(self):
        """Detach state from computation graph."""
        self.v = self.v.detach()
        if self.s is not None:
            self.s = self.s.detach()
        return self


# ============================================================================
# LIF Neuron (Leaky Integrate-and-Fire)
# ============================================================================

class LIF(nn.Module):
    """
    Leaky Integrate-and-Fire neuron with surrogate gradient.

    Discrete update:
        v[t+1] = α * v[t] + (1-α) * (W*x[t] + b) - v_th * s[t]
        s[t+1] = H(v[t+1] - v_th)

    where α = exp(-Δt / τ)

    Args:
        tau_ms: Membrane time constant in milliseconds
        v_threshold: Spike threshold
        v_reset: Reset potential (default: 0)
        dt_ms: Simulation time step in ms
        surrogate_type: Surrogate gradient type
        surrogate_k: Surrogate gradient width parameter
        trainable: Whether tau is learnable
    """

    def __init__(
        self,
        tau_ms: float = 20.0,
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        dt_ms: float = 1.0,
        surrogate_type: str = "plinear",
        surrogate_k: float = 1.0,
        trainable: bool = False,
    ):
        super().__init__()

        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.dt_ms = dt_ms
        self.surrogate_fn = get_surrogate_fn(surrogate_type, surrogate_k)

        # Decay factor α = exp(-dt / tau)
        if trainable:
            # Store log(tau) for better optimization
            self.log_tau = nn.Parameter(torch.tensor(math.log(tau_ms)))
        else:
            self.register_buffer("log_tau", torch.tensor(math.log(tau_ms)))

        self.trainable = trainable

    @property
    def tau(self):
        """Get current tau value."""
        return torch.exp(self.log_tau)

    @property
    def alpha(self):
        """Get decay factor α = exp(-dt / tau)."""
        return torch.exp(-self.dt_ms / self.tau)

    def forward(
        self,
        x_t: torch.Tensor,
        state: Optional[NeuronState] = None,
    ) -> Tuple[torch.Tensor, NeuronState]:
        """
        Forward pass for one time step.

        Args:
            x_t: Input current [batch, ...]
            state: Previous neuron state (None = initialize to zeros)

        Returns:
            s_t: Output spikes [batch, ...]
            new_state: Updated neuron state
        """
        # Initialize state if needed
        if state is None:
            v = torch.zeros_like(x_t)
        else:
            v = state.v

        # Integrate: v = α * v + (1-α) * x_t
        alpha = self.alpha
        v_new = alpha * v + (1 - alpha) * x_t

        # Spike generation with surrogate gradient
        spike_input = v_new - self.v_threshold
        s_t = self.surrogate_fn(spike_input)

        # Reset: v = v - v_threshold * s
        v_new = v_new - (self.v_threshold - self.v_reset) * s_t

        # Create new state
        new_state = NeuronState(v=v_new, s=s_t)

        return s_t, new_state

    def reset_state(
        self,
        batch_size: int,
        *shape,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> NeuronState:
        """Create fresh neuron state."""
        v = torch.zeros(batch_size, *shape, device=device, dtype=dtype)
        return NeuronState(v=v)


# ============================================================================
# PLIF Neuron (Parametric LIF with learnable tau)
# ============================================================================

class PLIF(nn.Module):
    """
    Parametric LIF with learnable time constant per neuron.

    Args:
        num_neurons: Number of neurons (for per-neuron tau)
        tau_init_ms: Initial tau value
        tau_min_ms: Minimum tau (for stability)
        v_threshold: Spike threshold
        v_reset: Reset potential
        dt_ms: Simulation time step
        surrogate_type: Surrogate gradient type
        surrogate_k: Surrogate gradient width
    """

    def __init__(
        self,
        num_neurons: int,
        tau_init_ms: float = 20.0,
        tau_min_ms: float = 1.0,
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        dt_ms: float = 1.0,
        surrogate_type: str = "plinear",
        surrogate_k: float = 1.0,
    ):
        super().__init__()

        self.num_neurons = num_neurons
        self.tau_min_ms = tau_min_ms
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.dt_ms = dt_ms
        self.surrogate_fn = get_surrogate_fn(surrogate_type, surrogate_k)

        # Learnable tau per neuron
        # Use inverse softplus parametrization for stability
        init_val = math.log(math.exp(tau_init_ms - tau_min_ms) - 1)
        self.tau_param = nn.Parameter(torch.full((num_neurons,), init_val))

    @property
    def tau(self):
        """Get current tau values: tau = softplus(param) + tau_min."""
        return F.softplus(self.tau_param) + self.tau_min_ms

    @property
    def alpha(self):
        """Get decay factors α = exp(-dt / tau)."""
        return torch.exp(-self.dt_ms / self.tau)

    def forward(
        self,
        x_t: torch.Tensor,
        state: Optional[NeuronState] = None,
    ) -> Tuple[torch.Tensor, NeuronState]:
        """
        Forward pass for one time step.

        Args:
            x_t: Input current [batch, num_neurons, ...]
            state: Previous neuron state

        Returns:
            s_t: Output spikes
            new_state: Updated state
        """
        # Initialize state
        if state is None:
            v = torch.zeros_like(x_t)
        else:
            v = state.v

        # Get alpha (broadcast to batch dimension)
        alpha = self.alpha.view(1, -1, *([1] * (x_t.dim() - 2)))

        # Integrate
        v_new = alpha * v + (1 - alpha) * x_t

        # Spike generation
        spike_input = v_new - self.v_threshold
        s_t = self.surrogate_fn(spike_input)

        # Reset
        v_new = v_new - (self.v_threshold - self.v_reset) * s_t

        new_state = NeuronState(v=v_new, s=s_t)

        return s_t, new_state

    def reset_state(
        self,
        batch_size: int,
        *shape,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> NeuronState:
        """Create fresh neuron state."""
        v = torch.zeros(batch_size, self.num_neurons, *shape, device=device, dtype=dtype)
        return NeuronState(v=v)


# ============================================================================
# Izhikevich Neuron (simplified, 2-variable)
# ============================================================================

class IzhikevichNeuron(nn.Module):
    """
    Simplified Izhikevich neuron model.

    Dynamics:
        v' = 0.04*v^2 + 5*v + 140 - u + I
        u' = a*(b*v - u)
        if v >= v_th: v = c, u = u + d

    Args:
        a, b, c, d: Izhikevich parameters
        v_threshold: Spike threshold
        dt_ms: Time step
        surrogate_type: Surrogate for spike generation
    """

    def __init__(
        self,
        a: float = 0.02,
        b: float = 0.2,
        c: float = -65.0,
        d: float = 8.0,
        v_threshold: float = 30.0,
        dt_ms: float = 1.0,
        surrogate_type: str = "plinear",
        surrogate_k: float = 1.0,
    ):
        super().__init__()

        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.v_threshold = v_threshold
        self.dt_ms = dt_ms
        self.surrogate_fn = get_surrogate_fn(surrogate_type, surrogate_k)

    def forward(
        self,
        I_t: torch.Tensor,
        state: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass.

        Args:
            I_t: Input current
            state: dict with 'v' and 'u'

        Returns:
            s_t: Spikes
            new_state: Updated {v, u}
        """
        if state is None:
            v = torch.full_like(I_t, self.c)
            u = torch.zeros_like(I_t)
        else:
            v = state['v']
            u = state['u']

        # Update v and u (Euler integration)
        dv = (0.04 * v**2 + 5 * v + 140 - u + I_t) * self.dt_ms
        du = self.a * (self.b * v - u) * self.dt_ms

        v_new = v + dv
        u_new = u + du

        # Spike detection
        spike_input = v_new - self.v_threshold
        s_t = self.surrogate_fn(spike_input)

        # Reset on spike
        v_new = torch.where(s_t > 0.5, torch.full_like(v_new, self.c), v_new)
        u_new = torch.where(s_t > 0.5, u_new + self.d, u_new)

        new_state = {'v': v_new, 'u': u_new, 's': s_t}

        return s_t, new_state


__all__ = [
    "LIF",
    "PLIF",
    "IzhikevichNeuron",
    "NeuronState",
    "get_surrogate_fn",
    "PiecewiseLinearSurrogate",
    "FastSigmoidSurrogate",
    "ATanSurrogate",
]
