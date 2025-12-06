"""
Binary Neural Network Core Primitives
======================================

The 1-bit neuron stack: ridiculously wide, cheap correlator layers
that live right next to your HPV (high-performance vector) compute.

Core operations:
    - pack_bits(): Float/int tensors → packed uint64 bitplanes
    - unpack_bits(): Packed → expanded
    - xnor_popcount(): The fundamental 1-bit dot product
    - BinaryDense: PyTorch module for binary matrix multiply

Hardware mapping:
    - CPU: numpy bit operations (fast enough for prototyping)
    - CUDA: custom kernels with __popcll intrinsic
    - FPGA: XNOR + popcount trees in fabric (see fpga_core.py)

Usage:
    from ara.neuro.binary.core import BinaryDense, pack_bits

    # Create a binary layer
    layer = BinaryDense(in_features=1024, out_features=256)

    # Pack input to binary
    x_packed = pack_bits(x_float, threshold=0.0)

    # Forward pass (XNOR + popcount internally)
    y = layer(x_packed)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np

log = logging.getLogger(__name__)

# Try to import torch, but don't require it
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


# =============================================================================
# BIT PACKING / UNPACKING
# =============================================================================

def pack_bits_numpy(x: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    Pack float/int array into uint64 bitplanes.

    Args:
        x: Array of shape (..., N) where N is feature dimension
        threshold: Values >= threshold become 1, else 0

    Returns:
        Packed array of shape (..., ceil(N/64)) with dtype uint64
    """
    # Binarize
    binary = (x >= threshold).astype(np.uint8)

    # Pad to multiple of 64
    *batch_dims, n_features = binary.shape
    pad_to = ((n_features + 63) // 64) * 64
    if pad_to > n_features:
        pad_width = [(0, 0)] * len(batch_dims) + [(0, pad_to - n_features)]
        binary = np.pad(binary, pad_width, mode='constant', constant_values=0)

    # Reshape to (..., n_words, 64)
    n_words = pad_to // 64
    binary = binary.reshape(*batch_dims, n_words, 64)

    # Pack into uint64
    # Each bit position contributes 2^i to the word
    powers = (1 << np.arange(64, dtype=np.uint64))
    packed = (binary.astype(np.uint64) * powers).sum(axis=-1)

    return packed


def unpack_bits_numpy(packed: np.ndarray, n_features: int) -> np.ndarray:
    """
    Unpack uint64 bitplanes back to binary array.

    Args:
        packed: Packed array of shape (..., n_words)
        n_features: Original number of features

    Returns:
        Binary array of shape (..., n_features) with values 0 or 1
    """
    *batch_dims, n_words = packed.shape

    # Unpack each word
    result = np.zeros((*batch_dims, n_words * 64), dtype=np.uint8)

    for i in range(64):
        bit_i = ((packed >> i) & 1).astype(np.uint8)
        result[..., i::64] = bit_i

    # Trim to original size
    return result[..., :n_features]


def xnor_popcount_numpy(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    XNOR + popcount: the core 1-bit dot product.

    For each pair of uint64 words:
        same = ~(x ^ w)     # XNOR: 1 where bits match
        matches = popcount(same)
        dot = matches * 2 - 64  # Scale to {-64, ..., 64}

    Args:
        x: Packed input of shape (batch, n_words)
        w: Packed weights of shape (n_neurons, n_words)

    Returns:
        Output of shape (batch, n_neurons) as int32
    """
    batch_size, n_words = x.shape
    n_neurons, _ = w.shape

    # Expand for broadcasting: (batch, 1, n_words) XNOR (1, n_neurons, n_words)
    x_exp = x[:, np.newaxis, :]
    w_exp = w[np.newaxis, :, :]

    # XNOR
    xnor = ~(x_exp ^ w_exp)

    # Popcount (numpy doesn't have native popcount, so we use a trick)
    # Count bits set in each uint64
    def popcount64(arr):
        """Popcount for uint64 array."""
        # Use lookup table for bytes, sum across 8 bytes
        # This is slow but correct; for production, use compiled extension
        result = np.zeros(arr.shape, dtype=np.int32)
        for byte_idx in range(8):
            byte_val = ((arr >> (byte_idx * 8)) & 0xFF).astype(np.uint8)
            # Bit count lookup
            result += np.array([bin(b).count('1') for b in byte_val.flat],
                               dtype=np.int32).reshape(arr.shape)
        return result

    # Vectorized popcount using bit manipulation
    # This is faster than the loop above
    def popcount64_fast(arr):
        """Fast popcount using parallel bit counting."""
        arr = arr.astype(np.uint64)
        arr = arr - ((arr >> 1) & 0x5555555555555555)
        arr = (arr & 0x3333333333333333) + ((arr >> 2) & 0x3333333333333333)
        arr = (arr + (arr >> 4)) & 0x0F0F0F0F0F0F0F0F
        arr = (arr * 0x0101010101010101) >> 56
        return arr.astype(np.int32)

    matches = popcount64_fast(xnor)

    # Sum across words and scale
    total_matches = matches.sum(axis=-1)  # (batch, n_neurons)
    total_bits = n_words * 64

    # Scale to signed: matches * 2 - total_bits
    output = total_matches * 2 - total_bits

    return output.astype(np.int32)


# =============================================================================
# TORCH IMPLEMENTATIONS
# =============================================================================

if TORCH_AVAILABLE:

    def pack_bits_torch(x: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
        """
        Pack float tensor into uint64 bitplanes (CUDA-friendly).

        Args:
            x: Tensor of shape (..., N)
            threshold: Binarization threshold

        Returns:
            Packed tensor of shape (..., ceil(N/64)) as int64
            (PyTorch doesn't have uint64, so we use int64 and interpret bits)
        """
        # Binarize
        binary = (x >= threshold).to(torch.int64)

        # Pad to multiple of 64
        *batch_dims, n_features = binary.shape
        pad_to = ((n_features + 63) // 64) * 64
        if pad_to > n_features:
            padding = (0, pad_to - n_features)
            binary = F.pad(binary, padding, value=0)

        # Reshape to (..., n_words, 64)
        n_words = pad_to // 64
        binary = binary.view(*batch_dims, n_words, 64)

        # Pack: multiply by powers of 2 and sum
        powers = (1 << torch.arange(64, device=x.device, dtype=torch.int64))
        packed = (binary * powers).sum(dim=-1)

        return packed

    def xnor_popcount_torch(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        XNOR + popcount for PyTorch tensors.

        This is the CPU fallback; for CUDA, we'd want a custom kernel.
        """
        batch_size, n_words = x.shape
        n_neurons, _ = w.shape

        # Expand for broadcasting
        x_exp = x.unsqueeze(1)  # (batch, 1, n_words)
        w_exp = w.unsqueeze(0)  # (1, n_neurons, n_words)

        # XNOR (using XOR then NOT via subtraction trick)
        xor_result = x_exp ^ w_exp
        xnor = ~xor_result  # Bitwise NOT

        # Popcount - PyTorch doesn't have native popcount
        # We use the parallel bit counting algorithm
        def popcount64_torch(arr):
            """Parallel popcount for int64 tensors."""
            # Cast to uint64-equivalent by treating as unsigned
            # (Python int64 can hold all uint64 bit patterns)
            v = arr.to(torch.int64)

            # Parallel popcount algorithm
            v = v - ((v >> 1) & 0x5555555555555555)
            v = (v & 0x3333333333333333) + ((v >> 2) & 0x3333333333333333)
            v = (v + (v >> 4)) & 0x0F0F0F0F0F0F0F0F
            v = (v * 0x0101010101010101) >> 56
            return v.to(torch.int32)

        matches = popcount64_torch(xnor)

        # Sum across words
        total_matches = matches.sum(dim=-1)  # (batch, n_neurons)
        total_bits = n_words * 64

        # Scale to signed
        output = total_matches * 2 - total_bits

        return output

    class BinaryDense(nn.Module):
        """
        Binary (1-bit) dense layer.

        Weights are stored as packed uint64 bitplanes.
        Forward pass uses XNOR + popcount.

        This is the drop-in module for your SNN/T-FAN stack.

        Args:
            in_features: Number of input features (will be padded to multiple of 64)
            out_features: Number of output neurons
            bias: Whether to include bias (added after binary matmul)
        """

        def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
        ):
            super().__init__()

            self.in_features = in_features
            self.out_features = out_features

            # Pad input to multiple of 64
            self.in_words = (in_features + 63) // 64
            self.in_padded = self.in_words * 64

            # Binary weights: store as int64 (bit patterns)
            # Shape: (out_features, in_words)
            self.register_buffer(
                'weight_packed',
                torch.zeros(out_features, self.in_words, dtype=torch.int64)
            )

            # Full-precision weights for training/initialization
            # We binarize these to update weight_packed
            self.weight = nn.Parameter(
                torch.randn(out_features, in_features) * 0.01
            )

            if bias:
                self.bias = nn.Parameter(torch.zeros(out_features))
            else:
                self.register_parameter('bias', None)

            # Scaling factor (learned or fixed)
            self.scale = nn.Parameter(torch.ones(1))

            # Pack initial weights
            self._pack_weights()

        def _pack_weights(self) -> None:
            """Pack float weights into binary representation."""
            with torch.no_grad():
                # Binarize: sign function
                binary = (self.weight >= 0).to(torch.int64)

                # Pad if needed
                if self.in_padded > self.in_features:
                    padding = (0, self.in_padded - self.in_features)
                    binary = F.pad(binary, padding, value=0)

                # Reshape to (out_features, in_words, 64)
                binary = binary.view(self.out_features, self.in_words, 64)

                # Pack
                powers = (1 << torch.arange(64, device=self.weight.device, dtype=torch.int64))
                self.weight_packed = (binary * powers).sum(dim=-1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass with XNOR + popcount.

            Args:
                x: Input tensor of shape (batch, in_features) - float
                   OR already packed (batch, in_words) - int64

            Returns:
                Output tensor of shape (batch, out_features)
            """
            # Update packed weights (for training)
            if self.training:
                self._pack_weights()

            # Check if input is already packed
            if x.dtype == torch.int64 and x.shape[-1] == self.in_words:
                x_packed = x
            else:
                # Pack input
                x_packed = pack_bits_torch(x, threshold=0.0)

            # XNOR + popcount
            output = xnor_popcount_torch(x_packed, self.weight_packed)

            # Scale and add bias
            output = output.float() * self.scale

            if self.bias is not None:
                output = output + self.bias

            return output

        def extra_repr(self) -> str:
            return (
                f'in_features={self.in_features}, '
                f'out_features={self.out_features}, '
                f'in_words={self.in_words}, '
                f'bias={self.bias is not None}'
            )


    class BinaryConv2d(nn.Module):
        """
        Binary 2D convolution layer.

        Same XNOR + popcount principle, applied convolutionally.
        Useful for binary vision encoders.
        """

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 1,
            bias: bool = True,
        ):
            super().__init__()

            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = kernel_size
            self.stride = stride
            self.padding = padding

            # Full-precision weights for training
            self.weight = nn.Parameter(
                torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01
            )

            if bias:
                self.bias = nn.Parameter(torch.zeros(out_channels))
            else:
                self.register_parameter('bias', None)

            self.scale = nn.Parameter(torch.ones(1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Binary convolution.

            For simplicity, we use the "straight-through estimator" approach:
            - Binarize weights and inputs
            - Do regular conv (which becomes XNOR + popcount for binary inputs)
            - Scale output
            """
            # Binarize input: sign function
            x_bin = torch.sign(x)
            x_bin = torch.where(x_bin == 0, torch.ones_like(x_bin), x_bin)

            # Binarize weights
            w_bin = torch.sign(self.weight)
            w_bin = torch.where(w_bin == 0, torch.ones_like(w_bin), w_bin)

            # Regular conv with binary values
            # This is equivalent to XNOR + popcount when values are in {-1, +1}
            output = F.conv2d(
                x_bin, w_bin,
                bias=None,
                stride=self.stride,
                padding=self.padding
            )

            # Scale
            output = output * self.scale

            if self.bias is not None:
                output = output + self.bias.view(1, -1, 1, 1)

            return output


# =============================================================================
# UNIFIED API
# =============================================================================

def pack_bits(
    x: Union[np.ndarray, "torch.Tensor"],
    threshold: float = 0.0
) -> Union[np.ndarray, "torch.Tensor"]:
    """
    Pack array/tensor to binary bitplanes.

    Dispatches to numpy or torch implementation.
    """
    if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        return pack_bits_torch(x, threshold)
    else:
        return pack_bits_numpy(np.asarray(x), threshold)


def xnor_popcount(
    x: Union[np.ndarray, "torch.Tensor"],
    w: Union[np.ndarray, "torch.Tensor"],
) -> Union[np.ndarray, "torch.Tensor"]:
    """
    XNOR + popcount dot product.

    Dispatches to numpy or torch implementation.
    """
    if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        return xnor_popcount_torch(x, w)
    else:
        return xnor_popcount_numpy(np.asarray(x), np.asarray(w))


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Numpy primitives
    'pack_bits_numpy',
    'unpack_bits_numpy',
    'xnor_popcount_numpy',

    # Unified API
    'pack_bits',
    'xnor_popcount',

    # Constants
    'TORCH_AVAILABLE',
]

# Conditionally export torch classes
if TORCH_AVAILABLE:
    __all__.extend([
        'pack_bits_torch',
        'xnor_popcount_torch',
        'BinaryDense',
        'BinaryConv2d',
    ])
