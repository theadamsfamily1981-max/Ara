"""
Binary Front-End Module
========================

The BinaryFrontEnd is the "sketching / hashing" layer that sits between
raw sensory input and the expensive SNN / T-FAN / QUANTA core.

Architecture:
    Raw Input → BinaryEncoder → Binary Code → [Optional: AssociativeMemory]
                     ↓
              TopologicalSketch → T-FAN / SNN Core

Benefits:
    - 10-100x cheaper than float matmul
    - Massive fan-out (millions of 1-bit synapses)
    - Natural interface to FPGA neuromorphic co-processors
    - Compressed codes for efficient storage/retrieval

Usage:
    from ara.neuro.binary.frontend import BinaryFrontEnd

    frontend = BinaryFrontEnd(
        input_dim=1024,
        hidden_dims=[2048, 4096, 2048],
        output_dim=512,
    )

    # Encode input to binary
    x = torch.randn(32, 1024)
    code, sketch = frontend(x)

    # code: (32, 512) binary output
    # sketch: (32, sketch_dim) topological features for T-FAN
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# Try torch import
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from .core import BinaryDense, pack_bits_torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


@dataclass
class FrontEndConfig:
    """Configuration for BinaryFrontEnd."""

    input_dim: int = 1024
    hidden_dims: List[int] = field(default_factory=lambda: [2048, 4096, 2048])
    output_dim: int = 512
    sketch_dim: int = 128  # Topological sketch for downstream

    # Architecture options
    use_batch_norm: bool = True
    dropout: float = 0.0
    activation: str = "hardtanh"  # hardtanh is binary-friendly

    # Training
    binary_regularization: float = 0.01  # Push weights toward {-1, +1}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "output_dim": self.output_dim,
            "sketch_dim": self.sketch_dim,
            "use_batch_norm": self.use_batch_norm,
            "dropout": self.dropout,
            "activation": self.activation,
        }


if TORCH_AVAILABLE:

    class BinaryEncoder(nn.Module):
        """
        Multi-layer binary encoder.

        Each layer: BinaryDense → BatchNorm (optional) → Activation
        """

        def __init__(self, config: FrontEndConfig):
            super().__init__()

            self.config = config
            dims = [config.input_dim] + config.hidden_dims + [config.output_dim]

            layers = []
            for i in range(len(dims) - 1):
                # Binary dense
                layers.append(BinaryDense(dims[i], dims[i + 1]))

                # Batch norm (optional, before activation)
                if config.use_batch_norm and i < len(dims) - 2:
                    layers.append(nn.BatchNorm1d(dims[i + 1]))

                # Activation
                if i < len(dims) - 2:
                    if config.activation == "hardtanh":
                        layers.append(nn.Hardtanh(-1, 1))
                    elif config.activation == "sign":
                        layers.append(SignActivation())
                    else:
                        layers.append(nn.ReLU())

                    # Dropout
                    if config.dropout > 0:
                        layers.append(nn.Dropout(config.dropout))

            self.layers = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Encode input to binary representation."""
            return self.layers(x)


    class SignActivation(nn.Module):
        """
        Sign activation with straight-through estimator for gradients.

        Forward: sign(x)
        Backward: identity (STE)
        """

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Sign function
            out = torch.sign(x)
            # Handle zeros
            out = torch.where(out == 0, torch.ones_like(out), out)

            if self.training:
                # Straight-through estimator
                return x + (out - x).detach()
            else:
                return out


    class TopologicalSketch(nn.Module):
        """
        Extract topological features from binary codes.

        This is a small network that produces a dense sketch
        suitable for feeding into T-FAN or other topology-aware models.

        Features extracted:
            - Population statistics (mean, variance of bits)
            - Local patterns (learned convolutions over bit sequence)
            - Global hash (compressed representation)
        """

        def __init__(self, code_dim: int, sketch_dim: int = 128):
            super().__init__()

            self.code_dim = code_dim
            self.sketch_dim = sketch_dim

            # 1D conv to detect local patterns in bit sequence
            self.local_conv = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=8, stride=4, padding=2),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
            )

            # Compute output size after convolutions
            # Input: code_dim, after conv1: (code_dim + 4 - 8) // 4 + 1
            # This is approximate; we'll use adaptive pooling
            self.pool = nn.AdaptiveAvgPool1d(16)

            # Project to sketch dim
            self.project = nn.Sequential(
                nn.Linear(64 * 16 + 3, sketch_dim),  # +3 for population stats
                nn.ReLU(),
                nn.Linear(sketch_dim, sketch_dim),
            )

        def forward(self, code: torch.Tensor) -> torch.Tensor:
            """
            Extract topological sketch from binary code.

            Args:
                code: Binary code of shape (batch, code_dim)

            Returns:
                Sketch of shape (batch, sketch_dim)
            """
            batch_size = code.shape[0]

            # Population statistics
            pop_mean = code.mean(dim=1, keepdim=True)
            pop_var = code.var(dim=1, keepdim=True)
            pop_density = (code > 0).float().mean(dim=1, keepdim=True)

            pop_stats = torch.cat([pop_mean, pop_var, pop_density], dim=1)

            # Local patterns via 1D conv
            code_1d = code.unsqueeze(1)  # (batch, 1, code_dim)
            local_features = self.local_conv(code_1d)  # (batch, 64, ?)
            local_features = self.pool(local_features)  # (batch, 64, 16)
            local_features = local_features.view(batch_size, -1)  # (batch, 64*16)

            # Combine and project
            combined = torch.cat([local_features, pop_stats], dim=1)
            sketch = self.project(combined)

            return sketch


    class BinaryFrontEnd(nn.Module):
        """
        Complete binary front-end for SNN/T-FAN integration.

        Components:
            1. BinaryEncoder: Multi-layer XNOR+popcount encoder
            2. TopologicalSketch: Extract topology features for downstream
            3. Optional: Hooks for FPGA offload

        Usage:
            frontend = BinaryFrontEnd(input_dim=1024, output_dim=512)
            code, sketch = frontend(x)

            # Send code to associative memory
            # Send sketch to T-FAN
        """

        def __init__(
            self,
            input_dim: int = 1024,
            hidden_dims: Optional[List[int]] = None,
            output_dim: int = 512,
            sketch_dim: int = 128,
            config: Optional[FrontEndConfig] = None,
        ):
            super().__init__()

            if config is None:
                config = FrontEndConfig(
                    input_dim=input_dim,
                    hidden_dims=hidden_dims or [2048, 4096, 2048],
                    output_dim=output_dim,
                    sketch_dim=sketch_dim,
                )

            self.config = config

            # Binary encoder
            self.encoder = BinaryEncoder(config)

            # Topological sketch extractor
            self.sketch = TopologicalSketch(config.output_dim, config.sketch_dim)

            # FPGA offload hooks (set externally)
            self._fpga_encoder = None
            self._use_fpga = False

            log.info(
                f"BinaryFrontEnd initialized: "
                f"{config.input_dim} → {config.hidden_dims} → {config.output_dim} "
                f"(sketch: {config.sketch_dim})"
            )

        def forward(
            self,
            x: torch.Tensor,
            return_packed: bool = False,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Encode input and extract topological sketch.

            Args:
                x: Input tensor of shape (batch, input_dim)
                return_packed: If True, return packed uint64 codes

            Returns:
                code: Binary code of shape (batch, output_dim)
                      or packed (batch, output_words) if return_packed
                sketch: Topological sketch of shape (batch, sketch_dim)
            """
            # Check for FPGA offload
            if self._use_fpga and self._fpga_encoder is not None:
                code = self._fpga_encoder(x)
            else:
                code = self.encoder(x)

            # Binarize output for clean codes
            code_binary = torch.sign(code)
            code_binary = torch.where(
                code_binary == 0,
                torch.ones_like(code_binary),
                code_binary
            )

            # Extract topological sketch
            sketch = self.sketch(code_binary)

            if return_packed:
                # Pack to uint64 for storage/transfer
                code_binary = pack_bits_torch(code_binary, threshold=0.0)

            return code_binary, sketch

        def encode_only(self, x: torch.Tensor) -> torch.Tensor:
            """Just encode, skip sketch extraction."""
            code = self.encoder(x)
            code_binary = torch.sign(code)
            return torch.where(code_binary == 0, torch.ones_like(code_binary), code_binary)

        def set_fpga_encoder(self, fpga_encoder: Any) -> None:
            """
            Set FPGA encoder for hardware offload.

            Args:
                fpga_encoder: Callable that takes tensor, returns encoded tensor
            """
            self._fpga_encoder = fpga_encoder
            self._use_fpga = True
            log.info("BinaryFrontEnd: FPGA encoder attached")

        def disable_fpga(self) -> None:
            """Disable FPGA offload, use GPU/CPU."""
            self._use_fpga = False
            log.info("BinaryFrontEnd: FPGA disabled, using local compute")

        def get_binary_regularization_loss(self) -> torch.Tensor:
            """
            Compute regularization loss to push weights toward {-1, +1}.

            This helps training converge to truly binary weights.
            """
            loss = torch.tensor(0.0, device=next(self.parameters()).device)

            for module in self.modules():
                if isinstance(module, BinaryDense):
                    # L2 distance from sign(weight)
                    w = module.weight
                    w_binary = torch.sign(w)
                    loss = loss + ((w - w_binary) ** 2).mean()

            return loss * self.config.binary_regularization


    class SensorySketcher(nn.Module):
        """
        Specialized front-end for sensory data (images, audio, telemetry).

        Combines:
            1. Modality-specific preprocessing
            2. Binary encoding
            3. Temporal integration (for sequences)
        """

        def __init__(
            self,
            modality: str = "generic",
            input_shape: Tuple[int, ...] = (1024,),
            code_dim: int = 512,
            temporal_window: int = 1,
        ):
            super().__init__()

            self.modality = modality
            self.input_shape = input_shape
            self.code_dim = code_dim
            self.temporal_window = temporal_window

            # Flatten input
            input_dim = int(np.prod(input_shape))

            # Modality-specific preprocessing
            if modality == "image":
                self.preprocess = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(input_dim, 2048),
                    nn.ReLU(),
                )
                input_dim = 2048
            elif modality == "audio":
                self.preprocess = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(input_dim, 1024),
                    nn.ReLU(),
                )
                input_dim = 1024
            else:
                self.preprocess = nn.Flatten()

            # Binary front-end
            self.frontend = BinaryFrontEnd(
                input_dim=input_dim,
                output_dim=code_dim,
            )

            # Temporal integration (if needed)
            if temporal_window > 1:
                self.temporal = nn.GRU(
                    input_size=code_dim,
                    hidden_size=code_dim,
                    batch_first=True,
                )
            else:
                self.temporal = None

        def forward(
            self,
            x: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Process sensory input.

            Args:
                x: Input tensor
                   - For single frame: (batch, *input_shape)
                   - For sequence: (batch, time, *input_shape)

            Returns:
                code: Binary code
                sketch: Topological sketch
            """
            if x.dim() == len(self.input_shape) + 2:
                # Sequence input
                batch, time, *_ = x.shape
                x_flat = x.view(batch * time, *self.input_shape)

                # Preprocess and encode
                x_pre = self.preprocess(x_flat)
                code, sketch = self.frontend(x_pre)

                # Reshape
                code = code.view(batch, time, -1)
                sketch = sketch.view(batch, time, -1)

                # Temporal integration
                if self.temporal is not None:
                    code, _ = self.temporal(code)
                    code = code[:, -1, :]  # Take last timestep
                    sketch = sketch[:, -1, :]

                return code, sketch
            else:
                # Single frame
                x_pre = self.preprocess(x)
                return self.frontend(x_pre)


# =============================================================================
# NUMPY-ONLY FALLBACK
# =============================================================================

class BinaryFrontEndNumpy:
    """
    NumPy-only implementation of BinaryFrontEnd.

    For environments without PyTorch.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dims: Optional[List[int]] = None,
        output_dim: int = 512,
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [2048, 4096, 2048]
        self.output_dim = output_dim

        # Initialize random binary weights
        dims = [input_dim] + self.hidden_dims + [output_dim]
        self.weights = []
        for i in range(len(dims) - 1):
            # Random binary weights: {-1, +1}
            w = np.sign(np.random.randn(dims[i + 1], dims[i]))
            w[w == 0] = 1
            self.weights.append(w)

        log.info(f"BinaryFrontEndNumpy: {dims}")

    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        Encode input to binary code.

        Args:
            x: Input array of shape (batch, input_dim)

        Returns:
            Binary code of shape (batch, output_dim)
        """
        h = x

        for i, w in enumerate(self.weights):
            # Binarize input
            h_bin = np.sign(h)
            h_bin[h_bin == 0] = 1

            # Binary matmul (equivalent to XNOR + popcount + scale)
            h = h_bin @ w.T

            # Activation (except last layer)
            if i < len(self.weights) - 1:
                h = np.clip(h, -1, 1)

        # Final binarization
        code = np.sign(h)
        code[code == 0] = 1

        return code


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'FrontEndConfig',
    'BinaryFrontEndNumpy',
]

if TORCH_AVAILABLE:
    __all__.extend([
        'BinaryEncoder',
        'SignActivation',
        'TopologicalSketch',
        'BinaryFrontEnd',
        'SensorySketcher',
    ])
