"""
EEG Encoder for AraBrain

Conv1D-based encoder for EEG time series that extracts hierarchical
temporal features and projects to a disentangled latent space.

Architecture:
    EEG (B, T, C) → Conv1D layers → Flatten → Dense → μ, logvar

The encoder treats:
    - Time as the sequence dimension (conv kernel slides over time)
    - Channels as input features at each timestep

This connects to the NeuroBalance framework by learning latent
representations that can be probed for precision-related features
(D_low, D_high, theta-gamma coupling).

Usage:
    from ara.neuro.arabrain.encoder import EEGEncoder, EEGDecoder

    encoder = EEGEncoder(latent_dim=32)
    decoder = EEGDecoder(latent_dim=32, output_shape=(256, 32))

    mu, logvar = encoder(x)  # x: (B, T, C)
    recon = decoder(z)       # z: (B, latent_dim)
"""

from __future__ import annotations

from typing import Tuple, Optional, Callable
from dataclasses import dataclass

try:
    import jax
    import jax.numpy as jnp
    from flax import linen as nn
    FLAX_AVAILABLE = True
except ImportError:
    FLAX_AVAILABLE = False
    nn = None
    jnp = None


# =============================================================================
# Encoder Architecture Configs
# =============================================================================

@dataclass
class EEGEncoderConfig:
    """Configuration for EEG encoder architecture."""

    # Latent dimension
    latent_dim: int = 32

    # Conv layers: (features, kernel_size, stride)
    conv_layers: Tuple[Tuple[int, int, int], ...] = (
        (32, 5, 1),   # First conv: 32 filters, kernel 5, stride 1
        (64, 5, 1),   # Second conv: 64 filters
        (128, 3, 1),  # Third conv: 128 filters
    )

    # Pool after each conv
    pool_size: int = 2
    pool_stride: int = 2

    # Dense layers after flatten
    dense_dims: Tuple[int, ...] = (256, 128)

    # Activation
    activation: str = "relu"

    # Dropout rate (0 = no dropout)
    dropout_rate: float = 0.1


@dataclass
class EEGDecoderConfig:
    """Configuration for EEG decoder architecture."""

    # Latent dimension (must match encoder)
    latent_dim: int = 32

    # Output shape (time, channels)
    output_shape: Tuple[int, int] = (256, 32)

    # Dense layers before reshape
    dense_dims: Tuple[int, ...] = (128, 256, 512)

    # Activation
    activation: str = "relu"


# =============================================================================
# EEG Encoder (Conv1D)
# =============================================================================

if FLAX_AVAILABLE:

    class EEGEncoder(nn.Module):
        """
        Conv1D encoder for EEG time series.

        Takes EEG input of shape (batch, time, channels) and outputs
        latent parameters (mu, logvar) for VAE reparameterization.

        Architecture:
            Input: (B, T, C)
            Conv1D + ReLU + MaxPool (repeat N times)
            Flatten: (B, T', C') → (B, T' * C')
            Dense layers
            Output: mu (B, latent_dim), logvar (B, latent_dim)
        """

        latent_dim: int = 32
        conv_features: Tuple[int, ...] = (32, 64, 128)
        kernel_sizes: Tuple[int, ...] = (5, 5, 3)
        pool_size: int = 2
        dense_dims: Tuple[int, ...] = (256, 128)
        dropout_rate: float = 0.1
        use_batch_norm: bool = False

        @nn.compact
        def __call__(
            self,
            x: jnp.ndarray,
            training: bool = True,
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """
            Encode EEG to latent parameters.

            Args:
                x: EEG input of shape (batch, time, channels)
                training: Whether in training mode (affects dropout)

            Returns:
                mu: Latent mean (batch, latent_dim)
                logvar: Latent log-variance (batch, latent_dim)
            """
            # === Temporal convolution layers ===
            for i, (features, kernel_size) in enumerate(
                zip(self.conv_features, self.kernel_sizes)
            ):
                # Conv1D over time dimension
                x = nn.Conv(
                    features=features,
                    kernel_size=(kernel_size,),
                    strides=(1,),
                    padding='SAME',
                    name=f'conv_{i}',
                )(x)

                # Optional batch norm
                if self.use_batch_norm:
                    x = nn.BatchNorm(use_running_average=not training)(x)

                # Activation
                x = nn.relu(x)

                # Max pooling over time
                x = nn.max_pool(
                    x,
                    window_shape=(self.pool_size,),
                    strides=(self.pool_size,),
                    padding='SAME',
                )

                # Dropout
                if self.dropout_rate > 0 and training:
                    x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

            # === Flatten temporal × feature dimensions ===
            batch_size = x.shape[0]
            x = x.reshape((batch_size, -1))  # (B, T' * features)

            # === Dense layers ===
            for i, dim in enumerate(self.dense_dims):
                x = nn.Dense(features=dim, name=f'dense_{i}')(x)
                x = nn.relu(x)
                if self.dropout_rate > 0 and training:
                    x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

            # === Latent projections ===
            mu = nn.Dense(features=self.latent_dim, name='latent_mu')(x)
            logvar = nn.Dense(features=self.latent_dim, name='latent_logvar')(x)

            return mu, logvar

        def encode(
            self,
            x: jnp.ndarray,
            rng: jnp.ndarray,
            training: bool = False,
        ) -> jnp.ndarray:
            """
            Encode to latent representation with reparameterization.

            Args:
                x: Input EEG
                rng: Random key for sampling
                training: Training mode flag

            Returns:
                z: Sampled latent vector
            """
            mu, logvar = self(x, training=training)
            std = jnp.exp(0.5 * logvar)
            eps = jax.random.normal(rng, std.shape)
            z = mu + eps * std
            return z


    class EEGDecoder(nn.Module):
        """
        Decoder for EEG reconstruction.

        Takes latent vector z and reconstructs EEG signal.
        Uses dense layers followed by reshape to (time, channels).

        For simplicity, uses MLP decoder. For better reconstruction,
        consider transposed convolutions (ConvTranspose).
        """

        latent_dim: int = 32
        output_shape: Tuple[int, int] = (256, 32)  # (time, channels)
        dense_dims: Tuple[int, ...] = (128, 256, 512)
        use_sigmoid: bool = True  # If input is normalized to [0, 1]

        @nn.compact
        def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
            """
            Decode latent to EEG reconstruction.

            Args:
                z: Latent vector of shape (batch, latent_dim)

            Returns:
                recon: Reconstructed EEG of shape (batch, time, channels)
            """
            time, channels = self.output_shape
            output_dim = time * channels

            x = z

            # Dense layers
            for i, dim in enumerate(self.dense_dims):
                x = nn.Dense(features=dim, name=f'dense_{i}')(x)
                x = nn.relu(x)

            # Final projection to output size
            x = nn.Dense(features=output_dim, name='output_projection')(x)

            # Activation for bounded output
            if self.use_sigmoid:
                x = nn.sigmoid(x)

            # Reshape to (batch, time, channels)
            batch_size = z.shape[0]
            x = x.reshape((batch_size, time, channels))

            return x


    class EEGDecoderConvTranspose(nn.Module):
        """
        Alternative decoder using transposed convolutions.

        Better for capturing temporal structure in reconstruction.
        """

        latent_dim: int = 32
        output_shape: Tuple[int, int] = (256, 32)
        hidden_time: int = 16  # Intermediate time dimension after dense
        hidden_features: int = 128
        conv_features: Tuple[int, ...] = (64, 32)
        kernel_sizes: Tuple[int, ...] = (5, 5)
        use_sigmoid: bool = True

        @nn.compact
        def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
            """Decode with transposed convolutions."""
            time, channels = self.output_shape
            batch_size = z.shape[0]

            # Project and reshape to (B, hidden_time, hidden_features)
            x = nn.Dense(features=self.hidden_time * self.hidden_features)(z)
            x = nn.relu(x)
            x = x.reshape((batch_size, self.hidden_time, self.hidden_features))

            # Transposed convolutions to upsample time
            for i, (features, kernel_size) in enumerate(
                zip(self.conv_features, self.kernel_sizes)
            ):
                # Upsample by factor of 2
                x = jax.image.resize(
                    x,
                    shape=(batch_size, x.shape[1] * 2, x.shape[2]),
                    method='bilinear',
                )
                x = nn.Conv(
                    features=features,
                    kernel_size=(kernel_size,),
                    padding='SAME',
                    name=f'conv_{i}',
                )(x)
                x = nn.relu(x)

            # Final resize to target time dimension
            if x.shape[1] != time:
                x = jax.image.resize(
                    x,
                    shape=(batch_size, time, x.shape[2]),
                    method='bilinear',
                )

            # Project to channel dimension
            x = nn.Conv(
                features=channels,
                kernel_size=(1,),
                padding='SAME',
                name='channel_projection',
            )(x)

            if self.use_sigmoid:
                x = nn.sigmoid(x)

            return x


    # =============================================================================
    # Specialized Encoders for Different EEG Paradigms
    # =============================================================================

    class SpatialTemporalEncoder(nn.Module):
        """
        Encoder that separates spatial (channel) and temporal processing.

        First applies spatial convolution across channels, then temporal.
        Better for capturing electrode-specific patterns.

        Input shape: (batch, time, channels)
        """

        latent_dim: int = 32
        spatial_features: int = 16  # Features per channel group
        temporal_features: Tuple[int, ...] = (32, 64)
        kernel_sizes: Tuple[int, ...] = (5, 5)
        pool_size: int = 2
        dense_dims: Tuple[int, ...] = (128,)

        @nn.compact
        def __call__(
            self,
            x: jnp.ndarray,
            training: bool = True,
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """
            Encode with spatial-then-temporal processing.

            Args:
                x: EEG input (batch, time, channels)

            Returns:
                mu, logvar: Latent parameters
            """
            batch_size, time_dim, n_channels = x.shape

            # === Spatial processing (1x1 conv across channels) ===
            # Treat as (batch * time, 1, channels) → conv → (batch * time, 1, spatial_features)
            x_flat = x.reshape((batch_size * time_dim, 1, n_channels))
            x_spatial = nn.Conv(
                features=self.spatial_features,
                kernel_size=(1,),
                name='spatial_conv',
            )(x_flat)
            x_spatial = nn.relu(x_spatial)
            x = x_spatial.reshape((batch_size, time_dim, self.spatial_features))

            # === Temporal processing ===
            for i, (features, kernel_size) in enumerate(
                zip(self.temporal_features, self.kernel_sizes)
            ):
                x = nn.Conv(
                    features=features,
                    kernel_size=(kernel_size,),
                    padding='SAME',
                    name=f'temporal_conv_{i}',
                )(x)
                x = nn.relu(x)
                x = nn.max_pool(
                    x,
                    window_shape=(self.pool_size,),
                    strides=(self.pool_size,),
                )

            # === Flatten and dense ===
            x = x.reshape((batch_size, -1))

            for i, dim in enumerate(self.dense_dims):
                x = nn.Dense(features=dim, name=f'dense_{i}')(x)
                x = nn.relu(x)

            # === Latent projection ===
            mu = nn.Dense(features=self.latent_dim, name='mu')(x)
            logvar = nn.Dense(features=self.latent_dim, name='logvar')(x)

            return mu, logvar


    class FrequencyAwareEncoder(nn.Module):
        """
        Encoder with multi-scale temporal convolutions.

        Uses different kernel sizes to capture different frequency bands:
        - Small kernels: High-frequency (gamma, beta)
        - Large kernels: Low-frequency (theta, alpha, delta)
        """

        latent_dim: int = 32
        # Multi-scale kernels: (features, kernel_size)
        scale_configs: Tuple[Tuple[int, int], ...] = (
            (16, 3),   # High-frequency (gamma-ish)
            (16, 7),   # Mid-frequency (beta/alpha)
            (16, 15),  # Low-frequency (theta)
            (16, 31),  # Very low (delta)
        )
        temporal_features: Tuple[int, ...] = (64, 128)
        dense_dims: Tuple[int, ...] = (128,)

        @nn.compact
        def __call__(
            self,
            x: jnp.ndarray,
            training: bool = True,
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """
            Multi-scale temporal encoding.

            Args:
                x: EEG input (batch, time, channels)

            Returns:
                mu, logvar: Latent parameters
            """
            # === Multi-scale convolutions ===
            scale_outputs = []
            for i, (features, kernel_size) in enumerate(self.scale_configs):
                branch = nn.Conv(
                    features=features,
                    kernel_size=(kernel_size,),
                    padding='SAME',
                    name=f'scale_{i}_k{kernel_size}',
                )(x)
                branch = nn.relu(branch)
                scale_outputs.append(branch)

            # Concatenate scales
            x = jnp.concatenate(scale_outputs, axis=-1)

            # Pool
            x = nn.max_pool(x, window_shape=(4,), strides=(4,))

            # === Further temporal processing ===
            for i, features in enumerate(self.temporal_features):
                x = nn.Conv(
                    features=features,
                    kernel_size=(5,),
                    padding='SAME',
                    name=f'temporal_{i}',
                )(x)
                x = nn.relu(x)
                x = nn.max_pool(x, window_shape=(2,), strides=(2,))

            # === Flatten and project ===
            batch_size = x.shape[0]
            x = x.reshape((batch_size, -1))

            for i, dim in enumerate(self.dense_dims):
                x = nn.Dense(features=dim, name=f'dense_{i}')(x)
                x = nn.relu(x)

            mu = nn.Dense(features=self.latent_dim, name='mu')(x)
            logvar = nn.Dense(features=self.latent_dim, name='logvar')(x)

            return mu, logvar


else:
    # Stub classes when Flax not available
    class EEGEncoder:
        def __init__(self, *args, **kwargs):
            raise ImportError("Flax required for EEGEncoder")

    class EEGDecoder:
        def __init__(self, *args, **kwargs):
            raise ImportError("Flax required for EEGDecoder")

    class EEGDecoderConvTranspose:
        def __init__(self, *args, **kwargs):
            raise ImportError("Flax required for EEGDecoderConvTranspose")

    class SpatialTemporalEncoder:
        def __init__(self, *args, **kwargs):
            raise ImportError("Flax required")

    class FrequencyAwareEncoder:
        def __init__(self, *args, **kwargs):
            raise ImportError("Flax required")


# =============================================================================
# Factory Functions
# =============================================================================

def create_encoder(
    encoder_type: str = "standard",
    latent_dim: int = 32,
    **kwargs,
) -> "EEGEncoder":
    """
    Create an EEG encoder of the specified type.

    Args:
        encoder_type: One of "standard", "spatial_temporal", "frequency_aware"
        latent_dim: Latent dimension
        **kwargs: Additional arguments passed to encoder

    Returns:
        Encoder module
    """
    if not FLAX_AVAILABLE:
        raise ImportError("Flax required for encoder creation")

    encoders = {
        "standard": EEGEncoder,
        "spatial_temporal": SpatialTemporalEncoder,
        "frequency_aware": FrequencyAwareEncoder,
    }

    if encoder_type not in encoders:
        raise ValueError(f"Unknown encoder type: {encoder_type}. "
                        f"Options: {list(encoders.keys())}")

    return encoders[encoder_type](latent_dim=latent_dim, **kwargs)


def create_decoder(
    decoder_type: str = "mlp",
    latent_dim: int = 32,
    output_shape: Tuple[int, int] = (256, 32),
    **kwargs,
) -> "EEGDecoder":
    """
    Create an EEG decoder of the specified type.

    Args:
        decoder_type: One of "mlp", "conv_transpose"
        latent_dim: Latent dimension
        output_shape: (time, channels) output shape
        **kwargs: Additional arguments

    Returns:
        Decoder module
    """
    if not FLAX_AVAILABLE:
        raise ImportError("Flax required for decoder creation")

    decoders = {
        "mlp": EEGDecoder,
        "conv_transpose": EEGDecoderConvTranspose,
    }

    if decoder_type not in decoders:
        raise ValueError(f"Unknown decoder type: {decoder_type}")

    return decoders[decoder_type](
        latent_dim=latent_dim,
        output_shape=output_shape,
        **kwargs,
    )


# =============================================================================
# Demo
# =============================================================================

def demo_encoder():
    """Demonstrate EEG encoder."""
    if not FLAX_AVAILABLE:
        print("Flax not available - skipping encoder demo")
        return

    import jax

    print("\n" + "=" * 70)
    print("EEG ENCODER DEMO")
    print("=" * 70)

    # Create encoder
    encoder = EEGEncoder(latent_dim=32)

    # Initialize with dummy input
    rng = jax.random.PRNGKey(0)
    batch_size, time_steps, channels = 4, 256, 32
    x = jax.random.normal(rng, (batch_size, time_steps, channels))

    # Initialize parameters
    params = encoder.init(rng, x)

    # Forward pass
    mu, logvar = encoder.apply(params, x)

    print(f"\nInput shape: {x.shape}")
    print(f"Output mu shape: {mu.shape}")
    print(f"Output logvar shape: {logvar.shape}")

    # Parameter count
    param_count = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"\nTotal parameters: {param_count:,}")

    # Test decoder
    print("\n" + "-" * 70)
    print("Testing decoder...")

    decoder = EEGDecoder(latent_dim=32, output_shape=(time_steps, channels))
    z = jax.random.normal(rng, (batch_size, 32))
    dec_params = decoder.init(rng, z)
    recon = decoder.apply(dec_params, z)

    print(f"Latent shape: {z.shape}")
    print(f"Reconstruction shape: {recon.shape}")

    dec_param_count = sum(p.size for p in jax.tree_util.tree_leaves(dec_params))
    print(f"Decoder parameters: {dec_param_count:,}")


if __name__ == "__main__":
    demo_encoder()
