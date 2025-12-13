"""
EEGAraBrain Model

β-VAE for EEG with telepathy head (cognitive overload detection).
Designed for 2D mesh parallelism (batch + model) and integration
with the NeuroBalance precision framework.

Architecture:
    EEG Input (B, T, C)
        ↓
    EEG Encoder (Conv1D)
        ↓
    Latent Space z ~ N(μ, σ²)
        ↓
    ┌───────────┬───────────┐
    │           │           │
    Decoder   Telepathy   Precision
    (Recon)    Head       Probe
                (D_low)    (HGF bridge)

The model learns disentangled representations of EEG dynamics,
with latent dimensions ideally capturing precision-related features.

Usage:
    from ara.neuro.arabrain.model import EEGAraBrain

    model = EEGAraBrain(
        latent_dim=32,
        time=256,
        channels=32,
        beta=4.0,
    )

    # Initialize
    params = model.init(rng, x_batch, rng_vae)

    # Forward pass with loss
    loss, outputs = model.apply(params, x_batch, rng, labels=y_batch)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

try:
    import jax
    import jax.numpy as jnp
    from flax import linen as nn
    from flax.training import train_state
    import optax
    FLAX_AVAILABLE = True
except ImportError:
    FLAX_AVAILABLE = False
    nn = None
    jnp = None
    train_state = None
    optax = None

from .encoder import (
    EEGEncoder,
    EEGDecoder,
    EEGDecoderConvTranspose,
    FrequencyAwareEncoder,
    SpatialTemporalEncoder,
)


# =============================================================================
# Model Configuration
# =============================================================================

@dataclass
class EEGAraBrainConfig:
    """Configuration for EEGAraBrain model."""

    # Architecture
    latent_dim: int = 32
    time: int = 256        # EEG window length (samples)
    channels: int = 32     # EEG channel count

    # Encoder type: "standard", "spatial_temporal", "frequency_aware"
    encoder_type: str = "standard"

    # Decoder type: "mlp", "conv_transpose"
    decoder_type: str = "mlp"

    # Loss weights
    beta: float = 4.0              # KL weight (β-VAE)
    telepathy_weight: float = 1.0  # Telepathy head weight

    # Encoder config
    conv_features: Tuple[int, ...] = (32, 64, 128)
    kernel_sizes: Tuple[int, ...] = (5, 5, 3)
    dense_dims: Tuple[int, ...] = (256, 128)
    dropout_rate: float = 0.1

    # Training
    learning_rate: float = 1e-3


# =============================================================================
# Telepathy Head (Cognitive Overload Detection)
# =============================================================================

if FLAX_AVAILABLE:

    class NeuroBalanceHead(nn.Module):
        """
        Telepathy head for cognitive overload detection.

        Predicts probability of cognitive overload from latent z.
        This is a proxy for D_high (sensory-dominated precision)
        in the NeuroBalance framework.

        Input: z (batch, latent_dim)
        Output: logits (batch, 1) for binary classification
        """

        hidden_dims: Tuple[int, ...] = (64, 32)
        dropout_rate: float = 0.1

        @nn.compact
        def __call__(
            self,
            z: jnp.ndarray,
            training: bool = True,
        ) -> jnp.ndarray:
            """
            Predict cognitive overload probability.

            Args:
                z: Latent representation (batch, latent_dim)
                training: Training mode flag

            Returns:
                logits: (batch, 1) overload logits
            """
            x = z

            for i, dim in enumerate(self.hidden_dims):
                x = nn.Dense(features=dim, name=f'dense_{i}')(x)
                x = nn.relu(x)
                if self.dropout_rate > 0 and training:
                    x = nn.Dropout(
                        rate=self.dropout_rate,
                        deterministic=not training,
                    )(x)

            logits = nn.Dense(features=1, name='output')(x)
            return logits


    class PrecisionProbeHead(nn.Module):
        """
        Head for probing precision-related features from latent space.

        Attempts to decode D_low, D_high, and delta_H from z.
        Trained with supervision from HGF-derived or EEG-derived targets.
        """

        hidden_dim: int = 64

        @nn.compact
        def __call__(self, z: jnp.ndarray) -> Dict[str, jnp.ndarray]:
            """
            Probe precision metrics from latent.

            Args:
                z: Latent (batch, latent_dim)

            Returns:
                Dict with 'D_low', 'D_high', 'delta_H' predictions
            """
            x = nn.Dense(features=self.hidden_dim)(z)
            x = nn.relu(x)

            # D metrics are positive, use softplus
            D_low = nn.Dense(features=1, name='D_low')(x)
            D_low = nn.softplus(D_low)

            D_high = nn.Dense(features=1, name='D_high')(x)
            D_high = nn.softplus(D_high)

            delta_H = nn.Dense(features=1, name='delta_H')(x)
            delta_H = nn.softplus(delta_H)

            return {
                'D_low': D_low.squeeze(-1),
                'D_high': D_high.squeeze(-1),
                'delta_H': delta_H.squeeze(-1),
            }


    # =============================================================================
    # EEG VAE
    # =============================================================================

    class EEGVAE(nn.Module):
        """
        β-VAE for EEG time series.

        Encoder-decoder architecture with KL-regularized latent space.
        Uses β > 1 for disentangled representations.
        """

        latent_dim: int = 32
        time: int = 256
        channels: int = 32
        beta: float = 4.0

        # Encoder config
        encoder_type: str = "standard"
        conv_features: Tuple[int, ...] = (32, 64, 128)
        kernel_sizes: Tuple[int, ...] = (5, 5, 3)
        dense_dims: Tuple[int, ...] = (256, 128)
        dropout_rate: float = 0.1

        # Decoder config
        decoder_type: str = "mlp"

        def setup(self):
            """Initialize encoder and decoder."""
            # Select encoder type
            if self.encoder_type == "standard":
                self.encoder = EEGEncoder(
                    latent_dim=self.latent_dim,
                    conv_features=self.conv_features,
                    kernel_sizes=self.kernel_sizes,
                    dense_dims=self.dense_dims,
                    dropout_rate=self.dropout_rate,
                )
            elif self.encoder_type == "spatial_temporal":
                self.encoder = SpatialTemporalEncoder(
                    latent_dim=self.latent_dim,
                )
            elif self.encoder_type == "frequency_aware":
                self.encoder = FrequencyAwareEncoder(
                    latent_dim=self.latent_dim,
                )
            else:
                raise ValueError(f"Unknown encoder type: {self.encoder_type}")

            # Select decoder type
            if self.decoder_type == "mlp":
                self.decoder = EEGDecoder(
                    latent_dim=self.latent_dim,
                    output_shape=(self.time, self.channels),
                )
            elif self.decoder_type == "conv_transpose":
                self.decoder = EEGDecoderConvTranspose(
                    latent_dim=self.latent_dim,
                    output_shape=(self.time, self.channels),
                )
            else:
                raise ValueError(f"Unknown decoder type: {self.decoder_type}")

        def __call__(
            self,
            x: jnp.ndarray,
            rng: jnp.ndarray,
            training: bool = True,
        ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            """
            Forward pass through VAE.

            Args:
                x: EEG input (batch, time, channels)
                rng: Random key for sampling
                training: Training mode flag

            Returns:
                recon: Reconstructed EEG
                mu: Latent mean
                logvar: Latent log-variance
                z: Sampled latent
            """
            # Encode
            mu, logvar = self.encoder(x, training=training)

            # Reparameterization
            std = jnp.exp(0.5 * logvar)
            eps = jax.random.normal(rng, std.shape)
            z = mu + eps * std

            # Decode
            recon = self.decoder(z)

            return recon, mu, logvar, z

        def loss(
            self,
            x: jnp.ndarray,
            rng: jnp.ndarray,
            training: bool = True,
        ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray], jnp.ndarray]:
            """
            Compute VAE loss.

            Args:
                x: EEG input (batch, time, channels)
                rng: Random key
                training: Training mode

            Returns:
                total_loss: Weighted sum of reconstruction and KL
                metrics: Dict with individual losses
                z: Sampled latent (for downstream heads)
            """
            recon, mu, logvar, z = self(x, rng, training=training)

            # Reconstruction loss (MSE for continuous EEG)
            # Alternative: BCE if input is normalized to [0, 1]
            recon_loss = jnp.mean(jnp.sum((x - recon) ** 2, axis=(1, 2)))

            # KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
            kl = -0.5 * jnp.sum(1 + logvar - jnp.square(mu) - jnp.exp(logvar), axis=1)
            kl_loss = jnp.mean(kl)

            # Total loss
            total_loss = recon_loss + self.beta * kl_loss

            metrics = {
                "recon_loss": recon_loss,
                "kl_loss": kl_loss,
                "vae_loss": total_loss,
            }

            return total_loss, metrics, z

        def encode(
            self,
            x: jnp.ndarray,
            rng: jnp.ndarray,
            training: bool = False,
        ) -> jnp.ndarray:
            """Encode to sampled latent z."""
            mu, logvar = self.encoder(x, training=training)
            std = jnp.exp(0.5 * logvar)
            eps = jax.random.normal(rng, std.shape)
            return mu + eps * std


    # =============================================================================
    # Full EEGAraBrain Model
    # =============================================================================

    class EEGAraBrain(nn.Module):
        """
        Full EEGAraBrain model: VAE + Telepathy head.

        Combines:
        - β-VAE for disentangled EEG representations
        - Telepathy head for cognitive overload detection
        - Optional precision probe for D metric estimation

        This is the main training target. Latent space z can be
        used for downstream MIG/DCI/EDI disentanglement metrics.
        """

        # VAE config
        latent_dim: int = 32
        time: int = 256
        channels: int = 32
        beta: float = 4.0

        # Architecture config
        encoder_type: str = "standard"
        decoder_type: str = "mlp"
        conv_features: Tuple[int, ...] = (32, 64, 128)
        kernel_sizes: Tuple[int, ...] = (5, 5, 3)
        dense_dims: Tuple[int, ...] = (256, 128)
        dropout_rate: float = 0.1

        # Loss weights
        telepathy_weight: float = 1.0
        precision_weight: float = 0.0  # Set > 0 to enable precision probe

        def setup(self):
            """Initialize sub-modules."""
            self.vae = EEGVAE(
                latent_dim=self.latent_dim,
                time=self.time,
                channels=self.channels,
                beta=self.beta,
                encoder_type=self.encoder_type,
                decoder_type=self.decoder_type,
                conv_features=self.conv_features,
                kernel_sizes=self.kernel_sizes,
                dense_dims=self.dense_dims,
                dropout_rate=self.dropout_rate,
            )

            self.telepathy = NeuroBalanceHead(
                dropout_rate=self.dropout_rate,
            )

            if self.precision_weight > 0:
                self.precision_probe = PrecisionProbeHead()

        def __call__(
            self,
            x: jnp.ndarray,
            rng: jnp.ndarray,
            labels: Optional[jnp.ndarray] = None,
            precision_targets: Optional[Dict[str, jnp.ndarray]] = None,
            training: bool = True,
        ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
            """
            Forward pass with loss computation.

            Args:
                x: EEG input (batch, time, channels)
                rng: Random key
                labels: Optional binary labels for telepathy (batch,)
                precision_targets: Optional dict with D_low, D_high, delta_H
                training: Training mode

            Returns:
                total_loss: Combined loss
                outputs: Dict with all metrics and intermediate values
            """
            # VAE forward
            vae_loss, vae_metrics, z = self.vae.loss(x, rng, training=training)

            outputs = {
                "z": z,
                **vae_metrics,
            }

            # Telepathy head
            telepathy_loss = jnp.array(0.0)
            if labels is not None:
                logits = self.telepathy(z, training=training)
                labels_reshaped = labels.reshape((-1, 1))
                probs = jax.nn.sigmoid(logits)

                # Binary cross-entropy
                bce = -(
                    labels_reshaped * jnp.log(probs + 1e-8) +
                    (1 - labels_reshaped) * jnp.log(1 - probs + 1e-8)
                )
                telepathy_loss = jnp.mean(bce)

                outputs["telepathy_loss"] = telepathy_loss
                outputs["telepathy_logits"] = logits
                outputs["overload_prob_mean"] = jnp.mean(probs)

                # Accuracy
                preds = (probs > 0.5).astype(jnp.float32)
                accuracy = jnp.mean(preds == labels_reshaped)
                outputs["telepathy_accuracy"] = accuracy

            # Precision probe (optional)
            precision_loss = jnp.array(0.0)
            if self.precision_weight > 0 and precision_targets is not None:
                precision_preds = self.precision_probe(z)

                # MSE on each metric
                for key in ['D_low', 'D_high', 'delta_H']:
                    if key in precision_targets:
                        target = precision_targets[key]
                        pred = precision_preds[key]
                        precision_loss += jnp.mean((pred - target) ** 2)

                outputs["precision_loss"] = precision_loss
                outputs["precision_preds"] = precision_preds

            # Total loss
            total_loss = (
                vae_loss +
                self.telepathy_weight * telepathy_loss +
                self.precision_weight * precision_loss
            )
            outputs["total_loss"] = total_loss

            return total_loss, outputs

        def encode(
            self,
            x: jnp.ndarray,
            rng: jnp.ndarray,
            training: bool = False,
        ) -> jnp.ndarray:
            """Encode EEG to latent z."""
            return self.vae.encode(x, rng, training=training)

        def reconstruct(
            self,
            x: jnp.ndarray,
            rng: jnp.ndarray,
            training: bool = False,
        ) -> jnp.ndarray:
            """Reconstruct EEG through VAE."""
            recon, _, _, _ = self.vae(x, rng, training=training)
            return recon

        def predict_overload(
            self,
            x: jnp.ndarray,
            rng: jnp.ndarray,
            training: bool = False,
        ) -> jnp.ndarray:
            """Predict cognitive overload probability."""
            z = self.encode(x, rng, training=training)
            logits = self.telepathy(z, training=training)
            return jax.nn.sigmoid(logits)


    # =============================================================================
    # Train State Factory
    # =============================================================================

    class EEGAraBrainTrainState(train_state.TrainState):
        """Extended train state with batch stats for batch norm."""
        batch_stats: Optional[Any] = None


    def create_train_state(
        rng: jnp.ndarray,
        model: EEGAraBrain,
        learning_rate: float,
        input_shape: Tuple[int, ...] = (1, 256, 32),
    ) -> EEGAraBrainTrainState:
        """
        Create initialized train state.

        Args:
            rng: Random key
            model: EEGAraBrain model
            learning_rate: Learning rate
            input_shape: (batch, time, channels) for initialization

        Returns:
            Initialized train state
        """
        rng, init_rng, sample_rng, dropout_rng = jax.random.split(rng, 4)

        # Dummy input for initialization
        x = jax.random.normal(init_rng, input_shape)

        # Dummy labels to ensure telepathy head is initialized
        dummy_labels = jnp.zeros(input_shape[0])

        # Initialize parameters (need dropout RNG for training mode)
        # Pass labels to ensure telepathy head params are created
        variables = model.init(
            {'params': rng, 'dropout': dropout_rng},
            x, sample_rng, labels=dummy_labels, training=True
        )
        params = variables.get('params', variables)
        batch_stats = variables.get('batch_stats', None)

        # Create optimizer
        tx = optax.adam(learning_rate)

        return EEGAraBrainTrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=tx,
            batch_stats=batch_stats,
        )


else:
    # Stub classes when Flax not available
    class NeuroBalanceHead:
        def __init__(self, *args, **kwargs):
            raise ImportError("Flax required")

    class PrecisionProbeHead:
        def __init__(self, *args, **kwargs):
            raise ImportError("Flax required")

    class EEGVAE:
        def __init__(self, *args, **kwargs):
            raise ImportError("Flax required")

    class EEGAraBrain:
        def __init__(self, *args, **kwargs):
            raise ImportError("Flax required")

    class EEGAraBrainTrainState:
        pass

    def create_train_state(*args, **kwargs):
        raise ImportError("Flax required")


# =============================================================================
# Demo
# =============================================================================

def demo_model():
    """Demonstrate EEGAraBrain model."""
    if not FLAX_AVAILABLE:
        print("Flax not available - skipping model demo")
        return

    print("\n" + "=" * 70)
    print("EEGAraBrain MODEL DEMO")
    print("=" * 70)

    # Config
    batch_size = 4
    time_steps = 256
    channels = 32
    latent_dim = 32

    # Create model
    model = EEGAraBrain(
        latent_dim=latent_dim,
        time=time_steps,
        channels=channels,
        beta=4.0,
        telepathy_weight=1.0,
    )

    # Initialize
    rng = jax.random.PRNGKey(42)
    rng, init_rng, sample_rng, dropout_rng = jax.random.split(rng, 4)

    x = jax.random.normal(init_rng, (batch_size, time_steps, channels))
    labels = jax.random.bernoulli(init_rng, 0.3, (batch_size,)).astype(jnp.float32)

    # Initialize with dropout RNG for training mode
    params = model.init(
        {'params': rng, 'dropout': dropout_rng},
        x, sample_rng, labels=labels, training=True
    )

    # Forward pass (training=False to skip dropout)
    loss, outputs = model.apply(params, x, sample_rng, labels=labels, training=False)

    print(f"\nInput shape: {x.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"\nOutputs:")
    for key, val in outputs.items():
        if hasattr(val, 'shape'):
            print(f"  {key}: shape={val.shape}, mean={float(jnp.mean(val)):.4f}")
        else:
            print(f"  {key}: {float(val):.4f}")

    # Parameter count
    param_count = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"\nTotal parameters: {param_count:,}")

    # Test encode (training=False)
    z = model.apply(params, x, sample_rng, training=False, method=model.encode)
    print(f"\nLatent z shape: {z.shape}")

    # Test predict_overload (training=False)
    probs = model.apply(params, x, sample_rng, training=False, method=model.predict_overload)
    print(f"Overload probs shape: {probs.shape}")
    print(f"Overload probs mean: {float(jnp.mean(probs)):.4f}")


if __name__ == "__main__":
    demo_model()
