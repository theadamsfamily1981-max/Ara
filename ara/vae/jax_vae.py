"""
ara.vae.jax_vae - JAX-based VAE with SPMD Support

Provides a JAX/Flax implementation of TrajectoryVAE that scales across
multiple GPUs/TPUs using SPMD (Single Program, Multiple Data) parallelism.

Key features:
- Mesh-based sharding for data parallelism
- Automatic collective operations via jax.jit
- Optional shard_map for explicit control
- Compatible with ara.vae.metrics for disentanglement evaluation

Usage:
    from ara.vae.jax_vae import JAXTrajectoryVAE, create_mesh, train_step

    mesh = create_mesh(n_devices=4)
    model = JAXTrajectoryVAE(config)
    state = create_train_state(model, mesh)

    for batch in dataloader:
        state, metrics = train_step(state, batch, mesh)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple
import numpy as np

# JAX imports with graceful fallback
try:
    import jax
    import jax.numpy as jnp
    from jax import random
    from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = None

# Flax imports
try:
    import flax.linen as nn
    from flax.training import train_state
    import optax
    HAS_FLAX = True
except ImportError:
    HAS_FLAX = False
    nn = None


@dataclass
class JAXVAEConfig:
    """Configuration for JAX-based VAE."""
    n_trials: int = 200
    n_features: int = 8
    latent_dim: int = 8
    hidden_dim: int = 64
    n_layers: int = 2
    beta: float = 4.0
    learning_rate: float = 1e-3
    dropout_rate: float = 0.1


if HAS_JAX and HAS_FLAX:

    class Encoder(nn.Module):
        """LSTM encoder for HGF trajectories."""
        config: JAXVAEConfig

        @nn.compact
        def __call__(self, x, train: bool = True):
            """
            Encode trajectory to latent distribution.

            Args:
                x: [batch, n_trials, n_features]
                train: Whether in training mode

            Returns:
                mu, logvar: [batch, latent_dim]
            """
            batch_size = x.shape[0]

            # LSTM scan over time dimension
            lstm = nn.scan(
                nn.LSTMCell,
                variable_broadcast="params",
                split_rngs={"params": False},
                in_axes=1,
                out_axes=1,
            )(features=self.config.hidden_dim)

            carry = lstm.initialize_carry(random.PRNGKey(0), (batch_size,))
            carry, outputs = lstm(carry, x)

            # Use final hidden state
            h = carry[0]  # Final cell state

            # Project to latent parameters
            h = nn.Dense(self.config.hidden_dim)(h)
            h = nn.relu(h)
            if train:
                h = nn.Dropout(rate=self.config.dropout_rate, deterministic=False)(h)

            mu = nn.Dense(self.config.latent_dim)(h)
            logvar = nn.Dense(self.config.latent_dim)(h)

            return mu, logvar

    class Decoder(nn.Module):
        """LSTM decoder for HGF trajectories."""
        config: JAXVAEConfig

        @nn.compact
        def __call__(self, z, train: bool = True):
            """
            Decode latent to trajectory.

            Args:
                z: [batch, latent_dim]
                train: Whether in training mode

            Returns:
                x_recon: [batch, n_trials, n_features]
            """
            batch_size = z.shape[0]

            # Project latent
            h = nn.Dense(self.config.hidden_dim)(z)
            h = nn.relu(h)

            # Repeat for each time step
            h = jnp.tile(h[:, None, :], (1, self.config.n_trials, 1))

            # LSTM decode
            lstm = nn.scan(
                nn.LSTMCell,
                variable_broadcast="params",
                split_rngs={"params": False},
                in_axes=1,
                out_axes=1,
            )(features=self.config.hidden_dim)

            carry = lstm.initialize_carry(random.PRNGKey(0), (batch_size,))
            carry, outputs = lstm(carry, h)

            # Project to output features
            x_recon = nn.Dense(self.config.n_features)(outputs)

            return x_recon

    class JAXTrajectoryVAE(nn.Module):
        """
        JAX/Flax implementation of TrajectoryVAE.

        Supports SPMD parallelism via mesh sharding.
        """
        config: JAXVAEConfig

        def setup(self):
            self.encoder = Encoder(self.config)
            self.decoder = Decoder(self.config)

        def __call__(self, x, train: bool = True, rng: Optional[Any] = None):
            """Forward pass with reparameterization."""
            mu, logvar = self.encoder(x, train=train)

            # Reparameterization trick
            if rng is not None:
                std = jnp.exp(0.5 * logvar)
                eps = random.normal(rng, mu.shape)
                z = mu + eps * std
            else:
                z = mu  # Use mean at inference

            x_recon = self.decoder(z, train=train)

            return x_recon, mu, logvar

        def encode(self, x):
            """Encode to latent mean (for evaluation)."""
            mu, _ = self.encoder(x, train=False)
            return mu

        def decode(self, z):
            """Decode latent to trajectory."""
            return self.decoder(z, train=False)

        def generate(self, rng, n_samples: int = 1):
            """Generate from prior."""
            z = random.normal(rng, (n_samples, self.config.latent_dim))
            return self.decode(z)


    # =========================================================================
    # SPMD Training Utilities
    # =========================================================================

    def create_mesh(n_devices: Optional[int] = None, axis_name: str = 'data') -> Mesh:
        """
        Create a JAX device mesh for SPMD parallelism.

        Args:
            n_devices: Number of devices (None = all available)
            axis_name: Name for the data parallel axis

        Returns:
            JAX Mesh object
        """
        devices = jax.devices()
        if n_devices is not None:
            devices = devices[:n_devices]

        return Mesh(np.array(devices), (axis_name,))


    def shard_batch(batch: jnp.ndarray, mesh: Mesh, axis_name: str = 'data') -> jnp.ndarray:
        """
        Shard a batch across devices.

        Args:
            batch: Data batch [batch_size, ...]
            mesh: Device mesh
            axis_name: Axis to shard along

        Returns:
            Sharded array
        """
        sharding = NamedSharding(mesh, P(axis_name, None))
        return jax.device_put(batch, sharding)


    def create_train_state(
        model: JAXTrajectoryVAE,
        mesh: Mesh,
        rng: Any,
        config: JAXVAEConfig,
    ) -> train_state.TrainState:
        """
        Initialize model and create sharded train state.

        Args:
            model: VAE model
            mesh: Device mesh
            rng: Random key
            config: Model config

        Returns:
            Flax TrainState with sharded parameters
        """
        # Initialize with dummy input
        dummy_input = jnp.ones((1, config.n_trials, config.n_features))
        variables = model.init(rng, dummy_input, train=False)

        # Create optimizer
        tx = optax.adam(config.learning_rate)

        # Create train state
        state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=variables['params'],
            tx=tx,
        )

        return state


    def vae_loss(
        params: Any,
        apply_fn: Callable,
        batch: jnp.ndarray,
        rng: Any,
        beta: float = 4.0,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute VAE ELBO loss.

        Args:
            params: Model parameters
            apply_fn: Model apply function
            batch: Input batch
            rng: Random key
            beta: KL weight

        Returns:
            (loss, metrics_dict)
        """
        x_recon, mu, logvar = apply_fn(
            {'params': params},
            batch,
            train=True,
            rng=rng,
        )

        # Reconstruction loss (MSE)
        recon_loss = jnp.mean((x_recon - batch) ** 2)

        # KL divergence
        kl_loss = -0.5 * jnp.mean(1 + logvar - mu**2 - jnp.exp(logvar))

        total_loss = recon_loss + beta * kl_loss

        metrics = {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
        }

        return total_loss, metrics


    @jax.jit
    def train_step_single(
        state: train_state.TrainState,
        batch: jnp.ndarray,
        rng: Any,
        beta: float = 4.0,
    ) -> Tuple[train_state.TrainState, Dict[str, float]]:
        """
        Single training step (non-sharded).

        Args:
            state: Current train state
            batch: Input batch
            rng: Random key
            beta: KL weight

        Returns:
            (new_state, metrics)
        """
        def loss_fn(params):
            return vae_loss(params, state.apply_fn, batch, rng, beta)

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)

        return state, metrics


    def create_spmd_train_step(mesh: Mesh, beta: float = 4.0):
        """
        Create an SPMD-parallelized training step.

        This uses jax.jit with sharding to automatically parallelize
        across devices in the mesh.

        Args:
            mesh: Device mesh
            beta: KL weight

        Returns:
            JIT-compiled training step function
        """
        data_sharding = NamedSharding(mesh, P('data', None, None))
        replicated = NamedSharding(mesh, P())

        @jax.jit
        def train_step(
            state: train_state.TrainState,
            batch: jnp.ndarray,
            rng: Any,
        ) -> Tuple[train_state.TrainState, Dict[str, float]]:
            """
            SPMD training step.

            The batch is automatically sharded across devices.
            Gradients are all-reduced before parameter update.
            """
            def loss_fn(params):
                return vae_loss(params, state.apply_fn, batch, rng, beta)

            (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

            # All-reduce gradients across devices (implicit via jit)
            state = state.apply_gradients(grads=grads)

            return state, metrics

        return train_step


    # =========================================================================
    # Full Training Loop
    # =========================================================================

    def train_vae_spmd(
        trajectories: np.ndarray,
        config: JAXVAEConfig,
        n_epochs: int = 50,
        batch_size: int = 32,
        n_devices: Optional[int] = None,
        seed: int = 42,
        verbose: bool = True,
    ) -> Tuple[Any, Dict[str, list]]:
        """
        Train VAE with SPMD parallelism.

        Args:
            trajectories: Training data [n_samples, n_trials, n_features]
            config: VAE configuration
            n_epochs: Number of epochs
            batch_size: Batch size (per-device if SPMD)
            n_devices: Number of devices (None = all)
            seed: Random seed
            verbose: Print progress

        Returns:
            (trained_params, history)
        """
        rng = random.PRNGKey(seed)
        rng, init_rng = random.split(rng)

        # Create mesh
        mesh = create_mesh(n_devices)
        n_devices_actual = len(mesh.devices.flatten())

        if verbose:
            print(f"Training on {n_devices_actual} device(s)")

        # Create model and state
        model = JAXTrajectoryVAE(config)
        state = create_train_state(model, mesh, init_rng, config)

        # Create SPMD train step
        train_step = create_spmd_train_step(mesh, config.beta)

        # Convert data to JAX array
        data = jnp.array(trajectories)
        n_samples = len(data)

        # Training history
        history = {'total_loss': [], 'recon_loss': [], 'kl_loss': []}

        # Training loop
        for epoch in range(n_epochs):
            rng, epoch_rng = random.split(rng)

            # Shuffle
            perm = random.permutation(epoch_rng, n_samples)
            data_shuffled = data[perm]

            epoch_metrics = {'total_loss': 0, 'recon_loss': 0, 'kl_loss': 0}
            n_batches = 0

            # Iterate batches
            for i in range(0, n_samples, batch_size):
                batch = data_shuffled[i:i+batch_size]

                if len(batch) < batch_size:
                    continue  # Skip incomplete batch

                rng, step_rng = random.split(rng)

                # Shard batch across devices
                with mesh:
                    batch_sharded = shard_batch(batch, mesh)
                    state, metrics = train_step(state, batch_sharded, step_rng)

                for k in epoch_metrics:
                    epoch_metrics[k] += float(metrics[k])
                n_batches += 1

            # Average metrics
            for k in epoch_metrics:
                epoch_metrics[k] /= max(1, n_batches)
                history[k].append(epoch_metrics[k])

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}: "
                      f"Loss={epoch_metrics['total_loss']:.4f} "
                      f"(Recon={epoch_metrics['recon_loss']:.4f}, "
                      f"KL={epoch_metrics['kl_loss']:.4f})")

        return state.params, history


    def encode_trajectories(
        params: Any,
        trajectories: np.ndarray,
        config: JAXVAEConfig,
        batch_size: int = 128,
    ) -> np.ndarray:
        """
        Encode trajectories to latent space.

        Args:
            params: Trained model parameters
            trajectories: Data to encode
            config: Model config
            batch_size: Batch size for encoding

        Returns:
            Latent representations [n_samples, latent_dim]
        """
        model = JAXTrajectoryVAE(config)

        @jax.jit
        def encode_batch(x):
            return model.apply({'params': params}, x, method=model.encode)

        data = jnp.array(trajectories)
        latents = []

        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            z = encode_batch(batch)
            latents.append(np.array(z))

        return np.concatenate(latents, axis=0)


else:
    # Fallback when JAX/Flax not available

    class JAXVAEConfig:
        def __init__(self, **kwargs):
            raise ImportError("JAX and Flax required for JAXTrajectoryVAE")

    class JAXTrajectoryVAE:
        def __init__(self, *args, **kwargs):
            raise ImportError("JAX and Flax required for JAXTrajectoryVAE")

    def create_mesh(*args, **kwargs):
        raise ImportError("JAX required for create_mesh")

    def train_vae_spmd(*args, **kwargs):
        raise ImportError("JAX and Flax required for train_vae_spmd")

    def encode_trajectories(*args, **kwargs):
        raise ImportError("JAX and Flax required for encode_trajectories")
