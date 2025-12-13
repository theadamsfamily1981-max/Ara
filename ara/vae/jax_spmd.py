"""
ara.vae.jax_spmd - Pure JAX β-VAE with SPMD Parallelism

No Flax/Haiku dependency. Uses explicit device mesh sharding and
lax.psum for gradient averaging across devices.

Mental model:
- mesh = Mesh(devices, ('data',)) → logical device axis
- P('data', None) → shard batch dim, replicate others
- lax.psum(..., 'data') → all-reduce across mesh axis

This is the cleanest way to scale β-VAE training across GPUs/TPUs.

Usage:
    from ara.vae.jax_spmd import (
        init_beta_vae_params,
        create_spmd_trainer,
        train_epoch,
        encode_to_latent,
    )

    # Setup
    mesh, train_step = create_spmd_trainer(
        latent_dim=10,
        beta=4.0,
        input_shape=(64, 64),
    )

    # Train
    with mesh:
        for epoch in range(100):
            params, metrics = train_epoch(params, data, train_step, mesh)

    # Extract latents for disentanglement evaluation
    z, mu, logvar = encode_to_latent(params, data)
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple
import numpy as np

# JAX imports
try:
    import jax
    import jax.numpy as jnp
    from jax import random, value_and_grad, lax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = np


@dataclass
class SPMDConfig:
    """Configuration for SPMD β-VAE."""
    latent_dim: int = 10
    hidden_dim: int = 256
    beta: float = 4.0
    learning_rate: float = 1e-3
    input_shape: Tuple[int, ...] = (64, 64)  # Image shape or (n_trials, n_features)

    @property
    def input_dim(self) -> int:
        result = 1
        for d in self.input_shape:
            result *= d
        return result


if HAS_JAX:

    # =========================================================================
    # Pure-Function β-VAE
    # =========================================================================

    def init_beta_vae_params(
        key: Any,
        config: SPMDConfig,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Initialize β-VAE parameters as nested dict.

        Architecture:
        - Encoder: input → hidden → hidden/2 → (mu, logvar)
        - Decoder: latent → hidden/2 → hidden → output
        """
        input_dim = config.input_dim
        latent_dim = config.latent_dim
        hidden_dim = config.hidden_dim

        keys = random.split(key, 7)

        params = {
            # Encoder layers
            "enc_1": {
                "W": random.normal(keys[0], (input_dim, hidden_dim)) * 0.01,
                "b": jnp.zeros((hidden_dim,)),
            },
            "enc_2": {
                "W": random.normal(keys[1], (hidden_dim, hidden_dim // 2)) * 0.01,
                "b": jnp.zeros((hidden_dim // 2,)),
            },
            # Latent projections
            "mu": {
                "W": random.normal(keys[2], (hidden_dim // 2, latent_dim)) * 0.01,
                "b": jnp.zeros((latent_dim,)),
            },
            "logvar": {
                "W": random.normal(keys[3], (hidden_dim // 2, latent_dim)) * 0.01,
                "b": jnp.zeros((latent_dim,)),
            },
            # Decoder layers
            "dec_1": {
                "W": random.normal(keys[4], (latent_dim, hidden_dim // 2)) * 0.01,
                "b": jnp.zeros((hidden_dim // 2,)),
            },
            "dec_2": {
                "W": random.normal(keys[5], (hidden_dim // 2, hidden_dim)) * 0.01,
                "b": jnp.zeros((hidden_dim,)),
            },
            "dec_out": {
                "W": random.normal(keys[6], (hidden_dim, input_dim)) * 0.01,
                "b": jnp.zeros((input_dim,)),
            },
        }

        return params

    def encoder(params: Dict, x_flat: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Encode input to latent distribution parameters.

        Args:
            params: Model parameters
            x_flat: Flattened input [batch, input_dim]

        Returns:
            mu, logvar: [batch, latent_dim]
        """
        h1 = jnp.tanh(x_flat @ params["enc_1"]["W"] + params["enc_1"]["b"])
        h2 = jnp.tanh(h1 @ params["enc_2"]["W"] + params["enc_2"]["b"])
        mu = h2 @ params["mu"]["W"] + params["mu"]["b"]
        logvar = h2 @ params["logvar"]["W"] + params["logvar"]["b"]
        return mu, logvar

    def reparameterize(
        key: Any,
        mu: jnp.ndarray,
        logvar: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Reparameterization trick: z = μ + σ * ε

        Args:
            key: Random key
            mu: Mean [batch, latent_dim]
            logvar: Log variance [batch, latent_dim]

        Returns:
            z: Sampled latent [batch, latent_dim]
        """
        std = jnp.exp(0.5 * logvar)
        eps = random.normal(key, mu.shape)
        return mu + eps * std

    def decoder(params: Dict, z: jnp.ndarray) -> jnp.ndarray:
        """
        Decode latent to reconstruction.

        Args:
            params: Model parameters
            z: Latent codes [batch, latent_dim]

        Returns:
            recon: Reconstruction [batch, input_dim] in (0, 1)
        """
        h1 = jnp.tanh(z @ params["dec_1"]["W"] + params["dec_1"]["b"])
        h2 = jnp.tanh(h1 @ params["dec_2"]["W"] + params["dec_2"]["b"])
        logits = h2 @ params["dec_out"]["W"] + params["dec_out"]["b"]
        recon = jax.nn.sigmoid(logits)
        return recon

    def beta_vae_forward(
        params: Dict,
        key: Any,
        x: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Full forward pass.

        Args:
            params: Model parameters
            key: Random key
            x: Input batch [batch, *input_shape]

        Returns:
            recon_flat, mu, logvar, x_flat
        """
        x_flat = x.reshape(x.shape[0], -1)
        mu, logvar = encoder(params, x_flat)
        key_z, _ = random.split(key)
        z = reparameterize(key_z, mu, logvar)
        recon_flat = decoder(params, z)
        return recon_flat, mu, logvar, x_flat

    def beta_vae_loss(
        params: Dict,
        key: Any,
        x: jnp.ndarray,
        beta: float,
    ) -> Tuple[float, Tuple[float, float]]:
        """
        Compute β-VAE ELBO loss.

        Loss = E[log p(x|z)] - β * KL(q(z|x) || p(z))

        Args:
            params: Model parameters
            key: Random key
            x: Input batch
            beta: KL weight (β > 1 for disentanglement)

        Returns:
            (total_loss, (recon_loss, kl_loss))
        """
        recon_flat, mu, logvar, x_flat = beta_vae_forward(params, key, x)

        # Bernoulli reconstruction (binary cross-entropy)
        bce = -jnp.sum(
            x_flat * jnp.log(recon_flat + 1e-8) +
            (1.0 - x_flat) * jnp.log(1.0 - recon_flat + 1e-8),
            axis=1,
        )
        recon_loss = jnp.mean(bce)

        # KL divergence: KL(N(μ,σ²) || N(0,1))
        kl = -0.5 * jnp.mean(
            jnp.sum(1 + logvar - mu**2 - jnp.exp(logvar), axis=1)
        )

        total_loss = recon_loss + beta * kl

        return total_loss, (recon_loss, kl)

    def beta_vae_loss_mse(
        params: Dict,
        key: Any,
        x: jnp.ndarray,
        beta: float,
    ) -> Tuple[float, Tuple[float, float]]:
        """
        β-VAE loss with MSE reconstruction (for continuous data like HGF trajectories).
        """
        recon_flat, mu, logvar, x_flat = beta_vae_forward(params, key, x)

        # MSE reconstruction
        recon_loss = jnp.mean(jnp.sum((recon_flat - x_flat) ** 2, axis=1))

        # KL divergence
        kl = -0.5 * jnp.mean(
            jnp.sum(1 + logvar - mu**2 - jnp.exp(logvar), axis=1)
        )

        total_loss = recon_loss + beta * kl

        return total_loss, (recon_loss, kl)

    # =========================================================================
    # SPMD Training Infrastructure
    # =========================================================================

    def create_mesh(axis_name: str = 'data') -> Tuple[Mesh, int]:
        """
        Create device mesh for SPMD parallelism.

        Returns:
            (mesh, n_devices)
        """
        devices = np.array(jax.devices())
        n_devices = devices.size
        mesh = Mesh(devices, (axis_name,))
        return mesh, n_devices

    def shard_batch(
        x: jnp.ndarray,
        mesh: Mesh,
        axis_name: str = 'data',
    ) -> jnp.ndarray:
        """
        Shard batch across devices.

        Args:
            x: Data [batch, ...]
            mesh: Device mesh
            axis_name: Mesh axis for batch sharding

        Returns:
            Sharded array
        """
        # Build PartitionSpec: shard first dim, replicate rest
        n_dims = len(x.shape)
        spec = P(axis_name, *([None] * (n_dims - 1)))
        sharding = NamedSharding(mesh, spec)
        return jax.device_put(x, sharding)

    def replicate_params(params: Dict, mesh: Mesh) -> Dict:
        """
        Replicate parameters across all devices.
        """
        rep_sharding = NamedSharding(mesh, P())
        return jax.tree_util.tree_map(
            lambda p: jax.device_put(p, rep_sharding),
            params,
        )

    def sgd_update(params: Dict, grads: Dict, lr: float) -> Dict:
        """Simple SGD parameter update."""
        return jax.tree_util.tree_map(
            lambda p, g: p - lr * g,
            params,
            grads,
        )

    def adam_update(
        params: Dict,
        grads: Dict,
        m: Dict,
        v: Dict,
        t: int,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Adam optimizer update.

        Returns:
            (new_params, new_m, new_v)
        """
        new_m = jax.tree_util.tree_map(
            lambda m_i, g: beta1 * m_i + (1 - beta1) * g,
            m, grads,
        )
        new_v = jax.tree_util.tree_map(
            lambda v_i, g: beta2 * v_i + (1 - beta2) * g**2,
            v, grads,
        )

        # Bias correction
        m_hat = jax.tree_util.tree_map(lambda m_i: m_i / (1 - beta1**t), new_m)
        v_hat = jax.tree_util.tree_map(lambda v_i: v_i / (1 - beta2**t), new_v)

        new_params = jax.tree_util.tree_map(
            lambda p, m_h, v_h: p - lr * m_h / (jnp.sqrt(v_h) + eps),
            params, m_hat, v_hat,
        )

        return new_params, new_m, new_v

    def create_spmd_train_step(
        config: SPMDConfig,
        n_devices: int,
        loss_fn: Callable = None,
    ) -> Callable:
        """
        Create JIT-compiled SPMD training step.

        The returned function:
        1. Computes loss and gradients on each device's shard
        2. All-reduces gradients via psum
        3. Updates parameters (replicated across devices)

        Args:
            config: Model configuration
            n_devices: Number of devices in mesh
            loss_fn: Loss function (default: beta_vae_loss)

        Returns:
            JIT-compiled train_step function
        """
        if loss_fn is None:
            loss_fn = beta_vae_loss

        beta = config.beta
        lr = config.learning_rate

        @partial(jax.jit, donate_argnums=(0,))
        def train_step(params, key, x):
            """
            Single SPMD training step.

            Args:
                params: Model params (replicated)
                key: Random key (replicated)
                x: Input batch (sharded on 'data')

            Returns:
                (new_params, metrics_dict)
            """
            def loss_for_grad(p, k, xb):
                loss, aux = loss_fn(p, k, xb, beta)
                return loss, aux

            # Compute loss and grads
            (loss_val, (recon_loss, kl)), grads = value_and_grad(
                loss_for_grad, has_aux=True
            )(params, key, x)

            # All-reduce across 'data' axis
            def avg_over_data(v):
                return lax.psum(v, 'data') / n_devices

            loss_val = avg_over_data(loss_val)
            recon_loss = avg_over_data(recon_loss)
            kl = avg_over_data(kl)
            grads = jax.tree_util.tree_map(avg_over_data, grads)

            # Update params
            new_params = sgd_update(params, grads, lr)

            metrics = {
                "loss": loss_val,
                "recon_loss": recon_loss,
                "kl": kl,
            }

            return new_params, metrics

        return train_step

    def create_spmd_trainer(
        config: SPMDConfig,
    ) -> Tuple[Mesh, Callable, int]:
        """
        Create complete SPMD training setup.

        Args:
            config: Model configuration

        Returns:
            (mesh, train_step, n_devices)
        """
        mesh, n_devices = create_mesh()
        train_step = create_spmd_train_step(config, n_devices)
        return mesh, train_step, n_devices

    # =========================================================================
    # Training Loop
    # =========================================================================

    def train_beta_vae_spmd(
        data: np.ndarray,
        config: SPMDConfig,
        n_epochs: int = 100,
        batch_size: int = 128,
        seed: int = 42,
        verbose: bool = True,
        loss_type: str = "bce",
    ) -> Tuple[Dict, Dict[str, list]]:
        """
        Train β-VAE with SPMD parallelism.

        Args:
            data: Training data [n_samples, *input_shape]
            config: Model configuration
            n_epochs: Number of epochs
            batch_size: Global batch size (split across devices)
            seed: Random seed
            verbose: Print progress
            loss_type: "bce" for binary, "mse" for continuous

        Returns:
            (trained_params, history)
        """
        # Setup
        key = random.PRNGKey(seed)
        key, init_key = random.split(key)

        # Create mesh and train step
        mesh, n_devices = create_mesh()

        if verbose:
            print(f"Training β-VAE (β={config.beta}) on {n_devices} device(s)")
            print(f"  Input shape: {config.input_shape}")
            print(f"  Latent dim: {config.latent_dim}")
            print(f"  Hidden dim: {config.hidden_dim}")

        # Select loss function
        loss_fn = beta_vae_loss if loss_type == "bce" else beta_vae_loss_mse
        train_step = create_spmd_train_step(config, n_devices, loss_fn)

        # Initialize and replicate params
        params = init_beta_vae_params(init_key, config)
        params = replicate_params(params, mesh)

        # Convert data to JAX
        data_jax = jnp.array(data)
        n_samples = len(data_jax)

        # Ensure batch size is divisible by n_devices
        effective_batch_size = (batch_size // n_devices) * n_devices

        # History
        history = {"loss": [], "recon_loss": [], "kl": []}

        # Training loop
        with mesh:
            for epoch in range(n_epochs):
                key, epoch_key, perm_key = random.split(key, 3)

                # Shuffle data
                perm = random.permutation(perm_key, n_samples)
                data_shuffled = data_jax[perm]

                epoch_metrics = {"loss": 0, "recon_loss": 0, "kl": 0}
                n_batches = 0

                # Iterate batches
                for i in range(0, n_samples, effective_batch_size):
                    batch = data_shuffled[i:i + effective_batch_size]

                    if len(batch) < effective_batch_size:
                        continue  # Skip incomplete batch

                    key, step_key = random.split(key)

                    # Shard and train
                    batch_sharded = shard_batch(batch, mesh)
                    params, metrics = train_step(params, step_key, batch_sharded)

                    for k in epoch_metrics:
                        epoch_metrics[k] += float(metrics[k])
                    n_batches += 1

                # Average
                for k in epoch_metrics:
                    epoch_metrics[k] /= max(1, n_batches)
                    history[k].append(epoch_metrics[k])

                if verbose and (epoch + 1) % 10 == 0:
                    print(
                        f"Epoch {epoch+1}/{n_epochs}: "
                        f"loss={epoch_metrics['loss']:.4f} "
                        f"recon={epoch_metrics['recon_loss']:.4f} "
                        f"kl={epoch_metrics['kl']:.4f}"
                    )

        return params, history

    # =========================================================================
    # Encoding / Evaluation
    # =========================================================================

    def encode_to_latent(
        params: Dict,
        data: np.ndarray,
        batch_size: int = 256,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Encode data to latent representations.

        Args:
            params: Trained model parameters
            data: Input data [n_samples, *input_shape]
            batch_size: Batch size for encoding

        Returns:
            (z, mu, logvar) as numpy arrays
        """

        @jax.jit
        def encode_batch(x):
            x_flat = x.reshape(x.shape[0], -1)
            mu, logvar = encoder(params, x_flat)
            return mu, logvar

        data_jax = jnp.array(data)
        n_samples = len(data_jax)

        mus = []
        logvars = []

        for i in range(0, n_samples, batch_size):
            batch = data_jax[i:i + batch_size]
            mu, logvar = encode_batch(batch)
            mus.append(np.array(mu))
            logvars.append(np.array(logvar))

        mu_all = np.concatenate(mus, axis=0)
        logvar_all = np.concatenate(logvars, axis=0)

        # Use mean as point estimate
        z_all = mu_all

        return z_all, mu_all, logvar_all

    def reconstruct(
        params: Dict,
        data: np.ndarray,
        key: Any,
        batch_size: int = 256,
    ) -> np.ndarray:
        """
        Reconstruct data through the VAE.

        Args:
            params: Model parameters
            data: Input data
            key: Random key for sampling
            batch_size: Batch size

        Returns:
            Reconstructions as numpy array
        """

        @jax.jit
        def recon_batch(x, k):
            recon_flat, _, _, _ = beta_vae_forward(params, k, x)
            return recon_flat

        data_jax = jnp.array(data)
        n_samples = len(data_jax)
        input_shape = data.shape[1:]

        recons = []

        for i in range(0, n_samples, batch_size):
            batch = data_jax[i:i + batch_size]
            key, batch_key = random.split(key)
            recon = recon_batch(batch, batch_key)
            recons.append(np.array(recon))

        recon_all = np.concatenate(recons, axis=0)

        # Reshape back to input shape
        return recon_all.reshape(-1, *input_shape)

    def sample_from_prior(
        params: Dict,
        key: Any,
        n_samples: int,
        config: SPMDConfig,
    ) -> np.ndarray:
        """
        Sample from prior and decode.

        Args:
            params: Model parameters
            key: Random key
            n_samples: Number of samples
            config: Model config

        Returns:
            Generated samples
        """

        @jax.jit
        def decode_z(z):
            return decoder(params, z)

        z = random.normal(key, (n_samples, config.latent_dim))
        recon_flat = decode_z(z)

        return np.array(recon_flat).reshape(-1, *config.input_shape)


else:
    # Fallback when JAX not available

    @dataclass
    class SPMDConfig:
        def __init__(self, **kwargs):
            raise ImportError("JAX required for SPMDConfig")

    def init_beta_vae_params(*args, **kwargs):
        raise ImportError("JAX required")

    def create_spmd_trainer(*args, **kwargs):
        raise ImportError("JAX required")

    def train_beta_vae_spmd(*args, **kwargs):
        raise ImportError("JAX required")

    def encode_to_latent(*args, **kwargs):
        raise ImportError("JAX required")
