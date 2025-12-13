"""
ara.vae.flax_brain - AraBrain: β-VAE + Telepathy Head in Flax

The complete Ara cognitive encoder combining:
1. β-VAE for disentangled latent representation
2. NeuroBalanceHead for comfort/overload classification
3. Joint training with combined loss

The telepathy head learns to shape the latent space so it's useful for
"is Croft ok?" decisions, not just compression.

Usage:
    from ara.vae.flax_brain import AraBrain, create_train_state, train_step

    model = AraBrain(latent_dim=32, input_dim=64*64, beta=4.0)
    state = create_train_state(rng, model, learning_rate=1e-3)

    for epoch in range(100):
        for x_batch, y_batch in dataloader:
            state, outputs = train_step(state, x_batch, y_batch, rng)

    # Extract latents for MIG/DCI/EDI
    latents = encode_dataset(state, dataset, rng)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np

# JAX imports
try:
    import jax
    import jax.numpy as jnp
    from jax import random
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = np

# Flax imports
try:
    from flax import linen as nn
    from flax.training import train_state
    import optax
    HAS_FLAX = True
except ImportError:
    HAS_FLAX = False


if HAS_JAX and HAS_FLAX:

    # =========================================================================
    # Model Components
    # =========================================================================

    class Encoder(nn.Module):
        """VAE Encoder: x → (μ, log σ²)"""
        latent_dim: int
        hidden_dims: Tuple[int, ...] = (256, 128)

        @nn.compact
        def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            # Flatten input
            x = x.reshape((x.shape[0], -1))

            # Hidden layers
            for dim in self.hidden_dims:
                x = nn.Dense(dim)(x)
                x = nn.relu(x)

            # Latent parameters
            mu = nn.Dense(self.latent_dim)(x)
            logvar = nn.Dense(self.latent_dim)(x)

            return mu, logvar

    class Decoder(nn.Module):
        """VAE Decoder: z → x̂"""
        output_dim: int
        hidden_dims: Tuple[int, ...] = (128, 256)

        @nn.compact
        def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
            x = z

            # Hidden layers
            for dim in self.hidden_dims:
                x = nn.Dense(dim)(x)
                x = nn.relu(x)

            # Output
            x = nn.Dense(self.output_dim)(x)
            x = nn.sigmoid(x)  # [0, 1] for images

            return x

    class BetaVAE(nn.Module):
        """β-VAE: Disentangled representation learning."""
        latent_dim: int
        input_dim: int
        beta: float = 4.0
        hidden_dims: Tuple[int, ...] = (256, 128)

        def setup(self):
            self.encoder = Encoder(
                latent_dim=self.latent_dim,
                hidden_dims=self.hidden_dims,
            )
            self.decoder = Decoder(
                output_dim=self.input_dim,
                hidden_dims=tuple(reversed(self.hidden_dims)),
            )

        def __call__(
            self,
            x: jnp.ndarray,
            rng: jax.Array,
        ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            """
            Forward pass with reparameterization.

            Returns:
                (reconstruction, mu, logvar, z)
            """
            # Flatten
            x_flat = x.reshape((x.shape[0], -1))

            # Encode
            mu, logvar = self.encoder(x)

            # Reparameterize
            std = jnp.exp(0.5 * logvar)
            eps = random.normal(rng, std.shape)
            z = mu + eps * std

            # Decode
            recon = self.decoder(z)

            return recon, mu, logvar, z

        def loss(
            self,
            x: jnp.ndarray,
            rng: jax.Array,
        ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray], jnp.ndarray]:
            """
            Compute β-VAE ELBO loss.

            Returns:
                (total_loss, metrics_dict, z)
            """
            x_flat = x.reshape((x.shape[0], -1))
            recon, mu, logvar, z = self(x, rng)

            # Reconstruction loss (BCE)
            bce = -(
                x_flat * jnp.log(recon + 1e-8) +
                (1.0 - x_flat) * jnp.log(1.0 - recon + 1e-8)
            )
            recon_loss = jnp.mean(jnp.sum(bce, axis=1))

            # KL divergence
            kl = -0.5 * jnp.sum(
                1 + logvar - jnp.square(mu) - jnp.exp(logvar),
                axis=1,
            )
            kl_loss = jnp.mean(kl)

            # Total VAE loss
            total = recon_loss + self.beta * kl_loss

            metrics = {
                "recon_loss": recon_loss,
                "kl_loss": kl_loss,
                "vae_loss": total,
            }

            return total, metrics, z

        def encode(self, x: jnp.ndarray) -> jnp.ndarray:
            """Encode to latent mean (for evaluation)."""
            mu, _ = self.encoder(x)
            return mu

    class NeuroBalanceHead(nn.Module):
        """
        Telepathy head: z → overload probability.

        Maps latent representation to comfort/overload classification.
        """
        hidden_dim: int = 64

        @nn.compact
        def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
            x = nn.Dense(self.hidden_dim)(z)
            x = nn.relu(x)
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.relu(x)
            logit = nn.Dense(1)(x)  # (batch, 1)
            return logit

    class AraBrain(nn.Module):
        """
        Complete Ara cognitive encoder.

        Combines:
        - β-VAE for disentangled representation
        - NeuroBalanceHead for comfort/overload classification

        The telepathy loss shapes the latent space to be useful for
        Ara's "is Croft ok?" decisions.
        """
        latent_dim: int
        input_dim: int
        beta: float = 4.0
        telepathy_weight: float = 1.0
        hidden_dims: Tuple[int, ...] = (256, 128)

        def setup(self):
            self.vae = BetaVAE(
                latent_dim=self.latent_dim,
                input_dim=self.input_dim,
                beta=self.beta,
                hidden_dims=self.hidden_dims,
            )
            self.telepathy = NeuroBalanceHead()

        def __call__(
            self,
            x: jnp.ndarray,
            rng: jax.Array,
            labels: Optional[jnp.ndarray] = None,
        ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
            """
            Forward pass with optional telepathy loss.

            Args:
                x: Input batch
                rng: Random key
                labels: Optional (batch,) binary labels (0=comfy, 1=overload)

            Returns:
                (total_loss, outputs_dict)
            """
            vae_rng, _ = random.split(rng)

            # VAE loss
            vae_loss, vae_metrics, z = self.vae.loss(x, vae_rng)

            outputs = {"z": z, **vae_metrics}

            # Telepathy loss (if labels provided)
            telepathy_loss = jnp.array(0.0)
            if labels is not None:
                logits = self.telepathy(z)  # (batch, 1)
                labels_reshaped = labels.reshape((-1, 1))
                probs = jax.nn.sigmoid(logits)

                # Binary cross-entropy
                bce = -(
                    labels_reshaped * jnp.log(probs + 1e-8) +
                    (1 - labels_reshaped) * jnp.log(1 - probs + 1e-8)
                )
                telepathy_loss = jnp.mean(bce)

                outputs["telepathy_loss"] = telepathy_loss
                outputs["overload_prob"] = jnp.mean(probs)
                outputs["overload_logits"] = logits

            # Total loss
            total_loss = vae_loss + self.telepathy_weight * telepathy_loss
            outputs["total_loss"] = total_loss

            return total_loss, outputs

        def encode(self, x: jnp.ndarray) -> jnp.ndarray:
            """Encode input to latent representation."""
            return self.vae.encode(x)

        def predict_overload(self, z: jnp.ndarray) -> jnp.ndarray:
            """Predict overload probability from latent."""
            logits = self.telepathy(z)
            return jax.nn.sigmoid(logits)

    # =========================================================================
    # Training Infrastructure
    # =========================================================================

    class AraBrainTrainState(train_state.TrainState):
        """Extended TrainState with model reference."""
        model: Any = None

    def create_train_state(
        rng: jax.Array,
        model: AraBrain,
        learning_rate: float = 1e-3,
    ) -> AraBrainTrainState:
        """
        Initialize model and create training state.

        Args:
            rng: Random key
            model: AraBrain instance
            learning_rate: Adam learning rate

        Returns:
            Initialized AraBrainTrainState
        """
        # Initialize with dummy input
        dummy_x = jnp.ones((1, model.input_dim))
        params = model.init(rng, dummy_x, rng, None)

        # Create optimizer
        tx = optax.adam(learning_rate)

        return AraBrainTrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=tx,
            model=model,
        )

    @jax.jit
    def train_step(
        state: AraBrainTrainState,
        x_batch: jnp.ndarray,
        y_batch: Optional[jnp.ndarray],
        rng: jax.Array,
    ) -> Tuple[AraBrainTrainState, Dict[str, Any]]:
        """
        Single training step.

        Args:
            state: Current training state
            x_batch: Input batch
            y_batch: Labels (or None for VAE-only)
            rng: Random key

        Returns:
            (new_state, outputs_dict)
        """
        def loss_fn(params):
            loss, outputs = state.apply_fn(params, x_batch, rng, y_batch)
            return loss, outputs

        (loss, outputs), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

        # Update
        state = state.apply_gradients(grads=grads)

        return state, outputs

    def train_ara_brain(
        data: np.ndarray,
        labels: Optional[np.ndarray],
        latent_dim: int = 32,
        beta: float = 4.0,
        telepathy_weight: float = 1.0,
        n_epochs: int = 50,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        seed: int = 42,
        verbose: bool = True,
    ) -> Tuple[AraBrainTrainState, Dict[str, list]]:
        """
        Train AraBrain on data.

        Args:
            data: Training data [n_samples, ...]
            labels: Binary labels [n_samples] or None
            latent_dim: Latent dimensionality
            beta: β-VAE weight
            telepathy_weight: Telepathy loss weight
            n_epochs: Training epochs
            batch_size: Batch size
            learning_rate: Adam LR
            seed: Random seed
            verbose: Print progress

        Returns:
            (trained_state, history)
        """
        rng = random.PRNGKey(seed)
        rng, init_rng = random.split(rng)

        # Flatten input dim
        input_dim = int(np.prod(data.shape[1:]))

        # Create model and state
        model = AraBrain(
            latent_dim=latent_dim,
            input_dim=input_dim,
            beta=beta,
            telepathy_weight=telepathy_weight,
        )
        state = create_train_state(init_rng, model, learning_rate)

        if verbose:
            print(f"Training AraBrain (β={beta}, z={latent_dim}, λ_tel={telepathy_weight})")
            print(f"  Input dim: {input_dim}")
            n_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
            print(f"  Parameters: {n_params:,}")

        # Convert to JAX
        data_jax = jnp.array(data.reshape(len(data), -1))
        labels_jax = jnp.array(labels) if labels is not None else None

        n_samples = len(data_jax)
        history = {"total_loss": [], "vae_loss": [], "telepathy_loss": [], "recon_loss": [], "kl_loss": []}

        # Training loop
        for epoch in range(n_epochs):
            rng, epoch_rng, perm_rng = random.split(rng, 3)

            # Shuffle
            perm = random.permutation(perm_rng, n_samples)
            data_shuffled = data_jax[perm]
            labels_shuffled = labels_jax[perm] if labels_jax is not None else None

            epoch_metrics = {k: 0.0 for k in history.keys()}
            n_batches = 0

            for i in range(0, n_samples, batch_size):
                x_batch = data_shuffled[i:i+batch_size]
                y_batch = labels_shuffled[i:i+batch_size] if labels_shuffled is not None else None

                if len(x_batch) < batch_size:
                    continue

                rng, step_rng = random.split(rng)
                state, outputs = train_step(state, x_batch, y_batch, step_rng)

                epoch_metrics["total_loss"] += float(outputs["total_loss"])
                epoch_metrics["vae_loss"] += float(outputs["vae_loss"])
                epoch_metrics["recon_loss"] += float(outputs["recon_loss"])
                epoch_metrics["kl_loss"] += float(outputs["kl_loss"])
                epoch_metrics["telepathy_loss"] += float(outputs.get("telepathy_loss", 0.0))
                n_batches += 1

            # Average
            for k in epoch_metrics:
                epoch_metrics[k] /= max(1, n_batches)
                history[k].append(epoch_metrics[k])

            if verbose and (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{n_epochs}: "
                    f"total={epoch_metrics['total_loss']:.4f} "
                    f"vae={epoch_metrics['vae_loss']:.4f} "
                    f"tel={epoch_metrics['telepathy_loss']:.4f}"
                )

        return state, history

    # =========================================================================
    # Encoding and Evaluation
    # =========================================================================

    def encode_dataset(
        state: AraBrainTrainState,
        data: np.ndarray,
        batch_size: int = 256,
    ) -> np.ndarray:
        """
        Encode full dataset to latent representations.

        Args:
            state: Trained state
            data: Data to encode
            batch_size: Batch size

        Returns:
            Latent representations [n_samples, latent_dim]
        """
        @jax.jit
        def encode_batch(params, x):
            return state.model.vae.apply(
                {'params': params['params']['vae']},
                x,
                method=state.model.vae.encode,
            )

        data_flat = data.reshape(len(data), -1)
        data_jax = jnp.array(data_flat)

        latents = []
        for i in range(0, len(data_jax), batch_size):
            batch = data_jax[i:i+batch_size]
            z = encode_batch(state.params, batch)
            latents.append(np.array(z))

        return np.concatenate(latents, axis=0)

    def predict_overload(
        state: AraBrainTrainState,
        latents: np.ndarray,
        batch_size: int = 256,
    ) -> np.ndarray:
        """
        Predict overload probability from latents.

        Args:
            state: Trained state
            latents: Latent representations
            batch_size: Batch size

        Returns:
            Overload probabilities [n_samples]
        """
        @jax.jit
        def predict_batch(params, z):
            logits = state.model.telepathy.apply(
                {'params': params['params']['telepathy']},
                z,
            )
            return jax.nn.sigmoid(logits)

        latents_jax = jnp.array(latents)

        probs = []
        for i in range(0, len(latents_jax), batch_size):
            batch = latents_jax[i:i+batch_size]
            p = predict_batch(state.params, batch)
            probs.append(np.array(p).flatten())

        return np.concatenate(probs)


else:
    # Fallback when JAX/Flax not available

    class AraBrain:
        def __init__(self, *args, **kwargs):
            raise ImportError("JAX and Flax required for AraBrain")

    def create_train_state(*args, **kwargs):
        raise ImportError("JAX and Flax required")

    def train_step(*args, **kwargs):
        raise ImportError("JAX and Flax required")

    def train_ara_brain(*args, **kwargs):
        raise ImportError("JAX and Flax required")

    def encode_dataset(*args, **kwargs):
        raise ImportError("JAX and Flax required")

    def predict_overload(*args, **kwargs):
        raise ImportError("JAX and Flax required")
