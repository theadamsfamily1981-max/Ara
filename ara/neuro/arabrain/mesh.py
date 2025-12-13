"""
2D Mesh Parallelism for AraBrain

Provides batch parallelism ('data' axis) + model parallelism ('model' axis)
across multiple devices. Enables scaling AraBrain to large hidden dimensions
and batch sizes beyond single-device memory.

Architecture:
    devices = (data_parallel, model_parallel) grid

    'data' axis  → which samples each device processes
    'model' axis → which slice of hidden dimensions each device owns

Example with 8 GPUs (4×2 grid):
    mesh = create_mesh(data_parallel=4, model_parallel=2)

    - Batch dimension split across 4 data-parallel groups
    - Hidden dimensions split across 2 model-parallel groups
    - Each GPU sees 1/4 of batch, 1/2 of hidden dims

Usage:
    from ara.neuro.arabrain.mesh import (
        create_mesh,
        get_data_sharding,
        get_hidden_sharding,
        shard_params,
    )

    mesh = create_mesh(data_parallel=4, model_parallel=2)

    with mesh:
        x = jax.device_put(x, get_data_sharding(mesh))
        state, outputs = train_step(state, x, y, rng)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Callable

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    Mesh = None
    P = None
    NamedSharding = None


# =============================================================================
# Mesh Creation
# =============================================================================

def create_mesh(
    data_parallel: int = 1,
    model_parallel: int = 1,
    devices: Optional[Any] = None,
) -> "Mesh":
    """
    Create a 2D device mesh for batch + model parallelism.

    Args:
        data_parallel: Number of devices for batch parallelism
        model_parallel: Number of devices for model parallelism
        devices: Optional explicit device array. If None, uses all available.

    Returns:
        JAX Mesh with axes ('data', 'model')

    Example:
        # 8 GPUs: 4 for data parallel, 2 for model parallel
        mesh = create_mesh(data_parallel=4, model_parallel=2)

        # Single GPU (no parallelism, but same API)
        mesh = create_mesh(data_parallel=1, model_parallel=1)
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX required for mesh parallelism")

    total_devices = data_parallel * model_parallel

    if devices is None:
        available = jax.devices()
        if len(available) < total_devices:
            # Fallback: use what we have
            print(f"Warning: Requested {total_devices} devices, "
                  f"only {len(available)} available. Adjusting mesh.")
            data_parallel = max(1, len(available))
            model_parallel = 1
            total_devices = data_parallel
        devices = np.array(available[:total_devices])

    # Reshape into 2D grid
    device_grid = devices.reshape(data_parallel, model_parallel)

    return Mesh(device_grid, axis_names=('data', 'model'))


def get_mesh_info(mesh: "Mesh") -> Dict[str, Any]:
    """Get information about a mesh configuration."""
    return {
        "axis_names": mesh.axis_names,
        "shape": mesh.devices.shape,
        "data_parallel": mesh.devices.shape[0],
        "model_parallel": mesh.devices.shape[1],
        "total_devices": mesh.devices.size,
        "device_ids": [str(d) for d in mesh.devices.flatten()],
    }


# =============================================================================
# Sharding Specifications
# =============================================================================

@dataclass
class ShardingConfig:
    """Configuration for how tensors should be sharded across the mesh."""

    # Batch dimension sharding
    batch_axis: str = 'data'

    # Hidden/feature dimension sharding
    hidden_axis: str = 'model'

    # Which layers to apply model parallelism to
    model_parallel_layers: Tuple[str, ...] = (
        'encoder/Dense_0',
        'encoder/Dense_1',
        'decoder/Dense_0',
        'latent_mu',
        'latent_logvar',
    )


def get_data_sharding(mesh: "Mesh") -> "NamedSharding":
    """
    Get sharding for batch data: split on 'data' axis, replicate features.

    For input x with shape (batch, time, channels):
        - batch split across data-parallel devices
        - time and channels replicated on each device
    """
    return NamedSharding(mesh, P('data', None, None))


def get_data_sharding_2d(mesh: "Mesh") -> "NamedSharding":
    """
    Get sharding for 2D batch data: (batch, features).
    """
    return NamedSharding(mesh, P('data', None))


def get_hidden_sharding(mesh: "Mesh") -> "NamedSharding":
    """
    Get sharding for hidden activations: (batch, hidden_dim).

    - Batch split across 'data' axis
    - Hidden dimension split across 'model' axis
    """
    return NamedSharding(mesh, P('data', 'model'))


def get_weight_sharding(mesh: "Mesh", axis: int = 1) -> "NamedSharding":
    """
    Get sharding for weight matrices.

    For W with shape (input_dim, output_dim):
        axis=0: Shard input dimension across 'model'
        axis=1: Shard output dimension across 'model' (default)
    """
    if axis == 0:
        return NamedSharding(mesh, P('model', None))
    else:
        return NamedSharding(mesh, P(None, 'model'))


def get_bias_sharding(mesh: "Mesh") -> "NamedSharding":
    """Get sharding for bias vectors: shard across 'model'."""
    return NamedSharding(mesh, P('model'))


def get_replicated_sharding(mesh: "Mesh") -> "NamedSharding":
    """Get sharding for fully replicated tensors."""
    return NamedSharding(mesh, P())


# =============================================================================
# Parameter Sharding
# =============================================================================

def shard_params(
    params: Dict,
    mesh: "Mesh",
    config: Optional[ShardingConfig] = None,
) -> Dict:
    """
    Apply sharding to model parameters.

    Large weight matrices in specified layers get split across 'model' axis.
    Other parameters stay replicated.

    Args:
        params: PyTree of model parameters
        mesh: Device mesh
        config: Sharding configuration

    Returns:
        Sharded parameter tree
    """
    if config is None:
        config = ShardingConfig()

    replicated = get_replicated_sharding(mesh)
    weight_sharded = get_weight_sharding(mesh, axis=1)
    bias_sharded = get_bias_sharding(mesh)

    def shard_leaf(path: str, param: Any) -> Any:
        """Determine sharding for a single parameter."""
        # Check if this layer should be model-parallel
        should_shard = any(layer in path for layer in config.model_parallel_layers)

        if not should_shard:
            return jax.device_put(param, replicated)

        # Shard weights and biases differently
        if 'kernel' in path and param.ndim == 2:
            return jax.device_put(param, weight_sharded)
        elif 'bias' in path and param.ndim == 1:
            return jax.device_put(param, bias_sharded)
        else:
            return jax.device_put(param, replicated)

    def _recursive_shard(tree: Dict, prefix: str = '') -> Dict:
        """Recursively shard a parameter tree."""
        result = {}
        for key, value in tree.items():
            path = f"{prefix}/{key}" if prefix else key
            if isinstance(value, dict):
                result[key] = _recursive_shard(value, path)
            else:
                result[key] = shard_leaf(path, value)
        return result

    with mesh:
        return _recursive_shard(params)


# =============================================================================
# Sharding Constraints for Model Code
# =============================================================================

def with_data_sharding(x: Any, mesh: "Mesh") -> Any:
    """Apply data sharding constraint to activation."""
    return jax.lax.with_sharding_constraint(
        x, NamedSharding(mesh, P('data', None, None))
    )


def with_hidden_sharding(x: Any, mesh: "Mesh") -> Any:
    """Apply hidden sharding constraint to activation (batch, hidden)."""
    return jax.lax.with_sharding_constraint(
        x, NamedSharding(mesh, P('data', 'model'))
    )


def with_replicated(x: Any, mesh: "Mesh") -> Any:
    """Apply replicated constraint (all devices have full copy)."""
    return jax.lax.with_sharding_constraint(
        x, NamedSharding(mesh, P())
    )


# =============================================================================
# Sharded Dense Layer Helper
# =============================================================================

def sharded_dense(
    x: Any,
    W: Any,
    b: Any,
    mesh: "Mesh",
    activation: Optional[Callable] = None,
) -> Any:
    """
    Apply dense layer with explicit sharding constraints.

    This ensures the compiler knows how to distribute computation:
    - x: (batch, in_features) sharded as P('data', 'model')
    - W: (in_features, out_features) sharded as P(None, 'model')
    - b: (out_features,) sharded as P('model')

    Args:
        x: Input tensor
        W: Weight matrix
        b: Bias vector
        mesh: Device mesh
        activation: Optional activation function

    Returns:
        Output tensor with sharding constraints applied
    """
    with mesh:
        # Apply sharding constraints
        x = jax.lax.with_sharding_constraint(
            x, NamedSharding(mesh, P('data', 'model'))
        )
        W = jax.lax.with_sharding_constraint(
            W, NamedSharding(mesh, P(None, 'model'))
        )
        b = jax.lax.with_sharding_constraint(
            b, NamedSharding(mesh, P('model'))
        )

        # Compute
        y = x @ W + b

        if activation is not None:
            y = activation(y)

        return y


# =============================================================================
# Gradient Aggregation
# =============================================================================

def aggregate_gradients(grads: Dict, mesh: "Mesh") -> Dict:
    """
    Aggregate gradients across data-parallel devices.

    After backward pass, each data-parallel device has gradients for its
    batch slice. This function averages them across the 'data' axis.
    """
    def _mean_across_data(x):
        return jax.lax.pmean(x, axis_name='data')

    return jax.tree_util.tree_map(_mean_across_data, grads)


# =============================================================================
# Convenience Context Manager
# =============================================================================

class MeshContext:
    """
    Context manager for mesh-based training.

    Usage:
        ctx = MeshContext(data_parallel=4, model_parallel=2)

        with ctx:
            x = ctx.shard_data(x_batch)
            state, outputs = train_step(state, x, y, rng)
    """

    def __init__(
        self,
        data_parallel: int = 1,
        model_parallel: int = 1,
        devices: Optional[Any] = None,
    ):
        self.mesh = create_mesh(data_parallel, model_parallel, devices)
        self._data_sharding = None
        self._hidden_sharding = None
        self._replicated_sharding = None

    def __enter__(self):
        self._data_sharding = get_data_sharding(self.mesh)
        self._hidden_sharding = get_hidden_sharding(self.mesh)
        self._replicated_sharding = get_replicated_sharding(self.mesh)
        self.mesh.__enter__()
        return self

    def __exit__(self, *args):
        return self.mesh.__exit__(*args)

    def shard_data(self, x: Any) -> Any:
        """Shard input data across 'data' axis."""
        return jax.device_put(x, self._data_sharding)

    def shard_data_2d(self, x: Any) -> Any:
        """Shard 2D input data (batch, features)."""
        return jax.device_put(x, get_data_sharding_2d(self.mesh))

    def shard_hidden(self, x: Any) -> Any:
        """Shard hidden activations across both axes."""
        return jax.device_put(x, self._hidden_sharding)

    def replicate(self, x: Any) -> Any:
        """Replicate tensor to all devices."""
        return jax.device_put(x, self._replicated_sharding)

    def shard_params(self, params: Dict, config: Optional[ShardingConfig] = None) -> Dict:
        """Shard model parameters."""
        return shard_params(params, self.mesh, config)

    @property
    def info(self) -> Dict[str, Any]:
        """Get mesh information."""
        return get_mesh_info(self.mesh)


# =============================================================================
# Demo
# =============================================================================

def demo_mesh():
    """Demonstrate mesh creation and sharding."""
    if not JAX_AVAILABLE:
        print("JAX not available - skipping mesh demo")
        return

    print("\n" + "=" * 70)
    print("2D MESH PARALLELISM DEMO")
    print("=" * 70)

    # Create mesh
    n_devices = len(jax.devices())
    print(f"\nAvailable devices: {n_devices}")

    if n_devices >= 2:
        data_p = max(1, n_devices // 2)
        model_p = n_devices // data_p
    else:
        data_p, model_p = 1, 1

    print(f"Creating mesh: data_parallel={data_p}, model_parallel={model_p}")

    ctx = MeshContext(data_parallel=data_p, model_parallel=model_p)

    with ctx:
        print(f"\nMesh info: {ctx.info}")

        # Demo data sharding
        batch_size = 8
        time_steps = 256
        channels = 32

        x = jnp.ones((batch_size, time_steps, channels))
        x_sharded = ctx.shard_data(x)

        print(f"\nInput shape: {x.shape}")
        print(f"Input sharding: {x_sharded.sharding}")

        # Demo hidden sharding
        hidden_dim = 128
        h = jnp.ones((batch_size, hidden_dim))
        h_sharded = ctx.shard_hidden(h)

        print(f"\nHidden shape: {h.shape}")
        print(f"Hidden sharding: {h_sharded.sharding}")

        # Demo weight sharding
        W = jnp.ones((hidden_dim, 256))
        W_sharded = jax.device_put(W, get_weight_sharding(ctx.mesh))

        print(f"\nWeight shape: {W.shape}")
        print(f"Weight sharding: {W_sharded.sharding}")

    print("\n" + "=" * 70)
    print("Demo complete!")


if __name__ == "__main__":
    demo_mesh()
