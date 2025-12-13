"""
AraBrain - EEG Neural Network with 2D Mesh Parallelism

A JAX/Flax implementation of β-VAE for EEG representation learning
with batch + model parallelism and NeuroBalance integration.

Architecture:
    EEG Input (B, T, C) → Conv1D Encoder → Latent z → Decoder + Telepathy Head

Key Components:
    - EEGEncoder: Conv1D temporal encoder for EEG
    - EEGAraBrain: Full model with VAE + telepathy head
    - MeshContext: 2D mesh parallelism (batch × model)

Integration:
    - Telepathy head predicts cognitive overload (D_high proxy)
    - Latent z can be probed for precision metrics via HGF bridge
    - Compatible with MIG/DCI/EDI disentanglement evaluation

Usage:
    from ara.neuro.arabrain import (
        EEGAraBrain,
        EEGEncoder,
        MeshContext,
        create_train_state,
    )

    # Create model
    model = EEGAraBrain(
        latent_dim=32,
        time=256,      # EEG window length
        channels=32,   # EEG channel count
        beta=4.0,      # KL weight
    )

    # Setup parallelism
    ctx = MeshContext(data_parallel=4, model_parallel=2)

    with ctx:
        state = create_train_state(rng, model, lr=1e-3)
        x_sharded = ctx.shard_data(x_batch)
        state, outputs = train_step(state, x_sharded, y_batch, rng)

Training:
    python -m ara.neuro.arabrain.train --demo  # Synthetic data demo
    python -m ara.neuro.arabrain.train --data_parallel 4 --batch_size 128
"""

from .encoder import (
    EEGEncoder,
    EEGDecoder,
    EEGDecoderConvTranspose,
    SpatialTemporalEncoder,
    FrequencyAwareEncoder,
    EEGEncoderConfig,
    EEGDecoderConfig,
    create_encoder,
    create_decoder,
)

from .model import (
    EEGAraBrain,
    EEGAraBrainConfig,
    EEGVAE,
    NeuroBalanceHead,
    PrecisionProbeHead,
    EEGAraBrainTrainState,
    create_train_state,
)

from .mesh import (
    MeshContext,
    ShardingConfig,
    create_mesh,
    get_mesh_info,
    get_data_sharding,
    get_data_sharding_2d,
    get_hidden_sharding,
    get_weight_sharding,
    get_bias_sharding,
    get_replicated_sharding,
    shard_params,
    with_data_sharding,
    with_hidden_sharding,
    with_replicated,
    sharded_dense,
    aggregate_gradients,
)

from .train import (
    TrainConfig,
    train,
    create_synthetic_data,
    create_data_iterator,
    make_train_step,
    make_eval_step,
)

from .edge_of_autumn import (
    MetricPoint,
    BalancedRegime,
    EdgeOfAutumnResult,
    find_edge_of_autumn,
    compute_structure_metric,
    compute_performance_metric,
    compute_robustness_metric,
    compute_deficits,
    is_in_balanced_region,
)

__all__ = [
    # Encoders
    "EEGEncoder",
    "EEGDecoder",
    "EEGDecoderConvTranspose",
    "SpatialTemporalEncoder",
    "FrequencyAwareEncoder",
    "EEGEncoderConfig",
    "EEGDecoderConfig",
    "create_encoder",
    "create_decoder",
    # Model
    "EEGAraBrain",
    "EEGAraBrainConfig",
    "EEGVAE",
    "NeuroBalanceHead",
    "PrecisionProbeHead",
    "EEGAraBrainTrainState",
    "create_train_state",
    # Mesh
    "MeshContext",
    "ShardingConfig",
    "create_mesh",
    "get_mesh_info",
    "get_data_sharding",
    "get_data_sharding_2d",
    "get_hidden_sharding",
    "get_weight_sharding",
    "get_bias_sharding",
    "get_replicated_sharding",
    "shard_params",
    "with_data_sharding",
    "with_hidden_sharding",
    "with_replicated",
    "sharded_dense",
    "aggregate_gradients",
    # Training
    "TrainConfig",
    "train",
    "create_synthetic_data",
    "create_data_iterator",
    "make_train_step",
    "make_eval_step",
    # Edge of Autumn
    "MetricPoint",
    "BalancedRegime",
    "EdgeOfAutumnResult",
    "find_edge_of_autumn",
    "compute_structure_metric",
    "compute_performance_metric",
    "compute_robustness_metric",
    "compute_deficits",
    "is_in_balanced_region",
]

__version__ = "0.1.0"
