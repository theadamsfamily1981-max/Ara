"""
ara.vae - Variational Autoencoder for HGF Trajectory Analysis

This package provides VAE models for learning disentangled latent representations
of HGF trajectories, enabling phenotype clustering and generative modeling of
computational psychiatry signatures.

Key Components:
- TrajectoryVAE: PyTorch VAE for HGF trajectory encoding
- JAXTrajectoryVAE: JAX/Flax VAE with SPMD parallelism
- Disentanglement metrics: DCI, MIG, SAP, EDI
- Parallel computation: MINE-based MI, ProcessPool DCI
"""

from ara.vae.core import (
    TrajectoryVAE,
    TrajectoryEncoder,
    TrajectoryDecoder,
    VAEConfig,
    VAELoss,
    SimpleVAE,
)
from ara.vae.metrics import (
    compute_dci,
    compute_mig,
    compute_sap,
    compute_edi,
    compute_dci_lite,
    DisentanglementReport,
    evaluate_disentanglement,
)
from ara.vae.data import (
    TrajectoryDataset,
    TrajectoryData,
    generate_phenotype_dataset,
    generate_parameter_sweep_dataset,
    PhenotypeLabel,
)

# Optional JAX imports (may not be available)
try:
    from ara.vae.jax_vae import (
        JAXTrajectoryVAE,
        JAXVAEConfig,
        create_mesh,
        shard_batch,
        train_vae_spmd,
        encode_trajectories,
    )
    HAS_JAX_VAE = True
except ImportError:
    HAS_JAX_VAE = False

# Parallel metrics (always available with ProcessPool fallback)
from ara.vae.parallel_metrics import (
    compute_dci_parallel,
    generate_dataset_parallel,
)

# Optional MINE-based parallel EDI
try:
    from ara.vae.parallel_metrics import (
        estimate_mi_mine,
        compute_mi_matrix_parallel,
        compute_edi_parallel,
    )
    HAS_MINE = True
except ImportError:
    HAS_MINE = False

__version__ = "0.1.0"
__all__ = [
    # Core VAE (PyTorch)
    "TrajectoryVAE",
    "TrajectoryEncoder",
    "TrajectoryDecoder",
    "VAEConfig",
    "VAELoss",
    "SimpleVAE",
    # Metrics
    "compute_dci",
    "compute_mig",
    "compute_sap",
    "compute_edi",
    "compute_dci_lite",
    "DisentanglementReport",
    "evaluate_disentanglement",
    # Data
    "TrajectoryDataset",
    "TrajectoryData",
    "generate_phenotype_dataset",
    "generate_parameter_sweep_dataset",
    "PhenotypeLabel",
    # Parallel (CPU)
    "compute_dci_parallel",
    "generate_dataset_parallel",
]

# Conditionally add JAX exports
if HAS_JAX_VAE:
    __all__.extend([
        "JAXTrajectoryVAE",
        "JAXVAEConfig",
        "create_mesh",
        "shard_batch",
        "train_vae_spmd",
        "encode_trajectories",
    ])

# Conditionally add MINE exports
if HAS_MINE:
    __all__.extend([
        "estimate_mi_mine",
        "compute_mi_matrix_parallel",
        "compute_edi_parallel",
    ])
