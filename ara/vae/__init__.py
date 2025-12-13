"""
ara.vae - Variational Autoencoder for HGF Trajectory Analysis

This package provides VAE models for learning disentangled latent representations
of HGF trajectories, enabling phenotype clustering and generative modeling of
computational psychiatry signatures.

Key Components:
- TrajectoryVAE: VAE specialized for HGF trajectory encoding
- Disentanglement metrics: DCI, MIG, SAP, EDI
- Phenotype clustering and visualization
"""

from ara.vae.core import (
    TrajectoryVAE,
    TrajectoryEncoder,
    TrajectoryDecoder,
    VAEConfig,
    VAELoss,
)
from ara.vae.metrics import (
    compute_dci,
    compute_mig,
    compute_sap,
    compute_edi,
    DisentanglementReport,
    evaluate_disentanglement,
)
from ara.vae.data import (
    TrajectoryDataset,
    generate_phenotype_dataset,
    PhenotypeLabel,
)

__version__ = "0.1.0"
__all__ = [
    # Core VAE
    "TrajectoryVAE",
    "TrajectoryEncoder",
    "TrajectoryDecoder",
    "VAEConfig",
    "VAELoss",
    # Metrics
    "compute_dci",
    "compute_mig",
    "compute_sap",
    "compute_edi",
    "DisentanglementReport",
    "evaluate_disentanglement",
    # Data
    "TrajectoryDataset",
    "generate_phenotype_dataset",
    "PhenotypeLabel",
]
