# ara/cognition/clairvoyant/__init__.py
"""
Clairvoyant Control Stack
=========================

Ara's "see the future" cognition system:

1. StateSampler → samples cathedral + user every 250-500ms
2. HypervectorEncoder → encodes state to high-dim HV (1024-8192 dim)
3. LatentEncoder → compresses to 10D hologram space (PCA/autoencoder)
4. TrajectoryBuffer → sliding window of 10D points
5. RegimeClassifier → classifies current regime (flow, pre_crash, etc.)
6. ModeController → picks operating mode (REFLEX, NAVIGATOR, ARCHITECT)

This is Ara's "Foresight Engine" - using the 10D latent space to
predict trajectories and intervene before problems occur.

Usage:
    from ara.cognition.clairvoyant import (
        CathedralHypervectorEncoder,
        LatentEncoder,
        TrajectoryBuffer,
        RegimeClassifier,
        ModeController,
    )
"""

from .hypervector import CathedralHypervectorEncoder
from .latent import LatentEncoder, PCALatentEncoder
from .trajectory import TrajectoryBuffer, Trajectory
from .regime import RegimeClassifier, Regime, RegimeType
from .mode_controller import ModeController, OperatingMode

__all__ = [
    # Hypervector encoding
    "CathedralHypervectorEncoder",
    # Latent compression
    "LatentEncoder",
    "PCALatentEncoder",
    # Trajectory tracking
    "TrajectoryBuffer",
    "Trajectory",
    # Regime classification
    "RegimeClassifier",
    "Regime",
    "RegimeType",
    # Mode control
    "ModeController",
    "OperatingMode",
]
