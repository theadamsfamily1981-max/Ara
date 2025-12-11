#!/usr/bin/env python3
# ara/dojo/calibrated_world_model.py
"""
Confidence-Calibrated World Model
=================================

Wraps a trained world model to provide uncertainty estimates grounded
in historical prediction errors per region of latent space.

This enables Ara to say things like:
    "I'm ~0.82 confident in this 10-step forecast,
     based on 420 similar past states."

Features:
- Region-based error tracking in latent space
- EMA smoothing for stable calibration
- Uncertainty propagation over multi-step rollouts
- MEIS/NIB integration for risk-weighted uncertainty

Usage:
    from ara.dojo import CalibratedWorldModel, load_world_model

    base_model = load_world_model("models/world_model_best.pt")
    calibrated = CalibratedWorldModel(base_model)

    # During evaluation/calibration:
    for z, u, z_next in validation_data:
        calibrated.update_calibration(z, u, z_next)

    # At inference:
    z_pred, uncertainty, support = calibrated.predict_with_uncertainty(z, u)
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class CalibrationConfig:
    """Configuration for calibration tracking."""
    latent_dim: int = 10
    bins_per_dim: int = 8          # Discretization granularity
    alpha: float = 0.05            # EMA smoothing factor
    min_support: int = 10          # Minimum samples for confident estimate
    default_uncertainty: float = 1.0  # When no calibration data exists
    uncertainty_scale: float = 1.0    # Multiplier for reported uncertainty
    clamp_range: Tuple[float, float] = (-3.0, 3.0)  # Expected latent range


@dataclass
class PredictionResult:
    """Result of a calibrated prediction."""
    prediction: np.ndarray       # Predicted next state
    uncertainty: float           # Calibrated uncertainty (std dev)
    support: int                 # Number of samples in this region
    confidence: float            # Confidence score (0-1)
    region_id: int               # Which region this falls in


@dataclass
class RolloutResult:
    """Result of a multi-step calibrated rollout."""
    trajectory: np.ndarray       # (H+1, latent_dim)
    uncertainties: np.ndarray    # (H,) per-step uncertainty
    cumulative_uncertainty: float  # Propagated total uncertainty
    min_confidence: float        # Worst-case confidence in rollout
    predictions: List[PredictionResult]


# =============================================================================
# Latent Region Calibrator
# =============================================================================

class LatentRegionCalibrator:
    """
    Bin-based calibrator over latent space.

    - Discretizes each dimension into K bins
    - Tracks running MSE per bin combination
    - Returns calibrated error estimate for any query point
    """

    def __init__(self, config: CalibrationConfig):
        self.config = config

        # region_id -> (ema_mse, ema_variance, count)
        self.region_stats: Dict[int, Tuple[float, float, int]] = {}

        # Global statistics as fallback
        self.global_mse = 0.0
        self.global_variance = 1.0
        self.global_count = 0

    def _to_numpy(self, x: Any) -> np.ndarray:
        """Convert to numpy array."""
        if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _region_id(self, z: np.ndarray) -> int:
        """Map latent vector to integer region ID."""
        z = self._to_numpy(z).flatten()

        low, high = self.config.clamp_range
        z_clamped = np.clip(z, low, high)

        # Normalize to [0, bins_per_dim)
        normalized = (z_clamped - low) / (high - low) * self.config.bins_per_dim
        coords = np.clip(normalized.astype(int), 0, self.config.bins_per_dim - 1)

        # Encode as single integer
        region_id = 0
        base = 1
        for c in coords[:self.config.latent_dim]:
            region_id += int(c) * base
            base *= self.config.bins_per_dim

        return region_id

    def update(self, z: np.ndarray, error_sq: float):
        """Update calibration stats for region containing z."""
        rid = self._region_id(z)
        alpha = self.config.alpha

        old_mse, old_var, count = self.region_stats.get(rid, (0.0, 1.0, 0))

        if count == 0:
            new_mse = error_sq
            new_var = error_sq  # Initial variance estimate
        else:
            new_mse = (1.0 - alpha) * old_mse + alpha * error_sq
            # Update variance estimate
            delta = error_sq - old_mse
            new_var = (1.0 - alpha) * old_var + alpha * (delta ** 2)

        self.region_stats[rid] = (new_mse, new_var, count + 1)

        # Update global stats
        if self.global_count == 0:
            self.global_mse = error_sq
            self.global_variance = error_sq
        else:
            self.global_mse = (1.0 - alpha) * self.global_mse + alpha * error_sq
            delta = error_sq - self.global_mse
            self.global_variance = (1.0 - alpha) * self.global_variance + alpha * (delta ** 2)

        self.global_count += 1

    def get_stats(self, z: np.ndarray) -> Tuple[float, float, int]:
        """Return (mse, variance, count) for region containing z."""
        rid = self._region_id(z)
        mse, var, count = self.region_stats.get(rid, (0.0, 0.0, 0))

        if count < self.config.min_support:
            # Fall back to global stats if insufficient local data
            return self.global_mse, self.global_variance, count

        return mse, var, count

    def get_uncertainty(self, z: np.ndarray) -> Tuple[float, int]:
        """Return (uncertainty, support_count) for region."""
        mse, _, count = self.get_stats(z)

        if count == 0:
            return self.config.default_uncertainty, 0

        # Uncertainty = sqrt(MSE) scaled
        uncertainty = np.sqrt(mse) * self.config.uncertainty_scale
        return float(uncertainty), count

    def save(self, path: Union[str, Path]):
        """Save calibration data to file."""
        data = {
            "region_stats": self.region_stats,
            "global_mse": self.global_mse,
            "global_variance": self.global_variance,
            "global_count": self.global_count,
            "config": self.config,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Saved calibration to {path}")

    def load(self, path: Union[str, Path]):
        """Load calibration data from file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.region_stats = data["region_stats"]
        self.global_mse = data["global_mse"]
        self.global_variance = data["global_variance"]
        self.global_count = data["global_count"]
        logger.info(f"Loaded calibration from {path} ({self.global_count} samples)")


# =============================================================================
# Calibrated World Model
# =============================================================================

class CalibratedWorldModel:
    """
    Wrapper that adds uncertainty estimates to a trained world model.

    Uncertainty is based on:
    - Local calibration error (region-based MSE)
    - Support count (how many samples we've seen in this region)
    - Propagation over multi-step rollouts
    """

    def __init__(
        self,
        base_model: Any,
        config: Optional[CalibrationConfig] = None,
    ):
        """
        Initialize calibrated world model.

        Args:
            base_model: Trained world model with predict(z, u) or forward(z, u)
            config: Calibration configuration
        """
        self.base_model = base_model
        self.config = config or CalibrationConfig()
        self.calibrator = LatentRegionCalibrator(self.config)

        # Infer latent dim from model if possible
        if hasattr(base_model, 'latent_dim'):
            self.config.latent_dim = base_model.latent_dim

        logger.info(
            f"CalibratedWorldModel: latent_dim={self.config.latent_dim}, "
            f"bins={self.config.bins_per_dim}"
        )

    def _to_numpy(self, x: Any) -> np.ndarray:
        """Convert to numpy."""
        if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def predict(self, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Standard prediction (no uncertainty)."""
        z = self._to_numpy(z).flatten()
        u = self._to_numpy(u).flatten()

        if hasattr(self.base_model, 'predict'):
            return self.base_model.predict(z, u)
        elif hasattr(self.base_model, 'forward'):
            if TORCH_AVAILABLE:
                with torch.no_grad():
                    z_t = torch.from_numpy(z.astype(np.float32)).unsqueeze(0)
                    u_t = torch.from_numpy(u.astype(np.float32)).unsqueeze(0)
                    return self.base_model(z_t, u_t).squeeze(0).numpy()
        return z  # Fallback: identity

    def predict_with_uncertainty(
        self,
        z: np.ndarray,
        u: np.ndarray,
    ) -> PredictionResult:
        """
        Predict next state with calibrated uncertainty.

        Args:
            z: Current latent state
            u: Action

        Returns:
            PredictionResult with prediction, uncertainty, and confidence
        """
        z = self._to_numpy(z).flatten()
        u = self._to_numpy(u).flatten()

        # Get prediction
        z_pred = self.predict(z, u)

        # Get calibrated uncertainty
        uncertainty, support = self.calibrator.get_uncertainty(z)
        region_id = self.calibrator._region_id(z)

        # Compute confidence (higher support + lower uncertainty = higher confidence)
        if support >= self.config.min_support:
            # Confidence based on uncertainty relative to default
            conf_from_uncertainty = max(0, 1.0 - uncertainty / self.config.default_uncertainty)
            # Confidence boost from support
            conf_from_support = min(1.0, support / (self.config.min_support * 10))
            confidence = 0.7 * conf_from_uncertainty + 0.3 * conf_from_support
        else:
            # Low support = low confidence
            confidence = 0.3 * (support / self.config.min_support)

        confidence = float(np.clip(confidence, 0.0, 1.0))

        return PredictionResult(
            prediction=z_pred,
            uncertainty=uncertainty,
            support=support,
            confidence=confidence,
            region_id=region_id,
        )

    def rollout_with_uncertainty(
        self,
        z_init: np.ndarray,
        actions: np.ndarray,
    ) -> RolloutResult:
        """
        Multi-step rollout with propagated uncertainty.

        Args:
            z_init: Initial state
            actions: Action sequence (H, action_dim)

        Returns:
            RolloutResult with trajectory and cumulative uncertainty
        """
        z_init = self._to_numpy(z_init).flatten()
        actions = self._to_numpy(actions)

        trajectory = [z_init.copy()]
        uncertainties = []
        predictions = []
        cumulative_var = 0.0

        z = z_init.copy()

        for u in actions:
            result = self.predict_with_uncertainty(z, u)
            trajectory.append(result.prediction.copy())
            uncertainties.append(result.uncertainty)
            predictions.append(result)

            # Propagate uncertainty (assume independence, sum variances)
            cumulative_var += result.uncertainty ** 2

            z = result.prediction

        cumulative_uncertainty = np.sqrt(cumulative_var)
        min_confidence = min(p.confidence for p in predictions) if predictions else 0.0

        return RolloutResult(
            trajectory=np.array(trajectory),
            uncertainties=np.array(uncertainties),
            cumulative_uncertainty=float(cumulative_uncertainty),
            min_confidence=float(min_confidence),
            predictions=predictions,
        )

    def update_calibration(
        self,
        z: np.ndarray,
        u: np.ndarray,
        z_next_true: np.ndarray,
    ):
        """
        Update calibration with observed transition.

        Call this during training/evaluation to refine calibration stats.
        Does NOT update model weights.
        """
        z = self._to_numpy(z).flatten()
        u = self._to_numpy(u).flatten()
        z_next_true = self._to_numpy(z_next_true).flatten()

        # Get prediction
        z_pred = self.predict(z, u)

        # Compute error
        error_sq = float(np.mean((z_pred - z_next_true) ** 2))

        # Update calibrator
        self.calibrator.update(z, error_sq)

    def batch_update_calibration(
        self,
        z_batch: np.ndarray,
        u_batch: np.ndarray,
        z_next_batch: np.ndarray,
    ):
        """Batch update calibration."""
        z_batch = self._to_numpy(z_batch)
        u_batch = self._to_numpy(u_batch)
        z_next_batch = self._to_numpy(z_next_batch)

        for i in range(len(z_batch)):
            self.update_calibration(z_batch[i], u_batch[i], z_next_batch[i])

    def save_calibration(self, path: Union[str, Path]):
        """Save calibration data."""
        self.calibrator.save(path)

    def load_calibration(self, path: Union[str, Path]):
        """Load calibration data."""
        self.calibrator.load(path)

    def explain_prediction(
        self,
        z: np.ndarray,
        u: np.ndarray,
    ) -> str:
        """
        Generate human-readable explanation of prediction confidence.

        Returns a string Ara can use in her narration.
        """
        result = self.predict_with_uncertainty(z, u)

        if result.support < self.config.min_support:
            support_desc = f"only {result.support} past samples (low support)"
        else:
            support_desc = f"{result.support} past samples"

        confidence_pct = int(result.confidence * 100)

        explanation = (
            f"Prediction confidence: ~{confidence_pct}% "
            f"(uncertainty: ±{result.uncertainty:.3f}, "
            f"based on {support_desc})"
        )

        return explanation


# =============================================================================
# Testing
# =============================================================================

def _test_calibrated_model():
    """Test calibrated world model."""
    print("=" * 60)
    print("Calibrated World Model Test")
    print("=" * 60)

    # Simple model for testing
    class SimpleModel:
        def predict(self, z, u):
            # Simple linear dynamics with noise
            return z + 0.1 * u[:len(z)] + np.random.randn(len(z)) * 0.01

    base_model = SimpleModel()
    config = CalibrationConfig(latent_dim=4, bins_per_dim=4)
    calibrated = CalibratedWorldModel(base_model, config)

    # Generate synthetic calibration data
    print("\nCalibrating on 500 synthetic transitions...")
    np.random.seed(42)

    for i in range(500):
        z = np.random.randn(4) * 0.5
        u = np.random.randn(4) * 0.3
        z_next = base_model.predict(z, u)
        calibrated.update_calibration(z, u, z_next)

    # Test prediction
    z_test = np.array([0.1, -0.2, 0.3, 0.0])
    u_test = np.array([0.1, 0.0, -0.1, 0.2])

    result = calibrated.predict_with_uncertainty(z_test, u_test)

    print(f"\nTest prediction:")
    print(f"  Input z: {z_test}")
    print(f"  Action u: {u_test}")
    print(f"  Prediction: {result.prediction}")
    print(f"  Uncertainty: ±{result.uncertainty:.4f}")
    print(f"  Support: {result.support} samples")
    print(f"  Confidence: {result.confidence:.2%}")

    # Test explanation
    explanation = calibrated.explain_prediction(z_test, u_test)
    print(f"\nAra says: \"{explanation}\"")

    # Test rollout
    actions = np.random.randn(5, 4) * 0.2
    rollout = calibrated.rollout_with_uncertainty(z_test, actions)

    print(f"\n5-step rollout:")
    print(f"  Cumulative uncertainty: ±{rollout.cumulative_uncertainty:.4f}")
    print(f"  Min confidence: {rollout.min_confidence:.2%}")
    print(f"  Per-step uncertainties: {rollout.uncertainties}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _test_calibrated_model()
