"""
Emotion prediction heads for VA (Valence-Arousal) or PAD (Pleasure-Arousal-Dominance).

Predicts emotional states from latent representations with:
- MSE or CCC (Concordance Correlation Coefficient) loss
- Temporal smoothness regularization
- Optional topological trajectory loss (β₀=1, β₁=0 on VA curve)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class EmotionPrediction:
    """Container for emotion predictions."""
    valence: Optional[torch.Tensor] = None  # (batch, seq_len) or (batch,)
    arousal: Optional[torch.Tensor] = None  # (batch, seq_len) or (batch,)
    dominance: Optional[torch.Tensor] = None  # (batch, seq_len) or (batch,)
    pleasure: Optional[torch.Tensor] = None  # Alias for valence in PAD
    confidence: torch.Tensor = None  # (batch, seq_len) or (batch,)


def concordance_correlation_coefficient(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute Concordance Correlation Coefficient (CCC) loss.

    CCC measures agreement between predictions and targets.
    CCC = 2 * cov(pred, target) / (var(pred) + var(target) + (mean(pred) - mean(target))²)

    Args:
        pred: Predictions (batch, ...)
        target: Targets (batch, ...)
        eps: Numerical stability constant

    Returns:
        CCC loss (scalar)
    """
    pred_mean = pred.mean()
    target_mean = target.mean()

    pred_var = pred.var()
    target_var = target.var()

    # Covariance
    covariance = ((pred - pred_mean) * (target - target_mean)).mean()

    # CCC
    ccc = (2 * covariance) / (pred_var + target_var + (pred_mean - target_mean) ** 2 + eps)

    # Return 1 - CCC as loss (CCC is in [-1, 1], we want to maximize it)
    return 1.0 - ccc


class EmotionHead(nn.Module):
    """
    Emotion prediction head.

    Predicts emotional states from latent representations.
    """

    def __init__(
        self,
        d_model: int,
        mode: str = "VA",
        loss_type: str = "CCC",
        temporal_smoothness_weight: float = 0.1,
        enable_topo_trajectory: bool = False,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_model: Input dimension
            mode: "VA" (valence-arousal) or "PAD" (pleasure-arousal-dominance)
            loss_type: "MSE" or "CCC"
            temporal_smoothness_weight: Weight for temporal smoothness penalty
            enable_topo_trajectory: Enable topological trajectory loss
            dropout: Dropout rate
        """
        super().__init__()
        self.d_model = d_model
        self.mode = mode
        self.loss_type = loss_type
        self.temporal_smoothness_weight = temporal_smoothness_weight
        self.enable_topo_trajectory = enable_topo_trajectory

        # Prediction heads
        if mode == "VA":
            self.valence_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
                nn.Tanh(),  # Output in [-1, 1]
            )
            self.arousal_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
                nn.Sigmoid(),  # Output in [0, 1]
            )
        elif mode == "PAD":
            self.pleasure_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
                nn.Tanh(),
            )
            self.arousal_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
                nn.Sigmoid(),
            )
            self.dominance_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
                nn.Tanh(),
            )
        else:
            raise ValueError(f"Unknown emotion mode: {mode}")

        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, latents: torch.Tensor) -> EmotionPrediction:
        """
        Predict emotion from latents.

        Args:
            latents: Latent representations (batch, seq_len, d_model) or (batch, d_model)

        Returns:
            EmotionPrediction with predicted states
        """
        # Predict emotions
        if self.mode == "VA":
            valence = self.valence_head(latents).squeeze(-1)  # (batch, seq_len) or (batch,)
            arousal = self.arousal_head(latents).squeeze(-1)
            confidence = self.confidence_head(latents).squeeze(-1)

            return EmotionPrediction(
                valence=valence,
                arousal=arousal,
                confidence=confidence,
            )
        elif self.mode == "PAD":
            pleasure = self.pleasure_head(latents).squeeze(-1)
            arousal = self.arousal_head(latents).squeeze(-1)
            dominance = self.dominance_head(latents).squeeze(-1)
            confidence = self.confidence_head(latents).squeeze(-1)

            return EmotionPrediction(
                pleasure=pleasure,
                arousal=arousal,
                dominance=dominance,
                confidence=confidence,
            )

    def compute_loss(
        self,
        predictions: EmotionPrediction,
        targets: EmotionPrediction,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute emotion prediction loss.

        Args:
            predictions: Predicted emotions
            targets: Target emotions

        Returns:
            (total_loss, loss_dict)
        """
        losses = {}

        # Main prediction loss
        if self.mode == "VA":
            if self.loss_type == "MSE":
                valence_loss = F.mse_loss(predictions.valence, targets.valence)
                arousal_loss = F.mse_loss(predictions.arousal, targets.arousal)
            elif self.loss_type == "CCC":
                valence_loss = concordance_correlation_coefficient(
                    predictions.valence, targets.valence
                )
                arousal_loss = concordance_correlation_coefficient(
                    predictions.arousal, targets.arousal
                )
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")

            losses["valence_loss"] = valence_loss.item()
            losses["arousal_loss"] = arousal_loss.item()
            main_loss = valence_loss + arousal_loss

        elif self.mode == "PAD":
            if self.loss_type == "MSE":
                pleasure_loss = F.mse_loss(predictions.pleasure, targets.pleasure)
                arousal_loss = F.mse_loss(predictions.arousal, targets.arousal)
                dominance_loss = F.mse_loss(predictions.dominance, targets.dominance)
            elif self.loss_type == "CCC":
                pleasure_loss = concordance_correlation_coefficient(
                    predictions.pleasure, targets.pleasure
                )
                arousal_loss = concordance_correlation_coefficient(
                    predictions.arousal, targets.arousal
                )
                dominance_loss = concordance_correlation_coefficient(
                    predictions.dominance, targets.dominance
                )
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")

            losses["pleasure_loss"] = pleasure_loss.item()
            losses["arousal_loss"] = arousal_loss.item()
            losses["dominance_loss"] = dominance_loss.item()
            main_loss = pleasure_loss + arousal_loss + dominance_loss

        # Temporal smoothness (if sequence)
        if predictions.valence.dim() > 1 and predictions.valence.shape[1] > 1:
            # Compute first-order differences
            if self.mode == "VA":
                valence_diff = predictions.valence[:, 1:] - predictions.valence[:, :-1]
                arousal_diff = predictions.arousal[:, 1:] - predictions.arousal[:, :-1]
                smoothness_loss = (valence_diff ** 2).mean() + (arousal_diff ** 2).mean()
            else:
                pleasure_diff = predictions.pleasure[:, 1:] - predictions.pleasure[:, :-1]
                arousal_diff = predictions.arousal[:, 1:] - predictions.arousal[:, :-1]
                dominance_diff = predictions.dominance[:, 1:] - predictions.dominance[:, :-1]
                smoothness_loss = (
                    (pleasure_diff ** 2).mean() +
                    (arousal_diff ** 2).mean() +
                    (dominance_diff ** 2).mean()
                )

            losses["smoothness_loss"] = smoothness_loss.item()
            main_loss = main_loss + self.temporal_smoothness_weight * smoothness_loss

        # Total loss
        losses["total_loss"] = main_loss.item()

        return main_loss, losses
