"""
Emotion prediction head for TF-A-N 7B.

Predicts valence, arousal, and confidence from latent states.
Integrates with FDT controller for emotion-modulated training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class EmotionHead(nn.Module):
    """
    Emotion prediction head.

    Predicts:
    - Valence: [-1, 1] (negative to positive emotion)
    - Arousal: [0, 1] (calm to excited)
    - Confidence: [0, 1] (prediction confidence)

    Uses final latent state (or pooled representation) for prediction.

    Args:
        hidden_size: Input hidden dimension
        hidden_dim: Intermediate MLP dimension (default: 512)
        dropout: Dropout probability (default: 0.1)
        pooling: Pooling method ("mean", "last", "max")
    """

    def __init__(
        self,
        hidden_size: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        pooling: str = "mean",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.hidden_dim = hidden_dim
        self.pooling = pooling

        assert pooling in ["mean", "last", "max"], f"Invalid pooling: {pooling}"

        # Shared MLP backbone
        self.backbone = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Prediction heads
        self.valence_head = nn.Linear(hidden_dim, 1)  # Output: tanh -> [-1, 1]
        self.arousal_head = nn.Linear(hidden_dim, 1)  # Output: sigmoid -> [0, 1]
        self.confidence_head = nn.Linear(hidden_dim, 1)  # Output: sigmoid -> [0, 1]

    def pool_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Pool sequence of latents to single vector.

        Args:
            latents: [batch, seq_len, hidden_size]

        Returns:
            pooled: [batch, hidden_size]
        """
        if self.pooling == "mean":
            return latents.mean(dim=1)
        elif self.pooling == "last":
            return latents[:, -1, :]
        elif self.pooling == "max":
            return latents.max(dim=1)[0]

    def forward(
        self,
        latents: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict emotion from latent states.

        Args:
            latents: [batch, seq_len, hidden_size]
            attention_mask: Optional mask for pooling [batch, seq_len]

        Returns:
            dict with:
                - valence: [batch, 1] in [-1, 1]
                - arousal: [batch, 1] in [0, 1]
                - confidence: [batch, 1] in [0, 1]
        """
        # Apply attention mask if provided (for mean pooling)
        if attention_mask is not None and self.pooling == "mean":
            # Mask out padding tokens
            mask_expanded = attention_mask.unsqueeze(-1)  # [batch, seq_len, 1]
            latents_masked = latents * mask_expanded
            pooled = latents_masked.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            pooled = self.pool_latents(latents)

        # Shared backbone
        features = self.backbone(pooled)  # [batch, hidden_dim]

        # Predict emotion components
        valence = torch.tanh(self.valence_head(features))  # [-1, 1]
        arousal = torch.sigmoid(self.arousal_head(features))  # [0, 1]
        confidence = torch.sigmoid(self.confidence_head(features))  # [0, 1]

        return {
            "valence": valence,
            "arousal": arousal,
            "confidence": confidence,
        }

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        loss_type: str = "mse",
        temporal_smoothness: float = 0.0,
    ) -> torch.Tensor:
        """
        Compute emotion prediction loss.

        Args:
            predictions: dict with valence, arousal, confidence
            targets: dict with target values
            loss_type: "mse" or "ccc" (concordance correlation coefficient)
            temporal_smoothness: Weight for temporal smoothness regularization

        Returns:
            loss: Total emotion loss
        """
        if loss_type == "mse":
            # MSE loss for each component
            valence_loss = F.mse_loss(predictions["valence"], targets["valence"])
            arousal_loss = F.mse_loss(predictions["arousal"], targets["arousal"])

            # Optional confidence weighting
            if "confidence" in targets:
                confidence_loss = F.mse_loss(
                    predictions["confidence"], targets["confidence"]
                )
                total_loss = valence_loss + arousal_loss + 0.5 * confidence_loss
            else:
                total_loss = valence_loss + arousal_loss

        elif loss_type == "ccc":
            # Concordance Correlation Coefficient loss (better for emotion)
            valence_loss = self._ccc_loss(predictions["valence"], targets["valence"])
            arousal_loss = self._ccc_loss(predictions["arousal"], targets["arousal"])
            total_loss = valence_loss + arousal_loss

        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        # Temporal smoothness regularization
        if temporal_smoothness > 0.0:
            # Penalize rapid changes in predictions (requires sequential data)
            # This would need to be computed across a batch of sequences
            # For now, skip in this simplified version
            pass

        return total_loss

    def _ccc_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Concordance Correlation Coefficient loss.

        CCC measures agreement between predictions and targets.
        Loss = 1 - CCC, where CCC in [-1, 1], so loss in [0, 2].

        Args:
            pred: Predictions [batch, 1]
            target: Targets [batch, 1]

        Returns:
            loss: 1 - CCC
        """
        pred_mean = pred.mean()
        target_mean = target.mean()

        pred_var = pred.var()
        target_var = target.var()

        covariance = ((pred - pred_mean) * (target - target_mean)).mean()

        ccc = (2 * covariance) / (pred_var + target_var + (pred_mean - target_mean) ** 2 + 1e-8)

        return 1.0 - ccc


class EmotionHeadWithHistory(nn.Module):
    """
    Emotion head with temporal history tracking.

    Maintains a history of emotion states for smoothness and FDT coupling.

    Args:
        hidden_size: Input hidden dimension
        hidden_dim: MLP dimension
        history_size: Number of past states to track
        dropout: Dropout probability
        pooling: Pooling method
    """

    def __init__(
        self,
        hidden_size: int,
        hidden_dim: int = 512,
        history_size: int = 10,
        dropout: float = 0.1,
        pooling: str = "mean",
    ):
        super().__init__()
        self.history_size = history_size

        # Base emotion head
        self.emotion_head = EmotionHead(
            hidden_size=hidden_size,
            hidden_dim=hidden_dim,
            dropout=dropout,
            pooling=pooling,
        )

        # History buffers
        self.register_buffer(
            "valence_history", torch.zeros(history_size)
        )
        self.register_buffer(
            "arousal_history", torch.zeros(history_size)
        )
        self.history_idx = 0

    def forward(
        self,
        latents: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        update_history: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward with history tracking.

        Args:
            latents: [batch, seq_len, hidden_size]
            attention_mask: Optional mask
            update_history: Whether to update history buffers

        Returns:
            dict with emotion predictions + smoothed values
        """
        # Get raw predictions
        predictions = self.emotion_head(latents, attention_mask)

        # Update history
        if update_history and self.training:
            self.valence_history[self.history_idx] = predictions["valence"].mean().item()
            self.arousal_history[self.history_idx] = predictions["arousal"].mean().item()
            self.history_idx = (self.history_idx + 1) % self.history_size

        # Add smoothed predictions (EMA of history)
        predictions["valence_smoothed"] = self.valence_history.mean()
        predictions["arousal_smoothed"] = self.arousal_history.mean()

        return predictions

    def get_emotion_state(self) -> Dict[str, float]:
        """
        Get current smoothed emotion state.

        Returns:
            dict with valence_smoothed, arousal_smoothed
        """
        return {
            "valence_smoothed": self.valence_history.mean().item(),
            "arousal_smoothed": self.arousal_history.mean().item(),
        }


def create_emotion_head(
    hidden_size: int,
    use_history: bool = False,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create emotion head.

    Args:
        hidden_size: Hidden dimension
        use_history: Whether to use history tracking
        **kwargs: Additional arguments for EmotionHead

    Returns:
        Emotion head module
    """
    if use_history:
        return EmotionHeadWithHistory(hidden_size=hidden_size, **kwargs)
    else:
        return EmotionHead(hidden_size=hidden_size, **kwargs)


__all__ = [
    "EmotionHead",
    "EmotionHeadWithHistory",
    "create_emotion_head",
]
