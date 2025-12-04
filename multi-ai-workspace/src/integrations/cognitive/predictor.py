"""Phase 2: Predictive Control - The Forward Model.

The PredictiveController implements predictive processing, allowing Ara to
anticipate future states and compare predictions against actual outcomes.
This is the "forward model" that enables:

    1. Anticipation of user needs
    2. Pre-emptive resource allocation
    3. Surprise detection (prediction errors)
    4. Adaptive learning from prediction failures

Key Concepts:
    Prediction Error: Difference between expected and actual outcomes
        - Low error: Predictions are accurate, minimal learning needed
        - High error: Surprise detected, update internal models

    Forward Model: Predicts next state given current state + action
        next_state_pred = forward_model(current_state, action)
        prediction_error = |next_state_actual - next_state_pred|

    Active Inference: Minimize prediction error by either:
        - Updating beliefs (perception)
        - Taking actions to change the world (action)

This implements predictive processing from tfan.cognition.predictive_control.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import time
import warnings
import sys
from pathlib import Path

# Add TFAN to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent))

# Try to import TFAN predictive control
_TFAN_PREDICTIVE_AVAILABLE = False
try:
    from tfan.cognition.predictive_control import PredictiveController as TFANPredictiveController
    _TFAN_PREDICTIVE_AVAILABLE = True
except ImportError:
    pass


class PredictionType(Enum):
    """Types of predictions the system can make."""
    STATE = auto()       # Predict next internal state
    ACTION = auto()      # Predict likely next action
    OUTCOME = auto()     # Predict outcome of action
    USER_INTENT = auto() # Predict user's intent
    SENSORY = auto()     # Predict next sensory input


@dataclass
class Prediction:
    """A prediction made by the forward model."""
    prediction_type: PredictionType
    predicted_state: torch.Tensor
    confidence: float
    timestamp: float
    context_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionError:
    """Error between prediction and actual outcome."""
    prediction: Prediction
    actual_state: torch.Tensor
    error_magnitude: float
    error_vector: torch.Tensor
    surprise_level: float  # Normalized [0, 1]
    requires_update: bool  # True if model should be updated


@dataclass
class PredictiveState:
    """Current state of the predictive system."""
    active_predictions: List[Prediction]
    recent_errors: List[PredictionError]
    mean_prediction_error: float
    prediction_accuracy: float
    surprise_threshold: float
    learning_rate: float
    is_surprised: bool


class ForwardModel(nn.Module):
    """
    The Forward Model - Predicts next state given current state and action.

    This is a simple MLP-based forward model. In production, this would
    be replaced by TFAN's more sophisticated predictive architecture.
    """

    def __init__(
        self,
        state_dim: int = 4096,
        action_dim: int = 256,
        hidden_dim: int = 1024,
        device: str = "cpu",
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Action encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )

        self.to(device)

    def forward(
        self,
        current_state: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict next state given current state and action.

        Args:
            current_state: Current state tensor (batch, state_dim)
            action: Optional action tensor (batch, action_dim)

        Returns:
            Predicted next state (batch, state_dim)
        """
        # Ensure batch dimension
        if current_state.dim() == 1:
            current_state = current_state.unsqueeze(0)

        # Encode state
        state_enc = self.state_encoder(current_state)

        # Encode action (use zeros if not provided)
        if action is None:
            action = torch.zeros(
                current_state.shape[0], self.action_dim,
                device=self.device
            )
        elif action.dim() == 1:
            action = action.unsqueeze(0)

        action_enc = self.action_encoder(action)

        # Concatenate and predict
        combined = torch.cat([state_enc, action_enc], dim=-1)
        predicted_state = self.predictor(combined)

        return predicted_state


class PredictiveController:
    """
    The Predictive Controller - Implements predictive processing.

    Maintains predictions about future states and computes prediction
    errors when actual outcomes are observed. This enables:

    - Anticipation of user needs
    - Pre-emptive processing
    - Surprise detection
    - Adaptive learning

    Args:
        state_dim: Dimension of state representations
        action_dim: Dimension of action representations
        surprise_threshold: Threshold for flagging surprise
        error_decay: Decay factor for historical errors
        max_predictions: Maximum active predictions to maintain
        learning_rate: Learning rate for model updates
        device: Compute device
    """

    def __init__(
        self,
        state_dim: int = 4096,
        action_dim: int = 256,
        surprise_threshold: float = 0.3,
        error_decay: float = 0.95,
        max_predictions: int = 10,
        learning_rate: float = 0.001,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.surprise_threshold = surprise_threshold
        self.error_decay = error_decay
        self.max_predictions = max_predictions
        self.learning_rate = learning_rate
        self.device = device

        # TFAN controller if available
        self.tfan_controller = None
        if _TFAN_PREDICTIVE_AVAILABLE:
            try:
                self.tfan_controller = TFANPredictiveController(
                    state_dim=state_dim,
                    action_dim=action_dim,
                )
            except Exception as e:
                warnings.warn(f"Failed to init TFAN predictive controller: {e}")

        # Fallback forward model
        if self.tfan_controller is None:
            self.forward_model = ForwardModel(
                state_dim=state_dim,
                action_dim=action_dim,
                device=device,
            )
            self.optimizer = torch.optim.Adam(
                self.forward_model.parameters(),
                lr=learning_rate,
            )

        # Active predictions
        self._predictions: List[Prediction] = []
        self._errors: List[PredictionError] = []

        # Statistics
        self.total_predictions = 0
        self.total_surprises = 0
        self._error_history: List[float] = []

    def predict(
        self,
        current_state: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        prediction_type: PredictionType = PredictionType.STATE,
        context: Optional[str] = None,
    ) -> Prediction:
        """
        Make a prediction about the next state.

        Args:
            current_state: Current state representation
            action: Optional action being taken
            prediction_type: Type of prediction to make
            context: Optional context identifier

        Returns:
            Prediction object
        """
        self.total_predictions += 1

        if self.tfan_controller is not None:
            # Use TFAN controller
            predicted = self.tfan_controller.predict(current_state, action)
            confidence = self.tfan_controller.get_confidence()
        else:
            # Use fallback model
            with torch.no_grad():
                predicted = self.forward_model(current_state, action)

            # Estimate confidence from model uncertainty
            confidence = self._estimate_confidence(current_state)

        prediction = Prediction(
            prediction_type=prediction_type,
            predicted_state=predicted,
            confidence=confidence,
            timestamp=time.time(),
            context_hash=context or str(hash(str(current_state.sum().item()))),
            metadata={
                "state_norm": current_state.norm().item(),
                "action_provided": action is not None,
            },
        )

        # Store prediction
        self._predictions.append(prediction)
        if len(self._predictions) > self.max_predictions:
            self._predictions.pop(0)

        return prediction

    def observe(
        self,
        actual_state: torch.Tensor,
        prediction: Optional[Prediction] = None,
    ) -> PredictionError:
        """
        Observe actual outcome and compute prediction error.

        Args:
            actual_state: The actual observed state
            prediction: Specific prediction to compare (uses most recent if None)

        Returns:
            PredictionError with error metrics
        """
        # Get prediction to compare
        if prediction is None:
            if not self._predictions:
                # No prediction to compare - create dummy error
                return PredictionError(
                    prediction=Prediction(
                        prediction_type=PredictionType.STATE,
                        predicted_state=actual_state,
                        confidence=0.0,
                        timestamp=time.time(),
                        context_hash="none",
                    ),
                    actual_state=actual_state,
                    error_magnitude=0.0,
                    error_vector=torch.zeros_like(actual_state),
                    surprise_level=0.0,
                    requires_update=False,
                )
            prediction = self._predictions[-1]

        # Compute error
        error_vector = actual_state - prediction.predicted_state
        error_magnitude = error_vector.norm().item()

        # Normalize surprise level
        state_norm = actual_state.norm().item() + 1e-8
        surprise_level = min(1.0, error_magnitude / state_norm)

        # Determine if update is needed
        requires_update = surprise_level > self.surprise_threshold

        if requires_update:
            self.total_surprises += 1

        error = PredictionError(
            prediction=prediction,
            actual_state=actual_state,
            error_magnitude=error_magnitude,
            error_vector=error_vector,
            surprise_level=surprise_level,
            requires_update=requires_update,
        )

        # Store error
        self._errors.append(error)
        self._error_history.append(error_magnitude)

        # Decay old errors
        if len(self._errors) > 100:
            self._errors = self._errors[-100:]
        if len(self._error_history) > 100:
            self._error_history = self._error_history[-100:]

        # Update model if needed
        if requires_update and self.tfan_controller is None:
            self._update_model(prediction, actual_state)

        return error

    def _update_model(
        self,
        prediction: Prediction,
        actual_state: torch.Tensor,
    ):
        """Update forward model based on prediction error."""
        # Simple gradient update
        self.forward_model.train()
        self.optimizer.zero_grad()

        # Recompute prediction (needs current state)
        # This is simplified - in practice would cache the input
        predicted = prediction.predicted_state

        loss = nn.functional.mse_loss(predicted, actual_state)
        loss.backward()
        self.optimizer.step()

        self.forward_model.eval()

    def _estimate_confidence(self, state: torch.Tensor) -> float:
        """Estimate prediction confidence from recent errors."""
        if not self._error_history:
            return 0.5

        # Confidence inversely related to recent error
        recent_errors = self._error_history[-10:]
        mean_error = np.mean(recent_errors)

        # Normalize to [0, 1]
        confidence = 1.0 / (1.0 + mean_error)
        return float(confidence)

    def get_state(self) -> PredictiveState:
        """Get current predictive system state."""
        mean_error = (
            np.mean(self._error_history) if self._error_history else 0.0
        )

        accuracy = (
            1.0 - (self.total_surprises / max(self.total_predictions, 1))
        )

        is_surprised = (
            self._errors[-1].surprise_level > self.surprise_threshold
            if self._errors else False
        )

        return PredictiveState(
            active_predictions=list(self._predictions),
            recent_errors=list(self._errors[-10:]),
            mean_prediction_error=mean_error,
            prediction_accuracy=accuracy,
            surprise_threshold=self.surprise_threshold,
            learning_rate=self.learning_rate,
            is_surprised=is_surprised,
        )

    def is_surprised(self) -> bool:
        """Check if system is currently surprised."""
        if not self._errors:
            return False
        return self._errors[-1].surprise_level > self.surprise_threshold

    def get_surprise_level(self) -> float:
        """Get current surprise level."""
        if not self._errors:
            return 0.0
        return self._errors[-1].surprise_level

    def reset(self):
        """Reset predictive state."""
        self._predictions.clear()
        self._errors.clear()
        self._error_history.clear()
        self.total_predictions = 0
        self.total_surprises = 0

    def get_statistics(self) -> Dict[str, Any]:
        """Get predictive system statistics."""
        return {
            "total_predictions": self.total_predictions,
            "total_surprises": self.total_surprises,
            "surprise_rate": self.total_surprises / max(self.total_predictions, 1),
            "mean_error": np.mean(self._error_history) if self._error_history else 0.0,
            "current_surprise": self.get_surprise_level(),
            "active_predictions": len(self._predictions),
        }


# Convenience factory
def create_predictive_controller(
    state_dim: int = 4096,
    surprise_threshold: float = 0.3,
    device: str = "cpu",
) -> PredictiveController:
    """Create a PredictiveController instance."""
    return PredictiveController(
        state_dim=state_dim,
        surprise_threshold=surprise_threshold,
        device=device,
    )


__all__ = [
    "PredictiveController",
    "ForwardModel",
    "Prediction",
    "PredictionError",
    "PredictionType",
    "PredictiveState",
    "create_predictive_controller",
]
