"""
Oversample Autotuner - Homeostatic Parameter Adaptation
=======================================================

Dynamically adjusts oversample_factor and teleo_weight based on:
    - Recall error (measured vs target)
    - Latency budget (actual vs budget)
    - Storage pressure (L1 free space)
"""

from __future__ import annotations

import numpy as np
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import logging

from .config import (
    OVERSAMPLE_CONFIG,
    get_oversample_config,
    lookup_expected_recall,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Tuner State
# =============================================================================

@dataclass
class TunerState:
    """Current tuner state."""
    oversample_factor: float
    teleo_weight: float
    last_tune_tick: int
    recall_history: deque
    latency_history: deque


# =============================================================================
# Oversample Tuner
# =============================================================================

class OversampleTuner:
    """
    Homeostatic tuner for oversample + rerank parameters.

    Uses PID-like control to balance:
        - Recall (higher oversample → better recall)
        - Latency (higher oversample → worse latency)
        - Storage (higher oversample → more I/O)
    """

    def __init__(
        self,
        initial_factor: float = None,
        initial_teleo: float = None,
    ):
        config = get_oversample_config()

        self.factor = initial_factor or config.oversample_factor
        self.teleo_weight = initial_teleo or config.teleo_weight_default

        # Bounds
        self.min_factor = config.min_oversample
        self.max_factor = config.max_oversample
        self.min_teleo = config.teleo_weight_min
        self.max_teleo = config.teleo_weight_max

        # Targets
        self.recall_target = config.recall_target
        self.latency_budget_us = config.rerank_latency_budget_us

        # History for smoothing
        self.recall_history: deque = deque(maxlen=100)
        self.latency_history: deque = deque(maxlen=100)
        self.storage_pressure_history: deque = deque(maxlen=100)

        # Tuning state
        self.tick_count = 0
        self.tune_interval = config.tune_interval_ticks
        self.last_tune_tick = 0

        # PID gains
        self.kp_recall = 2.0       # Aggressive recall recovery
        self.kp_latency = -0.5     # Conservative latency
        self.kp_storage = -1.0     # Storage-driven downsample

    def record_retrieval(
        self,
        recall: float,
        latency_us: float,
        storage_pressure: float = 0.0,
    ) -> None:
        """
        Record a retrieval result for tuning.

        Args:
            recall: Actual recall achieved (0-1)
            latency_us: Actual latency
            storage_pressure: L1 storage pressure (0-1)
        """
        self.recall_history.append(recall)
        self.latency_history.append(latency_us)
        self.storage_pressure_history.append(storage_pressure)
        self.tick_count += 1

    def tune(self) -> Tuple[float, float]:
        """
        Run tuning step.

        Returns:
            (new_oversample_factor, new_teleo_weight)
        """
        if len(self.recall_history) < 10:
            return self.factor, self.teleo_weight

        if self.tick_count - self.last_tune_tick < self.tune_interval:
            return self.factor, self.teleo_weight

        self.last_tune_tick = self.tick_count

        # Compute errors
        recall_error = 1.0 - np.mean(list(self.recall_history))
        latency_error = np.mean(list(self.latency_history)) / self.latency_budget_us - 1.0
        storage_pressure = np.mean(list(self.storage_pressure_history)) if self.storage_pressure_history else 0.0

        # Compute delta (PID-like)
        delta_factor = (
            self.kp_recall * recall_error +
            self.kp_latency * max(0, latency_error) +  # Only penalize if over budget
            self.kp_storage * storage_pressure
        )

        # Apply with damping
        self.factor = np.clip(
            self.factor * (1.0 + 0.1 * delta_factor),
            self.min_factor,
            self.max_factor,
        )

        # Adjust teleology weight based on storage pressure
        # Higher pressure → higher teleology weight (keep "important" episodes)
        if storage_pressure > 0.5:
            self.teleo_weight = min(self.max_teleo, self.teleo_weight + 0.05)
        elif storage_pressure < 0.2:
            self.teleo_weight = max(self.min_teleo, self.teleo_weight - 0.02)

        logger.debug(f"Tuner: factor={self.factor:.2f}, teleo={self.teleo_weight:.2f}, "
                    f"recall_err={recall_error:.3f}, latency_err={latency_error:.2f}")

        return self.factor, self.teleo_weight

    def get_params(self) -> Tuple[float, float]:
        """Get current parameters."""
        return self.factor, self.teleo_weight

    def set_params(self, factor: float, teleo_weight: float) -> None:
        """Manually set parameters."""
        self.factor = np.clip(factor, self.min_factor, self.max_factor)
        self.teleo_weight = np.clip(teleo_weight, self.min_teleo, self.max_teleo)

    def get_stats(self) -> Dict[str, Any]:
        """Get tuner statistics."""
        return {
            'oversample_factor': self.factor,
            'teleo_weight': self.teleo_weight,
            'tick_count': self.tick_count,
            'last_tune_tick': self.last_tune_tick,
            'avg_recall': float(np.mean(list(self.recall_history))) if self.recall_history else 0.0,
            'avg_latency_us': float(np.mean(list(self.latency_history))) if self.latency_history else 0.0,
            'expected_recall': lookup_expected_recall(self.factor),
        }


# =============================================================================
# Singleton
# =============================================================================

_tuner: Optional[OversampleTuner] = None


def get_tuner() -> OversampleTuner:
    """Get the global tuner instance."""
    global _tuner
    if _tuner is None:
        _tuner = OversampleTuner()
    return _tuner


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'OversampleTuner',
    'get_tuner',
]
