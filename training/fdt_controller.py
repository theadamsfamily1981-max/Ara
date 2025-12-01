"""
FDT (Fluctuation-Dissipation Theorem) PI-D Controller for TF-A-N training.

Maintains EPR-CV ≤ 0.15 through adaptive LR/temperature modulation.
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple
from collections import deque


class FDTController:
    """
    FDT PI-D controller for homeostatic training stability.

    Monitors EPR (Effective Parameter Ratio) = loss / grad_variance
    and adjusts learning rate and temperature to maintain EPR-CV ≤ 0.15.

    Args:
        kp: Proportional gain (default: 0.30)
        ki: Integral gain (default: 0.02)
        kd: Derivative gain (default: 0.10)
        ema_alpha: EMA smoothing for EPR (default: 0.85)
        target_epr_cv: Target EPR coefficient of variation (default: 0.15)
        history_size: Size of EPR history for CV computation (default: 100)
        lr_range: (min_mult, max_mult) for LR modulation (default: (0.7, 1.2))
        temp_range: (min_mult, max_mult) for temperature modulation (default: (0.8, 1.3))
    """

    def __init__(
        self,
        kp: float = 0.30,
        ki: float = 0.02,
        kd: float = 0.10,
        ema_alpha: float = 0.85,
        target_epr_cv: float = 0.15,
        history_size: int = 100,
        lr_range: Tuple[float, float] = (0.7, 1.2),
        temp_range: Tuple[float, float] = (0.8, 1.3),
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.ema_alpha = ema_alpha
        self.target_epr_cv = target_epr_cv
        self.history_size = history_size
        self.lr_range = lr_range
        self.temp_range = temp_range

        # State
        self.epr_history = deque(maxlen=history_size)
        self.epr_ema = None
        self.error_integral = 0.0
        self.prev_error = 0.0

        # Metrics
        self.current_epr = 0.0
        self.current_epr_cv = 0.0
        self.lr_multiplier = 1.0
        self.temp_multiplier = 1.0

    def compute_epr(
        self,
        loss: float,
        grad_variance: float,
        eps: float = 1e-8,
    ) -> float:
        """
        Compute Effective Parameter Ratio.

        EPR = loss / (grad_variance + eps)

        Args:
            loss: Current loss value
            grad_variance: Variance of gradients
            eps: Epsilon for numerical stability

        Returns:
            epr: Effective Parameter Ratio
        """
        return loss / (grad_variance + eps)

    def compute_epr_cv(self) -> float:
        """
        Compute coefficient of variation of EPR history.

        CV = std(EPR) / mean(EPR)

        Returns:
            epr_cv: Coefficient of variation
        """
        if len(self.epr_history) < 10:
            return 0.0

        epr_array = np.array(self.epr_history)
        mean_epr = np.mean(epr_array)
        std_epr = np.std(epr_array)

        if mean_epr < 1e-8:
            return 0.0

        return std_epr / mean_epr

    def step(
        self,
        loss: float,
        grad_variance: float,
        base_lr: float = 1e-4,
        base_temp: float = 1.0,
    ) -> Dict[str, float]:
        """
        Perform one step of FDT control.

        Updates EPR history, computes control signal, and returns modulated LR/temperature.

        Args:
            loss: Current loss value
            grad_variance: Gradient variance
            base_lr: Base learning rate
            base_temp: Base temperature

        Returns:
            dict with:
                - lr: Modulated learning rate
                - temperature: Modulated temperature
                - epr: Current EPR
                - epr_cv: Current EPR-CV
                - control_signal: PI-D control output
        """
        # Compute EPR
        epr = self.compute_epr(loss, grad_variance)
        self.current_epr = epr

        # Update EMA
        if self.epr_ema is None:
            self.epr_ema = epr
        else:
            self.epr_ema = self.ema_alpha * self.epr_ema + (1 - self.ema_alpha) * epr

        # Add to history (use EMA for stability)
        self.epr_history.append(self.epr_ema)

        # Compute EPR-CV
        epr_cv = self.compute_epr_cv()
        self.current_epr_cv = epr_cv

        # PI-D control
        error = epr_cv - self.target_epr_cv

        # Proportional term
        p_term = self.kp * error

        # Integral term
        self.error_integral += error
        i_term = self.ki * self.error_integral

        # Derivative term
        d_term = self.kd * (error - self.prev_error)
        self.prev_error = error

        # Control signal
        control_signal = p_term + i_term + d_term

        # Map control signal to LR and temperature multipliers
        # If EPR-CV is too high (unstable), reduce LR and increase temperature (more exploration)
        # If EPR-CV is too low (too stable), increase LR and reduce temperature (more exploitation)

        # LR multiplier: inverse relationship with control signal
        # High EPR-CV -> negative control signal -> reduce LR
        self.lr_multiplier = np.clip(
            1.0 - 0.5 * control_signal, self.lr_range[0], self.lr_range[1]
        )

        # Temperature multiplier: direct relationship with control signal
        # High EPR-CV -> increase temperature for exploration
        self.temp_multiplier = np.clip(
            1.0 + 0.3 * control_signal, self.temp_range[0], self.temp_range[1]
        )

        # Apply multipliers
        modulated_lr = base_lr * self.lr_multiplier
        modulated_temp = base_temp * self.temp_multiplier

        return {
            "lr": modulated_lr,
            "temperature": modulated_temp,
            "epr": epr,
            "epr_cv": epr_cv,
            "epr_ema": self.epr_ema,
            "control_signal": control_signal,
            "lr_multiplier": self.lr_multiplier,
            "temp_multiplier": self.temp_multiplier,
        }

    def get_metrics(self) -> Dict[str, float]:
        """
        Get current FDT metrics.

        Returns:
            dict with current metrics
        """
        return {
            "epr": self.current_epr,
            "epr_cv": self.current_epr_cv,
            "epr_ema": self.epr_ema if self.epr_ema is not None else 0.0,
            "lr_multiplier": self.lr_multiplier,
            "temp_multiplier": self.temp_multiplier,
            "error_integral": self.error_integral,
        }

    def reset(self):
        """Reset controller state."""
        self.epr_history.clear()
        self.epr_ema = None
        self.error_integral = 0.0
        self.prev_error = 0.0
        self.current_epr = 0.0
        self.current_epr_cv = 0.0
        self.lr_multiplier = 1.0
        self.temp_multiplier = 1.0


class FDTControllerWithEmotion:
    """
    FDT controller with emotion-based modulation.

    Integrates emotion signals (valence, arousal) for additional policy modulation.

    Args:
        fdt_controller: Base FDT controller
        emotion_weight: Weight for emotion modulation (default: 0.3)
        arousal_to_temp: Map arousal to temperature (default: True)
        valence_to_lr: Map valence to LR (default: True)
    """

    def __init__(
        self,
        fdt_controller: Optional[FDTController] = None,
        emotion_weight: float = 0.3,
        arousal_to_temp: bool = True,
        valence_to_lr: bool = True,
    ):
        self.fdt = fdt_controller or FDTController()
        self.emotion_weight = emotion_weight
        self.arousal_to_temp = arousal_to_temp
        self.valence_to_lr = valence_to_lr

    def step(
        self,
        loss: float,
        grad_variance: float,
        base_lr: float = 1e-4,
        base_temp: float = 1.0,
        emotion: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Step with emotion modulation.

        Args:
            loss: Current loss
            grad_variance: Gradient variance
            base_lr: Base learning rate
            base_temp: Base temperature
            emotion: Optional dict with valence, arousal

        Returns:
            dict with modulated LR, temperature, metrics
        """
        # Base FDT step
        fdt_outputs = self.fdt.step(loss, grad_variance, base_lr, base_temp)

        # Apply emotion modulation if provided
        if emotion is not None:
            valence = emotion.get("valence", 0.0)  # [-1, 1]
            arousal = emotion.get("arousal", 0.5)  # [0, 1]

            # Arousal -> Temperature (high arousal = high exploration)
            if self.arousal_to_temp:
                emotion_temp_mult = 0.8 + 0.5 * arousal  # [0.8, 1.3]
                fdt_outputs["temperature"] = (
                    (1 - self.emotion_weight) * fdt_outputs["temperature"]
                    + self.emotion_weight * base_temp * emotion_temp_mult
                )

            # Valence -> LR (positive valence = more confident = higher LR)
            if self.valence_to_lr:
                valence_norm = (valence + 1) / 2  # Map [-1, 1] -> [0, 1]
                emotion_lr_mult = 0.7 + 0.5 * valence_norm  # [0.7, 1.2]
                fdt_outputs["lr"] = (
                    (1 - self.emotion_weight) * fdt_outputs["lr"]
                    + self.emotion_weight * base_lr * emotion_lr_mult
                )

            # Add emotion info to outputs
            fdt_outputs["emotion_valence"] = valence
            fdt_outputs["emotion_arousal"] = arousal

        return fdt_outputs

    def get_metrics(self) -> Dict[str, float]:
        """Get FDT metrics."""
        return self.fdt.get_metrics()

    def reset(self):
        """Reset controller."""
        self.fdt.reset()


__all__ = [
    "FDTController",
    "FDTControllerWithEmotion",
]
