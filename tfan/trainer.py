"""
TF-A-N Trainer with FDT (Fluctuation-Dissipation Theorem) Homeostat.

Integrates all components:
- FDT-based stability control (PI-D controller)
- Emotion-based neuromodulation
- PGU formal safety
- Topology regularization
- Sparse attention
- Multi-modal fusion

Hard gate:
- EPR-CV ≤ 0.15 sustained
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import time
import warnings

from .config import TFANConfig
from .pgu import ProofGatedUpdater
from .topo import TopologyRegularizer
from .emotion.head import EmotionHead, EmotionPrediction
from .emotion.controller import EmotionController, ControlModulation


@dataclass
class FDTMetrics:
    """FDT homeostat metrics."""
    epr: float  # Energy-Perturbation Ratio
    epr_cv: float  # Coefficient of Variation of EPR
    grad_variance: float
    current_lr: float
    current_temperature: float
    lr_multiplier: float
    temp_multiplier: float


class FDTHomeostat:
    """
    Fluctuation-Dissipation Theorem based homeostat.

    Maintains training stability by coupling fluctuation (grad variance)
    to dissipation (learning rate and temperature).

    Uses PI-D controller:
    - Proportional: kp * error
    - Integral: ki * ∫error
    - Derivative: kd * Δerror
    """

    def __init__(
        self,
        kp: float = 0.30,
        ki: float = 0.02,
        kd: float = 0.10,
        ema_alpha: float = 0.95,
        epr_cv_max: float = 0.15,
        temperature_min: float = 0.5,
        temperature_max: float = 2.0,
        lr_min: float = 1e-6,
        lr_max: float = 1e-2,
    ):
        """
        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            ema_alpha: EMA smoothing factor
            epr_cv_max: Maximum allowed EPR-CV
            temperature_min: Minimum temperature
            temperature_max: Maximum temperature
            lr_min: Minimum learning rate
            lr_max: Maximum learning rate
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.ema_alpha = ema_alpha
        self.epr_cv_max = epr_cv_max
        self.temperature_min = temperature_min
        self.temperature_max = temperature_max
        self.lr_min = lr_min
        self.lr_max = lr_max

        # State
        self.epr_history: List[float] = []
        self.epr_ema: Optional[float] = None
        self.prev_error: float = 0.0
        self.integral: float = 0.0

        # Modulation outputs
        self.lr_multiplier: float = 1.0
        self.temp_multiplier: float = 1.0

    def compute_epr(
        self,
        loss: float,
        grad_variance: float,
        eps: float = 1e-8,
    ) -> float:
        """
        Compute Energy-Perturbation Ratio.

        EPR = loss / (grad_variance + eps)

        Args:
            loss: Current loss value
            grad_variance: Gradient variance
            eps: Numerical stability

        Returns:
            EPR value
        """
        return loss / (grad_variance + eps)

    def compute_epr_cv(self) -> float:
        """
        Compute Coefficient of Variation of EPR.

        CV = std(EPR) / mean(EPR)

        Returns:
            EPR-CV
        """
        if len(self.epr_history) < 2:
            return 0.0

        epr_array = np.array(self.epr_history)
        mean_epr = np.mean(epr_array)
        std_epr = np.std(epr_array)

        return std_epr / (mean_epr + 1e-8)

    def step(
        self,
        loss: float,
        grad_variance: float,
    ) -> FDTMetrics:
        """
        FDT homeostat step.

        Args:
            loss: Current loss
            grad_variance: Gradient variance

        Returns:
            FDTMetrics with current state
        """
        # Compute EPR
        epr = self.compute_epr(loss, grad_variance)
        self.epr_history.append(epr)

        # EMA of EPR
        if self.epr_ema is None:
            self.epr_ema = epr
        else:
            self.epr_ema = self.ema_alpha * self.epr_ema + (1 - self.ema_alpha) * epr

        # Compute CV
        epr_cv = self.compute_epr_cv()

        # PI-D control
        # Target: keep EPR-CV below threshold
        error = epr_cv - self.epr_cv_max

        # Proportional
        p_term = self.kp * error

        # Integral (with anti-windup)
        self.integral += error
        self.integral = np.clip(self.integral, -10.0, 10.0)  # Anti-windup
        i_term = self.ki * self.integral

        # Derivative
        d_term = self.kd * (error - self.prev_error)
        self.prev_error = error

        # Combined control signal
        control_signal = p_term + i_term + d_term

        # Map control signal to modulations
        # If EPR-CV is high (unstable), reduce LR and increase T
        if error > 0:  # CV above threshold
            # Cool down: reduce LR, increase T
            self.lr_multiplier = max(0.8, 1.0 - 0.2 * abs(control_signal))
            self.temp_multiplier = min(1.5, 1.0 + 0.3 * abs(control_signal))
        else:  # CV below threshold
            # Warm up: can increase LR, decrease T
            self.lr_multiplier = min(1.2, 1.0 + 0.1 * abs(control_signal))
            self.temp_multiplier = max(0.9, 1.0 - 0.1 * abs(control_signal))

        # Clamp to bounds
        self.lr_multiplier = np.clip(self.lr_multiplier, 0.5, 1.5)
        self.temp_multiplier = np.clip(self.temp_multiplier, 0.8, 1.5)

        metrics = FDTMetrics(
            epr=epr,
            epr_cv=epr_cv,
            grad_variance=grad_variance,
            current_lr=0.0,  # Will be filled by trainer
            current_temperature=0.0,  # Will be filled by trainer
            lr_multiplier=self.lr_multiplier,
            temp_multiplier=self.temp_multiplier,
        )

        return metrics

    def should_pause(self, epr_cv: float, threshold_mult: float = 2.0) -> bool:
        """
        Check if training should pause due to instability.

        Args:
            epr_cv: Current EPR-CV
            threshold_mult: Multiplier for threshold

        Returns:
            True if should pause
        """
        return epr_cv > (self.epr_cv_max * threshold_mult)


class TFANTrainer:
    """
    Main TF-A-N trainer.

    Orchestrates all components with FDT stability control.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TFANConfig,
        optimizer: Optional[optim.Optimizer] = None,
        pgu: Optional[ProofGatedUpdater] = None,
        emotion_head: Optional[EmotionHead] = None,
        emotion_controller: Optional[EmotionController] = None,
        topo_regularizer: Optional[TopologyRegularizer] = None,
    ):
        """
        Args:
            model: Model to train
            config: TF-A-N configuration
            optimizer: Optimizer (created if None)
            pgu: Proof-gated updater
            emotion_head: Emotion prediction head
            emotion_controller: Emotion-based controller
            topo_regularizer: Topology regularizer
        """
        self.model = model
        self.config = config
        self.device = torch.device(config.device)

        # Optimizer
        if optimizer is None:
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=config.base_lr,
                weight_decay=0.01,
            )
        else:
            self.optimizer = optimizer

        # Components
        self.pgu = pgu or ProofGatedUpdater(
            timeout_ms=config.pgu.timeout_ms,
            cache_size=config.pgu.cache_size,
            mode=config.pgu.mode,
        )

        self.emotion_head = emotion_head or EmotionHead(
            d_model=config.d_model,
            mode=config.emotion.mode,
            loss_type=config.emotion.loss_type,
        ).to(self.device)

        self.emotion_controller = emotion_controller or EmotionController(
            arousal_temp_coupling=config.emotion.arousal_temp_coupling,
            valence_lr_coupling=config.emotion.valence_lr_coupling,
            controller_weight=config.emotion.controller_weight,
        )

        self.topo_regularizer = topo_regularizer or TopologyRegularizer(
            lambda_topo=config.topology.lambda_topo,
            homology_degrees=config.topology.homology_degrees,
            device=config.device,
        ).to(self.device)

        # FDT homeostat
        self.fdt = FDTHomeostat(
            kp=config.fdt.kp,
            ki=config.fdt.ki,
            kd=config.fdt.kd,
            ema_alpha=config.fdt.ema_alpha,
            epr_cv_max=config.fdt.epr_cv_max,
            temperature_min=config.fdt.temperature_min,
            temperature_max=config.fdt.temperature_max,
            lr_min=config.fdt.lr_min,
            lr_max=config.fdt.lr_max,
        )

        # State
        self.current_step = 0
        self.current_epoch = 0
        self.base_lr = config.base_lr
        self.base_temperature = config.base_temperature
        self.current_lr = self.base_lr
        self.current_temperature = self.base_temperature

        # Metrics history
        self.metrics_history: List[Dict] = []

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        emotion_targets: Optional[EmotionPrediction] = None,
    ) -> Dict[str, float]:
        """
        Single training step.

        Args:
            batch: Input batch
            emotion_targets: Optional emotion labels for emotion head training

        Returns:
            Dictionary of metrics
        """
        self.model.train()
        start_time = time.time()

        # Forward pass
        outputs = self.model(batch)
        loss = outputs.get("loss", torch.tensor(0.0))

        # Get latent representations for emotion and topology
        latents = outputs.get("latents", None)

        # Emotion prediction
        emotion_loss = torch.tensor(0.0)
        emotion_pred = None
        if latents is not None and self.emotion_head is not None:
            emotion_pred = self.emotion_head(latents)

            if emotion_targets is not None:
                emotion_loss, emotion_metrics = self.emotion_head.compute_loss(
                    emotion_pred, emotion_targets
                )
                loss = loss + emotion_loss

        # Topology regularization
        topo_loss = torch.tensor(0.0)
        if latents is not None and self.topo_regularizer is not None:
            topo_loss, topo_metrics = self.topo_regularizer(latents)
            loss = loss + topo_loss

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Compute gradient variance for FDT
        grad_variance = self._compute_grad_variance()

        # FDT step
        fdt_metrics = self.fdt.step(loss.item(), grad_variance)

        # Emotion-based modulation
        modulation = ControlModulation()
        if emotion_pred is not None:
            fdt_metrics_dict = {
                "epr": fdt_metrics.epr,
                "epr_cv": fdt_metrics.epr_cv,
            }
            modulation = self.emotion_controller.modulate_policy(
                fdt_metrics_dict, emotion_pred, self.base_lr, self.base_temperature
            )

        # Combine FDT and emotion modulations
        final_lr_mult = fdt_metrics.lr_multiplier * modulation.lr_multiplier
        final_temp_mult = fdt_metrics.temp_multiplier * modulation.temperature_multiplier

        # Apply modulations
        self.current_lr = self.base_lr * final_lr_mult
        self.current_temperature = self.base_temperature * final_temp_mult

        # Clamp
        self.current_lr = np.clip(self.current_lr, self.fdt.lr_min, self.fdt.lr_max)
        self.current_temperature = np.clip(
            self.current_temperature, self.fdt.temperature_min, self.fdt.temperature_max
        )

        # Update optimizer LR
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.current_lr

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.fdt.grad_clip_norm
        )

        # PGU verification
        update_payload = {
            "param_name": "model",
            "step": self.current_step,
            "loss": loss.item(),
            "grad_norm": torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), float('inf')
            ).item(),
            "metadata": {"epr_cv": fdt_metrics.epr_cv},
        }

        pgu_result = self.pgu.verify_update(update_payload)

        # Optimizer step (if PGU allows)
        if pgu_result.proven:
            self.optimizer.step()
        else:
            warnings.warn(f"PGU blocked update: {pgu_result.rule_violations}")

        self.current_step += 1

        # Collect metrics
        elapsed_time = time.time() - start_time

        metrics = {
            "loss": loss.item(),
            "emotion_loss": emotion_loss.item() if isinstance(emotion_loss, torch.Tensor) else 0.0,
            "topo_loss": topo_loss.item() if isinstance(topo_loss, torch.Tensor) else 0.0,
            "epr": fdt_metrics.epr,
            "epr_cv": fdt_metrics.epr_cv,
            "grad_variance": grad_variance,
            "lr": self.current_lr,
            "temperature": self.current_temperature,
            "lr_mult_fdt": fdt_metrics.lr_multiplier,
            "temp_mult_fdt": fdt_metrics.temp_multiplier,
            "lr_mult_emotion": modulation.lr_multiplier,
            "temp_mult_emotion": modulation.temperature_multiplier,
            "pgu_proven": pgu_result.proven,
            "pgu_latency_ms": pgu_result.latency_ms,
            "step_time": elapsed_time,
        }

        self.metrics_history.append(metrics)

        return metrics

    def _compute_grad_variance(self) -> float:
        """
        Compute gradient variance across parameters.

        Returns:
            Gradient variance
        """
        grad_norms = []
        for param in self.model.parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())

        if len(grad_norms) == 0:
            return 0.0

        return float(np.var(grad_norms))

    def validate_gates(self) -> Tuple[bool, Dict[str, any]]:
        """
        Validate all hard gates.

        Returns:
            (passes_all_gates, gate_results)
        """
        results = {}

        # EPR-CV gate
        current_epr_cv = self.fdt.compute_epr_cv()
        results["epr_cv"] = {
            "value": current_epr_cv,
            "threshold": self.config.fdt.epr_cv_max,
            "passes": current_epr_cv <= self.config.fdt.epr_cv_max,
        }

        # PGU gates
        pgu_passes, pgu_metrics = self.pgu.validate_gates()
        results["pgu"] = {
            "passes": pgu_passes,
            **pgu_metrics,
        }

        # Overall
        passes_all = results["epr_cv"]["passes"] and results["pgu"]["passes"]

        return passes_all, results

    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        checkpoint = {
            "step": self.current_step,
            "epoch": self.current_epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "fdt_state": {
                "epr_history": self.fdt.epr_history,
                "epr_ema": self.fdt.epr_ema,
                "integral": self.fdt.integral,
            },
            "current_lr": self.current_lr,
            "current_temperature": self.current_temperature,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.current_step = checkpoint["step"]
        self.current_epoch = checkpoint["epoch"]
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])

        # Restore FDT state
        fdt_state = checkpoint.get("fdt_state", {})
        self.fdt.epr_history = fdt_state.get("epr_history", [])
        self.fdt.epr_ema = fdt_state.get("epr_ema", None)
        self.fdt.integral = fdt_state.get("integral", 0.0)

        self.current_lr = checkpoint.get("current_lr", self.base_lr)
        self.current_temperature = checkpoint.get("current_temperature", self.base_temperature)
