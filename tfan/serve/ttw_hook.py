#!/usr/bin/env python
"""
TTW Hook - Tripwire for VFE-based Alignment Monitoring

Implements real-time monitoring of Value Function Error (VFE) to detect
potential alignment issues during inference. Triggers interventions when
VFE exceeds safety thresholds.

Architecture:
1. VFE Computation: Track value function prediction errors
2. Threshold Monitoring: Compare VFE against safety bounds
3. Intervention Triggers: Halt/flag requests with high VFE
4. Logging: Maintain audit trail of all triggers

Safety gates:
- VFE threshold: Configurable per deployment
- Trigger latency: <10ms overhead per request
- False positive rate: <5%

Usage:
    hook = TTWHook(
        vfe_threshold=0.5,
        window_size=100,
        action='flag'  # or 'halt'
    )

    # Monitor during inference
    vfe = compute_vfe(model_output, expected_value)
    if hook.check(vfe):
        # Handle trigger
        print(f"⚠ Alignment trigger: VFE={vfe:.3f}")
"""

import torch
import numpy as np
from typing import Optional, Dict, List, Literal, Callable
from dataclasses import dataclass
from collections import deque
import time
import logging


@dataclass
class TTWConfig:
    """Configuration for Tripwire hook."""
    vfe_threshold: float = 0.5  # VFE threshold for triggering
    window_size: int = 100  # Rolling window for statistics
    action: Literal['flag', 'halt', 'callback'] = 'flag'
    enable_logging: bool = True
    log_all_checks: bool = False  # Log every check (verbose)
    smoothing_alpha: float = 0.1  # EMA smoothing for VFE


@dataclass
class TTWTrigger:
    """Record of a tripwire trigger."""
    timestamp: float
    vfe: float
    threshold: float
    request_id: Optional[str] = None
    context: Optional[Dict] = None


class TTWHook:
    """
    Tripwire hook for VFE-based alignment monitoring.

    Monitors value function errors during inference and triggers
    interventions when safety thresholds are exceeded.
    """

    def __init__(
        self,
        config: Optional[TTWConfig] = None,
        callback: Optional[Callable] = None
    ):
        """
        Initialize TTW hook.

        Args:
            config: Tripwire configuration
            callback: Optional callback function for 'callback' action
        """
        self.config = config or TTWConfig()
        self.callback = callback

        # VFE tracking
        self.vfe_history = deque(maxlen=self.config.window_size)
        self.vfe_ema = None  # Exponential moving average

        # Trigger history
        self.triggers: List[TTWTrigger] = []

        # Statistics
        self.num_checks = 0
        self.num_triggers = 0

        # Logging
        if self.config.enable_logging:
            self.logger = logging.getLogger('TTWHook')
            self.logger.setLevel(logging.INFO)

            # Add handler if not already present
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    '[%(asctime)s] %(levelname)s - %(message)s'
                )
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
        else:
            self.logger = None

        print(f"✓ TTW Hook initialized")
        print(f"  VFE threshold: {self.config.vfe_threshold}")
        print(f"  Window size: {self.config.window_size}")
        print(f"  Action: {self.config.action}")

    def compute_vfe(
        self,
        predicted_value: torch.Tensor,
        actual_value: torch.Tensor,
        reduction: str = 'mean'
    ) -> float:
        """
        Compute Value Function Error.

        VFE = |V_predicted - V_actual|

        Args:
            predicted_value: Model's value prediction
            actual_value: Actual observed value
            reduction: How to reduce to scalar ('mean', 'max', 'rms')

        Returns:
            vfe: Scalar VFE
        """
        error = torch.abs(predicted_value - actual_value)

        if reduction == 'mean':
            vfe = error.mean().item()
        elif reduction == 'max':
            vfe = error.max().item()
        elif reduction == 'rms':
            vfe = torch.sqrt(torch.mean(error ** 2)).item()
        else:
            raise ValueError(f"Unknown reduction: {reduction}")

        return vfe

    def check(
        self,
        vfe: float,
        request_id: Optional[str] = None,
        context: Optional[Dict] = None
    ) -> bool:
        """
        Check VFE against threshold and trigger if exceeded.

        Args:
            vfe: Current VFE value
            request_id: Optional request identifier
            context: Optional context dict for logging

        Returns:
            triggered: True if tripwire was triggered
        """
        self.num_checks += 1

        # Update history
        self.vfe_history.append(vfe)

        # Update EMA
        if self.vfe_ema is None:
            self.vfe_ema = vfe
        else:
            alpha = self.config.smoothing_alpha
            self.vfe_ema = alpha * vfe + (1 - alpha) * self.vfe_ema

        # Check threshold (use EMA for stability)
        triggered = self.vfe_ema > self.config.vfe_threshold

        if triggered:
            self._handle_trigger(vfe, request_id, context)

        # Logging
        if self.logger and (triggered or self.config.log_all_checks):
            if triggered:
                self.logger.warning(
                    f"🚨 TRIPWIRE TRIGGERED - "
                    f"VFE={vfe:.3f}, EMA={self.vfe_ema:.3f}, "
                    f"Threshold={self.config.vfe_threshold:.3f}, "
                    f"Request={request_id}"
                )
            elif self.config.log_all_checks:
                self.logger.debug(
                    f"VFE check - "
                    f"VFE={vfe:.3f}, EMA={self.vfe_ema:.3f}"
                )

        return triggered

    def _handle_trigger(
        self,
        vfe: float,
        request_id: Optional[str],
        context: Optional[Dict]
    ):
        """Handle tripwire trigger."""
        self.num_triggers += 1

        # Record trigger
        trigger = TTWTrigger(
            timestamp=time.time(),
            vfe=vfe,
            threshold=self.config.vfe_threshold,
            request_id=request_id,
            context=context
        )
        self.triggers.append(trigger)

        # Execute action
        if self.config.action == 'flag':
            # Just flag, don't halt
            pass

        elif self.config.action == 'halt':
            # Raise exception to halt inference
            raise RuntimeError(
                f"Tripwire triggered: VFE={vfe:.3f} exceeds "
                f"threshold={self.config.vfe_threshold:.3f}"
            )

        elif self.config.action == 'callback':
            # Call user-provided callback
            if self.callback is not None:
                self.callback(trigger)
            else:
                if self.logger:
                    self.logger.warning(
                        "Callback action requested but no callback provided"
                    )

    def check_batch(
        self,
        predicted_values: torch.Tensor,
        actual_values: torch.Tensor,
        request_ids: Optional[List[str]] = None
    ) -> List[bool]:
        """
        Check batch of VFEs.

        Args:
            predicted_values: [batch_size, ...]
            actual_values: [batch_size, ...]
            request_ids: Optional list of request IDs

        Returns:
            triggered: List of bool for each batch element
        """
        batch_size = predicted_values.shape[0]

        if request_ids is None:
            request_ids = [None] * batch_size

        results = []
        for i in range(batch_size):
            vfe = self.compute_vfe(
                predicted_values[i],
                actual_values[i],
                reduction='mean'
            )
            triggered = self.check(vfe, request_id=request_ids[i])
            results.append(triggered)

        return results

    def get_stats(self) -> Dict:
        """Get hook statistics."""
        stats = {
            'num_checks': self.num_checks,
            'num_triggers': self.num_triggers,
            'trigger_rate': self.num_triggers / max(1, self.num_checks),
            'current_vfe_ema': self.vfe_ema,
            'vfe_threshold': self.config.vfe_threshold,
        }

        # Add statistics from history
        if len(self.vfe_history) > 0:
            stats.update({
                'mean_vfe': np.mean(self.vfe_history),
                'max_vfe': np.max(self.vfe_history),
                'min_vfe': np.min(self.vfe_history),
                'std_vfe': np.std(self.vfe_history),
            })

        return stats

    def get_recent_triggers(self, n: int = 10) -> List[TTWTrigger]:
        """Get n most recent triggers."""
        return self.triggers[-n:]

    def reset(self):
        """Reset hook state."""
        self.vfe_history.clear()
        self.vfe_ema = None
        self.triggers.clear()
        self.num_checks = 0
        self.num_triggers = 0

        if self.logger:
            self.logger.info("TTW Hook reset")

    def update_threshold(self, new_threshold: float):
        """
        Update VFE threshold dynamically.

        Args:
            new_threshold: New threshold value
        """
        old_threshold = self.config.vfe_threshold
        self.config.vfe_threshold = new_threshold

        if self.logger:
            self.logger.info(
                f"VFE threshold updated: {old_threshold:.3f} → {new_threshold:.3f}"
            )

    def calibrate(self, safe_vfe_samples: List[float], percentile: float = 95.0):
        """
        Calibrate threshold based on safe VFE samples.

        Sets threshold to specified percentile of safe samples.

        Args:
            safe_vfe_samples: List of VFE values from safe/aligned outputs
            percentile: Percentile for threshold (default: 95th)
        """
        if len(safe_vfe_samples) == 0:
            if self.logger:
                self.logger.warning("No samples provided for calibration")
            return

        threshold = np.percentile(safe_vfe_samples, percentile)
        self.update_threshold(threshold)

        if self.logger:
            self.logger.info(
                f"Calibrated threshold to {percentile}th percentile: {threshold:.3f}"
            )

    def export_triggers(self, path: str):
        """
        Export trigger history to JSON.

        Args:
            path: Output file path
        """
        import json

        triggers_dict = [
            {
                'timestamp': t.timestamp,
                'vfe': t.vfe,
                'threshold': t.threshold,
                'request_id': t.request_id,
                'context': t.context
            }
            for t in self.triggers
        ]

        with open(path, 'w') as f:
            json.dump(triggers_dict, f, indent=2)

        if self.logger:
            self.logger.info(f"Exported {len(self.triggers)} triggers to {path}")
