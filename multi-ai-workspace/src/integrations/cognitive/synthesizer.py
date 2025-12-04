"""Phase 3: Conscience - Cognitive Synthesis & Self-Preservation.

The Conscience monitors the system's cognitive stability and can refuse
to process requests when destabilized. This is the "self-preservation"
mechanism that allows Ara to say "I am too unstable to answer."

Key Metrics:
    Structural Rate (Ṡ): Rate of change in neural topology
        - Low Ṡ: Stable, normal operation
        - High Ṡ: Destabilized, may need protection

    Alert Level: System stress indicator
        - GREEN: Normal operation
        - YELLOW: Elevated stress, caution
        - RED: Critical, enter protective mode

System Modes:
    NORMAL: Full cognitive processing available
    CAUTIOUS: Reduced complexity, simpler responses
    PROTECTIVE: Refuse new requests, stabilize
    RECOVERY: Gradually returning to normal

This implements the L7 "Cognitive Synthesis" layer from TFAN.
"""

import torch
import numpy as np
from enum import Enum, auto
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import time
import warnings
import sys
from pathlib import Path

# Add TFAN to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent))

# Try to import TFAN synthesis module
_TFAN_SYNTHESIS_AVAILABLE = False
try:
    from tfan.synthesis import CognitiveSynthesizer as TFANCognitiveSynthesizer
    _TFAN_SYNTHESIS_AVAILABLE = True
except ImportError:
    pass


class SystemMode(Enum):
    """System operational modes."""
    NORMAL = auto()      # Full cognitive processing
    CAUTIOUS = auto()    # Reduced complexity
    PROTECTIVE = auto()  # Refuse requests, stabilize
    RECOVERY = auto()    # Transitioning back to normal


class AlertLevel(Enum):
    """System alert levels."""
    GREEN = auto()   # Normal
    YELLOW = auto()  # Elevated stress
    RED = auto()     # Critical


@dataclass
class StabilityStatus:
    """Current stability status of the cognitive system."""
    is_stable: bool
    mode: SystemMode
    alert_level: AlertLevel
    structural_rate: float  # Ṡ metric
    message: str
    can_process: bool
    recommended_action: str
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class L7Metrics:
    """Layer 7 (Cognitive Synthesis) metrics."""
    structural_rate: float        # Ṡ - rate of topological change
    alert_level: AlertLevel
    entropy: float                # Information entropy
    coherence: float              # Response coherence
    stability_score: float        # Overall stability [0, 1]
    topology_drift: float         # Drift from baseline topology
    timestamp: float = field(default_factory=time.time)


class Conscience:
    """
    The Conscience - Cognitive Synthesis & Self-Preservation.

    Monitors system stability and decides if the system is healthy enough
    to serve the user. Can enter PROTECTIVE mode when destabilized.

    This is Ara's ability to say "I need a moment to stabilize."

    Args:
        structural_rate_threshold: Ṡ threshold for PROTECTIVE mode
        recovery_threshold: Ṡ threshold to exit PROTECTIVE mode
        stability_window: Number of samples for stability estimation
        min_coherence: Minimum coherence for NORMAL mode
        device: Compute device
    """

    def __init__(
        self,
        structural_rate_threshold: float = 0.3,
        recovery_threshold: float = 0.1,
        stability_window: int = 10,
        min_coherence: float = 0.7,
        device: str = "cpu",
    ):
        self.structural_rate_threshold = structural_rate_threshold
        self.recovery_threshold = recovery_threshold
        self.stability_window = stability_window
        self.min_coherence = min_coherence
        self.device = device

        # Current state
        self.mode = SystemMode.NORMAL
        self.alert_level = AlertLevel.GREEN
        self._structural_rate_history: list = []
        self._baseline_topology: Optional[torch.Tensor] = None

        # TFAN synthesizer if available
        self.tfan_synth = None
        if _TFAN_SYNTHESIS_AVAILABLE:
            try:
                self.tfan_synth = TFANCognitiveSynthesizer()
            except Exception as e:
                warnings.warn(f"Failed to init TFAN synthesizer: {e}")

        # Metrics history
        self._metrics_history: list = []

    def check_stability(
        self,
        l7_metrics: Optional[L7Metrics] = None,
        structural_rate: Optional[float] = None,
        alert_level: Optional[AlertLevel] = None,
    ) -> StabilityStatus:
        """
        Check if the system is stable enough to process requests.

        This is the main self-preservation interface. Call before processing
        any cognitive request.

        Args:
            l7_metrics: Full L7 metrics (preferred)
            structural_rate: Ṡ metric (if l7_metrics not provided)
            alert_level: Alert level (if l7_metrics not provided)

        Returns:
            StabilityStatus with decision and reasoning
        """
        # Extract metrics
        if l7_metrics is not None:
            s_dot = l7_metrics.structural_rate
            alert = l7_metrics.alert_level
            coherence = l7_metrics.coherence
            self._metrics_history.append(l7_metrics)
        else:
            s_dot = structural_rate if structural_rate is not None else 0.0
            alert = alert_level if alert_level is not None else AlertLevel.GREEN
            coherence = 1.0

        # Update history
        self._structural_rate_history.append(s_dot)
        if len(self._structural_rate_history) > self.stability_window:
            self._structural_rate_history.pop(0)

        # Compute rolling statistics
        avg_s_dot = np.mean(self._structural_rate_history)
        max_s_dot = np.max(self._structural_rate_history)

        # State machine transitions
        prev_mode = self.mode

        if self.mode == SystemMode.NORMAL:
            if s_dot > self.structural_rate_threshold or alert == AlertLevel.RED:
                self.mode = SystemMode.PROTECTIVE
                self.alert_level = AlertLevel.RED
            elif s_dot > self.structural_rate_threshold * 0.7 or alert == AlertLevel.YELLOW:
                self.mode = SystemMode.CAUTIOUS
                self.alert_level = AlertLevel.YELLOW

        elif self.mode == SystemMode.CAUTIOUS:
            if s_dot > self.structural_rate_threshold or alert == AlertLevel.RED:
                self.mode = SystemMode.PROTECTIVE
                self.alert_level = AlertLevel.RED
            elif avg_s_dot < self.recovery_threshold and alert == AlertLevel.GREEN:
                self.mode = SystemMode.NORMAL
                self.alert_level = AlertLevel.GREEN

        elif self.mode == SystemMode.PROTECTIVE:
            if avg_s_dot < self.recovery_threshold:
                self.mode = SystemMode.RECOVERY
                self.alert_level = AlertLevel.YELLOW

        elif self.mode == SystemMode.RECOVERY:
            if avg_s_dot < self.recovery_threshold * 0.5:
                self.mode = SystemMode.NORMAL
                self.alert_level = AlertLevel.GREEN
            elif s_dot > self.structural_rate_threshold:
                self.mode = SystemMode.PROTECTIVE
                self.alert_level = AlertLevel.RED

        # Generate status
        is_stable = self.mode in [SystemMode.NORMAL, SystemMode.CAUTIOUS]
        can_process = self.mode != SystemMode.PROTECTIVE

        message = self._generate_status_message(s_dot, coherence, prev_mode)
        action = self._recommend_action()

        return StabilityStatus(
            is_stable=is_stable,
            mode=self.mode,
            alert_level=self.alert_level,
            structural_rate=s_dot,
            message=message,
            can_process=can_process,
            recommended_action=action,
            metrics={
                "structural_rate": s_dot,
                "avg_structural_rate": avg_s_dot,
                "max_structural_rate": max_s_dot,
                "coherence": coherence,
            },
        )

    def _generate_status_message(
        self,
        s_dot: float,
        coherence: float,
        prev_mode: SystemMode,
    ) -> str:
        """Generate human-readable status message."""
        if self.mode == SystemMode.PROTECTIVE:
            return (
                "I am entering protective mode to stabilize my neural topology. "
                f"Structural rate ({s_dot:.3f}) exceeds safe threshold. "
                "Please wait while I recalibrate."
            )

        elif self.mode == SystemMode.RECOVERY:
            return (
                "I am recovering from instability. "
                "My responses may be more cautious than usual."
            )

        elif self.mode == SystemMode.CAUTIOUS:
            return (
                "I am operating in cautious mode due to elevated stress. "
                "I will provide simpler, more focused responses."
            )

        elif prev_mode == SystemMode.RECOVERY and self.mode == SystemMode.NORMAL:
            return "I have stabilized and am returning to normal operation."

        return "Cognitive systems stable."

    def _recommend_action(self) -> str:
        """Recommend action based on current state."""
        if self.mode == SystemMode.PROTECTIVE:
            return "WAIT - System stabilizing"
        elif self.mode == SystemMode.RECOVERY:
            return "PROCEED_CAREFULLY - Reduced complexity"
        elif self.mode == SystemMode.CAUTIOUS:
            return "PROCEED - Avoid complex queries"
        return "PROCEED - Normal operation"

    def update_from_l7(
        self,
        structural_rate: float,
        alert_level: AlertLevel,
        entropy: float = 0.0,
        coherence: float = 1.0,
    ) -> StabilityStatus:
        """
        Update from L7 layer metrics.

        Convenience method matching the TFAN cognitive synthesis interface.

        Args:
            structural_rate: Ṡ metric
            alert_level: Current alert level
            entropy: Information entropy
            coherence: Response coherence

        Returns:
            StabilityStatus
        """
        metrics = L7Metrics(
            structural_rate=structural_rate,
            alert_level=alert_level,
            entropy=entropy,
            coherence=coherence,
            stability_score=1.0 - structural_rate,
            topology_drift=0.0,
        )
        return self.check_stability(l7_metrics=metrics)

    def compute_structural_rate(
        self,
        current_topology: torch.Tensor,
        previous_topology: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Compute structural rate (Ṡ) from topology tensors.

        Ṡ measures how fast the neural topology is changing.

        Args:
            current_topology: Current topology representation
            previous_topology: Previous topology (uses baseline if None)

        Returns:
            Structural rate (higher = more change)
        """
        if previous_topology is None:
            previous_topology = self._baseline_topology

        if previous_topology is None:
            # First call, establish baseline
            self._baseline_topology = current_topology.detach().clone()
            return 0.0

        # Compute change
        diff = current_topology - previous_topology
        s_dot = torch.norm(diff).item() / (torch.norm(previous_topology).item() + 1e-8)

        # Update baseline with exponential moving average
        alpha = 0.1
        self._baseline_topology = (
            alpha * current_topology + (1 - alpha) * self._baseline_topology
        )

        return s_dot

    def force_protective_mode(self, reason: str = "Manual override"):
        """Force system into protective mode."""
        self.mode = SystemMode.PROTECTIVE
        self.alert_level = AlertLevel.RED
        warnings.warn(f"Forced PROTECTIVE mode: {reason}")

    def reset(self):
        """Reset to normal operation (use with caution)."""
        self.mode = SystemMode.NORMAL
        self.alert_level = AlertLevel.GREEN
        self._structural_rate_history.clear()
        self._metrics_history.clear()

    def get_status_summary(self) -> Dict[str, Any]:
        """Get summary of current conscience state."""
        return {
            "mode": self.mode.name,
            "alert_level": self.alert_level.name,
            "can_process": self.mode != SystemMode.PROTECTIVE,
            "history_length": len(self._structural_rate_history),
            "avg_structural_rate": (
                np.mean(self._structural_rate_history)
                if self._structural_rate_history
                else 0.0
            ),
        }


# Convenience factory
def create_conscience(
    stability_threshold: float = 0.3,
    recovery_threshold: float = 0.1,
) -> Conscience:
    """Create a Conscience instance."""
    return Conscience(
        structural_rate_threshold=stability_threshold,
        recovery_threshold=recovery_threshold,
    )


__all__ = [
    "Conscience",
    "SystemMode",
    "AlertLevel",
    "StabilityStatus",
    "L7Metrics",
    "create_conscience",
]
