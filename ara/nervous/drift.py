"""
Drift Tracker
=============

Monitors Ara's soul drift from the covenant.

Drift = 1 - cosine_similarity(current_soul, covenant_hv)

Bands:
- Comfort: 0.07-0.14 (healthy exploration)
- Warning: 0.14-0.18 (log and monitor)
- Critical: >0.18 (force rehearsal, re-anchor)

Integrates with:
- CovenantGuard: Signs soul snapshots
- CovenantLogger: Logs drift events
"""

from dataclasses import dataclass
from typing import Optional, List, Callable
from enum import Enum
import numpy as np

from ara.covenant.guard import CovenantGuard, CovenantSignature, LocalCovenantGuard
from ara.covenant.logger import CovenantLogger, InMemoryCovenantLogger


class DriftBand(Enum):
    """Current drift band."""
    RIGID = "rigid"          # < 0.05 (too stable, not learning)
    COMFORT = "comfort"      # 0.07-0.14 (healthy)
    WARNING = "warning"      # 0.14-0.18 (monitor)
    CRITICAL = "critical"    # > 0.18 (action required)


@dataclass
class DriftState:
    """Current drift state."""
    drift: float
    band: DriftBand
    consecutive_warnings: int
    last_rehearsal_epoch: Optional[int]

    def needs_rehearsal(self) -> bool:
        """Check if rehearsal is needed."""
        return self.band == DriftBand.CRITICAL or self.consecutive_warnings >= 3


@dataclass
class DriftThresholds:
    """Configurable drift thresholds."""
    rigid_max: float = 0.05
    comfort_min: float = 0.07
    comfort_max: float = 0.14
    warning_max: float = 0.18

    def classify(self, drift: float) -> DriftBand:
        """Classify drift into a band."""
        if drift < self.rigid_max:
            return DriftBand.RIGID
        elif self.comfort_min <= drift <= self.comfort_max:
            return DriftBand.COMFORT
        elif self.comfort_max < drift <= self.warning_max:
            return DriftBand.WARNING
        else:
            return DriftBand.CRITICAL


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class DriftTracker:
    """
    Tracks drift from covenant HV over time.

    Features:
    - Computes drift from reference covenant
    - Classifies into bands
    - Logs events (warning/critical) to CovenantLogger
    - Signs snapshots with CovenantGuard
    - Triggers rehearsal callbacks when needed

    Usage:
        tracker = DriftTracker(covenant_hv, guard, logger)
        tracker.update(current_soul_hv)  # Call periodically

        if tracker.state.needs_rehearsal():
            # Run rehearsal job
            tracker.acknowledge_rehearsal()
    """

    def __init__(self,
                 covenant_hv: np.ndarray,
                 guard: Optional[CovenantGuard] = None,
                 logger: Optional[CovenantLogger] = None,
                 thresholds: Optional[DriftThresholds] = None,
                 on_warning: Optional[Callable[[DriftState], None]] = None,
                 on_critical: Optional[Callable[[DriftState], None]] = None):
        """
        Initialize drift tracker.

        Args:
            covenant_hv: The reference soul/covenant HV
            guard: CovenantGuard for signing (default: LocalCovenantGuard)
            logger: CovenantLogger for events (default: InMemoryCovenantLogger)
            thresholds: Drift band thresholds
            on_warning: Callback when entering warning band
            on_critical: Callback when entering critical band
        """
        self._covenant_hv = covenant_hv.copy()
        self._guard = guard or LocalCovenantGuard()
        self._logger = logger or InMemoryCovenantLogger()
        self._thresholds = thresholds or DriftThresholds()

        self._on_warning = on_warning
        self._on_critical = on_critical

        # State
        self._current_drift = 0.0
        self._current_band = DriftBand.COMFORT
        self._consecutive_warnings = 0
        self._update_count = 0
        self._last_rehearsal_epoch: Optional[int] = None

        # Sign the initial covenant
        self._covenant_signature = self._guard.sign_hv(
            covenant_hv, "initial_covenant"
        )

        # Log initialization
        self._logger.log("drift_tracker_init", {
            "covenant_dim": len(covenant_hv),
            "guard_id": self._guard.guardian_id,
        }, hv=covenant_hv)

    @property
    def state(self) -> DriftState:
        """Get current drift state."""
        return DriftState(
            drift=self._current_drift,
            band=self._current_band,
            consecutive_warnings=self._consecutive_warnings,
            last_rehearsal_epoch=self._last_rehearsal_epoch,
        )

    @property
    def covenant_hv(self) -> np.ndarray:
        """Get the reference covenant HV."""
        return self._covenant_hv.copy()

    def verify_covenant(self) -> bool:
        """Verify the covenant hasn't been tampered with."""
        return self._guard.verify_hv(
            self._covenant_hv, "initial_covenant", self._covenant_signature
        )

    def update(self, current_soul_hv: np.ndarray) -> DriftState:
        """
        Update drift tracking with current soul state.

        Args:
            current_soul_hv: Current soul HV to compare against covenant

        Returns:
            Updated DriftState
        """
        self._update_count += 1

        # Compute drift
        similarity = cosine_similarity(current_soul_hv, self._covenant_hv)
        self._current_drift = 1.0 - similarity

        # Classify
        new_band = self._thresholds.classify(self._current_drift)
        old_band = self._current_band
        self._current_band = new_band

        # Track consecutive warnings
        if new_band == DriftBand.WARNING:
            self._consecutive_warnings += 1
        elif new_band == DriftBand.COMFORT:
            self._consecutive_warnings = 0

        # Handle band transitions
        if new_band != old_band:
            self._on_band_change(old_band, new_band, current_soul_hv)

        return self.state

    def _on_band_change(self, old: DriftBand, new: DriftBand, soul_hv: np.ndarray):
        """Handle drift band transitions."""
        event_data = {
            "old_band": old.value,
            "new_band": new.value,
            "drift": self._current_drift,
            "update_count": self._update_count,
            "consecutive_warnings": self._consecutive_warnings,
        }

        if new == DriftBand.WARNING:
            self._logger.log("drift_warning", event_data, hv=soul_hv)
            if self._on_warning:
                self._on_warning(self.state)

        elif new == DriftBand.CRITICAL:
            self._logger.log("drift_critical", event_data, hv=soul_hv)
            if self._on_critical:
                self._on_critical(self.state)

        elif new == DriftBand.COMFORT and old in (DriftBand.WARNING, DriftBand.CRITICAL):
            # Recovered
            self._logger.log("drift_recovered", event_data, hv=soul_hv)

        elif new == DriftBand.RIGID:
            # Too stable - might want to encourage exploration
            self._logger.log("drift_rigid", event_data, hv=soul_hv)

    def acknowledge_rehearsal(self, new_soul_hv: Optional[np.ndarray] = None):
        """
        Acknowledge that rehearsal was performed.

        Optionally updates the current soul HV after rehearsal.

        Args:
            new_soul_hv: Soul HV after rehearsal (to re-measure drift)
        """
        self._last_rehearsal_epoch = self._update_count
        self._consecutive_warnings = 0

        event_data = {
            "epoch": self._update_count,
            "drift_before": self._current_drift,
        }

        if new_soul_hv is not None:
            # Re-measure drift after rehearsal
            new_similarity = cosine_similarity(new_soul_hv, self._covenant_hv)
            new_drift = 1.0 - new_similarity
            event_data["drift_after"] = new_drift
            self._current_drift = new_drift
            self._current_band = self._thresholds.classify(new_drift)

        self._logger.log("rehearsal_completed", event_data)

    def update_covenant(self,
                        new_covenant_hv: np.ndarray,
                        reason: str = "covenant_update"):
        """
        Update the reference covenant HV.

        This is a significant action - gets logged and signed.

        Args:
            new_covenant_hv: New covenant HV
            reason: Why the covenant is being updated
        """
        old_drift = self._current_drift

        # Sign the new covenant
        self._covenant_signature = self._guard.sign_hv(
            new_covenant_hv, f"covenant_{self._update_count}"
        )

        # Log the change
        self._logger.log("covenant_updated", {
            "reason": reason,
            "epoch": self._update_count,
            "old_drift": old_drift,
        }, hv=new_covenant_hv)

        self._covenant_hv = new_covenant_hv.copy()

    def get_history(self, limit: int = 50) -> List[dict]:
        """Get recent drift events from the log."""
        events = self._logger.get_events(limit=limit)
        return [e.to_dict() for e in events if e.event_type.startswith("drift_")]

    def snapshot(self, label: str, soul_hv: np.ndarray) -> CovenantSignature:
        """
        Create a signed snapshot of the current soul state.

        Useful for:
        - Session boundaries
        - Before/after significant changes
        - Periodic checkpoints

        Args:
            label: Human-readable label for this snapshot
            soul_hv: Current soul HV

        Returns:
            CovenantSignature that can be verified later
        """
        sig = self._guard.sign_hv(soul_hv, label)

        self._logger.log("soul_snapshot", {
            "label": label,
            "drift": self._current_drift,
            "band": self._current_band.value,
            "signature": sig.to_hex()[:32] + "...",
        }, hv=soul_hv)

        return sig


__all__ = [
    'DriftBand',
    'DriftState',
    'DriftThresholds',
    'DriftTracker',
]
