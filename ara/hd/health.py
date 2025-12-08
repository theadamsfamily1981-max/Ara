"""
Ara HD Health Monitor - Runtime Soul Geometry Tracking
======================================================

Continuous monitoring of HTC health during operation.

This module provides:
- Periodic health checks
- Attractor usage tracking
- Drift detection
- Automatic alerts on geometry degradation

Usage:
    from ara.hd.health import SoulHealthMonitor, get_health_monitor

    monitor = get_health_monitor()
    monitor.record_attractor_activation(row_index)
    monitor.record_plasticity_event(context_hv, reward)

    # Periodic check
    if monitor.should_check():
        report = monitor.run_check()
        if not report.is_healthy:
            logger.warning(f"Soul degradation: {report.summary}")
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import deque
import logging

from .ops import DIM, cosine
from .vocab import get_vocab
from .diagnostics import (
    HealthThresholds,
    DEFAULT_THRESHOLDS,
    SoulHealthReport,
    check_attractor_diversity,
    log_health_metrics,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Health Metrics
# =============================================================================

@dataclass
class HealthMetrics:
    """Rolling health metrics for monitoring."""
    # Attractor usage (per row)
    attractor_hits: Dict[int, int] = field(default_factory=dict)
    attractor_last_hit: Dict[int, datetime] = field(default_factory=dict)

    # Plasticity events
    plasticity_count: int = 0
    reward_history: deque = field(default_factory=lambda: deque(maxlen=1000))

    # Drift tracking
    context_similarities: deque = field(default_factory=lambda: deque(maxlen=1000))
    best_match_scores: deque = field(default_factory=lambda: deque(maxlen=1000))

    # Timing
    last_check: Optional[datetime] = None
    checks_performed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plasticity_count": self.plasticity_count,
            "recent_rewards_mean": (
                sum(self.reward_history) / len(self.reward_history)
                if self.reward_history else 0.0
            ),
            "recent_best_match_mean": (
                sum(self.best_match_scores) / len(self.best_match_scores)
                if self.best_match_scores else 0.0
            ),
            "active_attractors": len(self.attractor_hits),
            "checks_performed": self.checks_performed,
            "last_check": self.last_check.isoformat() if self.last_check else None,
        }


# =============================================================================
# Soul Health Monitor
# =============================================================================

class SoulHealthMonitor:
    """
    Runtime health monitor for the HTC.

    Tracks:
    - Attractor activation patterns
    - Plasticity event statistics
    - Geometry drift over time
    """

    def __init__(
        self,
        n_attractors: int = 2048,
        check_interval: timedelta = timedelta(minutes=5),
        thresholds: HealthThresholds = DEFAULT_THRESHOLDS,
    ):
        """
        Initialize the health monitor.

        Args:
            n_attractors: Number of attractors to track
            check_interval: Minimum time between health checks
            thresholds: Health thresholds to enforce
        """
        self.n_attractors = n_attractors
        self.check_interval = check_interval
        self.thresholds = thresholds

        self.metrics = HealthMetrics()
        self._attractors: Optional[List[np.ndarray]] = None
        self._last_report: Optional[SoulHealthReport] = None

    def set_attractors(self, attractors: List[np.ndarray]) -> None:
        """Set the attractor bank for geometry checks."""
        self._attractors = attractors
        self.n_attractors = len(attractors)

    def record_attractor_activation(self, row_index: int) -> None:
        """Record an attractor hit."""
        now = datetime.utcnow()
        self.metrics.attractor_hits[row_index] = (
            self.metrics.attractor_hits.get(row_index, 0) + 1
        )
        self.metrics.attractor_last_hit[row_index] = now

    def record_plasticity_event(
        self,
        context_hv: np.ndarray,
        reward: float,
        best_match_score: Optional[float] = None,
    ) -> None:
        """Record a plasticity event for monitoring."""
        self.metrics.plasticity_count += 1
        self.metrics.reward_history.append(reward)

        if best_match_score is not None:
            self.metrics.best_match_scores.append(best_match_score)

    def record_context_similarity(
        self,
        context_hv: np.ndarray,
        reference_hv: np.ndarray,
    ) -> None:
        """Track context similarity for drift detection."""
        sim = cosine(context_hv, reference_hv)
        self.metrics.context_similarities.append(sim)

    def should_check(self) -> bool:
        """Check if it's time for a health check."""
        if self.metrics.last_check is None:
            return True
        elapsed = datetime.utcnow() - self.metrics.last_check
        return elapsed >= self.check_interval

    def run_check(self, force: bool = False) -> Optional[SoulHealthReport]:
        """
        Run a health check.

        Args:
            force: Run even if interval hasn't elapsed

        Returns:
            SoulHealthReport or None if skipped
        """
        if not force and not self.should_check():
            return None

        self.metrics.last_check = datetime.utcnow()
        self.metrics.checks_performed += 1

        # Get attractor usage counts
        usage_counts = [
            self.metrics.attractor_hits.get(i, 0)
            for i in range(self.n_attractors)
        ]

        # Run attractor diversity check if we have attractors
        if self._attractors is not None:
            report = check_attractor_diversity(
                self._attractors,
                usage_counts,
                self.thresholds,
            )

            # Add runtime metrics
            runtime_metrics = self.get_runtime_metrics()

            # Create full report
            full_report = SoulHealthReport(
                attractors=report,
                is_healthy=report.is_healthy,
                summary=report.violations[0] if report.violations else "Healthy",
            )

            self._last_report = full_report
            log_health_metrics(full_report, logger)
            return full_report

        return None

    def get_runtime_metrics(self) -> Dict[str, Any]:
        """Get current runtime metrics."""
        metrics = self.metrics.to_dict()

        # Add derived metrics
        if self.metrics.reward_history:
            rewards = list(self.metrics.reward_history)
            metrics["reward_variance"] = float(np.var(rewards))
            metrics["reward_trend"] = (
                sum(rewards[-100:]) / len(rewards[-100:])
                if len(rewards) >= 100 else None
            )

        # Usage distribution
        usage_counts = [
            self.metrics.attractor_hits.get(i, 0)
            for i in range(self.n_attractors)
        ]
        active = sum(1 for c in usage_counts if c > 0)
        metrics["usage_fraction"] = active / self.n_attractors
        metrics["usage_gini"] = self._gini_coefficient(usage_counts)

        return metrics

    def _gini_coefficient(self, values: List[int]) -> float:
        """
        Compute Gini coefficient for usage distribution.

        0 = perfectly equal, 1 = maximally unequal
        """
        if not values or sum(values) == 0:
            return 0.0

        sorted_values = sorted(values)
        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)
        return (2 * np.sum((np.arange(1, n + 1) * sorted_values)) - (n + 1) * cumsum[-1]) / (n * cumsum[-1])

    def get_last_report(self) -> Optional[SoulHealthReport]:
        """Get the most recent health report."""
        return self._last_report

    def reset_daily(self) -> None:
        """Reset daily counters (call at midnight or start of day)."""
        self.metrics.attractor_hits.clear()
        self.metrics.attractor_last_hit.clear()
        logger.info("SoulHealthMonitor: Daily metrics reset")

    def get_status(self) -> Dict[str, Any]:
        """Get monitor status for dashboards."""
        return {
            "n_attractors": self.n_attractors,
            "check_interval_seconds": self.check_interval.total_seconds(),
            "metrics": self.metrics.to_dict(),
            "runtime": self.get_runtime_metrics(),
            "last_report_healthy": (
                self._last_report.is_healthy if self._last_report else None
            ),
        }


# =============================================================================
# Alert System
# =============================================================================

@dataclass
class HealthAlert:
    """A health alert triggered by monitoring."""
    severity: str  # "warning", "critical"
    category: str  # "codebook", "bundling", "attractor", "drift"
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity,
            "category": self.category,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "metrics": self.metrics,
        }


class AlertManager:
    """Manages health alerts."""

    def __init__(self, max_alerts: int = 100):
        self.alerts: deque = deque(maxlen=max_alerts)
        self.callbacks: List[callable] = []

    def add_alert(self, alert: HealthAlert) -> None:
        """Add an alert and notify callbacks."""
        self.alerts.append(alert)

        if alert.severity == "critical":
            logger.critical(f"SOUL ALERT: {alert.message}")
        else:
            logger.warning(f"Soul alert: {alert.message}")

        for callback in self.callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    def register_callback(self, callback: callable) -> None:
        """Register a callback for alerts."""
        self.callbacks.append(callback)

    def get_recent(self, n: int = 10) -> List[HealthAlert]:
        """Get recent alerts."""
        return list(self.alerts)[-n:]

    def clear(self) -> None:
        """Clear all alerts."""
        self.alerts.clear()


# =============================================================================
# Singleton Access
# =============================================================================

_health_monitor: Optional[SoulHealthMonitor] = None
_alert_manager: Optional[AlertManager] = None


def get_health_monitor(n_attractors: int = 2048) -> SoulHealthMonitor:
    """Get the global health monitor."""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = SoulHealthMonitor(n_attractors=n_attractors)
    return _health_monitor


def get_alert_manager() -> AlertManager:
    """Get the global alert manager."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'HealthMetrics',
    'SoulHealthMonitor',
    'get_health_monitor',
    'HealthAlert',
    'AlertManager',
    'get_alert_manager',
]
