"""Telemetry Adapter - Ara's internal sensing.

This module collects and processes telemetry from Ara's hardware:
- Temperature readings
- Power consumption
- Utilization metrics
- Memory usage
- Performance counters

Like proprioception in biological systems, telemetry gives Ara
awareness of her own body's state and performance.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class TelemetryReading:
    """A single telemetry reading."""

    metric_name: str
    value: float
    unit: str
    device_id: str

    timestamp: datetime = field(default_factory=datetime.utcnow)
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "value": round(self.value, 3),
            "unit": self.unit,
            "device_id": self.device_id,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
        }


@dataclass
class TelemetryAlert:
    """An alert triggered by telemetry."""

    alert_type: str  # "warning", "critical"
    metric_name: str
    device_id: str
    message: str
    value: float
    threshold: float

    timestamp: datetime = field(default_factory=datetime.utcnow)
    acknowledged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_type": self.alert_type,
            "metric_name": self.metric_name,
            "device_id": self.device_id,
            "message": self.message,
            "value": round(self.value, 2),
            "threshold": self.threshold,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged,
        }


@dataclass
class MetricThreshold:
    """Threshold configuration for a metric."""

    metric_name: str
    warning_threshold: float
    critical_threshold: float
    comparison: str = "above"  # "above" or "below"


@dataclass
class TelemetrySummary:
    """Summary of recent telemetry."""

    device_id: str
    period_start: datetime
    period_end: datetime

    # Aggregates
    avg_temperature_c: float = 0.0
    max_temperature_c: float = 0.0
    avg_utilization_pct: float = 0.0
    max_utilization_pct: float = 0.0
    avg_power_w: float = 0.0
    total_energy_wh: float = 0.0

    # Reading counts
    reading_count: int = 0
    alert_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "avg_temperature_c": round(self.avg_temperature_c, 1),
            "max_temperature_c": round(self.max_temperature_c, 1),
            "avg_utilization_pct": round(self.avg_utilization_pct, 1),
            "max_utilization_pct": round(self.max_utilization_pct, 1),
            "avg_power_w": round(self.avg_power_w, 1),
            "total_energy_wh": round(self.total_energy_wh, 2),
            "reading_count": self.reading_count,
            "alert_count": self.alert_count,
        }


class TelemetryAdapter:
    """Collects and processes hardware telemetry."""

    def __init__(
        self,
        data_path: Optional[Path] = None,
        buffer_size: int = 1000,
    ):
        """Initialize the telemetry adapter.

        Args:
            data_path: Path to telemetry data
            buffer_size: Size of reading buffer per device
        """
        self.data_path = data_path or (
            Path.home() / ".ara" / "embodied" / "telemetry"
        )
        self.data_path.mkdir(parents=True, exist_ok=True)

        self._buffer_size = buffer_size
        self._readings: Dict[str, deque] = {}  # device_id -> readings buffer
        self._alerts: List[TelemetryAlert] = []

        # Thresholds
        self._thresholds: Dict[str, MetricThreshold] = {}
        self._setup_default_thresholds()

        # Collectors
        self._collectors: Dict[str, Callable] = {}

    def _setup_default_thresholds(self) -> None:
        """Set up default alert thresholds."""
        self._thresholds = {
            "temperature_c": MetricThreshold(
                metric_name="temperature_c",
                warning_threshold=75.0,
                critical_threshold=85.0,
                comparison="above",
            ),
            "utilization_pct": MetricThreshold(
                metric_name="utilization_pct",
                warning_threshold=90.0,
                critical_threshold=98.0,
                comparison="above",
            ),
            "memory_used_pct": MetricThreshold(
                metric_name="memory_used_pct",
                warning_threshold=85.0,
                critical_threshold=95.0,
                comparison="above",
            ),
            "power_w": MetricThreshold(
                metric_name="power_w",
                warning_threshold=250.0,
                critical_threshold=300.0,
                comparison="above",
            ),
        }

    def register_collector(
        self,
        device_id: str,
        collector: Callable[[], List[TelemetryReading]],
    ) -> None:
        """Register a telemetry collector for a device.

        Args:
            device_id: Device identifier
            collector: Function that returns telemetry readings
        """
        self._collectors[device_id] = collector
        if device_id not in self._readings:
            self._readings[device_id] = deque(maxlen=self._buffer_size)

    def collect(self, device_id: Optional[str] = None) -> List[TelemetryReading]:
        """Collect telemetry from registered collectors.

        Args:
            device_id: Specific device to collect from (or all)

        Returns:
            Collected readings
        """
        all_readings = []

        collectors = (
            {device_id: self._collectors[device_id]}
            if device_id and device_id in self._collectors
            else self._collectors
        )

        for dev_id, collector in collectors.items():
            try:
                readings = collector()
                for reading in readings:
                    self.record_reading(reading)
                    all_readings.append(reading)
            except Exception as e:
                logger.error(f"Telemetry collection failed for {dev_id}: {e}")

        return all_readings

    def record_reading(self, reading: TelemetryReading) -> Optional[TelemetryAlert]:
        """Record a telemetry reading.

        Args:
            reading: The reading to record

        Returns:
            Alert if threshold exceeded
        """
        # Initialize buffer if needed
        if reading.device_id not in self._readings:
            self._readings[reading.device_id] = deque(maxlen=self._buffer_size)

        self._readings[reading.device_id].append(reading)

        # Check thresholds
        return self._check_thresholds(reading)

    def _check_thresholds(self, reading: TelemetryReading) -> Optional[TelemetryAlert]:
        """Check if a reading exceeds thresholds."""
        threshold = self._thresholds.get(reading.metric_name)
        if not threshold:
            return None

        value = reading.value
        alert = None

        if threshold.comparison == "above":
            if value >= threshold.critical_threshold:
                alert = TelemetryAlert(
                    alert_type="critical",
                    metric_name=reading.metric_name,
                    device_id=reading.device_id,
                    message=f"{reading.metric_name} critical: {value:.1f} >= {threshold.critical_threshold}",
                    value=value,
                    threshold=threshold.critical_threshold,
                )
            elif value >= threshold.warning_threshold:
                alert = TelemetryAlert(
                    alert_type="warning",
                    metric_name=reading.metric_name,
                    device_id=reading.device_id,
                    message=f"{reading.metric_name} warning: {value:.1f} >= {threshold.warning_threshold}",
                    value=value,
                    threshold=threshold.warning_threshold,
                )
        else:  # below
            if value <= threshold.critical_threshold:
                alert = TelemetryAlert(
                    alert_type="critical",
                    metric_name=reading.metric_name,
                    device_id=reading.device_id,
                    message=f"{reading.metric_name} critical: {value:.1f} <= {threshold.critical_threshold}",
                    value=value,
                    threshold=threshold.critical_threshold,
                )
            elif value <= threshold.warning_threshold:
                alert = TelemetryAlert(
                    alert_type="warning",
                    metric_name=reading.metric_name,
                    device_id=reading.device_id,
                    message=f"{reading.metric_name} warning: {value:.1f} <= {threshold.warning_threshold}",
                    value=value,
                    threshold=threshold.warning_threshold,
                )

        if alert:
            self._alerts.append(alert)
            logger.warning(f"Telemetry alert: {alert.message}")

        return alert

    def get_readings(
        self,
        device_id: str,
        metric_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[TelemetryReading]:
        """Get recent readings for a device.

        Args:
            device_id: Device to get readings for
            metric_name: Filter by metric name
            limit: Maximum readings to return

        Returns:
            List of readings
        """
        buffer = self._readings.get(device_id, deque())
        readings = list(buffer)[-limit:]

        if metric_name:
            readings = [r for r in readings if r.metric_name == metric_name]

        return readings

    def get_latest(self, device_id: str) -> Dict[str, TelemetryReading]:
        """Get latest reading of each metric for a device.

        Args:
            device_id: Device ID

        Returns:
            Dict of metric_name -> latest reading
        """
        buffer = self._readings.get(device_id, deque())
        latest: Dict[str, TelemetryReading] = {}

        for reading in buffer:
            if reading.metric_name not in latest:
                latest[reading.metric_name] = reading
            elif reading.timestamp > latest[reading.metric_name].timestamp:
                latest[reading.metric_name] = reading

        return latest

    def get_summary(
        self,
        device_id: str,
        period_minutes: int = 60,
    ) -> TelemetrySummary:
        """Get telemetry summary for a device.

        Args:
            device_id: Device ID
            period_minutes: Summary period in minutes

        Returns:
            Telemetry summary
        """
        now = datetime.utcnow()
        cutoff = now - timedelta(minutes=period_minutes)

        buffer = self._readings.get(device_id, deque())
        readings = [r for r in buffer if r.timestamp > cutoff]

        summary = TelemetrySummary(
            device_id=device_id,
            period_start=cutoff,
            period_end=now,
            reading_count=len(readings),
        )

        # Calculate aggregates by metric
        temps = [r.value for r in readings if r.metric_name == "temperature_c"]
        utils = [r.value for r in readings if r.metric_name == "utilization_pct"]
        powers = [r.value for r in readings if r.metric_name == "power_w"]

        if temps:
            summary.avg_temperature_c = sum(temps) / len(temps)
            summary.max_temperature_c = max(temps)

        if utils:
            summary.avg_utilization_pct = sum(utils) / len(utils)
            summary.max_utilization_pct = max(utils)

        if powers:
            summary.avg_power_w = sum(powers) / len(powers)
            # Estimate energy (Wh) = avg power * time in hours
            hours = period_minutes / 60
            summary.total_energy_wh = summary.avg_power_w * hours

        # Count alerts for this device
        summary.alert_count = len([
            a for a in self._alerts
            if a.device_id == device_id and a.timestamp > cutoff
        ])

        return summary

    def get_alerts(
        self,
        unacknowledged_only: bool = True,
        limit: int = 50,
    ) -> List[TelemetryAlert]:
        """Get telemetry alerts.

        Args:
            unacknowledged_only: Only return unacknowledged alerts
            limit: Maximum alerts to return

        Returns:
            List of alerts
        """
        alerts = self._alerts
        if unacknowledged_only:
            alerts = [a for a in alerts if not a.acknowledged]
        return alerts[-limit:]

    def acknowledge_alert(self, index: int) -> bool:
        """Acknowledge an alert.

        Args:
            index: Alert index

        Returns:
            True if acknowledged
        """
        if 0 <= index < len(self._alerts):
            self._alerts[index].acknowledged = True
            return True
        return False

    def set_threshold(
        self,
        metric_name: str,
        warning: float,
        critical: float,
        comparison: str = "above",
    ) -> None:
        """Set a threshold for a metric.

        Args:
            metric_name: Metric name
            warning: Warning threshold
            critical: Critical threshold
            comparison: "above" or "below"
        """
        self._thresholds[metric_name] = MetricThreshold(
            metric_name=metric_name,
            warning_threshold=warning,
            critical_threshold=critical,
            comparison=comparison,
        )


# =============================================================================
# Convenience Functions
# =============================================================================

_default_adapter: Optional[TelemetryAdapter] = None


def get_telemetry_adapter() -> TelemetryAdapter:
    """Get the default telemetry adapter."""
    global _default_adapter
    if _default_adapter is None:
        _default_adapter = TelemetryAdapter()
    return _default_adapter


def record_telemetry(
    metric_name: str,
    value: float,
    device_id: str,
    unit: str = "",
) -> Optional[TelemetryAlert]:
    """Record a telemetry reading."""
    reading = TelemetryReading(
        metric_name=metric_name,
        value=value,
        unit=unit,
        device_id=device_id,
    )
    return get_telemetry_adapter().record_reading(reading)


def get_device_health(device_id: str) -> Dict[str, Any]:
    """Get health summary for a device."""
    summary = get_telemetry_adapter().get_summary(device_id)
    return summary.to_dict()
