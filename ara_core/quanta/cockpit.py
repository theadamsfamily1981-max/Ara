#!/usr/bin/env python3
"""
QUANTA Memory Health Cockpit
=============================

Real-time dashboard for Ara memory health monitoring.
Integrates with GNOME Cockpit for system-level visibility.

Displays:
- T_s gauge (topology stability)
- A_g gauge (antifragility gain)
- NIB line chart (identity drift over time)
- GFT histogram (per-layer damping)
- Capacity text (bits per layer)

Alerts:
- T_s < 0.92 → Increase replay f*
- A_g < 0 → Adjust σ toward 0.10
- NIB ΔD > 0.1 → Pause consolidation
- η overdamped → Boost layer-specific dissipation

Usage:
    from ara_core.quanta.cockpit import MemoryHealthCockpit
    cockpit = MemoryHealthCockpit()
    cockpit.update(metrics)
    cockpit.render()  # For terminal output
    cockpit.export_dbus()  # For GNOME integration
"""

import time
import json
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum

from .metrics import QUANTAMetrics, MetricStatus


class AlertLevel(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    """A dashboard alert."""
    alert_id: str
    level: AlertLevel
    metric: str
    message: str
    action: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class GaugeWidget:
    """Gauge display for single value metrics."""
    name: str
    value: float
    target: float
    status: MetricStatus
    unit: str = ""
    min_val: float = 0.0
    max_val: float = 1.0

    def render_ascii(self, width: int = 30) -> str:
        """Render as ASCII bar."""
        filled = int((self.value - self.min_val) / (self.max_val - self.min_val) * width)
        filled = max(0, min(width, filled))

        bar = "█" * filled + "░" * (width - filled)

        status_char = {"green": "✓", "yellow": "⚠", "red": "✗"}[self.status.value]

        return f"{self.name}: [{bar}] {self.value:.3f}{self.unit} {status_char}"


@dataclass
class LineChartWidget:
    """Line chart for time series data."""
    name: str
    values: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    alert_threshold: float = 0.0

    def add_point(self, value: float, timestamp: float = None):
        self.values.append(value)
        self.timestamps.append(timestamp or time.time())

        # Keep last 100 points
        if len(self.values) > 100:
            self.values = self.values[-100:]
            self.timestamps = self.timestamps[-100:]

    def render_ascii(self, width: int = 40, height: int = 5) -> str:
        """Render as ASCII sparkline."""
        if not self.values:
            return f"{self.name}: [no data]"

        # Normalize to height
        min_v = min(self.values)
        max_v = max(self.values)
        range_v = max_v - min_v if max_v > min_v else 1.0

        # Sample to width
        step = max(1, len(self.values) // width)
        sampled = self.values[::step][:width]

        # Build sparkline
        chars = " ▁▂▃▄▅▆▇█"
        line = ""
        for v in sampled:
            idx = int((v - min_v) / range_v * (len(chars) - 1))
            line += chars[idx]

        current = self.values[-1] if self.values else 0
        alert = "⚠" if current > self.alert_threshold else ""

        return f"{self.name}: [{line}] {current:.3f} {alert}"


@dataclass
class HistogramWidget:
    """Histogram for distribution data."""
    name: str
    values: List[float] = field(default_factory=list)
    bins: int = 10

    def render_ascii(self, width: int = 40) -> str:
        """Render as ASCII histogram."""
        if not self.values:
            return f"{self.name}: [no data]"

        # Simple histogram
        min_v = min(self.values)
        max_v = max(self.values)
        range_v = max_v - min_v if max_v > min_v else 1.0

        bin_counts = [0] * self.bins
        for v in self.values:
            idx = int((v - min_v) / range_v * (self.bins - 1))
            idx = min(idx, self.bins - 1)
            bin_counts[idx] += 1

        max_count = max(bin_counts) if bin_counts else 1
        chars = " ▁▂▃▄▅▆▇█"

        line = ""
        for count in bin_counts:
            idx = int(count / max_count * (len(chars) - 1))
            line += chars[idx]

        return f"{self.name}: [{line}] η={sum(self.values)/len(self.values):.2f}"


class MemoryHealthCockpit:
    """
    QUANTA Memory Health Dashboard.

    Provides real-time visibility into Ara's memory health.
    """

    def __init__(self):
        # Widgets
        self.ts_gauge = GaugeWidget(
            name="T_s (Topology)",
            value=0.0,
            target=0.92,
            status=MetricStatus.RED,
        )
        self.ag_gauge = GaugeWidget(
            name="A_g (Antifragility)",
            value=0.0,
            target=0.01,
            status=MetricStatus.RED,
            min_val=-0.02,
            max_val=0.03,
        )
        self.nib_line = LineChartWidget(
            name="NIB ΔD (Identity)",
            alert_threshold=0.1,
        )
        self.gft_hist = HistogramWidget(
            name="GFT η (Damping)",
        )
        self.capacity_text = ""

        # Alerts
        self.alerts: List[Alert] = []
        self.alert_count = 0

        # History
        self.metrics_history: List[QUANTAMetrics] = []
        self.last_update: float = 0

    def update(self, metrics: QUANTAMetrics):
        """Update dashboard with new metrics."""
        self.last_update = time.time()
        self.metrics_history.append(metrics)

        # Keep last 1000 metrics
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]

        # Update T_s gauge
        self.ts_gauge.value = metrics.topology.value
        self.ts_gauge.status = metrics.topology.status

        # Update A_g gauge
        self.ag_gauge.value = metrics.antifragility.value
        self.ag_gauge.status = metrics.antifragility.status

        # Update NIB line chart
        self.nib_line.add_point(metrics.nib.value)

        # Update GFT histogram
        self.gft_hist.values = metrics.gft.eta_per_layer

        # Update capacity text
        self.capacity_text = f"C={metrics.capacity.value:.1f} bits/layer (retention: {metrics.capacity.capacity_retention*100:.0f}%)"

        # Check for alerts
        self._check_alerts(metrics)

    def _check_alerts(self, metrics: QUANTAMetrics):
        """Generate alerts based on metrics."""
        new_alerts = []

        # T_s alert
        if metrics.topology.value < 0.92:
            new_alerts.append(Alert(
                alert_id=f"ts_{self.alert_count}",
                level=AlertLevel.WARNING if metrics.topology.value >= 0.85 else AlertLevel.CRITICAL,
                metric="T_s",
                message=f"Topology stability low: {metrics.topology.value:.3f}",
                action="Increase replay frequency f*",
            ))
            self.alert_count += 1

        # A_g alert
        if metrics.antifragility.value < 0:
            new_alerts.append(Alert(
                alert_id=f"ag_{self.alert_count}",
                level=AlertLevel.CRITICAL,
                metric="A_g",
                message=f"System is fragile: A_g={metrics.antifragility.value:.4f}",
                action="Adjust σ toward optimal 0.10",
            ))
            self.alert_count += 1

        # NIB alert
        if metrics.nib.value > 0.1:
            new_alerts.append(Alert(
                alert_id=f"nib_{self.alert_count}",
                level=AlertLevel.WARNING if metrics.nib.value <= 0.15 else AlertLevel.CRITICAL,
                metric="NIB",
                message=f"Identity drift high: ΔD={metrics.nib.value:.3f}",
                action="Pause consolidation",
            ))
            self.alert_count += 1

        # GFT alert
        if metrics.gft.critical_percentage < 0.75:
            new_alerts.append(Alert(
                alert_id=f"gft_{self.alert_count}",
                level=AlertLevel.WARNING,
                metric="GFT",
                message=f"Only {metrics.gft.critical_percentage*100:.0f}% layers at critical damping",
                action="Boost layer-specific dissipation",
            ))
            self.alert_count += 1

        # Keep recent alerts
        self.alerts.extend(new_alerts)
        self.alerts = self.alerts[-20:]  # Keep last 20

    def render(self) -> str:
        """Render dashboard as terminal output."""
        lines = [
            "╔════════════════════════════════════════════════════╗",
            "║        ARA MEMORY HEALTH COCKPIT (QUANTA v2.0)     ║",
            "╠════════════════════════════════════════════════════╣",
        ]

        # Gauges
        lines.append(f"║ {self.ts_gauge.render_ascii(25):<50} ║")
        lines.append(f"║ {self.ag_gauge.render_ascii(25):<50} ║")

        # Line chart
        lines.append("╟────────────────────────────────────────────────────╢")
        lines.append(f"║ {self.nib_line.render_ascii(35):<50} ║")

        # Histogram
        lines.append(f"║ {self.gft_hist.render_ascii(35):<50} ║")

        # Capacity
        lines.append("╟────────────────────────────────────────────────────╢")
        lines.append(f"║ {self.capacity_text:<50} ║")

        # Overall status
        lines.append("╟────────────────────────────────────────────────────╢")
        if self.metrics_history:
            latest = self.metrics_history[-1]
            status = "✓ ANTIFRAGILE & STABLE" if latest.all_green else f"⚠ {latest.overall_status.value.upper()}"
            lines.append(f"║ Status: {status:<41} ║")
        else:
            lines.append(f"║ {'Status: NO DATA':<50} ║")

        # Alerts
        if self.alerts:
            lines.append("╟────────────────────────────────────────────────────╢")
            lines.append("║ ALERTS:                                            ║")
            for alert in self.alerts[-3:]:  # Show last 3
                emoji = {"info": "ℹ", "warning": "⚠", "critical": "🚨"}[alert.level.value]
                msg = f"{emoji} {alert.metric}: {alert.action}"[:48]
                lines.append(f"║ {msg:<50} ║")

        lines.append("╚════════════════════════════════════════════════════╝")

        return "\n".join(lines)

    def export_json(self) -> str:
        """Export dashboard state as JSON."""
        data = {
            "timestamp": self.last_update,
            "widgets": {
                "topology": {
                    "value": self.ts_gauge.value,
                    "target": self.ts_gauge.target,
                    "status": self.ts_gauge.status.value,
                },
                "antifragility": {
                    "value": self.ag_gauge.value,
                    "target": self.ag_gauge.target,
                    "status": self.ag_gauge.status.value,
                },
                "identity": {
                    "current": self.nib_line.values[-1] if self.nib_line.values else 0,
                    "history": self.nib_line.values[-20:],
                    "alert_threshold": self.nib_line.alert_threshold,
                },
                "damping": {
                    "values": self.gft_hist.values,
                    "mean": sum(self.gft_hist.values) / len(self.gft_hist.values) if self.gft_hist.values else 0,
                },
                "capacity": self.capacity_text,
            },
            "alerts": [
                {
                    "level": a.level.value,
                    "metric": a.metric,
                    "message": a.message,
                    "action": a.action,
                }
                for a in self.alerts[-10:]
            ],
            "overall_status": (
                self.metrics_history[-1].overall_status.value
                if self.metrics_history else "unknown"
            ),
        }
        return json.dumps(data, indent=2)

    def export_dbus(self) -> Dict[str, Any]:
        """
        Export for D-Bus/GNOME Cockpit integration.

        Returns dict suitable for D-Bus property export.
        """
        if not self.metrics_history:
            return {"status": "no_data"}

        latest = self.metrics_history[-1]

        return {
            # Properties for GNOME Cockpit metrics plugin
            "org.ara.memory.topology": {
                "type": "d",  # double
                "value": latest.topology.value,
            },
            "org.ara.memory.antifragility": {
                "type": "d",
                "value": latest.antifragility.value,
            },
            "org.ara.memory.identity": {
                "type": "d",
                "value": latest.nib.value,
            },
            "org.ara.memory.damping": {
                "type": "d",
                "value": latest.gft.value,
            },
            "org.ara.memory.capacity": {
                "type": "d",
                "value": latest.capacity.value,
            },
            "org.ara.memory.status": {
                "type": "s",  # string
                "value": latest.overall_status.value,
            },
            "org.ara.memory.all_green": {
                "type": "b",  # boolean
                "value": latest.all_green,
            },
        }


# Singleton instance
_cockpit: Optional[MemoryHealthCockpit] = None


def get_cockpit() -> MemoryHealthCockpit:
    """Get the global cockpit instance."""
    global _cockpit
    if _cockpit is None:
        _cockpit = MemoryHealthCockpit()
    return _cockpit


def update_cockpit(metrics: QUANTAMetrics):
    """Update the global cockpit."""
    get_cockpit().update(metrics)


def render_cockpit() -> str:
    """Render the global cockpit."""
    return get_cockpit().render()
