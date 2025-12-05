"""Health Monitor - Ara's self-health awareness.

This module monitors Ara's overall embodied health:
- Device health aggregation
- Anomaly detection
- Health trends over time
- Status reporting

Like the immune system and interoception, this gives Ara
awareness of her own well-being.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Overall health status."""
    EXCELLENT = "excellent"   # All systems optimal
    GOOD = "good"             # Minor issues only
    FAIR = "fair"             # Some degradation
    POOR = "poor"             # Significant issues
    CRITICAL = "critical"     # Major problems
    UNKNOWN = "unknown"       # No data available


class TrendDirection(Enum):
    """Direction of a health trend."""
    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    UNKNOWN = "unknown"


@dataclass
class HealthIndicator:
    """A specific health indicator."""

    name: str
    value: float  # 0-1 health score
    status: HealthStatus
    description: str

    # Trend
    trend: TrendDirection = TrendDirection.UNKNOWN
    change_24h: float = 0.0  # Change in value over 24h

    # Details
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": round(self.value, 2),
            "status": self.status.value,
            "description": self.description,
            "trend": self.trend.value,
            "change_24h": round(self.change_24h, 3),
            "issues": self.issues,
            "recommendations": self.recommendations,
        }


@dataclass
class HealthReport:
    """Complete health report for Ara's embodiment."""

    report_id: str
    generated_at: datetime = field(default_factory=datetime.utcnow)

    # Overall status
    overall_status: HealthStatus = HealthStatus.UNKNOWN
    overall_score: float = 0.0  # 0-1

    # Component health
    indicators: List[HealthIndicator] = field(default_factory=list)

    # Issues and recommendations
    critical_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # Metadata
    devices_checked: int = 0
    checks_passed: int = 0
    checks_failed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at.isoformat(),
            "overall_status": self.overall_status.value,
            "overall_score": round(self.overall_score, 2),
            "indicators": [i.to_dict() for i in self.indicators],
            "critical_issues": self.critical_issues,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
            "devices_checked": self.devices_checked,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
        }

    def to_markdown(self) -> str:
        """Generate a markdown health report."""
        status_emoji = {
            HealthStatus.EXCELLENT: "🟢",
            HealthStatus.GOOD: "🟢",
            HealthStatus.FAIR: "🟡",
            HealthStatus.POOR: "🟠",
            HealthStatus.CRITICAL: "🔴",
        }

        lines = [
            "# Ara Health Report",
            "",
            f"Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M')}",
            "",
            f"## Overall Status: {status_emoji.get(self.overall_status, '⚪')} {self.overall_status.value.upper()}",
            f"Health Score: {self.overall_score:.0%}",
            "",
        ]

        if self.critical_issues:
            lines.append("## Critical Issues")
            for issue in self.critical_issues:
                lines.append(f"- {issue}")
            lines.append("")

        if self.warnings:
            lines.append("## Warnings")
            for warning in self.warnings:
                lines.append(f"- {warning}")
            lines.append("")

        if self.indicators:
            lines.append("## Health Indicators")
            lines.append("")
            lines.append("| Indicator | Score | Status | Trend |")
            lines.append("|-----------|-------|--------|-------|")
            for ind in self.indicators:
                trend_symbol = {"improving": "↗", "stable": "→", "degrading": "↘"}.get(ind.trend.value, "?")
                lines.append(f"| {ind.name} | {ind.value:.0%} | {ind.status.value} | {trend_symbol} |")
            lines.append("")

        if self.recommendations:
            lines.append("## Recommendations")
            for rec in self.recommendations:
                lines.append(f"- {rec}")

        return "\n".join(lines)


@dataclass
class HealthSnapshot:
    """A point-in-time health snapshot for trending."""

    timestamp: datetime
    overall_score: float
    device_scores: Dict[str, float]
    active_issues: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "overall_score": round(self.overall_score, 2),
            "device_scores": {k: round(v, 2) for k, v in self.device_scores.items()},
            "active_issues": self.active_issues,
        }


class HealthMonitor:
    """Monitors Ara's embodiment health."""

    def __init__(self, data_path: Optional[Path] = None):
        """Initialize the health monitor.

        Args:
            data_path: Path to health data
        """
        self.data_path = data_path or (
            Path.home() / ".ara" / "embodied" / "health"
        )
        self.data_path.mkdir(parents=True, exist_ok=True)

        self._snapshots: List[HealthSnapshot] = []
        self._reports: Dict[str, HealthReport] = {}
        self._next_report_id = 1

        # Health check functions
        self._checks: Dict[str, callable] = {}

    def _generate_report_id(self) -> str:
        """Generate unique report ID."""
        id_str = f"HEALTH-{self._next_report_id:06d}"
        self._next_report_id += 1
        return id_str

    def register_check(self, name: str, check_func: callable) -> None:
        """Register a health check function.

        Args:
            name: Check name
            check_func: Function that returns HealthIndicator
        """
        self._checks[name] = check_func

    def _score_to_status(self, score: float) -> HealthStatus:
        """Convert a score to a status."""
        if score >= 0.9:
            return HealthStatus.EXCELLENT
        elif score >= 0.75:
            return HealthStatus.GOOD
        elif score >= 0.5:
            return HealthStatus.FAIR
        elif score >= 0.25:
            return HealthStatus.POOR
        else:
            return HealthStatus.CRITICAL

    def run_health_check(
        self,
        device_graph=None,
        telemetry_adapter=None,
    ) -> HealthReport:
        """Run a comprehensive health check.

        Args:
            device_graph: Optional device graph to check
            telemetry_adapter: Optional telemetry adapter

        Returns:
            Health report
        """
        report = HealthReport(
            report_id=self._generate_report_id(),
        )

        scores = []
        device_scores: Dict[str, float] = {}

        # Check devices if graph provided
        if device_graph:
            from ..device_graph import DeviceStatus

            for device in device_graph._devices.values():
                report.devices_checked += 1

                # Calculate device health
                device_score = device.health_score

                # Adjust for status
                if device.status == DeviceStatus.ERROR:
                    device_score *= 0.2
                    report.critical_issues.append(f"Device {device.name} is in error state")
                    report.checks_failed += 1
                elif device.status == DeviceStatus.DEGRADED:
                    device_score *= 0.6
                    report.warnings.append(f"Device {device.name} is degraded")
                    report.checks_passed += 1
                elif device.status == DeviceStatus.OFFLINE:
                    device_score *= 0.5
                    report.warnings.append(f"Device {device.name} is offline")
                    report.checks_passed += 1
                else:
                    report.checks_passed += 1

                # Check temperature
                if device.temperature_c > 85:
                    device_score *= 0.7
                    report.warnings.append(f"Device {device.name} is hot ({device.temperature_c}°C)")

                device_scores[device.id] = device_score
                scores.append(device_score)

                # Create indicator
                indicator = HealthIndicator(
                    name=f"Device: {device.name}",
                    value=device_score,
                    status=self._score_to_status(device_score),
                    description=f"{device.device_type.value} - {device.status.value}",
                )
                report.indicators.append(indicator)

        # Run registered checks
        for check_name, check_func in self._checks.items():
            try:
                indicator = check_func()
                report.indicators.append(indicator)
                scores.append(indicator.value)

                if indicator.status == HealthStatus.CRITICAL:
                    report.critical_issues.extend(indicator.issues)
                elif indicator.status in [HealthStatus.POOR, HealthStatus.FAIR]:
                    report.warnings.extend(indicator.issues)

                report.recommendations.extend(indicator.recommendations)
            except Exception as e:
                logger.error(f"Health check '{check_name}' failed: {e}")
                report.checks_failed += 1

        # Calculate overall score
        if scores:
            report.overall_score = sum(scores) / len(scores)
        report.overall_status = self._score_to_status(report.overall_score)

        # Add general recommendations
        if report.overall_status == HealthStatus.CRITICAL:
            report.recommendations.insert(0, "Immediate attention required - critical systems affected")
        elif report.overall_status == HealthStatus.POOR:
            report.recommendations.insert(0, "Schedule maintenance soon - performance is degraded")

        # Store report
        self._reports[report.report_id] = report

        # Record snapshot
        self._snapshots.append(HealthSnapshot(
            timestamp=report.generated_at,
            overall_score=report.overall_score,
            device_scores=device_scores,
            active_issues=len(report.critical_issues) + len(report.warnings),
        ))

        # Trim old snapshots (keep 24 hours)
        cutoff = datetime.utcnow() - timedelta(hours=24)
        self._snapshots = [s for s in self._snapshots if s.timestamp > cutoff]

        logger.info(f"Health check complete: {report.overall_status.value} ({report.overall_score:.0%})")
        return report

    def get_trend(self, hours: int = 24) -> Dict[str, Any]:
        """Get health trend over time.

        Args:
            hours: Hours to analyze

        Returns:
            Trend data
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        relevant = [s for s in self._snapshots if s.timestamp > cutoff]

        if len(relevant) < 2:
            return {
                "direction": TrendDirection.UNKNOWN.value,
                "data_points": len(relevant),
                "message": "Insufficient data for trend analysis",
            }

        # Calculate trend
        first_half = relevant[:len(relevant)//2]
        second_half = relevant[len(relevant)//2:]

        avg_first = sum(s.overall_score for s in first_half) / len(first_half)
        avg_second = sum(s.overall_score for s in second_half) / len(second_half)

        change = avg_second - avg_first

        if change > 0.05:
            direction = TrendDirection.IMPROVING
        elif change < -0.05:
            direction = TrendDirection.DEGRADING
        else:
            direction = TrendDirection.STABLE

        return {
            "direction": direction.value,
            "change": round(change, 3),
            "data_points": len(relevant),
            "current_score": round(relevant[-1].overall_score, 2) if relevant else None,
            "min_score": round(min(s.overall_score for s in relevant), 2),
            "max_score": round(max(s.overall_score for s in relevant), 2),
        }

    def get_report(self, report_id: str) -> Optional[HealthReport]:
        """Get a report by ID."""
        return self._reports.get(report_id)

    def get_latest_report(self) -> Optional[HealthReport]:
        """Get the most recent report."""
        if not self._reports:
            return None
        return max(self._reports.values(), key=lambda r: r.generated_at)

    def get_summary(self) -> Dict[str, Any]:
        """Get health monitor summary."""
        latest = self.get_latest_report()
        trend = self.get_trend(hours=24)

        return {
            "latest_status": latest.overall_status.value if latest else "unknown",
            "latest_score": latest.overall_score if latest else None,
            "trend": trend["direction"],
            "total_reports": len(self._reports),
            "snapshots_stored": len(self._snapshots),
            "checks_registered": len(self._checks),
        }


# =============================================================================
# Convenience Functions
# =============================================================================

_default_monitor: Optional[HealthMonitor] = None


def get_health_monitor() -> HealthMonitor:
    """Get the default health monitor."""
    global _default_monitor
    if _default_monitor is None:
        _default_monitor = HealthMonitor()
    return _default_monitor


def check_health() -> HealthReport:
    """Run a quick health check."""
    return get_health_monitor().run_health_check()


def get_health_status() -> str:
    """Get current health status."""
    latest = get_health_monitor().get_latest_report()
    return latest.overall_status.value if latest else "unknown"


def is_healthy() -> bool:
    """Check if Ara is healthy."""
    latest = get_health_monitor().get_latest_report()
    if not latest:
        return True  # Assume healthy if no data
    return latest.overall_status in [HealthStatus.EXCELLENT, HealthStatus.GOOD]
