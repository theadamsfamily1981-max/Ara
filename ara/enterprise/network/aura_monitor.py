"""
Aura Monitor - Holographic Stethoscope for the Fleet
=====================================================

Iteration 31 Integration: The Choir meets the Fleet

Uses the ensemble HDC fabric as a holographic stethoscope:
- Each machine/device becomes a hypervector
- Global health field bundles all machine states
- Query for resonance with known patterns
- High agreement → auto-act, Low agreement → escalate

This is where Corporation Croft's Network Monitor gets superpowers:
- Not just SNMP polling, but holographic anomaly detection
- Not just thresholds, but "does this FEEL like a past outage?"
- Not just alerts, but confidence-weighted recommendations

Usage:
    from ara.enterprise.network.aura_monitor import AuraMonitor

    aura = AuraMonitor(org_chart)

    # Continuous monitoring loop
    while True:
        aura.sample_fleet()
        health = aura.check_health()

        if health["overall"] == "degraded":
            for alert in health["alerts"]:
                if alert["severity"] == "critical":
                    ara.escalate_to_human(alert)
                else:
                    ara.log_warning(alert)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ara.neuro.binary.ensemble import FieldMonitor, EnsembleChoir, get_ensemble_choir

log = logging.getLogger(__name__)


@dataclass
class FleetSnapshot:
    """Point-in-time snapshot of fleet health."""
    timestamp: float
    baseline_resonance: float
    machine_health: Dict[str, float]
    alerts: List[Dict[str, Any]]
    overall: str


@dataclass
class AnomalyPattern:
    """Learned anomaly pattern."""
    name: str
    description: str
    hypervector: np.ndarray
    first_seen: float
    times_matched: int = 0


class AuraMonitor:
    """
    Holographic health monitor for the Fleet.

    Integrates with:
    - OrgChart: Gets list of machines to monitor
    - NetworkMonitor: Gets raw telemetry
    - FieldMonitor: Does HDC resonance computations
    - Boardroom: Reports health status

    The "aura" is a bundled hypervector representing the gestalt
    state of all machines in the fleet.
    """

    def __init__(
        self,
        org_chart: Optional[Any] = None,
        choir: Optional[EnsembleChoir] = None,
        hv_dim: int = 8192,
        sample_interval_sec: float = 30.0,
        history_size: int = 100,
    ):
        """
        Initialize the aura monitor.

        Args:
            org_chart: OrgChart instance (optional, will try to load)
            choir: EnsembleChoir instance (optional, will create)
            hv_dim: Hypervector dimension
            sample_interval_sec: How often to sample
            history_size: How many snapshots to keep
        """
        self.hv_dim = hv_dim
        self.sample_interval_sec = sample_interval_sec

        # Try to get OrgChart
        if org_chart is None:
            try:
                from ara.enterprise import get_org_chart
                org_chart = get_org_chart()
            except ImportError:
                log.warning("AuraMonitor: OrgChart not available")
                org_chart = None

        self.org_chart = org_chart

        # Initialize choir and field monitor
        self.choir = choir or get_ensemble_choir()
        self.field_monitor = FieldMonitor(self.choir, hv_dim=hv_dim)

        # State
        self.last_sample_time = 0.0
        self.history: List[FleetSnapshot] = []
        self.history_size = history_size

        # Learned patterns
        self.anomaly_patterns: Dict[str, AnomalyPattern] = {}

        # Pre-register some common bad patterns
        self._init_default_patterns()

        log.info(f"AuraMonitor initialized: dim={hv_dim}, interval={sample_interval_sec}s")

    def _init_default_patterns(self) -> None:
        """Initialize default bad patterns."""
        # High entropy / chaos pattern (lots of random activity)
        chaos_pattern = np.random.default_rng(42).integers(0, 2, self.hv_dim, dtype=np.uint8)
        self.register_anomaly("chaos_entropy", "High entropy / chaotic state", chaos_pattern)

        # All-down pattern (everything offline)
        down_pattern = np.zeros(self.hv_dim, dtype=np.uint8)
        self.register_anomaly("all_down", "All systems down", down_pattern)

        # Overload pattern (everything maxed out)
        overload_pattern = np.ones(self.hv_dim, dtype=np.uint8)
        self.register_anomaly("overload", "All systems overloaded", overload_pattern)

    def register_anomaly(
        self,
        name: str,
        description: str,
        pattern: Optional[np.ndarray] = None,
    ) -> None:
        """
        Register a known anomaly pattern.

        Args:
            name: Unique name for this pattern
            description: Human-readable description
            pattern: Hypervector (if None, learns from current state)
        """
        if pattern is None:
            # Learn from current state
            pattern = self.field_monitor.health_field.copy()
            pattern = (pattern > 0).astype(np.uint8)

        self.anomaly_patterns[name] = AnomalyPattern(
            name=name,
            description=description,
            hypervector=pattern,
            first_seen=time.time(),
        )

        # Also register with field monitor
        self.field_monitor.add_bad_attractor(name, pattern)

        log.info(f"AuraMonitor: Registered anomaly pattern '{name}'")

    def sample_fleet(self) -> Dict[str, Any]:
        """
        Sample current state from all fleet machines.

        Returns:
            Dict of machine_id → state
        """
        states = {}

        if self.org_chart is None:
            # No OrgChart, return synthetic data
            states["synthetic-node-1"] = {
                "cpu": 45.0,
                "memory": 60.0,
                "errors": 0,
                "status": "online",
            }
            return states

        # Sample from each employee
        for emp_id, employee in self.org_chart.employees.items():
            state = self._sample_machine(employee)
            if state:
                states[emp_id] = state

        return states

    def _sample_machine(self, employee: Any) -> Optional[Dict[str, Any]]:
        """Sample state from a single machine."""
        # In a real implementation, this would:
        # 1. SSH to the machine and run health checks
        # 2. Query SNMP/metrics endpoints
        # 3. Check recent log patterns
        #
        # For now, we generate synthetic state based on employee properties

        try:
            state = {
                "status": employee.status,
                "role": employee.role.value if hasattr(employee.role, 'value') else str(employee.role),
                "capabilities": len(employee.capabilities),
                "allow_sudo": 1 if employee.allow_sudo else 0,
                "allow_internet": 1 if employee.allow_internet else 0,
            }

            # Add synthetic metrics
            if employee.status == "online":
                rng = np.random.default_rng(hash(employee.id) % (2**32))
                state["cpu"] = rng.uniform(10, 80)
                state["memory"] = rng.uniform(20, 70)
                state["errors"] = rng.integers(0, 5)
            else:
                state["cpu"] = 0.0
                state["memory"] = 0.0
                state["errors"] = 100  # High error count for offline

            return state

        except Exception as e:
            log.warning(f"AuraMonitor: Failed to sample {employee.id}: {e}")
            return None

    def update_aura(
        self,
        states: Optional[Dict[str, Any]] = None,
    ) -> FleetSnapshot:
        """
        Update the fleet aura from current states.

        Args:
            states: Machine states (if None, will sample)

        Returns:
            FleetSnapshot with health info
        """
        if states is None:
            states = self.sample_fleet()

        # Encode each machine state as hypervector
        for machine_id, state in states.items():
            self.field_monitor.encode_machine_state(machine_id, state)

        # Check health
        health = self.field_monitor.check_health()

        # Create snapshot
        snapshot = FleetSnapshot(
            timestamp=time.time(),
            baseline_resonance=health["baseline_resonance"],
            machine_health=health["machine_health"],
            alerts=health["alerts"],
            overall=health["overall"],
        )

        # Store in history
        self.history.append(snapshot)
        if len(self.history) > self.history_size:
            self.history.pop(0)

        self.last_sample_time = time.time()

        return snapshot

    def check_health(self) -> Dict[str, Any]:
        """
        Check current fleet health.

        Returns:
            Health report with resonances, alerts, recommendations
        """
        # Sample if needed
        if time.time() - self.last_sample_time > self.sample_interval_sec:
            self.update_aura()

        if not self.history:
            return {
                "overall": "unknown",
                "message": "No samples yet",
            }

        latest = self.history[-1]

        # Build report
        report = {
            "overall": latest.overall,
            "baseline_resonance": latest.baseline_resonance,
            "alerts": latest.alerts,
            "machine_health": latest.machine_health,
            "timestamp": latest.timestamp,
        }

        # Add recommendations based on confidence
        recommendations = []

        if latest.baseline_resonance < 0.5:
            recommendations.append({
                "action": "escalate",
                "reason": "Significant drift from healthy baseline",
                "confidence": 1.0 - latest.baseline_resonance,
            })
        elif latest.baseline_resonance < 0.7:
            recommendations.append({
                "action": "investigate",
                "reason": "Moderate drift from baseline",
                "confidence": 0.7 - latest.baseline_resonance + 0.3,
            })

        # Check for consistently unhealthy machines
        for machine_id, health in latest.machine_health.items():
            if health < 0.6:
                recommendations.append({
                    "action": "check_machine",
                    "target": machine_id,
                    "reason": f"Low health score ({health:.2f})",
                    "confidence": 1.0 - health,
                })

        report["recommendations"] = recommendations

        return report

    def learn_healthy_baseline(self) -> None:
        """
        Learn what "healthy" looks like from current state.

        Call this when the fleet is known to be in a good state.
        """
        # Sample current state
        self.update_aura()

        # Learn baseline
        self.field_monitor.learn_baseline()

        log.info("AuraMonitor: Learned healthy baseline from current state")

    def get_trend(self, window_size: int = 10) -> Dict[str, Any]:
        """
        Get health trend over recent history.

        Args:
            window_size: Number of samples to analyze

        Returns:
            Trend analysis
        """
        if len(self.history) < 2:
            return {"trend": "unknown", "samples": len(self.history)}

        recent = self.history[-window_size:]

        resonances = [s.baseline_resonance for s in recent]
        alert_counts = [len(s.alerts) for s in recent]

        # Calculate trend
        if len(resonances) >= 3:
            # Simple linear regression
            x = np.arange(len(resonances))
            slope = np.polyfit(x, resonances, 1)[0]

            if slope > 0.01:
                trend = "improving"
            elif slope < -0.01:
                trend = "degrading"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"

        return {
            "trend": trend,
            "current_resonance": resonances[-1] if resonances else 0.0,
            "avg_resonance": np.mean(resonances) if resonances else 0.0,
            "avg_alerts": np.mean(alert_counts) if alert_counts else 0.0,
            "samples": len(recent),
        }

    def get_machine_detail(self, machine_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed aura analysis for a specific machine."""
        return self.field_monitor.get_machine_aura(machine_id)

    def generate_report(self) -> str:
        """Generate human-readable health report."""
        health = self.check_health()
        trend = self.get_trend()

        lines = [
            "=" * 50,
            "FLEET AURA REPORT",
            "=" * 50,
            "",
            f"Overall Status: {health['overall'].upper()}",
            f"Baseline Resonance: {health.get('baseline_resonance', 0):.2f}",
            f"Trend: {trend['trend']}",
            "",
        ]

        if health.get("alerts"):
            lines.append("ALERTS:")
            for alert in health["alerts"]:
                lines.append(f"  [{alert['severity'].upper()}] {alert['message']}")
            lines.append("")

        if health.get("recommendations"):
            lines.append("RECOMMENDATIONS:")
            for rec in health["recommendations"]:
                target = f" ({rec['target']})" if rec.get('target') else ""
                lines.append(f"  - {rec['action'].upper()}{target}: {rec['reason']}")
                lines.append(f"    Confidence: {rec['confidence']:.2f}")
            lines.append("")

        if health.get("machine_health"):
            lines.append("MACHINE HEALTH:")
            for machine, score in sorted(health["machine_health"].items(),
                                         key=lambda x: x[1]):
                status = "OK" if score > 0.7 else "WARN" if score > 0.5 else "CRIT"
                lines.append(f"  {machine}: {score:.2f} [{status}]")

        lines.append("")
        lines.append("=" * 50)

        return "\n".join(lines)


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_global_aura_monitor: Optional[AuraMonitor] = None


def get_aura_monitor() -> AuraMonitor:
    """Get or create the global aura monitor."""
    global _global_aura_monitor
    if _global_aura_monitor is None:
        _global_aura_monitor = AuraMonitor()
    return _global_aura_monitor


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'FleetSnapshot',
    'AnomalyPattern',
    'AuraMonitor',
    'get_aura_monitor',
]
