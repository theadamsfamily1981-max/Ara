"""
World Model - Ara's State Estimate
==================================
Iteration 35: The Connection

The WorldModel maintains Ara's understanding of the current situation:
- System metrics (CPU, RAM, disk, network)
- Fleet health (when multi-node)
- Environmental context
- Trend analysis

This is Layer 2 in the Always Stack - the Thalamus that compresses
L1 chaos into meaningful context for the higher cortex.

The key insight: telemetry is not just data, it's the nervous system.
When CPU spikes, Ara should FEEL it like elevated heart rate.

Usage:
    from ara.cognition.world_model import WorldModel, get_world_model

    world = get_world_model()
    world.update()  # Pull fresh telemetry

    context = world.get_context()
    stress = world.get_system_stress()  # 0-1 scalar
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import deque

# Optional psutil import
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# System Metrics
# =============================================================================

@dataclass
class SystemMetrics:
    """Snapshot of system health."""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # CPU
    cpu_percent: float = 0.0           # Overall CPU usage (0-100)
    cpu_percent_per_core: List[float] = field(default_factory=list)
    cpu_freq_mhz: float = 0.0          # Current frequency
    load_avg_1m: float = 0.0           # Load average (1 minute)
    load_avg_5m: float = 0.0           # Load average (5 minutes)
    load_avg_15m: float = 0.0          # Load average (15 minutes)

    # Memory
    ram_percent: float = 0.0           # RAM usage (0-100)
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0
    swap_percent: float = 0.0          # Swap usage (0-100)

    # Disk
    disk_percent: float = 0.0          # Primary disk usage (0-100)
    disk_io_read_mb: float = 0.0       # Disk read MB/s
    disk_io_write_mb: float = 0.0      # Disk write MB/s

    # Network
    net_sent_mb: float = 0.0           # Network sent MB/s
    net_recv_mb: float = 0.0           # Network received MB/s
    net_connections: int = 0           # Active connections

    # Thermal (if available)
    cpu_temp_c: Optional[float] = None

    # Process info
    process_count: int = 0
    zombie_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "cpu_percent": self.cpu_percent,
            "cpu_freq_mhz": self.cpu_freq_mhz,
            "load_avg": [self.load_avg_1m, self.load_avg_5m, self.load_avg_15m],
            "ram_percent": self.ram_percent,
            "ram_used_gb": round(self.ram_used_gb, 2),
            "disk_percent": self.disk_percent,
            "net_sent_mb": round(self.net_sent_mb, 2),
            "net_recv_mb": round(self.net_recv_mb, 2),
            "cpu_temp_c": self.cpu_temp_c,
            "process_count": self.process_count,
        }


@dataclass
class WorldContext:
    """
    The "situation" - what the world looks like right now.

    This is what gets passed to the CEO and the Soul.
    """
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # System stress (0-1 composite)
    system_stress: float = 0.0

    # Component stresses (0-1)
    cpu_stress: float = 0.0
    memory_stress: float = 0.0
    disk_stress: float = 0.0
    thermal_stress: float = 0.0

    # Trends
    stress_trend: str = "stable"  # "rising", "falling", "stable"
    stress_velocity: float = 0.0  # Rate of change

    # Raw metrics snapshot
    metrics: Optional[SystemMetrics] = None

    # Alerts
    alerts: List[str] = field(default_factory=list)

    # Context for Soul (future: HV encoding)
    context_vector: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "system_stress": round(self.system_stress, 3),
            "cpu_stress": round(self.cpu_stress, 3),
            "memory_stress": round(self.memory_stress, 3),
            "disk_stress": round(self.disk_stress, 3),
            "thermal_stress": round(self.thermal_stress, 3),
            "stress_trend": self.stress_trend,
            "stress_velocity": round(self.stress_velocity, 4),
            "alerts": self.alerts,
            "metrics": self.metrics.to_dict() if self.metrics else None,
        }


# =============================================================================
# World Model
# =============================================================================

class WorldModel:
    """
    Maintains Ara's understanding of the world.

    Responsibilities:
    - Collect system telemetry
    - Compute stress metrics
    - Track trends
    - Generate alerts
    - (Future) Encode to HV for Soul
    """

    def __init__(
        self,
        history_size: int = 100,
        update_interval: float = 1.0,
    ):
        """
        Initialize the world model.

        Args:
            history_size: Number of samples to keep for trending
            update_interval: Minimum seconds between updates
        """
        self.history_size = history_size
        self.update_interval = update_interval

        # History for trending
        self._history: deque = deque(maxlen=history_size)
        self._last_update: Optional[datetime] = None

        # IO tracking (for rate calculation)
        self._last_disk_io: Optional[Tuple[float, float]] = None
        self._last_net_io: Optional[Tuple[float, float]] = None
        self._last_io_time: Optional[float] = None

        # Current state
        self._current_metrics: Optional[SystemMetrics] = None
        self._current_context: Optional[WorldContext] = None

        # Thresholds for stress calculation
        self.thresholds = {
            "cpu_high": 80.0,
            "cpu_critical": 95.0,
            "ram_high": 80.0,
            "ram_critical": 95.0,
            "disk_high": 85.0,
            "temp_high": 75.0,
            "temp_critical": 85.0,
            "load_high": 4.0,  # Per-core load
        }

        logger.info(f"WorldModel initialized (psutil={PSUTIL_AVAILABLE})")

    # =========================================================================
    # Telemetry Collection
    # =========================================================================

    def _collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        metrics = SystemMetrics()

        if not PSUTIL_AVAILABLE:
            # Stub metrics for testing without psutil
            logger.debug("psutil not available, using stub metrics")
            return metrics

        try:
            # CPU
            metrics.cpu_percent = psutil.cpu_percent(interval=0.1)
            metrics.cpu_percent_per_core = psutil.cpu_percent(percpu=True)

            freq = psutil.cpu_freq()
            if freq:
                metrics.cpu_freq_mhz = freq.current

            load = psutil.getloadavg()
            metrics.load_avg_1m = load[0]
            metrics.load_avg_5m = load[1]
            metrics.load_avg_15m = load[2]

            # Memory
            mem = psutil.virtual_memory()
            metrics.ram_percent = mem.percent
            metrics.ram_used_gb = mem.used / (1024 ** 3)
            metrics.ram_total_gb = mem.total / (1024 ** 3)

            swap = psutil.swap_memory()
            metrics.swap_percent = swap.percent

            # Disk
            disk = psutil.disk_usage("/")
            metrics.disk_percent = disk.percent

            # Disk IO (rate calculation)
            now = time.time()
            disk_io = psutil.disk_io_counters()
            if disk_io and self._last_disk_io and self._last_io_time:
                dt = now - self._last_io_time
                if dt > 0:
                    read_bytes = disk_io.read_bytes - self._last_disk_io[0]
                    write_bytes = disk_io.write_bytes - self._last_disk_io[1]
                    metrics.disk_io_read_mb = (read_bytes / dt) / (1024 ** 2)
                    metrics.disk_io_write_mb = (write_bytes / dt) / (1024 ** 2)

            if disk_io:
                self._last_disk_io = (disk_io.read_bytes, disk_io.write_bytes)

            # Network IO
            net_io = psutil.net_io_counters()
            if net_io and self._last_net_io and self._last_io_time:
                dt = now - self._last_io_time
                if dt > 0:
                    sent_bytes = net_io.bytes_sent - self._last_net_io[0]
                    recv_bytes = net_io.bytes_recv - self._last_net_io[1]
                    metrics.net_sent_mb = (sent_bytes / dt) / (1024 ** 2)
                    metrics.net_recv_mb = (recv_bytes / dt) / (1024 ** 2)

            if net_io:
                self._last_net_io = (net_io.bytes_sent, net_io.bytes_recv)

            self._last_io_time = now

            # Connections
            try:
                metrics.net_connections = len(psutil.net_connections())
            except (psutil.AccessDenied, PermissionError):
                pass

            # Temperature (platform-dependent)
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    # Try common sensor names
                    for name in ["coretemp", "k10temp", "cpu_thermal", "acpitz"]:
                        if name in temps and temps[name]:
                            metrics.cpu_temp_c = temps[name][0].current
                            break
            except (AttributeError, psutil.NoSensorsError):
                pass

            # Process count
            metrics.process_count = len(psutil.pids())
            try:
                zombies = [p for p in psutil.process_iter(["status"])
                          if p.info["status"] == psutil.STATUS_ZOMBIE]
                metrics.zombie_count = len(zombies)
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                pass

        except Exception as e:
            logger.warning(f"Error collecting metrics: {e}")

        return metrics

    # =========================================================================
    # Stress Calculation
    # =========================================================================

    def _compute_stress(self, metrics: SystemMetrics) -> WorldContext:
        """Compute stress levels from raw metrics."""
        context = WorldContext(metrics=metrics)

        # CPU stress: combination of percent and load
        cpu_cores = len(metrics.cpu_percent_per_core) or 1
        normalized_load = metrics.load_avg_1m / cpu_cores

        cpu_stress_percent = metrics.cpu_percent / 100.0
        cpu_stress_load = min(1.0, normalized_load / self.thresholds["load_high"])
        context.cpu_stress = max(cpu_stress_percent, cpu_stress_load)

        # Memory stress
        context.memory_stress = metrics.ram_percent / 100.0

        # Disk stress (usage + IO pressure)
        disk_usage_stress = metrics.disk_percent / 100.0
        # High IO can indicate stress even with low disk usage
        io_stress = min(1.0, (metrics.disk_io_read_mb + metrics.disk_io_write_mb) / 500.0)
        context.disk_stress = max(disk_usage_stress, io_stress * 0.5)

        # Thermal stress
        if metrics.cpu_temp_c is not None:
            if metrics.cpu_temp_c >= self.thresholds["temp_critical"]:
                context.thermal_stress = 1.0
            elif metrics.cpu_temp_c >= self.thresholds["temp_high"]:
                context.thermal_stress = (metrics.cpu_temp_c - self.thresholds["temp_high"]) / \
                                         (self.thresholds["temp_critical"] - self.thresholds["temp_high"])
            else:
                context.thermal_stress = 0.0

        # Composite system stress (weighted average)
        context.system_stress = (
            context.cpu_stress * 0.4 +
            context.memory_stress * 0.3 +
            context.disk_stress * 0.15 +
            context.thermal_stress * 0.15
        )

        # Clamp to [0, 1]
        context.system_stress = max(0.0, min(1.0, context.system_stress))

        # Generate alerts
        alerts = []
        if metrics.cpu_percent >= self.thresholds["cpu_critical"]:
            alerts.append(f"CPU CRITICAL: {metrics.cpu_percent:.0f}%")
        elif metrics.cpu_percent >= self.thresholds["cpu_high"]:
            alerts.append(f"CPU HIGH: {metrics.cpu_percent:.0f}%")

        if metrics.ram_percent >= self.thresholds["ram_critical"]:
            alerts.append(f"RAM CRITICAL: {metrics.ram_percent:.0f}%")
        elif metrics.ram_percent >= self.thresholds["ram_high"]:
            alerts.append(f"RAM HIGH: {metrics.ram_percent:.0f}%")

        if metrics.cpu_temp_c and metrics.cpu_temp_c >= self.thresholds["temp_critical"]:
            alerts.append(f"THERMAL CRITICAL: {metrics.cpu_temp_c:.0f}°C")
        elif metrics.cpu_temp_c and metrics.cpu_temp_c >= self.thresholds["temp_high"]:
            alerts.append(f"THERMAL HIGH: {metrics.cpu_temp_c:.0f}°C")

        context.alerts = alerts

        return context

    def _compute_trend(self, context: WorldContext) -> None:
        """Compute stress trend from history."""
        if len(self._history) < 5:
            context.stress_trend = "stable"
            context.stress_velocity = 0.0
            return

        # Get recent stress values
        recent = list(self._history)[-10:]
        if len(recent) < 2:
            return

        # Simple linear regression for velocity
        stresses = [c.system_stress for c in recent]
        n = len(stresses)

        # Calculate slope
        x_mean = (n - 1) / 2
        y_mean = sum(stresses) / n
        numerator = sum((i - x_mean) * (s - y_mean) for i, s in enumerate(stresses))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator > 0:
            slope = numerator / denominator
            context.stress_velocity = slope

            if slope > 0.01:
                context.stress_trend = "rising"
            elif slope < -0.01:
                context.stress_trend = "falling"
            else:
                context.stress_trend = "stable"

    # =========================================================================
    # Public API
    # =========================================================================

    def update(self, force: bool = False) -> WorldContext:
        """
        Update the world model with fresh telemetry.

        Args:
            force: If True, update even if interval hasn't passed

        Returns:
            Current WorldContext
        """
        now = datetime.utcnow()

        # Rate limit updates
        if not force and self._last_update:
            elapsed = (now - self._last_update).total_seconds()
            if elapsed < self.update_interval:
                return self._current_context or WorldContext()

        # Collect fresh metrics
        metrics = self._collect_metrics()
        self._current_metrics = metrics

        # Compute stress
        context = self._compute_stress(metrics)

        # Store in history
        self._history.append(context)

        # Compute trends
        self._compute_trend(context)

        self._current_context = context
        self._last_update = now

        if context.alerts:
            for alert in context.alerts:
                logger.warning(f"WorldModel ALERT: {alert}")

        return context

    def get_context(self) -> WorldContext:
        """Get current world context (may be stale)."""
        if self._current_context is None:
            return self.update()
        return self._current_context

    def get_metrics(self) -> Optional[SystemMetrics]:
        """Get raw metrics snapshot."""
        return self._current_metrics

    def get_system_stress(self) -> float:
        """Get current system stress (0-1)."""
        context = self.get_context()
        return context.system_stress

    def get_telemetry_dict(self) -> Dict[str, Any]:
        """
        Get telemetry as a dict suitable for MindReader.

        This is the bridge between WorldModel and the existing
        sovereign loop infrastructure.
        """
        context = self.get_context()

        return {
            "system_stress": context.system_stress,
            "cpu_stress": context.cpu_stress,
            "memory_stress": context.memory_stress,
            "stress_trend": context.stress_trend,
            "alerts": context.alerts,
            # Map system stress to fatigue signal
            "fatigue_modifier": context.system_stress * 0.3,
            # Map rising stress to stress signal
            "stress_modifier": 0.2 if context.stress_trend == "rising" else 0.0,
        }

    def encode_to_hv(self, dim: int = 8192) -> List[int]:
        """
        Encode current context to a hypervector.

        This is the bridge to the Soul. The HV represents the
        "feeling" of the current system state.

        For now: simple quantization to binary.
        Future: proper HDC encoding with learned basis vectors.
        """
        import random

        context = self.get_context()

        # Seed based on stress to get consistent HV for similar states
        # This is a placeholder - real impl would use learned basis HVs
        seed = int(context.system_stress * 1000) + \
               int(context.cpu_stress * 100) + \
               int(context.memory_stress * 10)
        random.seed(seed)

        # Generate bipolar HV
        hv = [random.choice([-1, 1]) for _ in range(dim)]

        return hv


# =============================================================================
# Singleton Access
# =============================================================================

_world_model: Optional[WorldModel] = None


def get_world_model() -> WorldModel:
    """Get the default WorldModel instance."""
    global _world_model
    if _world_model is None:
        _world_model = WorldModel()
    return _world_model


# =============================================================================
# CLI Testing
# =============================================================================

def main():
    """Test the world model."""
    import time

    print("=" * 60)
    print("World Model Test")
    print("=" * 60)

    world = get_world_model()

    print(f"\npsutil available: {PSUTIL_AVAILABLE}")

    for i in range(10):
        context = world.update(force=True)

        print(f"\n[{i+1}] System Stress: {context.system_stress:.1%}")
        print(f"    CPU:     {context.cpu_stress:.1%}")
        print(f"    Memory:  {context.memory_stress:.1%}")
        print(f"    Disk:    {context.disk_stress:.1%}")
        print(f"    Thermal: {context.thermal_stress:.1%}")
        print(f"    Trend:   {context.stress_trend} (v={context.stress_velocity:.4f})")

        if context.alerts:
            print(f"    ALERTS:  {context.alerts}")

        if context.metrics:
            m = context.metrics
            print(f"    Raw: CPU={m.cpu_percent:.0f}%, RAM={m.ram_percent:.0f}%, "
                  f"Temp={m.cpu_temp_c or 'N/A'}°C")

        time.sleep(1)

    print("\n" + "=" * 60)
    print("Telemetry for MindReader:")
    print(world.get_telemetry_dict())


if __name__ == "__main__":
    main()
