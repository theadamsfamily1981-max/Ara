"""
Cathedral Resource Scheduler
=============================

Optimizes routing between classical/quantum paths to maximize
useful work per clock and per joule.

Metrics tracked:
    - Per-device utilization (GPU, FPGA, QPU)
    - Latency distributions (p50, p95, p99)
    - Waste ratio (idle + failed / total)
    - Over/under-routing rates
    - Energy efficiency (J/op)

Knobs:
    - Max quantum calls per job
    - Max wall-clock per kernel
    - Retry budget
    - Dynamic backoff thresholds
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Deque
from collections import deque
from enum import Enum
from datetime import datetime, timedelta


class DeviceType(str, Enum):
    """Compute device types."""
    GPU = "gpu"
    FPGA = "fpga"
    QPU = "qpu"       # Quantum processing unit
    CPU = "cpu"


class JobStatus(str, Enum):
    """Job execution status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    RERUN = "rerun"      # Had to be redone
    FALLBACK = "fallback"  # Fell back to classical


@dataclass
class JobTiming:
    """Timing record for a single job."""
    job_id: str
    device: DeviceType
    start_time: float
    end_time: float = 0.0
    status: JobStatus = JobStatus.PENDING
    quantum_calls: int = 0
    retries: int = 0
    energy_joules: float = 0.0

    @property
    def latency_ms(self) -> float:
        if self.end_time > 0:
            return (self.end_time - self.start_time) * 1000
        return 0.0

    @property
    def is_waste(self) -> bool:
        return self.status in [JobStatus.FAILED, JobStatus.RERUN]


@dataclass
class DeviceStats:
    """Statistics for a single device."""
    device: DeviceType
    total_time_sec: float = 0.0
    active_time_sec: float = 0.0
    idle_time_sec: float = 0.0
    failed_time_sec: float = 0.0
    jobs_completed: int = 0
    jobs_failed: int = 0
    total_energy_joules: float = 0.0

    @property
    def utilization(self) -> float:
        if self.total_time_sec == 0:
            return 0.0
        return self.active_time_sec / self.total_time_sec

    @property
    def waste_ratio(self) -> float:
        if self.total_time_sec == 0:
            return 0.0
        return (self.idle_time_sec + self.failed_time_sec) / self.total_time_sec

    @property
    def energy_per_op(self) -> float:
        if self.jobs_completed == 0:
            return float('inf')
        return self.total_energy_joules / self.jobs_completed


@dataclass
class RoutingMetrics:
    """Metrics for routing quality."""
    total_routed: int = 0
    quantum_routed: int = 0
    classical_routed: int = 0

    # Over-routing: went quantum but had to redo classically
    over_routed: int = 0

    # Under-routing: stayed classical but would have been faster quantum
    # (detected retrospectively)
    under_routed: int = 0

    @property
    def over_routing_rate(self) -> float:
        if self.quantum_routed == 0:
            return 0.0
        return self.over_routed / self.quantum_routed

    @property
    def under_routing_rate(self) -> float:
        if self.classical_routed == 0:
            return 0.0
        return self.under_routed / self.classical_routed


class LatencyTracker:
    """Track latency distributions."""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.latencies: Dict[DeviceType, Deque[float]] = {
            d: deque(maxlen=window_size) for d in DeviceType
        }

    def record(self, device: DeviceType, latency_ms: float):
        self.latencies[device].append(latency_ms)

    def percentile(self, device: DeviceType, p: float) -> float:
        """Get latency percentile for device."""
        vals = list(self.latencies[device])
        if not vals:
            return 0.0
        return float(np.percentile(vals, p))

    def p50(self, device: DeviceType) -> float:
        return self.percentile(device, 50)

    def p95(self, device: DeviceType) -> float:
        return self.percentile(device, 95)

    def p99(self, device: DeviceType) -> float:
        return self.percentile(device, 99)

    def stats(self, device: DeviceType) -> Dict[str, float]:
        return {
            "p50": self.p50(device),
            "p95": self.p95(device),
            "p99": self.p99(device),
            "count": len(self.latencies[device]),
        }


@dataclass
class SchedulerConfig:
    """Configuration for scheduler knobs."""
    # Hard caps
    max_quantum_calls_per_job: int = 10
    max_kernel_wall_clock_ms: float = 5000.0
    max_retry_budget: int = 3

    # Thresholds for dynamic backoff
    queue_depth_threshold: int = 100
    latency_p99_threshold_ms: float = 2000.0
    waste_ratio_threshold: float = 0.2

    # Energy budgets
    max_energy_per_job_joules: float = 100.0

    # Routing preferences
    prefer_quantum_speedup_min: float = 1.5  # Only route to quantum if expected speedup > 1.5x


class ResourceScheduler:
    """
    Main scheduler for routing jobs between classical and quantum paths.

    Maximizes useful work per second and per joule while avoiding clock abuse.
    """

    def __init__(self, config: Optional[SchedulerConfig] = None):
        self.config = config or SchedulerConfig()

        # Device stats
        self.device_stats: Dict[DeviceType, DeviceStats] = {
            d: DeviceStats(device=d) for d in DeviceType
        }

        # Latency tracking
        self.latency_tracker = LatencyTracker()

        # Routing metrics
        self.routing = RoutingMetrics()

        # Job history
        self.job_history: Deque[JobTiming] = deque(maxlen=10000)

        # Queue depths
        self.queue_depths: Dict[DeviceType, int] = {d: 0 for d in DeviceType}

        # Backoff state
        self.quantum_backoff_until: float = 0.0

    def should_route_quantum(self, job_type: str, expected_speedup: float) -> bool:
        """Decide if job should be routed to quantum path."""
        # Check backoff
        if time.time() < self.quantum_backoff_until:
            return False

        # Check queue depth
        if self.queue_depths[DeviceType.QPU] > self.config.queue_depth_threshold:
            return False

        # Check latency threshold
        if self.latency_tracker.p99(DeviceType.QPU) > self.config.latency_p99_threshold_ms:
            return False

        # Check waste ratio
        qpu_stats = self.device_stats[DeviceType.QPU]
        if qpu_stats.waste_ratio > self.config.waste_ratio_threshold:
            return False

        # Check expected speedup
        if expected_speedup < self.config.prefer_quantum_speedup_min:
            return False

        return True

    def route_job(self, job_id: str, job_type: str,
                  expected_quantum_speedup: float = 1.0) -> DeviceType:
        """Route a job to appropriate device."""
        self.routing.total_routed += 1

        if self.should_route_quantum(job_type, expected_quantum_speedup):
            self.routing.quantum_routed += 1
            self.queue_depths[DeviceType.QPU] += 1
            return DeviceType.QPU
        else:
            self.routing.classical_routed += 1
            # Choose between GPU and FPGA based on availability
            if self.queue_depths[DeviceType.GPU] <= self.queue_depths[DeviceType.FPGA]:
                self.queue_depths[DeviceType.GPU] += 1
                return DeviceType.GPU
            else:
                self.queue_depths[DeviceType.FPGA] += 1
                return DeviceType.FPGA

    def start_job(self, job_id: str, device: DeviceType) -> JobTiming:
        """Record job start."""
        timing = JobTiming(
            job_id=job_id,
            device=device,
            start_time=time.time(),
            status=JobStatus.RUNNING,
        )
        return timing

    def complete_job(self, timing: JobTiming, success: bool,
                     energy_joules: float = 0.0,
                     quantum_calls: int = 0):
        """Record job completion."""
        timing.end_time = time.time()
        timing.energy_joules = energy_joules
        timing.quantum_calls = quantum_calls

        if success:
            timing.status = JobStatus.SUCCESS
        else:
            timing.status = JobStatus.FAILED

        # Update stats
        self._update_stats(timing)

        # Record latency
        self.latency_tracker.record(timing.device, timing.latency_ms)

        # Update queue depth
        self.queue_depths[timing.device] = max(0, self.queue_depths[timing.device] - 1)

        # Store in history
        self.job_history.append(timing)

        # Check if we need to trigger backoff
        self._check_backoff()

    def record_fallback(self, job_id: str, from_device: DeviceType):
        """Record a fallback from quantum to classical."""
        self.routing.over_routed += 1

        # Update queue
        self.queue_depths[from_device] = max(0, self.queue_depths[from_device] - 1)

    def record_retrospective_underroute(self, job_id: str):
        """Record that a classical job would have been faster on quantum."""
        self.routing.under_routed += 1

    def _update_stats(self, timing: JobTiming):
        """Update device statistics."""
        stats = self.device_stats[timing.device]
        duration = timing.end_time - timing.start_time

        stats.total_time_sec += duration
        stats.total_energy_joules += timing.energy_joules

        if timing.status == JobStatus.SUCCESS:
            stats.active_time_sec += duration
            stats.jobs_completed += 1
        else:
            stats.failed_time_sec += duration
            stats.jobs_failed += 1

    def _check_backoff(self):
        """Check if we should back off from quantum routing."""
        qpu_stats = self.device_stats[DeviceType.QPU]

        # Trigger backoff if waste ratio too high
        if qpu_stats.waste_ratio > self.config.waste_ratio_threshold:
            # Back off for 10 seconds
            self.quantum_backoff_until = time.time() + 10.0

        # Trigger backoff if p99 latency too high
        if self.latency_tracker.p99(DeviceType.QPU) > self.config.latency_p99_threshold_ms:
            self.quantum_backoff_until = time.time() + 5.0

    def get_metrics(self) -> Dict[str, Any]:
        """Get all scheduler metrics."""
        return {
            "devices": {
                d.value: {
                    "utilization": self.device_stats[d].utilization,
                    "waste_ratio": self.device_stats[d].waste_ratio,
                    "energy_per_op": self.device_stats[d].energy_per_op,
                    "jobs_completed": self.device_stats[d].jobs_completed,
                    "jobs_failed": self.device_stats[d].jobs_failed,
                    "latency": self.latency_tracker.stats(d),
                }
                for d in DeviceType
            },
            "routing": {
                "total": self.routing.total_routed,
                "quantum_fraction": (self.routing.quantum_routed / max(1, self.routing.total_routed)),
                "over_routing_rate": self.routing.over_routing_rate,
                "under_routing_rate": self.routing.under_routing_rate,
            },
            "queue_depths": {d.value: self.queue_depths[d] for d in DeviceType},
            "backoff_active": time.time() < self.quantum_backoff_until,
        }

    def render_dashboard(self) -> str:
        """Render scheduler status as ASCII dashboard."""
        metrics = self.get_metrics()
        lines = []

        lines.append("╔══════════════════════════════════════════════════════════════════════════╗")
        lines.append("║  CATHEDRAL RESOURCE SCHEDULER                                            ║")
        lines.append("╠══════════════════════════════════════════════════════════════════════════╣")

        # Per-device stats
        for d in [DeviceType.GPU, DeviceType.FPGA, DeviceType.QPU]:
            dm = metrics["devices"][d.value]
            util_bar = self._bar(dm["utilization"])
            waste_bar = self._bar(dm["waste_ratio"])
            p99 = dm["latency"]["p99"]

            lines.append(f"║  {d.value.upper():<4}  util={dm['utilization']:.1%} {util_bar}  waste={dm['waste_ratio']:.1%} {waste_bar}  p99={p99:.0f}ms  ║")

        lines.append("╠══════════════════════════════════════════════════════════════════════════╣")

        # Routing stats
        rm = metrics["routing"]
        lines.append(f"║  ROUTING  quantum={rm['quantum_fraction']:.1%}  over={rm['over_routing_rate']:.1%}  under={rm['under_routing_rate']:.1%}         ║")

        backoff = "BACKOFF" if metrics["backoff_active"] else "NORMAL"
        lines.append(f"║  STATUS   {backoff:<8}  queues: GPU={metrics['queue_depths']['gpu']} FPGA={metrics['queue_depths']['fpga']} QPU={metrics['queue_depths']['qpu']}       ║")

        lines.append("╚══════════════════════════════════════════════════════════════════════════╝")

        return "\n".join(lines)

    def _bar(self, value: float, width: int = 8) -> str:
        filled = int(value * width)
        return "[" + "█" * filled + "░" * (width - filled) + "]"


# =============================================================================
# CONVENIENCE / SINGLETON
# =============================================================================

_scheduler: Optional[ResourceScheduler] = None


def get_scheduler() -> ResourceScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = ResourceScheduler()
    return _scheduler


def route_job(job_id: str, job_type: str, expected_speedup: float = 1.0) -> DeviceType:
    """Route a job to appropriate device."""
    return get_scheduler().route_job(job_id, job_type, expected_speedup)


def render_scheduler_dashboard() -> str:
    """Render scheduler status."""
    return get_scheduler().render_dashboard()
