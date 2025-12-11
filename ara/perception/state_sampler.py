# ara/perception/state_sampler.py
"""
State Sampler - Cathedral + User State at Every Tick
=====================================================

Samples the full "now" state every 250-500ms for the clairvoyant control loop:
- Hardware: GPU load/temp/power, CPU load/temp, FPGA status
- System: IO pressure, network, disk queue, active jobs
- Agents: Running count, blocked, erroring, HiveHD workflow state
- User: Input rate, window focus, neurostate (stress, focus, fatigue)

This produces the raw feature vector f_t that gets hypervector-encoded.

Usage:
    from ara.perception.state_sampler import StateSampler, get_state_sampler

    sampler = get_state_sampler()
    features = await sampler.sample()  # Dict[str, float]
    # {'gpu0.load': 0.72, 'gpu0.temp': 68.0, 'user.input_rate': 0.3, ...}
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from collections import deque

logger = logging.getLogger(__name__)


# =============================================================================
# Hardware Metrics (GPU, FPGA, etc.)
# =============================================================================

@dataclass
class GPUMetrics:
    """Metrics for a single GPU."""
    index: int = 0
    name: str = "unknown"
    load_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    temperature_c: float = 0.0
    power_watts: float = 0.0
    fan_percent: float = 0.0


@dataclass
class FPGAMetrics:
    """Metrics for FPGA accelerator."""
    online: bool = False
    link_status: str = "unknown"
    throughput_gbps: float = 0.0
    error_count: int = 0
    temperature_c: float = 0.0


def _collect_gpu_metrics() -> List[GPUMetrics]:
    """Collect GPU metrics via pynvml or fallback."""
    gpus = []

    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)

            gpu = GPUMetrics(index=i)
            gpu.name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(gpu.name, bytes):
                gpu.name = gpu.name.decode('utf-8')

            # Utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu.load_percent = util.gpu

            # Memory
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu.memory_used_mb = mem.used / (1024 ** 2)
            gpu.memory_total_mb = mem.total / (1024 ** 2)
            gpu.memory_percent = (mem.used / mem.total) * 100 if mem.total > 0 else 0

            # Temperature
            try:
                gpu.temperature_c = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
            except:
                pass

            # Power
            try:
                gpu.power_watts = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            except:
                pass

            # Fan
            try:
                gpu.fan_percent = pynvml.nvmlDeviceGetFanSpeed(handle)
            except:
                pass

            gpus.append(gpu)

        pynvml.nvmlShutdown()
    except ImportError:
        logger.debug("pynvml not available, GPU metrics disabled")
    except Exception as e:
        logger.debug(f"GPU metrics error: {e}")

    return gpus


def _collect_fpga_metrics() -> Optional[FPGAMetrics]:
    """Collect FPGA metrics (placeholder for now)."""
    # TODO: Integrate with actual FPGA monitoring (OpenCL, Xilinx runtime, etc.)
    return None


# =============================================================================
# User Metrics (Input Rate, Focus, etc.)
# =============================================================================

@dataclass
class UserMetrics:
    """Metrics about the user's current state."""
    input_rate: float = 0.0        # Keys/clicks per second (0-1 normalized)
    window_focus: str = "unknown"  # Active window category
    idle_seconds: float = 0.0      # Seconds since last input

    # From NeuroState (if brainlink connected)
    attention: float = 0.5
    stress: float = 0.5
    engagement: float = 0.5
    fatigue: float = 0.5


class InputTracker:
    """Tracks user input rate over time."""

    def __init__(self, window_seconds: float = 5.0):
        self.window_seconds = window_seconds
        self._events: deque = deque()
        self._last_event_time: float = time.time()

    def record_event(self, event_type: str = "key"):
        """Record an input event."""
        now = time.time()
        self._events.append(now)
        self._last_event_time = now
        self._prune_old()

    def _prune_old(self):
        """Remove events outside the window."""
        cutoff = time.time() - self.window_seconds
        while self._events and self._events[0] < cutoff:
            self._events.popleft()

    def get_rate(self) -> float:
        """Get events per second, normalized to 0-1."""
        self._prune_old()
        rate = len(self._events) / self.window_seconds
        # Normalize: 0 events = 0, 10+ events/sec = 1
        return min(1.0, rate / 10.0)

    def get_idle_seconds(self) -> float:
        """Get seconds since last input."""
        return time.time() - self._last_event_time


# =============================================================================
# Agent Metrics (HiveHD, Jobs, etc.)
# =============================================================================

@dataclass
class AgentMetrics:
    """Metrics about running agents and jobs."""
    agents_running: int = 0
    agents_blocked: int = 0
    agents_erroring: int = 0
    jobs_queued: int = 0
    jobs_active: int = 0
    current_workflow: str = "idle"
    workflow_progress: float = 0.0


def _collect_agent_metrics() -> AgentMetrics:
    """Collect agent/job metrics from HiveHD (placeholder)."""
    # TODO: Integrate with actual HiveHD job scheduler
    return AgentMetrics()


# =============================================================================
# State Sampler
# =============================================================================

@dataclass
class StateSample:
    """Complete state sample at a single tick."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    features: Dict[str, float] = field(default_factory=dict)

    # Raw metrics (for debugging/logging)
    gpus: List[GPUMetrics] = field(default_factory=list)
    fpga: Optional[FPGAMetrics] = None
    user: UserMetrics = field(default_factory=UserMetrics)
    agents: AgentMetrics = field(default_factory=AgentMetrics)
    system: Dict[str, float] = field(default_factory=dict)


class StateSampler:
    """
    Samples the full cathedral + user state at every tick.

    This is the "sensory cortex" that feeds the clairvoyant control loop.
    Every 250-500ms, it produces a feature dict that gets hypervector-encoded.
    """

    def __init__(
        self,
        tick_interval: float = 0.5,
        enable_gpu: bool = True,
        enable_fpga: bool = False,
        enable_neurostate: bool = True,
    ):
        """
        Initialize the state sampler.

        Args:
            tick_interval: Seconds between samples (default 0.5)
            enable_gpu: Enable GPU metrics via pynvml
            enable_fpga: Enable FPGA metrics
            enable_neurostate: Enable neurostate from brainlink
        """
        self.tick_interval = tick_interval
        self.enable_gpu = enable_gpu
        self.enable_fpga = enable_fpga
        self.enable_neurostate = enable_neurostate

        # Input tracking
        self._input_tracker = InputTracker()

        # World model (for CPU/RAM/disk)
        self._world_model = None

        # Neurostate extractor
        self._neurostate_extractor = None

        # Last sample
        self._last_sample: Optional[StateSample] = None
        self._last_sample_time: float = 0.0

        logger.info(f"StateSampler initialized: tick={tick_interval}s")

    def _get_world_model(self):
        """Lazy load world model."""
        if self._world_model is None:
            try:
                from ara.cognition.world_model import get_world_model
                self._world_model = get_world_model()
            except ImportError:
                logger.warning("WorldModel not available")
        return self._world_model

    def _get_neurostate_extractor(self):
        """Lazy load neurostate extractor."""
        if self._neurostate_extractor is None and self.enable_neurostate:
            try:
                from ara.perception.neurostate import get_neurostate_extractor
                self._neurostate_extractor = get_neurostate_extractor()
            except ImportError:
                logger.debug("NeuroState extractor not available")
        return self._neurostate_extractor

    def record_input(self, event_type: str = "key"):
        """Record a user input event (called from input hooks)."""
        self._input_tracker.record_event(event_type)

    async def sample(self, force: bool = False) -> Dict[str, float]:
        """
        Sample current state and return feature dict.

        Args:
            force: Sample even if tick interval hasn't elapsed

        Returns:
            Dict mapping feature names to normalized values.
            Feature names follow the pattern: category.metric
            e.g., 'gpu0.load', 'user.stress', 'system.cpu_percent'
        """
        now = time.time()

        # Rate limit unless forced
        if not force and (now - self._last_sample_time) < self.tick_interval:
            if self._last_sample:
                return self._last_sample.features
            return {}

        features: Dict[str, float] = {}
        sample = StateSample()

        # =================================================================
        # GPU Metrics
        # =================================================================
        if self.enable_gpu:
            gpus = _collect_gpu_metrics()
            sample.gpus = gpus

            for gpu in gpus:
                prefix = f"gpu{gpu.index}"
                features[f"{prefix}.load"] = gpu.load_percent / 100.0
                features[f"{prefix}.memory"] = gpu.memory_percent / 100.0
                features[f"{prefix}.temp"] = min(1.0, gpu.temperature_c / 100.0)
                features[f"{prefix}.power"] = min(1.0, gpu.power_watts / 500.0)  # Normalize to 500W max

        # =================================================================
        # FPGA Metrics
        # =================================================================
        if self.enable_fpga:
            fpga = _collect_fpga_metrics()
            sample.fpga = fpga

            if fpga:
                features["fpga.online"] = 1.0 if fpga.online else 0.0
                features["fpga.throughput"] = min(1.0, fpga.throughput_gbps / 100.0)
                features["fpga.errors"] = min(1.0, fpga.error_count / 100.0)

        # =================================================================
        # System Metrics (CPU, RAM, Disk, Network)
        # =================================================================
        world = self._get_world_model()
        if world:
            ctx = world.update(force=True)
            sample.system = {
                "cpu_percent": ctx.metrics.cpu_percent if ctx.metrics else 0,
                "ram_percent": ctx.metrics.ram_percent if ctx.metrics else 0,
                "disk_percent": ctx.metrics.disk_percent if ctx.metrics else 0,
            }

            features["system.cpu"] = ctx.cpu_stress
            features["system.memory"] = ctx.memory_stress
            features["system.disk"] = ctx.disk_stress
            features["system.thermal"] = ctx.thermal_stress
            features["system.stress"] = ctx.system_stress

            # Load average (normalized to core count)
            if ctx.metrics and ctx.metrics.cpu_percent_per_core:
                core_count = len(ctx.metrics.cpu_percent_per_core)
                if core_count > 0 and ctx.metrics.load_avg_1m:
                    features["system.load"] = min(1.0, ctx.metrics.load_avg_1m / core_count)

            # IO rates (normalized)
            if ctx.metrics:
                features["system.disk_io"] = min(1.0,
                    (ctx.metrics.disk_io_read_mb + ctx.metrics.disk_io_write_mb) / 500.0
                )
                features["system.net_io"] = min(1.0,
                    (ctx.metrics.net_sent_mb + ctx.metrics.net_recv_mb) / 100.0
                )

        # =================================================================
        # User Metrics
        # =================================================================
        user = UserMetrics()
        user.input_rate = self._input_tracker.get_rate()
        user.idle_seconds = self._input_tracker.get_idle_seconds()

        # Neurostate (if available)
        neuro = self._get_neurostate_extractor()
        if neuro:
            try:
                state = await neuro.get_current_state()
                if state:
                    user.attention = state.attention
                    user.stress = state.stress
                    user.engagement = state.engagement
                    user.fatigue = state.fatigue
            except Exception as e:
                logger.debug(f"NeuroState error: {e}")

        sample.user = user

        features["user.input_rate"] = user.input_rate
        features["user.idle"] = min(1.0, user.idle_seconds / 300.0)  # Normalize to 5min
        features["user.attention"] = user.attention
        features["user.stress"] = user.stress
        features["user.engagement"] = user.engagement
        features["user.fatigue"] = user.fatigue

        # =================================================================
        # Agent/Job Metrics
        # =================================================================
        agents = _collect_agent_metrics()
        sample.agents = agents

        features["agents.running"] = min(1.0, agents.agents_running / 100.0)
        features["agents.blocked"] = min(1.0, agents.agents_blocked / 10.0)
        features["agents.erroring"] = min(1.0, agents.agents_erroring / 10.0)
        features["jobs.queued"] = min(1.0, agents.jobs_queued / 100.0)
        features["jobs.active"] = min(1.0, agents.jobs_active / 50.0)
        features["workflow.progress"] = agents.workflow_progress

        # =================================================================
        # Finalize
        # =================================================================
        sample.features = features
        sample.timestamp = datetime.utcnow()

        self._last_sample = sample
        self._last_sample_time = now

        return features

    def get_last_sample(self) -> Optional[StateSample]:
        """Get the most recent full sample."""
        return self._last_sample

    def get_feature_names(self) -> List[str]:
        """Get list of all feature names (for encoder initialization)."""
        # Return canonical list of features
        names = [
            # GPU (assuming up to 4)
            "gpu0.load", "gpu0.memory", "gpu0.temp", "gpu0.power",
            "gpu1.load", "gpu1.memory", "gpu1.temp", "gpu1.power",
            "gpu2.load", "gpu2.memory", "gpu2.temp", "gpu2.power",
            "gpu3.load", "gpu3.memory", "gpu3.temp", "gpu3.power",
            # FPGA
            "fpga.online", "fpga.throughput", "fpga.errors",
            # System
            "system.cpu", "system.memory", "system.disk", "system.thermal",
            "system.stress", "system.load", "system.disk_io", "system.net_io",
            # User
            "user.input_rate", "user.idle", "user.attention", "user.stress",
            "user.engagement", "user.fatigue",
            # Agents/Jobs
            "agents.running", "agents.blocked", "agents.erroring",
            "jobs.queued", "jobs.active", "workflow.progress",
        ]
        return names


# =============================================================================
# Singleton Access
# =============================================================================

_state_sampler: Optional[StateSampler] = None


def get_state_sampler(**kwargs) -> StateSampler:
    """Get the default StateSampler instance."""
    global _state_sampler
    if _state_sampler is None:
        _state_sampler = StateSampler(**kwargs)
    return _state_sampler


# =============================================================================
# CLI Testing
# =============================================================================

async def _test_sampler():
    """Test the state sampler."""
    print("=" * 60)
    print("State Sampler Test")
    print("=" * 60)

    sampler = get_state_sampler()

    for i in range(5):
        features = await sampler.sample(force=True)

        print(f"\n[{i+1}] Sampled {len(features)} features:")

        # Group by category
        categories: Dict[str, List[str]] = {}
        for name in sorted(features.keys()):
            cat = name.split(".")[0]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(name)

        for cat, names in categories.items():
            values = [f"{n.split('.')[1]}={features[n]:.2f}" for n in names]
            print(f"  {cat}: {', '.join(values)}")

        await asyncio.sleep(1)


def main():
    asyncio.run(_test_sampler())


if __name__ == "__main__":
    main()
