"""
HSF Telemetry Sources
======================

Fake telemetry generators for testing the Hypervector Spiking Field.

Each source simulates realistic telemetry with:
- Normal operating range with noise
- Occasional anomalies (spikes, drift, correlation breaks)
- Time-correlated behavior (not just random)

Real integration would tap:
- nvidia-smi / NVML for GPU
- SNMP / NETCONF for network
- Prometheus / metrics endpoints for services
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from abc import ABC, abstractmethod
import time


@dataclass
class FakeTelemetrySource(ABC):
    """Base class for fake telemetry generators."""
    name: str
    features: List[str]
    _rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())
    _tick: int = 0
    _anomaly_mode: Optional[str] = None
    _anomaly_start: int = 0
    _anomaly_duration: int = 0

    @abstractmethod
    def _generate_normal(self) -> Dict[str, float]:
        """Generate normal operating telemetry."""
        pass

    @abstractmethod
    def _generate_anomaly(self, mode: str) -> Dict[str, float]:
        """Generate anomalous telemetry for given mode."""
        pass

    def inject_anomaly(self, mode: str, duration: int = 10):
        """Inject an anomaly for the next N ticks."""
        self._anomaly_mode = mode
        self._anomaly_start = self._tick
        self._anomaly_duration = duration

    def sample(self) -> Dict[str, float]:
        """Get next telemetry sample."""
        self._tick += 1

        # Check if anomaly is active
        if self._anomaly_mode is not None:
            if self._tick - self._anomaly_start < self._anomaly_duration:
                return self._generate_anomaly(self._anomaly_mode)
            else:
                self._anomaly_mode = None

        return self._generate_normal()


@dataclass
class GPUTelemetry(FakeTelemetrySource):
    """
    Simulates GPU telemetry (nvidia-smi style).

    Features:
    - temp: GPU temperature (30-90°C)
    - util: GPU utilization (0-100%)
    - mem: Memory utilization (0-100%)
    - power: Power draw (50-350W)
    - fan: Fan speed (0-100%)

    Correlations:
    - Higher util → higher temp, power
    - Higher temp → higher fan
    """
    name: str = "gpu"
    features: List[str] = field(default_factory=lambda: ["temp", "util", "mem", "power", "fan"])

    # Internal state for time-correlation
    _base_util: float = 0.3
    _thermal_mass: float = 45.0  # Smoothed temperature

    def _generate_normal(self) -> Dict[str, float]:
        # Slowly wandering base utilization
        self._base_util += self._rng.normal(0, 0.02)
        self._base_util = np.clip(self._base_util, 0.1, 0.9)

        # Actual util with noise
        util = self._base_util + self._rng.normal(0, 0.05)
        util = np.clip(util, 0, 1)

        # Temperature correlates with util (thermal lag)
        target_temp = 40 + util * 45 + self._rng.normal(0, 2)
        self._thermal_mass = 0.9 * self._thermal_mass + 0.1 * target_temp

        # Memory usually tracks util loosely
        mem = util * 0.8 + self._rng.normal(0, 0.1)
        mem = np.clip(mem, 0, 1)

        # Power correlates with util
        power = 80 + util * 250 + self._rng.normal(0, 10)

        # Fan responds to temperature
        fan = max(0.3, (self._thermal_mass - 40) / 50)
        fan = np.clip(fan + self._rng.normal(0, 0.05), 0, 1)

        return {
            "temp": self._thermal_mass,
            "util": util,
            "mem": mem,
            "power": power,
            "fan": fan,
        }

    def _generate_anomaly(self, mode: str) -> Dict[str, float]:
        normal = self._generate_normal()

        if mode == "thermal_runaway":
            # Temperature climbing despite low util
            normal["temp"] = 85 + self._rng.normal(0, 3)
            normal["fan"] = 1.0  # Fan at max
        elif mode == "memory_leak":
            # Memory climbing regardless of util
            progress = (self._tick - self._anomaly_start) / self._anomaly_duration
            normal["mem"] = 0.5 + progress * 0.5
        elif mode == "power_spike":
            # Random power spikes
            normal["power"] = 320 + self._rng.normal(0, 20)
        elif mode == "util_saturation":
            # Pegged at 100%
            normal["util"] = 0.99
            normal["mem"] = 0.95
            normal["temp"] = 80 + self._rng.normal(0, 2)

        return normal


@dataclass
class NetworkTelemetry(FakeTelemetrySource):
    """
    Simulates network interface telemetry (SNMP/Juniper style).

    Features:
    - bps_in: Inbound bandwidth (0-10Gbps)
    - bps_out: Outbound bandwidth (0-10Gbps)
    - pps_in: Packets per second in
    - pps_out: Packets per second out
    - errors: Error counter (should be near 0)
    - drops: Drop counter (should be near 0)
    - latency_ms: Round-trip latency
    """
    name: str = "network"
    features: List[str] = field(default_factory=lambda: [
        "bps_in", "bps_out", "pps_in", "pps_out", "errors", "drops", "latency_ms"
    ])

    # Internal state
    _base_load: float = 0.2

    def _generate_normal(self) -> Dict[str, float]:
        # Slowly wandering base load
        self._base_load += self._rng.normal(0, 0.01)
        self._base_load = np.clip(self._base_load, 0.05, 0.7)

        # Bandwidth with bursty behavior
        burst = self._rng.exponential(0.1)
        load = self._base_load + burst
        load = np.clip(load, 0, 1)

        bps_in = load * 10e9 + self._rng.normal(0, 1e8)
        bps_out = load * 5e9 + self._rng.normal(0, 5e7)  # Asymmetric

        # Packets correlate with bandwidth (assume ~1KB avg packet)
        pps_in = bps_in / 8000 + self._rng.normal(0, 1000)
        pps_out = bps_out / 8000 + self._rng.normal(0, 500)

        # Errors and drops are rare in normal operation
        errors = max(0, self._rng.poisson(0.1))
        drops = max(0, self._rng.poisson(0.05))

        # Latency is usually stable
        latency = 0.5 + self._rng.exponential(0.1)

        return {
            "bps_in": max(0, bps_in),
            "bps_out": max(0, bps_out),
            "pps_in": max(0, pps_in),
            "pps_out": max(0, pps_out),
            "errors": errors,
            "drops": drops,
            "latency_ms": latency,
        }

    def _generate_anomaly(self, mode: str) -> Dict[str, float]:
        normal = self._generate_normal()

        if mode == "congestion":
            # High drops, increased latency
            normal["drops"] = self._rng.poisson(50)
            normal["latency_ms"] = 10 + self._rng.exponential(5)
        elif mode == "link_flap":
            # Intermittent zeros
            if self._rng.random() < 0.3:
                normal["bps_in"] = 0
                normal["bps_out"] = 0
                normal["pps_in"] = 0
                normal["pps_out"] = 0
        elif mode == "error_storm":
            # Lots of errors
            normal["errors"] = self._rng.poisson(100)
        elif mode == "ddos":
            # Asymmetric flood
            normal["bps_in"] = 9.5e9 + self._rng.normal(0, 1e8)
            normal["pps_in"] = 1e7 + self._rng.normal(0, 1e5)
            normal["drops"] = self._rng.poisson(1000)

        return normal


@dataclass
class ServiceTelemetry(FakeTelemetrySource):
    """
    Simulates service-level telemetry (Prometheus/metrics style).

    Features:
    - request_rate: Requests per second
    - error_rate: Error percentage (0-1)
    - latency_p50: 50th percentile latency (ms)
    - latency_p99: 99th percentile latency (ms)
    - queue_depth: Pending requests
    - cpu: Service CPU usage (0-1)
    - memory: Service memory usage (0-1)
    """
    name: str = "service"
    features: List[str] = field(default_factory=lambda: [
        "request_rate", "error_rate", "latency_p50", "latency_p99",
        "queue_depth", "cpu", "memory"
    ])

    # Internal state
    _base_rate: float = 100.0

    def _generate_normal(self) -> Dict[str, float]:
        # Request rate with daily pattern (simplified)
        hour_factor = 0.5 + 0.5 * np.sin(self._tick * 0.01)  # Fake diurnal
        self._base_rate += self._rng.normal(0, 5)
        self._base_rate = np.clip(self._base_rate, 50, 500)

        request_rate = self._base_rate * hour_factor + self._rng.normal(0, 10)

        # Error rate is usually very low
        error_rate = 0.001 + self._rng.exponential(0.0005)
        error_rate = min(error_rate, 0.1)

        # Latency
        latency_p50 = 5 + self._rng.exponential(2)
        latency_p99 = latency_p50 * 3 + self._rng.exponential(10)

        # Queue depth correlates with load
        load_factor = request_rate / 500
        queue_depth = load_factor * 20 + self._rng.exponential(2)

        # Resources correlate with load
        cpu = 0.2 + load_factor * 0.5 + self._rng.normal(0, 0.05)
        cpu = np.clip(cpu, 0, 1)
        memory = 0.4 + self._rng.normal(0, 0.02)  # Memory more stable

        return {
            "request_rate": max(0, request_rate),
            "error_rate": np.clip(error_rate, 0, 1),
            "latency_p50": max(1, latency_p50),
            "latency_p99": max(latency_p50, latency_p99),
            "queue_depth": max(0, queue_depth),
            "cpu": np.clip(cpu, 0, 1),
            "memory": np.clip(memory, 0, 1),
        }

    def _generate_anomaly(self, mode: str) -> Dict[str, float]:
        normal = self._generate_normal()

        if mode == "overload":
            # High latency, queue depth, errors
            normal["latency_p50"] = 100 + self._rng.exponential(50)
            normal["latency_p99"] = 500 + self._rng.exponential(200)
            normal["queue_depth"] = 100 + self._rng.exponential(50)
            normal["error_rate"] = 0.1 + self._rng.exponential(0.05)
            normal["cpu"] = 0.95 + self._rng.normal(0, 0.02)
        elif mode == "memory_exhaustion":
            # Memory climbing, GC stalls
            progress = (self._tick - self._anomaly_start) / self._anomaly_duration
            normal["memory"] = 0.7 + progress * 0.28
            normal["latency_p99"] = 100 + progress * 400  # GC pauses
        elif mode == "upstream_failure":
            # High error rate, normal load
            normal["error_rate"] = 0.3 + self._rng.exponential(0.1)
        elif mode == "traffic_spike":
            # Sudden load increase
            normal["request_rate"] = 800 + self._rng.exponential(100)
            normal["queue_depth"] = 50 + self._rng.exponential(20)
            normal["cpu"] = 0.85 + self._rng.normal(0, 0.05)

        return normal


@dataclass
class TelemetryMux:
    """
    Multiplexer for multiple telemetry sources.

    Aggregates all sources and provides unified sampling.
    """
    sources: Dict[str, FakeTelemetrySource] = field(default_factory=dict)

    def add_source(self, source: FakeTelemetrySource):
        """Add a telemetry source."""
        self.sources[source.name] = source

    def sample_all(self) -> Dict[str, Dict[str, float]]:
        """Sample from all sources."""
        return {name: source.sample() for name, source in self.sources.items()}

    def inject_anomaly(self, source_name: str, mode: str, duration: int = 10):
        """Inject anomaly into specific source."""
        if source_name in self.sources:
            self.sources[source_name].inject_anomaly(mode, duration)

    @classmethod
    def create_default(cls) -> "TelemetryMux":
        """Create mux with standard sources."""
        mux = cls()
        mux.add_source(GPUTelemetry())
        mux.add_source(NetworkTelemetry())
        mux.add_source(ServiceTelemetry())
        return mux
