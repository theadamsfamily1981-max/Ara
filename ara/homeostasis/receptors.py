"""
Ara Receptor Daemon - Sensory Input from All Subsystems
========================================================

The ReceptorDaemon is the organism's sensory system - gathering telemetry
from all sources and building the moment hypervector (H_moment).

Architecture:
    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    │  Thermal    │  │    LAN      │  │   Visual    │
    │  Sensors    │  │   Stats     │  │   Input     │
    └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
           │                │                │
           └────────────────┼────────────────┘
                            ▼
                  ┌─────────────────┐
                  │ ReceptorDaemon  │
                  │  (5 kHz loop)   │
                  │                 │
                  │  → Telemetry    │
                  │  → H_moment     │
                  └────────┬────────┘
                           │
                           ▼
                    [Sovereign Loop]

Rate: 5 kHz (200 µs period)

The receptor daemon is the fastest loop - it must never block.
"""

from __future__ import annotations

import asyncio
import time
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable, List
from queue import Queue, Empty
import logging

from .state import Telemetry, HomeostaticState
from .config import Setpoints, HomeostaticConfig


logger = logging.getLogger(__name__)


# =============================================================================
# Receptor Sources
# =============================================================================

class ReceptorSource:
    """Base class for receptor sources."""

    def __init__(self, name: str, rate_hz: float = 1000.0):
        self.name = name
        self.rate_hz = rate_hz
        self.last_sample_time: float = 0.0
        self.sample_count: int = 0
        self.error_count: int = 0

    def sample(self) -> Dict[str, Any]:
        """Sample this receptor source. Override in subclasses."""
        return {}

    def sample_period(self) -> float:
        """Minimum period between samples (seconds)."""
        return 1.0 / self.rate_hz


class ThermalReceptor(ReceptorSource):
    """Thermal sensor receptor."""

    def __init__(self):
        super().__init__("thermal", rate_hz=100.0)
        self._fpga_temp_path = "/sys/class/hwmon/hwmon0/temp1_input"
        self._cpu_temp_path = "/sys/class/thermal/thermal_zone0/temp"

    def sample(self) -> Dict[str, Any]:
        """Sample thermal sensors."""
        result = {
            'fpga_temp': 45.0,  # Default
            'cpu_temp': 40.0,
            'ambient_temp': 25.0,
        }

        try:
            # Try to read actual sensors (Linux)
            try:
                with open(self._cpu_temp_path, 'r') as f:
                    result['cpu_temp'] = float(f.read().strip()) / 1000.0
            except (FileNotFoundError, PermissionError):
                pass

            # FPGA temp would come from FPGA driver
            # For now, simulate based on activity
            result['fpga_temp'] = 45.0 + np.random.normal(0, 2)

        except Exception as e:
            self.error_count += 1
            logger.debug(f"Thermal sample error: {e}")

        self.sample_count += 1
        self.last_sample_time = time.time()
        return result


class NetworkReceptor(ReceptorSource):
    """Network/LAN receptor."""

    def __init__(self):
        super().__init__("network", rate_hz=1000.0)
        self._last_rx_packets = 0
        self._last_tx_packets = 0
        self._last_errors = 0

    def sample(self) -> Dict[str, Any]:
        """Sample network statistics."""
        result = {
            'packet_rate': 0.0,
            'packet_loss_rate': 0.0,
            'flow_hit_rate': 0.95,
            'reflex_triggers': 0,
        }

        try:
            # Try to get actual network stats (Linux)
            try:
                with open('/proc/net/dev', 'r') as f:
                    for line in f:
                        if 'eth0' in line or 'enp' in line:
                            parts = line.split()
                            rx_packets = int(parts[2])
                            tx_packets = int(parts[10])
                            errors = int(parts[3]) + int(parts[11])

                            if self._last_rx_packets > 0:
                                delta_packets = (rx_packets + tx_packets -
                                               self._last_rx_packets - self._last_tx_packets)
                                delta_errors = errors - self._last_errors
                                dt = time.time() - self.last_sample_time

                                if dt > 0:
                                    result['packet_rate'] = delta_packets / dt
                                    if delta_packets > 0:
                                        result['packet_loss_rate'] = delta_errors / delta_packets

                            self._last_rx_packets = rx_packets
                            self._last_tx_packets = tx_packets
                            self._last_errors = errors
                            break
            except (FileNotFoundError, PermissionError):
                pass

        except Exception as e:
            self.error_count += 1
            logger.debug(f"Network sample error: {e}")

        self.sample_count += 1
        self.last_sample_time = time.time()
        return result


class CognitiveReceptor(ReceptorSource):
    """Cognitive load receptor (from HTC)."""

    def __init__(self):
        super().__init__("cognitive", rate_hz=200.0)
        self._htc_search = None  # Will be connected to HTCSearchFPGA

    def connect_htc(self, htc_search) -> None:
        """Connect to HTC search module."""
        self._htc_search = htc_search

    def sample(self) -> Dict[str, Any]:
        """Sample cognitive state."""
        result = {
            'hd_query_rate': 0.0,
            'active_attractors': 0,
            'working_memory_size': 0,
            'cognitive_load': 0.0,
            'last_hd_latency_us': 0.0,
            'avg_hd_latency_us': 0.0,
            'current_resonance': 0.0,
            'top_attractor_id': -1,
        }

        try:
            if self._htc_search is not None:
                stats = self._htc_search.get_stats()
                result['hd_query_rate'] = stats.get('query_rate', 0.0)
                result['avg_hd_latency_us'] = stats.get('avg_latency_us', 0.0)
                result['last_hd_latency_us'] = stats.get('last_latency_us', 0.0)

            # Simulate cognitive load based on query rate
            # Load increases with query rate
            result['cognitive_load'] = min(1.0, result['hd_query_rate'] / 10000.0)

        except Exception as e:
            self.error_count += 1
            logger.debug(f"Cognitive sample error: {e}")

        self.sample_count += 1
        self.last_sample_time = time.time()
        return result


class CathedralReceptor(ReceptorSource):
    """Cathedral/memory receptor."""

    def __init__(self):
        super().__init__("cathedral", rate_hz=10.0)  # Low rate - cathedral is slow
        self._cathedral = None  # Will be connected to cathedral

    def connect_cathedral(self, cathedral) -> None:
        """Connect to cathedral module."""
        self._cathedral = cathedral

    def sample(self) -> Dict[str, Any]:
        """Sample cathedral state."""
        result = {
            'consolidation_rate': 0.0,
            'episode_count': 0,
            'attractor_count': 0,
            'attractor_diversity': 1.0,
        }

        try:
            if self._cathedral is not None:
                stats = self._cathedral.get_stats()
                result['consolidation_rate'] = stats.get('consolidation_rate', 0.0)
                result['episode_count'] = stats.get('episode_count', 0)
                result['attractor_count'] = stats.get('attractor_count', 0)
                result['attractor_diversity'] = stats.get('diversity', 1.0)

        except Exception as e:
            self.error_count += 1
            logger.debug(f"Cathedral sample error: {e}")

        self.sample_count += 1
        self.last_sample_time = time.time()
        return result


# =============================================================================
# Moment Builder
# =============================================================================

class MomentBuilder:
    """
    Builds the moment hypervector (H_moment) from telemetry.

    H_moment = Σ role_i ⊗ encode(value_i)

    This is the "what is happening right now" encoding.
    """

    def __init__(self, dim: int = 16384):
        self.dim = dim
        self._rng = np.random.default_rng(seed=42)

        # Role hypervectors (fixed random)
        self.role_hvs: Dict[str, np.ndarray] = {}
        self._init_role_hvs()

        # Value encoders
        self._value_levels = 256  # Discretization levels

    def _init_role_hvs(self) -> None:
        """Initialize role hypervectors."""
        roles = [
            'THERMAL', 'COGNITIVE', 'LATENCY', 'NETWORK',
            'CATHEDRAL', 'REWARD', 'MODE', 'FOUNDER',
            'ERROR', 'TIME', 'RESONANCE'
        ]
        for role in roles:
            self.role_hvs[role] = self._random_hv()

    def _random_hv(self) -> np.ndarray:
        """Generate random bipolar hypervector."""
        bits = self._rng.integers(0, 2, size=self.dim, dtype=np.uint8)
        return bits.astype(np.int8) * 2 - 1

    def _encode_scalar(self, value: float, min_val: float, max_val: float) -> np.ndarray:
        """Encode scalar value as hypervector using thermometer encoding."""
        # Normalize to [0, 1]
        normalized = np.clip((value - min_val) / (max_val - min_val + 1e-10), 0, 1)

        # Thermometer encoding: flip bits up to normalized position
        flip_count = int(normalized * self.dim)
        hv = np.ones(self.dim, dtype=np.int8)
        if flip_count > 0:
            # Use deterministic flip pattern based on value
            indices = np.arange(self.dim)
            self._rng.shuffle(indices)
            hv[indices[:flip_count]] = -1

        return hv

    def _bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Bind two hypervectors (XOR for bipolar = element-wise multiply)."""
        return (a * b).astype(np.int8)

    def _bundle(self, hvs: List[np.ndarray]) -> np.ndarray:
        """Bundle hypervectors (majority vote)."""
        if not hvs:
            return np.zeros(self.dim, dtype=np.int8)

        stacked = np.stack(hvs)
        summed = np.sum(stacked, axis=0)

        # Majority vote with random tie-breaking
        result = np.sign(summed).astype(np.int8)
        ties = (summed == 0)
        if np.any(ties):
            result[ties] = self._rng.choice([-1, 1], size=np.sum(ties))

        return result

    def build(self, telemetry: Telemetry, state: Optional[HomeostaticState] = None) -> np.ndarray:
        """
        Build moment hypervector from telemetry.

        Args:
            telemetry: Current telemetry
            state: Optional full state for additional context

        Returns:
            H_moment as numpy array
        """
        components = []

        # Thermal component
        thermal_hv = self._encode_scalar(telemetry.fpga_temp, 20.0, 100.0)
        components.append(self._bind(self.role_hvs['THERMAL'], thermal_hv))

        # Cognitive component
        cognitive_hv = self._encode_scalar(telemetry.cognitive_load, 0.0, 1.0)
        components.append(self._bind(self.role_hvs['COGNITIVE'], cognitive_hv))

        # Latency component
        latency_hv = self._encode_scalar(telemetry.avg_hd_latency_us, 0.0, 10.0)
        components.append(self._bind(self.role_hvs['LATENCY'], latency_hv))

        # Network component
        network_hv = self._encode_scalar(telemetry.packet_rate, 0.0, 100000.0)
        components.append(self._bind(self.role_hvs['NETWORK'], network_hv))

        # Cathedral component
        cathedral_hv = self._encode_scalar(telemetry.consolidation_rate, 0.0, 1.0)
        components.append(self._bind(self.role_hvs['CATHEDRAL'], cathedral_hv))

        # Reward component
        reward_hv = self._encode_scalar(telemetry.smoothed_reward, -1.0, 1.0)
        components.append(self._bind(self.role_hvs['REWARD'], reward_hv))

        # Resonance component
        resonance_hv = self._encode_scalar(telemetry.current_resonance, 0.0, 1.0)
        components.append(self._bind(self.role_hvs['RESONANCE'], resonance_hv))

        # Time component (circadian encoding)
        hour = (time.time() % 86400) / 3600  # Hour of day
        time_hv = self._encode_scalar(hour, 0.0, 24.0)
        components.append(self._bind(self.role_hvs['TIME'], time_hv))

        # Bundle all components
        h_moment = self._bundle(components)

        return h_moment


# =============================================================================
# Receptor Daemon
# =============================================================================

class ReceptorDaemon:
    """
    The receptor daemon - gathers telemetry at 5 kHz.

    This is the organism's sensory system.
    """

    def __init__(
        self,
        config: HomeostaticConfig,
        output_queue: Queue,
        target_hz: float = 5000.0,
    ):
        """
        Initialize receptor daemon.

        Args:
            config: Homeostatic configuration
            output_queue: Queue for telemetry output
            target_hz: Target sampling rate
        """
        self.config = config
        self.output_queue = output_queue
        self.target_hz = target_hz
        self.period = 1.0 / target_hz

        # Receptors
        self.thermal = ThermalReceptor()
        self.network = NetworkReceptor()
        self.cognitive = CognitiveReceptor()
        self.cathedral = CathedralReceptor()

        # Moment builder
        self.moment_builder = MomentBuilder()

        # State
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._telemetry = Telemetry()
        self._h_moment: Optional[np.ndarray] = None

        # Statistics
        self._loop_count = 0
        self._total_time = 0.0
        self._max_loop_time = 0.0
        self._overruns = 0

    def connect_modules(
        self,
        htc_search=None,
        cathedral=None,
    ) -> None:
        """Connect receptor sources to their data providers."""
        if htc_search:
            self.cognitive.connect_htc(htc_search)
        if cathedral:
            self.cathedral.connect_cathedral(cathedral)

    def start(self) -> None:
        """Start the receptor daemon."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info(f"ReceptorDaemon started at {self.target_hz} Hz")

    def stop(self) -> None:
        """Stop the receptor daemon."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        logger.info("ReceptorDaemon stopped")

    def _run_loop(self) -> None:
        """Main receptor loop."""
        next_time = time.perf_counter()

        while self._running:
            loop_start = time.perf_counter()

            # Sample all receptors
            self._sample_all()

            # Build H_moment
            self._h_moment = self.moment_builder.build(self._telemetry)

            # Push to output queue (non-blocking)
            try:
                self.output_queue.put_nowait({
                    'telemetry': self._telemetry,
                    'h_moment': self._h_moment,
                    'timestamp': time.time(),
                })
            except:
                pass  # Queue full, drop sample

            # Statistics
            loop_time = time.perf_counter() - loop_start
            self._loop_count += 1
            self._total_time += loop_time
            self._max_loop_time = max(self._max_loop_time, loop_time)

            # Timing control
            next_time += self.period
            sleep_time = next_time - time.perf_counter()

            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # Overrun - we're behind
                self._overruns += 1
                next_time = time.perf_counter()

    def _sample_all(self) -> None:
        """Sample all receptor sources."""
        now = time.time()
        self._telemetry.timestamp = now

        # Thermal (low rate)
        if now - self.thermal.last_sample_time >= self.thermal.sample_period():
            data = self.thermal.sample()
            self._telemetry.fpga_temp = data['fpga_temp']
            self._telemetry.cpu_temp = data['cpu_temp']
            self._telemetry.ambient_temp = data['ambient_temp']

        # Network (high rate)
        if now - self.network.last_sample_time >= self.network.sample_period():
            data = self.network.sample()
            self._telemetry.packet_rate = data['packet_rate']
            self._telemetry.packet_loss_rate = data['packet_loss_rate']
            self._telemetry.flow_hit_rate = data['flow_hit_rate']
            self._telemetry.reflex_triggers = data['reflex_triggers']

        # Cognitive
        if now - self.cognitive.last_sample_time >= self.cognitive.sample_period():
            data = self.cognitive.sample()
            self._telemetry.hd_query_rate = data['hd_query_rate']
            self._telemetry.cognitive_load = data['cognitive_load']
            self._telemetry.last_hd_latency_us = data['last_hd_latency_us']
            self._telemetry.avg_hd_latency_us = data['avg_hd_latency_us']
            self._telemetry.current_resonance = data['current_resonance']
            self._telemetry.top_attractor_id = data['top_attractor_id']

        # Cathedral (low rate)
        if now - self.cathedral.last_sample_time >= self.cathedral.sample_period():
            data = self.cathedral.sample()
            self._telemetry.consolidation_rate = data['consolidation_rate']
            self._telemetry.episode_count = data['episode_count']
            self._telemetry.attractor_count = data['attractor_count']
            self._telemetry.attractor_diversity = data['attractor_diversity']

    @property
    def telemetry(self) -> Telemetry:
        """Get current telemetry snapshot."""
        return self._telemetry

    @property
    def h_moment(self) -> Optional[np.ndarray]:
        """Get current moment hypervector."""
        return self._h_moment

    def get_stats(self) -> Dict[str, Any]:
        """Get receptor daemon statistics."""
        avg_loop_time = self._total_time / max(self._loop_count, 1)
        return {
            'loop_count': self._loop_count,
            'avg_loop_time_us': avg_loop_time * 1e6,
            'max_loop_time_us': self._max_loop_time * 1e6,
            'overruns': self._overruns,
            'target_hz': self.target_hz,
            'actual_hz': self._loop_count / max(self._total_time, 0.001),
            'thermal_samples': self.thermal.sample_count,
            'network_samples': self.network.sample_count,
            'cognitive_samples': self.cognitive.sample_count,
            'cathedral_samples': self.cathedral.sample_count,
        }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'ReceptorSource',
    'ThermalReceptor',
    'NetworkReceptor',
    'CognitiveReceptor',
    'CathedralReceptor',
    'MomentBuilder',
    'ReceptorDaemon',
]
