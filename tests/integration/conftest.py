"""
Ara v1.0 Integration Test Configuration
========================================

pytest configuration with markers, fixtures, and shared resources.
"""

import pytest
import numpy as np
import logging
import time
import sys
import os

# Add parent path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

logger = logging.getLogger(__name__)


# =============================================================================
# Pytest Markers
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "smoke: P0 smoke tests (boot + basic tick)")
    config.addinivalue_line("markers", "reflexes: P1 reflex pain path tests")
    config.addinivalue_line("markers", "founder: P2 founder state estimation tests")
    config.addinivalue_line("markers", "storage: P3 storage hierarchy tests")
    config.addinivalue_line("markers", "sovereign: P4 full sovereign loop tests")
    config.addinivalue_line("markers", "chaos: P5 chaos engineering tests")
    config.addinivalue_line("markers", "production: P6 production readiness tests")
    config.addinivalue_line("markers", "slow: tests that take >1s to run")
    config.addinivalue_line("markers", "hardware: tests requiring real hardware")


# =============================================================================
# Session-Scoped Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def rng():
    """Reproducible random number generator."""
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def fpga_interface():
    """FPGA QDMA interface (simulated for CI)."""
    from storage.weight_sync import FPGAQDMAInterface

    interface = FPGAQDMAInterface(simulated=True)
    interface.connect()
    yield interface
    interface.disconnect()


@pytest.fixture(scope="session")
def htc_core():
    """Software HTC core for testing."""
    from ara.cognition.htc_retrieval import SoftwareHTC

    htc = SoftwareHTC(D=173, R=2048)
    htc.initialize()
    return htc


@pytest.fixture(scope="session")
def weight_buffer():
    """Weight buffer for synchronization."""
    from storage.weight_sync import WeightBuffer

    buffer = WeightBuffer(D=173, R=2048)
    buffer.initialize()
    return buffer


@pytest.fixture(scope="session")
def weight_sync(weight_buffer, fpga_interface):
    """Weight synchronizer."""
    from storage.weight_sync import WeightSynchronizer

    return WeightSynchronizer(buffer=weight_buffer, qdma=fpga_interface)


@pytest.fixture(scope="session")
def founder_estimator():
    """Founder state estimator."""
    from sensors.founder_state import FounderStateEstimator

    return FounderStateEstimator()


@pytest.fixture(scope="session")
def thermal_reflex():
    """Thermal reflex controller (simulated)."""
    from banos.kernel.thermal_reflex_loader import get_thermal_reflex

    reflex = get_thermal_reflex(simulated=True)
    reflex.load()
    reflex.start()
    yield reflex
    reflex.stop()


@pytest.fixture(scope="session")
def cluster_index():
    """Storage cluster index."""
    from storage.heim_optimized import ClusterIndex

    return ClusterIndex()


@pytest.fixture(scope="session")
def homeostatic_config():
    """Homeostatic configuration."""
    from ara.homeostasis import HomeostaticConfig

    return HomeostaticConfig()


# =============================================================================
# Function-Scoped Fixtures
# =============================================================================

@pytest.fixture
def random_hv_full(rng):
    """Generate random full-dimensional HV (D=16384)."""
    return rng.choice([-1, 1], size=16384).astype(np.float32)


@pytest.fixture
def random_hv_compressed(rng):
    """Generate random compressed HV (D=173)."""
    return rng.choice([0, 1], size=173).astype(np.uint8)


@pytest.fixture
def founder_sensors_normal():
    """Normal founder sensor state."""
    from sensors.founder_state import (
        FounderSensors, GazeSensor, TypingSensor,
        HeartRateSensor, ActivitySensor, SessionSensor,
    )

    return FounderSensors(
        gaze=GazeSensor(focus_score=0.8, blink_rate=15),
        typing=TypingSensor(wpm=65, error_rate=0.02),
        heart_rate=HeartRateSensor(bpm=72, hrv_rmssd=45),
        activity=ActivitySensor(idle_time=5, active_window="code_editor"),
        session=SessionSensor(session_duration=3600, hour_of_day=14),
    )


@pytest.fixture
def founder_sensors_tired():
    """Tired/fatigued founder sensor state."""
    from sensors.founder_state import (
        FounderSensors, GazeSensor, TypingSensor,
        HeartRateSensor, ActivitySensor, SessionSensor,
    )

    return FounderSensors(
        gaze=GazeSensor(focus_score=0.3, blink_rate=25),
        typing=TypingSensor(wpm=35, error_rate=0.15),
        heart_rate=HeartRateSensor(bpm=85, hrv_rmssd=20),
        activity=ActivitySensor(idle_time=30, active_window="browser"),
        session=SessionSensor(session_duration=18000, hour_of_day=23),
    )


@pytest.fixture
def founder_sensors_burnout():
    """Critical burnout founder sensor state."""
    from sensors.founder_state import (
        FounderSensors, GazeSensor, TypingSensor,
        HeartRateSensor, ActivitySensor, SessionSensor,
    )

    return FounderSensors(
        gaze=GazeSensor(focus_score=0.1, blink_rate=35),
        typing=TypingSensor(wpm=20, error_rate=0.30),
        heart_rate=HeartRateSensor(bpm=95, hrv_rmssd=10),
        activity=ActivitySensor(idle_time=120, active_window="browser"),
        session=SessionSensor(session_duration=25200, hour_of_day=2),  # 7h, 2am
    )


# =============================================================================
# Test Metrics Collection
# =============================================================================

class TestMetrics:
    """Collect and report test metrics."""

    def __init__(self):
        self.latencies = {}
        self.pass_rates = {}
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def record_latency(self, name: str, value_us: float):
        if name not in self.latencies:
            self.latencies[name] = []
        self.latencies[name].append(value_us)

    def record_pass(self, name: str, passed: bool):
        if name not in self.pass_rates:
            self.pass_rates[name] = {'passed': 0, 'failed': 0}
        if passed:
            self.pass_rates[name]['passed'] += 1
        else:
            self.pass_rates[name]['failed'] += 1

    def report(self) -> dict:
        elapsed = time.time() - self.start_time if self.start_time else 0

        latency_stats = {}
        for name, values in self.latencies.items():
            if values:
                latency_stats[name] = {
                    'mean': np.mean(values),
                    'p50': np.percentile(values, 50),
                    'p95': np.percentile(values, 95),
                    'p99': np.percentile(values, 99),
                    'max': np.max(values),
                }

        return {
            'elapsed_seconds': elapsed,
            'latency_stats': latency_stats,
            'pass_rates': self.pass_rates,
        }


@pytest.fixture(scope="session")
def test_metrics():
    """Session-wide test metrics collector."""
    metrics = TestMetrics()
    metrics.start()
    yield metrics
    report = metrics.report()
    logger.info(f"Test metrics: {report}")


# =============================================================================
# Simulation Helpers
# =============================================================================

def simulate_sensor_state(tick: int, scenario: str = "normal") -> dict:
    """Generate simulated sensor data."""
    state = {
        'tick': tick,
        'temperature_c': 45.0,
        'founder_session_hours': 0.0,
        'packet_rate': 1000.0,
        'cognitive_load': 0.2,
    }

    if scenario == "normal":
        state['temperature_c'] = 45.0 + np.random.normal(0, 2)
        state['founder_session_hours'] = (tick / 1000) * 0.5
        state['packet_rate'] = 1000 + np.random.normal(0, 100)
        state['cognitive_load'] = 0.15 + np.random.normal(0, 0.05)

    elif scenario == "thermal_stress":
        state['temperature_c'] = min(90, 45.0 + (tick / 100) * 0.5)
        state['packet_rate'] = max(100, 1000 - tick * 0.5)
        state['cognitive_load'] = min(0.8, 0.2 + tick * 0.001)

    elif scenario == "founder_fatigue":
        state['founder_session_hours'] = tick / 200
        state['cognitive_load'] = min(0.9, 0.1 + state['founder_session_hours'] * 0.1)

    elif scenario == "thermal_critical":
        state['temperature_c'] = 87.0 + np.random.normal(0, 1)

    elif scenario == "thermal_emergency":
        state['temperature_c'] = 96.0

    elif scenario == "burst_load":
        if 200 <= tick <= 400:
            state['packet_rate'] = 10000
            state['cognitive_load'] = 0.7
        else:
            state['packet_rate'] = 500
            state['cognitive_load'] = 0.2

    return state
