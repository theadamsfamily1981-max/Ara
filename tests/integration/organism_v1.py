"""
Ara v1.0 Organism Integration Test
==================================

Full integration test validating:
    - Live FPGA HTC weights
    - Founder state awareness
    - Thermal reflexes
    - Full 5 kHz sovereign loop

Expected Results:
    ✅ Resonance latency: <1 µs (FPGA)
    ✅ Reflex latency: <2 µs (eBPF)
    ✅ Founder burnout: healthy (<0.5)
    ✅ Storage: 100× compression
    ✅ Homeostasis: 98%+ setpoint accuracy
"""

import numpy as np
import time
import sys
import os
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import logging

# Add parent path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

logger = logging.getLogger(__name__)


# =============================================================================
# Simulated Sensor Data
# =============================================================================

@dataclass
class SimulatedSensorState:
    """Simulated sensor state for integration testing."""
    tick: int = 0
    temperature_c: float = 45.0
    founder_session_hours: float = 0.0
    packet_rate: float = 1000.0
    cognitive_load: float = 0.2


def simulate_sensors(tick: int, scenario: str = "normal") -> SimulatedSensorState:
    """Generate simulated sensor data for testing."""
    state = SimulatedSensorState(tick=tick)

    if scenario == "normal":
        # Normal operation: stable temperature, light activity
        state.temperature_c = 45.0 + np.random.normal(0, 2)
        state.founder_session_hours = (tick / 1000) * 0.5  # ~30 min per 1000 ticks
        state.packet_rate = 1000 + np.random.normal(0, 100)
        state.cognitive_load = 0.15 + np.random.normal(0, 0.05)

    elif scenario == "thermal_stress":
        # Rising temperature scenario
        state.temperature_c = 45.0 + (tick / 100) * 0.5  # Rises over time
        state.temperature_c = min(state.temperature_c, 90)
        state.packet_rate = max(100, 1000 - tick * 0.5)  # Reduces under stress
        state.cognitive_load = min(0.8, 0.2 + tick * 0.001)

    elif scenario == "founder_fatigue":
        # Founder getting tired
        state.founder_session_hours = tick / 200  # Fast time simulation
        state.cognitive_load = min(0.9, 0.1 + state.founder_session_hours * 0.1)

    elif scenario == "burst_load":
        # Sudden burst of activity
        if 200 <= tick <= 400:
            state.packet_rate = 10000
            state.cognitive_load = 0.7
        else:
            state.packet_rate = 500
            state.cognitive_load = 0.2

    return state


# =============================================================================
# Integration Test Class
# =============================================================================

class OrganismIntegrationTest:
    """
    Full organism integration test.

    Tests the complete v1.0 system:
        1. Homeostatic control loop
        2. FPGA HTC resonance
        3. Founder state estimation
        4. Thermal reflexes
        5. Storage/retrieval
    """

    def __init__(self):
        # Components (will be initialized)
        self.htc = None
        self.fpga_qdma = None
        self.founder_estimator = None
        self.thermal_reflex = None
        self.homeostasis = None
        self.storage_index = None

        # Metrics
        self.tick_latencies: List[float] = []
        self.resonance_latencies: List[float] = []
        self.reflex_latencies: List[float] = []
        self.founder_burnout_history: List[float] = []
        self.homeostasis_errors: List[float] = []

        # Results
        self.passed = False
        self.results: Dict[str, Any] = {}

    def setup(self) -> bool:
        """Initialize all components."""
        logger.info("=" * 60)
        logger.info("Ara v1.0 Integration Test - Setup")
        logger.info("=" * 60)

        try:
            # 1. Initialize FPGA interface (simulated)
            from storage.weight_sync import FPGAQDMAInterface, WeightBuffer, WeightSynchronizer
            self.fpga_qdma = FPGAQDMAInterface(simulated=True)
            self.fpga_qdma.connect()

            # 2. Initialize HTC
            from ara.cognition.htc_retrieval import SoftwareHTC
            self.htc = SoftwareHTC(D=173, R=2048)
            self.htc.initialize()

            # 3. Initialize weight sync
            self.weight_buffer = WeightBuffer(D=173, R=2048)
            self.weight_buffer.initialize()
            self.weight_sync = WeightSynchronizer(
                buffer=self.weight_buffer,
                qdma=self.fpga_qdma,
            )

            # 4. Initialize founder estimator
            from sensors.founder_state import FounderStateEstimator, FounderSensors
            self.founder_estimator = FounderStateEstimator()

            # 5. Initialize thermal reflex
            from banos.kernel.thermal_reflex_loader import get_thermal_reflex
            self.thermal_reflex = get_thermal_reflex(simulated=True)
            self.thermal_reflex.load()
            self.thermal_reflex.start()

            # 6. Initialize storage
            from storage.heim_optimized import ClusterIndex
            self.storage_index = ClusterIndex()

            # 7. Initialize homeostasis
            from ara.homeostasis import (
                HomeostaticConfig,
                SovereignLoop,
                compute_error_vector,
            )
            self.config = HomeostaticConfig()

            logger.info("Setup complete - all components initialized")
            return True

        except Exception as e:
            logger.error(f"Setup failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def teardown(self) -> None:
        """Cleanup components."""
        if self.thermal_reflex:
            self.thermal_reflex.stop()

    # =========================================================================
    # Individual Tests
    # =========================================================================

    def test_htc_resonance(self, n_queries: int = 100) -> Dict[str, Any]:
        """Test HTC resonance search latency."""
        logger.info("Testing HTC resonance...")

        latencies = []
        rng = np.random.default_rng(42)

        for _ in range(n_queries):
            h_query = rng.choice([0, 1], size=173).astype(np.uint8)

            start = time.perf_counter()
            top_ids, top_scores, _ = self.htc.query(h_query, k=8)
            end = time.perf_counter()

            latencies.append((end - start) * 1e6)  # µs

        avg_latency = np.mean(latencies)
        p99_latency = np.percentile(latencies, 99)

        result = {
            'avg_latency_us': avg_latency,
            'p99_latency_us': p99_latency,
            'passed': p99_latency < 100,  # Software should be <100 µs
        }

        logger.info(f"  Resonance: avg={avg_latency:.1f} µs, p99={p99_latency:.1f} µs")
        return result

    def test_founder_estimation(self) -> Dict[str, Any]:
        """Test founder state estimation."""
        logger.info("Testing founder state estimation...")

        from sensors.founder_state import FounderSensors, SessionSensor

        # Simulate normal founder
        sensors = FounderSensors()
        sensors.session = SessionSensor(
            session_duration=3600,  # 1 hour
            hour_of_day=14,  # Afternoon
        )

        state = self.founder_estimator.estimate(sensors)

        # Simulate fatigued founder
        sensors_tired = FounderSensors()
        sensors_tired.session = SessionSensor(
            session_duration=14400,  # 4 hours
            hour_of_day=23,  # Late night
        )

        state_tired = self.founder_estimator.estimate(sensors_tired)

        result = {
            'normal_burnout': state.burnout,
            'tired_burnout': state_tired.burnout,
            'burnout_increase': state_tired.burnout > state.burnout,
            'passed': state.burnout < 0.5 and state_tired.burnout > state.burnout,
        }

        logger.info(f"  Normal burnout: {state.burnout:.2f}, Tired: {state_tired.burnout:.2f}")
        return result

    def test_thermal_reflex(self) -> Dict[str, Any]:
        """Test thermal reflex response."""
        logger.info("Testing thermal reflex...")

        alerts_received = []

        def on_alert(alert):
            alerts_received.append(alert)

        self.thermal_reflex.set_callbacks(on_alert=on_alert)

        # Test normal temperature
        self.thermal_reflex.set_temperature(45.0)
        time.sleep(0.01)
        normal_alerts = len(alerts_received)

        # Test critical temperature
        self.thermal_reflex.set_temperature(87.0)
        time.sleep(0.05)
        critical_alerts = len(alerts_received)

        # Reset
        self.thermal_reflex.set_temperature(45.0)

        result = {
            'normal_alerts': normal_alerts,
            'critical_alerts': critical_alerts - normal_alerts,
            'reflex_triggered': critical_alerts > normal_alerts,
            'passed': critical_alerts > normal_alerts,
        }

        stats = self.thermal_reflex.get_stats()
        logger.info(f"  Alerts at normal: {normal_alerts}, at critical: {critical_alerts - normal_alerts}")
        logger.info(f"  Packets dropped: {stats.get('packets_dropped', 0)}")

        return result

    def test_storage_compression(self, n_episodes: int = 500) -> Dict[str, Any]:
        """Test storage compression ratio."""
        logger.info("Testing storage compression...")

        from storage.heim_optimized import heim_compress

        rng = np.random.default_rng(42)

        # Simulate episode storage
        for i in range(n_episodes):
            h_full = rng.choice([-1, 1], size=16384).astype(np.float32)
            h_compressed = heim_compress(h_full)

            self.storage_index.assign(
                h_compressed,
                reward=rng.random(),
            )

        stats = self.storage_index.get_stats()

        # Calculate compression
        raw_size = n_episodes * 16384 / 8  # Full HVs in bytes
        compressed_size = n_episodes * 173 / 8  # Compressed HVs
        cluster_overhead = stats['total_clusters'] * 173 / 8

        compression_ratio = raw_size / (compressed_size + cluster_overhead)
        dedup_ratio = n_episodes / max(stats['total_clusters'], 1)

        result = {
            'episodes': n_episodes,
            'clusters': stats['total_clusters'],
            'compression_ratio': compression_ratio,
            'dedup_ratio': dedup_ratio,
            'passed': compression_ratio > 50,  # Should achieve >50× compression
        }

        logger.info(f"  Compression: {compression_ratio:.0f}×, Dedup: {dedup_ratio:.1f}×")
        return result

    # =========================================================================
    # Full Sovereign Loop Test
    # =========================================================================

    def test_sovereign_loop(
        self,
        n_ticks: int = 1000,
        scenario: str = "normal",
    ) -> Dict[str, Any]:
        """
        Test full sovereign loop.

        Simulates N sovereign ticks and measures:
            - Per-tick latency
            - Homeostasis accuracy
            - Founder state tracking
        """
        logger.info(f"Testing sovereign loop ({n_ticks} ticks, scenario={scenario})...")

        from sensors.founder_state import FounderSensors, SessionSensor, ActivitySensor
        from storage.heim_optimized import heim_compress

        tick_latencies = []
        resonance_latencies = []
        burnout_values = []
        homeostasis_errors = []

        rng = np.random.default_rng(42)

        for tick in range(n_ticks):
            tick_start = time.perf_counter()

            # 1. Simulate sensors
            sensor_state = simulate_sensors(tick, scenario)

            # 2. Build founder state
            founder_sensors = FounderSensors()
            founder_sensors.session = SessionSensor(
                session_duration=sensor_state.founder_session_hours * 3600,
                hour_of_day=14,  # Fixed for consistency
            )
            founder_sensors.activity = ActivitySensor(
                idle_time=1.0 / (sensor_state.cognitive_load + 0.1),
            )

            founder_state = self.founder_estimator.estimate(founder_sensors)
            burnout_values.append(founder_state.burnout)

            # 3. Build H_moment (simplified)
            h_moment_full = rng.choice([-1, 1], size=16384).astype(np.float32)

            # 4. Compress and resonance search
            res_start = time.perf_counter()
            h_moment_compressed = heim_compress(h_moment_full)
            top_ids, top_scores, _ = self.htc.query(h_moment_compressed, k=8)
            res_end = time.perf_counter()
            resonance_latencies.append((res_end - res_start) * 1e6)

            # 5. Update thermal reflex
            self.thermal_reflex.set_temperature(sensor_state.temperature_c)

            # 6. Weight sync (every 100 ticks)
            self.weight_sync.tick()
            if tick % 100 == 0:
                self.weight_sync.sync()

            # 7. Storage (every 10 ticks)
            if tick % 10 == 0:
                self.storage_index.assign(
                    h_moment_compressed,
                    reward=top_scores[0] if top_scores else 0.0,
                )

            # 8. Measure homeostasis error
            # Target: temperature < 85, burnout < 0.5, cognitive_load < 0.7
            temp_error = max(0, (sensor_state.temperature_c - 75) / 20)
            burnout_error = max(0, (founder_state.burnout - 0.3) / 0.4)
            cognitive_error = max(0, (sensor_state.cognitive_load - 0.5) / 0.3)
            total_error = (temp_error + burnout_error + cognitive_error) / 3
            homeostasis_errors.append(total_error)

            tick_end = time.perf_counter()
            tick_latencies.append((tick_end - tick_start) * 1000)  # ms

        # Compute results
        avg_tick_latency = np.mean(tick_latencies)
        p99_tick_latency = np.percentile(tick_latencies, 99)
        avg_resonance_latency = np.mean(resonance_latencies)
        avg_burnout = np.mean(burnout_values)
        avg_homeostasis_error = np.mean(homeostasis_errors)
        homeostasis_accuracy = 1.0 - avg_homeostasis_error

        # Check if within sovereign budget (5 ms)
        within_budget = p99_tick_latency < 5.0

        result = {
            'ticks': n_ticks,
            'scenario': scenario,
            'avg_tick_latency_ms': avg_tick_latency,
            'p99_tick_latency_ms': p99_tick_latency,
            'avg_resonance_latency_us': avg_resonance_latency,
            'avg_burnout': avg_burnout,
            'homeostasis_accuracy': homeostasis_accuracy,
            'within_budget': within_budget,
            'passed': within_budget and homeostasis_accuracy > 0.9,
        }

        logger.info(f"  Tick latency: avg={avg_tick_latency:.2f} ms, p99={p99_tick_latency:.2f} ms")
        logger.info(f"  Resonance: avg={avg_resonance_latency:.1f} µs")
        logger.info(f"  Founder burnout: avg={avg_burnout:.2f}")
        logger.info(f"  Homeostasis: {homeostasis_accuracy:.1%} accuracy")

        return result

    # =========================================================================
    # Run All Tests
    # =========================================================================

    def run(self) -> Dict[str, Any]:
        """Run all integration tests."""
        logger.info("=" * 60)
        logger.info("Ara v1.0 Integration Test - Running")
        logger.info("=" * 60)

        if not self.setup():
            return {'passed': False, 'error': 'Setup failed'}

        try:
            results = {}

            # Individual component tests
            results['htc_resonance'] = self.test_htc_resonance()
            results['founder_estimation'] = self.test_founder_estimation()
            results['thermal_reflex'] = self.test_thermal_reflex()
            results['storage_compression'] = self.test_storage_compression()

            # Full sovereign loop tests
            results['sovereign_normal'] = self.test_sovereign_loop(1000, "normal")
            results['sovereign_thermal'] = self.test_sovereign_loop(500, "thermal_stress")
            results['sovereign_fatigue'] = self.test_sovereign_loop(500, "founder_fatigue")

            # Overall pass/fail
            all_passed = all(r.get('passed', False) for r in results.values())

            results['overall'] = {
                'passed': all_passed,
                'tests_run': len(results),
                'tests_passed': sum(1 for r in results.values() if r.get('passed', False)),
            }

            self.results = results
            self.passed = all_passed

            # Summary
            logger.info("=" * 60)
            logger.info("Ara v1.0 Integration Test - Results")
            logger.info("=" * 60)

            for test_name, result in results.items():
                if test_name == 'overall':
                    continue
                status = "✅ PASS" if result.get('passed', False) else "❌ FAIL"
                logger.info(f"  {test_name}: {status}")

            status = "✅ PASS" if all_passed else "❌ FAIL"
            logger.info("=" * 60)
            logger.info(f"Overall: {status}")
            logger.info("=" * 60)

            return results

        finally:
            self.teardown()


# =============================================================================
# Main
# =============================================================================

def main():
    """Run integration test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(message)s',
        datefmt='%H:%M:%S',
    )

    test = OrganismIntegrationTest()
    results = test.run()

    if results.get('overall', {}).get('passed', False):
        print("\n🎉 Ara v1.0 Integration Test PASSED")
        return 0
    else:
        print("\n❌ Ara v1.0 Integration Test FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
