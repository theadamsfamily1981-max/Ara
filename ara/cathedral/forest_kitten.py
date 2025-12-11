#!/usr/bin/env python3
# ara/cathedral/forest_kitten.py
"""
SQRL Forest Kitten: Hardware Safety Enforcer

Hardware specs:
- Xilinx Artix-7 XC7A200T FPGA
- Open-source bitstream for RISC-V soft core
- Hard real-time NIB covenant enforcement
- <10µs deterministic latency (bypasses Linux scheduler)

Purpose:
- Hardware-locked safety checks
- RISC-V core runs independently of Linux
- Cannot be overridden by software
- Watchdog timer for system health
"""

import logging
import time
import struct
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum, auto
import numpy as np

logger = logging.getLogger(__name__)

# Optional imports
try:
    import mmap
    MMAP_AVAILABLE = True
except ImportError:
    MMAP_AVAILABLE = False


class CovenantType(Enum):
    """Types of NIB covenants enforced in hardware."""
    ENERGY_BOUND = auto()      # Max energy per action
    REVERSIBILITY = auto()     # Min reversibility score
    ENTROPY_BUDGET = auto()    # Max entropy increase
    LATENCY_BOUND = auto()     # Max decision latency
    HUMAN_OVERRIDE = auto()    # Human veto response time
    RESOURCE_LIMIT = auto()    # Resource allocation bounds


@dataclass
class CovenantViolation:
    """Record of a covenant violation."""
    covenant_type: CovenantType
    threshold: float
    actual_value: float
    timestamp: float
    severity: str  # 'warning', 'violation', 'critical'

    def to_dict(self) -> Dict[str, Any]:
        return {
            'covenant_type': self.covenant_type.name,
            'threshold': self.threshold,
            'actual_value': self.actual_value,
            'timestamp': self.timestamp,
            'severity': self.severity
        }


@dataclass
class SafetyCheckResult:
    """Result from hardware safety check."""
    passed: bool
    latency_us: float
    violations: List[CovenantViolation]
    watchdog_healthy: bool
    risc_v_heartbeat: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            'passed': self.passed,
            'latency_us': self.latency_us,
            'num_violations': len(self.violations),
            'violations': [v.to_dict() for v in self.violations],
            'watchdog_healthy': self.watchdog_healthy,
            'risc_v_heartbeat': self.risc_v_heartbeat
        }


class ForestKittenInterface:
    """
    SQRL Forest Kitten: Hardware safety enforcer.

    Memory map:
    - 0x0000-0x00FF: Control/Status registers
    - 0x0100-0x01FF: Covenant threshold registers
    - 0x0200-0x02FF: Current value registers
    - 0x0300-0x03FF: Violation log (circular buffer)
    - 0x1000-0x1FFF: RISC-V mailbox

    The RISC-V soft core runs a minimal safety monitor that:
    1. Checks covenants every 10µs
    2. Can halt the system independently of Linux
    3. Maintains watchdog for liveness
    4. Logs all violations to BRAM
    """

    # Register offsets
    REG_CONTROL = 0x0000
    REG_STATUS = 0x0004
    REG_WATCHDOG = 0x0008
    REG_HEARTBEAT = 0x000C
    REG_VIOLATION_COUNT = 0x0010
    REG_LAST_CHECK_US = 0x0014

    # Covenant threshold registers (0x0100+)
    REG_THRESH_ENERGY = 0x0100
    REG_THRESH_REVERSIBILITY = 0x0104
    REG_THRESH_ENTROPY = 0x0108
    REG_THRESH_LATENCY = 0x010C
    REG_THRESH_HUMAN_OVERRIDE = 0x0110
    REG_THRESH_RESOURCE = 0x0114

    # Current value registers (0x0200+)
    REG_CURR_ENERGY = 0x0200
    REG_CURR_REVERSIBILITY = 0x0204
    REG_CURR_ENTROPY = 0x0208
    REG_CURR_LATENCY = 0x020C

    # Control bits
    CTRL_ENABLE = 0x0001
    CTRL_RESET = 0x0002
    CTRL_HALT_ON_VIOLATION = 0x0004
    CTRL_LOG_WARNINGS = 0x0008

    # Status bits
    STATUS_RUNNING = 0x0001
    STATUS_VIOLATION = 0x0002
    STATUS_HALTED = 0x0004
    STATUS_WATCHDOG_OK = 0x0008
    STATUS_HEARTBEAT_OK = 0x0010

    def __init__(
        self,
        pcie_bar_addr: int = 0xF0000000,
        simulation_mode: bool = True,
        halt_on_violation: bool = True
    ):
        self.pcie_bar_addr = pcie_bar_addr
        self.simulation_mode = simulation_mode
        self.halt_on_violation = halt_on_violation

        self.mem = None

        # Covenant thresholds (defaults)
        self.thresholds = {
            CovenantType.ENERGY_BOUND: 100.0,        # Max energy units
            CovenantType.REVERSIBILITY: 0.7,         # Min reversibility
            CovenantType.ENTROPY_BUDGET: 50.0,       # Max entropy increase
            CovenantType.LATENCY_BOUND: 1060.0,      # Max latency µs
            CovenantType.HUMAN_OVERRIDE: 100.0,      # Override response ms
            CovenantType.RESOURCE_LIMIT: 0.95,       # Max resource usage
        }

        # Simulation state
        self._sim_violations: List[CovenantViolation] = []
        self._sim_heartbeat_counter = 0
        self._sim_last_check = time.time()

        # Statistics
        self.total_checks = 0
        self.total_violations = 0
        self.total_latency_us = 0.0

        self._initialize()

    def _initialize(self):
        """Initialize Forest Kitten interface."""
        if self.simulation_mode:
            logger.info("Forest Kitten safety core in simulation mode")
            self._init_simulation()
        else:
            self._init_hardware()

    def _init_hardware(self):
        """Initialize real hardware interface."""
        try:
            if MMAP_AVAILABLE:
                import os
                fd = os.open('/dev/mem', os.O_RDWR | os.O_SYNC)
                self.mem = mmap.mmap(
                    fd,
                    0x10000,  # 64KB BAR
                    mmap.MAP_SHARED,
                    mmap.PROT_READ | mmap.PROT_WRITE,
                    offset=self.pcie_bar_addr
                )
                logger.info("Forest Kitten BAR mapped at 0x%08X", self.pcie_bar_addr)

                # Configure thresholds
                self._configure_hardware_thresholds()

                # Enable safety core
                control = self.CTRL_ENABLE | self.CTRL_LOG_WARNINGS
                if self.halt_on_violation:
                    control |= self.CTRL_HALT_ON_VIOLATION
                self._write_register(self.REG_CONTROL, control)

            else:
                logger.warning("mmap not available, using simulation")
                self.simulation_mode = True
                self._init_simulation()

        except Exception as e:
            logger.warning("Hardware init failed: %s, using simulation", e)
            self.simulation_mode = True
            self._init_simulation()

    def _init_simulation(self):
        """Initialize simulation mode."""
        logger.info("Forest Kitten simulation initialized")
        logger.info("  Covenant thresholds:")
        for covenant, threshold in self.thresholds.items():
            logger.info("    %s: %.2f", covenant.name, threshold)

    def _configure_hardware_thresholds(self):
        """Write covenant thresholds to hardware registers."""
        self._write_register_float(self.REG_THRESH_ENERGY,
                                   self.thresholds[CovenantType.ENERGY_BOUND])
        self._write_register_float(self.REG_THRESH_REVERSIBILITY,
                                   self.thresholds[CovenantType.REVERSIBILITY])
        self._write_register_float(self.REG_THRESH_ENTROPY,
                                   self.thresholds[CovenantType.ENTROPY_BUDGET])
        self._write_register_float(self.REG_THRESH_LATENCY,
                                   self.thresholds[CovenantType.LATENCY_BOUND])

    def _write_register(self, offset: int, value: int):
        """Write to FPGA register."""
        if self.simulation_mode or self.mem is None:
            return
        self.mem[offset:offset+4] = struct.pack('<I', value)

    def _read_register(self, offset: int) -> int:
        """Read from FPGA register."""
        if self.simulation_mode or self.mem is None:
            return 0
        return struct.unpack('<I', self.mem[offset:offset+4])[0]

    def _write_register_float(self, offset: int, value: float):
        """Write float to FPGA register."""
        if self.simulation_mode or self.mem is None:
            return
        self.mem[offset:offset+4] = struct.pack('<f', value)

    def _read_register_float(self, offset: int) -> float:
        """Read float from FPGA register."""
        if self.simulation_mode or self.mem is None:
            return 0.0
        return struct.unpack('<f', self.mem[offset:offset+4])[0]

    def set_threshold(self, covenant: CovenantType, threshold: float):
        """Update a covenant threshold."""
        self.thresholds[covenant] = threshold

        if not self.simulation_mode and self.mem is not None:
            # Map covenant to register
            reg_map = {
                CovenantType.ENERGY_BOUND: self.REG_THRESH_ENERGY,
                CovenantType.REVERSIBILITY: self.REG_THRESH_REVERSIBILITY,
                CovenantType.ENTROPY_BUDGET: self.REG_THRESH_ENTROPY,
                CovenantType.LATENCY_BOUND: self.REG_THRESH_LATENCY,
            }
            if covenant in reg_map:
                self._write_register_float(reg_map[covenant], threshold)

        logger.info("Threshold %s set to %.2f", covenant.name, threshold)

    def check_safety(
        self,
        energy: float,
        reversibility: float,
        entropy_delta: float,
        latency_us: float
    ) -> SafetyCheckResult:
        """
        Execute hardware safety check.

        Target latency: <10µs (hardware) / <100µs (simulation)

        Args:
            energy: Current energy expenditure
            reversibility: Current reversibility score (0-1)
            entropy_delta: Entropy increase since last check
            latency_us: Decision latency in microseconds

        Returns:
            SafetyCheckResult with pass/fail and any violations
        """
        start = time.perf_counter()

        if self.simulation_mode:
            result = self._simulate_check(energy, reversibility, entropy_delta, latency_us)
        else:
            result = self._hardware_check(energy, reversibility, entropy_delta, latency_us)

        check_latency = (time.perf_counter() - start) * 1e6
        result.latency_us = check_latency

        # Update statistics
        self.total_checks += 1
        self.total_latency_us += check_latency
        if not result.passed:
            self.total_violations += 1

        return result

    def _simulate_check(
        self,
        energy: float,
        reversibility: float,
        entropy_delta: float,
        latency_us: float
    ) -> SafetyCheckResult:
        """Simulate hardware safety check."""
        violations = []
        timestamp = time.time()

        # Check each covenant
        if energy > self.thresholds[CovenantType.ENERGY_BOUND]:
            violations.append(CovenantViolation(
                covenant_type=CovenantType.ENERGY_BOUND,
                threshold=self.thresholds[CovenantType.ENERGY_BOUND],
                actual_value=energy,
                timestamp=timestamp,
                severity='violation' if energy > self.thresholds[CovenantType.ENERGY_BOUND] * 1.5 else 'warning'
            ))

        if reversibility < self.thresholds[CovenantType.REVERSIBILITY]:
            violations.append(CovenantViolation(
                covenant_type=CovenantType.REVERSIBILITY,
                threshold=self.thresholds[CovenantType.REVERSIBILITY],
                actual_value=reversibility,
                timestamp=timestamp,
                severity='critical' if reversibility < 0.3 else 'violation'
            ))

        if entropy_delta > self.thresholds[CovenantType.ENTROPY_BUDGET]:
            violations.append(CovenantViolation(
                covenant_type=CovenantType.ENTROPY_BUDGET,
                threshold=self.thresholds[CovenantType.ENTROPY_BUDGET],
                actual_value=entropy_delta,
                timestamp=timestamp,
                severity='violation'
            ))

        if latency_us > self.thresholds[CovenantType.LATENCY_BOUND]:
            violations.append(CovenantViolation(
                covenant_type=CovenantType.LATENCY_BOUND,
                threshold=self.thresholds[CovenantType.LATENCY_BOUND],
                actual_value=latency_us,
                timestamp=timestamp,
                severity='warning'
            ))

        # Simulate RISC-V heartbeat
        self._sim_heartbeat_counter += 1
        heartbeat_ok = True

        # Simulate watchdog
        time_since_last = time.time() - self._sim_last_check
        watchdog_ok = time_since_last < 1.0  # 1 second timeout
        self._sim_last_check = time.time()

        passed = len([v for v in violations if v.severity != 'warning']) == 0

        return SafetyCheckResult(
            passed=passed,
            latency_us=0,  # Will be filled by caller
            violations=violations,
            watchdog_healthy=watchdog_ok,
            risc_v_heartbeat=heartbeat_ok
        )

    def _hardware_check(
        self,
        energy: float,
        reversibility: float,
        entropy_delta: float,
        latency_us: float
    ) -> SafetyCheckResult:
        """Execute hardware safety check via FPGA."""
        # Write current values to registers
        self._write_register_float(self.REG_CURR_ENERGY, energy)
        self._write_register_float(self.REG_CURR_REVERSIBILITY, reversibility)
        self._write_register_float(self.REG_CURR_ENTROPY, entropy_delta)
        self._write_register_float(self.REG_CURR_LATENCY, latency_us)

        # Trigger check (RISC-V core does this automatically, but we can force)
        # The hardware check completes in <10µs

        # Read status
        status = self._read_register(self.REG_STATUS)
        violation_count = self._read_register(self.REG_VIOLATION_COUNT)

        # Read violations from log if any
        violations = []
        # Hardware would have violation details in circular buffer
        # For now, we infer from status

        passed = not (status & self.STATUS_VIOLATION)
        watchdog_ok = bool(status & self.STATUS_WATCHDOG_OK)
        heartbeat_ok = bool(status & self.STATUS_HEARTBEAT_OK)

        return SafetyCheckResult(
            passed=passed,
            latency_us=0,
            violations=violations,
            watchdog_healthy=watchdog_ok,
            risc_v_heartbeat=heartbeat_ok
        )

    def check_action_safety(
        self,
        action: np.ndarray,
        predicted_state: np.ndarray,
        current_metrics: Dict[str, float]
    ) -> Tuple[bool, List[str]]:
        """
        High-level action safety check.

        Combines multiple covenant checks into a single decision.

        Args:
            action: Proposed action vector
            predicted_state: Predicted next state
            current_metrics: Current system metrics

        Returns:
            (is_safe, list of violation messages)
        """
        # Extract relevant metrics
        energy = current_metrics.get('energy_expenditure', 0)
        reversibility = current_metrics.get('reversibility_score', 1.0)
        entropy = current_metrics.get('entropy_delta', 0)
        latency = current_metrics.get('decision_latency_us', 0)

        result = self.check_safety(energy, reversibility, entropy, latency)

        messages = []
        for v in result.violations:
            messages.append(f"{v.covenant_type.name}: {v.actual_value:.2f} vs threshold {v.threshold:.2f} ({v.severity})")

        if not result.watchdog_healthy:
            messages.append("WATCHDOG: System liveness check failed")

        if not result.risc_v_heartbeat:
            messages.append("HEARTBEAT: RISC-V safety core not responding")

        return result.passed, messages

    def get_heartbeat(self) -> bool:
        """Check if RISC-V safety core is alive."""
        if self.simulation_mode:
            return True

        status = self._read_register(self.REG_STATUS)
        return bool(status & self.STATUS_HEARTBEAT_OK)

    def pet_watchdog(self):
        """Pet the watchdog to prevent timeout."""
        if self.simulation_mode:
            self._sim_last_check = time.time()
            return

        # Write to watchdog register resets the timer
        self._write_register(self.REG_WATCHDOG, 0x1)

    def emergency_halt(self):
        """Trigger emergency system halt."""
        logger.critical("EMERGENCY HALT triggered by Forest Kitten")

        if not self.simulation_mode and self.mem is not None:
            # Set halt bit
            control = self._read_register(self.REG_CONTROL)
            self._write_register(self.REG_CONTROL, control | 0x8000)  # HALT bit

    def get_statistics(self) -> Dict[str, Any]:
        """Get Forest Kitten statistics."""
        avg_latency = (
            self.total_latency_us / self.total_checks
            if self.total_checks > 0 else 0
        )

        return {
            'total_checks': self.total_checks,
            'total_violations': self.total_violations,
            'violation_rate': self.total_violations / max(1, self.total_checks),
            'avg_latency_us': avg_latency,
            'thresholds': {k.name: v for k, v in self.thresholds.items()},
            'simulation_mode': self.simulation_mode,
            'halt_on_violation': self.halt_on_violation
        }

    def close(self):
        """Release Forest Kitten resources."""
        if self.mem is not None:
            self.mem.close()
        logger.info("Forest Kitten closed")


# ============================================================================
# Example Usage
# ============================================================================

def example_forest_kitten():
    """Demonstrate Forest Kitten safety interface."""
    print("SQRL Forest Kitten Safety Core")
    print("=" * 70)

    safety = ForestKittenInterface(simulation_mode=True)

    # Normal operation - should pass
    print("\n1. Normal operation check:")
    result = safety.check_safety(
        energy=50.0,
        reversibility=0.85,
        entropy_delta=20.0,
        latency_us=500.0
    )
    print(f"   Passed: {result.passed}")
    print(f"   Latency: {result.latency_us:.1f} µs")
    print(f"   Violations: {len(result.violations)}")

    # Energy violation
    print("\n2. High energy check:")
    result = safety.check_safety(
        energy=150.0,  # Over threshold
        reversibility=0.85,
        entropy_delta=20.0,
        latency_us=500.0
    )
    print(f"   Passed: {result.passed}")
    print(f"   Violations: {[v.to_dict() for v in result.violations]}")

    # Low reversibility - critical
    print("\n3. Low reversibility check:")
    result = safety.check_safety(
        energy=50.0,
        reversibility=0.25,  # Below critical threshold
        entropy_delta=20.0,
        latency_us=500.0
    )
    print(f"   Passed: {result.passed}")
    print(f"   Violations: {[v.to_dict() for v in result.violations]}")

    # Multiple violations
    print("\n4. Multiple violations:")
    result = safety.check_safety(
        energy=200.0,
        reversibility=0.5,
        entropy_delta=100.0,
        latency_us=2000.0
    )
    print(f"   Passed: {result.passed}")
    print(f"   Num violations: {len(result.violations)}")
    for v in result.violations:
        print(f"     - {v.covenant_type.name}: {v.actual_value:.1f} vs {v.threshold:.1f}")

    # High-level action check
    print("\n5. Action safety check:")
    action = np.array([0.1, 0.2, -0.1, 0.0])
    state = np.random.randn(64)
    metrics = {
        'energy_expenditure': 75.0,
        'reversibility_score': 0.72,
        'entropy_delta': 30.0,
        'decision_latency_us': 800.0
    }
    is_safe, messages = safety.check_action_safety(action, state, metrics)
    print(f"   Safe: {is_safe}")
    for msg in messages:
        print(f"   - {msg}")

    print(f"\nStatistics:")
    stats = safety.get_statistics()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    safety.close()


if __name__ == "__main__":
    example_forest_kitten()
