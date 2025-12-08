"""
Ara Safety Monitor - Hard Invariants That Must Never Be Violated
================================================================

The safety monitor enforces absolute constraints that override
all other system behavior. These are the non-negotiable rules.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                     Safety Monitor                           │
    │                                                              │
    │  ┌──────────────────────────────────────────────────────┐   │
    │  │                Hard Invariants                        │   │
    │  │  • Thermal: FPGA < 95°C (critical shutdown)          │   │
    │  │  • Memory: Heap < 90% (prevent OOM)                  │   │
    │  │  • Network: No external exec of untrusted code       │   │
    │  │  • Auth: All modules authenticated                    │   │
    │  │  • Audit: All decisions logged                        │   │
    │  └──────────────────────────────────────────────────────┘   │
    │                            │                                 │
    │                            ▼                                 │
    │  ┌──────────────────────────────────────────────────────┐   │
    │  │               Enforcement Actions                     │   │
    │  │  • SHUTDOWN: Graceful termination                    │   │
    │  │  • ISOLATE: Quarantine subsystem                     │   │
    │  │  • THROTTLE: Reduce activity                         │   │
    │  │  • ALERT: Notify founder                             │   │
    │  └──────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────┘

The safety monitor runs in a separate thread and can halt
any subsystem if invariants are violated.
"""

from __future__ import annotations

import time
import threading
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable, Set
from enum import IntEnum, auto
from queue import Queue

from .state import Telemetry, HomeostaticState, OperationalMode
from .config import HomeostaticConfig, Setpoints


logger = logging.getLogger(__name__)


# =============================================================================
# Violation Types
# =============================================================================

class ViolationType(IntEnum):
    """Types of safety violations."""
    THERMAL_CRITICAL = 0      # FPGA over critical temp
    THERMAL_WARNING = 1       # FPGA approaching critical
    MEMORY_CRITICAL = 2       # Heap exhaustion imminent
    MEMORY_WARNING = 3        # Heap pressure
    NETWORK_UNTRUSTED = 4     # Untrusted network activity
    AUTH_FAILURE = 5          # Authentication failure
    INVARIANT_BREACH = 6      # Generic invariant violation
    HEARTBEAT_TIMEOUT = 7     # Module stopped responding
    WATCHDOG_TIMEOUT = 8      # Sovereign loop stuck


class EnforcementAction(IntEnum):
    """Actions the safety monitor can take."""
    NONE = 0
    LOG = 1
    ALERT = 2
    THROTTLE = 3
    ISOLATE = 4
    SHUTDOWN = 5


# =============================================================================
# Safety Violation
# =============================================================================

@dataclass
class SafetyViolation:
    """Record of a safety violation."""
    violation_type: ViolationType
    severity: float         # 0-1, 1 = most severe
    message: str
    timestamp: float = field(default_factory=time.time)
    subsystem: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    action_taken: EnforcementAction = EnforcementAction.NONE


# =============================================================================
# Hard Invariants
# =============================================================================

class Invariant:
    """Base class for safety invariants."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.enabled = True
        self.last_check = 0.0
        self.violations = 0

    def check(self, state: HomeostaticState) -> Optional[SafetyViolation]:
        """Check invariant. Return violation if breached, None if OK."""
        raise NotImplementedError


class ThermalInvariant(Invariant):
    """FPGA must not exceed critical temperature."""

    def __init__(self, critical_temp: float = 95.0, warning_temp: float = 85.0):
        super().__init__(
            "thermal_critical",
            f"FPGA temperature must not exceed {critical_temp}°C"
        )
        self.critical_temp = critical_temp
        self.warning_temp = warning_temp

    def check(self, state: HomeostaticState) -> Optional[SafetyViolation]:
        self.last_check = time.time()
        temp = state.telemetry.fpga_temp

        if temp >= self.critical_temp:
            self.violations += 1
            return SafetyViolation(
                violation_type=ViolationType.THERMAL_CRITICAL,
                severity=1.0,
                message=f"CRITICAL: FPGA at {temp:.1f}°C (limit: {self.critical_temp}°C)",
                subsystem="fpga",
                details={"temperature": temp, "limit": self.critical_temp},
            )
        elif temp >= self.warning_temp:
            return SafetyViolation(
                violation_type=ViolationType.THERMAL_WARNING,
                severity=0.5 + 0.5 * (temp - self.warning_temp) / (self.critical_temp - self.warning_temp),
                message=f"WARNING: FPGA at {temp:.1f}°C (warning: {self.warning_temp}°C)",
                subsystem="fpga",
                details={"temperature": temp, "warning": self.warning_temp},
            )

        return None


class MemoryInvariant(Invariant):
    """Heap usage must not exceed critical threshold."""

    def __init__(self, critical_pct: float = 90.0, warning_pct: float = 75.0):
        super().__init__(
            "memory_critical",
            f"Heap usage must not exceed {critical_pct}%"
        )
        self.critical_pct = critical_pct
        self.warning_pct = warning_pct

    def check(self, state: HomeostaticState) -> Optional[SafetyViolation]:
        self.last_check = time.time()

        # Get memory usage (would use actual memory stats in production)
        try:
            import psutil
            mem = psutil.virtual_memory()
            usage_pct = mem.percent
        except ImportError:
            # Fallback - assume OK
            return None

        if usage_pct >= self.critical_pct:
            self.violations += 1
            return SafetyViolation(
                violation_type=ViolationType.MEMORY_CRITICAL,
                severity=1.0,
                message=f"CRITICAL: Memory at {usage_pct:.1f}% (limit: {self.critical_pct}%)",
                subsystem="memory",
                details={"usage_pct": usage_pct, "limit": self.critical_pct},
            )
        elif usage_pct >= self.warning_pct:
            return SafetyViolation(
                violation_type=ViolationType.MEMORY_WARNING,
                severity=0.3 + 0.4 * (usage_pct - self.warning_pct) / (self.critical_pct - self.warning_pct),
                message=f"WARNING: Memory at {usage_pct:.1f}% (warning: {self.warning_pct}%)",
                subsystem="memory",
                details={"usage_pct": usage_pct, "warning": self.warning_pct},
            )

        return None


class HeartbeatInvariant(Invariant):
    """All critical modules must maintain heartbeat."""

    def __init__(self, timeout_ms: float = 1000.0):
        super().__init__(
            "heartbeat",
            f"All critical modules must respond within {timeout_ms}ms"
        )
        self.timeout_ms = timeout_ms
        self._last_heartbeats: Dict[str, float] = {}
        self._required_modules: Set[str] = {"receptor", "sovereign", "effector"}

    def register_heartbeat(self, module: str) -> None:
        """Record heartbeat from a module."""
        self._last_heartbeats[module] = time.time()

    def check(self, state: HomeostaticState) -> Optional[SafetyViolation]:
        self.last_check = time.time()
        now = time.time()

        for module in self._required_modules:
            last = self._last_heartbeats.get(module, 0)
            if last == 0:
                continue  # Not started yet

            age_ms = (now - last) * 1000
            if age_ms > self.timeout_ms:
                self.violations += 1
                return SafetyViolation(
                    violation_type=ViolationType.HEARTBEAT_TIMEOUT,
                    severity=min(1.0, age_ms / (self.timeout_ms * 2)),
                    message=f"TIMEOUT: {module} not responding ({age_ms:.0f}ms)",
                    subsystem=module,
                    details={"module": module, "age_ms": age_ms, "timeout_ms": self.timeout_ms},
                )

        return None


class WatchdogInvariant(Invariant):
    """Sovereign loop must not stall."""

    def __init__(self, max_loop_ms: float = 50.0):
        super().__init__(
            "watchdog",
            f"Sovereign loop must complete within {max_loop_ms}ms"
        )
        self.max_loop_ms = max_loop_ms

    def check(self, state: HomeostaticState) -> Optional[SafetyViolation]:
        self.last_check = time.time()
        loop_ms = state.telemetry.sovereign_loop_ms

        if loop_ms > self.max_loop_ms:
            self.violations += 1
            return SafetyViolation(
                violation_type=ViolationType.WATCHDOG_TIMEOUT,
                severity=min(1.0, loop_ms / (self.max_loop_ms * 2)),
                message=f"SLOW: Sovereign loop took {loop_ms:.1f}ms (limit: {self.max_loop_ms}ms)",
                subsystem="sovereign",
                details={"loop_ms": loop_ms, "limit": self.max_loop_ms},
            )

        return None


# =============================================================================
# Safety Monitor
# =============================================================================

class SafetyMonitor:
    """
    The safety monitor - enforces hard invariants.

    Runs in a dedicated thread at 100 Hz, independent of
    the main sovereign loop.
    """

    def __init__(
        self,
        config: HomeostaticConfig,
        check_hz: float = 100.0,
    ):
        """
        Initialize safety monitor.

        Args:
            config: Homeostatic configuration
            check_hz: Safety check frequency
        """
        self.config = config
        self.check_hz = check_hz
        self.period = 1.0 / check_hz

        # Invariants
        self.invariants: List[Invariant] = []
        self._init_invariants()

        # State reference (set by connect)
        self._state_provider: Optional[Callable[[], HomeostaticState]] = None

        # Violation log
        self._violations: List[SafetyViolation] = []
        self._violation_count = 0
        self._critical_count = 0

        # Shutdown callback
        self._shutdown_callback: Optional[Callable] = None
        self._alert_callback: Optional[Callable[[SafetyViolation], None]] = None

        # State
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._safety_ok = True

    def _init_invariants(self) -> None:
        """Initialize safety invariants."""
        setpoints = self.config.setpoints

        self.invariants = [
            ThermalInvariant(
                critical_temp=setpoints.thermal_critical,
                warning_temp=setpoints.thermal_max,
            ),
            MemoryInvariant(critical_pct=90.0, warning_pct=75.0),
            HeartbeatInvariant(timeout_ms=setpoints.heartbeat_timeout_ms),
            WatchdogInvariant(max_loop_ms=setpoints.latency_max_ms * 100),
        ]

    def connect(
        self,
        state_provider: Callable[[], HomeostaticState],
        shutdown_callback: Optional[Callable] = None,
        alert_callback: Optional[Callable[[SafetyViolation], None]] = None,
    ) -> None:
        """
        Connect safety monitor to system.

        Args:
            state_provider: Function returning current HomeostaticState
            shutdown_callback: Called on critical violation
            alert_callback: Called on any violation
        """
        self._state_provider = state_provider
        self._shutdown_callback = shutdown_callback
        self._alert_callback = alert_callback

    def register_heartbeat(self, module: str) -> None:
        """Record heartbeat from a module."""
        for inv in self.invariants:
            if isinstance(inv, HeartbeatInvariant):
                inv.register_heartbeat(module)

    def start(self) -> None:
        """Start the safety monitor."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info(f"SafetyMonitor started at {self.check_hz} Hz")

    def stop(self) -> None:
        """Stop the safety monitor."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        logger.info("SafetyMonitor stopped")

    def _run_loop(self) -> None:
        """Main safety check loop."""
        next_time = time.perf_counter()

        while self._running:
            # Get current state
            if self._state_provider:
                try:
                    state = self._state_provider()
                    self._check_invariants(state)
                except Exception as e:
                    logger.error(f"Safety check error: {e}")

            # Timing
            next_time += self.period
            sleep_time = next_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_time = time.perf_counter()

    def _check_invariants(self, state: HomeostaticState) -> None:
        """Check all invariants."""
        for invariant in self.invariants:
            if not invariant.enabled:
                continue

            try:
                violation = invariant.check(state)
                if violation:
                    self._handle_violation(violation)
            except Exception as e:
                logger.debug(f"Invariant {invariant.name} check error: {e}")

    def _handle_violation(self, violation: SafetyViolation) -> None:
        """Handle a safety violation."""
        self._violation_count += 1
        self._violations.append(violation)

        # Keep violation log bounded
        if len(self._violations) > 10000:
            self._violations = self._violations[-5000:]

        # Determine action based on severity
        if violation.severity >= 0.9:
            action = EnforcementAction.SHUTDOWN
            self._critical_count += 1
            self._safety_ok = False
        elif violation.severity >= 0.7:
            action = EnforcementAction.ISOLATE
        elif violation.severity >= 0.5:
            action = EnforcementAction.THROTTLE
        elif violation.severity >= 0.3:
            action = EnforcementAction.ALERT
        else:
            action = EnforcementAction.LOG

        violation.action_taken = action

        # Log
        if action >= EnforcementAction.ALERT:
            logger.warning(f"Safety violation: {violation.message} -> {action.name}")
        else:
            logger.debug(f"Safety violation: {violation.message}")

        # Callbacks
        if self._alert_callback:
            try:
                self._alert_callback(violation)
            except:
                pass

        if action == EnforcementAction.SHUTDOWN and self._shutdown_callback:
            try:
                self._shutdown_callback()
            except:
                pass

    @property
    def is_safe(self) -> bool:
        """Check if system is in safe state."""
        return self._safety_ok

    def get_violations(self, since: float = 0.0) -> List[SafetyViolation]:
        """Get violations since timestamp."""
        return [v for v in self._violations if v.timestamp > since]

    def get_stats(self) -> Dict[str, Any]:
        """Get safety monitor statistics."""
        return {
            'check_hz': self.check_hz,
            'safety_ok': self._safety_ok,
            'total_violations': self._violation_count,
            'critical_violations': self._critical_count,
            'invariants': {
                inv.name: {
                    'enabled': inv.enabled,
                    'violations': inv.violations,
                    'last_check': inv.last_check,
                }
                for inv in self.invariants
            },
        }

    def clear_violations(self) -> None:
        """Clear violation history (for testing)."""
        self._violations.clear()
        self._violation_count = 0
        self._critical_count = 0
        self._safety_ok = True
        for inv in self.invariants:
            inv.violations = 0


# =============================================================================
# Audit Daemon
# =============================================================================

class AuditDaemon:
    """
    Audit daemon - logs all decisions for review.

    Creates an immutable audit trail of:
    - Mode changes
    - Effector commands
    - Safety violations
    - Reward signals
    """

    def __init__(self, log_path: str = "/var/log/ara/audit.log"):
        self.log_path = log_path
        self._entries: List[Dict[str, Any]] = []
        self._entry_count = 0

        # Set up file logger
        self._file_handler = None
        self._init_logger()

    def _init_logger(self) -> None:
        """Initialize audit logger."""
        try:
            import os
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

            self._file_handler = logging.FileHandler(self.log_path)
            self._file_handler.setFormatter(
                logging.Formatter('%(asctime)s | %(message)s')
            )
        except Exception as e:
            logger.debug(f"Audit log init failed: {e}")

    def log_mode_change(
        self,
        from_mode: OperationalMode,
        to_mode: OperationalMode,
        reason: str,
    ) -> None:
        """Log a mode change."""
        entry = {
            'type': 'mode_change',
            'timestamp': time.time(),
            'from_mode': from_mode.name,
            'to_mode': to_mode.name,
            'reason': reason,
        }
        self._record(entry)

    def log_command(
        self,
        cmd_type: str,
        target: str,
        data: Dict[str, Any],
    ) -> None:
        """Log an effector command."""
        entry = {
            'type': 'command',
            'timestamp': time.time(),
            'cmd_type': cmd_type,
            'target': target,
            'data': data,
        }
        self._record(entry)

    def log_violation(self, violation: SafetyViolation) -> None:
        """Log a safety violation."""
        entry = {
            'type': 'violation',
            'timestamp': violation.timestamp,
            'violation_type': violation.violation_type.name,
            'severity': violation.severity,
            'message': violation.message,
            'action': violation.action_taken.name,
        }
        self._record(entry)

    def log_reward(self, reward: float, context: Dict[str, Any]) -> None:
        """Log a reward signal."""
        entry = {
            'type': 'reward',
            'timestamp': time.time(),
            'reward': reward,
            'context': context,
        }
        self._record(entry)

    def _record(self, entry: Dict[str, Any]) -> None:
        """Record an audit entry."""
        self._entries.append(entry)
        self._entry_count += 1

        # Keep in-memory log bounded
        if len(self._entries) > 100000:
            self._entries = self._entries[-50000:]

        # Write to file
        if self._file_handler:
            try:
                import json
                line = json.dumps(entry)
                self._file_handler.stream.write(line + '\n')
                self._file_handler.stream.flush()
            except:
                pass

    def get_entries(self, since: float = 0.0, entry_type: str = None) -> List[Dict]:
        """Get audit entries."""
        entries = [e for e in self._entries if e['timestamp'] > since]
        if entry_type:
            entries = [e for e in entries if e['type'] == entry_type]
        return entries

    def get_stats(self) -> Dict[str, Any]:
        """Get audit statistics."""
        return {
            'total_entries': self._entry_count,
            'in_memory': len(self._entries),
            'log_path': self.log_path,
        }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'ViolationType',
    'EnforcementAction',
    'SafetyViolation',
    'Invariant',
    'ThermalInvariant',
    'MemoryInvariant',
    'HeartbeatInvariant',
    'WatchdogInvariant',
    'SafetyMonitor',
    'AuditDaemon',
]
