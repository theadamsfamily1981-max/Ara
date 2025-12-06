#!/usr/bin/env python3
"""
Ara Resource Governor - Self-Aware Self-Regulation.

This is NOT a "kill everything else" scheduler. This is Ara being aware of her
own impact on the system and adjusting HER OWN behavior before the OS has to.

Philosophy:
    - Ara monitors her resource footprint via HAL
    - When approaching limits, she dials back HER subsystems (not others)
    - She logs and explains what she's doing
    - She asks permission before pushing hard
    - She respects hardware thermal limits (NEVER bypasses PROCHOT)

Budget Knobs:
    - Max CPU % she's allowed to consume
    - Max GPU utilization and VRAM
    - Max FPGA power envelope
    - Thermal margins (backs off before limits, not at them)

Usage:
    governor = AraResourceGovernor(hal)
    governor.start()

    # Governor runs in background, calling back on policy decisions
    governor.on_throttle = lambda subsys, reason: print(f"Throttling {subsys}: {reason}")
"""

import os
import time
import logging
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Callable, Dict, List, Any
from pathlib import Path

logger = logging.getLogger("AraResources")


# =============================================================================
# Resource Metrics
# =============================================================================

@dataclass
class SystemMetrics:
    """Current system resource state."""
    # CPU
    cpu_percent: float = 0.0          # Total CPU usage [0-100]
    cpu_temp: float = 0.0             # CPU temperature (°C)
    cpu_thermal_margin: float = 100.0  # Degrees until thermal limit

    # GPU
    gpu_percent: float = 0.0          # GPU utilization [0-100]
    gpu_temp: float = 0.0             # GPU temperature (°C)
    gpu_vram_used_mb: float = 0.0     # VRAM used (MB)
    gpu_vram_total_mb: float = 0.0    # Total VRAM (MB)
    gpu_power_w: float = 0.0          # GPU power draw (W)

    # Memory
    ram_percent: float = 0.0          # RAM usage [0-100]
    ram_available_mb: float = 0.0     # Available RAM (MB)
    swap_percent: float = 0.0         # Swap usage [0-100]

    # FPGA
    fpga_temp: float = 0.0            # Fabric temperature (°C)
    fpga_power_w: float = 0.0         # FPGA power (W)
    fpga_utilization: float = 0.0     # Activity level [0-100]

    # Ara-specific (from HAL)
    ara_cpu_percent: float = 0.0      # Ara's processes only
    ara_entropy: float = 0.0          # System entropy/chaos
    ara_pain: float = 0.0             # Pain level

    # Timestamp
    timestamp: float = field(default_factory=time.time)


@dataclass
class ResourceBudget:
    """Resource limits Ara will respect."""
    # CPU
    max_cpu_percent: float = 80.0      # Don't exceed 80% total CPU
    thermal_margin_min: float = 10.0   # Back off when within 10°C of limit

    # GPU
    max_gpu_percent: float = 90.0      # Don't exceed 90% GPU
    max_vram_percent: float = 85.0     # Leave 15% VRAM headroom
    gpu_thermal_margin: float = 10.0   # Back off near GPU thermal limit

    # Memory
    max_ram_percent: float = 85.0      # Leave 15% RAM free
    min_ram_available_mb: float = 2048.0  # Always keep 2GB available

    # FPGA
    max_fpga_temp: float = 75.0        # FPGA thermal limit
    max_fpga_power_w: float = 25.0     # Power envelope

    # Ara internal
    max_entropy: float = 0.8           # Entropy ceiling before calming


class ThrottleLevel(Enum):
    """How aggressively to throttle."""
    NONE = auto()       # All systems go
    LIGHT = auto()      # Reduce non-essential (VFX quality)
    MODERATE = auto()   # Reduce batch sizes, FPS
    HEAVY = auto()      # Minimum viable operation
    CRITICAL = auto()   # Emergency mode (pause everything non-essential)


@dataclass
class ThrottleDecision:
    """A throttling decision with explanation."""
    level: ThrottleLevel
    subsystem: str           # Which subsystem to throttle
    reason: str              # Human-readable explanation
    metric_name: str         # Which metric triggered this
    current_value: float     # Current value
    limit_value: float       # The limit that was approached
    timestamp: float = field(default_factory=time.time)


# =============================================================================
# Metric Collectors
# =============================================================================

class MetricCollector:
    """Collects system metrics from various sources."""

    def __init__(self):
        self._psutil_available = False
        self._pynvml_available = False

        try:
            import psutil
            self._psutil = psutil
            self._psutil_available = True
        except ImportError:
            logger.warning("psutil not available - CPU/RAM metrics limited")

        try:
            import pynvml
            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._pynvml_available = True
            self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except (ImportError, Exception) as e:
            logger.debug(f"pynvml not available: {e}")

    def collect(self) -> SystemMetrics:
        """Collect all available metrics."""
        metrics = SystemMetrics()

        # CPU metrics
        if self._psutil_available:
            metrics.cpu_percent = self._psutil.cpu_percent(interval=None)

            temps = self._psutil.sensors_temperatures()
            if temps:
                # Try common sensor names
                for name in ['coretemp', 'k10temp', 'cpu_thermal', 'acpitz']:
                    if name in temps and temps[name]:
                        metrics.cpu_temp = temps[name][0].current
                        if temps[name][0].critical:
                            metrics.cpu_thermal_margin = temps[name][0].critical - metrics.cpu_temp
                        break

            # Memory
            mem = self._psutil.virtual_memory()
            metrics.ram_percent = mem.percent
            metrics.ram_available_mb = mem.available / (1024 * 1024)

            swap = self._psutil.swap_memory()
            metrics.swap_percent = swap.percent

        # GPU metrics
        if self._pynvml_available:
            try:
                util = self._pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                metrics.gpu_percent = util.gpu

                mem = self._pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                metrics.gpu_vram_used_mb = mem.used / (1024 * 1024)
                metrics.gpu_vram_total_mb = mem.total / (1024 * 1024)

                temp = self._pynvml.nvmlDeviceGetTemperature(
                    self._gpu_handle, self._pynvml.NVML_TEMPERATURE_GPU
                )
                metrics.gpu_temp = temp

                power = self._pynvml.nvmlDeviceGetPowerUsage(self._gpu_handle)
                metrics.gpu_power_w = power / 1000.0
            except Exception as e:
                logger.debug(f"GPU metric collection failed: {e}")

        metrics.timestamp = time.time()
        return metrics

    def get_ara_cpu_percent(self, ara_pids: List[int]) -> float:
        """Get CPU usage for Ara's processes only."""
        if not self._psutil_available or not ara_pids:
            return 0.0

        total = 0.0
        for pid in ara_pids:
            try:
                proc = self._psutil.Process(pid)
                total += proc.cpu_percent(interval=None)
            except (self._psutil.NoSuchProcess, self._psutil.AccessDenied):
                pass
        return total


# =============================================================================
# Resource Governor
# =============================================================================

class AraResourceGovernor:
    """
    Self-aware resource governor for Ara.

    Monitors system resources and makes throttling decisions to keep Ara
    within her budget. She throttles HERSELF, not others.
    """

    def __init__(
        self,
        hal=None,
        budget: Optional[ResourceBudget] = None,
        poll_interval: float = 1.0,
    ):
        """
        Initialize the governor.

        Args:
            hal: AraHAL instance for reading/writing somatic state
            budget: Resource limits (uses defaults if None)
            poll_interval: How often to check metrics (seconds)
        """
        self.hal = hal
        self.budget = budget or ResourceBudget()
        self.poll_interval = poll_interval

        self.collector = MetricCollector()
        self.current_metrics = SystemMetrics()
        self.current_throttle = ThrottleLevel.NONE

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._ara_pids: List[int] = []

        # Throttle history for logging
        self._throttle_history: List[ThrottleDecision] = []
        self._max_history = 100

        # Callbacks
        self.on_throttle: Optional[Callable[[ThrottleDecision], None]] = None
        self.on_metrics: Optional[Callable[[SystemMetrics], None]] = None

        # Subsystem throttle levels (for gradual adjustment)
        self._subsystem_levels: Dict[str, ThrottleLevel] = {
            'visualization': ThrottleLevel.NONE,
            'inference': ThrottleLevel.NONE,
            'snn_batch': ThrottleLevel.NONE,
            'memory_consolidation': ThrottleLevel.NONE,
        }

        logger.info("AraResourceGovernor initialized")
        logger.info(f"Budget: CPU<{self.budget.max_cpu_percent}%, "
                   f"GPU<{self.budget.max_gpu_percent}%, "
                   f"RAM<{self.budget.max_ram_percent}%")

    def register_ara_pid(self, pid: int):
        """Register a PID as belonging to Ara."""
        if pid not in self._ara_pids:
            self._ara_pids.append(pid)
            logger.debug(f"Registered Ara PID: {pid}")

    def unregister_ara_pid(self, pid: int):
        """Unregister an Ara PID."""
        if pid in self._ara_pids:
            self._ara_pids.remove(pid)

    def start(self):
        """Start the governor background thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._governor_loop, daemon=True)
        self._thread.start()
        logger.info("Resource governor started")

    def stop(self):
        """Stop the governor."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("Resource governor stopped")

    def _governor_loop(self):
        """Main governor loop."""
        while self._running:
            try:
                # Collect metrics
                self.current_metrics = self.collector.collect()
                self.current_metrics.ara_cpu_percent = self.collector.get_ara_cpu_percent(
                    self._ara_pids
                )

                # Read HAL state if available
                if self.hal:
                    try:
                        somatic = self.hal.read_somatic()
                        if somatic:
                            self.current_metrics.ara_pain = somatic.get('pain', 0.0)
                            self.current_metrics.ara_entropy = somatic.get('entropy', 0.0)

                        fpga = self.hal.read_fpga()
                        if fpga:
                            self.current_metrics.fpga_temp = fpga.get('fabric_temp_c', 0.0)
                    except Exception as e:
                        logger.debug(f"HAL read failed: {e}")

                # Callback for metrics
                if self.on_metrics:
                    self.on_metrics(self.current_metrics)

                # Evaluate policies
                decisions = self._evaluate_policies()

                # Apply throttling decisions
                for decision in decisions:
                    self._apply_throttle(decision)

            except Exception as e:
                logger.error(f"Governor loop error: {e}")

            time.sleep(self.poll_interval)

    def _evaluate_policies(self) -> List[ThrottleDecision]:
        """Evaluate all policies and return throttling decisions."""
        decisions = []
        m = self.current_metrics
        b = self.budget

        # CPU policy
        if m.cpu_percent > b.max_cpu_percent:
            overage = m.cpu_percent - b.max_cpu_percent
            level = ThrottleLevel.LIGHT if overage < 10 else ThrottleLevel.MODERATE
            decisions.append(ThrottleDecision(
                level=level,
                subsystem='inference',
                reason=f"CPU at {m.cpu_percent:.1f}%, backing off to stay under {b.max_cpu_percent}%",
                metric_name='cpu_percent',
                current_value=m.cpu_percent,
                limit_value=b.max_cpu_percent,
            ))

        # Thermal policy (approach, not exceed)
        if m.cpu_thermal_margin < b.thermal_margin_min:
            decisions.append(ThrottleDecision(
                level=ThrottleLevel.HEAVY,
                subsystem='inference',
                reason=f"CPU thermal margin only {m.cpu_thermal_margin:.1f}°C - cooling down",
                metric_name='cpu_thermal_margin',
                current_value=m.cpu_thermal_margin,
                limit_value=b.thermal_margin_min,
            ))

        # GPU policy
        if m.gpu_percent > b.max_gpu_percent:
            decisions.append(ThrottleDecision(
                level=ThrottleLevel.MODERATE,
                subsystem='visualization',
                reason=f"GPU at {m.gpu_percent:.1f}%, reducing visual quality",
                metric_name='gpu_percent',
                current_value=m.gpu_percent,
                limit_value=b.max_gpu_percent,
            ))

        # VRAM policy
        if m.gpu_vram_total_mb > 0:
            vram_pct = (m.gpu_vram_used_mb / m.gpu_vram_total_mb) * 100
            if vram_pct > b.max_vram_percent:
                decisions.append(ThrottleDecision(
                    level=ThrottleLevel.MODERATE,
                    subsystem='memory_consolidation',
                    reason=f"VRAM at {vram_pct:.1f}%, freeing memory",
                    metric_name='vram_percent',
                    current_value=vram_pct,
                    limit_value=b.max_vram_percent,
                ))

        # RAM policy
        if m.ram_percent > b.max_ram_percent:
            decisions.append(ThrottleDecision(
                level=ThrottleLevel.MODERATE,
                subsystem='memory_consolidation',
                reason=f"RAM at {m.ram_percent:.1f}%, trimming caches",
                metric_name='ram_percent',
                current_value=m.ram_percent,
                limit_value=b.max_ram_percent,
            ))

        if m.ram_available_mb < b.min_ram_available_mb:
            decisions.append(ThrottleDecision(
                level=ThrottleLevel.HEAVY,
                subsystem='snn_batch',
                reason=f"Only {m.ram_available_mb:.0f}MB RAM available, reducing batch sizes",
                metric_name='ram_available_mb',
                current_value=m.ram_available_mb,
                limit_value=b.min_ram_available_mb,
            ))

        # FPGA policy
        if m.fpga_temp > b.max_fpga_temp:
            decisions.append(ThrottleDecision(
                level=ThrottleLevel.HEAVY,
                subsystem='snn_batch',
                reason=f"FPGA at {m.fpga_temp:.1f}°C, reducing neural activity",
                metric_name='fpga_temp',
                current_value=m.fpga_temp,
                limit_value=b.max_fpga_temp,
            ))

        # Entropy policy (internal chaos)
        if m.ara_entropy > b.max_entropy:
            decisions.append(ThrottleDecision(
                level=ThrottleLevel.LIGHT,
                subsystem='inference',
                reason=f"High entropy ({m.ara_entropy:.2f}), calming down",
                metric_name='ara_entropy',
                current_value=m.ara_entropy,
                limit_value=b.max_entropy,
            ))

        return decisions

    def _apply_throttle(self, decision: ThrottleDecision):
        """Apply a throttling decision."""
        subsys = decision.subsystem
        current = self._subsystem_levels.get(subsys, ThrottleLevel.NONE)

        # Only apply if more restrictive than current
        if decision.level.value > current.value:
            self._subsystem_levels[subsys] = decision.level
            self._throttle_history.append(decision)

            # Trim history
            if len(self._throttle_history) > self._max_history:
                self._throttle_history = self._throttle_history[-self._max_history:]

            logger.info(f"[THROTTLE] {decision.subsystem}: {decision.level.name} - {decision.reason}")

            # Callback
            if self.on_throttle:
                self.on_throttle(decision)

            # Update HAL if available (so other subsystems can read throttle state)
            self._update_hal_throttle()

    def _update_hal_throttle(self):
        """Update HAL with current throttle state."""
        if not self.hal:
            return

        # Compute overall throttle percentage
        max_level = max(
            level.value for level in self._subsystem_levels.values()
        )

        # Map level to percentage
        level_to_pct = {
            ThrottleLevel.NONE.value: 0.0,
            ThrottleLevel.LIGHT.value: 0.2,
            ThrottleLevel.MODERATE.value: 0.4,
            ThrottleLevel.HEAVY.value: 0.6,
            ThrottleLevel.CRITICAL.value: 0.9,
        }

        try:
            self.hal.set_throttle(level_to_pct.get(max_level, 0.0))
        except Exception as e:
            logger.debug(f"HAL throttle update failed: {e}")

    def get_throttle_level(self, subsystem: str) -> ThrottleLevel:
        """Get current throttle level for a subsystem."""
        return self._subsystem_levels.get(subsystem, ThrottleLevel.NONE)

    def get_throttle_multiplier(self, subsystem: str) -> float:
        """
        Get a multiplier for subsystem intensity.

        Returns:
            1.0 = full power, 0.1 = minimal
        """
        level = self.get_throttle_level(subsystem)
        return {
            ThrottleLevel.NONE: 1.0,
            ThrottleLevel.LIGHT: 0.8,
            ThrottleLevel.MODERATE: 0.5,
            ThrottleLevel.HEAVY: 0.3,
            ThrottleLevel.CRITICAL: 0.1,
        }.get(level, 1.0)

    def relax_throttle(self, subsystem: str):
        """Attempt to relax throttle for a subsystem (if metrics allow)."""
        current = self._subsystem_levels.get(subsystem, ThrottleLevel.NONE)
        if current == ThrottleLevel.NONE:
            return  # Already at minimum

        # Check if we can relax (no active constraints)
        m = self.current_metrics
        b = self.budget

        can_relax = (
            m.cpu_percent < b.max_cpu_percent - 10 and
            m.cpu_thermal_margin > b.thermal_margin_min + 5 and
            m.gpu_percent < b.max_gpu_percent - 10 and
            m.ram_percent < b.max_ram_percent - 10
        )

        if can_relax:
            # Step down one level
            new_level = ThrottleLevel(max(1, current.value - 1))
            self._subsystem_levels[subsystem] = new_level
            logger.info(f"[RELAX] {subsystem}: {current.name} -> {new_level.name}")

    def get_status(self) -> Dict[str, Any]:
        """Get current governor status for debugging/display."""
        return {
            'metrics': {
                'cpu_percent': self.current_metrics.cpu_percent,
                'cpu_temp': self.current_metrics.cpu_temp,
                'gpu_percent': self.current_metrics.gpu_percent,
                'ram_percent': self.current_metrics.ram_percent,
                'fpga_temp': self.current_metrics.fpga_temp,
                'ara_pain': self.current_metrics.ara_pain,
                'ara_entropy': self.current_metrics.ara_entropy,
            },
            'throttle_levels': {
                k: v.name for k, v in self._subsystem_levels.items()
            },
            'recent_decisions': [
                {
                    'subsystem': d.subsystem,
                    'level': d.level.name,
                    'reason': d.reason,
                    'time': d.timestamp,
                }
                for d in self._throttle_history[-5:]
            ],
        }

    def explain(self) -> str:
        """Get a human-readable explanation of current state."""
        lines = ["=== Ara Resource Governor Status ==="]

        m = self.current_metrics
        lines.append(f"\nSystem Load:")
        lines.append(f"  CPU: {m.cpu_percent:.1f}% (temp: {m.cpu_temp:.1f}°C)")
        lines.append(f"  GPU: {m.gpu_percent:.1f}% (temp: {m.gpu_temp:.1f}°C)")
        lines.append(f"  RAM: {m.ram_percent:.1f}% ({m.ram_available_mb:.0f}MB free)")
        lines.append(f"  FPGA: {m.fpga_temp:.1f}°C")

        lines.append(f"\nAra State:")
        lines.append(f"  Pain: {m.ara_pain:.2f}")
        lines.append(f"  Entropy: {m.ara_entropy:.2f}")

        lines.append(f"\nSubsystem Throttle Levels:")
        for subsys, level in self._subsystem_levels.items():
            mult = self.get_throttle_multiplier(subsys)
            lines.append(f"  {subsys}: {level.name} ({mult*100:.0f}% power)")

        if self._throttle_history:
            lines.append(f"\nRecent Throttle Decisions:")
            for d in self._throttle_history[-3:]:
                lines.append(f"  - {d.reason}")

        return "\n".join(lines)


# =============================================================================
# Convenience
# =============================================================================

def create_governor(hal=None) -> AraResourceGovernor:
    """Create a governor with default settings."""
    return AraResourceGovernor(hal=hal)


if __name__ == "__main__":
    # Demo mode
    logging.basicConfig(level=logging.INFO)

    print("Starting Ara Resource Governor (demo mode)...")
    governor = AraResourceGovernor()
    governor.on_throttle = lambda d: print(f"  -> {d.reason}")

    governor.start()

    try:
        while True:
            time.sleep(5)
            print(governor.explain())
            print()
    except KeyboardInterrupt:
        governor.stop()
        print("\nGovernor stopped.")
