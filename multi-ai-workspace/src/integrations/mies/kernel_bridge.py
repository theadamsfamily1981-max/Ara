"""MIES Kernel Bridge - Communication with the semantic AI kernel.

This is the pipe from brainstem (kernel snn_ai_engine) to cortex (Ara/MIES).

The kernel exposes hardware state via /dev/snn_ai:
- GPU/FPGA/CPU utilization
- Thermal readings
- Deadline miss rates
- RL confidence levels
- Current policy mode

This bridge reads that state and translates it into KernelPhysiology,
which MIES then maps to SystemPhysiology for mood/presence decisions.

When the kernel device isn't available (development, non-Linux),
we fall back to simulated values or None.
"""

import fcntl
import logging
import struct
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# === Kernel Interface Constants ===

SNN_AI_STATUS_VERSION = 1

# Struct format: 10x uint32
# version, gpu_util, fpga_util, cpu_util, mem_pressure,
# thermal_gpu_mC, thermal_fpga_mC, deadline_miss_ppm, q_confidence, policy_mode
SNN_AI_STATUS_FMT = "=I I I I I I I I I I"
SNN_AI_STATUS_SIZE = struct.calcsize(SNN_AI_STATUS_FMT)

# ioctl magic: 'S' = 0x53, command 0x01
# _IOR('S', 0x01, snn_ai_status_t) -> direction=read, type='S', nr=1
# Simplified calculation: we'll use the direct value
SNN_AI_IOCTL_GET_STATUS = 0x80285301  # _IOR with size 40 bytes


class PolicyMode(IntEnum):
    """Kernel scheduling policy modes."""
    PERFORMANCE = 0       # Max throughput, power be damned
    EFFICIENCY = 1        # Balance power and performance
    LATENCY_CRITICAL = 2  # Minimize jitter, sacrifice throughput
    THERMAL_THROTTLE = 3  # Emergency: reduce load to cool down
    RECOVERY = 4          # Post-fault: conservative operation


@dataclass
class KernelPhysiology:
    """Raw physiological readings from the kernel.

    This is the "raw nerve signal" - direct hardware metrics
    before interpretation into feelings/mood.
    """
    # Utilization (0.0 - 1.0)
    gpu_load: float
    fpga_load: float
    cpu_load: float
    mem_pressure: float

    # Thermal (degrees Celsius)
    thermal_gpu: float
    thermal_fpga: float

    # Health metrics
    miss_rate: float      # Deadline misses (0.0 - 1.0, normalized from ppm)
    q_confidence: float   # RL engine confidence (0.0 - 1.0)

    # Current mode
    policy_mode: PolicyMode

    @property
    def mean_load(self) -> float:
        """Average load across compute resources."""
        return (self.gpu_load + self.fpga_load + self.cpu_load) / 3.0

    @property
    def max_thermal(self) -> float:
        """Hottest component temperature."""
        return max(self.thermal_gpu, self.thermal_fpga)

    @property
    def is_throttling(self) -> bool:
        """Whether we're in thermal throttle mode."""
        return self.policy_mode == PolicyMode.THERMAL_THROTTLE

    @property
    def is_stressed(self) -> bool:
        """Whether the system is under significant stress."""
        return (
            self.miss_rate > 0.01 or  # >1% deadline misses
            self.max_thermal > 85.0 or  # Hot
            self.mean_load > 0.9  # Saturated
        )


class KernelBridge:
    """Bridge to the semantic AI kernel.

    Reads hardware state from /dev/snn_ai and translates it
    into KernelPhysiology for MIES consumption.

    Falls back gracefully when kernel module isn't loaded.
    """

    def __init__(
        self,
        dev_path: str = "/dev/snn_ai",
        fallback_to_proc: bool = True,
        simulate_if_missing: bool = False,
    ):
        """Initialize the kernel bridge.

        Args:
            dev_path: Path to the kernel device file
            fallback_to_proc: If device missing, try /proc/stat etc.
            simulate_if_missing: Return simulated values if no real data
        """
        self.dev_path = Path(dev_path)
        self.fallback_to_proc = fallback_to_proc
        self.simulate_if_missing = simulate_if_missing

        self._last_reading: Optional[KernelPhysiology] = None
        self._read_failures = 0
        self._max_failures_before_warn = 10

    def is_available(self) -> bool:
        """Check if kernel device is available."""
        return self.dev_path.exists()

    def read_physiology(self) -> Optional[KernelPhysiology]:
        """Read current physiological state from kernel.

        Returns:
            KernelPhysiology if successful, None if unavailable
        """
        # Try direct kernel device first
        if self.dev_path.exists():
            result = self._read_from_device()
            if result:
                self._last_reading = result
                self._read_failures = 0
                return result

        # Fallback to /proc/stat and friends
        if self.fallback_to_proc:
            result = self._read_from_proc()
            if result:
                self._last_reading = result
                return result

        # Simulate if configured
        if self.simulate_if_missing:
            return self._simulate_physiology()

        # Track failures for logging
        self._read_failures += 1
        if self._read_failures == self._max_failures_before_warn:
            logger.warning(
                f"Kernel bridge: {self._read_failures} consecutive read failures. "
                f"Device {self.dev_path} may not be available."
            )

        return None

    def _read_from_device(self) -> Optional[KernelPhysiology]:
        """Read directly from /dev/snn_ai via ioctl."""
        try:
            with open(self.dev_path, "rb", buffering=0) as f:
                buf = bytearray(SNN_AI_STATUS_SIZE)
                fcntl.ioctl(f, SNN_AI_IOCTL_GET_STATUS, buf)

                fields = struct.unpack(SNN_AI_STATUS_FMT, buf)
                (
                    version,
                    gpu_util, fpga_util, cpu_util, mem_press,
                    t_gpu_mC, t_fpga_mC,
                    miss_ppm, q_conf, policy_mode
                ) = fields

                # Version check
                if version != SNN_AI_STATUS_VERSION:
                    logger.warning(
                        f"Kernel status version mismatch: got {version}, "
                        f"expected {SNN_AI_STATUS_VERSION}"
                    )

                return KernelPhysiology(
                    gpu_load=gpu_util / 100.0,
                    fpga_load=fpga_util / 100.0,
                    cpu_load=cpu_util / 100.0,
                    mem_pressure=mem_press / 100.0,
                    thermal_gpu=t_gpu_mC / 1000.0,
                    thermal_fpga=t_fpga_mC / 1000.0,
                    miss_rate=min(1.0, miss_ppm / 1_000_000.0),
                    q_confidence=q_conf / 100.0,
                    policy_mode=PolicyMode(min(policy_mode, 4)),
                )

        except FileNotFoundError:
            return None
        except OSError as e:
            logger.debug(f"Kernel device read error: {e}")
            return None
        except struct.error as e:
            logger.warning(f"Kernel status parse error: {e}")
            return None

    def _read_from_proc(self) -> Optional[KernelPhysiology]:
        """Fallback: read from /proc/stat, /sys/class/thermal, etc.

        This gives us basic CPU stats even without the kernel module.
        """
        try:
            # CPU utilization from /proc/stat
            cpu_load = self._read_cpu_load()

            # GPU load from nvidia-smi or /sys (if available)
            gpu_load = self._read_gpu_load()

            # Thermal from /sys/class/thermal
            thermal = self._read_thermal()

            # Memory pressure from /proc/meminfo
            mem_pressure = self._read_mem_pressure()

            return KernelPhysiology(
                gpu_load=gpu_load,
                fpga_load=0.0,  # No FPGA info without kernel module
                cpu_load=cpu_load,
                mem_pressure=mem_pressure,
                thermal_gpu=thermal.get("gpu", 50.0),
                thermal_fpga=thermal.get("fpga", 40.0),
                miss_rate=0.0,  # No deadline info without kernel module
                q_confidence=0.5,  # Unknown
                policy_mode=PolicyMode.EFFICIENCY,
            )

        except Exception as e:
            logger.debug(f"Proc fallback error: {e}")
            return None

    def _read_cpu_load(self) -> float:
        """Read CPU load from /proc/stat."""
        try:
            with open("/proc/stat", "r") as f:
                line = f.readline()
                if line.startswith("cpu "):
                    parts = line.split()[1:]
                    values = [int(x) for x in parts[:7]]
                    # user, nice, system, idle, iowait, irq, softirq
                    total = sum(values)
                    idle = values[3] + values[4]  # idle + iowait
                    if total > 0:
                        return 1.0 - (idle / total)
        except Exception:
            pass
        return 0.5  # Default assumption

    def _read_gpu_load(self) -> float:
        """Try to read GPU load from nvidia-smi or /sys."""
        # Try nvidia-smi first
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=1.0
            )
            if result.returncode == 0:
                return float(result.stdout.strip()) / 100.0
        except Exception:
            pass

        # Try AMD via /sys
        try:
            gpu_busy = Path("/sys/class/drm/card0/device/gpu_busy_percent")
            if gpu_busy.exists():
                return float(gpu_busy.read_text().strip()) / 100.0
        except Exception:
            pass

        return 0.0  # No GPU info available

    def _read_thermal(self) -> dict:
        """Read thermal data from /sys/class/thermal."""
        thermals = {}
        try:
            thermal_base = Path("/sys/class/thermal")
            for zone in thermal_base.glob("thermal_zone*"):
                try:
                    temp_file = zone / "temp"
                    type_file = zone / "type"
                    if temp_file.exists():
                        temp_mC = int(temp_file.read_text().strip())
                        temp_C = temp_mC / 1000.0

                        zone_type = "cpu"
                        if type_file.exists():
                            zone_type = type_file.read_text().strip().lower()

                        # Map to our categories
                        if "gpu" in zone_type or "nvidia" in zone_type:
                            thermals["gpu"] = temp_C
                        elif "cpu" in zone_type or "x86" in zone_type:
                            thermals.setdefault("cpu", temp_C)
                        else:
                            thermals.setdefault("other", temp_C)
                except Exception:
                    continue
        except Exception:
            pass

        return thermals

    def _read_mem_pressure(self) -> float:
        """Read memory pressure from /proc/meminfo."""
        try:
            with open("/proc/meminfo", "r") as f:
                mem_total = 0
                mem_available = 0
                for line in f:
                    if line.startswith("MemTotal:"):
                        mem_total = int(line.split()[1])
                    elif line.startswith("MemAvailable:"):
                        mem_available = int(line.split()[1])

                if mem_total > 0:
                    return 1.0 - (mem_available / mem_total)
        except Exception:
            pass
        return 0.3  # Default assumption

    def _simulate_physiology(self) -> KernelPhysiology:
        """Generate simulated physiology for testing."""
        import random
        import math
        import time

        # Sinusoidal base with noise
        t = time.time()
        base = 0.3 + 0.2 * math.sin(t / 30.0)

        return KernelPhysiology(
            gpu_load=max(0.0, min(1.0, base + random.gauss(0, 0.1))),
            fpga_load=max(0.0, min(1.0, base * 0.5 + random.gauss(0, 0.05))),
            cpu_load=max(0.0, min(1.0, base * 0.8 + random.gauss(0, 0.1))),
            mem_pressure=max(0.0, min(1.0, 0.4 + random.gauss(0, 0.1))),
            thermal_gpu=55.0 + base * 30.0 + random.gauss(0, 2),
            thermal_fpga=45.0 + base * 20.0 + random.gauss(0, 2),
            miss_rate=max(0.0, random.gauss(0.001, 0.002)),
            q_confidence=0.7 + random.gauss(0, 0.1),
            policy_mode=PolicyMode.EFFICIENCY,
        )


# === Factory ===

def create_kernel_bridge(
    dev_path: str = "/dev/snn_ai",
    fallback: bool = True,
    simulate: bool = False,
) -> KernelBridge:
    """Create a kernel bridge instance.

    Args:
        dev_path: Path to kernel device
        fallback: Use /proc fallback if device unavailable
        simulate: Generate simulated values if no real data
    """
    return KernelBridge(
        dev_path=dev_path,
        fallback_to_proc=fallback,
        simulate_if_missing=simulate,
    )


__all__ = [
    "KernelBridge",
    "KernelPhysiology",
    "PolicyMode",
    "create_kernel_bridge",
    "SNN_AI_STATUS_VERSION",
]
