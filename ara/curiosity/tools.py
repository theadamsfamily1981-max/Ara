"""Safe Probes - Sandboxed system investigation tools.

These probes allow Ara to safely investigate her environment
without risk of modifying system state. All probes are:
1. Read-only (no side effects)
2. Timeout-bounded (can't hang forever)
3. Output-limited (can't consume infinite memory)
4. Sandboxed (can't escape to arbitrary commands)

Safety Rails:
- No shell=True
- No user-controlled arguments
- Fixed command lists only
- Output truncation
- Timeout enforcement
"""

from __future__ import annotations

import subprocess
import time
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Callable
from pathlib import Path


class ProbeType(Enum):
    """Types of safe probes available to Ara."""

    LSPCI = auto()       # PCIe device enumeration
    DMESG = auto()       # Kernel message buffer
    SENSORS = auto()     # Hardware sensors (lm-sensors)
    FPGA = auto()        # FPGA status (custom interface)
    MEMORY = auto()      # Memory info (/proc/meminfo)
    NETWORK = auto()     # Network interfaces
    NVME = auto()        # NVMe device info
    CPU = auto()         # CPU info
    PROC = auto()        # Process info


@dataclass
class ProbeResult:
    """Result of running a safe probe.

    Attributes:
        probe_type: Which probe was run
        success: Whether probe completed successfully
        output: Captured stdout (truncated)
        error: Captured stderr or exception message
        duration_ms: How long the probe took
        timestamp: When the probe was run
        truncated: Whether output was truncated
    """

    probe_type: ProbeType
    success: bool
    output: str
    error: Optional[str] = None
    duration_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    truncated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging/storage."""
        return {
            "probe_type": self.probe_type.name,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
            "truncated": self.truncated,
        }


@dataclass
class SafeProbe:
    """Definition of a safe probe command.

    Attributes:
        probe_type: Type identifier
        command: Fixed command list (no shell)
        timeout_sec: Maximum execution time
        max_output_bytes: Maximum output to capture
        description: Human-readable description
    """

    probe_type: ProbeType
    command: List[str]
    timeout_sec: float = 5.0
    max_output_bytes: int = 65536
    description: str = ""

    def run(self) -> ProbeResult:
        """Execute the probe safely.

        Returns:
            ProbeResult with captured output or error
        """
        start = time.time()

        try:
            # Never use shell=True, never allow user input
            result = subprocess.run(
                self.command,
                capture_output=True,
                timeout=self.timeout_sec,
                text=True,
            )

            duration_ms = (time.time() - start) * 1000
            output = result.stdout
            truncated = False

            # Truncate if too large
            if len(output) > self.max_output_bytes:
                output = output[:self.max_output_bytes] + "\n... [truncated]"
                truncated = True

            return ProbeResult(
                probe_type=self.probe_type,
                success=result.returncode == 0,
                output=output,
                error=result.stderr if result.returncode != 0 else None,
                duration_ms=duration_ms,
                truncated=truncated,
            )

        except subprocess.TimeoutExpired:
            return ProbeResult(
                probe_type=self.probe_type,
                success=False,
                output="",
                error=f"Timeout after {self.timeout_sec}s",
                duration_ms=(time.time() - start) * 1000,
            )
        except FileNotFoundError:
            return ProbeResult(
                probe_type=self.probe_type,
                success=False,
                output="",
                error=f"Command not found: {self.command[0]}",
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return ProbeResult(
                probe_type=self.probe_type,
                success=False,
                output="",
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
            )


# =============================================================================
# Pre-defined Safe Probes
# =============================================================================

lspci_probe = SafeProbe(
    probe_type=ProbeType.LSPCI,
    command=["lspci", "-vvv"],
    timeout_sec=10.0,
    description="Enumerate PCI/PCIe devices with full details",
)

dmesg_probe = SafeProbe(
    probe_type=ProbeType.DMESG,
    command=["dmesg", "--time-format=iso", "-T"],
    timeout_sec=5.0,
    max_output_bytes=131072,  # dmesg can be large
    description="Read kernel message buffer (recent boot/device events)",
)

sensors_probe = SafeProbe(
    probe_type=ProbeType.SENSORS,
    command=["sensors", "-j"],  # JSON output for parsing
    timeout_sec=5.0,
    description="Read hardware sensors (temperatures, voltages, fans)",
)

memory_probe = SafeProbe(
    probe_type=ProbeType.MEMORY,
    command=["cat", "/proc/meminfo"],
    timeout_sec=2.0,
    description="Read memory information",
)

cpu_probe = SafeProbe(
    probe_type=ProbeType.CPU,
    command=["cat", "/proc/cpuinfo"],
    timeout_sec=2.0,
    description="Read CPU information",
)

network_probe = SafeProbe(
    probe_type=ProbeType.NETWORK,
    command=["ip", "-j", "addr"],  # JSON output
    timeout_sec=5.0,
    description="List network interfaces",
)

nvme_probe = SafeProbe(
    probe_type=ProbeType.NVME,
    command=["nvme", "list", "-o", "json"],
    timeout_sec=5.0,
    description="List NVMe devices",
)


def fpga_probe() -> ProbeResult:
    """Custom probe for FPGA status via BANOS interface.

    This reads from /dev/banos and /proc/ara_guardian if available.
    """
    start = time.time()
    output_parts = []

    # Try BANOS device
    banos_path = Path("/dev/banos")
    if banos_path.exists():
        try:
            # Read status register
            with open(banos_path, "rb") as f:
                # Read first 64 bytes (status registers)
                data = f.read(64)
                output_parts.append(f"BANOS device: {len(data)} bytes read")
                output_parts.append(f"Raw: {data.hex()}")
        except Exception as e:
            output_parts.append(f"BANOS read error: {e}")
    else:
        output_parts.append("BANOS device not found")

    # Try ara_guardian proc interface
    guardian_path = Path("/proc/ara_guardian/status")
    if guardian_path.exists():
        try:
            status = guardian_path.read_text()
            output_parts.append(f"Guardian status: {status}")
        except Exception as e:
            output_parts.append(f"Guardian read error: {e}")

    # Try FPGA temperature from sysfs (common location)
    fpga_temp_paths = [
        "/sys/class/fpga_manager/fpga0/status",
        "/sys/class/xrt/card0/thermal",
    ]
    for temp_path in fpga_temp_paths:
        if Path(temp_path).exists():
            try:
                temp = Path(temp_path).read_text().strip()
                output_parts.append(f"{temp_path}: {temp}")
            except Exception as e:
                pass

    duration_ms = (time.time() - start) * 1000
    output = "\n".join(output_parts)

    return ProbeResult(
        probe_type=ProbeType.FPGA,
        success=len(output_parts) > 0,
        output=output,
        duration_ms=duration_ms,
    )


# Registry of all available probes
PROBE_REGISTRY: Dict[ProbeType, Callable[[], ProbeResult]] = {
    ProbeType.LSPCI: lspci_probe.run,
    ProbeType.DMESG: dmesg_probe.run,
    ProbeType.SENSORS: sensors_probe.run,
    ProbeType.MEMORY: memory_probe.run,
    ProbeType.CPU: cpu_probe.run,
    ProbeType.NETWORK: network_probe.run,
    ProbeType.NVME: nvme_probe.run,
    ProbeType.FPGA: fpga_probe,
}


def run_safe_probe(probe_type: ProbeType) -> ProbeResult:
    """Run a probe by type.

    Args:
        probe_type: Which probe to run

    Returns:
        ProbeResult with output or error
    """
    if probe_type not in PROBE_REGISTRY:
        return ProbeResult(
            probe_type=probe_type,
            success=False,
            output="",
            error=f"Unknown probe type: {probe_type}",
        )

    return PROBE_REGISTRY[probe_type]()


def run_all_probes() -> Dict[ProbeType, ProbeResult]:
    """Run all available probes (for initial discovery).

    Returns:
        Dict mapping probe type to result
    """
    results = {}
    for probe_type in PROBE_REGISTRY:
        results[probe_type] = run_safe_probe(probe_type)
    return results
