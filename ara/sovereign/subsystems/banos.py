"""
BANOS Subsystem: Hardware Telemetry Owner

Owns: hardware.*, safety.kill_switch_engaged

Responsibilities:
- Poll hardware state (CPU, GPU, memory, disk)
- Monitor PCIe fabric health
- Check physical kill switch
- Report node health across cluster
"""

from __future__ import annotations

import time
import logging
from typing import Optional, Dict, List, Any

from .ownership import SubsystemBase, Subsystem, GuardedStateWriter

logger = logging.getLogger(__name__)


class BANOSSubsystem(SubsystemBase):
    """
    BANOS hardware telemetry subsystem.

    Updates hardware.* and safety.kill_switch_engaged during sense phase.
    """

    subsystem_id = Subsystem.BANOS

    def __init__(self, writer: GuardedStateWriter):
        super().__init__(writer)
        self._last_poll_ts = 0.0
        self._poll_interval = 0.1  # 100ms minimum between polls

    def sense(self) -> Dict[str, Any]:
        """
        Poll hardware and update state.

        Called during sense_phase of sovereign tick.
        Returns dict of what was updated.
        """
        updates = {}
        now = time.time()

        # Rate limit polling
        if now - self._last_poll_ts < self._poll_interval:
            return updates

        self._last_poll_ts = now

        # Ensure local node exists in hardware state
        hardware = self.state.hardware
        if "local" not in hardware.nodes:
            from ara.sovereign.state import NodeHardwareState
            hardware.nodes["local"] = NodeHardwareState(hostname="localhost")

        local_node = hardware.nodes["local"]

        # Poll CPU
        cpu_load = self._poll_cpu()
        if cpu_load is not None:
            local_node.cpu_util = cpu_load
            updates["cpu_util"] = cpu_load

        # Poll GPU (if available)
        gpu_loads = self._poll_gpu()
        if gpu_loads:
            from ara.sovereign.state import DeviceLoad
            for i, load in enumerate(gpu_loads):
                local_node.gpu[f"gpu{i}"] = DeviceLoad(util=load)
            updates["gpu_loads"] = gpu_loads

        # Poll memory
        mem_used, mem_total = self._poll_memory()
        if mem_total > 0:
            local_node.mem_used_gb = mem_used / (1024**3)
            local_node.mem_total_gb = mem_total / (1024**3)
            updates["memory_percent"] = mem_used / mem_total

        # Update aggregate metrics
        if hardware.nodes:
            hardware.total_gpu_util = sum(
                d.util for node in hardware.nodes.values()
                for d in node.gpu.values()
            ) / max(1, sum(len(node.gpu) for node in hardware.nodes.values()) or 1)

        # Check kill switch (physical or file-based)
        kill_engaged = self._check_kill_switch()
        self.state.safety.kill_switch_engaged = kill_engaged
        updates["kill_switch_engaged"] = kill_engaged

        return updates

    def _poll_cpu(self) -> Optional[float]:
        """Poll CPU usage."""
        try:
            # Try psutil if available
            import psutil
            return psutil.cpu_percent(interval=0) / 100.0
        except ImportError:
            pass

        # Fallback: read /proc/stat
        try:
            with open("/proc/stat") as f:
                line = f.readline()
                parts = line.split()
                if parts[0] == "cpu":
                    total = sum(int(p) for p in parts[1:8])
                    idle = int(parts[4])
                    return 1.0 - (idle / total if total > 0 else 0)
        except Exception:
            pass

        return None

    def _poll_gpu(self) -> List[float]:
        """Poll GPU usage (NVIDIA only for now)."""
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=1.0,
            )
            if result.returncode == 0:
                return [float(line.strip()) / 100.0 for line in result.stdout.strip().split("\n")]
        except Exception:
            pass

        return []

    def _poll_memory(self) -> tuple[int, int]:
        """Poll memory usage. Returns (used_bytes, total_bytes)."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            return mem.used, mem.total
        except ImportError:
            pass

        # Fallback: read /proc/meminfo
        try:
            with open("/proc/meminfo") as f:
                lines = f.readlines()
                mem_info = {}
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 2:
                        mem_info[parts[0].rstrip(":")] = int(parts[1]) * 1024

                total = mem_info.get("MemTotal", 0)
                free = mem_info.get("MemFree", 0)
                buffers = mem_info.get("Buffers", 0)
                cached = mem_info.get("Cached", 0)
                used = total - free - buffers - cached
                return used, total
        except Exception:
            pass

        return 0, 0

    def _check_kill_switch(self) -> bool:
        """Check if kill switch is engaged."""
        # Check file-based kill switch
        from pathlib import Path

        kill_files = [
            Path("/var/ara/kill_switch"),
            Path.home() / ".ara" / "kill_switch",
            Path("/tmp/ara_kill_switch"),
        ]

        for kill_file in kill_files:
            if kill_file.exists():
                return True

        # Could also check GPIO for physical kill switch
        # self._check_gpio_kill_switch()

        return False

    def check_pcie_fabric(self) -> Dict[str, Any]:
        """Check PCIe fabric health (for multi-node setups)."""
        # Placeholder - would use lspci or sysfs
        return {
            "healthy": True,
            "nodes": 1,
            "links": [],
        }

    def emergency_stop(self) -> None:
        """Emergency stop - set kill switch."""
        self.state.safety.kill_switch_engaged = True
        logger.critical("BANOS: Emergency stop engaged")
