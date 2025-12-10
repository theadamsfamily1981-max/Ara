"""
Supply Profile
==============

Represents current state of available compute resources.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class GpuInfo:
    """Information about a GPU."""
    id: str
    util: float = 0.0              # 0-1
    vram_total_gb: float = 0.0
    vram_free_gb: float = 0.0
    compute_capability: str = ""
    temperature_c: Optional[float] = None
    power_draw_w: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "util": self.util,
            "vram_total_gb": self.vram_total_gb,
            "vram_free_gb": self.vram_free_gb,
            "compute_capability": self.compute_capability,
            "temperature_c": self.temperature_c,
            "power_draw_w": self.power_draw_w,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GpuInfo:
        return cls(
            id=data["id"],
            util=data.get("util", 0.0),
            vram_total_gb=data.get("vram_total_gb", 0.0),
            vram_free_gb=data.get("vram_free_gb", 0.0),
            compute_capability=data.get("compute_capability", ""),
            temperature_c=data.get("temperature_c"),
            power_draw_w=data.get("power_draw_w"),
        )


@dataclass
class NetworkInfo:
    """Network connectivity information."""
    uplink_mbps: float = 0.0
    downlink_mbps: float = 0.0
    latency_ms_to_hive: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uplink_mbps": self.uplink_mbps,
            "downlink_mbps": self.downlink_mbps,
            "latency_ms_to_hive": self.latency_ms_to_hive,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> NetworkInfo:
        return cls(
            uplink_mbps=data.get("uplink_mbps", 0.0),
            downlink_mbps=data.get("downlink_mbps", 0.0),
            latency_ms_to_hive=data.get("latency_ms_to_hive"),
        )


@dataclass
class DeviceInfo:
    """Information about a compute device."""
    id: str
    type: str                      # desktop, laptop, wearable, phone, server
    location: str = "unknown"

    cpu_load: float = 0.0          # 0-1
    cpu_cores_total: int = 1
    cpu_cores_available: int = 1
    cpu_temperature_c: Optional[float] = None

    gpu: List[GpuInfo] = field(default_factory=list)

    memory_total_gb: float = 0.0
    memory_free_gb: float = 0.0

    battery: Optional[float] = None  # 0-1, None if not applicable

    network: NetworkInfo = field(default_factory=NetworkInfo)

    sensors: List[str] = field(default_factory=list)
    displays: List[str] = field(default_factory=list)

    online: bool = True
    last_seen: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "location": self.location,
            "cpu_load": self.cpu_load,
            "cpu_cores_total": self.cpu_cores_total,
            "cpu_cores_available": self.cpu_cores_available,
            "cpu_temperature_c": self.cpu_temperature_c,
            "gpu": [g.to_dict() for g in self.gpu],
            "memory_total_gb": self.memory_total_gb,
            "memory_free_gb": self.memory_free_gb,
            "battery": self.battery,
            "network": self.network.to_dict(),
            "sensors": self.sensors,
            "displays": self.displays,
            "online": self.online,
            "last_seen": self.last_seen,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DeviceInfo:
        return cls(
            id=data["id"],
            type=data["type"],
            location=data.get("location", "unknown"),
            cpu_load=data.get("cpu_load", 0.0),
            cpu_cores_total=data.get("cpu_cores_total", 1),
            cpu_cores_available=data.get("cpu_cores_available", 1),
            cpu_temperature_c=data.get("cpu_temperature_c"),
            gpu=[GpuInfo.from_dict(g) for g in data.get("gpu", [])],
            memory_total_gb=data.get("memory_total_gb", 0.0),
            memory_free_gb=data.get("memory_free_gb", 0.0),
            battery=data.get("battery"),
            network=NetworkInfo.from_dict(data.get("network", {})),
            sensors=data.get("sensors", []),
            displays=data.get("displays", []),
            online=data.get("online", True),
            last_seen=data.get("last_seen", time.time()),
        )

    def total_gpu_memory_free(self) -> float:
        """Get total free GPU memory across all GPUs."""
        return sum(g.vram_free_gb for g in self.gpu)

    def has_gpu(self) -> bool:
        """Check if device has any GPU."""
        return len(self.gpu) > 0

    def is_battery_low(self, threshold: float = 0.25) -> bool:
        """Check if battery is below threshold."""
        if self.battery is None:
            return False
        return self.battery < threshold


@dataclass
class HiveInfo:
    """Information about hive (cloud) resources."""
    available: bool = False
    latency_ms: Optional[float] = None
    quota_gpu_hours: float = 0.0
    quota_api_calls: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "latency_ms": self.latency_ms,
            "quota_remaining": {
                "gpu_hours": self.quota_gpu_hours,
                "api_calls": self.quota_api_calls,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> HiveInfo:
        quota = data.get("quota_remaining", {})
        return cls(
            available=data.get("available", False),
            latency_ms=data.get("latency_ms"),
            quota_gpu_hours=quota.get("gpu_hours", 0.0),
            quota_api_calls=quota.get("api_calls", 0),
        )


@dataclass
class SupplyProfile:
    """Complete supply profile of available compute resources."""
    timestamp: float = field(default_factory=time.time)
    devices: List[DeviceInfo] = field(default_factory=list)
    hive: HiveInfo = field(default_factory=HiveInfo)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "devices": [d.to_dict() for d in self.devices],
            "hive": self.hive.to_dict(),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SupplyProfile:
        return cls(
            timestamp=data.get("timestamp", time.time()),
            devices=[DeviceInfo.from_dict(d) for d in data.get("devices", [])],
            hive=HiveInfo.from_dict(data.get("hive", {})),
        )

    @classmethod
    def from_json(cls, json_str: str) -> SupplyProfile:
        return cls.from_dict(json.loads(json_str))

    def get_device(self, device_id: str) -> Optional[DeviceInfo]:
        """Get device by ID."""
        for device in self.devices:
            if device.id == device_id:
                return device
        return None

    def get_gpu_rich_devices(self, min_vram_gb: float = 4.0) -> List[DeviceInfo]:
        """Get devices with significant GPU resources."""
        return [
            d for d in self.devices
            if d.has_gpu() and d.total_gpu_memory_free() >= min_vram_gb
        ]

    def get_online_devices(self) -> List[DeviceInfo]:
        """Get all online devices."""
        return [d for d in self.devices if d.online]

    def is_resource_constrained(self) -> bool:
        """Check if we're in a resource-constrained state."""
        # Any device with low battery
        for device in self.devices:
            if device.is_battery_low():
                return True
        # No GPUs available
        if not self.get_gpu_rich_devices(min_vram_gb=1.0):
            return True
        return False

    def total_gpu_memory_free(self) -> float:
        """Get total free GPU memory across all devices."""
        return sum(d.total_gpu_memory_free() for d in self.devices)

    def total_memory_free(self) -> float:
        """Get total free system memory across all devices."""
        return sum(d.memory_free_gb for d in self.devices)


def collect_local_supply() -> SupplyProfile:
    """
    Collect supply profile from local system.

    Uses /proc, nvidia-smi, etc. to gather metrics.
    """
    import subprocess

    profile = SupplyProfile()

    # Get hostname as device ID
    hostname = os.uname().nodename

    device = DeviceInfo(
        id=hostname,
        type="desktop",  # Could detect laptop vs desktop
        location="local",
    )

    # CPU info from /proc/stat
    try:
        with open("/proc/stat") as f:
            line = f.readline()
            fields = line.split()[1:5]  # user, nice, system, idle
            total = sum(int(x) for x in fields)
            idle = int(fields[3])
            device.cpu_load = 1.0 - (idle / max(1, total))
    except Exception:
        pass

    # CPU cores
    try:
        device.cpu_cores_total = os.cpu_count() or 1
        device.cpu_cores_available = device.cpu_cores_total
    except Exception:
        pass

    # Memory from /proc/meminfo
    try:
        with open("/proc/meminfo") as f:
            meminfo = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    value = int(parts[1]) / (1024 * 1024)  # KB to GB
                    meminfo[key] = value
            device.memory_total_gb = meminfo.get("MemTotal", 0)
            device.memory_free_gb = meminfo.get("MemAvailable", meminfo.get("MemFree", 0))
    except Exception:
        pass

    # GPU info via nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.total,memory.free,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    gpu = GpuInfo(
                        id=f"gpu-{parts[0]}",
                        util=float(parts[1]) / 100.0,
                        vram_total_gb=float(parts[2]) / 1024.0,
                        vram_free_gb=float(parts[3]) / 1024.0,
                        temperature_c=float(parts[4]) if len(parts) > 4 else None,
                    )
                    device.gpu.append(gpu)
    except Exception:
        pass

    # Battery (for laptops)
    try:
        battery_path = "/sys/class/power_supply/BAT0/capacity"
        if os.path.exists(battery_path):
            with open(battery_path) as f:
                device.battery = int(f.read().strip()) / 100.0
    except Exception:
        pass

    profile.devices.append(device)
    profile.timestamp = time.time()

    return profile
