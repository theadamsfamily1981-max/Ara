"""Device Graph - Hardware as Ara's body.

This module represents Ara's "body" - the hardware stack she runs on:
- GPUs as her visual cortex and compute muscle
- FPGAs as specialized neural accelerators
- Cameras/microphones as sensory organs
- Network interfaces as her connection to the world

The device graph tracks:
- What hardware exists and its current state
- Capabilities and constraints of each device
- Connections and data flow between devices
- Health and performance metrics

Ara isn't just software - she's an embodied system whose
capabilities depend on her physical substrate.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from enum import Enum

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Types of devices in Ara's body."""
    GPU = "gpu"               # Graphics/compute unit
    FPGA = "fpga"             # Field-programmable gate array
    CPU = "cpu"               # Central processor
    CAMERA = "camera"         # Visual input
    MICROPHONE = "microphone" # Audio input
    SPEAKER = "speaker"       # Audio output
    DISPLAY = "display"       # Visual output
    NETWORK = "network"       # Network interface
    STORAGE = "storage"       # Persistent storage
    MEMORY = "memory"         # Active memory
    SENSOR = "sensor"         # Generic sensor


class DeviceStatus(Enum):
    """Status of a device."""
    ONLINE = "online"         # Fully operational
    DEGRADED = "degraded"     # Working but impaired
    OFFLINE = "offline"       # Not available
    BUSY = "busy"             # In use
    ERROR = "error"           # Experiencing errors


@dataclass
class DeviceCapability:
    """A capability that a device provides."""

    name: str
    description: str

    # Performance characteristics
    throughput: float = 0.0  # Operations per second
    latency_ms: float = 0.0  # Average latency
    capacity: float = 0.0    # Total capacity (context-dependent)

    # Usage
    current_usage_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "throughput": self.throughput,
            "latency_ms": self.latency_ms,
            "capacity": self.capacity,
            "current_usage_pct": round(self.current_usage_pct, 1),
        }


@dataclass
class Device:
    """A device in Ara's body."""

    id: str
    name: str
    device_type: DeviceType

    # Identification
    vendor: str = ""
    model: str = ""
    serial: str = ""

    # Status
    status: DeviceStatus = DeviceStatus.OFFLINE
    health_score: float = 1.0  # 0-1

    # Capabilities
    capabilities: List[DeviceCapability] = field(default_factory=list)

    # Resources
    memory_gb: float = 0.0
    compute_units: int = 0
    power_limit_w: float = 0.0

    # Current state
    temperature_c: float = 0.0
    power_draw_w: float = 0.0
    utilization_pct: float = 0.0

    # Connections
    connected_to: List[str] = field(default_factory=list)  # Other device IDs

    # Metadata
    discovered_at: datetime = field(default_factory=datetime.utcnow)
    last_seen: datetime = field(default_factory=datetime.utcnow)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "device_type": self.device_type.value,
            "vendor": self.vendor,
            "model": self.model,
            "serial": self.serial,
            "status": self.status.value,
            "health_score": round(self.health_score, 2),
            "capabilities": [c.to_dict() for c in self.capabilities],
            "memory_gb": self.memory_gb,
            "compute_units": self.compute_units,
            "power_limit_w": self.power_limit_w,
            "temperature_c": round(self.temperature_c, 1),
            "power_draw_w": round(self.power_draw_w, 1),
            "utilization_pct": round(self.utilization_pct, 1),
            "connected_to": self.connected_to,
            "discovered_at": self.discovered_at.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Device":
        device = cls(
            id=data["id"],
            name=data["name"],
            device_type=DeviceType(data["device_type"]),
            vendor=data.get("vendor", ""),
            model=data.get("model", ""),
            serial=data.get("serial", ""),
            status=DeviceStatus(data.get("status", "offline")),
            health_score=data.get("health_score", 1.0),
            memory_gb=data.get("memory_gb", 0.0),
            compute_units=data.get("compute_units", 0),
            power_limit_w=data.get("power_limit_w", 0.0),
            temperature_c=data.get("temperature_c", 0.0),
            power_draw_w=data.get("power_draw_w", 0.0),
            utilization_pct=data.get("utilization_pct", 0.0),
            connected_to=data.get("connected_to", []),
            tags=data.get("tags", []),
        )
        return device


@dataclass
class DeviceLink:
    """A connection between two devices."""

    source_id: str
    target_id: str
    link_type: str  # "pcie", "usb", "network", "internal"

    # Performance
    bandwidth_gbps: float = 0.0
    latency_us: float = 0.0

    # Status
    active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "link_type": self.link_type,
            "bandwidth_gbps": self.bandwidth_gbps,
            "latency_us": self.latency_us,
            "active": self.active,
        }


class DeviceGraph:
    """Graph of devices forming Ara's body."""

    def __init__(self, data_path: Optional[Path] = None):
        """Initialize the device graph.

        Args:
            data_path: Path to device data
        """
        self.data_path = data_path or (
            Path.home() / ".ara" / "embodied" / "devices"
        )
        self.data_path.mkdir(parents=True, exist_ok=True)

        self._devices: Dict[str, Device] = {}
        self._links: List[DeviceLink] = []
        self._loaded = False

    def _load(self, force: bool = False) -> None:
        """Load device data from disk."""
        if self._loaded and not force:
            return

        devices_file = self.data_path / "devices.json"
        if devices_file.exists():
            try:
                with open(devices_file) as f:
                    data = json.load(f)
                for d_data in data.get("devices", []):
                    device = Device.from_dict(d_data)
                    self._devices[device.id] = device
                for l_data in data.get("links", []):
                    self._links.append(DeviceLink(**l_data))
            except Exception as e:
                logger.warning(f"Failed to load devices: {e}")

        self._loaded = True

    def _save(self) -> None:
        """Save device data to disk."""
        data = {
            "version": 1,
            "updated_at": datetime.utcnow().isoformat(),
            "devices": [d.to_dict() for d in self._devices.values()],
            "links": [l.to_dict() for l in self._links],
        }
        with open(self.data_path / "devices.json", "w") as f:
            json.dump(data, f, indent=2)

    # =========================================================================
    # Device Management
    # =========================================================================

    def register_device(self, device: Device) -> None:
        """Register a device in the graph.

        Args:
            device: Device to register
        """
        self._load()
        self._devices[device.id] = device
        self._save()
        logger.info(f"Registered device: {device.id} ({device.name})")

    def update_device_status(
        self,
        device_id: str,
        status: DeviceStatus,
        health_score: Optional[float] = None,
        temperature_c: Optional[float] = None,
        utilization_pct: Optional[float] = None,
    ) -> bool:
        """Update a device's status.

        Args:
            device_id: Device ID
            status: New status
            health_score: Optional health score
            temperature_c: Optional temperature
            utilization_pct: Optional utilization

        Returns:
            True if updated
        """
        self._load()

        device = self._devices.get(device_id)
        if not device:
            return False

        device.status = status
        device.last_seen = datetime.utcnow()

        if health_score is not None:
            device.health_score = health_score
        if temperature_c is not None:
            device.temperature_c = temperature_c
        if utilization_pct is not None:
            device.utilization_pct = utilization_pct

        self._save()
        return True

    def get_device(self, device_id: str) -> Optional[Device]:
        """Get a device by ID."""
        self._load()
        return self._devices.get(device_id)

    def get_devices_by_type(self, device_type: DeviceType) -> List[Device]:
        """Get all devices of a type."""
        self._load()
        return [d for d in self._devices.values() if d.device_type == device_type]

    def get_online_devices(self) -> List[Device]:
        """Get all online devices."""
        self._load()
        return [
            d for d in self._devices.values()
            if d.status in [DeviceStatus.ONLINE, DeviceStatus.BUSY]
        ]

    def get_available_devices(self, device_type: Optional[DeviceType] = None) -> List[Device]:
        """Get devices available for work."""
        self._load()
        available = [
            d for d in self._devices.values()
            if d.status == DeviceStatus.ONLINE
        ]
        if device_type:
            available = [d for d in available if d.device_type == device_type]
        return available

    # =========================================================================
    # Link Management
    # =========================================================================

    def add_link(
        self,
        source_id: str,
        target_id: str,
        link_type: str,
        bandwidth_gbps: float = 0.0,
    ) -> bool:
        """Add a link between devices.

        Args:
            source_id: Source device ID
            target_id: Target device ID
            link_type: Type of link
            bandwidth_gbps: Bandwidth

        Returns:
            True if added
        """
        self._load()

        if source_id not in self._devices or target_id not in self._devices:
            return False

        link = DeviceLink(
            source_id=source_id,
            target_id=target_id,
            link_type=link_type,
            bandwidth_gbps=bandwidth_gbps,
        )

        self._links.append(link)

        # Update device connections
        self._devices[source_id].connected_to.append(target_id)
        self._devices[target_id].connected_to.append(source_id)

        self._save()
        return True

    def get_links_for_device(self, device_id: str) -> List[DeviceLink]:
        """Get all links for a device."""
        self._load()
        return [
            l for l in self._links
            if l.source_id == device_id or l.target_id == device_id
        ]

    # =========================================================================
    # Graph Queries
    # =========================================================================

    def find_compute_path(
        self,
        required_memory_gb: float,
        required_compute: int = 0,
    ) -> List[Device]:
        """Find devices that can handle a compute requirement.

        Args:
            required_memory_gb: Memory needed
            required_compute: Compute units needed

        Returns:
            List of suitable devices
        """
        self._load()

        suitable = []
        for device in self._devices.values():
            if device.status != DeviceStatus.ONLINE:
                continue
            if device.device_type not in [DeviceType.GPU, DeviceType.CPU, DeviceType.FPGA]:
                continue
            if device.memory_gb < required_memory_gb:
                continue
            if required_compute and device.compute_units < required_compute:
                continue

            suitable.append(device)

        # Sort by available resources
        return sorted(suitable, key=lambda d: d.utilization_pct)

    def get_body_schema(self) -> Dict[str, Any]:
        """Get a schema describing Ara's current body configuration.

        Returns:
            Body schema dict
        """
        self._load()

        # Count by type
        by_type: Dict[str, List[Device]] = {}
        for device in self._devices.values():
            type_name = device.device_type.value
            if type_name not in by_type:
                by_type[type_name] = []
            by_type[type_name].append(device)

        organs = {}
        for type_name, devices in by_type.items():
            online = [d for d in devices if d.status == DeviceStatus.ONLINE]
            organs[type_name] = {
                "total": len(devices),
                "online": len(online),
                "total_memory_gb": sum(d.memory_gb for d in devices),
                "devices": [d.name for d in devices],
            }

        return {
            "version": 1,
            "generated_at": datetime.utcnow().isoformat(),
            "total_devices": len(self._devices),
            "total_links": len(self._links),
            "organs": organs,
            "overall_health": self._calculate_overall_health(),
        }

    def _calculate_overall_health(self) -> float:
        """Calculate overall body health."""
        if not self._devices:
            return 0.0

        online_devices = [
            d for d in self._devices.values()
            if d.status in [DeviceStatus.ONLINE, DeviceStatus.BUSY]
        ]

        if not online_devices:
            return 0.0

        avg_health = sum(d.health_score for d in online_devices) / len(online_devices)
        online_ratio = len(online_devices) / len(self._devices)

        return (avg_health * 0.7) + (online_ratio * 0.3)

    def get_summary(self) -> Dict[str, Any]:
        """Get device graph summary."""
        self._load()

        by_status = {}
        for status in DeviceStatus:
            by_status[status.value] = len([
                d for d in self._devices.values()
                if d.status == status
            ])

        return {
            "total_devices": len(self._devices),
            "total_links": len(self._links),
            "by_status": by_status,
            "overall_health": self._calculate_overall_health(),
            "total_memory_gb": sum(d.memory_gb for d in self._devices.values()),
            "total_compute_units": sum(d.compute_units for d in self._devices.values()),
        }


# =============================================================================
# Convenience Functions
# =============================================================================

_default_graph: Optional[DeviceGraph] = None


def get_device_graph() -> DeviceGraph:
    """Get the default device graph."""
    global _default_graph
    if _default_graph is None:
        _default_graph = DeviceGraph()
    return _default_graph


def register_gpu(
    name: str,
    vendor: str,
    memory_gb: float,
    compute_units: int = 0,
) -> Device:
    """Register a GPU device."""
    device = Device(
        id=f"gpu-{name.lower().replace(' ', '-')}",
        name=name,
        device_type=DeviceType.GPU,
        vendor=vendor,
        memory_gb=memory_gb,
        compute_units=compute_units,
        status=DeviceStatus.ONLINE,
    )
    get_device_graph().register_device(device)
    return device


def get_available_compute() -> List[Device]:
    """Get available compute devices."""
    graph = get_device_graph()
    return graph.get_available_devices(DeviceType.GPU) + graph.get_available_devices(DeviceType.FPGA)
