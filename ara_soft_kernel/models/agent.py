"""
Agent Models
============

Agent specifications and running instances.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from enum import Enum


class AgentKind(str, Enum):
    """Types of agents."""
    SERVICE = "service"      # Long-running service
    TASK = "task"            # One-shot task
    DAEMON = "daemon"        # Background daemon
    EPHEMERAL = "ephemeral"  # Short-lived helper


class AgentPriority(str, Enum):
    """Agent execution priority."""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    IDLE = "idle"


class AgentState(str, Enum):
    """Agent instance states."""
    PENDING = "pending"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class AgentHealth(str, Enum):
    """Agent health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class IntelligenceSpec:
    """Intelligence level requirements."""
    min: float = 0.0     # 0 = deterministic, 1 = full LLM
    max: float = 1.0
    preferred: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {"min": self.min, "max": self.max, "preferred": self.preferred}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> IntelligenceSpec:
        return cls(
            min=data.get("min", 0.0),
            max=data.get("max", 1.0),
            preferred=data.get("preferred", 0.5),
        )


@dataclass
class PlacementSpec:
    """Where an agent can run."""
    preferred: str = "any"  # gpu_rich, cpu_rich, low_latency, any
    required_device_types: List[str] = field(default_factory=list)
    excluded_device_types: List[str] = field(default_factory=list)
    can_migrate: bool = True
    preferred_device_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "preferred": self.preferred,
            "required_device_types": self.required_device_types,
            "excluded_device_types": self.excluded_device_types,
            "can_migrate": self.can_migrate,
            "preferred_device_id": self.preferred_device_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PlacementSpec:
        return cls(
            preferred=data.get("preferred", "any"),
            required_device_types=data.get("required_device_types", []),
            excluded_device_types=data.get("excluded_device_types", []),
            can_migrate=data.get("can_migrate", True),
            preferred_device_id=data.get("preferred_device_id"),
        )


@dataclass
class ResourceSpec:
    """Resource requirements."""
    cpu_cores: float = 1.0
    memory_gb: float = 1.0
    gpu_mem_gb: float = 0.0
    gpu_compute_units: int = 0
    network_mbps: float = 0.0
    priority: AgentPriority = AgentPriority.NORMAL

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cpu_cores": self.cpu_cores,
            "memory_gb": self.memory_gb,
            "gpu_mem_gb": self.gpu_mem_gb,
            "gpu_compute_units": self.gpu_compute_units,
            "network_mbps": self.network_mbps,
            "priority": self.priority.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ResourceSpec:
        priority = data.get("priority", "normal")
        if isinstance(priority, str):
            priority = AgentPriority(priority)
        return cls(
            cpu_cores=data.get("cpu_cores", 1.0),
            memory_gb=data.get("memory_gb", 1.0),
            gpu_mem_gb=data.get("gpu_mem_gb", 0.0),
            gpu_compute_units=data.get("gpu_compute_units", 0),
            network_mbps=data.get("network_mbps", 0.0),
            priority=priority,
        )


@dataclass
class PermissionSpec:
    """Agent permissions."""
    filesystem: List[str] = field(default_factory=list)  # "read:path", "write:path"
    network: List[str] = field(default_factory=list)     # domains or empty
    sensors: List[str] = field(default_factory=list)     # sensor names
    system: List[str] = field(default_factory=list)      # system capabilities
    agents: List[str] = field(default_factory=list)      # other agents

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filesystem": self.filesystem,
            "network": self.network,
            "sensors": self.sensors,
            "system": self.system,
            "agents": self.agents,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PermissionSpec:
        return cls(
            filesystem=data.get("filesystem", []),
            network=data.get("network", []),
            sensors=data.get("sensors", []),
            system=data.get("system", []),
            agents=data.get("agents", []),
        )

    def can_read(self, path: str) -> bool:
        """Check if read access to path is allowed."""
        for perm in self.filesystem:
            if perm.startswith("read:"):
                pattern = perm[5:]
                if self._matches_pattern(path, pattern):
                    return True
        return False

    def can_write(self, path: str) -> bool:
        """Check if write access to path is allowed."""
        for perm in self.filesystem:
            if perm.startswith("write:"):
                pattern = perm[6:]
                if self._matches_pattern(path, pattern):
                    return True
        return False

    def _matches_pattern(self, path: str, pattern: str) -> bool:
        """Simple glob-like pattern matching."""
        if pattern.endswith("*"):
            return path.startswith(pattern[:-1])
        return path == pattern


@dataclass
class LifecycleSpec:
    """Agent lifecycle configuration."""
    startup_timeout_ms: int = 5000
    shutdown_timeout_ms: int = 2000
    health_check_interval_ms: int = 1000
    restart_policy: str = "on_failure"  # never, on_failure, always
    max_restarts: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "startup_timeout_ms": self.startup_timeout_ms,
            "shutdown_timeout_ms": self.shutdown_timeout_ms,
            "health_check_interval_ms": self.health_check_interval_ms,
            "restart_policy": self.restart_policy,
            "max_restarts": self.max_restarts,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LifecycleSpec:
        return cls(
            startup_timeout_ms=data.get("startup_timeout_ms", 5000),
            shutdown_timeout_ms=data.get("shutdown_timeout_ms", 2000),
            health_check_interval_ms=data.get("health_check_interval_ms", 1000),
            restart_policy=data.get("restart_policy", "on_failure"),
            max_restarts=data.get("max_restarts", 3),
        )


@dataclass
class AgentSpec:
    """Complete specification of an agent."""
    name: str
    version: str = "0.1.0"
    kind: AgentKind = AgentKind.SERVICE
    description: str = ""

    intelligence: IntelligenceSpec = field(default_factory=IntelligenceSpec)
    placement: PlacementSpec = field(default_factory=PlacementSpec)
    resources: ResourceSpec = field(default_factory=ResourceSpec)
    permissions: PermissionSpec = field(default_factory=PermissionSpec)
    lifecycle: LifecycleSpec = field(default_factory=LifecycleSpec)

    # Execution
    command: Optional[str] = None
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    working_dir: Optional[str] = None

    # Interfaces
    inputs: List[Dict[str, Any]] = field(default_factory=list)
    outputs: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "kind": self.kind.value,
            "description": self.description,
            "intelligence": self.intelligence.to_dict(),
            "placement": self.placement.to_dict(),
            "resources": self.resources.to_dict(),
            "permissions": self.permissions.to_dict(),
            "lifecycle": self.lifecycle.to_dict(),
            "command": self.command,
            "args": self.args,
            "env": self.env,
            "working_dir": self.working_dir,
            "inputs": self.inputs,
            "outputs": self.outputs,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AgentSpec:
        kind = data.get("kind", "service")
        if isinstance(kind, str):
            kind = AgentKind(kind)
        return cls(
            name=data["name"],
            version=data.get("version", "0.1.0"),
            kind=kind,
            description=data.get("description", ""),
            intelligence=IntelligenceSpec.from_dict(data.get("intelligence", {})),
            placement=PlacementSpec.from_dict(data.get("placement", {})),
            resources=ResourceSpec.from_dict(data.get("resources", {})),
            permissions=PermissionSpec.from_dict(data.get("permissions", {})),
            lifecycle=LifecycleSpec.from_dict(data.get("lifecycle", {})),
            command=data.get("command"),
            args=data.get("args", []),
            env=data.get("env", {}),
            working_dir=data.get("working_dir"),
            inputs=data.get("inputs", []),
            outputs=data.get("outputs", []),
        )

    @classmethod
    def from_json(cls, json_str: str) -> AgentSpec:
        return cls.from_dict(json.loads(json_str))

    def requires_gpu(self) -> bool:
        """Check if agent requires GPU."""
        return self.resources.gpu_mem_gb > 0 or self.resources.gpu_compute_units > 0


@dataclass
class ResourceUsage:
    """Current resource usage of an agent."""
    cpu_percent: float = 0.0
    memory_gb: float = 0.0
    gpu_mem_gb: float = 0.0
    gpu_util_percent: float = 0.0
    network_mbps: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cpu_percent": self.cpu_percent,
            "memory_gb": self.memory_gb,
            "gpu_mem_gb": self.gpu_mem_gb,
            "gpu_util_percent": self.gpu_util_percent,
            "network_mbps": self.network_mbps,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ResourceUsage:
        return cls(
            cpu_percent=data.get("cpu_percent", 0.0),
            memory_gb=data.get("memory_gb", 0.0),
            gpu_mem_gb=data.get("gpu_mem_gb", 0.0),
            gpu_util_percent=data.get("gpu_util_percent", 0.0),
            network_mbps=data.get("network_mbps", 0.0),
        )


@dataclass
class AgentInstance:
    """A running instance of an agent."""
    instance_id: str
    agent_name: str
    agent_version: str
    spec: AgentSpec

    state: AgentState = AgentState.PENDING
    device_id: str = ""
    pid: Optional[int] = None

    started_at: Optional[float] = None
    stopped_at: Optional[float] = None
    last_health_check: Optional[float] = None
    health: AgentHealth = AgentHealth.UNKNOWN

    resource_usage: ResourceUsage = field(default_factory=ResourceUsage)
    restart_count: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "agent_name": self.agent_name,
            "agent_version": self.agent_version,
            "state": self.state.value,
            "device_id": self.device_id,
            "pid": self.pid,
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
            "last_health_check": self.last_health_check,
            "health": self.health.value,
            "resource_usage": self.resource_usage.to_dict(),
            "restart_count": self.restart_count,
            "error": self.error,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_spec(cls, spec: AgentSpec, instance_id: str, device_id: str) -> AgentInstance:
        """Create an instance from a spec."""
        return cls(
            instance_id=instance_id,
            agent_name=spec.name,
            agent_version=spec.version,
            spec=spec,
            device_id=device_id,
        )

    def is_running(self) -> bool:
        """Check if agent is running."""
        return self.state == AgentState.RUNNING

    def is_healthy(self) -> bool:
        """Check if agent is healthy."""
        return self.health == AgentHealth.HEALTHY

    def uptime_seconds(self) -> float:
        """Get agent uptime in seconds."""
        if self.started_at is None:
            return 0.0
        end = self.stopped_at or time.time()
        return end - self.started_at
