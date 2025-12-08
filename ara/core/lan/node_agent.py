"""
Ara Node Agent - Spinal Node Contract
=====================================

Defines the NodeAgent interface for spinal nodes in Ara's nervous system.

Every node in Ara's cluster runs a NodeAgent that:
- Collects local metrics (temp, load, errors)
- Encodes state as hypervectors for HTC
- Emits heartbeats to central soul
- Receives and applies policy updates

Lost heartbeats = "numbness" detection in Sovereign.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import numpy as np

from ara.hd.ops import bind, bundle, DIM
from ara.hd.vocab import get_vocab
from ara.core.lan.soulmesh_protocol import (
    SoulMeshMessage,
    MessageType,
    hash_hv,
    create_somatic_message,
)


# =============================================================================
# Node State
# =============================================================================

@dataclass
class NodeState:
    """Current state of a node in Ara's cluster."""
    node_id: str
    node_type: str = "generic"      # "fpga", "gpu", "cpu", "storage", "edge"

    # Resource utilization (0-1)
    cpu_load: float = 0.0
    gpu_load: float = 0.0
    fpga_load: float = 0.0
    memory_used: float = 0.0

    # Health metrics
    max_temp: float = 40.0          # Celsius
    power_watts: float = 0.0
    error_rate: float = 0.0
    uptime_hours: float = 0.0

    # Network metrics
    rx_bytes_sec: float = 0.0
    tx_bytes_sec: float = 0.0
    latency_ms: float = 0.0

    # Timestamps
    last_update: datetime = field(default_factory=datetime.utcnow)
    last_heartbeat: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "cpu_load": self.cpu_load,
            "gpu_load": self.gpu_load,
            "fpga_load": self.fpga_load,
            "memory_used": self.memory_used,
            "max_temp": self.max_temp,
            "power_watts": self.power_watts,
            "error_rate": self.error_rate,
            "uptime_hours": self.uptime_hours,
            "rx_bytes_sec": self.rx_bytes_sec,
            "tx_bytes_sec": self.tx_bytes_sec,
            "latency_ms": self.latency_ms,
            "last_update": self.last_update.isoformat(),
        }

    def is_healthy(self) -> bool:
        """Quick health check."""
        return (
            self.max_temp < 85 and
            self.error_rate < 0.05 and
            self.cpu_load < 0.95
        )


# =============================================================================
# Node Agent Interface
# =============================================================================

class NodeAgent(ABC):
    """
    Abstract interface for node agents in Ara's nervous system.

    Each physical node runs a NodeAgent that:
    1. Collects local metrics
    2. Encodes state as HVs
    3. Emits heartbeats
    4. Applies policy updates
    """

    @property
    @abstractmethod
    def node_id(self) -> str:
        """Unique node identifier."""
        ...

    @property
    @abstractmethod
    def node_type(self) -> str:
        """Node type (fpga, gpu, cpu, etc.)."""
        ...

    @abstractmethod
    def collect_metrics(self) -> Dict[str, float]:
        """Collect current node metrics."""
        ...

    @abstractmethod
    def get_state(self) -> NodeState:
        """Get current node state."""
        ...

    def get_state_hv(self) -> np.ndarray:
        """Encode current state as hypervector."""
        state = self.get_state()
        return encode_node_state_hv(state)

    def emit_heartbeat(self) -> SoulMeshMessage:
        """Generate a heartbeat message for the central soul."""
        state = self.get_state()
        hv = self.get_state_hv()

        return create_somatic_message(
            node_id=self.node_id,
            metrics=state.to_dict(),
            context_hv=hv,
        )

    @abstractmethod
    def apply_policy(self, policy: Dict[str, Any]) -> bool:
        """
        Apply a policy update from Sovereign.

        Returns True if policy was applied successfully.
        """
        ...


# =============================================================================
# Dummy Node Agent (for testing)
# =============================================================================

class DummyNodeAgent(NodeAgent):
    """
    Dummy node agent for testing without real hardware.

    Generates synthetic metrics that can be controlled for testing.
    """

    def __init__(
        self,
        node_id: str = "dummy-01",
        node_type: str = "cpu",
    ):
        self._node_id = node_id
        self._node_type = node_type
        self._state = NodeState(
            node_id=node_id,
            node_type=node_type,
        )

        # Controllable parameters for testing
        self._base_cpu = 0.3
        self._base_temp = 45.0
        self._noise_level = 0.1

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def node_type(self) -> str:
        return self._node_type

    def collect_metrics(self) -> Dict[str, float]:
        """Generate synthetic metrics."""
        import random

        noise = lambda: (random.random() - 0.5) * 2 * self._noise_level

        return {
            "cpu_load": max(0, min(1, self._base_cpu + noise())),
            "gpu_load": max(0, min(1, 0.2 + noise())),
            "memory_used": max(0, min(1, 0.4 + noise())),
            "max_temp": max(30, min(100, self._base_temp + noise() * 10)),
            "error_rate": max(0, 0.001 + noise() * 0.01),
            "rx_bytes_sec": max(0, 1e6 + noise() * 1e6),
            "tx_bytes_sec": max(0, 5e5 + noise() * 5e5),
            "latency_ms": max(0, 5 + noise() * 10),
        }

    def get_state(self) -> NodeState:
        """Get current state with fresh metrics."""
        metrics = self.collect_metrics()
        self._state.cpu_load = metrics["cpu_load"]
        self._state.gpu_load = metrics["gpu_load"]
        self._state.memory_used = metrics["memory_used"]
        self._state.max_temp = metrics["max_temp"]
        self._state.error_rate = metrics["error_rate"]
        self._state.rx_bytes_sec = metrics["rx_bytes_sec"]
        self._state.tx_bytes_sec = metrics["tx_bytes_sec"]
        self._state.latency_ms = metrics["latency_ms"]
        self._state.last_update = datetime.utcnow()
        return self._state

    def apply_policy(self, policy: Dict[str, Any]) -> bool:
        """Apply policy (just logs for dummy)."""
        print(f"[DummyAgent {self._node_id}] Applying policy: {policy}")
        return True

    def simulate_stress(self, cpu: float = 0.9, temp: float = 75.0):
        """Simulate stressed state for testing."""
        self._base_cpu = cpu
        self._base_temp = temp

    def simulate_normal(self):
        """Return to normal state."""
        self._base_cpu = 0.3
        self._base_temp = 45.0


# =============================================================================
# HV Encoding
# =============================================================================

def encode_node_state_hv(state: NodeState) -> np.ndarray:
    """
    Encode a node state into a hypervector.

    H_node = ROLE_PROPRIOCEPTION ⊕ H_ID ⊕ H_TYPE ⊕ H_CPU ⊕ H_TEMP ⊕ H_ERR
    """
    vocab = get_vocab()

    # Bin continuous values
    def bin_load(v: float) -> str:
        if v < 0.3: return "LOW"
        if v < 0.6: return "MED"
        if v < 0.85: return "HIGH"
        return "CRITICAL"

    def bin_temp(v: float) -> str:
        if v < 50: return "LOW"
        if v < 65: return "MED"
        if v < 80: return "HIGH"
        return "CRITICAL"

    def bin_error(v: float) -> str:
        if v < 0.001: return "ZERO"
        if v < 0.01: return "LOW"
        if v < 0.05: return "MED"
        return "CRITICAL"

    # Create component HVs
    from ara.hd.ops import random_hv_from_string

    h_id = random_hv_from_string(f"NODE:{state.node_id}")
    h_type = random_hv_from_string(f"NODE_TYPE:{state.node_type}")

    components = [
        bind(vocab.feature("NODE_ID"), h_id),
        bind(vocab.feature("NODE_TYPE"), h_type),
        bind(vocab.feature("CPU_LOAD"), vocab.bin(bin_load(state.cpu_load))),
        bind(vocab.feature("GPU_UTIL"), vocab.bin(bin_load(state.gpu_load))),
        bind(vocab.feature("MEMORY_USED"), vocab.bin(bin_load(state.memory_used))),
        bind(vocab.feature("TEMP"), vocab.bin(bin_temp(state.max_temp))),
        bind(vocab.feature("ERR_RATE"), vocab.bin(bin_error(state.error_rate))),
    ]

    h_node = bundle(components)
    h_bound = bind(vocab.role("PROPRIOCEPTION"), h_node)

    return h_bound


# =============================================================================
# Node Registry
# =============================================================================

class NodeRegistry:
    """
    Registry of known nodes in Ara's cluster.

    Tracks node states and detects numbness (lost heartbeats).
    """

    def __init__(self, heartbeat_timeout: timedelta = timedelta(seconds=30)):
        self._nodes: Dict[str, NodeState] = {}
        self._timeout = heartbeat_timeout

    def update(self, state: NodeState) -> None:
        """Update a node's state."""
        state.last_heartbeat = datetime.utcnow()
        self._nodes[state.node_id] = state

    def get(self, node_id: str) -> Optional[NodeState]:
        """Get a node's state."""
        return self._nodes.get(node_id)

    def get_all(self) -> List[NodeState]:
        """Get all known node states."""
        return list(self._nodes.values())

    def get_numb_nodes(self) -> List[str]:
        """Get list of nodes that have gone numb (missed heartbeats)."""
        now = datetime.utcnow()
        numb = []
        for node_id, state in self._nodes.items():
            if state.last_heartbeat is None:
                continue
            if now - state.last_heartbeat > self._timeout:
                numb.append(node_id)
        return numb

    def get_unhealthy_nodes(self) -> List[str]:
        """Get list of unhealthy nodes."""
        return [
            node_id for node_id, state in self._nodes.items()
            if not state.is_healthy()
        ]


__all__ = [
    'NodeState',
    'NodeAgent',
    'DummyNodeAgent',
    'encode_node_state_hv',
    'NodeRegistry',
]
