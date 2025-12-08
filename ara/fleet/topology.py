"""
Fleet Topology - Node Roles and Structure
==========================================

Defines the organizational structure of the fleet:
- Roles (what a node is for)
- Auth levels (what it can do)
- Capabilities (what hardware it has)
- Topology (how nodes relate)

Fleet roles map to the OrgChart:
    INTERN      - Expendable, fast to rebuild
    WORKER      - Does real work, medium trust
    CONSULTANT  - Critical infrastructure, high trust, gated changes
    MEDIC       - Safety controller, can only turn things off
    BRAINSTEM   - Orchestrator, never runs experiments
    ARCHIVIST   - Storage, read-heavy, writes via pipelines
    SHIELD      - Observer, reads metrics, never pushes config
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum, auto
import time
import hashlib


class FleetRole(Enum):
    """Roles a node can have in the fleet."""
    # Core compute
    INTERN = auto()         # Sacrificial, experimental
    WORKER = auto()         # Production workloads
    CONSULTANT = auto()     # Critical infrastructure

    # Robustness layer
    MEDIC = auto()          # Out-of-band safety (Watcher)
    BRAINSTEM = auto()      # Always-on orchestrator
    ARCHIVIST = auto()      # Storage with snapshots
    SHIELD = auto()         # Network observer

    # Special
    SENSOR_HUB = auto()     # Environmental sensors
    POWER_SPINE = auto()    # UPS/PDU controller


class AuthLevel(Enum):
    """Authorization levels for actions."""
    NONE = 0            # No access
    READ_ONLY = 1       # Can observe
    EXECUTE = 2         # Can run predefined actions
    MODIFY = 3          # Can change configuration
    ADMIN = 4           # Full control (requires signature)


class NodeCapability(Enum):
    """Hardware capabilities a node might have."""
    # Compute
    GPU = auto()            # Has GPU(s)
    FPGA = auto()           # Has FPGA(s)
    HIGH_RAM = auto()       # >64GB RAM
    FAST_STORAGE = auto()   # NVMe SSD

    # Robustness
    UPS_BACKED = auto()     # On UPS power
    OUT_OF_BAND = auto()    # Has OOB management (IPMI, iLO, etc.)
    RELAY_CONTROL = auto()  # Can control power relays

    # Network
    DUAL_NIC = auto()       # Multiple network interfaces
    SPAN_PORT = auto()      # Connected to mirror port

    # Sensors
    TEMP_SENSOR = auto()
    POWER_METER = auto()
    ENV_SENSOR = auto()     # Humidity, light, etc.


@dataclass
class FleetNode:
    """A node in the fleet."""
    node_id: str
    hostname: str
    role: FleetRole
    capabilities: Set[NodeCapability] = field(default_factory=set)

    # Network
    ip_address: str = ""
    mac_address: str = ""

    # Authorization
    auth_level: AuthLevel = AuthLevel.READ_ONLY

    # Power management
    power_outlet: Optional[str] = None  # PDU outlet ID
    on_ups: bool = False

    # Health
    last_seen: float = field(default_factory=time.time)
    health_score: float = 1.0

    # Relationships
    controlled_by: Optional[str] = None     # Node ID of controller
    controls: List[str] = field(default_factory=list)  # Node IDs we control

    # Metadata
    description: str = ""
    tags: Set[str] = field(default_factory=set)

    def is_alive(self, timeout_seconds: float = 60.0) -> bool:
        """Check if node has been seen recently."""
        return time.time() - self.last_seen < timeout_seconds

    def can_execute(self, action: str, required_level: AuthLevel) -> bool:
        """Check if node can execute an action."""
        return self.auth_level.value >= required_level.value

    def update_health(self, score: float):
        """Update health score."""
        self.health_score = max(0.0, min(1.0, score))
        self.last_seen = time.time()


@dataclass
class FleetTopology:
    """
    The complete fleet topology.

    Maps nodes, their relationships, and provides queries
    for finding nodes by role/capability.
    """
    nodes: Dict[str, FleetNode] = field(default_factory=dict)

    # Role-based indices
    _by_role: Dict[FleetRole, Set[str]] = field(default_factory=dict)
    _by_capability: Dict[NodeCapability, Set[str]] = field(default_factory=dict)

    def add_node(self, node: FleetNode):
        """Add a node to the topology."""
        self.nodes[node.node_id] = node

        # Index by role
        if node.role not in self._by_role:
            self._by_role[node.role] = set()
        self._by_role[node.role].add(node.node_id)

        # Index by capability
        for cap in node.capabilities:
            if cap not in self._by_capability:
                self._by_capability[cap] = set()
            self._by_capability[cap].add(node.node_id)

    def remove_node(self, node_id: str):
        """Remove a node from the topology."""
        if node_id not in self.nodes:
            return

        node = self.nodes[node_id]

        # Remove from role index
        if node.role in self._by_role:
            self._by_role[node.role].discard(node_id)

        # Remove from capability indices
        for cap in node.capabilities:
            if cap in self._by_capability:
                self._by_capability[cap].discard(node_id)

        del self.nodes[node_id]

    def get_by_role(self, role: FleetRole) -> List[FleetNode]:
        """Get all nodes with a specific role."""
        node_ids = self._by_role.get(role, set())
        return [self.nodes[nid] for nid in node_ids if nid in self.nodes]

    def get_by_capability(self, cap: NodeCapability) -> List[FleetNode]:
        """Get all nodes with a specific capability."""
        node_ids = self._by_capability.get(cap, set())
        return [self.nodes[nid] for nid in node_ids if nid in self.nodes]

    def get_interns(self) -> List[FleetNode]:
        """Get all intern nodes (sacrificial)."""
        return self.get_by_role(FleetRole.INTERN)

    def get_workers(self) -> List[FleetNode]:
        """Get all worker nodes."""
        return self.get_by_role(FleetRole.WORKER)

    def get_consultants(self) -> List[FleetNode]:
        """Get all consultant nodes (critical)."""
        return self.get_by_role(FleetRole.CONSULTANT)

    def get_brainstem(self) -> Optional[FleetNode]:
        """Get the brainstem node (should be exactly one)."""
        nodes = self.get_by_role(FleetRole.BRAINSTEM)
        return nodes[0] if nodes else None

    def get_watcher(self) -> Optional[FleetNode]:
        """Get the watcher/medic node."""
        nodes = self.get_by_role(FleetRole.MEDIC)
        return nodes[0] if nodes else None

    def get_archivist(self) -> Optional[FleetNode]:
        """Get the archivist/NAS node."""
        nodes = self.get_by_role(FleetRole.ARCHIVIST)
        return nodes[0] if nodes else None

    def get_shield(self) -> Optional[FleetNode]:
        """Get the shield/network observer node."""
        nodes = self.get_by_role(FleetRole.SHIELD)
        return nodes[0] if nodes else None

    def get_healthy_nodes(self, threshold: float = 0.5) -> List[FleetNode]:
        """Get all nodes with health above threshold."""
        return [n for n in self.nodes.values() if n.health_score >= threshold]

    def get_gpu_nodes(self) -> List[FleetNode]:
        """Get all nodes with GPUs."""
        return self.get_by_capability(NodeCapability.GPU)

    def get_fpga_nodes(self) -> List[FleetNode]:
        """Get all nodes with FPGAs."""
        return self.get_by_capability(NodeCapability.FPGA)

    def get_ups_backed_nodes(self) -> List[FleetNode]:
        """Get all nodes on UPS power."""
        return self.get_by_capability(NodeCapability.UPS_BACKED)

    def get_control_tree(self, node_id: str) -> Dict[str, Any]:
        """Get the control tree for a node (what it controls)."""
        if node_id not in self.nodes:
            return {}

        node = self.nodes[node_id]
        tree = {
            "node": node.node_id,
            "role": node.role.name,
            "controls": [],
        }

        for child_id in node.controls:
            if child_id in self.nodes:
                tree["controls"].append(self.get_control_tree(child_id))

        return tree

    def summary(self) -> Dict[str, Any]:
        """Get topology summary."""
        role_counts = {}
        for role in FleetRole:
            count = len(self.get_by_role(role))
            if count > 0:
                role_counts[role.name] = count

        capability_counts = {}
        for cap in NodeCapability:
            count = len(self.get_by_capability(cap))
            if count > 0:
                capability_counts[cap.name] = count

        healthy = len(self.get_healthy_nodes())
        total = len(self.nodes)

        return {
            "total_nodes": total,
            "healthy_nodes": healthy,
            "roles": role_counts,
            "capabilities": capability_counts,
        }


def create_default_topology() -> FleetTopology:
    """
    Create a default fleet topology matching the cathedral architecture.

    This represents a typical Ara deployment:
    - 1 Brainstem (orchestrator)
    - 1 Watcher (safety controller)
    - 1 Archivist (NAS)
    - 1 Shield (network observer)
    - 1+ GPU Workers
    - 1+ FPGA Workers
    - 1+ Interns (VMs)
    """
    topology = FleetTopology()

    # === Brainstem ===
    brainstem = FleetNode(
        node_id="brainstem-01",
        hostname="brainstem",
        role=FleetRole.BRAINSTEM,
        capabilities={
            NodeCapability.UPS_BACKED,
        },
        auth_level=AuthLevel.ADMIN,
        on_ups=True,
        description="Always-on orchestrator. Holds global view.",
        tags={"critical", "orchestrator"},
    )
    topology.add_node(brainstem)

    # === Watcher (Medic) ===
    watcher = FleetNode(
        node_id="watcher-01",
        hostname="watcher",
        role=FleetRole.MEDIC,
        capabilities={
            NodeCapability.OUT_OF_BAND,
            NodeCapability.RELAY_CONTROL,
            NodeCapability.POWER_METER,
            NodeCapability.UPS_BACKED,
        },
        auth_level=AuthLevel.EXECUTE,  # Can execute power actions only
        on_ups=True,
        description="Out-of-band safety controller. Can kill power.",
        tags={"safety", "medic"},
        controlled_by="brainstem-01",
    )
    topology.add_node(watcher)

    # === Archivist (NAS) ===
    archivist = FleetNode(
        node_id="archivist-01",
        hostname="nas",
        role=FleetRole.ARCHIVIST,
        capabilities={
            NodeCapability.FAST_STORAGE,
            NodeCapability.UPS_BACKED,
        },
        auth_level=AuthLevel.READ_ONLY,  # Writes via pipelines only
        on_ups=True,
        description="Storage with snapshots. Memory that survives.",
        tags={"storage", "backup"},
        controlled_by="brainstem-01",
    )
    topology.add_node(archivist)

    # === Shield (Network Observer) ===
    shield = FleetNode(
        node_id="shield-01",
        hostname="shield",
        role=FleetRole.SHIELD,
        capabilities={
            NodeCapability.SPAN_PORT,
            NodeCapability.DUAL_NIC,
        },
        auth_level=AuthLevel.READ_ONLY,  # Never pushes config
        description="Network tap. Observes Juniper without control.",
        tags={"network", "observer"},
        controlled_by="brainstem-01",
    )
    topology.add_node(shield)

    # === GPU Worker ===
    gpu_worker = FleetNode(
        node_id="gpu-worker-01",
        hostname="threadripper",
        role=FleetRole.WORKER,
        capabilities={
            NodeCapability.GPU,
            NodeCapability.HIGH_RAM,
            NodeCapability.FAST_STORAGE,
        },
        auth_level=AuthLevel.EXECUTE,
        power_outlet="pdu-01:outlet-1",
        description="Main GPU compute node. Runs LLM workloads.",
        tags={"compute", "gpu", "llm"},
        controlled_by="brainstem-01",
    )
    topology.add_node(gpu_worker)

    # === FPGA Worker (HSF Card) ===
    fpga_worker = FleetNode(
        node_id="fpga-worker-01",
        hostname="hsf-card",
        role=FleetRole.WORKER,
        capabilities={
            NodeCapability.FPGA,
        },
        auth_level=AuthLevel.EXECUTE,
        power_outlet="pdu-01:outlet-2",
        description="FPGA neuromorphic card. SNN/HDC subcortex.",
        tags={"compute", "fpga", "neuromorphic"},
        controlled_by="brainstem-01",
    )
    topology.add_node(fpga_worker)

    # === Intern Cluster (VM Host) ===
    intern_host = FleetNode(
        node_id="intern-host-01",
        hostname="vm-host",
        role=FleetRole.INTERN,
        capabilities={
            NodeCapability.HIGH_RAM,
        },
        auth_level=AuthLevel.MODIFY,  # Can be messed with
        power_outlet="pdu-01:outlet-3",
        description="Sacrificial VM host. Wild experiments go here.",
        tags={"experimental", "vm"},
        controlled_by="brainstem-01",
    )
    topology.add_node(intern_host)

    # === Juniper (Consultant) ===
    juniper = FleetNode(
        node_id="juniper-01",
        hostname="juniper",
        role=FleetRole.CONSULTANT,
        capabilities={
            NodeCapability.UPS_BACKED,
        },
        auth_level=AuthLevel.READ_ONLY,  # Config changes require signature
        on_ups=True,
        description="Core router. Config changes gated by signatures.",
        tags={"network", "critical", "infrastructure"},
        controlled_by="brainstem-01",
    )
    topology.add_node(juniper)

    # === Sensor Hub ===
    sensor_hub = FleetNode(
        node_id="sensor-hub-01",
        hostname="sensors",
        role=FleetRole.SENSOR_HUB,
        capabilities={
            NodeCapability.TEMP_SENSOR,
            NodeCapability.ENV_SENSOR,
        },
        auth_level=AuthLevel.READ_ONLY,
        description="Environmental sensors. Room temp, humidity, etc.",
        tags={"sensors", "environment"},
        controlled_by="brainstem-01",
    )
    topology.add_node(sensor_hub)

    # Set up control relationships
    brainstem.controls = [
        "watcher-01", "archivist-01", "shield-01",
        "gpu-worker-01", "fpga-worker-01", "intern-host-01",
        "juniper-01", "sensor-hub-01",
    ]

    # Watcher controls power to compute nodes
    watcher.controls = ["gpu-worker-01", "fpga-worker-01", "intern-host-01"]

    return topology
