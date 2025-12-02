"""
L4 Cognitive Memory: CXL-Backed Knowledge Graph

This module implements the structured knowledge storage layer that enables
the system to maintain a persistent world model with:
- Lab-state knowledge graph (boards, GPUs, FPGAs, workloads, experiments)
- Belief updates as first-class operations with uncertainty tracking
- Temporal versioning for reasoning about state changes
- Integration with L6 reasoning orchestrator for intelligent retrieval

The KG provides the "world it believes in" - a structured, queryable
representation of everything the system knows about its environment.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict
import json
import hashlib


class NodeType(str, Enum):
    """Types of nodes in the knowledge graph."""
    # Hardware
    BOARD = "board"
    GPU = "gpu"
    FPGA = "fpga"
    CPU = "cpu"
    MEMORY = "memory"

    # Software/Config
    WORKLOAD = "workload"
    CONFIG = "config"
    PROFILE = "profile"
    KERNEL = "kernel"

    # Events/Episodes
    EXPERIMENT = "experiment"
    FAILURE = "failure"
    SUCCESS = "success"
    EPISODE = "episode"

    # Metrics
    METRIC = "metric"
    OBSERVATION = "observation"

    # Abstract concepts
    CONCEPT = "concept"
    AXIOM = "axiom"
    TRUTH = "truth"


class EdgeType(str, Enum):
    """Types of relationships between nodes."""
    # Hardware relationships
    RAN_ON = "ran_on"
    REQUIRES = "requires"
    CONNECTED_TO = "connected_to"
    PART_OF = "part_of"

    # Causal relationships
    CAUSED = "caused"
    IMPROVED = "improved"
    DEGRADED = "degraded"
    FAILED_WITH = "failed_with"

    # Temporal relationships
    PRECEDED = "preceded"
    FOLLOWED = "followed"
    CONCURRENT_WITH = "concurrent_with"

    # Logical relationships
    IMPLIES = "implies"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"
    DERIVED_FROM = "derived_from"

    # Membership
    INSTANCE_OF = "instance_of"
    HAS_PROPERTY = "has_property"
    CONFIGURED_WITH = "configured_with"


class HealthStatus(str, Enum):
    """Health status for hardware/system nodes."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass
class Belief:
    """
    A belief represents a fact with associated uncertainty and provenance.

    This allows the system to reason about "what it thinks is true"
    with appropriate confidence levels.
    """
    statement: str
    confidence: float  # 0.0 to 1.0
    source: str  # Where this belief came from
    timestamp: datetime
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)
    last_verified: Optional[datetime] = None

    def update_confidence(self, delta: float, reason: str) -> None:
        """Update confidence with bounded adjustment."""
        old_conf = self.confidence
        self.confidence = max(0.0, min(1.0, self.confidence + delta))
        if delta > 0:
            self.supporting_evidence.append(f"{reason} (+{delta:.2f})")
        else:
            self.contradicting_evidence.append(f"{reason} ({delta:.2f})")
        self.last_verified = datetime.now()

    def is_uncertain(self, threshold: float = 0.7) -> bool:
        """Check if belief is below confidence threshold."""
        return self.confidence < threshold


@dataclass
class KGNode:
    """A node in the knowledge graph with properties and beliefs."""
    id: str
    node_type: NodeType
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    beliefs: List[Belief] = field(default_factory=list)
    health: HealthStatus = HealthStatus.UNKNOWN
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: int = 1

    def add_belief(self, statement: str, confidence: float, source: str) -> Belief:
        """Add a new belief to this node."""
        belief = Belief(
            statement=statement,
            confidence=confidence,
            source=source,
            timestamp=datetime.now()
        )
        self.beliefs.append(belief)
        self.updated_at = datetime.now()
        self.version += 1
        return belief

    def get_belief(self, statement: str) -> Optional[Belief]:
        """Find a belief by statement."""
        for b in self.beliefs:
            if b.statement == statement:
                return b
        return None

    def update_health(self, new_health: HealthStatus, reason: str) -> None:
        """Update node health with reason tracking."""
        self.health = new_health
        self.add_belief(
            f"Health status: {new_health.value}",
            confidence=0.9,
            source=reason
        )
        self.updated_at = datetime.now()


@dataclass
class KGEdge:
    """An edge connecting two nodes with typed relationship."""
    source_id: str
    target_id: str
    edge_type: EdgeType
    properties: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0  # Strength of relationship
    confidence: float = 1.0  # How certain we are this edge exists
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def id(self) -> str:
        """Unique edge identifier."""
        return f"{self.source_id}--{self.edge_type.value}-->{self.target_id}"


class CognitiveKnowledgeGraph:
    """
    The core knowledge graph for cognitive memory.

    This provides the "world model" that the system reasons over,
    with support for:
    - Structured fact storage with uncertainty
    - Temporal versioning
    - Belief propagation
    - Intelligent retrieval based on L3 risk state
    """

    def __init__(self, persist_path: Optional[str] = None):
        self._nodes: Dict[str, KGNode] = {}
        self._edges: Dict[str, KGEdge] = {}
        self._node_index: Dict[NodeType, Set[str]] = defaultdict(set)
        self._edge_index: Dict[str, Set[str]] = defaultdict(set)  # node_id -> edge_ids
        self._persist_path = persist_path
        self._version = 0

        # Track verified truths (cached from PGU)
        self._verified_truths: Dict[str, Tuple[bool, float]] = {}  # statement -> (is_true, confidence)

    # ========== Node Operations ==========

    def add_node(
        self,
        node_type: NodeType,
        name: str,
        properties: Optional[Dict[str, Any]] = None,
        node_id: Optional[str] = None
    ) -> KGNode:
        """Add a node to the knowledge graph."""
        if node_id is None:
            node_id = self._generate_id(node_type.value, name)

        node = KGNode(
            id=node_id,
            node_type=node_type,
            name=name,
            properties=properties or {}
        )

        self._nodes[node_id] = node
        self._node_index[node_type].add(node_id)
        self._version += 1

        return node

    def get_node(self, node_id: str) -> Optional[KGNode]:
        """Retrieve a node by ID."""
        return self._nodes.get(node_id)

    def find_nodes(
        self,
        node_type: Optional[NodeType] = None,
        name_contains: Optional[str] = None,
        health: Optional[HealthStatus] = None,
        min_confidence: float = 0.0
    ) -> List[KGNode]:
        """Find nodes matching criteria."""
        candidates = set()

        if node_type:
            candidates = self._node_index.get(node_type, set())
        else:
            candidates = set(self._nodes.keys())

        results = []
        for node_id in candidates:
            node = self._nodes[node_id]

            if name_contains and name_contains.lower() not in node.name.lower():
                continue
            if health and node.health != health:
                continue

            # Check belief confidence
            if min_confidence > 0:
                max_belief_conf = max((b.confidence for b in node.beliefs), default=1.0)
                if max_belief_conf < min_confidence:
                    continue

            results.append(node)

        return results

    def update_node(self, node_id: str, properties: Dict[str, Any]) -> Optional[KGNode]:
        """Update node properties."""
        node = self._nodes.get(node_id)
        if node:
            node.properties.update(properties)
            node.updated_at = datetime.now()
            node.version += 1
            self._version += 1
        return node

    # ========== Edge Operations ==========

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        properties: Optional[Dict[str, Any]] = None,
        weight: float = 1.0,
        confidence: float = 1.0
    ) -> Optional[KGEdge]:
        """Add an edge between two nodes."""
        if source_id not in self._nodes or target_id not in self._nodes:
            return None

        edge = KGEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            properties=properties or {},
            weight=weight,
            confidence=confidence
        )

        self._edges[edge.id] = edge
        self._edge_index[source_id].add(edge.id)
        self._edge_index[target_id].add(edge.id)
        self._version += 1

        return edge

    def get_edges(
        self,
        node_id: str,
        edge_type: Optional[EdgeType] = None,
        direction: str = "both"  # "outgoing", "incoming", "both"
    ) -> List[KGEdge]:
        """Get edges connected to a node."""
        edge_ids = self._edge_index.get(node_id, set())
        results = []

        for edge_id in edge_ids:
            edge = self._edges.get(edge_id)
            if not edge:
                continue

            if edge_type and edge.edge_type != edge_type:
                continue

            if direction == "outgoing" and edge.source_id != node_id:
                continue
            if direction == "incoming" and edge.target_id != node_id:
                continue

            results.append(edge)

        return results

    def get_neighbors(
        self,
        node_id: str,
        edge_type: Optional[EdgeType] = None,
        direction: str = "both"
    ) -> List[KGNode]:
        """Get neighboring nodes."""
        edges = self.get_edges(node_id, edge_type, direction)
        neighbor_ids = set()

        for edge in edges:
            if edge.source_id == node_id:
                neighbor_ids.add(edge.target_id)
            else:
                neighbor_ids.add(edge.source_id)

        return [self._nodes[nid] for nid in neighbor_ids if nid in self._nodes]

    # ========== Belief Operations ==========

    def add_belief(
        self,
        node_id: str,
        statement: str,
        confidence: float,
        source: str
    ) -> Optional[Belief]:
        """Add a belief to a node."""
        node = self._nodes.get(node_id)
        if node:
            return node.add_belief(statement, confidence, source)
        return None

    def update_belief(
        self,
        node_id: str,
        statement: str,
        delta: float,
        reason: str
    ) -> Optional[Belief]:
        """Update confidence in an existing belief."""
        node = self._nodes.get(node_id)
        if node:
            belief = node.get_belief(statement)
            if belief:
                belief.update_confidence(delta, reason)
                return belief
        return None

    def get_uncertain_beliefs(self, threshold: float = 0.7) -> List[Tuple[KGNode, Belief]]:
        """Find all beliefs below confidence threshold."""
        uncertain = []
        for node in self._nodes.values():
            for belief in node.beliefs:
                if belief.is_uncertain(threshold):
                    uncertain.append((node, belief))
        return uncertain

    # ========== Verified Truth Cache (PGU Integration) ==========

    def cache_verified_truth(
        self,
        statement: str,
        is_true: bool,
        confidence: float = 1.0
    ) -> None:
        """Cache a truth verified by PGU."""
        self._verified_truths[statement] = (is_true, confidence)

    def check_verified_truth(self, statement: str) -> Optional[Tuple[bool, float]]:
        """Check if a statement has been verified."""
        return self._verified_truths.get(statement)

    def get_all_verified_truths(self) -> Dict[str, Tuple[bool, float]]:
        """Get all cached verified truths."""
        return dict(self._verified_truths)

    # ========== Query Operations ==========

    def query(
        self,
        query_type: str,
        **kwargs
    ) -> List[Any]:
        """
        Execute a structured query on the knowledge graph.

        Supported query types:
        - "path": Find path between two nodes
        - "subgraph": Extract subgraph around a node
        - "pattern": Match a pattern of nodes and edges
        - "similar": Find nodes similar to a given node
        """
        if query_type == "path":
            return self._query_path(kwargs.get("source"), kwargs.get("target"), kwargs.get("max_depth", 5))
        elif query_type == "subgraph":
            return self._query_subgraph(kwargs.get("center"), kwargs.get("radius", 2))
        elif query_type == "pattern":
            return self._query_pattern(kwargs.get("pattern"))
        else:
            return []

    def _query_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int
    ) -> List[List[str]]:
        """BFS to find paths between nodes."""
        if source_id not in self._nodes or target_id not in self._nodes:
            return []

        queue = [(source_id, [source_id])]
        visited = {source_id}
        paths = []

        while queue:
            current, path = queue.pop(0)

            if len(path) > max_depth:
                continue

            if current == target_id:
                paths.append(path)
                continue

            for neighbor in self.get_neighbors(current):
                if neighbor.id not in visited:
                    visited.add(neighbor.id)
                    queue.append((neighbor.id, path + [neighbor.id]))

        return paths

    def _query_subgraph(
        self,
        center_id: str,
        radius: int
    ) -> Tuple[List[KGNode], List[KGEdge]]:
        """Extract subgraph within radius of center."""
        if center_id not in self._nodes:
            return [], []

        visited_nodes = {center_id}
        current_layer = {center_id}

        for _ in range(radius):
            next_layer = set()
            for node_id in current_layer:
                for neighbor in self.get_neighbors(node_id):
                    if neighbor.id not in visited_nodes:
                        visited_nodes.add(neighbor.id)
                        next_layer.add(neighbor.id)
            current_layer = next_layer

        nodes = [self._nodes[nid] for nid in visited_nodes]

        # Get edges within subgraph
        edges = []
        for edge in self._edges.values():
            if edge.source_id in visited_nodes and edge.target_id in visited_nodes:
                edges.append(edge)

        return nodes, edges

    def _query_pattern(self, pattern: Dict[str, Any]) -> List[Dict[str, KGNode]]:
        """Match a pattern of node types and relationships."""
        # Simple pattern matching: {"node1": NodeType.X, "node2": NodeType.Y, "edge": EdgeType.Z}
        # Returns bindings that satisfy the pattern
        results = []

        node1_type = pattern.get("node1_type")
        node2_type = pattern.get("node2_type")
        edge_type = pattern.get("edge_type")

        if not all([node1_type, node2_type, edge_type]):
            return results

        for node1 in self.find_nodes(node_type=node1_type):
            for edge in self.get_edges(node1.id, edge_type=edge_type, direction="outgoing"):
                node2 = self._nodes.get(edge.target_id)
                if node2 and node2.node_type == node2_type:
                    results.append({"node1": node1, "node2": node2, "edge": edge})

        return results

    # ========== Lab State Helpers ==========

    def record_experiment(
        self,
        name: str,
        config: Dict[str, Any],
        results: Dict[str, Any],
        hardware_used: List[str]
    ) -> KGNode:
        """Record an experiment as a node with relationships."""
        exp_node = self.add_node(
            NodeType.EXPERIMENT,
            name,
            properties={
                "config": config,
                "results": results,
                "timestamp": datetime.now().isoformat()
            }
        )

        # Link to hardware
        for hw_id in hardware_used:
            if hw_id in self._nodes:
                self.add_edge(exp_node.id, hw_id, EdgeType.RAN_ON)

        # Add beliefs about results
        if "af_score" in results:
            exp_node.add_belief(
                f"Achieved AF score {results['af_score']:.2f}",
                confidence=0.95,
                source="certification"
            )

        return exp_node

    def record_failure(
        self,
        hardware_id: str,
        failure_type: str,
        details: Dict[str, Any]
    ) -> Optional[KGNode]:
        """Record a hardware failure event."""
        hw_node = self._nodes.get(hardware_id)
        if not hw_node:
            return None

        failure_node = self.add_node(
            NodeType.FAILURE,
            f"Failure on {hw_node.name}: {failure_type}",
            properties=details
        )

        self.add_edge(failure_node.id, hardware_id, EdgeType.CAUSED)

        # Update hardware health
        hw_node.update_health(HealthStatus.DEGRADED, f"Failure: {failure_type}")

        return failure_node

    def get_hardware_health_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get health summary of all hardware nodes."""
        summary = {}

        for node_type in [NodeType.BOARD, NodeType.GPU, NodeType.FPGA, NodeType.CPU]:
            for node in self.find_nodes(node_type=node_type):
                failures = self.get_edges(node.id, EdgeType.CAUSED, direction="incoming")
                experiments = self.get_edges(node.id, EdgeType.RAN_ON, direction="incoming")

                summary[node.id] = {
                    "name": node.name,
                    "type": node_type.value,
                    "health": node.health.value,
                    "failure_count": len(failures),
                    "experiment_count": len(experiments),
                    "last_updated": node.updated_at.isoformat()
                }

        return summary

    # ========== Persistence ==========

    def save(self, path: Optional[str] = None) -> None:
        """Save knowledge graph to JSON."""
        save_path = path or self._persist_path
        if not save_path:
            return

        data = {
            "version": self._version,
            "nodes": {
                nid: {
                    "id": n.id,
                    "node_type": n.node_type.value,
                    "name": n.name,
                    "properties": n.properties,
                    "health": n.health.value,
                    "beliefs": [
                        {
                            "statement": b.statement,
                            "confidence": b.confidence,
                            "source": b.source,
                            "timestamp": b.timestamp.isoformat()
                        }
                        for b in n.beliefs
                    ],
                    "created_at": n.created_at.isoformat(),
                    "updated_at": n.updated_at.isoformat()
                }
                for nid, n in self._nodes.items()
            },
            "edges": {
                eid: {
                    "source_id": e.source_id,
                    "target_id": e.target_id,
                    "edge_type": e.edge_type.value,
                    "properties": e.properties,
                    "weight": e.weight,
                    "confidence": e.confidence
                }
                for eid, e in self._edges.items()
            },
            "verified_truths": self._verified_truths
        }

        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _generate_id(self, prefix: str, name: str) -> str:
        """Generate unique node ID."""
        hash_input = f"{prefix}:{name}:{datetime.now().isoformat()}"
        return f"{prefix}_{hashlib.md5(hash_input.encode()).hexdigest()[:8]}"

    # ========== Statistics ==========

    @property
    def stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return {
            "node_count": len(self._nodes),
            "edge_count": len(self._edges),
            "verified_truths": len(self._verified_truths),
            "version": self._version,
            "nodes_by_type": {
                nt.value: len(self._node_index.get(nt, set()))
                for nt in NodeType
            }
        }


# ============================================================
# Convenience functions for creating and using the KG
# ============================================================

def create_lab_kg() -> CognitiveKnowledgeGraph:
    """Create a knowledge graph pre-populated with common lab entities."""
    kg = CognitiveKnowledgeGraph()

    # Add common hardware nodes
    kg.add_node(NodeType.FPGA, "Alveo U250", {"target": "xcu250-figd2104-2L-e", "clock": "250MHz"})
    kg.add_node(NodeType.GPU, "A10 GPU", {"memory": "24GB", "compute": "FP32"})
    kg.add_node(NodeType.BOARD, "K10 Board", {"status": "active"})

    # Add common profiles
    for profile in ["cautious_stable", "reactive_adaptive", "balanced_general", "exploratory_creative"]:
        kg.add_node(NodeType.PROFILE, profile, {"source": "L5_meta_learning"})

    return kg


def query_for_task(
    kg: CognitiveKnowledgeGraph,
    task_type: str,
    risk_level: str
) -> Dict[str, Any]:
    """
    Query KG with L3-aware retrieval strategy.

    High risk: Conservative, structured retrieval
    Low risk: Fast, exploratory retrieval
    """
    if risk_level == "high":
        # Structured, fact-checked retrieval
        # Look for verified truths and high-confidence beliefs
        relevant_nodes = []
        for node in kg._nodes.values():
            high_conf_beliefs = [b for b in node.beliefs if b.confidence > 0.8]
            if high_conf_beliefs:
                relevant_nodes.append({
                    "node": node.name,
                    "type": node.node_type.value,
                    "beliefs": [b.statement for b in high_conf_beliefs]
                })
        return {"mode": "conservative", "results": relevant_nodes[:10]}
    else:
        # Fast, exploratory retrieval
        # Broader search, lower confidence threshold
        relevant_nodes = kg.find_nodes(min_confidence=0.5)
        return {
            "mode": "exploratory",
            "results": [{"node": n.name, "type": n.node_type.value} for n in relevant_nodes[:20]]
        }
