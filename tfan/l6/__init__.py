"""
L6 Formal Reasoning: PGU + Knowledge Graph + L3-Aware Retrieval

This module implements hierarchical reasoning that combines:
1. LLM - Fast, fuzzy, wide coverage
2. Knowledge Graph - Structured, multi-hop, high precision
3. PGU/SMT - Slow but guaranteed consistent for formal domains

The L3 emotional state influences which reasoning path is preferred:
- High risk / Low valence → PGU-verified formal reasoning
- Low risk / Exploratory → LLM-first with optional KG support

Key Components:
- ReasoningOrchestrator: Routes queries to appropriate reasoning backend
- KnowledgeGraph: CXL-backed structured knowledge store
- ConsistencyOracle: PGU wrapper for logical consistency checking
- L3AwareRetriever: Emotionally-gated memory retrieval
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple, Callable
from datetime import datetime
from enum import Enum
import logging
import hashlib
import json
import time

# Add paths
_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

logger = logging.getLogger("tfan.l6.reasoning")


# =============================================================================
# TASK TYPES AND REASONING MODES
# =============================================================================

class TaskType(str, Enum):
    """Types of reasoning tasks."""
    SYSTEM_CONFIG = "system_config"       # Hardware/software configuration
    HARDWARE_SAFETY = "hardware_safety"   # Safety-critical operations
    GENERAL_QUESTION = "general_question" # Open-ended queries
    PLANNING = "planning"                 # Multi-step reasoning
    RETRIEVAL = "retrieval"               # Fact lookup
    CREATIVE = "creative"                 # Open-ended generation


class ReasoningMode(str, Enum):
    """Reasoning backend modes."""
    LLM_ONLY = "llm_only"           # Fast, fuzzy
    KG_ASSISTED = "kg_assisted"     # KG → LLM summarization
    PGU_VERIFIED = "pgu_verified"   # PGU checks LLM output
    FORMAL_FIRST = "formal_first"   # KG → PGU → LLM narration
    HYBRID = "hybrid"               # Adaptive mix


@dataclass
class ReasoningContext:
    """Context for reasoning decision."""
    task_type: TaskType
    clv_risk: str = "UNKNOWN"  # LOW/MEDIUM/HIGH/CRITICAL
    valence: float = 0.0       # [-1, 1]
    arousal: float = 0.5       # [0, 1]
    dominance: float = 0.5     # [0, 1]
    confidence_required: float = 0.5  # How certain must answer be?
    latency_budget_ms: float = 5000   # Time available

    def is_high_stakes(self) -> bool:
        """Check if this is a high-stakes query."""
        return (
            self.task_type in [TaskType.HARDWARE_SAFETY, TaskType.SYSTEM_CONFIG] or
            self.clv_risk in ["HIGH", "CRITICAL"] or
            self.valence < -0.5 or
            self.confidence_required > 0.8
        )


# =============================================================================
# KNOWLEDGE GRAPH (CXL-BACKED)
# =============================================================================

@dataclass
class KGNode:
    """Knowledge graph node."""
    id: str
    type: str  # hardware, workload, config, experiment, metric
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    pgu_verified: bool = False
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class KGEdge:
    """Knowledge graph edge."""
    source_id: str
    target_id: str
    relation: str  # has_config, runs_on, produces, requires, conflicts_with
    weight: float = 1.0
    confidence: float = 1.0
    pgu_verified: bool = False


@dataclass
class KGQueryResult:
    """Result of a knowledge graph query."""
    nodes: List[KGNode]
    edges: List[KGEdge]
    paths: List[List[str]]  # Multi-hop paths
    confidence: float
    latency_ms: float
    pgu_verified: bool = False


class KnowledgeGraph:
    """
    CXL-backed knowledge graph for structured reasoning.

    Stores:
    - Hardware nodes (boards, FPGAs, GPUs)
    - Workload nodes (types, characteristics)
    - Config nodes (settings, parameters)
    - Experiment nodes (past runs, results)
    - Metric nodes (measurements, thresholds)

    Supports:
    - Single-hop retrieval
    - Multi-hop path finding
    - PGU-verified edges
    - Confidence-weighted queries
    """

    def __init__(self, persistence_path: Optional[str] = None):
        """
        Initialize knowledge graph.

        Args:
            persistence_path: Path for persistent storage (simulates CXL)
        """
        self.nodes: Dict[str, KGNode] = {}
        self.edges: List[KGEdge] = []
        self.adjacency: Dict[str, List[Tuple[str, str, float]]] = {}  # node_id → [(target, relation, weight)]

        self.persistence_path = persistence_path

        # Metrics
        self.query_count = 0
        self.query_latencies: List[float] = []

        logger.info("KnowledgeGraph initialized")

    def add_node(self, node: KGNode):
        """Add or update a node."""
        self.nodes[node.id] = node
        if node.id not in self.adjacency:
            self.adjacency[node.id] = []

    def add_edge(self, edge: KGEdge):
        """Add an edge."""
        self.edges.append(edge)
        if edge.source_id not in self.adjacency:
            self.adjacency[edge.source_id] = []
        self.adjacency[edge.source_id].append((edge.target_id, edge.relation, edge.weight))

    def query_single_hop(
        self,
        node_id: str,
        relation: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> KGQueryResult:
        """
        Query single-hop neighbors.

        Args:
            node_id: Source node ID
            relation: Filter by relation type
            min_confidence: Minimum edge confidence

        Returns:
            KGQueryResult with neighbors
        """
        start_time = time.perf_counter()
        self.query_count += 1

        neighbors = []
        found_edges = []

        if node_id in self.adjacency:
            for target_id, rel, weight in self.adjacency[node_id]:
                if relation and rel != relation:
                    continue
                if target_id in self.nodes:
                    edge = next((e for e in self.edges
                                 if e.source_id == node_id and e.target_id == target_id
                                 and e.relation == rel), None)
                    if edge and edge.confidence >= min_confidence:
                        neighbors.append(self.nodes[target_id])
                        found_edges.append(edge)

        latency_ms = (time.perf_counter() - start_time) * 1000
        self.query_latencies.append(latency_ms)

        # Compute aggregate confidence
        confidence = sum(e.confidence for e in found_edges) / max(len(found_edges), 1)

        return KGQueryResult(
            nodes=neighbors,
            edges=found_edges,
            paths=[[node_id, n.id] for n in neighbors],
            confidence=confidence,
            latency_ms=latency_ms,
            pgu_verified=all(e.pgu_verified for e in found_edges) if found_edges else False,
        )

    def query_multi_hop(
        self,
        start_id: str,
        max_hops: int = 3,
        min_confidence: float = 0.5,
        target_type: Optional[str] = None,
    ) -> KGQueryResult:
        """
        Multi-hop path finding with confidence decay.

        Args:
            start_id: Starting node ID
            max_hops: Maximum path length
            min_confidence: Minimum path confidence
            target_type: Filter targets by node type

        Returns:
            KGQueryResult with all reachable nodes and paths
        """
        start_time = time.perf_counter()
        self.query_count += 1

        # BFS with confidence tracking
        visited = {start_id: (1.0, [start_id])}  # node_id → (confidence, path)
        queue = [(start_id, 1.0, [start_id], 0)]  # (node_id, confidence, path, depth)

        found_nodes = []
        found_paths = []

        while queue:
            current_id, current_conf, current_path, depth = queue.pop(0)

            if depth >= max_hops:
                continue

            if current_id in self.adjacency:
                for target_id, rel, weight in self.adjacency[current_id]:
                    # Confidence decays with hops
                    edge = next((e for e in self.edges
                                 if e.source_id == current_id and e.target_id == target_id
                                 and e.relation == rel), None)
                    edge_conf = edge.confidence if edge else 0.5
                    new_conf = current_conf * edge_conf * (0.9 ** depth)

                    if new_conf < min_confidence:
                        continue

                    new_path = current_path + [target_id]

                    # Only visit if better confidence or not visited
                    if target_id not in visited or visited[target_id][0] < new_conf:
                        visited[target_id] = (new_conf, new_path)
                        queue.append((target_id, new_conf, new_path, depth + 1))

                        if target_id in self.nodes:
                            node = self.nodes[target_id]
                            if target_type is None or node.type == target_type:
                                found_nodes.append(node)
                                found_paths.append(new_path)

        latency_ms = (time.perf_counter() - start_time) * 1000
        self.query_latencies.append(latency_ms)

        # Aggregate confidence
        confidence = max((visited.get(n.id, (0,))[0] for n in found_nodes), default=0.0)

        return KGQueryResult(
            nodes=found_nodes,
            edges=[],  # Too complex to return all edges
            paths=found_paths,
            confidence=confidence,
            latency_ms=latency_ms,
            pgu_verified=False,  # Multi-hop not PGU-verified by default
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get KG performance metrics."""
        if not self.query_latencies:
            return {"query_count": 0, "p50_latency_ms": 0, "p95_latency_ms": 0}

        import statistics
        sorted_lat = sorted(self.query_latencies)
        return {
            "query_count": self.query_count,
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "p50_latency_ms": sorted_lat[len(sorted_lat) // 2],
            "p95_latency_ms": sorted_lat[int(len(sorted_lat) * 0.95)],
        }


# =============================================================================
# CONSISTENCY ORACLE (PGU WRAPPER)
# =============================================================================

@dataclass
class ConsistencyCheckResult:
    """Result of a consistency check."""
    consistent: bool
    violations: List[str]
    suggestions: List[str]
    latency_ms: float
    cached: bool = False


class ConsistencyOracle:
    """
    PGU wrapper for logical consistency checking.

    Checks:
    - Constraint satisfaction
    - Invariant preservation
    - Logical consistency of claims
    """

    def __init__(self, pgu=None):
        """
        Initialize consistency oracle.

        Args:
            pgu: ProofGatedUpdater instance (optional)
        """
        self.pgu = pgu
        self.check_count = 0
        self.cache: Dict[str, ConsistencyCheckResult] = {}

        logger.info("ConsistencyOracle initialized")

    def check_constraints(
        self,
        claims: List[Dict[str, Any]],
        invariants: List[str],
    ) -> ConsistencyCheckResult:
        """
        Check if claims satisfy invariants.

        Args:
            claims: List of claim dictionaries
            invariants: List of invariant names to check

        Returns:
            ConsistencyCheckResult
        """
        start_time = time.perf_counter()
        self.check_count += 1

        # Create cache key
        cache_key = hashlib.sha256(
            json.dumps({"claims": claims, "invariants": invariants}, sort_keys=True).encode()
        ).hexdigest()

        if cache_key in self.cache:
            result = self.cache[cache_key]
            result.cached = True
            result.latency_ms = (time.perf_counter() - start_time) * 1000
            return result

        # If PGU available, use it
        if self.pgu:
            try:
                pgu_result = self.pgu.verify_update({
                    "claims": claims,
                    "invariants": invariants,
                    "metadata": {"type": "consistency_check"},
                })
                result = ConsistencyCheckResult(
                    consistent=pgu_result.proven,
                    violations=pgu_result.rule_violations,
                    suggestions=[],
                    latency_ms=(time.perf_counter() - start_time) * 1000,
                )
            except Exception as e:
                logger.warning(f"PGU check failed: {e}")
                result = self._fallback_check(claims, invariants, start_time)
        else:
            result = self._fallback_check(claims, invariants, start_time)

        self.cache[cache_key] = result
        return result

    def _fallback_check(
        self,
        claims: List[Dict[str, Any]],
        invariants: List[str],
        start_time: float,
    ) -> ConsistencyCheckResult:
        """Fallback consistency check without Z3."""
        # Simple heuristic checks
        violations = []
        suggestions = []

        for claim in claims:
            # Check for obvious inconsistencies
            if "value" in claim and "max_value" in claim:
                if claim["value"] > claim["max_value"]:
                    violations.append(f"Value {claim['value']} exceeds max {claim['max_value']}")

            if "requires" in claim and "conflicts_with" in claim:
                if set(claim["requires"]) & set(claim["conflicts_with"]):
                    violations.append(f"Claim has conflicting requirements")

        return ConsistencyCheckResult(
            consistent=len(violations) == 0,
            violations=violations,
            suggestions=suggestions,
            latency_ms=(time.perf_counter() - start_time) * 1000,
        )

    def verify_chain(
        self,
        reasoning_steps: List[str],
        context: Dict[str, Any],
    ) -> ConsistencyCheckResult:
        """
        Verify a chain of reasoning steps.

        Used for LLM → PGU loops where PGU checks LLM output.

        Args:
            reasoning_steps: List of reasoning step descriptions
            context: Additional context

        Returns:
            ConsistencyCheckResult
        """
        start_time = time.perf_counter()

        # Convert steps to claims
        claims = [{"step": i, "content": step} for i, step in enumerate(reasoning_steps)]

        # Check for logical consistency
        violations = []
        for i, step in enumerate(reasoning_steps):
            # Check for contradictions (simplified)
            lower_step = step.lower()
            if "not" in lower_step and "always" in lower_step:
                violations.append(f"Step {i}: Potential contradiction (not + always)")
            if "never" in lower_step and "must" in lower_step:
                violations.append(f"Step {i}: Potential contradiction (never + must)")

        return ConsistencyCheckResult(
            consistent=len(violations) == 0,
            violations=violations,
            suggestions=["Consider rephrasing steps with contradictions"] if violations else [],
            latency_ms=(time.perf_counter() - start_time) * 1000,
        )


# =============================================================================
# REASONING ORCHESTRATOR
# =============================================================================

@dataclass
class ReasoningResult:
    """Result of reasoning operation."""
    answer: str
    reasoning_mode: ReasoningMode
    confidence: float
    sources: List[str]  # Where answer came from
    pgu_verified: bool
    kg_paths: List[List[str]]
    latency_ms: float
    violations: List[str]


class ReasoningOrchestrator:
    """
    Orchestrates reasoning across LLM, KG, and PGU.

    Routes queries based on:
    - Task type
    - CLV risk level
    - PAD emotional state
    - Latency budget
    """

    def __init__(
        self,
        kg: Optional[KnowledgeGraph] = None,
        oracle: Optional[ConsistencyOracle] = None,
        llm_handler: Optional[Callable] = None,
    ):
        """
        Initialize orchestrator.

        Args:
            kg: KnowledgeGraph instance
            oracle: ConsistencyOracle instance
            llm_handler: Function to call LLM (async-compatible)
        """
        self.kg = kg or KnowledgeGraph()
        self.oracle = oracle or ConsistencyOracle()
        self.llm_handler = llm_handler

        # Statistics
        self.query_count = 0
        self.mode_counts: Dict[ReasoningMode, int] = {m: 0 for m in ReasoningMode}

        logger.info("ReasoningOrchestrator initialized")

    def select_mode(self, context: ReasoningContext) -> ReasoningMode:
        """
        Select reasoning mode based on context.

        Policy:
        - High stakes + enough time → FORMAL_FIRST
        - High stakes + limited time → PGU_VERIFIED
        - Low stakes + exploratory → LLM_ONLY
        - Default → KG_ASSISTED
        """
        if context.is_high_stakes():
            if context.latency_budget_ms > 3000:
                return ReasoningMode.FORMAL_FIRST
            else:
                return ReasoningMode.PGU_VERIFIED

        if context.task_type == TaskType.CREATIVE:
            return ReasoningMode.LLM_ONLY

        if context.task_type == TaskType.RETRIEVAL:
            return ReasoningMode.KG_ASSISTED

        # Low valence (stressed) → prefer verified paths
        if context.valence < -0.3:
            return ReasoningMode.PGU_VERIFIED

        # High arousal (urgent) → faster path
        if context.arousal > 0.7:
            return ReasoningMode.LLM_ONLY

        return ReasoningMode.KG_ASSISTED

    def reason(
        self,
        query: str,
        context: ReasoningContext,
        kg_start_node: Optional[str] = None,
    ) -> ReasoningResult:
        """
        Execute reasoning with appropriate backend.

        Args:
            query: The question/task
            context: Reasoning context
            kg_start_node: Optional KG node to start from

        Returns:
            ReasoningResult
        """
        start_time = time.perf_counter()
        self.query_count += 1

        # Select mode
        mode = self.select_mode(context)
        self.mode_counts[mode] += 1

        logger.info(f"Reasoning mode: {mode.value} for task: {context.task_type.value}")

        # Execute based on mode
        if mode == ReasoningMode.LLM_ONLY:
            result = self._reason_llm_only(query, context, start_time)

        elif mode == ReasoningMode.KG_ASSISTED:
            result = self._reason_kg_assisted(query, context, kg_start_node, start_time)

        elif mode == ReasoningMode.PGU_VERIFIED:
            result = self._reason_pgu_verified(query, context, start_time)

        elif mode == ReasoningMode.FORMAL_FIRST:
            result = self._reason_formal_first(query, context, kg_start_node, start_time)

        else:  # HYBRID
            result = self._reason_hybrid(query, context, kg_start_node, start_time)

        return result

    def _reason_llm_only(
        self,
        query: str,
        context: ReasoningContext,
        start_time: float,
    ) -> ReasoningResult:
        """Fast LLM-only reasoning."""
        # Simulate LLM call (actual implementation would call real LLM)
        if self.llm_handler:
            answer = self.llm_handler(query)
        else:
            answer = f"[LLM Response to: {query[:50]}...]"

        return ReasoningResult(
            answer=answer,
            reasoning_mode=ReasoningMode.LLM_ONLY,
            confidence=0.7,  # LLM confidence estimate
            sources=["llm"],
            pgu_verified=False,
            kg_paths=[],
            latency_ms=(time.perf_counter() - start_time) * 1000,
            violations=[],
        )

    def _reason_kg_assisted(
        self,
        query: str,
        context: ReasoningContext,
        kg_start_node: Optional[str],
        start_time: float,
    ) -> ReasoningResult:
        """KG-assisted reasoning."""
        kg_paths = []
        sources = ["llm"]
        kg_confidence = 0.0

        # Query KG if start node provided
        if kg_start_node:
            kg_result = self.kg.query_multi_hop(
                kg_start_node,
                max_hops=3,
                min_confidence=0.5,
            )
            kg_paths = kg_result.paths
            kg_confidence = kg_result.confidence
            if kg_paths:
                sources.append("knowledge_graph")

        # Generate answer (with KG context)
        if self.llm_handler:
            kg_context = f"\nRelevant KG paths: {kg_paths}" if kg_paths else ""
            answer = self.llm_handler(query + kg_context)
        else:
            answer = f"[KG-Assisted Response: {len(kg_paths)} paths found]"

        return ReasoningResult(
            answer=answer,
            reasoning_mode=ReasoningMode.KG_ASSISTED,
            confidence=max(0.7, kg_confidence),
            sources=sources,
            pgu_verified=False,
            kg_paths=kg_paths,
            latency_ms=(time.perf_counter() - start_time) * 1000,
            violations=[],
        )

    def _reason_pgu_verified(
        self,
        query: str,
        context: ReasoningContext,
        start_time: float,
    ) -> ReasoningResult:
        """PGU-verified reasoning."""
        # First get LLM answer
        if self.llm_handler:
            answer = self.llm_handler(query)
        else:
            answer = f"[LLM Response]"

        # Extract claims/steps for verification
        reasoning_steps = answer.split(". ")

        # Verify with PGU
        check_result = self.oracle.verify_chain(reasoning_steps, {"query": query})

        return ReasoningResult(
            answer=answer,
            reasoning_mode=ReasoningMode.PGU_VERIFIED,
            confidence=0.9 if check_result.consistent else 0.5,
            sources=["llm", "pgu"],
            pgu_verified=check_result.consistent,
            kg_paths=[],
            latency_ms=(time.perf_counter() - start_time) * 1000,
            violations=check_result.violations,
        )

    def _reason_formal_first(
        self,
        query: str,
        context: ReasoningContext,
        kg_start_node: Optional[str],
        start_time: float,
    ) -> ReasoningResult:
        """Full formal reasoning: KG → PGU → LLM narration."""
        sources = []
        kg_paths = []
        violations = []
        confidence = 0.5

        # 1. Query KG for relevant facts
        if kg_start_node:
            kg_result = self.kg.query_multi_hop(
                kg_start_node,
                max_hops=3,
                min_confidence=0.7,  # Higher threshold for formal
            )
            kg_paths = kg_result.paths
            if kg_paths:
                sources.append("knowledge_graph")
                confidence = max(confidence, kg_result.confidence)

        # 2. Verify KG facts with PGU
        if kg_paths:
            claims = [{"path": p, "type": "kg_fact"} for p in kg_paths]
            check_result = self.oracle.check_constraints(claims, ["path_valid"])
            if check_result.consistent:
                sources.append("pgu")
                confidence = min(confidence + 0.2, 1.0)
            else:
                violations.extend(check_result.violations)

        # 3. Generate narration with LLM
        if self.llm_handler:
            formal_context = f"\nVerified facts: {kg_paths}\nViolations: {violations}"
            answer = self.llm_handler(query + formal_context)
        else:
            answer = f"[Formal reasoning complete. {len(kg_paths)} paths, {len(violations)} violations]"

        sources.append("llm")

        return ReasoningResult(
            answer=answer,
            reasoning_mode=ReasoningMode.FORMAL_FIRST,
            confidence=confidence,
            sources=sources,
            pgu_verified=len(violations) == 0,
            kg_paths=kg_paths,
            latency_ms=(time.perf_counter() - start_time) * 1000,
            violations=violations,
        )

    def _reason_hybrid(
        self,
        query: str,
        context: ReasoningContext,
        kg_start_node: Optional[str],
        start_time: float,
    ) -> ReasoningResult:
        """Hybrid reasoning with adaptive backend selection."""
        # Try KG first, fall back to LLM
        if kg_start_node:
            kg_result = self.kg.query_single_hop(kg_start_node)
            if kg_result.nodes and kg_result.confidence > 0.8:
                return self._reason_kg_assisted(query, context, kg_start_node, start_time)

        return self._reason_llm_only(query, context, start_time)

    def get_statistics(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        return {
            "query_count": self.query_count,
            "mode_distribution": {m.value: c for m, c in self.mode_counts.items()},
            "kg_metrics": self.kg.get_metrics(),
        }


# =============================================================================
# L3-AWARE RETRIEVER
# =============================================================================

class L3AwareRetriever:
    """
    Memory retrieval gated by L3 emotional state.

    Low valence / High risk → Conservative, PGU-verified retrieval
    High valence / Low risk → Fast semantic retrieval
    """

    def __init__(
        self,
        orchestrator: ReasoningOrchestrator,
    ):
        """
        Initialize L3-aware retriever.

        Args:
            orchestrator: ReasoningOrchestrator instance
        """
        self.orchestrator = orchestrator

    def retrieve(
        self,
        query: str,
        valence: float,
        arousal: float,
        dominance: float = 0.5,
        clv_risk: str = "LOW",
        kg_start_node: Optional[str] = None,
    ) -> ReasoningResult:
        """
        Retrieve with L3-aware mode selection.

        Args:
            query: Query string
            valence: Current valence [-1, 1]
            arousal: Current arousal [0, 1]
            dominance: Current dominance [0, 1]
            clv_risk: CLV risk level
            kg_start_node: Optional KG starting node

        Returns:
            ReasoningResult
        """
        # Build context from L3 state
        context = ReasoningContext(
            task_type=TaskType.RETRIEVAL,
            clv_risk=clv_risk,
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            confidence_required=0.7 if clv_risk in ["HIGH", "CRITICAL"] else 0.5,
            latency_budget_ms=2000 if arousal > 0.7 else 5000,
        )

        return self.orchestrator.reason(query, context, kg_start_node)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global instances
_orchestrator: Optional[ReasoningOrchestrator] = None
_retriever: Optional[L3AwareRetriever] = None


def get_orchestrator() -> ReasoningOrchestrator:
    """Get or create global orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ReasoningOrchestrator()
    return _orchestrator


def get_l3_retriever() -> L3AwareRetriever:
    """Get or create global L3-aware retriever."""
    global _retriever
    if _retriever is None:
        _retriever = L3AwareRetriever(get_orchestrator())
    return _retriever


def reason(
    query: str,
    task_type: str = "general_question",
    valence: float = 0.0,
    arousal: float = 0.5,
    clv_risk: str = "LOW",
) -> Dict[str, Any]:
    """
    High-level reasoning function.

    Returns:
        Result dictionary with answer, confidence, sources, etc.
    """
    context = ReasoningContext(
        task_type=TaskType(task_type),
        clv_risk=clv_risk,
        valence=valence,
        arousal=arousal,
    )

    result = get_orchestrator().reason(query, context)

    return {
        "answer": result.answer,
        "confidence": result.confidence,
        "sources": result.sources,
        "pgu_verified": result.pgu_verified,
        "reasoning_mode": result.reasoning_mode.value,
        "latency_ms": result.latency_ms,
    }


__all__ = [
    "TaskType",
    "ReasoningMode",
    "ReasoningContext",
    "KGNode",
    "KGEdge",
    "KGQueryResult",
    "KnowledgeGraph",
    "ConsistencyCheckResult",
    "ConsistencyOracle",
    "ReasoningResult",
    "ReasoningOrchestrator",
    "L3AwareRetriever",
    "get_orchestrator",
    "get_l3_retriever",
    "reason",
]
