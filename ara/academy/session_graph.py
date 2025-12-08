# ara/academy/session_graph.py
"""
Semantic Session Graph - Topological Understanding of Teaching Sessions
========================================================================

Lifts raw transcripts into a graph structure so Ara can detect patterns
even when teachers are messy. No more brittle regex - structural understanding.

Node Types:
    - USER_INTENT: What the user wanted
    - TOOL_CALL: Invocation of a tool/teacher
    - TOOL_RESULT: Outcome of tool call
    - TEACHER_EXPLANATION: Reasoning/explanation from teacher
    - PLAN_STEP: Structured planning output
    - REFINEMENT: User refining their request

Edge Types:
    - attempts_to_solve: Intent → Tool
    - results_in: Tool → Result
    - explained_by: Result → Explanation
    - refined_by: Intent → Refined Intent
    - answered_by: Intent → Explanation
    - causes: Generic causal link

Usage:
    from ara.academy.session_graph import SessionGraphBuilder

    builder = SessionGraphBuilder()
    graph = builder.build_from_transcript(session_id, transcript)

    # Now mine patterns on subgraphs, not regex
    patterns = graph.find_subgraph_pattern(
        "USER_INTENT -> TOOL_CALL[success=False] -> USER_INTENT -> TOOL_CALL[success=True]"
    )
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import List, Dict, Any, Optional, Set, Tuple

log = logging.getLogger("Ara.SessionGraph")


class NodeType(Enum):
    """Types of nodes in the session graph."""
    USER_INTENT = auto()        # What the user wanted to accomplish
    TOOL_CALL = auto()          # Invocation of tool/teacher
    TOOL_RESULT = auto()        # Output from tool
    TEACHER_EXPLANATION = auto() # Reasoning/explanation
    PLAN_STEP = auto()          # Structured planning
    REFINEMENT = auto()         # User clarification/refinement
    CODE_BLOCK = auto()         # Code artifact
    ERROR = auto()              # Error/failure
    OTHER = auto()              # Catch-all


class EdgeType(str, Enum):
    """Types of edges connecting nodes."""
    ATTEMPTS_TO_SOLVE = "attempts_to_solve"  # Intent → Tool
    RESULTS_IN = "results_in"                 # Tool → Result
    EXPLAINED_BY = "explained_by"             # Result → Explanation
    REFINED_BY = "refined_by"                 # Intent → Refined Intent
    ANSWERED_BY = "answered_by"               # Intent → Explanation
    CAUSES = "causes"                         # Generic causal
    FOLLOWS = "follows"                       # Sequential
    DEPENDS_ON = "depends_on"                 # Dependency


@dataclass
class Node:
    """A node in the session graph."""
    id: str
    type: NodeType
    text: str
    meta: Dict[str, Any] = field(default_factory=dict)

    # Computed features
    embedding: Optional[List[float]] = None  # For similarity matching

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.name,
            "text": self.text[:200] + "..." if len(self.text) > 200 else self.text,
            "meta": self.meta,
        }

    def signature(self) -> str:
        """Short signature for pattern matching."""
        tool = self.meta.get("tool", "")
        success = self.meta.get("success", "")
        return f"{self.type.name}[tool={tool},success={success}]" if tool else self.type.name


@dataclass
class Edge:
    """An edge connecting two nodes."""
    src: str
    dst: str
    label: EdgeType
    weight: float = 1.0
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "src": self.src,
            "dst": self.dst,
            "label": self.label.value,
            "weight": self.weight,
        }


@dataclass
class SessionGraph:
    """
    Graph representation of a teaching session.

    Nodes represent events (intents, tool calls, explanations).
    Edges represent relationships (causes, refines, answers).
    """
    session_id: str
    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)

    # Computed indices
    _adjacency: Dict[str, List[str]] = field(default_factory=dict)
    _reverse_adjacency: Dict[str, List[str]] = field(default_factory=dict)

    def add_node(self, node: Node) -> None:
        """Add a node to the graph."""
        self.nodes[node.id] = node
        if node.id not in self._adjacency:
            self._adjacency[node.id] = []
        if node.id not in self._reverse_adjacency:
            self._reverse_adjacency[node.id] = []

    def add_edge(self, src: str, dst: str, label: EdgeType, weight: float = 1.0) -> None:
        """Add an edge to the graph."""
        edge = Edge(src=src, dst=dst, label=label, weight=weight)
        self.edges.append(edge)
        self._adjacency.setdefault(src, []).append(dst)
        self._reverse_adjacency.setdefault(dst, []).append(src)

    def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by ID."""
        return self.nodes.get(node_id)

    def successors(self, node_id: str) -> List[Node]:
        """Get all successor nodes."""
        return [self.nodes[nid] for nid in self._adjacency.get(node_id, []) if nid in self.nodes]

    def predecessors(self, node_id: str) -> List[Node]:
        """Get all predecessor nodes."""
        return [self.nodes[nid] for nid in self._reverse_adjacency.get(node_id, []) if nid in self.nodes]

    def nodes_by_type(self, node_type: NodeType) -> List[Node]:
        """Get all nodes of a given type."""
        return [n for n in self.nodes.values() if n.type == node_type]

    def tool_calls(self) -> List[Node]:
        """Get all tool call nodes."""
        return self.nodes_by_type(NodeType.TOOL_CALL)

    def intents(self) -> List[Node]:
        """Get all user intent nodes."""
        return self.nodes_by_type(NodeType.USER_INTENT)

    # =========================================================================
    # PATTERN DETECTION
    # =========================================================================

    def find_paths(self, start_type: NodeType, end_type: NodeType, max_length: int = 5) -> List[List[Node]]:
        """Find all paths from nodes of start_type to nodes of end_type."""
        paths = []
        start_nodes = self.nodes_by_type(start_type)

        for start in start_nodes:
            self._dfs_paths(start.id, end_type, [start], paths, max_length)

        return paths

    def _dfs_paths(
        self,
        current: str,
        end_type: NodeType,
        path: List[Node],
        paths: List[List[Node]],
        max_length: int
    ) -> None:
        """DFS helper for path finding."""
        if len(path) > max_length:
            return

        current_node = self.nodes.get(current)
        if current_node and current_node.type == end_type and len(path) > 1:
            paths.append(list(path))
            return

        for succ_id in self._adjacency.get(current, []):
            if succ_id not in [n.id for n in path]:  # Avoid cycles
                succ = self.nodes.get(succ_id)
                if succ:
                    path.append(succ)
                    self._dfs_paths(succ_id, end_type, path, paths, max_length)
                    path.pop()

    def find_retry_patterns(self) -> List[Dict[str, Any]]:
        """
        Find patterns where user retried after failure.

        Pattern: INTENT -> TOOL[fail] -> REFINEMENT -> TOOL[success]
        """
        patterns = []

        for intent in self.intents():
            # Find tool calls from this intent
            tool_calls = [n for n in self.successors(intent.id) if n.type == NodeType.TOOL_CALL]

            for tool in tool_calls:
                if tool.meta.get("success") is False:
                    # Look for refinement after this failure
                    for sibling in self.successors(intent.id):
                        if sibling.type == NodeType.REFINEMENT:
                            # Look for successful retry
                            retry_tools = [n for n in self.successors(sibling.id)
                                          if n.type == NodeType.TOOL_CALL and n.meta.get("success") is True]
                            for retry in retry_tools:
                                patterns.append({
                                    "type": "retry_success",
                                    "original_intent": intent.id,
                                    "failed_tool": tool.id,
                                    "refinement": sibling.id,
                                    "successful_tool": retry.id,
                                    "tool_name": tool.meta.get("tool"),
                                })

        return patterns

    def find_socratic_loops(self) -> List[Dict[str, Any]]:
        """
        Find Socratic teaching patterns.

        Pattern: INTENT -> EXPLANATION -> REFINEMENT -> EXPLANATION (repeat)
        """
        patterns = []

        for intent in self.intents():
            explanations = [n for n in self.successors(intent.id)
                           if n.type == NodeType.TEACHER_EXPLANATION]

            for exp in explanations:
                # Look for refinement after explanation
                refinements = [n for n in self.successors(exp.id)
                              if n.type == NodeType.REFINEMENT]

                for ref in refinements:
                    # Look for another explanation
                    followup_exp = [n for n in self.successors(ref.id)
                                   if n.type == NodeType.TEACHER_EXPLANATION]

                    if followup_exp:
                        patterns.append({
                            "type": "socratic_loop",
                            "initial_intent": intent.id,
                            "first_explanation": exp.id,
                            "refinement": ref.id,
                            "followup_explanation": followup_exp[0].id,
                            "depth": 1,  # Could recurse to find deeper loops
                        })

        return patterns

    # =========================================================================
    # CONTEXT EXTRACTION
    # =========================================================================

    def extract_context_features(self) -> Dict[str, Any]:
        """
        Extract features for causal mining.

        Returns a dict that can be hashed for context matching.
        """
        all_text = " ".join(n.text.lower() for n in self.nodes.values()
                           if n.type in [NodeType.USER_INTENT, NodeType.REFINEMENT])

        features = {}

        # Task type detection
        if any(kw in all_text for kw in ["architecture", "design", "structure", "system"]):
            features["task_type"] = "architecture"
        elif any(kw in all_text for kw in ["bug", "error", "fix", "crash", "fail"]):
            features["task_type"] = "debugging"
        elif any(kw in all_text for kw in ["implement", "create", "build", "add"]):
            features["task_type"] = "implementation"
        elif any(kw in all_text for kw in ["explain", "understand", "how", "why"]):
            features["task_type"] = "learning"
        else:
            features["task_type"] = "general"

        # Complexity estimation
        features["node_count"] = len(self.nodes)
        features["tool_count"] = len(self.tool_calls())
        features["intent_count"] = len(self.intents())
        features["retry_count"] = len(self.find_retry_patterns())

        if features["node_count"] > 20 or features["tool_count"] > 5:
            features["complexity"] = "high"
        elif features["node_count"] > 10 or features["tool_count"] > 2:
            features["complexity"] = "medium"
        else:
            features["complexity"] = "low"

        # Tools used
        tools_used = set()
        for tool_node in self.tool_calls():
            tool_name = tool_node.meta.get("tool")
            if tool_name:
                tools_used.add(tool_name)
        features["tools_used"] = sorted(tools_used)

        # Success rate
        tool_nodes = self.tool_calls()
        if tool_nodes:
            successes = sum(1 for t in tool_nodes if t.meta.get("success") is True)
            features["success_rate"] = successes / len(tool_nodes)
        else:
            features["success_rate"] = None

        return features

    def context_hash(self) -> str:
        """Generate a hash for context matching."""
        features = self.extract_context_features()
        # Only hash stable features (not counts which vary)
        stable = {
            "task_type": features["task_type"],
            "complexity": features["complexity"],
        }
        return hashlib.md5(json.dumps(stable, sort_keys=True).encode()).hexdigest()[:12]

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "session_id": self.session_id,
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
            "edges": [e.to_dict() for e in self.edges],
            "context": self.extract_context_features(),
        }

    def summary(self) -> str:
        """Human-readable summary."""
        features = self.extract_context_features()
        return (
            f"SessionGraph[{self.session_id}]: "
            f"{len(self.nodes)} nodes, {len(self.edges)} edges, "
            f"task={features['task_type']}, complexity={features['complexity']}, "
            f"success_rate={features.get('success_rate', 'N/A')}"
        )


class SessionGraphBuilder:
    """
    Builds SessionGraph from raw transcripts.

    Takes a linear transcript (list of events) and infers causal structure:
        - UserIntent -> ToolCall -> ToolResult
        - ToolResult -> TeacherExplanation
        - TeacherExplanation -> Refinement -> NewToolCall
    """

    # Heuristics for detecting node types from text
    PLAN_INDICATORS = ["step 1", "1.", "plan:", "here is a plan", "i'll", "let me", "first,"]
    CODE_INDICATORS = ["```", "def ", "class ", "import ", "function "]
    ERROR_INDICATORS = ["error:", "exception:", "failed:", "traceback", "cannot", "unable"]

    def build_from_transcript(
        self,
        session_id: str,
        transcript: List[Dict[str, Any]]
    ) -> SessionGraph:
        """
        Build graph from transcript.

        Args:
            session_id: Unique session identifier
            transcript: List of events like:
                {"role": "user", "text": "..."}
                {"role": "assistant", "tool": "nova", "text": "...", "success": True}
                {"role": "assistant", "text": "Explanation ..."}

        Returns:
            SessionGraph with inferred structure
        """
        graph = SessionGraph(session_id=session_id)

        last_intent_node: Optional[Node] = None
        last_tool_node: Optional[Node] = None
        last_any_node: Optional[Node] = None

        for idx, evt in enumerate(transcript):
            nid = f"n{idx}"
            role = evt.get("role", "unknown")
            text = evt.get("text", "") or ""
            tool = evt.get("tool")
            success = evt.get("success")

            node: Optional[Node] = None

            if role == "user":
                # Determine if this is initial intent or refinement
                if last_intent_node and last_tool_node:
                    node_type = NodeType.REFINEMENT
                else:
                    node_type = NodeType.USER_INTENT

                node = Node(
                    id=nid,
                    type=node_type,
                    text=text,
                    meta={"role": role, "raw_index": idx}
                )
                graph.add_node(node)

                # Edge from previous intent if refining
                if node_type == NodeType.REFINEMENT and last_intent_node:
                    graph.add_edge(last_intent_node.id, node.id, EdgeType.REFINED_BY)

                if node_type == NodeType.USER_INTENT:
                    last_intent_node = node
                    last_tool_node = None

            elif role == "assistant" and tool:
                # Tool call
                node = Node(
                    id=nid,
                    type=NodeType.TOOL_CALL,
                    text=text,
                    meta={"tool": tool, "success": success, "role": role, "raw_index": idx}
                )
                graph.add_node(node)

                # Edge from intent
                if last_intent_node:
                    graph.add_edge(last_intent_node.id, node.id, EdgeType.ATTEMPTS_TO_SOLVE)

                last_tool_node = node

            elif role == "assistant" and not tool:
                # Determine type from content
                node_type = self._classify_assistant_text(text)

                node = Node(
                    id=nid,
                    type=node_type,
                    text=text,
                    meta={"role": role, "raw_index": idx}
                )
                graph.add_node(node)

                # Edges based on context
                if last_tool_node:
                    graph.add_edge(last_tool_node.id, node.id, EdgeType.EXPLAINED_BY)
                elif last_intent_node:
                    graph.add_edge(last_intent_node.id, node.id, EdgeType.ANSWERED_BY)

            else:
                # Unknown role
                node = Node(
                    id=nid,
                    type=NodeType.OTHER,
                    text=text,
                    meta={"role": role, "raw_index": idx}
                )
                graph.add_node(node)

            # Sequential edge
            if node and last_any_node:
                graph.add_edge(last_any_node.id, node.id, EdgeType.FOLLOWS, weight=0.5)

            if node:
                last_any_node = node

        log.info(f"Built {graph.summary()}")
        return graph

    def _classify_assistant_text(self, text: str) -> NodeType:
        """Classify assistant text into node type."""
        text_lower = text.strip().lower()

        # Check for plan structure
        if any(text_lower.startswith(ind) for ind in self.PLAN_INDICATORS):
            return NodeType.PLAN_STEP

        # Check for code blocks
        if any(ind in text for ind in self.CODE_INDICATORS):
            return NodeType.CODE_BLOCK

        # Check for errors
        if any(ind in text_lower for ind in self.ERROR_INDICATORS):
            return NodeType.ERROR

        # Default to explanation
        return NodeType.TEACHER_EXPLANATION


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'NodeType',
    'EdgeType',
    'Node',
    'Edge',
    'SessionGraph',
    'SessionGraphBuilder',
]
