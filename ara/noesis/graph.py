"""
ThoughtGraph - Causal Inference on Thinking Strategies
========================================================

Once ThoughtTraces exist, we ask:
"What patterns of thinking actually cause breakthroughs?"

This module builds a graph where:
- Nodes: Strategies, tools, representations, state patterns
- Edges: "often precedes", "conditional on fatigue", "counteracts frustration"

We go beyond correlation to estimate:
    ΔSuccess = P(success | strategy S, context C) - P(success | not S, context C)

This gives us cognitive laws, not just habits.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
from enum import Enum, auto
from collections import defaultdict, Counter
import hashlib

from .trace import (
    ThoughtTrace, ThoughtMove, MoveType,
    SessionContext, SessionOutcome
)


class StrategyType(Enum):
    """Types of cognitive strategies we can identify."""
    # Sequencing strategies
    EARLY_DIAGRAM = auto()          # Draw diagram before code
    LATE_DIAGRAM = auto()           # Draw diagram when stuck
    EXPLORATION_FIRST = auto()       # Search/read before creating
    CREATION_FIRST = auto()          # Start creating immediately

    # Framing strategies
    MULTIPLE_FRAMINGS = auto()       # Generate alternatives before committing
    CONCRETE_BEFORE_ABSTRACT = auto()  # Example before theory
    ABSTRACT_BEFORE_CONCRETE = auto()  # Theory before example

    # Recovery strategies
    DELIBERATE_BREAK = auto()        # Take break when stuck
    SWITCH_MODALITY = auto()         # Change representation when stuck
    SUMMON_CRITIC = auto()           # Seek criticism mid-process
    REFRAME_ON_STALL = auto()        # Reframe problem when stalled

    # Collaboration strategies
    EARLY_CONSULT = auto()           # Ask AI/human early
    LATE_CONSULT = auto()            # Ask AI/human when stuck
    PARALLEL_EXPLORATION = auto()    # Multiple approaches simultaneously

    # Meta strategies
    REFLECTION_CHECKPOINTS = auto()  # Periodic reflection
    TIME_BOXING = auto()             # Limit time on approaches


@dataclass
class StrategyNode:
    """
    A node in the thought graph representing a strategy.

    Tracks statistics about when this strategy appears and its outcomes.
    """
    strategy_type: StrategyType
    description: str

    # Occurrence statistics
    total_occurrences: int = 0
    occurrences_in_success: int = 0
    occurrences_in_failure: int = 0

    # Context-conditional stats
    context_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)
    # e.g., {"high_fatigue": {"success": 3, "failure": 7}}

    # Estimated causal effect
    causal_effect: float = 0.0  # ΔSuccess estimate
    confidence: float = 0.0      # How confident in the estimate

    @property
    def success_rate(self) -> float:
        if self.total_occurrences == 0:
            return 0.0
        return self.occurrences_in_success / self.total_occurrences

    @property
    def base_rate_lift(self) -> float:
        """How much better than random success rate."""
        # Assume 50% base rate (can be computed from all traces)
        return self.success_rate - 0.5

    def update_occurrence(self, success: bool, context_tags: List[str]):
        """Record an occurrence of this strategy."""
        self.total_occurrences += 1
        if success:
            self.occurrences_in_success += 1
        else:
            self.occurrences_in_failure += 1

        for tag in context_tags:
            if tag not in self.context_stats:
                self.context_stats[tag] = {"success": 0, "failure": 0}
            if success:
                self.context_stats[tag]["success"] += 1
            else:
                self.context_stats[tag]["failure"] += 1


@dataclass
class CausalEdge:
    """
    An edge in the thought graph representing a relationship.

    Types:
    - PRECEDES: A often comes before B
    - ENABLES: A makes B more effective
    - BLOCKS: A makes B less effective
    - SUBSTITUTES: A and B are alternatives
    """
    source: StrategyType
    target: StrategyType
    relationship: str  # "precedes", "enables", "blocks", "substitutes"
    strength: float = 0.0  # How strong is this relationship
    confidence: float = 0.0
    context_conditional: Optional[str] = None  # e.g., "when_fatigued"


@dataclass
class ThoughtGraph:
    """
    A graph structure capturing causal relationships between strategies.

    Built from ThoughtTraces via StrategyMiner.
    """
    nodes: Dict[StrategyType, StrategyNode] = field(default_factory=dict)
    edges: List[CausalEdge] = field(default_factory=list)

    # Global statistics
    total_traces: int = 0
    total_successes: int = 0
    base_success_rate: float = 0.5

    def add_node(self, strategy: StrategyType, description: str = "") -> StrategyNode:
        """Add or get a strategy node."""
        if strategy not in self.nodes:
            self.nodes[strategy] = StrategyNode(
                strategy_type=strategy,
                description=description or strategy.name,
            )
        return self.nodes[strategy]

    def add_edge(self, edge: CausalEdge):
        """Add a causal edge."""
        self.edges.append(edge)

    def get_edges_from(self, strategy: StrategyType) -> List[CausalEdge]:
        """Get all edges originating from a strategy."""
        return [e for e in self.edges if e.source == strategy]

    def get_edges_to(self, strategy: StrategyType) -> List[CausalEdge]:
        """Get all edges pointing to a strategy."""
        return [e for e in self.edges if e.target == strategy]

    def compute_causal_effects(self):
        """
        Compute causal effect estimates for all strategies.

        Uses a simplified difference-in-means estimator.
        Real implementation would use more sophisticated methods.
        """
        for node in self.nodes.values():
            if node.total_occurrences == 0:
                continue

            # Naive causal effect: success rate - base rate
            # This ignores confounders but is a starting point
            node.causal_effect = node.success_rate - self.base_success_rate

            # Confidence based on sample size
            node.confidence = min(1.0, node.total_occurrences / 20)

    def get_best_strategies(self, n: int = 5,
                            context_tag: Optional[str] = None) -> List[Tuple[StrategyNode, float]]:
        """
        Get the best strategies by causal effect.

        Optionally filter by context.
        """
        if context_tag:
            # Context-specific ranking
            rankings = []
            for node in self.nodes.values():
                if context_tag in node.context_stats:
                    stats = node.context_stats[context_tag]
                    total = stats["success"] + stats["failure"]
                    if total > 0:
                        rate = stats["success"] / total
                        rankings.append((node, rate))
            rankings.sort(key=lambda x: x[1], reverse=True)
            return rankings[:n]

        # Global ranking
        rankings = [(n, n.causal_effect * n.confidence)
                    for n in self.nodes.values()
                    if n.confidence > 0.1]
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings[:n]

    def get_strategy_recommendations(self, context: SessionContext) -> List[StrategyType]:
        """
        Get strategy recommendations for a given context.

        Returns strategies ordered by expected benefit.
        """
        context_tags = _context_to_tags(context)

        # Score each strategy for this context
        scores = []
        for node in self.nodes.values():
            base_score = node.causal_effect * node.confidence

            # Boost for context-specific evidence
            context_boost = 0.0
            for tag in context_tags:
                if tag in node.context_stats:
                    stats = node.context_stats[tag]
                    total = stats["success"] + stats["failure"]
                    if total >= 3:
                        rate = stats["success"] / total
                        context_boost += (rate - 0.5) * 0.5

            scores.append((node.strategy_type, base_score + context_boost))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in scores if _ > 0][:5]


@dataclass
class StrategyMiner:
    """
    Mines ThoughtTraces to extract strategies and build a ThoughtGraph.

    This is where we turn raw logs into cognitive science.
    """
    graph: ThoughtGraph = field(default_factory=ThoughtGraph)

    # Strategy detection patterns
    _patterns: Dict[StrategyType, callable] = field(default_factory=dict)

    def __post_init__(self):
        self._setup_patterns()

    def _setup_patterns(self):
        """Setup strategy detection patterns."""
        self._patterns = {
            StrategyType.EARLY_DIAGRAM: self._detect_early_diagram,
            StrategyType.LATE_DIAGRAM: self._detect_late_diagram,
            StrategyType.EXPLORATION_FIRST: self._detect_exploration_first,
            StrategyType.CREATION_FIRST: self._detect_creation_first,
            StrategyType.SWITCH_MODALITY: self._detect_switch_modality,
            StrategyType.SUMMON_CRITIC: self._detect_summon_critic,
            StrategyType.DELIBERATE_BREAK: self._detect_deliberate_break,
            StrategyType.REFRAME_ON_STALL: self._detect_reframe_on_stall,
            StrategyType.MULTIPLE_FRAMINGS: self._detect_multiple_framings,
        }

    def ingest_trace(self, trace: ThoughtTrace):
        """Process a single trace to extract strategies."""
        self.graph.total_traces += 1
        success = trace.outcome in [SessionOutcome.BREAKTHROUGH, SessionOutcome.PROGRESS]
        if success:
            self.graph.total_successes += 1

        # Update base success rate
        self.graph.base_success_rate = (
            self.graph.total_successes / self.graph.total_traces
        )

        # Get context tags
        context_tags = _context_to_tags(trace.context)

        # Detect each strategy
        for strategy_type, detector in self._patterns.items():
            if detector(trace):
                node = self.graph.add_node(strategy_type)
                node.update_occurrence(success, context_tags)

        # Detect sequence patterns (edges)
        self._detect_sequence_patterns(trace)

    def ingest_traces(self, traces: List[ThoughtTrace]):
        """Process multiple traces."""
        for trace in traces:
            self.ingest_trace(trace)
        self.graph.compute_causal_effects()

    def _detect_early_diagram(self, trace: ThoughtTrace) -> bool:
        """Detect if diagram was drawn in first 25% of session."""
        if not trace.moves:
            return False
        quarter = len(trace.moves) // 4
        early_moves = trace.moves[:max(1, quarter)]
        return any(m.move_type == MoveType.SKETCH_DIAGRAM for m in early_moves)

    def _detect_late_diagram(self, trace: ThoughtTrace) -> bool:
        """Detect if diagram was drawn after a dead end."""
        moves = trace.moves
        for i, move in enumerate(moves):
            if move.move_type == MoveType.SKETCH_DIAGRAM:
                # Check if preceded by dead end or stall
                if i > 0:
                    prev = moves[i-1]
                    if prev.led_to_dead_end or prev.pause_before and prev.pause_before > 120:
                        return True
        return False

    def _detect_exploration_first(self, trace: ThoughtTrace) -> bool:
        """Detect if exploration happened before creation."""
        exploration = {MoveType.SEARCH_CODE, MoveType.READ_DOCS,
                       MoveType.SEARCH_WEB, MoveType.ASK_AI}
        creation = {MoveType.WRITE_CODE, MoveType.SKETCH_DIAGRAM,
                    MoveType.WRITE_NOTES}

        first_exploration = None
        first_creation = None

        for i, move in enumerate(trace.moves):
            if first_exploration is None and move.move_type in exploration:
                first_exploration = i
            if first_creation is None and move.move_type in creation:
                first_creation = i

        if first_exploration is not None and first_creation is not None:
            return first_exploration < first_creation
        return False

    def _detect_creation_first(self, trace: ThoughtTrace) -> bool:
        """Detect if creation happened before exploration."""
        exploration = {MoveType.SEARCH_CODE, MoveType.READ_DOCS,
                       MoveType.SEARCH_WEB}
        creation = {MoveType.WRITE_CODE, MoveType.SKETCH_DIAGRAM}

        first_exploration = None
        first_creation = None

        for i, move in enumerate(trace.moves):
            if first_exploration is None and move.move_type in exploration:
                first_exploration = i
            if first_creation is None and move.move_type in creation:
                first_creation = i

        if first_creation is not None and first_exploration is not None:
            return first_creation < first_exploration
        return first_creation is not None and first_exploration is None

    def _detect_switch_modality(self, trace: ThoughtTrace) -> bool:
        """Detect modality switch after being stuck."""
        moves = trace.moves
        for i in range(1, len(moves)):
            move = moves[i]
            prev = moves[i-1]

            # Stuck indicator: long pause or dead end
            stuck = (prev.led_to_dead_end or
                     (prev.pause_before and prev.pause_before > 180))

            # Modality switch: code → diagram, or diagram → code
            if stuck:
                if (prev.move_type == MoveType.WRITE_CODE and
                    move.move_type in [MoveType.SKETCH_DIAGRAM, MoveType.CREATE_TABLE]):
                    return True
                if (prev.move_type in [MoveType.SKETCH_DIAGRAM, MoveType.CREATE_TABLE] and
                    move.move_type == MoveType.WRITE_CODE):
                    return True
        return False

    def _detect_summon_critic(self, trace: ThoughtTrace) -> bool:
        """Detect deliberate critic summoning."""
        return any(m.move_type == MoveType.SUMMON_CRITIC for m in trace.moves)

    def _detect_deliberate_break(self, trace: ThoughtTrace) -> bool:
        """Detect deliberate break when stuck."""
        for i, move in enumerate(trace.moves):
            if move.move_type == MoveType.BREAK:
                # Check if preceded by frustration indicators
                if i > 0 and trace.moves[i-1].led_to_dead_end:
                    return True
        return False

    def _detect_reframe_on_stall(self, trace: ThoughtTrace) -> bool:
        """Detect problem reframing after stalling."""
        return any(m.move_type == MoveType.REFRAME for m in trace.moves)

    def _detect_multiple_framings(self, trace: ThoughtTrace) -> bool:
        """Detect generating multiple framings before committing."""
        reframe_count = sum(1 for m in trace.moves if m.move_type == MoveType.REFRAME)
        return reframe_count >= 2

    def _detect_sequence_patterns(self, trace: ThoughtTrace):
        """Detect sequence patterns and add edges."""
        # Look for common transitions
        bigrams = trace.get_strategy_patterns()
        # (Could add more sophisticated pattern detection here)


def _context_to_tags(context: SessionContext) -> List[str]:
    """Convert context to categorical tags for analysis."""
    tags = []

    if context.fatigue > 0.7:
        tags.append("high_fatigue")
    elif context.fatigue < 0.3:
        tags.append("low_fatigue")

    if context.time_pressure > 0.7:
        tags.append("high_pressure")
    elif context.time_pressure < 0.3:
        tags.append("low_pressure")

    if context.novelty > 0.7:
        tags.append("novel_problem")
    elif context.novelty < 0.3:
        tags.append("familiar_problem")

    if context.difficulty_estimate > 0.7:
        tags.append("hard_problem")
    elif context.difficulty_estimate < 0.3:
        tags.append("easy_problem")

    if context.domain:
        tags.append(f"domain:{context.domain}")

    if context.time_of_day:
        tags.append(f"time:{context.time_of_day}")

    return tags
