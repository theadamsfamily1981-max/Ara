# ara/meta/causal_miner.py
"""
Causal Pattern Miner - From Superstition to Understanding
==========================================================

Moves beyond simple P(success | tool) to contrastive causal scores:

    Δ = P(success | tool, context) - P(success | ~tool, context)

This stops cargo-cult learning like "Claude is good for code" and
starts thinking in "Claude is good for code WHEN the problem is
architectural AND complex."

Key Concepts:
    - ToolOutcome: A single observation of tool use in context
    - ContextHash: Stable hash of context features for bucketing
    - CausalScore: Δ in [-1, 1] showing tool's causal effect
    - Contrastive: Compares tool vs baseline in same context

Usage:
    from ara.meta.causal_miner import CausalPatternMiner, ToolOutcome

    miner = CausalPatternMiner()

    # Log outcomes as they happen
    miner.log_outcome(ToolOutcome(
        tool="claude",
        context_hash="arch_complex",
        context_features={"task_type": "architecture", "complexity": "high"},
        success=True
    ))

    # Query causal effectiveness
    score = miner.estimate_causal_score("claude", "arch_complex")
    # score > 0 means Claude helps in this context
    # score < 0 means Claude hurts in this context
    # score ~ 0 means Claude is irrelevant

    # Get ranked tools for a context
    rankings = miner.rank_tools_for_context("arch_complex")
    # [("claude", 0.4), ("nova", 0.2), ("grep", -0.1)]
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Tuple, Optional, Set
from collections import defaultdict

log = logging.getLogger("Ara.CausalMiner")


@dataclass
class ToolOutcome:
    """
    A single observation of tool use.

    Captures what tool was used, in what context, and whether it succeeded.
    """
    tool: str                              # Tool/teacher name
    context_hash: str                      # Hash of context features
    context_features: Dict[str, Any]       # Raw context features
    success: bool                          # Did it achieve the goal?
    ts: float = field(default_factory=time.time)

    # Optional metadata
    duration_ms: Optional[float] = None    # How long did it take?
    retries: int = 0                       # How many retries needed?
    simulated: bool = False                # Is this a counterfactual simulation?

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CausalEstimate:
    """
    Causal effect estimate for a tool in a context.
    """
    tool: str
    context_hash: str
    delta: float                           # Causal effect Δ in [-1, 1]
    confidence: float                      # Confidence in estimate [0, 1]
    p_tool: float                          # P(success | tool, context)
    p_baseline: float                      # P(success | any tool, context)
    n_observations: int                    # Number of observations
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ToolStats:
    """Statistics for a (tool, context) pair."""
    success: int = 0
    total: int = 0
    total_duration_ms: float = 0.0
    total_retries: int = 0


class CausalPatternMiner:
    """
    Causal-aware pattern miner.

    Augments simple success rates with contrastive causal scores:
        Δ(tool, context) = P(Success | tool, context) - P(Success | ~tool, context)

    This distinguishes between:
        - "Tool X works everywhere" (high P, low Δ)
        - "Tool X is magic for this context" (high Δ)
        - "Tool X is harmful here" (negative Δ)

    Uses Beta-Bernoulli posteriors for smoothing with low counts.
    """

    def __init__(
        self,
        alpha: float = 1.0,     # Beta prior success count
        beta: float = 1.0,      # Beta prior failure count
        min_observations: int = 2,  # Min observations for estimate
        simulated_weight: float = 0.3,  # Weight for simulated outcomes
    ):
        """
        Initialize the causal miner.

        Args:
            alpha: Beta prior for successes (higher = more optimistic)
            beta: Beta prior for failures (higher = more pessimistic)
            min_observations: Minimum observations before estimating
            simulated_weight: Weight for simulated counterfactual outcomes
        """
        self.alpha = alpha
        self.beta = beta
        self.min_observations = min_observations
        self.simulated_weight = simulated_weight

        # Stats keyed by (tool, context_hash)
        self._tool_stats: Dict[Tuple[str, str], ToolStats] = defaultdict(ToolStats)

        # Baseline stats keyed by context_hash (all tools combined)
        self._context_stats: Dict[str, ToolStats] = defaultdict(ToolStats)

        # Context features cache (context_hash -> features)
        self._context_features: Dict[str, Dict[str, Any]] = {}

        # Raw outcome history for analysis
        self._outcomes: List[ToolOutcome] = []

        # Tools seen per context
        self._tools_per_context: Dict[str, Set[str]] = defaultdict(set)

        log.info("CausalPatternMiner initialized (alpha=%.1f, beta=%.1f)", alpha, beta)

    # =========================================================================
    # LOGGING OUTCOMES
    # =========================================================================

    def log_outcome(self, outcome: ToolOutcome) -> None:
        """
        Log an observed outcome.

        Args:
            outcome: The tool outcome to log
        """
        key = (outcome.tool, outcome.context_hash)
        weight = self.simulated_weight if outcome.simulated else 1.0

        # Update tool-specific stats
        stats = self._tool_stats[key]
        stats.total += weight
        if outcome.success:
            stats.success += weight
        if outcome.duration_ms:
            stats.total_duration_ms += outcome.duration_ms * weight
        stats.total_retries += outcome.retries

        # Update context baseline stats
        ctx_stats = self._context_stats[outcome.context_hash]
        ctx_stats.total += weight
        if outcome.success:
            ctx_stats.success += weight

        # Cache context features
        if outcome.context_hash not in self._context_features:
            self._context_features[outcome.context_hash] = outcome.context_features

        # Track tools per context
        self._tools_per_context[outcome.context_hash].add(outcome.tool)

        # Store raw outcome
        self._outcomes.append(outcome)

        log.debug(
            "Logged outcome: tool=%s ctx=%s success=%s (n=%d)",
            outcome.tool, outcome.context_hash, outcome.success, int(stats.total)
        )

    def log_from_session_graph(self, graph: "SessionGraph") -> None:
        """
        Log outcomes from a SessionGraph.

        Extracts tool calls and their success from the graph structure.
        """
        from ara.academy.session_graph import NodeType

        context_hash = graph.context_hash()
        context_features = graph.extract_context_features()

        for tool_node in graph.tool_calls():
            tool = tool_node.meta.get("tool")
            success = tool_node.meta.get("success")

            if tool and success is not None:
                self.log_outcome(ToolOutcome(
                    tool=tool,
                    context_hash=context_hash,
                    context_features=context_features,
                    success=success,
                ))

    # =========================================================================
    # CAUSAL ESTIMATION
    # =========================================================================

    def _beta_mean(self, successes: float, total: float) -> float:
        """
        Beta-Bernoulli posterior mean.

        (successes + alpha) / (total + alpha + beta)

        Provides smoothing when counts are low.
        """
        return (successes + self.alpha) / (total + self.alpha + self.beta)

    def estimate_causal_score(
        self,
        tool: str,
        context_hash: str
    ) -> Optional[CausalEstimate]:
        """
        Estimate the causal effect of a tool in a context.

        Returns:
            CausalEstimate with Δ in [-1, 1]:
                Δ > 0: Tool helps in this context
                Δ ~ 0: Tool is irrelevant
                Δ < 0: Tool hurts in this context

            Returns None if insufficient data.
        """
        ctx_stats = self._context_stats.get(context_hash)
        if not ctx_stats or ctx_stats.total < self.min_observations:
            return None  # Not enough context data

        tool_stats = self._tool_stats.get((tool, context_hash))
        if not tool_stats or tool_stats.total < self.min_observations:
            return None  # Not enough tool data

        # P(success | any tool, context) - baseline
        p_baseline = self._beta_mean(ctx_stats.success, ctx_stats.total)

        # P(success | this tool, context)
        p_tool = self._beta_mean(tool_stats.success, tool_stats.total)

        # Causal effect
        delta = p_tool - p_baseline

        # Confidence based on sample size (tanh to cap at 1)
        confidence = math.tanh(tool_stats.total / 10.0)

        # Scale delta by confidence
        weighted_delta = delta * confidence

        return CausalEstimate(
            tool=tool,
            context_hash=context_hash,
            delta=weighted_delta,
            confidence=confidence,
            p_tool=p_tool,
            p_baseline=p_baseline,
            n_observations=int(tool_stats.total),
        )

    def rank_tools_for_context(
        self,
        context_hash: str,
        min_confidence: float = 0.3
    ) -> List[Tuple[str, CausalEstimate]]:
        """
        Rank all tools by causal effectiveness for a context.

        Args:
            context_hash: Context to rank for
            min_confidence: Minimum confidence to include

        Returns:
            List of (tool, estimate) sorted by delta descending
        """
        tools = self._tools_per_context.get(context_hash, set())
        estimates = []

        for tool in tools:
            estimate = self.estimate_causal_score(tool, context_hash)
            if estimate and estimate.confidence >= min_confidence:
                estimates.append((tool, estimate))

        # Sort by causal effect (delta) descending
        estimates.sort(key=lambda x: x[1].delta, reverse=True)
        return estimates

    def recommend_tool(
        self,
        context_hash: str,
        exclude_tools: Optional[Set[str]] = None
    ) -> Optional[Tuple[str, CausalEstimate]]:
        """
        Recommend the best tool for a context.

        Args:
            context_hash: Context to recommend for
            exclude_tools: Tools to exclude from recommendation

        Returns:
            (tool_name, estimate) or None if no good recommendation
        """
        rankings = self.rank_tools_for_context(context_hash)
        exclude = exclude_tools or set()

        for tool, estimate in rankings:
            if tool not in exclude and estimate.delta > 0:
                return (tool, estimate)

        return None

    # =========================================================================
    # COUNTERFACTUAL SIMULATION
    # =========================================================================

    def simulate_counterfactual(
        self,
        tool: str,
        context_hash: str,
        assumed_success: bool
    ) -> None:
        """
        Simulate a counterfactual outcome.

        Useful for running "what if" scenarios when actual experiments
        are expensive. These get weighted lower than real observations.

        Args:
            tool: Tool to simulate
            context_hash: Context for simulation
            assumed_success: What we assume would happen
        """
        features = self._context_features.get(context_hash, {})

        self.log_outcome(ToolOutcome(
            tool=tool,
            context_hash=context_hash,
            context_features=features,
            success=assumed_success,
            simulated=True,
        ))

        log.debug(
            "Simulated counterfactual: tool=%s ctx=%s success=%s",
            tool, context_hash, assumed_success
        )

    # =========================================================================
    # ANALYSIS & REPORTING
    # =========================================================================

    def get_tool_summary(self, tool: str) -> Dict[str, Any]:
        """Get summary statistics for a tool across all contexts."""
        contexts_with_tool = [
            ctx for ctx, tools in self._tools_per_context.items()
            if tool in tools
        ]

        estimates = []
        for ctx in contexts_with_tool:
            est = self.estimate_causal_score(tool, ctx)
            if est:
                estimates.append(est)

        if not estimates:
            return {"tool": tool, "status": "insufficient_data"}

        avg_delta = sum(e.delta for e in estimates) / len(estimates)
        avg_p_tool = sum(e.p_tool for e in estimates) / len(estimates)
        total_obs = sum(e.n_observations for e in estimates)

        # Best and worst contexts
        by_delta = sorted(estimates, key=lambda e: e.delta, reverse=True)
        best_ctx = by_delta[0] if by_delta else None
        worst_ctx = by_delta[-1] if by_delta else None

        return {
            "tool": tool,
            "n_contexts": len(estimates),
            "total_observations": total_obs,
            "avg_causal_effect": avg_delta,
            "avg_success_rate": avg_p_tool,
            "best_context": best_ctx.context_hash if best_ctx else None,
            "best_delta": best_ctx.delta if best_ctx else None,
            "worst_context": worst_ctx.context_hash if worst_ctx else None,
            "worst_delta": worst_ctx.delta if worst_ctx else None,
        }

    def get_context_summary(self, context_hash: str) -> Dict[str, Any]:
        """Get summary statistics for a context."""
        ctx_stats = self._context_stats.get(context_hash)
        if not ctx_stats or ctx_stats.total < self.min_observations:
            return {"context_hash": context_hash, "status": "insufficient_data"}

        rankings = self.rank_tools_for_context(context_hash)
        features = self._context_features.get(context_hash, {})

        return {
            "context_hash": context_hash,
            "features": features,
            "baseline_success_rate": self._beta_mean(ctx_stats.success, ctx_stats.total),
            "total_observations": int(ctx_stats.total),
            "n_tools": len(rankings),
            "tool_rankings": [
                {"tool": t, "delta": e.delta, "confidence": e.confidence}
                for t, e in rankings[:5]  # Top 5
            ],
        }

    def generate_insights(self) -> List[str]:
        """
        Generate natural language insights from the data.

        This is what Ara would tell Croft about tool effectiveness.
        """
        insights = []

        # Get all tools
        all_tools = set()
        for ctx, tools in self._tools_per_context.items():
            all_tools.update(tools)

        # Analyze each tool
        for tool in all_tools:
            summary = self.get_tool_summary(tool)
            if summary.get("status") == "insufficient_data":
                continue

            delta = summary["avg_causal_effect"]
            n_obs = summary["total_observations"]

            if delta > 0.2:
                insights.append(
                    f"**{tool}** is consistently helpful (Δ={delta:.2f}, n={n_obs}). "
                    f"Best in context '{summary['best_context']}' (Δ={summary['best_delta']:.2f})."
                )
            elif delta < -0.1:
                insights.append(
                    f"**{tool}** may be counterproductive (Δ={delta:.2f}, n={n_obs}). "
                    f"Particularly weak in '{summary['worst_context']}'."
                )
            elif n_obs > 10:
                insights.append(
                    f"**{tool}** is context-neutral (Δ≈0, n={n_obs}). "
                    f"Consider when to use it more carefully."
                )

        return insights

    def export_state(self) -> Dict[str, Any]:
        """Export state for persistence."""
        return {
            "tool_stats": {
                f"{k[0]}:{k[1]}": asdict(v) if hasattr(v, '__dataclass_fields__') else {
                    "success": v.success,
                    "total": v.total,
                    "total_duration_ms": v.total_duration_ms,
                    "total_retries": v.total_retries,
                }
                for k, v in self._tool_stats.items()
            },
            "context_stats": {
                k: {
                    "success": v.success,
                    "total": v.total,
                }
                for k, v in self._context_stats.items()
            },
            "context_features": self._context_features,
            "outcomes_count": len(self._outcomes),
        }


# =============================================================================
# Convenience Functions
# =============================================================================

_default_miner: Optional[CausalPatternMiner] = None


def get_causal_miner() -> CausalPatternMiner:
    """Get the default CausalPatternMiner instance."""
    global _default_miner
    if _default_miner is None:
        _default_miner = CausalPatternMiner()
    return _default_miner


def hash_context(features: Dict[str, Any]) -> str:
    """Generate a stable hash for context features."""
    # Only hash stable features
    stable = {k: v for k, v in sorted(features.items())
              if not k.endswith("_count")}  # Exclude counts
    return hashlib.md5(json.dumps(stable, sort_keys=True).encode()).hexdigest()[:12]


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'ToolOutcome',
    'CausalEstimate',
    'ToolStats',
    'CausalPatternMiner',
    'get_causal_miner',
    'hash_context',
]
