"""Pattern Miner - Discover what works from interaction history.

Analyzes Ara's interaction logs to find:
- Tool preferences by task type
- Strategy success rates
- Workflow patterns (golden paths)
- Failure modes to avoid
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

from .schemas import InteractionRecord, PatternSuggestion
from .meta_logger import get_meta_logger, MetaLogger

logger = logging.getLogger(__name__)


@dataclass
class ToolPerformance:
    """Performance stats for a tool."""

    tool_name: str
    total_calls: int = 0
    successful_calls: int = 0
    total_latency_ms: float = 0.0
    avg_quality: float = 0.0
    quality_samples: int = 0

    @property
    def success_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.successful_calls / self.total_calls

    @property
    def avg_latency_ms(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.total_latency_ms / self.total_calls

    def update(
        self,
        success: bool,
        latency_ms: Optional[float] = None,
        quality: Optional[float] = None,
    ) -> None:
        self.total_calls += 1
        if success:
            self.successful_calls += 1
        if latency_ms is not None:
            self.total_latency_ms += latency_ms
        if quality is not None:
            self.quality_samples += 1
            # Exponential moving average
            alpha = 1.0 / self.quality_samples
            self.avg_quality = alpha * quality + (1 - alpha) * self.avg_quality


@dataclass
class WorkflowPattern:
    """A discovered workflow pattern."""

    pattern_id: str
    tool_sequence: List[str]
    context_tags: List[str]
    occurrences: int = 0
    success_count: int = 0
    avg_quality: float = 0.0
    quality_samples: int = 0

    @property
    def success_rate(self) -> float:
        if self.occurrences == 0:
            return 0.0
        return self.success_count / self.occurrences

    @property
    def is_golden_path(self) -> bool:
        """Golden path = 80%+ success rate with 3+ occurrences."""
        return self.success_rate >= 0.8 and self.occurrences >= 3


@dataclass
class StrategyStats:
    """Stats for a strategy."""

    strategy: str
    uses: int = 0
    successes: int = 0
    avg_quality: float = 0.0
    quality_samples: int = 0

    @property
    def success_rate(self) -> float:
        if self.uses == 0:
            return 0.0
        return self.successes / self.uses


class PatternMiner:
    """Mines patterns from interaction history.

    Discovers:
    - Which tools work best for which tasks
    - Which strategies succeed
    - Workflow patterns that lead to success
    - Failure modes to avoid
    """

    def __init__(self, meta_logger: Optional[MetaLogger] = None):
        """Initialize the pattern miner.

        Args:
            meta_logger: Logger to read from (uses default if None)
        """
        self.meta_logger = meta_logger or get_meta_logger()

        # Caches (rebuilt on analyze())
        self._tool_stats: Dict[str, ToolPerformance] = {}
        self._tool_by_context: Dict[str, Dict[str, ToolPerformance]] = {}
        self._strategy_stats: Dict[str, StrategyStats] = {}
        self._workflow_patterns: Dict[str, WorkflowPattern] = {}
        self._last_analyzed: Optional[datetime] = None

    def analyze(self, days: int = 30, force: bool = False) -> Dict[str, Any]:
        """Analyze interaction history.

        Args:
            days: Number of days to analyze
            force: Force re-analysis even if recently done

        Returns:
            Analysis summary
        """
        # Skip if recently analyzed
        if not force and self._last_analyzed:
            if datetime.utcnow() - self._last_analyzed < timedelta(hours=1):
                return self._get_summary()

        # Reset caches
        self._tool_stats.clear()
        self._tool_by_context.clear()
        self._strategy_stats.clear()
        self._workflow_patterns.clear()

        # Query interactions
        since = datetime.utcnow() - timedelta(days=days)
        records = self.meta_logger.query(since=since, limit=10000)

        logger.info(f"Analyzing {len(records)} interactions from last {days} days")

        for record in records:
            self._process_record(record)

        self._last_analyzed = datetime.utcnow()
        return self._get_summary()

    def _process_record(self, record: InteractionRecord) -> None:
        """Process a single interaction record."""
        quality = record.outcome_quality
        success = quality is not None and quality >= 0.5

        # Strategy stats
        if record.chosen_strategy:
            if record.chosen_strategy not in self._strategy_stats:
                self._strategy_stats[record.chosen_strategy] = StrategyStats(
                    strategy=record.chosen_strategy
                )
            stats = self._strategy_stats[record.chosen_strategy]
            stats.uses += 1
            if success:
                stats.successes += 1
            if quality is not None:
                stats.quality_samples += 1
                alpha = 1.0 / stats.quality_samples
                stats.avg_quality = alpha * quality + (1 - alpha) * stats.avg_quality

        # Tool stats
        for tc in record.tools_used:
            # Global tool stats
            if tc.tool_name not in self._tool_stats:
                self._tool_stats[tc.tool_name] = ToolPerformance(tool_name=tc.tool_name)
            self._tool_stats[tc.tool_name].update(tc.success, tc.latency_ms, quality)

            # Per-context tool stats
            for tag in record.context_tags:
                if tag not in self._tool_by_context:
                    self._tool_by_context[tag] = {}
                if tc.tool_name not in self._tool_by_context[tag]:
                    self._tool_by_context[tag][tc.tool_name] = ToolPerformance(
                        tool_name=tc.tool_name
                    )
                self._tool_by_context[tag][tc.tool_name].update(
                    tc.success, tc.latency_ms, quality
                )

        # Workflow patterns
        if len(record.tools_used) >= 2:
            tool_seq = tuple(tc.tool_name for tc in record.tools_used)
            pattern_id = "->".join(tool_seq)
            if pattern_id not in self._workflow_patterns:
                self._workflow_patterns[pattern_id] = WorkflowPattern(
                    pattern_id=pattern_id,
                    tool_sequence=list(tool_seq),
                    context_tags=record.context_tags[:],
                )
            pattern = self._workflow_patterns[pattern_id]
            pattern.occurrences += 1
            if success:
                pattern.success_count += 1
            if quality is not None:
                pattern.quality_samples += 1
                alpha = 1.0 / pattern.quality_samples
                pattern.avg_quality = alpha * quality + (1 - alpha) * pattern.avg_quality

    def _get_summary(self) -> Dict[str, Any]:
        """Get analysis summary."""
        return {
            "analyzed_at": self._last_analyzed.isoformat() if self._last_analyzed else None,
            "tools": {
                name: {
                    "success_rate": stats.success_rate,
                    "avg_latency_ms": stats.avg_latency_ms,
                    "avg_quality": stats.avg_quality,
                    "total_calls": stats.total_calls,
                }
                for name, stats in self._tool_stats.items()
            },
            "strategies": {
                name: {
                    "success_rate": stats.success_rate,
                    "avg_quality": stats.avg_quality,
                    "uses": stats.uses,
                }
                for name, stats in self._strategy_stats.items()
            },
            "golden_paths": [
                {
                    "pattern": p.pattern_id,
                    "success_rate": p.success_rate,
                    "occurrences": p.occurrences,
                }
                for p in self._workflow_patterns.values()
                if p.is_golden_path
            ],
        }

    def get_tool_ranking(self, context_tag: Optional[str] = None) -> List[Tuple[str, float]]:
        """Get tools ranked by performance.

        Args:
            context_tag: Optional context filter

        Returns:
            List of (tool_name, score) tuples
        """
        if context_tag and context_tag in self._tool_by_context:
            stats = self._tool_by_context[context_tag]
        else:
            stats = self._tool_stats

        rankings = []
        for name, perf in stats.items():
            # Score = weighted combo of success rate and quality
            score = 0.4 * perf.success_rate + 0.6 * perf.avg_quality
            rankings.append((name, score))

        return sorted(rankings, key=lambda x: x[1], reverse=True)

    def get_best_tool(self, context_tag: Optional[str] = None) -> Optional[str]:
        """Get the best performing tool.

        Args:
            context_tag: Optional context filter

        Returns:
            Tool name or None
        """
        rankings = self.get_tool_ranking(context_tag)
        return rankings[0][0] if rankings else None

    def get_golden_paths(self, min_success_rate: float = 0.8) -> List[WorkflowPattern]:
        """Get successful workflow patterns.

        Args:
            min_success_rate: Minimum success rate

        Returns:
            List of golden path patterns
        """
        return [
            p for p in self._workflow_patterns.values()
            if p.success_rate >= min_success_rate and p.occurrences >= 3
        ]

    def suggest_patterns(self, max_suggestions: int = 5) -> List[PatternSuggestion]:
        """Generate pattern-based suggestions.

        Args:
            max_suggestions: Maximum suggestions to return

        Returns:
            List of pattern suggestions
        """
        self.analyze()  # Ensure analysis is fresh
        suggestions = []

        # Suggest tool preferences
        for context, tools in self._tool_by_context.items():
            if len(tools) < 2:
                continue

            rankings = sorted(
                tools.values(),
                key=lambda t: 0.4 * t.success_rate + 0.6 * t.avg_quality,
                reverse=True,
            )
            best = rankings[0]

            if best.success_rate >= 0.8 and best.total_calls >= 5:
                suggestions.append(
                    PatternSuggestion(
                        scope="tool_routing",
                        confidence=min(0.9, 0.5 + 0.1 * best.total_calls),
                        description=f"For {context} tasks, {best.tool_name} performs best",
                        recommendation=f"When context includes '{context}', prefer {best.tool_name} (success rate: {best.success_rate:.0%})",
                        evidence={
                            "context": context,
                            "tool": best.tool_name,
                            "success_rate": best.success_rate,
                            "samples": best.total_calls,
                        },
                        safe_to_auto_apply=best.success_rate >= 0.9,
                    )
                )

        # Suggest golden paths
        for pattern in self.get_golden_paths():
            suggestions.append(
                PatternSuggestion(
                    scope="workflow",
                    confidence=min(0.9, 0.5 + 0.05 * pattern.occurrences),
                    description=f"Workflow pattern: {pattern.pattern_id}",
                    recommendation=f"This sequence works well: {' -> '.join(pattern.tool_sequence)}",
                    evidence={
                        "pattern": pattern.pattern_id,
                        "success_rate": pattern.success_rate,
                        "occurrences": pattern.occurrences,
                    },
                    safe_to_auto_apply=False,
                )
            )

        # Sort by confidence and limit
        suggestions.sort(key=lambda s: s.confidence, reverse=True)
        return suggestions[:max_suggestions]

    def get_failure_modes(self) -> List[Dict[str, Any]]:
        """Identify failure modes to avoid.

        Returns:
            List of failure patterns
        """
        failures = []

        # Tools with low success rate
        for name, stats in self._tool_stats.items():
            if stats.success_rate < 0.5 and stats.total_calls >= 5:
                failures.append({
                    "type": "tool_failure",
                    "tool": name,
                    "success_rate": stats.success_rate,
                    "samples": stats.total_calls,
                    "recommendation": f"Consider avoiding {name} or investigating failures",
                })

        # Strategies that don't work
        for name, stats in self._strategy_stats.items():
            if stats.success_rate < 0.5 and stats.uses >= 5:
                failures.append({
                    "type": "strategy_failure",
                    "strategy": name,
                    "success_rate": stats.success_rate,
                    "samples": stats.uses,
                    "recommendation": f"Strategy '{name}' has low success rate",
                })

        # Patterns that fail
        for pattern_id, pattern in self._workflow_patterns.items():
            if pattern.success_rate < 0.3 and pattern.occurrences >= 3:
                failures.append({
                    "type": "workflow_failure",
                    "pattern": pattern_id,
                    "success_rate": pattern.success_rate,
                    "samples": pattern.occurrences,
                    "recommendation": f"Avoid this workflow: {pattern_id}",
                })

        return failures


# =============================================================================
# Convenience Functions
# =============================================================================

_default_miner: Optional[PatternMiner] = None


def get_miner() -> PatternMiner:
    """Get the default pattern miner."""
    global _default_miner
    if _default_miner is None:
        _default_miner = PatternMiner()
    return _default_miner


def mine_patterns(days: int = 30) -> Dict[str, Any]:
    """Mine patterns from recent history.

    Args:
        days: Days of history to analyze

    Returns:
        Analysis summary
    """
    return get_miner().analyze(days=days)


def get_tool_stats(context: Optional[str] = None) -> List[Tuple[str, float]]:
    """Get tool performance stats.

    Args:
        context: Optional context filter

    Returns:
        Tool rankings
    """
    get_miner().analyze()
    return get_miner().get_tool_ranking(context)


def get_suggestions() -> List[PatternSuggestion]:
    """Get pattern-based suggestions.

    Returns:
        List of suggestions
    """
    return get_miner().suggest_patterns()
