"""
Swarm Stats - Optimization Analytics
=====================================

Compute layer and pattern statistics for optimization.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from collections import defaultdict
import json
from pathlib import Path

from .schema import AgentLayer, JobRecord, JobOutcome, RiskLevel


@dataclass
class LayerStats:
    """Statistics for a single agent layer."""
    layer: AgentLayer
    total_jobs: int = 0
    successes: int = 0
    failures: int = 0
    corrections_made: int = 0  # Times this layer fixed lower layers
    corrections_received: int = 0  # Times this layer was fixed by higher
    total_cost: float = 0.0
    total_latency_ms: float = 0.0

    @property
    def success_rate(self) -> float:
        if self.total_jobs == 0:
            return 0.0
        return self.successes / self.total_jobs

    @property
    def correction_rate(self) -> float:
        if self.total_jobs == 0:
            return 0.0
        return self.corrections_received / self.total_jobs

    @property
    def avg_cost(self) -> float:
        if self.total_jobs == 0:
            return 0.0
        return self.total_cost / self.total_jobs

    @property
    def avg_latency_ms(self) -> float:
        if self.total_jobs == 0:
            return 0.0
        return self.total_latency_ms / self.total_jobs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer.value,
            "total_jobs": self.total_jobs,
            "success_rate": round(self.success_rate, 4),
            "correction_rate": round(self.correction_rate, 4),
            "avg_cost": round(self.avg_cost, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "corrections_made": self.corrections_made,
        }


@dataclass
class PatternStats:
    """Statistics for a job pattern."""
    pattern_id: str
    total_runs: int = 0
    wins: int = 0
    total_cost: float = 0.0
    total_latency_ms: float = 0.0

    @property
    def win_rate(self) -> float:
        if self.total_runs == 0:
            return 0.0
        return self.wins / self.total_runs

    @property
    def avg_cost(self) -> float:
        if self.total_runs == 0:
            return 0.0
        return self.total_cost / self.total_runs

    @property
    def avg_latency_ms(self) -> float:
        if self.total_runs == 0:
            return 0.0
        return self.total_latency_ms / self.total_runs

    @property
    def score(self) -> float:
        """Score for ranking (win_rate / cost)."""
        return self.win_rate / (self.avg_cost + 1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "total_runs": self.total_runs,
            "win_rate": round(self.win_rate, 4),
            "avg_cost": round(self.avg_cost, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "score": round(self.score, 6),
        }


def compute_layer_stats(jobs: List[JobRecord]) -> Dict[AgentLayer, LayerStats]:
    """Compute statistics per layer from job records."""
    stats = {layer: LayerStats(layer=layer) for layer in AgentLayer}

    for job in jobs:
        # Track agent runs per layer
        layers_in_job = set()
        for agent in job.agents:
            layer = agent.layer
            layers_in_job.add(layer)
            stats[layer].total_jobs += 1
            stats[layer].total_cost += agent.cost
            stats[layer].total_latency_ms += agent.latency_ms

            if agent.outcome == JobOutcome.SUCCESS:
                stats[layer].successes += 1
            else:
                stats[layer].failures += 1

        # Track corrections
        for fix in job.fixes:
            stats[fix.from_layer].corrections_made += 1
            stats[fix.to_layer].corrections_received += 1

    return stats


def compute_pattern_stats(jobs: List[JobRecord]) -> Dict[str, PatternStats]:
    """Compute statistics per pattern from job records."""
    stats: Dict[str, PatternStats] = defaultdict(lambda: PatternStats(pattern_id=""))

    for job in jobs:
        pattern_id = job.pattern_id
        if pattern_id not in stats:
            stats[pattern_id] = PatternStats(pattern_id=pattern_id)

        p_stats = stats[pattern_id]
        p_stats.total_runs += 1
        p_stats.total_cost += job.total_cost
        p_stats.total_latency_ms += job.total_latency_ms

        if job.outcome == JobOutcome.SUCCESS:
            p_stats.wins += 1

    return dict(stats)


@dataclass
class OptimizationSuggestion:
    """A suggested optimization action."""
    action: str  # "promote", "demote", "deprecate", "clone"
    target: str  # job_type, pattern_id, or agent_id
    reason: str
    priority: str  # "high", "medium", "low"
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "target": self.target,
            "reason": self.reason,
            "priority": self.priority,
            "details": self.details,
        }


def get_optimization_suggestions(
    layer_stats: Dict[AgentLayer, LayerStats],
    pattern_stats: Dict[str, PatternStats],
    thresholds: Optional[Dict[str, float]] = None,
) -> List[OptimizationSuggestion]:
    """Generate optimization suggestions based on stats."""

    if thresholds is None:
        thresholds = {
            "low_success_rate": 0.5,
            "high_correction_rate": 0.3,
            "high_cost_threshold": 500.0,
            "deprecate_win_rate": 0.3,
            "min_runs_for_suggestion": 10,
        }

    suggestions = []

    # Check layer performance
    for layer, stats in layer_stats.items():
        if stats.total_jobs < thresholds["min_runs_for_suggestion"]:
            continue

        # High correction rate + low success = promote job types to higher layer
        if (stats.correction_rate > thresholds["high_correction_rate"] and
            stats.success_rate < thresholds["low_success_rate"]):
            suggestions.append(OptimizationSuggestion(
                action="promote",
                target=f"layer_{layer.value}_jobs",
                reason=f"L{layer.value} has {stats.correction_rate:.1%} correction rate, {stats.success_rate:.1%} success",
                priority="high",
                details={
                    "current_layer": layer.value,
                    "success_rate": stats.success_rate,
                    "correction_rate": stats.correction_rate,
                },
            ))

        # High success + high cost = try cloning to lower layer
        if (stats.success_rate > 0.9 and
            stats.avg_cost > thresholds["high_cost_threshold"] and
            layer.value > 0):
            suggestions.append(OptimizationSuggestion(
                action="demote",
                target=f"layer_{layer.value}_easy_jobs",
                reason=f"L{layer.value} has {stats.success_rate:.1%} success but high cost ({stats.avg_cost:.0f})",
                priority="medium",
                details={
                    "current_layer": layer.value,
                    "avg_cost": stats.avg_cost,
                    "potential_savings": stats.avg_cost * 0.5,
                },
            ))

    # Check pattern performance
    for pattern_id, stats in pattern_stats.items():
        if stats.total_runs < thresholds["min_runs_for_suggestion"]:
            continue

        # Low win rate = deprecate
        if stats.win_rate < thresholds["deprecate_win_rate"]:
            suggestions.append(OptimizationSuggestion(
                action="deprecate",
                target=pattern_id,
                reason=f"Pattern {pattern_id} has only {stats.win_rate:.1%} win rate",
                priority="high",
                details={
                    "win_rate": stats.win_rate,
                    "total_runs": stats.total_runs,
                },
            ))

    # Sort by priority
    priority_order = {"high": 0, "medium": 1, "low": 2}
    suggestions.sort(key=lambda s: priority_order.get(s.priority, 2))

    return suggestions


def load_jobs_from_jsonl(path: str) -> List[JobRecord]:
    """Load job records from JSONL file."""
    jobs = []
    p = Path(path)
    if not p.exists():
        return jobs

    with open(p, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    jobs.append(JobRecord.from_dict(data))
                except (json.JSONDecodeError, KeyError):
                    continue

    return jobs


def save_jobs_to_jsonl(jobs: List[JobRecord], path: str):
    """Save job records to JSONL file."""
    with open(path, 'a') as f:
        for job in jobs:
            f.write(job.to_jsonl() + '\n')
