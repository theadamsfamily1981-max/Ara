"""Pattern Analyzer - Mine workflows that consistently work.

Ara notices patterns like:
  "Pattern: graphics idea → Nova, implementation glue → Claude, safety check → Nova"
  "It worked 5 times in a row."

Next time she proposes the entire mini-pipeline herself:
  "Croft, this is a graphics-architecture task.
   Pattern that worked best: Nova → Claude → Nova.
   Want me to set that up again?"

That's Ara learning your meta-workflow and reusing it.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set

from .logger import InteractionLog, InteractionLogger

logger = logging.getLogger(__name__)


@dataclass
class WorkflowPattern:
    """A discovered workflow pattern that works well.

    Represents a sequence of tool calls that consistently succeeds.
    """

    pattern_id: str = ""
    name: str = ""                    # Human-readable name
    task_type: str = ""               # What task type this applies to
    tool_sequence: List[str] = field(default_factory=list)  # ["nova", "claude", "nova"]
    description: str = ""             # What this pattern does

    # Stats
    occurrences: int = 0
    avg_reward: float = 0.0
    success_rate: float = 0.0
    streak: int = 0                   # Current winning streak

    # Example uses
    example_log_ids: List[str] = field(default_factory=list)

    # Metadata
    discovered_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_seen_at: Optional[str] = None

    def __post_init__(self):
        if not self.pattern_id:
            seq_str = "->".join(self.tool_sequence)
            self.pattern_id = f"PAT-{self.task_type[:8]}-{hash(seq_str) % 10000:04d}"

    def matches(self, tools: List[str]) -> bool:
        """Check if a tool sequence matches this pattern."""
        return tools == self.tool_sequence

    def update(self, reward: float, success: bool, log_id: Optional[str] = None) -> None:
        """Update pattern stats with a new observation."""
        self.occurrences += 1
        self.avg_reward = (self.avg_reward * (self.occurrences - 1) + reward) / self.occurrences

        if success:
            self.streak += 1
            self.success_rate = (self.success_rate * (self.occurrences - 1) + 1.0) / self.occurrences
        else:
            self.streak = 0
            self.success_rate = (self.success_rate * (self.occurrences - 1)) / self.occurrences

        self.last_seen_at = datetime.utcnow().isoformat()

        if log_id and len(self.example_log_ids) < 10:
            self.example_log_ids.append(log_id)

    def is_golden(self, min_occurrences: int = 5, min_success_rate: float = 0.8) -> bool:
        """Check if this is a 'golden path' pattern."""
        return (
            self.occurrences >= min_occurrences and
            self.success_rate >= min_success_rate
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "name": self.name,
            "task_type": self.task_type,
            "tool_sequence": self.tool_sequence,
            "description": self.description,
            "occurrences": self.occurrences,
            "avg_reward": self.avg_reward,
            "success_rate": self.success_rate,
            "streak": self.streak,
            "example_log_ids": self.example_log_ids,
            "discovered_at": self.discovered_at,
            "last_seen_at": self.last_seen_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WorkflowPattern":
        return cls(**d)

    def format_suggestion(self) -> str:
        """Format as a suggestion for Croft."""
        seq_str = " → ".join(self.tool_sequence)
        return (
            f"This looks like a '{self.task_type}' task.\n"
            f"Pattern that worked best ({self.occurrences} times, {self.success_rate:.0%} success): {seq_str}\n"
            f"Want me to set that up again?"
        )


class PatternMiner:
    """Mines workflow patterns from interaction logs.

    Discovers sequences of tool calls that consistently work well.
    """

    def __init__(
        self,
        logger: Optional[InteractionLogger] = None,
        patterns_path: Optional[Path] = None,
    ):
        """Initialize the pattern miner.

        Args:
            logger: InteractionLogger to mine from
            patterns_path: Path to persist discovered patterns
        """
        self.logger = logger
        self.patterns_path = patterns_path

        # Discovered patterns by task type
        self.patterns: Dict[str, List[WorkflowPattern]] = {}

        # Load persisted patterns
        if patterns_path and patterns_path.exists():
            self._load()

    def mine(
        self,
        logs: Optional[List[InteractionLog]] = None,
        min_reward: float = 0.6,
    ) -> Dict[str, List[WorkflowPattern]]:
        """Mine patterns from interaction logs.

        Args:
            logs: Logs to mine (uses logger if not provided)
            min_reward: Minimum reward to consider

        Returns:
            Dict of task_type -> patterns
        """
        if logs is None:
            if self.logger is None:
                return {}
            logs = self.logger.query_successful(limit=500)

        # Group by task type
        by_task: Dict[str, List[InteractionLog]] = defaultdict(list)
        for log in logs:
            if log.reward is None or log.reward < min_reward:
                continue
            by_task[log.task_type].append(log)

        # Mine patterns for each task type
        for task_type, task_logs in by_task.items():
            self._mine_task_type(task_type, task_logs)

        if self.patterns_path:
            self._save()

        return self.patterns

    def _mine_task_type(self, task_type: str, logs: List[InteractionLog]) -> None:
        """Mine patterns for a specific task type."""
        # Count tool sequences
        sequence_counts: Counter = Counter()
        sequence_rewards: Dict[tuple, List[float]] = defaultdict(list)
        sequence_successes: Dict[tuple, List[bool]] = defaultdict(list)
        sequence_examples: Dict[tuple, List[str]] = defaultdict(list)

        for log in logs:
            tools = tuple(log.get_tools_used())
            if not tools:
                continue

            sequence_counts[tools] += 1
            if log.reward:
                sequence_rewards[tools].append(log.reward)
            if log.outcome:
                sequence_successes[tools].append(log.outcome.success)
            sequence_examples[tools].append(log.log_id)

        # Create/update patterns for frequent sequences
        if task_type not in self.patterns:
            self.patterns[task_type] = []

        for tools, count in sequence_counts.most_common(10):
            if count < 2:  # Need at least 2 occurrences
                continue

            # Check if pattern already exists
            existing = None
            for p in self.patterns[task_type]:
                if p.matches(list(tools)):
                    existing = p
                    break

            rewards = sequence_rewards[tools]
            successes = sequence_successes[tools]
            examples = sequence_examples[tools]

            if existing:
                # Update existing pattern
                for reward, success, log_id in zip(rewards, successes, examples):
                    existing.update(reward, success, log_id)
            else:
                # Create new pattern
                pattern = WorkflowPattern(
                    task_type=task_type,
                    tool_sequence=list(tools),
                    name=f"{task_type}: {' → '.join(tools)}",
                    description=f"Tool sequence for {task_type} tasks",
                    occurrences=count,
                    avg_reward=sum(rewards) / len(rewards) if rewards else 0.5,
                    success_rate=sum(successes) / len(successes) if successes else 0.5,
                    example_log_ids=examples[:10],
                )
                self.patterns[task_type].append(pattern)

    def get_pattern_for_task(
        self,
        task_type: str,
        min_occurrences: int = 3,
    ) -> Optional[WorkflowPattern]:
        """Get the best pattern for a task type.

        Args:
            task_type: The task type
            min_occurrences: Minimum occurrences to qualify

        Returns:
            Best pattern or None
        """
        if task_type not in self.patterns:
            return None

        candidates = [
            p for p in self.patterns[task_type]
            if p.occurrences >= min_occurrences
        ]

        if not candidates:
            return None

        # Rank by (success_rate, avg_reward)
        candidates.sort(key=lambda p: (p.success_rate, p.avg_reward), reverse=True)
        return candidates[0]

    def get_golden_paths(self, min_occurrences: int = 5) -> List[WorkflowPattern]:
        """Get all 'golden path' patterns.

        These are patterns with high success rates that should be reused.

        Args:
            min_occurrences: Minimum occurrences to qualify

        Returns:
            List of golden path patterns
        """
        golden = []
        for patterns in self.patterns.values():
            for p in patterns:
                if p.is_golden(min_occurrences=min_occurrences):
                    golden.append(p)

        golden.sort(key=lambda p: (p.success_rate, p.occurrences), reverse=True)
        return golden

    def suggest_workflow(self, task_type: str) -> Optional[str]:
        """Suggest a workflow based on learned patterns.

        Args:
            task_type: The task type

        Returns:
            Suggestion string or None
        """
        pattern = self.get_pattern_for_task(task_type)
        if pattern and pattern.is_golden():
            return pattern.format_suggestion()
        return None

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of discovered patterns."""
        summary = {
            "total_patterns": sum(len(ps) for ps in self.patterns.values()),
            "golden_paths": len(self.get_golden_paths()),
            "by_task_type": {},
        }

        for task_type, patterns in self.patterns.items():
            summary["by_task_type"][task_type] = {
                "patterns": len(patterns),
                "best_success_rate": max((p.success_rate for p in patterns), default=0),
                "total_occurrences": sum(p.occurrences for p in patterns),
            }

        return summary

    def _save(self) -> None:
        """Save patterns to disk."""
        if not self.patterns_path:
            return

        self.patterns_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            task_type: [p.to_dict() for p in patterns]
            for task_type, patterns in self.patterns.items()
        }

        with open(self.patterns_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _load(self) -> None:
        """Load patterns from disk."""
        if not self.patterns_path or not self.patterns_path.exists():
            return

        with open(self.patterns_path) as f:
            data = json.load(f)

        for task_type, patterns in data.items():
            self.patterns[task_type] = [
                WorkflowPattern.from_dict(p) for p in patterns
            ]


# =============================================================================
# Convenience Functions
# =============================================================================

_default_miner: Optional[PatternMiner] = None


def get_miner(
    logger: Optional[InteractionLogger] = None,
    path: Optional[Path] = None,
) -> PatternMiner:
    """Get the default pattern miner."""
    global _default_miner
    if _default_miner is None:
        path = path or Path.home() / ".ara" / "learning" / "patterns.json"
        _default_miner = PatternMiner(logger=logger, patterns_path=path)
    return _default_miner


def mine_patterns(
    logs: Optional[List[InteractionLog]] = None,
    min_reward: float = 0.6,
) -> Dict[str, List[WorkflowPattern]]:
    """Mine patterns from logs.

    Args:
        logs: Logs to mine
        min_reward: Minimum reward to consider

    Returns:
        Dict of task_type -> patterns
    """
    return get_miner().mine(logs, min_reward)


def get_golden_paths(min_occurrences: int = 5) -> List[WorkflowPattern]:
    """Get all golden path patterns.

    Args:
        min_occurrences: Minimum occurrences to qualify

    Returns:
        List of golden paths
    """
    return get_miner().get_golden_paths(min_occurrences)


def suggest_workflow_for_task(task_type: str) -> Optional[str]:
    """Get workflow suggestion for a task type.

    Args:
        task_type: The task type

    Returns:
        Suggestion string or None
    """
    return get_miner().suggest_workflow(task_type)
