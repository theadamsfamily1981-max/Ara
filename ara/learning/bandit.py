"""Tool Bandit - Adaptive tool selection using multi-armed bandit.

For each task type, Ara keeps an estimate of which tool works best.
This module implements epsilon-greedy bandit that:
- Exploits (90%): Uses the tool with best average reward
- Explores (10%): Tries a random tool to learn more

Over time, Ara automatically migrates to:
- "Claude for tight production code"
- "Nova for high-risk architecture decisions"
- "Gemini for speculative physics graphics stuff"

...because her own history says so.
"""

from __future__ import annotations

import json
import random
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolStats:
    """Statistics for a tool on a specific task type."""

    tool: str
    task_type: str
    count: int = 0
    total_reward: float = 0.0
    avg_reward: float = 0.5  # Prior: neutral

    # Additional stats for analysis
    min_reward: float = 1.0
    max_reward: float = 0.0
    recent_rewards: List[float] = field(default_factory=list)  # Last N rewards

    def update(self, reward: float, max_recent: int = 10) -> None:
        """Update stats with a new reward observation.

        Args:
            reward: The observed reward
            max_recent: Max recent rewards to keep
        """
        self.count += 1
        self.total_reward += reward
        self.avg_reward = self.total_reward / self.count

        self.min_reward = min(self.min_reward, reward)
        self.max_reward = max(self.max_reward, reward)

        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > max_recent:
            self.recent_rewards.pop(0)

    def get_recent_avg(self) -> float:
        """Get average of recent rewards (more responsive to changes)."""
        if not self.recent_rewards:
            return self.avg_reward
        return sum(self.recent_rewards) / len(self.recent_rewards)

    def get_ucb_score(self, total_pulls: int, c: float = 2.0) -> float:
        """Get Upper Confidence Bound score.

        UCB balances exploitation (high avg) with exploration (uncertainty).

        Args:
            total_pulls: Total pulls across all tools
            c: Exploration constant (higher = more exploration)

        Returns:
            UCB score
        """
        import math
        if self.count == 0:
            return float('inf')  # Never tried → infinite optimism
        exploration_bonus = c * math.sqrt(math.log(total_pulls) / self.count)
        return self.avg_reward + exploration_bonus

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool,
            "task_type": self.task_type,
            "count": self.count,
            "total_reward": self.total_reward,
            "avg_reward": self.avg_reward,
            "min_reward": self.min_reward,
            "max_reward": self.max_reward,
            "recent_rewards": self.recent_rewards,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ToolStats":
        return cls(
            tool=d["tool"],
            task_type=d["task_type"],
            count=d.get("count", 0),
            total_reward=d.get("total_reward", 0.0),
            avg_reward=d.get("avg_reward", 0.5),
            min_reward=d.get("min_reward", 1.0),
            max_reward=d.get("max_reward", 0.0),
            recent_rewards=d.get("recent_rewards", []),
        )


class ToolBandit:
    """Multi-armed bandit for adaptive tool selection.

    Manages per-task-type statistics for each tool and selects
    tools using epsilon-greedy or UCB strategy.
    """

    def __init__(
        self,
        available_tools: List[str],
        stats_path: Optional[Path] = None,
        epsilon: float = 0.1,
        strategy: str = "epsilon_greedy",
    ):
        """Initialize the bandit.

        Args:
            available_tools: List of available tool names
            stats_path: Path to persist stats
            epsilon: Exploration rate for epsilon-greedy
            strategy: "epsilon_greedy" or "ucb"
        """
        self.available_tools = available_tools
        self.stats_path = stats_path
        self.epsilon = epsilon
        self.strategy = strategy

        # Stats: task_type -> tool -> ToolStats
        self.stats: Dict[str, Dict[str, ToolStats]] = {}

        # Load persisted stats
        if stats_path and stats_path.exists():
            self._load_stats()

    def _get_or_create_stats(self, task_type: str, tool: str) -> ToolStats:
        """Get or create stats for a task/tool pair."""
        if task_type not in self.stats:
            self.stats[task_type] = {}

        if tool not in self.stats[task_type]:
            self.stats[task_type][tool] = ToolStats(tool=tool, task_type=task_type)

        return self.stats[task_type][tool]

    def choose(self, task_type: str, exclude: Optional[List[str]] = None) -> str:
        """Choose a tool for a task type.

        Args:
            task_type: The type of task
            exclude: Tools to exclude from selection

        Returns:
            Selected tool name
        """
        available = [t for t in self.available_tools if t not in (exclude or [])]
        if not available:
            raise ValueError("No tools available")

        if self.strategy == "ucb":
            return self._choose_ucb(task_type, available)
        else:
            return self._choose_epsilon_greedy(task_type, available)

    def _choose_epsilon_greedy(self, task_type: str, available: List[str]) -> str:
        """Epsilon-greedy selection.

        With probability epsilon: explore (random tool)
        With probability 1-epsilon: exploit (best average tool)
        """
        if random.random() < self.epsilon:
            # Explore
            choice = random.choice(available)
            logger.debug(f"Bandit exploring: {choice} for {task_type}")
            return choice

        # Exploit: find best average
        best_tool = None
        best_avg = -1.0

        for tool in available:
            stats = self._get_or_create_stats(task_type, tool)
            if stats.avg_reward > best_avg:
                best_avg = stats.avg_reward
                best_tool = tool

        if best_tool is None:
            best_tool = random.choice(available)

        logger.debug(f"Bandit exploiting: {best_tool} (avg={best_avg:.2f}) for {task_type}")
        return best_tool

    def _choose_ucb(self, task_type: str, available: List[str]) -> str:
        """Upper Confidence Bound selection.

        Balances exploitation with exploration based on uncertainty.
        """
        total_pulls = sum(
            self._get_or_create_stats(task_type, t).count
            for t in available
        )
        total_pulls = max(1, total_pulls)  # Avoid log(0)

        best_tool = None
        best_ucb = -1.0

        for tool in available:
            stats = self._get_or_create_stats(task_type, tool)
            ucb = stats.get_ucb_score(total_pulls)
            if ucb > best_ucb:
                best_ucb = ucb
                best_tool = tool

        if best_tool is None:
            best_tool = random.choice(available)

        logger.debug(f"Bandit UCB: {best_tool} (ucb={best_ucb:.2f}) for {task_type}")
        return best_tool

    def update(self, task_type: str, tool: str, reward: float) -> None:
        """Update stats after observing a reward.

        Args:
            task_type: The task type
            tool: The tool that was used
            reward: The observed reward
        """
        stats = self._get_or_create_stats(task_type, tool)
        stats.update(reward)
        logger.info(f"Bandit update: {tool} for {task_type} = {reward:.2f} (avg={stats.avg_reward:.2f}, n={stats.count})")

        # Persist
        if self.stats_path:
            self._save_stats()

    def get_rankings(self, task_type: str) -> List[tuple]:
        """Get tools ranked by average reward for a task type.

        Returns:
            List of (tool, avg_reward, count) tuples, sorted best first
        """
        if task_type not in self.stats:
            return [(t, 0.5, 0) for t in self.available_tools]

        rankings = []
        for tool in self.available_tools:
            stats = self._get_or_create_stats(task_type, tool)
            rankings.append((tool, stats.avg_reward, stats.count))

        return sorted(rankings, key=lambda x: x[1], reverse=True)

    def get_best_tool(self, task_type: str) -> Optional[str]:
        """Get the best tool for a task type (pure exploitation)."""
        rankings = self.get_rankings(task_type)
        if rankings:
            return rankings[0][0]
        return None

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all stats."""
        summary = {}
        for task_type, tools in self.stats.items():
            summary[task_type] = {
                tool: {
                    "avg_reward": stats.avg_reward,
                    "count": stats.count,
                    "recent_avg": stats.get_recent_avg(),
                }
                for tool, stats in tools.items()
            }
        return summary

    def _save_stats(self) -> None:
        """Save stats to disk."""
        if not self.stats_path:
            return

        self.stats_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            task_type: {
                tool: stats.to_dict()
                for tool, stats in tools.items()
            }
            for task_type, tools in self.stats.items()
        }

        with open(self.stats_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_stats(self) -> None:
        """Load stats from disk."""
        if not self.stats_path or not self.stats_path.exists():
            return

        with open(self.stats_path) as f:
            data = json.load(f)

        for task_type, tools in data.items():
            self.stats[task_type] = {}
            for tool, stats_dict in tools.items():
                self.stats[task_type][tool] = ToolStats.from_dict(stats_dict)


# =============================================================================
# Convenience Functions
# =============================================================================

_default_bandit: Optional[ToolBandit] = None


def get_bandit(
    tools: Optional[List[str]] = None,
    stats_path: Optional[Path] = None,
) -> ToolBandit:
    """Get the default tool bandit."""
    global _default_bandit
    if _default_bandit is None:
        tools = tools or ["claude", "nova", "gemini", "local"]
        stats_path = stats_path or Path.home() / ".ara" / "learning" / "bandit_stats.json"
        _default_bandit = ToolBandit(
            available_tools=tools,
            stats_path=stats_path,
        )
    return _default_bandit


def choose_tool(task_type: str, exclude: Optional[List[str]] = None) -> str:
    """Choose a tool for a task type.

    Args:
        task_type: The type of task
        exclude: Tools to exclude

    Returns:
        Selected tool name
    """
    return get_bandit().choose(task_type, exclude)


def update_tool_stats(task_type: str, tool: str, reward: float) -> None:
    """Update stats after using a tool.

    Args:
        task_type: The task type
        tool: The tool used
        reward: The observed reward
    """
    get_bandit().update(task_type, tool, reward)
