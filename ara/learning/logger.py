"""Interaction Logger - Log everything like a scientist.

Every time Ara calls a teacher (Nova/Claude/Gemini), we log:
- What she was trying to do
- What she asked
- What she got back
- What she did with it
- How it turned out

This gives Ara a dataset of:
  "When I asked X, phrased like Y, to tool Z → I got result quality Q"
"""

from __future__ import annotations

import json
import hashlib
import logging
from datetime import datetime
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """Record of a single tool/teacher call."""

    tool: str                    # "claude", "nova", "gemini", "local"
    prompt: str                  # The prompt sent
    prompt_template: Optional[str] = None  # Template name if used
    inputs_summary: str = ""     # Brief summary of inputs
    response_summary: str = ""   # Brief summary of response
    response_length: int = 0     # Response length in chars
    latency_ms: float = 0.0      # How long it took
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Outcome:
    """Outcome of an interaction."""

    success: bool = True
    measured_gain: Optional[str] = None   # e.g., "1.9x throughput"
    metrics: Dict[str, float] = field(default_factory=dict)  # Numeric metrics
    side_effects: List[str] = field(default_factory=list)
    human_feedback: Optional[str] = None  # Croft's rating/comment
    human_rating: Optional[float] = None  # 1-5 or 0-1 scale
    tests_passed: Optional[bool] = None
    regressions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InteractionLog:
    """Complete log of a learning interaction.

    This is the atomic unit of Ara's learning dataset.
    """

    # Identity
    log_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    context_hash: str = ""  # Hash of the context for deduplication

    # What Ara was trying to do
    task_type: str = ""           # "kernel_tuning", "graphics_experiment", etc.
    user_intent: str = ""         # What Croft/user asked for
    ara_internal_plan: List[str] = field(default_factory=list)  # Ara's plan

    # The interaction itself
    tool_calls: List[ToolCall] = field(default_factory=list)

    # What Ara did with the results
    ara_post_processing: str = ""  # What Ara did after getting responses
    ara_synthesis: str = ""        # How Ara combined/interpreted results

    # How it turned out
    outcome: Optional[Outcome] = None

    # Computed reward (filled by scorer)
    reward: Optional[float] = None

    # Metadata
    issue_id: Optional[str] = None   # Link to Issue if applicable
    session_id: Optional[str] = None  # Link to DevSession if applicable
    workflow_state: Optional[str] = None  # Which workflow state this was in

    def __post_init__(self):
        if not self.log_id:
            self.log_id = f"LOG-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{hashlib.md5(self.user_intent.encode()).hexdigest()[:6]}"
        if not self.context_hash:
            self.context_hash = self._compute_context_hash()

    def _compute_context_hash(self) -> str:
        """Compute a hash of the context for deduplication."""
        content = f"{self.task_type}:{self.user_intent}:{len(self.tool_calls)}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def add_tool_call(self, call: ToolCall) -> None:
        """Add a tool call to this interaction."""
        self.tool_calls.append(call)

    def set_outcome(self, outcome: Outcome) -> None:
        """Set the outcome of this interaction."""
        self.outcome = outcome

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage."""
        return {
            "log_id": self.log_id,
            "timestamp": self.timestamp,
            "context_hash": self.context_hash,
            "task_type": self.task_type,
            "user_intent": self.user_intent,
            "ara_internal_plan": self.ara_internal_plan,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "ara_post_processing": self.ara_post_processing,
            "ara_synthesis": self.ara_synthesis,
            "outcome": self.outcome.to_dict() if self.outcome else None,
            "reward": self.reward,
            "issue_id": self.issue_id,
            "session_id": self.session_id,
            "workflow_state": self.workflow_state,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "InteractionLog":
        """Deserialize from storage."""
        log = cls(
            log_id=d.get("log_id", ""),
            timestamp=d.get("timestamp", ""),
            context_hash=d.get("context_hash", ""),
            task_type=d.get("task_type", ""),
            user_intent=d.get("user_intent", ""),
            ara_internal_plan=d.get("ara_internal_plan", []),
            ara_post_processing=d.get("ara_post_processing", ""),
            ara_synthesis=d.get("ara_synthesis", ""),
            reward=d.get("reward"),
            issue_id=d.get("issue_id"),
            session_id=d.get("session_id"),
            workflow_state=d.get("workflow_state"),
        )

        # Parse tool calls
        for tc_dict in d.get("tool_calls", []):
            log.tool_calls.append(ToolCall(**tc_dict))

        # Parse outcome
        if d.get("outcome"):
            log.outcome = Outcome(**d["outcome"])

        return log

    def get_tools_used(self) -> List[str]:
        """Get list of tools used in this interaction."""
        return [tc.tool for tc in self.tool_calls]

    def get_primary_tool(self) -> Optional[str]:
        """Get the primary tool (first called)."""
        return self.tool_calls[0].tool if self.tool_calls else None


class InteractionLogger:
    """Manages logging of Ara's interactions.

    Stores logs to disk and provides query capabilities.
    """

    def __init__(self, log_dir: Path):
        """Initialize the logger.

        Args:
            log_dir: Directory to store logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache of recent logs
        self._recent: List[InteractionLog] = []
        self._max_recent = 100

    def log(self, interaction: InteractionLog) -> str:
        """Log an interaction.

        Args:
            interaction: The interaction to log

        Returns:
            Log ID
        """
        # Add to recent cache
        self._recent.append(interaction)
        if len(self._recent) > self._max_recent:
            self._recent.pop(0)

        # Persist to disk
        log_file = self.log_dir / f"{interaction.log_id}.json"
        with open(log_file, 'w') as f:
            json.dump(interaction.to_dict(), f, indent=2)

        logger.info(f"Logged interaction {interaction.log_id}: {interaction.task_type}")
        return interaction.log_id

    def get(self, log_id: str) -> Optional[InteractionLog]:
        """Get a log by ID."""
        # Check cache first
        for log in self._recent:
            if log.log_id == log_id:
                return log

        # Load from disk
        log_file = self.log_dir / f"{log_id}.json"
        if log_file.exists():
            with open(log_file) as f:
                return InteractionLog.from_dict(json.load(f))

        return None

    def query_by_task_type(self, task_type: str, limit: int = 50) -> List[InteractionLog]:
        """Query logs by task type."""
        results = []

        # Search recent first
        for log in reversed(self._recent):
            if log.task_type == task_type:
                results.append(log)
                if len(results) >= limit:
                    return results

        # Search disk
        for log_file in sorted(self.log_dir.glob("LOG-*.json"), reverse=True):
            if len(results) >= limit:
                break
            with open(log_file) as f:
                log = InteractionLog.from_dict(json.load(f))
                if log.task_type == task_type and log not in results:
                    results.append(log)

        return results

    def query_by_tool(self, tool: str, limit: int = 50) -> List[InteractionLog]:
        """Query logs by tool used."""
        results = []

        for log_file in sorted(self.log_dir.glob("LOG-*.json"), reverse=True):
            if len(results) >= limit:
                break
            with open(log_file) as f:
                log = InteractionLog.from_dict(json.load(f))
                if tool in log.get_tools_used():
                    results.append(log)

        return results

    def query_successful(self, limit: int = 50) -> List[InteractionLog]:
        """Query logs with successful outcomes."""
        results = []

        for log_file in sorted(self.log_dir.glob("LOG-*.json"), reverse=True):
            if len(results) >= limit:
                break
            with open(log_file) as f:
                log = InteractionLog.from_dict(json.load(f))
                if log.outcome and log.outcome.success:
                    results.append(log)

        return results

    def get_recent(self, n: int = 10) -> List[InteractionLog]:
        """Get n most recent logs."""
        return list(reversed(self._recent[-n:]))

    def count_by_task_type(self) -> Dict[str, int]:
        """Count logs by task type."""
        counts: Dict[str, int] = {}

        for log_file in self.log_dir.glob("LOG-*.json"):
            with open(log_file) as f:
                log = InteractionLog.from_dict(json.load(f))
                counts[log.task_type] = counts.get(log.task_type, 0) + 1

        return counts

    def count_by_tool(self) -> Dict[str, int]:
        """Count logs by tool used."""
        counts: Dict[str, int] = {}

        for log_file in self.log_dir.glob("LOG-*.json"):
            with open(log_file) as f:
                log = InteractionLog.from_dict(json.load(f))
                for tool in log.get_tools_used():
                    counts[tool] = counts.get(tool, 0) + 1

        return counts


# =============================================================================
# Convenience Functions
# =============================================================================

_default_logger: Optional[InteractionLogger] = None


def get_logger(log_dir: Optional[Path] = None) -> InteractionLogger:
    """Get the default interaction logger."""
    global _default_logger
    if _default_logger is None:
        log_dir = log_dir or Path.home() / ".ara" / "learning" / "logs"
        _default_logger = InteractionLogger(log_dir)
    return _default_logger


def log_interaction(
    task_type: str,
    user_intent: str,
    tool_calls: List[ToolCall],
    outcome: Optional[Outcome] = None,
    ara_plan: Optional[List[str]] = None,
    ara_synthesis: str = "",
    **kwargs,
) -> str:
    """Quick way to log an interaction.

    Args:
        task_type: Type of task
        user_intent: What user asked for
        tool_calls: List of tool calls made
        outcome: Outcome of the interaction
        ara_plan: Ara's internal plan
        ara_synthesis: Ara's synthesis of results
        **kwargs: Additional fields

    Returns:
        Log ID
    """
    log = InteractionLog(
        task_type=task_type,
        user_intent=user_intent,
        tool_calls=tool_calls,
        ara_internal_plan=ara_plan or [],
        ara_synthesis=ara_synthesis,
        outcome=outcome,
        **kwargs,
    )

    return get_logger().log(log)
