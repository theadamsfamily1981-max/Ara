"""Meta Logger - Structured logging for Ara's learning layer.

Logs every interaction with teachers to JSONL for later analysis.
This is Ara's lab notebook - every experiment recorded.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Iterator, Dict, Any

from .schemas import InteractionRecord, ToolCall

logger = logging.getLogger(__name__)


class MetaLogger:
    """Structured interaction logger for meta-learning.

    Writes interactions as JSONL for:
    - Pattern mining
    - Reward analysis
    - Research agenda progress tracking
    """

    def __init__(
        self,
        log_path: Optional[Path] = None,
        enabled: bool = True,
        min_quality_to_log: float = 0.0,
    ):
        """Initialize the meta logger.

        Args:
            log_path: Path to the JSONL log file
            enabled: Whether logging is enabled
            min_quality_to_log: Minimum outcome quality to log (filter noise)
        """
        self.log_path = log_path or Path.home() / ".ara" / "meta" / "interactions.jsonl"
        self.enabled = enabled
        self.min_quality_to_log = min_quality_to_log

        # Ensure directory exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # In-memory buffer for recent interactions
        self._recent_buffer: List[InteractionRecord] = []
        self._buffer_max = 100

    def log_interaction(self, record: InteractionRecord) -> bool:
        """Log an interaction record.

        Args:
            record: The interaction to log

        Returns:
            True if logged, False if filtered out
        """
        if not self.enabled:
            return False

        # Quality filter
        if record.outcome_quality is not None:
            if record.outcome_quality < self.min_quality_to_log:
                logger.debug(f"Skipping low-quality interaction: {record.id}")
                return False

        # Add to recent buffer
        self._recent_buffer.append(record)
        if len(self._recent_buffer) > self._buffer_max:
            self._recent_buffer.pop(0)

        # Persist to JSONL
        try:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(record.model_dump_json() + "\n")
            logger.info(f"Logged interaction: {record.id}")
            return True
        except Exception as e:
            logger.error(f"Failed to log interaction: {e}")
            return False

    def log_tool_call(
        self,
        tool_name: str,
        role: str = "teacher",
        input_summary: str = "",
        output_summary: str = "",
        success: bool = True,
        latency_ms: Optional[float] = None,
        error_type: Optional[str] = None,
    ) -> ToolCall:
        """Create and return a tool call record.

        Convenience method for building ToolCall objects.
        """
        return ToolCall(
            tool_name=tool_name,
            role=role,
            input_summary=input_summary[:500],  # Truncate for safety
            output_summary=output_summary[:500],
            success=success,
            latency_ms=latency_ms,
            error_type=error_type,
        )

    def get_recent(self, n: int = 10) -> List[InteractionRecord]:
        """Get recent interactions from buffer.

        Args:
            n: Number of recent interactions

        Returns:
            List of recent interactions
        """
        return self._recent_buffer[-n:]

    def iter_all(self) -> Iterator[InteractionRecord]:
        """Iterate over all logged interactions.

        Yields:
            InteractionRecord objects from the log
        """
        if not self.log_path.exists():
            return

        with self.log_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    yield InteractionRecord(**data)
                except (json.JSONDecodeError, Exception) as e:
                    logger.warning(f"Failed to parse log line: {e}")
                    continue

    def query(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        strategy: Optional[str] = None,
        tool: Optional[str] = None,
        min_quality: Optional[float] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[InteractionRecord]:
        """Query interactions with filters.

        Args:
            since: Start time filter
            until: End time filter
            strategy: Strategy name filter
            tool: Tool name filter
            min_quality: Minimum quality filter
            tags: Required context tags
            limit: Maximum results

        Returns:
            Matching interactions
        """
        results = []

        for record in self.iter_all():
            # Time filters
            if since and record.timestamp < since:
                continue
            if until and record.timestamp > until:
                continue

            # Strategy filter
            if strategy and record.chosen_strategy != strategy:
                continue

            # Tool filter
            if tool:
                tool_names = record.get_tool_names()
                if tool not in tool_names:
                    continue

            # Quality filter
            if min_quality is not None:
                if record.outcome_quality is None or record.outcome_quality < min_quality:
                    continue

            # Tags filter (all must match)
            if tags:
                if not all(t in record.context_tags for t in tags):
                    continue

            results.append(record)
            if len(results) >= limit:
                break

        return results

    def get_stats(self, days: int = 7) -> Dict[str, Any]:
        """Get summary statistics.

        Args:
            days: Number of days to analyze

        Returns:
            Statistics dictionary
        """
        since = datetime.utcnow() - timedelta(days=days)
        records = self.query(since=since, limit=10000)

        if not records:
            return {
                "total_interactions": 0,
                "period_days": days,
                "avg_quality": None,
                "tools_used": {},
                "strategies_used": {},
                "success_rate": None,
            }

        # Compute stats
        qualities = [r.outcome_quality for r in records if r.outcome_quality is not None]

        tool_counts: Dict[str, int] = {}
        strategy_counts: Dict[str, int] = {}
        total_calls = 0
        successful_calls = 0

        for record in records:
            # Strategy tracking
            if record.chosen_strategy:
                strategy_counts[record.chosen_strategy] = strategy_counts.get(record.chosen_strategy, 0) + 1

            # Tool tracking
            for tc in record.tools_used:
                tool_counts[tc.tool_name] = tool_counts.get(tc.tool_name, 0) + 1
                total_calls += 1
                if tc.success:
                    successful_calls += 1

        return {
            "total_interactions": len(records),
            "period_days": days,
            "avg_quality": sum(qualities) / len(qualities) if qualities else None,
            "tools_used": tool_counts,
            "strategies_used": strategy_counts,
            "success_rate": successful_calls / total_calls if total_calls > 0 else None,
        }

    def get_by_session(self, session_id: str) -> List[InteractionRecord]:
        """Get all interactions for a session.

        Args:
            session_id: The session ID

        Returns:
            Interactions in the session
        """
        return [r for r in self.iter_all() if r.session_id == session_id]

    def get_by_issue(self, issue_id: str) -> List[InteractionRecord]:
        """Get all interactions for an issue.

        Args:
            issue_id: The issue ID

        Returns:
            Interactions related to the issue
        """
        return [r for r in self.iter_all() if r.issue_id == issue_id]

    def clear(self) -> None:
        """Clear all logs (use with caution)."""
        if self.log_path.exists():
            self.log_path.unlink()
        self._recent_buffer.clear()
        logger.warning("Meta logs cleared")


# =============================================================================
# Convenience Functions
# =============================================================================

_default_logger: Optional[MetaLogger] = None


def get_meta_logger(path: Optional[Path] = None) -> MetaLogger:
    """Get the default meta logger."""
    global _default_logger
    if _default_logger is None:
        _default_logger = MetaLogger(log_path=path)
    return _default_logger


def log_interaction(record: InteractionRecord) -> bool:
    """Log an interaction using the default logger.

    Args:
        record: The interaction record

    Returns:
        True if logged successfully
    """
    return get_meta_logger().log_interaction(record)


def create_interaction(
    user_query: str,
    strategy: str = "",
    context_tags: Optional[List[str]] = None,
    session_id: Optional[str] = None,
    issue_id: Optional[str] = None,
) -> InteractionRecord:
    """Create a new interaction record.

    Convenience function for starting an interaction.

    Args:
        user_query: What the user asked
        strategy: The chosen strategy
        context_tags: Context classification
        session_id: Session ID
        issue_id: Related issue ID

    Returns:
        New InteractionRecord
    """
    return InteractionRecord(
        user_query=user_query,
        chosen_strategy=strategy,
        context_tags=context_tags or [],
        session_id=session_id,
        issue_id=issue_id,
    )
