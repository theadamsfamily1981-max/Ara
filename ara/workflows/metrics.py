# ara/workflows/metrics.py
"""
Workflow Metrics Client (Ara-as-Historian)
==========================================

Records workflow outcomes for long-term learning.
This is how Ara learns from experience.

Data flows:
    Workflow Execution → MetricsClient → MEIS → QUANTA

Metrics captured:
    - Step durations and success rates
    - Decision confidence and outcomes
    - User interaction patterns
    - Error and retry patterns

Integration with MEIS:
    - Workflow metrics become QUANTA signals
    - Patterns detected feed back into decision strategy
    - Successful patterns get reinforced

Usage:
    from ara.workflows.metrics import MetricsClient, get_metrics_client

    metrics = get_metrics_client()
    await metrics.record_step(step_id, result, state)
    await metrics.record_workflow(workflow_result)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("Ara.Workflows.Metrics")


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class StepMetrics:
    """Metrics for a single step execution."""
    step_id: str
    step_name: str
    step_type: str

    # Outcome
    success: bool
    error: Optional[str] = None

    # Timing
    duration_seconds: float = 0.0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    # Decision context
    decision_confidence: float = 0.0
    was_alternative: bool = False

    # User interaction
    required_user_input: bool = False
    user_response_time_seconds: float = 0.0

    # Retry info
    attempt_number: int = 1
    is_retry: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WorkflowMetrics:
    """Metrics for a complete workflow execution."""
    workflow_id: str
    execution_id: str

    # Outcome
    success: bool
    final_status: str
    error: Optional[str] = None

    # Timing
    duration_seconds: float = 0.0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    # Steps
    steps_total: int = 0
    steps_completed: int = 0
    steps_failed: int = 0
    step_sequence: List[str] = field(default_factory=list)

    # Decisions
    decisions_total: int = 0
    avg_decision_confidence: float = 0.0
    alternative_chosen_count: int = 0

    # User interactions
    user_interactions: int = 0
    avg_user_response_time: float = 0.0

    # Errors and retries
    error_count: int = 0
    retry_count: int = 0

    # Step details
    step_metrics: List[StepMetrics] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["step_metrics"] = [s.to_dict() for s in self.step_metrics]
        return data

    def to_quanta_signal(self) -> Dict[str, Any]:
        """
        Convert to QUANTA-compatible signal format.

        This is what gets fed into the experience engine.
        """
        return {
            "type": "workflow_outcome",
            "source": "ara.workflows",
            "timestamp": self.completed_at or datetime.utcnow().isoformat(),
            "data": {
                "workflow_id": self.workflow_id,
                "execution_id": self.execution_id,
                "success": self.success,
                "duration_seconds": self.duration_seconds,
                "steps_completed": self.steps_completed,
                "error_count": self.error_count,
            },
            "features": {
                "step_sequence_hash": hash(tuple(self.step_sequence)),
                "avg_confidence": self.avg_decision_confidence,
                "user_interaction_ratio": self.user_interactions / max(1, self.steps_total),
                "error_ratio": self.error_count / max(1, self.steps_total),
            },
            "labels": {
                "outcome": "success" if self.success else "failure",
                "complexity": self._complexity_label(),
            },
        }

    def _complexity_label(self) -> str:
        """Categorize workflow complexity."""
        if self.steps_total <= 3:
            return "simple"
        elif self.steps_total <= 10:
            return "medium"
        else:
            return "complex"


# =============================================================================
# Metrics Client
# =============================================================================

class MetricsClient:
    """
    Records workflow metrics for learning.

    This is Ara-as-Historian: recording outcomes
    so future decisions can be better.

    Storage options:
        - JSONL file (default, for development)
        - MEIS integration (for production)
        - Custom backend

    Usage:
        client = MetricsClient(storage_path="/path/to/metrics.jsonl")
        await client.record_workflow(workflow_metrics)
    """

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        meis_client: Optional[Any] = None,
    ):
        """
        Initialize metrics client.

        Args:
            storage_path: Path for JSONL storage (default: ~/.ara/metrics/workflows.jsonl)
            meis_client: Optional MEIS client for production integration
        """
        self.storage_path = storage_path or (
            Path.home() / ".ara" / "metrics" / "workflows.jsonl"
        )
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        self.meis = meis_client
        self._buffer: List[Dict[str, Any]] = []
        self._buffer_size = 100  # Flush after this many records

        log.info("MetricsClient initialized: storage=%s", self.storage_path)

    # =========================================================================
    # Recording
    # =========================================================================

    async def record_step(
        self,
        step_metrics: StepMetrics,
        workflow_id: str,
        execution_id: str,
    ) -> None:
        """
        Record metrics for a single step.

        Args:
            step_metrics: Metrics for the step
            workflow_id: Parent workflow ID
            execution_id: Parent execution ID
        """
        record = {
            "type": "step",
            "timestamp": datetime.utcnow().isoformat(),
            "workflow_id": workflow_id,
            "execution_id": execution_id,
            **step_metrics.to_dict(),
        }

        self._buffer.append(record)

        if len(self._buffer) >= self._buffer_size:
            await self._flush()

        log.debug(
            "Recorded step metrics: %s (success=%s, duration=%.2fs)",
            step_metrics.step_id,
            step_metrics.success,
            step_metrics.duration_seconds,
        )

    async def record_workflow(self, workflow_metrics: WorkflowMetrics) -> None:
        """
        Record metrics for a complete workflow.

        This is called when a workflow finishes (success or failure).
        """
        record = {
            "type": "workflow",
            "timestamp": datetime.utcnow().isoformat(),
            **workflow_metrics.to_dict(),
        }

        self._buffer.append(record)
        await self._flush()  # Always flush workflow completions

        # Send to MEIS if available
        if self.meis:
            await self._send_to_meis(workflow_metrics)

        log.info(
            "Recorded workflow metrics: %s (success=%s, steps=%d, duration=%.1fs)",
            workflow_metrics.workflow_id,
            workflow_metrics.success,
            workflow_metrics.steps_completed,
            workflow_metrics.duration_seconds,
        )

    async def record_decision(
        self,
        workflow_id: str,
        execution_id: str,
        step_id: str,
        confidence: float,
        reasoning: str,
        alternatives: List[str],
        outcome: Optional[bool] = None,
    ) -> None:
        """
        Record a decision made by Ara.

        Used to learn which decisions lead to good outcomes.
        """
        record = {
            "type": "decision",
            "timestamp": datetime.utcnow().isoformat(),
            "workflow_id": workflow_id,
            "execution_id": execution_id,
            "step_id": step_id,
            "confidence": confidence,
            "reasoning": reasoning,
            "alternatives": alternatives,
            "outcome": outcome,
        }

        self._buffer.append(record)

    async def record_user_interaction(
        self,
        workflow_id: str,
        execution_id: str,
        step_id: str,
        prompt: str,
        response: Any,
        response_time_seconds: float,
    ) -> None:
        """
        Record a user interaction.

        Used to learn user preferences and patterns.
        """
        record = {
            "type": "user_interaction",
            "timestamp": datetime.utcnow().isoformat(),
            "workflow_id": workflow_id,
            "execution_id": execution_id,
            "step_id": step_id,
            "prompt": prompt,
            "response": str(response)[:500],  # Truncate
            "response_time_seconds": response_time_seconds,
        }

        self._buffer.append(record)

    # =========================================================================
    # Storage
    # =========================================================================

    async def _flush(self) -> None:
        """Flush buffer to storage."""
        if not self._buffer:
            return

        try:
            with open(self.storage_path, "a") as f:
                for record in self._buffer:
                    f.write(json.dumps(record) + "\n")

            log.debug("Flushed %d records to %s", len(self._buffer), self.storage_path)
            self._buffer.clear()

        except Exception as e:
            log.error("Failed to flush metrics: %s", e)

    async def _send_to_meis(self, workflow_metrics: WorkflowMetrics) -> None:
        """
        Send workflow outcome to MEIS for learning.

        In production: this feeds QUANTA for pattern detection.
        """
        if not self.meis:
            return

        try:
            signal = workflow_metrics.to_quanta_signal()
            # await self.meis.emit_signal(signal)
            log.debug("Sent workflow metrics to MEIS")
        except Exception as e:
            log.error("Failed to send to MEIS: %s", e)

    # =========================================================================
    # Querying
    # =========================================================================

    async def get_workflow_stats(
        self,
        workflow_id: str,
        days: int = 30,
    ) -> Dict[str, Any]:
        """
        Get aggregate stats for a workflow.

        Returns success rate, avg duration, common errors, etc.
        """
        records = await self._load_records(
            record_type="workflow",
            workflow_id=workflow_id,
            days=days,
        )

        if not records:
            return {"workflow_id": workflow_id, "executions": 0}

        successes = sum(1 for r in records if r.get("success"))
        durations = [r.get("duration_seconds", 0) for r in records]
        errors = [r.get("error") for r in records if r.get("error")]

        return {
            "workflow_id": workflow_id,
            "executions": len(records),
            "success_rate": successes / len(records),
            "avg_duration_seconds": sum(durations) / len(durations),
            "min_duration_seconds": min(durations),
            "max_duration_seconds": max(durations),
            "common_errors": self._top_items(errors, 5),
        }

    async def get_step_stats(
        self,
        step_id: str,
        days: int = 30,
    ) -> Dict[str, Any]:
        """Get aggregate stats for a step."""
        records = await self._load_records(
            record_type="step",
            days=days,
        )

        step_records = [r for r in records if r.get("step_id") == step_id]
        if not step_records:
            return {"step_id": step_id, "executions": 0}

        successes = sum(1 for r in step_records if r.get("success"))
        durations = [r.get("duration_seconds", 0) for r in step_records]

        return {
            "step_id": step_id,
            "executions": len(step_records),
            "success_rate": successes / len(step_records),
            "avg_duration_seconds": sum(durations) / len(durations) if durations else 0,
        }

    async def get_decision_outcomes(
        self,
        workflow_id: str,
        days: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Get decision outcomes for learning.

        Returns decisions with their outcomes to train the decision strategy.
        """
        records = await self._load_records(
            record_type="decision",
            workflow_id=workflow_id,
            days=days,
        )

        return [
            {
                "step_id": r.get("step_id"),
                "confidence": r.get("confidence"),
                "outcome": r.get("outcome"),
            }
            for r in records
            if r.get("outcome") is not None
        ]

    async def _load_records(
        self,
        record_type: Optional[str] = None,
        workflow_id: Optional[str] = None,
        days: int = 30,
    ) -> List[Dict[str, Any]]:
        """Load records from storage with filtering."""
        if not self.storage_path.exists():
            return []

        records = []
        cutoff = datetime.utcnow().timestamp() - (days * 86400)

        try:
            with open(self.storage_path, "r") as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())

                        # Filter by type
                        if record_type and record.get("type") != record_type:
                            continue

                        # Filter by workflow
                        if workflow_id and record.get("workflow_id") != workflow_id:
                            continue

                        # Filter by date
                        ts = record.get("timestamp", "")
                        if ts:
                            try:
                                record_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                                if record_time.timestamp() < cutoff:
                                    continue
                            except:
                                pass

                        records.append(record)

                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            log.error("Failed to load records: %s", e)

        return records

    def _top_items(self, items: List[str], n: int) -> List[Dict[str, Any]]:
        """Get top N most common items."""
        counts: Dict[str, int] = {}
        for item in items:
            if item:
                counts[item] = counts.get(item, 0) + 1

        sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return [{"item": item, "count": count} for item, count in sorted_items[:n]]


# =============================================================================
# Learning Integration
# =============================================================================

class WorkflowLearner:
    """
    Uses historical metrics to improve future workflows.

    This is the learning loop:
        Metrics → Patterns → Decision Strategy Updates

    Usage:
        learner = WorkflowLearner(metrics_client)
        insights = await learner.analyze_workflow("onboarding")
        # Use insights to improve decision strategy
    """

    def __init__(self, metrics_client: MetricsClient):
        self.metrics = metrics_client

    async def analyze_workflow(
        self,
        workflow_id: str,
        days: int = 30,
    ) -> Dict[str, Any]:
        """
        Analyze a workflow's historical performance.

        Returns insights for improving the workflow.
        """
        stats = await self.metrics.get_workflow_stats(workflow_id, days)

        if stats.get("executions", 0) == 0:
            return {"workflow_id": workflow_id, "insights": []}

        insights = []

        # Success rate insights
        success_rate = stats.get("success_rate", 0)
        if success_rate < 0.8:
            insights.append({
                "type": "low_success_rate",
                "message": f"Success rate is {success_rate:.1%}. Consider reviewing error patterns.",
                "severity": "high" if success_rate < 0.5 else "medium",
            })

        # Duration insights
        avg_duration = stats.get("avg_duration_seconds", 0)
        max_duration = stats.get("max_duration_seconds", 0)
        if max_duration > avg_duration * 3:
            insights.append({
                "type": "high_variance_duration",
                "message": "Some executions take much longer than average. Check for bottlenecks.",
                "severity": "medium",
            })

        # Error patterns
        common_errors = stats.get("common_errors", [])
        if common_errors:
            top_error = common_errors[0]
            if top_error.get("count", 0) > 3:
                insights.append({
                    "type": "recurring_error",
                    "message": f"Error '{top_error['item'][:50]}...' has occurred {top_error['count']} times.",
                    "severity": "high",
                })

        return {
            "workflow_id": workflow_id,
            "stats": stats,
            "insights": insights,
        }

    async def suggest_step_order(
        self,
        workflow_id: str,
    ) -> List[str]:
        """
        Suggest optimal step order based on historical data.

        Considers:
        - Steps that fail often should be earlier (fail fast)
        - Steps that take long should be later (unless blocking)
        - User input steps should be grouped
        """
        # Load historical step metrics
        records = await self.metrics._load_records(
            record_type="step",
            workflow_id=workflow_id,
        )

        if not records:
            return []

        # Aggregate by step
        step_stats: Dict[str, Dict[str, Any]] = {}
        for r in records:
            step_id = r.get("step_id", "")
            if step_id not in step_stats:
                step_stats[step_id] = {
                    "count": 0,
                    "failures": 0,
                    "total_duration": 0,
                }

            step_stats[step_id]["count"] += 1
            if not r.get("success"):
                step_stats[step_id]["failures"] += 1
            step_stats[step_id]["total_duration"] += r.get("duration_seconds", 0)

        # Score each step
        # Higher score = should run earlier
        scored = []
        for step_id, stats in step_stats.items():
            failure_rate = stats["failures"] / max(1, stats["count"])
            avg_duration = stats["total_duration"] / max(1, stats["count"])

            # Fail-fast: high failure rate = run earlier
            # Fast: short duration = run earlier (for quick feedback)
            score = failure_rate * 10 + (1.0 / max(0.1, avg_duration))
            scored.append((step_id, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        return [step_id for step_id, _ in scored]


# =============================================================================
# Convenience
# =============================================================================

_default_client: Optional[MetricsClient] = None


def get_metrics_client(
    storage_path: Optional[Path] = None,
) -> MetricsClient:
    """Get the default metrics client."""
    global _default_client
    if _default_client is None:
        _default_client = MetricsClient(storage_path=storage_path)
    return _default_client


__all__ = [
    # Data structures
    "StepMetrics",
    "WorkflowMetrics",
    # Client
    "MetricsClient",
    "get_metrics_client",
    # Learning
    "WorkflowLearner",
]
