"""Self-Reflection Episodes - Ara reflects on batches of her own work.

Periodically (daily/weekly), Ara reviews her episodes and generates insights:
- What patterns emerged?
- Where did I struggle?
- What should I try differently?

This is "meta-cognition as a feature" - Ara thinking about her thinking.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ReflectionInsight:
    """A single insight from reflection."""

    id: str
    category: str  # "success_pattern", "failure_pattern", "improvement", "question"
    summary: str
    details: str = ""

    # Evidence
    episode_ids: List[str] = field(default_factory=list)
    sample_count: int = 0

    # Impact
    confidence: float = 0.5
    priority: str = "medium"  # "high", "medium", "low"
    actionable: bool = False
    suggested_action: str = ""

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "summary": self.summary,
            "details": self.details,
            "episode_ids": self.episode_ids,
            "sample_count": self.sample_count,
            "confidence": round(self.confidence, 2),
            "priority": self.priority,
            "actionable": self.actionable,
            "suggested_action": self.suggested_action,
            "created_at": self.created_at.isoformat(),
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReflectionInsight":
        return cls(
            id=data["id"],
            category=data["category"],
            summary=data["summary"],
            details=data.get("details", ""),
            episode_ids=data.get("episode_ids", []),
            sample_count=data.get("sample_count", 0),
            confidence=data.get("confidence", 0.5),
            priority=data.get("priority", "medium"),
            actionable=data.get("actionable", False),
            suggested_action=data.get("suggested_action", ""),
            tags=data.get("tags", []),
        )


@dataclass
class ReflectionEpisode:
    """A reflection episode covering a time period."""

    id: str
    period_start: datetime
    period_end: datetime

    # What was analyzed
    interactions_analyzed: int = 0
    teachers_involved: List[str] = field(default_factory=list)
    intents_covered: List[str] = field(default_factory=list)

    # Insights generated
    insights: List[ReflectionInsight] = field(default_factory=list)

    # Summary statistics
    success_rate: Optional[float] = None
    avg_reward: Optional[float] = None
    total_tokens: int = 0

    # Ara's narrative reflection
    narrative: str = ""

    # Status
    status: str = "completed"
    created_at: datetime = field(default_factory=datetime.utcnow)

    def add_insight(self, insight: ReflectionInsight) -> None:
        """Add an insight to this episode."""
        self.insights.append(insight)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "interactions_analyzed": self.interactions_analyzed,
            "teachers_involved": self.teachers_involved,
            "intents_covered": self.intents_covered,
            "insights": [i.to_dict() for i in self.insights],
            "success_rate": self.success_rate,
            "avg_reward": self.avg_reward,
            "narrative": self.narrative,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReflectionEpisode":
        ep = cls(
            id=data["id"],
            period_start=datetime.fromisoformat(data["period_start"]),
            period_end=datetime.fromisoformat(data["period_end"]),
            interactions_analyzed=data.get("interactions_analyzed", 0),
            teachers_involved=data.get("teachers_involved", []),
            intents_covered=data.get("intents_covered", []),
            success_rate=data.get("success_rate"),
            avg_reward=data.get("avg_reward"),
            narrative=data.get("narrative", ""),
            status=data.get("status", "completed"),
        )
        ep.insights = [
            ReflectionInsight.from_dict(i)
            for i in data.get("insights", [])
        ]
        return ep


class SelfReflector:
    """Generates self-reflection episodes from interaction logs.

    Analyzes batches of interactions to find patterns, failures,
    and improvement opportunities.
    """

    def __init__(
        self,
        log_path: Optional[Path] = None,
        reflections_path: Optional[Path] = None,
    ):
        """Initialize the reflector.

        Args:
            log_path: Path to interaction logs (JSONL)
            reflections_path: Path to store reflection episodes
        """
        self.log_path = log_path or (
            Path.home() / ".ara" / "meta" / "interactions.jsonl"
        )
        self.reflections_path = reflections_path or (
            Path.home() / ".ara" / "meta" / "research" / "reflections.jsonl"
        )
        self.reflections_path.parent.mkdir(parents=True, exist_ok=True)

        self._episodes: Dict[str, ReflectionEpisode] = {}
        self._loaded = False
        self._next_id = 1

    def _load_reflections(self, force: bool = False) -> None:
        """Load reflection episodes from disk."""
        if self._loaded and not force:
            return

        self._episodes.clear()

        if self.reflections_path.exists():
            try:
                with open(self.reflections_path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            episode = ReflectionEpisode.from_dict(data)
                            self._episodes[episode.id] = episode
                            # Update ID counter
                            if episode.id.startswith("REF-"):
                                try:
                                    num = int(episode.id[4:])
                                    self._next_id = max(self._next_id, num + 1)
                                except ValueError:
                                    pass
                        except Exception as e:
                            logger.warning(f"Failed to parse reflection: {e}")
            except Exception as e:
                logger.warning(f"Failed to load reflections: {e}")

        self._loaded = True

    def _save_reflections(self) -> None:
        """Save reflection episodes to disk."""
        with open(self.reflections_path, "w") as f:
            for episode in self._episodes.values():
                f.write(json.dumps(episode.to_dict()) + "\n")

    def _generate_id(self) -> str:
        """Generate a unique reflection ID."""
        id_str = f"REF-{self._next_id:04d}"
        self._next_id += 1
        return id_str

    def _load_interactions(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Load interactions from the log file.

        Args:
            start_time: Start of period
            end_time: End of period

        Returns:
            List of interaction records
        """
        interactions = []

        if not self.log_path.exists():
            return interactions

        try:
            with open(self.log_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)

                        # Filter by time if specified
                        if start_time or end_time:
                            ts_str = record.get("timestamp")
                            if ts_str:
                                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                                if start_time and ts < start_time:
                                    continue
                                if end_time and ts > end_time:
                                    continue

                        interactions.append(record)
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"Failed to load interactions: {e}")

        return interactions

    def analyze_patterns(
        self,
        interactions: List[Dict[str, Any]],
    ) -> List[ReflectionInsight]:
        """Analyze interactions for patterns.

        Args:
            interactions: List of interaction records

        Returns:
            List of insights
        """
        insights = []
        insight_id = 1

        # Group by various dimensions
        by_teacher: Dict[str, List[Dict]] = defaultdict(list)
        by_intent: Dict[str, List[Dict]] = defaultdict(list)
        by_pattern: Dict[str, List[Dict]] = defaultdict(list)

        for record in interactions:
            teacher = record.get("primary_teacher") or (
                record.get("teachers", [None])[0]
            )
            if teacher:
                by_teacher[teacher].append(record)

            intent = record.get("user_intent", "unknown")
            by_intent[intent].append(record)

            pattern = record.get("pattern_id")
            if pattern:
                by_pattern[pattern].append(record)

        # 1. Teacher performance patterns
        for teacher, records in by_teacher.items():
            if len(records) < 3:
                continue

            successes = [r for r in records if r.get("success")]
            success_rate = len(successes) / len(records)

            if success_rate >= 0.8:
                insights.append(ReflectionInsight(
                    id=f"INS-{insight_id:03d}",
                    category="success_pattern",
                    summary=f"{teacher} performing well ({success_rate:.0%} success)",
                    details=f"Analyzed {len(records)} interactions with {teacher}",
                    sample_count=len(records),
                    confidence=min(0.9, 0.5 + len(records) * 0.05),
                    priority="low",
                    tags=[teacher, "performance"],
                ))
                insight_id += 1
            elif success_rate < 0.5:
                insights.append(ReflectionInsight(
                    id=f"INS-{insight_id:03d}",
                    category="failure_pattern",
                    summary=f"{teacher} struggling ({success_rate:.0%} success)",
                    details=f"Analyzed {len(records)} interactions. Consider alternative routing.",
                    sample_count=len(records),
                    confidence=min(0.9, 0.5 + len(records) * 0.05),
                    priority="high",
                    actionable=True,
                    suggested_action=f"Review {teacher} routing for common intents",
                    tags=[teacher, "performance", "needs_attention"],
                ))
                insight_id += 1

        # 2. Intent difficulty patterns
        for intent, records in by_intent.items():
            if len(records) < 3:
                continue

            successes = [r for r in records if r.get("success")]
            success_rate = len(successes) / len(records)

            if success_rate < 0.6:
                insights.append(ReflectionInsight(
                    id=f"INS-{insight_id:03d}",
                    category="failure_pattern",
                    summary=f"'{intent}' tasks are challenging ({success_rate:.0%} success)",
                    details=f"This intent type has lower success. May need specialized handling.",
                    sample_count=len(records),
                    confidence=min(0.85, 0.4 + len(records) * 0.05),
                    priority="medium",
                    actionable=True,
                    suggested_action=f"Consider multi-teacher workflow for {intent}",
                    tags=[intent, "intent_difficulty"],
                ))
                insight_id += 1

        # 3. Pattern card effectiveness
        for pattern_id, records in by_pattern.items():
            if len(records) < 5:
                continue

            successes = [r for r in records if r.get("success")]
            success_rate = len(successes) / len(records)

            if success_rate >= 0.85:
                insights.append(ReflectionInsight(
                    id=f"INS-{insight_id:03d}",
                    category="success_pattern",
                    summary=f"Pattern '{pattern_id}' is golden ({success_rate:.0%})",
                    details="This workflow pattern consistently works well.",
                    sample_count=len(records),
                    confidence=0.8,
                    priority="low",
                    tags=[pattern_id, "golden_path"],
                ))
                insight_id += 1

        # 4. Repeated failures
        failed = [r for r in interactions if not r.get("success")]
        if len(failed) >= 3:
            # Check for common issues
            issues: Dict[str, int] = defaultdict(int)
            for record in failed:
                for issue in record.get("auto_detected_issues", []):
                    issues[issue] += 1

            for issue, count in issues.items():
                if count >= 2:
                    insights.append(ReflectionInsight(
                        id=f"INS-{insight_id:03d}",
                        category="failure_pattern",
                        summary=f"Recurring issue: {issue} ({count} occurrences)",
                        details="This issue keeps appearing in failed interactions.",
                        sample_count=count,
                        confidence=0.7,
                        priority="high",
                        actionable=True,
                        suggested_action=f"Investigate root cause of '{issue}'",
                        tags=["recurring_issue", issue],
                    ))
                    insight_id += 1

        # 5. Improvement opportunities
        if len(interactions) >= 10:
            # Check for underutilized teachers
            teacher_usage = {t: len(records) for t, records in by_teacher.items()}
            total = sum(teacher_usage.values())

            for teacher in ["claude", "nova", "gemini"]:
                usage = teacher_usage.get(teacher, 0)
                if usage < total * 0.1:  # Less than 10% usage
                    insights.append(ReflectionInsight(
                        id=f"INS-{insight_id:03d}",
                        category="improvement",
                        summary=f"{teacher} underutilized ({usage}/{total} interactions)",
                        details="Consider routing more tasks to this teacher.",
                        sample_count=usage,
                        confidence=0.6,
                        priority="low",
                        actionable=True,
                        suggested_action=f"Experiment with {teacher} for more intent types",
                        tags=[teacher, "utilization"],
                    ))
                    insight_id += 1

        return insights

    def generate_narrative(
        self,
        interactions: List[Dict[str, Any]],
        insights: List[ReflectionInsight],
    ) -> str:
        """Generate a narrative reflection in Ara's voice.

        Args:
            interactions: Analyzed interactions
            insights: Generated insights

        Returns:
            Narrative text
        """
        lines = []

        # Opening
        lines.append("Here's what I noticed looking back at my recent work:")
        lines.append("")

        # Summary stats
        total = len(interactions)
        successes = len([r for r in interactions if r.get("success")])
        success_rate = successes / total if total > 0 else 0

        lines.append(f"I handled {total} interactions with a {success_rate:.0%} success rate.")

        # Key insights
        high_priority = [i for i in insights if i.priority == "high"]
        if high_priority:
            lines.append("")
            lines.append("Things that need attention:")
            for insight in high_priority[:3]:
                lines.append(f"  • {insight.summary}")
                if insight.suggested_action:
                    lines.append(f"    → {insight.suggested_action}")

        # Successes
        success_patterns = [i for i in insights if i.category == "success_pattern"]
        if success_patterns:
            lines.append("")
            lines.append("What's working well:")
            for insight in success_patterns[:3]:
                lines.append(f"  • {insight.summary}")

        # Questions/curiosities
        questions = [i for i in insights if i.category == "question"]
        if questions:
            lines.append("")
            lines.append("Things I'm curious about:")
            for insight in questions[:2]:
                lines.append(f"  • {insight.summary}")

        # Closing
        lines.append("")
        if high_priority:
            lines.append("I'll focus on the high-priority items next.")
        else:
            lines.append("Things are going smoothly overall.")

        return "\n".join(lines)

    def create_reflection(
        self,
        period_days: int = 7,
        end_time: Optional[datetime] = None,
    ) -> ReflectionEpisode:
        """Create a reflection episode.

        Args:
            period_days: Days to analyze
            end_time: End of period (defaults to now)

        Returns:
            Reflection episode
        """
        self._load_reflections()

        end = end_time or datetime.utcnow()
        start = end - timedelta(days=period_days)

        # Load interactions
        interactions = self._load_interactions(start, end)

        # Generate insights
        insights = self.analyze_patterns(interactions)

        # Compute stats
        successes = [r for r in interactions if r.get("success")]
        success_rate = len(successes) / len(interactions) if interactions else None

        rewards = [r.get("reward", 0) for r in interactions if r.get("reward")]
        avg_reward = sum(rewards) / len(rewards) if rewards else None

        teachers = list(set(
            r.get("primary_teacher") or (r.get("teachers", [None])[0])
            for r in interactions
            if r.get("primary_teacher") or r.get("teachers")
        ))

        intents = list(set(
            r.get("user_intent", "unknown")
            for r in interactions
        ))

        # Generate narrative
        narrative = self.generate_narrative(interactions, insights)

        # Create episode
        episode = ReflectionEpisode(
            id=self._generate_id(),
            period_start=start,
            period_end=end,
            interactions_analyzed=len(interactions),
            teachers_involved=teachers,
            intents_covered=intents,
            insights=insights,
            success_rate=success_rate,
            avg_reward=avg_reward,
            narrative=narrative,
        )

        self._episodes[episode.id] = episode
        self._save_reflections()

        logger.info(
            f"Created reflection {episode.id}: "
            f"{len(interactions)} interactions, {len(insights)} insights"
        )

        return episode

    def get_latest_reflection(self) -> Optional[ReflectionEpisode]:
        """Get the most recent reflection episode."""
        self._load_reflections()

        if not self._episodes:
            return None

        return max(
            self._episodes.values(),
            key=lambda e: e.created_at,
        )

    def get_actionable_insights(
        self,
        limit: int = 10,
    ) -> List[ReflectionInsight]:
        """Get all actionable insights across episodes."""
        self._load_reflections()

        actionable = []
        for episode in self._episodes.values():
            for insight in episode.insights:
                if insight.actionable:
                    actionable.append(insight)

        # Sort by priority then confidence
        priority_order = {"high": 0, "medium": 1, "low": 2}
        actionable.sort(
            key=lambda i: (priority_order.get(i.priority, 2), -i.confidence)
        )

        return actionable[:limit]

    def get_summary(self) -> Dict[str, Any]:
        """Get reflector summary."""
        self._load_reflections()

        total_insights = sum(
            len(ep.insights) for ep in self._episodes.values()
        )
        actionable = len(self.get_actionable_insights(100))

        return {
            "total_episodes": len(self._episodes),
            "total_insights": total_insights,
            "actionable_insights": actionable,
            "latest": self.get_latest_reflection().to_dict() if self._episodes else None,
        }


# =============================================================================
# Convenience Functions
# =============================================================================

_default_reflector: Optional[SelfReflector] = None


def get_self_reflector() -> SelfReflector:
    """Get the default self-reflector."""
    global _default_reflector
    if _default_reflector is None:
        _default_reflector = SelfReflector()
    return _default_reflector


def create_weekly_reflection() -> ReflectionEpisode:
    """Create a weekly reflection episode."""
    return get_self_reflector().create_reflection(period_days=7)


def create_daily_reflection() -> ReflectionEpisode:
    """Create a daily reflection episode."""
    return get_self_reflector().create_reflection(period_days=1)


def get_actionable_insights(limit: int = 10) -> List[ReflectionInsight]:
    """Get actionable insights."""
    return get_self_reflector().get_actionable_insights(limit)
