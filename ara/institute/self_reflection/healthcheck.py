"""Skill Health Check - Ara monitors her learned skills.

Like a doctor checking vitals, this module monitors the health of
Ara's internalized skills:
- Are they still being used?
- Are they succeeding?
- Have they drifted from their training data?
- Should they be deprecated or retrained?

A healthy skill ecosystem means knowing when to retire old skills
and when to invest in improving struggling ones.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


class SkillHealth(Enum):
    """Health status of a skill."""
    HEALTHY = "healthy"           # Working well
    DEGRADED = "degraded"         # Performance declining
    DORMANT = "dormant"           # Not being used
    FAILING = "failing"           # Low success rate
    UNKNOWN = "unknown"           # Insufficient data


class RecommendedAction(Enum):
    """Recommended action for a skill."""
    NONE = "none"                 # No action needed
    MONITOR = "monitor"           # Watch closely
    RETRAIN = "retrain"           # Needs retraining
    DEPRECATE = "deprecate"       # Consider retiring
    INVESTIGATE = "investigate"   # Needs analysis


@dataclass
class SkillMetrics:
    """Metrics for a single skill."""

    skill_id: str
    skill_name: str

    # Usage
    total_invocations: int = 0
    invocations_last_7_days: int = 0
    invocations_last_30_days: int = 0
    last_invoked: Optional[datetime] = None

    # Performance
    success_rate: float = 0.0
    success_rate_last_7_days: float = 0.0
    avg_latency_ms: float = 0.0
    error_count: int = 0

    # Trends
    usage_trend: str = "stable"  # "increasing", "stable", "decreasing"
    performance_trend: str = "stable"

    # Health
    health: SkillHealth = SkillHealth.UNKNOWN
    recommended_action: RecommendedAction = RecommendedAction.NONE
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "skill_name": self.skill_name,
            "total_invocations": self.total_invocations,
            "invocations_last_7_days": self.invocations_last_7_days,
            "invocations_last_30_days": self.invocations_last_30_days,
            "last_invoked": self.last_invoked.isoformat() if self.last_invoked else None,
            "success_rate": round(self.success_rate, 3),
            "success_rate_last_7_days": round(self.success_rate_last_7_days, 3),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "error_count": self.error_count,
            "usage_trend": self.usage_trend,
            "performance_trend": self.performance_trend,
            "health": self.health.value,
            "recommended_action": self.recommended_action.value,
            "notes": self.notes,
        }


@dataclass
class HealthCheckReport:
    """Complete health check report for all skills."""

    id: str
    generated_at: datetime = field(default_factory=datetime.utcnow)

    # Skill metrics
    skill_metrics: List[SkillMetrics] = field(default_factory=list)

    # Summary counts
    total_skills: int = 0
    healthy_skills: int = 0
    degraded_skills: int = 0
    dormant_skills: int = 0
    failing_skills: int = 0

    # Overall stats
    total_invocations: int = 0
    overall_success_rate: float = 0.0

    # Actionable items
    skills_to_retrain: List[str] = field(default_factory=list)
    skills_to_deprecate: List[str] = field(default_factory=list)
    skills_to_investigate: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "generated_at": self.generated_at.isoformat(),
            "skill_metrics": [m.to_dict() for m in self.skill_metrics],
            "total_skills": self.total_skills,
            "healthy_skills": self.healthy_skills,
            "degraded_skills": self.degraded_skills,
            "dormant_skills": self.dormant_skills,
            "failing_skills": self.failing_skills,
            "total_invocations": self.total_invocations,
            "overall_success_rate": round(self.overall_success_rate, 3),
            "skills_to_retrain": self.skills_to_retrain,
            "skills_to_deprecate": self.skills_to_deprecate,
            "skills_to_investigate": self.skills_to_investigate,
        }

    def to_markdown(self) -> str:
        """Generate a markdown report."""
        lines = [
            "# Ara Skill Health Check",
            "",
            f"Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M')}",
            "",
            "## Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Skills | {self.total_skills} |",
            f"| Healthy | {self.healthy_skills} |",
            f"| Degraded | {self.degraded_skills} |",
            f"| Dormant | {self.dormant_skills} |",
            f"| Failing | {self.failing_skills} |",
            f"| Overall Success Rate | {self.overall_success_rate:.1%} |",
            "",
        ]

        # Skills needing attention
        if self.skills_to_retrain or self.skills_to_deprecate or self.skills_to_investigate:
            lines.extend([
                "## Action Items",
                "",
            ])

            if self.skills_to_retrain:
                lines.append("### Skills to Retrain")
                lines.extend([f"- {s}" for s in self.skills_to_retrain])
                lines.append("")

            if self.skills_to_deprecate:
                lines.append("### Skills to Consider Deprecating")
                lines.extend([f"- {s}" for s in self.skills_to_deprecate])
                lines.append("")

            if self.skills_to_investigate:
                lines.append("### Skills to Investigate")
                lines.extend([f"- {s}" for s in self.skills_to_investigate])
                lines.append("")

        # Individual skill details
        if self.skill_metrics:
            lines.extend([
                "## Skill Details",
                "",
                "| Skill | Health | Uses (7d) | Success Rate | Action |",
                "|-------|--------|-----------|--------------|--------|",
            ])

            for m in sorted(self.skill_metrics, key=lambda x: x.health.value):
                lines.append(
                    f"| {m.skill_name} | {m.health.value} | {m.invocations_last_7_days} | "
                    f"{m.success_rate:.0%} | {m.recommended_action.value} |"
                )

        return "\n".join(lines)


class SkillHealthChecker:
    """Monitors and reports on skill health."""

    def __init__(self, data_path: Optional[Path] = None):
        """Initialize the health checker.

        Args:
            data_path: Path to health check data
        """
        self.data_path = data_path or (
            Path.home() / ".ara" / "institute" / "health_checks"
        )
        self.data_path.mkdir(parents=True, exist_ok=True)

        self._reports: Dict[str, HealthCheckReport] = {}
        self._loaded = False
        self._next_id = 1

        # Thresholds
        self.dormant_days = 14  # Days without use to be dormant
        self.failing_threshold = 0.5  # Success rate below this = failing
        self.degraded_threshold = 0.7  # Below this = degraded

    def _load(self, force: bool = False) -> None:
        """Load data from disk."""
        if self._loaded and not force:
            return

        reports_file = self.data_path / "reports.json"
        if reports_file.exists():
            try:
                with open(reports_file) as f:
                    data = json.load(f)
                self._next_id = data.get("next_id", 1)
            except Exception as e:
                logger.warning(f"Failed to load health reports: {e}")

        self._loaded = True

    def _save(self) -> None:
        """Save data to disk."""
        data = {
            "version": 1,
            "updated_at": datetime.utcnow().isoformat(),
            "next_id": self._next_id,
            "reports": [r.to_dict() for r in self._reports.values()],
        }
        with open(self.data_path / "reports.json", "w") as f:
            json.dump(data, f, indent=2)

    def _generate_id(self) -> str:
        """Generate unique report ID."""
        id_str = f"HC-{self._next_id:04d}"
        self._next_id += 1
        return id_str

    def evaluate_skill_health(
        self,
        skill_id: str,
        skill_name: str,
        total_invocations: int,
        invocations_7d: int,
        invocations_30d: int,
        successes: int,
        successes_7d: int,
        last_invoked: Optional[datetime] = None,
        avg_latency_ms: float = 0.0,
    ) -> SkillMetrics:
        """Evaluate health of a single skill.

        Args:
            skill_id: Skill identifier
            skill_name: Human-readable name
            total_invocations: All-time invocation count
            invocations_7d: Invocations in last 7 days
            invocations_30d: Invocations in last 30 days
            successes: All-time success count
            successes_7d: Successes in last 7 days
            last_invoked: When skill was last used
            avg_latency_ms: Average response time

        Returns:
            Skill metrics with health assessment
        """
        metrics = SkillMetrics(
            skill_id=skill_id,
            skill_name=skill_name,
            total_invocations=total_invocations,
            invocations_last_7_days=invocations_7d,
            invocations_last_30_days=invocations_30d,
            last_invoked=last_invoked,
            avg_latency_ms=avg_latency_ms,
        )

        # Calculate success rates
        if total_invocations > 0:
            metrics.success_rate = successes / total_invocations
        if invocations_7d > 0:
            metrics.success_rate_last_7_days = successes_7d / invocations_7d

        # Determine usage trend
        if invocations_30d > 0:
            weekly_avg = invocations_30d / 4
            if invocations_7d > weekly_avg * 1.5:
                metrics.usage_trend = "increasing"
            elif invocations_7d < weekly_avg * 0.5:
                metrics.usage_trend = "decreasing"

        # Determine performance trend
        if total_invocations > 10 and invocations_7d > 3:
            if metrics.success_rate_last_7_days > metrics.success_rate + 0.1:
                metrics.performance_trend = "improving"
            elif metrics.success_rate_last_7_days < metrics.success_rate - 0.1:
                metrics.performance_trend = "declining"

        # Evaluate health
        metrics.health = self._evaluate_health(metrics)
        metrics.recommended_action = self._recommend_action(metrics)

        return metrics

    def _evaluate_health(self, metrics: SkillMetrics) -> SkillHealth:
        """Determine health status from metrics."""
        # Check for dormancy
        if metrics.last_invoked:
            days_since_use = (datetime.utcnow() - metrics.last_invoked).days
            if days_since_use > self.dormant_days:
                return SkillHealth.DORMANT

        # Check for insufficient data
        if metrics.total_invocations < 5:
            return SkillHealth.UNKNOWN

        # Check success rate
        if metrics.success_rate < self.failing_threshold:
            return SkillHealth.FAILING

        if metrics.success_rate < self.degraded_threshold:
            return SkillHealth.DEGRADED

        # Check for declining performance
        if (metrics.performance_trend == "declining" and
                metrics.success_rate_last_7_days < self.degraded_threshold):
            return SkillHealth.DEGRADED

        return SkillHealth.HEALTHY

    def _recommend_action(self, metrics: SkillMetrics) -> RecommendedAction:
        """Recommend action based on health."""
        if metrics.health == SkillHealth.FAILING:
            # Failing but still used -> retrain
            if metrics.invocations_last_7_days > 0:
                return RecommendedAction.RETRAIN
            else:
                return RecommendedAction.DEPRECATE

        if metrics.health == SkillHealth.DORMANT:
            # Long dormant -> consider deprecating
            if metrics.total_invocations < 10:
                return RecommendedAction.DEPRECATE
            else:
                return RecommendedAction.INVESTIGATE

        if metrics.health == SkillHealth.DEGRADED:
            if metrics.performance_trend == "declining":
                return RecommendedAction.RETRAIN
            else:
                return RecommendedAction.MONITOR

        if metrics.health == SkillHealth.UNKNOWN:
            return RecommendedAction.MONITOR

        return RecommendedAction.NONE

    def run_health_check(
        self,
        skills_data: List[Dict[str, Any]],
    ) -> HealthCheckReport:
        """Run a complete health check on all skills.

        Args:
            skills_data: List of skill data dicts with:
                - id, name
                - total_invocations, invocations_7d, invocations_30d
                - successes, successes_7d
                - last_invoked (optional)

        Returns:
            Health check report
        """
        self._load()

        report = HealthCheckReport(
            id=self._generate_id(),
            total_skills=len(skills_data),
        )

        for skill_data in skills_data:
            metrics = self.evaluate_skill_health(
                skill_id=skill_data["id"],
                skill_name=skill_data.get("name", skill_data["id"]),
                total_invocations=skill_data.get("total_invocations", 0),
                invocations_7d=skill_data.get("invocations_7d", 0),
                invocations_30d=skill_data.get("invocations_30d", 0),
                successes=skill_data.get("successes", 0),
                successes_7d=skill_data.get("successes_7d", 0),
                last_invoked=skill_data.get("last_invoked"),
                avg_latency_ms=skill_data.get("avg_latency_ms", 0.0),
            )

            report.skill_metrics.append(metrics)
            report.total_invocations += metrics.total_invocations

            # Count by health
            if metrics.health == SkillHealth.HEALTHY:
                report.healthy_skills += 1
            elif metrics.health == SkillHealth.DEGRADED:
                report.degraded_skills += 1
            elif metrics.health == SkillHealth.DORMANT:
                report.dormant_skills += 1
            elif metrics.health == SkillHealth.FAILING:
                report.failing_skills += 1

            # Track action items
            if metrics.recommended_action == RecommendedAction.RETRAIN:
                report.skills_to_retrain.append(metrics.skill_name)
            elif metrics.recommended_action == RecommendedAction.DEPRECATE:
                report.skills_to_deprecate.append(metrics.skill_name)
            elif metrics.recommended_action == RecommendedAction.INVESTIGATE:
                report.skills_to_investigate.append(metrics.skill_name)

        # Calculate overall success rate
        total_successes = sum(
            m.success_rate * m.total_invocations
            for m in report.skill_metrics
        )
        if report.total_invocations > 0:
            report.overall_success_rate = total_successes / report.total_invocations

        self._reports[report.id] = report
        self._save()

        logger.info(f"Health check complete: {report.healthy_skills}/{report.total_skills} healthy")
        return report

    def get_report(self, report_id: str) -> Optional[HealthCheckReport]:
        """Get a report by ID."""
        self._load()
        return self._reports.get(report_id)

    def get_latest_report(self) -> Optional[HealthCheckReport]:
        """Get the most recent report."""
        self._load()
        if not self._reports:
            return None
        return max(self._reports.values(), key=lambda r: r.generated_at)


# =============================================================================
# Convenience Functions
# =============================================================================

_default_checker: Optional[SkillHealthChecker] = None


def get_skill_health_checker() -> SkillHealthChecker:
    """Get the default health checker."""
    global _default_checker
    if _default_checker is None:
        _default_checker = SkillHealthChecker()
    return _default_checker


def check_skill_health(skills_data: List[Dict[str, Any]]) -> HealthCheckReport:
    """Quick health check on skills."""
    return get_skill_health_checker().run_health_check(skills_data)
