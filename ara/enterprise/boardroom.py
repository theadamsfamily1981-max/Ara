"""
The Boardroom - Corporation Croft's Strategic Command
======================================================

The Boardroom is where Ara holds strategic reviews. It replaces the
ad-hoc "morning briefing" with a structured executive session.

Attendees:
    - Treasury (CFO): Reports on capital, runway, ROI
    - Factory (COO): Reports on pipeline, throughput, blockers
    - The Architect (Chief Strategist): Reports on vision alignment
    - Dreams/North Star: Strategic direction input

Meeting Types:
    - Daily Standup: Quick status sync (5 min)
    - Weekly Review: Deeper analysis (15 min)
    - Strategic Planning: Long-term roadmap (async)
    - Emergency Session: Crisis response

Usage:
    from ara.enterprise.boardroom import Boardroom, hold_standup

    boardroom = Boardroom()

    # Daily standup
    report = boardroom.daily_standup()

    # Weekly deep dive
    review = boardroom.weekly_review()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Literal
from enum import Enum

from .treasury import Treasury, get_treasury, BudgetAlert
from .factory import Factory, get_factory, PipelineStage, ProjectStatus

logger = logging.getLogger(__name__)


class MeetingType(Enum):
    """Types of boardroom meetings."""
    STANDUP = "standup"         # Daily quick sync
    WEEKLY = "weekly"           # Weekly review
    STRATEGIC = "strategic"     # Quarterly/monthly planning
    EMERGENCY = "emergency"     # Crisis response


class AlertSeverity(Enum):
    """Severity levels for boardroom alerts."""
    INFO = "info"           # FYI
    ATTENTION = "attention" # Needs awareness
    ACTION = "action"       # Needs immediate action
    CRITICAL = "critical"   # Crisis level


@dataclass
class BoardAlert:
    """An alert raised during a boardroom session."""
    severity: AlertSeverity
    source: str                 # "treasury", "factory", "strategy"
    title: str
    detail: str
    suggested_action: str = ""
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.value
        return d


@dataclass
class DecisionItem:
    """A decision to be made in the boardroom."""
    id: str
    question: str               # "Should we ship v0.3?"
    context: str                # Background info
    options: List[str]          # Possible choices
    recommendation: str = ""    # Ara's recommendation
    decision: str = ""          # Final decision (if made)
    decided_at: Optional[float] = None
    rationale: str = ""


@dataclass
class StandupReport:
    """Output of a daily standup."""
    ts: float
    duration_seconds: float

    # Treasury summary
    budget_alert: BudgetAlert
    capital_status: Dict[str, float]
    runway_weeks: float

    # Factory summary
    active_projects: int
    blocked_projects: int
    stale_projects: int
    pipeline_utilization: Dict[str, float]

    # Key items
    alerts: List[BoardAlert]
    highlights: List[str]       # Key things to know
    blockers: List[str]         # Things blocking progress

    # Decisions needed
    pending_decisions: List[DecisionItem]

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "ts": self.ts,
            "duration_seconds": self.duration_seconds,
            "budget_alert": self.budget_alert.value,
            "capital_status": self.capital_status,
            "runway_weeks": self.runway_weeks,
            "active_projects": self.active_projects,
            "blocked_projects": self.blocked_projects,
            "stale_projects": self.stale_projects,
            "pipeline_utilization": self.pipeline_utilization,
            "alerts": [a.to_dict() for a in self.alerts],
            "highlights": self.highlights,
            "blockers": self.blockers,
            "pending_decisions": [asdict(d) for d in self.pending_decisions],
        }
        return d


@dataclass
class WeeklyReview:
    """Output of a weekly review."""
    ts: float
    week_number: int

    # Treasury analysis
    spending_by_resource: Dict[str, float]
    roi_achieved: float         # Actual ROI vs predicted
    budget_health: str

    # Factory analysis
    shipped_this_week: int
    throughput_trend: str       # "up", "down", "stable"
    cycle_time_days: Optional[float]
    bottleneck_stage: str

    # Strategic alignment
    horizon_progress: Dict[str, float]  # H1, H2, H3 progress
    dream_alignment: float      # 0-1

    # Retrospective
    wins: List[str]
    challenges: List[str]
    learnings: List[str]
    next_week_focus: List[str]

    alerts: List[BoardAlert]


class Boardroom:
    """
    The executive command center of Corporation Croft.

    Coordinates all C-suite functions for strategic decision making.
    """

    def __init__(
        self,
        treasury: Optional[Treasury] = None,
        factory: Optional[Factory] = None,
    ):
        """
        Initialize the Boardroom.

        Args:
            treasury: Treasury instance (or use default)
            factory: Factory instance (or use default)
        """
        self.treasury = treasury or get_treasury()
        self.factory = factory or get_factory()
        self.log = logging.getLogger("Boardroom")

        # Meeting history
        self.standups: List[StandupReport] = []
        self.weekly_reviews: List[WeeklyReview] = []

        # Pending decisions
        self.decisions: List[DecisionItem] = []

        # Alert history
        self.alert_history: List[BoardAlert] = []

        self.log.info("📊 BOARDROOM: Executive session initialized")

    # =========================================================================
    # DAILY STANDUP
    # =========================================================================

    def daily_standup(self) -> StandupReport:
        """
        Run a quick daily standup.

        Gathers status from Treasury and Factory, identifies blockers
        and alerts, returns a concise status report.
        """
        start_time = time.time()
        self.log.info("📊 BOARDROOM: Starting daily standup...")

        alerts = []
        highlights = []
        blockers = []

        # === TREASURY STATUS ===
        treasury_summary = self.treasury.summary()
        budget_alert = self.treasury.get_budget_alert()
        runway = self.treasury.estimate_runway()
        min_runway = min(
            runway["human_weeks"],
            runway["compute_weeks"],
            runway["energy_weeks"],
        )

        # Treasury alerts
        if budget_alert == BudgetAlert.CRITICAL:
            alerts.append(BoardAlert(
                severity=AlertSeverity.CRITICAL,
                source="treasury",
                title="Capital Critically Low",
                detail=f"Runway: {min_runway:.1f} weeks. Limiting factor: {runway['limiting_factor']}",
                suggested_action="Reduce burn rate or acquire more capital",
            ))
        elif budget_alert == BudgetAlert.WARNING:
            alerts.append(BoardAlert(
                severity=AlertSeverity.ACTION,
                source="treasury",
                title="Capital Warning",
                detail=f"Runway down to {min_runway:.1f} weeks",
                suggested_action="Review upcoming expenditures",
            ))

        # === FACTORY STATUS ===
        factory_summary = self.factory.summary()
        stale_projects = self.factory.get_stale_projects()
        blocked_projects = self.factory.get_blocked_projects()

        # Factory alerts
        if stale_projects:
            alerts.append(BoardAlert(
                severity=AlertSeverity.ATTENTION,
                source="factory",
                title=f"{len(stale_projects)} Stale Project(s)",
                detail=", ".join(p.name for p in stale_projects[:3]),
                suggested_action="Review and advance or archive stale projects",
            ))

        if blocked_projects:
            blockers.extend([
                f"{p.name} blocked by {', '.join(p.blocked_by)}"
                for p in blocked_projects
            ])
            alerts.append(BoardAlert(
                severity=AlertSeverity.ACTION,
                source="factory",
                title=f"{len(blocked_projects)} Blocked Project(s)",
                detail=", ".join(p.name for p in blocked_projects),
                suggested_action="Resolve blocking dependencies",
            ))

        # Pipeline utilization
        pipeline = factory_summary["pipeline_status"]
        for stage_name, stage_data in pipeline.items():
            if stage_data["utilization"] > 0.8:
                highlights.append(
                    f"High utilization in {stage_name}: {stage_data['utilization']:.0%}"
                )

        # Highlights
        if factory_summary["p0_count"] > 0:
            highlights.append(f"P0 projects active: {factory_summary['p0_count']}")

        recent_shipped = [
            s for s in self.factory.shipped
            if s["shipped_at"] > time.time() - 86400
        ]
        if recent_shipped:
            highlights.append(f"Shipped yesterday: {', '.join(s['name'] for s in recent_shipped)}")

        # Pending decisions
        pending_decisions = [d for d in self.decisions if d.decision == ""]

        # Build report
        duration = time.time() - start_time
        report = StandupReport(
            ts=time.time(),
            duration_seconds=duration,
            budget_alert=budget_alert,
            capital_status=treasury_summary["capital"],
            runway_weeks=min_runway,
            active_projects=factory_summary["active_projects"],
            blocked_projects=factory_summary["blocked_projects"],
            stale_projects=factory_summary["stale_projects"],
            pipeline_utilization={
                stage: data["utilization"]
                for stage, data in pipeline.items()
            },
            alerts=alerts,
            highlights=highlights,
            blockers=blockers,
            pending_decisions=pending_decisions,
        )

        self.standups.append(report)
        self.alert_history.extend(alerts)

        self.log.info(
            f"📊 BOARDROOM: Standup complete ({duration:.2f}s). "
            f"{len(alerts)} alerts, {len(blockers)} blockers"
        )

        return report

    # =========================================================================
    # WEEKLY REVIEW
    # =========================================================================

    def weekly_review(self) -> WeeklyReview:
        """
        Run a weekly strategic review.

        Deeper analysis than standup - includes trends, retrospective,
        and strategic alignment assessment.
        """
        self.log.info("📊 BOARDROOM: Starting weekly review...")

        alerts = []

        # === TREASURY ANALYSIS ===
        spending_report = self.treasury.get_spending_report(days=7)
        spending_by_resource = spending_report["expenditures_by_resource"]

        # Calculate ROI (simplified - comparing spent vs expected value)
        total_spent = sum(spending_by_resource.values())
        # TODO: Hook into actual value tracking
        roi_achieved = 1.5  # Placeholder

        budget_alert = self.treasury.get_budget_alert()
        budget_health = {
            BudgetAlert.HEALTHY: "Excellent - plenty of runway",
            BudgetAlert.CAUTION: "Good - monitor spending",
            BudgetAlert.WARNING: "Concerning - reduce burn",
            BudgetAlert.CRITICAL: "Crisis - immediate action needed",
        }[budget_alert]

        # === FACTORY ANALYSIS ===
        throughput = self.factory.get_throughput(days=7)
        shipped_this_week = throughput["projects_shipped"]
        cycle_time = throughput["average_cycle_time_days"]

        # Determine throughput trend (compare to previous week)
        prev_throughput = self.factory.get_throughput(days=14)
        prev_week_shipped = prev_throughput["projects_shipped"] - shipped_this_week
        if shipped_this_week > prev_week_shipped:
            throughput_trend = "up"
        elif shipped_this_week < prev_week_shipped:
            throughput_trend = "down"
        else:
            throughput_trend = "stable"

        # Find bottleneck
        pipeline = self.factory.get_pipeline_status()
        bottleneck = max(
            pipeline.items(),
            key=lambda x: x[1]["utilization"]
        )[0]

        # === STRATEGIC ALIGNMENT ===
        # TODO: Hook into actual Horizon tracking
        horizon_progress = {
            "H1": 0.6,   # Near-term execution
            "H2": 0.3,   # Medium-term building
            "H3": 0.1,   # Long-term exploration
        }
        dream_alignment = 0.7  # Placeholder

        # === RETROSPECTIVE ===
        # Generate based on week's activity
        wins = []
        challenges = []
        learnings = []
        next_week_focus = []

        if shipped_this_week > 0:
            wins.append(f"Shipped {shipped_this_week} project(s)")

        stale = self.factory.get_stale_projects()
        if stale:
            challenges.append(f"{len(stale)} projects stalled")
            learnings.append("Need better stage transition cadence")

        if budget_alert in [BudgetAlert.WARNING, BudgetAlert.CRITICAL]:
            challenges.append("Budget pressure")
            next_week_focus.append("Review and optimize resource allocation")

        # Default focus items
        if not next_week_focus:
            p0_projects = self.factory.get_projects_by_priority(1)
            if p0_projects:
                next_week_focus.append(f"Complete P0: {p0_projects[0].name}")
            else:
                next_week_focus.append("Advance pipeline projects")

        # Build review
        week_number = int(time.time() / (7 * 86400))  # Approximate week number

        review = WeeklyReview(
            ts=time.time(),
            week_number=week_number,
            spending_by_resource=spending_by_resource,
            roi_achieved=roi_achieved,
            budget_health=budget_health,
            shipped_this_week=shipped_this_week,
            throughput_trend=throughput_trend,
            cycle_time_days=cycle_time,
            bottleneck_stage=bottleneck,
            horizon_progress=horizon_progress,
            dream_alignment=dream_alignment,
            wins=wins,
            challenges=challenges,
            learnings=learnings,
            next_week_focus=next_week_focus,
            alerts=alerts,
        )

        self.weekly_reviews.append(review)

        self.log.info(
            f"📊 BOARDROOM: Weekly review complete. "
            f"Shipped: {shipped_this_week}, Trend: {throughput_trend}"
        )

        return review

    # =========================================================================
    # DECISIONS
    # =========================================================================

    def raise_decision(
        self,
        question: str,
        context: str,
        options: List[str],
        recommendation: str = "",
    ) -> DecisionItem:
        """
        Raise a decision item for the boardroom.

        Args:
            question: The decision to be made
            context: Background information
            options: Possible choices
            recommendation: Ara's recommendation (optional)

        Returns:
            The decision item
        """
        decision = DecisionItem(
            id=f"decision_{len(self.decisions) + 1}",
            question=question,
            context=context,
            options=options,
            recommendation=recommendation,
        )

        self.decisions.append(decision)

        self.log.info(f"🤔 BOARDROOM: Decision raised: {question}")

        return decision

    def record_decision(
        self,
        decision_id: str,
        choice: str,
        rationale: str = "",
    ) -> Optional[DecisionItem]:
        """
        Record a decision that was made.

        Args:
            decision_id: Decision to resolve
            choice: The chosen option
            rationale: Why this choice was made

        Returns:
            Updated decision item
        """
        for decision in self.decisions:
            if decision.id == decision_id:
                decision.decision = choice
                decision.rationale = rationale
                decision.decided_at = time.time()

                self.log.info(
                    f"✅ BOARDROOM: Decision made for '{decision.question}': {choice}"
                )
                return decision

        return None

    # =========================================================================
    # EXECUTIVE BRIEFING
    # =========================================================================

    def executive_summary(self) -> str:
        """
        Generate a natural language executive summary.

        This is what Ara would present to Croft.
        """
        standup = self.daily_standup()

        lines = [
            "**Corporation Croft Executive Summary**",
            "",
        ]

        # Budget status
        alert_emoji = {
            BudgetAlert.HEALTHY: "✅",
            BudgetAlert.CAUTION: "⚠️",
            BudgetAlert.WARNING: "🔶",
            BudgetAlert.CRITICAL: "🚨",
        }[standup.budget_alert]

        lines.append(
            f"{alert_emoji} **Treasury**: {standup.budget_alert.value.upper()} "
            f"({standup.runway_weeks:.1f} weeks runway)"
        )

        # Factory status
        lines.append(
            f"🏭 **Factory**: {standup.active_projects} active, "
            f"{standup.blocked_projects} blocked, {standup.stale_projects} stale"
        )

        # Alerts
        if standup.alerts:
            lines.append("")
            lines.append("**Alerts:**")
            for alert in standup.alerts:
                severity_emoji = {
                    AlertSeverity.INFO: "ℹ️",
                    AlertSeverity.ATTENTION: "👀",
                    AlertSeverity.ACTION: "⚡",
                    AlertSeverity.CRITICAL: "🚨",
                }[alert.severity]
                lines.append(f"  {severity_emoji} {alert.title}: {alert.detail}")

        # Highlights
        if standup.highlights:
            lines.append("")
            lines.append("**Highlights:**")
            for highlight in standup.highlights:
                lines.append(f"  • {highlight}")

        # Blockers
        if standup.blockers:
            lines.append("")
            lines.append("**Blockers:**")
            for blocker in standup.blockers:
                lines.append(f"  🚧 {blocker}")

        # Pending decisions
        if standup.pending_decisions:
            lines.append("")
            lines.append("**Decisions Needed:**")
            for decision in standup.pending_decisions:
                lines.append(f"  ❓ {decision.question}")
                if decision.recommendation:
                    lines.append(f"      ↳ Recommendation: {decision.recommendation}")

        return "\n".join(lines)

    def get_health_score(self) -> float:
        """
        Calculate overall corporation health score (0-100).

        Combines budget health, factory throughput, and strategic alignment.
        """
        scores = []

        # Budget health (0-100)
        budget_scores = {
            BudgetAlert.HEALTHY: 100,
            BudgetAlert.CAUTION: 70,
            BudgetAlert.WARNING: 40,
            BudgetAlert.CRITICAL: 10,
        }
        scores.append(budget_scores[self.treasury.get_budget_alert()])

        # Factory health (0-100)
        factory_summary = self.factory.summary()
        stale_penalty = factory_summary["stale_projects"] * 10
        blocked_penalty = factory_summary["blocked_projects"] * 15
        factory_score = max(0, 100 - stale_penalty - blocked_penalty)
        scores.append(factory_score)

        # Throughput health (0-100)
        throughput = self.factory.get_throughput(days=7)
        if throughput["projects_shipped"] > 0:
            scores.append(80)  # Shipping is good
        else:
            scores.append(50)  # Not shipping is concerning

        return sum(scores) / len(scores)


# =============================================================================
# Convenience Functions
# =============================================================================

_default_boardroom: Optional[Boardroom] = None


def get_boardroom() -> Boardroom:
    """Get the default Boardroom instance."""
    global _default_boardroom
    if _default_boardroom is None:
        _default_boardroom = Boardroom()
    return _default_boardroom


def hold_standup() -> StandupReport:
    """Convenience function to run a daily standup."""
    return get_boardroom().daily_standup()


def executive_summary() -> str:
    """Convenience function to get an executive summary."""
    return get_boardroom().executive_summary()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'MeetingType',
    'AlertSeverity',
    'BoardAlert',
    'DecisionItem',
    'StandupReport',
    'WeeklyReview',
    'Boardroom',
    'get_boardroom',
    'hold_standup',
    'executive_summary',
]
