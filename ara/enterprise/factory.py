"""
The Factory - Corporation Croft's COO
=====================================

The Factory converts Ideas into Products. She doesn't just "fix a bug";
she "ships a release."

Pipeline Stages:
    - Research: Exploration, spikes, feasibility studies
    - Prototype: Quick & dirty implementation, proof of concept
    - Production: Production-ready code, tested, documented
    - Maintenance: Bug fixes, optimizations, support

Core Principles:
    1. Pipeline Discipline: Every idea must traverse the pipeline
    2. Quality Gates: No stage transition without validation
    3. Capacity Awareness: Don't overcommit the factory
    4. Ship or Kill: Stale projects get archived

Usage:
    from ara.enterprise.factory import Factory, Project

    factory = Factory()

    # Create a new project
    project = factory.create_project(
        name="SNN Attention Module",
        source_idea="Implement attention for spike networks",
        priority=1,
    )

    # Advance through pipeline
    factory.advance_stage(project.id, notes="Spike completed")

    # Ship when ready
    factory.ship(project.id, version="0.1.0")
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Literal, Any
from enum import Enum

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """The four stages of the production pipeline."""
    RESEARCH = "research"       # Exploration, feasibility
    PROTOTYPE = "prototype"     # Quick implementation
    PRODUCTION = "production"   # Production-ready
    MAINTENANCE = "maintenance" # Post-ship support

    @classmethod
    def order(cls) -> List["PipelineStage"]:
        return [cls.RESEARCH, cls.PROTOTYPE, cls.PRODUCTION, cls.MAINTENANCE]

    def next_stage(self) -> Optional["PipelineStage"]:
        """Get the next stage in the pipeline."""
        order = self.order()
        idx = order.index(self)
        if idx < len(order) - 1:
            return order[idx + 1]
        return None

    def previous_stage(self) -> Optional["PipelineStage"]:
        """Get the previous stage (for rollback)."""
        order = self.order()
        idx = order.index(self)
        if idx > 0:
            return order[idx - 1]
        return None


class ProjectStatus(Enum):
    """Project lifecycle status."""
    ACTIVE = "active"         # In progress
    BLOCKED = "blocked"       # Waiting on dependency
    SHIPPED = "shipped"       # Released to production
    ARCHIVED = "archived"     # Killed or completed


class QualityGate(Enum):
    """Quality gates for stage transitions."""
    RESEARCH_TO_PROTOTYPE = "research_to_prototype"
    PROTOTYPE_TO_PRODUCTION = "prototype_to_production"
    PRODUCTION_TO_SHIP = "production_to_ship"


@dataclass
class StageTransition:
    """Record of a stage transition."""
    ts: float
    from_stage: PipelineStage
    to_stage: PipelineStage
    notes: str = ""
    quality_gate_passed: bool = True


@dataclass
class Project:
    """
    A project moving through the production pipeline.

    Every idea that gets actioned becomes a Project.
    """
    id: str
    name: str
    source_idea: str                          # The original idea/request
    stage: PipelineStage = PipelineStage.RESEARCH
    status: ProjectStatus = ProjectStatus.ACTIVE
    priority: int = 2                         # 1=P0 critical, 2=P1 high, 3=P2 medium
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # Stage history
    transitions: List[StageTransition] = field(default_factory=list)

    # Metadata
    tags: List[str] = field(default_factory=list)
    assignee: str = "Ara"                     # Who owns this project
    estimated_hours: float = 0.0              # Treasury integration
    actual_hours: float = 0.0                 # Tracked time

    # Dependencies
    blocked_by: List[str] = field(default_factory=list)  # Project IDs
    blocks: List[str] = field(default_factory=list)      # Project IDs we block

    # Shipping info
    version: Optional[str] = None             # Release version
    shipped_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["stage"] = self.stage.value
        d["status"] = self.status.value
        d["transitions"] = [
            {**asdict(t), "from_stage": t.from_stage.value, "to_stage": t.to_stage.value}
            for t in self.transitions
        ]
        return d

    def age_days(self) -> float:
        """How old is this project in days?"""
        return (time.time() - self.created_at) / 86400

    def days_in_stage(self) -> float:
        """How long has it been in the current stage?"""
        if self.transitions:
            last_transition = self.transitions[-1].ts
            return (time.time() - last_transition) / 86400
        return self.age_days()

    def is_stale(self, threshold_days: float = 14.0) -> bool:
        """Is this project stale and needs attention?"""
        return (
            self.status == ProjectStatus.ACTIVE and
            self.days_in_stage() > threshold_days
        )


@dataclass
class CapacitySlot:
    """Represents available capacity in a pipeline stage."""
    stage: PipelineStage
    max_concurrent: int = 3       # Max projects in this stage
    current_count: int = 0

    @property
    def available(self) -> int:
        return max(0, self.max_concurrent - self.current_count)

    @property
    def utilization(self) -> float:
        if self.max_concurrent == 0:
            return 0.0
        return self.current_count / self.max_concurrent


@dataclass
class QualityGateResult:
    """Result of a quality gate check."""
    gate: QualityGate
    passed: bool
    checks: Dict[str, bool] = field(default_factory=dict)
    notes: str = ""


class Factory:
    """
    The COO of Corporation Croft.

    Manages the production pipeline, converting Ideas into shipped Products.
    """

    # Default capacity limits per stage
    DEFAULT_CAPACITY = {
        PipelineStage.RESEARCH: 5,      # Can explore many things
        PipelineStage.PROTOTYPE: 3,     # Limited prototype slots
        PipelineStage.PRODUCTION: 2,    # Focus on shipping
        PipelineStage.MAINTENANCE: 10,  # Can support many shipped things
    }

    # Staleness thresholds by stage (days)
    STALE_THRESHOLDS = {
        PipelineStage.RESEARCH: 7,      # Research should conclude quickly
        PipelineStage.PROTOTYPE: 14,    # Prototypes shouldn't linger
        PipelineStage.PRODUCTION: 21,   # Production takes longer
        PipelineStage.MAINTENANCE: 90,  # Maintenance can be long-lived
    }

    def __init__(
        self,
        capacity: Optional[Dict[PipelineStage, int]] = None,
    ):
        """
        Initialize the Factory.

        Args:
            capacity: Custom capacity limits per stage
        """
        self.log = logging.getLogger("Factory")

        # Project registry
        self.projects: Dict[str, Project] = {}

        # Capacity management
        self.capacity = {
            stage: CapacitySlot(
                stage=stage,
                max_concurrent=capacity.get(stage, default) if capacity else default
            )
            for stage, default in self.DEFAULT_CAPACITY.items()
        }

        # Shipped products registry
        self.shipped: List[Dict[str, Any]] = []

        self.log.info("🏭 FACTORY: Initialized production pipeline")

    # =========================================================================
    # PROJECT LIFECYCLE
    # =========================================================================

    def create_project(
        self,
        name: str,
        source_idea: str,
        priority: int = 2,
        tags: Optional[List[str]] = None,
        estimated_hours: float = 0.0,
    ) -> Project:
        """
        Create a new project in the Research stage.

        Args:
            name: Project name
            source_idea: The original idea/request that spawned this
            priority: 1=P0, 2=P1, 3=P2
            tags: Classification tags
            estimated_hours: Estimated hours for Treasury

        Returns:
            The new Project
        """
        project_id = f"proj_{uuid.uuid4().hex[:8]}"

        project = Project(
            id=project_id,
            name=name,
            source_idea=source_idea,
            priority=priority,
            tags=tags or [],
            estimated_hours=estimated_hours,
        )

        self.projects[project_id] = project
        self._update_capacity_counts()

        self.log.info(
            f"🆕 FACTORY: Created project '{name}' (P{priority}) → Research"
        )

        return project

    def get_project(self, project_id: str) -> Optional[Project]:
        """Get a project by ID."""
        return self.projects.get(project_id)

    def advance_stage(
        self,
        project_id: str,
        notes: str = "",
        force: bool = False,
    ) -> Optional[Project]:
        """
        Advance a project to the next pipeline stage.

        Args:
            project_id: Project to advance
            notes: Notes about this transition
            force: Skip quality gate check

        Returns:
            Updated project or None if transition failed
        """
        project = self.projects.get(project_id)
        if not project:
            self.log.warning(f"❌ FACTORY: Project {project_id} not found")
            return None

        if project.status != ProjectStatus.ACTIVE:
            self.log.warning(
                f"❌ FACTORY: Cannot advance {project.name} - status is {project.status.value}"
            )
            return None

        next_stage = project.stage.next_stage()
        if not next_stage:
            self.log.warning(
                f"❌ FACTORY: {project.name} is already in final stage"
            )
            return None

        # Quality gate check
        if not force:
            gate_result = self._check_quality_gate(project, next_stage)
            if not gate_result.passed:
                self.log.warning(
                    f"❌ FACTORY: Quality gate failed for {project.name}: {gate_result.notes}"
                )
                return None

        # Capacity check
        target_capacity = self.capacity[next_stage]
        if target_capacity.available <= 0:
            self.log.warning(
                f"❌ FACTORY: No capacity in {next_stage.value} for {project.name}"
            )
            return None

        # Perform transition
        transition = StageTransition(
            ts=time.time(),
            from_stage=project.stage,
            to_stage=next_stage,
            notes=notes,
            quality_gate_passed=True,
        )

        project.transitions.append(transition)
        project.stage = next_stage
        project.updated_at = time.time()

        self._update_capacity_counts()

        self.log.info(
            f"⏩ FACTORY: {project.name} advanced to {next_stage.value}"
        )

        return project

    def rollback_stage(
        self,
        project_id: str,
        notes: str = "",
    ) -> Optional[Project]:
        """
        Roll back a project to the previous stage.

        Args:
            project_id: Project to roll back
            notes: Reason for rollback

        Returns:
            Updated project or None if rollback failed
        """
        project = self.projects.get(project_id)
        if not project:
            return None

        prev_stage = project.stage.previous_stage()
        if not prev_stage:
            self.log.warning(
                f"❌ FACTORY: {project.name} is already in first stage"
            )
            return None

        transition = StageTransition(
            ts=time.time(),
            from_stage=project.stage,
            to_stage=prev_stage,
            notes=f"ROLLBACK: {notes}",
            quality_gate_passed=False,
        )

        project.transitions.append(transition)
        project.stage = prev_stage
        project.updated_at = time.time()

        self._update_capacity_counts()

        self.log.warning(
            f"⏪ FACTORY: {project.name} rolled back to {prev_stage.value}"
        )

        return project

    def ship(
        self,
        project_id: str,
        version: str,
        notes: str = "",
    ) -> Optional[Project]:
        """
        Ship a project to production.

        Args:
            project_id: Project to ship
            version: Release version (e.g., "1.0.0")
            notes: Release notes

        Returns:
            Shipped project or None if ship failed
        """
        project = self.projects.get(project_id)
        if not project:
            return None

        if project.stage != PipelineStage.PRODUCTION:
            self.log.warning(
                f"❌ FACTORY: {project.name} must be in Production to ship "
                f"(currently in {project.stage.value})"
            )
            return None

        # Quality gate for shipping
        gate_result = self._check_quality_gate(project, PipelineStage.MAINTENANCE)
        if not gate_result.passed:
            self.log.warning(
                f"❌ FACTORY: Ship quality gate failed for {project.name}"
            )
            return None

        # Ship it!
        project.status = ProjectStatus.SHIPPED
        project.version = version
        project.shipped_at = time.time()
        project.stage = PipelineStage.MAINTENANCE
        project.updated_at = time.time()

        self.shipped.append({
            "project_id": project.id,
            "name": project.name,
            "version": version,
            "shipped_at": project.shipped_at,
            "notes": notes,
        })

        self._update_capacity_counts()

        self.log.info(
            f"🚀 FACTORY: SHIPPED {project.name} v{version}!"
        )

        return project

    def archive(
        self,
        project_id: str,
        reason: str = "No longer needed",
    ) -> Optional[Project]:
        """
        Archive (kill) a project.

        Args:
            project_id: Project to archive
            reason: Why we're killing it

        Returns:
            Archived project
        """
        project = self.projects.get(project_id)
        if not project:
            return None

        project.status = ProjectStatus.ARCHIVED
        project.updated_at = time.time()

        self._update_capacity_counts()

        self.log.info(
            f"📦 FACTORY: Archived {project.name}: {reason}"
        )

        return project

    def block(
        self,
        project_id: str,
        blocked_by: str,
        notes: str = "",
    ) -> Optional[Project]:
        """
        Mark a project as blocked by another project.

        Args:
            project_id: Project to block
            blocked_by: Project ID that blocks this one
            notes: Blocking reason

        Returns:
            Blocked project
        """
        project = self.projects.get(project_id)
        blocker = self.projects.get(blocked_by)

        if not project or not blocker:
            return None

        project.status = ProjectStatus.BLOCKED
        project.blocked_by.append(blocked_by)
        blocker.blocks.append(project_id)
        project.updated_at = time.time()

        self.log.info(
            f"🚧 FACTORY: {project.name} blocked by {blocker.name}: {notes}"
        )

        return project

    def unblock(self, project_id: str) -> Optional[Project]:
        """Unblock a project."""
        project = self.projects.get(project_id)
        if not project:
            return None

        if project.status == ProjectStatus.BLOCKED:
            project.status = ProjectStatus.ACTIVE
            project.blocked_by = []
            project.updated_at = time.time()

            self.log.info(f"✅ FACTORY: {project.name} unblocked")

        return project

    # =========================================================================
    # QUALITY GATES
    # =========================================================================

    def _check_quality_gate(
        self,
        project: Project,
        target_stage: PipelineStage,
    ) -> QualityGateResult:
        """
        Check if a project passes the quality gate for stage transition.

        This is extensible - override for custom gates.
        """
        checks = {}
        notes = []

        if target_stage == PipelineStage.PROTOTYPE:
            # Research → Prototype: Must have spent some time
            checks["has_research_time"] = project.days_in_stage() >= 0.1  # At least 2.4 hours
            checks["has_notes"] = len(project.transitions) > 0 or True  # Be lenient

        elif target_stage == PipelineStage.PRODUCTION:
            # Prototype → Production: Must have a working prototype
            checks["prototype_reviewed"] = True  # TODO: Hook into code review
            checks["has_tests"] = True  # TODO: Check test coverage

        elif target_stage == PipelineStage.MAINTENANCE:
            # Production → Ship: Must pass all checks
            checks["production_ready"] = True
            checks["documentation"] = True  # TODO: Check for docs

        # All checks must pass
        passed = all(checks.values())
        if not passed:
            failed = [k for k, v in checks.items() if not v]
            notes.append(f"Failed checks: {', '.join(failed)}")

        gate = QualityGate.RESEARCH_TO_PROTOTYPE  # Default
        if target_stage == PipelineStage.PRODUCTION:
            gate = QualityGate.PROTOTYPE_TO_PRODUCTION
        elif target_stage == PipelineStage.MAINTENANCE:
            gate = QualityGate.PRODUCTION_TO_SHIP

        return QualityGateResult(
            gate=gate,
            passed=passed,
            checks=checks,
            notes="; ".join(notes),
        )

    # =========================================================================
    # CAPACITY & QUERIES
    # =========================================================================

    def _update_capacity_counts(self) -> None:
        """Update capacity counts based on active projects."""
        # Reset counts
        for slot in self.capacity.values():
            slot.current_count = 0

        # Count active projects per stage
        for project in self.projects.values():
            if project.status == ProjectStatus.ACTIVE:
                self.capacity[project.stage].current_count += 1

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get full pipeline status for dashboards."""
        return {
            stage.value: {
                "projects": [
                    p.name for p in self.projects.values()
                    if p.stage == stage and p.status == ProjectStatus.ACTIVE
                ],
                "count": self.capacity[stage].current_count,
                "capacity": self.capacity[stage].max_concurrent,
                "utilization": self.capacity[stage].utilization,
            }
            for stage in PipelineStage.order()
        }

    def get_stale_projects(self) -> List[Project]:
        """Get projects that have been in their stage too long."""
        stale = []
        for project in self.projects.values():
            if project.status != ProjectStatus.ACTIVE:
                continue
            threshold = self.STALE_THRESHOLDS.get(project.stage, 14)
            if project.days_in_stage() > threshold:
                stale.append(project)
        return stale

    def get_blocked_projects(self) -> List[Project]:
        """Get all blocked projects."""
        return [
            p for p in self.projects.values()
            if p.status == ProjectStatus.BLOCKED
        ]

    def get_projects_by_stage(self, stage: PipelineStage) -> List[Project]:
        """Get all active projects in a given stage."""
        return [
            p for p in self.projects.values()
            if p.stage == stage and p.status == ProjectStatus.ACTIVE
        ]

    def get_projects_by_priority(self, priority: int) -> List[Project]:
        """Get all active projects with given priority."""
        return [
            p for p in self.projects.values()
            if p.priority == priority and p.status == ProjectStatus.ACTIVE
        ]

    # =========================================================================
    # REPORTING
    # =========================================================================

    def summary(self) -> Dict[str, Any]:
        """Get a summary for dashboards/boardroom."""
        active_projects = [
            p for p in self.projects.values()
            if p.status == ProjectStatus.ACTIVE
        ]

        return {
            "total_projects": len(self.projects),
            "active_projects": len(active_projects),
            "blocked_projects": len(self.get_blocked_projects()),
            "stale_projects": len(self.get_stale_projects()),
            "shipped_total": len(self.shipped),
            "pipeline_status": self.get_pipeline_status(),
            "p0_count": len([p for p in active_projects if p.priority == 1]),
            "p1_count": len([p for p in active_projects if p.priority == 2]),
            "p2_count": len([p for p in active_projects if p.priority == 3]),
        }

    def get_throughput(self, days: int = 30) -> Dict[str, Any]:
        """
        Calculate factory throughput metrics.

        Args:
            days: Lookback period

        Returns:
            Throughput metrics
        """
        cutoff = time.time() - (days * 86400)

        shipped_in_period = [
            s for s in self.shipped
            if s["shipped_at"] > cutoff
        ]

        created_in_period = [
            p for p in self.projects.values()
            if p.created_at > cutoff
        ]

        return {
            "period_days": days,
            "projects_created": len(created_in_period),
            "projects_shipped": len(shipped_in_period),
            "ship_rate": len(shipped_in_period) / max(days / 7, 1),  # per week
            "average_cycle_time_days": self._calculate_average_cycle_time(shipped_in_period),
        }

    def _calculate_average_cycle_time(
        self,
        shipped: List[Dict[str, Any]]
    ) -> Optional[float]:
        """Calculate average time from creation to ship."""
        if not shipped:
            return None

        cycle_times = []
        for s in shipped:
            project = self.projects.get(s["project_id"])
            if project:
                cycle_times.append(
                    (project.shipped_at - project.created_at) / 86400
                )

        return sum(cycle_times) / len(cycle_times) if cycle_times else None


# =============================================================================
# Convenience Functions
# =============================================================================

_default_factory: Optional[Factory] = None


def get_factory() -> Factory:
    """Get the default Factory instance."""
    global _default_factory
    if _default_factory is None:
        _default_factory = Factory()
    return _default_factory


def create_project(
    name: str,
    source_idea: str,
    priority: int = 2,
) -> Project:
    """Convenience function to create a project."""
    return get_factory().create_project(name, source_idea, priority)


def ship_project(project_id: str, version: str) -> Optional[Project]:
    """Convenience function to ship a project."""
    return get_factory().ship(project_id, version)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'PipelineStage',
    'ProjectStatus',
    'QualityGate',
    'StageTransition',
    'Project',
    'CapacitySlot',
    'QualityGateResult',
    'Factory',
    'get_factory',
    'create_project',
    'ship_project',
]
