"""
Idea Registry - Ara's Self-Improvement Proposal System
=======================================================

This is where Ara writes her own "Change Requests" when she senses
something could be better. Ideas are stored as YAML files that specify:
- The problem (symptoms, metrics, context)
- The interface for solutions (what files to touch, what tests to pass)
- Risk bounds (safety constraints)

Ideas progress through states:
    DRAFT -> OPEN -> IN_REVIEW -> TESTING -> ACCEPTED/REJECTED -> MERGED

The EvolutionDaemon picks open ideas and summons teachers to propose patches.

Usage:
    registry = IdeaRegistry("/var/lib/ara/ideas")

    # Ara senses a problem and creates an idea
    idea = registry.create_idea(
        title="Reduce DMA latency for pain signals",
        symptom="High prediction error during heavy I/O",
        current_metrics={"latency_ms": 14.2},
        target_metrics={"latency_ms": 1.0},
        affected_files=["banos/fpga/rtl/axi_dma_ring.sv"],
        risk_bounds={"max_thermal_delta_c": 5}
    )

    # Later, EvolutionDaemon picks it up
    open_ideas = registry.get_open_ideas()
"""

import os
import yaml
import hashlib
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any
import shutil


logger = logging.getLogger(__name__)


class IdeaState(str, Enum):
    """Lifecycle states for improvement ideas."""
    DRAFT = "draft"           # Being formulated
    OPEN = "open"             # Ready for teacher proposals
    IN_REVIEW = "in_review"   # Proposals received, under evaluation
    TESTING = "testing"       # Best proposal being tested in sandbox
    ACCEPTED = "accepted"     # Passed all tests, ready for merge
    REJECTED = "rejected"     # Failed tests or violated constraints
    MERGED = "merged"         # Successfully integrated
    ARCHIVED = "archived"     # Closed without action


class IdeaPriority(str, Enum):
    """Priority levels based on pain/impact."""
    CRITICAL = "critical"     # Causing frequent pain, blocking
    HIGH = "high"            # Significant quality-of-life issue
    MEDIUM = "medium"        # Nice to have improvement
    LOW = "low"              # Minor optimization
    EXPLORATORY = "exploratory"  # Pure curiosity/learning


@dataclass
class RiskBounds:
    """Safety constraints that any solution must respect."""
    max_thermal_delta_c: float = 5.0
    max_cpu_util_percent: float = 20.0
    max_memory_delta_mb: float = 500.0
    max_latency_increase_ms: float = 0.0
    forbidden_syscalls: List[str] = field(default_factory=list)
    require_rollback_plan: bool = True
    max_lines_changed: int = 500

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RiskBounds':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ProposalInterface:
    """Defines what a solution proposal should look like."""
    input_artifacts: List[str]      # Files teachers can read/modify
    output_type: str = "patch"      # patch, config, script
    safety_tests: List[str] = field(default_factory=list)
    perf_tests: List[str] = field(default_factory=list)
    integration_tests: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProposalInterface':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class TeacherProposal:
    """A proposed solution from a teacher (LLM)."""
    teacher_id: str              # e.g., "claude-opus", "gemini-pro", "local-ara"
    submitted_at: datetime
    patch_content: str           # The actual diff/code
    rationale: str               # Why this approach
    estimated_improvement: Dict[str, float]
    confidence: float            # 0.0 to 1.0
    test_results: Optional[Dict[str, bool]] = None
    score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['submitted_at'] = self.submitted_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TeacherProposal':
        data['submitted_at'] = datetime.fromisoformat(data['submitted_at'])
        return cls(**data)


@dataclass
class Idea:
    """
    A self-improvement proposal.

    This is Ara saying: "I noticed something could be better, here's
    what I want, and here's how to safely try to fix it."
    """
    id: str
    title: str
    state: IdeaState
    priority: IdeaPriority

    # Problem description
    symptom: str
    context: str = ""
    trigger_event: Optional[str] = None  # What caused Ara to notice

    # Metrics
    current_metrics: Dict[str, float] = field(default_factory=dict)
    target_metrics: Dict[str, float] = field(default_factory=dict)

    # Solution interface
    proposal_interface: ProposalInterface = field(default_factory=ProposalInterface)
    risk_bounds: RiskBounds = field(default_factory=RiskBounds)

    # Proposals from teachers
    proposals: List[TeacherProposal] = field(default_factory=list)
    winning_proposal_idx: Optional[int] = None

    # Lifecycle
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None

    # Lineage
    parent_idea_id: Optional[str] = None  # If this is a refinement
    related_episode_ids: List[str] = field(default_factory=list)

    # Outcome
    outcome_notes: str = ""
    actual_improvement: Dict[str, float] = field(default_factory=dict)

    def to_yaml(self) -> str:
        """Serialize to YAML for storage."""
        data = {
            'id': self.id,
            'title': self.title,
            'state': self.state.value,
            'priority': self.priority.value,
            'symptom': self.symptom,
            'context': self.context,
            'trigger_event': self.trigger_event,
            'current_metrics': self.current_metrics,
            'target_metrics': self.target_metrics,
            'proposal_interface': self.proposal_interface.to_dict(),
            'risk_bounds': self.risk_bounds.to_dict(),
            'proposals': [p.to_dict() for p in self.proposals],
            'winning_proposal_idx': self.winning_proposal_idx,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'parent_idea_id': self.parent_idea_id,
            'related_episode_ids': self.related_episode_ids,
            'outcome_notes': self.outcome_notes,
            'actual_improvement': self.actual_improvement,
        }
        return yaml.dump(data, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> 'Idea':
        """Deserialize from YAML."""
        data = yaml.safe_load(yaml_str)
        return cls(
            id=data['id'],
            title=data['title'],
            state=IdeaState(data['state']),
            priority=IdeaPriority(data['priority']),
            symptom=data['symptom'],
            context=data.get('context', ''),
            trigger_event=data.get('trigger_event'),
            current_metrics=data.get('current_metrics', {}),
            target_metrics=data.get('target_metrics', {}),
            proposal_interface=ProposalInterface.from_dict(data.get('proposal_interface', {})),
            risk_bounds=RiskBounds.from_dict(data.get('risk_bounds', {})),
            proposals=[TeacherProposal.from_dict(p) for p in data.get('proposals', [])],
            winning_proposal_idx=data.get('winning_proposal_idx'),
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            resolved_at=datetime.fromisoformat(data['resolved_at']) if data.get('resolved_at') else None,
            parent_idea_id=data.get('parent_idea_id'),
            related_episode_ids=data.get('related_episode_ids', []),
            outcome_notes=data.get('outcome_notes', ''),
            actual_improvement=data.get('actual_improvement', {}),
        )


class IdeaRegistry:
    """
    Manages the collection of self-improvement ideas.

    Ideas are stored as YAML files in a directory structure:
        ideas/
            open/
                00042_reduce_dma_latency.yaml
            testing/
                00041_optimize_fan_curve.yaml
            merged/
                00040_fix_thermal_spike.yaml
            ...
    """

    def __init__(self, base_dir: str = "/var/lib/ara/ideas"):
        self.base_dir = Path(base_dir)
        self.log = logging.getLogger("IdeaRegistry")

        # Create directory structure
        for state in IdeaState:
            (self.base_dir / state.value).mkdir(parents=True, exist_ok=True)

        # Counter for generating IDs
        self._next_id = self._find_max_id() + 1

    def _find_max_id(self) -> int:
        """Find the highest existing idea ID."""
        max_id = 0
        for state_dir in self.base_dir.iterdir():
            if state_dir.is_dir():
                for yaml_file in state_dir.glob("*.yaml"):
                    try:
                        # Extract ID from filename like "00042_title.yaml"
                        id_str = yaml_file.stem.split('_')[0]
                        max_id = max(max_id, int(id_str))
                    except (ValueError, IndexError):
                        continue
        return max_id

    def _generate_id(self) -> str:
        """Generate a new unique idea ID."""
        idea_id = f"{self._next_id:05d}"
        self._next_id += 1
        return idea_id

    def _slugify(self, title: str) -> str:
        """Convert title to filesystem-safe slug."""
        slug = title.lower()
        slug = ''.join(c if c.isalnum() or c == ' ' else '' for c in slug)
        slug = '_'.join(slug.split()[:5])  # First 5 words
        return slug

    def _get_idea_path(self, idea: Idea) -> Path:
        """Get the file path for an idea."""
        filename = f"{idea.id}_{self._slugify(idea.title)}.yaml"
        return self.base_dir / idea.state.value / filename

    def _find_idea_file(self, idea_id: str) -> Optional[Path]:
        """Find an idea file by ID across all state directories."""
        for state_dir in self.base_dir.iterdir():
            if state_dir.is_dir():
                for yaml_file in state_dir.glob(f"{idea_id}_*.yaml"):
                    return yaml_file
        return None

    def create_idea(
        self,
        title: str,
        symptom: str,
        current_metrics: Dict[str, float],
        target_metrics: Dict[str, float],
        affected_files: List[str],
        priority: IdeaPriority = IdeaPriority.MEDIUM,
        context: str = "",
        trigger_event: Optional[str] = None,
        risk_bounds: Optional[RiskBounds] = None,
        safety_tests: Optional[List[str]] = None,
        perf_tests: Optional[List[str]] = None,
    ) -> Idea:
        """
        Create a new improvement idea.

        This is called when Ara senses something could be better.
        """
        idea = Idea(
            id=self._generate_id(),
            title=title,
            state=IdeaState.OPEN,  # Start as open, ready for proposals
            priority=priority,
            symptom=symptom,
            context=context,
            trigger_event=trigger_event,
            current_metrics=current_metrics,
            target_metrics=target_metrics,
            proposal_interface=ProposalInterface(
                input_artifacts=affected_files,
                safety_tests=safety_tests or [],
                perf_tests=perf_tests or [],
            ),
            risk_bounds=risk_bounds or RiskBounds(),
        )

        self._save_idea(idea)
        self.log.info(f"Created idea {idea.id}: {title}")
        return idea

    def _save_idea(self, idea: Idea) -> None:
        """Save an idea to disk."""
        idea.updated_at = datetime.now()
        path = self._get_idea_path(idea)
        path.write_text(idea.to_yaml())

    def get_idea(self, idea_id: str) -> Optional[Idea]:
        """Load an idea by ID."""
        path = self._find_idea_file(idea_id)
        if path and path.exists():
            return Idea.from_yaml(path.read_text())
        return None

    def update_idea(self, idea: Idea) -> None:
        """Update an existing idea, moving it to new state directory if needed."""
        old_path = self._find_idea_file(idea.id)
        new_path = self._get_idea_path(idea)

        # If state changed, move the file
        if old_path and old_path != new_path:
            old_path.unlink()

        self._save_idea(idea)

    def transition_state(self, idea_id: str, new_state: IdeaState, notes: str = "") -> Optional[Idea]:
        """Transition an idea to a new state."""
        idea = self.get_idea(idea_id)
        if not idea:
            return None

        old_state = idea.state
        idea.state = new_state

        if new_state in [IdeaState.ACCEPTED, IdeaState.REJECTED, IdeaState.MERGED]:
            idea.resolved_at = datetime.now()

        if notes:
            idea.outcome_notes += f"\n[{datetime.now().isoformat()}] {old_state.value} -> {new_state.value}: {notes}"

        self.update_idea(idea)
        self.log.info(f"Idea {idea_id}: {old_state.value} -> {new_state.value}")
        return idea

    def add_proposal(self, idea_id: str, proposal: TeacherProposal) -> Optional[Idea]:
        """Add a teacher proposal to an idea."""
        idea = self.get_idea(idea_id)
        if not idea:
            return None

        idea.proposals.append(proposal)

        # Transition to IN_REVIEW if this is the first proposal
        if idea.state == IdeaState.OPEN and len(idea.proposals) == 1:
            idea.state = IdeaState.IN_REVIEW

        self.update_idea(idea)
        self.log.info(f"Idea {idea_id}: Received proposal from {proposal.teacher_id}")
        return idea

    def get_open_ideas(self, priority: Optional[IdeaPriority] = None) -> List[Idea]:
        """Get all ideas that are open for proposals."""
        ideas = []
        open_dir = self.base_dir / IdeaState.OPEN.value

        for yaml_file in open_dir.glob("*.yaml"):
            try:
                idea = Idea.from_yaml(yaml_file.read_text())
                if priority is None or idea.priority == priority:
                    ideas.append(idea)
            except Exception as e:
                self.log.error(f"Failed to load {yaml_file}: {e}")

        # Sort by priority (critical first), then by age
        priority_order = {p: i for i, p in enumerate(IdeaPriority)}
        ideas.sort(key=lambda x: (priority_order[x.priority], x.created_at))

        return ideas

    def get_ideas_in_state(self, state: IdeaState) -> List[Idea]:
        """Get all ideas in a specific state."""
        ideas = []
        state_dir = self.base_dir / state.value

        for yaml_file in state_dir.glob("*.yaml"):
            try:
                ideas.append(Idea.from_yaml(yaml_file.read_text()))
            except Exception as e:
                self.log.error(f"Failed to load {yaml_file}: {e}")

        return ideas

    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        stats = {"total": 0, "by_state": {}, "by_priority": {}}

        for state in IdeaState:
            count = len(list((self.base_dir / state.value).glob("*.yaml")))
            stats["by_state"][state.value] = count
            stats["total"] += count

        # Count by priority across open/in_review
        for priority in IdeaPriority:
            count = 0
            for idea in self.get_open_ideas():
                if idea.priority == priority:
                    count += 1
            stats["by_priority"][priority.value] = count

        return stats


# =============================================================================
# Convenience functions
# =============================================================================

_registry: Optional[IdeaRegistry] = None


def get_idea_registry(base_dir: str = "/var/lib/ara/ideas") -> IdeaRegistry:
    """Get the global idea registry instance."""
    global _registry
    if _registry is None:
        _registry = IdeaRegistry(base_dir)
    return _registry


def propose_improvement(
    title: str,
    symptom: str,
    current_metrics: Dict[str, float],
    target_metrics: Dict[str, float],
    affected_files: List[str],
    **kwargs
) -> Idea:
    """
    Quick helper for Ara to propose an improvement.

    Called from pain detection, prediction error spikes, etc.
    """
    return get_idea_registry().create_idea(
        title=title,
        symptom=symptom,
        current_metrics=current_metrics,
        target_metrics=target_metrics,
        affected_files=affected_files,
        **kwargs
    )


__all__ = [
    "IdeaRegistry",
    "Idea",
    "IdeaState",
    "IdeaPriority",
    "RiskBounds",
    "ProposalInterface",
    "TeacherProposal",
    "get_idea_registry",
    "propose_improvement",
]
