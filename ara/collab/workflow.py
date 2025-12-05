"""Workflow state machine for Ara's dev sessions.

This defines the workflow Ara follows when tackling a problem:

    OBSERVE_ISSUE → TRIAGE → IDEATE → SPECIFY → IMPLEMENT → VERIFY → REPORT_TO_CROFT

Each state has:
- Entry conditions (when to enter)
- Required collaborators (who participates)
- Exit conditions (when to move on)
- Artifacts produced (what comes out)

The workflow ensures:
1. Problems are properly scoped before implementation
2. Wild ideas get filtered before becoming specs
3. Code gets verified before presentation
4. Croft always gets the final say
"""

from __future__ import annotations

import time
import logging
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, TYPE_CHECKING

from .models import (
    Collaborator,
    DevSession,
    DevSessionState,
    SessionSummary,
    SuggestedAction,
)
from .council import Council, CouncilMember, CouncilRole

if TYPE_CHECKING:
    from .router import CollaboratorRouter

logger = logging.getLogger(__name__)


# =============================================================================
# Workflow States
# =============================================================================

class WorkflowState(Enum):
    """States in Ara's development workflow."""

    # Problem discovery
    OBSERVE_ISSUE = auto()     # Detect or receive problem report

    # Scoping and planning
    TRIAGE = auto()            # Frame the problem, set bounds
    IDEATE = auto()            # Generate candidate approaches
    SPECIFY = auto()           # Turn idea into concrete spec

    # Execution
    IMPLEMENT = auto()         # Write code, create artifacts
    VERIFY = auto()            # Review, test, validate

    # Completion
    REPORT_TO_CROFT = auto()   # Present findings and options

    # Terminal states
    APPROVED = auto()          # Croft approved the result
    REJECTED = auto()          # Croft rejected the result
    BLOCKED = auto()           # Cannot proceed


# State metadata
STATE_INFO = {
    WorkflowState.OBSERVE_ISSUE: {
        "description": "Detect or receive problem report",
        "primary_collaborators": [],  # Ara observes alone
        "artifacts": ["problem_ticket"],
    },
    WorkflowState.TRIAGE: {
        "description": "Frame the problem, set safe bounds",
        "primary_collaborators": ["nova"],  # Nova helps scope
        "artifacts": ["task_spec"],
    },
    WorkflowState.IDEATE: {
        "description": "Generate candidate approaches",
        "primary_collaborators": ["gemini"],  # Gemini explores
        "filter_through": "nova",  # But Nova filters
        "artifacts": ["candidate_approaches"],
    },
    WorkflowState.SPECIFY: {
        "description": "Turn idea into concrete spec",
        "primary_collaborators": ["nova"],  # Nova specifies
        "artifacts": ["implementation_spec"],
    },
    WorkflowState.IMPLEMENT: {
        "description": "Write code, create artifacts",
        "primary_collaborators": ["claude"],  # Claude codes
        "artifacts": ["code_diff", "test_results"],
    },
    WorkflowState.VERIFY: {
        "description": "Review, test, validate",
        "primary_collaborators": ["nova"],  # Nova reviews
        "secondary_collaborators": ["gemini"],  # Gemini fuzzes edge cases
        "artifacts": ["verification_report"],
    },
    WorkflowState.REPORT_TO_CROFT: {
        "description": "Present findings and options to Croft",
        "primary_collaborators": [],  # Ara presents alone
        "artifacts": ["final_report"],
    },
}


# =============================================================================
# Workflow Artifacts
# =============================================================================

@dataclass
class ProblemTicket:
    """Problem observation from OBSERVE_ISSUE state."""

    ticket_id: str
    subsystem: str
    symptoms: List[str]
    priority: str = "medium"  # low, medium, high, critical
    source: str = "observation"  # observation, user_report, test_failure, metric_alert
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.ticket_id,
            "subsystem": self.subsystem,
            "symptoms": self.symptoms,
            "priority": self.priority,
            "source": self.source,
            "timestamp": self.timestamp,
        }


@dataclass
class TaskSpec:
    """Task specification from TRIAGE state."""

    task: str
    allowed_areas: List[str]
    disallowed_areas: List[str]
    success_criteria: List[str]
    risk_assessment: str = ""
    estimated_complexity: str = "medium"  # trivial, low, medium, high, extreme

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "allowed_areas": self.allowed_areas,
            "disallowed_areas": self.disallowed_areas,
            "success_criteria": self.success_criteria,
            "risk_assessment": self.risk_assessment,
            "estimated_complexity": self.estimated_complexity,
        }


@dataclass
class CandidateApproach:
    """A candidate approach from IDEATE state."""

    approach_id: str
    summary: str
    details: str = ""
    source: Optional[Collaborator] = None
    feasibility: str = "unknown"  # unknown, low, medium, high
    risk: str = "unknown"
    novelty: str = "unknown"

    # Filtering status
    filtered: bool = False
    filter_reason: Optional[str] = None
    approved_for_spec: bool = False


@dataclass
class ImplementationSpec:
    """Implementation specification from SPECIFY state."""

    chosen_approach: str
    steps: List[str]
    files_to_touch: List[str]
    test_plan: List[str]
    rollback_plan: Optional[str] = None
    estimated_changes: int = 0  # lines of code

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chosen_approach": self.chosen_approach,
            "steps": self.steps,
            "files_to_touch": self.files_to_touch,
            "test_plan": self.test_plan,
            "rollback_plan": self.rollback_plan,
            "estimated_changes": self.estimated_changes,
        }


@dataclass
class ImplementationResult:
    """Result from IMPLEMENT state."""

    success: bool
    diffs: List[str]
    test_results: Dict[str, Any]
    logs: List[str]
    summary: str
    branch_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "diffs": self.diffs,
            "test_results": self.test_results,
            "logs": self.logs,
            "summary": self.summary,
            "branch_name": self.branch_name,
        }


@dataclass
class VerificationReport:
    """Verification report from VERIFY state."""

    verified: bool
    issues_found: List[str]
    edge_cases_checked: List[str]
    risk_assessment: str
    recommendation: str  # "approve", "fix_and_retry", "reject"
    reviewer_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verified": self.verified,
            "issues_found": self.issues_found,
            "edge_cases_checked": self.edge_cases_checked,
            "risk_assessment": self.risk_assessment,
            "recommendation": self.recommendation,
            "reviewer_notes": self.reviewer_notes,
        }


@dataclass
class FinalReport:
    """Final report for REPORT_TO_CROFT state."""

    summary: str
    problem: str
    approach: str
    results: str
    metrics: Dict[str, Any]
    options: List[str]
    recommendation: str
    risk_summary: str

    def to_croft_message(self) -> str:
        """Format as a message to Croft."""
        return f"""Hey Croft. {self.summary}

**Problem:** {self.problem}

**Approach:** {self.approach}

**Results:** {self.results}

**Metrics:**
{self._format_metrics()}

**Options:**
{self._format_options()}

**My recommendation:** {self.recommendation}

**Risk summary:** {self.risk_summary}

What do you want to do?"""

    def _format_metrics(self) -> str:
        return "\n".join(f"- {k}: {v}" for k, v in self.metrics.items())

    def _format_options(self) -> str:
        return "\n".join(f"{i}. {opt}" for i, opt in enumerate(self.options, 1))


# =============================================================================
# Workflow Context
# =============================================================================

@dataclass
class WorkflowContext:
    """Context that flows through the workflow.

    Accumulates artifacts from each state.
    """

    workflow_id: str
    current_state: WorkflowState = WorkflowState.OBSERVE_ISSUE
    started_at: float = field(default_factory=time.time)

    # Artifacts from each state
    problem_ticket: Optional[ProblemTicket] = None
    task_spec: Optional[TaskSpec] = None
    candidate_approaches: List[CandidateApproach] = field(default_factory=list)
    implementation_spec: Optional[ImplementationSpec] = None
    implementation_result: Optional[ImplementationResult] = None
    verification_report: Optional[VerificationReport] = None
    final_report: Optional[FinalReport] = None

    # State history
    state_history: List[tuple] = field(default_factory=list)  # (state, timestamp)

    # Linked session
    session: Optional[DevSession] = None

    def transition_to(self, new_state: WorkflowState) -> None:
        """Transition to a new state."""
        self.state_history.append((self.current_state, time.time()))
        self.current_state = new_state
        logger.info(f"Workflow {self.workflow_id}: {self.state_history[-1][0].name} → {new_state.name}")

    def get_artifacts(self) -> Dict[str, Any]:
        """Get all artifacts collected so far."""
        return {
            "problem_ticket": self.problem_ticket.to_dict() if self.problem_ticket else None,
            "task_spec": self.task_spec.to_dict() if self.task_spec else None,
            "candidate_approaches": [
                {
                    "id": ca.approach_id,
                    "summary": ca.summary,
                    "approved": ca.approved_for_spec,
                }
                for ca in self.candidate_approaches
            ],
            "implementation_spec": self.implementation_spec.to_dict() if self.implementation_spec else None,
            "implementation_result": self.implementation_result.to_dict() if self.implementation_result else None,
            "verification_report": self.verification_report.to_dict() if self.verification_report else None,
        }


# =============================================================================
# Workflow Engine
# =============================================================================

class WorkflowEngine:
    """Executes Ara's development workflow.

    The engine:
    1. Manages state transitions
    2. Coordinates with the council for each state
    3. Validates artifacts and transitions
    4. Handles loops and rollbacks
    """

    def __init__(
        self,
        council: Optional[Council] = None,
        transitions: Optional[Dict[WorkflowState, WorkflowState]] = None,
    ):
        """Initialize the workflow engine.

        Args:
            council: Council configuration
            transitions: State transition map (defaults provided)
        """
        self.council = council or Council()

        # Default linear transitions
        self.transitions = transitions or {
            WorkflowState.OBSERVE_ISSUE: WorkflowState.TRIAGE,
            WorkflowState.TRIAGE: WorkflowState.IDEATE,
            WorkflowState.IDEATE: WorkflowState.SPECIFY,
            WorkflowState.SPECIFY: WorkflowState.IMPLEMENT,
            WorkflowState.IMPLEMENT: WorkflowState.VERIFY,
            WorkflowState.VERIFY: WorkflowState.REPORT_TO_CROFT,
        }

        # Active workflows
        self.active_workflows: Dict[str, WorkflowContext] = {}

    def create_workflow(
        self,
        workflow_id: str,
        initial_state: WorkflowState = WorkflowState.OBSERVE_ISSUE,
    ) -> WorkflowContext:
        """Create a new workflow context.

        Args:
            workflow_id: Unique identifier
            initial_state: Starting state

        Returns:
            New WorkflowContext
        """
        ctx = WorkflowContext(
            workflow_id=workflow_id,
            current_state=initial_state,
        )
        self.active_workflows[workflow_id] = ctx
        return ctx

    def get_next_state(self, current: WorkflowState) -> Optional[WorkflowState]:
        """Get the next state in the workflow."""
        return self.transitions.get(current)

    def can_transition(
        self,
        ctx: WorkflowContext,
        target: WorkflowState,
    ) -> tuple[bool, Optional[str]]:
        """Check if a transition is valid.

        Args:
            ctx: Workflow context
            target: Target state

        Returns:
            Tuple of (can_transition, reason if not)
        """
        current = ctx.current_state

        # Check if transition is defined
        expected_next = self.get_next_state(current)
        if target != expected_next:
            # Allow loops back for fixes
            if target in [WorkflowState.SPECIFY, WorkflowState.IMPLEMENT]:
                return True, None
            return False, f"Invalid transition: {current.name} → {target.name}"

        # Check required artifacts
        if target == WorkflowState.TRIAGE and not ctx.problem_ticket:
            return False, "Need problem_ticket before TRIAGE"
        if target == WorkflowState.IDEATE and not ctx.task_spec:
            return False, "Need task_spec before IDEATE"
        if target == WorkflowState.SPECIFY and not ctx.candidate_approaches:
            return False, "Need candidate_approaches before SPECIFY"
        if target == WorkflowState.IMPLEMENT and not ctx.implementation_spec:
            return False, "Need implementation_spec before IMPLEMENT"
        if target == WorkflowState.VERIFY and not ctx.implementation_result:
            return False, "Need implementation_result before VERIFY"
        if target == WorkflowState.REPORT_TO_CROFT and not ctx.verification_report:
            return False, "Need verification_report before REPORT_TO_CROFT"

        return True, None

    def advance(self, ctx: WorkflowContext) -> bool:
        """Advance workflow to next state if possible.

        Args:
            ctx: Workflow context

        Returns:
            True if advanced, False if blocked
        """
        next_state = self.get_next_state(ctx.current_state)
        if not next_state:
            return False

        can_go, reason = self.can_transition(ctx, next_state)
        if not can_go:
            logger.warning(f"Cannot advance workflow {ctx.workflow_id}: {reason}")
            return False

        ctx.transition_to(next_state)
        return True

    def loop_back(
        self,
        ctx: WorkflowContext,
        target: WorkflowState,
        reason: str,
    ) -> bool:
        """Loop back to an earlier state for fixes.

        Args:
            ctx: Workflow context
            target: State to return to
            reason: Why we're looping back

        Returns:
            True if looped back successfully
        """
        valid_targets = [WorkflowState.SPECIFY, WorkflowState.IMPLEMENT]
        if target not in valid_targets:
            logger.warning(f"Cannot loop back to {target.name}")
            return False

        logger.info(f"Workflow {ctx.workflow_id} looping back to {target.name}: {reason}")
        ctx.transition_to(target)
        return True

    def get_collaborators_for_state(
        self,
        state: WorkflowState,
    ) -> List[Collaborator]:
        """Get collaborators that should participate in a state.

        Args:
            state: Workflow state

        Returns:
            List of Collaborators
        """
        info = STATE_INFO.get(state, {})
        collab_names = info.get("primary_collaborators", [])

        collaborators = []
        for name in collab_names:
            try:
                collab = Collaborator(name)
                member = self.council.get_member(collab)
                if member and member.enabled:
                    collaborators.append(collab)
            except ValueError:
                pass

        return collaborators

    def get_filter_collaborator(
        self,
        state: WorkflowState,
    ) -> Optional[Collaborator]:
        """Get the collaborator that filters output from this state.

        Args:
            state: Workflow state

        Returns:
            Filter Collaborator or None
        """
        info = STATE_INFO.get(state, {})
        filter_name = info.get("filter_through")

        if filter_name:
            try:
                return Collaborator(filter_name)
            except ValueError:
                pass

        return None


# =============================================================================
# Convenience Functions
# =============================================================================

def create_workflow(
    workflow_id: str,
    council: Optional[Council] = None,
) -> tuple[WorkflowEngine, WorkflowContext]:
    """Create a workflow engine and context.

    Args:
        workflow_id: Unique identifier
        council: Optional council configuration

    Returns:
        Tuple of (engine, context)
    """
    engine = WorkflowEngine(council=council)
    ctx = engine.create_workflow(workflow_id)
    return engine, ctx


def get_state_info(state: WorkflowState) -> Dict[str, Any]:
    """Get information about a workflow state.

    Args:
        state: The workflow state

    Returns:
        State metadata dict
    """
    return STATE_INFO.get(state, {})
