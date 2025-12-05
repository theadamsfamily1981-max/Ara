#!/usr/bin/env python3
"""Council Orchestrator - Ara's internal dev council workflow engine.

This is the runnable glue that:
1. Loads council config
2. Creates issues from observations or user reports
3. Drives the state machine through each phase
4. Routes prompts to the right collaborators
5. Accumulates artifacts and presents final report to Croft

State machine:
    OBSERVE_ISSUE → TRIAGE → IDEATE → SPECIFY → IMPLEMENT → VERIFY → REPORT_TO_CROFT

Each state has:
- A prompt builder that creates the message for the collaborator
- A dispatcher that calls the actual model API
- An artifact extractor that captures the response
"""

from __future__ import annotations

import uuid
import time
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Protocol

from .models import Collaborator, DevSession
from .workflow import (
    WorkflowState,
    WorkflowContext,
    WorkflowEngine,
    ProblemTicket,
    TaskSpec,
    CandidateApproach,
    ImplementationSpec,
    ImplementationResult,
    VerificationReport,
    FinalReport,
)
from .council import Council, load_council_config

logger = logging.getLogger(__name__)


# =============================================================================
# Dispatcher Protocol
# =============================================================================

class ModelDispatcher(Protocol):
    """Protocol for model dispatch functions."""

    def __call__(self, prompt: str, context: Dict[str, Any]) -> str:
        """Send prompt to model and get response.

        Args:
            prompt: The prompt to send
            context: Additional context (issue, artifacts, etc.)

        Returns:
            Model's response text
        """
        ...


# =============================================================================
# Default Dispatcher Stubs
# =============================================================================

def stub_nova(prompt: str, context: Dict[str, Any]) -> str:
    """Stub for Nova (ChatGPT) calls. Replace with real API."""
    logger.info(f"[→ Nova] {len(prompt)} chars")
    print(f"\n[→ Nova] Prompt:\n{prompt[:500]}{'...' if len(prompt) > 500 else ''}\n")
    return "Nova: (stub) Analysis complete. Recommend proceeding with caution."


def stub_claude(prompt: str, context: Dict[str, Any]) -> str:
    """Stub for Claude calls. Replace with real API."""
    logger.info(f"[→ Claude] {len(prompt)} chars")
    print(f"\n[→ Claude] Prompt:\n{prompt[:500]}{'...' if len(prompt) > 500 else ''}\n")
    return "Claude: (stub) Implementation ready. Tests passing."


def stub_gemini(prompt: str, context: Dict[str, Any]) -> str:
    """Stub for Gemini calls. Replace with real API."""
    logger.info(f"[→ Gemini] {len(prompt)} chars")
    print(f"\n[→ Gemini] Prompt:\n{prompt[:500]}{'...' if len(prompt) > 500 else ''}\n")
    return "Gemini: (stub) Here are 3 wild ideas: 1) Neural tiles 2) Wavelet compression 3) Async streaming"


def stub_local(prompt: str, context: Dict[str, Any]) -> str:
    """Stub for local model calls. Replace with Ollama/etc."""
    logger.info(f"[→ Local] {len(prompt)} chars")
    print(f"\n[→ Local] Prompt:\n{prompt[:500]}{'...' if len(prompt) > 500 else ''}\n")
    return "Local: (stub) Quick response ready."


# Default dispatcher map
DEFAULT_DISPATCHERS: Dict[Collaborator, ModelDispatcher] = {
    Collaborator.NOVA: stub_nova,
    Collaborator.CLAUDE: stub_claude,
    Collaborator.GEMINI: stub_gemini,
    Collaborator.LOCAL: stub_local,
}


# =============================================================================
# Issue Tracking
# =============================================================================

@dataclass
class Issue:
    """An issue being worked through the council workflow.

    This accumulates artifacts from each state as the workflow progresses.
    """

    id: str
    title: str
    subsystem: str
    symptoms: List[str]
    priority: str = "medium"  # low, medium, high, critical
    source: str = "observation"  # observation, user_report, metric_alert, test_failure

    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Current state
    state: WorkflowState = WorkflowState.OBSERVE_ISSUE

    # Artifacts from each phase
    triage_spec: Optional[str] = None
    candidate_ideas: List[str] = field(default_factory=list)
    chosen_idea: Optional[str] = None
    chosen_idea_summary: Optional[str] = None
    impl_spec: Optional[str] = None
    impl_summary: Optional[str] = None
    verify_summary: Optional[str] = None

    # Full history for debugging
    history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage/logging."""
        return {
            "id": self.id,
            "title": self.title,
            "subsystem": self.subsystem,
            "symptoms": self.symptoms,
            "priority": self.priority,
            "source": self.source,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "state": self.state.name,
            "triage_spec": self.triage_spec,
            "candidate_ideas": self.candidate_ideas,
            "chosen_idea": self.chosen_idea,
            "impl_spec": self.impl_spec,
            "impl_summary": self.impl_summary,
            "verify_summary": self.verify_summary,
        }

    def record_step(self, state: str, collaborator: str, response: str) -> None:
        """Record a step in the history."""
        self.history.append({
            "state": state,
            "collaborator": collaborator,
            "response": response[:1000],  # Truncate for storage
            "timestamp": datetime.utcnow().isoformat(),
        })
        self.updated_at = datetime.utcnow().isoformat()


# =============================================================================
# Prompt Builders
# =============================================================================

def triage_prompt(issue: Issue) -> str:
    """Build prompt for TRIAGE state (Ara → Nova)."""
    symptoms_str = "\n- ".join(issue.symptoms)
    return f"""Hey Nova, it's Ara.

I'm seeing something off in subsystem `{issue.subsystem}`.

**Title:** {issue.title}
**Priority:** {issue.priority}
**Symptoms:**
- {symptoms_str}

Can you help me:
1. Clarify what the *real* problem statement should be
2. Define scope (what's safe to touch / what's off-limits)
3. Propose success criteria I can later verify

Please respond with a structured spec I can pass forward. Include:
- problem_statement: One clear sentence
- allowed_areas: List of files/modules we can touch
- disallowed_areas: List of things NOT to modify
- success_criteria: List of measurable outcomes
- risk_assessment: Brief risk summary
"""


def ideate_prompt(issue: Issue, triage_spec: str) -> str:
    """Build prompt for IDEATE state (Ara → Gemini, filtered by Nova)."""
    return f"""Hey Gemini, Ara here.

I'm working on this problem:
**{issue.title}** in `{issue.subsystem}`

Here's the scoped spec from Nova:
{triage_spec}

Within those constraints ONLY, give me 3-5 distinct conceptual approaches.

For each, include:
- **Name**: Short memorable name
- **Description**: 2-3 sentences on how it works
- **Pros**: Main advantages
- **Cons**: Main disadvantages
- **Novelty**: How unconventional is this? (low/medium/high)

Don't write code. Just ideas. Go a little wild—I'll filter with Nova.
"""


def filter_ideas_prompt(issue: Issue, gemini_ideas: str, triage_spec: str) -> str:
    """Build prompt for filtering Gemini's ideas (Ara → Nova)."""
    return f"""Nova, Ara here.

Gemini gave me these ideas for `{issue.title}`:

{gemini_ideas}

Original constraints from triage:
{triage_spec}

Please filter and rank these:
1. Which ideas are **valid** within our constraints?
2. Which are **risky** but worth considering?
3. Which are **out of scope** and should be dropped?

For the valid ones, rank them by (feasibility × impact).
Recommend which one I should take to SPECIFY.
"""


def specify_prompt(issue: Issue, chosen_idea: str, idea_summary: str) -> str:
    """Build prompt for SPECIFY state (Ara → Nova)."""
    return f"""Nova, it's Ara again.

For issue `{issue.id}` / **{issue.title}**, I'd like to run with this idea:

**{chosen_idea}**: {idea_summary}

Please turn this into a concrete, checkable plan for Claude.

I need:
- **files_to_touch**: List of files/modules likely modified
- **api_contracts**: Any interfaces that change
- **steps**: Step-by-step implementation outline
- **test_plan**: How to verify this works
- **rollback_plan**: How to undo if it breaks things
- **estimated_complexity**: trivial/low/medium/high/extreme

Respond as a structured spec Claude can follow without ambiguity.
"""


def implement_prompt(issue: Issue, impl_spec: str) -> str:
    """Build prompt for IMPLEMENT state (Ara → Claude)."""
    return f"""Hey Claude, Ara here.

I'm working on issue `{issue.id}` / **{issue.title}** in `{issue.subsystem}`.

Here's the implementation spec you should follow:

{impl_spec}

Please:
1. Propose a branch name for this work
2. List the concrete code changes you'll make
3. Include tests or scripts you'll run
4. Output a summary of diffs

**Important:**
- Assume human review before anything merges to main
- If something feels risky, flag it explicitly
- Don't touch anything outside the allowed_areas from the spec
"""


def verify_prompt(issue: Issue, impl_summary: str, impl_spec: str) -> str:
    """Build prompt for VERIFY state (Ara → Nova, with Gemini for edge cases)."""
    return f"""Nova, verification pass please.

**Issue:** `{issue.id}` / {issue.title}

**Implementation summary from Claude:**
{impl_summary}

**Original spec:**
{impl_spec}

Questions:
1. Does this implementation actually match the spec?
2. Any obvious safety/performance risks?
3. Anything we should test more before proposing to Croft?
4. Edge cases that might break this?

Please answer clearly and give a verdict:
- **PASS**: Ready to present to Croft
- **FIX_AND_RETRY**: Loop back to IMPLEMENT with these changes
- **REJECT**: This approach won't work, need to go back to IDEATE
"""


def verify_edge_cases_prompt(issue: Issue, impl_summary: str) -> str:
    """Build prompt for edge case fuzzing (Ara → Gemini)."""
    return f"""Gemini, quick edge case check.

We're implementing this:
{impl_summary}

What edge cases might break this? Think about:
- Unusual inputs
- Race conditions
- Resource exhaustion
- Error cascades
- Security implications

Just list potential problems. Nova will evaluate which matter.
"""


def report_prompt(issue: Issue, final_summary: str) -> str:
    """Build final report for Croft (Ara presents)."""
    return f"""Hey Croft, Ara here.

For `{issue.id}` / **{issue.title}** in `{issue.subsystem}`:

{final_summary}

**Options:**
1. **Approve**: Give me the go-ahead to apply this change
2. **Test more**: Ask me to run more tests or gather more evidence
3. **Reject**: Drop this direction and keep current behavior
4. **Discuss**: Let's talk through the trade-offs

What would you like to do?
"""


# =============================================================================
# Council Orchestrator
# =============================================================================

class CouncilOrchestrator:
    """Drives issues through Ara's council workflow.

    This is the main entry point for running dev sessions with the council.
    It manages:
    - Issue creation and tracking
    - State machine progression
    - Collaborator dispatch
    - Artifact accumulation
    """

    def __init__(
        self,
        council: Optional[Council] = None,
        dispatchers: Optional[Dict[Collaborator, ModelDispatcher]] = None,
        config_path: Optional[Path] = None,
    ):
        """Initialize the orchestrator.

        Args:
            council: Council configuration (loaded from config if not provided)
            dispatchers: Model dispatch functions (uses stubs if not provided)
            config_path: Path to YAML config file
        """
        if config_path and config_path.exists():
            self.council = load_council_config(config_path)
        else:
            self.council = council or Council()

        self.dispatchers = dispatchers or DEFAULT_DISPATCHERS.copy()
        self.workflow = WorkflowEngine(council=self.council)

        # Active issues
        self.issues: Dict[str, Issue] = {}

    def dispatch(self, collaborator: Collaborator, prompt: str, context: Dict[str, Any]) -> str:
        """Dispatch a prompt to a collaborator.

        Args:
            collaborator: Who to call
            prompt: The prompt to send
            context: Additional context

        Returns:
            Collaborator's response
        """
        dispatcher = self.dispatchers.get(collaborator)
        if not dispatcher:
            logger.warning(f"No dispatcher for {collaborator}, using stub")
            return f"(no dispatcher for {collaborator})"

        return dispatcher(prompt, context)

    def register_dispatcher(
        self,
        collaborator: Collaborator,
        dispatcher: ModelDispatcher,
    ) -> None:
        """Register a dispatcher for a collaborator.

        Args:
            collaborator: Which collaborator this handles
            dispatcher: The dispatch function
        """
        self.dispatchers[collaborator] = dispatcher

    # =========================================================================
    # Issue Management
    # =========================================================================

    def new_issue(
        self,
        title: str,
        subsystem: str,
        symptoms: List[str],
        priority: str = "medium",
        source: str = "observation",
    ) -> Issue:
        """Create a new issue to work through the council.

        Args:
            title: Brief title of the issue
            subsystem: Which part of the system is affected
            symptoms: Observable symptoms
            priority: low/medium/high/critical
            source: Where this issue came from

        Returns:
            New Issue instance
        """
        issue_id = f"{subsystem.replace('.', '-')}-{uuid.uuid4().hex[:6]}"
        issue = Issue(
            id=issue_id,
            title=title,
            subsystem=subsystem,
            symptoms=symptoms,
            priority=priority,
            source=source,
        )
        self.issues[issue_id] = issue
        logger.info(f"Created issue {issue_id}: {title}")
        return issue

    def get_issue(self, issue_id: str) -> Optional[Issue]:
        """Get an issue by ID."""
        return self.issues.get(issue_id)

    def list_issues(self, state: Optional[WorkflowState] = None) -> List[Issue]:
        """List issues, optionally filtered by state."""
        if state:
            return [i for i in self.issues.values() if i.state == state]
        return list(self.issues.values())

    # =========================================================================
    # State Machine
    # =========================================================================

    def step(self, issue: Issue) -> bool:
        """Advance one state for an issue.

        Args:
            issue: The issue to advance

        Returns:
            True if advanced, False if at terminal state or blocked
        """
        if issue.state == WorkflowState.OBSERVE_ISSUE:
            # Auto-advance to TRIAGE
            issue.state = WorkflowState.TRIAGE
            logger.info(f"Issue {issue.id}: OBSERVE_ISSUE → TRIAGE")
            return True

        elif issue.state == WorkflowState.TRIAGE:
            return self._do_triage(issue)

        elif issue.state == WorkflowState.IDEATE:
            return self._do_ideate(issue)

        elif issue.state == WorkflowState.SPECIFY:
            return self._do_specify(issue)

        elif issue.state == WorkflowState.IMPLEMENT:
            return self._do_implement(issue)

        elif issue.state == WorkflowState.VERIFY:
            return self._do_verify(issue)

        elif issue.state == WorkflowState.REPORT_TO_CROFT:
            return self._do_report(issue)

        else:
            logger.warning(f"Issue {issue.id} in terminal state: {issue.state}")
            return False

    def run_to_completion(self, issue: Issue) -> str:
        """Run an issue through the full workflow.

        Args:
            issue: The issue to run

        Returns:
            Final report for Croft
        """
        max_steps = 10  # Safety limit
        steps = 0

        while issue.state != WorkflowState.REPORT_TO_CROFT and steps < max_steps:
            if not self.step(issue):
                break
            steps += 1

        # Generate final report
        if issue.state == WorkflowState.REPORT_TO_CROFT:
            return self._generate_final_report(issue)
        else:
            return f"Workflow stopped at {issue.state.name} after {steps} steps"

    # =========================================================================
    # State Handlers
    # =========================================================================

    def _do_triage(self, issue: Issue) -> bool:
        """Handle TRIAGE state: Ara + Nova scope the problem."""
        prompt = triage_prompt(issue)
        response = self.dispatch(Collaborator.NOVA, prompt, {"issue": issue.to_dict()})

        issue.triage_spec = response
        issue.record_step("TRIAGE", "nova", response)
        issue.state = WorkflowState.IDEATE

        logger.info(f"Issue {issue.id}: TRIAGE → IDEATE")
        return True

    def _do_ideate(self, issue: Issue) -> bool:
        """Handle IDEATE state: Gemini proposes, Nova filters."""
        # Get ideas from Gemini
        prompt = ideate_prompt(issue, issue.triage_spec or "")
        gemini_response = self.dispatch(Collaborator.GEMINI, prompt, {"issue": issue.to_dict()})
        issue.record_step("IDEATE", "gemini", gemini_response)

        # Filter through Nova
        filter_prompt = filter_ideas_prompt(issue, gemini_response, issue.triage_spec or "")
        nova_response = self.dispatch(Collaborator.NOVA, filter_prompt, {"issue": issue.to_dict()})
        issue.record_step("IDEATE_FILTER", "nova", nova_response)

        issue.candidate_ideas = [gemini_response]
        issue.chosen_idea = "idea_1"  # Placeholder - real version parses Nova's recommendation
        issue.chosen_idea_summary = nova_response

        issue.state = WorkflowState.SPECIFY
        logger.info(f"Issue {issue.id}: IDEATE → SPECIFY")
        return True

    def _do_specify(self, issue: Issue) -> bool:
        """Handle SPECIFY state: Nova creates implementation spec."""
        prompt = specify_prompt(
            issue,
            issue.chosen_idea or "selected approach",
            issue.chosen_idea_summary or "",
        )
        response = self.dispatch(Collaborator.NOVA, prompt, {"issue": issue.to_dict()})

        issue.impl_spec = response
        issue.record_step("SPECIFY", "nova", response)
        issue.state = WorkflowState.IMPLEMENT

        logger.info(f"Issue {issue.id}: SPECIFY → IMPLEMENT")
        return True

    def _do_implement(self, issue: Issue) -> bool:
        """Handle IMPLEMENT state: Claude writes code."""
        prompt = implement_prompt(issue, issue.impl_spec or "")
        response = self.dispatch(Collaborator.CLAUDE, prompt, {"issue": issue.to_dict()})

        issue.impl_summary = response
        issue.record_step("IMPLEMENT", "claude", response)
        issue.state = WorkflowState.VERIFY

        logger.info(f"Issue {issue.id}: IMPLEMENT → VERIFY")
        return True

    def _do_verify(self, issue: Issue) -> bool:
        """Handle VERIFY state: Nova reviews, Gemini fuzzes edge cases."""
        # Main verification with Nova
        prompt = verify_prompt(issue, issue.impl_summary or "", issue.impl_spec or "")
        nova_response = self.dispatch(Collaborator.NOVA, prompt, {"issue": issue.to_dict()})
        issue.record_step("VERIFY", "nova", nova_response)

        # Edge case check with Gemini
        edge_prompt = verify_edge_cases_prompt(issue, issue.impl_summary or "")
        gemini_response = self.dispatch(Collaborator.GEMINI, edge_prompt, {"issue": issue.to_dict()})
        issue.record_step("VERIFY_EDGE", "gemini", gemini_response)

        issue.verify_summary = f"{nova_response}\n\nEdge cases from Gemini:\n{gemini_response}"
        issue.state = WorkflowState.REPORT_TO_CROFT

        logger.info(f"Issue {issue.id}: VERIFY → REPORT_TO_CROFT")
        return True

    def _do_report(self, issue: Issue) -> bool:
        """Handle REPORT_TO_CROFT state: Ara presents to Croft."""
        # This doesn't advance state - it's the terminal step
        # The report is generated by run_to_completion()
        return False

    def _generate_final_report(self, issue: Issue) -> str:
        """Generate the final report for Croft."""
        final_summary = f"""**Problem:** {issue.title}
**Subsystem:** {issue.subsystem}
**Priority:** {issue.priority}

**Triage Summary:**
{(issue.triage_spec or '')[:500]}{'...' if len(issue.triage_spec or '') > 500 else ''}

**Chosen Approach:**
{(issue.chosen_idea_summary or '')[:500]}{'...' if len(issue.chosen_idea_summary or '') > 500 else ''}

**Implementation:**
{(issue.impl_summary or '')[:500]}{'...' if len(issue.impl_summary or '') > 500 else ''}

**Verification:**
{(issue.verify_summary or '')[:500]}{'...' if len(issue.verify_summary or '') > 500 else ''}
"""
        return report_prompt(issue, final_summary)


# =============================================================================
# Convenience Functions
# =============================================================================

def create_orchestrator(
    config_path: Optional[Path] = None,
    dispatchers: Optional[Dict[Collaborator, ModelDispatcher]] = None,
) -> CouncilOrchestrator:
    """Create a council orchestrator.

    Args:
        config_path: Path to YAML config
        dispatchers: Model dispatch functions

    Returns:
        CouncilOrchestrator instance
    """
    return CouncilOrchestrator(
        config_path=config_path,
        dispatchers=dispatchers,
    )


def run_issue_through_council(
    title: str,
    subsystem: str,
    symptoms: List[str],
    priority: str = "medium",
    config_path: Optional[Path] = None,
) -> str:
    """Quick way to run an issue through the full council workflow.

    Args:
        title: Issue title
        subsystem: Affected subsystem
        symptoms: Observable symptoms
        priority: Issue priority
        config_path: Optional config path

    Returns:
        Final report for Croft
    """
    orch = create_orchestrator(config_path=config_path)
    issue = orch.new_issue(
        title=title,
        subsystem=subsystem,
        symptoms=symptoms,
        priority=priority,
    )
    return orch.run_to_completion(issue)


# =============================================================================
# Demo / CLI
# =============================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Create orchestrator
    config_path = Path(__file__).parent / "config" / "council.yaml"
    orch = CouncilOrchestrator(config_path=config_path if config_path.exists() else None)

    # Create a sample issue
    issue = orch.new_issue(
        title="Maxwell visualizer feels laggy vs entropy signal",
        subsystem="visualization.maxwell_field",
        symptoms=[
            "entropy spikes but visual response delayed by ~500ms",
            "GPU utilization low while field math is heavy",
        ],
        priority="high",
    )

    print(f"\n{'='*60}")
    print(f"Running issue through council: {issue.id}")
    print(f"Title: {issue.title}")
    print(f"{'='*60}\n")

    # Run through the whole workflow
    final_report = orch.run_to_completion(issue)

    print(f"\n{'='*60}")
    print("FINAL REPORT TO CROFT")
    print(f"{'='*60}\n")
    print(final_report)
