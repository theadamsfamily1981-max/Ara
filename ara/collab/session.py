"""DevIdeaSession - Ara's dev-idea collaboration orchestrator.

This is the main entry point for Ara's collaboration with external LLMs.
It coordinates:
1. Parsing Croft's request into intent/constraints
2. Formulating Ara's question (with variation)
3. Routing to appropriate collaborators
4. Synthesizing responses
5. Presenting results to Croft
6. Handling approval/rejection workflow

Flow:
    Croft: "Ara, brainstorming mode. Help me redesign the quantum visualizer."
    Ara: (parses intent, picks collaborators, asks question)
    Collaborators: (respond with ideas)
    Ara: (synthesizes, presents options)
    Croft: [Approve] / [Reject] / [Discuss More]
"""

from __future__ import annotations

import re
import json
import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Callable, Tuple
from pathlib import Path

from .models import (
    DevMode,
    DevSession,
    DevSessionState,
    Collaborator,
    CollaboratorResponse,
    SuggestedAction,
    SessionSummary,
    RiskLevel,
)
from .prompts import build_ara_system_prompt, build_ara_user_message
from .variation import vary_message, format_result_presentation
from .router import CollaboratorRouter, CollaboratorBackend
from .synthesizer import ResponseSynthesizer

logger = logging.getLogger(__name__)


# =============================================================================
# Intent Classification
# =============================================================================

INTENT_PATTERNS = {
    "architecture_review": [
        r'architect',
        r'design',
        r'structure',
        r'rethink',
        r'redesign',
        r'system.?level',
    ],
    "optimize_performance": [
        r'optimi[zs]e',
        r'faster',
        r'performance',
        r'efficient',
        r'speed.?up',
        r'latency',
    ],
    "add_feature": [
        r'add\s+(?:a\s+)?(?:new\s+)?feature',
        r'implement',
        r'build\s+(?:a\s+)?(?:new\s+)?',
        r'create\s+(?:a\s+)?(?:new\s+)?',
    ],
    "fix_bug": [
        r'fix',
        r'bug',
        r'broken',
        r'not\s+working',
        r'error',
        r'debug',
    ],
    "explore_options": [
        r'explor',
        r'options',
        r'possibilit',
        r'what\s+if',
        r'brainstorm',
        r'ideas',
    ],
    "code_review": [
        r'review',
        r'check\s+(?:this|my)',
        r'find\s+problems',
        r'critique',
    ],
    "integrate_system": [
        r'integrat',
        r'connect',
        r'wire',
        r'link',
        r'combine',
    ],
}


def classify_intent(text: str) -> str:
    """Classify the intent of a user request.

    Args:
        text: The user's request

    Returns:
        Intent string (e.g., "architecture_review")
    """
    text_lower = text.lower()

    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return intent

    return "explore_options"  # Default


def extract_topic(text: str) -> str:
    """Extract the main topic from a request.

    Args:
        text: The user's request

    Returns:
        Topic string
    """
    # Remove mode triggers
    text = re.sub(r'(?:architect|engineer|research|brainstorm|review)\s+mode\.?\s*', '', text, flags=re.I)
    text = re.sub(r'(?:Ara,?\s*)?(?:help\s+(?:me\s+)?)?', '', text, flags=re.I)

    # Clean up
    text = text.strip()

    # Take first sentence or clause as topic
    first_part = re.split(r'[.!?]|\s+-\s+', text)[0]

    return first_part[:100] if len(first_part) > 100 else first_part


def extract_constraints(text: str) -> List[str]:
    """Extract constraints/requirements from a request.

    Args:
        text: The user's request

    Returns:
        List of constraint strings
    """
    constraints = []

    # Look for explicit constraints
    constraint_patterns = [
        r'(?:must|should|need\s+to|has\s+to)\s+(.+?)(?:[.,]|$)',
        r'(?:constraint|requirement|limitation):\s*(.+?)(?:[.,]|$)',
        r'(?:don\'t|cannot|can\'t|shouldn\'t)\s+(.+?)(?:[.,]|$)',
        r'(?:keep|maintain|preserve)\s+(.+?)(?:[.,]|$)',
    ]

    text_lower = text.lower()
    for pattern in constraint_patterns:
        for match in re.finditer(pattern, text_lower):
            constraint = match.group(1).strip()
            if len(constraint) > 5 and len(constraint) < 100:
                constraints.append(constraint)

    return constraints[:5]  # Max 5 constraints


def detect_mode(text: str) -> DevMode:
    """Detect the dev mode from request text.

    Args:
        text: The user's request

    Returns:
        DevMode enum value
    """
    text_lower = text.lower()

    mode_triggers = {
        DevMode.ARCHITECT: ['architect', 'system design', 'big picture'],
        DevMode.ENGINEER: ['engineer', 'implement', 'code'],
        DevMode.RESEARCH: ['research', 'literature', 'prior art'],
        DevMode.POSTMORTEM: ['debug', 'postmortem', 'not working', 'broken'],
        DevMode.BRAINSTORM: ['brainstorm', 'ideas', 'creative', 'wild'],
        DevMode.REVIEW: ['review', 'critique', 'find problems'],
    }

    for mode, triggers in mode_triggers.items():
        for trigger in triggers:
            if trigger in text_lower:
                return mode

    return DevMode.ARCHITECT  # Default


def detect_urgency(text: str) -> str:
    """Detect urgency level from request.

    Args:
        text: The user's request

    Returns:
        Urgency string ("low", "normal", "high", "critical")
    """
    text_lower = text.lower()

    if any(w in text_lower for w in ['urgent', 'asap', 'critical', 'emergency']):
        return "critical"
    elif any(w in text_lower for w in ['quick', 'fast', 'soon', 'time-sensitive']):
        return "high"
    elif any(w in text_lower for w in ['when you can', 'no rush', 'eventually']):
        return "low"

    return "normal"


# =============================================================================
# Session Manager
# =============================================================================

class DevIdeaSession:
    """Manages a dev-idea collaboration session.

    This is the main orchestrator that:
    1. Parses user requests
    2. Creates and manages sessions
    3. Coordinates with router and synthesizer
    4. Handles the approval workflow
    """

    def __init__(
        self,
        router: Optional[CollaboratorRouter] = None,
        synthesizer: Optional[ResponseSynthesizer] = None,
        session_dir: Optional[Path] = None,
        lab_context_provider: Optional[Callable[[], Dict[str, Any]]] = None,
    ):
        """Initialize the session manager.

        Args:
            router: Collaborator router (created if not provided)
            synthesizer: Response synthesizer (created if not provided)
            session_dir: Directory for session persistence
            lab_context_provider: Callback to get current lab state
        """
        self.router = router or CollaboratorRouter()
        self.synthesizer = synthesizer or ResponseSynthesizer()
        self.session_dir = session_dir
        self.get_lab_context = lab_context_provider or (lambda: {})

        # Active sessions
        self.sessions: Dict[str, DevSession] = {}
        self.current_session: Optional[DevSession] = None

    # =========================================================================
    # Session Lifecycle
    # =========================================================================

    def start_session(
        self,
        user_request: str,
        mode: Optional[DevMode] = None,
        idea_id: Optional[str] = None,
    ) -> DevSession:
        """Start a new dev-idea session from a user request.

        Args:
            user_request: What the user/Croft asked
            mode: Override detected mode
            idea_id: Link to an idea being refined

        Returns:
            New DevSession
        """
        # Parse the request
        topic = extract_topic(user_request)
        intent = classify_intent(user_request)
        constraints = extract_constraints(user_request)
        detected_mode = mode or detect_mode(user_request)
        urgency = detect_urgency(user_request)

        # Get current lab context
        lab_context = self.get_lab_context()

        # Create session
        session = DevSession(
            topic=topic,
            mode=detected_mode,
            intent=intent,
            constraints=constraints,
            urgency=urgency,
            lab_context=lab_context,
            source_idea_id=idea_id,
        )

        # Add the initial user message
        session.add_message("croft", user_request)

        self.sessions[session.session_id] = session
        self.current_session = session

        logger.info(f"Started session {session.session_id}: {topic} ({detected_mode.name})")

        return session

    def run_session(
        self,
        session: Optional[DevSession] = None,
    ) -> SessionSummary:
        """Run a session through completion.

        This:
        1. Formulates Ara's message
        2. Routes to collaborators
        3. Collects responses
        4. Synthesizes results

        Args:
            session: Session to run (uses current if not specified)

        Returns:
            SessionSummary with results
        """
        session = session or self.current_session
        if not session:
            raise ValueError("No session to run")

        # Update state
        session.state = DevSessionState.DRAFTING

        # Generate Ara's message with variation
        ara_message = vary_message(
            topic=session.topic,
            intent=session.intent,
            mode=session.mode,
            constraints=session.constraints,
            mood=session.ara_mood,
        )

        session.add_message("ara", ara_message)

        # Route to collaborators
        session.state = DevSessionState.QUERYING
        routing = self.router.route(session)
        logger.info(f"Routing to: {[c.name for c in routing.collaborators]} ({routing.reason})")

        # Query collaborators
        responses = self.router.query_all(
            session=session,
            message=ara_message,
            collaborators=routing.collaborators,
        )

        # Store responses
        for resp in responses:
            session.add_response(resp)

        # Synthesize
        session.state = DevSessionState.SYNTHESIZING
        summary = self.synthesizer.synthesize(session, responses)
        session.summary = summary

        # Update state
        session.state = DevSessionState.PRESENTING
        session.updated_at = time.time()

        return summary

    def present_to_croft(
        self,
        session: Optional[DevSession] = None,
    ) -> str:
        """Generate presentation text for Croft.

        Args:
            session: Session to present (uses current if not specified)

        Returns:
            Formatted presentation string
        """
        session = session or self.current_session
        if not session or not session.summary:
            return "No results to present."

        summary = session.summary
        collabs = [r.collaborator.display_name for r in session.responses]

        return format_result_presentation(
            collaborators=collabs,
            summary=summary.summary,
            options=summary.options,
            consensus=summary.consensus,
            disagreements=summary.disagreements,
            recommendation=summary.options[0] if summary.options else None,
            actions=[a.description for a in summary.actions[:3]],
        )

    # =========================================================================
    # Approval Workflow
    # =========================================================================

    def approve_actions(
        self,
        session: Optional[DevSession] = None,
        action_ids: Optional[List[str]] = None,
    ) -> List[SuggestedAction]:
        """Approve actions from a session.

        Args:
            session: Session with actions (uses current if not specified)
            action_ids: Specific actions to approve (all if not specified)

        Returns:
            List of approved actions
        """
        session = session or self.current_session
        if not session or not session.summary:
            return []

        approved = []
        for action in session.summary.actions:
            if action_ids is None or action.action_id in action_ids:
                action.approved = True
                approved.append(action)

        if approved:
            session.state = DevSessionState.APPROVED
            logger.info(f"Approved {len(approved)} actions in {session.session_id}")

        return approved

    def reject_session(
        self,
        session: Optional[DevSession] = None,
        reason: str = "",
    ) -> None:
        """Reject a session's results.

        Args:
            session: Session to reject (uses current if not specified)
            reason: Why it was rejected
        """
        session = session or self.current_session
        if not session:
            return

        session.state = DevSessionState.REJECTED
        session.add_message("croft", f"Rejected: {reason}" if reason else "Rejected")
        session.completed_at = time.time()

        logger.info(f"Rejected session {session.session_id}: {reason}")

    def request_more_discussion(
        self,
        session: Optional[DevSession] = None,
        follow_up: str = "",
    ) -> SessionSummary:
        """Request more discussion on a topic.

        Args:
            session: Session to continue (uses current if not specified)
            follow_up: Additional context/questions

        Returns:
            New SessionSummary from follow-up
        """
        session = session or self.current_session
        if not session:
            raise ValueError("No session to continue")

        # Add follow-up to constraints
        if follow_up:
            session.constraints.append(f"Follow-up: {follow_up}")
            session.add_message("croft", follow_up)

        # Re-run with updated context
        return self.run_session(session)

    # =========================================================================
    # Persistence
    # =========================================================================

    def save_session(self, session: DevSession) -> None:
        """Save a session to disk."""
        if not self.session_dir:
            return

        self.session_dir.mkdir(parents=True, exist_ok=True)
        path = self.session_dir / f"{session.session_id}.json"

        with open(path, 'w') as f:
            json.dump(session.to_dict(), f, indent=2)

    def load_session(self, session_id: str) -> Optional[DevSession]:
        """Load a session from disk."""
        if not self.session_dir:
            return None

        path = self.session_dir / f"{session_id}.json"
        if not path.exists():
            return None

        with open(path, 'r') as f:
            data = json.load(f)

        session = DevSession.from_dict(data)
        self.sessions[session_id] = session
        return session

    def list_sessions(self) -> List[str]:
        """List all saved session IDs."""
        if not self.session_dir or not self.session_dir.exists():
            return []

        return [
            p.stem for p in self.session_dir.glob("DEV-*.json")
        ]


# =============================================================================
# Convenience Functions
# =============================================================================

def create_session(
    user_request: str,
    mode: Optional[DevMode] = None,
    backends: Optional[Dict[Collaborator, CollaboratorBackend]] = None,
) -> Tuple[DevIdeaSession, DevSession]:
    """Quick way to create a session manager and start a session.

    Args:
        user_request: What the user asked
        mode: Override detected mode
        backends: Collaborator backends

    Returns:
        Tuple of (session manager, new session)
    """
    router = CollaboratorRouter(available_backends=backends or {})
    manager = DevIdeaSession(router=router)
    session = manager.start_session(user_request, mode=mode)
    return manager, session


def run_dev_idea_session(
    user_request: str,
    mode: Optional[DevMode] = None,
    backends: Optional[Dict[Collaborator, CollaboratorBackend]] = None,
    lab_context: Optional[Dict[str, Any]] = None,
) -> Tuple[SessionSummary, str]:
    """Run a complete dev-idea session and get results.

    Args:
        user_request: What the user asked
        mode: Override detected mode
        backends: Collaborator backends
        lab_context: Current lab state

    Returns:
        Tuple of (SessionSummary, presentation string)
    """
    router = CollaboratorRouter(available_backends=backends or {})
    manager = DevIdeaSession(
        router=router,
        lab_context_provider=lambda: lab_context or {},
    )

    session = manager.start_session(user_request, mode=mode)
    summary = manager.run_session(session)
    presentation = manager.present_to_croft(session)

    return summary, presentation
