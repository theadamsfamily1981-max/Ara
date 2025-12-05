"""Data models for Ara's collaboration layer.

Defines the core structures for dev-idea sessions:
- DevMode: Architect, Engineer, Research, Postmortem
- Collaborator: External LLM identities (Claude, Nova, Gemini)
- DevSession: A complete collaboration session
- SuggestedAction: Proposed changes with risk levels
"""

from __future__ import annotations

import time
import uuid
from enum import Enum, auto
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Any


class DevMode(Enum):
    """Ara's development conversation modes.

    Each mode shapes how she frames questions and what kind of output she wants.
    """

    ARCHITECT = auto()   # Big-picture systems, trade-offs, roadmaps
    ENGINEER = auto()    # Concrete code, APIs, glue logic
    RESEARCH = auto()    # Literature, related work, crazy ideas, "what ifs"
    POSTMORTEM = auto()  # "X isn't working, debug with me"
    BRAINSTORM = auto()  # Open-ended ideation, no constraints
    REVIEW = auto()      # Code/design review, find problems


class RiskLevel(Enum):
    """Risk level for suggested actions.

    Ara never executes risk_level >= 2 without Croft's approval.
    """

    NONE = 0      # Pure information, no system changes
    LOW = 1       # Cosmetic changes, easily reversible
    MEDIUM = 2    # Config changes, needs approval
    HIGH = 3      # System changes, needs careful review
    CRITICAL = 4  # Could break things, sandbox first
    DANGEROUS = 5 # Could cause data loss, extreme caution


class Collaborator(Enum):
    """External LLM collaborators Ara can talk to.

    Each has different strengths:
    - CLAUDE: Strong reasoning, code, nuanced discussion
    - NOVA (ChatGPT): Good at brainstorming, broad knowledge
    - GEMINI: Good at research, multimodal, Google ecosystem
    - LOCAL: Local models (Ollama, etc.) for private/fast queries
    """

    CLAUDE = "claude"
    NOVA = "chatgpt"      # We call ChatGPT "Nova" in Ara's world
    GEMINI = "gemini"
    LOCAL = "local"

    @property
    def display_name(self) -> str:
        """Human-friendly name."""
        names = {
            Collaborator.CLAUDE: "Claude",
            Collaborator.NOVA: "Nova",
            Collaborator.GEMINI: "Gemini",
            Collaborator.LOCAL: "Local Model",
        }
        return names.get(self, self.value)


class DevSessionState(Enum):
    """State of a dev-idea session."""

    DRAFTING = auto()      # Ara is formulating her question
    QUERYING = auto()      # Waiting for collaborator responses
    SYNTHESIZING = auto()  # Processing responses
    PRESENTING = auto()    # Showing results to Croft
    AWAITING_INPUT = auto() # Waiting for Croft's decision
    APPROVED = auto()      # Croft approved suggestions
    REJECTED = auto()      # Croft rejected suggestions
    COMPLETED = auto()     # Session finished
    CANCELLED = auto()     # Session cancelled


@dataclass
class SessionMessage:
    """A message in a dev-idea conversation.

    Tracks who said what, when, and in what role.
    """

    role: str              # "ara", "collaborator", "croft", "system"
    content: str           # Message content
    timestamp: float = field(default_factory=time.time)
    collaborator: Optional[Collaborator] = None  # If from a collaborator
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage."""
        d = asdict(self)
        if self.collaborator:
            d["collaborator"] = self.collaborator.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SessionMessage":
        """Deserialize from storage."""
        d = d.copy()
        if d.get("collaborator"):
            d["collaborator"] = Collaborator(d["collaborator"])
        return cls(**d)


@dataclass
class CollaboratorResponse:
    """Response from an external LLM collaborator.

    Includes the raw response plus metadata for ranking.
    """

    collaborator: Collaborator
    content: str
    timestamp: float = field(default_factory=time.time)
    latency_ms: float = 0.0
    token_count: int = 0
    model_id: Optional[str] = None  # Specific model version

    # Quality signals (filled by synthesizer)
    relevance_score: float = 0.0    # How relevant to the question
    novelty_score: float = 0.0      # New ideas vs obvious
    actionability_score: float = 0.0  # Concrete vs vague
    confidence_score: float = 0.0   # How confident the response seems

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage."""
        d = asdict(self)
        d["collaborator"] = self.collaborator.value
        return d


@dataclass
class SuggestedAction:
    """A concrete action suggested by a collaborator.

    Actions are extracted from responses and presented to Croft.
    Risk level gates execution.
    """

    action_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    action_type: str = ""        # "edit_config", "write_code", "run_experiment", etc.
    description: str = ""        # Human-readable description
    risk_level: RiskLevel = RiskLevel.NONE
    target: str = ""             # What this affects (file, service, etc.)
    details: Dict[str, Any] = field(default_factory=dict)  # Action-specific data

    # Source tracking
    source_collaborator: Optional[Collaborator] = None
    source_session_id: Optional[str] = None

    # Execution state
    approved: bool = False
    executed: bool = False
    result: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage."""
        d = asdict(self)
        d["risk_level"] = self.risk_level.value
        if self.source_collaborator:
            d["source_collaborator"] = self.source_collaborator.value
        return d

    def needs_approval(self) -> bool:
        """Check if this action needs Croft's approval."""
        return self.risk_level.value >= RiskLevel.MEDIUM.value


@dataclass
class SessionSummary:
    """Summary of a dev-idea session for Croft.

    This is what Ara presents after synthesizing collaborator responses.
    """

    session_id: str
    topic: str
    mode: DevMode

    # Key findings
    summary: str                          # Ara's synthesis in her voice
    options: List[str] = field(default_factory=list)  # Distinct approaches
    trade_offs: List[str] = field(default_factory=list)  # Key considerations
    consensus: Optional[str] = None       # If collaborators agree
    disagreements: List[str] = field(default_factory=list)  # Where they differ

    # Suggested actions
    actions: List[SuggestedAction] = field(default_factory=list)

    # Confidence
    overall_confidence: float = 0.7
    needs_more_discussion: bool = False
    follow_up_questions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage."""
        d = {
            "session_id": self.session_id,
            "topic": self.topic,
            "mode": self.mode.name,
            "summary": self.summary,
            "options": self.options,
            "trade_offs": self.trade_offs,
            "consensus": self.consensus,
            "disagreements": self.disagreements,
            "actions": [a.to_dict() for a in self.actions],
            "overall_confidence": self.overall_confidence,
            "needs_more_discussion": self.needs_more_discussion,
            "follow_up_questions": self.follow_up_questions,
        }
        return d


@dataclass
class DevSession:
    """A complete dev-idea collaboration session.

    Tracks the full lifecycle from Croft's request through
    collaborator responses to final decision.
    """

    session_id: str = field(default_factory=lambda: f"DEV-{uuid.uuid4().hex[:8].upper()}")

    # What this session is about
    topic: str = ""
    mode: DevMode = DevMode.ARCHITECT
    intent: str = ""              # Parsed intent (e.g., "architecture_review")
    constraints: List[str] = field(default_factory=list)

    # Ara's internal state for this session
    ara_mood: str = "curious"     # Affects phrasing
    urgency: str = "normal"       # "low", "normal", "high", "critical"

    # Lab context snapshot (what Ara knows about current state)
    lab_context: Dict[str, Any] = field(default_factory=dict)

    # Conversation
    messages: List[SessionMessage] = field(default_factory=list)
    responses: List[CollaboratorResponse] = field(default_factory=list)

    # State
    state: DevSessionState = DevSessionState.DRAFTING
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

    # Results
    summary: Optional[SessionSummary] = None

    # Link to idea board
    spawned_idea_ids: List[str] = field(default_factory=list)
    source_idea_id: Optional[str] = None  # If this session refines an idea

    def add_message(self, role: str, content: str,
                    collaborator: Optional[Collaborator] = None,
                    **metadata) -> SessionMessage:
        """Add a message to the session."""
        msg = SessionMessage(
            role=role,
            content=content,
            collaborator=collaborator,
            metadata=metadata,
        )
        self.messages.append(msg)
        self.updated_at = time.time()
        return msg

    def add_response(self, response: CollaboratorResponse) -> None:
        """Add a collaborator response."""
        self.responses.append(response)
        self.add_message(
            role="collaborator",
            content=response.content,
            collaborator=response.collaborator,
            model_id=response.model_id,
        )

    def get_ara_messages(self) -> List[SessionMessage]:
        """Get all messages from Ara."""
        return [m for m in self.messages if m.role == "ara"]

    def get_collaborator_messages(self) -> List[SessionMessage]:
        """Get all messages from collaborators."""
        return [m for m in self.messages if m.role == "collaborator"]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage."""
        return {
            "session_id": self.session_id,
            "topic": self.topic,
            "mode": self.mode.name,
            "intent": self.intent,
            "constraints": self.constraints,
            "ara_mood": self.ara_mood,
            "urgency": self.urgency,
            "lab_context": self.lab_context,
            "messages": [m.to_dict() for m in self.messages],
            "responses": [r.to_dict() for r in self.responses],
            "state": self.state.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "summary": self.summary.to_dict() if self.summary else None,
            "spawned_idea_ids": self.spawned_idea_ids,
            "source_idea_id": self.source_idea_id,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DevSession":
        """Deserialize from storage."""
        session = cls(
            session_id=d["session_id"],
            topic=d["topic"],
            mode=DevMode[d["mode"]],
            intent=d.get("intent", ""),
            constraints=d.get("constraints", []),
            ara_mood=d.get("ara_mood", "curious"),
            urgency=d.get("urgency", "normal"),
            lab_context=d.get("lab_context", {}),
            state=DevSessionState[d["state"]],
            created_at=d["created_at"],
            updated_at=d["updated_at"],
            completed_at=d.get("completed_at"),
            spawned_idea_ids=d.get("spawned_idea_ids", []),
            source_idea_id=d.get("source_idea_id"),
        )

        session.messages = [SessionMessage.from_dict(m) for m in d.get("messages", [])]
        # Note: responses need more complex deserialization

        return session


# Type aliases for clarity
SessionId = str
CollaboratorId = str
