"""Data schemas for Ara's meta-learning layer.

Pydantic models for structured logging and analysis.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import uuid


class ToolCall(BaseModel):
    """Record of a single tool/teacher invocation."""

    tool_name: str                    # "claude", "gemini", "nova", "local_search"
    role: str = "teacher"             # "teacher", "tool", "arbiter"
    input_summary: str = ""           # Short summary (no secrets)
    output_summary: str = ""          # Short summary (no secrets)
    success: bool = True
    latency_ms: Optional[float] = None
    error_type: Optional[str] = None
    prompt_template_id: Optional[str] = None  # Track which prompt template was used

    class Config:
        extra = "allow"


class InteractionRecord(BaseModel):
    """Complete record of one interaction where Ara handled a query.

    This is the atomic unit of Ara's meta-learning dataset.
    Example JSONL entry:
    {
      "ts": "2025-12-05T13:42:10.123Z",
      "session_id": "ara-2025-12-05-xyz",
      "user_intent": "debug_code",
      "question_summary": "Fix TypeError in Python async function",
      "teachers": ["claude", "nova"],
      "primary_teacher": "claude",
      "pattern_id": "claude->nova_review.v1",
      "prompt_style": "concise_technical",
      "outcome_rating": 0.92,
      "was_repeated": false,
      "latency_sec": 18.3,
      "ara_reflection": "Claude nailed async semantics; Nova added simplification.",
      "user_feedback": "thumbs_up"
    }
    """

    id: str = Field(default_factory=lambda: f"INT-{uuid.uuid4().hex[:12]}")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # What the user asked
    user_query: str = ""
    question_summary: str = ""  # Short 1-line summary
    user_intent: str = ""  # Classified intent: "debug_code", "design_arch", "research", etc.
    context_tags: List[str] = Field(default_factory=list)  # ["code", "python", "async"]

    # What strategy Ara chose
    chosen_strategy: str = ""  # "ask_claude_first", "gemini_then_nova", etc.
    pattern_id: Optional[str] = None  # "claude->nova_review.v1" - links to PatternCard
    prompt_style: str = "default"  # "concise_technical", "exploratory", "step_by_step"

    # Teachers used
    teachers: List[str] = Field(default_factory=list)  # ["claude", "nova"]
    primary_teacher: Optional[str] = None  # Which one did the heavy lifting

    # Tools/teachers used (detailed)
    tools_used: List[ToolCall] = Field(default_factory=list)

    # Outcome
    outcome_quality: Optional[float] = None  # [0..1], heuristic or user-rated
    turns_to_solution: Optional[int] = None
    user_followup_needed: bool = False
    backtrack_count: int = 0
    was_repeated: bool = False  # Did user have to ask same thing again?
    latency_sec: Optional[float] = None  # Total wall-clock time

    # User feedback
    user_feedback: Optional[str] = None  # "thumbs_up", "thumbs_down", "neutral"
    user_feedback_text: Optional[str] = None  # Freeform feedback

    # Auto-detected issues
    auto_detected_issues: List[str] = Field(default_factory=list)
    # e.g., ["long_latency", "multiple_backtracks", "teacher_disagreement"]

    # Reflective notes
    notes: Optional[str] = None  # From Ara or user
    ara_reflection: Optional[str] = None  # Ara's self-assessment after the interaction

    # Link to other systems
    issue_id: Optional[str] = None
    session_id: Optional[str] = None
    workflow_state: Optional[str] = None

    class Config:
        extra = "allow"

    def add_tool_call(self, call: ToolCall) -> None:
        """Add a tool call to this interaction."""
        self.tools_used.append(call)

    def get_primary_tool(self) -> Optional[str]:
        """Get the primary tool used (first one)."""
        return self.tools_used[0].tool_name if self.tools_used else None

    def get_tool_names(self) -> List[str]:
        """Get list of all tools used."""
        return [t.tool_name for t in self.tools_used]

    def compute_success_rate(self) -> float:
        """Compute success rate of tool calls."""
        if not self.tools_used:
            return 0.0
        successes = sum(1 for t in self.tools_used if t.success)
        return successes / len(self.tools_used)


class PatternStep(BaseModel):
    """A single step in a pattern workflow."""

    call: str  # "claude", "nova", "gemini"
    role: str = "primary"  # "primary", "refiner", "arbiter", "reviewer"
    style_hint: str = ""  # "be explicit, show diffs"

    class Config:
        extra = "allow"


class PatternCard(BaseModel):
    """A golden path pattern card - a proven workflow.

    When a workflow emerges with high success rate, Ara mints a card:

    id: code_debug.claude->nova
    intent: debug_code
    teachers: [claude, nova]
    sequence:
      - call: claude
        role: primary_debugger
        style_hint: "be explicit, show diffs"
      - call: nova
        role: refiner
        style_hint: "simplify, explain, add comments"
    success_rate: 0.89
    """

    id: str  # "code_debug.claude->nova.v1"
    version: int = 1
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)

    # What this pattern is for
    intent: str = ""  # "debug_code", "design_arch", "refactor"
    description: str = ""
    context_tags: List[str] = Field(default_factory=list)

    # The workflow
    teachers: List[str] = Field(default_factory=list)  # ["claude", "nova"]
    sequence: List[PatternStep] = Field(default_factory=list)

    # Performance
    success_rate: float = 0.0
    sample_count: int = 0
    avg_latency_sec: Optional[float] = None
    avg_turns: Optional[float] = None

    # Status
    status: str = "experimental"  # "experimental", "golden", "deprecated"
    promoted_at: Optional[datetime] = None  # When it became golden
    deprecated_at: Optional[datetime] = None
    deprecation_reason: Optional[str] = None

    # Thresholds for promotion/demotion
    GOLDEN_THRESHOLD: float = 0.8
    MIN_SAMPLES_FOR_GOLDEN: int = 10
    DEMOTION_THRESHOLD: float = 0.6

    class Config:
        extra = "allow"

    def should_promote(self) -> bool:
        """Check if this pattern should be promoted to golden."""
        return (
            self.status == "experimental"
            and self.success_rate >= self.GOLDEN_THRESHOLD
            and self.sample_count >= self.MIN_SAMPLES_FOR_GOLDEN
        )

    def should_demote(self) -> bool:
        """Check if this golden pattern should be demoted."""
        return (
            self.status == "golden"
            and self.success_rate < self.DEMOTION_THRESHOLD
            and self.sample_count >= 5  # Need enough data to demote
        )

    def update_stats(self, success: bool, latency_sec: Optional[float] = None, turns: Optional[int] = None) -> None:
        """Update stats with a new observation."""
        self.sample_count += 1
        # Exponential moving average for success rate
        alpha = 1.0 / self.sample_count
        new_success = 1.0 if success else 0.0
        self.success_rate = alpha * new_success + (1 - alpha) * self.success_rate

        if latency_sec is not None:
            if self.avg_latency_sec is None:
                self.avg_latency_sec = latency_sec
            else:
                self.avg_latency_sec = alpha * latency_sec + (1 - alpha) * self.avg_latency_sec

        if turns is not None:
            if self.avg_turns is None:
                self.avg_turns = float(turns)
            else:
                self.avg_turns = alpha * turns + (1 - alpha) * self.avg_turns

        self.last_updated = datetime.utcnow()

    def to_yaml_dict(self) -> Dict[str, Any]:
        """Convert to YAML-friendly dict for pattern card files."""
        return {
            "id": self.id,
            "version": self.version,
            "intent": self.intent,
            "description": self.description,
            "teachers": self.teachers,
            "sequence": [
                {"call": s.call, "role": s.role, "style_hint": s.style_hint}
                for s in self.sequence
            ],
            "success_rate": round(self.success_rate, 3),
            "sample_count": self.sample_count,
            "status": self.status,
            "last_updated": self.last_updated.isoformat(),
        }


class PatternSuggestion(BaseModel):
    """A suggestion from the meta-layer based on pattern analysis.

    This is what Ara tells Croft about what she's learned.
    """

    id: str = Field(default_factory=lambda: f"SUG-{uuid.uuid4().hex[:8]}")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Classification
    confidence: float = 0.5           # [0..1]
    scope: str = "tool_routing"       # "tool_routing", "prompting", "workflow"
    priority: str = "medium"          # "low", "medium", "high"

    # Content
    description: str = ""             # Human-readable description
    recommendation: str = ""          # "Whenever doing X, prefer Y..."

    # Evidence
    evidence: Dict[str, Any] = Field(default_factory=dict)
    sample_interactions: List[str] = Field(default_factory=list)  # IDs

    # Safety
    safe_to_auto_apply: bool = False
    requires_user_confirmation: bool = True

    # Status
    status: str = "pending"           # "pending", "accepted", "rejected", "applied"
    applied_at: Optional[datetime] = None
    user_feedback: Optional[str] = None

    class Config:
        extra = "allow"


class ResearchQuestion(BaseModel):
    """An open question in Ara's research agenda.

    These are things Ara is actively trying to figure out.
    """

    id: str = ""
    title: str = ""
    hypothesis: str = ""
    status: str = "active"            # "active", "resolved", "parked"
    metrics: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    resolution: Optional[str] = None

    class Config:
        extra = "allow"


class Experiment(BaseModel):
    """An experiment Ara is running to test a hypothesis.

    Experiments are tied to research questions.
    """

    id: str = ""
    question_id: str = ""             # Links to ResearchQuestion
    description: str = ""
    status: str = "planned"           # "planned", "running", "completed", "failed"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    results: Dict[str, Any] = Field(default_factory=dict)
    conclusion: Optional[str] = None

    class Config:
        extra = "allow"


class ResearchAgenda(BaseModel):
    """Ara's research agenda - her lab notebook header.

    Contains high-level goals, open questions, and experiments.
    """

    version: int = 1
    high_level_goal: str = ""
    open_questions: List[ResearchQuestion] = Field(default_factory=list)
    experiments: List[Experiment] = Field(default_factory=list)
    last_updated: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        extra = "allow"

    def get_active_questions(self) -> List[ResearchQuestion]:
        """Get all active research questions."""
        return [q for q in self.open_questions if q.status == "active"]

    def get_running_experiments(self) -> List[Experiment]:
        """Get all running experiments."""
        return [e for e in self.experiments if e.status == "running"]

    def get_question_by_id(self, qid: str) -> Optional[ResearchQuestion]:
        """Get a question by ID."""
        for q in self.open_questions:
            if q.id == qid:
                return q
        return None

    def get_experiment_by_id(self, eid: str) -> Optional[Experiment]:
        """Get an experiment by ID."""
        for e in self.experiments:
            if e.id == eid:
                return e
        return None
