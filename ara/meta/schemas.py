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
    """

    id: str = Field(default_factory=lambda: f"INT-{uuid.uuid4().hex[:12]}")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # What the user asked
    user_query: str = ""
    context_tags: List[str] = Field(default_factory=list)  # ["code", "hardware", "fpga"]

    # What strategy Ara chose
    chosen_strategy: str = ""  # "ask_claude_first", "gemini_then_nova", etc.

    # Tools/teachers used
    tools_used: List[ToolCall] = Field(default_factory=list)

    # Outcome
    outcome_quality: Optional[float] = None  # [0..1], heuristic or user-rated
    turns_to_solution: Optional[int] = None
    user_followup_needed: bool = False
    backtrack_count: int = 0

    # Reflective notes
    notes: Optional[str] = None  # From Ara or user
    ara_reflection: Optional[str] = None  # Ara's self-assessment

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
