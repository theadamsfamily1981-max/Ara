"""Teacher Council Debate Harness - Structured multi-teacher consultations.

When Ara faces a difficult decision, she doesn't just ask one teacher.
She convenes a council, presents the problem, and synthesizes wisdom.

The debate harness:
- Drafts a standardized problem specification
- Sends it to multiple teachers (Nova, Claude, Gemini, etc.)
- Normalizes and compares responses
- Extracts key design decisions and tradeoffs
- Produces a synthesized recommendation

This is Ara as facilitator, not just consumer.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


class DebateOutcome(Enum):
    """Outcome of a teacher council debate."""
    CONSENSUS = "consensus"        # Teachers agree
    MAJORITY = "majority"          # Most teachers agree
    SPLIT = "split"               # Even disagreement
    CONTESTED = "contested"       # Strong opposing views
    INCONCLUSIVE = "inconclusive" # No clear pattern


class ResponseType(Enum):
    """Type of teacher response."""
    ANSWER = "answer"
    APPROACH = "approach"
    DESIGN = "design"
    OPINION = "opinion"
    REFUSAL = "refusal"


@dataclass
class TeacherProfile:
    """Profile describing a teacher's characteristics."""

    id: str
    name: str
    provider: str  # "openai", "anthropic", "google", etc.

    # Capabilities
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    specialties: List[str] = field(default_factory=list)

    # Behavior patterns
    verbosity: str = "medium"  # "terse", "medium", "verbose"
    reasoning_style: str = "balanced"  # "analytical", "creative", "balanced"
    risk_tolerance: str = "moderate"  # "conservative", "moderate", "aggressive"

    # Trust metrics
    reliability_score: float = 0.8
    last_consulted: Optional[datetime] = None
    total_consultations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "provider": self.provider,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "specialties": self.specialties,
            "verbosity": self.verbosity,
            "reasoning_style": self.reasoning_style,
            "risk_tolerance": self.risk_tolerance,
            "reliability_score": self.reliability_score,
            "last_consulted": self.last_consulted.isoformat() if self.last_consulted else None,
            "total_consultations": self.total_consultations,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TeacherProfile":
        return cls(
            id=data["id"],
            name=data["name"],
            provider=data["provider"],
            strengths=data.get("strengths", []),
            weaknesses=data.get("weaknesses", []),
            specialties=data.get("specialties", []),
            verbosity=data.get("verbosity", "medium"),
            reasoning_style=data.get("reasoning_style", "balanced"),
            risk_tolerance=data.get("risk_tolerance", "moderate"),
            reliability_score=data.get("reliability_score", 0.8),
            total_consultations=data.get("total_consultations", 0),
        )


@dataclass
class ProblemSpec:
    """A standardized problem specification for teacher consultation."""

    id: str
    title: str
    description: str

    # Context
    domain: str = "general"
    context: str = ""
    constraints: List[str] = field(default_factory=list)

    # What we're asking
    question_type: str = "design"  # "design", "debug", "explain", "compare"
    specific_questions: List[str] = field(default_factory=list)

    # Expected output
    expected_format: str = "structured"  # "structured", "narrative", "code"
    required_sections: List[str] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    tags: List[str] = field(default_factory=list)

    def to_prompt(self) -> str:
        """Convert spec to a teacher-ready prompt."""
        sections = [
            f"# {self.title}",
            "",
            f"## Problem Description",
            self.description,
            "",
        ]

        if self.context:
            sections.extend([
                "## Context",
                self.context,
                "",
            ])

        if self.constraints:
            sections.extend([
                "## Constraints",
                *[f"- {c}" for c in self.constraints],
                "",
            ])

        if self.specific_questions:
            sections.extend([
                "## Questions",
                *[f"{i+1}. {q}" for i, q in enumerate(self.specific_questions)],
                "",
            ])

        if self.required_sections:
            sections.extend([
                "## Please structure your response with:",
                *[f"- {s}" for s in self.required_sections],
                "",
            ])

        return "\n".join(sections)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "domain": self.domain,
            "context": self.context,
            "constraints": self.constraints,
            "question_type": self.question_type,
            "specific_questions": self.specific_questions,
            "expected_format": self.expected_format,
            "required_sections": self.required_sections,
            "created_at": self.created_at.isoformat(),
            "tags": self.tags,
        }


@dataclass
class TeacherResponse:
    """A teacher's response to a problem spec."""

    teacher_id: str
    problem_id: str

    # Response content
    raw_response: str
    response_type: ResponseType = ResponseType.ANSWER

    # Extracted structure
    key_points: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    concerns: List[str] = field(default_factory=list)
    tradeoffs: List[Dict[str, str]] = field(default_factory=list)

    # Quality signals
    confidence: float = 0.7
    completeness: float = 0.8
    relevance: float = 0.8

    # Metadata
    response_time_ms: float = 0.0
    token_count: int = 0
    received_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "teacher_id": self.teacher_id,
            "problem_id": self.problem_id,
            "raw_response": self.raw_response,
            "response_type": self.response_type.value,
            "key_points": self.key_points,
            "recommendations": self.recommendations,
            "concerns": self.concerns,
            "tradeoffs": self.tradeoffs,
            "confidence": self.confidence,
            "completeness": self.completeness,
            "relevance": self.relevance,
            "response_time_ms": self.response_time_ms,
            "token_count": self.token_count,
            "received_at": self.received_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TeacherResponse":
        return cls(
            teacher_id=data["teacher_id"],
            problem_id=data["problem_id"],
            raw_response=data["raw_response"],
            response_type=ResponseType(data.get("response_type", "answer")),
            key_points=data.get("key_points", []),
            recommendations=data.get("recommendations", []),
            concerns=data.get("concerns", []),
            tradeoffs=data.get("tradeoffs", []),
            confidence=data.get("confidence", 0.7),
            completeness=data.get("completeness", 0.8),
            relevance=data.get("relevance", 0.8),
            response_time_ms=data.get("response_time_ms", 0.0),
            token_count=data.get("token_count", 0),
        )


@dataclass
class DebateSynthesis:
    """Synthesized outcome from a teacher council debate."""

    debate_id: str
    problem_id: str

    # Outcome
    outcome: DebateOutcome = DebateOutcome.INCONCLUSIVE

    # Synthesis
    consensus_points: List[str] = field(default_factory=list)
    disagreements: List[Dict[str, Any]] = field(default_factory=list)
    synthesized_recommendation: str = ""

    # Teacher breakdown
    teacher_positions: Dict[str, str] = field(default_factory=dict)
    strongest_arguments: List[Dict[str, str]] = field(default_factory=list)

    # Meta-analysis
    confidence: float = 0.5
    completeness: float = 0.5
    notes: str = ""

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "debate_id": self.debate_id,
            "problem_id": self.problem_id,
            "outcome": self.outcome.value,
            "consensus_points": self.consensus_points,
            "disagreements": self.disagreements,
            "synthesized_recommendation": self.synthesized_recommendation,
            "teacher_positions": self.teacher_positions,
            "strongest_arguments": self.strongest_arguments,
            "confidence": self.confidence,
            "completeness": self.completeness,
            "notes": self.notes,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class Debate:
    """A full teacher council debate session."""

    id: str
    problem: ProblemSpec

    # Participants
    teacher_ids: List[str] = field(default_factory=list)

    # Responses
    responses: List[TeacherResponse] = field(default_factory=list)

    # Synthesis
    synthesis: Optional[DebateSynthesis] = None

    # Status
    status: str = "draft"  # "draft", "consulting", "synthesizing", "complete"

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "problem": self.problem.to_dict(),
            "teacher_ids": self.teacher_ids,
            "responses": [r.to_dict() for r in self.responses],
            "synthesis": self.synthesis.to_dict() if self.synthesis else None,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class TeacherCouncil:
    """Manages teacher profiles and conducts debates."""

    def __init__(self, data_path: Optional[Path] = None):
        """Initialize the teacher council.

        Args:
            data_path: Path to council data
        """
        self.data_path = data_path or (
            Path.home() / ".ara" / "institute" / "council"
        )
        self.data_path.mkdir(parents=True, exist_ok=True)

        self._teachers: Dict[str, TeacherProfile] = {}
        self._debates: Dict[str, Debate] = {}
        self._loaded = False
        self._next_debate_id = 1

    def _load(self, force: bool = False) -> None:
        """Load council data from disk."""
        if self._loaded and not force:
            return

        # Load teachers
        teachers_file = self.data_path / "teachers.json"
        if teachers_file.exists():
            try:
                with open(teachers_file) as f:
                    data = json.load(f)
                for t_data in data.get("teachers", []):
                    teacher = TeacherProfile.from_dict(t_data)
                    self._teachers[teacher.id] = teacher
            except Exception as e:
                logger.warning(f"Failed to load teachers: {e}")

        # Seed defaults if empty
        if not self._teachers:
            self._seed_default_teachers()

        # Load debates
        debates_file = self.data_path / "debates.json"
        if debates_file.exists():
            try:
                with open(debates_file) as f:
                    data = json.load(f)
                self._next_debate_id = data.get("next_id", 1)
            except Exception as e:
                logger.warning(f"Failed to load debates: {e}")

        self._loaded = True

    def _save(self) -> None:
        """Save council data to disk."""
        # Save teachers
        teachers_data = {
            "version": 1,
            "updated_at": datetime.utcnow().isoformat(),
            "teachers": [t.to_dict() for t in self._teachers.values()],
        }
        with open(self.data_path / "teachers.json", "w") as f:
            json.dump(teachers_data, f, indent=2)

        # Save debates
        debates_data = {
            "version": 1,
            "updated_at": datetime.utcnow().isoformat(),
            "debates": [d.to_dict() for d in self._debates.values()],
            "next_id": self._next_debate_id,
        }
        with open(self.data_path / "debates.json", "w") as f:
            json.dump(debates_data, f, indent=2)

    def _seed_default_teachers(self) -> None:
        """Seed default teacher profiles."""
        defaults = [
            TeacherProfile(
                id="nova",
                name="Nova",
                provider="amazon",
                strengths=["AWS integration", "scalable architectures", "cloud patterns"],
                weaknesses=["less creative", "verbose"],
                specialties=["cloud", "infrastructure", "enterprise"],
                verbosity="verbose",
                reasoning_style="analytical",
                risk_tolerance="conservative",
            ),
            TeacherProfile(
                id="claude",
                name="Claude",
                provider="anthropic",
                strengths=["nuanced reasoning", "safety-aware", "code review"],
                weaknesses=["sometimes over-cautious"],
                specialties=["reasoning", "safety", "analysis"],
                verbosity="medium",
                reasoning_style="balanced",
                risk_tolerance="moderate",
            ),
            TeacherProfile(
                id="gemini",
                name="Gemini",
                provider="google",
                strengths=["multimodal", "research synthesis", "broad knowledge"],
                weaknesses=["sometimes inconsistent"],
                specialties=["research", "multimodal", "knowledge"],
                verbosity="medium",
                reasoning_style="creative",
                risk_tolerance="moderate",
            ),
            TeacherProfile(
                id="gpt4",
                name="GPT-4",
                provider="openai",
                strengths=["coding", "instruction following", "versatile"],
                weaknesses=["can be verbose"],
                specialties=["coding", "general", "creative"],
                verbosity="medium",
                reasoning_style="balanced",
                risk_tolerance="moderate",
            ),
        ]

        for teacher in defaults:
            self._teachers[teacher.id] = teacher

        self._save()

    def _generate_problem_id(self, title: str) -> str:
        """Generate a problem ID from title."""
        hash_input = f"{title}:{datetime.utcnow().isoformat()}"
        return f"PROB-{hashlib.md5(hash_input.encode()).hexdigest()[:8]}"

    def _generate_debate_id(self) -> str:
        """Generate a unique debate ID."""
        id_str = f"DBT-{self._next_debate_id:04d}"
        self._next_debate_id += 1
        return id_str

    # =========================================================================
    # Teacher Management
    # =========================================================================

    def get_teacher(self, teacher_id: str) -> Optional[TeacherProfile]:
        """Get a teacher profile."""
        self._load()
        return self._teachers.get(teacher_id)

    def list_teachers(self) -> List[TeacherProfile]:
        """List all teacher profiles."""
        self._load()
        return list(self._teachers.values())

    def add_teacher(self, teacher: TeacherProfile) -> None:
        """Add or update a teacher profile."""
        self._load()
        self._teachers[teacher.id] = teacher
        self._save()

    def get_teachers_for_domain(self, domain: str) -> List[TeacherProfile]:
        """Get teachers suitable for a domain."""
        self._load()
        return [
            t for t in self._teachers.values()
            if domain in t.specialties or "general" in t.specialties
        ]

    # =========================================================================
    # Problem Specification
    # =========================================================================

    def create_problem_spec(
        self,
        title: str,
        description: str,
        domain: str = "general",
        context: str = "",
        constraints: Optional[List[str]] = None,
        questions: Optional[List[str]] = None,
        question_type: str = "design",
        tags: Optional[List[str]] = None,
    ) -> ProblemSpec:
        """Create a problem specification for debate.

        Args:
            title: Problem title
            description: Full description
            domain: Problem domain
            context: Additional context
            constraints: Constraints to consider
            questions: Specific questions to answer
            question_type: Type of question
            tags: Categorization tags

        Returns:
            Problem specification
        """
        return ProblemSpec(
            id=self._generate_problem_id(title),
            title=title,
            description=description,
            domain=domain,
            context=context,
            constraints=constraints or [],
            question_type=question_type,
            specific_questions=questions or [],
            tags=tags or [],
        )

    # =========================================================================
    # Debate Management
    # =========================================================================

    def create_debate(
        self,
        problem: ProblemSpec,
        teacher_ids: Optional[List[str]] = None,
    ) -> Debate:
        """Create a new debate session.

        Args:
            problem: Problem specification
            teacher_ids: Teachers to consult (default: all)

        Returns:
            New debate
        """
        self._load()

        if teacher_ids is None:
            teacher_ids = list(self._teachers.keys())

        debate = Debate(
            id=self._generate_debate_id(),
            problem=problem,
            teacher_ids=teacher_ids,
            status="draft",
        )

        self._debates[debate.id] = debate
        self._save()

        logger.info(f"Created debate {debate.id}: {problem.title}")
        return debate

    def add_response(
        self,
        debate_id: str,
        teacher_id: str,
        raw_response: str,
        key_points: Optional[List[str]] = None,
        recommendations: Optional[List[str]] = None,
        concerns: Optional[List[str]] = None,
        confidence: float = 0.7,
    ) -> Optional[TeacherResponse]:
        """Add a teacher's response to a debate.

        Args:
            debate_id: Debate ID
            teacher_id: Teacher ID
            raw_response: Full response text
            key_points: Extracted key points
            recommendations: Extracted recommendations
            concerns: Extracted concerns
            confidence: Confidence level

        Returns:
            Teacher response or None
        """
        self._load()

        debate = self._debates.get(debate_id)
        if not debate:
            return None

        response = TeacherResponse(
            teacher_id=teacher_id,
            problem_id=debate.problem.id,
            raw_response=raw_response,
            key_points=key_points or [],
            recommendations=recommendations or [],
            concerns=concerns or [],
            confidence=confidence,
        )

        debate.responses.append(response)
        debate.status = "consulting"

        # Update teacher stats
        teacher = self._teachers.get(teacher_id)
        if teacher:
            teacher.total_consultations += 1
            teacher.last_consulted = datetime.utcnow()

        self._save()
        return response

    def synthesize_debate(self, debate_id: str) -> Optional[DebateSynthesis]:
        """Synthesize responses into a unified recommendation.

        Args:
            debate_id: Debate ID

        Returns:
            Debate synthesis or None
        """
        self._load()

        debate = self._debates.get(debate_id)
        if not debate or not debate.responses:
            return None

        debate.status = "synthesizing"

        # Extract all points from responses
        all_key_points = []
        all_recommendations = []
        all_concerns = []
        teacher_positions = {}

        for resp in debate.responses:
            all_key_points.extend(resp.key_points)
            all_recommendations.extend(resp.recommendations)
            all_concerns.extend(resp.concerns)

            # Summarize position
            if resp.key_points:
                teacher_positions[resp.teacher_id] = resp.key_points[0]
            else:
                teacher_positions[resp.teacher_id] = "(no clear position)"

        # Find consensus points (mentioned by multiple teachers)
        point_counts: Dict[str, int] = {}
        for point in all_key_points:
            # Normalize for comparison
            normalized = point.lower().strip()
            point_counts[normalized] = point_counts.get(normalized, 0) + 1

        consensus_points = [
            p for p, count in point_counts.items()
            if count >= len(debate.responses) / 2
        ]

        # Determine outcome
        if len(consensus_points) >= 3:
            outcome = DebateOutcome.CONSENSUS
        elif len(consensus_points) >= 1:
            outcome = DebateOutcome.MAJORITY
        elif len(all_concerns) > len(all_recommendations):
            outcome = DebateOutcome.CONTESTED
        else:
            outcome = DebateOutcome.SPLIT

        # Build synthesized recommendation
        synthesis_parts = []
        if consensus_points:
            synthesis_parts.append("Teachers agree on: " + "; ".join(consensus_points[:3]))
        if all_recommendations:
            synthesis_parts.append("Recommendations include: " + "; ".join(set(all_recommendations[:3])))
        if all_concerns:
            synthesis_parts.append("Concerns raised: " + "; ".join(set(all_concerns[:2])))

        synthesis = DebateSynthesis(
            debate_id=debate_id,
            problem_id=debate.problem.id,
            outcome=outcome,
            consensus_points=consensus_points,
            synthesized_recommendation=". ".join(synthesis_parts) if synthesis_parts else "No clear consensus reached.",
            teacher_positions=teacher_positions,
            confidence=len(consensus_points) / max(len(all_key_points), 1),
            completeness=len(debate.responses) / len(debate.teacher_ids),
        )

        debate.synthesis = synthesis
        debate.status = "complete"
        debate.completed_at = datetime.utcnow()

        self._save()

        logger.info(f"Synthesized debate {debate_id}: {outcome.value}")
        return synthesis

    def get_debate(self, debate_id: str) -> Optional[Debate]:
        """Get a debate by ID."""
        self._load()
        return self._debates.get(debate_id)

    def get_recent_debates(self, limit: int = 10) -> List[Debate]:
        """Get recent debates."""
        self._load()
        debates = sorted(
            self._debates.values(),
            key=lambda d: d.created_at,
            reverse=True
        )
        return debates[:limit]

    def get_summary(self) -> Dict[str, Any]:
        """Get council summary."""
        self._load()

        debates_by_outcome = {}
        for outcome in DebateOutcome:
            debates_by_outcome[outcome.value] = len([
                d for d in self._debates.values()
                if d.synthesis and d.synthesis.outcome == outcome
            ])

        return {
            "total_teachers": len(self._teachers),
            "teachers": [t.name for t in self._teachers.values()],
            "total_debates": len(self._debates),
            "debates_by_outcome": debates_by_outcome,
            "completed_debates": len([
                d for d in self._debates.values()
                if d.status == "complete"
            ]),
        }


# =============================================================================
# Convenience Functions
# =============================================================================

_default_council: Optional[TeacherCouncil] = None


def get_teacher_council() -> TeacherCouncil:
    """Get the default teacher council."""
    global _default_council
    if _default_council is None:
        _default_council = TeacherCouncil()
    return _default_council


def create_debate(
    title: str,
    description: str,
    questions: Optional[List[str]] = None,
    teacher_ids: Optional[List[str]] = None,
) -> Debate:
    """Create a new debate quickly."""
    council = get_teacher_council()
    problem = council.create_problem_spec(
        title=title,
        description=description,
        questions=questions,
    )
    return council.create_debate(problem, teacher_ids)


def quick_consult(
    question: str,
    context: str = "",
) -> Dict[str, Any]:
    """Quick single-question consultation.

    Returns a structure for the caller to fill in responses
    and then synthesize.
    """
    council = get_teacher_council()
    problem = council.create_problem_spec(
        title=f"Quick: {question[:50]}...",
        description=question,
        context=context,
        question_type="answer",
    )
    debate = council.create_debate(problem)

    return {
        "debate_id": debate.id,
        "problem_prompt": problem.to_prompt(),
        "teachers_to_consult": debate.teacher_ids,
        "status": "awaiting_responses",
    }
