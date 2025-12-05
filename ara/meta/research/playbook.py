"""Teacher Playbook Generator - Ara writes docs on how to talk to each teacher.

Based on everything Ara has learned, she can generate a "playbook" for each
teacher - a living document that captures:
- What they're good at
- How to prompt them effectively
- When to use them vs alternatives
- Common pitfalls to avoid

Example playbook excerpt:
  ## Claude
  **Best for**: Code surgery, debugging, implementation
  **Prompt style**: Structured with [TASK], [CODE], [CONSTRAINTS]
  **Pitfall**: Tends to over-explain; ask for concise responses
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict

from .templates import get_template_learner, PromptTemplate
from .self_reflection import get_self_reflector

logger = logging.getLogger(__name__)


@dataclass
class TeacherStrength:
    """A strength/capability of a teacher."""

    category: str  # "code", "design", "research", etc.
    description: str
    confidence: float  # 0-1
    evidence_count: int = 0
    success_rate: Optional[float] = None
    example_intents: List[str] = field(default_factory=list)


@dataclass
class TeacherWeakness:
    """A weakness/pitfall of a teacher."""

    category: str
    description: str
    confidence: float
    evidence_count: int = 0
    workaround: str = ""
    example_failures: List[str] = field(default_factory=list)


@dataclass
class PromptGuideline:
    """A guideline for prompting a teacher."""

    guideline: str
    rationale: str
    confidence: float
    evidence_count: int = 0
    example: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class TeacherPlaybook:
    """A playbook for working with a teacher."""

    teacher: str
    version: int = 1

    # Overview
    summary: str = ""
    best_for: List[str] = field(default_factory=list)
    avoid_for: List[str] = field(default_factory=list)

    # Detailed sections
    strengths: List[TeacherStrength] = field(default_factory=list)
    weaknesses: List[TeacherWeakness] = field(default_factory=list)
    prompt_guidelines: List[PromptGuideline] = field(default_factory=list)

    # Statistics
    total_interactions: int = 0
    overall_success_rate: Optional[float] = None
    avg_latency_sec: Optional[float] = None

    # Best templates
    top_templates: List[str] = field(default_factory=list)  # Template IDs

    # Metadata
    generated_at: datetime = field(default_factory=datetime.utcnow)
    data_period_days: int = 30

    def to_dict(self) -> Dict[str, Any]:
        return {
            "teacher": self.teacher,
            "version": self.version,
            "summary": self.summary,
            "best_for": self.best_for,
            "avoid_for": self.avoid_for,
            "strengths": [
                {
                    "category": s.category,
                    "description": s.description,
                    "confidence": round(s.confidence, 2),
                    "evidence_count": s.evidence_count,
                    "success_rate": s.success_rate,
                    "example_intents": s.example_intents,
                }
                for s in self.strengths
            ],
            "weaknesses": [
                {
                    "category": w.category,
                    "description": w.description,
                    "confidence": round(w.confidence, 2),
                    "workaround": w.workaround,
                }
                for w in self.weaknesses
            ],
            "prompt_guidelines": [
                {
                    "guideline": g.guideline,
                    "rationale": g.rationale,
                    "confidence": round(g.confidence, 2),
                    "example": g.example,
                }
                for g in self.prompt_guidelines
            ],
            "total_interactions": self.total_interactions,
            "overall_success_rate": self.overall_success_rate,
            "top_templates": self.top_templates,
            "generated_at": self.generated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TeacherPlaybook":
        pb = cls(
            teacher=data["teacher"],
            version=data.get("version", 1),
            summary=data.get("summary", ""),
            best_for=data.get("best_for", []),
            avoid_for=data.get("avoid_for", []),
            total_interactions=data.get("total_interactions", 0),
            overall_success_rate=data.get("overall_success_rate"),
            top_templates=data.get("top_templates", []),
            data_period_days=data.get("data_period_days", 30),
        )

        for s in data.get("strengths", []):
            pb.strengths.append(TeacherStrength(
                category=s["category"],
                description=s["description"],
                confidence=s.get("confidence", 0.5),
                evidence_count=s.get("evidence_count", 0),
                success_rate=s.get("success_rate"),
                example_intents=s.get("example_intents", []),
            ))

        for w in data.get("weaknesses", []):
            pb.weaknesses.append(TeacherWeakness(
                category=w["category"],
                description=w["description"],
                confidence=w.get("confidence", 0.5),
                workaround=w.get("workaround", ""),
            ))

        for g in data.get("prompt_guidelines", []):
            pb.prompt_guidelines.append(PromptGuideline(
                guideline=g["guideline"],
                rationale=g.get("rationale", ""),
                confidence=g.get("confidence", 0.5),
                example=g.get("example", ""),
            ))

        return pb

    def format_markdown(self) -> str:
        """Format the playbook as markdown."""
        lines = []

        lines.append(f"# {self.teacher.title()} Playbook")
        lines.append(f"*Generated: {self.generated_at.strftime('%Y-%m-%d')} | "
                    f"v{self.version} | {self.total_interactions} interactions*")
        lines.append("")

        # Summary
        if self.summary:
            lines.append("## Overview")
            lines.append(self.summary)
            lines.append("")

        # Best for
        if self.best_for:
            lines.append("## Best For")
            for item in self.best_for:
                lines.append(f"- {item}")
            lines.append("")

        # Avoid for
        if self.avoid_for:
            lines.append("## Avoid For")
            for item in self.avoid_for:
                lines.append(f"- {item}")
            lines.append("")

        # Strengths
        if self.strengths:
            lines.append("## Strengths")
            for strength in self.strengths:
                conf_str = f" ({strength.confidence:.0%} confidence)" if strength.confidence < 1.0 else ""
                lines.append(f"### {strength.category.title()}{conf_str}")
                lines.append(strength.description)
                if strength.success_rate:
                    lines.append(f"*Success rate: {strength.success_rate:.0%}*")
                lines.append("")

        # Weaknesses
        if self.weaknesses:
            lines.append("## Pitfalls & Workarounds")
            for weakness in self.weaknesses:
                lines.append(f"### {weakness.category.title()}")
                lines.append(f"**Issue**: {weakness.description}")
                if weakness.workaround:
                    lines.append(f"**Workaround**: {weakness.workaround}")
                lines.append("")

        # Prompt guidelines
        if self.prompt_guidelines:
            lines.append("## Prompt Guidelines")
            for i, guideline in enumerate(self.prompt_guidelines, 1):
                lines.append(f"### {i}. {guideline.guideline}")
                if guideline.rationale:
                    lines.append(guideline.rationale)
                if guideline.example:
                    lines.append("")
                    lines.append("**Example:**")
                    lines.append(f"```\n{guideline.example}\n```")
                lines.append("")

        # Statistics
        lines.append("## Statistics")
        if self.overall_success_rate:
            lines.append(f"- Overall success rate: {self.overall_success_rate:.0%}")
        if self.avg_latency_sec:
            lines.append(f"- Average latency: {self.avg_latency_sec:.1f}s")
        lines.append(f"- Total interactions analyzed: {self.total_interactions}")
        lines.append("")

        return "\n".join(lines)


class PlaybookGenerator:
    """Generates playbooks from learned data."""

    # Default teacher characteristics (priors)
    TEACHER_PRIORS = {
        "claude": {
            "summary": "Claude excels at code implementation, debugging, and detailed technical explanations.",
            "best_for": ["code_implementation", "debugging", "code_review", "technical_writing"],
            "avoid_for": ["hardware_specifics", "real-time_data"],
            "default_strengths": [
                TeacherStrength(
                    category="code",
                    description="Excellent at writing clean, well-structured code",
                    confidence=0.7,
                ),
                TeacherStrength(
                    category="debugging",
                    description="Strong at tracing through code to find issues",
                    confidence=0.7,
                ),
            ],
            "default_guidelines": [
                PromptGuideline(
                    guideline="Use structured prompts with clear sections",
                    rationale="Claude responds well to organized input",
                    confidence=0.6,
                    example="[TASK]\\n...\\n[CODE]\\n...\\n[CONSTRAINTS]\\n...",
                ),
            ],
        },
        "nova": {
            "summary": "Nova is the arbiter - great for review, validation, and catching edge cases.",
            "best_for": ["code_review", "design_validation", "edge_case_analysis"],
            "avoid_for": ["creative_tasks", "open-ended_research"],
            "default_strengths": [
                TeacherStrength(
                    category="review",
                    description="Thorough at reviewing and finding issues",
                    confidence=0.7,
                ),
            ],
            "default_guidelines": [
                PromptGuideline(
                    guideline="Present work for review with specific questions",
                    rationale="Nova works best with focused review tasks",
                    confidence=0.6,
                ),
            ],
        },
        "gemini": {
            "summary": "Gemini is the R&D gremlin - excellent for research, exploration, and creative solutions.",
            "best_for": ["research", "exploration", "creative_solutions", "broad_analysis"],
            "avoid_for": ["precise_implementation", "strict_formatting"],
            "default_strengths": [
                TeacherStrength(
                    category="research",
                    description="Strong at exploring topics and finding connections",
                    confidence=0.7,
                ),
            ],
            "default_guidelines": [
                PromptGuideline(
                    guideline="Allow room for exploration",
                    rationale="Gemini thrives with open-ended prompts",
                    confidence=0.6,
                ),
            ],
        },
    }

    def __init__(self, playbooks_path: Optional[Path] = None):
        """Initialize the generator.

        Args:
            playbooks_path: Path to store playbooks
        """
        self.playbooks_path = playbooks_path or (
            Path.home() / ".ara" / "meta" / "research" / "playbooks.json"
        )
        self.playbooks_path.parent.mkdir(parents=True, exist_ok=True)

        self._playbooks: Dict[str, TeacherPlaybook] = {}
        self._loaded = False

    def _load(self, force: bool = False) -> None:
        """Load playbooks from disk."""
        if self._loaded and not force:
            return

        self._playbooks.clear()

        if self.playbooks_path.exists():
            try:
                with open(self.playbooks_path) as f:
                    data = json.load(f)
                for pb_data in data.get("playbooks", []):
                    pb = TeacherPlaybook.from_dict(pb_data)
                    self._playbooks[pb.teacher] = pb
            except Exception as e:
                logger.warning(f"Failed to load playbooks: {e}")

        self._loaded = True

    def _save(self) -> None:
        """Save playbooks to disk."""
        data = {
            "version": 1,
            "updated_at": datetime.utcnow().isoformat(),
            "playbooks": [pb.to_dict() for pb in self._playbooks.values()],
        }
        with open(self.playbooks_path, "w") as f:
            json.dump(data, f, indent=2)

    def get_playbook(self, teacher: str) -> Optional[TeacherPlaybook]:
        """Get a playbook by teacher name."""
        self._load()
        return self._playbooks.get(teacher)

    def get_all_playbooks(self) -> List[TeacherPlaybook]:
        """Get all playbooks."""
        self._load()
        return list(self._playbooks.values())

    def _load_interaction_stats(
        self,
        teacher: str,
    ) -> Dict[str, Any]:
        """Load interaction statistics for a teacher.

        Args:
            teacher: Teacher name

        Returns:
            Statistics dict
        """
        log_path = Path.home() / ".ara" / "meta" / "interactions.jsonl"
        stats: Dict[str, Any] = {
            "total": 0,
            "successes": 0,
            "by_intent": defaultdict(lambda: {"total": 0, "successes": 0}),
            "latencies": [],
        }

        if not log_path.exists():
            return stats

        try:
            with open(log_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        primary = record.get("primary_teacher") or (
                            record.get("teachers", [None])[0]
                        )
                        if primary != teacher:
                            continue

                        stats["total"] += 1
                        if record.get("success"):
                            stats["successes"] += 1

                        intent = record.get("user_intent", "unknown")
                        stats["by_intent"][intent]["total"] += 1
                        if record.get("success"):
                            stats["by_intent"][intent]["successes"] += 1

                        latency = record.get("latency_sec")
                        if latency:
                            stats["latencies"].append(latency)

                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"Failed to load interaction stats: {e}")

        return stats

    def generate_playbook(
        self,
        teacher: str,
        force_regenerate: bool = False,
    ) -> TeacherPlaybook:
        """Generate a playbook for a teacher.

        Args:
            teacher: Teacher name
            force_regenerate: Force regeneration even if exists

        Returns:
            Generated playbook
        """
        self._load()

        existing = self._playbooks.get(teacher)
        if existing and not force_regenerate:
            return existing

        # Start with priors
        priors = self.TEACHER_PRIORS.get(teacher, {})

        playbook = TeacherPlaybook(
            teacher=teacher,
            version=(existing.version + 1) if existing else 1,
            summary=priors.get("summary", f"{teacher.title()} teacher playbook"),
            best_for=priors.get("best_for", []),
            avoid_for=priors.get("avoid_for", []),
        )

        # Add default strengths
        for strength in priors.get("default_strengths", []):
            playbook.strengths.append(strength)

        # Add default guidelines
        for guideline in priors.get("default_guidelines", []):
            playbook.prompt_guidelines.append(guideline)

        # Load real statistics
        stats = self._load_interaction_stats(teacher)
        playbook.total_interactions = stats["total"]

        if stats["total"] > 0:
            playbook.overall_success_rate = stats["successes"] / stats["total"]

        if stats["latencies"]:
            playbook.avg_latency_sec = sum(stats["latencies"]) / len(stats["latencies"])

        # Analyze intent performance
        for intent, intent_stats in stats["by_intent"].items():
            if intent_stats["total"] < 3:
                continue

            success_rate = intent_stats["successes"] / intent_stats["total"]

            if success_rate >= 0.8:
                # This is a strength
                existing_strength = next(
                    (s for s in playbook.strengths if s.category == intent),
                    None,
                )
                if existing_strength:
                    existing_strength.success_rate = success_rate
                    existing_strength.evidence_count = intent_stats["total"]
                    existing_strength.confidence = min(
                        0.95, 0.5 + intent_stats["total"] * 0.05
                    )
                else:
                    playbook.strengths.append(TeacherStrength(
                        category=intent,
                        description=f"Performs well on {intent} tasks",
                        confidence=min(0.9, 0.5 + intent_stats["total"] * 0.05),
                        evidence_count=intent_stats["total"],
                        success_rate=success_rate,
                        example_intents=[intent],
                    ))

                if intent not in playbook.best_for:
                    playbook.best_for.append(intent)

            elif success_rate < 0.5:
                # This is a weakness
                playbook.weaknesses.append(TeacherWeakness(
                    category=intent,
                    description=f"Struggles with {intent} tasks ({success_rate:.0%} success)",
                    confidence=min(0.9, 0.5 + intent_stats["total"] * 0.05),
                    evidence_count=intent_stats["total"],
                    workaround=f"Consider using a different teacher for {intent}",
                ))

                if intent not in playbook.avoid_for:
                    playbook.avoid_for.append(intent)

        # Get top templates
        template_learner = get_template_learner()
        templates = template_learner.get_templates_for(teacher, "*")
        if templates:
            sorted_templates = sorted(
                [t for t in templates if t.sample_count >= 3],
                key=lambda t: (t.success_rate or 0, t.avg_reward or 0),
                reverse=True,
            )
            playbook.top_templates = [t.id for t in sorted_templates[:5]]

            # Add template-based guidelines
            for template in sorted_templates[:3]:
                if template.success_rate and template.success_rate >= 0.8:
                    style = template.tags[0] if template.tags else "standard"
                    playbook.prompt_guidelines.append(PromptGuideline(
                        guideline=f"Use {style} prompts for {template.intent}",
                        rationale=f"Template {template.name} has {template.success_rate:.0%} success",
                        confidence=0.7,
                        example=template.skeleton[:200] if template.skeleton else "",
                        evidence_count=template.sample_count,
                    ))

        self._playbooks[teacher] = playbook
        self._save()

        logger.info(f"Generated playbook for {teacher} (v{playbook.version})")
        return playbook

    def generate_all_playbooks(self) -> List[TeacherPlaybook]:
        """Generate playbooks for all teachers."""
        playbooks = []
        for teacher in ["claude", "nova", "gemini"]:
            pb = self.generate_playbook(teacher, force_regenerate=True)
            playbooks.append(pb)
        return playbooks

    def export_markdown(
        self,
        teacher: Optional[str] = None,
        output_dir: Optional[Path] = None,
    ) -> List[Path]:
        """Export playbooks as markdown files.

        Args:
            teacher: Specific teacher or None for all
            output_dir: Output directory

        Returns:
            List of created file paths
        """
        self._load()

        output_dir = output_dir or (Path.home() / ".ara" / "meta" / "playbooks")
        output_dir.mkdir(parents=True, exist_ok=True)

        teachers = [teacher] if teacher else list(self._playbooks.keys())
        created = []

        for t in teachers:
            pb = self._playbooks.get(t)
            if not pb:
                continue

            filepath = output_dir / f"{t}_playbook.md"
            with open(filepath, "w") as f:
                f.write(pb.format_markdown())

            created.append(filepath)
            logger.info(f"Exported playbook: {filepath}")

        return created

    def get_summary(self) -> Dict[str, Any]:
        """Get generator summary."""
        self._load()

        return {
            "total_playbooks": len(self._playbooks),
            "teachers": list(self._playbooks.keys()),
            "playbooks": [
                {
                    "teacher": pb.teacher,
                    "version": pb.version,
                    "interactions": pb.total_interactions,
                    "success_rate": pb.overall_success_rate,
                    "strengths": len(pb.strengths),
                    "weaknesses": len(pb.weaknesses),
                }
                for pb in self._playbooks.values()
            ],
        }


# =============================================================================
# Convenience Functions
# =============================================================================

_default_generator: Optional[PlaybookGenerator] = None


def get_playbook_generator() -> PlaybookGenerator:
    """Get the default playbook generator."""
    global _default_generator
    if _default_generator is None:
        _default_generator = PlaybookGenerator()
    return _default_generator


def generate_playbook(teacher: str) -> TeacherPlaybook:
    """Generate a playbook for a teacher."""
    return get_playbook_generator().generate_playbook(teacher)


def get_playbook(teacher: str) -> Optional[TeacherPlaybook]:
    """Get a playbook for a teacher."""
    return get_playbook_generator().get_playbook(teacher)


def export_all_playbooks() -> List[Path]:
    """Export all playbooks as markdown."""
    return get_playbook_generator().export_markdown()
