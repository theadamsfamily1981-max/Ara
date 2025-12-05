"""Skill Distiller - Bootstrap Ara's own knowledge from interactions.

Every time a teacher gives great code/design:
- Save: problem description, solution, why it was chosen
- Add to "skills" memory

Next time Ara gets a similar question:
1. Search her skill DB first
2. If close match: adapt directly
3. Only if weak match: call external council

Over time, Ara's "needing a second opinion" curve bends downward.
"""

from __future__ import annotations

import json
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from .logger import InteractionLog

logger = logging.getLogger(__name__)


@dataclass
class Skill:
    """A distilled skill from a successful interaction.

    This is Ara's internal knowledge, learned from the teachers.
    """

    skill_id: str = ""
    name: str = ""                    # Short name for the skill
    domain: str = ""                  # Domain area (graphics, fpga, snn, etc.)
    task_type: str = ""               # Task type this applies to

    # The knowledge
    problem_pattern: str = ""         # What kind of problem this solves
    solution_summary: str = ""        # Summary of the solution approach
    solution_code: Optional[str] = None  # Code if applicable
    rationale: str = ""               # Why this approach works

    # Applicability
    prerequisites: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)

    # Provenance
    source_teacher: str = ""          # Which teacher this came from
    source_log_id: Optional[str] = None
    original_reward: float = 0.0

    # Usage tracking
    times_applied: int = 0
    success_rate: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_applied_at: Optional[str] = None

    def __post_init__(self):
        if not self.skill_id:
            content = f"{self.domain}:{self.task_type}:{self.problem_pattern[:50]}"
            self.skill_id = f"SKILL-{hashlib.md5(content.encode()).hexdigest()[:8]}"

    def apply(self, success: bool) -> None:
        """Record an application of this skill."""
        self.times_applied += 1
        # Exponential moving average
        alpha = 0.2
        self.success_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * self.success_rate
        self.last_applied_at = datetime.utcnow().isoformat()

    def match_score(self, query: str, domain: Optional[str] = None) -> float:
        """Compute match score for a query.

        Args:
            query: Problem description
            domain: Optional domain filter

        Returns:
            Match score [0, 1]
        """
        score = 0.0
        query_lower = query.lower()
        query_words = set(query_lower.split())

        # Domain match
        if domain and self.domain == domain:
            score += 0.2

        # Keyword match (strong signal)
        keyword_matches = sum(1 for kw in self.keywords if kw.lower() in query_lower)
        if self.keywords:
            score += 0.4 * (keyword_matches / len(self.keywords))

        # Problem pattern similarity
        pattern_words = set(self.problem_pattern.lower().split())
        overlap = len(query_words & pattern_words)
        if pattern_words:
            score += 0.3 * (overlap / len(pattern_words))

        # Bonus for proven skills
        if self.times_applied > 0:
            score += 0.1 * min(1.0, self.success_rate)

        return min(1.0, score)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "domain": self.domain,
            "task_type": self.task_type,
            "problem_pattern": self.problem_pattern,
            "solution_summary": self.solution_summary,
            "solution_code": self.solution_code,
            "rationale": self.rationale,
            "prerequisites": self.prerequisites,
            "constraints": self.constraints,
            "keywords": self.keywords,
            "source_teacher": self.source_teacher,
            "source_log_id": self.source_log_id,
            "original_reward": self.original_reward,
            "times_applied": self.times_applied,
            "success_rate": self.success_rate,
            "created_at": self.created_at,
            "last_applied_at": self.last_applied_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Skill":
        return cls(**d)


class SkillLibrary:
    """Ara's library of distilled skills.

    Provides storage and retrieval of learned skills.
    """

    def __init__(self, library_path: Optional[Path] = None):
        """Initialize the skill library.

        Args:
            library_path: Path to persist the library
        """
        self.library_path = library_path

        # Skills indexed by domain
        self.skills: Dict[str, List[Skill]] = {}

        # Load persisted library
        if library_path and library_path.exists():
            self._load()

    def add(self, skill: Skill) -> None:
        """Add a skill to the library.

        Args:
            skill: The skill to add
        """
        if skill.domain not in self.skills:
            self.skills[skill.domain] = []

        # Check for duplicate
        for s in self.skills[skill.domain]:
            if s.skill_id == skill.skill_id:
                logger.debug(f"Skill {skill.skill_id} already exists")
                return

        self.skills[skill.domain].append(skill)
        logger.info(f"Added skill {skill.skill_id}: {skill.name}")

        if self.library_path:
            self._save()

    def search(
        self,
        query: str,
        domain: Optional[str] = None,
        task_type: Optional[str] = None,
        top_k: int = 5,
        min_score: float = 0.3,
    ) -> List[Tuple[Skill, float]]:
        """Search for relevant skills.

        Args:
            query: Problem description
            domain: Optional domain filter
            task_type: Optional task type filter
            top_k: Max results
            min_score: Minimum match score

        Returns:
            List of (skill, match_score) tuples
        """
        candidates = []

        for d, skills in self.skills.items():
            if domain and d != domain:
                continue

            for skill in skills:
                if task_type and skill.task_type != task_type:
                    continue

                score = skill.match_score(query, domain)
                if score >= min_score:
                    candidates.append((skill, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

    def get_best(
        self,
        query: str,
        domain: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> Optional[Tuple[Skill, float]]:
        """Get the single best matching skill.

        Args:
            query: Problem description
            domain: Optional domain filter
            task_type: Optional task type filter

        Returns:
            (skill, score) or None
        """
        results = self.search(query, domain, task_type, top_k=1)
        return results[0] if results else None

    def get_by_domain(self, domain: str) -> List[Skill]:
        """Get all skills for a domain."""
        return self.skills.get(domain, [])

    def update_application(self, skill_id: str, success: bool) -> bool:
        """Record a skill application.

        Args:
            skill_id: The skill ID
            success: Whether it worked

        Returns:
            True if found and updated
        """
        for skills in self.skills.values():
            for skill in skills:
                if skill.skill_id == skill_id:
                    skill.apply(success)
                    if self.library_path:
                        self._save()
                    return True
        return False

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of the library."""
        summary = {
            "total_skills": sum(len(s) for s in self.skills.values()),
            "domains": {},
        }

        for domain, skills in self.skills.items():
            summary["domains"][domain] = {
                "count": len(skills),
                "avg_success_rate": sum(s.success_rate for s in skills) / len(skills) if skills else 0,
                "total_applications": sum(s.times_applied for s in skills),
            }

        return summary

    def _save(self) -> None:
        """Save library to disk."""
        if not self.library_path:
            return

        self.library_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            domain: [s.to_dict() for s in skills]
            for domain, skills in self.skills.items()
        }

        with open(self.library_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _load(self) -> None:
        """Load library from disk."""
        if not self.library_path or not self.library_path.exists():
            return

        with open(self.library_path) as f:
            data = json.load(f)

        for domain, skills in data.items():
            self.skills[domain] = [Skill.from_dict(s) for s in skills]


# =============================================================================
# Distillation from Interactions
# =============================================================================

def distill_from_interaction(
    log: InteractionLog,
    min_reward: float = 0.7,
) -> Optional[Skill]:
    """Distill a skill from a successful interaction.

    Only distills if the interaction was successful enough.

    Args:
        log: The interaction log
        min_reward: Minimum reward to distill

    Returns:
        Skill if distillable, None otherwise
    """
    # Check if worth distilling
    if not log.outcome or not log.outcome.success:
        return None
    if log.reward is not None and log.reward < min_reward:
        return None

    # Need actual content
    if not log.tool_calls:
        return None

    # Extract the main tool call
    main_call = log.tool_calls[0]

    # Build skill
    skill = Skill(
        name=f"Learned: {log.task_type}",
        domain=_infer_domain(log.task_type),
        task_type=log.task_type,
        problem_pattern=log.user_intent,
        solution_summary=log.ara_synthesis or main_call.response_summary,
        rationale=f"Worked for: {log.user_intent[:100]}",
        keywords=_extract_keywords(log.user_intent),
        source_teacher=main_call.tool,
        source_log_id=log.log_id,
        original_reward=log.reward or 0.7,
    )

    # Check for code in response
    if "```" in main_call.response_summary:
        skill.solution_code = main_call.response_summary

    return skill


def _infer_domain(task_type: str) -> str:
    """Infer domain from task type."""
    task_lower = task_type.lower()

    domain_keywords = {
        "graphics": ["graphics", "visual", "shader", "render", "display"],
        "fpga": ["fpga", "verilog", "hdl", "firmware", "register"],
        "snn": ["snn", "neural", "spike", "neuron", "synapse"],
        "kernel": ["kernel", "cuda", "opencl", "compute", "parallel"],
        "optimization": ["optim", "perf", "benchmark", "speed"],
        "systems": ["system", "daemon", "service", "config"],
    }

    for domain, keywords in domain_keywords.items():
        if any(kw in task_lower for kw in keywords):
            return domain

    return "general"


def _extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract keywords from text."""
    import re

    # Simple keyword extraction
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())

    # Filter common words
    stopwords = {
        "this", "that", "with", "from", "have", "been", "were", "being",
        "should", "would", "could", "about", "into", "which", "their",
        "what", "when", "where", "there", "these", "those", "some", "more",
    }
    keywords = [w for w in words if w not in stopwords]

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique.append(kw)

    return unique[:max_keywords]


# =============================================================================
# Convenience Functions
# =============================================================================

_default_library: Optional[SkillLibrary] = None


def get_library(path: Optional[Path] = None) -> SkillLibrary:
    """Get the default skill library."""
    global _default_library
    if _default_library is None:
        path = path or Path.home() / ".ara" / "learning" / "skills.json"
        _default_library = SkillLibrary(path)
    return _default_library


def search_skills(
    query: str,
    domain: Optional[str] = None,
    task_type: Optional[str] = None,
) -> List[Tuple[Skill, float]]:
    """Search for skills matching a query.

    Args:
        query: Problem description
        domain: Optional domain filter
        task_type: Optional task type filter

    Returns:
        List of (skill, score) tuples
    """
    return get_library().search(query, domain, task_type)
