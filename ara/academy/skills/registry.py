"""Skill Registry - Ara's learned, hardened skills.

Each skill represents something Ara learned from teachers and has
internalized into a reusable, local capability:
- What it does
- Who she learned it from
- Success rate
- When to use it vs escalate to teachers

Example:
  skill_id: benchmark_harness_py
  learned_from: ["nova", "claude"]
  success_rate: 0.91
  implementation: template+policy
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class SkillImplementation:
    """How a skill is actually executed."""

    type: str  # "template", "script", "policy", "template+policy"
    entrypoint: Optional[str] = None  # Python path or file
    template: Optional[str] = None  # Template string for simple skills
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "entrypoint": self.entrypoint,
            "template": self.template,
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillImplementation":
        return cls(
            type=data.get("type", "template"),
            entrypoint=data.get("entrypoint"),
            template=data.get("template"),
            parameters=data.get("parameters", {}),
        )


@dataclass
class EscalationPolicy:
    """When to escalate to teachers instead of using local skill."""

    escalate_on_error: bool = True
    escalate_on_low_confidence: bool = True
    confidence_threshold: float = 0.7
    escalate_on_tags: List[str] = field(default_factory=list)  # e.g., ["novel", "safety_critical"]
    max_autonomous_uses: Optional[int] = None  # Require review after N uses
    preferred_escalation_teacher: Optional[str] = None

    def should_escalate(
        self,
        confidence: float,
        tags: List[str],
        autonomous_uses: int,
    ) -> tuple[bool, str]:
        """Check if we should escalate.

        Returns (should_escalate, reason)
        """
        if self.escalate_on_low_confidence and confidence < self.confidence_threshold:
            return True, f"confidence {confidence:.0%} < threshold {self.confidence_threshold:.0%}"

        for tag in tags:
            if tag in self.escalate_on_tags:
                return True, f"tag '{tag}' requires escalation"

        if self.max_autonomous_uses and autonomous_uses >= self.max_autonomous_uses:
            return True, f"reached {self.max_autonomous_uses} autonomous uses, time for review"

        return False, ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "escalate_on_error": self.escalate_on_error,
            "escalate_on_low_confidence": self.escalate_on_low_confidence,
            "confidence_threshold": self.confidence_threshold,
            "escalate_on_tags": self.escalate_on_tags,
            "max_autonomous_uses": self.max_autonomous_uses,
            "preferred_escalation_teacher": self.preferred_escalation_teacher,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EscalationPolicy":
        return cls(
            escalate_on_error=data.get("escalate_on_error", True),
            escalate_on_low_confidence=data.get("escalate_on_low_confidence", True),
            confidence_threshold=data.get("confidence_threshold", 0.7),
            escalate_on_tags=data.get("escalate_on_tags", []),
            max_autonomous_uses=data.get("max_autonomous_uses"),
            preferred_escalation_teacher=data.get("preferred_escalation_teacher"),
        )


@dataclass
class LearnedSkill:
    """A skill Ara has learned and internalized."""

    id: str
    name: str
    description: str

    # Learning provenance
    learned_from: List[str] = field(default_factory=list)  # Teachers
    examples_seen: int = 0
    first_learned: datetime = field(default_factory=datetime.utcnow)

    # Implementation
    implementation: Optional[SkillImplementation] = None
    escalation_policy: EscalationPolicy = field(default_factory=EscalationPolicy)

    # Performance tracking
    uses: int = 0
    successes: int = 0
    autonomous_uses: int = 0  # Uses without teacher help
    last_used: Optional[datetime] = None

    # Categorization
    category: str = "general"  # "codegen", "analysis", "viz", "orchestration"
    tags: List[str] = field(default_factory=list)
    triggers: List[str] = field(default_factory=list)  # Keywords that activate

    # Status
    status: str = "draft"  # "draft", "active", "deprecated", "needs_review"
    version: str = "1.0.0"

    @property
    def success_rate(self) -> Optional[float]:
        if self.uses == 0:
            return None
        return self.successes / self.uses

    def record_use(self, success: bool, autonomous: bool = False) -> None:
        """Record a use of this skill."""
        self.uses += 1
        if success:
            self.successes += 1
        if autonomous:
            self.autonomous_uses += 1
        self.last_used = datetime.utcnow()

    def should_escalate(self, confidence: float, tags: List[str]) -> tuple[bool, str]:
        """Check if this use should escalate to teachers."""
        return self.escalation_policy.should_escalate(
            confidence, tags, self.autonomous_uses
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "learned_from": self.learned_from,
            "examples_seen": self.examples_seen,
            "first_learned": self.first_learned.isoformat(),
            "implementation": self.implementation.to_dict() if self.implementation else None,
            "escalation_policy": self.escalation_policy.to_dict(),
            "uses": self.uses,
            "successes": self.successes,
            "success_rate": self.success_rate,
            "autonomous_uses": self.autonomous_uses,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "category": self.category,
            "tags": self.tags,
            "triggers": self.triggers,
            "status": self.status,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LearnedSkill":
        skill = cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            learned_from=data.get("learned_from", []),
            examples_seen=data.get("examples_seen", 0),
            uses=data.get("uses", 0),
            successes=data.get("successes", 0),
            autonomous_uses=data.get("autonomous_uses", 0),
            category=data.get("category", "general"),
            tags=data.get("tags", []),
            triggers=data.get("triggers", []),
            status=data.get("status", "draft"),
            version=data.get("version", "1.0.0"),
        )

        if data.get("implementation"):
            skill.implementation = SkillImplementation.from_dict(data["implementation"])

        if data.get("escalation_policy"):
            skill.escalation_policy = EscalationPolicy.from_dict(data["escalation_policy"])

        return skill


class SkillRegistry:
    """Registry of all learned skills."""

    def __init__(self, registry_path: Optional[Path] = None):
        """Initialize the registry.

        Args:
            registry_path: Path to registry JSON file
        """
        self.registry_path = registry_path or (
            Path.home() / ".ara" / "academy" / "skills" / "registry.json"
        )
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

        self._skills: Dict[str, LearnedSkill] = {}
        self._loaded = False
        self._next_id = 1

    def _load(self, force: bool = False) -> None:
        """Load skills from disk."""
        if self._loaded and not force:
            return

        self._skills.clear()

        if self.registry_path.exists():
            try:
                with open(self.registry_path) as f:
                    data = json.load(f)
                for skill_data in data.get("skills", []):
                    skill = LearnedSkill.from_dict(skill_data)
                    self._skills[skill.id] = skill
                    # Update ID counter
                    if skill.id.startswith("LSKILL-"):
                        try:
                            num = int(skill.id[7:11])
                            self._next_id = max(self._next_id, num + 1)
                        except ValueError:
                            pass
            except Exception as e:
                logger.warning(f"Failed to load skill registry: {e}")

        self._loaded = True

    def _save(self) -> None:
        """Save skills to disk."""
        data = {
            "version": 1,
            "updated_at": datetime.utcnow().isoformat(),
            "skills": [s.to_dict() for s in self._skills.values()],
        }
        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=2)

    def _generate_id(self, category: str) -> str:
        """Generate a unique skill ID."""
        cat_abbr = category[:4].lower()
        id_str = f"LSKILL-{self._next_id:04d}-{cat_abbr}"
        self._next_id += 1
        return id_str

    def get_skill(self, skill_id: str) -> Optional[LearnedSkill]:
        """Get a skill by ID."""
        self._load()
        return self._skills.get(skill_id)

    def get_all_skills(self) -> List[LearnedSkill]:
        """Get all skills."""
        self._load()
        return list(self._skills.values())

    def get_active_skills(self) -> List[LearnedSkill]:
        """Get all active skills."""
        self._load()
        return [s for s in self._skills.values() if s.status == "active"]

    def get_skills_by_category(self, category: str) -> List[LearnedSkill]:
        """Get skills in a category."""
        self._load()
        return [s for s in self._skills.values() if s.category == category]

    def find_skill_for_task(
        self,
        query: str,
        tags: Optional[List[str]] = None,
        min_success_rate: float = 0.7,
    ) -> Optional[LearnedSkill]:
        """Find a skill that matches a task.

        Args:
            query: Task description
            tags: Task tags
            min_success_rate: Minimum required success rate

        Returns:
            Best matching skill or None
        """
        self._load()

        query_lower = query.lower()
        tags = tags or []
        candidates = []

        for skill in self._skills.values():
            if skill.status != "active":
                continue

            # Check success rate
            if skill.success_rate and skill.success_rate < min_success_rate:
                continue

            # Score based on trigger matches
            score = 0.0
            for trigger in skill.triggers:
                if trigger.lower() in query_lower:
                    score += 0.3

            for tag in tags:
                if tag in skill.tags:
                    score += 0.2

            if score > 0:
                candidates.append((skill, score))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    def register_skill(
        self,
        name: str,
        description: str,
        category: str,
        learned_from: List[str],
        implementation: Optional[SkillImplementation] = None,
        triggers: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        examples_seen: int = 1,
    ) -> LearnedSkill:
        """Register a new learned skill.

        Args:
            name: Skill name
            description: What it does
            category: Category (codegen, analysis, viz, etc.)
            learned_from: Teachers it was learned from
            implementation: How to execute it
            triggers: Keywords that activate it
            tags: Categorization tags
            examples_seen: Number of examples observed

        Returns:
            The new skill
        """
        self._load()

        skill = LearnedSkill(
            id=self._generate_id(category),
            name=name,
            description=description,
            category=category,
            learned_from=learned_from,
            implementation=implementation,
            triggers=triggers or [],
            tags=tags or [],
            examples_seen=examples_seen,
            status="draft",
        )

        self._skills[skill.id] = skill
        self._save()
        logger.info(f"Registered new skill: {skill.id} ({skill.name})")

        return skill

    def promote_skill(self, skill_id: str) -> bool:
        """Promote a skill from draft to active."""
        self._load()

        skill = self._skills.get(skill_id)
        if not skill:
            return False

        if skill.status == "draft":
            skill.status = "active"
            self._save()
            logger.info(f"Promoted skill to active: {skill_id}")
            return True

        return False

    def deprecate_skill(self, skill_id: str, reason: str = "") -> bool:
        """Deprecate a skill."""
        self._load()

        skill = self._skills.get(skill_id)
        if not skill:
            return False

        skill.status = "deprecated"
        self._save()
        logger.info(f"Deprecated skill: {skill_id} ({reason})")
        return True

    def record_skill_use(
        self,
        skill_id: str,
        success: bool,
        autonomous: bool = False,
    ) -> bool:
        """Record a use of a skill."""
        self._load()

        skill = self._skills.get(skill_id)
        if not skill:
            return False

        skill.record_use(success, autonomous)

        # Auto-deprecate if performing poorly
        if skill.uses >= 10 and (skill.success_rate or 0) < 0.5:
            skill.status = "needs_review"
            logger.warning(f"Skill {skill_id} needs review (success rate: {skill.success_rate:.0%})")

        self._save()
        return True

    def get_summary(self) -> Dict[str, Any]:
        """Get registry summary."""
        self._load()

        by_status = defaultdict(int)
        by_category = defaultdict(int)
        by_teacher = defaultdict(int)

        for skill in self._skills.values():
            by_status[skill.status] += 1
            by_category[skill.category] += 1
            for teacher in skill.learned_from:
                by_teacher[teacher] += 1

        # Top performers
        active = [s for s in self._skills.values() if s.status == "active" and s.uses >= 3]
        top = sorted(active, key=lambda s: s.success_rate or 0, reverse=True)[:5]

        return {
            "total_skills": len(self._skills),
            "by_status": dict(by_status),
            "by_category": dict(by_category),
            "by_teacher": dict(by_teacher),
            "top_performers": [
                {
                    "id": s.id,
                    "name": s.name,
                    "success_rate": s.success_rate,
                    "uses": s.uses,
                }
                for s in top
            ],
        }


# =============================================================================
# Default Skills
# =============================================================================

DEFAULT_SKILLS = [
    {
        "id": "LSKILL-0001-code",
        "name": "benchmark_harness_py",
        "description": "Generate Python benchmark harness with timing, CLI, and structured logging",
        "category": "codegen",
        "learned_from": ["nova", "claude"],
        "examples_seen": 12,
        "triggers": ["benchmark", "harness", "timing", "performance test"],
        "tags": ["python", "benchmarking"],
        "status": "active",
        "implementation": {
            "type": "template+policy",
            "template": "# Benchmark: {name}\nimport time\nimport argparse\n\ndef benchmark_{name}():\n    # TODO: Fill in\n    pass\n",
        },
    },
    {
        "id": "LSKILL-0002-anal",
        "name": "log_summarizer",
        "description": "Summarize interaction logs into patterns and insights",
        "category": "analysis",
        "learned_from": ["claude"],
        "examples_seen": 8,
        "triggers": ["summarize", "logs", "patterns", "insights"],
        "tags": ["logs", "analysis"],
        "status": "active",
    },
    {
        "id": "LSKILL-0003-viz",
        "name": "shader_param_tuner",
        "description": "Auto-tune shader parameters based on performance metrics",
        "category": "viz",
        "learned_from": ["nova"],
        "examples_seen": 5,
        "triggers": ["shader", "tune", "parameters", "webgl"],
        "tags": ["shaders", "graphics", "optimization"],
        "status": "draft",
    },
]


def seed_default_skills(registry: SkillRegistry) -> int:
    """Seed default skills.

    Args:
        registry: Skill registry

    Returns:
        Number seeded
    """
    seeded = 0
    for skill_data in DEFAULT_SKILLS:
        if not registry.get_skill(skill_data["id"]):
            registry._load()
            skill = LearnedSkill.from_dict(skill_data)
            registry._skills[skill.id] = skill
            seeded += 1

    if seeded:
        registry._save()

    return seeded


# =============================================================================
# Convenience Functions
# =============================================================================

_default_registry: Optional[SkillRegistry] = None


def get_skill_registry() -> SkillRegistry:
    """Get the default skill registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = SkillRegistry()
    return _default_registry


def find_skill(query: str, tags: Optional[List[str]] = None) -> Optional[LearnedSkill]:
    """Find a skill for a task."""
    return get_skill_registry().find_skill_for_task(query, tags)


def register_skill(
    name: str,
    description: str,
    category: str,
    learned_from: List[str],
) -> LearnedSkill:
    """Register a new skill."""
    return get_skill_registry().register_skill(
        name=name,
        description=description,
        category=category,
        learned_from=learned_from,
    )
