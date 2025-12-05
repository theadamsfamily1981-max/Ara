"""Skill Capsules - Ara packages reusable skills.

A skill capsule is a self-contained, transferable package of knowledge:
- What the skill does
- Which teachers work best for it
- Prompt templates that succeed
- Success criteria and metrics
- Example interactions

Think of it like Ara "minting" a skill she's mastered into a portable format.

Example capsule:
  skill_id: debug_python_async
  description: "Debug async/await issues in Python code"
  best_teachers: ["claude"]
  template_ids: ["TPL-0001-cla-debu"]
  success_rate: 0.87
  triggers: ["async", "await", "asyncio", "coroutine"]
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class SkillExample:
    """An example interaction for a skill."""

    query: str
    response_summary: str
    teachers_used: List[str]
    success: bool
    reward: float = 0.0
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "response_summary": self.response_summary,
            "teachers_used": self.teachers_used,
            "success": self.success,
            "reward": self.reward,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillExample":
        return cls(
            query=data["query"],
            response_summary=data.get("response_summary", ""),
            teachers_used=data.get("teachers_used", []),
            success=data.get("success", True),
            reward=data.get("reward", 0.0),
            notes=data.get("notes", ""),
        )


@dataclass
class SkillCapsule:
    """A packaged, reusable skill definition."""

    id: str
    name: str
    description: str

    # What triggers this skill
    intent: str  # Primary intent
    triggers: List[str] = field(default_factory=list)  # Keywords
    patterns: List[str] = field(default_factory=list)  # Regex patterns

    # How to execute it
    best_teachers: List[str] = field(default_factory=list)
    workflow: List[str] = field(default_factory=list)  # Default workflow
    template_ids: List[str] = field(default_factory=list)

    # Success criteria
    success_rate: float = 0.0
    sample_count: int = 0
    min_confidence: float = 0.7

    # Examples
    examples: List[SkillExample] = field(default_factory=list)

    # Metadata
    version: str = "1.0.0"
    author: str = "ara"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    tags: List[str] = field(default_factory=list)

    # Status
    status: str = "active"  # "draft", "active", "deprecated"

    def matches_query(self, query: str) -> float:
        """Check if this skill matches a query.

        Returns confidence score 0-1.
        """
        query_lower = query.lower()
        score = 0.0

        # Check triggers
        trigger_matches = sum(1 for t in self.triggers if t.lower() in query_lower)
        if self.triggers:
            score += 0.4 * (trigger_matches / len(self.triggers))

        # Check patterns
        import re
        pattern_matches = 0
        for pattern in self.patterns:
            try:
                if re.search(pattern, query, re.IGNORECASE):
                    pattern_matches += 1
            except re.error:
                pass

        if self.patterns:
            score += 0.4 * (pattern_matches / len(self.patterns))

        # Boost if we have high success rate
        if self.sample_count >= 5 and self.success_rate >= 0.8:
            score += 0.2

        return min(1.0, score)

    def record_usage(self, success: bool, reward: float = 0.0) -> None:
        """Record a usage of this skill."""
        old_total = self.success_rate * self.sample_count
        self.sample_count += 1
        if success:
            self.success_rate = (old_total + 1) / self.sample_count
        else:
            self.success_rate = old_total / self.sample_count
        self.updated_at = datetime.utcnow()

    def add_example(self, example: SkillExample) -> None:
        """Add an example interaction."""
        self.examples.append(example)
        if len(self.examples) > 10:
            # Keep best examples
            self.examples.sort(key=lambda e: e.reward, reverse=True)
            self.examples = self.examples[:10]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "intent": self.intent,
            "triggers": self.triggers,
            "patterns": self.patterns,
            "best_teachers": self.best_teachers,
            "workflow": self.workflow,
            "template_ids": self.template_ids,
            "success_rate": round(self.success_rate, 3),
            "sample_count": self.sample_count,
            "min_confidence": self.min_confidence,
            "examples": [e.to_dict() for e in self.examples],
            "version": self.version,
            "author": self.author,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "tags": self.tags,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillCapsule":
        capsule = cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            intent=data.get("intent", "general"),
            triggers=data.get("triggers", []),
            patterns=data.get("patterns", []),
            best_teachers=data.get("best_teachers", []),
            workflow=data.get("workflow", []),
            template_ids=data.get("template_ids", []),
            success_rate=data.get("success_rate", 0.0),
            sample_count=data.get("sample_count", 0),
            min_confidence=data.get("min_confidence", 0.7),
            version=data.get("version", "1.0.0"),
            author=data.get("author", "ara"),
            tags=data.get("tags", []),
            status=data.get("status", "active"),
        )

        for ex_data in data.get("examples", []):
            capsule.examples.append(SkillExample.from_dict(ex_data))

        return capsule

    def format_yaml(self) -> str:
        """Format as YAML for human readability."""
        lines = [
            f"# Skill Capsule: {self.name}",
            f"id: {self.id}",
            f"version: {self.version}",
            f"status: {self.status}",
            "",
            f"description: |",
            f"  {self.description}",
            "",
            f"intent: {self.intent}",
            f"triggers: {self.triggers}",
            "",
            "execution:",
            f"  best_teachers: {self.best_teachers}",
            f"  workflow: {self.workflow}",
            f"  template_ids: {self.template_ids}",
            "",
            "performance:",
            f"  success_rate: {self.success_rate:.0%}",
            f"  sample_count: {self.sample_count}",
            "",
            f"tags: {self.tags}",
        ]

        if self.examples:
            lines.append("")
            lines.append("examples:")
            for i, ex in enumerate(self.examples[:3], 1):
                lines.append(f"  - query: \"{ex.query[:50]}...\"")
                lines.append(f"    success: {ex.success}")

        return "\n".join(lines)


class CapsuleManager:
    """Manages skill capsules."""

    def __init__(self, capsules_path: Optional[Path] = None):
        """Initialize the manager.

        Args:
            capsules_path: Path to capsules JSON file
        """
        self.capsules_path = capsules_path or (
            Path.home() / ".ara" / "meta" / "toolsmith" / "capsules.json"
        )
        self.capsules_path.parent.mkdir(parents=True, exist_ok=True)

        self._capsules: Dict[str, SkillCapsule] = {}
        self._loaded = False
        self._next_id = 1

    def _load(self, force: bool = False) -> None:
        """Load capsules from disk."""
        if self._loaded and not force:
            return

        self._capsules.clear()

        if self.capsules_path.exists():
            try:
                with open(self.capsules_path) as f:
                    data = json.load(f)
                for cap_data in data.get("capsules", []):
                    capsule = SkillCapsule.from_dict(cap_data)
                    self._capsules[capsule.id] = capsule
                    # Update ID counter
                    if capsule.id.startswith("SKILL-"):
                        try:
                            num = int(capsule.id[6:10])
                            self._next_id = max(self._next_id, num + 1)
                        except ValueError:
                            pass
            except Exception as e:
                logger.warning(f"Failed to load capsules: {e}")

        self._loaded = True

    def _save(self) -> None:
        """Save capsules to disk."""
        data = {
            "version": 1,
            "updated_at": datetime.utcnow().isoformat(),
            "capsules": [c.to_dict() for c in self._capsules.values()],
        }
        with open(self.capsules_path, "w") as f:
            json.dump(data, f, indent=2)

    def _generate_id(self, name: str) -> str:
        """Generate a unique capsule ID."""
        slug = name.lower().replace(" ", "_")[:10]
        id_str = f"SKILL-{self._next_id:04d}-{slug}"
        self._next_id += 1
        return id_str

    def get_capsule(self, capsule_id: str) -> Optional[SkillCapsule]:
        """Get a capsule by ID."""
        self._load()
        return self._capsules.get(capsule_id)

    def get_all_capsules(self) -> List[SkillCapsule]:
        """Get all capsules."""
        self._load()
        return list(self._capsules.values())

    def get_active_capsules(self) -> List[SkillCapsule]:
        """Get all active capsules."""
        self._load()
        return [c for c in self._capsules.values() if c.status == "active"]

    def get_capsules_for_intent(self, intent: str) -> List[SkillCapsule]:
        """Get capsules matching an intent."""
        self._load()
        return [c for c in self._capsules.values() if c.intent == intent]

    def find_matching_capsules(
        self,
        query: str,
        min_confidence: float = 0.3,
        limit: int = 5,
    ) -> List[tuple[SkillCapsule, float]]:
        """Find capsules matching a query.

        Args:
            query: The query to match
            min_confidence: Minimum confidence threshold
            limit: Maximum results

        Returns:
            List of (capsule, confidence) tuples
        """
        self._load()

        matches = []
        for capsule in self._capsules.values():
            if capsule.status != "active":
                continue

            confidence = capsule.matches_query(query)
            if confidence >= min_confidence:
                matches.append((capsule, confidence))

        # Sort by confidence
        matches.sort(key=lambda x: x[1], reverse=True)

        return matches[:limit]

    def create_capsule(
        self,
        name: str,
        description: str,
        intent: str,
        triggers: Optional[List[str]] = None,
        best_teachers: Optional[List[str]] = None,
        workflow: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
    ) -> SkillCapsule:
        """Create a new skill capsule.

        Args:
            name: Human-readable name
            description: What the skill does
            intent: Primary intent
            triggers: Keywords that trigger this skill
            best_teachers: Preferred teachers
            workflow: Default workflow
            tags: Categorization tags

        Returns:
            The new capsule
        """
        self._load()

        capsule = SkillCapsule(
            id=self._generate_id(name),
            name=name,
            description=description,
            intent=intent,
            triggers=triggers or [],
            best_teachers=best_teachers or [],
            workflow=workflow or best_teachers or [],
            tags=tags or [],
            status="draft",
        )

        self._capsules[capsule.id] = capsule
        self._save()
        logger.info(f"Created skill capsule: {capsule.id}")

        return capsule

    def mint_from_interactions(
        self,
        name: str,
        description: str,
        intent: str,
        interactions: List[Dict[str, Any]],
    ) -> Optional[SkillCapsule]:
        """Mint a capsule from successful interactions.

        Args:
            name: Skill name
            description: What it does
            intent: Primary intent
            interactions: List of interaction records

        Returns:
            New capsule if enough data, None otherwise
        """
        if len(interactions) < 3:
            return None

        # Analyze interactions
        successful = [i for i in interactions if i.get("success")]
        if len(successful) < 2:
            return None

        # Find common teachers
        teacher_counts: Dict[str, int] = defaultdict(int)
        for i in successful:
            for t in i.get("teachers", []):
                teacher_counts[t] += 1

        best_teachers = sorted(
            teacher_counts.keys(),
            key=lambda t: teacher_counts[t],
            reverse=True,
        )[:3]

        # Extract triggers from queries
        triggers = set()
        for i in successful:
            query = i.get("query", "").lower()
            # Extract significant words (simple approach)
            words = query.split()
            for word in words:
                if len(word) > 4 and word.isalpha():
                    triggers.add(word)

        # Create capsule
        capsule = self.create_capsule(
            name=name,
            description=description,
            intent=intent,
            triggers=list(triggers)[:10],
            best_teachers=best_teachers,
            workflow=best_teachers[:2] if len(best_teachers) > 1 else best_teachers,
        )

        # Add examples
        for i in successful[:5]:
            capsule.add_example(SkillExample(
                query=i.get("query", "")[:200],
                response_summary=i.get("response_summary", ""),
                teachers_used=i.get("teachers", []),
                success=True,
                reward=i.get("reward", 0.0),
            ))

        # Set statistics
        capsule.sample_count = len(interactions)
        capsule.success_rate = len(successful) / len(interactions)

        # Promote to active if good enough
        if capsule.success_rate >= 0.7 and capsule.sample_count >= 5:
            capsule.status = "active"

        self._save()
        return capsule

    def update_capsule_status(
        self,
        capsule_id: str,
        status: str,
    ) -> bool:
        """Update capsule status.

        Args:
            capsule_id: Capsule ID
            status: New status

        Returns:
            True if updated
        """
        self._load()

        capsule = self._capsules.get(capsule_id)
        if not capsule:
            return False

        capsule.status = status
        capsule.updated_at = datetime.utcnow()
        self._save()
        return True

    def record_capsule_usage(
        self,
        capsule_id: str,
        success: bool,
        reward: float = 0.0,
    ) -> bool:
        """Record usage of a capsule.

        Args:
            capsule_id: Capsule ID
            success: Whether it succeeded
            reward: Quality score

        Returns:
            True if recorded
        """
        self._load()

        capsule = self._capsules.get(capsule_id)
        if not capsule:
            return False

        capsule.record_usage(success, reward)

        # Auto-deprecate if performing poorly
        if capsule.sample_count >= 10 and capsule.success_rate < 0.5:
            capsule.status = "deprecated"
            logger.info(f"Auto-deprecated capsule {capsule_id} (success rate: {capsule.success_rate:.0%})")

        self._save()
        return True

    def export_capsule(
        self,
        capsule_id: str,
        output_dir: Optional[Path] = None,
    ) -> Optional[Path]:
        """Export a capsule to YAML.

        Args:
            capsule_id: Capsule to export
            output_dir: Output directory

        Returns:
            Path to exported file
        """
        self._load()

        capsule = self._capsules.get(capsule_id)
        if not capsule:
            return None

        output_dir = output_dir or (Path.home() / ".ara" / "meta" / "toolsmith" / "exports")
        output_dir.mkdir(parents=True, exist_ok=True)

        filepath = output_dir / f"{capsule.id}.yaml"
        with open(filepath, "w") as f:
            f.write(capsule.format_yaml())

        return filepath

    def get_summary(self) -> Dict[str, Any]:
        """Get manager summary."""
        self._load()

        active = [c for c in self._capsules.values() if c.status == "active"]
        drafts = [c for c in self._capsules.values() if c.status == "draft"]
        deprecated = [c for c in self._capsules.values() if c.status == "deprecated"]

        # Top performing
        top = sorted(
            [c for c in active if c.sample_count >= 3],
            key=lambda c: c.success_rate,
            reverse=True,
        )[:5]

        return {
            "total_capsules": len(self._capsules),
            "active": len(active),
            "drafts": len(drafts),
            "deprecated": len(deprecated),
            "top_performers": [
                {
                    "id": c.id,
                    "name": c.name,
                    "success_rate": c.success_rate,
                    "samples": c.sample_count,
                }
                for c in top
            ],
        }


# =============================================================================
# Default Skill Capsules
# =============================================================================

DEFAULT_CAPSULES = [
    {
        "id": "SKILL-0001-debug_pyth",
        "name": "Python Debugging",
        "description": "Debug Python code issues including errors, exceptions, and logic bugs",
        "intent": "debug_code",
        "triggers": ["python", "error", "exception", "traceback", "debug", "bug"],
        "best_teachers": ["claude"],
        "workflow": ["claude"],
        "tags": ["python", "debugging"],
    },
    {
        "id": "SKILL-0002-arch_desi",
        "name": "Architecture Design",
        "description": "Design system architectures and review design decisions",
        "intent": "design_arch",
        "triggers": ["architecture", "design", "system", "structure", "pattern"],
        "best_teachers": ["nova", "gemini"],
        "workflow": ["nova", "gemini"],
        "tags": ["architecture", "design"],
    },
    {
        "id": "SKILL-0003-code_revi",
        "name": "Code Review",
        "description": "Review code for issues, best practices, and improvements",
        "intent": "review",
        "triggers": ["review", "check", "audit", "improve", "refactor"],
        "best_teachers": ["nova", "claude"],
        "workflow": ["claude", "nova"],
        "tags": ["review", "quality"],
    },
    {
        "id": "SKILL-0004-impl_feat",
        "name": "Feature Implementation",
        "description": "Implement new features and functionality",
        "intent": "implement",
        "triggers": ["implement", "create", "build", "add", "feature"],
        "best_teachers": ["claude"],
        "workflow": ["claude", "nova"],
        "tags": ["implementation", "features"],
    },
    {
        "id": "SKILL-0005-research",
        "name": "Technical Research",
        "description": "Research technical topics and explore solutions",
        "intent": "research",
        "triggers": ["research", "explore", "investigate", "learn", "understand"],
        "best_teachers": ["gemini"],
        "workflow": ["gemini", "claude"],
        "tags": ["research", "exploration"],
    },
]


def seed_default_capsules(manager: CapsuleManager) -> int:
    """Seed default skill capsules.

    Args:
        manager: Capsule manager

    Returns:
        Number seeded
    """
    seeded = 0
    for cap_data in DEFAULT_CAPSULES:
        if not manager.get_capsule(cap_data["id"]):
            manager._load()
            capsule = SkillCapsule.from_dict(cap_data)
            capsule.status = "active"
            manager._capsules[capsule.id] = capsule
            seeded += 1

    if seeded:
        manager._save()

    return seeded


# =============================================================================
# Convenience Functions
# =============================================================================

_default_manager: Optional[CapsuleManager] = None


def get_capsule_manager() -> CapsuleManager:
    """Get the default capsule manager."""
    global _default_manager
    if _default_manager is None:
        _default_manager = CapsuleManager()
    return _default_manager


def find_skill_for_query(
    query: str,
    min_confidence: float = 0.3,
) -> Optional[tuple[SkillCapsule, float]]:
    """Find the best skill capsule for a query.

    Args:
        query: The query to match
        min_confidence: Minimum confidence

    Returns:
        (capsule, confidence) or None
    """
    matches = get_capsule_manager().find_matching_capsules(
        query, min_confidence, limit=1
    )
    return matches[0] if matches else None


def mint_skill_capsule(
    name: str,
    description: str,
    intent: str,
    interactions: List[Dict[str, Any]],
) -> Optional[SkillCapsule]:
    """Mint a new skill capsule from interactions."""
    return get_capsule_manager().mint_from_interactions(
        name, description, intent, interactions
    )
