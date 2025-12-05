"""Internalization Curriculum - What Ara should learn vs outsource.

Not everything should be internalized. This module defines:
- Criteria for what makes something worth internalizing
- What should always stay with external teachers
- Thresholds and policies for skill promotion

The goal: Ara builds local skills for repetitive, low-risk tasks,
and keeps using teachers for novel, complex, or safety-critical work.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set

logger = logging.getLogger(__name__)


@dataclass
class InternalizationCriteria:
    """Criteria for when to internalize a skill."""

    id: str
    name: str
    description: str

    # Frequency requirements
    min_frequency_per_week: int = 3
    min_total_examples: int = 5

    # Quality requirements
    min_success_rate: float = 0.8
    max_risk_level: str = "medium"  # "low", "medium", "high"

    # Context requirements
    requires_tags: List[str] = field(default_factory=list)  # Must have ALL
    forbids_tags: List[str] = field(default_factory=list)  # Must have NONE

    # Teacher requirements
    min_teachers_agreeing: int = 1  # How many teachers should have similar patterns
    preferred_teachers: List[str] = field(default_factory=list)

    def matches(
        self,
        frequency_per_week: float,
        total_examples: int,
        success_rate: float,
        risk_level: str,
        tags: List[str],
        teachers_with_pattern: int,
    ) -> tuple[bool, List[str]]:
        """Check if a skill candidate matches these criteria.

        Returns (matches, reasons)
        """
        reasons = []
        risk_order = ["low", "medium", "high"]

        if frequency_per_week < self.min_frequency_per_week:
            reasons.append(f"frequency {frequency_per_week:.1f}/week < {self.min_frequency_per_week}")

        if total_examples < self.min_total_examples:
            reasons.append(f"examples {total_examples} < {self.min_total_examples}")

        if success_rate < self.min_success_rate:
            reasons.append(f"success rate {success_rate:.0%} < {self.min_success_rate:.0%}")

        if risk_order.index(risk_level) > risk_order.index(self.max_risk_level):
            reasons.append(f"risk level {risk_level} > {self.max_risk_level}")

        for required_tag in self.requires_tags:
            if required_tag not in tags:
                reasons.append(f"missing required tag: {required_tag}")

        for forbidden_tag in self.forbids_tags:
            if forbidden_tag in tags:
                reasons.append(f"has forbidden tag: {forbidden_tag}")

        if teachers_with_pattern < self.min_teachers_agreeing:
            reasons.append(f"only {teachers_with_pattern} teachers have pattern, need {self.min_teachers_agreeing}")

        return len(reasons) == 0, reasons

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "min_frequency_per_week": self.min_frequency_per_week,
            "min_total_examples": self.min_total_examples,
            "min_success_rate": self.min_success_rate,
            "max_risk_level": self.max_risk_level,
            "requires_tags": self.requires_tags,
            "forbids_tags": self.forbids_tags,
            "min_teachers_agreeing": self.min_teachers_agreeing,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InternalizationCriteria":
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            min_frequency_per_week=data.get("min_frequency_per_week", 3),
            min_total_examples=data.get("min_total_examples", 5),
            min_success_rate=data.get("min_success_rate", 0.8),
            max_risk_level=data.get("max_risk_level", "medium"),
            requires_tags=data.get("requires_tags", []),
            forbids_tags=data.get("forbids_tags", []),
            min_teachers_agreeing=data.get("min_teachers_agreeing", 1),
        )


@dataclass
class AvoidanceRule:
    """Rule for what should NOT be internalized."""

    id: str
    name: str
    description: str
    reason: str

    # Match criteria
    match_tags: List[str] = field(default_factory=list)  # Match ANY
    match_intents: List[str] = field(default_factory=list)
    match_keywords: List[str] = field(default_factory=list)

    # Action
    always_escalate_to: Optional[str] = None  # Specific teacher to use

    def matches(
        self,
        tags: List[str],
        intent: str,
        keywords: List[str],
    ) -> bool:
        """Check if this avoidance rule matches."""
        # Check tags
        for tag in self.match_tags:
            if tag in tags:
                return True

        # Check intent
        if intent in self.match_intents:
            return True

        # Check keywords
        for kw in self.match_keywords:
            if kw in keywords:
                return True

        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "reason": self.reason,
            "match_tags": self.match_tags,
            "match_intents": self.match_intents,
            "match_keywords": self.match_keywords,
            "always_escalate_to": self.always_escalate_to,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AvoidanceRule":
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            reason=data.get("reason", ""),
            match_tags=data.get("match_tags", []),
            match_intents=data.get("match_intents", []),
            match_keywords=data.get("match_keywords", []),
            always_escalate_to=data.get("always_escalate_to"),
        )


@dataclass
class Curriculum:
    """The complete internalization curriculum."""

    version: str = "1.0.0"

    # What to internalize
    priorities: List[InternalizationCriteria] = field(default_factory=list)

    # What to avoid
    avoid_rules: List[AvoidanceRule] = field(default_factory=list)

    # General settings
    default_review_period_days: int = 7
    auto_promote_threshold: float = 0.9
    auto_demote_threshold: float = 0.5

    def should_internalize(
        self,
        frequency_per_week: float,
        total_examples: int,
        success_rate: float,
        risk_level: str,
        tags: List[str],
        intent: str,
        keywords: List[str],
        teachers_with_pattern: int,
    ) -> tuple[bool, str]:
        """Decide if something should be internalized.

        Returns (should_internalize, reason)
        """
        # First check avoidance rules
        for rule in self.avoid_rules:
            if rule.matches(tags, intent, keywords):
                return False, f"Avoidance rule '{rule.name}': {rule.reason}"

        # Check priorities
        for criteria in self.priorities:
            matches, reasons = criteria.matches(
                frequency_per_week=frequency_per_week,
                total_examples=total_examples,
                success_rate=success_rate,
                risk_level=risk_level,
                tags=tags,
                teachers_with_pattern=teachers_with_pattern,
            )
            if matches:
                return True, f"Matches priority '{criteria.name}'"

        return False, "No matching priority criteria"

    def get_escalation_teacher(
        self,
        tags: List[str],
        intent: str,
        keywords: List[str],
    ) -> Optional[str]:
        """Get the teacher to escalate to for a given context."""
        for rule in self.avoid_rules:
            if rule.matches(tags, intent, keywords) and rule.always_escalate_to:
                return rule.always_escalate_to
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "priorities": [p.to_dict() for p in self.priorities],
            "avoid_rules": [r.to_dict() for r in self.avoid_rules],
            "default_review_period_days": self.default_review_period_days,
            "auto_promote_threshold": self.auto_promote_threshold,
            "auto_demote_threshold": self.auto_demote_threshold,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Curriculum":
        curriculum = cls(
            version=data.get("version", "1.0.0"),
            default_review_period_days=data.get("default_review_period_days", 7),
            auto_promote_threshold=data.get("auto_promote_threshold", 0.9),
            auto_demote_threshold=data.get("auto_demote_threshold", 0.5),
        )

        for p_data in data.get("priorities", []):
            curriculum.priorities.append(InternalizationCriteria.from_dict(p_data))

        for r_data in data.get("avoid_rules", []):
            curriculum.avoid_rules.append(AvoidanceRule.from_dict(r_data))

        return curriculum


class CurriculumManager:
    """Manages the internalization curriculum."""

    def __init__(self, curriculum_path: Optional[Path] = None):
        """Initialize the manager.

        Args:
            curriculum_path: Path to curriculum YAML/JSON file
        """
        self.curriculum_path = curriculum_path or (
            Path.home() / ".ara" / "academy" / "curriculum" / "what_to_internalize.json"
        )
        self.curriculum_path.parent.mkdir(parents=True, exist_ok=True)

        self._curriculum: Optional[Curriculum] = None
        self._loaded = False

    def _load(self, force: bool = False) -> None:
        """Load curriculum from disk."""
        if self._loaded and not force:
            return

        if self.curriculum_path.exists():
            try:
                with open(self.curriculum_path) as f:
                    data = json.load(f)
                self._curriculum = Curriculum.from_dict(data)
            except Exception as e:
                logger.warning(f"Failed to load curriculum: {e}")
                self._curriculum = self._create_default()
        else:
            self._curriculum = self._create_default()
            self._save()

        self._loaded = True

    def _save(self) -> None:
        """Save curriculum to disk."""
        if self._curriculum:
            with open(self.curriculum_path, "w") as f:
                json.dump(self._curriculum.to_dict(), f, indent=2)

    def _create_default(self) -> Curriculum:
        """Create default curriculum."""
        curriculum = Curriculum()

        # Priority: Fast inner loop tasks
        curriculum.priorities.append(InternalizationCriteria(
            id="PRI-001",
            name="fast_inner_loop",
            description="Frequent, low-risk tasks that block the user",
            min_frequency_per_week=5,
            min_total_examples=10,
            min_success_rate=0.8,
            max_risk_level="low",
        ))

        # Priority: Tool glue
        curriculum.priorities.append(InternalizationCriteria(
            id="PRI-002",
            name="tool_glue",
            description="Glue code between Ara and lab tools",
            min_frequency_per_week=3,
            min_total_examples=5,
            min_success_rate=0.75,
            max_risk_level="medium",
            requires_tags=["tooling", "automation"],
        ))

        # Priority: Boilerplate generation
        curriculum.priorities.append(InternalizationCriteria(
            id="PRI-003",
            name="boilerplate_codegen",
            description="Standard boilerplate code generation",
            min_frequency_per_week=2,
            min_total_examples=8,
            min_success_rate=0.85,
            max_risk_level="low",
            requires_tags=["codegen"],
        ))

        # Avoidance: Novel theory
        curriculum.avoid_rules.append(AvoidanceRule(
            id="AVOID-001",
            name="novel_theory_derivation",
            description="Novel theoretical work",
            reason="Requires deep reasoning and creativity",
            match_tags=["theory", "derivation", "proof", "novel"],
            match_intents=["research", "derive"],
            always_escalate_to="gemini",
        ))

        # Avoidance: Safety-critical
        curriculum.avoid_rules.append(AvoidanceRule(
            id="AVOID-002",
            name="safety_critical_mods",
            description="Safety-critical modifications",
            reason="Requires human review and teacher verification",
            match_tags=["safety", "critical", "hardware", "flash"],
            match_intents=["flash", "modify_hardware"],
            always_escalate_to="nova",
        ))

        # Avoidance: Hardware procedures
        curriculum.avoid_rules.append(AvoidanceRule(
            id="AVOID-003",
            name="hardware_flash_procedures",
            description="Hardware flashing and low-level procedures",
            reason="High risk of hardware damage",
            match_keywords=["flash", "bitstream", "firmware", "overclock"],
            always_escalate_to="nova",
        ))

        return curriculum

    def get_curriculum(self) -> Curriculum:
        """Get the curriculum."""
        self._load()
        return self._curriculum

    def should_internalize(
        self,
        frequency_per_week: float,
        total_examples: int,
        success_rate: float,
        risk_level: str,
        tags: List[str],
        intent: str,
        keywords: List[str],
        teachers_with_pattern: int,
    ) -> tuple[bool, str]:
        """Decide if something should be internalized."""
        self._load()
        return self._curriculum.should_internalize(
            frequency_per_week=frequency_per_week,
            total_examples=total_examples,
            success_rate=success_rate,
            risk_level=risk_level,
            tags=tags,
            intent=intent,
            keywords=keywords,
            teachers_with_pattern=teachers_with_pattern,
        )

    def add_priority(self, criteria: InternalizationCriteria) -> None:
        """Add a priority criterion."""
        self._load()
        self._curriculum.priorities.append(criteria)
        self._save()

    def add_avoidance_rule(self, rule: AvoidanceRule) -> None:
        """Add an avoidance rule."""
        self._load()
        self._curriculum.avoid_rules.append(rule)
        self._save()

    def get_summary(self) -> Dict[str, Any]:
        """Get curriculum summary."""
        self._load()
        return {
            "version": self._curriculum.version,
            "priorities": len(self._curriculum.priorities),
            "avoid_rules": len(self._curriculum.avoid_rules),
            "auto_promote_threshold": self._curriculum.auto_promote_threshold,
            "auto_demote_threshold": self._curriculum.auto_demote_threshold,
        }


# =============================================================================
# Convenience Functions
# =============================================================================

_default_manager: Optional[CurriculumManager] = None


def get_curriculum_manager() -> CurriculumManager:
    """Get the default curriculum manager."""
    global _default_manager
    if _default_manager is None:
        _default_manager = CurriculumManager()
    return _default_manager


def should_internalize(
    frequency_per_week: float,
    total_examples: int,
    success_rate: float,
    tags: List[str],
    intent: str,
) -> tuple[bool, str]:
    """Check if a skill should be internalized."""
    return get_curriculum_manager().should_internalize(
        frequency_per_week=frequency_per_week,
        total_examples=total_examples,
        success_rate=success_rate,
        risk_level="medium",
        tags=tags,
        intent=intent,
        keywords=[],
        teachers_with_pattern=1,
    )
