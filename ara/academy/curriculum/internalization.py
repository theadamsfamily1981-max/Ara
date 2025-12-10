"""Internalization Curriculum - What Ara should learn vs outsource.

Not everything should be internalized. This module defines:
- Criteria for what makes something worth internalizing
- What should always stay with external teachers
- Thresholds and policies for skill promotion

The goal: Ara builds local skills for repetitive, low-risk tasks,
and keeps using teachers for novel, complex, or safety-critical work.

ITERATION 27 UPDATE (The Sovereign):
=====================================
Skills are now scored not just by frequency × success, but by:

    Score = base_productivity × vision_factor

Where:
    base_productivity = normalized_frequency × success_rate
    vision_factor = 1.0 + teleology_alignment

This means:
- "Thermal Recovery" (rare but critical) gets prioritized
- "Clear Cache" (frequent but mundane) gets deprioritized
- Cathedral/antifragility skills get hard floors even if seen once

The TeleologyEngine is now wired into all internalization decisions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set

# Import the TeleologyEngine for vision-aware scoring
from ara.cognition.teleology_engine import TeleologyEngine, get_teleology_engine

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
# Vision-Aware Internalization (Iteration 27: The Sovereign)
# =============================================================================

@dataclass
class SkillCandidate:
    """A candidate skill being evaluated for internalization."""

    name: str
    description: str

    # Usage statistics
    frequency_per_week: float = 0.0
    total_examples: int = 0
    success_rate: float = 0.0

    # Classification
    tags: Dict[str, float] = field(default_factory=dict)  # tag -> relevance
    risk_level: str = "medium"
    intent: str = ""
    keywords: List[str] = field(default_factory=list)

    # Teacher info
    teachers_with_pattern: int = 0
    primary_teacher: Optional[str] = None


class VisionAwareInternalization:
    """
    Vision-aware internalization scoring.

    This replaces the old frequency × success formula with:

        Score = base_productivity × vision_factor × (1 + alignment_floor)

    Where:
        - base_productivity = min(1.0, freq/7) × success_rate
        - vision_factor = 1.0 + strategic_priority
        - alignment_floor = hard minimum for critical skills

    The key insight: A skill that serves the Cathedral (even if rare)
    should be prioritized over a skill that's just frequent.
    """

    def __init__(
        self,
        teleology: Optional[TeleologyEngine] = None,
        curriculum: Optional[Curriculum] = None,
        base_threshold: float = 0.5,
    ):
        """
        Initialize vision-aware internalization.

        Args:
            teleology: TeleologyEngine for alignment scoring
            curriculum: Traditional curriculum for avoidance rules
            base_threshold: Minimum score to internalize
        """
        self.teleology = teleology or get_teleology_engine()
        self.curriculum = curriculum
        self.base_threshold = base_threshold

    def compute_score(self, candidate: SkillCandidate) -> Dict[str, Any]:
        """
        Compute internalization score for a skill candidate.

        Returns detailed breakdown including:
        - final_score: The overall score
        - should_internalize: Boolean decision
        - classification: sovereign/strategic/operational/secretary
        - breakdown: Component scores
        """
        # Normalize frequency (7+ times/week = saturated)
        norm_freq = min(1.0, candidate.frequency_per_week / 7.0)

        # Base productivity: how often × how well
        base_productivity = norm_freq * candidate.success_rate

        # Get alignment from teleology
        alignment = self.teleology.alignment_score(candidate.tags)
        strategic_priority = self.teleology.strategic_priority(candidate.tags)
        classification = self.teleology.classify_skill(candidate.tags)

        # Vision factor: boosts (or dampens) based on alignment
        # Neutral (0.5) = no change, aligned (1.0) = 2× boost
        vision_factor = 1.0 + strategic_priority

        # Calculate raw score
        raw_score = base_productivity * vision_factor

        # Apply alignment floor for critical skills
        # If a skill is highly aligned (sovereign/strategic) and has
        # decent success rate, force it above threshold
        alignment_floor = 0.0
        if classification in ("sovereign", "strategic"):
            if candidate.success_rate >= 0.6:
                # Rare but critical = hard floor
                alignment_floor = self.base_threshold * 1.1
            elif candidate.success_rate >= 0.4:
                alignment_floor = self.base_threshold * 0.8

        # Final score (for threshold comparison)
        final_score = max(raw_score, alignment_floor)

        # Priority score (for sorting/ranking)
        # Tier bonus ensures sovereign skills sort above secretary skills
        tier_bonus = {
            "sovereign": 10.0,    # Critical infrastructure always top priority
            "strategic": 5.0,     # Research, hardware mastery
            "operational": 2.0,   # Automation, organization
            "secretary": 0.0,     # Admin, mundane - no bonus
        }.get(classification, 0.0)
        priority_score = final_score + tier_bonus

        # Check avoidance rules if curriculum provided
        avoidance_reason = None
        if self.curriculum:
            for rule in self.curriculum.avoid_rules:
                if rule.matches(
                    list(candidate.tags.keys()),
                    candidate.intent,
                    candidate.keywords
                ):
                    avoidance_reason = f"{rule.name}: {rule.reason}"
                    break

        # Decision
        should_internalize = (
            final_score >= self.base_threshold and
            avoidance_reason is None
        )

        return {
            "final_score": round(final_score, 3),
            "priority_score": round(priority_score, 3),  # Use for sorting
            "should_internalize": should_internalize,
            "classification": classification,
            "avoidance_reason": avoidance_reason,
            "breakdown": {
                "frequency_per_week": candidate.frequency_per_week,
                "normalized_frequency": round(norm_freq, 3),
                "success_rate": candidate.success_rate,
                "base_productivity": round(base_productivity, 3),
                "alignment": round(alignment, 3),
                "strategic_priority": round(strategic_priority, 3),
                "vision_factor": round(vision_factor, 3),
                "alignment_floor": round(alignment_floor, 3),
                "raw_score": round(raw_score, 3),
                "tier_bonus": tier_bonus,
            },
        }

    def should_internalize(self, candidate: SkillCandidate) -> tuple[bool, str]:
        """
        Simplified interface: should we internalize this skill?

        Returns:
            (should_internalize, reason)
        """
        result = self.compute_score(candidate)

        if result["avoidance_reason"]:
            return False, f"Avoided: {result['avoidance_reason']}"

        if result["should_internalize"]:
            return True, (
                f"Score {result['final_score']:.2f} >= threshold "
                f"({result['classification']}, priority={result['breakdown']['strategic_priority']:.2f})"
            )
        else:
            return False, (
                f"Score {result['final_score']:.2f} < threshold "
                f"({result['classification']})"
            )

    def should_internalize_detailed(self, proposal: Any) -> tuple[bool, float, Optional[str]]:
        """
        Detailed interface for pipeline: should we internalize this skill?

        This interface is used by the SkillPipeline and triggers priority
        evolution for high-value skills.

        Args:
            proposal: A SkillProposal or similar object with name, pattern, etc.

        Returns:
            (should_internalize, score, rejection_reason or None)
        """
        # Extract candidate info from proposal
        pattern = getattr(proposal, 'pattern', None)
        candidate = SkillCandidate(
            name=getattr(proposal, 'name', 'unknown'),
            description=getattr(proposal, 'description', ''),
            frequency_per_week=getattr(proposal, 'frequency', 0.0),
            success_rate=getattr(proposal, 'success_rate', 0.0),
            tags={tag: 1.0 for tag in (getattr(pattern, 'tags', []) if pattern else [])},
            intent=getattr(pattern, 'intent', '') if pattern else '',
        )

        result = self.compute_score(candidate)
        should = result["should_internalize"]
        score = result["final_score"]
        reason = result["avoidance_reason"]

        # CRITICAL: If this is a high-value skill (sovereign/strategic),
        # trigger immediate hardening via the Ouroboros evolution engine
        if should and result["classification"] in ("sovereign", "strategic"):
            self._trigger_priority_evolution(candidate.name, result)

        return should, score, reason

    def _trigger_priority_evolution(self, skill_name: str, result: Dict[str, Any]) -> None:
        """
        Trigger priority evolution for high-value skills.

        When a sovereign or strategic skill is identified for internalization,
        we don't want to wait for the nightly batch - it should be hardened
        immediately by the Ouroboros.
        """
        try:
            from ara.meta.evolution import get_evolution_engine
            engine = get_evolution_engine()
            engine.schedule_priority_evolution(skill_name)
            logger.info(
                "Triggered priority evolution for high-value skill '%s' (%s)",
                skill_name, result["classification"]
            )
        except Exception as e:
            # Don't let evolution trigger failures block internalization
            logger.warning("Could not trigger priority evolution for '%s': %s", skill_name, e)

    def rank_candidates(
        self,
        candidates: List[SkillCandidate],
    ) -> List[tuple[SkillCandidate, Dict[str, Any]]]:
        """
        Rank multiple candidates by internalization priority.

        Returns list of (candidate, score_result) sorted by final_score descending.
        """
        scored = [(c, self.compute_score(c)) for c in candidates]
        return sorted(scored, key=lambda x: x[1]["final_score"], reverse=True)

    def compare_old_vs_new(self, candidate: SkillCandidate) -> Dict[str, Any]:
        """
        Compare old (frequency × success) vs new (vision-aware) scoring.

        Useful for understanding how the new system changes priorities.
        """
        # Old scoring
        old_score = candidate.frequency_per_week * candidate.success_rate / 7.0

        # New scoring
        new_result = self.compute_score(candidate)

        return {
            "candidate": candidate.name,
            "old_score": round(old_score, 3),
            "old_would_internalize": old_score >= self.base_threshold,
            "new_score": new_result["final_score"],
            "new_would_internalize": new_result["should_internalize"],
            "classification": new_result["classification"],
            "score_change": round(new_result["final_score"] - old_score, 3),
            "priority_shift": "↑" if new_result["final_score"] > old_score else "↓" if new_result["final_score"] < old_score else "→",
        }


# =============================================================================
# Convenience Functions
# =============================================================================

_default_manager: Optional[CurriculumManager] = None
_default_vision_aware: Optional[VisionAwareInternalization] = None


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
    """Check if a skill should be internalized (legacy interface)."""
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


def get_vision_aware_internalization() -> VisionAwareInternalization:
    """Get the default vision-aware internalization scorer."""
    global _default_vision_aware
    if _default_vision_aware is None:
        manager = get_curriculum_manager()
        _default_vision_aware = VisionAwareInternalization(
            curriculum=manager.get_curriculum()
        )
    return _default_vision_aware


def should_internalize_vision_aware(
    name: str,
    description: str,
    frequency_per_week: float,
    success_rate: float,
    tags: Dict[str, float],
    intent: str = "",
) -> tuple[bool, str, Dict[str, Any]]:
    """
    Check if a skill should be internalized using vision-aware scoring.

    This is the new preferred interface that uses teleology alignment.

    Args:
        name: Skill name
        description: Skill description
        frequency_per_week: How often this pattern occurs
        success_rate: Success rate [0, 1]
        tags: Dict of semantic tags with relevance scores
        intent: User intent classification

    Returns:
        (should_internalize, reason, full_score_breakdown)
    """
    scorer = get_vision_aware_internalization()

    candidate = SkillCandidate(
        name=name,
        description=description,
        frequency_per_week=frequency_per_week,
        success_rate=success_rate,
        tags=tags,
        intent=intent,
    )

    should, reason = scorer.should_internalize(candidate)
    breakdown = scorer.compute_score(candidate)

    return should, reason, breakdown


def demo_scoring_comparison():
    """
    Demonstrate how the new vision-aware scoring differs from the old.

    Shows why "Thermal Recovery" (rare but critical) gets prioritized
    over "Clear Cache" (frequent but mundane).
    """
    scorer = get_vision_aware_internalization()

    test_cases = [
        SkillCandidate(
            name="Thermal Recovery",
            description="Recover system from thermal event",
            frequency_per_week=0.5,  # Rare: once every 2 weeks
            success_rate=0.8,
            tags={"thermal": 1.0, "recovery": 0.9, "antifragility": 0.8},
        ),
        SkillCandidate(
            name="Clear Cache",
            description="Clear various caches",
            frequency_per_week=35.0,  # Very frequent: 5x/day
            success_rate=0.95,
            tags={"clear_cache": 1.0, "admin": 0.7, "cleanup": 0.5},
        ),
        SkillCandidate(
            name="SNN Kernel Optimization",
            description="Optimize SNN kernels for FPGA",
            frequency_per_week=2.0,  # Moderate
            success_rate=0.7,
            tags={"snn": 1.0, "optimization": 0.8, "fpga": 0.7, "neuromorphic": 0.6},
        ),
        SkillCandidate(
            name="Rename Files",
            description="Batch rename files",
            frequency_per_week=14.0,  # Twice a day
            success_rate=0.99,
            tags={"rename": 1.0, "admin": 0.8, "mundane": 0.5},
        ),
    ]

    print("=" * 70)
    print("Vision-Aware Internalization: Old vs New Scoring")
    print("=" * 70)

    for candidate in test_cases:
        comparison = scorer.compare_old_vs_new(candidate)

        print(f"\n{candidate.name}:")
        print(f"  Freq: {candidate.frequency_per_week}/week, Success: {candidate.success_rate:.0%}")
        print(f"  Old Score: {comparison['old_score']:.3f} → Would internalize: {comparison['old_would_internalize']}")
        print(f"  New Score: {comparison['new_score']:.3f} → Would internalize: {comparison['new_would_internalize']}")
        print(f"  Classification: {comparison['classification']}")
        print(f"  Priority Shift: {comparison['priority_shift']} ({comparison['score_change']:+.3f})")

    print("\n" + "=" * 70)
    print("KEY INSIGHT: Thermal Recovery (rare) now outranks Clear Cache (frequent)")
    print("because it serves the Cathedral vision of antifragility.")
    print("=" * 70)
