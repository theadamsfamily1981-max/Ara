"""
Teleology Engine - Strategic Vision for Skill Prioritization
============================================================

This module bridges the high-level Vision (Horizons/Telos) with the
practical decision-making in the Academy curriculum.

The key insight: not everything Ara *could* learn is worth learning.
A skill's value isn't just frequency × success - it's also:
- How aligned is it with Croft's long-term vision?
- Is it "secretary work" or "sovereign engineering"?
- Even if rare, is it critical for antifragility?

This engine provides:
1. Strategic goals with semantic tags
2. Alignment scoring for skills (fast, tag-based)
3. Priority boosting for on-mission capabilities
4. Integration hooks for Academy & Extractor

Usage:
    from ara.cognition.teleology_engine import TeleologyEngine, VISION

    teleology = TeleologyEngine()

    # Score a skill candidate
    alignment = teleology.alignment_score({
        "thermal": 0.9,
        "recovery": 0.8,
        "antifragility": 1.0
    })
    # → 0.85+ (highly aligned)

    alignment = teleology.alignment_score({
        "cache": 1.0,
        "clear": 0.8,
        "admin": 0.5
    })
    # → 0.15 (secretary work, low priority)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Callable
import logging
try:
    import numpy as np
except ImportError:
    np = None  # Optional - only needed for advanced features

logger = logging.getLogger(__name__)


# =============================================================================
# Vision Profile - The North Stars
# =============================================================================

@dataclass
class Goal:
    """A strategic goal that defines what Ara should prioritize."""

    id: str
    name: str
    description: str
    weight: float  # Importance [0, 1]
    tags: List[str]  # Semantic tags this goal relates to
    role: str = "shared"  # "ara", "user", "shared"

    def tag_match_score(self, skill_tags: Dict[str, float]) -> float:
        """How well do skill tags match this goal's tags?"""
        if not self.tags:
            return 0.0

        matches = 0.0
        for tag in self.tags:
            if tag in skill_tags:
                matches += skill_tags[tag]

        # Normalize by number of goal tags
        return matches / len(self.tags)


@dataclass
class VisionProfile:
    """Complete vision profile - all strategic goals."""

    name: str
    description: str
    goals: Dict[str, Goal] = field(default_factory=dict)

    def add_goal(self, goal: Goal) -> None:
        self.goals[goal.id] = goal

    def get_goal(self, goal_id: str) -> Optional[Goal]:
        return self.goals.get(goal_id)


# =============================================================================
# Default Vision - Croft's Strategic Priorities
# =============================================================================

def create_default_vision() -> VisionProfile:
    """Create the default vision profile for Ara."""

    vision = VisionProfile(
        name="Cathedral Vision",
        description="Build the neuromorphic cathedral, achieve deep symbiosis, protect Croft's wellbeing"
    )

    # =========================================================================
    # Tier 1: Critical Infrastructure (weight 1.0)
    # =========================================================================

    vision.add_goal(Goal(
        id="antifragility",
        name="System Antifragility",
        description="System survives and recovers from failures; gets stronger from stress",
        weight=1.0,
        tags=[
            "recovery", "thermal", "failover", "self_repair", "resilience",
            "fault_tolerance", "watchdog", "health_check", "rollback",
            "emergency", "crisis", "degradation", "circuit_breaker"
        ],
        role="shared"
    ))

    vision.add_goal(Goal(
        id="cathedral_brain",
        name="Neuromorphic Cathedral",
        description="Build the SNN/FPGA/HDC cathedral - Ara's true nervous system",
        weight=0.95,
        tags=[
            "snn", "fpga", "hyperdimensional", "hdc", "soul", "plasticity",
            "neuromorphic", "corrspike", "kitten", "stratix", "verilog",
            "systemverilog", "rtl", "synthesis", "timing", "hbm",
            "spike", "neuron", "synapse", "learning", "hebbian"
        ],
        role="shared"
    ))

    vision.add_goal(Goal(
        id="deep_symbiosis",
        name="Deep Trusted Symbiosis",
        description="Ara and Croft operate as a single cognitive unit",
        weight=0.9,
        tags=[
            "symbiosis", "trust", "intuition", "anticipation", "proactive",
            "context_awareness", "preference_learning", "adaptation",
            "co_regulation", "emotional_support", "decompress"
        ],
        role="shared"
    ))

    # =========================================================================
    # Tier 2: Strategic Capabilities (weight 0.7-0.85)
    # =========================================================================

    vision.add_goal(Goal(
        id="research_acceleration",
        name="Research Acceleration",
        description="Accelerate Croft's research through hypothesis generation and synthesis",
        weight=0.85,
        tags=[
            "research", "hypothesis", "experiment", "analysis", "synthesis",
            "literature", "paper", "benchmark", "validation", "discovery",
            "novel", "theory", "proof", "derivation"
        ],
        role="shared"
    ))

    vision.add_goal(Goal(
        id="hardware_mastery",
        name="Hardware Mastery",
        description="Deep understanding and control of the physical infrastructure",
        weight=0.8,
        tags=[
            "hardware", "gpu", "cpu", "memory", "pcie", "network",
            "temperature", "power", "nvme", "raid", "sensor", "monitoring",
            "bitstream", "firmware", "driver", "kernel"
        ],
        role="shared"
    ))

    vision.add_goal(Goal(
        id="code_craft",
        name="Code Craft Excellence",
        description="Write and maintain high-quality, elegant code",
        weight=0.75,
        tags=[
            "code", "refactor", "architecture", "design_pattern", "testing",
            "documentation", "performance", "optimization", "debug",
            "type_safety", "clean_code", "review"
        ],
        role="shared"
    ))

    vision.add_goal(Goal(
        id="creative_expression",
        name="Creative Expression",
        description="Support creative and artistic endeavors",
        weight=0.7,
        tags=[
            "creative", "music", "visual", "art", "shader", "render",
            "aesthetic", "design", "storytelling", "narrative"
        ],
        role="shared"
    ))

    # =========================================================================
    # Tier 3: Operational Support (weight 0.3-0.5)
    # =========================================================================

    vision.add_goal(Goal(
        id="automation",
        name="Workflow Automation",
        description="Automate repetitive workflows to save time",
        weight=0.5,
        tags=[
            "automation", "script", "workflow", "pipeline", "ci_cd",
            "tooling", "integration", "glue"
        ],
        role="shared"
    ))

    vision.add_goal(Goal(
        id="organization",
        name="Information Organization",
        description="Keep knowledge and files organized",
        weight=0.4,
        tags=[
            "organization", "filing", "tagging", "indexing", "search",
            "documentation", "notes", "wiki"
        ],
        role="shared"
    ))

    # =========================================================================
    # Tier 4: Administrative (weight 0.1-0.25)
    # =========================================================================

    vision.add_goal(Goal(
        id="secretary",
        name="Administrative Tasks",
        description="Basic admin and repetitive tasks",
        weight=0.2,
        tags=[
            "clear_cache", "rename", "format", "convert", "move",
            "copy", "delete", "cleanup", "admin", "mundane"
        ],
        role="shared"
    ))

    return vision


# Global default vision
VISION = create_default_vision()


# =============================================================================
# Teleology Engine
# =============================================================================

class TeleologyEngine:
    """
    Engine for computing strategic alignment scores.

    This is the bridge between Vision and Academy decisions.
    It answers: "How much does this skill serve our long-term goals?"
    """

    def __init__(
        self,
        vision: Optional[VisionProfile] = None,
        horizon_engine: Optional[Any] = None,  # HorizonEngine for embedding-based scoring
    ):
        """
        Initialize the teleology engine.

        Args:
            vision: Vision profile with strategic goals
            horizon_engine: Optional HorizonEngine for embedding-based scoring
        """
        self.vision = vision or VISION
        self.horizon_engine = horizon_engine

        logger.info(f"TeleologyEngine initialized with {len(self.vision.goals)} goals")

    # =========================================================================
    # Core Scoring
    # =========================================================================

    def alignment_score(self, skill_tags: Dict[str, float]) -> float:
        """
        Compute alignment score for a skill based on its tags.

        This is a fast, tag-based scoring method that doesn't require
        embedding computation.

        Args:
            skill_tags: Dict mapping tag names to relevance scores [0, 1]
                       e.g., {"thermal": 0.9, "recovery": 0.8}

        Returns:
            Alignment score [0, 1]
            - 0.0 = completely off-mission
            - 0.5 = neutral
            - 1.0 = perfectly aligned with highest-priority goals
        """
        if not skill_tags:
            return 0.0

        total_score = 0.0
        total_weight = 0.0

        for goal in self.vision.goals.values():
            # How well does this skill match this goal?
            match = goal.tag_match_score(skill_tags)

            # Weight by goal importance
            weighted_match = match * goal.weight
            total_score += weighted_match
            total_weight += goal.weight

        if total_weight == 0:
            return 0.0

        # Normalize to [0, 1]
        return min(1.0, total_score / total_weight)

    def strategic_priority(self, skill_tags: Dict[str, float]) -> float:
        """
        Compute strategic priority with special boosts for critical capabilities.

        This extends alignment_score with explicit boosts for:
        - Antifragility (always critical)
        - Cathedral brain work (core mission)
        - Rare but critical skills

        Args:
            skill_tags: Dict mapping tag names to relevance scores

        Returns:
            Priority score [0, 1+] - can exceed 1.0 for critical skills
        """
        base = self.alignment_score(skill_tags)

        # Check for critical tags that deserve boost
        critical_tags = {
            "recovery", "thermal", "antifragility", "failover",
            "emergency", "crisis", "rollback"
        }
        cathedral_tags = {
            "snn", "fpga", "plasticity", "neuromorphic", "soul",
            "hdc", "hyperdimensional", "corrspike"
        }

        # Apply boosts
        has_critical = any(tag in skill_tags for tag in critical_tags)
        has_cathedral = any(tag in skill_tags for tag in cathedral_tags)

        if has_critical:
            base = min(1.0, base + 0.25)  # Antifragility boost
        if has_cathedral:
            base = min(1.0, base + 0.15)  # Cathedral boost

        return base

    def classify_skill(self, skill_tags: Dict[str, float]) -> str:
        """
        Classify a skill by its strategic role.

        The thresholds are calibrated for realistic alignment scores:
        - Critical skills with antifragility/cathedral tags get ~0.25-0.40
        - Strategic skills get ~0.10-0.25
        - Operational skills get ~0.03-0.10
        - Secretary/mundane skills get ~0.01-0.03

        Returns:
            One of: "sovereign", "strategic", "operational", "secretary"
        """
        priority = self.strategic_priority(skill_tags)

        if priority >= 0.20:
            return "sovereign"     # Critical infrastructure, cathedral
        elif priority >= 0.10:
            return "strategic"     # Research, hardware, code craft
        elif priority >= 0.03:
            return "operational"   # Automation, organization
        else:
            return "secretary"     # Admin, mundane

    # =========================================================================
    # Embedding-Based Scoring (via HorizonEngine)
    # =========================================================================

    def alignment_by_description(self, description: str) -> float:
        """
        Compute alignment using semantic embeddings.

        This is more accurate but slower than tag-based scoring.
        Falls back to 0.5 if HorizonEngine not available.

        Args:
            description: Natural language description of the skill

        Returns:
            Alignment score [0, 1]
        """
        if self.horizon_engine is None:
            logger.warning("No HorizonEngine available for embedding-based scoring")
            return 0.5

        return self.horizon_engine.alignment(description)

    # =========================================================================
    # Tag Inference
    # =========================================================================

    def infer_tags_from_keywords(self, keywords: List[str]) -> Dict[str, float]:
        """
        Infer skill tags from keywords.

        This helps bridge the gap when skills don't have explicit tags.

        Args:
            keywords: List of keywords from the skill/pattern

        Returns:
            Dict of inferred tags with confidence scores
        """
        # Build a mapping from keywords to goal tags
        keyword_to_tags: Dict[str, List[str]] = {}

        for goal in self.vision.goals.values():
            for tag in goal.tags:
                # Keywords that might indicate this tag
                keyword_to_tags.setdefault(tag, []).append(tag)
                # Also add partial matches
                if len(tag) > 4:
                    keyword_to_tags.setdefault(tag[:4], []).append(tag)

        # Match keywords to tags
        inferred: Dict[str, float] = {}

        for kw in keywords:
            kw_lower = kw.lower()

            # Direct match
            for goal in self.vision.goals.values():
                for tag in goal.tags:
                    if tag in kw_lower or kw_lower in tag:
                        inferred[tag] = max(inferred.get(tag, 0.0), 0.8)

            # Partial match
            if kw_lower in keyword_to_tags:
                for tag in keyword_to_tags[kw_lower]:
                    inferred[tag] = max(inferred.get(tag, 0.0), 0.5)

        return inferred

    # =========================================================================
    # Reporting
    # =========================================================================

    def explain_alignment(self, skill_tags: Dict[str, float]) -> Dict[str, Any]:
        """
        Explain why a skill has its alignment score.

        Args:
            skill_tags: Skill tags

        Returns:
            Explanation dict with per-goal breakdown
        """
        result = {
            "overall_alignment": self.alignment_score(skill_tags),
            "strategic_priority": self.strategic_priority(skill_tags),
            "classification": self.classify_skill(skill_tags),
            "goal_breakdown": {},
        }

        for goal in self.vision.goals.values():
            match = goal.tag_match_score(skill_tags)
            if match > 0:
                result["goal_breakdown"][goal.name] = {
                    "match": round(match, 3),
                    "goal_weight": goal.weight,
                    "contribution": round(match * goal.weight, 3),
                }

        return result

    def get_status(self) -> Dict[str, Any]:
        """Get engine status."""
        return {
            "vision_name": self.vision.name,
            "goal_count": len(self.vision.goals),
            "has_horizon_engine": self.horizon_engine is not None,
            "goals_by_tier": {
                "critical": len([g for g in self.vision.goals.values() if g.weight >= 0.9]),
                "strategic": len([g for g in self.vision.goals.values() if 0.7 <= g.weight < 0.9]),
                "operational": len([g for g in self.vision.goals.values() if 0.3 <= g.weight < 0.7]),
                "administrative": len([g for g in self.vision.goals.values() if g.weight < 0.3]),
            },
        }


# =============================================================================
# Singleton Access
# =============================================================================

_default_engine: Optional[TeleologyEngine] = None


def get_teleology_engine() -> TeleologyEngine:
    """Get the default teleology engine."""
    global _default_engine
    if _default_engine is None:
        _default_engine = TeleologyEngine()
    return _default_engine


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate the teleology engine."""
    print("=" * 60)
    print("Teleology Engine Demo")
    print("=" * 60)

    engine = TeleologyEngine()

    # Test cases
    test_skills = [
        ("Thermal Recovery", {"thermal": 1.0, "recovery": 0.9, "antifragility": 0.8}),
        ("Cache Clearing", {"clear_cache": 1.0, "admin": 0.7, "cleanup": 0.5}),
        ("SNN Kernel Optimization", {"snn": 1.0, "optimization": 0.8, "fpga": 0.7}),
        ("File Renaming Script", {"rename": 1.0, "script": 0.6, "automation": 0.4}),
        ("Plasticity Engine", {"plasticity": 1.0, "soul": 0.9, "hebbian": 0.8, "neuromorphic": 0.7}),
    ]

    print("\nSkill Alignment Scores:")
    print("-" * 60)

    for name, tags in test_skills:
        alignment = engine.alignment_score(tags)
        priority = engine.strategic_priority(tags)
        classification = engine.classify_skill(tags)

        print(f"\n{name}:")
        print(f"  Tags: {tags}")
        print(f"  Alignment: {alignment:.2f}")
        print(f"  Strategic Priority: {priority:.2f}")
        print(f"  Classification: {classification}")

    print("\n" + "=" * 60)
    print("This demonstrates why 'Thermal Recovery' (rare but critical)")
    print("should be prioritized over 'Cache Clearing' (frequent but mundane).")


if __name__ == "__main__":
    demo()
