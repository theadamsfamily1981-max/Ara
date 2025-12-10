#!/usr/bin/env python3
"""
A-KTP Allegory Generation Module (AGM)
=======================================

Generates allegorical narratives that map domain-specific problems
to universal relational structures for zero-shot transfer.

Key insight (Structure-Mapping Theory, Gentner 1983):
Allegories preserve relational structure while abstracting surface details,
enabling knowledge transfer across domains.

Example mappings:
- "Monolith → Microservices" ↔ "Mighty River → River Network"
- "GPU cluster scaling" ↔ "Beehive expansion"
- "Neural network training" ↔ "Garden cultivation"
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import json


class AllegoryArchetype(str, Enum):
    """Universal story archetypes that map to problem patterns."""
    # Transformation archetypes
    RIVER_NETWORK = "river_network"         # Distributed from monolith
    GARDEN_GROWTH = "garden_growth"         # Cultivation/training
    HIVE_EXPANSION = "hive_expansion"       # Scaling swarm/cluster

    # Conflict archetypes
    BRIDGE_BUILDING = "bridge_building"     # Integration challenges
    STORM_WEATHERING = "storm_weathering"   # Resilience/recovery

    # Discovery archetypes
    TREASURE_MAP = "treasure_map"           # Search/optimization
    CONSTELLATION = "constellation"          # Pattern discovery

    # Balance archetypes
    ECOSYSTEM = "ecosystem"                 # Multi-objective balance
    SEASONS = "seasons"                     # Cyclic optimization


@dataclass
class StructuralMapping:
    """Maps domain concepts to allegory elements."""
    domain_entity: str           # e.g., "microservice"
    allegory_entity: str         # e.g., "tributary"
    relation_type: str           # e.g., "transforms_into"
    constraints: List[str] = field(default_factory=list)
    # e.g., ["flow_must_continue", "no_floods"]


@dataclass
class Allegory:
    """A complete allegorical narrative for knowledge transfer."""
    allegory_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = ""
    archetype: AllegoryArchetype = AllegoryArchetype.RIVER_NETWORK

    # The narrative
    narrative: str = ""
    moral: str = ""              # The transferable insight

    # Structural mappings
    mappings: List[StructuralMapping] = field(default_factory=list)

    # Source problem
    source_domain: str = ""
    source_problem: str = ""

    # Metadata
    created_at: float = field(default_factory=time.time)
    transfer_confidence: float = 0.5       # How well it transfers

    # Ethical flags
    is_hypothetical: bool = False          # Unverified domain
    bias_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "allegory_id": self.allegory_id,
            "title": self.title,
            "archetype": self.archetype.value,
            "narrative": self.narrative,
            "moral": self.moral,
            "mappings": [
                {
                    "domain": m.domain_entity,
                    "allegory": m.allegory_entity,
                    "relation": m.relation_type,
                    "constraints": m.constraints,
                }
                for m in self.mappings
            ],
            "source": {
                "domain": self.source_domain,
                "problem": self.source_problem,
            },
            "transfer_confidence": self.transfer_confidence,
            "is_hypothetical": self.is_hypothetical,
            "bias_warnings": self.bias_warnings,
        }


# Pre-defined allegory templates for common patterns
ALLEGORY_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "monolith_to_microservices": {
        "archetype": AllegoryArchetype.RIVER_NETWORK,
        "title": "The Mighty River and the River Network",
        "narrative": """
        Once there was a Mighty River that carried all the water of the land.
        It was powerful but rigid—when one part flooded, the whole river suffered.
        The wise engineers saw that many smaller tributaries could carry the same water,
        each adapting to its terrain, each recovering independently from storms.
        The transformation was gradual: first the headwaters split, then the main channel,
        until a resilient network flowed where once a single river dominated.
        """,
        "moral": "Distributed paths are more resilient than single channels.",
        "mappings": [
            ("monolith", "mighty_river", "transforms_into"),
            ("microservice", "tributary", "becomes"),
            ("api_gateway", "confluence", "serves_as"),
            ("database", "underground_aquifer", "persists_as"),
            ("scaling", "rainfall", "triggers"),
            ("failure", "drought", "represents"),
        ],
    },
    "gpu_cluster_scaling": {
        "archetype": AllegoryArchetype.HIVE_EXPANSION,
        "title": "The Growing Hive",
        "narrative": """
        A single bee can gather nectar, but a hive multiplies that power.
        When the colony grows, new workers must learn the dances of their sisters.
        Too many bees in one flower patch creates contention; wise bees spread out.
        The queen coordinates not by command, but by pheromone—a shared state
        that each bee reads and responds to independently.
        """,
        "moral": "Coordination without centralization scales.",
        "mappings": [
            ("gpu", "worker_bee", "acts_as"),
            ("scheduler", "queen_pheromone", "coordinates_via"),
            ("job_queue", "flower_patch", "represents"),
            ("load_balancing", "bee_dance", "achieves"),
            ("thermal_throttling", "hive_overheating", "analogous_to"),
        ],
    },
    "model_training": {
        "archetype": AllegoryArchetype.GARDEN_GROWTH,
        "title": "The Patient Gardener",
        "narrative": """
        A gardener plants seeds in prepared soil. Not all will sprout.
        Too much water drowns the roots; too little starves them.
        The gardener learns which plants thrive together, which compete.
        Pruning removes what is weak to strengthen what remains.
        The harvest comes not from force, but from patient cultivation.
        """,
        "moral": "Growth requires balance, patience, and selective pressure.",
        "mappings": [
            ("model", "garden", "grows_as"),
            ("data", "seeds", "starts_from"),
            ("learning_rate", "watering", "controlled_by"),
            ("regularization", "pruning", "achieved_through"),
            ("overfitting", "overcrowding", "manifests_as"),
            ("convergence", "harvest", "results_in"),
        ],
    },
    "system_integration": {
        "archetype": AllegoryArchetype.BRIDGE_BUILDING,
        "title": "The Bridge Between Kingdoms",
        "narrative": """
        Two kingdoms prospered on opposite shores, each with unique resources.
        Building a bridge was simple in concept, complex in execution:
        the foundations must anchor in both soils, the span must weather storms,
        the traffic must flow both ways without collision.
        The builders learned: test the foundations first, then the span,
        then the traffic—never all at once.
        """,
        "moral": "Integration requires solid foundations and incremental trust.",
        "mappings": [
            ("legacy_system", "old_kingdom", "represents"),
            ("new_system", "new_kingdom", "represents"),
            ("api", "bridge", "connects_via"),
            ("data_migration", "trade_routes", "establishes"),
            ("rollback", "retreat_path", "preserves"),
        ],
    },
    "optimization_search": {
        "archetype": AllegoryArchetype.TREASURE_MAP,
        "title": "The Treasure Seekers",
        "narrative": """
        Many seekers set out with the same map, but each reads it differently.
        Some dig where X marks the spot; others follow the riddles.
        The wisest seekers share what they find: "nothing here" is data too.
        In time, they triangulate—the treasure lies not where any map said,
        but where all the clues converge.
        """,
        "moral": "Parallel exploration with communication finds solutions faster.",
        "mappings": [
            ("search_agent", "treasure_seeker", "acts_as"),
            ("objective_function", "treasure", "optimizes_toward"),
            ("exploration", "digging", "performs"),
            ("exploitation", "triangulation", "refines_via"),
            ("local_optima", "false_treasure", "avoids"),
        ],
    },
}


class AllegoryGenerator:
    """
    Generates allegories from problem descriptions.

    Usage:
        agm = AllegoryGenerator()
        allegory = agm.generate(
            problem="How should we scale our GPU cluster?",
            domain="infrastructure"
        )
    """

    def __init__(self, templates: Dict[str, Dict] = None):
        self.templates = templates or ALLEGORY_TEMPLATES
        self.generated: List[Allegory] = []

    def generate(self, problem: str, domain: str = None,
                 hint_archetype: AllegoryArchetype = None) -> Allegory:
        """Generate an allegory for the given problem."""

        # Find best matching template
        template_key, template = self._match_template(problem, domain, hint_archetype)

        if template:
            allegory = self._from_template(template_key, template, problem, domain)
        else:
            allegory = self._generate_novel(problem, domain, hint_archetype)

        # Check for hypothetical/unverified domains
        allegory.is_hypothetical = self._check_hypothetical(domain, problem)
        if allegory.is_hypothetical:
            allegory.bias_warnings.append("HYPOTHETICAL - Domain contains unsolved problems")

        self.generated.append(allegory)
        return allegory

    def _match_template(self, problem: str, domain: str,
                        hint: AllegoryArchetype) -> Tuple[str, Optional[Dict]]:
        """Find best matching template for problem."""
        problem_lower = problem.lower()

        # Keyword matching
        if "microservice" in problem_lower or "monolith" in problem_lower:
            return "monolith_to_microservices", self.templates.get("monolith_to_microservices")

        if "gpu" in problem_lower or "cluster" in problem_lower or "scale" in problem_lower:
            return "gpu_cluster_scaling", self.templates.get("gpu_cluster_scaling")

        if "train" in problem_lower or "model" in problem_lower or "learning" in problem_lower:
            return "model_training", self.templates.get("model_training")

        if "integrat" in problem_lower or "legacy" in problem_lower or "migrat" in problem_lower:
            return "system_integration", self.templates.get("system_integration")

        if "optim" in problem_lower or "search" in problem_lower or "find" in problem_lower:
            return "optimization_search", self.templates.get("optimization_search")

        # Use hint if provided
        if hint:
            for key, tmpl in self.templates.items():
                if tmpl.get("archetype") == hint:
                    return key, tmpl

        return "", None

    def _from_template(self, key: str, template: Dict,
                       problem: str, domain: str) -> Allegory:
        """Create allegory from template."""
        mappings = [
            StructuralMapping(
                domain_entity=m[0],
                allegory_entity=m[1],
                relation_type=m[2],
                constraints=[],
            )
            for m in template.get("mappings", [])
        ]

        return Allegory(
            title=template.get("title", key),
            archetype=template.get("archetype", AllegoryArchetype.RIVER_NETWORK),
            narrative=template.get("narrative", "").strip(),
            moral=template.get("moral", ""),
            mappings=mappings,
            source_domain=domain or "general",
            source_problem=problem,
            transfer_confidence=0.75,  # Templates are pre-validated
        )

    def _generate_novel(self, problem: str, domain: str,
                        hint: AllegoryArchetype) -> Allegory:
        """Generate a novel allegory (placeholder for LLM integration)."""
        archetype = hint or AllegoryArchetype.ECOSYSTEM

        return Allegory(
            title=f"The {domain or 'Unknown'} Challenge",
            archetype=archetype,
            narrative=f"A novel situation arose: {problem}. "
                      f"The wise ones gathered to find patterns in the chaos...",
            moral="Novel problems require synthesized wisdom.",
            source_domain=domain or "unknown",
            source_problem=problem,
            transfer_confidence=0.3,  # Lower confidence for novel
        )

    def _check_hypothetical(self, domain: str, problem: str) -> bool:
        """Check if domain contains unsolved/hypothetical elements."""
        hypothetical_keywords = [
            "riemann", "p vs np", "collatz", "hodge",
            "unproven", "conjecture", "unsolved",
        ]
        text = f"{domain or ''} {problem}".lower()
        return any(kw in text for kw in hypothetical_keywords)

    def get_allegory_for_transfer(self, source_allegory: Allegory,
                                   target_domain: str) -> Dict[str, Any]:
        """
        Prepare allegory for transfer to a new domain.

        Returns mapping hints for the target domain.
        """
        return {
            "source_allegory": source_allegory.title,
            "target_domain": target_domain,
            "transferable_moral": source_allegory.moral,
            "mapping_hints": [
                {
                    "allegory_element": m.allegory_entity,
                    "original_domain": m.domain_entity,
                    "suggested_target": f"{target_domain}_{m.allegory_entity}",
                }
                for m in source_allegory.mappings
            ],
            "confidence": source_allegory.transfer_confidence * 0.8,  # Decay for transfer
        }


# Convenience functions
def generate_allegory(problem: str, domain: str = None) -> Allegory:
    """Quick allegory generation."""
    agm = AllegoryGenerator()
    return agm.generate(problem, domain)


def get_template_allegory(template_key: str) -> Optional[Allegory]:
    """Get a pre-defined allegory template."""
    if template_key not in ALLEGORY_TEMPLATES:
        return None

    template = ALLEGORY_TEMPLATES[template_key]
    agm = AllegoryGenerator()
    return agm._from_template(template_key, template, "", "template")
