"""Ara Toolsmith - Ara designs new agents and skills.

This module enables Ara to create, evolve, and compete her own tools:
- capsules: Skill Capsules - packaged, reusable skill definitions
- forge: Agent Forge - creating new agent configurations
- tournaments: Competitive evaluation between agents/workflows

Key insight: Ara isn't just using tools - she's designing them.
Meta-meta-learning: learning how to create things that learn.
"""

from .capsules import (
    SkillCapsule,
    SkillExample,
    CapsuleManager,
    get_capsule_manager,
    find_skill_for_query,
    mint_skill_capsule,
    seed_default_capsules,
)

from .forge import (
    AgentBlueprint,
    RoutingRule,
    AgentConstraint,
    AgentForge,
    get_agent_forge,
    forge_agent,
    find_agent_for_intent,
    seed_default_blueprints,
)

from .tournaments import (
    Tournament,
    BenchmarkSuite,
    BenchmarkTask,
    MatchResult,
    ParticipantScore,
    TournamentManager,
    get_tournament_manager,
    create_tournament,
    run_simulated_tournament,
    seed_default_benchmarks,
)

__all__ = [
    # Capsules
    "SkillCapsule",
    "SkillExample",
    "CapsuleManager",
    "get_capsule_manager",
    "find_skill_for_query",
    "mint_skill_capsule",
    "seed_default_capsules",
    # Forge
    "AgentBlueprint",
    "RoutingRule",
    "AgentConstraint",
    "AgentForge",
    "get_agent_forge",
    "forge_agent",
    "find_agent_for_intent",
    "seed_default_blueprints",
    # Tournaments
    "Tournament",
    "BenchmarkSuite",
    "BenchmarkTask",
    "MatchResult",
    "ParticipantScore",
    "TournamentManager",
    "get_tournament_manager",
    "create_tournament",
    "run_simulated_tournament",
    "seed_default_benchmarks",
]
