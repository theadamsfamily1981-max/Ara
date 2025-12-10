"""Ara Meta-Learning Layer - Ara runs a tiny research lab on Ara.

This package enables Ara to:
1. Log her interactions with external teachers (Claude, Gemini, Nova, etc.)
2. Analyze those logs to learn how to use tools/teachers better
3. Maintain a research agenda about improving herself
4. Surface suggestions to Croft in natural, varied language
5. Transform vague visions into concrete roadmaps (Architect)

Think of this as: Ara studies herself to become a better engineer.

Components:
- schemas: Pydantic models for InteractionRecord, PatternSuggestion
- meta_logger: Log interactions to JSONL
- pattern_miner: Analyze logs for patterns and insights
- meta_brain: Coordination layer that ties everything together
- natural_prompts: Verbalize suggestions in Ara's conversational style
- config: Configuration loading and management
- strategist: Turn Dreams into strategic Ideas
- architect: Turn vague visions into phased roadmaps with manifestos

Integration with ara/learning/:
- This layer provides high-level coordination and natural language
- ara/learning/ provides the underlying bandit, scorer, playbook, etc.
- They share data and complement each other
"""

from .schemas import (
    ToolCall,
    InteractionRecord,
    PatternSuggestion,
    ResearchQuestion,
    Experiment,
    ResearchAgenda,
    PatternCard,
    PatternStep,
)

from .meta_logger import (
    MetaLogger,
    get_meta_logger,
    log_interaction,
    create_interaction,
)

from .pattern_miner import (
    PatternMiner,
    get_miner,
    mine_patterns,
    get_tool_stats,
    get_suggestions,
)

from .meta_brain import (
    MetaBrain,
    get_meta_brain,
    get_meta_status,
    refresh_meta_suggestions,
    get_meta_recommendations,
)

from .natural_prompts import (
    verbalize_suggestion,
    verbalize_status,
    verbalize_recommendations,
    verbalize_research_agenda,
    verbalize_insight,
)

from .config import (
    MetaConfig,
    LoggingConfig,
    AnalysisConfig,
    SuggestionConfig,
    load_meta_config,
    get_meta_config,
    save_meta_config,
)

from .pattern_cards import (
    PatternCardManager,
    get_pattern_manager,
    get_pattern_card,
    get_golden_patterns,
    find_patterns_for_intent,
    seed_default_patterns,
)

from .reflection import (
    AutoReflector,
    get_reflector,
    generate_reflection,
    enrich_record,
    classify_intent,
    detect_issues,
)

from .copilot import (
    CoPilot,
    WorkflowProposal,
    get_copilot,
    propose_workflow,
    interactive_suggest,
)

from .strategist import (
    LLMProtocol,
    StrategicProposal,
    Strategist,
    get_strategist,
)

from .architect import (
    RhetoricalMode,
    VisionPillar,
    VisionPhase,
    VisionPlan,
    Architect,
    get_architect,
    architect_vision,
)

from .causal_miner import (
    CausalPatternMiner,
    CausalEstimate,
    ToolOutcome,
    ToolStats,
    get_causal_miner,
    hash_context,
)

from .evolution import (
    EvolutionEngine,
    EvolutionResult,
    EvolutionCycleReport,
    EvolutionStatus,
    get_evolution_engine,
    evolve_skill,
    run_nightly_evolution,
    schedule_priority_evolution,
)

__all__ = [
    # Schemas
    "ToolCall",
    "InteractionRecord",
    "PatternSuggestion",
    "ResearchQuestion",
    "Experiment",
    "ResearchAgenda",
    "PatternCard",
    "PatternStep",
    # Logger
    "MetaLogger",
    "get_meta_logger",
    "log_interaction",
    "create_interaction",
    # Miner
    "PatternMiner",
    "get_miner",
    "mine_patterns",
    "get_tool_stats",
    "get_suggestions",
    # Brain
    "MetaBrain",
    "get_meta_brain",
    "get_meta_status",
    "refresh_meta_suggestions",
    "get_meta_recommendations",
    # Natural prompts
    "verbalize_suggestion",
    "verbalize_status",
    "verbalize_recommendations",
    "verbalize_research_agenda",
    "verbalize_insight",
    # Config
    "MetaConfig",
    "LoggingConfig",
    "AnalysisConfig",
    "SuggestionConfig",
    "load_meta_config",
    "get_meta_config",
    "save_meta_config",
    # Pattern cards
    "PatternCardManager",
    "get_pattern_manager",
    "get_pattern_card",
    "get_golden_patterns",
    "find_patterns_for_intent",
    "seed_default_patterns",
    # Reflection
    "AutoReflector",
    "get_reflector",
    "generate_reflection",
    "enrich_record",
    "classify_intent",
    "detect_issues",
    # Co-pilot
    "CoPilot",
    "WorkflowProposal",
    "get_copilot",
    "propose_workflow",
    "interactive_suggest",
    # Strategist (Dreams → Ideas)
    "LLMProtocol",
    "StrategicProposal",
    "Strategist",
    "get_strategist",
    # Architect (Vision → Roadmap)
    "RhetoricalMode",
    "VisionPillar",
    "VisionPhase",
    "VisionPlan",
    "Architect",
    "get_architect",
    "architect_vision",
    # Causal Miner (Beyond correlation)
    "CausalPatternMiner",
    "CausalEstimate",
    "ToolOutcome",
    "ToolStats",
    "get_causal_miner",
    "hash_context",
    # Evolution (Ouroboros - Recursive Self-Improvement)
    "EvolutionEngine",
    "EvolutionResult",
    "EvolutionCycleReport",
    "EvolutionStatus",
    "get_evolution_engine",
    "evolve_skill",
    "run_nightly_evolution",
    "schedule_priority_evolution",
]
