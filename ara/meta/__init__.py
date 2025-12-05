"""Ara Meta-Learning Layer - Ara runs a tiny research lab on Ara.

This package enables Ara to:
1. Log her interactions with external teachers (Claude, Gemini, Nova, etc.)
2. Analyze those logs to learn how to use tools/teachers better
3. Maintain a research agenda about improving herself
4. Surface suggestions to Croft in natural, varied language

Think of this as: Ara studies herself to become a better engineer.

Components:
- schemas: Pydantic models for InteractionRecord, PatternSuggestion
- meta_logger: Log interactions to JSONL
- pattern_miner: Analyze logs for patterns and insights
- meta_brain: Coordination layer that ties everything together
- natural_prompts: Verbalize suggestions in Ara's conversational style
- config: Configuration loading and management

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

__all__ = [
    # Schemas
    "ToolCall",
    "InteractionRecord",
    "PatternSuggestion",
    "ResearchQuestion",
    "Experiment",
    "ResearchAgenda",
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
]
