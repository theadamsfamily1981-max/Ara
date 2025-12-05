"""Ara Self-Learning Layer - The student analyzing the teachers.

This package enables Ara to learn from her interactions with the council:
1. Log every interaction with structured metrics
2. Score outcomes (what worked, what failed)
3. Adapt tool selection using bandit algorithms
4. Build a prompt playbook of what works best
5. Distill knowledge into reusable skills

The key insight: We can't retrain the LLM, but we can:
- Profile the teachers (which tool works best for what)
- Learn prompt patterns (how to talk to each collaborator)
- Skip steps (internalize workflows that consistently work)
- Bootstrap skills (reuse past solutions, only ask for novel pieces)

Over time, Ara's "needing a second opinion" curve bends downward.
"""

from .logger import (
    InteractionLog,
    InteractionLogger,
    ToolCall,
    Outcome,
    log_interaction,
)

from .scorer import (
    RewardSignal,
    RewardScorer,
    compute_reward,
    normalize_perf_gain,
)

from .bandit import (
    ToolStats,
    ToolBandit,
    choose_tool,
    update_tool_stats,
)

from .playbook import (
    PromptExemplar,
    PromptPlaybook,
    retrieve_best_prompt,
    save_exemplar,
    seed_default_exemplars,
)

from .distiller import (
    Skill,
    SkillLibrary,
    distill_from_interaction,
    search_skills,
)

from .analyzer import (
    PatternMiner,
    WorkflowPattern,
    mine_patterns,
    get_golden_paths,
)

__all__ = [
    # Logger
    "InteractionLog",
    "InteractionLogger",
    "ToolCall",
    "Outcome",
    "log_interaction",
    # Scorer
    "RewardSignal",
    "RewardScorer",
    "compute_reward",
    "normalize_perf_gain",
    # Bandit
    "ToolStats",
    "ToolBandit",
    "choose_tool",
    "update_tool_stats",
    # Playbook
    "PromptExemplar",
    "PromptPlaybook",
    "retrieve_best_prompt",
    "save_exemplar",
    "seed_default_exemplars",
    # Distiller
    "Skill",
    "SkillLibrary",
    "distill_from_interaction",
    "search_skills",
    # Analyzer
    "PatternMiner",
    "WorkflowPattern",
    "mine_patterns",
    "get_golden_paths",
]
