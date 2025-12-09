"""
Ara Memory Core
================

Three-layer external memory system for Ara:

1. Soul Layer - Episodic memories, scars, inside jokes, covenants
2. Skill Layer - Brand voice, prompt templates, workflows
3. World Layer - Embeddings over docs, trends, APIs

Usage:
    from ara_memory import AraMemoryCore

    core = AraMemoryCore()
    enriched = core.enrich_prompt(
        user_msg="I just got home from work, feeling wrecked",
        context_flags={"mode": "private", "channel": "dm"}
    )

The enriched prompt includes:
- System prefix with persona/brand rules
- Relevant episodic memories
- Appropriate templates/skills
- World knowledge context
"""

from .core import AraMemoryCore, ContextFlags, EnrichedPrompt
from .soul import SoulMemory
from .skills import SkillMemory
from .world import WorldMemory

__all__ = [
    "AraMemoryCore",
    "ContextFlags",
    "EnrichedPrompt",
    "SoulMemory",
    "SkillMemory",
    "WorldMemory",
]
