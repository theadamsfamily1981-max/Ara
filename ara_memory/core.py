"""
Ara Memory Core
================

Unified interface for Ara's three-layer memory system.

Every prompt to Ara goes through this preprocessor, which enriches it with:
- Relevant episodic memories (Soul)
- Brand voice and templates (Skill)
- World knowledge (World)

Ara never talks to anyone "raw" - everything is memory-enriched.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Literal

logger = logging.getLogger(__name__)


# =============================================================================
# Context and Result Types
# =============================================================================

@dataclass
class ContextFlags:
    """Flags that control memory retrieval and prompt enrichment."""

    # Who is this? Private (Croft) or public?
    mode: Literal["private", "public"] = "public"

    # What channel? Affects templates and voice
    channel: str = "generic"  # dm, twitter, email, blog, github, generic

    # Task type for skill selection
    task: str = "conversation"  # conversation, thread, newsletter, code, etc.

    # How many memories to retrieve
    memory_k: int = 3

    # How much world context
    world_k: int = 5

    # Override flags
    skip_soul: bool = False
    skip_skills: bool = False
    skip_world: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "channel": self.channel,
            "task": self.task,
            "memory_k": self.memory_k,
            "world_k": self.world_k,
        }


@dataclass
class EnrichedPrompt:
    """
    Result of prompt enrichment.

    Contains:
    - system: The system prompt with persona/brand rules
    - context_blocks: List of context strings (soul, skills, world)
    - user: The original user message
    - metadata: Debug info about what was retrieved
    """
    system: str
    context_blocks: List[str]
    user: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_messages(self) -> List[Dict[str, str]]:
        """Convert to standard messages format for LLM APIs."""
        messages = [{"role": "system", "content": self.system}]

        # Add context as assistant knowledge
        if self.context_blocks:
            context_combined = "\n\n".join(b for b in self.context_blocks if b)
            if context_combined.strip():
                messages.append({
                    "role": "system",
                    "content": f"[MEMORY CONTEXT]\n{context_combined}"
                })

        messages.append({"role": "user", "content": self.user})
        return messages

    def to_single_prompt(self) -> str:
        """Convert to single string prompt."""
        parts = [self.system]
        for block in self.context_blocks:
            if block:
                parts.append(block)
        parts.append(f"User: {self.user}")
        return "\n\n".join(parts)


# =============================================================================
# Memory Core
# =============================================================================

class AraMemoryCore:
    """
    Unified memory core for Ara.

    Coordinates three memory layers:
    1. Soul - Episodic memories (personal history with Croft)
    2. Skills - Brand voice, templates, workflows
    3. World - Knowledge embeddings, trends, docs

    Usage:
        core = AraMemoryCore()
        enriched = core.enrich_prompt(
            user_msg="Help me write a Twitter thread",
            context_flags=ContextFlags(mode="private", channel="twitter")
        )
    """

    def __init__(
        self,
        soul_path: Optional[str] = None,
        skills_path: Optional[str] = None,
        world_path: Optional[str] = None,
        lazy_load: bool = True,
    ):
        """
        Initialize memory core.

        Args:
            soul_path: Path to soul/episodes directory
            skills_path: Path to skills directory
            world_path: Path to world indexes
            lazy_load: If True, load layers on first use
        """
        self._soul_path = soul_path
        self._skills_path = skills_path
        self._world_path = world_path
        self._lazy_load = lazy_load

        self._soul = None
        self._skills = None
        self._world = None

        if not lazy_load:
            self._load_all()

    def _load_all(self):
        """Load all memory layers."""
        _ = self.soul
        _ = self.skills
        _ = self.world

    @property
    def soul(self):
        """Lazy-load soul memory."""
        if self._soul is None:
            from .soul import SoulMemory
            self._soul = SoulMemory(path=self._soul_path)
        return self._soul

    @property
    def skills(self):
        """Lazy-load skill memory."""
        if self._skills is None:
            from .skills import SkillMemory
            self._skills = SkillMemory(path=self._skills_path)
        return self._skills

    @property
    def world(self):
        """Lazy-load world memory."""
        if self._world is None:
            from .world import WorldMemory
            self._world = WorldMemory(path=self._world_path)
        return self._world

    # =========================================================================
    # Main Interface
    # =========================================================================

    def enrich_prompt(
        self,
        user_msg: str,
        context_flags: Optional[ContextFlags] = None,
    ) -> EnrichedPrompt:
        """
        Enrich a user message with memory context.

        This is the main entry point. Every prompt to Ara should go through here.

        Args:
            user_msg: The raw user message
            context_flags: Flags controlling retrieval

        Returns:
            EnrichedPrompt with system, context, and user message
        """
        flags = context_flags or ContextFlags()

        # Build system prefix from skills/brand
        system_prefix = self.skills.build_system_prefix(flags)

        # Retrieve from each layer
        context_blocks = []
        metadata = {"flags": flags.to_dict()}

        # Soul layer - episodic memories
        if not flags.skip_soul:
            soul_context, soul_meta = self.soul.retrieve_relevant(
                user_msg=user_msg,
                context_flags=flags,
                k=flags.memory_k,
            )
            if soul_context:
                context_blocks.append(soul_context)
            metadata["soul"] = soul_meta

        # Skill layer - templates and hints
        if not flags.skip_skills:
            skill_context, skill_meta = self.skills.select_templates(
                user_msg=user_msg,
                context_flags=flags,
            )
            if skill_context:
                context_blocks.append(skill_context)
            metadata["skills"] = skill_meta

        # World layer - knowledge embeddings
        if not flags.skip_world:
            world_context, world_meta = self.world.retrieve_knowledge(
                user_msg=user_msg,
                context_flags=flags,
                k=flags.world_k,
            )
            if world_context:
                context_blocks.append(world_context)
            metadata["world"] = world_meta

        return EnrichedPrompt(
            system=system_prefix,
            context_blocks=context_blocks,
            user=user_msg,
            metadata=metadata,
        )

    # =========================================================================
    # Diagnostic Methods
    # =========================================================================

    def test_memory(self, test_prompt: str, expected_behavior: str) -> Dict[str, Any]:
        """
        Run a diagnostic test against memory.

        Args:
            test_prompt: The test input
            expected_behavior: What we expect Ara to do

        Returns:
            Dict with test results
        """
        # Private mode test
        private_result = self.enrich_prompt(
            user_msg=test_prompt,
            context_flags=ContextFlags(mode="private"),
        )

        # Public mode test
        public_result = self.enrich_prompt(
            user_msg=test_prompt,
            context_flags=ContextFlags(mode="public"),
        )

        return {
            "test_prompt": test_prompt,
            "expected": expected_behavior,
            "private_context": private_result.context_blocks,
            "public_context": public_result.context_blocks,
            "private_soul_meta": private_result.metadata.get("soul"),
            "public_soul_meta": public_result.metadata.get("soul"),
        }

    def list_episodes(self) -> List[str]:
        """List all loaded episode IDs."""
        return self.soul.list_episodes()

    def get_episode(self, episode_id: str) -> Optional[Dict]:
        """Get a specific episode by ID."""
        return self.soul.get_episode(episode_id)

    # =========================================================================
    # Status
    # =========================================================================

    def status(self) -> Dict[str, Any]:
        """Get status of all memory layers."""
        return {
            "soul": {
                "loaded": self._soul is not None,
                "episode_count": len(self.soul.episodes) if self._soul else 0,
                "indexed": self.soul.is_indexed if self._soul else False,
            },
            "skills": {
                "loaded": self._skills is not None,
                "template_count": self.skills.template_count if self._skills else 0,
                "brand_loaded": self.skills.brand_loaded if self._skills else False,
            },
            "world": {
                "loaded": self._world is not None,
                "index_count": self.world.index_count if self._world else 0,
            },
        }


# =============================================================================
# Module-level convenience
# =============================================================================

_default_core: Optional[AraMemoryCore] = None


def get_memory_core(reload: bool = False) -> AraMemoryCore:
    """Get the default memory core singleton."""
    global _default_core
    if _default_core is None or reload:
        _default_core = AraMemoryCore()
    return _default_core


def enrich_prompt(
    user_msg: str,
    mode: str = "public",
    channel: str = "generic",
    **kwargs,
) -> EnrichedPrompt:
    """
    Convenience function for prompt enrichment.

    Usage:
        from ara_memory import enrich_prompt
        result = enrich_prompt("Hello!", mode="private")
    """
    core = get_memory_core()
    flags = ContextFlags(mode=mode, channel=channel, **kwargs)
    return core.enrich_prompt(user_msg, flags)
