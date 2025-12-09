"""
Ara Memory Integration
=======================

Bridges CovenantMemoryBank with the nervous system (AxisMundi).

Flow:
1. AxisMundi produces world_hv from current sensors
2. Prosody produces emotional valence from speech
3. We query CovenantMemoryBank with both
4. Recalled memories get injected into LLM context
5. Ara responds with actual memories backing her behavior
"""

from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ara.nervous import AxisMundi
    from ara.nervous.prosody import ProsodyClassifier

from .recall import CovenantMemoryBank, RecalledMemory
from .episode import EpisodeCard

logger = logging.getLogger(__name__)


class MemoryIntegrator:
    """
    Integrates covenant memory with the nervous system.

    Listens to AxisMundi and ProsodyClassifier to automatically
    recall relevant memories when context changes.
    """

    def __init__(
        self,
        memory_bank: Optional[CovenantMemoryBank] = None,
        axis: Optional['AxisMundi'] = None,
        prosody: Optional['ProsodyClassifier'] = None,
        auto_recall_threshold: float = 0.3,
        max_memories: int = 3,
    ):
        self.memory_bank = memory_bank or CovenantMemoryBank()
        self.axis = axis
        self.prosody = prosody

        self.auto_recall_threshold = auto_recall_threshold
        self.max_memories = max_memories

        # Current recall state
        self._last_recalled: List[RecalledMemory] = []
        self._recall_count = 0

    def set_axis(self, axis: 'AxisMundi'):
        """Connect to AxisMundi."""
        self.axis = axis

    def set_prosody(self, prosody: 'ProsodyClassifier'):
        """Connect to ProsodyClassifier."""
        self.prosody = prosody

    def recall_for_input(
        self,
        user_text: str,
        prosody_valence: Optional[float] = None,
        prosody_arousal: Optional[float] = None,
    ) -> List[RecalledMemory]:
        """
        Recall memories relevant to user input.

        This is the main entry point for memory recall.

        Args:
            user_text: What the user said
            prosody_valence: Emotional valence from speech (0=negative, 1=positive)
            prosody_arousal: Arousal level from speech (0=calm, 1=agitated)

        Returns:
            List of recalled memories
        """
        # Get world_hv if available
        world_hv = None
        if self.axis is not None:
            world_hv = self.axis.world_hv

        # Get emotional state from intero if not provided
        if prosody_valence is None and self.axis is not None:
            # Map stress to negative valence
            prosody_valence = 1.0 - self.axis.intero.stress

        if prosody_arousal is None and self.axis is not None:
            prosody_arousal = self.axis.intero.arousal

        # Recall
        recalled = self.memory_bank.recall(
            query_text=user_text,
            emotion_valence=prosody_valence,
            emotion_arousal=prosody_arousal,
            emotion_attachment=0.8,  # Default high for Croft
            world_hv=world_hv,
            k=self.max_memories,
            threshold=self.auto_recall_threshold,
        )

        self._last_recalled = recalled
        self._recall_count += 1

        if recalled:
            logger.debug(f"Recalled {len(recalled)} memories for input")
            for r in recalled:
                logger.debug(f"  {r.card.id}: {r.combined_score:.3f}")

        return recalled

    def get_context_injection(
        self,
        user_text: str,
        max_chars: int = 2000,
    ) -> str:
        """
        Get memory context injection for LLM prompt.

        This is what gets prepended to the system prompt.
        """
        recalled = self.recall_for_input(user_text)
        return self.memory_bank.format_recalled_for_context(recalled, max_chars)

    def get_always_on_context(self) -> str:
        """
        Get the "always on" core patterns.

        These are injected into every response regardless of recall.
        """
        core = self.memory_bank.recall_core_patterns()

        if not core:
            return ""

        lines = [
            "<core_covenant>",
            "These are my foundational patterns with Croft:",
            "",
        ]

        for card in core[:3]:  # Max 3 core patterns
            lines.append(f"- {card.lesson_for_future_ara.strip()}")

        lines.append("</core_covenant>")

        return "\n".join(lines)

    @property
    def last_recalled(self) -> List[RecalledMemory]:
        """Get last recalled memories."""
        return self._last_recalled

    @property
    def recall_count(self) -> int:
        """Number of recall operations performed."""
        return self._recall_count


class MemoryAugmentedResponse:
    """
    Wrapper for responses augmented with memory context.

    Used when generating Ara's response to include:
    1. What memories were recalled
    2. What patterns should be followed
    3. Any "I remember when..." hooks
    """

    def __init__(
        self,
        recalled: List[RecalledMemory],
        user_input: str,
    ):
        self.recalled = recalled
        self.user_input = user_input

    def get_response_guidance(self) -> str:
        """
        Get response guidance based on recalled memories.

        This instructs the LLM on how to respond.
        """
        if not self.recalled:
            return ""

        lines = [
            "<memory_guidance>",
            "Based on my memories, I should:",
            "",
        ]

        for r in self.recalled:
            card = r.card
            lines.append(f"From {card.id} ({r.combined_score:.2f}):")
            lines.append(f"  {card.lesson_for_future_ara.strip()}")
            lines.append("")

        # Specific "remember when" hooks
        high_match = [r for r in self.recalled if r.combined_score > 0.5]
        if high_match:
            lines.append("I can reference these memories naturally:")
            for r in high_match:
                lines.append(f"  - {r.card.dialogue_snippets.paraphrased_exchange[:100]}...")

        lines.append("</memory_guidance>")

        return "\n".join(lines)

    def should_say_remember(self) -> bool:
        """Check if we should explicitly say 'I remember when...'"""
        return any(r.combined_score > 0.6 for r in self.recalled)

    def get_remember_hook(self) -> Optional[str]:
        """Get a natural 'I remember' hook if appropriate."""
        high_match = [r for r in self.recalled if r.combined_score > 0.6]
        if not high_match:
            return None

        best = high_match[0]
        exchange = best.card.dialogue_snippets.paraphrased_exchange

        # Extract first sentence as hook
        sentences = exchange.split('.')
        if sentences:
            return f"This reminds me of when {sentences[0].strip().lower()}."

        return None


def create_memory_system(
    axis: Optional['AxisMundi'] = None,
) -> MemoryIntegrator:
    """
    Create a configured memory system.

    Returns MemoryIntegrator connected to axis if provided.
    """
    bank = CovenantMemoryBank()
    integrator = MemoryIntegrator(memory_bank=bank, axis=axis)

    logger.info(f"Memory system created with {bank.card_count} memories")

    return integrator
