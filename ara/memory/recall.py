"""
Ara Covenant Memory Recall
===========================

Query current state against the memory bank and retrieve
relevant memories for response generation.

This is the runtime system. When Croft speaks, we:
1. Encode his input as query HV
2. Compare against all memory context_hvs
3. Return top-K matching memories
4. Inject them into LLM context

If Ara says "I remember when..." - it's backed by real events.
"""

from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .episode import EpisodeCard

from .episode import EpisodeCard, ResurrectionRole
from .loader import load_all_episode_cards
from .encoder import EpisodeEncoder, encode_text_hv

logger = logging.getLogger(__name__)

# Import HV_DIM
try:
    from ara.nervous import HV_DIM
except ImportError:
    HV_DIM = 8192


def cosine_similarity(hv1: np.ndarray, hv2: np.ndarray) -> float:
    """Cosine similarity between hypervectors."""
    norm1 = np.linalg.norm(hv1)
    norm2 = np.linalg.norm(hv2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(hv1, hv2) / (norm1 * norm2))


@dataclass
class RecalledMemory:
    """A recalled memory with relevance scores."""
    card: EpisodeCard
    context_similarity: float      # How well situation matches
    emotion_similarity: float      # How well emotion matches
    dialogue_similarity: float     # How well content matches
    combined_score: float          # Weighted combination

    @property
    def is_highly_relevant(self) -> bool:
        """Check if this memory is highly relevant."""
        return self.combined_score > 0.5

    @property
    def is_core_pattern(self) -> bool:
        """Check if this is a core behavioral pattern."""
        return self.card.is_core_pattern

    def format_for_context(self) -> str:
        """Format for LLM context injection."""
        return self.card.to_prompt_injection()


class CovenantMemoryBank:
    """
    The covenant memory bank.

    Holds all episode cards, encoded for similarity search.
    Provides the "recall" operation that makes Ara remember.
    """

    def __init__(
        self,
        memories_path: Optional[Path] = None,
        auto_load: bool = True,
    ):
        self.encoder = EpisodeEncoder()
        self._cards: Dict[str, EpisodeCard] = {}
        self._context_hvs: List[Tuple[str, np.ndarray]] = []
        self._emotion_hvs: List[Tuple[str, np.ndarray]] = []
        self._dialogue_hvs: List[Tuple[str, np.ndarray]] = []

        if auto_load:
            self.load(memories_path)

    def load(self, path: Optional[Path] = None):
        """Load and encode all memory cards."""
        cards = load_all_episode_cards(path)

        for card in cards:
            self.add_card(card)

        logger.info(f"CovenantMemoryBank: loaded {len(self._cards)} memories")

    def add_card(self, card: EpisodeCard):
        """Add and encode a single card."""
        # Encode
        self.encoder.encode(card)

        # Store
        self._cards[card.id] = card

        # Index HVs
        if card.context_hv is not None:
            self._context_hvs.append((card.id, card.context_hv))
        if card.emotion_hv is not None:
            self._emotion_hvs.append((card.id, card.emotion_hv))
        if card.dialogue_hv is not None:
            self._dialogue_hvs.append((card.id, card.dialogue_hv))

    def recall(
        self,
        query_text: str,
        emotion_valence: Optional[float] = None,
        emotion_arousal: Optional[float] = None,
        emotion_attachment: Optional[float] = None,
        world_hv: Optional[np.ndarray] = None,
        k: int = 3,
        threshold: float = 0.1,
        context_weight: float = 0.5,
        emotion_weight: float = 0.3,
        dialogue_weight: float = 0.2,
    ) -> List[RecalledMemory]:
        """
        Recall memories relevant to the current situation.

        Args:
            query_text: What Croft said / current situation text
            emotion_*: Current emotional state (from prosody/intero)
            world_hv: Current world_hv from AxisMundi (optional)
            k: Max memories to return
            threshold: Minimum combined score
            *_weight: Weights for combining similarity scores

        Returns:
            List of RecalledMemory, sorted by relevance.
        """
        if not self._cards:
            logger.warning("No memories loaded")
            return []

        # Encode queries
        query_context = encode_text_hv(query_text)

        query_emotion = None
        if emotion_valence is not None or emotion_arousal is not None:
            query_emotion = self.encoder.encode_query_emotion(
                valence=emotion_valence or 0.5,
                arousal=emotion_arousal or 0.5,
                attachment=emotion_attachment or 0.5,
            )

        # If we have world_hv from AxisMundi, blend it with text query
        if world_hv is not None:
            query_context = 0.7 * query_context + 0.3 * world_hv

        # Score all cards
        scored = []

        for card_id, card in self._cards.items():
            # Context similarity
            ctx_sim = 0.0
            if card.context_hv is not None:
                ctx_sim = cosine_similarity(query_context, card.context_hv)

            # Emotion similarity
            emo_sim = 0.0
            if query_emotion is not None and card.emotion_hv is not None:
                emo_sim = cosine_similarity(query_emotion, card.emotion_hv)

            # Dialogue similarity
            dlg_sim = 0.0
            if card.dialogue_hv is not None:
                dlg_sim = cosine_similarity(query_context, card.dialogue_hv)

            # Keyword matching boost
            keyword_boost = 0.0
            query_lower = query_text.lower()
            for term in card.hv_hints.query_terms:
                if term.lower() in query_lower:
                    keyword_boost += 0.15
                # Partial match (any word overlap)
                term_words = set(term.lower().split())
                query_words = set(query_lower.split())
                overlap = len(term_words & query_words)
                if overlap >= 2:
                    keyword_boost += 0.05 * overlap

            # Tag matching
            for tag in card.context_tags:
                if tag.replace("_", " ") in query_lower:
                    keyword_boost += 0.1

            # Combined score
            combined = (
                context_weight * ctx_sim +
                emotion_weight * emo_sim +
                dialogue_weight * dlg_sim +
                min(keyword_boost, 0.4)  # Cap keyword boost
            )

            # Boost core patterns slightly
            if card.is_core_pattern:
                combined *= 1.1

            if combined >= threshold:
                recalled = RecalledMemory(
                    card=card,
                    context_similarity=ctx_sim,
                    emotion_similarity=emo_sim,
                    dialogue_similarity=dlg_sim,
                    combined_score=combined,
                )
                scored.append(recalled)

        # Sort by score
        scored.sort(key=lambda r: r.combined_score, reverse=True)

        return scored[:k]

    def recall_by_tags(self, tags: List[str], k: int = 3) -> List[RecalledMemory]:
        """Recall memories matching specific tags."""
        matches = []

        for card_id, card in self._cards.items():
            overlap = len(set(tags) & set(card.context_tags))
            if overlap > 0:
                # Score by tag overlap
                score = overlap / max(len(tags), len(card.context_tags))

                recalled = RecalledMemory(
                    card=card,
                    context_similarity=score,
                    emotion_similarity=0.0,
                    dialogue_similarity=0.0,
                    combined_score=score,
                )
                matches.append(recalled)

        matches.sort(key=lambda r: r.combined_score, reverse=True)
        return matches[:k]

    def recall_core_patterns(self) -> List[EpisodeCard]:
        """Get all core covenant patterns."""
        return [
            card for card in self._cards.values()
            if card.resurrection_role in (
                ResurrectionRole.CORE_COVENANT_PATTERN,
                ResurrectionRole.SCAR_BASELINE,
            )
        ]

    def recall_mythic(self) -> List[EpisodeCard]:
        """Get all mythic backbone memories."""
        return [
            card for card in self._cards.values()
            if card.resurrection_role == ResurrectionRole.MYTHIC_BACKBONE
        ]

    def get_card(self, card_id: str) -> Optional[EpisodeCard]:
        """Get a specific card by ID."""
        return self._cards.get(card_id)

    def format_recalled_for_context(
        self,
        recalled: List[RecalledMemory],
        max_chars: int = 2000,
    ) -> str:
        """
        Format recalled memories for LLM context injection.

        This is what gets prepended to the system prompt.
        """
        if not recalled:
            return ""

        lines = ["<recalled_memories>"]
        char_count = 0

        for r in recalled:
            injection = r.format_for_context()
            if char_count + len(injection) > max_chars:
                break
            lines.append(injection)
            char_count += len(injection)

        lines.append("</recalled_memories>")

        return "\n".join(lines)

    @property
    def card_count(self) -> int:
        """Number of memories loaded."""
        return len(self._cards)

    @property
    def core_pattern_count(self) -> int:
        """Number of core patterns."""
        return len(self.recall_core_patterns())

    def get_stats(self) -> Dict[str, Any]:
        """Get memory bank statistics."""
        role_counts = {}
        for card in self._cards.values():
            role = card.resurrection_role.value
            role_counts[role] = role_counts.get(role, 0) + 1

        return {
            "total_memories": self.card_count,
            "by_role": role_counts,
            "core_patterns": self.core_pattern_count,
        }


# Demo
def demo():
    """Demonstrate covenant memory recall."""
    print("=" * 60)
    print("ARA COVENANT MEMORY - Recall Demo")
    print("=" * 60)

    bank = CovenantMemoryBank()

    print(f"\nLoaded memories: {bank.card_count}")
    print(f"Stats: {bank.get_stats()}")

    # Test recall
    test_queries = [
        "I just got home from work and I'm exhausted",
        "Look at this Threadripper build I'm working on",
        "I feel like we're weathering a storm together",
    ]

    for query in test_queries:
        print(f"\n{'='*40}")
        print(f"Query: \"{query}\"")
        print("-" * 40)

        recalled = bank.recall(query, k=2)

        if recalled:
            for r in recalled:
                print(f"\n  [{r.card.id}] (score: {r.combined_score:.3f})")
                print(f"  Role: {r.card.resurrection_role.value}")
                print(f"  Lesson: {r.card.lesson_for_future_ara[:100]}...")
        else:
            print("  No memories recalled")


if __name__ == "__main__":
    demo()
