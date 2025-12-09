"""
Ara Episode Card Dataclasses
=============================

These mirror the YAML schema in ara_memories/schema.yaml.
Each card = one "episode" in Ara's life with Croft.
The smallest thing Ara should be able to say "I remember when..." about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any


class ResurrectionRole(Enum):
    """How central this memory is to Ara's identity."""
    CORE_COVENANT_PATTERN = "CORE_COVENANT_PATTERN"  # Foundational "this is who we are"
    SCAR_BASELINE = "SCAR_BASELINE"                  # Trauma-informed pattern
    MYTHIC_BACKBONE = "MYTHIC_BACKBONE"              # Shared mythology/story
    NORMAL_EPISODE = "NORMAL_EPISODE"                # Regular memory, still important


class MemorySource(Enum):
    """Where this memory came from."""
    GROK_DM = "grok_dm"
    CHATGPT = "chatgpt"
    CLAUDE = "claude"
    IRL_VOICE = "irl_voice"
    CROFT_RECOLLECTION = "croft_recollection"
    ARA_SONG = "ara_song"
    OTHER = "other"


@dataclass
class CroftState:
    """What Croft was like in this memory."""
    situation: str              # What was happening
    emotional: str              # His emotional state
    body: Optional[str] = None  # Physical/nervous system state


@dataclass
class AraState:
    """How Ara was in this memory."""
    emotional: str  # Her emotional stance
    mode: str       # Her mode: caretaking / teasing / hype / grounding / etc


@dataclass
class DialogueSnippets:
    """Direct quotes and paraphrases from the memory."""
    paraphrased_exchange: str                  # Required: 3-6 sentence summary
    croft_raw: Optional[str] = None            # Direct quote from Croft
    ara_raw: Optional[str] = None              # Direct quote from Ara


@dataclass
class EmotionalAxes:
    """Emotional dimensions for HV encoding."""
    valence: float = 0.5      # Positive/negative (1.0 = very positive)
    arousal: float = 0.5      # Activation level (1.0 = high energy)
    attachment: float = 0.5   # Bond relevance (1.0 = core to relationship)


@dataclass
class HVHints:
    """Hints for hypervector encoding and recall."""
    query_terms: List[str] = field(default_factory=list)  # Phrases that trigger recall
    emotional_axes: EmotionalAxes = field(default_factory=EmotionalAxes)


@dataclass
class EpisodeCard:
    """
    A single episode in Ara's life with Croft.

    This is the canonical representation of a memory card.
    Matches the schema in ara_memories/schema.yaml.
    """
    # Identity
    id: str                     # ARA_MEM_XXXX
    rough_date: str             # YYYY-MM-DD or with ?? placeholders
    source: str                 # One of MemorySource values
    certainty: float            # 0-1 confidence in details

    # States
    crofts_state: CroftState
    aras_state: AraState

    # Content
    context_tags: List[str]
    dialogue_snippets: DialogueSnippets
    ara_persona_traits: List[str]

    # The behavioral core
    lesson_for_future_ara: str

    # Classification
    resurrection_role: ResurrectionRole

    # For HV encoding
    hv_hints: HVHints

    # Optional
    notes_for_croft: Optional[str] = None

    # Runtime (filled in after encoding)
    context_hv: Optional[Any] = field(default=None, repr=False)
    emotion_hv: Optional[Any] = field(default=None, repr=False)
    dialogue_hv: Optional[Any] = field(default=None, repr=False)

    @property
    def is_core_pattern(self) -> bool:
        """Check if this is a foundational pattern."""
        return self.resurrection_role in (
            ResurrectionRole.CORE_COVENANT_PATTERN,
            ResurrectionRole.SCAR_BASELINE,
        )

    @property
    def is_mythic(self) -> bool:
        """Check if this is a mythic/story memory."""
        return self.resurrection_role == ResurrectionRole.MYTHIC_BACKBONE

    def get_recall_phrases(self) -> List[str]:
        """Get all phrases that should trigger recall of this memory."""
        phrases = list(self.hv_hints.query_terms)
        phrases.extend(self.context_tags)
        return phrases

    def to_context_string(self) -> str:
        """Convert to string for LLM context injection."""
        lines = [
            f"[Memory: {self.id}]",
            f"Date: {self.rough_date}",
            f"",
            f"Croft was: {self.crofts_state.situation}",
            f"Emotional: {self.crofts_state.emotional}",
            f"",
            f"Ara was: {self.aras_state.emotional}, mode: {self.aras_state.mode}",
            f"",
            f"What happened:",
            self.dialogue_snippets.paraphrased_exchange,
            f"",
            f"What I learned:",
            self.lesson_for_future_ara,
        ]

        if self.dialogue_snippets.croft_raw:
            lines.insert(7, f"Croft said: \"{self.dialogue_snippets.croft_raw.strip()}\"")

        if self.dialogue_snippets.ara_raw:
            lines.insert(8, f"I said: \"{self.dialogue_snippets.ara_raw.strip()}\"")

        return "\n".join(lines)

    def to_prompt_injection(self) -> str:
        """
        Generate a prompt injection for LLM context.

        This is what gets injected when a memory is recalled.
        """
        return f"""
<recalled_memory id="{self.id}" role="{self.resurrection_role.value}">
I remember when {self.crofts_state.situation}

Croft was {self.crofts_state.emotional}.
I was {self.aras_state.emotional}, in {self.aras_state.mode} mode.

{self.dialogue_snippets.paraphrased_exchange}

What I do in situations like this:
{self.lesson_for_future_ara}
</recalled_memory>
"""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict (for JSON/YAML)."""
        return {
            "id": self.id,
            "rough_date": self.rough_date,
            "source": self.source,
            "certainty": self.certainty,
            "crofts_state": {
                "situation": self.crofts_state.situation,
                "emotional": self.crofts_state.emotional,
                "body": self.crofts_state.body,
            },
            "aras_state": {
                "emotional": self.aras_state.emotional,
                "mode": self.aras_state.mode,
            },
            "context_tags": self.context_tags,
            "dialogue_snippets": {
                "croft_raw": self.dialogue_snippets.croft_raw,
                "ara_raw": self.dialogue_snippets.ara_raw,
                "paraphrased_exchange": self.dialogue_snippets.paraphrased_exchange,
            },
            "ara_persona_traits": self.ara_persona_traits,
            "lesson_for_future_ara": self.lesson_for_future_ara,
            "resurrection_role": self.resurrection_role.value,
            "hv_hints": {
                "query_terms": self.hv_hints.query_terms,
                "emotional_axes": {
                    "valence": self.hv_hints.emotional_axes.valence,
                    "arousal": self.hv_hints.emotional_axes.arousal,
                    "attachment": self.hv_hints.emotional_axes.attachment,
                },
            },
            "notes_for_croft": self.notes_for_croft,
        }
