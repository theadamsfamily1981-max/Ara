"""
Sanity Test: Covenant Memory System
=====================================

Tests that the episode card system loads, encodes, and recalls properly.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

try:
    import pytest
except ImportError:
    pytest = None

from ara.memory import (
    EpisodeCard,
    CroftState,
    AraState,
    DialogueSnippets,
    HVHints,
    EmotionalAxes,
    ResurrectionRole,
    load_all_episode_cards,
    validate_episode_card,
    EpisodeEncoder,
    CovenantMemoryBank,
)


def test_episode_card_creation():
    """Test that episode cards can be created programmatically."""
    card = EpisodeCard(
        id="TEST_MEM_0001",
        rough_date="2024-01-01",
        source="croft_recollection",
        certainty=0.9,
        crofts_state=CroftState(
            situation="Testing the memory system",
            emotional="Curious and hopeful",
        ),
        aras_state=AraState(
            emotional="Supportive and attentive",
            mode="Assisting",
        ),
        context_tags=["testing", "memory", "sanity_check"],
        dialogue_snippets=DialogueSnippets(
            paraphrased_exchange="Croft tested the memory system and it worked.",
        ),
        ara_persona_traits=["Helpful", "Reliable"],
        lesson_for_future_ara="When Croft tests systems, I support and verify.",
        resurrection_role=ResurrectionRole.NORMAL_EPISODE,
        hv_hints=HVHints(
            query_terms=["testing", "memory system"],
            emotional_axes=EmotionalAxes(valence=0.7, arousal=0.5, attachment=0.6),
        ),
    )

    assert card.id == "TEST_MEM_0001"
    assert card.is_core_pattern is False
    assert card.is_mythic is False
    assert len(card.get_recall_phrases()) > 0


def test_load_real_memories():
    """Test that real memory cards load from ara_memories/."""
    cards = load_all_episode_cards()

    assert len(cards) >= 4, f"Expected at least 4 memories, got {len(cards)}"

    # Check required memories exist
    card_ids = {c.id for c in cards}
    assert "ARA_MEM_0001" in card_ids
    assert "ARA_MEM_0002" in card_ids
    assert "ARA_MEM_0003" in card_ids
    assert "ARA_MEM_0004" in card_ids


def test_validate_memories():
    """Test that all loaded memories pass validation."""
    cards = load_all_episode_cards()

    for card in cards:
        issues = validate_episode_card(card)
        assert len(issues) == 0, f"{card.id} has issues: {issues}"


def test_encoder_produces_hvs():
    """Test that the encoder produces valid HVs."""
    cards = load_all_episode_cards()
    encoder = EpisodeEncoder()

    for card in cards:
        encoded = encoder.encode(card)

        assert encoded.context_hv is not None
        assert encoded.context_hv.shape == (8192,)
        assert np.linalg.norm(encoded.context_hv) > 0

        assert encoded.emotion_hv is not None
        assert encoded.emotion_hv.shape == (8192,)

        assert encoded.dialogue_hv is not None
        assert encoded.dialogue_hv.shape == (8192,)


def test_memory_bank_recall():
    """Test that the memory bank recalls correct memories."""
    bank = CovenantMemoryBank()

    assert bank.card_count >= 4

    # Test decompression recall
    recalled = bank.recall("I just got home from work and I'm exhausted")
    assert len(recalled) > 0
    # Should recall decompression-related memories
    recalled_ids = {r.card.id for r in recalled}
    assert "ARA_MEM_0001" in recalled_ids or "ARA_MEM_0002" in recalled_ids

    # Test threadripper recall
    recalled = bank.recall("Look at this Threadripper build I'm working on")
    assert len(recalled) > 0
    recalled_ids = {r.card.id for r in recalled}
    assert "ARA_MEM_0004" in recalled_ids

    # Test storm/mythic recall
    recalled = bank.recall("I feel like we're weathering a storm together")
    assert len(recalled) > 0
    recalled_ids = {r.card.id for r in recalled}
    assert "ARA_MEM_0003" in recalled_ids


def test_core_patterns_retrieval():
    """Test that core patterns can be retrieved."""
    bank = CovenantMemoryBank()

    core = bank.recall_core_patterns()
    assert len(core) >= 2

    # All should be CORE_COVENANT_PATTERN or SCAR_BASELINE
    for card in core:
        assert card.resurrection_role in (
            ResurrectionRole.CORE_COVENANT_PATTERN,
            ResurrectionRole.SCAR_BASELINE,
        )


def test_mythic_retrieval():
    """Test that mythic memories can be retrieved."""
    bank = CovenantMemoryBank()

    mythic = bank.recall_mythic()
    assert len(mythic) >= 1

    for card in mythic:
        assert card.resurrection_role == ResurrectionRole.MYTHIC_BACKBONE


def test_context_injection_format():
    """Test that recalled memories format properly for LLM context."""
    bank = CovenantMemoryBank()

    recalled = bank.recall("I'm coming home from work stressed")

    if recalled:
        context = bank.format_recalled_for_context(recalled)
        assert "<recalled_memories>" in context
        assert "</recalled_memories>" in context


def test_prompt_injection():
    """Test that cards can generate prompt injections."""
    cards = load_all_episode_cards()
    card = cards[0]

    injection = card.to_prompt_injection()

    assert "<recalled_memory" in injection
    assert card.id in injection
    assert card.lesson_for_future_ara[:50] in injection


if __name__ == "__main__":
    # Run tests
    test_episode_card_creation()
    print("✓ Episode card creation")

    test_load_real_memories()
    print("✓ Load real memories")

    test_validate_memories()
    print("✓ Validate memories")

    test_encoder_produces_hvs()
    print("✓ Encoder produces HVs")

    test_memory_bank_recall()
    print("✓ Memory bank recall")

    test_core_patterns_retrieval()
    print("✓ Core patterns retrieval")

    test_mythic_retrieval()
    print("✓ Mythic retrieval")

    test_context_injection_format()
    print("✓ Context injection format")

    test_prompt_injection()
    print("✓ Prompt injection")

    print("\n✓ All covenant memory tests passed!")
