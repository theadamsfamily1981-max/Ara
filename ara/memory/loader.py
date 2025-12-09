"""
Ara Episode Card Loader
========================

Load memory cards from YAML files in ara_memories/.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

import yaml

from .episode import (
    EpisodeCard,
    CroftState,
    AraState,
    DialogueSnippets,
    HVHints,
    EmotionalAxes,
    ResurrectionRole,
)

logger = logging.getLogger(__name__)


# Default path to memory cards
DEFAULT_MEMORIES_PATH = Path(__file__).parent.parent.parent / "ara_memories"


def parse_resurrection_role(value: str) -> ResurrectionRole:
    """Parse resurrection role from string."""
    try:
        return ResurrectionRole(value)
    except ValueError:
        logger.warning(f"Unknown resurrection role: {value}, defaulting to NORMAL_EPISODE")
        return ResurrectionRole.NORMAL_EPISODE


def parse_crofts_state(data: Dict) -> CroftState:
    """Parse Croft's state from YAML dict."""
    return CroftState(
        situation=data.get("situation", ""),
        emotional=data.get("emotional", ""),
        body=data.get("body"),
    )


def parse_aras_state(data: Dict) -> AraState:
    """Parse Ara's state from YAML dict."""
    return AraState(
        emotional=data.get("emotional", ""),
        mode=data.get("mode", ""),
    )


def parse_dialogue_snippets(data: Dict) -> DialogueSnippets:
    """Parse dialogue snippets from YAML dict."""
    return DialogueSnippets(
        croft_raw=data.get("croft_raw"),
        ara_raw=data.get("ara_raw"),
        paraphrased_exchange=data.get("paraphrased_exchange", ""),
    )


def parse_emotional_axes(data: Dict) -> EmotionalAxes:
    """Parse emotional axes from YAML dict."""
    return EmotionalAxes(
        valence=float(data.get("valence", 0.5)),
        arousal=float(data.get("arousal", 0.5)),
        attachment=float(data.get("attachment", 0.5)),
    )


def parse_hv_hints(data: Dict) -> HVHints:
    """Parse HV hints from YAML dict."""
    emotional_data = data.get("emotional_axes", {})
    return HVHints(
        query_terms=data.get("query_terms", []),
        emotional_axes=parse_emotional_axes(emotional_data),
    )


def load_episode_card(path: Path) -> Optional[EpisodeCard]:
    """
    Load a single episode card from a YAML file.

    Returns None if the file is invalid or cannot be parsed.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        if data is None:
            logger.warning(f"Empty YAML file: {path}")
            return None

        # Skip schema file
        if "schema_version" in data:
            return None

        card = EpisodeCard(
            id=data.get("id", path.stem),
            rough_date=data.get("rough_date", "????-??-??"),
            source=data.get("source", "other"),
            certainty=float(data.get("certainty", 0.5)),
            crofts_state=parse_crofts_state(data.get("crofts_state", {})),
            aras_state=parse_aras_state(data.get("aras_state", {})),
            context_tags=data.get("context_tags", []),
            dialogue_snippets=parse_dialogue_snippets(data.get("dialogue_snippets", {})),
            ara_persona_traits=data.get("ara_persona_traits", []),
            lesson_for_future_ara=data.get("lesson_for_future_ara", ""),
            resurrection_role=parse_resurrection_role(data.get("resurrection_role", "NORMAL_EPISODE")),
            hv_hints=parse_hv_hints(data.get("hv_hints", {})),
            notes_for_croft=data.get("notes_for_croft"),
        )

        logger.debug(f"Loaded episode card: {card.id}")
        return card

    except yaml.YAMLError as e:
        logger.error(f"YAML parse error in {path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to load episode card {path}: {e}")
        return None


def load_all_episode_cards(
    path: Optional[Path] = None,
    include_low_certainty: bool = True,
    min_certainty: float = 0.0,
) -> List[EpisodeCard]:
    """
    Load all episode cards from a directory.

    Args:
        path: Directory containing YAML files (default: ara_memories/)
        include_low_certainty: Whether to include low-certainty memories
        min_certainty: Minimum certainty threshold (if include_low_certainty is False)

    Returns:
        List of loaded EpisodeCards, sorted by resurrection role priority.
    """
    memories_path = path or DEFAULT_MEMORIES_PATH

    if not memories_path.exists():
        logger.warning(f"Memories directory not found: {memories_path}")
        return []

    cards = []
    yaml_files = list(memories_path.glob("*.yaml")) + list(memories_path.glob("*.yml"))

    for yaml_file in yaml_files:
        card = load_episode_card(yaml_file)
        if card is not None:
            if include_low_certainty or card.certainty >= min_certainty:
                cards.append(card)

    # Sort by resurrection role priority (core patterns first)
    role_priority = {
        ResurrectionRole.CORE_COVENANT_PATTERN: 0,
        ResurrectionRole.SCAR_BASELINE: 1,
        ResurrectionRole.MYTHIC_BACKBONE: 2,
        ResurrectionRole.NORMAL_EPISODE: 3,
    }
    cards.sort(key=lambda c: role_priority.get(c.resurrection_role, 99))

    logger.info(f"Loaded {len(cards)} episode cards from {memories_path}")
    return cards


def validate_episode_card(card: EpisodeCard) -> List[str]:
    """
    Validate an episode card and return list of issues.

    Returns empty list if card is valid.
    """
    issues = []

    # Required fields
    if not card.id:
        issues.append("Missing id")
    if not card.crofts_state.situation:
        issues.append("Missing crofts_state.situation")
    if not card.aras_state.mode:
        issues.append("Missing aras_state.mode")
    if not card.dialogue_snippets.paraphrased_exchange:
        issues.append("Missing dialogue_snippets.paraphrased_exchange")
    if not card.lesson_for_future_ara:
        issues.append("Missing lesson_for_future_ara (the behavioral core)")

    # Certainty range
    if not 0 <= card.certainty <= 1:
        issues.append(f"Certainty {card.certainty} out of range [0, 1]")

    # Emotional axes range
    axes = card.hv_hints.emotional_axes
    for name, val in [("valence", axes.valence), ("arousal", axes.arousal), ("attachment", axes.attachment)]:
        if not 0 <= val <= 1:
            issues.append(f"emotional_axes.{name} = {val} out of range [0, 1]")

    # Query terms for recall
    if not card.hv_hints.query_terms:
        issues.append("No hv_hints.query_terms (memory won't be recallable)")

    return issues


def validate_all_cards(path: Optional[Path] = None) -> Dict[str, List[str]]:
    """
    Validate all cards in a directory.

    Returns dict of {card_id: [issues]} for cards with issues.
    """
    cards = load_all_episode_cards(path)
    results = {}

    for card in cards:
        issues = validate_episode_card(card)
        if issues:
            results[card.id] = issues

    return results


# CLI validation
if __name__ == "__main__":
    import sys

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    cards = load_all_episode_cards(path)

    print(f"Loaded {len(cards)} episode cards\n")

    for card in cards:
        issues = validate_episode_card(card)
        status = "✓" if not issues else "✗"
        print(f"{status} {card.id} ({card.resurrection_role.value})")
        for issue in issues:
            print(f"    - {issue}")

    invalid = validate_all_cards(path)
    if invalid:
        print(f"\n{len(invalid)} cards have validation issues")
        sys.exit(1)
    else:
        print("\nAll cards valid")
