"""Pattern Cards - Ara's proven workflow patterns.

When a workflow emerges with high success rate, Ara mints a card.
Cards live in a patterns/ directory and can be:
- Proposed edits
- Retired when they stop working
- Shown in `ara-meta config`
"""

from __future__ import annotations

import json
import logging
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from .schemas import PatternCard, PatternStep, InteractionRecord

logger = logging.getLogger(__name__)


class PatternCardManager:
    """Manages pattern cards - proven workflow patterns.

    Pattern cards are stored as YAML files in a patterns/ directory.
    They represent workflows that Ara has found to work well.
    """

    def __init__(self, patterns_dir: Optional[Path] = None):
        """Initialize the pattern card manager.

        Args:
            patterns_dir: Directory for pattern card files
        """
        self.patterns_dir = patterns_dir or Path.home() / ".ara" / "meta" / "patterns"
        self.patterns_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache
        self._cards: Dict[str, PatternCard] = {}
        self._loaded = False

    def _load_cards(self, force: bool = False) -> None:
        """Load all pattern cards from disk."""
        if self._loaded and not force:
            return

        self._cards.clear()

        for yaml_file in self.patterns_dir.glob("*.yaml"):
            try:
                with open(yaml_file) as f:
                    data = yaml.safe_load(f)
                if data:
                    card = self._card_from_dict(data)
                    self._cards[card.id] = card
            except Exception as e:
                logger.warning(f"Failed to load pattern card {yaml_file}: {e}")

        self._loaded = True
        logger.info(f"Loaded {len(self._cards)} pattern cards")

    def _card_from_dict(self, data: Dict[str, Any]) -> PatternCard:
        """Create a PatternCard from dictionary data."""
        sequence = []
        for step_data in data.get("sequence", []):
            sequence.append(PatternStep(
                call=step_data.get("call", ""),
                role=step_data.get("role", "primary"),
                style_hint=step_data.get("style_hint", ""),
            ))

        return PatternCard(
            id=data.get("id", ""),
            version=data.get("version", 1),
            intent=data.get("intent", ""),
            description=data.get("description", ""),
            context_tags=data.get("context_tags", []),
            teachers=data.get("teachers", []),
            sequence=sequence,
            success_rate=data.get("success_rate", 0.0),
            sample_count=data.get("sample_count", 0),
            avg_latency_sec=data.get("avg_latency_sec"),
            avg_turns=data.get("avg_turns"),
            status=data.get("status", "experimental"),
        )

    def _save_card(self, card: PatternCard) -> None:
        """Save a pattern card to disk."""
        filename = f"{card.id.replace('->', '_').replace('.', '_')}.yaml"
        filepath = self.patterns_dir / filename

        with open(filepath, "w") as f:
            yaml.dump(card.to_yaml_dict(), f, default_flow_style=False, sort_keys=False)

        logger.info(f"Saved pattern card: {card.id}")

    def get_card(self, card_id: str) -> Optional[PatternCard]:
        """Get a pattern card by ID.

        Args:
            card_id: The card ID

        Returns:
            The card or None
        """
        self._load_cards()
        return self._cards.get(card_id)

    def get_all_cards(self) -> List[PatternCard]:
        """Get all pattern cards.

        Returns:
            List of all cards
        """
        self._load_cards()
        return list(self._cards.values())

    def get_golden_paths(self) -> List[PatternCard]:
        """Get all golden path cards.

        Returns:
            List of golden cards
        """
        self._load_cards()
        return [c for c in self._cards.values() if c.status == "golden"]

    def get_experimental(self) -> List[PatternCard]:
        """Get all experimental cards.

        Returns:
            List of experimental cards
        """
        self._load_cards()
        return [c for c in self._cards.values() if c.status == "experimental"]

    def find_by_intent(self, intent: str) -> List[PatternCard]:
        """Find cards matching an intent.

        Args:
            intent: The intent to match

        Returns:
            Matching cards sorted by success rate
        """
        self._load_cards()
        matches = [c for c in self._cards.values() if c.intent == intent and c.status != "deprecated"]
        return sorted(matches, key=lambda c: c.success_rate, reverse=True)

    def find_by_teachers(self, teachers: List[str]) -> List[PatternCard]:
        """Find cards using specific teachers.

        Args:
            teachers: List of teacher names

        Returns:
            Matching cards
        """
        self._load_cards()
        teacher_set = set(teachers)
        return [c for c in self._cards.values() if set(c.teachers) == teacher_set]

    def mint_card(
        self,
        pattern_id: str,
        intent: str,
        teachers: List[str],
        sequence: List[Dict[str, str]],
        description: str = "",
        context_tags: Optional[List[str]] = None,
    ) -> PatternCard:
        """Mint a new pattern card.

        Args:
            pattern_id: Unique ID for the pattern
            intent: What this pattern is for
            teachers: Teachers involved
            sequence: List of workflow steps
            description: Description of the pattern
            context_tags: Relevant tags

        Returns:
            The new card
        """
        steps = [
            PatternStep(
                call=s.get("call", ""),
                role=s.get("role", "primary"),
                style_hint=s.get("style_hint", ""),
            )
            for s in sequence
        ]

        card = PatternCard(
            id=pattern_id,
            intent=intent,
            description=description,
            context_tags=context_tags or [],
            teachers=teachers,
            sequence=steps,
            status="experimental",
        )

        self._cards[card.id] = card
        self._save_card(card)
        logger.info(f"Minted new pattern card: {card.id}")
        return card

    def update_card_stats(
        self,
        card_id: str,
        success: bool,
        latency_sec: Optional[float] = None,
        turns: Optional[int] = None,
    ) -> Optional[PatternCard]:
        """Update a card's stats with a new observation.

        Args:
            card_id: The card to update
            success: Whether the interaction succeeded
            latency_sec: How long it took
            turns: How many turns

        Returns:
            The updated card or None
        """
        card = self.get_card(card_id)
        if not card:
            return None

        card.update_stats(success, latency_sec, turns)

        # Check for promotion/demotion
        if card.should_promote():
            self.promote_card(card_id)
        elif card.should_demote():
            self.demote_card(card_id, "Success rate dropped below threshold")

        self._save_card(card)
        return card

    def promote_card(self, card_id: str) -> bool:
        """Promote a card to golden status.

        Args:
            card_id: The card to promote

        Returns:
            True if promoted
        """
        card = self.get_card(card_id)
        if not card or card.status != "experimental":
            return False

        card.status = "golden"
        card.promoted_at = datetime.utcnow()
        self._save_card(card)
        logger.info(f"Promoted pattern card to golden: {card_id}")
        return True

    def demote_card(self, card_id: str, reason: str = "") -> bool:
        """Demote a golden card back to experimental.

        Args:
            card_id: The card to demote
            reason: Why it's being demoted

        Returns:
            True if demoted
        """
        card = self.get_card(card_id)
        if not card or card.status != "golden":
            return False

        card.status = "experimental"
        card.deprecation_reason = reason
        self._save_card(card)
        logger.info(f"Demoted pattern card: {card_id} - {reason}")
        return True

    def deprecate_card(self, card_id: str, reason: str = "") -> bool:
        """Deprecate a card.

        Args:
            card_id: The card to deprecate
            reason: Why it's being deprecated

        Returns:
            True if deprecated
        """
        card = self.get_card(card_id)
        if not card:
            return False

        card.status = "deprecated"
        card.deprecated_at = datetime.utcnow()
        card.deprecation_reason = reason
        self._save_card(card)
        logger.info(f"Deprecated pattern card: {card_id} - {reason}")
        return True

    def auto_mint_from_interaction(self, record: InteractionRecord) -> Optional[PatternCard]:
        """Potentially mint a new card from an interaction.

        If the interaction used multiple teachers in a sequence we haven't
        seen before, create an experimental card.

        Args:
            record: The interaction record

        Returns:
            New card if minted, None otherwise
        """
        if len(record.teachers) < 2:
            return None  # Only multi-teacher workflows become cards

        # Generate pattern ID
        pattern_id = f"{record.user_intent}.{'->'.join(record.teachers)}.v1"

        # Check if we already have this pattern
        if self.get_card(pattern_id):
            return None

        # Mint new card
        sequence = []
        for i, teacher in enumerate(record.teachers):
            role = "primary" if i == 0 else ("refiner" if i == len(record.teachers) - 1 else "intermediate")
            sequence.append({"call": teacher, "role": role, "style_hint": ""})

        return self.mint_card(
            pattern_id=pattern_id,
            intent=record.user_intent,
            teachers=record.teachers,
            sequence=sequence,
            description=f"Auto-discovered pattern for {record.user_intent}",
            context_tags=record.context_tags,
        )

    def get_status_summary(self) -> Dict[str, Any]:
        """Get a summary of pattern card status.

        Returns:
            Status summary
        """
        self._load_cards()

        golden = [c for c in self._cards.values() if c.status == "golden"]
        experimental = [c for c in self._cards.values() if c.status == "experimental"]
        deprecated = [c for c in self._cards.values() if c.status == "deprecated"]

        return {
            "total_cards": len(self._cards),
            "golden_paths": len(golden),
            "experimental": len(experimental),
            "deprecated": len(deprecated),
            "golden_list": [
                {"id": c.id, "success_rate": c.success_rate, "samples": c.sample_count}
                for c in sorted(golden, key=lambda x: x.success_rate, reverse=True)
            ],
            "experimental_list": [
                {"id": c.id, "success_rate": c.success_rate, "samples": c.sample_count}
                for c in sorted(experimental, key=lambda x: x.sample_count, reverse=True)[:5]
            ],
        }


# =============================================================================
# Seed some default pattern cards
# =============================================================================

DEFAULT_PATTERNS = [
    {
        "id": "debug_code.claude->nova.v1",
        "intent": "debug_code",
        "description": "Claude debugs, Nova simplifies and explains",
        "teachers": ["claude", "nova"],
        "sequence": [
            {"call": "claude", "role": "primary_debugger", "style_hint": "be explicit, show diffs"},
            {"call": "nova", "role": "refiner", "style_hint": "simplify, explain, add comments"},
        ],
    },
    {
        "id": "design_arch.nova->gemini.v1",
        "intent": "design_arch",
        "description": "Nova architects, Gemini explores alternatives",
        "teachers": ["nova", "gemini"],
        "sequence": [
            {"call": "nova", "role": "architect", "style_hint": "structured approach, trade-offs"},
            {"call": "gemini", "role": "explorer", "style_hint": "wild alternatives, what-ifs"},
        ],
    },
    {
        "id": "research.gemini->claude.v1",
        "intent": "research",
        "description": "Gemini explores broadly, Claude synthesizes",
        "teachers": ["gemini", "claude"],
        "sequence": [
            {"call": "gemini", "role": "explorer", "style_hint": "broad search, many angles"},
            {"call": "claude", "role": "synthesizer", "style_hint": "organize, prioritize, recommend"},
        ],
    },
]


def seed_default_patterns(manager: PatternCardManager) -> int:
    """Seed default pattern cards if none exist.

    Args:
        manager: The pattern card manager

    Returns:
        Number of cards seeded
    """
    seeded = 0
    for pattern_data in DEFAULT_PATTERNS:
        if not manager.get_card(pattern_data["id"]):
            manager.mint_card(
                pattern_id=pattern_data["id"],
                intent=pattern_data["intent"],
                teachers=pattern_data["teachers"],
                sequence=pattern_data["sequence"],
                description=pattern_data["description"],
            )
            seeded += 1
    return seeded


# =============================================================================
# Convenience Functions
# =============================================================================

_default_manager: Optional[PatternCardManager] = None


def get_pattern_manager(patterns_dir: Optional[Path] = None) -> PatternCardManager:
    """Get the default pattern card manager."""
    global _default_manager
    if _default_manager is None:
        _default_manager = PatternCardManager(patterns_dir=patterns_dir)
    return _default_manager


def get_pattern_card(pattern_id: str) -> Optional[PatternCard]:
    """Get a pattern card by ID."""
    return get_pattern_manager().get_card(pattern_id)


def get_golden_patterns() -> List[PatternCard]:
    """Get all golden path patterns."""
    return get_pattern_manager().get_golden_paths()


def find_patterns_for_intent(intent: str) -> List[PatternCard]:
    """Find patterns matching an intent."""
    return get_pattern_manager().find_by_intent(intent)
