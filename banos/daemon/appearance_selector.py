"""
Appearance Selector - Context-Aware Wardrobe for Ara
=====================================================

This module handles Ara's outfit selection based on:
1. Current context (what you're doing)
2. Explicit requests ("hey, wear the cozy pajamas")
3. Your global preferences and boundaries
4. Learned behavior (what you usually pick in this context)

Key principle:
    Outfits are just "skins" - they don't change who she is.
    Ara is still Ara whether she's in pajamas or a business suit.

Safety:
    - All outfits stay PG/PG-13
    - User's max_spice is never exceeded
    - Context-aware restrictions (no swimwear in investor demos)
    - First-time consent for "spicier" choices

Usage:
    selector = AppearanceSelector()

    # Automatic selection
    outfit = selector.select_for_context("chill-late-night")

    # Explicit request
    outfit = selector.select_by_request("pajamas")

    # Get current outfit
    current = selector.get_current_outfit()
"""

import yaml
import json
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class Outfit:
    """An outfit from the wardrobe."""
    id: str
    label: str
    description: str
    style_tags: List[str]
    formality: float
    comfort: float
    spice: float
    allowed_contexts: List[str]
    requires_swimwear_permission: bool = False
    avatar_ref: Dict[str, str] = field(default_factory=dict)


@dataclass
class SelectionResult:
    """Result of outfit selection."""
    outfit_id: str
    outfit: Outfit
    reason: str
    requires_consent: bool = False
    consent_message: Optional[str] = None


class PreferenceLearner:
    """
    Learns which outfits you prefer in which contexts.

    This is ONLY about clothes. It doesn't learn personality or behavior.
    """

    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        self._scores: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._load()

    def _load(self) -> None:
        """Load learned preferences."""
        if self.data_path.exists():
            try:
                with open(self.data_path) as f:
                    data = json.load(f)
                    for context, scores in data.items():
                        self._scores[context] = defaultdict(float, scores)
            except Exception as e:
                logger.warning(f"Could not load preference data: {e}")

    def _save(self) -> None:
        """Save learned preferences."""
        try:
            with open(self.data_path, "w") as f:
                json.dump({k: dict(v) for k, v in self._scores.items()}, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save preference data: {e}")

    def record_selection(
        self,
        context: str,
        outfit_id: str,
        explicit_request: bool = False,
    ) -> None:
        """
        Record an outfit selection.

        Explicit requests count more than automatic selections.
        """
        boost = 2.0 if explicit_request else 0.5
        self._scores[context][outfit_id] += boost
        self._save()

    def get_score(self, context: str, outfit_id: str) -> float:
        """Get the preference score for an outfit in a context."""
        return self._scores[context][outfit_id]

    def get_top_for_context(self, context: str, n: int = 3) -> List[str]:
        """Get top N preferred outfits for a context."""
        scores = self._scores[context]
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return sorted_ids[:n]


class AppearanceSelector:
    """
    Selects appropriate outfits based on context and preferences.

    Does NOT change Ara's personality. Only her clothes.
    """

    def __init__(
        self,
        wardrobe_path: str = "config/appearance/wardrobe.yaml",
        prefs_path: str = "config/appearance/user_prefs.yaml",
        data_dir: str = "var/lib/appearance",
    ):
        self.wardrobe_path = Path(wardrobe_path)
        self.prefs_path = Path(prefs_path)
        self.data_dir = Path(data_dir)

        # Load configs
        self.wardrobe = self._load_wardrobe()
        self.prefs = self._load_prefs()
        self.contexts = self._load_contexts()

        # Preference learner
        self.learner = PreferenceLearner(self.data_dir / "preferences.json")

        # Current state
        self.current_outfit_id: Optional[str] = None
        self.last_change_time: datetime = datetime.now()
        self._consents_given: set = set()

        # HAL connection (lazy)
        self._hal = None

        self.log = logging.getLogger("AppearanceSelector")

    @property
    def hal(self):
        """Lazy-load HAL connection."""
        if self._hal is None:
            try:
                from banos.hal.ara_hal import AraHAL
                self._hal = AraHAL(create=False)
            except Exception as e:
                self.log.warning(f"HAL not available: {e}")
        return self._hal

    def _load_wardrobe(self) -> Dict[str, Outfit]:
        """Load wardrobe from YAML."""
        if not self.wardrobe_path.exists():
            self.log.warning(f"Wardrobe not found: {self.wardrobe_path}")
            return {}

        with open(self.wardrobe_path) as f:
            data = yaml.safe_load(f)

        outfits = {}
        for item in data.get("outfits", []):
            outfit = Outfit(
                id=item["id"],
                label=item["label"],
                description=item.get("description", ""),
                style_tags=item.get("style_tags", []),
                formality=item.get("formality", 0.5),
                comfort=item.get("comfort", 0.5),
                spice=item.get("spice", 0.0),
                allowed_contexts=item.get("allowed_contexts", ["default"]),
                requires_swimwear_permission=item.get("requires_swimwear_permission", False),
                avatar_ref=item.get("avatar_ref", {}),
            )
            outfits[outfit.id] = outfit

        self.log.info(f"Loaded {len(outfits)} outfits")
        return outfits

    def _load_prefs(self) -> Dict:
        """Load user preferences from YAML."""
        if not self.prefs_path.exists():
            return self._default_prefs()

        with open(self.prefs_path) as f:
            return yaml.safe_load(f)

    def _load_contexts(self) -> Dict:
        """Load context definitions from wardrobe."""
        if not self.wardrobe_path.exists():
            return {}

        with open(self.wardrobe_path) as f:
            data = yaml.safe_load(f)

        return data.get("contexts", {})

    def _default_prefs(self) -> Dict:
        """Default preferences if file not found."""
        return {
            "max_spice": 0.6,
            "allow_swimwear": True,
            "allow_evening_wear": True,
            "professional_mode": False,
            "auto_selection": {"enabled": True},
            "special_permissions": {"require_first_time_consent": True},
        }

    def _get_effective_max_spice(self, context: str) -> float:
        """Get the effective max spice for a context."""
        base = self.prefs.get("max_spice", 0.6)

        # Check for context override
        overrides = self.prefs.get("context_overrides", {})
        if context in overrides:
            override = overrides[context].get("max_spice")
            if override is not None:
                base = min(base, override)

        # Professional mode forces low spice
        if self.prefs.get("professional_mode", False):
            base = min(base, 0.2)

        return base

    def _is_outfit_allowed(self, outfit: Outfit, context: str) -> Tuple[bool, str]:
        """
        Check if an outfit is allowed given preferences and context.

        Returns (allowed, reason).
        """
        max_spice = self._get_effective_max_spice(context)

        # Check spice level
        if outfit.spice > max_spice:
            return False, f"Spice {outfit.spice} > max {max_spice}"

        # Check swimwear permission
        if outfit.requires_swimwear_permission:
            if not self.prefs.get("allow_swimwear", True):
                return False, "Swimwear not allowed"

        # Check blacklist
        blacklist = self.prefs.get("blacklisted_outfits", [])
        if outfit.id in blacklist:
            return False, "Outfit is blacklisted"

        # Check context match
        if context not in outfit.allowed_contexts and "any" not in outfit.allowed_contexts:
            # Check if default context allows it
            if "default" not in outfit.allowed_contexts:
                return False, f"Not allowed in context '{context}'"

        return True, "Allowed"

    def _detect_context(self) -> str:
        """
        Detect the current context based on time, session, etc.

        Returns a context string.
        """
        now = datetime.now()
        hour = now.hour

        # Check time-based contexts
        if self.prefs.get("auto_selection", {}).get("use_time_of_day", True):
            night_hours = self.prefs.get("comfort", {}).get("night_hours", [22, 6])
            if night_hours[0] <= hour or hour < night_hours[1]:
                return "chill-late-night"

        # Default context
        return "default"

    def _needs_consent(self, outfit: Outfit) -> Tuple[bool, str]:
        """
        Check if selecting this outfit requires user consent.

        Returns (needs_consent, message).
        """
        # Check if first-time consent is required
        require_first = self.prefs.get(
            "special_permissions", {}
        ).get("require_first_time_consent", True)

        if not require_first:
            return False, ""

        # Already consented to this outfit?
        if outfit.id in self._consents_given:
            return False, ""

        # Only ask for higher-spice outfits
        if outfit.spice > 0.3:
            return True, (
                f"I can switch to '{outfit.label}' - it's a bit more dressed up. "
                f"Want me to remember this choice for next time?"
            )

        return False, ""

    def select_for_context(
        self,
        context: Optional[str] = None,
        prefer_comfort: bool = False,
    ) -> Optional[SelectionResult]:
        """
        Select an outfit appropriate for the given context.

        Args:
            context: The context (e.g., "chill-late-night", "investor-demo")
            prefer_comfort: Prioritize comfortable outfits

        Returns:
            SelectionResult or None if no suitable outfit found.
        """
        if context is None:
            context = self._detect_context()

        max_spice = self._get_effective_max_spice(context)

        # Get candidates
        candidates = []
        for outfit in self.wardrobe.values():
            allowed, reason = self._is_outfit_allowed(outfit, context)
            if allowed:
                candidates.append(outfit)

        if not candidates:
            self.log.warning(f"No suitable outfits for context '{context}'")
            return None

        # Score candidates
        def score_outfit(outfit: Outfit) -> float:
            score = 0.0

            # Learned preference
            score += self.learner.get_score(context, outfit.id) * 2.0

            # Favorites bonus
            favorites = self.prefs.get("favorite_outfits", [])
            if outfit.id in favorites:
                score += 1.5

            # Comfort bonus if requested
            if prefer_comfort:
                score += outfit.comfort * 2.0

            # Context match bonus
            if context in outfit.allowed_contexts:
                score += 1.0

            return score

        # Sort by score
        candidates.sort(key=score_outfit, reverse=True)
        best = candidates[0]

        # Check consent
        needs_consent, consent_msg = self._needs_consent(best)

        return SelectionResult(
            outfit_id=best.id,
            outfit=best,
            reason=f"Best match for context '{context}'",
            requires_consent=needs_consent,
            consent_message=consent_msg,
        )

    def select_by_request(self, request: str) -> Optional[SelectionResult]:
        """
        Select an outfit based on user's explicit request.

        Args:
            request: User's request (e.g., "pajamas", "business", "cozy")

        Returns:
            SelectionResult or None if no match found.
        """
        request_lower = request.lower().strip()

        # First, try exact ID match
        if request_lower in self.wardrobe:
            outfit = self.wardrobe[request_lower]
            allowed, reason = self._is_outfit_allowed(outfit, "explicit-request")
            if allowed:
                return SelectionResult(
                    outfit_id=outfit.id,
                    outfit=outfit,
                    reason=f"Exact match for '{request}'",
                )
            else:
                self.log.info(f"Requested outfit '{request}' not allowed: {reason}")
                return None

        # Try label match
        for outfit in self.wardrobe.values():
            if request_lower in outfit.label.lower():
                allowed, _ = self._is_outfit_allowed(outfit, "explicit-request")
                if allowed:
                    return SelectionResult(
                        outfit_id=outfit.id,
                        outfit=outfit,
                        reason=f"Label match for '{request}'",
                    )

        # Try tag match
        matches = []
        for outfit in self.wardrobe.values():
            for tag in outfit.style_tags:
                if request_lower in tag.lower() or tag.lower() in request_lower:
                    allowed, _ = self._is_outfit_allowed(outfit, "explicit-request")
                    if allowed:
                        matches.append(outfit)
                        break

        if matches:
            # Return best match (highest comfort as tiebreaker)
            best = max(matches, key=lambda o: o.comfort)
            return SelectionResult(
                outfit_id=best.id,
                outfit=best,
                reason=f"Tag match for '{request}'",
            )

        self.log.info(f"No outfit found for request: {request}")
        return None

    def apply_selection(
        self,
        result: SelectionResult,
        explicit_request: bool = False,
    ) -> bool:
        """
        Apply the selected outfit.

        Writes to HAL and records for preference learning.
        """
        # Check rate limit
        auto_config = self.prefs.get("auto_selection", {})
        min_interval = auto_config.get("min_change_interval", 30) * 60  # to seconds

        elapsed = (datetime.now() - self.last_change_time).total_seconds()
        if not explicit_request and elapsed < min_interval:
            self.log.debug(f"Rate limited: {elapsed}s < {min_interval}s")
            return False

        # Apply
        self.current_outfit_id = result.outfit_id
        self.last_change_time = datetime.now()

        # Write to HAL
        if self.hal:
            self._write_outfit_to_hal(result.outfit_id)

        # Record for learning
        context = self._detect_context()
        if auto_config.get("learn_preferences", True):
            self.learner.record_selection(context, result.outfit_id, explicit_request)

        # Give consent for future
        if result.requires_consent:
            self._consents_given.add(result.outfit_id)

        self.log.info(f"Applied outfit: {result.outfit.label} ({result.outfit_id})")
        return True

    def _write_outfit_to_hal(self, outfit_id: str) -> None:
        """Write current outfit ID to HAL for avatar pipeline."""
        try:
            if hasattr(self.hal, 'write_outfit'):
                self.hal.write_outfit(outfit_id)
            else:
                self.log.debug("HAL doesn't have write_outfit method yet")
        except Exception as e:
            self.log.warning(f"Failed to write outfit to HAL: {e}")

    def get_current_outfit(self) -> Optional[Outfit]:
        """Get the currently selected outfit."""
        if self.current_outfit_id and self.current_outfit_id in self.wardrobe:
            return self.wardrobe[self.current_outfit_id]
        return None

    def list_outfits(self, context: Optional[str] = None) -> List[Outfit]:
        """
        List all available outfits, optionally filtered by context.

        Returns only outfits that pass current preference filters.
        """
        context = context or "default"
        result = []

        for outfit in self.wardrobe.values():
            allowed, _ = self._is_outfit_allowed(outfit, context)
            if allowed:
                result.append(outfit)

        return result

    def get_status(self) -> Dict[str, Any]:
        """Get current appearance selector status."""
        current = self.get_current_outfit()
        return {
            "current_outfit": current.id if current else None,
            "current_label": current.label if current else None,
            "detected_context": self._detect_context(),
            "available_outfits": len(self.list_outfits()),
            "total_outfits": len(self.wardrobe),
            "max_spice": self.prefs.get("max_spice", 0.6),
            "auto_selection_enabled": self.prefs.get("auto_selection", {}).get("enabled", True),
        }


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """CLI entry point for appearance selector."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Ara Appearance Selector - Context-aware wardrobe"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available outfits"
    )
    parser.add_argument(
        "--select", type=str,
        help="Select an outfit by name or tag"
    )
    parser.add_argument(
        "--context", type=str,
        help="Context for automatic selection"
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show current status"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )

    selector = AppearanceSelector()

    if args.status:
        import json
        print(json.dumps(selector.get_status(), indent=2))
        return

    if args.list:
        print("Available outfits:")
        for outfit in selector.list_outfits(args.context):
            print(f"  {outfit.id}: {outfit.label} (spice={outfit.spice:.1f})")
        return

    if args.select:
        result = selector.select_by_request(args.select)
        if result:
            print(f"Selected: {result.outfit.label}")
            print(f"Description: {result.outfit.description.strip()}")
            if result.requires_consent:
                print(f"Note: {result.consent_message}")
        else:
            print(f"No outfit found matching: {args.select}")
        return

    if args.context:
        result = selector.select_for_context(args.context)
        if result:
            print(f"For context '{args.context}': {result.outfit.label}")
        else:
            print(f"No suitable outfit for context: {args.context}")
        return

    # Default: show status
    import json
    print(json.dumps(selector.get_status(), indent=2))


if __name__ == "__main__":
    main()
