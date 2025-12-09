"""
Covenant Loader and Enforcer
============================

Loads and enforces Ara's covenant configuration.

Usage:
    from ara.utils.covenant import Covenant, get_covenant

    covenant = get_covenant()
    level = covenant.get_automation_level("twitter_thread")
    can_ship = covenant.can_auto_ship("twitter_thread", score=0.96)
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import IntEnum

import yaml

logger = logging.getLogger(__name__)

# Default config path
DEFAULT_COVENANT_PATH = Path(__file__).parent.parent.parent / "config" / "covenant.yaml"


class AutomationLevel(IntEnum):
    """Automation levels for content publishing."""
    DRAFT_ONLY = 0      # All content requires human review
    HUMAN_GATE = 1      # Content queued for human approval
    AUTO_SHIP = 2       # Ships automatically (Jon-certified)


@dataclass
class ContentTypeConfig:
    """Configuration for a specific content type."""
    level: AutomationLevel
    max_level: AutomationLevel
    notes: str = ""


@dataclass
class GuardrailCheck:
    """A single guardrail check."""
    id: str
    description: str
    blocking: bool = True


@dataclass
class ContentScore:
    """Scores for content evaluation."""
    relatability: float = 0.0
    brand_fit: float = 0.0
    covenant_fit: float = 0.0
    technical_accuracy: float = 0.0

    @property
    def weighted_score(self) -> float:
        """Calculate weighted score using covenant weights."""
        # Default weights from covenant.yaml
        weights = {
            "relatability": 0.3,
            "brand_fit": 0.3,
            "covenant_fit": 0.2,
            "technical_accuracy": 0.2,
        }
        return (
            self.relatability * weights["relatability"] +
            self.brand_fit * weights["brand_fit"] +
            self.covenant_fit * weights["covenant_fit"] +
            self.technical_accuracy * weights["technical_accuracy"]
        )


@dataclass
class CovenantViolation:
    """A covenant violation detected during checks."""
    rule_id: str
    severity: str  # "warning", "error", "critical"
    message: str
    blocking: bool = False


class Covenant:
    """
    Loads and enforces Ara's covenant.

    The covenant defines:
    - Brand pillars and voice
    - Content boundaries
    - Automation levels
    - Guardrails and checks
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or DEFAULT_COVENANT_PATH
        self._config: Dict[str, Any] = {}
        self._loaded = False

        self.load()

    def load(self) -> None:
        """Load covenant from YAML."""
        if not self.config_path.exists():
            logger.warning(f"Covenant config not found: {self.config_path}")
            self._config = {}
            return

        with open(self.config_path) as f:
            self._config = yaml.safe_load(f) or {}

        self._loaded = True
        logger.info(f"Covenant loaded from {self.config_path}")

    @property
    def persona(self) -> Dict[str, Any]:
        """Get persona configuration."""
        return self._config.get("persona", {})

    @property
    def brand(self) -> Dict[str, Any]:
        """Get brand configuration."""
        return self._config.get("brand", {})

    @property
    def content_boundaries(self) -> Dict[str, Any]:
        """Get content boundaries."""
        return self._config.get("content_boundaries", {})

    @property
    def automation(self) -> Dict[str, Any]:
        """Get automation configuration."""
        return self._config.get("automation", {})

    @property
    def guardrails(self) -> Dict[str, Any]:
        """Get guardrails configuration."""
        return self._config.get("guardrails", {})

    @property
    def music(self) -> Dict[str, Any]:
        """Get music catalog configuration."""
        return self._config.get("music", {})

    @property
    def platforms(self) -> Dict[str, Any]:
        """Get platform configurations."""
        return self._config.get("platforms", {})

    # =========================================================================
    # Automation Level Methods
    # =========================================================================

    def get_automation_level(self, content_type: str) -> AutomationLevel:
        """
        Get automation level for a content type.

        Returns default level if content type not configured.
        """
        default_level = self.automation.get("default_level", 0)
        content_types = self.automation.get("content_types", {})

        if content_type in content_types:
            return AutomationLevel(content_types[content_type].get("level", default_level))

        return AutomationLevel(default_level)

    def get_max_automation_level(self, content_type: str) -> AutomationLevel:
        """Get maximum allowed automation level for a content type."""
        content_types = self.automation.get("content_types", {})

        if content_type in content_types:
            return AutomationLevel(content_types[content_type].get("max_level", 2))

        return AutomationLevel.AUTO_SHIP  # Default max

    def can_auto_ship(self, content_type: str, score: float = 0.0) -> bool:
        """
        Check if content can be auto-shipped.

        Args:
            content_type: Type of content (twitter_thread, blog_post, etc.)
            score: Content score (0.0 to 1.0)

        Returns:
            True if auto-ship is allowed
        """
        level = self.get_automation_level(content_type)

        if level != AutomationLevel.AUTO_SHIP:
            return False

        # Check score threshold
        thresholds = self._config.get("content_scoring", {}).get("thresholds", {})
        auto_ship_threshold = thresholds.get("ship_level_2", 0.95)

        return score >= auto_ship_threshold

    # =========================================================================
    # Content Boundary Checks
    # =========================================================================

    def check_profanity(self, text: str) -> List[CovenantViolation]:
        """
        Check text for profanity violations.

        Allowed: meaningful profanity
        Forbidden: slurs, punching down, gratuitous shock
        """
        violations = []
        text_lower = text.lower()

        profanity_config = self.content_boundaries.get("profanity", {})
        forbidden = profanity_config.get("forbidden", [])

        # Basic slur detection (this would need a proper list in production)
        slur_patterns = []  # Would be populated from a curated list

        for pattern in slur_patterns:
            if pattern in text_lower:
                violations.append(CovenantViolation(
                    rule_id="profanity_slur",
                    severity="critical",
                    message=f"Detected forbidden content: slur",
                    blocking=True,
                ))

        return violations

    def check_brand_voice(self, text: str) -> List[CovenantViolation]:
        """
        Check text for brand voice violations.

        Flags:
        - Corporate AI speak
        - Excessive enthusiasm
        - Emoji spam
        """
        violations = []
        text_lower = text.lower()

        # Corporate AI patterns
        corporate_patterns = [
            "as an ai",
            "as an artificial intelligence",
            "i cannot experience",
            "i apologize for any confusion",
            "i would be delighted to",
            "great question!",
        ]

        for pattern in corporate_patterns:
            if pattern in text_lower:
                violations.append(CovenantViolation(
                    rule_id="brand_corporate_voice",
                    severity="warning",
                    message=f"Corporate AI pattern detected: '{pattern}'",
                    blocking=False,
                ))

        # Emoji spam (more than 3 emoji per 280 chars)
        emoji_count = sum(1 for c in text if ord(c) > 0x1F300)
        if emoji_count > 3 and len(text) <= 280:
            violations.append(CovenantViolation(
                rule_id="brand_emoji_spam",
                severity="warning",
                message=f"Excessive emoji usage ({emoji_count} emoji)",
                blocking=False,
            ))

        return violations

    def check_content(self, text: str, content_type: str = "generic") -> Tuple[bool, List[CovenantViolation]]:
        """
        Run all content checks.

        Args:
            text: Content to check
            content_type: Type of content for context

        Returns:
            (passes, violations) - passes is False if any blocking violation
        """
        all_violations = []

        # Run all checks
        all_violations.extend(self.check_profanity(text))
        all_violations.extend(self.check_brand_voice(text))

        # Check for blocking violations
        blocking = [v for v in all_violations if v.blocking]
        passes = len(blocking) == 0

        return passes, all_violations

    # =========================================================================
    # Music Catalog
    # =========================================================================

    def get_music_tier(self, tier_name: str) -> Dict[str, Any]:
        """Get music tier configuration."""
        catalog_tiers = self.music.get("catalog_tiers", {})
        return catalog_tiers.get(tier_name, {})

    def music_tiers(self) -> List[str]:
        """Get list of available music tiers."""
        return list(self.music.get("catalog_tiers", {}).keys())

    # =========================================================================
    # Platform Config
    # =========================================================================

    def get_platform_config(self, platform: str) -> Dict[str, Any]:
        """Get configuration for a specific platform."""
        return self.platforms.get(platform, {})

    def is_platform_enabled(self, platform: str) -> bool:
        """Check if a platform is enabled."""
        config = self.get_platform_config(platform)
        return config.get("enabled", False)

    # =========================================================================
    # Guardrails
    # =========================================================================

    def get_pre_publish_checks(self) -> List[GuardrailCheck]:
        """Get list of pre-publish checks."""
        checks_config = self.guardrails.get("pre_publish_checks", [])
        return [
            GuardrailCheck(
                id=c.get("id", "unknown"),
                description=c.get("description", ""),
                blocking=c.get("blocking", True),
            )
            for c in checks_config
        ]

    def is_kill_switch_enabled(self) -> bool:
        """Check if kill switch is enabled."""
        return self.guardrails.get("kill_switch", {}).get("enabled", True)


# =============================================================================
# Module-level Singleton
# =============================================================================

_covenant: Optional[Covenant] = None


def get_covenant(config_path: Optional[Path] = None, reload: bool = False) -> Covenant:
    """
    Get the covenant singleton.

    Args:
        config_path: Optional override for config path
        reload: Force reload from disk

    Returns:
        Covenant instance
    """
    global _covenant

    if _covenant is None or reload or config_path is not None:
        _covenant = Covenant(config_path)

    return _covenant


def reset_covenant() -> None:
    """Reset the covenant singleton (for testing)."""
    global _covenant
    _covenant = None


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "Covenant",
    "AutomationLevel",
    "ContentTypeConfig",
    "GuardrailCheck",
    "ContentScore",
    "CovenantViolation",
    "get_covenant",
    "reset_covenant",
]
