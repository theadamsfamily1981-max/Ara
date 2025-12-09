# =============================================================================
# ARA PROMOTION SETTINGS
# =============================================================================
"""
User-level promotion preferences and state tracking.

This module manages:
- Global promotion toggle (on/off/ask_only)
- Per-product muting
- Rate limit tracking (last promo timestamps)
- Promotion event history

Key principle: User is always in control.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class GlobalPromoMode(str, Enum):
    """User's global promotion preference."""
    ON = "on"              # Proactive suggestions allowed
    OFF = "off"            # No suggestions unless explicitly asked
    ASK_ONLY = "ask_only"  # Only when user asks about products


class PromoDismissReason(str, Enum):
    """Why a promo was dismissed."""
    NOT_NOW = "not_now"           # User said "not now"
    MUTED = "muted"               # User muted this product
    NOT_INTERESTED = "not_interested"  # User explicitly not interested
    ALREADY_HAVE = "already_have"      # User already has this product
    IRRELEVANT = "irrelevant"          # User said this wasn't relevant


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ProductPromoSettings:
    """Per-product promotion settings."""
    product_id: str
    muted: bool = False
    muted_at: Optional[float] = None
    last_promo_ts: float = 0.0
    last_dismiss_ts: float = 0.0
    last_dismiss_reason: Optional[PromoDismissReason] = None
    times_shown: int = 0
    times_dismissed: int = 0
    times_clicked: int = 0

    def mute(self, reason: Optional[PromoDismissReason] = None) -> None:
        """Mute this product's promotions."""
        self.muted = True
        self.muted_at = time.time()
        if reason:
            self.last_dismiss_reason = reason
        logger.info(f"Muted promotions for {self.product_id}")

    def unmute(self) -> None:
        """Unmute this product's promotions."""
        self.muted = False
        self.muted_at = None
        logger.info(f"Unmuted promotions for {self.product_id}")

    def record_shown(self) -> None:
        """Record that a promo was shown."""
        self.last_promo_ts = time.time()
        self.times_shown += 1

    def record_dismissed(self, reason: PromoDismissReason) -> None:
        """Record that a promo was dismissed."""
        self.last_dismiss_ts = time.time()
        self.last_dismiss_reason = reason
        self.times_dismissed += 1

        # Auto-mute if dismissed too many times
        if self.times_dismissed >= 3 and reason != PromoDismissReason.NOT_NOW:
            self.mute(reason)

    def record_clicked(self) -> None:
        """Record that user clicked "learn more"."""
        self.times_clicked += 1

    def days_since_last_promo(self) -> float:
        """Days since last promo was shown."""
        if self.last_promo_ts == 0:
            return float('inf')
        return (time.time() - self.last_promo_ts) / 86400

    def days_since_last_dismiss(self) -> float:
        """Days since last dismissal."""
        if self.last_dismiss_ts == 0:
            return float('inf')
        return (time.time() - self.last_dismiss_ts) / 86400


@dataclass
class PromoEvent:
    """Record of a promotion event."""
    timestamp: float
    product_id: str
    template_id: str
    surface: str
    trigger_reason: str
    outcome: str  # "shown", "clicked", "dismissed", "blocked"
    dismiss_reason: Optional[str] = None
    context_snapshot: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserPromoSettings:
    """
    Complete user promotion preferences and history.

    This is persisted per-user and loaded on session start.
    """
    user_id: str

    # Global settings
    global_mode: GlobalPromoMode = GlobalPromoMode.ON
    global_mode_changed_at: Optional[float] = None

    # Timestamps
    last_global_promo_ts: float = 0.0
    first_promo_ever_ts: Optional[float] = None

    # Per-product settings
    per_product: Dict[str, ProductPromoSettings] = field(default_factory=dict)

    # Event history (recent, for analysis)
    recent_events: List[PromoEvent] = field(default_factory=list)
    max_events: int = 100

    # Stats
    total_promos_shown: int = 0
    total_promos_clicked: int = 0
    total_promos_dismissed: int = 0

    def get_product_settings(self, product_id: str) -> ProductPromoSettings:
        """Get or create settings for a product."""
        if product_id not in self.per_product:
            self.per_product[product_id] = ProductPromoSettings(product_id=product_id)
        return self.per_product[product_id]

    def set_global_mode(self, mode: GlobalPromoMode) -> None:
        """Update global promotion mode."""
        self.global_mode = mode
        self.global_mode_changed_at = time.time()
        logger.info(f"Global promo mode set to: {mode.value}")

    def promos_allowed(self) -> bool:
        """Check if proactive promos are allowed globally."""
        return self.global_mode == GlobalPromoMode.ON

    def explicit_promos_only(self) -> bool:
        """Check if only explicit (user-requested) promos allowed."""
        return self.global_mode in (GlobalPromoMode.OFF, GlobalPromoMode.ASK_ONLY)

    def record_promo_shown(
        self,
        product_id: str,
        template_id: str,
        surface: str,
        trigger_reason: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record that a promo was shown."""
        now = time.time()

        # Update global state
        self.last_global_promo_ts = now
        self.total_promos_shown += 1

        if self.first_promo_ever_ts is None:
            self.first_promo_ever_ts = now

        # Update product state
        product_settings = self.get_product_settings(product_id)
        product_settings.record_shown()

        # Record event
        event = PromoEvent(
            timestamp=now,
            product_id=product_id,
            template_id=template_id,
            surface=surface,
            trigger_reason=trigger_reason,
            outcome="shown",
            context_snapshot=context or {},
        )
        self._add_event(event)

    def record_promo_clicked(self, product_id: str) -> None:
        """Record that user clicked to learn more."""
        self.total_promos_clicked += 1
        self.get_product_settings(product_id).record_clicked()

    def record_promo_dismissed(
        self,
        product_id: str,
        reason: PromoDismissReason,
    ) -> None:
        """Record that a promo was dismissed."""
        self.total_promos_dismissed += 1
        self.get_product_settings(product_id).record_dismissed(reason)

    def mute_product(
        self,
        product_id: str,
        reason: Optional[PromoDismissReason] = None,
    ) -> None:
        """Mute a specific product's promotions."""
        self.get_product_settings(product_id).mute(reason)

    def unmute_product(self, product_id: str) -> None:
        """Unmute a specific product's promotions."""
        self.get_product_settings(product_id).unmute()

    def is_product_muted(self, product_id: str) -> bool:
        """Check if a product is muted."""
        if product_id not in self.per_product:
            return False
        return self.per_product[product_id].muted

    def days_since_last_promo(self) -> float:
        """Days since any promo was shown."""
        if self.last_global_promo_ts == 0:
            return float('inf')
        return (time.time() - self.last_global_promo_ts) / 86400

    def promos_today(self) -> int:
        """Count promos shown today."""
        today_start = time.time() - (time.time() % 86400)
        return sum(
            1 for e in self.recent_events
            if e.timestamp >= today_start and e.outcome == "shown"
        )

    def promos_this_week(self) -> int:
        """Count promos shown this week."""
        week_start = time.time() - (7 * 86400)
        return sum(
            1 for e in self.recent_events
            if e.timestamp >= week_start and e.outcome == "shown"
        )

    def is_first_promo_ever(self) -> bool:
        """Check if this would be the first promo ever shown."""
        return self.first_promo_ever_ts is None

    def _add_event(self, event: PromoEvent) -> None:
        """Add an event, maintaining max size."""
        self.recent_events.append(event)
        if len(self.recent_events) > self.max_events:
            self.recent_events = self.recent_events[-self.max_events:]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for storage."""
        return {
            "user_id": self.user_id,
            "global_mode": self.global_mode.value,
            "global_mode_changed_at": self.global_mode_changed_at,
            "last_global_promo_ts": self.last_global_promo_ts,
            "first_promo_ever_ts": self.first_promo_ever_ts,
            "per_product": {
                pid: {
                    "product_id": ps.product_id,
                    "muted": ps.muted,
                    "muted_at": ps.muted_at,
                    "last_promo_ts": ps.last_promo_ts,
                    "last_dismiss_ts": ps.last_dismiss_ts,
                    "last_dismiss_reason": ps.last_dismiss_reason.value if ps.last_dismiss_reason else None,
                    "times_shown": ps.times_shown,
                    "times_dismissed": ps.times_dismissed,
                    "times_clicked": ps.times_clicked,
                }
                for pid, ps in self.per_product.items()
            },
            "total_promos_shown": self.total_promos_shown,
            "total_promos_clicked": self.total_promos_clicked,
            "total_promos_dismissed": self.total_promos_dismissed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserPromoSettings":
        """Deserialize from dictionary."""
        settings = cls(user_id=data.get("user_id", "unknown"))

        settings.global_mode = GlobalPromoMode(data.get("global_mode", "on"))
        settings.global_mode_changed_at = data.get("global_mode_changed_at")
        settings.last_global_promo_ts = data.get("last_global_promo_ts", 0.0)
        settings.first_promo_ever_ts = data.get("first_promo_ever_ts")
        settings.total_promos_shown = data.get("total_promos_shown", 0)
        settings.total_promos_clicked = data.get("total_promos_clicked", 0)
        settings.total_promos_dismissed = data.get("total_promos_dismissed", 0)

        for pid, ps_data in data.get("per_product", {}).items():
            ps = ProductPromoSettings(product_id=pid)
            ps.muted = ps_data.get("muted", False)
            ps.muted_at = ps_data.get("muted_at")
            ps.last_promo_ts = ps_data.get("last_promo_ts", 0.0)
            ps.last_dismiss_ts = ps_data.get("last_dismiss_ts", 0.0)
            if ps_data.get("last_dismiss_reason"):
                ps.last_dismiss_reason = PromoDismissReason(ps_data["last_dismiss_reason"])
            ps.times_shown = ps_data.get("times_shown", 0)
            ps.times_dismissed = ps_data.get("times_dismissed", 0)
            ps.times_clicked = ps_data.get("times_clicked", 0)
            settings.per_product[pid] = ps

        return settings


# =============================================================================
# Persistence
# =============================================================================

def save_promo_settings(settings: UserPromoSettings, path: Path) -> None:
    """Save settings to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(settings.to_dict(), f, indent=2)
    logger.debug(f"Saved promo settings to {path}")


def load_promo_settings(user_id: str, path: Path) -> UserPromoSettings:
    """Load settings from file, or create default."""
    if path.exists():
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            return UserPromoSettings.from_dict(data)
        except Exception as e:
            logger.warning(f"Failed to load promo settings: {e}")

    return UserPromoSettings(user_id=user_id)
