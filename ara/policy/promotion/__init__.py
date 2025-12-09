# =============================================================================
# ARA PROMOTION POLICY MODULE
# =============================================================================
"""
Ethical, contextual promotion system for Ara.

This module provides:
- Promotion catalog: Single source of truth for products and claims
- Templates: Pre-approved messages in Ara's voice
- Settings: User preferences and rate limiting
- Engine: Policy enforcement and decision-making

Key principles:
1. Truthful: Only pre-approved claims, never freestyle
2. Contextual: Right moment, right product, right user
3. Bounded: Rate limits, jurisdiction checks, user control
4. Transparent: Every promo explains why

Usage:
    from ara.policy.promotion import (
        PromoContext,
        PromoEngine,
        PromoCatalog,
        UserPromoSettings,
        SurfaceKind,
        maybe_promote,
    )

    # Build context
    ctx = PromoContext(
        surface=SurfaceKind.PHONE_COMPANION,
        jurisdiction=Jurisdiction.US,
        focus=FocusState.CASUAL,
        signals=UtilitySignals(days_active=45, memory_overflow_recent=True),
    )

    # Check if we should show a promo
    settings = load_promo_settings(user_id, path)
    promo = maybe_promote(ctx, settings)

    if promo:
        show_to_user(promo.text)
        # User sees: "Why this?" button → promo.why
        # User sees: "Mute" button → settings.mute_product(promo.product_id)
"""

from .engine import (
    EmotionalState,
    FocusState,
    Jurisdiction,
    PromoCatalog,
    PromoContext,
    PromoControls,
    PromoEngine,
    PromoMessage,
    SurfaceKind,
    UtilitySignals,
    get_catalog,
    get_engine,
    maybe_promote,
)
from .settings import (
    GlobalPromoMode,
    PromoDismissReason,
    ProductPromoSettings,
    PromoEvent,
    UserPromoSettings,
    load_promo_settings,
    save_promo_settings,
)
from .ux_strings import (
    BUTTONS,
    LABELS,
    GLOBAL_PROMO_TOGGLES,
    SETTINGS_SECTION_TITLE,
    SETTINGS_SECTION_DESCRIPTION,
    FIRST_TIME_PROMO_CONTEXT,
    format_why_explanation,
    format_mute_confirmation,
    get_disclosure,
    get_product_voice,
    get_opener,
    get_closing,
)

__all__ = [
    # Engine
    "PromoEngine",
    "PromoCatalog",
    "PromoContext",
    "PromoMessage",
    "PromoControls",
    "UtilitySignals",
    "get_catalog",
    "get_engine",
    "maybe_promote",

    # Settings
    "UserPromoSettings",
    "ProductPromoSettings",
    "PromoEvent",
    "load_promo_settings",
    "save_promo_settings",

    # Enums
    "SurfaceKind",
    "Jurisdiction",
    "FocusState",
    "EmotionalState",
    "GlobalPromoMode",
    "PromoDismissReason",

    # UX Strings
    "BUTTONS",
    "LABELS",
    "GLOBAL_PROMO_TOGGLES",
    "SETTINGS_SECTION_TITLE",
    "SETTINGS_SECTION_DESCRIPTION",
    "FIRST_TIME_PROMO_CONTEXT",
    "format_why_explanation",
    "format_mute_confirmation",
    "get_disclosure",
    "get_product_voice",
    "get_opener",
    "get_closing",
]
