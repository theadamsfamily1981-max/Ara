# =============================================================================
# ARA PROMOTION POLICY ENGINE
# =============================================================================
"""
Core engine for ethical, contextual promotions.

This module enforces:
- Truthfulness: Only pre-approved claims
- Context: Never during distress, focus, or emergencies
- Boundaries: Rate limits, jurisdiction checks, user preferences
- Transparency: Every promo explains why

Key invariant: "If in doubt, don't promote."
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml

from .settings import (
    GlobalPromoMode,
    PromoDismissReason,
    UserPromoSettings,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Types and Enums
# =============================================================================

class SurfaceKind(str, Enum):
    """Where Ara is running."""
    PHONE_COMPANION = "phone_companion"
    DESKTOP_COMPANION = "desktop_companion"
    ELDER_DEVICE = "elder_device"
    HEALTHCARE = "healthcare"
    DATACENTER = "datacenter"
    ENTERPRISE_CONSOLE = "enterprise_console"
    DEVELOPER_CONSOLE = "developer_console"
    CAREGIVER_DASHBOARD = "caregiver_dashboard"
    CLINICIAN_DASHBOARD = "clinician_dashboard"


class Jurisdiction(str, Enum):
    """Geographic/regulatory jurisdiction."""
    US = "us"
    EU = "eu"
    UK = "uk"
    GLOBAL = "global"


class FocusState(str, Enum):
    """User's current focus level."""
    HIGH_FOCUS = "high_focus"     # Deep work, no interruptions
    MODERATE = "moderate"          # Normal engagement
    CASUAL = "casual"              # Relaxed, browsing
    DISTRESSED = "distressed"      # Emotional distress
    UNKNOWN = "unknown"


class EmotionalState(str, Enum):
    """User's emotional state (from mood module)."""
    CALM = "calm"
    HAPPY = "happy"
    SAD = "sad"
    ANXIOUS = "anxious"
    GRIEVING = "grieving"
    CRISIS = "crisis"
    UNKNOWN = "unknown"


# =============================================================================
# Context
# =============================================================================

@dataclass
class UtilitySignals:
    """Signals that indicate a product might be useful."""
    # Usage patterns
    days_active: int = 0
    daily_interactions: int = 0
    memory_overflow_recent: bool = False
    cross_device_interest: bool = False
    hit_autonomy_limit: bool = False

    # Contextual signals
    caregiver_context: bool = False
    elder_parent_mentions: int = 0
    loneliness_mentions: int = 0
    developer_context: bool = False

    # Infrastructure signals
    gpu_idle_percent: float = 0.0
    manual_tuning_detected: bool = False
    scaling_struggles: bool = False

    # Explicit asks
    explicit_product_inquiry: bool = False
    explicit_health_inquiry: bool = False


@dataclass
class PromoContext:
    """
    Complete context for promo decision-making.

    Gathered at the moment we consider showing a promo.
    """
    # Where and when
    surface: SurfaceKind
    jurisdiction: Jurisdiction
    now: float = field(default_factory=time.time)

    # User state
    focus: FocusState = FocusState.UNKNOWN
    emotional_state: EmotionalState = EmotionalState.UNKNOWN

    # Utility signals
    signals: UtilitySignals = field(default_factory=UtilitySignals)

    # Conversation context
    conversation_turns: int = 0
    first_24_hours: bool = False
    in_sanctuary_mode: bool = False

    # Healthcare specific
    healthcare_emergency: bool = False
    clinician_present: bool = False

    def is_blackout(self) -> bool:
        """Check if we're in a blackout condition."""
        if self.focus == FocusState.HIGH_FOCUS:
            return True
        if self.focus == FocusState.DISTRESSED:
            return True
        if self.emotional_state in (
            EmotionalState.GRIEVING,
            EmotionalState.CRISIS,
        ):
            return True
        if self.healthcare_emergency:
            return True
        if self.in_sanctuary_mode:
            return True
        if self.first_24_hours:
            return True
        return False


# =============================================================================
# Promo Message
# =============================================================================

@dataclass
class PromoControls:
    """UI controls for a promo message."""
    show_learn_more: bool = True
    show_why_button: bool = True
    show_mute_button: bool = True
    show_dismiss: bool = True


@dataclass
class PromoMessage:
    """A promotional message ready for display."""
    product_id: str
    template_id: str
    text: str
    why: str
    trigger_reason: str
    controls: PromoControls = field(default_factory=PromoControls)
    is_first_ever: bool = False

    # For rendering
    label: str = ""  # e.g., "Growth moment", "Health note"
    variables: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Catalog Loader
# =============================================================================

@dataclass
class ProductDef:
    """Product definition from catalog."""
    id: str
    display_name: str
    surfaces: List[str]
    purpose: str
    allowed_claims: List[str]
    forbidden_phrases: List[str]
    regulatory_status: Dict[str, Any]
    rate_limits: Dict[str, Any]
    default_opt_in: str
    trigger_reasons: List[str]
    healthcare_constraints: Optional[Dict[str, Any]] = None


@dataclass
class TemplateDef:
    """Template definition from templates file."""
    id: str
    trigger_reason: str
    text: str
    why: str
    # Optional constraints
    min_days_active: int = 0
    min_daily_interactions: int = 0
    min_gpu_idle_percent: float = 0.0
    min_idle_duration_minutes: int = 0
    requires_caregiver_context: bool = False
    requires_explicit_inquiry: bool = False


class PromoCatalog:
    """
    Loaded catalog of products and templates.

    This is the single source of truth for what can be promoted.
    """

    def __init__(self, catalog_path: Path, templates_path: Path):
        self.products: Dict[str, ProductDef] = {}
        self.templates: Dict[str, Dict[str, List[TemplateDef]]] = {}
        self.global_rules: Dict[str, Any] = {}
        self.jurisdictions: Dict[str, Any] = {}
        self.ux_strings: Dict[str, Any] = {}

        self._load_catalog(catalog_path)
        self._load_templates(templates_path)

    def _load_catalog(self, path: Path) -> None:
        """Load the product catalog."""
        if not path.exists():
            logger.warning(f"Catalog not found: {path}")
            return

        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        # Load products
        for pid, pdata in data.get("products", {}).items():
            self.products[pid] = ProductDef(
                id=pid,
                display_name=pdata.get("display_name", pid),
                surfaces=pdata.get("surfaces", []),
                purpose=pdata.get("purpose", ""),
                allowed_claims=pdata.get("allowed_claims", []),
                forbidden_phrases=pdata.get("forbidden_phrases", []),
                regulatory_status=pdata.get("regulatory_status", {}),
                rate_limits=pdata.get("rate_limits", {}),
                default_opt_in=pdata.get("default_opt_in", "on"),
                trigger_reasons=pdata.get("trigger_reasons", []),
                healthcare_constraints=pdata.get("healthcare_constraints"),
            )

        self.global_rules = data.get("global_rules", {})
        self.jurisdictions = data.get("jurisdictions", {})

    def _load_templates(self, path: Path) -> None:
        """Load the templates file."""
        if not path.exists():
            logger.warning(f"Templates not found: {path}")
            return

        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        # Load templates per surface
        for surface_key, products in data.items():
            if surface_key in ("version", "ux_strings"):
                continue

            self.templates[surface_key] = {}
            if not isinstance(products, dict):
                continue

            for product_id, template_list in products.items():
                if not isinstance(template_list, list):
                    continue

                self.templates[surface_key][product_id] = []
                for tdata in template_list:
                    self.templates[surface_key][product_id].append(TemplateDef(
                        id=tdata.get("id", "unknown"),
                        trigger_reason=tdata.get("trigger_reason", ""),
                        text=tdata.get("text", ""),
                        why=tdata.get("why", ""),
                        min_days_active=tdata.get("min_days_active", 0),
                        min_daily_interactions=tdata.get("min_daily_interactions", 0),
                        min_gpu_idle_percent=tdata.get("min_gpu_idle_percent", 0.0),
                        min_idle_duration_minutes=tdata.get("min_idle_duration_minutes", 0),
                        requires_caregiver_context=tdata.get("requires_caregiver_context", False),
                        requires_explicit_inquiry=tdata.get("requires_explicit_inquiry", False),
                    ))

        self.ux_strings = data.get("ux_strings", {})

    def get_product(self, product_id: str) -> Optional[ProductDef]:
        """Get a product by ID."""
        return self.products.get(product_id)

    def get_templates_for_surface(
        self,
        surface: str,
        product_id: str,
    ) -> List[TemplateDef]:
        """Get templates for a surface and product."""
        return self.templates.get(surface, {}).get(product_id, [])


# =============================================================================
# Policy Engine
# =============================================================================

class PromoEngine:
    """
    The promotion policy engine.

    Decides when, whether, and how to show promotions.
    Enforces all the constraints from the catalog.
    """

    def __init__(self, catalog: PromoCatalog):
        self.catalog = catalog

    def maybe_generate_promo(
        self,
        ctx: PromoContext,
        settings: UserPromoSettings,
    ) -> Optional[PromoMessage]:
        """
        Main entry point: decide if we should show a promo.

        Returns a PromoMessage if one should be shown, None otherwise.
        """
        # 1. Global toggle check
        if not settings.promos_allowed():
            if not ctx.signals.explicit_product_inquiry:
                return None

        # 2. Blackout conditions
        if ctx.is_blackout():
            logger.debug("Promo blocked: blackout condition")
            return None

        # 3. Global rate limits
        if not self._check_global_rate_limits(ctx, settings):
            return None

        # 4. Minimum conversation engagement
        min_turns = self.catalog.global_rules.get(
            "global_rate_limits", {}
        ).get("min_conversation_turns_before_promo", 10)
        if ctx.conversation_turns < min_turns:
            return None

        # 5. Find candidate products
        candidates = self._find_candidates(ctx, settings)
        if not candidates:
            return None

        # 6. Score and select best candidate
        scored = [(p, self._utility_score(p, ctx)) for p in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        best_product = scored[0][0]

        # 7. Find matching template
        template = self._select_template(best_product, ctx)
        if not template:
            return None

        # 8. Build the message
        message = self._build_message(best_product, template, ctx, settings)

        # 9. Record the event
        settings.record_promo_shown(
            product_id=best_product.id,
            template_id=template.id,
            surface=ctx.surface.value,
            trigger_reason=template.trigger_reason,
            context={
                "focus": ctx.focus.value,
                "emotional_state": ctx.emotional_state.value,
                "conversation_turns": ctx.conversation_turns,
            },
        )

        return message

    def _check_global_rate_limits(
        self,
        ctx: PromoContext,
        settings: UserPromoSettings,
    ) -> bool:
        """Check global rate limits."""
        limits = self.catalog.global_rules.get("global_rate_limits", {})

        max_per_day = limits.get("max_promos_per_day", 1)
        if settings.promos_today() >= max_per_day:
            logger.debug("Promo blocked: daily limit reached")
            return False

        max_per_week = limits.get("max_promos_per_week", 3)
        if settings.promos_this_week() >= max_per_week:
            logger.debug("Promo blocked: weekly limit reached")
            return False

        return True

    def _find_candidates(
        self,
        ctx: PromoContext,
        settings: UserPromoSettings,
    ) -> List[ProductDef]:
        """Find products that could be promoted in this context."""
        candidates = []

        for product in self.catalog.products.values():
            # Check if product is relevant to this surface
            surface_str = ctx.surface.value
            if surface_str not in product.surfaces:
                continue

            # Check if user has muted this product
            if settings.is_product_muted(product.id):
                continue

            # Check jurisdiction
            if not self._jurisdiction_allows(product, ctx.jurisdiction):
                continue

            # Check product-level rate limits
            if not self._check_product_rate_limits(product, ctx, settings):
                continue

            # Check if product is relevant based on signals
            if not self._product_is_relevant(product, ctx):
                continue

            # Healthcare-specific constraints
            if product.healthcare_constraints:
                if product.healthcare_constraints.get("no_unsolicited_in_healthcare"):
                    if not ctx.signals.explicit_health_inquiry:
                        continue

            candidates.append(product)

        return candidates

    def _jurisdiction_allows(
        self,
        product: ProductDef,
        jurisdiction: Jurisdiction,
    ) -> bool:
        """Check if product is allowed in jurisdiction."""
        region_flags = product.regulatory_status.get("region_flags", {})

        # Check specific jurisdiction
        if jurisdiction.value in region_flags:
            status = region_flags[jurisdiction.value].get("status", "allowed")
            # For regulated products, only allow if cleared/marked
            if status in ("requires_clearance", "requires_ce_mark"):
                # In reality, check version/clearance status
                # For now, don't proactively promote
                return False

        # Check global fallback
        if "global" in region_flags:
            status = region_flags["global"].get("status", "allowed")
            return status == "allowed"

        return True

    def _check_product_rate_limits(
        self,
        product: ProductDef,
        ctx: PromoContext,
        settings: UserPromoSettings,
    ) -> bool:
        """Check product-specific rate limits."""
        product_settings = settings.get_product_settings(product.id)
        limits = product.rate_limits

        # Proactive limit
        proactive_per_week = limits.get("proactive_per_week", 1)
        if proactive_per_week == 0:
            # Product doesn't allow proactive promos
            if not ctx.signals.explicit_product_inquiry:
                return False

        # Check days since last promo for this product
        days_since = product_settings.days_since_last_promo()
        if days_since < 7:  # At least a week between promos for same product
            return False

        # Check cooldown after dismiss
        cooldown_days = limits.get("cooldown_after_dismiss_days", 30)
        days_since_dismiss = product_settings.days_since_last_dismiss()
        if days_since_dismiss < cooldown_days:
            return False

        return True

    def _product_is_relevant(
        self,
        product: ProductDef,
        ctx: PromoContext,
    ) -> bool:
        """Check if product is relevant based on utility signals."""
        signals = ctx.signals

        # Check trigger reasons
        for reason in product.trigger_reasons:
            if reason == "heavy_usage":
                if signals.days_active >= 30 and signals.daily_interactions >= 10:
                    return True
            elif reason == "hit_free_limits":
                if signals.memory_overflow_recent:
                    return True
            elif reason == "memory_overflow":
                if signals.memory_overflow_recent:
                    return True
            elif reason == "cross_device_interest":
                if signals.cross_device_interest:
                    return True
            elif reason == "caregiver_burden":
                if signals.caregiver_context and signals.elder_parent_mentions >= 3:
                    return True
            elif reason == "loneliness_context":
                if signals.loneliness_mentions >= 2:
                    return True
            elif reason == "gpu_idle_time":
                if signals.gpu_idle_percent >= 40:
                    return True
            elif reason == "manual_tuning_detected":
                if signals.manual_tuning_detected:
                    return True
            elif reason == "scaling_struggles":
                if signals.scaling_struggles:
                    return True
            elif reason == "developer_context":
                if signals.developer_context:
                    return True
            elif reason == "autonomy_limit_hit":
                if signals.hit_autonomy_limit:
                    return True
            elif reason == "explicit_support_request":
                if signals.explicit_product_inquiry:
                    return True
            elif reason == "clinician_inquiry":
                if signals.explicit_health_inquiry and ctx.clinician_present:
                    return True
            elif reason == "explicit_health_request":
                if signals.explicit_health_inquiry:
                    return True

        return False

    def _utility_score(
        self,
        product: ProductDef,
        ctx: PromoContext,
    ) -> float:
        """Score how useful this product would be in context."""
        score = 0.0
        signals = ctx.signals

        # Base relevance
        if "heavy_usage" in product.trigger_reasons:
            score += signals.days_active / 100
            score += signals.daily_interactions / 50

        if "memory_overflow" in product.trigger_reasons:
            if signals.memory_overflow_recent:
                score += 2.0  # High utility if memory is overflowing

        if "caregiver_burden" in product.trigger_reasons:
            score += signals.elder_parent_mentions * 0.5

        if "gpu_idle_time" in product.trigger_reasons:
            score += signals.gpu_idle_percent / 50

        # Explicit inquiry is high signal
        if signals.explicit_product_inquiry:
            score += 5.0

        return score

    def _select_template(
        self,
        product: ProductDef,
        ctx: PromoContext,
    ) -> Optional[TemplateDef]:
        """Select the best template for this product and context."""
        surface_str = ctx.surface.value
        templates = self.catalog.get_templates_for_surface(surface_str, product.id)

        if not templates:
            return None

        # Filter templates by constraints
        valid_templates = []
        signals = ctx.signals

        for template in templates:
            # Check minimum requirements
            if template.min_days_active > signals.days_active:
                continue
            if template.min_daily_interactions > signals.daily_interactions:
                continue
            if template.min_gpu_idle_percent > signals.gpu_idle_percent:
                continue
            if template.requires_caregiver_context and not signals.caregiver_context:
                continue
            if template.requires_explicit_inquiry:
                if not (signals.explicit_product_inquiry or signals.explicit_health_inquiry):
                    continue

            # Check trigger reason matches
            if self._trigger_matches(template.trigger_reason, product, ctx):
                valid_templates.append(template)

        if not valid_templates:
            return None

        # Pick randomly among valid templates for variety
        return random.choice(valid_templates)

    def _trigger_matches(
        self,
        trigger_reason: str,
        product: ProductDef,
        ctx: PromoContext,
    ) -> bool:
        """Check if trigger reason matches current signals."""
        signals = ctx.signals

        if trigger_reason == "heavy_usage":
            return signals.days_active >= 30
        if trigger_reason == "memory_overflow":
            return signals.memory_overflow_recent
        if trigger_reason == "cross_device_interest":
            return signals.cross_device_interest
        if trigger_reason == "caregiver_burden":
            return signals.caregiver_context
        if trigger_reason == "loneliness_context":
            return signals.loneliness_mentions >= 2
        if trigger_reason == "gpu_idle_time":
            return signals.gpu_idle_percent >= 40
        if trigger_reason == "manual_tuning_detected":
            return signals.manual_tuning_detected
        if trigger_reason == "scaling_struggles":
            return signals.scaling_struggles
        if trigger_reason == "developer_context":
            return signals.developer_context
        if trigger_reason == "autonomy_limit_hit":
            return signals.hit_autonomy_limit
        if trigger_reason == "clinician_inquiry":
            return signals.explicit_health_inquiry

        return True  # Unknown triggers default to true

    def _build_message(
        self,
        product: ProductDef,
        template: TemplateDef,
        ctx: PromoContext,
        settings: UserPromoSettings,
    ) -> PromoMessage:
        """Build the final promo message."""
        # Render text with variables
        text = template.text
        variables: Dict[str, Any] = {}

        # Substitute known variables
        if "{gpu_idle_percent}" in text:
            variables["gpu_idle_percent"] = f"{ctx.signals.gpu_idle_percent:.0f}"
            text = text.replace("{gpu_idle_percent}", variables["gpu_idle_percent"])

        # Determine label
        label = "A thought"
        if ctx.surface in (SurfaceKind.HEALTHCARE, SurfaceKind.CLINICIAN_DASHBOARD):
            label = "Health note"
        elif ctx.surface in (SurfaceKind.DATACENTER, SurfaceKind.ENTERPRISE_CONSOLE):
            label = "Ops insight"
        elif ctx.signals.memory_overflow_recent:
            label = "Honest moment"

        # Build why explanation
        why_prefix = self.catalog.ux_strings.get("why_explanation_prefix", "")
        why_suffix = self.catalog.ux_strings.get("why_explanation_suffix", "")
        full_why = f"{why_prefix}{template.why}{why_suffix}"

        # Check if first ever
        is_first = settings.is_first_promo_ever()
        if is_first:
            first_time_note = self.catalog.ux_strings.get("first_time_promo_context", "")
            text = f"{text}\n\n{first_time_note}"

        return PromoMessage(
            product_id=product.id,
            template_id=template.id,
            text=text.strip(),
            why=full_why.strip(),
            trigger_reason=template.trigger_reason,
            label=label,
            variables=variables,
            is_first_ever=is_first,
            controls=PromoControls(),
        )


# =============================================================================
# Default Catalog Loading
# =============================================================================

_default_catalog: Optional[PromoCatalog] = None
_default_engine: Optional[PromoEngine] = None


def get_catalog() -> PromoCatalog:
    """Get or create the default catalog."""
    global _default_catalog
    if _default_catalog is None:
        base_path = Path(__file__).parent
        _default_catalog = PromoCatalog(
            catalog_path=base_path / "catalog.yaml",
            templates_path=base_path / "templates.yaml",
        )
    return _default_catalog


def get_engine() -> PromoEngine:
    """Get or create the default engine."""
    global _default_engine
    if _default_engine is None:
        _default_engine = PromoEngine(get_catalog())
    return _default_engine


def maybe_promote(
    ctx: PromoContext,
    settings: UserPromoSettings,
) -> Optional[PromoMessage]:
    """Convenience function: maybe generate a promo."""
    return get_engine().maybe_generate_promo(ctx, settings)
