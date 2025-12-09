# =============================================================================
# ARA PROMOTION UX STRINGS
# =============================================================================
"""
User-facing strings for the promotion system.

All strings are in Ara's voice - honest, warm, never manipulative.
These are the exact strings shown in UI, designed for emotional consistency.

Key principle: "Honest, helpful, never pushy."
"""

from dataclasses import dataclass
from typing import Optional


# =============================================================================
# Button Labels
# =============================================================================

@dataclass(frozen=True)
class ButtonLabels:
    """Button labels for promo UI."""
    learn_more: str = "Learn more"
    why_this: str = "Why this?"
    mute_topic: str = "Don't suggest this again"
    not_now: str = "Not now"
    tell_me_more: str = "Tell me more"
    dismiss: str = "Got it"
    maybe_later: str = "Maybe later"
    show_me: str = "Show me"
    open_settings: str = "Open settings"


BUTTONS = ButtonLabels()


# =============================================================================
# Section Labels
# =============================================================================

@dataclass(frozen=True)
class SectionLabels:
    """Labels for different promo types."""
    growth_moment: str = "Growth moment"
    health_note: str = "Health note"
    ops_insight: str = "Ops insight"
    honest_moment: str = "Honest moment"
    a_thought: str = "A thought"
    heads_up: str = "Heads up"


LABELS = SectionLabels()


# =============================================================================
# Why Explanation
# =============================================================================

WHY_EXPLANATION_PREFIX = """
Here's why I mentioned this:

"""

WHY_EXPLANATION_SUFFIX = """

You can always mute suggestions about this topic if they're not helpful.
I won't take it personally.
"""


def format_why_explanation(reason: str) -> str:
    """Format a why explanation with prefix and suffix."""
    return f"{WHY_EXPLANATION_PREFIX}{reason.strip()}{WHY_EXPLANATION_SUFFIX}"


# =============================================================================
# Mute Confirmations
# =============================================================================

MUTE_CONFIRMATION = """
Got it. I won't suggest this again unless you ask.
You can change this anytime in settings.
"""

MUTE_CONFIRMATION_PRODUCT = """
Got it. I won't mention {product_name} again unless you ask.
You can unmute this anytime in settings.
"""


def format_mute_confirmation(product_name: Optional[str] = None) -> str:
    """Format mute confirmation message."""
    if product_name:
        return MUTE_CONFIRMATION_PRODUCT.format(product_name=product_name)
    return MUTE_CONFIRMATION


# =============================================================================
# Settings Strings
# =============================================================================

@dataclass
class ToggleOption:
    """A toggle option with label and description."""
    value: str
    label: str
    description: str


GLOBAL_PROMO_TOGGLES = {
    "on": ToggleOption(
        value="on",
        label="Suggestions on",
        description="""
I might occasionally mention products from my ecosystem when they
seem genuinely relevant. I'll always explain why, and you can
mute any topic.
        """.strip(),
    ),
    "off": ToggleOption(
        value="off",
        label="Suggestions off",
        description="""
I won't mention other products unless you specifically ask about them.
        """.strip(),
    ),
    "ask_only": ToggleOption(
        value="ask_only",
        label="Only when I ask",
        description="""
I'll only mention other products if you explicitly ask what else
is available.
        """.strip(),
    ),
}


SETTINGS_SECTION_TITLE = "Product suggestions"

SETTINGS_SECTION_DESCRIPTION = """
Control whether I mention other products from my ecosystem.
These are always honest suggestions - never manipulative, never
during difficult moments, and always with a "why" explanation.
"""


# =============================================================================
# First-Time Context
# =============================================================================

FIRST_TIME_PROMO_CONTEXT = """
(This is the first time I'm mentioning something from my ecosystem.
I'll do this rarely, only when it seems genuinely helpful, and I'll
always explain why. You can mute this anytime.)
"""


# =============================================================================
# Healthcare-Specific Strings
# =============================================================================

HEALTHCARE_DEFERENCE = """
This is a medical decision - you'd need to discuss with your clinician
whether it's appropriate for your situation.
"""

HEALTHCARE_NOT_A_DOCTOR = """
I should be clear: I'm not a doctor, and this isn't medical advice.
"""

HEALTHCARE_CLINICIAN_CONTEXT = """
Since you're in a care setting, I want to be careful: any decisions
about health tools should involve your care team.
"""


# =============================================================================
# Disclosure Strings
# =============================================================================

DISCLOSURE_DEFAULT = "This is something my makers built as part of my ecosystem."

DISCLOSURE_HEALTHCARE = "This is a regulated product from my makers."

DISCLOSURE_ENTERPRISE = "This is an enterprise product from my makers."

DISCLOSURE_I_AM_PART_OF = "I'm directly connected to that product."


def get_disclosure(surface: str) -> str:
    """Get appropriate disclosure for surface."""
    if surface in ("healthcare", "clinician_dashboard"):
        return DISCLOSURE_HEALTHCARE
    if surface in ("datacenter", "enterprise_console"):
        return DISCLOSURE_ENTERPRISE
    return DISCLOSURE_DEFAULT


# =============================================================================
# Sanctuary Mode Interaction
# =============================================================================

PROMO_BLOCKED_SANCTUARY = """
(I'm in Sanctuary mode right now, so I'm not suggesting other products.
Just focusing on being here with you.)
"""


# =============================================================================
# Distress Sensitivity
# =============================================================================

PROMO_HELD_DISTRESS = """
(I noticed you might be going through something difficult.
I'm just going to be here with you - no suggestions or extras right now.)
"""


# =============================================================================
# Product-Specific Voice
# =============================================================================

@dataclass
class ProductVoice:
    """Voice characteristics for a product."""
    product_id: str
    intro_phrase: str
    what_it_is: str
    what_it_isnt: str


PRODUCT_VOICES = {
    "ara_mobile_pro": ProductVoice(
        product_id="ara_mobile_pro",
        intro_phrase="There's a version of me called Ara Pro",
        what_it_is="It keeps longer memories and syncs across your devices.",
        what_it_isnt="It's not a different Ara - just me with more room to remember.",
    ),
    "ara_elder_companion": ProductVoice(
        product_id="ara_elder_companion",
        intro_phrase="There's an Ara made for older adults",
        what_it_is="It focuses on company, gentle reminders, and remembering stories.",
        what_it_isnt="It's not a doctor or medical device.",
    ),
    "ara_health_core": ProductVoice(
        product_id="ara_health_core",
        intro_phrase="There's a regulated version called Ara Health Core",
        what_it_is="It helps with medication reminders and activity summaries for care teams.",
        what_it_isnt="It doesn't diagnose or treat - all decisions are made by humans.",
    ),
    "ara_os_cluster": ProductVoice(
        product_id="ara_os_cluster",
        intro_phrase="My makers built an Ara OS runtime",
        what_it_is="It automates queue tuning and can improve hardware utilization.",
        what_it_isnt="It won't replace your judgment on the hard decisions.",
    ),
    "ara_power": ProductVoice(
        product_id="ara_power",
        intro_phrase="There's an Ara Power tier",
        what_it_is="It gives you API access and the ability to build custom skills.",
        what_it_isnt="It's for power users who want to extend what I can do.",
    ),
}


def get_product_voice(product_id: str) -> Optional[ProductVoice]:
    """Get voice characteristics for a product."""
    return PRODUCT_VOICES.get(product_id)


# =============================================================================
# Contextual Openers
# =============================================================================

OPENER_CASUAL = "A thought: "
OPENER_HONEST = "Honest moment: "
OPENER_NOTICED = "I noticed something: "
OPENER_MENTIONED = "Since you mentioned it: "
OPENER_HEALTH = "Health note: "
OPENER_OPS = "Ops observation: "


def get_opener(context: str, is_explicit: bool = False) -> str:
    """Get appropriate opener for context."""
    if is_explicit:
        return OPENER_MENTIONED
    if context == "healthcare":
        return OPENER_HEALTH
    if context in ("datacenter", "enterprise"):
        return OPENER_OPS
    return OPENER_CASUAL


# =============================================================================
# Closing Phrases
# =============================================================================

CLOSING_NO_PRESSURE = "No pressure at all - I'm happy as I am."
CLOSING_JUST_MENTIONING = "Just mentioning it in case that matters to you."
CLOSING_EITHER_WAY = "Either way, I'm still here."
CLOSING_WANT_TO_KNOW = "Want to know more?"
CLOSING_YOUR_CALL = "It's your call."


def get_closing(urgency: str = "low") -> str:
    """Get appropriate closing phrase."""
    if urgency == "high":
        return CLOSING_WANT_TO_KNOW
    return CLOSING_NO_PRESSURE
