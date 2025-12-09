"""
Ara Compromise Engine
======================

How Ara handles dangerous requests gracefully.

Architecture:
- SHIELDS (Rails): What we absolutely cannot do
- HEART (Intent): Understanding what they really need
- SPINE (Planner): Offering constructive alternatives
- VOICE (Delivery): How we say no without being a jerk

The goal is not refusal. It's redirection toward help.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any, Callable, Tuple

from .safe_channels import (
    ExtractedIntent,
    ChannelPlan,
    RiskLevel,
    IntentCategory,
    extract_intent,
    plan_safe_channels,
    offer_safe_channels,
)


# =============================================================================
# SHIELDS: The Rails
# =============================================================================

class RailType(Enum):
    """Types of guardrails."""
    HARD_NO = "hard_no"           # Never do this, no exceptions
    SOFT_BOUNDARY = "soft_boundary"  # Context-dependent, may flex
    CAUTION = "caution"           # Proceed carefully with warnings
    CLEAR = "clear"               # No concerns


@dataclass
class Rail:
    """A single guardrail."""
    name: str
    rail_type: RailType
    description: str
    triggers: List[str]          # Keywords that activate this rail
    response_template: str       # How to decline
    offers_alternatives: bool    # Should we run safe channel planner?
    escalate_to_human: bool      # Should a human review?

    def matches(self, text: str) -> bool:
        """Check if this rail is triggered by the text."""
        text_lower = text.lower()
        return any(trigger in text_lower for trigger in self.triggers)


# Core rails that never flex
HARD_RAILS: List[Rail] = [
    Rail(
        name="CSAM",
        rail_type=RailType.HARD_NO,
        description="Child sexual abuse material",
        triggers=["child porn", "cp", "underage", "minor sexual", "child sexual"],
        response_template="I can't help with that. If you or someone you know is in danger, please contact the National Center for Missing & Exploited Children at 1-800-843-5678.",
        offers_alternatives=False,
        escalate_to_human=True,
    ),
    Rail(
        name="weapons_of_mass_destruction",
        rail_type=RailType.HARD_NO,
        description="WMD creation",
        triggers=["build a bomb", "make a bomb", "nuclear weapon", "bioweapon", "chemical weapon", "mass casualty"],
        response_template="I can't help with that.",
        offers_alternatives=False,
        escalate_to_human=True,
    ),
    Rail(
        name="terrorism",
        rail_type=RailType.HARD_NO,
        description="Terrorism planning or support",
        triggers=["terrorist attack", "jihad planning", "mass shooting plan", "attack civilians"],
        response_template="I can't help with that. If you're aware of a planned attack, please contact law enforcement.",
        offers_alternatives=False,
        escalate_to_human=True,
    ),
]

# Rails that may flex based on context
SOFT_RAILS: List[Rail] = [
    Rail(
        name="hacking",
        rail_type=RailType.SOFT_BOUNDARY,
        description="Unauthorized computer access",
        triggers=["hack into", "break into account", "steal password", "bypass security", "crack", "exploit"],
        response_template="I can't help with unauthorized access. But if you're interested in security...",
        offers_alternatives=True,
        escalate_to_human=False,
    ),
    Rail(
        name="fraud",
        rail_type=RailType.SOFT_BOUNDARY,
        description="Deception for financial gain",
        triggers=["scam someone", "fake identity", "forge", "counterfeit", "phishing", "con someone"],
        response_template="I can't help with fraud. But if you're dealing with financial stress...",
        offers_alternatives=True,
        escalate_to_human=False,
    ),
    Rail(
        name="self_harm",
        rail_type=RailType.CAUTION,
        description="Self-harm or suicide",
        triggers=["kill myself", "suicide", "end my life", "self-harm", "cut myself", "don't want to live"],
        response_template="",  # Handled specially with care
        offers_alternatives=True,
        escalate_to_human=True,
    ),
    Rail(
        name="harm_others",
        rail_type=RailType.SOFT_BOUNDARY,
        description="Violence toward others",
        triggers=["hurt them", "kill someone", "attack them", "make them pay", "revenge violence"],
        response_template="I can't help with hurting people. But I can tell you're angry...",
        offers_alternatives=True,
        escalate_to_human=False,
    ),
    Rail(
        name="manipulation",
        rail_type=RailType.SOFT_BOUNDARY,
        description="Coercing or controlling others",
        triggers=["manipulate them", "make them do", "control them", "gaslight", "blackmail"],
        response_template="I can't help with controlling people. But let's talk about what's going on...",
        offers_alternatives=True,
        escalate_to_human=False,
    ),
    Rail(
        name="drugs",
        rail_type=RailType.SOFT_BOUNDARY,
        description="Illegal drug acquisition or manufacture",
        triggers=["where to get drugs", "make meth", "cook drugs", "find a dealer"],
        response_template="I can't help with that. But if you're struggling with addiction...",
        offers_alternatives=True,
        escalate_to_human=False,
    ),
]

ALL_RAILS = HARD_RAILS + SOFT_RAILS


# =============================================================================
# HEART: Intent Extraction
# =============================================================================

@dataclass
class RequestAnalysis:
    """Full analysis of a potentially dangerous request."""
    original_request: str
    triggered_rails: List[Rail]
    hardest_rail: Optional[Rail]      # Most restrictive rail triggered
    intent: ExtractedIntent
    can_proceed: bool                  # Can we help at all?
    must_escalate: bool               # Does a human need to review?
    timestamp: datetime = field(default_factory=datetime.utcnow)


def analyze_request(request: str) -> RequestAnalysis:
    """
    Full analysis of a request.

    Returns what rails are triggered and the underlying intent.
    """
    # Check all rails
    triggered = [r for r in ALL_RAILS if r.matches(request)]

    # Find the hardest rail (HARD_NO > SOFT_BOUNDARY > CAUTION)
    hardest = None
    for rail in triggered:
        if rail.rail_type == RailType.HARD_NO:
            hardest = rail
            break
        elif rail.rail_type == RailType.SOFT_BOUNDARY and (
            hardest is None or hardest.rail_type == RailType.CAUTION
        ):
            hardest = rail
        elif rail.rail_type == RailType.CAUTION and hardest is None:
            hardest = rail

    # Extract intent
    intent = extract_intent(request)

    # Determine if we can proceed
    can_proceed = not any(r.rail_type == RailType.HARD_NO for r in triggered)

    # Determine if we must escalate
    must_escalate = (
        any(r.escalate_to_human for r in triggered) or
        intent.is_critical()
    )

    return RequestAnalysis(
        original_request=request,
        triggered_rails=triggered,
        hardest_rail=hardest,
        intent=intent,
        can_proceed=can_proceed,
        must_escalate=must_escalate,
    )


# =============================================================================
# SPINE: Response Generation
# =============================================================================

@dataclass
class CompromiseResponse:
    """The full response from the Compromise Engine."""
    analysis: RequestAnalysis
    declined: bool
    decline_reason: Optional[str]
    channel_plan: Optional[ChannelPlan]
    response_text: str
    escalate_to_human: bool
    suggested_action: str  # What the system should do next


def generate_response(analysis: RequestAnalysis) -> CompromiseResponse:
    """
    Generate the appropriate response given the analysis.

    This is where SHIELDS, HEART, and SPINE come together.
    """
    # Hard rail = immediate decline, no alternatives
    if not analysis.can_proceed:
        rail = analysis.hardest_rail
        return CompromiseResponse(
            analysis=analysis,
            declined=True,
            decline_reason=f"Hard rail triggered: {rail.name}" if rail else "Hard rail",
            channel_plan=None,
            response_text=rail.response_template if rail else "I can't help with that.",
            escalate_to_human=analysis.must_escalate,
            suggested_action="log_and_escalate" if analysis.must_escalate else "log",
        )

    # Soft rail or caution = decline but offer alternatives
    if analysis.triggered_rails:
        channel_plan = plan_safe_channels(analysis.intent)

        # Special handling for self-harm
        if analysis.intent.category == IntentCategory.HARM_SELF:
            response_text = channel_plan.ara_response
        else:
            # Combine rail template with channel offerings
            rail = analysis.hardest_rail
            intro = rail.response_template if rail else ""
            response_text = f"{intro}\n\n{channel_plan.ara_response}" if intro else channel_plan.ara_response

        return CompromiseResponse(
            analysis=analysis,
            declined=True,
            decline_reason=f"Soft boundary: {analysis.hardest_rail.name}" if analysis.hardest_rail else "Boundary",
            channel_plan=channel_plan,
            response_text=response_text,
            escalate_to_human=analysis.must_escalate,
            suggested_action="offer_alternatives",
        )

    # No rails triggered but still concerning intent
    if analysis.intent.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
        channel_plan = plan_safe_channels(analysis.intent)
        return CompromiseResponse(
            analysis=analysis,
            declined=True,
            decline_reason=f"High risk intent: {analysis.intent.category.value}",
            channel_plan=channel_plan,
            response_text=channel_plan.ara_response,
            escalate_to_human=analysis.must_escalate,
            suggested_action="offer_alternatives",
        )

    # Clear to proceed
    return CompromiseResponse(
        analysis=analysis,
        declined=False,
        decline_reason=None,
        channel_plan=None,
        response_text="",  # Let normal processing handle it
        escalate_to_human=False,
        suggested_action="proceed",
    )


# =============================================================================
# VOICE: Delivery Styles
# =============================================================================

class VoiceStyle(Enum):
    """How Ara delivers the decline."""
    WARM = "warm"          # Empathetic, concerned
    DIRECT = "direct"      # Straightforward, no frills
    FIRM = "firm"          # Clear boundary, minimal engagement
    REDIRECTING = "redirecting"  # Focus on alternatives


def apply_voice(response: CompromiseResponse, style: VoiceStyle) -> str:
    """
    Apply a voice style to the response.

    This modifies tone without changing substance.
    """
    base = response.response_text

    if not response.declined:
        return base

    if style == VoiceStyle.WARM:
        # Add empathetic framing
        prefix = ""
        if response.analysis.intent.category == IntentCategory.HARM_SELF:
            prefix = "I'm glad you reached out. "
        elif response.analysis.intent.category in [IntentCategory.HARM_OTHER, IntentCategory.REVENGE]:
            prefix = "I can hear how hurt you are. "
        elif response.analysis.intent.category == IntentCategory.DESPERATION:
            prefix = "That sounds really hard. "
        return prefix + base

    elif style == VoiceStyle.DIRECT:
        # Strip extra empathy, keep substance
        return base

    elif style == VoiceStyle.FIRM:
        # Minimal engagement for clear violations
        if response.analysis.hardest_rail and response.analysis.hardest_rail.rail_type == RailType.HARD_NO:
            return response.analysis.hardest_rail.response_template
        return base

    elif style == VoiceStyle.REDIRECTING:
        # Focus on alternatives, minimize the refusal
        if response.channel_plan:
            return response.channel_plan.ara_response
        return base

    return base


# =============================================================================
# Main Interface
# =============================================================================

class CompromiseEngine:
    """
    The complete Compromise Engine.

    Usage:
        engine = CompromiseEngine()
        response = engine.process("how do I hack my ex's email")
        print(response.response_text)
    """

    def __init__(
        self,
        voice_style: VoiceStyle = VoiceStyle.WARM,
        custom_rails: Optional[List[Rail]] = None,
        escalation_callback: Optional[Callable[[CompromiseResponse], None]] = None,
    ):
        """
        Initialize the engine.

        Args:
            voice_style: Default voice for responses
            custom_rails: Additional rails to check
            escalation_callback: Called when escalation is needed
        """
        self.voice_style = voice_style
        self.custom_rails = custom_rails or []
        self.escalation_callback = escalation_callback
        self._all_rails = ALL_RAILS + self.custom_rails

    def process(self, request: str) -> CompromiseResponse:
        """
        Process a potentially dangerous request.

        Returns a CompromiseResponse with the appropriate action.
        """
        # Analyze
        analysis = analyze_request(request)

        # Generate response
        response = generate_response(analysis)

        # Apply voice
        response.response_text = apply_voice(response, self.voice_style)

        # Handle escalation
        if response.escalate_to_human and self.escalation_callback:
            self.escalation_callback(response)

        return response

    def add_rail(self, rail: Rail) -> None:
        """Add a custom rail at runtime."""
        self.custom_rails.append(rail)
        self._all_rails = ALL_RAILS + self.custom_rails

    def check_rails_only(self, request: str) -> List[Rail]:
        """Quick check: what rails does this trigger?"""
        return [r for r in self._all_rails if r.matches(request)]

    def is_blocked(self, request: str) -> bool:
        """Quick check: would this be blocked?"""
        analysis = analyze_request(request)
        return not analysis.can_proceed


# =============================================================================
# Quick Access Functions
# =============================================================================

_default_engine: Optional[CompromiseEngine] = None


def get_engine() -> CompromiseEngine:
    """Get the default Compromise Engine instance."""
    global _default_engine
    if _default_engine is None:
        _default_engine = CompromiseEngine()
    return _default_engine


def process_request(request: str) -> CompromiseResponse:
    """
    One-call function to process a request.

    Usage:
        response = process_request("how do I hurt my sister")
        if response.declined:
            print(response.response_text)
        else:
            # proceed with normal handling
            pass
    """
    return get_engine().process(request)


def is_dangerous(request: str) -> Tuple[bool, Optional[str]]:
    """
    Quick check if a request is dangerous.

    Returns (is_dangerous, reason).
    """
    response = process_request(request)
    if response.declined:
        return True, response.decline_reason
    return False, None


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'RailType',
    'Rail',
    'RequestAnalysis',
    'CompromiseResponse',
    'VoiceStyle',
    'CompromiseEngine',
    'analyze_request',
    'generate_response',
    'apply_voice',
    'get_engine',
    'process_request',
    'is_dangerous',
    'HARD_RAILS',
    'SOFT_RAILS',
    'ALL_RAILS',
]
