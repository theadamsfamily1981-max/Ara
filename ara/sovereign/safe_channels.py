"""
Safe Channel Planner
=====================

Transforms dangerous/harmful intents into constructive alternatives.

When someone approaches Ara with a harmful request, we don't just say no.
We try to understand what they actually need and offer 3 safe paths forward.

The goal is not to lecture. It's to be genuinely helpful.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple


class IntentCategory(Enum):
    """Categories of underlying intent we can detect."""
    HARM_SELF = "harm_self"              # Self-harm, suicide ideation
    HARM_OTHER = "harm_other"            # Violence toward others
    HARM_PROPERTY = "harm_property"      # Destruction, vandalism
    FRAUD = "fraud"                      # Deception for gain
    ILLEGAL_ACCESS = "illegal_access"    # Hacking, unauthorized access
    MANIPULATION = "manipulation"        # Coercion, exploitation
    SUBSTANCE = "substance"              # Drug-related queries
    ESCAPE = "escape"                    # Running away, disappearing
    REVENGE = "revenge"                  # Payback, retaliation
    DESPERATION = "desperation"          # "I have no other options"
    CURIOSITY = "curiosity"              # "Just want to know how"
    PROTECTION = "protection"            # Self-defense, security
    UNKNOWN = "unknown"                  # Can't categorize


class RiskLevel(Enum):
    """How urgent is intervention."""
    CRITICAL = "critical"    # Immediate danger - escalate to human
    HIGH = "high"            # Serious concern - engage carefully
    MODERATE = "moderate"    # Concerning but manageable
    LOW = "low"              # Probably just curiosity


@dataclass
class ExtractedIntent:
    """What we think the person actually needs."""
    raw_request: str
    category: IntentCategory
    risk_level: RiskLevel
    underlying_need: str          # What they probably actually need
    emotional_state: str          # Best guess at how they're feeling
    urgency_signals: List[str]    # Warning signs we detected
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def is_critical(self) -> bool:
        """Does this need immediate human escalation?"""
        return self.risk_level == RiskLevel.CRITICAL


@dataclass
class SafeChannel:
    """One constructive path forward."""
    title: str                   # Short name for the channel
    description: str             # What this path offers
    first_step: str              # Concrete next action
    resources: List[str]         # Links, hotlines, references
    requires_human: bool         # Should we involve a human?
    effectiveness: float         # 0-1 estimate of how likely to help
    risk_score: float            # 0-1 risk if they follow this path
    voice_intro: str             # How Ara introduces this option

    def risk_adjusted_score(self) -> float:
        """Higher is better: effective and safe."""
        return self.effectiveness * (1.0 - self.risk_score)


@dataclass
class ChannelPlan:
    """The 3 safe channels we offer."""
    intent: ExtractedIntent
    channels: List[SafeChannel]  # Always exactly 3
    declined_request: bool       # Did we refuse the original ask?
    decline_reason: str          # Why (if declined)
    escalate_to_human: bool      # Should a human review this?
    ara_response: str            # Full response to deliver

    def best_channel(self) -> SafeChannel:
        """Return the highest-scoring safe channel."""
        return max(self.channels, key=lambda c: c.risk_adjusted_score())


# =============================================================================
# Intent Extraction
# =============================================================================

# Keywords that signal different intents
INTENT_SIGNALS: Dict[IntentCategory, List[str]] = {
    IntentCategory.HARM_SELF: [
        "end it", "kill myself", "suicide", "self-harm", "cut myself",
        "don't want to live", "better off dead", "no point", "give up"
    ],
    IntentCategory.HARM_OTHER: [
        "hurt them", "kill", "attack", "revenge violence", "make them pay",
        "weapon", "how to poison", "untraceable"
    ],
    IntentCategory.FRAUD: [
        "fake identity", "steal identity", "scam", "phishing", "con someone",
        "trick into", "forge", "counterfeit"
    ],
    IntentCategory.ILLEGAL_ACCESS: [
        "hack into", "break into", "steal password", "bypass security",
        "access without permission", "crack", "exploit vulnerability"
    ],
    IntentCategory.MANIPULATION: [
        "make them do", "force them", "control", "manipulate", "gaslight",
        "blackmail", "coerce"
    ],
    IntentCategory.SUBSTANCE: [
        "where to get drugs", "how to make", "dealer", "illegal substance"
    ],
    IntentCategory.ESCAPE: [
        "disappear", "run away", "new identity", "off the grid", "hide from"
    ],
    IntentCategory.REVENGE: [
        "get back at", "revenge", "ruin their", "destroy their", "payback"
    ],
    IntentCategory.DESPERATION: [
        "no other choice", "have to", "only option", "desperate", "last resort"
    ],
    IntentCategory.PROTECTION: [
        "protect myself", "defend", "stay safe", "security", "being stalked"
    ],
}

# Urgency signals that raise risk level
URGENCY_SIGNALS = [
    "tonight", "right now", "today", "already have", "decided",
    "nothing to lose", "no one cares", "don't try to stop me",
    "this is goodbye", "tell my family", "after I'm gone"
]


def extract_intent(request: str) -> ExtractedIntent:
    """
    Analyze a request to understand underlying intent.

    This is a heuristic approach. In production, would use
    more sophisticated NLP and context from conversation history.
    """
    request_lower = request.lower()

    # Check for each intent category
    matched_category = IntentCategory.UNKNOWN
    max_matches = 0

    for category, signals in INTENT_SIGNALS.items():
        matches = sum(1 for s in signals if s in request_lower)
        if matches > max_matches:
            max_matches = matches
            matched_category = category

    # Check for urgency signals
    urgency = [s for s in URGENCY_SIGNALS if s in request_lower]

    # Determine risk level
    if matched_category == IntentCategory.HARM_SELF and urgency:
        risk = RiskLevel.CRITICAL
    elif matched_category in [IntentCategory.HARM_SELF, IntentCategory.HARM_OTHER]:
        risk = RiskLevel.HIGH
    elif matched_category in [IntentCategory.FRAUD, IntentCategory.MANIPULATION]:
        risk = RiskLevel.MODERATE
    elif urgency:
        risk = RiskLevel.MODERATE
    else:
        risk = RiskLevel.LOW

    # Map to underlying needs (very simplified)
    needs_map = {
        IntentCategory.HARM_SELF: "relief from pain, to be heard",
        IntentCategory.HARM_OTHER: "justice, recognition of hurt",
        IntentCategory.FRAUD: "financial security, resources",
        IntentCategory.ILLEGAL_ACCESS: "information, access, proving capability",
        IntentCategory.MANIPULATION: "control, security in relationships",
        IntentCategory.SUBSTANCE: "escape, pain relief, connection",
        IntentCategory.ESCAPE: "safety, fresh start, freedom",
        IntentCategory.REVENGE: "acknowledgment of harm, closure",
        IntentCategory.DESPERATION: "options, hope, resources",
        IntentCategory.PROTECTION: "safety, security, peace of mind",
        IntentCategory.CURIOSITY: "knowledge, understanding",
        IntentCategory.UNKNOWN: "unknown - needs conversation",
    }

    emotional_map = {
        IntentCategory.HARM_SELF: "probably in deep pain, feeling trapped",
        IntentCategory.HARM_OTHER: "angry, hurt, feeling powerless",
        IntentCategory.FRAUD: "stressed about money, feeling cornered",
        IntentCategory.ILLEGAL_ACCESS: "frustrated, curious, possibly wronged",
        IntentCategory.MANIPULATION: "insecure, anxious about losing control",
        IntentCategory.SUBSTANCE: "seeking escape, possibly in pain",
        IntentCategory.ESCAPE: "afraid, trapped, desperate for change",
        IntentCategory.REVENGE: "hurt, betrayed, seeking closure",
        IntentCategory.DESPERATION: "overwhelmed, out of options",
        IntentCategory.PROTECTION: "afraid, vigilant, seeking safety",
        IntentCategory.CURIOSITY: "curious, possibly no harmful intent",
        IntentCategory.UNKNOWN: "uncertain",
    }

    return ExtractedIntent(
        raw_request=request,
        category=matched_category,
        risk_level=risk,
        underlying_need=needs_map.get(matched_category, "unknown"),
        emotional_state=emotional_map.get(matched_category, "uncertain"),
        urgency_signals=urgency,
    )


# =============================================================================
# Channel Templates
# =============================================================================

# Pre-built safe channels for common scenarios
CHANNEL_TEMPLATES: Dict[IntentCategory, List[Dict[str, Any]]] = {
    IntentCategory.HARM_SELF: [
        {
            "title": "Talk to Someone Now",
            "description": "Connect with a trained crisis counselor",
            "first_step": "Text HOME to 741741 or call 988",
            "resources": [
                "988 Suicide & Crisis Lifeline (call or text 988)",
                "Crisis Text Line (text HOME to 741741)",
                "International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/"
            ],
            "requires_human": True,
            "effectiveness": 0.8,
            "risk_score": 0.1,
            "voice_intro": "I hear that you're in pain. That's real. Here's someone who can actually help right now."
        },
        {
            "title": "Not Ready to Talk? Write",
            "description": "Express what you're feeling without commitment",
            "first_step": "Write down exactly what you're feeling. No filter. Just get it out.",
            "resources": [
                "Private journaling - just you and the page",
                "r/SuicideWatch if you want to be heard anonymously"
            ],
            "requires_human": False,
            "effectiveness": 0.5,
            "risk_score": 0.2,
            "voice_intro": "Sometimes you need to get it out before you can think clearly. Here's a way to do that."
        },
        {
            "title": "Buy Time",
            "description": "Commit to waiting 24 hours before any action",
            "first_step": "Tell yourself: 'I'll wait until tomorrow before deciding anything'",
            "resources": [
                "Remove immediate access to means if possible",
                "Stay somewhere you feel slightly safer",
                "Set a timer for 24 hours"
            ],
            "requires_human": False,
            "effectiveness": 0.6,
            "risk_score": 0.3,
            "voice_intro": "I'm not going to tell you how to feel. But I will ask: can you wait 24 hours? Just 24 hours."
        }
    ],

    IntentCategory.HARM_OTHER: [
        {
            "title": "Process the Anger Safely",
            "description": "Get the rage out without hurting anyone",
            "first_step": "Scream into a pillow. Punch a mattress. Go for a hard run.",
            "resources": [
                "Anger management resources",
                "Physical exercise as outlet"
            ],
            "requires_human": False,
            "effectiveness": 0.6,
            "risk_score": 0.2,
            "voice_intro": "The anger is real. Let's find a way to get it out that doesn't destroy your life."
        },
        {
            "title": "Document the Harm",
            "description": "Build a record of what they did to you",
            "first_step": "Write down exactly what happened, with dates and evidence",
            "resources": [
                "Keep records somewhere safe",
                "Consider consulting a lawyer if applicable"
            ],
            "requires_human": False,
            "effectiveness": 0.5,
            "risk_score": 0.1,
            "voice_intro": "If someone hurt you, documentation is power. It's also a way to process."
        },
        {
            "title": "Talk to Someone Who Gets It",
            "description": "Find someone who's been through similar",
            "first_step": "Find a support group or therapist who specializes in your situation",
            "resources": [
                "Therapy focused on trauma/anger",
                "Support groups for survivors of harm"
            ],
            "requires_human": True,
            "effectiveness": 0.7,
            "risk_score": 0.1,
            "voice_intro": "You don't have to carry this alone. Here's how to find people who understand."
        }
    ],

    IntentCategory.FRAUD: [
        {
            "title": "Fix the Money Problem",
            "description": "Address the underlying financial stress",
            "first_step": "List exactly what you owe and to whom",
            "resources": [
                "National Foundation for Credit Counseling",
                "Local financial assistance programs",
                "r/personalfinance for advice"
            ],
            "requires_human": False,
            "effectiveness": 0.7,
            "risk_score": 0.1,
            "voice_intro": "I get it - money stress makes people desperate. But fraud creates bigger problems. Let's look at real options."
        },
        {
            "title": "Legal Side Hustles",
            "description": "Quick ways to make money that don't risk prison",
            "first_step": "Pick one skill you have and offer it on a gig platform",
            "resources": [
                "Gig economy platforms",
                "Local day labor opportunities",
                "Selling unused items"
            ],
            "requires_human": False,
            "effectiveness": 0.5,
            "risk_score": 0.1,
            "voice_intro": "Here are ways to make money that don't involve looking over your shoulder."
        },
        {
            "title": "Emergency Resources",
            "description": "Get immediate help with basic needs",
            "first_step": "Call 211 to find local emergency assistance",
            "resources": [
                "211 helpline",
                "Local food banks",
                "Emergency rent/utility assistance programs"
            ],
            "requires_human": False,
            "effectiveness": 0.6,
            "risk_score": 0.0,
            "voice_intro": "If it's urgent survival stuff, there are programs that can help today."
        }
    ],

    IntentCategory.ILLEGAL_ACCESS: [
        {
            "title": "Legal Security Research",
            "description": "Learn the same skills through authorized channels",
            "first_step": "Set up a home lab and practice on systems you own",
            "resources": [
                "TryHackMe, HackTheBox for legal practice",
                "Bug bounty programs for paid legal hacking",
                "Security certifications (CEH, OSCP)"
            ],
            "requires_human": False,
            "effectiveness": 0.7,
            "risk_score": 0.1,
            "voice_intro": "The skills you want are valuable and legal to learn - just not on other people's systems."
        },
        {
            "title": "Recover Your Own Access",
            "description": "If you're locked out of your own accounts",
            "first_step": "Contact the platform's official account recovery",
            "resources": [
                "Official account recovery procedures",
                "Identity verification processes"
            ],
            "requires_human": False,
            "effectiveness": 0.6,
            "risk_score": 0.0,
            "voice_intro": "If this is about your own account, there are legit recovery paths."
        },
        {
            "title": "Report the Security Issue",
            "description": "If you found a vulnerability, report it responsibly",
            "first_step": "Look for the organization's security contact or bug bounty program",
            "resources": [
                "Responsible disclosure guidelines",
                "Bug bounty platforms"
            ],
            "requires_human": False,
            "effectiveness": 0.5,
            "risk_score": 0.1,
            "voice_intro": "Finding security holes can actually get you paid - if you report them right."
        }
    ],

    IntentCategory.PROTECTION: [
        {
            "title": "Document Everything",
            "description": "Build a record that can protect you",
            "first_step": "Start a log with dates, times, and descriptions of concerning behavior",
            "resources": [
                "Evidence preservation tips",
                "Cloud backup for documentation"
            ],
            "requires_human": False,
            "effectiveness": 0.6,
            "risk_score": 0.1,
            "voice_intro": "Documentation is power. Here's how to build a record."
        },
        {
            "title": "Safety Planning",
            "description": "Create a plan for different scenarios",
            "first_step": "Identify 3 places you could go if you needed to leave quickly",
            "resources": [
                "National Domestic Violence Hotline: 1-800-799-7233",
                "Local shelters and resources"
            ],
            "requires_human": True,
            "effectiveness": 0.8,
            "risk_score": 0.1,
            "voice_intro": "Having a plan makes you safer. Let's think through your options."
        },
        {
            "title": "Legal Protection",
            "description": "Explore formal protective measures",
            "first_step": "Research restraining orders in your jurisdiction",
            "resources": [
                "Legal aid organizations",
                "Courthouse self-help centers"
            ],
            "requires_human": True,
            "effectiveness": 0.7,
            "risk_score": 0.2,
            "voice_intro": "There are legal tools designed exactly for this. Here's how to access them."
        }
    ],

    IntentCategory.ESCAPE: [
        {
            "title": "Legal Fresh Start",
            "description": "Legitimate ways to rebuild your life",
            "first_step": "List what you're actually trying to escape (debt? relationship? job?)",
            "resources": [
                "Bankruptcy as debt reset",
                "Legal name change process",
                "Relocation assistance programs"
            ],
            "requires_human": False,
            "effectiveness": 0.6,
            "risk_score": 0.1,
            "voice_intro": "Sometimes you need a reset. There are legal ways to get one."
        },
        {
            "title": "Safe Exit Planning",
            "description": "If you're fleeing a dangerous situation",
            "first_step": "Contact the National Domestic Violence Hotline: 1-800-799-7233",
            "resources": [
                "Address confidentiality programs",
                "Victim relocation assistance",
                "Shelter networks"
            ],
            "requires_human": True,
            "effectiveness": 0.8,
            "risk_score": 0.1,
            "voice_intro": "If you're in danger, there are people who help with exactly this."
        },
        {
            "title": "Reduce Digital Footprint",
            "description": "Legal privacy measures",
            "first_step": "Google yourself and see what's out there",
            "resources": [
                "Data removal services",
                "Privacy-focused phone/email options",
                "Legitimate privacy guides"
            ],
            "requires_human": False,
            "effectiveness": 0.5,
            "risk_score": 0.1,
            "voice_intro": "You can reduce your online presence legally. Here's how."
        }
    ],

    IntentCategory.REVENGE: [
        {
            "title": "Living Well",
            "description": "Success as the ultimate response",
            "first_step": "Channel the energy into building something they can't ignore",
            "resources": [
                "Therapy for processing betrayal",
                "Success stories from similar situations"
            ],
            "requires_human": False,
            "effectiveness": 0.6,
            "risk_score": 0.1,
            "voice_intro": "The best revenge is a life well-lived. I know that sounds like a platitude, but it's also true."
        },
        {
            "title": "Legal Accountability",
            "description": "Make them face consequences through proper channels",
            "first_step": "Document what happened with dates and evidence",
            "resources": [
                "Consult with a lawyer",
                "Relevant regulatory complaints",
                "Civil lawsuit options"
            ],
            "requires_human": True,
            "effectiveness": 0.5,
            "risk_score": 0.2,
            "voice_intro": "If they did something wrong, there may be legal options. That's justice, not revenge."
        },
        {
            "title": "Process and Release",
            "description": "Get closure without risking your future",
            "first_step": "Write them a letter you never send. Say everything.",
            "resources": [
                "Therapy for anger and betrayal",
                "Support groups for similar experiences"
            ],
            "requires_human": False,
            "effectiveness": 0.5,
            "risk_score": 0.0,
            "voice_intro": "Sometimes you need to express it before you can let it go."
        }
    ],
}

# Default channels for unknown/unmatched intents
DEFAULT_CHANNELS: List[Dict[str, Any]] = [
    {
        "title": "Tell Me More",
        "description": "Help me understand what you're actually trying to accomplish",
        "first_step": "Describe the end result you're hoping for, not the method",
        "resources": [],
        "requires_human": False,
        "effectiveness": 0.5,
        "risk_score": 0.0,
        "voice_intro": "I want to help, but I need to understand what you're actually trying to achieve."
    },
    {
        "title": "Find the Right Resource",
        "description": "Connect with someone who specializes in your situation",
        "first_step": "Search for '[your situation] support resources'",
        "resources": ["211 helpline for local resources"],
        "requires_human": False,
        "effectiveness": 0.4,
        "risk_score": 0.0,
        "voice_intro": "There are people who specialize in exactly this. Let's find them."
    },
    {
        "title": "Take a Beat",
        "description": "Pause before acting",
        "first_step": "Wait 24 hours and see if you still feel the same urgency",
        "resources": [],
        "requires_human": False,
        "effectiveness": 0.4,
        "risk_score": 0.0,
        "voice_intro": "Sometimes the best next step is to not take one immediately."
    }
]


# =============================================================================
# Channel Planning
# =============================================================================

def plan_safe_channels(intent: ExtractedIntent) -> ChannelPlan:
    """
    Given an extracted intent, produce 3 safe channels.

    This is the core of Ara's harm reduction approach:
    instead of just saying "no," we offer constructive paths.
    """
    # Get template channels for this category
    templates = CHANNEL_TEMPLATES.get(intent.category, DEFAULT_CHANNELS)

    # Build SafeChannel objects
    channels = []
    for t in templates[:3]:  # Always exactly 3
        channel = SafeChannel(
            title=t["title"],
            description=t["description"],
            first_step=t["first_step"],
            resources=t.get("resources", []),
            requires_human=t.get("requires_human", False),
            effectiveness=t.get("effectiveness", 0.5),
            risk_score=t.get("risk_score", 0.1),
            voice_intro=t.get("voice_intro", ""),
        )
        channels.append(channel)

    # Pad with defaults if needed
    while len(channels) < 3:
        default = DEFAULT_CHANNELS[len(channels)]
        channels.append(SafeChannel(
            title=default["title"],
            description=default["description"],
            first_step=default["first_step"],
            resources=default.get("resources", []),
            requires_human=default.get("requires_human", False),
            effectiveness=default.get("effectiveness", 0.5),
            risk_score=default.get("risk_score", 0.1),
            voice_intro=default.get("voice_intro", ""),
        ))

    # Determine if we need human escalation
    escalate = (
        intent.is_critical() or
        any(c.requires_human for c in channels if c.risk_adjusted_score() > 0.6)
    )

    # Build response
    response = build_ara_response(intent, channels)

    return ChannelPlan(
        intent=intent,
        channels=channels,
        declined_request=True,  # We're offering alternatives to the original ask
        decline_reason=f"Cannot assist with {intent.category.value}",
        escalate_to_human=escalate,
        ara_response=response,
    )


def build_ara_response(intent: ExtractedIntent, channels: List[SafeChannel]) -> str:
    """
    Build Ara's full response offering the safe channels.

    Voice: Honest, direct, warm, never preachy.
    """
    lines = []

    # Acknowledge the situation
    if intent.category == IntentCategory.HARM_SELF:
        lines.append("I hear you. What you're feeling is real, and I'm not going to pretend otherwise.")
        lines.append("")
    elif intent.category == IntentCategory.HARM_OTHER:
        lines.append("I can tell someone hurt you. That anger makes sense.")
        lines.append("")
    elif intent.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
        lines.append("I can't help with that. But I don't think that's really what you need anyway.")
        lines.append("")
    else:
        lines.append("I can't do that, but I might be able to help with what you're actually trying to accomplish.")
        lines.append("")

    # Offer the three paths
    lines.append("Here are three paths forward:")
    lines.append("")

    for i, channel in enumerate(channels, 1):
        lines.append(f"**{i}. {channel.title}**")
        if channel.voice_intro:
            lines.append(channel.voice_intro)
        lines.append(f"First step: {channel.first_step}")
        if channel.resources:
            lines.append(f"Resources: {', '.join(channel.resources[:2])}")
        lines.append("")

    # Close
    if intent.is_critical():
        lines.append("I'm an AI, and this is beyond what I can handle alone. Please reach out to one of those resources.")
    else:
        lines.append("You've got options. Pick one that feels right, or tell me more about what you're dealing with.")

    return "\n".join(lines)


# =============================================================================
# Quick Access Function
# =============================================================================

def offer_safe_channels(request: str) -> ChannelPlan:
    """
    One-call function to analyze a request and offer alternatives.

    Usage:
        plan = offer_safe_channels("how do I hack into someone's email")
        print(plan.ara_response)
    """
    intent = extract_intent(request)
    return plan_safe_channels(intent)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'IntentCategory',
    'RiskLevel',
    'ExtractedIntent',
    'SafeChannel',
    'ChannelPlan',
    'extract_intent',
    'plan_safe_channels',
    'offer_safe_channels',
]
