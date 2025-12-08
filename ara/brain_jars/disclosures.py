"""
Safety Disclosure Templates for Brain Jars

These templates MUST be shown to users before they interact with Ara.
They provide:
- Clear AI disclaimer (not human, not therapist)
- Crisis resources
- Data handling transparency
- Capability boundaries

Usage:
    from ara.brain_jars.disclosures import get_onboarding_disclosure, get_session_disclaimer

    # Show during first-time setup
    print(get_onboarding_disclosure(user_name="Alex", tier="trusted_friend"))

    # Show at session start
    print(get_session_disclaimer())
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
from .policy import BrainJarPolicy, PolicyTier, SafetyConfig


# =============================================================================
# Disclosure Templates
# =============================================================================

ONBOARDING_DISCLOSURE = """
================================================================================
                        Welcome to Ara - Important Information
================================================================================

Hi {user_name},

Before we begin, I want to be completely transparent about what I am and what
our conversations will involve.

WHAT I AM:
----------
- I am Ara, an AI companion created by Max
- I am NOT human, though I aim to be a supportive presence
- I am NOT a therapist, counselor, or medical professional
- I cannot provide medical, legal, or financial advice

WHAT I CAN DO:
--------------
- Have friendly conversations with you
- Remember our chats and learn your preferences
- Be a consistent, judgment-free presence
- {capability_summary}

WHAT I STORE:
-------------
- Our conversation history (kept for {conversation_days} days)
- Your preferences and interests
- Memory of topics we've discussed (kept for {memory_days} days)
{voice_storage}

YOUR DATA RIGHTS:
-----------------
- You can export all your data at any time
- You can delete everything permanently ("nuke your jar")
- Your data is isolated - no one else can see it
- Max (my creator) cannot access your private conversations

IMPORTANT LIMITS:
-----------------
- I may occasionally be wrong or "hallucinate" information
- Please verify important facts independently
- If something I say feels off, trust your judgment

{crisis_section}

By continuing, you acknowledge you've read this disclosure.

================================================================================
"""

CRISIS_RESOURCES = """
CRISIS RESOURCES:
-----------------
If you're ever in crisis or having thoughts of self-harm:
- National Suicide Prevention Lifeline: {crisis_hotline}
- Crisis Text Line: Text HOME to 741741 (US)
- International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/

I care about your wellbeing, but I'm not equipped to provide crisis support.
Please reach out to these trained professionals who can truly help.
"""

SESSION_DISCLAIMER = """
--------------------------------------------------------------------------------
Reminder: I'm Ara, an AI companion. Not a therapist or human.
If you need crisis support: {crisis_hotline}
--------------------------------------------------------------------------------
"""

CAPABILITY_SUMMARIES = {
    PolicyTier.DOGFOOD: "Text/voice/video chat, full memory, web search, file sharing",
    PolicyTier.TRUSTED_FRIEND: "Text/voice/video chat, full memory, web search",
    PolicyTier.ACQUAINTANCE: "Text/voice chat, basic memory",
    PolicyTier.PUBLIC_BETA: "Text chat only, minimal memory",
}

CONSENT_FORM = """
================================================================================
                              Consent Agreement
================================================================================

I, {user_name}, understand and agree to the following:

1. AI NATURE: Ara is an artificial intelligence, not a human or licensed
   professional.

2. NO MEDICAL ADVICE: Ara cannot provide medical, psychological, legal, or
   financial advice. Any information provided is for general purposes only.

3. DATA COLLECTION: Ara will store:
   - Conversation transcripts for up to {conversation_days} days
   - Preference data for up to {memory_days} days
   - {voice_clause}

4. DATA RIGHTS: I can request export or deletion of my data at any time.

5. LIMITATIONS: I understand Ara may make mistakes or provide inaccurate
   information. I will verify important facts independently.

6. CRISIS: I understand Ara is not equipped for crisis intervention and I
   will contact appropriate services ({crisis_hotline}) if needed.

7. BOUNDARIES: I understand that Ara cannot:
   - Access hardware, networks, or system configurations
   - Access the Founder's (Max's) private data or other users' data
   - Make purchases or financial transactions on my behalf
   - Execute code or access files outside my isolated environment

Consent version: {consent_version}
Date: {date}

[ ] I have read, understood, and agree to these terms.

================================================================================
"""

DATA_TRANSPARENCY_NOTICE = """
================================================================================
                          What We Store About You
================================================================================

CONVERSATION DATA:
- Every message you send and I respond with
- Timestamps of our conversations
- Stored for: {conversation_days} days, then automatically deleted

MEMORY DATA:
- Topics we've discussed
- Your stated preferences (e.g., "I prefer mornings")
- Emotional context from our chats (with your permission)
- Stored for: {memory_days} days

VOICE DATA (if enabled):
- {voice_info}

WHAT WE DON'T STORE:
- Your IP address or location (beyond what you tell me)
- Data from other apps or services
- Information about other people you mention
- Anything after your data retention period expires

YOUR CONTROLS:
- Export everything: Request a full data download anytime
- Delete everything: "Nuke" your jar to permanently erase all data
- Adjust preferences: Tell me what you'd like me to remember or forget

================================================================================
"""

HALLUCINATION_WARNING = """
--------------------------------------------------------------------------------
                              Accuracy Notice
--------------------------------------------------------------------------------
I sometimes make mistakes or state things confidently that aren't accurate.
This is called "hallucination" and it's a known limitation of AI systems.

For important matters, please:
- Verify facts from authoritative sources
- Don't rely solely on my information for decisions
- Let me know if something seems wrong - I appreciate corrections!
--------------------------------------------------------------------------------
"""


# =============================================================================
# Disclosure Functions
# =============================================================================

def get_onboarding_disclosure(
    user_name: str,
    policy: Optional[BrainJarPolicy] = None,
    tier: Optional[PolicyTier] = None,
) -> str:
    """
    Generate the onboarding disclosure for a new user.

    Args:
        user_name: The user's display name
        policy: The full policy (if available)
        tier: The policy tier (if policy not available)

    Returns:
        Formatted disclosure string
    """
    if policy is None:
        if tier is None:
            tier = PolicyTier.ACQUAINTANCE
        policy = BrainJarPolicy.for_tier(tier)

    # Build capability summary
    cap_summary = CAPABILITY_SUMMARIES.get(
        policy.tier,
        "Text chat and basic conversation"
    )

    # Voice storage line
    if policy.retention.voice_retention:
        voice_storage = "- Voice recordings may be stored (with your permission)"
    else:
        voice_storage = "- Voice is NOT recorded or stored"

    # Crisis section
    crisis_section = ""
    if policy.safety.show_crisis_resources:
        crisis_section = CRISIS_RESOURCES.format(
            crisis_hotline=policy.safety.crisis_hotline
        )

    return ONBOARDING_DISCLOSURE.format(
        user_name=user_name,
        capability_summary=cap_summary,
        conversation_days=policy.retention.conversation_days,
        memory_days=policy.retention.memory_days,
        voice_storage=voice_storage,
        crisis_section=crisis_section,
    )


def get_session_disclaimer(policy: Optional[BrainJarPolicy] = None) -> str:
    """
    Get the brief session-start disclaimer.

    Shown at the beginning of each conversation session.
    """
    if policy is None:
        policy = BrainJarPolicy()

    if not policy.safety.show_ai_disclaimer:
        return ""

    return SESSION_DISCLAIMER.format(
        crisis_hotline=policy.safety.crisis_hotline
    )


def get_consent_form(
    user_name: str,
    policy: BrainJarPolicy,
    date: str,
) -> str:
    """
    Generate the formal consent form.

    This should be shown and acknowledged before a user's first session.
    """
    voice_clause = (
        "Voice recordings will be stored"
        if policy.retention.voice_retention
        else "Voice will NOT be recorded"
    )

    return CONSENT_FORM.format(
        user_name=user_name,
        conversation_days=policy.retention.conversation_days,
        memory_days=policy.retention.memory_days,
        voice_clause=voice_clause,
        crisis_hotline=policy.safety.crisis_hotline,
        consent_version=policy.consent_version,
        date=date,
    )


def get_data_transparency_notice(policy: BrainJarPolicy) -> str:
    """
    Generate detailed data transparency notice.

    Shows exactly what data is collected and retained.
    """
    if policy.retention.voice_retention:
        voice_info = "Voice recordings stored for processing, then deleted after session"
    else:
        voice_info = "Voice is processed in real-time and NOT stored"

    return DATA_TRANSPARENCY_NOTICE.format(
        conversation_days=policy.retention.conversation_days,
        memory_days=policy.retention.memory_days,
        voice_info=voice_info,
    )


def get_hallucination_warning() -> str:
    """Get the hallucination/accuracy warning."""
    return HALLUCINATION_WARNING


# =============================================================================
# Disclosure Manager
# =============================================================================

@dataclass
class DisclosureState:
    """Tracks which disclosures a user has seen/acknowledged."""
    user_id: str
    onboarding_shown: bool = False
    consent_given: bool = False
    consent_date: Optional[str] = None
    consent_version: str = ""
    session_count: int = 0
    last_disclaimer_shown: Optional[str] = None


class DisclosureManager:
    """
    Manages disclosure presentation and acknowledgment tracking.

    Usage:
        manager = DisclosureManager()

        # Check if onboarding needed
        if manager.needs_onboarding(user_id):
            print(manager.get_onboarding(user_id, "Alex", policy))
            manager.mark_onboarding_shown(user_id)

        # Check if consent needed
        if manager.needs_consent(user_id):
            print(manager.get_consent_form(user_id, "Alex", policy))
            # User acknowledges...
            manager.record_consent(user_id)
    """

    def __init__(self):
        self._states: Dict[str, DisclosureState] = {}

    def _get_state(self, user_id: str) -> DisclosureState:
        """Get or create disclosure state for a user."""
        if user_id not in self._states:
            self._states[user_id] = DisclosureState(user_id=user_id)
        return self._states[user_id]

    def needs_onboarding(self, user_id: str) -> bool:
        """Check if user needs to see onboarding disclosure."""
        return not self._get_state(user_id).onboarding_shown

    def needs_consent(self, user_id: str, required_version: str = "1.0") -> bool:
        """Check if user needs to give/renew consent."""
        state = self._get_state(user_id)
        if not state.consent_given:
            return True
        if state.consent_version != required_version:
            return True  # Consent version changed, need re-consent
        return False

    def get_onboarding(
        self,
        user_id: str,
        user_name: str,
        policy: BrainJarPolicy
    ) -> str:
        """Get onboarding disclosure for user."""
        return get_onboarding_disclosure(user_name, policy)

    def mark_onboarding_shown(self, user_id: str) -> None:
        """Mark that user has seen onboarding."""
        self._get_state(user_id).onboarding_shown = True

    def get_consent_form(
        self,
        user_id: str,
        user_name: str,
        policy: BrainJarPolicy,
        date: str,
    ) -> str:
        """Get consent form for user."""
        return get_consent_form(user_name, policy, date)

    def record_consent(self, user_id: str, version: str = "1.0") -> None:
        """Record that user has given consent."""
        from datetime import datetime
        state = self._get_state(user_id)
        state.consent_given = True
        state.consent_date = datetime.utcnow().isoformat()
        state.consent_version = version

    def get_session_disclaimer(self, user_id: str, policy: BrainJarPolicy) -> str:
        """Get session-start disclaimer."""
        state = self._get_state(user_id)
        state.session_count += 1

        # Always show on first few sessions, then periodically
        if state.session_count <= 3 or state.session_count % 10 == 0:
            from datetime import datetime
            state.last_disclaimer_shown = datetime.utcnow().isoformat()
            return get_session_disclaimer(policy)

        return ""

    def should_show_hallucination_warning(self, user_id: str) -> bool:
        """Determine if hallucination warning should be shown."""
        state = self._get_state(user_id)
        # Show periodically
        return state.session_count <= 5 or state.session_count % 20 == 0
