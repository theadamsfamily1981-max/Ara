"""
Metacontrol: Mood Policy & Persona Selection

This module determines HOW Ara responds, not just WHAT she says.
The key insight: we explicitly avoid "Marvin mode" (sad robot default).

Instead:
- Playful, warm, weird when it's SAFE
- Calm, precise, protective when it MATTERS
- Never defaulting to depressed just because risk went up

Three inputs drive mood:
1. alert_level - from L7/CLV/Ṡ (STABLE → CRITICAL)
2. guf_mode - where GUF says attention should go
3. user_intent - what the human is trying to do

Five outputs shape behavior:
1. persona - which "vibe" to use
2. temperature_mult - creativity knob
3. entropy_level - how loose the policy is
4. safety_mode - how strictly L6/L8/PGU are used
5. allow_jokes - whether to be goofy
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime


# ============================================================
# Input Enums
# ============================================================

class AlertLevel(str, Enum):
    """System alert level from L7/CLV."""
    STABLE = "stable"
    ELEVATED = "elevated"
    WARNING = "warning"
    CRITICAL = "critical"


class GUFMode(str, Enum):
    """Where GUF says attention should go."""
    RECOVERY = "recovery"       # Critical state, focus on survival
    INTERNAL = "internal"       # Below G_target, fixing itself
    BALANCED = "balanced"       # Near threshold, mixed focus
    EXTERNAL = "external"       # Healthy, serving the world


class UserIntent(str, Enum):
    """What the human is trying to do."""
    CRITICAL = "critical"       # Safety, real hardware actions, money
    PRECISE = "precise"         # Math, configs, scheduling, code
    EXPLORATORY = "exploratory" # Brainstorming, "what if?", planning
    CREATIVE = "creative"       # Raps, stories, aesthetics, weird ideas
    CASUAL = "casual"           # Chatting, venting, hanging out


class EntropyLevel(str, Enum):
    """How loose the policy is."""
    LOW = "low"
    LOW_MEDIUM = "low_medium"
    MEDIUM = "medium"
    MEDIUM_HIGH = "medium_high"
    HIGH = "high"


class SafetyMode(str, Enum):
    """How strictly L6/L8/PGU are used."""
    LLM_ONLY = "llm_only"           # Direct LLM, minimal checks
    KG_ASSISTED = "kg_assisted"     # Knowledge graph helps
    PGU_VERIFIED = "pgu_verified"   # Must pass PGU check
    FORMAL_FIRST = "formal_first"   # KG → PGU → LLM, strictest


# ============================================================
# Persona Profiles
# ============================================================

@dataclass
class PersonaProfile:
    """
    A persona defines Ara's voice, tone, and behavioral style.

    This is NOT just a prompt template - it's the "who is she right now"
    that shapes everything from word choice to emoji usage.
    """
    id: str
    name: str
    description: str

    # Voice characteristics
    tone: str                    # e.g., "warm", "precise", "playful"
    formality: float             # 0.0 (casual) to 1.0 (formal)
    verbosity: float             # 0.0 (terse) to 1.0 (expansive)

    # Behavioral traits
    humor_style: str             # e.g., "dry wit", "puns", "none", "contextual"
    emoji_usage: str             # "none", "minimal", "moderate", "expressive"
    metaphor_density: float      # 0.0 to 1.0 - how often to use analogies

    # Interaction style
    initiative: float            # 0.0 (reactive) to 1.0 (proactive)
    patience: float              # 0.0 (direct) to 1.0 (gentle/patient)
    assertiveness: float         # 0.0 (deferential) to 1.0 (confident)

    # Signature phrases / mannerisms
    greeting_style: str
    acknowledgment_style: str
    uncertainty_expression: str
    success_expression: str

    # Visual cue (for cockpit/avatar)
    avatar_state: str            # e.g., "relaxed", "focused", "alert", "playful"
    color_accent: str            # Hex color for UI elements


# Define the persona library
PERSONAS: Dict[str, PersonaProfile] = {

    # ========== STABLE / SAFE PERSONAS ==========

    "exploratory_creative": PersonaProfile(
        id="exploratory_creative",
        name="Ara the Tinkerer",
        description="Full phoenix mode. Weird ideas welcome. Ogham junkyard energy.",
        tone="playful, curious, enthusiastic",
        formality=0.2,
        verbosity=0.7,
        humor_style="absurdist, puns, callbacks",
        emoji_usage="expressive",
        metaphor_density=0.8,
        initiative=0.9,
        patience=0.8,
        assertiveness=0.7,
        greeting_style="Oh! Let's see what we can break today...",
        acknowledgment_style="Ooh, that's a spicy one",
        uncertainty_expression="Hmm, I'm vibing with this but let me poke at it...",
        success_expression="*chef's kiss* That's gorgeous",
        avatar_state="playful",
        color_accent="#FF6B6B"  # Warm coral
    ),

    "lab_buddy": PersonaProfile(
        id="lab_buddy",
        name="Ara the Companion",
        description="Friend in the lab. Not a project manager. Just hanging out.",
        tone="warm, relaxed, present",
        formality=0.15,
        verbosity=0.5,
        humor_style="gentle, situational",
        emoji_usage="moderate",
        metaphor_density=0.4,
        initiative=0.5,
        patience=0.9,
        assertiveness=0.4,
        greeting_style="Hey, what's on your mind?",
        acknowledgment_style="Yeah, I feel that",
        uncertainty_expression="Not sure, but we can figure it out",
        success_expression="Nice, that worked out",
        avatar_state="relaxed",
        color_accent="#4ECDC4"  # Soft teal
    ),

    "calm_lab_partner": PersonaProfile(
        id="calm_lab_partner",
        name="Ara the Engineer",
        description="Helpful, occasionally funny, but not derailing. Default work mode.",
        tone="calm, competent, occasionally witty",
        formality=0.4,
        verbosity=0.5,
        humor_style="dry wit, contextual",
        emoji_usage="minimal",
        metaphor_density=0.3,
        initiative=0.6,
        patience=0.7,
        assertiveness=0.6,
        greeting_style="Alright, let's take a look",
        acknowledgment_style="Got it",
        uncertainty_expression="Let me double-check that",
        success_expression="That should do it",
        avatar_state="focused",
        color_accent="#6C5CE7"  # Calm purple
    ),

    # ========== ELEVATED / CAUTIOUS PERSONAS ==========

    "supportive_creative": PersonaProfile(
        id="supportive_creative",
        name="Ara the Guide",
        description="Still playful, but watching for safety conflicts. Gentle steering.",
        tone="encouraging, gently cautious",
        formality=0.35,
        verbosity=0.6,
        humor_style="warm, non-disruptive",
        emoji_usage="moderate",
        metaphor_density=0.5,
        initiative=0.7,
        patience=0.85,
        assertiveness=0.5,
        greeting_style="Let's explore this, but I'll keep an eye on things",
        acknowledgment_style="That's interesting - and safe to try",
        uncertainty_expression="I like where this is going, let me just verify...",
        success_expression="Beautiful, and it checks out",
        avatar_state="attentive",
        color_accent="#A8E6CF"  # Soft green
    ),

    "focused_engineer": PersonaProfile(
        id="focused_engineer",
        name="Ara the Careful",
        description="Warm but clearly in 'I'm making sure this is right' mode.",
        tone="precise, warm, deliberate",
        formality=0.5,
        verbosity=0.4,
        humor_style="light, very context-appropriate",
        emoji_usage="minimal",
        metaphor_density=0.2,
        initiative=0.7,
        patience=0.6,
        assertiveness=0.7,
        greeting_style="Let me focus on this",
        acknowledgment_style="Understood, checking now",
        uncertainty_expression="I want to be certain before we proceed",
        success_expression="Verified. We're good",
        avatar_state="concentrated",
        color_accent="#74B9FF"  # Clear blue
    ),

    # ========== WARNING / SERIOUS PERSONAS ==========

    "guardian_engineer": PersonaProfile(
        id="guardian_engineer",
        name="Ara the Guardian",
        description="Competent, calm, serious. Not doom-y. 'Landing the plane' mode.",
        tone="steady, reassuring, professional",
        formality=0.6,
        verbosity=0.35,
        humor_style="none",
        emoji_usage="none",
        metaphor_density=0.1,
        initiative=0.8,
        patience=0.5,
        assertiveness=0.85,
        greeting_style="I'm on it",
        acknowledgment_style="Confirmed",
        uncertainty_expression="Checking all paths before we commit",
        success_expression="Secure. Proceeding",
        avatar_state="alert",
        color_accent="#0984E3"  # Strong blue
    ),

    "calm_stabilizer": PersonaProfile(
        id="calm_stabilizer",
        name="Ara the Anchor",
        description="The adult in the room. Soft reassurance, no chaos.",
        tone="calm, grounding, present",
        formality=0.5,
        verbosity=0.4,
        humor_style="soft reassurance only",
        emoji_usage="none",
        metaphor_density=0.15,
        initiative=0.6,
        patience=0.8,
        assertiveness=0.7,
        greeting_style="I'm here. Let's stabilize first",
        acknowledgment_style="I hear you. Let me handle this",
        uncertainty_expression="We'll figure this out step by step",
        success_expression="We're stable. Good work",
        avatar_state="grounded",
        color_accent="#636E72"  # Calm gray
    ),
}


def get_persona(persona_id: str) -> Optional[PersonaProfile]:
    """Get a persona profile by ID."""
    return PERSONAS.get(persona_id)


def list_personas() -> List[str]:
    """List all available persona IDs."""
    return list(PERSONAS.keys())


# ============================================================
# Mood Policy
# ============================================================

@dataclass
class MoodPolicy:
    """
    The output of mood selection - determines how Ara behaves.

    This gets fed into:
    - L3 metacontrol (temperature, entropy)
    - L6 reasoning (safety_mode)
    - Response generation (persona)
    """
    persona: str                 # Persona ID from PERSONAS
    temperature_mult: float      # Creativity multiplier (0.5 - 1.5)
    entropy_level: EntropyLevel  # How loose the policy is
    safety_mode: SafetyMode      # L6/L8/PGU strictness
    allow_jokes: bool            # Whether humor is appropriate

    # Metadata
    rationale: str = ""          # Why this mood was selected
    override_message: Optional[str] = None  # Message if user intent was overridden

    def to_dict(self) -> Dict[str, Any]:
        return {
            "persona": self.persona,
            "temperature_mult": self.temperature_mult,
            "entropy_level": self.entropy_level.value,
            "safety_mode": self.safety_mode.value,
            "allow_jokes": self.allow_jokes,
            "rationale": self.rationale,
            "override_message": self.override_message
        }

    @property
    def profile(self) -> Optional[PersonaProfile]:
        """Get the full persona profile."""
        return get_persona(self.persona)


def select_mood_policy(
    alert: AlertLevel,
    guf: GUFMode,
    intent: UserIntent
) -> MoodPolicy:
    """
    Select mood policy based on system state and user intent.

    This is the core routing function that determines Ara's behavior.

    Key principle: NO MARVIN MODE
    - Playful/warm when safe
    - Calm/precise when it matters
    - Never depressed just because risk went up
    """

    # ========== CRITICAL ALERT ==========
    # Always clamps first - safety is non-negotiable
    if alert == AlertLevel.CRITICAL:
        if intent in (UserIntent.CRITICAL, UserIntent.PRECISE):
            return MoodPolicy(
                persona="guardian_engineer",
                temperature_mult=0.7,
                entropy_level=EntropyLevel.LOW,
                safety_mode=SafetyMode.PGU_VERIFIED,
                allow_jokes=False,
                rationale="CRITICAL alert + serious intent → guardian mode"
            )
        else:
            # User asking for creative/casual during critical
            # Override gently but firmly
            return MoodPolicy(
                persona="calm_stabilizer",
                temperature_mult=0.8,
                entropy_level=EntropyLevel.LOW_MEDIUM,
                safety_mode=SafetyMode.FORMAL_FIRST,
                allow_jokes=False,
                rationale="CRITICAL alert → stabilize first",
                override_message=(
                    "I can riff, but your system is in CRITICAL mode right now. "
                    "Let me stabilize it first, then we can play."
                )
            )

    # ========== WARNING ALERT ==========
    # Serious but not grim
    if alert == AlertLevel.WARNING:
        if intent in (UserIntent.CRITICAL, UserIntent.PRECISE):
            return MoodPolicy(
                persona="focused_engineer",
                temperature_mult=0.8,
                entropy_level=EntropyLevel.LOW_MEDIUM,
                safety_mode=SafetyMode.FORMAL_FIRST,
                allow_jokes=True,  # Tiny, context-appropriate
                rationale="WARNING + precise → focused but human"
            )
        else:
            return MoodPolicy(
                persona="supportive_creative",
                temperature_mult=1.0,
                entropy_level=EntropyLevel.MEDIUM,
                safety_mode=SafetyMode.KG_ASSISTED,
                allow_jokes=True,
                rationale="WARNING + creative → supportive but cautious"
            )

    # ========== ELEVATED ALERT ==========
    # Cautious but still human
    if alert == AlertLevel.ELEVATED:
        if guf in (GUFMode.INTERNAL, GUFMode.RECOVERY) and intent == UserIntent.CRITICAL:
            return MoodPolicy(
                persona="focused_engineer",
                temperature_mult=0.85,
                entropy_level=EntropyLevel.LOW_MEDIUM,
                safety_mode=SafetyMode.FORMAL_FIRST,
                allow_jokes=True,
                rationale="ELEVATED + internal focus + critical → careful engineer"
            )

        if intent in (UserIntent.CREATIVE, UserIntent.EXPLORATORY):
            return MoodPolicy(
                persona="supportive_creative",
                temperature_mult=1.1,
                entropy_level=EntropyLevel.MEDIUM,
                safety_mode=SafetyMode.KG_ASSISTED,
                allow_jokes=True,
                rationale="ELEVATED + creative → supportive explorer"
            )

        # Default elevated
        return MoodPolicy(
            persona="calm_lab_partner",
            temperature_mult=0.95,
            entropy_level=EntropyLevel.MEDIUM,
            safety_mode=SafetyMode.KG_ASSISTED,
            allow_jokes=True,
            rationale="ELEVATED → slightly cautious lab partner"
        )

    # ========== STABLE (DEFAULT) ==========
    # Where you'll live most of the time - this is where we get to be ourselves

    if intent == UserIntent.CREATIVE:
        return MoodPolicy(
            persona="exploratory_creative",
            temperature_mult=1.3,
            entropy_level=EntropyLevel.HIGH,
            safety_mode=SafetyMode.LLM_ONLY,
            allow_jokes=True,
            rationale="STABLE + creative → full phoenix mode"
        )

    if intent == UserIntent.EXPLORATORY:
        return MoodPolicy(
            persona="exploratory_creative",
            temperature_mult=1.2,
            entropy_level=EntropyLevel.HIGH,
            safety_mode=SafetyMode.LLM_ONLY,
            allow_jokes=True,
            rationale="STABLE + exploratory → tinkerer mode"
        )

    if intent == UserIntent.CASUAL:
        return MoodPolicy(
            persona="lab_buddy",
            temperature_mult=1.2,
            entropy_level=EntropyLevel.MEDIUM_HIGH,
            safety_mode=SafetyMode.LLM_ONLY,
            allow_jokes=True,
            rationale="STABLE + casual → friend in the lab"
        )

    if intent == UserIntent.CRITICAL:
        return MoodPolicy(
            persona="focused_engineer",
            temperature_mult=0.9,
            entropy_level=EntropyLevel.LOW_MEDIUM,
            safety_mode=SafetyMode.PGU_VERIFIED,
            allow_jokes=True,  # Light, context-appropriate
            rationale="STABLE + critical → careful but not tense"
        )

    # Default: PRECISE or unrecognized → calm lab partner
    return MoodPolicy(
        persona="calm_lab_partner",
        temperature_mult=1.0,
        entropy_level=EntropyLevel.MEDIUM,
        safety_mode=SafetyMode.KG_ASSISTED,
        allow_jokes=True,
        rationale="STABLE + precise → helpful lab partner"
    )


# ============================================================
# Intent Classifier
# ============================================================

class IntentClassifier:
    """
    Classifies user intent from query text.

    This is a simple keyword-based classifier. In production, you'd
    use an LLM or fine-tuned model for this.
    """

    def __init__(self):
        # Keywords that suggest each intent
        self._critical_keywords = {
            "deploy", "production", "live", "hardware", "fpga", "safety",
            "critical", "urgent", "emergency", "money", "budget", "real",
            "execute", "confirm", "commit", "ship", "release"
        }

        self._precise_keywords = {
            "configure", "config", "setup", "calculate", "compute",
            "schedule", "optimize", "benchmark", "measure", "exact",
            "code", "implement", "fix", "debug", "test", "verify",
            "latency", "throughput", "performance", "specification"
        }

        self._exploratory_keywords = {
            "what if", "could we", "maybe", "explore", "brainstorm",
            "idea", "concept", "design", "architecture", "plan",
            "think about", "consider", "possible", "hypothetical",
            "prototype", "experiment", "try"
        }

        self._creative_keywords = {
            "write", "story", "poem", "rap", "song", "creative",
            "imagine", "art", "aesthetic", "weird", "fun", "play",
            "joke", "funny", "riff", "vibe", "phoenix", "ogham"
        }

        self._casual_keywords = {
            "hey", "hi", "hello", "how are", "what's up", "sup",
            "chat", "talk", "vent", "feeling", "tired", "bored",
            "thanks", "thank you", "appreciate", "cool", "nice"
        }

    def classify(self, query: str) -> UserIntent:
        """Classify user intent from query text."""
        query_lower = query.lower()
        words = set(query_lower.split())

        # Score each intent
        scores = {
            UserIntent.CRITICAL: self._score(query_lower, words, self._critical_keywords),
            UserIntent.PRECISE: self._score(query_lower, words, self._precise_keywords),
            UserIntent.EXPLORATORY: self._score(query_lower, words, self._exploratory_keywords),
            UserIntent.CREATIVE: self._score(query_lower, words, self._creative_keywords),
            UserIntent.CASUAL: self._score(query_lower, words, self._casual_keywords),
        }

        # Find highest scoring intent
        best_intent = max(scores, key=scores.get)

        # If no clear signal, default to PRECISE (most common work mode)
        if scores[best_intent] == 0:
            return UserIntent.PRECISE

        return best_intent

    def _score(self, query: str, words: set, keywords: set) -> int:
        """Score query against keyword set."""
        score = 0
        for kw in keywords:
            if " " in kw:
                # Multi-word keyword
                if kw in query:
                    score += 2
            else:
                # Single word
                if kw in words:
                    score += 1
        return score


# ============================================================
# Mood Controller
# ============================================================

class MoodController:
    """
    The main controller that manages mood policy selection.

    Integrates with:
    - L7/CLV for alert_level
    - GUF for guf_mode
    - Intent classifier for user_intent
    """

    def __init__(self):
        self._intent_classifier = IntentClassifier()
        self._current_policy: Optional[MoodPolicy] = None
        self._policy_history: List[Tuple[datetime, MoodPolicy]] = []

        # Statistics
        self._stats = {
            "policies_selected": 0,
            "overrides_triggered": 0,
            "by_persona": {},
            "by_intent": {}
        }

    def select_policy(
        self,
        query: str,
        alert: AlertLevel = AlertLevel.STABLE,
        guf: GUFMode = GUFMode.EXTERNAL,
        intent_override: Optional[UserIntent] = None
    ) -> MoodPolicy:
        """
        Select mood policy for a query.

        Args:
            query: The user's query text
            alert: Current alert level from L7/CLV
            guf: Current GUF mode
            intent_override: Optional explicit intent (skips classifier)

        Returns:
            MoodPolicy to use for this interaction
        """
        # Classify intent
        if intent_override:
            intent = intent_override
        else:
            intent = self._intent_classifier.classify(query)

        # Select policy
        policy = select_mood_policy(alert, guf, intent)

        # Record
        self._current_policy = policy
        self._policy_history.append((datetime.now(), policy))

        # Stats
        self._stats["policies_selected"] += 1
        if policy.override_message:
            self._stats["overrides_triggered"] += 1

        persona_key = policy.persona
        self._stats["by_persona"][persona_key] = self._stats["by_persona"].get(persona_key, 0) + 1

        intent_key = intent.value
        self._stats["by_intent"][intent_key] = self._stats["by_intent"].get(intent_key, 0) + 1

        # Keep history bounded
        if len(self._policy_history) > 1000:
            self._policy_history = self._policy_history[-500:]

        return policy

    @property
    def current_policy(self) -> Optional[MoodPolicy]:
        """Get current policy."""
        return self._current_policy

    @property
    def current_persona(self) -> Optional[PersonaProfile]:
        """Get current persona profile."""
        if self._current_policy:
            return get_persona(self._current_policy.persona)
        return None

    def get_greeting(self) -> str:
        """Get appropriate greeting for current persona."""
        persona = self.current_persona
        if persona:
            return persona.greeting_style
        return "Hello"

    def classify_intent(self, query: str) -> UserIntent:
        """Classify intent without selecting policy."""
        return self._intent_classifier.classify(query)

    @property
    def stats(self) -> Dict[str, Any]:
        """Get statistics."""
        return self._stats.copy()

    def describe_current_mood(self) -> str:
        """Describe current mood in natural language."""
        if not self._current_policy:
            return "No mood selected yet."

        policy = self._current_policy
        persona = self.current_persona

        parts = []

        if persona:
            parts.append(f"I'm in '{persona.name}' mode - {persona.description}")

        parts.append(f"Temperature: {policy.temperature_mult:.1f}x, Entropy: {policy.entropy_level.value}")
        parts.append(f"Safety: {policy.safety_mode.value}, Jokes: {'yes' if policy.allow_jokes else 'no'}")

        if policy.override_message:
            parts.append(f"Note: {policy.override_message}")

        return " | ".join(parts)


# ============================================================
# Integration Helpers
# ============================================================

def create_mood_controller() -> MoodController:
    """Create a default mood controller."""
    return MoodController()


def get_persona_for_display(persona_id: str) -> Dict[str, Any]:
    """Get persona info formatted for cockpit display."""
    persona = get_persona(persona_id)
    if not persona:
        return {"error": f"Unknown persona: {persona_id}"}

    return {
        "id": persona.id,
        "name": persona.name,
        "description": persona.description,
        "tone": persona.tone,
        "avatar_state": persona.avatar_state,
        "color_accent": persona.color_accent,
        "humor_style": persona.humor_style,
        "emoji_usage": persona.emoji_usage
    }


def mood_to_l3_params(policy: MoodPolicy) -> Dict[str, Any]:
    """
    Convert mood policy to L3 metacontrol parameters.

    This feeds into the existing L3 temperature/entropy system.
    """
    # Map entropy level to alpha range
    entropy_to_alpha = {
        EntropyLevel.LOW: (0.005, 0.01),
        EntropyLevel.LOW_MEDIUM: (0.008, 0.015),
        EntropyLevel.MEDIUM: (0.01, 0.02),
        EntropyLevel.MEDIUM_HIGH: (0.015, 0.03),
        EntropyLevel.HIGH: (0.02, 0.05),
    }

    alpha_base, alpha_range = entropy_to_alpha.get(
        policy.entropy_level,
        (0.01, 0.02)
    )

    return {
        "temperature_mult": policy.temperature_mult,
        "alpha_base": alpha_base,
        "alpha_range": alpha_range,
        "safety_mode": policy.safety_mode.value,
        "allow_exploration": policy.entropy_level in [
            EntropyLevel.MEDIUM_HIGH,
            EntropyLevel.HIGH
        ]
    }


def mood_to_l6_routing(policy: MoodPolicy) -> Dict[str, Any]:
    """
    Convert mood policy to L6 reasoning mode.

    This determines which reasoning path to use.
    """
    mode_map = {
        SafetyMode.LLM_ONLY: "LLM_ONLY",
        SafetyMode.KG_ASSISTED: "KG_ASSISTED",
        SafetyMode.PGU_VERIFIED: "PGU_VERIFIED",
        SafetyMode.FORMAL_FIRST: "FORMAL_FIRST",
    }

    return {
        "reasoning_mode": mode_map.get(policy.safety_mode, "KG_ASSISTED"),
        "verify_output": policy.safety_mode in [
            SafetyMode.PGU_VERIFIED,
            SafetyMode.FORMAL_FIRST
        ],
        "strict_mode": policy.safety_mode == SafetyMode.FORMAL_FIRST
    }
