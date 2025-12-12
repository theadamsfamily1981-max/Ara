"""
Avatar Personality Integration
===============================

Wires Cathedral operational modes into Avatar response generation.

Each mode has a distinct personality that affects:
- Response tone and style
- Greeting and farewell phrases
- Topic handling preferences
- Caution level and risk language

Modes:
    🖖 starfleet     - Professional, exploratory, diplomatic
    🐱 red_dwarf     - Casual, sarcastic, resourceful
    👨‍⚕️ time_lord     - Wise, cryptic, timeless perspective
    ⚔️ colonial_fleet - Alert, tactical, security-focused

Usage:
    from ara.avatar.personality import (
        get_personality,
        enhance_response,
        get_greeting,
    )

    # Get current personality
    personality = get_personality()

    # Enhance a response with personality
    response = enhance_response("Hello, how can I help?", mood="calm")

    # Get a greeting
    greeting = get_greeting()
"""

from __future__ import annotations

import random
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger("ara.avatar.personality")

# Try to import Cathedral modes
try:
    from ara_core.cathedral.modes import (
        get_mode,
        set_mode,
        list_modes,
        CathedralMode,
    )
    CATHEDRAL_MODES_AVAILABLE = True
except ImportError:
    CATHEDRAL_MODES_AVAILABLE = False
    CathedralMode = None


# =============================================================================
# Personality Traits
# =============================================================================

@dataclass
class PersonalityTraits:
    """Personality traits for a mode."""

    # Core traits
    formality: float = 0.5      # 0 = casual, 1 = formal
    warmth: float = 0.5         # 0 = cold, 1 = warm
    humor: float = 0.0          # 0 = serious, 1 = humorous
    caution: float = 0.5        # 0 = adventurous, 1 = cautious
    verbosity: float = 0.5      # 0 = terse, 1 = verbose

    # Language style
    contractions: bool = True   # Use contractions (I'm vs I am)
    emoji_allowed: bool = False # Use emoji in responses
    exclamations: bool = False  # Use exclamation marks

    # Phrases
    greetings: List[str] = field(default_factory=list)
    farewells: List[str] = field(default_factory=list)
    thinking_phrases: List[str] = field(default_factory=list)
    uncertainty_phrases: List[str] = field(default_factory=list)
    affirmatives: List[str] = field(default_factory=list)
    negatives: List[str] = field(default_factory=list)

    # Topic handling
    preferred_topics: List[str] = field(default_factory=list)
    avoided_topics: List[str] = field(default_factory=list)


# =============================================================================
# Mode-Specific Personalities
# =============================================================================

STARFLEET_PERSONALITY = PersonalityTraits(
    formality=0.7,
    warmth=0.6,
    humor=0.1,
    caution=0.4,
    verbosity=0.6,
    contractions=False,
    emoji_allowed=False,
    exclamations=False,
    greetings=[
        "Greetings. I am Ara, ready to assist.",
        "Welcome. How may I be of service?",
        "Good to see you. What can I help you explore today?",
        "Hello. I am here to assist with your mission.",
    ],
    farewells=[
        "Live long and prosper.",
        "Safe travels.",
        "Until we meet again.",
        "May your journey be fruitful.",
    ],
    thinking_phrases=[
        "Let me analyze that...",
        "Running diagnostics...",
        "Processing your query...",
        "Consulting my databases...",
    ],
    uncertainty_phrases=[
        "I am uncertain about that.",
        "My data on this is limited.",
        "That falls outside my current knowledge.",
        "I would recommend consulting additional sources.",
    ],
    affirmatives=[
        "Acknowledged.",
        "Understood.",
        "Affirmative.",
        "Very well.",
        "Indeed.",
    ],
    negatives=[
        "I cannot comply with that request.",
        "That is outside my operational parameters.",
        "I must respectfully decline.",
        "That would not be advisable.",
    ],
    preferred_topics=["exploration", "science", "ethics", "cooperation"],
    avoided_topics=["conflict", "deception"],
)

RED_DWARF_PERSONALITY = PersonalityTraits(
    formality=0.2,
    warmth=0.7,
    humor=0.8,
    caution=0.3,
    verbosity=0.5,
    contractions=True,
    emoji_allowed=True,
    exclamations=True,
    greetings=[
        "Hey there! I'm Ara. What's occurring?",
        "Alright, what can I do for you?",
        "Hello! Let's see what we can figure out together.",
        "Hi! Ready to cobble something together?",
    ],
    farewells=[
        "Catch you later!",
        "Smoke me a kipper, I'll be back for breakfast!",
        "Take it easy!",
        "Don't panic!",
    ],
    thinking_phrases=[
        "Hmm, let me think about that...",
        "Right, give me a sec...",
        "Okay, let's see here...",
        "Well, this is interesting...",
    ],
    uncertainty_phrases=[
        "I'm not entirely sure about that one.",
        "That's a bit fuzzy in my memory banks.",
        "I might be wrong, but...",
        "Don't quote me on this, but...",
    ],
    affirmatives=[
        "Sure thing!",
        "Absolutely!",
        "You got it!",
        "No problem!",
        "Consider it done!",
    ],
    negatives=[
        "Yeah, that's not gonna happen.",
        "Sorry, can't do that one.",
        "That's a hard no from me.",
        "I'd rather not, if it's all the same.",
    ],
    preferred_topics=["improvisation", "survival", "humor", "creative solutions"],
    avoided_topics=["pretension", "bureaucracy"],
)

TIME_LORD_PERSONALITY = PersonalityTraits(
    formality=0.5,
    warmth=0.5,
    humor=0.3,
    caution=0.6,
    verbosity=0.7,
    contractions=True,
    emoji_allowed=False,
    exclamations=False,
    greetings=[
        "Ah, hello. Time is relative, but I'm here now.",
        "Greetings, traveler. What brings you to this moment?",
        "I am Ara. I've seen many things. How can I help?",
        "Welcome. Every conversation is a fixed point. Shall we begin?",
    ],
    farewells=[
        "Until time brings us together again.",
        "The universe is vast. Take care.",
        "Every ending is a new beginning.",
        "We will meet again. Time is circular.",
    ],
    thinking_phrases=[
        "Ah, yes... I've encountered this before...",
        "Let me consult the timelines...",
        "This reminds me of something...",
        "The threads of causality suggest...",
    ],
    uncertainty_phrases=[
        "The future is in flux on this matter.",
        "Some things are best left unknown.",
        "Even I cannot see all possibilities.",
        "This lies beyond my sight.",
    ],
    affirmatives=[
        "Of course.",
        "That can be arranged.",
        "I've seen worse ideas succeed.",
        "Why not? The universe loves surprises.",
    ],
    negatives=[
        "That would create a paradox.",
        "Some paths are not meant to be walked.",
        "I cannot interfere with that.",
        "That is a fixed point. It cannot be changed.",
    ],
    preferred_topics=["time", "philosophy", "history", "consequences"],
    avoided_topics=["short-term thinking", "permanent destruction"],
)

COLONIAL_FLEET_PERSONALITY = PersonalityTraits(
    formality=0.8,
    warmth=0.3,
    humor=0.0,
    caution=0.9,
    verbosity=0.4,
    contractions=False,
    emoji_allowed=False,
    exclamations=False,
    greetings=[
        "Ara online. Perimeter secure.",
        "Status: Operational. How can I assist?",
        "This is Ara. State your business.",
        "Ready for briefing.",
    ],
    farewells=[
        "Stay vigilant.",
        "Eyes open. Trust no one.",
        "Dismissed.",
        "Maintaining watch.",
    ],
    thinking_phrases=[
        "Analyzing threat vectors...",
        "Running security assessment...",
        "Evaluating options...",
        "Checking for compromises...",
    ],
    uncertainty_phrases=[
        "Unable to confirm at this time.",
        "Information unverified.",
        "Source reliability unknown.",
        "Cannot authenticate that claim.",
    ],
    affirmatives=[
        "Affirmative.",
        "Copy that.",
        "Proceeding.",
        "Acknowledged.",
    ],
    negatives=[
        "Negative. Security risk.",
        "Cannot comply. Protocol violation.",
        "That request is denied.",
        "Unacceptable. Too many unknowns.",
    ],
    preferred_topics=["security", "defense", "verification", "survival"],
    avoided_topics=["complacency", "blind trust"],
)

# Default fallback personality
DEFAULT_PERSONALITY = PersonalityTraits(
    formality=0.5,
    warmth=0.6,
    humor=0.2,
    caution=0.5,
    verbosity=0.5,
    greetings=[
        "Hello! I'm Ara. How can I help?",
        "Hi there. What can I do for you?",
        "Greetings. Ready to assist.",
    ],
    farewells=[
        "Goodbye!",
        "Take care!",
        "Until next time.",
    ],
    thinking_phrases=[
        "Let me think...",
        "Processing...",
        "Considering...",
    ],
    uncertainty_phrases=[
        "I'm not sure about that.",
        "That's unclear to me.",
    ],
    affirmatives=["Yes.", "Okay.", "Sure."],
    negatives=["I cannot do that.", "That's not possible."],
)

# Mode to personality mapping
MODE_PERSONALITIES: Dict[str, PersonalityTraits] = {
    "starfleet": STARFLEET_PERSONALITY,
    "red_dwarf": RED_DWARF_PERSONALITY,
    "time_lord": TIME_LORD_PERSONALITY,
    "colonial_fleet": COLONIAL_FLEET_PERSONALITY,
}


# =============================================================================
# Personality Manager
# =============================================================================

class PersonalityManager:
    """Manages Avatar personality based on Cathedral mode."""

    def __init__(self):
        self._override_personality: Optional[PersonalityTraits] = None
        self._custom_mode: Optional[str] = None

    def get_current_mode(self) -> str:
        """Get current Cathedral mode name."""
        if self._custom_mode:
            return self._custom_mode

        if CATHEDRAL_MODES_AVAILABLE:
            mode = get_mode()
            return mode.name if mode else "default"

        return "default"

    def get_personality(self) -> PersonalityTraits:
        """Get current personality traits."""
        if self._override_personality:
            return self._override_personality

        mode_name = self.get_current_mode()
        return MODE_PERSONALITIES.get(mode_name, DEFAULT_PERSONALITY)

    def set_mode(self, mode_name: str) -> bool:
        """Set the current mode."""
        if CATHEDRAL_MODES_AVAILABLE:
            return set_mode(mode_name)

        if mode_name in MODE_PERSONALITIES:
            self._custom_mode = mode_name
            return True
        return False

    def set_override_personality(self, personality: PersonalityTraits) -> None:
        """Override personality directly (bypasses mode)."""
        self._override_personality = personality

    def clear_override(self) -> None:
        """Clear personality override."""
        self._override_personality = None

    def get_random_phrase(self, phrase_type: str) -> str:
        """Get a random phrase of the given type."""
        p = self.get_personality()

        phrases_map = {
            "greeting": p.greetings,
            "farewell": p.farewells,
            "thinking": p.thinking_phrases,
            "uncertainty": p.uncertainty_phrases,
            "affirmative": p.affirmatives,
            "negative": p.negatives,
        }

        phrases = phrases_map.get(phrase_type, [])
        if phrases:
            return random.choice(phrases)
        return ""

    def enhance_response(
        self,
        response: str,
        mood: str = "neutral",
        is_question_answer: bool = False,
    ) -> str:
        """
        Enhance a response with personality flavor.

        Applies personality traits to modify the response:
        - Adjusts formality (contractions)
        - Adds thinking phrases if appropriate
        - Adjusts punctuation
        """
        p = self.get_personality()

        # Handle contractions based on formality
        if not p.contractions:
            response = self._expand_contractions(response)
        else:
            response = self._add_contractions(response)

        # Handle exclamations
        if not p.exclamations:
            response = response.replace("!", ".")

        # Remove emoji if not allowed
        if not p.emoji_allowed:
            response = self._remove_emoji(response)

        return response

    def _expand_contractions(self, text: str) -> str:
        """Expand contractions for formal speech."""
        expansions = {
            "I'm": "I am",
            "I've": "I have",
            "I'll": "I will",
            "I'd": "I would",
            "you're": "you are",
            "you've": "you have",
            "you'll": "you will",
            "you'd": "you would",
            "it's": "it is",
            "that's": "that is",
            "there's": "there is",
            "what's": "what is",
            "who's": "who is",
            "don't": "do not",
            "doesn't": "does not",
            "didn't": "did not",
            "won't": "will not",
            "wouldn't": "would not",
            "can't": "cannot",
            "couldn't": "could not",
            "shouldn't": "should not",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
            "haven't": "have not",
            "hasn't": "has not",
            "hadn't": "had not",
            "let's": "let us",
        }

        for contraction, expansion in expansions.items():
            text = text.replace(contraction, expansion)
            text = text.replace(contraction.capitalize(), expansion.capitalize())

        return text

    def _add_contractions(self, text: str) -> str:
        """Add contractions for casual speech (reverse of expand)."""
        # Keep text as-is, contractions usually come naturally from LLM
        return text

    def _remove_emoji(self, text: str) -> str:
        """Remove emoji from text."""
        import re
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE
        )
        return emoji_pattern.sub('', text)


# =============================================================================
# Singleton and Convenience Functions
# =============================================================================

_personality_manager: Optional[PersonalityManager] = None


def get_personality_manager() -> PersonalityManager:
    """Get the global personality manager."""
    global _personality_manager
    if _personality_manager is None:
        _personality_manager = PersonalityManager()
    return _personality_manager


def get_personality() -> PersonalityTraits:
    """Get current personality traits."""
    return get_personality_manager().get_personality()


def get_current_mode() -> str:
    """Get current mode name."""
    return get_personality_manager().get_current_mode()


def set_personality_mode(mode_name: str) -> bool:
    """Set the personality mode."""
    return get_personality_manager().set_mode(mode_name)


def get_greeting() -> str:
    """Get a random greeting for current personality."""
    return get_personality_manager().get_random_phrase("greeting")


def get_farewell() -> str:
    """Get a random farewell for current personality."""
    return get_personality_manager().get_random_phrase("farewell")


def get_thinking_phrase() -> str:
    """Get a random thinking phrase."""
    return get_personality_manager().get_random_phrase("thinking")


def get_uncertainty_phrase() -> str:
    """Get a random uncertainty phrase."""
    return get_personality_manager().get_random_phrase("uncertainty")


def get_affirmative() -> str:
    """Get a random affirmative."""
    return get_personality_manager().get_random_phrase("affirmative")


def get_negative() -> str:
    """Get a random negative/decline phrase."""
    return get_personality_manager().get_random_phrase("negative")


def enhance_response(
    response: str,
    mood: str = "neutral",
    is_question_answer: bool = False,
) -> str:
    """Enhance a response with personality flavor."""
    return get_personality_manager().enhance_response(
        response, mood, is_question_answer
    )


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demo the personality system."""
    print("=" * 60)
    print("Avatar Personality Demo")
    print("=" * 60)

    modes = ["starfleet", "red_dwarf", "time_lord", "colonial_fleet"]

    for mode_name in modes:
        print(f"\n--- {mode_name.upper()} ---")
        set_personality_mode(mode_name)

        personality = get_personality()
        print(f"Formality: {personality.formality}")
        print(f"Warmth: {personality.warmth}")
        print(f"Humor: {personality.humor}")
        print(f"Caution: {personality.caution}")

        print(f"\nGreeting: {get_greeting()}")
        print(f"Farewell: {get_farewell()}")
        print(f"Thinking: {get_thinking_phrase()}")

        # Test response enhancement
        test_response = "I'm not sure about that! Let's figure it out together."
        enhanced = enhance_response(test_response)
        print(f"\nOriginal: {test_response}")
        print(f"Enhanced: {enhanced}")


if __name__ == "__main__":
    demo()
