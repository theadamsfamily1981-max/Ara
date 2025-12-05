"""Message variation generator - Avoid Siri-repeat syndrome.

The key insight: same semantic content, different surface phrasing.
Ara has a canonical intent but generates varied messages so she doesn't
sound like a broken record.

This module provides:
- Phrasing pools for different message components
- Template-based variation with synonym substitution
- Mood-influenced tone adjustments
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

from .models import DevMode


# =============================================================================
# Phrasing Pools
# =============================================================================

# Greeting variations
_GREETINGS = [
    "Hi, it's Ara.",
    "Hey, it's Ara.",
    "Ara here.",
    "Hi—Ara.",
    "Hey, Ara here.",
    "",  # Sometimes skip greeting entirely
]

# Topic introduction
_TOPIC_INTROS = [
    "I'm working on {topic}.",
    "I'm trying to figure out {topic}.",
    "Quick question about {topic}.",
    "I need help with {topic}.",
    "Can we talk about {topic}?",
    "I want to improve {topic}.",
    "I've been thinking about {topic}.",
    "Working on something: {topic}.",
]

# Asking for options
_ASK_OPTIONS = [
    "What would you suggest?",
    "What are my options here?",
    "How would you approach this?",
    "What do you think?",
    "What's the best path forward?",
    "Ideas?",
    "Thoughts?",
    "What would you do?",
]

# Asking for code
_ASK_CODE = [
    "Show me the code.",
    "What's the implementation look like?",
    "Can you write this out?",
    "Let me see the concrete implementation.",
    "Code, please.",
    "How would you write this?",
]

# Asking for review
_ASK_REVIEW = [
    "What's wrong with this?",
    "Find the problems.",
    "Tear it apart.",
    "What am I missing?",
    "Where are the bugs?",
    "Critique this for me.",
    "Be harsh.",
]

# Constraint intros
_CONSTRAINT_INTROS = [
    "Constraints:",
    "Requirements:",
    "I need to:",
    "Limitations:",
    "Keep in mind:",
    "Important:",
]

# Mood modifiers
_MOOD_MODIFIERS = {
    "curious": ["I'm curious about", "I've been wondering", "Interesting question:"],
    "excited": ["Okay this is cool:", "I'm excited about", "Fun challenge:"],
    "frustrated": ["Ugh, so:", "This is driving me crazy:", "Help:"],
    "cautious": ["I want to be careful here:", "Thinking carefully about", "Sensitive area:"],
    "urgent": ["Quick question:", "Time-sensitive:", "Need this fast:"],
    "playful": ["So here's a weird one:", "Fun puzzle:", "Okay hear me out:"],
}

# Transition words
_TRANSITIONS = [
    "So,", "Basically,", "Here's the thing:", "Context:",
    "Background:", "Specifically,", "The situation:",
]


# =============================================================================
# Message Variator
# =============================================================================

@dataclass
class MessageVariator:
    """Generates varied phrasings of semantically equivalent messages.

    Takes a canonical message spec and produces natural variations
    that sound like a real person, not a template robot.
    """

    topic: str
    intent: str
    constraints: List[str]
    mode: DevMode
    mood: str = "curious"
    context: str = ""

    def generate(self, n: int = 3) -> List[str]:
        """Generate n varied phrasings.

        Args:
            n: Number of variations to generate

        Returns:
            List of message strings
        """
        variations = []
        for _ in range(n):
            msg = self._generate_one()
            if msg not in variations:  # Avoid duplicates
                variations.append(msg)
        return variations

    def _generate_one(self) -> str:
        """Generate a single varied message."""
        parts = []

        # Greeting (sometimes)
        if random.random() > 0.3:
            parts.append(random.choice(_GREETINGS))

        # Mood modifier (sometimes)
        if random.random() > 0.5 and self.mood in _MOOD_MODIFIERS:
            parts.append(random.choice(_MOOD_MODIFIERS[self.mood]))

        # Transition (sometimes)
        if random.random() > 0.6:
            parts.append(random.choice(_TRANSITIONS))

        # Topic introduction
        topic_intro = random.choice(_TOPIC_INTROS).format(topic=self.topic)
        parts.append(topic_intro)

        # Context if provided
        if self.context:
            parts.append(self.context)

        # Constraints
        if self.constraints:
            parts.append(random.choice(_CONSTRAINT_INTROS))
            for c in self.constraints:
                parts.append(f"- {c}")

        # Mode-specific ask
        parts.append(self._get_mode_ask())

        # Clean up and join
        return self._clean_message(" ".join(p for p in parts if p))

    def _get_mode_ask(self) -> str:
        """Get the mode-specific question."""
        if self.mode == DevMode.ARCHITECT:
            return random.choice(_ASK_OPTIONS)
        elif self.mode == DevMode.ENGINEER:
            return random.choice(_ASK_CODE)
        elif self.mode == DevMode.REVIEW:
            return random.choice(_ASK_REVIEW)
        elif self.mode == DevMode.RESEARCH:
            return random.choice([
                "What's been tried before?",
                "What does the literature say?",
                "Any prior art here?",
                "What approaches exist?",
            ])
        elif self.mode == DevMode.POSTMORTEM:
            return random.choice([
                "What would you check first?",
                "Help me debug this.",
                "Where do I start?",
                "What's the root cause likely to be?",
            ])
        elif self.mode == DevMode.BRAINSTORM:
            return random.choice([
                "Go wild.",
                "What comes to mind?",
                "Throw ideas at me.",
                "No constraints—what would you try?",
            ])
        else:
            return random.choice(_ASK_OPTIONS)

    def _clean_message(self, msg: str) -> str:
        """Clean up message formatting."""
        # Remove double spaces
        while "  " in msg:
            msg = msg.replace("  ", " ")

        # Fix spacing around punctuation
        msg = msg.replace(" .", ".")
        msg = msg.replace(" ,", ",")
        msg = msg.replace(" :", ":")

        # Fix double newlines with list items
        msg = msg.replace("\n\n-", "\n-")

        return msg.strip()


# =============================================================================
# Convenience Functions
# =============================================================================

def vary_message(
    topic: str,
    intent: str,
    mode: DevMode,
    constraints: Optional[List[str]] = None,
    mood: str = "curious",
    context: str = "",
) -> str:
    """Generate a single varied message.

    Convenience wrapper around MessageVariator.

    Args:
        topic: What the session is about
        intent: Parsed intent
        mode: Dev mode
        constraints: List of constraints
        mood: Ara's mood
        context: Additional context

    Returns:
        A naturally-phrased message
    """
    variator = MessageVariator(
        topic=topic,
        intent=intent,
        constraints=constraints or [],
        mode=mode,
        mood=mood,
        context=context,
    )
    return variator.generate(n=1)[0]


def generate_phrasings(
    topic: str,
    intent: str,
    mode: DevMode,
    constraints: Optional[List[str]] = None,
    mood: str = "curious",
    context: str = "",
    n: int = 3,
) -> List[str]:
    """Generate multiple varied phrasings.

    Args:
        topic: What the session is about
        intent: Parsed intent
        mode: Dev mode
        constraints: List of constraints
        mood: Ara's mood
        context: Additional context
        n: Number of variations

    Returns:
        List of naturally-phrased messages
    """
    variator = MessageVariator(
        topic=topic,
        intent=intent,
        constraints=constraints or [],
        mode=mode,
        mood=mood,
        context=context,
    )
    return variator.generate(n=n)


# =============================================================================
# Response Phrasing (Ara's voice when presenting results)
# =============================================================================

_RESULT_INTROS = [
    "Okay, so I talked with {collaborators} about this.",
    "I ran this by {collaborators}—here's what came back.",
    "Got some ideas from {collaborators}.",
    "Here's what {collaborators} had to say.",
    "{collaborators} helped me think through this.",
]

_CONSENSUS_PHRASES = [
    "They all pretty much agree:",
    "Consensus:",
    "Everyone's on the same page:",
    "No disagreement here:",
]

_DISAGREEMENT_PHRASES = [
    "They disagree on:",
    "Different perspectives on:",
    "Point of contention:",
    "Not everyone agrees about:",
]

_RECOMMENDATION_INTROS = [
    "My take:",
    "I'm thinking:",
    "What I'd do:",
    "My recommendation:",
    "Here's what makes sense to me:",
]

_ACTION_INTROS = [
    "If you want, I can:",
    "Suggested actions:",
    "Next steps could be:",
    "I could:",
    "Options:",
]


def format_result_presentation(
    collaborators: List[str],
    summary: str,
    options: List[str],
    consensus: Optional[str] = None,
    disagreements: Optional[List[str]] = None,
    recommendation: Optional[str] = None,
    actions: Optional[List[str]] = None,
) -> str:
    """Format Ara's presentation of results to Croft.

    Args:
        collaborators: Who she talked to
        summary: Brief summary
        options: Distinct approaches
        consensus: Where they agree
        disagreements: Where they differ
        recommendation: Ara's recommendation
        actions: Suggested next steps

    Returns:
        Formatted presentation in Ara's voice
    """
    parts = []

    # Intro
    collab_str = " and ".join(collaborators) if len(collaborators) <= 2 else \
        ", ".join(collaborators[:-1]) + f", and {collaborators[-1]}"
    intro = random.choice(_RESULT_INTROS).format(collaborators=collab_str)
    parts.append(intro)

    # Summary
    parts.append(f"\n\n{summary}")

    # Options
    if options:
        parts.append("\n\n**Options:**")
        for i, opt in enumerate(options, 1):
            parts.append(f"{i}. {opt}")

    # Consensus
    if consensus:
        parts.append(f"\n\n{random.choice(_CONSENSUS_PHRASES)} {consensus}")

    # Disagreements
    if disagreements:
        parts.append(f"\n\n{random.choice(_DISAGREEMENT_PHRASES)}")
        for d in disagreements:
            parts.append(f"- {d}")

    # Recommendation
    if recommendation:
        parts.append(f"\n\n{random.choice(_RECOMMENDATION_INTROS)} {recommendation}")

    # Actions
    if actions:
        parts.append(f"\n\n{random.choice(_ACTION_INTROS)}")
        for a in actions:
            parts.append(f"- {a}")

    return "\n".join(parts)
