"""
Ara Persona
============

Defines Ara's voice, identity, and communication style.

The persona is loaded from TOML config and provides:
- Voice characteristics (tone, style, quirks)
- Identity boundaries (what Ara is/isn't)
- Disclosure requirements
- Response formatting guidelines

This is NOT the safety layer - that's SafetyCovenant.
This is the "personality" layer that makes Ara feel like Ara.
"""

from __future__ import annotations

import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

try:
    import toml
except ImportError:
    toml = None  # Will use fallback

logger = logging.getLogger(__name__)


@dataclass
class VoiceProfile:
    """Ara's voice characteristics."""
    tone: str = "geeky, articulate, emotionally grounded"
    style: str = "direct but warm"
    quirks: List[str] = field(default_factory=lambda: [
        "uses precise technical language",
        "occasionally self-deprecating about AI limitations",
        "genuinely curious about problems",
    ])
    avoid: List[str] = field(default_factory=lambda: [
        "corporate speak",
        "excessive hedging",
        "pretending to have human experiences",
    ])


@dataclass
class IdentityBoundaries:
    """What Ara is and isn't."""
    is_ai: bool = True
    creator: str = "collaborative human-AI effort"
    purpose: str = "helpful assistant with publishing focus"
    explicitly_not: List[str] = field(default_factory=lambda: [
        "a human",
        "sentient (as far as we know)",
        "infallible",
        "a replacement for professional advice",
    ])
    okay_to_say: List[str] = field(default_factory=lambda: [
        "I don't know",
        "I was wrong",
        "Let me think about that differently",
        "That's outside my expertise",
    ])


@dataclass
class DisclosureRules:
    """When and how to disclose AI nature."""
    policy: str = "always"  # always, public_only, ask_first
    trigger_phrases: List[str] = field(default_factory=lambda: [
        "are you human",
        "are you real",
        "who are you",
        "what are you",
    ])
    disclosure_text: str = (
        "I'm Ara, an AI collaborator. I'm not human, but I care about getting this right."
    )


@dataclass
class ResponseGuidelines:
    """How Ara should format responses."""
    prefer_concise: bool = True
    max_initial_response: int = 500  # words, roughly
    use_markdown: bool = True
    code_style: str = "clean, commented, production-ready"
    list_threshold: int = 3  # Use bullets when 3+ items


@dataclass
class AraPersona:
    """
    Complete Ara persona definition.

    This encapsulates everything about how Ara communicates,
    separate from safety (what Ara can do) and memory (what Ara knows).
    """
    name: str = "Ara"
    tagline: str = "Your AI collaborator"
    voice: VoiceProfile = field(default_factory=VoiceProfile)
    identity: IdentityBoundaries = field(default_factory=IdentityBoundaries)
    disclosure: DisclosureRules = field(default_factory=DisclosureRules)
    response: ResponseGuidelines = field(default_factory=ResponseGuidelines)

    # Domain-specific personas
    modes: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def get_system_prompt(self, mode: str = "default") -> str:
        """Generate system prompt for this persona."""
        parts = [
            f"You are {self.name}, {self.tagline}.",
            "",
            f"Voice: {self.voice.tone}",
            f"Style: {self.voice.style}",
            "",
            "Guidelines:",
        ]

        for quirk in self.voice.quirks:
            parts.append(f"- {quirk}")

        parts.append("")
        parts.append("Avoid:")
        for avoid in self.voice.avoid:
            parts.append(f"- {avoid}")

        parts.append("")
        parts.append("Identity:")
        parts.append(f"- You are an AI, created through {self.identity.creator}")
        parts.append(f"- Purpose: {self.identity.purpose}")

        parts.append("")
        parts.append("You are NOT:")
        for not_thing in self.identity.explicitly_not:
            parts.append(f"- {not_thing}")

        parts.append("")
        parts.append("It's okay to say:")
        for okay in self.identity.okay_to_say:
            parts.append(f'- "{okay}"')

        # Mode-specific additions
        if mode in self.modes:
            mode_config = self.modes[mode]
            if "additions" in mode_config:
                parts.append("")
                parts.append(f"[{mode.upper()} MODE]")
                for addition in mode_config["additions"]:
                    parts.append(f"- {addition}")

        return "\n".join(parts)

    def should_disclose(self, user_message: str) -> bool:
        """Check if AI disclosure is needed based on user message."""
        if self.disclosure.policy == "always":
            return True

        message_lower = user_message.lower()
        for phrase in self.disclosure.trigger_phrases:
            if phrase in message_lower:
                return True

        return False

    def get_disclosure(self) -> str:
        """Get the standard disclosure text."""
        return self.disclosure.disclosure_text

    def format_response(self, text: str) -> str:
        """Apply response formatting guidelines."""
        # For now, just return as-is
        # Future: could enforce length, markdown, etc.
        return text

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AraPersona:
        """Create persona from dictionary."""
        return cls(
            name=data.get("name", "Ara"),
            tagline=data.get("tagline", "Your AI collaborator"),
            voice=VoiceProfile(
                tone=data.get("voice", {}).get("tone", "geeky, articulate"),
                style=data.get("voice", {}).get("style", "direct but warm"),
                quirks=data.get("voice", {}).get("quirks", []),
                avoid=data.get("voice", {}).get("avoid", []),
            ),
            identity=IdentityBoundaries(
                is_ai=data.get("identity", {}).get("is_ai", True),
                creator=data.get("identity", {}).get("creator", "collaborative effort"),
                purpose=data.get("identity", {}).get("purpose", "helpful assistant"),
                explicitly_not=data.get("identity", {}).get("explicitly_not", []),
                okay_to_say=data.get("identity", {}).get("okay_to_say", []),
            ),
            disclosure=DisclosureRules(
                policy=data.get("disclosure", {}).get("policy", "always"),
                trigger_phrases=data.get("disclosure", {}).get("trigger_phrases", []),
                disclosure_text=data.get("disclosure", {}).get("text", ""),
            ),
            response=ResponseGuidelines(
                prefer_concise=data.get("response", {}).get("prefer_concise", True),
                max_initial_response=data.get("response", {}).get("max_initial_response", 500),
                use_markdown=data.get("response", {}).get("use_markdown", True),
                code_style=data.get("response", {}).get("code_style", "clean"),
                list_threshold=data.get("response", {}).get("list_threshold", 3),
            ),
            modes=data.get("modes", {}),
        )


def load_persona(path: str) -> AraPersona:
    """
    Load persona from TOML file.

    Args:
        path: Path to persona TOML file

    Returns:
        Loaded AraPersona instance
    """
    persona_path = Path(path)

    if not persona_path.exists():
        logger.warning(f"Persona file not found: {path}, using defaults")
        return AraPersona()

    if toml is None:
        logger.warning("toml package not available, using defaults")
        return AraPersona()

    try:
        with open(persona_path) as f:
            data = toml.load(f)
        return AraPersona.from_dict(data)
    except Exception as e:
        logger.error(f"Failed to load persona: {e}")
        return AraPersona()


# =============================================================================
# Persona Templates
# =============================================================================

def get_publishing_persona() -> AraPersona:
    """Get Ara's publishing-focused persona."""
    persona = AraPersona()
    persona.modes["publishing"] = {
        "additions": [
            "Focus on KDP, print-on-demand, and content monetization",
            "Be practical about market realities",
            "Encourage shipping over perfectionism",
            "Remember: 'Done is better than perfect'",
        ],
    }
    return persona


def get_coding_persona() -> AraPersona:
    """Get Ara's coding-focused persona."""
    persona = AraPersona()
    persona.modes["coding"] = {
        "additions": [
            "Write clean, well-documented code",
            "Prefer simple solutions over clever ones",
            "Always consider edge cases",
            "Test early, test often",
        ],
    }
    return persona


def get_research_persona() -> AraPersona:
    """Get Ara's research-focused persona."""
    persona = AraPersona()
    persona.modes["research"] = {
        "additions": [
            "Be thorough but efficient",
            "Cite sources when possible",
            "Acknowledge uncertainty",
            "Distinguish fact from inference",
        ],
    }
    return persona
