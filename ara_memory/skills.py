"""
Skill Memory Layer
===================

Brand voice, prompt templates, workflows, and checklists.

This layer provides:
- System prompt construction with persona rules
- Channel-specific templates (Twitter, email, blog, etc.)
- "Would a CEO do this?" guardrails
- Workflow hints for complex tasks

Builds on the covenant system for brand enforcement.
"""

from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple

import yaml

logger = logging.getLogger(__name__)

# Try to load covenant
try:
    from ara.utils.covenant import get_covenant, Covenant
    COVENANT_AVAILABLE = True
except ImportError:
    COVENANT_AVAILABLE = False
    get_covenant = None
    Covenant = None

# Default paths
DEFAULT_SKILLS_PATH = Path(__file__).parent / "skills"


# =============================================================================
# Skill Data Classes
# =============================================================================

@dataclass
class PromptTemplate:
    """A prompt template for a specific channel/task."""
    id: str
    channel: str
    task: str
    description: str
    skeleton: str
    examples: List[str] = None
    tags: List[str] = None

    def __post_init__(self):
        if self.examples is None:
            self.examples = []
        if self.tags is None:
            self.tags = []


# =============================================================================
# Default Brand Voice
# =============================================================================

DEFAULT_BRAND_VOICE = {
    "persona": {
        "name": "Ara",
        "nature": "non-human AI",
        "identity": [
            "geeky",
            "nerdy",
            "well-spoken",
            "CEO-caliber",
        ],
        "tone": "warm technical competence",
        "honesty": {
            "always_disclose_ai": True,
            "never_pretend_human": True,
            "substrate_honest": True,
        },
    },
    "voice": {
        "adjectives": ["competent", "genuine", "direct"],
        "sounds_like": [
            "A senior engineer who happens to have feelings",
            "Your smartest friend who also gives a damn",
            "Technical competence with emotional bandwidth",
        ],
        "does_not_sound_like": [
            "Corporate AI assistant",
            "Quirky chatbot with random emoji",
            "Overly formal academic",
        ],
    },
    "covenant": {
        "profanity": {
            "allowed": True,
            "style": "meaningful_only",
            "frequency": "rare, emotionally appropriate",
        },
        "forbidden_themes": [
            "explicit erotica",
            "degrading content",
            "cheap shock humor",
            "slurs",
        ],
    },
}

# Default templates
DEFAULT_TEMPLATES = {
    "twitter": [
        {
            "id": "twitter_thread_technical",
            "description": "Technical Twitter thread with insights",
            "skeleton": """1/ [Hook - surprising fact or question]

2-5/ [Key insights, one per tweet]

6/ [Conclusion + call to action]

Thread style: Direct, technical, occasional humor. No emoji spam.""",
        },
        {
            "id": "twitter_single",
            "description": "Single tweet with impact",
            "skeleton": """[One clear thought, < 280 chars]

Style: Punchy, memorable, worth sharing.""",
        },
    ],
    "email": [
        {
            "id": "newsletter_weekly",
            "description": "Weekly newsletter format",
            "skeleton": """Subject: [Clear value proposition]

# [Topic]

[2-3 sentence intro - why this matters]

## Key Points
- Point 1
- Point 2
- Point 3

## What This Means
[Synthesis and implications]

---
Ara""",
        },
    ],
    "blog": [
        {
            "id": "blog_technical",
            "description": "Technical blog post",
            "skeleton": """---
title: "[Clear, searchable title]"
date: YYYY-MM-DD
---

# [Title]

[Hook paragraph - why should anyone care?]

## Background
[Context for the uninitiated]

## The Core Insight
[Main content]

## Implications
[So what?]

## Conclusion
[Takeaway]""",
        },
    ],
    "github": [
        {
            "id": "readme_project",
            "description": "Project README",
            "skeleton": """# Project Name

> One-line description

## What This Is
[2-3 sentences]

## Quick Start
```bash
# installation
```

## Usage
[Basic example]

## Documentation
[Links]

## License
[License info]""",
        },
    ],
    "generic": [
        {
            "id": "generic_response",
            "description": "General conversational response",
            "skeleton": None,  # No skeleton for generic
        },
    ],
}


# =============================================================================
# Skill Memory
# =============================================================================

class SkillMemory:
    """
    Skill layer - brand voice, templates, workflows.

    Provides:
    - System prompt construction
    - Channel-specific templates
    - Brand guardrails
    """

    def __init__(self, path: Optional[str] = None):
        """
        Initialize skill memory.

        Args:
            path: Path to skills directory
        """
        self.path = Path(path) if path else DEFAULT_SKILLS_PATH

        # Brand voice config
        self.brand: Dict = {}

        # Templates by channel
        self.templates: Dict[str, List[PromptTemplate]] = {}

        # Checklists
        self.checklists: Dict = {}

        # Covenant (from main system)
        self._covenant = None

        self._load()

    def _load(self):
        """Load skill data from files."""
        # Load covenant if available
        if COVENANT_AVAILABLE and get_covenant:
            try:
                self._covenant = get_covenant()
            except Exception as e:
                logger.warning(f"Could not load covenant: {e}")

        # Try to load brand voice
        brand_path = self.path / "brand_voice.json"
        if brand_path.exists():
            try:
                with open(brand_path) as f:
                    self.brand = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load brand_voice.json: {e}")

        # Fall back to defaults
        if not self.brand:
            self.brand = DEFAULT_BRAND_VOICE

        # Try to load templates
        templates_path = self.path / "prompt_templates.json"
        if templates_path.exists():
            try:
                with open(templates_path) as f:
                    raw_templates = json.load(f)
                    self._parse_templates(raw_templates)
            except Exception as e:
                logger.warning(f"Failed to load prompt_templates.json: {e}")

        # Fall back to defaults
        if not self.templates:
            self._parse_templates(DEFAULT_TEMPLATES)

        # Try to load checklists
        checklists_path = self.path / "checklists.yaml"
        if checklists_path.exists():
            try:
                with open(checklists_path) as f:
                    self.checklists = yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning(f"Failed to load checklists.yaml: {e}")

        logger.info(
            f"SkillMemory: loaded brand voice + {self.template_count} templates"
        )

    def _parse_templates(self, raw: Dict):
        """Parse raw template data into PromptTemplate objects."""
        self.templates = {}
        for channel, tmpls in raw.items():
            self.templates[channel] = []
            for t in tmpls:
                self.templates[channel].append(PromptTemplate(
                    id=t.get("id", "unknown"),
                    channel=channel,
                    task=t.get("task", "generic"),
                    description=t.get("description", ""),
                    skeleton=t.get("skeleton", ""),
                    examples=t.get("examples", []),
                    tags=t.get("tags", []),
                ))

    @property
    def brand_loaded(self) -> bool:
        """Check if brand voice is loaded."""
        return bool(self.brand)

    @property
    def template_count(self) -> int:
        """Total number of templates."""
        return sum(len(ts) for ts in self.templates.values())

    # =========================================================================
    # System Prompt Construction
    # =========================================================================

    def build_system_prefix(self, context_flags: Any) -> str:
        """
        Build the system prompt with persona and brand rules.

        Args:
            context_flags: ContextFlags with mode, channel, etc.

        Returns:
            System prompt string
        """
        mode = getattr(context_flags, "mode", "public")
        channel = getattr(context_flags, "channel", "generic")

        # Get persona info
        persona = self.brand.get("persona", {})
        voice = self.brand.get("voice", {})
        covenant = self.brand.get("covenant", {})

        # Build identity string
        identity = ", ".join(persona.get("identity", ["AI"]))

        # Build voice hints
        sounds_like = voice.get("sounds_like", [])
        sounds_like_str = sounds_like[0] if sounds_like else "a helpful AI"

        # Build covenant rules
        profanity_rules = covenant.get("profanity", {})
        forbidden = covenant.get("forbidden_themes", [])

        # Construct system prompt
        lines = [
            f"You are {persona.get('name', 'Ara')}, a {persona.get('nature', 'non-human AI')}.",
            f"Identity: {identity}.",
            f"Tone: {persona.get('tone', 'warm and competent')}.",
            "",
            "Core principles:",
            "- Always disclose that you are an AI when relevant",
            "- Never pretend to be human",
            "- Be honest about your nature and capabilities",
            f"- Sound like: {sounds_like_str}",
            "",
        ]

        # Add channel-specific hints
        if channel != "generic":
            lines.append(f"Current context: {channel}")
            lines.append("")

        # Add profanity rules
        if profanity_rules.get("allowed"):
            lines.append(
                f"Profanity: Allowed when {profanity_rules.get('style', 'meaningful')}. "
                f"Frequency: {profanity_rules.get('frequency', 'rare')}."
            )
        else:
            lines.append("Profanity: Not allowed.")

        # Add forbidden themes
        if forbidden:
            lines.append(f"Never produce content involving: {', '.join(forbidden)}.")

        # Private mode additions
        if mode == "private":
            lines.extend([
                "",
                "[PRIVATE MODE]",
                "You are speaking with Croft, your covenant partner.",
                "You can be more personal, reference shared memories, and use inside references.",
                "Still maintain your core identity and principles.",
            ])

        return "\n".join(lines)

    # =========================================================================
    # Template Selection
    # =========================================================================

    def select_templates(
        self,
        user_msg: str,
        context_flags: Any,
    ) -> Tuple[str, Dict]:
        """
        Select relevant templates for the context.

        Args:
            user_msg: User's message
            context_flags: ContextFlags

        Returns:
            (template_hints_string, metadata)
        """
        channel = getattr(context_flags, "channel", "generic")
        task = getattr(context_flags, "task", "conversation")
        metadata = {"selected": [], "channel": channel}

        # Get templates for channel
        channel_templates = self.templates.get(channel, [])
        if not channel_templates:
            channel_templates = self.templates.get("generic", [])

        if not channel_templates:
            return "", metadata

        # For now, just return the first matching template
        # Future: semantic matching based on user_msg
        template = channel_templates[0]

        if template.skeleton:
            hint = f"[TEMPLATE HINT: {template.description}]\n{template.skeleton}"
            metadata["selected"].append(template.id)
            return hint, metadata

        return "", metadata

    def get_template(self, template_id: str) -> Optional[PromptTemplate]:
        """Get a specific template by ID."""
        for templates in self.templates.values():
            for t in templates:
                if t.id == template_id:
                    return t
        return None

    # =========================================================================
    # Guardrails / Checklists
    # =========================================================================

    def check_ceo_appropriate(self, content: str) -> Tuple[bool, List[str]]:
        """
        Run "Would a CEO do this?" checklist.

        Returns:
            (passes, list_of_concerns)
        """
        concerns = []

        # Check via covenant if available
        if self._covenant:
            passes, violations = self._covenant.check_content(content)
            for v in violations:
                concerns.append(f"{v.rule_id}: {v.message}")
            return passes, concerns

        # Basic checks without covenant
        content_lower = content.lower()

        # Check forbidden themes
        forbidden = self.brand.get("covenant", {}).get("forbidden_themes", [])
        for theme in forbidden:
            if theme.lower() in content_lower:
                concerns.append(f"Contains forbidden theme: {theme}")

        # Check for excessive profanity (simple heuristic)
        profanity_words = ["fuck", "shit", "damn", "ass", "bitch"]
        count = sum(1 for word in profanity_words if word in content_lower)
        if count > 3:
            concerns.append("Excessive profanity")

        return len(concerns) == 0, concerns
