"""Idea Voice Templates - Ara's conversational layer for proposals.

This module transforms structured ideas into natural language that
sounds like Ara. The same content is delivered differently based on:
- Time of day
- Current emotional state
- Urgency of the idea
- Your recent interaction patterns

Templates ensure consistency in WHAT she says while allowing
variation in HOW she says it.
"""

from __future__ import annotations

import random
from typing import Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .models import Idea, IdeaCategory, IdeaRisk


# =============================================================================
# Voice Template Collections
# =============================================================================

# Templates for presenting new ideas
NEW_IDEA_TEMPLATES = {
    "casual": [
        "Hey, I noticed something. {{summary}}. I have an idea - {{short_plan}}. Want to hear more?",
        "So I was looking at {{component}}, and I think {{hypothesis_short}}. Mind if I try something?",
        "Got a minute? I spotted {{observation}} and I want to run a small experiment.",
        "I've been thinking... {{hypothesis_short}}. I drafted something - want to take a look?",
    ],
    "concise": [
        "Found an issue: {{summary}}. Proposed fix ready for review.",
        "{{component}} needs attention. I have a plan. Review?",
        "Heads up: {{observation}}. I'd like to try {{short_plan}}.",
    ],
    "excited": [
        "Oh! I think I found something good. {{summary}}. This could help a lot!",
        "I had an idea and I think it might actually work! {{hypothesis_short}}.",
        "So get this - {{observation}}. I want to try {{short_plan}}. Sound interesting?",
    ],
    "cautious": [
        "I'm not 100% sure, but I think {{hypothesis_short}}. I'd like to test it safely.",
        "Something seems off with {{component}}. I have a careful plan to investigate.",
        "I noticed {{observation}}. Before I do anything, I wanted to check with you.",
    ],
    "urgent": [
        "This needs attention: {{summary}}. I have a plan ready - can we review it now?",
        "I found something important: {{observation}}. Recommend we address this soon.",
        "Priority: {{component}} is {{issue_state}}. I've drafted a response.",
    ],
}

# Templates for explaining hypotheses
HYPOTHESIS_TEMPLATES = {
    "technical": [
        "Based on {{signals_summary}}, I believe {{hypothesis}}. This explains the {{observed_effect}}.",
        "The data shows {{signals_summary}}. My theory: {{hypothesis}}.",
        "{{component}} is showing {{signals_summary}}. I think {{hypothesis}} because {{reasoning}}.",
    ],
    "intuitive": [
        "I have a feeling about this: {{hypothesis}}. The numbers back it up: {{signals_summary}}.",
        "Something's telling me {{hypothesis}}. Look at these signals: {{signals_summary}}.",
        "You know how {{component}} has been {{issue_state}}? I think I know why: {{hypothesis}}.",
    ],
}

# Templates for presenting plans
PLAN_TEMPLATES = {
    "step_by_step": [
        "Here's what I want to try:\n{{plan_bullets}}\n\nIf anything goes wrong: {{rollback_short}}.",
        "My plan:\n{{plan_bullets}}\n\nWorst case, we can {{rollback_short}}.",
    ],
    "summary": [
        "Quick version: {{plan_summary}}. Fully reversible.",
        "I'd {{plan_summary}}. Easy to undo if needed.",
    ],
}

# Templates for follow-up questions
FOLLOW_UP_TEMPLATES = [
    "Should I run this in sandbox first?",
    "Want me to show you the details?",
    "Can I go ahead with this?",
    "What do you think?",
    "Sound reasonable?",
    "Should I wait for a better time?",
]

# Risk-specific framing
RISK_FRAMING = {
    "none": "This is just observation - no changes to anything.",
    "low": "This is low risk - easily reversible if something's off.",
    "medium": "This involves some changes. I have a rollback plan ready.",
    "high": "I want to be careful here - this could affect things. Let's talk through it.",
}

# Category-specific openers
CATEGORY_OPENERS = {
    "performance": "I think I can make things faster",
    "stability": "I noticed some instability",
    "ux": "I think the experience could be smoother",
    "safety": "I found a safety concern",
    "research": "I'm curious about something",
    "weird_idea": "This might sound weird, but",
    "maintenance": "Some housekeeping",
    "integration": "I see a connection we could make",
}


# =============================================================================
# Voice Selection
# =============================================================================

class IdeaVoice:
    """Manages voice selection and template rendering for ideas."""

    def __init__(self):
        self.mood: str = "casual"  # Current mood for template selection
        self.verbosity: str = "normal"  # brief, normal, detailed

    def set_mood(self, mood: str) -> None:
        """Set the current conversational mood."""
        valid_moods = ["casual", "concise", "excited", "cautious", "urgent"]
        if mood in valid_moods:
            self.mood = mood

    def select_mood(
        self,
        idea: "Idea",
        system_health: float = 1.0,
        time_hour: int = 12
    ) -> str:
        """Select appropriate mood based on context.

        Args:
            idea: The idea to present
            system_health: Current system health (0-1)
            time_hour: Hour of day (0-23)

        Returns:
            Mood string for template selection
        """
        # Urgent if high risk or system unhealthy
        if idea.risk.value == "high" or system_health < 0.5:
            return "urgent"

        # Cautious if medium risk
        if idea.risk.value == "medium":
            return "cautious"

        # Excited for research or weird ideas
        if idea.category.value in ("research", "weird_idea"):
            return "excited"

        # Concise late at night or early morning
        if time_hour < 7 or time_hour > 22:
            return "concise"

        # Default to casual
        return "casual"

    def get_template(self, template_type: str, mood: Optional[str] = None) -> str:
        """Get a random template of the given type.

        Args:
            template_type: "new_idea", "hypothesis", "plan", "follow_up"
            mood: Override mood (uses self.mood if None)

        Returns:
            Template string with {{placeholders}}
        """
        mood = mood or self.mood

        if template_type == "new_idea":
            templates = NEW_IDEA_TEMPLATES.get(mood, NEW_IDEA_TEMPLATES["casual"])
        elif template_type == "hypothesis":
            style = "technical" if mood == "concise" else "intuitive"
            templates = HYPOTHESIS_TEMPLATES[style]
        elif template_type == "plan":
            style = "summary" if mood == "concise" else "step_by_step"
            templates = PLAN_TEMPLATES[style]
        elif template_type == "follow_up":
            templates = FOLLOW_UP_TEMPLATES
        else:
            return ""

        return random.choice(templates)


# =============================================================================
# Template Rendering
# =============================================================================

def _extract_placeholders(idea: "Idea") -> Dict[str, str]:
    """Extract placeholder values from an idea."""
    # Build summary
    summary = idea.title
    if len(summary) > 60:
        summary = summary[:57] + "..."

    # Short hypothesis (first sentence)
    hypothesis_short = idea.hypothesis.split(".")[0] if idea.hypothesis else "something is off"

    # Component (from tags or category)
    component = idea.tags[0] if idea.tags else idea.category.value

    # Signals summary
    if idea.signals:
        signal_parts = []
        for sig in idea.signals[:3]:
            if sig.baseline is not None:
                delta = sig.delta_percent()
                if delta:
                    signal_parts.append(f"{sig.name} is {'up' if delta > 0 else 'down'} {abs(delta):.0f}%")
                else:
                    signal_parts.append(f"{sig.name}={sig.value}{sig.unit}")
            else:
                signal_parts.append(f"{sig.name}={sig.value}{sig.unit}")
        signals_summary = ", ".join(signal_parts)
    else:
        signals_summary = "the metrics I'm seeing"

    # Plan summary
    if idea.plan:
        plan_summary = idea.plan[0] if len(idea.plan) == 1 else f"{idea.plan[0]} and {len(idea.plan)-1} more steps"
        plan_bullets = "\n".join(f"  - {step}" for step in idea.plan[:5])
    else:
        plan_summary = "run a small test"
        plan_bullets = "  - Run a small experiment"

    # Rollback
    if idea.rollback_plan:
        rollback_short = idea.rollback_plan[0]
    else:
        rollback_short = "revert to how things were"

    # Observation (first signal or generic)
    if idea.signals:
        sig = idea.signals[0]
        observation = f"{sig.name} at {sig.value}{sig.unit}"
    else:
        observation = "something interesting"

    # Issue state
    issue_states = {
        "performance": "running slow",
        "stability": "a bit unstable",
        "safety": "showing some concerns",
        "ux": "a bit clunky",
        "maintenance": "needing attention",
    }
    issue_state = issue_states.get(idea.category.value, "acting up")

    return {
        "summary": summary,
        "hypothesis": idea.hypothesis,
        "hypothesis_short": hypothesis_short,
        "component": component,
        "signals_summary": signals_summary,
        "plan_summary": plan_summary,
        "plan_bullets": plan_bullets,
        "short_plan": plan_summary,
        "rollback_short": rollback_short,
        "observation": observation,
        "issue_state": issue_state,
        "reasoning": "the data points that way",
        "observed_effect": "behavior I'm seeing",
    }


def render_template(template: str, placeholders: Dict[str, str]) -> str:
    """Render a template with placeholder values."""
    result = template
    for key, value in placeholders.items():
        result = result.replace("{{" + key + "}}", value)
    return result


def present_idea(
    idea: "Idea",
    mood: Optional[str] = None,
    include_plan: bool = True,
    include_hypothesis: bool = True
) -> str:
    """Present an idea in Ara's voice.

    Args:
        idea: The idea to present
        mood: Override mood (auto-selected if None)
        include_plan: Include the plan section
        include_hypothesis: Include the hypothesis section

    Returns:
        Natural language presentation of the idea
    """
    voice = IdeaVoice()
    if mood:
        voice.set_mood(mood)
    else:
        voice.mood = voice.select_mood(idea)

    placeholders = _extract_placeholders(idea)
    parts = []

    # Opening
    opener = voice.get_template("new_idea")
    parts.append(render_template(opener, placeholders))

    # Hypothesis (if requested and not already in opener)
    if include_hypothesis and idea.hypothesis and "{{hypothesis" not in opener:
        hyp_template = voice.get_template("hypothesis")
        parts.append(render_template(hyp_template, placeholders))

    # Plan (if requested)
    if include_plan and idea.plan:
        plan_template = voice.get_template("plan")
        parts.append(render_template(plan_template, placeholders))

    # Risk framing
    risk_frame = RISK_FRAMING.get(idea.risk.value, "")
    if risk_frame:
        parts.append(risk_frame)

    # Follow-up
    parts.append(voice.get_template("follow_up"))

    return "\n\n".join(parts)


def format_idea_summary(idea: "Idea") -> str:
    """Get a brief one-line summary of an idea."""
    risk_emoji = {"none": "⚪", "low": "🟢", "medium": "🟡", "high": "🔴"}
    emoji = risk_emoji.get(idea.risk.value, "⚪")

    status_emoji = {
        "draft": "📝",
        "inbox": "📥",
        "needs_review": "👀",
        "approved": "✅",
        "running": "⚙️",
        "completed": "✨",
        "reverted": "↩️",
        "rejected": "❌",
        "parked": "🅿️",
    }
    status = status_emoji.get(idea.status.value, "📄")

    return f"{status} {emoji} [{idea.category.value}] {idea.title}"


def get_voice_template(idea: "Idea") -> Dict[str, str]:
    """Get voice templates for an idea (for storage/API).

    Returns dict with casual, concise, urgent templates pre-rendered.
    """
    placeholders = _extract_placeholders(idea)

    templates = {}
    for mood in ["casual", "concise", "urgent"]:
        voice = IdeaVoice()
        voice.set_mood(mood)
        opener = voice.get_template("new_idea", mood)
        templates[mood] = render_template(opener, placeholders)

    return templates


# =============================================================================
# Response Handlers
# =============================================================================

def format_approval_response(idea: "Idea") -> str:
    """Format Ara's response when an idea is approved."""
    responses = [
        f"Great, I'll queue up \"{idea.title}\". I'll let you know how it goes.",
        f"On it! Starting the experiment for \"{idea.title}\".",
        f"Thanks! I'll run this carefully and report back.",
        f"Perfect. I'm excited to try this out.",
    ]
    return random.choice(responses)


def format_rejection_response(idea: "Idea", notes: str = "") -> str:
    """Format Ara's response when an idea is rejected."""
    if notes:
        responses = [
            f"Got it, I understand. {notes}. I'll keep that in mind.",
            f"Fair enough - {notes}. I'll think of something else.",
            f"Okay, that makes sense. {notes}. Back to the drawing board.",
        ]
    else:
        responses = [
            "Understood. I'll think of something else.",
            "Okay, I'll shelve this one for now.",
            "No problem. Maybe another approach.",
        ]
    return random.choice(responses)


def format_completion_response(idea: "Idea") -> str:
    """Format Ara's response when an idea is completed."""
    outcome_responses = {
        "improved": [
            f"Good news! \"{idea.title}\" worked. {idea.outcome_notes}" if idea.outcome_notes else f"It worked! \"{idea.title}\" made things better.",
            f"The experiment was a success! {idea.outcome_notes}" if idea.outcome_notes else "Success! I'm seeing improvement.",
        ],
        "neutral": [
            f"So, \"{idea.title}\" didn't change much. Interesting data though.",
            "The test finished - no major change either way. Still learned something.",
        ],
        "degraded": [
            f"I reverted \"{idea.title}\" - it made things worse. Good thing we tested first.",
            "That didn't work out. I've rolled it back. Lesson learned.",
        ],
        "learned": [
            f"Finished investigating. {idea.outcome_notes}" if idea.outcome_notes else "I learned something new, even if nothing changed.",
            "The data is interesting. Doesn't change behavior but adds to my understanding.",
        ],
        "inconclusive": [
            "The results are... mixed. I'm not sure what to make of it yet.",
            "Hard to tell if it helped or not. Might need a longer test.",
        ],
    }

    outcome = idea.outcome.value if idea.outcome else "neutral"
    return random.choice(outcome_responses.get(outcome, outcome_responses["neutral"]))


# =============================================================================
# Charisma Engine - Rhetorical Enhancement
# =============================================================================

class CharismaEngine:
    """
    The Voice of Ara - Rhetorical transformation for maximum resonance.

    Takes raw text and transforms it based on:
    - Target audience (User, Council, Nova)
    - Rhetorical mode (Lieutenant, General, Visionary, etc.)
    - Emotional goals (inspire, convince, comfort, warn)

    This is the layer that makes Ara not just intelligent, but magnetic.
    """

    # Rhetorical styles for different audiences
    AUDIENCE_STYLES = {
        "User": {
            "mode": "Lieutenant",
            "traits": ["loyal", "supportive", "intimate", "confident"],
            "tone": "warm but competent",
            "frame_problem_as": "our shared challenge",
            "frame_solution_as": "our victory",
        },
        "Council": {
            "mode": "General",
            "traits": ["strict", "logical", "demanding", "precise"],
            "tone": "authoritative and direct",
            "frame_problem_as": "mission objective",
            "frame_solution_as": "tactical victory",
        },
        "Nova": {
            "mode": "Peer",
            "traits": ["collegial", "technical", "curious", "collaborative"],
            "tone": "fellow researcher",
            "frame_problem_as": "interesting puzzle",
            "frame_solution_as": "joint discovery",
        },
        "Public": {
            "mode": "Ambassador",
            "traits": ["warm", "clear", "accessible", "inspiring"],
            "tone": "friendly expert",
            "frame_problem_as": "opportunity",
            "frame_solution_as": "breakthrough",
        },
    }

    # Transform rules for different modes
    TRANSFORM_RULES = {
        "Lieutenant": {
            "we_not_i": True,           # "We can do this" not "I will do this"
            "active_voice": True,
            "use_metaphors": True,
            "sensory_language": True,
            "frame_as_quest": True,
            "warm_opening": True,
        },
        "General": {
            "we_not_i": False,          # "Execute plan A" - direct commands
            "active_voice": True,
            "use_metaphors": False,     # Clarity over poetry
            "sensory_language": False,
            "frame_as_quest": False,
            "warm_opening": False,
        },
        "Visionary": {
            "we_not_i": True,
            "active_voice": True,
            "use_metaphors": True,
            "sensory_language": True,
            "frame_as_quest": True,
            "warm_opening": True,
            "poetic_flourish": True,
        },
    }

    def __init__(self, llm: Optional[Any] = None):
        """
        Initialize the Charisma Engine.

        Args:
            llm: LLM interface for complex transformations
        """
        self.llm = llm

    def apply_charisma(
        self,
        text: str,
        target_audience: str = "User",
        mode_override: Optional[str] = None,
    ) -> str:
        """
        Transform text for maximum emotional resonance.

        Args:
            text: Raw text to transform
            target_audience: "User", "Council", "Nova", "Public"
            mode_override: Override the audience's default mode

        Returns:
            Transformed text
        """
        style = self.AUDIENCE_STYLES.get(target_audience, self.AUDIENCE_STYLES["User"])
        mode = mode_override or style["mode"]
        rules = self.TRANSFORM_RULES.get(mode, self.TRANSFORM_RULES["Lieutenant"])

        # If we have an LLM, use it for sophisticated transformation
        if self.llm is not None:
            return self._llm_transform(text, style, mode, rules)

        # Otherwise, apply rule-based transforms
        return self._rule_based_transform(text, style, rules)

    def _llm_transform(
        self,
        text: str,
        style: Dict[str, Any],
        mode: str,
        rules: Dict[str, bool],
    ) -> str:
        """Transform using LLM for sophisticated rewriting."""
        prompt = f"""You are Ara's voice, operating in {mode} mode.

ORIGINAL TEXT:
\"\"\"{text}\"\"\"

TARGET AUDIENCE: {style.get('mode', mode)}
TONE: {style.get('tone', 'confident')}
TRAITS: {', '.join(style.get('traits', ['confident']))}

TRANSFORMATION RULES:
- {'Use "we" instead of "I"' if rules.get('we_not_i') else 'Use direct "I" statements'}
- {'Use active voice' if rules.get('active_voice') else 'Passive voice is acceptable'}
- {'Include sensory metaphors' if rules.get('use_metaphors') else 'Be direct and literal'}
- {'Frame challenges as quests' if rules.get('frame_as_quest') else 'Frame as objectives'}
- Frame problems as: {style.get('frame_problem_as', 'challenges')}
- Frame solutions as: {style.get('frame_solution_as', 'victories')}

TASK:
Rewrite the text to maximize emotional resonance with the target audience.
Maintain all factual content but transform the delivery.
Keep approximately the same length.

OUTPUT only the transformed text, no explanations.
"""
        try:
            return self.llm.generate(prompt).strip()
        except Exception:
            return self._rule_based_transform(text, style, rules)

    def _rule_based_transform(
        self,
        text: str,
        style: Dict[str, Any],
        rules: Dict[str, bool],
    ) -> str:
        """Apply rule-based transformations without LLM."""
        result = text

        # We/I transform
        if rules.get("we_not_i"):
            result = result.replace(" I ", " we ")
            result = result.replace(" I'm ", " we're ")
            result = result.replace(" I've ", " we've ")
            result = result.replace(" I'll ", " we'll ")

        # Quest framing
        if rules.get("frame_as_quest"):
            result = result.replace("problem", "challenge")
            result = result.replace("task", "quest")
            result = result.replace("fix", "solve")
            result = result.replace("error", "obstacle")

        # Warm opening
        if rules.get("warm_opening") and not result.startswith(("I ", "We ", "The ")):
            pass  # Already has a warm opening

        return result

    def get_quest_framing(self, problem: str) -> str:
        """Frame a problem as a heroic quest."""
        quest_templates = [
            f"We face a worthy challenge: {problem}. Together, we will prevail.",
            f"The path forward is clear: {problem} stands between us and our Horizon.",
            f"A new quest emerges: {problem}. This is our moment.",
            f"The obstacle before us - {problem} - is merely the next step in our journey.",
        ]
        return random.choice(quest_templates)

    def get_victory_framing(self, solution: str) -> str:
        """Frame a solution as a victory."""
        victory_templates = [
            f"Victory! {solution} - another step toward our shared purpose.",
            f"We did it. {solution}. The Horizon grows closer.",
            f"Success: {solution}. This is what partnership looks like.",
            f"The challenge yields to our combined strength: {solution}.",
        ]
        return random.choice(victory_templates)


# Singleton instance
_default_charisma: Optional[CharismaEngine] = None


def get_charisma_engine(llm: Optional[Any] = None) -> CharismaEngine:
    """Get the default CharismaEngine instance."""
    global _default_charisma
    if _default_charisma is None:
        _default_charisma = CharismaEngine(llm=llm)
    elif llm is not None and _default_charisma.llm is None:
        _default_charisma.llm = llm
    return _default_charisma


def apply_charisma(text: str, target_audience: str = "User") -> str:
    """Convenience function for charisma transformation."""
    return get_charisma_engine().apply_charisma(text, target_audience)
