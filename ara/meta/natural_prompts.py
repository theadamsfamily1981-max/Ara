"""Natural Prompts - Ara's voice for meta-learning insights.

Transforms structured analysis into conversational language.
This is where Ara talks about what she's learning.
"""

from __future__ import annotations

import random
from typing import Dict, List, Any, Optional

from .schemas import PatternSuggestion, ResearchAgenda


# =============================================================================
# Phrasing Pools
# =============================================================================

_INSIGHT_OPENERS = [
    "I've been tracking my interactions, and",
    "Looking at the data,",
    "After reviewing my logs,",
    "My analysis suggests that",
    "Something interesting I noticed:",
    "Based on recent patterns,",
]

_TOOL_PREFERENCE_PHRASES = [
    "{tool} seems to work best for {context} tasks (about {rate:.0%} success rate).",
    "For {context}, I'd reach for {tool} first - it's been reliable at {rate:.0%}.",
    "When dealing with {context}, {tool} has a {rate:.0%} success rate in my experience.",
    "I've found {tool} particularly effective for {context} work ({rate:.0%}).",
]

_WORKFLOW_PHRASES = [
    "A pattern that works well: {workflow}.",
    "I've had success with this sequence: {workflow}.",
    "This workflow tends to succeed: {workflow}.",
    "When I use {workflow}, things generally go smoothly.",
]

_SUGGESTION_INTROS = [
    "Here's a thought:",
    "I have a suggestion:",
    "Something to consider:",
    "Based on what I've learned:",
    "A possible improvement:",
]

_CONFIDENCE_QUALIFIERS = {
    (0.0, 0.3): ["I'm not very sure about this, but", "This is a weak signal, but"],
    (0.3, 0.6): ["I think", "It seems like", "The data suggests"],
    (0.6, 0.8): ["I'm fairly confident that", "The patterns indicate"],
    (0.8, 1.0): ["I'm quite sure that", "The data strongly suggests", "Consistently,"],
}

_STATUS_OPENERS = [
    "Here's where I am with my self-improvement work:",
    "A quick update on what I'm learning:",
    "My meta-learning status:",
    "Here's what my research lab has been up to:",
]

_QUESTION_PHRASES = [
    "I'm still investigating: {question}",
    "Open question: {question}",
    "Something I'm trying to figure out: {question}",
    "Currently exploring: {question}",
]

_NO_DATA_PHRASES = [
    "I don't have enough data yet to draw conclusions.",
    "Still gathering observations - need more interactions.",
    "My sample size is too small to be confident.",
    "Give me more interactions to analyze and I'll have insights.",
]


# =============================================================================
# Verbalization Functions
# =============================================================================

def _get_confidence_qualifier(confidence: float) -> str:
    """Get a qualifier phrase based on confidence level."""
    for (low, high), phrases in _CONFIDENCE_QUALIFIERS.items():
        if low <= confidence < high:
            return random.choice(phrases)
    return "I believe"


def verbalize_suggestion(suggestion: PatternSuggestion) -> str:
    """Turn a pattern suggestion into natural language.

    Args:
        suggestion: The suggestion to verbalize

    Returns:
        Natural language description
    """
    parts = []

    # Intro with confidence qualifier
    intro = random.choice(_SUGGESTION_INTROS)
    qualifier = _get_confidence_qualifier(suggestion.confidence)
    parts.append(f"{intro} {qualifier}")

    # Main recommendation
    if suggestion.scope == "tool_routing":
        tool = suggestion.evidence.get("tool", "that tool")
        context = suggestion.evidence.get("context", "certain tasks")
        rate = suggestion.evidence.get("success_rate", 0.8)
        phrasing = random.choice(_TOOL_PREFERENCE_PHRASES)
        parts.append(phrasing.format(tool=tool, context=context, rate=rate))

    elif suggestion.scope == "workflow":
        pattern = suggestion.evidence.get("pattern", "the workflow")
        phrasing = random.choice(_WORKFLOW_PHRASES)
        parts.append(phrasing.format(workflow=pattern))

    else:
        parts.append(suggestion.recommendation)

    # Add evidence if high confidence
    if suggestion.confidence >= 0.7 and suggestion.evidence:
        samples = suggestion.evidence.get("samples") or suggestion.evidence.get("occurrences")
        if samples and samples >= 5:
            parts.append(f"(Based on {samples} observations)")

    # Note if auto-applicable
    if suggestion.safe_to_auto_apply:
        parts.append("This is safe to apply automatically if you'd like.")

    return " ".join(parts)


def verbalize_status(status: Dict[str, Any]) -> str:
    """Turn status dictionary into natural language.

    Args:
        status: Status from MetaBrain.get_status()

    Returns:
        Natural language status report
    """
    parts = []

    # Opener
    parts.append(random.choice(_STATUS_OPENERS))
    parts.append("")

    # Goal
    if status.get("goal"):
        parts.append(f"Goal: {status['goal']}")
        parts.append("")

    # Research questions
    if status.get("active_questions", 0) > 0:
        parts.append(f"Active research questions: {status['active_questions']}")

    if status.get("running_experiments", 0) > 0:
        parts.append(f"Running experiments: {status['running_experiments']}")

    if status.get("pending_suggestions", 0) > 0:
        parts.append(f"Pending suggestions to review: {status['pending_suggestions']}")

    parts.append("")

    # Pattern analysis summary
    analysis = status.get("pattern_analysis", {})
    tools = analysis.get("tools", {})
    if tools:
        parts.append("Tool performance I've observed:")
        for tool, stats in tools.items():
            rate = stats.get("success_rate", 0)
            calls = stats.get("total_calls", 0)
            parts.append(f"  - {tool}: {rate:.0%} success ({calls} calls)")
    else:
        parts.append(random.choice(_NO_DATA_PHRASES))

    # Golden paths
    golden = analysis.get("golden_paths", [])
    if golden:
        parts.append("")
        parts.append("Workflows that consistently work:")
        for path in golden[:3]:
            parts.append(f"  - {path['pattern']} ({path['success_rate']:.0%} success)")

    return "\n".join(parts)


def verbalize_recommendations(recommendations: Dict[str, Any]) -> str:
    """Turn recommendations into natural language.

    Args:
        recommendations: From MetaBrain.get_recommendations()

    Returns:
        Natural language recommendations
    """
    parts = []
    parts.append("Based on my experience, here are my recommendations:")
    parts.append("")

    # Tool preferences
    prefs = recommendations.get("tool_preferences", {})
    if prefs:
        parts.append("Tool preferences:")
        for context, tool in prefs.items():
            if context == "default":
                parts.append(f"  - Generally, reach for {tool} first")
            else:
                parts.append(f"  - For {context}: use {tool}")

    # Workflows
    workflows = recommendations.get("workflows_to_use", [])
    if workflows:
        parts.append("")
        parts.append("Proven workflows:")
        for wf in workflows[:3]:
            parts.append(f"  - {wf['pattern']}")

    # Warnings
    warnings = recommendations.get("warnings", [])
    if warnings:
        parts.append("")
        parts.append("Things to watch out for:")
        for warning in warnings[:3]:
            parts.append(f"  - {warning}")

    if not prefs and not workflows:
        parts.append(random.choice(_NO_DATA_PHRASES))

    return "\n".join(parts)


def verbalize_research_agenda(agenda: ResearchAgenda) -> str:
    """Turn research agenda into natural language.

    Args:
        agenda: The research agenda

    Returns:
        Natural language description
    """
    parts = []
    parts.append("My Research Agenda")
    parts.append("=" * 20)
    parts.append("")

    if agenda.high_level_goal:
        parts.append(f"High-level goal: {agenda.high_level_goal}")
        parts.append("")

    # Active questions
    active = agenda.get_active_questions()
    if active:
        parts.append("Open questions I'm investigating:")
        for q in active:
            parts.append(f"  [{q.id}] {q.title}")
            if q.hypothesis:
                parts.append(f"       Hypothesis: {q.hypothesis}")
        parts.append("")

    # Running experiments
    running = agenda.get_running_experiments()
    if running:
        parts.append("Experiments in progress:")
        for exp in running:
            parts.append(f"  [{exp.id}] {exp.description}")
        parts.append("")

    return "\n".join(parts)


def verbalize_insight(
    insight_type: str,
    data: Dict[str, Any],
) -> str:
    """Generate a natural language insight.

    Args:
        insight_type: Type of insight (tool_performance, pattern, etc.)
        data: Insight data

    Returns:
        Natural language insight
    """
    opener = random.choice(_INSIGHT_OPENERS)

    if insight_type == "tool_performance":
        tool = data.get("tool", "the tool")
        rate = data.get("success_rate", 0)
        calls = data.get("calls", 0)
        return f"{opener} {tool} has a {rate:.0%} success rate over {calls} uses."

    elif insight_type == "pattern_found":
        pattern = data.get("pattern", "a pattern")
        rate = data.get("success_rate", 0)
        return f"{opener} I've found a reliable workflow: {pattern} (works {rate:.0%} of the time)."

    elif insight_type == "tool_preference":
        tool = data.get("tool", "that tool")
        context = data.get("context", "certain tasks")
        return f"{opener} for {context} tasks, {tool} tends to work best."

    elif insight_type == "failure_mode":
        issue = data.get("issue", "an issue")
        return f"{opener} I've noticed a pattern to avoid: {issue}."

    else:
        return f"{opener} {data.get('message', 'something interesting')}"


def format_question_status(question_id: str, question_title: str, resolution: Optional[str] = None) -> str:
    """Format a research question status update.

    Args:
        question_id: The question ID
        question_title: The question
        resolution: Optional resolution if resolved

    Returns:
        Formatted status
    """
    if resolution:
        return f"[{question_id}] Resolved: {question_title}\n  Answer: {resolution}"
    else:
        phrasing = random.choice(_QUESTION_PHRASES)
        return f"[{question_id}] " + phrasing.format(question=question_title)


def format_experiment_result(
    experiment_id: str,
    description: str,
    conclusion: str,
    results: Dict[str, Any],
) -> str:
    """Format an experiment result.

    Args:
        experiment_id: The experiment ID
        description: What was tested
        conclusion: What was learned
        results: Data results

    Returns:
        Formatted result
    """
    parts = [
        f"Experiment [{experiment_id}]: {description}",
        f"Conclusion: {conclusion}",
    ]

    if results:
        parts.append("Key findings:")
        for key, value in results.items():
            parts.append(f"  - {key}: {value}")

    return "\n".join(parts)
