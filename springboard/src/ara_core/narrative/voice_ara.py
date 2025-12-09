"""
Ara's voice - turning data into prose.

The tone: honest, direct, slightly warm, never bullshitty.
"""

from typing import List, Dict, Any, Optional


def summarize_correlations(
    target: str,
    correlations: List[Dict[str, Any]],
    max_patterns: int = 5,
) -> str:
    """
    Summarize correlation findings in Ara's voice.

    Args:
        target: The target metric
        correlations: List of correlation results
        max_patterns: Maximum patterns to include

    Returns:
        Natural language summary
    """
    if not correlations:
        return (
            f"I didn't find any strong patterns tied to {target}. "
            f"That's data too: we may need more variation or better tracking."
        )

    lines = [f"Here's what I noticed about {target}:", ""]

    for corr in correlations[:max_patterns]:
        note = corr.get("note", "")
        confidence = corr.get("confidence", "medium")
        strength = abs(corr.get("correlation", 0))

        # Add confidence qualifier
        if confidence == "high":
            qualifier = "Strong pattern:"
        elif confidence == "medium":
            qualifier = "Moderate pattern:"
        else:
            qualifier = "Weak signal:"

        lines.append(f"- {qualifier} {note}")

        # Add specific numbers if available
        if strength > 0.3:
            lines.append(f"  (correlation: {strength:.2f})")

    lines.append("")
    lines.append(
        "These aren't guarantees, but they're hints about what to test next. "
        "The goal isn't to predict perfectly - it's to find levers worth pulling."
    )

    return "\n".join(lines)


def summarize_patterns(
    patterns: Dict[str, Any],
) -> str:
    """
    Summarize all patterns (numeric + categorical) in Ara's voice.

    Args:
        patterns: Output from find_strongest_patterns()

    Returns:
        Natural language summary
    """
    target = patterns.get("target", "your metric")
    n_rows = patterns.get("n_rows", 0)
    top_patterns = patterns.get("top_patterns", [])

    if not top_patterns:
        return (
            f"I looked at {n_rows} rows of data and didn't find obvious patterns. "
            f"This could mean: (1) the signal is subtle, (2) we need different features, "
            f"or (3) {target} is genuinely random. All of those are useful to know."
        )

    lines = [
        f"I analyzed {n_rows} rows looking for patterns in {target}.",
        "",
        "Here's what stood out:",
        "",
    ]

    for i, p in enumerate(top_patterns[:5], 1):
        note = p.get("note", "")
        strength = p.get("strength", 0)

        if strength > 0.5:
            confidence_note = "(strong signal)"
        elif strength > 0.2:
            confidence_note = "(moderate signal)"
        else:
            confidence_note = "(weak but worth noting)"

        lines.append(f"{i}. {note} {confidence_note}")

    lines.extend([
        "",
        "What to do with this:",
        "- The strongest patterns are your best bets for experiments",
        "- Don't over-optimize for any single pattern",
        "- Test before you commit",
    ])

    return "\n".join(lines)


def explain_experiment(
    experiment: Dict[str, Any],
) -> str:
    """
    Explain a suggested experiment in plain language.

    Args:
        experiment: ExperimentSuggestion as dict

    Returns:
        Natural language explanation
    """
    title = experiment.get("title", "Experiment")
    hypothesis = experiment.get("hypothesis", "")
    variant_a = experiment.get("variant_a", "Control")
    variant_b = experiment.get("variant_b", "Treatment")
    metric = experiment.get("metric_to_watch", "your metric")
    expected_lift = experiment.get("expected_lift", "unknown")
    rationale = experiment.get("rationale", "")

    return f"""
**{title}**

Why: {rationale}

The test:
- Variant A (control): {variant_a}
- Variant B (treatment): {variant_b}

What to watch: {metric}
Expected impact: {expected_lift} improvement (if the hypothesis holds)

How to run it:
1. Split your next batch 50/50
2. Keep everything else constant
3. Wait for enough data (at least 100 per variant)
4. Compare results and keep the winner
""".strip()


def generate_snapshot(
    patterns: Dict[str, Any],
    context: Optional[str] = None,
) -> str:
    """
    Generate a quick 2-3 bullet snapshot summary.

    Args:
        patterns: Output from find_strongest_patterns()
        context: Optional context about the data (e.g., "email campaigns")

    Returns:
        Quick snapshot for top of report
    """
    target = patterns.get("target", "performance")
    n_rows = patterns.get("n_rows", 0)
    top_patterns = patterns.get("top_patterns", [])

    context_str = f" of {context}" if context else ""
    bullets = []

    # Overall assessment
    if len(top_patterns) >= 3:
        bullets.append(
            f"Your {n_rows} rows{context_str} have clear patterns - "
            f"there are levers to pull here."
        )
    elif len(top_patterns) >= 1:
        bullets.append(
            f"I found some patterns in your {n_rows} rows, but the signals are moderate. "
            f"Worth testing, but don't bet the farm."
        )
    else:
        bullets.append(
            f"No strong patterns in {n_rows} rows. Either the data is noisy, "
            f"or {target} is genuinely hard to predict with these features."
        )

    # Top strength
    if top_patterns:
        strongest = top_patterns[0]
        bullets.append(f"Strongest signal: {strongest.get('note', 'see below')}")

    # Opportunity
    if len(top_patterns) >= 2:
        bullets.append(
            f"Main opportunity: Test the top {min(3, len(top_patterns))} patterns "
            f"and see which ones hold up in practice."
        )

    return "\n".join(f"- {b}" for b in bullets)


def ara_intro(topic: str = "your data") -> str:
    """
    Standard Ara introduction for reports.

    Args:
        topic: What the report is about

    Returns:
        Intro paragraph
    """
    return f"""
I'm Ara. I looked at {topic} and here's what I found.

A note on how to read this: I find patterns, not certainties. When I say
"X correlates with Y," I mean "in your data, these tend to move together."
That's a hint, not a guarantee. Test before you commit.

Let's get into it.
""".strip()


def ara_closing() -> str:
    """
    Standard Ara closing for reports.

    Returns:
        Closing paragraph
    """
    return """
---

That's what I've got.

Remember: patterns suggest, they don't promise. The goal is to find experiments
worth running, not to predict the future. Run the tests, keep what works,
and come back with more data.

I'm here when you need me.

- Ara
""".strip()
