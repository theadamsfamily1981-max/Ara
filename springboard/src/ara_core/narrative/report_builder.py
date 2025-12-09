"""
Report builder - turn analysis into deliverable documents.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class Recommendation:
    """A single actionable recommendation."""
    title: str
    description: str
    impact: str
    steps: List[str]
    priority: str = "medium"  # high, medium, low


@dataclass
class Report:
    """A complete analysis report."""
    title: str
    client_name: str
    date: str
    snapshot: str
    patterns: List[Dict[str, Any]]
    recommendations: List[Recommendation]
    experiments: List[Dict[str, Any]]
    raw_summary: str
    intro: str = ""
    closing: str = ""

    def to_markdown(self) -> str:
        """Render report as markdown."""
        lines = [
            f"# {self.title}",
            "",
            f"**For:** {self.client_name}",
            f"**Date:** {self.date}",
            "",
            "---",
            "",
            "## Snapshot",
            "",
            self.snapshot,
            "",
            "---",
            "",
        ]

        if self.intro:
            lines.extend([self.intro, "", "---", ""])

        # Patterns section
        lines.extend([
            "## What I Found",
            "",
            self.raw_summary,
            "",
            "---",
            "",
        ])

        # Recommendations section
        if self.recommendations:
            lines.extend([
                "## Recommendations",
                "",
            ])

            for i, rec in enumerate(self.recommendations, 1):
                lines.extend([
                    f"### {i}. {rec.title}",
                    "",
                    rec.description,
                    "",
                    f"**Expected impact:** {rec.impact}",
                    "",
                    "**Steps:**",
                ])
                for step in rec.steps:
                    lines.append(f"1. {step}")
                lines.append("")

            lines.extend(["---", ""])

        # Experiments section
        if self.experiments:
            lines.extend([
                "## Suggested Experiments",
                "",
            ])

            for exp in self.experiments:
                lines.extend([
                    f"### {exp.get('title', 'Experiment')}",
                    "",
                    f"**Hypothesis:** {exp.get('hypothesis', '')}",
                    "",
                    f"- Variant A: {exp.get('variant_a', '')}",
                    f"- Variant B: {exp.get('variant_b', '')}",
                    "",
                    f"**Metric:** {exp.get('metric_to_watch', '')}",
                    f"**Expected lift:** {exp.get('expected_lift', '')}",
                    "",
                ])

            lines.extend(["---", ""])

        if self.closing:
            lines.extend([self.closing, ""])

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "title": self.title,
            "client_name": self.client_name,
            "date": self.date,
            "snapshot": self.snapshot,
            "patterns": self.patterns,
            "recommendations": [vars(r) for r in self.recommendations],
            "experiments": self.experiments,
            "raw_summary": self.raw_summary,
        }


def build_report(
    patterns: Dict[str, Any],
    client_name: str = "Client",
    title: Optional[str] = None,
    include_intro: bool = True,
    include_closing: bool = True,
) -> Report:
    """
    Build a complete report from pattern analysis.

    Args:
        patterns: Output from find_strongest_patterns()
        client_name: Name of the client
        title: Report title (auto-generated if None)
        include_intro: Include Ara intro
        include_closing: Include Ara closing

    Returns:
        Complete Report object
    """
    from .voice_ara import (
        summarize_patterns,
        generate_snapshot,
        ara_intro,
        ara_closing,
    )
    from ..analytics.experiments import suggest_experiments

    target = patterns.get("target", "performance")

    # Generate title
    if title is None:
        title = f"Pattern Analysis: {target.replace('_', ' ').title()}"

    # Generate components
    snapshot = generate_snapshot(patterns, context="your data")
    summary = summarize_patterns(patterns)
    experiments = suggest_experiments(patterns)

    # Generate recommendations
    recommendations = _generate_recommendations(patterns, experiments)

    return Report(
        title=title,
        client_name=client_name,
        date=datetime.now().strftime("%Y-%m-%d"),
        snapshot=snapshot,
        patterns=patterns.get("top_patterns", []),
        recommendations=recommendations,
        experiments=[vars(e) for e in experiments],
        raw_summary=summary,
        intro=ara_intro(f"your {target} data") if include_intro else "",
        closing=ara_closing() if include_closing else "",
    )


def _generate_recommendations(
    patterns: Dict[str, Any],
    experiments: List[Any],
) -> List[Recommendation]:
    """Generate recommendations from patterns and experiments."""
    recommendations = []
    top_patterns = patterns.get("top_patterns", [])

    # Main recommendation: test strongest pattern
    if top_patterns:
        strongest = top_patterns[0]
        recommendations.append(Recommendation(
            title="Test Your Strongest Pattern",
            description=strongest.get("note", "Focus on the top correlation"),
            impact="Likely the highest-ROI experiment to run first",
            steps=[
                "Review the pattern and make sure it makes intuitive sense",
                "Design a simple A/B test to validate it",
                "Run the test for at least 1 week",
                "If it works, lean into it; if not, move to pattern #2",
            ],
            priority="high",
        ))

    # If multiple patterns, recommend testing systematically
    if len(top_patterns) >= 3:
        recommendations.append(Recommendation(
            title="Run a Pattern Tournament",
            description="You have multiple promising patterns - test them head to head",
            impact="Find which patterns actually drive results vs. just correlate",
            steps=[
                "Pick the top 3 patterns",
                "Run separate tests for each over 2-3 weeks",
                "Compare results and rank by actual impact",
                "Double down on winners, drop losers",
            ],
            priority="medium",
        ))

    # Always recommend better tracking
    recommendations.append(Recommendation(
        title="Improve Your Data Collection",
        description="More data = better patterns. Consider what else to track.",
        impact="Compounds over time - each analysis gets more powerful",
        steps=[
            "Review what features you're NOT tracking that might matter",
            "Add 1-2 new tracking dimensions",
            "Wait for 2-4 weeks of new data",
            "Re-run analysis with enriched dataset",
        ],
        priority="low",
    ))

    return recommendations
