"""
Email Subject Line Oracle - analyze what makes subjects perform.

This is the first sellable workflow:
- Client sends CSV of past campaigns
- You run this analysis
- They get patterns + recommendations + subject line ideas
"""

import pandas as pd
import re
from typing import Dict, Any, List, Optional
from ara_core.analytics.correlations import find_strongest_patterns
from ara_core.analytics.experiments import suggest_experiments
from ara_core.narrative.voice_ara import summarize_patterns, generate_snapshot
from ara_core.narrative.report_builder import build_report, Report


def analyze_email_subjects(
    csv_path: str,
    target_col: str = "open_rate",
    subject_col: str = "subject",
    client_name: str = "Client",
) -> Dict[str, Any]:
    """
    Analyze email campaign data for subject line patterns.

    Args:
        csv_path: Path to CSV file
        target_col: Column with performance metric (e.g., open_rate)
        subject_col: Column with subject lines
        client_name: Client name for report

    Returns:
        Dictionary with analysis results
    """
    df = pd.read_csv(csv_path)

    # Enrich with subject line features
    df = _enrich_subject_features(df, subject_col)

    # Find patterns
    patterns = find_strongest_patterns(df, target_col)

    # Generate experiments
    experiments = suggest_experiments(patterns)

    # Build report
    report = build_report(
        patterns,
        client_name=client_name,
        title="Email Subject Line Analysis",
    )

    # Generate subject line suggestions
    suggestions = _generate_subject_suggestions(df, patterns, subject_col, target_col)

    return {
        "report": report.to_dict(),
        "report_markdown": report.to_markdown(),
        "patterns": patterns,
        "experiments": [vars(e) for e in experiments],
        "subject_suggestions": suggestions,
        "summary": summarize_patterns(patterns),
        "snapshot": generate_snapshot(patterns, "email campaigns"),
    }


def _enrich_subject_features(
    df: pd.DataFrame,
    subject_col: str,
) -> pd.DataFrame:
    """
    Add derived features from subject lines.

    Features:
    - Length (characters and words)
    - Has emoji
    - Has number
    - Has question mark
    - Starts with "How"
    - Has urgency words
    - Has personalization tokens
    """
    df = df.copy()

    if subject_col not in df.columns:
        return df

    subjects = df[subject_col].fillna("")

    # Length features
    df["subject_length_chars"] = subjects.str.len()
    df["subject_length_words"] = subjects.str.split().str.len()

    # Content features
    df["has_emoji"] = subjects.apply(_has_emoji).astype(int)
    df["has_number"] = subjects.str.contains(r'\d', regex=True).astype(int)
    df["has_question"] = subjects.str.contains(r'\?').astype(int)
    df["starts_with_how"] = subjects.str.lower().str.startswith("how").astype(int)
    df["has_urgency"] = subjects.str.lower().str.contains(
        r'\b(now|today|urgent|last chance|don\'t miss|limited)\b',
        regex=True
    ).astype(int)
    df["has_personalization"] = subjects.str.contains(
        r'\{|first_name|{{',
        regex=True
    ).astype(int)

    # Categorize length
    df["length_category"] = pd.cut(
        df["subject_length_chars"],
        bins=[0, 30, 50, 70, 100, float('inf')],
        labels=["very_short", "short", "medium", "long", "very_long"]
    )

    return df


def _has_emoji(text: str) -> bool:
    """Check if text contains emoji."""
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
    return bool(emoji_pattern.search(str(text)))


def _generate_subject_suggestions(
    df: pd.DataFrame,
    patterns: Dict[str, Any],
    subject_col: str,
    target_col: str,
    n_suggestions: int = 10,
) -> List[Dict[str, str]]:
    """
    Generate subject line suggestions based on patterns.

    This is a simple template-based approach.
    For production, you'd want to use an LLM.
    """
    suggestions = []

    # Get top performing subjects for reference
    df_sorted = df.sort_values(target_col, ascending=False)
    top_subjects = df_sorted[subject_col].head(5).tolist()

    # Extract patterns
    top_patterns = patterns.get("top_patterns", [])

    suggestions.append({
        "category": "Reference: Your Top Performers",
        "note": "These are your actual best-performing subjects",
        "subjects": top_subjects[:5],
    })

    # Generate suggestions based on patterns
    pattern_suggestions = []

    for p in top_patterns[:3]:
        feature = p.get("feature", "")
        note = p.get("note", "")

        if "length" in feature.lower() and "short" in note.lower():
            pattern_suggestions.append(
                "Try shorter subjects (under 40 characters)"
            )
        if "emoji" in feature.lower() and "positive" in str(p.get("direction", "")):
            pattern_suggestions.append(
                "Add an emoji to increase engagement"
            )
        if "question" in feature.lower():
            pattern_suggestions.append(
                "Try framing as a question"
            )
        if "number" in feature.lower():
            pattern_suggestions.append(
                "Include a specific number (e.g., '3 ways to...')"
            )

    if pattern_suggestions:
        suggestions.append({
            "category": "Based on Your Patterns",
            "note": "These tactics work for your specific audience",
            "subjects": pattern_suggestions[:5],
        })

    # Generic high-performing templates
    suggestions.append({
        "category": "Proven Templates",
        "note": "Adapt these to your content",
        "subjects": [
            "[Number] ways to [benefit]",
            "How I [achieved result] in [timeframe]",
            "The [adjective] guide to [topic]",
            "Why [common belief] is wrong",
            "What [authority] taught me about [topic]",
        ],
    })

    return suggestions


def quick_analyze(csv_path: str, target_col: str = "open_rate") -> str:
    """
    Quick analysis that returns just the summary.

    For CLI or quick checks.
    """
    result = analyze_email_subjects(csv_path, target_col)
    return result["summary"]
