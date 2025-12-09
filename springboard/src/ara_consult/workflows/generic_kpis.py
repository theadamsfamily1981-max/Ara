"""
Generic KPI Reader - analyze any performance dashboard export.

Works with any CSV that has:
- A target metric column
- Some feature columns
"""

import pandas as pd
from typing import Dict, Any, Optional
from ara_core.analytics.correlations import find_strongest_patterns
from ara_core.analytics.segments import segment_by_performance
from ara_core.analytics.experiments import suggest_experiments
from ara_core.narrative.voice_ara import summarize_patterns, generate_snapshot
from ara_core.narrative.report_builder import build_report


def analyze_kpis(
    csv_path: str,
    target_col: str,
    client_name: str = "Client",
    title: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze any KPI data for patterns.

    Args:
        csv_path: Path to CSV file
        target_col: Column with the metric to analyze
        client_name: Client name for report
        title: Optional report title

    Returns:
        Dictionary with analysis results
    """
    df = pd.read_csv(csv_path)

    # Basic data cleaning
    df = _clean_data(df)

    # Find patterns
    patterns = find_strongest_patterns(df, target_col)

    # Segment by performance
    segments = segment_by_performance(df, target_col)

    # Generate experiments
    experiments = suggest_experiments(patterns)

    # Build report
    report_title = title or f"{target_col.replace('_', ' ').title()} Analysis"
    report = build_report(
        patterns,
        client_name=client_name,
        title=report_title,
    )

    return {
        "report": report.to_dict(),
        "report_markdown": report.to_markdown(),
        "patterns": patterns,
        "segments": segments,
        "experiments": [vars(e) for e in experiments],
        "summary": summarize_patterns(patterns),
        "snapshot": generate_snapshot(patterns),
        "data_overview": {
            "rows": len(df),
            "columns": len(df.columns),
            "target": target_col,
            "target_mean": float(pd.to_numeric(df[target_col], errors="coerce").mean()),
            "target_std": float(pd.to_numeric(df[target_col], errors="coerce").std()),
        },
    }


def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic data cleaning."""
    df = df.copy()

    # Drop completely empty columns
    df = df.dropna(axis=1, how="all")

    # Drop duplicate rows
    df = df.drop_duplicates()

    # Try to infer datetime columns
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                df[col] = pd.to_datetime(df[col])
                df[f"{col}_day_of_week"] = df[col].dt.dayofweek
                df[f"{col}_hour"] = df[col].dt.hour
            except (ValueError, TypeError):
                pass

    return df


def quick_analyze(csv_path: str, target_col: str) -> str:
    """
    Quick analysis that returns just the summary.

    For CLI or quick checks.
    """
    result = analyze_kpis(csv_path, target_col)
    return result["summary"]
