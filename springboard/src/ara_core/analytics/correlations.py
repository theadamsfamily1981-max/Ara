"""
Correlation analysis for small datasets.

Focus: explainable, actionable patterns - not statistical significance.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class CorrelationResult:
    """A single correlation finding."""
    feature: str
    correlation: float
    direction: str  # "positive" or "negative"
    note: str
    confidence: str  # "high", "medium", "low"
    sample_size: int


def simple_correlations(
    df: pd.DataFrame,
    target: str,
    min_correlation: float = 0.1,
) -> List[CorrelationResult]:
    """
    Find simple correlations between features and target.

    Args:
        df: Input DataFrame
        target: Target column name
        min_correlation: Minimum absolute correlation to report

    Returns:
        List of CorrelationResult, sorted by strength
    """
    results = []

    if target not in df.columns:
        return results

    target_series = pd.to_numeric(df[target], errors="coerce")
    if target_series.isna().all():
        return results

    for col in df.columns:
        if col == target:
            continue

        try:
            # Try numeric correlation first
            feature_series = pd.to_numeric(df[col], errors="coerce")

            if feature_series.isna().sum() > len(df) * 0.5:
                continue  # Skip if too many NaN

            valid_mask = ~(feature_series.isna() | target_series.isna())
            if valid_mask.sum() < 10:
                continue  # Need at least 10 data points

            corr = feature_series[valid_mask].corr(target_series[valid_mask])

            if pd.isna(corr) or abs(corr) < min_correlation:
                continue

            direction = "positive" if corr > 0 else "negative"
            direction_word = "higher" if corr > 0 else "lower"

            note = f"When {col} is higher, {target} tends to be {direction_word}."

            # Confidence based on correlation strength and sample size
            sample_size = valid_mask.sum()
            if abs(corr) > 0.5 and sample_size > 30:
                confidence = "high"
            elif abs(corr) > 0.3 or sample_size > 50:
                confidence = "medium"
            else:
                confidence = "low"

            results.append(CorrelationResult(
                feature=col,
                correlation=float(corr),
                direction=direction,
                note=note,
                confidence=confidence,
                sample_size=sample_size,
            ))

        except Exception:
            continue  # Skip problematic columns

    # Sort by absolute correlation
    results.sort(key=lambda x: abs(x.correlation), reverse=True)
    return results


def categorical_correlations(
    df: pd.DataFrame,
    target: str,
    categorical_cols: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Find patterns in categorical variables vs target.

    Args:
        df: Input DataFrame
        target: Target column name (numeric)
        categorical_cols: Columns to analyze (auto-detect if None)

    Returns:
        List of categorical pattern findings
    """
    results = []

    target_series = pd.to_numeric(df[target], errors="coerce")
    if target_series.isna().all():
        return results

    # Auto-detect categorical columns
    if categorical_cols is None:
        categorical_cols = []
        for col in df.columns:
            if col == target:
                continue
            if df[col].dtype == 'object' or df[col].nunique() < 10:
                categorical_cols.append(col)

    for col in categorical_cols:
        try:
            grouped = df.groupby(col)[target].agg(['mean', 'count', 'std'])
            grouped = grouped[grouped['count'] >= 5]  # Min 5 samples per group

            if len(grouped) < 2:
                continue

            overall_mean = target_series.mean()
            best_group = grouped['mean'].idxmax()
            worst_group = grouped['mean'].idxmin()

            best_lift = (grouped.loc[best_group, 'mean'] - overall_mean) / overall_mean * 100
            worst_lift = (grouped.loc[worst_group, 'mean'] - overall_mean) / overall_mean * 100

            if abs(best_lift) > 5 or abs(worst_lift) > 5:  # At least 5% difference
                results.append({
                    "feature": col,
                    "type": "categorical",
                    "best_value": best_group,
                    "best_lift_pct": round(best_lift, 1),
                    "worst_value": worst_group,
                    "worst_lift_pct": round(worst_lift, 1),
                    "note": f"'{best_group}' performs {abs(best_lift):.1f}% better than average for {target}",
                })

        except Exception:
            continue

    return results


def find_strongest_patterns(
    df: pd.DataFrame,
    target: str,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Find the strongest patterns in data, combining numeric and categorical.

    Args:
        df: Input DataFrame
        target: Target column name
        top_k: Number of top patterns to return

    Returns:
        Dictionary with numeric_patterns, categorical_patterns, summary
    """
    numeric = simple_correlations(df, target)[:top_k]
    categorical = categorical_correlations(df, target)[:top_k]

    # Generate summary
    all_patterns = []
    for n in numeric:
        all_patterns.append({
            "type": "numeric",
            "feature": n.feature,
            "strength": abs(n.correlation),
            "note": n.note,
        })
    for c in categorical:
        all_patterns.append({
            "type": "categorical",
            "feature": c["feature"],
            "strength": abs(c["best_lift_pct"]) / 100,
            "note": c["note"],
        })

    all_patterns.sort(key=lambda x: x["strength"], reverse=True)

    return {
        "numeric_patterns": [vars(n) for n in numeric],
        "categorical_patterns": categorical,
        "top_patterns": all_patterns[:top_k],
        "target": target,
        "n_rows": len(df),
    }
