"""
Pricing Pattern Sniffer - analyze what makes pricing work.

Workflow for analyzing:
- Price point performance
- Discount effectiveness
- Bundle vs. single pricing
"""

import pandas as pd
from typing import Dict, Any, List, Optional
from ara_core.analytics.correlations import find_strongest_patterns
from ara_core.analytics.experiments import suggest_experiments
from ara_core.narrative.voice_ara import summarize_patterns, generate_snapshot
from ara_core.narrative.report_builder import build_report


def analyze_pricing(
    csv_path: str,
    target_col: str = "conversion_rate",
    price_col: str = "price",
    client_name: str = "Client",
) -> Dict[str, Any]:
    """
    Analyze pricing data for patterns.

    Args:
        csv_path: Path to CSV file
        target_col: Column with performance metric
        price_col: Column with price values
        client_name: Client name for report

    Returns:
        Dictionary with analysis results
    """
    df = pd.read_csv(csv_path)

    # Enrich with pricing features
    df = _enrich_pricing_features(df, price_col)

    # Find patterns
    patterns = find_strongest_patterns(df, target_col)

    # Generate experiments
    experiments = suggest_experiments(patterns)

    # Build report
    report = build_report(
        patterns,
        client_name=client_name,
        title="Pricing Pattern Analysis",
    )

    # Generate pricing suggestions
    suggestions = _generate_pricing_suggestions(df, patterns, price_col, target_col)

    return {
        "report": report.to_dict(),
        "report_markdown": report.to_markdown(),
        "patterns": patterns,
        "experiments": [vars(e) for e in experiments],
        "pricing_suggestions": suggestions,
        "summary": summarize_patterns(patterns),
        "snapshot": generate_snapshot(patterns, "pricing data"),
    }


def _enrich_pricing_features(
    df: pd.DataFrame,
    price_col: str,
) -> pd.DataFrame:
    """Add derived features from pricing data."""
    df = df.copy()

    if price_col not in df.columns:
        return df

    prices = pd.to_numeric(df[price_col], errors="coerce")

    # Price tier
    df["price_tier"] = pd.cut(
        prices,
        bins=[0, 10, 25, 50, 100, 250, float('inf')],
        labels=["micro", "low", "medium", "high", "premium", "enterprise"]
    )

    # Psychological pricing features
    df["ends_in_9"] = (prices % 10 == 9).astype(int) | (prices % 1 == 0.99).astype(int)
    df["ends_in_0"] = (prices % 10 == 0).astype(int)
    df["is_round_number"] = (prices % 5 == 0).astype(int)

    # Relative to mean
    mean_price = prices.mean()
    df["above_average_price"] = (prices > mean_price).astype(int)

    # Check for discount columns
    if "original_price" in df.columns:
        orig = pd.to_numeric(df["original_price"], errors="coerce")
        df["discount_pct"] = ((orig - prices) / orig * 100).fillna(0)
        df["has_discount"] = (df["discount_pct"] > 0).astype(int)

    return df


def _generate_pricing_suggestions(
    df: pd.DataFrame,
    patterns: Dict[str, Any],
    price_col: str,
    target_col: str,
) -> List[Dict[str, Any]]:
    """Generate pricing suggestions based on patterns."""
    suggestions = []

    top_patterns = patterns.get("top_patterns", [])

    # Find optimal price ranges
    df_sorted = df.sort_values(target_col, ascending=False)
    prices = pd.to_numeric(df_sorted[price_col], errors="coerce")
    top_prices = prices.head(int(len(df) * 0.2))

    suggestions.append({
        "category": "Optimal Price Range",
        "note": "Prices of your top-performing 20%",
        "range": f"${top_prices.min():.2f} - ${top_prices.max():.2f}",
        "median": f"${top_prices.median():.2f}",
    })

    # Pattern-based suggestions
    for p in top_patterns[:3]:
        feature = p.get("feature", "")
        note = p.get("note", "")

        if "ends_in_9" in feature:
            suggestions.append({
                "category": "Charm Pricing",
                "note": note,
                "action": "Test $X9 or $X.99 endings",
            })
        if "tier" in feature.lower():
            suggestions.append({
                "category": "Price Tier",
                "note": note,
                "action": "Consider adjusting your price tier mix",
            })
        if "discount" in feature.lower():
            suggestions.append({
                "category": "Discount Strategy",
                "note": note,
                "action": "Adjust discount frequency or depth",
            })

    return suggestions
