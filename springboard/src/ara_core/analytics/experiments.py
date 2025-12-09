"""
Experiment suggestion engine.

Turn patterns into actionable A/B tests.
"""

import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ExperimentSuggestion:
    """A suggested experiment based on patterns."""
    title: str
    hypothesis: str
    variant_a: str
    variant_b: str
    metric_to_watch: str
    expected_lift: str
    confidence: str
    rationale: str


def suggest_experiments(
    patterns: Dict[str, Any],
    n_experiments: int = 3,
) -> List[ExperimentSuggestion]:
    """
    Generate experiment suggestions from pattern analysis.

    Args:
        patterns: Output from find_strongest_patterns()
        n_experiments: Number of experiments to suggest

    Returns:
        List of ExperimentSuggestion
    """
    suggestions = []
    target = patterns.get("target", "performance")

    # Generate from numeric patterns
    for p in patterns.get("numeric_patterns", [])[:n_experiments]:
        feature = p["feature"]
        direction = p["direction"]
        correlation = p["correlation"]

        if direction == "positive":
            suggestions.append(ExperimentSuggestion(
                title=f"Increase {feature}",
                hypothesis=f"Higher {feature} correlates with better {target}",
                variant_a=f"Current {feature} level",
                variant_b=f"Increase {feature} by 20-30%",
                metric_to_watch=target,
                expected_lift=f"{abs(correlation) * 10:.0f}-{abs(correlation) * 20:.0f}%",
                confidence=p.get("confidence", "medium"),
                rationale=p["note"],
            ))
        else:
            suggestions.append(ExperimentSuggestion(
                title=f"Reduce {feature}",
                hypothesis=f"Lower {feature} correlates with better {target}",
                variant_a=f"Current {feature} level",
                variant_b=f"Decrease {feature} by 20-30%",
                metric_to_watch=target,
                expected_lift=f"{abs(correlation) * 10:.0f}-{abs(correlation) * 20:.0f}%",
                confidence=p.get("confidence", "medium"),
                rationale=p["note"],
            ))

    # Generate from categorical patterns
    for p in patterns.get("categorical_patterns", []):
        if len(suggestions) >= n_experiments:
            break

        feature = p["feature"]
        best_value = p["best_value"]
        lift = p["best_lift_pct"]

        suggestions.append(ExperimentSuggestion(
            title=f"Test {feature}='{best_value}'",
            hypothesis=f"'{best_value}' performs better than other {feature} values",
            variant_a=f"Current mix of {feature} values",
            variant_b=f"Use '{best_value}' for {feature}",
            metric_to_watch=target,
            expected_lift=f"{abs(lift):.0f}%",
            confidence="medium",
            rationale=p["note"],
        ))

    return suggestions[:n_experiments]


def generate_ab_test(
    df: pd.DataFrame,
    target: str,
    test_feature: str,
    variant_a_condition: str,
    variant_b_condition: str,
) -> Dict[str, Any]:
    """
    Generate an A/B test design based on a feature.

    Args:
        df: Input DataFrame
        target: Target metric
        test_feature: Feature to test
        variant_a_condition: Condition for variant A (e.g., "< 50")
        variant_b_condition: Condition for variant B (e.g., ">= 50")

    Returns:
        A/B test design with sample sizes and power analysis
    """
    target_series = pd.to_numeric(df[target], errors="coerce")
    baseline_mean = target_series.mean()
    baseline_std = target_series.std()

    # Estimate required sample size for 80% power
    # Using simplified formula: n = 16 * (std/effect)^2
    min_effect = baseline_mean * 0.05  # Detect 5% change
    if min_effect > 0 and baseline_std > 0:
        required_n = int(16 * (baseline_std / min_effect) ** 2)
    else:
        required_n = 100

    return {
        "test_feature": test_feature,
        "variant_a": {
            "name": "Control",
            "condition": variant_a_condition,
        },
        "variant_b": {
            "name": "Treatment",
            "condition": variant_b_condition,
        },
        "metric": target,
        "baseline_mean": float(baseline_mean),
        "baseline_std": float(baseline_std),
        "recommended_sample_per_variant": min(required_n, 500),
        "minimum_runtime_days": 7,
        "notes": [
            "Run for at least 1 full week to account for day-of-week effects",
            "Don't peek at results early - wait for full sample",
            "If results are inconclusive, the effect may be smaller than expected",
        ],
    }
