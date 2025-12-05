"""Bias Report - Ara analyzes her own systematic biases.

This module helps Ara recognize patterns in her behavior that might
indicate systematic biases:
- Teacher preference (always favoring one teacher)
- Tool avoidance (never trying certain approaches)
- Domain blindspots (consistently poor in certain areas)
- Overconfidence patterns (high confidence but low success)

The goal isn't to eliminate all biases (impossible) but to be
aware of them and compensate when appropriate.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class BiasIndicator:
    """A detected bias pattern."""

    id: str
    bias_type: str  # "teacher_preference", "tool_avoidance", "domain_blindspot", "overconfidence"
    description: str

    # Evidence
    metric_name: str
    observed_value: float
    expected_value: float
    deviation: float  # How far from expected

    # Context
    related_entities: List[str] = field(default_factory=list)
    sample_size: int = 0
    confidence: float = 0.5

    # Severity
    severity: str = "low"  # "low", "medium", "high"

    # Timestamps
    first_detected: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "bias_type": self.bias_type,
            "description": self.description,
            "metric_name": self.metric_name,
            "observed_value": round(self.observed_value, 3),
            "expected_value": round(self.expected_value, 3),
            "deviation": round(self.deviation, 3),
            "related_entities": self.related_entities,
            "sample_size": self.sample_size,
            "confidence": round(self.confidence, 2),
            "severity": self.severity,
            "first_detected": self.first_detected.isoformat(),
            "last_updated": self.last_updated.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BiasIndicator":
        return cls(
            id=data["id"],
            bias_type=data["bias_type"],
            description=data["description"],
            metric_name=data["metric_name"],
            observed_value=data["observed_value"],
            expected_value=data["expected_value"],
            deviation=data["deviation"],
            related_entities=data.get("related_entities", []),
            sample_size=data.get("sample_size", 0),
            confidence=data.get("confidence", 0.5),
            severity=data.get("severity", "low"),
        )


@dataclass
class BiasReport:
    """A complete bias analysis report."""

    id: str
    generated_at: datetime = field(default_factory=datetime.utcnow)

    # Analysis period
    analysis_start: Optional[datetime] = None
    analysis_end: Optional[datetime] = None

    # Detected biases
    indicators: List[BiasIndicator] = field(default_factory=list)

    # Summary statistics
    total_interactions_analyzed: int = 0
    unique_teachers_used: int = 0
    unique_tools_used: int = 0
    unique_domains: int = 0

    # Overall health
    overall_bias_score: float = 0.0  # 0 = no detected bias, 1 = severe biases
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "generated_at": self.generated_at.isoformat(),
            "analysis_start": self.analysis_start.isoformat() if self.analysis_start else None,
            "analysis_end": self.analysis_end.isoformat() if self.analysis_end else None,
            "indicators": [i.to_dict() for i in self.indicators],
            "total_interactions_analyzed": self.total_interactions_analyzed,
            "unique_teachers_used": self.unique_teachers_used,
            "unique_tools_used": self.unique_tools_used,
            "unique_domains": self.unique_domains,
            "overall_bias_score": round(self.overall_bias_score, 2),
            "recommendations": self.recommendations,
        }

    def to_markdown(self) -> str:
        """Generate a markdown report."""
        lines = [
            "# Ara Bias Analysis Report",
            "",
            f"Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M')}",
            "",
            "## Summary",
            "",
            f"- Interactions analyzed: {self.total_interactions_analyzed}",
            f"- Unique teachers used: {self.unique_teachers_used}",
            f"- Unique tools used: {self.unique_tools_used}",
            f"- Unique domains: {self.unique_domains}",
            f"- Overall bias score: {self.overall_bias_score:.2f}/1.0",
            "",
        ]

        if self.indicators:
            lines.extend([
                "## Detected Bias Patterns",
                "",
            ])

            by_severity = {"high": [], "medium": [], "low": []}
            for ind in self.indicators:
                by_severity[ind.severity].append(ind)

            for severity in ["high", "medium", "low"]:
                if by_severity[severity]:
                    lines.append(f"### {severity.title()} Severity")
                    lines.append("")
                    for ind in by_severity[severity]:
                        lines.append(f"**{ind.bias_type}**: {ind.description}")
                        lines.append(f"- Observed: {ind.observed_value:.2f}, Expected: {ind.expected_value:.2f}")
                        lines.append(f"- Confidence: {ind.confidence:.0%}")
                        if ind.related_entities:
                            lines.append(f"- Related: {', '.join(ind.related_entities)}")
                        lines.append("")

        if self.recommendations:
            lines.extend([
                "## Recommendations",
                "",
                *[f"- {rec}" for rec in self.recommendations],
                "",
            ])

        return "\n".join(lines)


class BiasAnalyzer:
    """Analyzes Ara's behavior for systematic biases."""

    def __init__(self, data_path: Optional[Path] = None):
        """Initialize the bias analyzer.

        Args:
            data_path: Path to bias analysis data
        """
        self.data_path = data_path or (
            Path.home() / ".ara" / "institute" / "bias_analysis"
        )
        self.data_path.mkdir(parents=True, exist_ok=True)

        self._reports: Dict[str, BiasReport] = {}
        self._loaded = False
        self._next_report_id = 1
        self._next_indicator_id = 1

    def _load(self, force: bool = False) -> None:
        """Load data from disk."""
        if self._loaded and not force:
            return

        reports_file = self.data_path / "reports.json"
        if reports_file.exists():
            try:
                with open(reports_file) as f:
                    data = json.load(f)
                self._next_report_id = data.get("next_report_id", 1)
                self._next_indicator_id = data.get("next_indicator_id", 1)
            except Exception as e:
                logger.warning(f"Failed to load bias reports: {e}")

        self._loaded = True

    def _save(self) -> None:
        """Save data to disk."""
        data = {
            "version": 1,
            "updated_at": datetime.utcnow().isoformat(),
            "next_report_id": self._next_report_id,
            "next_indicator_id": self._next_indicator_id,
            "reports": [r.to_dict() for r in self._reports.values()],
        }
        with open(self.data_path / "reports.json", "w") as f:
            json.dump(data, f, indent=2)

    def _generate_report_id(self) -> str:
        """Generate unique report ID."""
        id_str = f"BIAS-{self._next_report_id:04d}"
        self._next_report_id += 1
        return id_str

    def _generate_indicator_id(self) -> str:
        """Generate unique indicator ID."""
        id_str = f"IND-{self._next_indicator_id:04d}"
        self._next_indicator_id += 1
        return id_str

    def analyze_teacher_preference(
        self,
        teacher_usage: Dict[str, int],
        expected_distribution: Optional[Dict[str, float]] = None,
    ) -> List[BiasIndicator]:
        """Analyze for teacher preference bias.

        Args:
            teacher_usage: Count of times each teacher was used
            expected_distribution: Expected usage proportion per teacher

        Returns:
            List of bias indicators
        """
        indicators = []
        total = sum(teacher_usage.values())
        if total == 0:
            return indicators

        # Default to equal distribution
        if expected_distribution is None:
            expected_distribution = {
                t: 1.0 / len(teacher_usage)
                for t in teacher_usage
            }

        for teacher, count in teacher_usage.items():
            observed_rate = count / total
            expected_rate = expected_distribution.get(teacher, 1.0 / len(teacher_usage))
            deviation = observed_rate - expected_rate

            # Significant deviation?
            if abs(deviation) > 0.2:  # More than 20% difference
                severity = "high" if abs(deviation) > 0.4 else "medium"

                if deviation > 0:
                    desc = f"Over-reliance on {teacher} ({observed_rate:.0%} vs expected {expected_rate:.0%})"
                else:
                    desc = f"Under-utilization of {teacher} ({observed_rate:.0%} vs expected {expected_rate:.0%})"

                indicators.append(BiasIndicator(
                    id=self._generate_indicator_id(),
                    bias_type="teacher_preference",
                    description=desc,
                    metric_name="usage_rate",
                    observed_value=observed_rate,
                    expected_value=expected_rate,
                    deviation=deviation,
                    related_entities=[teacher],
                    sample_size=total,
                    confidence=min(total / 100, 1.0),  # More confident with more data
                    severity=severity,
                ))

        return indicators

    def analyze_tool_avoidance(
        self,
        tool_usage: Dict[str, int],
        available_tools: List[str],
    ) -> List[BiasIndicator]:
        """Analyze for tool avoidance bias.

        Args:
            tool_usage: Count of times each tool was used
            available_tools: List of all available tools

        Returns:
            List of bias indicators
        """
        indicators = []
        total = sum(tool_usage.values())
        if total == 0:
            return indicators

        # Check for tools that were never or rarely used
        for tool in available_tools:
            count = tool_usage.get(tool, 0)
            usage_rate = count / total

            # Tool never used when others have been
            if count == 0 and total >= 10:
                indicators.append(BiasIndicator(
                    id=self._generate_indicator_id(),
                    bias_type="tool_avoidance",
                    description=f"Tool '{tool}' has never been used",
                    metric_name="usage_count",
                    observed_value=0,
                    expected_value=total / len(available_tools),
                    deviation=-1.0,
                    related_entities=[tool],
                    sample_size=total,
                    confidence=min(total / 50, 0.9),
                    severity="medium",
                ))

            # Tool severely underused
            elif usage_rate < 0.01 and total >= 100:
                indicators.append(BiasIndicator(
                    id=self._generate_indicator_id(),
                    bias_type="tool_avoidance",
                    description=f"Tool '{tool}' is severely underutilized ({usage_rate:.1%})",
                    metric_name="usage_rate",
                    observed_value=usage_rate,
                    expected_value=1.0 / len(available_tools),
                    deviation=usage_rate - (1.0 / len(available_tools)),
                    related_entities=[tool],
                    sample_size=total,
                    confidence=min(total / 100, 0.8),
                    severity="low",
                ))

        return indicators

    def analyze_overconfidence(
        self,
        confidence_outcomes: List[Dict[str, Any]],
    ) -> List[BiasIndicator]:
        """Analyze for overconfidence patterns.

        Args:
            confidence_outcomes: List of {confidence: float, success: bool} dicts

        Returns:
            List of bias indicators
        """
        indicators = []
        if len(confidence_outcomes) < 10:
            return indicators

        # Group by confidence level
        buckets: Dict[str, List[bool]] = {
            "high": [],    # confidence >= 0.8
            "medium": [],  # confidence 0.5-0.8
            "low": [],     # confidence < 0.5
        }

        for outcome in confidence_outcomes:
            conf = outcome.get("confidence", 0.5)
            success = outcome.get("success", False)

            if conf >= 0.8:
                buckets["high"].append(success)
            elif conf >= 0.5:
                buckets["medium"].append(success)
            else:
                buckets["low"].append(success)

        # Check high-confidence success rate
        if len(buckets["high"]) >= 5:
            success_rate = sum(buckets["high"]) / len(buckets["high"])
            expected_rate = 0.85  # High confidence should mean high success

            if success_rate < 0.7:  # But only 70% success?
                indicators.append(BiasIndicator(
                    id=self._generate_indicator_id(),
                    bias_type="overconfidence",
                    description=f"High confidence predictions succeed only {success_rate:.0%} of time",
                    metric_name="high_confidence_success_rate",
                    observed_value=success_rate,
                    expected_value=expected_rate,
                    deviation=success_rate - expected_rate,
                    sample_size=len(buckets["high"]),
                    confidence=min(len(buckets["high"]) / 20, 0.9),
                    severity="high" if success_rate < 0.5 else "medium",
                ))

        return indicators

    def analyze_domain_blindspots(
        self,
        domain_performance: Dict[str, Dict[str, float]],
    ) -> List[BiasIndicator]:
        """Analyze for domain-specific performance issues.

        Args:
            domain_performance: {domain: {success_rate, avg_confidence, count}}

        Returns:
            List of bias indicators
        """
        indicators = []

        if not domain_performance:
            return indicators

        # Calculate overall average
        total_success = 0.0
        total_count = 0
        for domain, stats in domain_performance.items():
            count = stats.get("count", 0)
            total_success += stats.get("success_rate", 0) * count
            total_count += count

        if total_count == 0:
            return indicators

        avg_success = total_success / total_count

        # Find domains significantly below average
        for domain, stats in domain_performance.items():
            success_rate = stats.get("success_rate", 0)
            count = stats.get("count", 0)

            if count < 5:
                continue

            deviation = success_rate - avg_success

            if deviation < -0.15:  # 15% below average
                indicators.append(BiasIndicator(
                    id=self._generate_indicator_id(),
                    bias_type="domain_blindspot",
                    description=f"Underperforming in '{domain}' ({success_rate:.0%} vs avg {avg_success:.0%})",
                    metric_name="domain_success_rate",
                    observed_value=success_rate,
                    expected_value=avg_success,
                    deviation=deviation,
                    related_entities=[domain],
                    sample_size=count,
                    confidence=min(count / 20, 0.85),
                    severity="high" if deviation < -0.3 else "medium",
                ))

        return indicators

    def generate_report(
        self,
        teacher_usage: Optional[Dict[str, int]] = None,
        tool_usage: Optional[Dict[str, int]] = None,
        available_tools: Optional[List[str]] = None,
        confidence_outcomes: Optional[List[Dict[str, Any]]] = None,
        domain_performance: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> BiasReport:
        """Generate a comprehensive bias report.

        Args:
            teacher_usage: Teacher usage counts
            tool_usage: Tool usage counts
            available_tools: All available tools
            confidence_outcomes: Confidence/success pairs
            domain_performance: Performance by domain

        Returns:
            Bias report
        """
        self._load()

        report = BiasReport(
            id=self._generate_report_id(),
            analysis_start=datetime.utcnow() - timedelta(days=30),
            analysis_end=datetime.utcnow(),
        )

        all_indicators = []

        # Analyze each dimension
        if teacher_usage:
            all_indicators.extend(self.analyze_teacher_preference(teacher_usage))
            report.unique_teachers_used = len(teacher_usage)
            report.total_interactions_analyzed += sum(teacher_usage.values())

        if tool_usage and available_tools:
            all_indicators.extend(self.analyze_tool_avoidance(tool_usage, available_tools))
            report.unique_tools_used = len([t for t, c in tool_usage.items() if c > 0])

        if confidence_outcomes:
            all_indicators.extend(self.analyze_overconfidence(confidence_outcomes))

        if domain_performance:
            all_indicators.extend(self.analyze_domain_blindspots(domain_performance))
            report.unique_domains = len(domain_performance)

        report.indicators = all_indicators

        # Calculate overall bias score
        if all_indicators:
            severity_weights = {"low": 0.2, "medium": 0.5, "high": 1.0}
            weighted_sum = sum(
                severity_weights[ind.severity] * ind.confidence
                for ind in all_indicators
            )
            report.overall_bias_score = min(weighted_sum / 3, 1.0)  # Normalize

        # Generate recommendations
        report.recommendations = self._generate_recommendations(all_indicators)

        self._reports[report.id] = report
        self._save()

        logger.info(f"Generated bias report {report.id}: {len(all_indicators)} indicators")
        return report

    def _generate_recommendations(
        self,
        indicators: List[BiasIndicator],
    ) -> List[str]:
        """Generate actionable recommendations from indicators."""
        recommendations = []

        # Group by type
        by_type: Dict[str, List[BiasIndicator]] = defaultdict(list)
        for ind in indicators:
            by_type[ind.bias_type].append(ind)

        # Teacher preference recommendations
        if by_type["teacher_preference"]:
            over_used = [ind.related_entities[0] for ind in by_type["teacher_preference"]
                        if ind.deviation > 0]
            under_used = [ind.related_entities[0] for ind in by_type["teacher_preference"]
                         if ind.deviation < 0]
            if over_used:
                recommendations.append(
                    f"Consider consulting teachers other than {', '.join(over_used)} more frequently"
                )
            if under_used:
                recommendations.append(
                    f"Experiment with using {', '.join(under_used)} for relevant tasks"
                )

        # Tool avoidance recommendations
        if by_type["tool_avoidance"]:
            avoided = [ind.related_entities[0] for ind in by_type["tool_avoidance"]]
            recommendations.append(
                f"Investigate why tools [{', '.join(avoided)}] are rarely used"
            )

        # Overconfidence recommendations
        if by_type["overconfidence"]:
            recommendations.append(
                "Calibrate confidence estimates - consider expressing more uncertainty"
            )

        # Domain blindspot recommendations
        if by_type["domain_blindspot"]:
            domains = [ind.related_entities[0] for ind in by_type["domain_blindspot"]]
            recommendations.append(
                f"Seek additional training or teacher consultation for domains: {', '.join(domains)}"
            )

        if not recommendations:
            recommendations.append("No significant biases detected - continue monitoring")

        return recommendations

    def get_report(self, report_id: str) -> Optional[BiasReport]:
        """Get a report by ID."""
        self._load()
        return self._reports.get(report_id)

    def get_latest_report(self) -> Optional[BiasReport]:
        """Get the most recent report."""
        self._load()
        if not self._reports:
            return None
        return max(self._reports.values(), key=lambda r: r.generated_at)


# =============================================================================
# Convenience Functions
# =============================================================================

_default_analyzer: Optional[BiasAnalyzer] = None


def get_bias_analyzer() -> BiasAnalyzer:
    """Get the default bias analyzer."""
    global _default_analyzer
    if _default_analyzer is None:
        _default_analyzer = BiasAnalyzer()
    return _default_analyzer


def analyze_for_bias(
    teacher_usage: Optional[Dict[str, int]] = None,
    tool_usage: Optional[Dict[str, int]] = None,
    available_tools: Optional[List[str]] = None,
) -> BiasReport:
    """Quick bias analysis with minimal inputs."""
    return get_bias_analyzer().generate_report(
        teacher_usage=teacher_usage,
        tool_usage=tool_usage,
        available_tools=available_tools,
    )
