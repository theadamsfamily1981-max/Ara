#!/usr/bin/env python3
# ara/narrative/dojo_hook.py
"""
Dojo Narrative Hook: Publishes narrative updates during evolution.

Connects the NarrativeEngine to MEIS/NIB governance loops,
enabling fitness bonuses for agents that can explain themselves.
"""

import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .interface import (
    NarrativeEngine,
    NarrativeStreamer,
    SystemMetrics,
    LifecyclePhase,
    PHASE_PROFILES,
)

logger = logging.getLogger(__name__)


@dataclass
class NarrativeTrace:
    """A recorded narrative event for MEIS analysis."""
    generation: int
    timestamp: float
    phase: str
    efficiency: Dict[str, float]
    narrative_quality: float  # 0-1 score
    explanation_clarity: float  # 0-1 score


class NarrativeGovernanceAdapter:
    """
    Adapter that translates Dojo metrics -> Narrative reports -> MEIS scores.

    This enables the evolutionary loop to reward agents that can
    explain their decisions clearly (narrative fitness bonus).
    """

    def __init__(
        self,
        narrative_engine: Optional[NarrativeEngine] = None,
        explanation_weight: float = 0.1
    ):
        """
        Initialize the governance adapter.

        Args:
            narrative_engine: NarrativeEngine instance (creates default if None)
            explanation_weight: Weight for narrative bonus in fitness (0-1)
        """
        self.engine = narrative_engine or NarrativeEngine()
        self.streamer = NarrativeStreamer(self.engine)
        self.explanation_weight = explanation_weight

        # Narrative history for MEIS
        self.narrative_trace: List[NarrativeTrace] = []
        self.generation_reports: Dict[int, Dict[str, Any]] = {}

        logger.info(
            "NarrativeGovernanceAdapter initialized (explanation_weight=%.2f)",
            explanation_weight
        )

    def on_generation_complete(
        self,
        generation: int,
        population_metrics: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Called by Dojo after each generation evaluation.

        Converts population metrics to SystemMetrics and generates narrative.

        Args:
            generation: Current generation number
            population_metrics: Dict containing:
                - agents: List of agent metric dicts
                - avg_latency_ms: Average latency
                - cpu_util: CPU utilization (0-100)
                - avg_horizon: Average planning horizon
                - avg_futures: Average futures explored
                - safety_rate: Safety prevention rate (0-1)
                - total_violations: Total covenant violations
                - avg_entropy: Average entropy cost
                - avg_reversibility: Average reversibility score
                - total_decisions: Total decisions across population
                - uptime_hours: Hours since training started

        Returns:
            Dict of narrative reports by audience type
        """
        # Aggregate population metrics
        system_metrics = self._aggregate_metrics(population_metrics)

        # Publish narrative
        reports = self.streamer.publish_update(system_metrics)

        # Store for MEIS analysis
        efficiency = reports['operator'].get('efficiency', {})
        trace = NarrativeTrace(
            generation=generation,
            timestamp=time.time(),
            phase=reports['operator'].get('phase', 'unknown'),
            efficiency=efficiency,
            narrative_quality=self._assess_narrative_quality(reports),
            explanation_clarity=self._assess_explanation_clarity(reports)
        )
        self.narrative_trace.append(trace)
        self.generation_reports[generation] = reports

        logger.info(
            "Generation %d narrative: phase=%s, efficiency=%.1f%%",
            generation,
            trace.phase,
            efficiency.get('overall_efficiency', 0)
        )

        return reports

    def _aggregate_metrics(
        self,
        pop_metrics: Dict[str, Any]
    ) -> SystemMetrics:
        """Convert population-level metrics to system-level."""

        agents = pop_metrics.get('agents', [])
        n_agents = len(agents) if agents else 1

        # Average across population
        if agents:
            avg_throughput = sum(
                a.get('throughput', 0) for a in agents
            ) / n_agents
            avg_accuracy = sum(
                a.get('prediction_accuracy', 0) for a in agents
            ) / n_agents
        else:
            avg_throughput = pop_metrics.get('avg_throughput', 0)
            avg_accuracy = pop_metrics.get('avg_accuracy', 0)

        total_decisions = pop_metrics.get('total_decisions', 0)
        uptime = pop_metrics.get('uptime_hours', 0)

        # Determine phase
        efficiency_estimate = pop_metrics.get('avg_efficiency', 50)
        phase = self.engine.determine_phase(total_decisions, efficiency_estimate)

        return SystemMetrics(
            timestamp=time.time(),
            throughput_agents_per_sec=avg_throughput,
            latency_ms=pop_metrics.get('avg_latency_ms', 50),
            cpu_utilization_pct=pop_metrics.get('cpu_util', 50),
            planning_horizon_steps=int(pop_metrics.get('avg_horizon', 5)),
            futures_explored_count=int(pop_metrics.get('avg_futures', 100)),
            safety_prevented_pct=pop_metrics.get('safety_rate', 0.8) * 100,
            prediction_accuracy_pct=avg_accuracy * 100,
            covenant_violations=pop_metrics.get('total_violations', 0),
            entropy_cost_bits=pop_metrics.get('avg_entropy', 1.0),
            reversibility_score=pop_metrics.get('avg_reversibility', 0.7),
            total_decisions=total_decisions,
            current_phase=phase,
            hours_since_deployment=uptime
        )

    def _assess_narrative_quality(
        self,
        reports: Dict[str, Dict[str, Any]]
    ) -> float:
        """
        Assess overall narrative quality.

        Higher scores for:
        - Complete information coverage
        - Consistent phase characterization
        - Clear struggle articulation
        """
        score = 0.0

        # Check that all audiences generated narratives
        for audience in ['operator', 'public', 'technical', 'mythic']:
            if audience in reports and reports[audience].get('narrative'):
                score += 0.2

        # Check efficiency data completeness
        efficiency = reports.get('operator', {}).get('efficiency', {})
        if all(k in efficiency for k in ['overall_efficiency', 'throughput_pct', 'cognitive_pct']):
            score += 0.2

        return min(score, 1.0)

    def _assess_explanation_clarity(
        self,
        reports: Dict[str, Dict[str, Any]]
    ) -> float:
        """
        Assess how well the narrative explains the system state.

        Looks for:
        - Specific metrics mentioned
        - Clear bottleneck identification
        - Actionable information
        """
        technical = reports.get('technical', {}).get('narrative', '')
        operator = reports.get('operator', {}).get('narrative', '')

        score = 0.0

        # Technical report has specific numbers
        if any(c.isdigit() for c in technical):
            score += 0.3

        # Operator report identifies struggle/bottleneck
        if 'Struggle' in operator or 'bottleneck' in operator.lower():
            score += 0.3

        # Reports have status assessment
        if any(marker in operator for marker in ['[OK]', '[WARN]', '[CRIT]']):
            score += 0.2

        # Phase is characterized
        if reports.get('operator', {}).get('phase'):
            score += 0.2

        return min(score, 1.0)

    def compute_narrative_fitness(
        self,
        agent_metrics: Dict[str, Any],
        generation: Optional[int] = None
    ) -> float:
        """
        MEIS extension: Fitness bonus for coherent self-narration.

        Agents that can explain their decisions clearly get fitness boost.
        This encourages development of interpretable behaviors.

        Args:
            agent_metrics: Individual agent metrics
            generation: Generation number (uses latest if None)

        Returns:
            Fitness bonus (0 to explanation_weight)
        """
        # Get relevant generation report
        if generation is not None and generation in self.generation_reports:
            reports = self.generation_reports[generation]
        elif self.narrative_trace:
            latest = self.narrative_trace[-1]
            reports = self.generation_reports.get(latest.generation, {})
        else:
            return 0.0

        # Base clarity from narrative quality
        explanation_clarity = self._assess_explanation_clarity(reports)

        # Agent-specific bonus based on their contribution to clarity
        # Agents with lower variance in predictions are easier to explain
        prediction_variance = agent_metrics.get('prediction_variance', 1.0)
        stability_bonus = max(0, 1 - prediction_variance)

        # Combined score
        raw_score = 0.6 * explanation_clarity + 0.4 * stability_bonus

        return self.explanation_weight * raw_score

    def get_phase_transition_history(self) -> List[Dict[str, Any]]:
        """Get history of phase transitions for visualization."""
        return self.engine.phase_transitions

    def get_narrative_summary(self, n_recent: int = 5) -> Dict[str, Any]:
        """
        Get summary of recent narrative trace for dashboard.

        Args:
            n_recent: Number of recent traces to include

        Returns:
            Summary dict with trends and current state
        """
        if not self.narrative_trace:
            return {
                'current_phase': 'unknown',
                'efficiency_trend': [],
                'recent_traces': [],
                'phase_transitions': []
            }

        recent = self.narrative_trace[-n_recent:]
        current = recent[-1]

        efficiency_trend = [
            {
                'generation': t.generation,
                'efficiency': t.efficiency.get('overall_efficiency', 0)
            }
            for t in recent
        ]

        return {
            'current_phase': current.phase,
            'current_efficiency': current.efficiency,
            'efficiency_trend': efficiency_trend,
            'narrative_quality': current.narrative_quality,
            'explanation_clarity': current.explanation_clarity,
            'recent_traces': [
                {
                    'generation': t.generation,
                    'phase': t.phase,
                    'efficiency': t.efficiency.get('overall_efficiency', 0)
                }
                for t in recent
            ],
            'phase_transitions': self.get_phase_transition_history()
        }

    def reset(self) -> None:
        """Reset trace history (for new training runs)."""
        self.narrative_trace.clear()
        self.generation_reports.clear()
        self.engine.phase_transitions.clear()
        self.engine.last_phase = None
        logger.info("NarrativeGovernanceAdapter reset")


# ============================================================================
# Factory Function
# ============================================================================

def create_narrative_adapter(
    metrics_path: Optional[str] = None,
    explanation_weight: float = 0.1
) -> NarrativeGovernanceAdapter:
    """
    Create a configured NarrativeGovernanceAdapter.

    Args:
        metrics_path: Path to baseline metrics JSON
        explanation_weight: Weight for narrative fitness bonus

    Returns:
        Configured adapter instance
    """
    engine = NarrativeEngine(metrics_path=metrics_path)
    return NarrativeGovernanceAdapter(
        narrative_engine=engine,
        explanation_weight=explanation_weight
    )


__all__ = [
    'NarrativeGovernanceAdapter',
    'NarrativeTrace',
    'create_narrative_adapter',
]
