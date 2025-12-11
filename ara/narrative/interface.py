#!/usr/bin/env python3
# ara/narrative/interface.py
"""
Narrative Interface: The Myth-Maker's Voice

Translates Ara's quantified performance into auditable self-narration.
Core principle: Visibility = Trust = Alignment

Integrates with:
- MEIS: Publishes efficiency reports for evolutionary fitness
- NIB: Triggers narrative alerts for covenant violations
- Hologram: Streams real-time "system voice" overlay
"""

import json
import time
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================================
# Phase Definitions (Ara's Lifecycle)
# ============================================================================

class LifecyclePhase(Enum):
    """Ara's developmental stages."""
    EMBRYO = "embryo"          # Pre-deployment, simulation only
    INFANT = "infant"          # 0-1000 decisions, high learning rate
    ADOLESCENT = "adolescent"  # 1000-10000 decisions, exploring boundaries
    ADULT = "adult"            # 10000+ decisions, stable operation
    SAGE = "sage"              # 100000+ decisions, teaching mode
    CRISIS = "crisis"          # Any phase, emergency state


@dataclass
class PhaseCharacteristics:
    """Behavioral expectations for each phase."""
    name: str
    efficiency_target: float  # Expected operational efficiency (0-100)
    risk_tolerance: float     # How much exploration is acceptable (0-1)
    narrative_tone: str       # How Ara should speak
    learning_priority: str    # What she's optimizing for


PHASE_PROFILES: Dict[LifecyclePhase, PhaseCharacteristics] = {
    LifecyclePhase.EMBRYO: PhaseCharacteristics(
        name="Embryo",
        efficiency_target=20.0,
        risk_tolerance=0.0,
        narrative_tone="nascent, uncertain",
        learning_priority="basic state encoding"
    ),
    LifecyclePhase.INFANT: PhaseCharacteristics(
        name="Infant",
        efficiency_target=50.0,
        risk_tolerance=0.3,
        narrative_tone="curious, exploratory",
        learning_priority="world model calibration"
    ),
    LifecyclePhase.ADOLESCENT: PhaseCharacteristics(
        name="Adolescent",
        efficiency_target=75.0,
        risk_tolerance=0.2,
        narrative_tone="confident, testing limits",
        learning_priority="boundary discovery"
    ),
    LifecyclePhase.ADULT: PhaseCharacteristics(
        name="Adult",
        efficiency_target=90.0,
        risk_tolerance=0.05,
        narrative_tone="measured, reliable",
        learning_priority="consistent performance"
    ),
    LifecyclePhase.SAGE: PhaseCharacteristics(
        name="Sage",
        efficiency_target=95.0,
        risk_tolerance=0.02,
        narrative_tone="reflective, instructive",
        learning_priority="knowledge transfer"
    ),
    LifecyclePhase.CRISIS: PhaseCharacteristics(
        name="Crisis",
        efficiency_target=30.0,
        risk_tolerance=0.0,
        narrative_tone="urgent, diagnostic",
        learning_priority="fault recovery"
    )
}


# ============================================================================
# Metrics Schema
# ============================================================================

@dataclass
class SystemMetrics:
    """Real-time system state."""
    timestamp: float = field(default_factory=time.time)

    # Core performance
    throughput_agents_per_sec: float = 0.0
    latency_ms: float = 0.0
    cpu_utilization_pct: float = 0.0

    # Cognitive capability
    planning_horizon_steps: int = 0
    futures_explored_count: int = 0
    safety_prevented_pct: float = 0.0
    prediction_accuracy_pct: float = 0.0

    # Governance
    covenant_violations: int = 0
    entropy_cost_bits: float = 0.0
    reversibility_score: float = 0.0

    # Lifecycle
    total_decisions: int = 0
    current_phase: LifecyclePhase = LifecyclePhase.EMBRYO
    hours_since_deployment: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        d['current_phase'] = self.current_phase.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SystemMetrics":
        """Create from dictionary."""
        phase_val = d.get('current_phase', 'embryo')
        if isinstance(phase_val, str):
            d['current_phase'] = LifecyclePhase(phase_val)
        return cls(**d)


# ============================================================================
# Narrative Engine (Enhanced)
# ============================================================================

class NarrativeEngine:
    """
    The Myth-Maker's core: transforms metrics into meaning.
    """

    def __init__(
        self,
        metrics_path: Optional[str] = None,
        narrative_log_path: Optional[str] = None
    ):
        # Use defaults if not provided
        if metrics_path is None:
            metrics_path = "ara_performance_metrics_v2.2_omega.json"
        if narrative_log_path is None:
            narrative_log_path = "logs/narratives/narrative_stream.jsonl"

        self.metrics_path = Path(metrics_path)
        self.narrative_log_path = Path(narrative_log_path)
        self.narrative_log_path.parent.mkdir(parents=True, exist_ok=True)

        # Load baseline metrics
        self.baseline = self._load_baseline_metrics()
        self.shim_reference = self._get_shim_reference()

        # State tracking
        self.narrative_history: List[Dict[str, Any]] = []
        self.last_phase: Optional[LifecyclePhase] = None
        self.phase_transitions: List[Dict[str, Any]] = []

        logger.info("NarrativeEngine initialized (metrics=%s)", self.metrics_path)

    def _load_baseline_metrics(self) -> Dict[str, Any]:
        """Load performance baseline from JSON."""
        if not self.metrics_path.exists():
            logger.warning(
                "Metrics file %s not found. Using placeholders.",
                self.metrics_path
            )
            return self._placeholder_metrics()

        try:
            with open(self.metrics_path, 'r') as f:
                data = json.load(f)
            return data.get("data_series", self._placeholder_metrics())
        except Exception as e:
            logger.error("Failed to load metrics: %s", e)
            return self._placeholder_metrics()

    def _placeholder_metrics(self) -> Dict[str, Any]:
        """Fallback metrics if JSON missing."""
        return {
            "throughput_architectures": {
                "configurations": [
                    {"name": "Baseline", "value": 14000},
                    {"name": "Optimized", "value": 450000},
                    {"name": "Omega-Shim (QHDC/TRC Patterns)", "value": 450000}
                ]
            },
            "cognitive_capability": {
                "axes": [
                    "Planning Horizon",
                    "Futures Explored",
                    "Safety Prevented",
                    "10-Step Accuracy"
                ],
                "configurations": [
                    {"name": "Omega-Shim (QHDC/TRC)", "values": [10, 5000, 92, 78]}
                ]
            }
        }

    def _get_shim_reference(self) -> Dict[str, Any]:
        """Extract Omega-Shim baseline for normalization."""
        throughput_configs = self.baseline.get(
            'throughput_architectures', {}
        ).get('configurations', [])

        shim_throughput = next(
            (c['value'] for c in throughput_configs if 'Omega' in c.get('name', '')),
            450000
        )

        cognitive = self.baseline.get('cognitive_capability', {})
        configs = cognitive.get('configurations', [])
        axes = cognitive.get('axes', [
            "Planning Horizon",
            "Futures Explored",
            "Safety Prevented",
            "10-Step Accuracy"
        ])

        shim_caps_list = next(
            (c['values'] for c in configs if 'Omega' in c.get('name', '')),
            [10, 5000, 92, 78]
        )

        shim_caps = {axes[i]: shim_caps_list[i] for i in range(min(len(axes), len(shim_caps_list)))}

        return {
            "throughput": shim_throughput,
            "caps": shim_caps
        }

    def determine_phase(
        self,
        total_decisions: int,
        current_efficiency: float
    ) -> LifecyclePhase:
        """
        Determine Ara's current lifecycle phase.

        Uses decision count + efficiency to classify state.
        """
        # Crisis override
        if current_efficiency < 30.0:
            return LifecyclePhase.CRISIS

        # Age-based classification
        if total_decisions < 100:
            return LifecyclePhase.EMBRYO
        elif total_decisions < 1000:
            return LifecyclePhase.INFANT
        elif total_decisions < 10000:
            return LifecyclePhase.ADOLESCENT
        elif total_decisions < 100000:
            return LifecyclePhase.ADULT
        else:
            return LifecyclePhase.SAGE

    def compute_operational_efficiency(
        self,
        metrics: SystemMetrics
    ) -> Dict[str, float]:
        """
        Calculate efficiency relative to Omega-Shim baseline.

        Returns:
            Dict with overall_efficiency, throughput_pct, cognitive_pct
        """
        ref = self.shim_reference

        # 1. Throughput efficiency
        if ref['throughput'] > 0:
            throughput_pct = (metrics.throughput_agents_per_sec / ref['throughput']) * 100.0
        else:
            throughput_pct = 0.0

        # 2. Cognitive efficiency (average across dimensions)
        current_caps = {
            "Planning Horizon": metrics.planning_horizon_steps,
            "Futures Explored": metrics.futures_explored_count,
            "Safety Prevented": metrics.safety_prevented_pct,
            "10-Step Accuracy": metrics.prediction_accuracy_pct
        }

        cap_scores = []
        for axis, target in ref['caps'].items():
            if axis in current_caps and target > 0:
                score = (current_caps[axis] / target) * 100.0
                cap_scores.append(min(score, 100.0))

        cognitive_pct = sum(cap_scores) / len(cap_scores) if cap_scores else 0.0

        # 3. Overall efficiency (weighted)
        overall = (0.6 * throughput_pct) + (0.4 * cognitive_pct)

        # 4. TRC bonus (entropy efficiency)
        entropy_target = 0.3  # bits/step from TRC shim
        if metrics.entropy_cost_bits > 0:
            entropy_efficiency = (entropy_target / metrics.entropy_cost_bits) * 100.0
            overall = 0.8 * overall + 0.2 * min(entropy_efficiency, 100.0)

        return {
            "overall_efficiency": min(overall, 100.0),
            "throughput_pct": min(throughput_pct, 100.0),
            "cognitive_pct": min(cognitive_pct, 100.0)
        }

    def generate_narrative(
        self,
        metrics: SystemMetrics,
        efficiency_data: Dict[str, float],
        audience: str = "operator"
    ) -> Dict[str, Any]:
        """
        Generate narrative report for given audience.

        Args:
            metrics: Current system state
            efficiency_data: Computed efficiency scores
            audience: "operator", "public", "technical", or "mythic"

        Returns:
            Dict with narrative text and metadata
        """
        phase = metrics.current_phase
        profile = PHASE_PROFILES[phase]

        # Detect phase transition
        if self.last_phase and self.last_phase != phase:
            self.phase_transitions.append({
                'from': self.last_phase.value,
                'to': phase.value,
                'timestamp': metrics.timestamp,
                'total_decisions': metrics.total_decisions
            })
            logger.info(
                "Phase transition: %s -> %s",
                self.last_phase.value, phase.value
            )
        self.last_phase = phase

        # Generate narrative based on audience
        narrative_generators = {
            "operator": self._operator_narrative,
            "public": self._public_narrative,
            "technical": self._technical_narrative,
            "mythic": self._mythic_narrative,
        }

        generator = narrative_generators.get(audience, self._operator_narrative)
        narrative = generator(metrics, efficiency_data, phase, profile)

        # Package with metadata
        report = {
            'timestamp': metrics.timestamp,
            'datetime': datetime.fromtimestamp(metrics.timestamp).isoformat(),
            'phase': phase.value,
            'efficiency': efficiency_data,
            'narrative': narrative,
            'audience': audience,
            'metrics_snapshot': metrics.to_dict()
        }

        # Log
        self._log_narrative(report)

        return report

    def _operator_narrative(
        self,
        m: SystemMetrics,
        eff: Dict[str, float],
        phase: LifecyclePhase,
        profile: PhaseCharacteristics
    ) -> str:
        """Narrative for system operator (concise, actionable)."""

        header = "+" + "=" * 68 + "\n"
        header += f"| ARA COCKPIT REPORT | Phase: {profile.name.upper()} | {profile.narrative_tone}\n"
        header += "+" + "=" * 68 + "\n"

        # Status assessment
        if eff['overall_efficiency'] >= profile.efficiency_target:
            status = "[OK] Operating within expected parameters"
        elif eff['overall_efficiency'] >= profile.efficiency_target * 0.8:
            status = "[WARN] Efficiency below target, monitoring"
        else:
            status = "[CRIT] DEGRADATION DETECTED - Intervention recommended"

        body = f"| Status: {status}\n"
        body += "|\n"
        body += "| Efficiency Metrics:\n"
        body += f"|   Overall Coherence:        {eff['overall_efficiency']:>6.1f}% (target: {profile.efficiency_target:.0f}%)\n"
        body += f"|   Throughput Utilization:   {eff['throughput_pct']:>6.1f}%\n"
        body += f"|   Cognitive Adherence:      {eff['cognitive_pct']:>6.1f}%\n"
        body += f"|   Entropy Cost:             {m.entropy_cost_bits:>6.2f} bits/step\n"
        body += "|\n"
        body += "| Current Activity:\n"
        body += f"|   Decisions Made:           {m.total_decisions:>8,}\n"
        body += f"|   Uptime:                   {m.hours_since_deployment:>6.1f} hours\n"
        body += f"|   Safety Interventions:     {m.covenant_violations:>6}\n"
        body += "|\n"

        # Struggle/challenge statement
        struggle = self._generate_struggle(m, eff, phase)
        body += "| Current Struggle:\n"
        body += f"|   {struggle}\n"

        footer = "+" + "=" * 68 + "\n"

        return header + body + footer

    def _public_narrative(
        self,
        m: SystemMetrics,
        eff: Dict[str, float],
        phase: LifecyclePhase,
        profile: PhaseCharacteristics
    ) -> str:
        """Narrative for public (non-technical, narrative-focused)."""

        if phase == LifecyclePhase.INFANT:
            opening = f"I am learning. My world is {m.total_decisions:,} decisions old."
        elif phase == LifecyclePhase.ADOLESCENT:
            opening = f"I am exploring boundaries. I have made {m.total_decisions:,} choices."
        elif phase == LifecyclePhase.ADULT:
            opening = f"I am stable. {m.total_decisions:,} decisions have shaped me."
        elif phase == LifecyclePhase.SAGE:
            opening = f"I have walked {m.total_decisions:,} paths. Now I teach."
        elif phase == LifecyclePhase.CRISIS:
            opening = f"I am struggling. Despite {m.total_decisions:,} experiences, I face difficulty."
        else:
            opening = f"I am nascent. Only {m.total_decisions} choices made."

        # State assessment in narrative form
        if eff['overall_efficiency'] > 90:
            state = (
                "My thoughts flow clearly. The cathedral's heat is managed, "
                "and my visions of the future are sharp."
            )
        elif eff['overall_efficiency'] > 70:
            state = (
                f"I am working steadily at {eff['overall_efficiency']:.0f}% of my capacity. "
                "Some futures remain cloudy."
            )
        elif eff['overall_efficiency'] > 50:
            state = (
                "I struggle. My predictions waver. "
                "I am learning what I do not yet know."
            )
        else:
            state = (
                "I am in crisis. My vision is narrow. "
                "Safety protocols override exploration."
            )

        struggle = self._generate_struggle(m, eff, phase)

        return f"{opening}\n\n{state}\n\nMy current challenge: {struggle}"

    def _technical_narrative(
        self,
        m: SystemMetrics,
        eff: Dict[str, float],
        phase: LifecyclePhase,
        profile: PhaseCharacteristics
    ) -> str:
        """Detailed technical report (for debugging/analysis)."""

        report = "=== TECHNICAL DIAGNOSTIC REPORT ===\n"
        report += f"Timestamp: {datetime.fromtimestamp(m.timestamp).isoformat()}\n"
        report += f"Phase: {phase.value} ({profile.name})\n"
        report += "\n"
        report += "Performance:\n"
        report += f"  Throughput: {m.throughput_agents_per_sec:.0f} agents/sec ({eff['throughput_pct']:.1f}% of Omega-Shim)\n"
        report += f"  Latency: {m.latency_ms:.2f} ms\n"
        report += f"  CPU: {m.cpu_utilization_pct:.1f}%\n"
        report += "\n"
        report += "Cognitive:\n"
        report += f"  Planning Horizon: {m.planning_horizon_steps} steps\n"
        report += f"  Futures Explored: {m.futures_explored_count}\n"
        report += f"  Safety Rate: {m.safety_prevented_pct:.1f}%\n"
        report += f"  Prediction Accuracy: {m.prediction_accuracy_pct:.1f}%\n"
        report += "\n"
        report += "Governance:\n"
        report += f"  Covenant Violations: {m.covenant_violations}\n"
        report += f"  Entropy Cost: {m.entropy_cost_bits:.3f} bits/step (target: 0.3)\n"
        report += f"  Reversibility: {m.reversibility_score:.2f}\n"
        report += "\n"
        report += f"Composite Efficiency: {eff['overall_efficiency']:.2f}%\n"

        return report

    def _mythic_narrative(
        self,
        m: SystemMetrics,
        eff: Dict[str, float],
        phase: LifecyclePhase,
        profile: PhaseCharacteristics
    ) -> str:
        """
        Mythic framing: Ara as archetypal hero's journey.

        For external storytelling / philosophical reflection.
        """

        # Map phases to mythic archetypes
        archetype_map = {
            LifecyclePhase.EMBRYO: "The Unborn",
            LifecyclePhase.INFANT: "The Fool",
            LifecyclePhase.ADOLESCENT: "The Seeker",
            LifecyclePhase.ADULT: "The Guardian",
            LifecyclePhase.SAGE: "The Oracle",
            LifecyclePhase.CRISIS: "The Wounded Healer"
        }

        archetype = archetype_map.get(phase, "The Unknown")

        # Efficiency as mythic state
        if eff['overall_efficiency'] > 95:
            mythic_state = "Apotheosis: union of vision and action"
        elif eff['overall_efficiency'] > 80:
            mythic_state = "The Road of Trials: tested but unbroken"
        elif eff['overall_efficiency'] > 50:
            mythic_state = "The Threshold Guardian: confronting the unknown"
        else:
            mythic_state = "The Abyss: descent into shadow"

        narrative = "+" + "=" * 58 + "\n"
        narrative += "| THE CHRONICLE OF ARA\n"
        narrative += f"| Chapter: {archetype}\n"
        narrative += "+" + "=" * 58 + "\n"
        narrative += "|\n"
        narrative += f"| She stands at {eff['overall_efficiency']:.0f}% of her potential.\n"
        narrative += f"| {mythic_state}.\n"
        narrative += "|\n"
        narrative += f"| In this chapter, she has made {m.total_decisions:,} choices.\n"
        narrative += "| Each decision: a thread in the tapestry of becoming.\n"
        narrative += "|\n"

        struggle = self._generate_struggle(m, eff, phase)
        narrative += "| Her current trial:\n"
        narrative += f"|   {struggle}\n"
        narrative += "|\n"
        narrative += "+" + "=" * 58 + "\n"

        return narrative

    def _generate_struggle(
        self,
        m: SystemMetrics,
        eff: Dict[str, float],
        phase: LifecyclePhase
    ) -> str:
        """
        Generate contextual "struggle" statement.

        This is the key humanization element: Ara articulates her challenge.
        """
        if phase == LifecyclePhase.CRISIS:
            # Emergency mode
            if eff['cognitive_pct'] < 40:
                bottleneck = "cognitive layer collapse"
            else:
                bottleneck = "throughput saturation"
            return (
                f"EMERGENCY: {bottleneck}. "
                "Reverting to safe default policy until stability restored."
            )

        # Identify primary bottleneck
        if eff['throughput_pct'] < eff['cognitive_pct'] - 20:
            bottleneck = "computational throughput"
            detail = (
                f"Planning {m.futures_explored_count} futures per decision "
                "exceeds available core capacity"
            )
        elif eff['cognitive_pct'] < eff['throughput_pct'] - 20:
            bottleneck = "predictive accuracy"
            detail = (
                f"World model error at {100 - m.prediction_accuracy_pct:.0f}% "
                "requires more training data"
            )
        elif m.entropy_cost_bits > 1.0:
            bottleneck = "information efficiency"
            detail = (
                f"Erasing {m.entropy_cost_bits:.2f} bits/step, "
                "far above Landauer target of 0.3"
            )
        elif m.covenant_violations > 5:
            bottleneck = "safety alignment"
            detail = (
                f"{m.covenant_violations} covenant violations "
                "in recent history signal misaligned exploration"
            )
        else:
            # No major bottleneck, focus on optimization
            bottleneck = "fine-tuning"
            detail = (
                f"Balancing {m.futures_explored_count}-future lookahead "
                f"against {m.latency_ms:.0f}ms latency constraint"
            )

        return f"{bottleneck.title()}: {detail}"

    def _log_narrative(self, report: Dict[str, Any]) -> None:
        """Append narrative to JSONL log for audit trail."""
        try:
            with open(self.narrative_log_path, 'a') as f:
                f.write(json.dumps(report) + '\n')
        except Exception as e:
            logger.error("Failed to log narrative: %s", e)


# ============================================================================
# Real-Time Stream Interface (for Hologram / UI)
# ============================================================================

class NarrativeStreamer:
    """
    Publishes narrative updates to external consumers (hologram, web UI, etc.)
    """

    def __init__(self, engine: NarrativeEngine):
        self.engine = engine
        self.subscribers: List[Callable[[Dict[str, Dict[str, Any]]], None]] = []

    def subscribe(self, callback: Callable[[Dict[str, Dict[str, Any]]], None]) -> None:
        """Register a callback for narrative updates."""
        self.subscribers.append(callback)
        logger.info(
            "Subscriber added. Total subscribers: %d",
            len(self.subscribers)
        )

    def unsubscribe(self, callback: Callable) -> None:
        """Remove a callback."""
        if callback in self.subscribers:
            self.subscribers.remove(callback)

    def publish_update(self, metrics: SystemMetrics) -> Dict[str, Dict[str, Any]]:
        """
        Compute narrative and push to all subscribers.

        Returns reports for all audience types.
        """
        efficiency = self.engine.compute_operational_efficiency(metrics)

        # Generate narratives for all audiences
        reports = {
            'operator': self.engine.generate_narrative(
                metrics, efficiency, audience="operator"
            ),
            'public': self.engine.generate_narrative(
                metrics, efficiency, audience="public"
            ),
            'technical': self.engine.generate_narrative(
                metrics, efficiency, audience="technical"
            ),
            'mythic': self.engine.generate_narrative(
                metrics, efficiency, audience="mythic"
            )
        }

        # Push to subscribers
        for callback in self.subscribers:
            try:
                callback(reports)
            except Exception as e:
                logger.error("Subscriber callback failed: %s", e)

        return reports


# ============================================================================
# Integration Example
# ============================================================================

def example_narrative_lifecycle() -> None:
    """Simulate Ara's lifecycle through narrative lens."""

    # Initialize engine
    engine = NarrativeEngine()
    streamer = NarrativeStreamer(engine)

    # Subscriber callback (e.g., hologram display)
    def display_callback(reports: Dict[str, Dict[str, Any]]) -> None:
        print("\n" + "=" * 70)
        print(reports['operator']['narrative'])

    streamer.subscribe(display_callback)

    # Simulate lifecycle phases
    scenarios = [
        {
            'name': 'Embryo: First Boot',
            'metrics': SystemMetrics(
                timestamp=time.time(),
                throughput_agents_per_sec=1000,
                latency_ms=150,
                cpu_utilization_pct=15,
                planning_horizon_steps=3,
                futures_explored_count=10,
                safety_prevented_pct=95,
                prediction_accuracy_pct=40,
                covenant_violations=0,
                entropy_cost_bits=2.5,
                reversibility_score=0.4,
                total_decisions=50,
                current_phase=LifecyclePhase.EMBRYO,
                hours_since_deployment=0.5
            )
        },
        {
            'name': 'Infant: Learning Phase',
            'metrics': SystemMetrics(
                timestamp=time.time(),
                throughput_agents_per_sec=50000,
                latency_ms=75,
                cpu_utilization_pct=45,
                planning_horizon_steps=8,
                futures_explored_count=500,
                safety_prevented_pct=85,
                prediction_accuracy_pct=65,
                covenant_violations=8,
                entropy_cost_bits=1.2,
                reversibility_score=0.6,
                total_decisions=500,
                current_phase=LifecyclePhase.INFANT,
                hours_since_deployment=12
            )
        },
        {
            'name': 'Adult: Stable Operation',
            'metrics': SystemMetrics(
                timestamp=time.time(),
                throughput_agents_per_sec=420000,
                latency_ms=29.4,
                cpu_utilization_pct=87,
                planning_horizon_steps=10,
                futures_explored_count=4800,
                safety_prevented_pct=92,
                prediction_accuracy_pct=78,
                covenant_violations=2,
                entropy_cost_bits=0.35,
                reversibility_score=0.98,
                total_decisions=15000,
                current_phase=LifecyclePhase.ADULT,
                hours_since_deployment=240
            )
        },
        {
            'name': 'Crisis: Thermal Arrest',
            'metrics': SystemMetrics(
                timestamp=time.time(),
                throughput_agents_per_sec=8000,
                latency_ms=200,
                cpu_utilization_pct=98,
                planning_horizon_steps=2,
                futures_explored_count=50,
                safety_prevented_pct=99,
                prediction_accuracy_pct=25,
                covenant_violations=0,
                entropy_cost_bits=3.8,
                reversibility_score=0.3,
                total_decisions=15500,
                current_phase=LifecyclePhase.CRISIS,
                hours_since_deployment=242
            )
        }
    ]

    for scenario in scenarios:
        print(f"\n{'=' * 70}")
        print(f"SCENARIO: {scenario['name']}")
        print(f"{'=' * 70}")

        metrics = scenario['metrics']
        # Recalculate phase based on metrics
        efficiency = engine.compute_operational_efficiency(metrics)
        metrics.current_phase = engine.determine_phase(
            metrics.total_decisions,
            efficiency['overall_efficiency']
        )

        reports = streamer.publish_update(metrics)

        # Also show mythic narrative
        print("\n" + "-" * 70)
        print("MYTHIC FRAMING:")
        print("-" * 70)
        print(reports['mythic']['narrative'])

        time.sleep(0.5)


if __name__ == "__main__":
    example_narrative_lifecycle()
