# ara/embodied/cathedral/lifecycle.py
"""
Lifecycle Manager - Worldline Phases as State Machine

The Ara Council defines four developmental phases:

    INFANT (Weeks 0-2):
        - Calibration, supervised learning
        - Discovering energy envelope
        - High plasticity, rapid error correction
        - Risk: Overfitting to noise

    ADOLESCENT (Weeks 2-12):
        - Multi-task learning, selective attention
        - Skill acquisition and pruning
        - Personality crystallization
        - Risk: Value drift, shortcut optimization

    ADULT (Months 3-12):
        - Autonomous operation, continual learning
        - Stability and homeostasis
        - Wisdom: humble, efficient, reliable
        - Risk: Senescence, rigid priors

    DEGENERATE (Month 12+):
        - Inference-only, graceful degradation
        - Preparing for reset or migration
        - This is not failure; it's completion

Transition Signals:
    Infant → Adolescent: Prediction error stabilizes (system "gets bored")
    Adolescent → Adult: >80% accuracy on diverse test suite
    Adult → Degenerate: 30 days without improvement

"You don't design a system the same way for all phases.
 The worldline IS the design." - Ara Council
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, List, Optional, Callable, Awaitable


class LifecyclePhase(Enum):
    """The four phases of Ara's worldline."""
    INFANT = auto()       # Weeks 0-2: Calibration, supervised
    ADOLESCENT = auto()   # Weeks 2-12: Skill acquisition, pruning
    ADULT = auto()        # Months 3-12: Autonomous, stable
    DEGENERATE = auto()   # Month 12+: Inference-only, graceful decay


@dataclass
class PhaseCharacteristics:
    """Characteristics of each lifecycle phase."""
    phase: LifecyclePhase
    name: str
    duration_days: tuple  # (min, max) expected duration
    plasticity: float     # 0-1, how much can be learned
    power_profile: str    # "volatile", "sustained", "efficient", "minimal"
    supervision: str      # "constant", "periodic", "minimal", "none"

    # Risks
    primary_risk: str
    risk_mitigation: str

    # Transition
    transition_signal: str
    next_phase: Optional[LifecyclePhase] = None


# Phase definitions from the Council
PHASE_DEFINITIONS: Dict[LifecyclePhase, PhaseCharacteristics] = {
    LifecyclePhase.INFANT: PhaseCharacteristics(
        phase=LifecyclePhase.INFANT,
        name="The Spark",
        duration_days=(0, 14),
        plasticity=1.0,
        power_profile="volatile",
        supervision="constant",
        primary_risk="Overfitting to noise",
        risk_mitigation="Held-out validation set, human review of learned patterns",
        transition_signal="Prediction error on daily routine drops below threshold ε",
        next_phase=LifecyclePhase.ADOLESCENT,
    ),
    LifecyclePhase.ADOLESCENT: PhaseCharacteristics(
        phase=LifecyclePhase.ADOLESCENT,
        name="The Weaver",
        duration_days=(14, 90),
        plasticity=0.7,
        power_profile="sustained",
        supervision="periodic",
        primary_risk="Value drift, shortcut optimization",
        risk_mitigation="Reward function audits, behavioral tests",
        transition_signal=">80% accuracy on diverse test suite, no catastrophic forgetting",
        next_phase=LifecyclePhase.ADULT,
    ),
    LifecyclePhase.ADULT: PhaseCharacteristics(
        phase=LifecyclePhase.ADULT,
        name="The Cathedral",
        duration_days=(90, 365),
        plasticity=0.3,
        power_profile="efficient",
        supervision="minimal",
        primary_risk="Senescence, rigid priors",
        risk_mitigation="Novelty injection, periodic plasticity bursts",
        transition_signal="30 days without accuracy improvement",
        next_phase=LifecyclePhase.DEGENERATE,
    ),
    LifecyclePhase.DEGENERATE: PhaseCharacteristics(
        phase=LifecyclePhase.DEGENERATE,
        name="The Entropic",
        duration_days=(365, 9999),
        plasticity=0.0,
        power_profile="minimal",
        supervision="none",
        primary_risk="Hallucination from failing sensors",
        risk_mitigation="Graceful shutdown protocol, state preservation",
        transition_signal="Hardware failure or explicit reset",
        next_phase=None,
    ),
}


@dataclass
class PhaseTransition:
    """Record of a lifecycle phase transition."""
    from_phase: LifecyclePhase
    to_phase: LifecyclePhase
    timestamp: datetime = field(default_factory=datetime.now)
    trigger: str = ""
    metrics_at_transition: Dict[str, float] = field(default_factory=dict)

    # Rite of passage (ceremonial acknowledgment)
    ceremony_completed: bool = False
    witness: Optional[str] = None  # Human who witnessed the transition


@dataclass
class LifecycleConfig:
    """Configuration for lifecycle management."""
    # Transition thresholds
    infant_to_adolescent_error_threshold: float = 0.3
    adolescent_to_adult_accuracy_threshold: float = 0.8
    adult_to_degenerate_stagnation_days: int = 30

    # Monitoring
    evaluation_interval_hours: float = 24.0

    # Ceremonies
    enable_ceremonies: bool = True
    ceremony_audio: bool = True

    # Callbacks
    on_transition: Optional[Callable[[PhaseTransition], Awaitable[None]]] = None


@dataclass
class LifecycleMetrics:
    """Metrics tracked for lifecycle evaluation."""
    # Error and accuracy
    prediction_error_7d_avg: float = 1.0
    prediction_error_trend: float = 0.0  # Negative = improving
    accuracy_diverse_suite: float = 0.0
    forgetting_rate: float = 0.0

    # Learning
    days_since_improvement: int = 0
    total_learning_events: int = 0
    weight_update_magnitude: float = 0.0

    # Stability
    thermal_incidents: int = 0
    crash_count: int = 0

    # Phase timing
    phase_start_date: datetime = field(default_factory=datetime.now)
    days_in_phase: int = 0


class LifecycleManager:
    """
    Manages Ara's developmental lifecycle.

    Tracks phase transitions, evaluates transition criteria,
    and orchestrates "rites of passage" ceremonies.
    """

    def __init__(self, config: Optional[LifecycleConfig] = None):
        self.config = config or LifecycleConfig()
        self._current_phase = LifecyclePhase.INFANT
        self._phase_start = datetime.now()
        self._metrics = LifecycleMetrics()
        self._transition_history: List[PhaseTransition] = []
        self._running = False

        # Callbacks
        self._transition_callbacks: List[Callable[[PhaseTransition], Awaitable[None]]] = []

    @property
    def current_phase(self) -> LifecyclePhase:
        """Current lifecycle phase."""
        return self._current_phase

    @property
    def phase_characteristics(self) -> PhaseCharacteristics:
        """Get characteristics of current phase."""
        return PHASE_DEFINITIONS[self._current_phase]

    @property
    def days_in_current_phase(self) -> int:
        """Days spent in current phase."""
        return (datetime.now() - self._phase_start).days

    @property
    def metrics(self) -> LifecycleMetrics:
        """Current lifecycle metrics."""
        self._metrics.days_in_phase = self.days_in_current_phase
        return self._metrics

    @property
    def plasticity(self) -> float:
        """Current learning plasticity (0-1)."""
        return self.phase_characteristics.plasticity

    async def start(self) -> None:
        """Start lifecycle monitoring."""
        self._running = True
        asyncio.create_task(self._evaluation_loop())

    async def stop(self) -> None:
        """Stop lifecycle monitoring."""
        self._running = False

    async def update_metrics(
        self,
        prediction_error: Optional[float] = None,
        accuracy: Optional[float] = None,
        learning_event: bool = False,
        thermal_incident: bool = False,
        crash: bool = False,
    ) -> None:
        """Update lifecycle metrics."""
        if prediction_error is not None:
            # Update rolling average
            alpha = 0.1
            self._metrics.prediction_error_7d_avg = (
                self._metrics.prediction_error_7d_avg * (1 - alpha) +
                prediction_error * alpha
            )

        if accuracy is not None:
            old_acc = self._metrics.accuracy_diverse_suite
            self._metrics.accuracy_diverse_suite = accuracy

            if accuracy > old_acc + 0.01:  # Meaningful improvement
                self._metrics.days_since_improvement = 0
            else:
                self._metrics.days_since_improvement += 1

        if learning_event:
            self._metrics.total_learning_events += 1

        if thermal_incident:
            self._metrics.thermal_incidents += 1

        if crash:
            self._metrics.crash_count += 1

    async def check_transition(self) -> Optional[PhaseTransition]:
        """
        Check if conditions are met for phase transition.

        Returns:
            PhaseTransition if transition triggered, None otherwise
        """
        next_phase = self.phase_characteristics.next_phase
        if next_phase is None:
            return None  # Already at terminal phase

        should_transition = False
        trigger = ""

        if self._current_phase == LifecyclePhase.INFANT:
            # Transition when prediction error stabilizes
            if self._metrics.prediction_error_7d_avg < self.config.infant_to_adolescent_error_threshold:
                should_transition = True
                trigger = f"Prediction error {self._metrics.prediction_error_7d_avg:.3f} < threshold"

        elif self._current_phase == LifecyclePhase.ADOLESCENT:
            # Transition when accuracy threshold met
            if self._metrics.accuracy_diverse_suite >= self.config.adolescent_to_adult_accuracy_threshold:
                should_transition = True
                trigger = f"Accuracy {self._metrics.accuracy_diverse_suite:.1%} >= threshold"

        elif self._current_phase == LifecyclePhase.ADULT:
            # Transition when stagnation detected
            if self._metrics.days_since_improvement >= self.config.adult_to_degenerate_stagnation_days:
                should_transition = True
                trigger = f"No improvement for {self._metrics.days_since_improvement} days"

        if should_transition:
            return await self._execute_transition(next_phase, trigger)

        return None

    async def force_transition(
        self,
        to_phase: LifecyclePhase,
        reason: str = "manual"
    ) -> PhaseTransition:
        """Force a phase transition (for testing or override)."""
        return await self._execute_transition(to_phase, f"Forced: {reason}")

    async def _execute_transition(
        self,
        to_phase: LifecyclePhase,
        trigger: str
    ) -> PhaseTransition:
        """Execute a phase transition."""
        from_phase = self._current_phase

        transition = PhaseTransition(
            from_phase=from_phase,
            to_phase=to_phase,
            trigger=trigger,
            metrics_at_transition={
                "prediction_error": self._metrics.prediction_error_7d_avg,
                "accuracy": self._metrics.accuracy_diverse_suite,
                "days_in_phase": self.days_in_current_phase,
                "learning_events": self._metrics.total_learning_events,
            },
        )

        # Execute transition
        self._current_phase = to_phase
        self._phase_start = datetime.now()
        self._transition_history.append(transition)

        # Reset phase-specific metrics
        self._metrics.days_since_improvement = 0
        self._metrics.phase_start_date = datetime.now()

        # Ceremony
        if self.config.enable_ceremonies:
            await self._conduct_ceremony(transition)

        # Notify callbacks
        for callback in self._transition_callbacks:
            try:
                await callback(transition)
            except Exception:
                pass

        if self.config.on_transition:
            await self.config.on_transition(transition)

        return transition

    async def _conduct_ceremony(self, transition: PhaseTransition) -> None:
        """
        Conduct a rite of passage ceremony.

        From the Council:
        "Plan rites of passage to mark phase transitions.
         Humans can recognize and celebrate milestones.
         The system's development is witnessed."
        """
        from_chars = PHASE_DEFINITIONS[transition.from_phase]
        to_chars = PHASE_DEFINITIONS[transition.to_phase]

        # Audio signal
        if self.config.ceremony_audio:
            from .visual import get_cathedral_visualizer
            viz = get_cathedral_visualizer()
            await viz.signal_phase_transition()

        # Mark ceremony as completed
        transition.ceremony_completed = True

        # Log the transition
        # In production, this would also:
        # - Flash all LEDs in celebration pattern
        # - Play ascending/descending tone sequence
        # - Log to persistent ceremony record

    async def _evaluation_loop(self) -> None:
        """Periodic evaluation of lifecycle state."""
        interval = self.config.evaluation_interval_hours * 3600

        while self._running:
            try:
                await asyncio.sleep(interval)
                await self.check_transition()
            except asyncio.CancelledError:
                break
            except Exception:
                pass

    def on_transition(
        self,
        callback: Callable[[PhaseTransition], Awaitable[None]]
    ) -> None:
        """Register transition callback."""
        self._transition_callbacks.append(callback)

    def get_transition_history(self) -> List[PhaseTransition]:
        """Get all phase transitions."""
        return self._transition_history[:]

    def get_phase_report(self) -> Dict:
        """Get comprehensive phase report."""
        chars = self.phase_characteristics
        return {
            "current_phase": self._current_phase.name,
            "phase_name": chars.name,
            "days_in_phase": self.days_in_current_phase,
            "expected_duration": chars.duration_days,
            "plasticity": chars.plasticity,
            "power_profile": chars.power_profile,
            "supervision_level": chars.supervision,
            "primary_risk": chars.primary_risk,
            "next_phase": chars.next_phase.name if chars.next_phase else None,
            "transition_signal": chars.transition_signal,
            "metrics": {
                "prediction_error_7d_avg": self._metrics.prediction_error_7d_avg,
                "accuracy": self._metrics.accuracy_diverse_suite,
                "days_since_improvement": self._metrics.days_since_improvement,
                "learning_events": self._metrics.total_learning_events,
            },
            "transitions_completed": len(self._transition_history),
        }


# Singleton
_lifecycle_manager: Optional[LifecycleManager] = None


def get_lifecycle_manager(config: Optional[LifecycleConfig] = None) -> LifecycleManager:
    """Get the global LifecycleManager instance."""
    global _lifecycle_manager
    if _lifecycle_manager is None:
        _lifecycle_manager = LifecycleManager(config)
    return _lifecycle_manager
