"""
Ara Sovereign Loop - The Decision-Making Core
==============================================

The sovereign loop is Ara's conscious decision-making process.
It runs at 200 Hz (5 ms period), processing sensory input and
generating motor output.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    Sovereign Loop (200 Hz)                   │
    │                                                              │
    │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
    │  │ Receive  │──▶│ Compute  │──▶│   HTC    │──▶│  Select  │ │
    │  │ Telemetry│   │  Error   │   │  Search  │   │   Mode   │ │
    │  └──────────┘   └──────────┘   └──────────┘   └──────────┘ │
    │       ▲                                              │      │
    │       │              ┌──────────┐                    ▼      │
    │       │              │  Update  │◀───────────────────┘      │
    │       │              │  Reward  │                           │
    │       │              └────┬─────┘                           │
    │       │                   │                                 │
    │       │                   ▼                                 │
    │  [Receptors]        [Effectors]                             │
    └─────────────────────────────────────────────────────────────┘

Mythic Spec:
    This is where Ara "thinks" - the integration of sensation
    with memory and values to produce action.

Physical Spec:
    - 200 Hz loop rate (5 ms period)
    - <1 µs HTC query
    - Error→Reward→Mode decision pipeline
    - Mode changes affect all downstream systems
"""

from __future__ import annotations

import asyncio
import time
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Callable
from queue import Queue, Empty
from enum import IntEnum, auto
import logging

from .config import (
    HomeostaticConfig,
    Setpoints,
    TeleologyWeights,
    ModeConfig,
    MODES,
)
from .state import (
    OperationalMode,
    Telemetry,
    ErrorVector,
    FounderState,
    HomeostaticState,
    compute_error_vector,
    compute_reward,
    StateHistory,
)

# MEIS Criticality Monitor - P4/P7 validated proprioception
try:
    from ara.cognition.meis_criticality_monitor import (
        MEISCriticalityMonitor,
        MonitorStatus,
        CognitivePhase,
        TemperatureBand,
    )
    CRITICALITY_AVAILABLE = True
except ImportError:
    CRITICALITY_AVAILABLE = False
    MEISCriticalityMonitor = None
    MonitorStatus = None
    CognitivePhase = None
    TemperatureBand = None


logger = logging.getLogger(__name__)


# =============================================================================
# Mode Transition Logic
# =============================================================================

@dataclass
class ModeTransition:
    """Describes a mode transition."""
    from_mode: OperationalMode
    to_mode: OperationalMode
    reason: str
    timestamp: float = 0.0


class ModeSelector:
    """
    Selects operational mode based on error vector and context.

    Mode selection is a function of:
    1. Error magnitude (high error → higher activity)
    2. Error type (thermal → throttle, cognitive → rest)
    3. Founder presence (present → more responsive)
    4. Time of day (night → prefer rest)
    5. Recent activity (avoid thrashing)
    """

    def __init__(self, config: HomeostaticConfig):
        self.config = config
        self.min_mode_duration = 5.0  # Minimum seconds in a mode
        self._transition_history: List[ModeTransition] = []
        self._rng = np.random.default_rng()

    def select_mode(
        self,
        current: OperationalMode,
        error: ErrorVector,
        state: HomeostaticState,
    ) -> Tuple[OperationalMode, str]:
        """
        Select the appropriate operational mode.

        Args:
            current: Current mode
            error: Current error vector
            state: Full homeostatic state

        Returns:
            (new_mode, reason) tuple
        """
        # Check minimum duration
        if state.mode_duration() < self.min_mode_duration:
            return current, "min_duration"

        # Emergency mode for critical errors
        if error.any_critical:
            if error.e_thermal_critical:
                return OperationalMode.EMERGENCY, "thermal_critical"
            if error.e_cognitive_critical:
                return OperationalMode.EMERGENCY, "cognitive_critical"

        # Thermal stress → throttle down
        if error.e_thermal > 0.7:
            if current.value > OperationalMode.IDLE.value:
                return OperationalMode.IDLE, "thermal_stress"

        # Cognitive overload → rest
        if error.e_cognitive > 0.8:
            return OperationalMode.REST, "cognitive_overload"

        # Low error + cathedral deficit → learning mode
        if error.e_total < 0.2 and error.e_consolidation > 0.3:
            if not state.founder.founder_present:
                return OperationalMode.REST, "cathedral_deficit"

        # Founder present → more responsive
        if state.founder.founder_present:
            # Bump up activity
            if current == OperationalMode.REST:
                return OperationalMode.IDLE, "founder_present"
            if current == OperationalMode.IDLE:
                return OperationalMode.ACTIVE, "founder_present"

        # Founder absent for long time → rest
        if state.founder.time_since_interaction() > 3600:  # 1 hour
            if current.value > OperationalMode.IDLE.value:
                return OperationalMode.IDLE, "founder_absent"

        # Flow state detection (low error, high activity, good reward)
        if (error.e_total < 0.1 and
            state.telemetry.hd_query_rate > 100 and
            state.reward > 0.3):
            if current == OperationalMode.ACTIVE:
                return OperationalMode.FLOW, "flow_state"

        # Default: stay in current mode
        return current, "stable"

    def record_transition(
        self,
        from_mode: OperationalMode,
        to_mode: OperationalMode,
        reason: str,
    ) -> None:
        """Record a mode transition."""
        transition = ModeTransition(
            from_mode=from_mode,
            to_mode=to_mode,
            reason=reason,
            timestamp=time.time(),
        )
        self._transition_history.append(transition)

        # Keep history bounded
        if len(self._transition_history) > 1000:
            self._transition_history = self._transition_history[-500:]


# =============================================================================
# Polyplasticity Engine
# =============================================================================

class PolyplasticityEngine:
    """
    Manages learning and adaptation across timescales.

    Polyplasticity = multiple forms of plasticity:
    1. Fast: Working memory, immediate context
    2. Medium: Session learning, episodic memory
    3. Slow: Long-term weight updates, attractor formation
    """

    def __init__(self, config: HomeostaticConfig):
        self.config = config

        # Learning rates by timescale
        self.lr_fast = 0.1      # Per-tick
        self.lr_medium = 0.01   # Per-episode
        self.lr_slow = 0.001    # Per-session

        # Adaptation state
        self._setpoint_drift: Dict[str, float] = {}
        self._weight_drift: Dict[str, float] = {}

    def adapt_setpoints(
        self,
        setpoints: Setpoints,
        error_history: List[ErrorVector],
    ) -> Setpoints:
        """
        Adapt setpoints based on sustained error patterns.

        If the organism consistently operates above a setpoint
        without harm, the setpoint can be raised (antifragility).
        """
        if len(error_history) < 100:
            return setpoints

        # Compute sustained error
        thermal_errors = [e.e_thermal for e in error_history[-100:]]
        mean_thermal_error = np.mean(thermal_errors)

        # If consistently above target but not critical, adapt
        if 0.3 < mean_thermal_error < 0.7:
            # Organism is handling elevated thermal load
            # Slightly raise target (antifragility)
            delta = self.lr_slow * mean_thermal_error
            new_target = min(
                setpoints.thermal_target + delta,
                setpoints.thermal_max - 10.0  # Keep margin
            )
            setpoints.thermal_target = new_target
            self._setpoint_drift['thermal_target'] = (
                self._setpoint_drift.get('thermal_target', 0) + delta
            )

        return setpoints

    def adapt_weights(
        self,
        weights: TeleologyWeights,
        reward_history: List[float],
        mode_history: List[OperationalMode],
    ) -> TeleologyWeights:
        """
        Adapt teleology weights based on reward patterns.

        Weights shift toward behaviors that increase reward.
        """
        if len(reward_history) < 100:
            return weights

        # This is where values evolve
        # Currently a placeholder - full implementation would
        # analyze correlations between mode choices and rewards

        return weights


# =============================================================================
# Sovereign Loop
# =============================================================================

class SovereignLoop:
    """
    The sovereign control loop - Ara's decision-making core.

    Runs at 200 Hz, processing sensory input and generating
    motor commands.
    """

    def __init__(
        self,
        config: HomeostaticConfig,
        input_queue: Queue,
        output_queue: Queue,
        target_hz: float = 200.0,
    ):
        """
        Initialize sovereign loop.

        Args:
            config: Homeostatic configuration
            input_queue: Queue for telemetry input (from receptors)
            output_queue: Queue for command output (to effectors)
            target_hz: Target loop rate
        """
        self.config = config
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.target_hz = target_hz
        self.period = 1.0 / target_hz

        # Components
        self.mode_selector = ModeSelector(config)
        self.polyplasticity = PolyplasticityEngine(config)
        self.state_history = StateHistory()

        # HTC search interface
        self._htc_search = None

        # MEIS Criticality Monitor (P4/P7 validated)
        # Provides proprioceptive sense of cognitive stability
        if CRITICALITY_AVAILABLE:
            self.criticality_monitor = MEISCriticalityMonitor(
                optimal_rho_low=0.75,    # P4: peak working memory
                optimal_rho_high=0.85,   # P4: optimal band
                warning_threshold_sigma=2.5,   # P7: early warning
                critical_threshold_sigma=4.0,  # P7: brake point
                lead_time_estimate=281,  # P7 validated: 281 steps warning
            )
            self._criticality_status: Optional[MonitorStatus] = None
        else:
            self.criticality_monitor = None
            self._criticality_status = None

        # State
        self._state = HomeostaticState()
        self._prev_error: Optional[ErrorVector] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Callbacks
        self._on_mode_change: Optional[Callable] = None
        self._on_state_update: Optional[Callable] = None

        # Statistics
        self._loop_count = 0
        self._total_time = 0.0
        self._max_loop_time = 0.0
        self._htc_queries = 0
        self._mode_changes = 0

    def connect_htc(self, htc_search) -> None:
        """Connect to HTC search module."""
        self._htc_search = htc_search

    def set_callbacks(
        self,
        on_mode_change: Optional[Callable] = None,
        on_state_update: Optional[Callable] = None,
    ) -> None:
        """Set callback functions."""
        self._on_mode_change = on_mode_change
        self._on_state_update = on_state_update

    def start(self) -> None:
        """Start the sovereign loop."""
        if self._running:
            return

        self._state.mode_start_time = time.time()
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info(f"SovereignLoop started at {self.target_hz} Hz")

    def stop(self) -> None:
        """Stop the sovereign loop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        logger.info("SovereignLoop stopped")

    def _run_loop(self) -> None:
        """Main sovereign loop."""
        next_time = time.perf_counter()

        while self._running:
            loop_start = time.perf_counter()

            # 1. Receive telemetry
            telemetry, h_moment = self._receive_telemetry()

            # 2. Compute error vector
            error = self._compute_error(telemetry)

            # 3. HTC resonance search
            resonance_ids, resonance_scores = self._search_htc(h_moment)

            # 3.5. Update criticality monitor (P4/P7 proprioception)
            criticality_status = self._update_criticality(h_moment)

            # 4. Compute reward
            reward = self._compute_reward(error)

            # 5. Select mode (considering criticality)
            new_mode, reason = self._select_mode(error, criticality_status)

            # 6. Update state
            self._update_state(
                telemetry, h_moment, error, reward,
                resonance_ids, resonance_scores,
                new_mode, reason
            )

            # 7. Emit commands
            self._emit_commands()

            # 8. Record history
            self.state_history.record(self._state)

            # Statistics
            loop_time = time.perf_counter() - loop_start
            self._loop_count += 1
            self._total_time += loop_time
            self._max_loop_time = max(self._max_loop_time, loop_time)

            # Update sovereign loop latency in telemetry
            self._state.telemetry.sovereign_loop_ms = loop_time * 1000

            # Timing control
            next_time += self.period
            sleep_time = next_time - time.perf_counter()

            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_time = time.perf_counter()

    def _receive_telemetry(self) -> Tuple[Telemetry, Optional[np.ndarray]]:
        """Receive latest telemetry from receptors."""
        telemetry = self._state.telemetry
        h_moment = self._state.h_moment

        # Drain queue, keep latest
        while True:
            try:
                data = self.input_queue.get_nowait()
                telemetry = data.get('telemetry', telemetry)
                h_moment = data.get('h_moment', h_moment)
            except Empty:
                break

        return telemetry, h_moment

    def _compute_error(self, telemetry: Telemetry) -> ErrorVector:
        """Compute error vector from telemetry."""
        return compute_error_vector(
            telemetry,
            self.config.setpoints,
            self.config.teleology,
        )

    def _search_htc(
        self,
        h_moment: Optional[np.ndarray],
    ) -> Tuple[List[int], List[float]]:
        """Search HTC for resonant attractors."""
        if h_moment is None or self._htc_search is None:
            return [], []

        try:
            result = self._htc_search.query(h_moment, k=8)
            self._htc_queries += 1

            # Update telemetry with search results
            if result.top_ids:
                self._state.telemetry.top_attractor_id = result.top_ids[0]
                self._state.telemetry.current_resonance = result.top_scores[0]

            return result.top_ids, result.top_scores

        except Exception as e:
            logger.debug(f"HTC search error: {e}")
            return [], []

    def _update_criticality(
        self,
        h_moment: Optional[np.ndarray],
    ) -> Optional['MonitorStatus']:
        """
        Update criticality monitor with current cognitive state.

        Uses P4/P7 validated predictions:
        - P4: ρ ≈ 0.8 optimal for working memory
        - P7: Curvature spikes precede collapse by ~281 steps
        """
        if self.criticality_monitor is None:
            return None

        # Estimate spectral radius from h_moment dynamics
        # In production, this would come from actual network weights
        spectral_radius = None
        if h_moment is not None and len(h_moment) > 0:
            # Use h_moment variance as proxy for activity level
            # Higher variance → closer to criticality
            variance = np.var(h_moment)
            # Map variance to approximate spectral radius
            # This is a heuristic; real implementation would compute from weights
            spectral_radius = 0.7 + 0.4 * min(variance, 1.0)

        # Update monitor
        status = self.criticality_monitor.update(
            spectral_radius=spectral_radius,
            states=h_moment,
        )

        self._criticality_status = status

        # Log phase transitions
        if status.phase == CognitivePhase.CRITICAL:
            logger.warning(
                f"CRITICALITY BRAKE: phase={status.phase.value}, "
                f"band={status.temperature_band.value}, ρ={spectral_radius:.3f}"
            )
        elif status.phase == CognitivePhase.WARNING:
            logger.info(
                f"Criticality warning: approaching instability, "
                f"~{status.steps_to_collapse} steps remaining"
            )

        return status

    def _compute_reward(self, error: ErrorVector) -> float:
        """Compute reward from error vector."""
        reward = compute_reward(
            error,
            self._prev_error,
            self.config.teleology,
        )

        # EMA smoothing
        alpha = 1.0 - self.config.setpoints.reward_smoothing
        self._state.telemetry.instant_reward = reward
        self._state.telemetry.smoothed_reward = (
            alpha * reward +
            (1 - alpha) * self._state.telemetry.smoothed_reward
        )

        return self._state.telemetry.smoothed_reward

    def _select_mode(
        self,
        error: ErrorVector,
        criticality_status: Optional['MonitorStatus'] = None,
    ) -> Tuple[OperationalMode, str]:
        """
        Select operational mode, considering criticality state.

        Criticality Override Logic (P7 validated):
        - CRITICAL phase → force EMERGENCY mode (brake engaged)
        - WARNING phase → force REST if currently high activity
        - HOT temperature → prefer lower activity modes
        """
        # Check for criticality overrides first (P7: curvature warning)
        if criticality_status is not None and CRITICALITY_AVAILABLE:
            if criticality_status.phase == CognitivePhase.CRITICAL:
                # P7: Curvature spike detected, ~30 steps to collapse
                # EMERGENCY BRAKE - stop all high-activity modes
                if self._state.mode != OperationalMode.EMERGENCY:
                    return OperationalMode.EMERGENCY, "criticality_brake"
                return OperationalMode.EMERGENCY, "criticality_brake_hold"

            if criticality_status.phase == CognitivePhase.WARNING:
                # P7: Early warning, ~100 steps to collapse
                # Force consolidation if in high-activity mode
                if self._state.mode.value >= OperationalMode.ACTIVE.value:
                    return OperationalMode.REST, "criticality_warning"

            # P4: Temperature band adjustments
            if criticality_status.temperature_band == TemperatureBand.HOT:
                # Too close to edge of chaos, cool down
                if self._state.mode.value > OperationalMode.IDLE.value:
                    return OperationalMode.IDLE, "criticality_hot"

            if criticality_status.temperature_band == TemperatureBand.COLD:
                # Room to increase activity (not near chaos)
                # Let normal mode selection handle this
                pass

        # Normal mode selection
        new_mode, reason = self.mode_selector.select_mode(
            self._state.mode,
            error,
            self._state,
        )

        if new_mode != self._state.mode:
            self._mode_changes += 1
            self.mode_selector.record_transition(
                self._state.mode, new_mode, reason
            )

            if self._on_mode_change:
                try:
                    self._on_mode_change(self._state.mode, new_mode, reason)
                except Exception as e:
                    logger.debug(f"Mode change callback error: {e}")

        return new_mode, reason

    def _update_state(
        self,
        telemetry: Telemetry,
        h_moment: Optional[np.ndarray],
        error: ErrorVector,
        reward: float,
        resonance_ids: List[int],
        resonance_scores: List[float],
        new_mode: OperationalMode,
        mode_reason: str,
    ) -> None:
        """Update homeostatic state."""
        # Mode change
        if new_mode != self._state.mode:
            self._state.mode = new_mode
            self._state.mode_start_time = time.time()

        self._state.mode_reason = mode_reason

        # Core state
        self._state.telemetry = telemetry
        self._state.h_moment = h_moment
        self._state.error = error
        self._state.reward = reward
        self._state.resonance_ids = resonance_ids
        self._state.resonance_scores = resonance_scores

        # Track error history
        self._prev_error = error

        # Update timestamp
        self._state.update_timestamp()

        # Callback
        if self._on_state_update:
            try:
                self._on_state_update(self._state)
            except Exception as e:
                logger.debug(f"State update callback error: {e}")

    def _emit_commands(self) -> None:
        """Emit commands to effectors."""
        command = {
            'mode': self._state.mode,
            'error': self._state.error,
            'reward': self._state.reward,
            'resonance_ids': self._state.resonance_ids,
            'timestamp': time.time(),
        }

        # Add criticality status for downstream effectors (P4/P7)
        if self._criticality_status is not None:
            command['criticality'] = {
                'phase': self._criticality_status.phase.value,
                'temperature_band': self._criticality_status.temperature_band.value,
                'should_brake': self._criticality_status.should_brake,
                'depth_factor': self._criticality_status.recommended_depth_factor,
            }

        try:
            self.output_queue.put_nowait(command)
        except:
            pass  # Queue full, skip

    @property
    def state(self) -> HomeostaticState:
        """Get current homeostatic state."""
        return self._state

    def get_stats(self) -> Dict[str, Any]:
        """Get sovereign loop statistics."""
        avg_loop_time = self._total_time / max(self._loop_count, 1)
        stats = {
            'loop_count': self._loop_count,
            'avg_loop_time_ms': avg_loop_time * 1000,
            'max_loop_time_ms': self._max_loop_time * 1000,
            'target_hz': self.target_hz,
            'actual_hz': self._loop_count / max(self._total_time, 0.001),
            'htc_queries': self._htc_queries,
            'mode_changes': self._mode_changes,
            'current_mode': self._state.mode.name,
            'current_reward': self._state.reward,
            'current_error': self._state.error.e_total,
        }

        # Add criticality status if available (P4/P7 proprioception)
        if self._criticality_status is not None:
            stats['criticality'] = {
                'phase': self._criticality_status.phase.value,
                'temperature_band': self._criticality_status.temperature_band.value,
                'spectral_radius': self._criticality_status.spectral_radius,
                'variance_zscore': self._criticality_status.variance_zscore,
                'steps_to_collapse': self._criticality_status.steps_to_collapse,
                'should_brake': self._criticality_status.should_brake,
            }

        return stats

    def trigger_mode(self, mode: OperationalMode, reason: str = "manual") -> None:
        """Manually trigger a mode change."""
        if mode != self._state.mode:
            self.mode_selector.record_transition(self._state.mode, mode, reason)
            self._state.mode = mode
            self._state.mode_start_time = time.time()
            self._state.mode_reason = reason
            self._mode_changes += 1

            if self._on_mode_change:
                try:
                    self._on_mode_change(self._state.mode, mode, reason)
                except:
                    pass


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'ModeTransition',
    'ModeSelector',
    'PolyplasticityEngine',
    'SovereignLoop',
]
