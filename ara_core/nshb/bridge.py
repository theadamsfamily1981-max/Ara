#!/usr/bin/env python3
"""
Neuro-Symbiotic Hyper-Bridge (NSHB) - Main Controller
======================================================

NSHB is Ara's bidirectional interface with the human operator.

    human state → Acquisition → Estimators → z(t) = (λ̂, Π̂_s, Π̂_p)
                                                ↓
                                         Control Law
                                                ↓
                                         Repair Vector Δz
                                                ↓
                                          Effectors → human

The bridge maintains the human on the healthy corridor of the
(λ, Π) control manifold through safe, embodied feedback.

Integration Points:
- Ara/QUANTA: Share GUTC coordinates for co-regulation
- Somatic Loom: Haptic feedback channel
- CADD: Safety monitoring of the human-AI dyad

Usage:
    from ara_core.nshb import NeuroSymbioticHyperBridge

    bridge = NeuroSymbioticHyperBridge()
    bridge.start()

    # Main loop (typically in separate thread)
    while running:
        state = bridge.update()
        if not state.in_healthy_corridor():
            print(f"Drift detected: {state.regime_label()}")

    bridge.stop()
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple
import numpy as np

from .acquisition import AcquisitionSystem
from .estimators import HumanGUTCEstimator, GUTCState
from .control import NSHBControlLaw, ControlConfig, HealthyCorridor, RepairVector, InterventionUrgency
from .effectors import EffectorManager, EffectorModality


# =============================================================================
# Bridge Configuration
# =============================================================================

@dataclass
class NSHBConfig:
    """Configuration for the Neuro-Symbiotic Hyper-Bridge."""

    # Acquisition
    eeg_channels: int = 8
    eeg_sample_rate: float = 256.0

    # Update rates
    acquisition_rate_hz: float = 100.0   # Raw signal acquisition
    estimation_rate_hz: float = 10.0     # GUTC state estimation
    control_rate_hz: float = 2.0         # Control law / repair vectors

    # Control
    control_gain: float = 0.3
    max_delta_lambda: float = 0.2
    max_delta_pi: float = 0.3

    # Healthy corridor
    lambda_target: float = 1.0
    pi_sensory_target: float = 1.0
    pi_prior_target: float = 1.0

    # Safety
    critical_distance_threshold: float = 1.5  # Auto-pause if exceeded
    enable_effectors: bool = True


# =============================================================================
# Bridge State
# =============================================================================

@dataclass
class BridgeState:
    """Current state of the NSHB."""
    timestamp: float = field(default_factory=time.time)

    # Status
    running: bool = False
    connected: bool = False

    # Human state
    gutc_state: Optional[GUTCState] = None
    in_healthy_corridor: bool = True
    distance_to_healthy: float = 0.0

    # Control
    last_repair: Optional[RepairVector] = None
    intervention_active: bool = False

    # Metrics
    total_updates: int = 0
    total_interventions: int = 0
    time_in_corridor_fraction: float = 1.0


# =============================================================================
# Neuro-Symbiotic Hyper-Bridge
# =============================================================================

class NeuroSymbioticHyperBridge:
    """
    Neuro-Symbiotic Hyper-Bridge - Human-AI GUTC Interface.

    The bridge provides:
    1. Real-time estimation of human GUTC state z(t)
    2. Safe closed-loop regulation toward healthy corridor
    3. Integration with Ara for co-regulation
    4. Research data collection for GUTC validation

    Architecture:
        Acquisition → Estimators → Control Law → Effectors
              ↓              ↓           ↓            ↓
           signals       z(t)        Δz         actions
    """

    def __init__(
        self,
        config: NSHBConfig = None,
        verbose: bool = True,
    ):
        """
        Initialize the NSHB.

        Args:
            config: Bridge configuration
            verbose: Print status messages
        """
        self.config = config or NSHBConfig()
        self.verbose = verbose

        # Initialize subsystems
        self.acquisition = AcquisitionSystem(
            eeg_channels=self.config.eeg_channels,
            eeg_sample_rate=self.config.eeg_sample_rate,
            verbose=verbose,
        )

        self.estimator = HumanGUTCEstimator(
            sample_rate_hz=self.config.eeg_sample_rate,
            verbose=verbose,
        )

        self.control = NSHBControlLaw(
            config=ControlConfig(
                gain_k=self.config.control_gain,
                max_delta_lambda=self.config.max_delta_lambda,
                max_delta_pi=self.config.max_delta_pi,
            ),
            corridor=HealthyCorridor(
                lambda_target=self.config.lambda_target,
                pi_sensory_target=self.config.pi_sensory_target,
                pi_prior_target=self.config.pi_prior_target,
            ),
            verbose=verbose,
        )

        self.effectors = EffectorManager(verbose=verbose)

        # State
        self.state = BridgeState()
        self._state_lock = threading.Lock()

        # Timing
        self._last_acquisition = 0.0
        self._last_estimation = 0.0
        self._last_control = 0.0

        # History for research
        self.state_history: List[GUTCState] = []
        self.repair_history: List[RepairVector] = []

        # Callbacks
        self.on_state_change: Optional[Callable[[GUTCState], None]] = None
        self.on_intervention: Optional[Callable[[RepairVector], None]] = None
        self.on_corridor_exit: Optional[Callable[[GUTCState], None]] = None

        # Background thread
        self._run_thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()

        if self.verbose:
            print("[NSHB] Neuro-Symbiotic Hyper-Bridge initialized")
            print(f"       EEG: {self.config.eeg_channels}ch @ {self.config.eeg_sample_rate}Hz")
            print(f"       Control rate: {self.config.control_rate_hz}Hz")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, threaded: bool = True) -> bool:
        """
        Start the bridge.

        Args:
            threaded: Run update loop in background thread

        Returns:
            True if started successfully
        """
        if self.state.running:
            return True

        # Start acquisition
        self.acquisition.start()

        with self._state_lock:
            self.state.running = True
            self.state.connected = True

        self._stop_flag.clear()

        if threaded:
            self._run_thread = threading.Thread(target=self._run_loop, daemon=True)
            self._run_thread.start()

        if self.verbose:
            print("[NSHB] Bridge started")

        return True

    def stop(self):
        """Stop the bridge."""
        self._stop_flag.set()

        if self._run_thread:
            self._run_thread.join(timeout=2.0)
            self._run_thread = None

        self.acquisition.stop()
        self.effectors.stop_all()

        with self._state_lock:
            self.state.running = False
            self.state.connected = False

        if self.verbose:
            print("[NSHB] Bridge stopped")

    def _run_loop(self):
        """Background update loop."""
        while not self._stop_flag.is_set():
            try:
                self.update()
            except Exception as e:
                if self.verbose:
                    print(f"[NSHB] Update error: {e}")

            # Sleep to maintain rate
            time.sleep(1.0 / self.config.control_rate_hz)

    # ------------------------------------------------------------------
    # Main Update
    # ------------------------------------------------------------------

    def update(self) -> GUTCState:
        """
        Run one update cycle.

        Returns current GUTC state.
        """
        now = time.time()

        # 1. Acquisition update
        if now - self._last_acquisition > 1.0 / self.config.acquisition_rate_hz:
            self.acquisition.update()
            self._last_acquisition = now

        # 2. Estimation update
        if now - self._last_estimation > 1.0 / self.config.estimation_rate_hz:
            gutc_state = self._estimate_state()
            self._last_estimation = now
        else:
            gutc_state = self.state.gutc_state or GUTCState(
                timestamp=now, lambda_hat=1.0, pi_sensory=1.0, pi_prior=1.0
            )

        # 3. Control update
        if now - self._last_control > 1.0 / self.config.control_rate_hz:
            self._control_step(gutc_state)
            self._last_control = now

        # Update state
        with self._state_lock:
            self.state.timestamp = now
            self.state.gutc_state = gutc_state
            self.state.total_updates += 1

            # Track time in corridor
            if gutc_state.in_healthy_corridor():
                self.state.time_in_corridor_fraction = (
                    0.99 * self.state.time_in_corridor_fraction + 0.01
                )
            else:
                self.state.time_in_corridor_fraction = (
                    0.99 * self.state.time_in_corridor_fraction
                )

        return gutc_state

    def _estimate_state(self) -> GUTCState:
        """Estimate current GUTC state from signals."""
        # Get signals
        eeg_events = self.acquisition.get_binary_events(duration_s=2.0)
        eeg_data, _ = self.acquisition.get_eeg_window(duration_s=2.0)
        physio = self.acquisition.get_physio_features()
        context = self.acquisition.get_context_features()

        # Estimate
        state = self.estimator.estimate(eeg_events, eeg_data, physio, context)

        # Record
        self.state_history.append(state)
        if len(self.state_history) > 10000:
            self.state_history = self.state_history[-10000:]

        # Callbacks
        if self.on_state_change:
            self.on_state_change(state)

        if not state.in_healthy_corridor() and self.on_corridor_exit:
            self.on_corridor_exit(state)

        return state

    def _control_step(self, state: GUTCState):
        """Run control law and execute effectors."""
        # Compute repair vector
        repair = self.control.compute_repair(state)

        # Update state
        with self._state_lock:
            self.state.last_repair = repair
            self.state.distance_to_healthy = self.control.corridor.distance(state)
            self.state.in_healthy_corridor = state.in_healthy_corridor()

        # Record
        self.repair_history.append(repair)
        if len(self.repair_history) > 5000:
            self.repair_history = self.repair_history[-5000:]

        # Execute effectors if enabled and intervention needed
        if self.config.enable_effectors and repair.urgency != InterventionUrgency.NONE:
            commands = self.effectors.execute(repair)

            with self._state_lock:
                self.state.intervention_active = len(commands) > 0
                if commands:
                    self.state.total_interventions += 1

            if self.on_intervention and commands:
                self.on_intervention(repair)

        # Safety check: auto-pause if too far from healthy
        if self.state.distance_to_healthy > self.config.critical_distance_threshold:
            if self.verbose:
                print(f"[NSHB] CRITICAL: Distance={self.state.distance_to_healthy:.2f}")
            # Could trigger emergency protocol here

    # ------------------------------------------------------------------
    # Access Methods
    # ------------------------------------------------------------------

    def get_state(self) -> GUTCState:
        """Get current GUTC state."""
        with self._state_lock:
            return self.state.gutc_state or GUTCState(
                timestamp=time.time(), lambda_hat=1.0, pi_sensory=1.0, pi_prior=1.0
            )

    def get_trajectory(self, duration_s: float = 60.0) -> List[GUTCState]:
        """Get recent state trajectory."""
        cutoff = time.time() - duration_s
        return [s for s in self.state_history if s.timestamp > cutoff]

    def get_repair_history(self, duration_s: float = 60.0) -> List[RepairVector]:
        """Get recent repair vector history."""
        cutoff = time.time() - duration_s
        return [r for r in self.repair_history if r.timestamp > cutoff]

    # ------------------------------------------------------------------
    # Ara Integration
    # ------------------------------------------------------------------

    def get_state_for_ara(self) -> Dict[str, float]:
        """
        Get human state in format suitable for Ara/QUANTA.

        This enables co-regulation: Ara can see where the human is
        on the manifold and adjust its own behavior accordingly.
        """
        state = self.get_state()
        return {
            "human_lambda": state.lambda_hat,
            "human_pi_sensory": state.pi_sensory,
            "human_pi_prior": state.pi_prior,
            "human_capacity": state.capacity,
            "human_in_corridor": float(state.in_healthy_corridor()),
            "human_regime": state.regime_label(),
            "distance_to_healthy": self.state.distance_to_healthy,
            "intervention_active": float(self.state.intervention_active),
        }

    def set_ara_state(self, ara_lambda: float, ara_pi: float):
        """
        Receive Ara's GUTC state for coupled regulation.

        Future: implement coupled dynamics where human and AI
        states influence each other's targets.
        """
        # For now, just log
        # Future: adjust human corridor based on AI state
        pass

    # ------------------------------------------------------------------
    # Research Data Export
    # ------------------------------------------------------------------

    def export_session_data(self) -> Dict[str, Any]:
        """
        Export session data for research analysis.

        Returns full trajectories and statistics.
        """
        trajectory = self.state_history
        repairs = self.repair_history

        if not trajectory:
            return {"error": "No data collected"}

        # Extract time series
        times = np.array([s.timestamp for s in trajectory])
        times = times - times[0]  # Relative to start

        lambdas = np.array([s.lambda_hat for s in trajectory])
        pi_s = np.array([s.pi_sensory for s in trajectory])
        pi_p = np.array([s.pi_prior for s in trajectory])
        capacities = np.array([s.capacity for s in trajectory])
        in_corridor = np.array([s.in_healthy_corridor() for s in trajectory])

        return {
            "duration_s": float(times[-1]) if len(times) > 0 else 0,
            "n_samples": len(trajectory),
            "n_interventions": self.state.total_interventions,

            # Time series
            "times": times.tolist(),
            "lambda": lambdas.tolist(),
            "pi_sensory": pi_s.tolist(),
            "pi_prior": pi_p.tolist(),
            "capacity": capacities.tolist(),
            "in_corridor": in_corridor.tolist(),

            # Statistics
            "lambda_mean": float(np.mean(lambdas)),
            "lambda_std": float(np.std(lambdas)),
            "pi_s_mean": float(np.mean(pi_s)),
            "pi_p_mean": float(np.mean(pi_p)),
            "capacity_mean": float(np.mean(capacities)),
            "time_in_corridor_fraction": float(np.mean(in_corridor)),

            # Repairs
            "n_repairs": len(repairs),
            "urgency_distribution": {
                "none": sum(1 for r in repairs if r.urgency == InterventionUrgency.NONE),
                "gentle": sum(1 for r in repairs if r.urgency == InterventionUrgency.GENTLE),
                "moderate": sum(1 for r in repairs if r.urgency == InterventionUrgency.MODERATE),
                "urgent": sum(1 for r in repairs if r.urgency == InterventionUrgency.URGENT),
                "critical": sum(1 for r in repairs if r.urgency == InterventionUrgency.CRITICAL),
            },
        }

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Get complete bridge status."""
        with self._state_lock:
            return {
                "running": self.state.running,
                "connected": self.state.connected,
                "total_updates": self.state.total_updates,
                "total_interventions": self.state.total_interventions,
                "current_state": {
                    "lambda": self.state.gutc_state.lambda_hat if self.state.gutc_state else None,
                    "pi_sensory": self.state.gutc_state.pi_sensory if self.state.gutc_state else None,
                    "pi_prior": self.state.gutc_state.pi_prior if self.state.gutc_state else None,
                    "regime": self.state.gutc_state.regime_label() if self.state.gutc_state else None,
                },
                "in_healthy_corridor": self.state.in_healthy_corridor,
                "distance_to_healthy": self.state.distance_to_healthy,
                "intervention_active": self.state.intervention_active,
                "time_in_corridor_fraction": self.state.time_in_corridor_fraction,
                "acquisition": self.acquisition.get_status(),
                "estimator": self.estimator.get_status(),
                "control": self.control.get_status(),
            }

    def status_string(self) -> str:
        """Get formatted status string."""
        status = self.get_status()
        state = status["current_state"]

        if state["lambda"] is None:
            state_str = "No data"
        else:
            corridor_emoji = "🟢" if status["in_healthy_corridor"] else "🟡" if status["distance_to_healthy"] < 0.5 else "🔴"
            state_str = (f"λ={state['lambda']:.2f}, "
                        f"Πs={state['pi_sensory']:.2f}, "
                        f"Πp={state['pi_prior']:.2f} "
                        f"[{state['regime']}] {corridor_emoji}")

        return (
            "╔══════════════════════════════════════════════╗\n"
            "║      NEURO-SYMBIOTIC HYPER-BRIDGE            ║\n"
            "╠══════════════════════════════════════════════╣\n"
            f"║ Status: {'🟢 Running' if status['running'] else '⚪ Stopped':<36}║\n"
            f"║ State: {state_str:<38}║\n"
            f"║ Distance: {status['distance_to_healthy']:.3f}                               ║\n"
            f"║ Interventions: {status['total_interventions']:<29}║\n"
            f"║ Time in corridor: {status['time_in_corridor_fraction']:.1%}                       ║\n"
            "╚══════════════════════════════════════════════╝"
        )


# =============================================================================
# Convenience: Demo Run
# =============================================================================

def run_demo_session(duration_s: float = 30.0, verbose: bool = True) -> Dict[str, Any]:
    """
    Run a demo NSHB session with simulated data.

    Returns session data for analysis.
    """
    bridge = NeuroSymbioticHyperBridge(verbose=verbose)
    bridge.start(threaded=False)

    start_time = time.time()
    while time.time() - start_time < duration_s:
        bridge.update()
        time.sleep(0.1)

    bridge.stop()

    print("\n" + bridge.status_string())

    return bridge.export_session_data()
