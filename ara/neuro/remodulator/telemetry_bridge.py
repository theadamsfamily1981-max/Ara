"""
Brain Remodulator - Telemetry Bridge

Connects the BrainRemodulator to the Cognitive Cockpit HUD,
enabling real-time visualization of:
- Precision state (D, Π_prior, Π_sensory)
- Disorder pattern detection
- Active interventions
- Risk indicators

Also provides a way to feed real sensor data into the remodulator.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional
import threading

from .core import (
    BrainRemodulator,
    BrainState,
    DisorderPattern,
    Intervention,
    InterventionType,
)

logger = logging.getLogger("ara.neuro.remodulator.bridge")

# Try to import telemetry
try:
    from hud.cognitive_cockpit import (
        get_cognitive_telemetry,
        CognitiveHUDTelemetry,
        MentalMode,
    )
    TELEMETRY_AVAILABLE = True
except ImportError:
    TELEMETRY_AVAILABLE = False
    logger.warning("Cognitive Cockpit telemetry not available")


class RemodulatorTelemetryBridge:
    """
    Bridges BrainRemodulator with Cognitive Cockpit HUD.

    Responsibilities:
    - Push remodulator state to HUD telemetry
    - Map disorder patterns to HUD display
    - Update ticker with intervention status
    """

    def __init__(
        self,
        remodulator: BrainRemodulator,
        telemetry: Optional["CognitiveHUDTelemetry"] = None,
        update_interval: float = 0.5,
    ):
        self.remodulator = remodulator
        self.update_interval = update_interval

        # Get or create telemetry
        if telemetry:
            self.telemetry = telemetry
        elif TELEMETRY_AVAILABLE:
            self.telemetry = get_cognitive_telemetry()
        else:
            self.telemetry = None
            logger.warning("No telemetry available - bridge will log only")

        # Set up callbacks on remodulator
        self.remodulator.set_on_pattern_change(self._on_pattern_change)
        self.remodulator.set_on_intervention(self._on_intervention)

        # Background update thread
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Step counter
        self._step = 0

    def start(self):
        """Start background telemetry updates."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()
        logger.info("RemodulatorTelemetryBridge started")

    def stop(self):
        """Stop background updates."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("RemodulatorTelemetryBridge stopped")

    def _update_loop(self):
        """Background update loop."""
        while self._running:
            self._push_to_telemetry()
            time.sleep(self.update_interval)

    def _push_to_telemetry(self):
        """Push current remodulator state to HUD telemetry."""
        if not self.telemetry:
            return

        self._step += 1
        state = self.remodulator.get_state()

        # Update delusion metrics
        self.telemetry.update_delusion(
            force_prior=state.precision.pi_prior,
            force_reality=state.precision.pi_sensory,
            guardrail_active=state.pattern != DisorderPattern.HEALTHY,
            hallucination_flag=state.hallucination_risk > 0.5,
        )

        # Update criticality
        self.telemetry.update_criticality(
            rho=state.criticality.rho,
            tau=state.criticality.tau,
        )

        # Update precision
        self.telemetry.update_precision(
            pi_y=state.precision.pi_prior,
            pi_mu=state.precision.pi_sensory,
        )

        # Map pattern to mental mode for display
        mode_map = {
            DisorderPattern.HEALTHY: MentalMode.WORKER,
            DisorderPattern.SCHIZOPHRENIA_LIKE: MentalMode.SCIENTIST,  # High internal
            DisorderPattern.ASD_LIKE: MentalMode.CHILL,  # Low abstraction
            DisorderPattern.MIXED: MentalMode.WORKER,
            DisorderPattern.UNKNOWN: MentalMode.WORKER,
        }

        if state.pattern in mode_map:
            self.telemetry.update_mental_mode(
                mode=mode_map[state.pattern],
                extrinsic_weight=state.precision.pi_sensory / 2,
                intrinsic_weight=state.precision.pi_prior / 2,
                energy_budget=1.0 - max(state.hallucination_risk, state.sensory_overwhelm_risk),
            )

        # Emit
        self.telemetry.emit(step=self._step)

    def _on_pattern_change(self, old: DisorderPattern, new: DisorderPattern):
        """Handle pattern change - update HUD ticker."""
        logger.warning(f"Pattern change: {old.value} → {new.value}")

        # This would update the ticker message in the HUD
        # The microcopy generator in telemetry will pick up the guardrail state

    def _on_intervention(self, intervention: Intervention):
        """Handle intervention - could update HUD with intervention info."""
        logger.info(f"Intervention: {intervention.type.name} → {intervention.target_parameter}")

    def push_once(self):
        """Push current state to telemetry once (manual mode)."""
        self._push_to_telemetry()

    def get_hud_state(self) -> Dict[str, Any]:
        """Get state formatted for HUD display."""
        state = self.remodulator.get_state()
        interventions = self.remodulator.get_pending_interventions()

        return {
            "pattern": state.pattern.value,
            "pattern_label": self._pattern_to_label(state.pattern),
            "D": state.precision.D,
            "rho": state.criticality.rho,
            "risks": {
                "hallucination": state.hallucination_risk,
                "overwhelm": state.sensory_overwhelm_risk,
                "instability": state.instability_risk,
            },
            "interventions_pending": len(interventions),
            "intervention_summary": [i.rationale for i in interventions[:3]],
        }

    def _pattern_to_label(self, pattern: DisorderPattern) -> str:
        """Convert pattern to HUD display label."""
        labels = {
            DisorderPattern.HEALTHY: "BALANCED",
            DisorderPattern.SCHIZOPHRENIA_LIKE: "PRIOR-DOMINATED",
            DisorderPattern.ASD_LIKE: "SENSORY-DOMINATED",
            DisorderPattern.MIXED: "UNSTABLE",
            DisorderPattern.UNKNOWN: "UNKNOWN",
        }
        return labels.get(pattern, "UNKNOWN")


def create_bridge(remodulator: BrainRemodulator) -> Optional[RemodulatorTelemetryBridge]:
    """Factory function to create telemetry bridge."""
    if not TELEMETRY_AVAILABLE:
        logger.warning("Telemetry not available, bridge not created")
        return None

    return RemodulatorTelemetryBridge(remodulator)


__all__ = [
    "RemodulatorTelemetryBridge",
    "create_bridge",
    "TELEMETRY_AVAILABLE",
]
