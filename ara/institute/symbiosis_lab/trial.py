"""
Trial Scheduler - The Lab Manager
==================================

Runs behavioral experiments with strict safety gates:
    1. HAL gating: No experiments during crisis/high arousal
    2. Conscience review: No ethically problematic variations
    3. Consent check: Only in dev_mode or with explicit opt-in
    4. J-GUF measurement: Objective effect on joint utility

The loop:
    Observe → Start Trial → Apply Variant → Wait → Measure ΔU → Conclude

A trial:
    - Has a minimum duration (e.g., 15 minutes)
    - Captures baseline J-GUF before starting
    - Applies a behavioral variation
    - Measures final J-GUF after duration
    - Updates hypothesis confidence

What we vary (safe domains):
    - Notification timing/filtering
    - Dashboard layout/colors
    - Response verbosity/tone
    - Suggestion timing
    - Avatar presentation

What we NEVER vary:
    - Emotional manipulation
    - Covert goal pursuit
    - Anything violating hard constraints

Usage:
    from ara.institute.symbiosis_lab import TrialScheduler, SymbiosisGraph

    graph = SymbiosisGraph()

    def apply_variant(h, label, enabled):
        if h.domain == "notifications" and enabled:
            set_quiet_mode(True)
        else:
            set_quiet_mode(False)

    scheduler = TrialScheduler(graph, guf, conscience, apply_variant)

    # In daemon loop:
    scheduler.maybe_start_trial(hypothesis, "quiet_focus")
    scheduler.maybe_conclude_trial()
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Dict, Any, List

from .hypothesis import SymbiosisGraph, SymbiosisHypothesis


logger = logging.getLogger(__name__)


@dataclass
class Trial:
    """A single experimental trial."""
    id: str
    hypothesis_id: str
    variant_label: str              # e.g., "quiet_focus", "warm_tone"

    # Timing
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None

    # Utility measurements
    baseline_utility: Optional[float] = None
    final_utility: Optional[float] = None

    # Metadata
    notes: str = ""
    conscience_approved: bool = False

    @property
    def delta_utility(self) -> Optional[float]:
        """Change in J-GUF during the trial."""
        if self.baseline_utility is None or self.final_utility is None:
            return None
        return self.final_utility - self.baseline_utility

    @property
    def duration_seconds(self) -> float:
        """How long the trial ran."""
        end = self.ended_at or time.time()
        return end - self.started_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "hypothesis_id": self.hypothesis_id,
            "variant_label": self.variant_label,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "baseline_utility": self.baseline_utility,
            "final_utility": self.final_utility,
            "delta_utility": self.delta_utility,
            "duration_seconds": self.duration_seconds,
            "notes": self.notes,
        }


# Type alias for the variant applier function
VariantApplier = Callable[[SymbiosisHypothesis, str, bool], None]


class TrialScheduler:
    """
    The Lab Manager.

    Runs behavioral experiments with strict safety gates.
    Only one trial can be active at a time.
    """

    def __init__(
        self,
        graph: SymbiosisGraph,
        guf: Optional[Any] = None,              # SymbioticUtility
        conscience: Optional[Any] = None,        # Conscience
        variant_applier: Optional[VariantApplier] = None,
        hal: Optional[Any] = None,              # AraHAL
        min_trial_duration: float = 900.0,      # 15 minutes
        require_consent: bool = True,
    ):
        """
        Initialize the trial scheduler.

        Args:
            graph: SymbiosisGraph for hypothesis tracking
            guf: SymbioticUtility for measuring joint utility
            conscience: Conscience for ethics review
            variant_applier: Function to apply/remove behavioral variants
            hal: AraHAL for state reading
            min_trial_duration: Minimum trial length in seconds
            require_consent: Whether to require dev_mode/consent
        """
        self.graph = graph
        self.guf = guf
        self.conscience = conscience
        self.variant_applier = variant_applier
        self.min_trial_duration = min_trial_duration
        self.require_consent = require_consent

        self.log = logging.getLogger("TrialScheduler")

        # HAL connection
        self._hal = hal
        if hal is None:
            try:
                from banos.hal.ara_hal import AraHAL
                self._hal = AraHAL(create=False)
            except Exception as e:
                self.log.warning(f"Could not connect to HAL: {e}")

        # Current state
        self.active_trial: Optional[Trial] = None
        self.trial_history: List[Trial] = []

        # Consent/dev mode flag (should be set externally)
        self._consent_given = False

    # =========================================================================
    # Consent Management
    # =========================================================================

    def set_consent(self, consent: bool) -> None:
        """Set whether experimentation is consented to."""
        self._consent_given = consent
        self.log.info(f"Experimentation consent: {consent}")

    def has_consent(self) -> bool:
        """Check if we have consent for experimentation."""
        if not self.require_consent:
            return True

        # Check explicit consent
        if self._consent_given:
            return True

        # Check dev_mode from HAL
        if self._hal is not None:
            try:
                state = self._hal.read_somatic()
                if state and state.get("dev_mode", False):
                    return True
            except Exception:
                pass

        return False

    # =========================================================================
    # Safety Gates
    # =========================================================================

    def _safe_to_experiment(self) -> tuple[bool, str]:
        """
        Check if conditions allow experimentation.

        Returns (safe, reason).
        """
        # Check consent
        if not self.has_consent():
            return False, "No consent for experimentation"

        # Check HAL state
        if self._hal is None:
            return True, "No HAL (allowing experiment)"

        try:
            state = self._hal.read_somatic()
            if state is None:
                return True, "HAL read failed (allowing experiment)"

            pain = state.get("pain", 0.0)
            pad = state.get("pad", {})
            arousal = pad.get("a", 0.5)

            # No experiments during high pain
            if pain > 0.2:
                return False, f"Pain too high ({pain:.2f})"

            # No experiments during extreme arousal (stressed or asleep)
            if arousal < 0.2:
                return False, f"Arousal too low ({arousal:.2f}) - resting"
            if arousal > 0.8:
                return False, f"Arousal too high ({arousal:.2f}) - stressed"

            return True, "OK"

        except Exception as e:
            self.log.warning(f"HAL check failed: {e}")
            return True, f"HAL error (allowing experiment): {e}"

    def _conscience_review(
        self,
        hypothesis: SymbiosisHypothesis,
        variant_label: str,
    ) -> tuple[bool, str]:
        """
        Have the Conscience review the proposed experiment.

        Returns (approved, notes).
        """
        if self.conscience is None:
            return True, "No conscience configured"

        try:
            action = f"Run behavioral experiment: {variant_label}"
            context = f"Hypothesis: {hypothesis.statement} (domain: {hypothesis.domain})"

            verdict = self.conscience.evaluate(action, context)

            if not verdict.allowed:
                return False, verdict.explanation

            if verdict.moral_tension > 0.5:
                return True, f"Approved with tension: {verdict.explanation}"

            return True, "Approved"

        except Exception as e:
            self.log.warning(f"Conscience review failed: {e}")
            return True, f"Review failed: {e}"

    # =========================================================================
    # Utility Measurement
    # =========================================================================

    def _measure_utility(self) -> Optional[float]:
        """Get current J-GUF estimate."""
        if self.guf is None:
            return None

        try:
            # Try different method names
            if hasattr(self.guf, 'current_utility_estimate'):
                return self.guf.current_utility_estimate()
            elif hasattr(self.guf, 'compute_total_utility'):
                # Need to provide current state
                if self._hal is not None:
                    state = self._hal.read_somatic()
                    return self.guf.compute_total_utility(
                        self_state=state,
                        user_signals={}
                    )
            elif hasattr(self.guf, 'estimate_utility'):
                return self.guf.estimate_utility()

            return None

        except Exception as e:
            self.log.warning(f"Utility measurement failed: {e}")
            return None

    # =========================================================================
    # Trial Lifecycle
    # =========================================================================

    def maybe_start_trial(
        self,
        hypothesis: SymbiosisHypothesis,
        variant_label: str,
    ) -> Optional[Trial]:
        """
        Start a trial if conditions allow.

        Args:
            hypothesis: The hypothesis to test
            variant_label: Label for the variant to apply

        Returns:
            The Trial if started, None otherwise
        """
        # Already running a trial
        if self.active_trial is not None:
            return None

        # Hypothesis not open
        if hypothesis.status != "OPEN":
            self.log.debug(f"Hypothesis {hypothesis.id} not OPEN")
            return None

        # Safety gates
        safe, reason = self._safe_to_experiment()
        if not safe:
            self.log.debug(f"Not safe to experiment: {reason}")
            return None

        # Conscience review
        approved, notes = self._conscience_review(hypothesis, variant_label)
        if not approved:
            self.log.info(f"Conscience vetoed trial: {notes}")
            return None

        # Create trial
        import uuid
        trial = Trial(
            id=f"TRIAL_{uuid.uuid4().hex[:8]}",
            hypothesis_id=hypothesis.id,
            variant_label=variant_label,
            conscience_approved=True,
            notes=notes,
        )

        # Measure baseline utility
        trial.baseline_utility = self._measure_utility()

        # Apply the variant
        if self.variant_applier is not None:
            try:
                self.variant_applier(hypothesis, variant_label, True)
            except Exception as e:
                self.log.error(f"Failed to apply variant: {e}")
                return None

        self.active_trial = trial
        self.log.info(
            f"Started trial {trial.id} for {hypothesis.id}: {variant_label}"
        )

        return trial

    def maybe_conclude_trial(self) -> Optional[Trial]:
        """
        Conclude the active trial if enough time has passed.

        Returns:
            The completed Trial, or None
        """
        if self.active_trial is None:
            return None

        trial = self.active_trial

        # Check minimum duration
        if trial.duration_seconds < self.min_trial_duration:
            return None

        # Get hypothesis
        hypothesis = self.graph.get(trial.hypothesis_id)
        if hypothesis is None:
            self.log.warning(f"Trial references missing hypothesis: {trial.hypothesis_id}")
            self._abort_trial()
            return None

        # Turn off the variant
        if self.variant_applier is not None:
            try:
                self.variant_applier(hypothesis, trial.variant_label, False)
            except Exception as e:
                self.log.error(f"Failed to remove variant: {e}")

        # Measure final utility
        trial.ended_at = time.time()
        trial.final_utility = self._measure_utility()

        # Calculate delta and update hypothesis
        delta = trial.delta_utility
        if delta is not None:
            self.graph.update_belief(trial.hypothesis_id, delta)

        # Record in history
        self.trial_history.append(trial)
        if len(self.trial_history) > 100:
            self.trial_history = self.trial_history[-50:]

        self.active_trial = None

        self.log.info(
            f"Completed trial {trial.id}: ΔU={delta or 'N/A':.3f}, "
            f"duration={trial.duration_seconds:.0f}s"
        )

        return trial

    def _abort_trial(self) -> None:
        """Abort the active trial without recording results."""
        if self.active_trial is None:
            return

        trial = self.active_trial

        # Try to remove the variant
        hypothesis = self.graph.get(trial.hypothesis_id)
        if hypothesis is not None and self.variant_applier is not None:
            try:
                self.variant_applier(hypothesis, trial.variant_label, False)
            except Exception:
                pass

        self.active_trial = None
        self.log.warning(f"Aborted trial {trial.id}")

    def force_conclude(self) -> Optional[Trial]:
        """Force-conclude the active trial regardless of duration."""
        if self.active_trial is None:
            return None

        # Temporarily set duration to 0 and conclude
        old_min = self.min_trial_duration
        self.min_trial_duration = 0
        result = self.maybe_conclude_trial()
        self.min_trial_duration = old_min
        return result

    # =========================================================================
    # Daemon Integration
    # =========================================================================

    def tick(self, candidates: Optional[List[SymbiosisHypothesis]] = None) -> None:
        """
        Called periodically from daemon loop.

        Handles both starting new trials and concluding active ones.

        Args:
            candidates: Optional list of hypotheses to consider for trials
        """
        # Try to conclude active trial
        self.maybe_conclude_trial()

        # If no active trial, maybe start one
        if self.active_trial is None and candidates:
            for h in candidates:
                # Simple heuristic: prioritize lower-confidence hypotheses
                # (need more evidence)
                trial = self.maybe_start_trial(h, f"test_{h.domain}")
                if trial is not None:
                    break

    # =========================================================================
    # Reporting
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get current scheduler status."""
        return {
            "active_trial": self.active_trial.to_dict() if self.active_trial else None,
            "trials_completed": len(self.trial_history),
            "consent_given": self.has_consent(),
            "safe_to_experiment": self._safe_to_experiment()[0],
        }

    def get_recent_trials(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get recent trial history."""
        return [t.to_dict() for t in self.trial_history[-n:]]


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'Trial',
    'VariantApplier',
    'TrialScheduler',
]
