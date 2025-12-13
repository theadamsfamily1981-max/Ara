"""
Brain Remodulator - Precision Thermostat for Neural Dynamics

A theoretical and implementable framework for correcting aberrant precision
weighting in predictive processing systems, inspired by computational
psychiatry models of schizophrenia and autism spectrum disorder (ASD).

THEORETICAL BASIS
=================

Predictive Processing Framework:
- The brain maintains a generative model that predicts sensory input
- Prediction errors (PE) = sensory input - predicted input
- Precision (Π) = confidence/weight assigned to signals
- Balance between prior beliefs and sensory evidence determines perception

Delusion Index (D):
    D = Π_prior / Π_sensory = force_prior / force_reality

    D = 1.0  → Balanced (healthy)
    D >> 1   → Prior-dominated (schizophrenia-like: hallucinations, delusions)
    D << 1   → Sensory-dominated (ASD-like: overwhelm, hyper-literalism)

Criticality (ρ):
    ρ < 0.7  → Subcritical: rigid, stereotyped, poor generalization
    ρ ≈ 0.85 → Near-critical: optimal information processing
    ρ > 1.1  → Supercritical: chaotic, unstable, noise amplification

DISORDER MODELS
===============

Schizophrenia Spectrum:
- Aberrantly high precision on priors/predictions
- Reduced precision on sensory prediction errors
- Results: false perceptions treated as real, delusional beliefs resistant to evidence
- Neural: reduced NMDA receptor function, dopamine dysregulation
- GUTC interpretation: D >> 1, possibly supercritical ρ

Autism Spectrum:
- Aberrantly high precision on sensory input
- Reduced precision on priors/predictions
- Results: sensory overwhelm, difficulty with abstraction, hyper-literal thinking
- Neural: altered E/I balance, reduced predictive smoothing
- GUTC interpretation: D << 1, possibly subcritical ρ

THE REMODULATOR
===============

A control system that:
1. Continuously monitors D (delusion index) and ρ (criticality)
2. Detects drift from the "critical corridor" (healthy operating range)
3. Computes corrective interventions to rebalance precision weighting
4. Can interface with various intervention modalities:
   - Neurofeedback (real-time brain state feedback)
   - tDCS/TMS (non-invasive brain stimulation)
   - Pharmacological models (simulated drug effects)
   - Behavioral interventions (attention training)

References:
- Friston, K. (2010). The free-energy principle: a unified brain theory?
- Adams, R.A., et al. (2013). Computational psychiatry: towards a mathematically informed understanding of mental illness
- Lawson, R.P., et al. (2014). An aberrant precision account of autism
- Sterzer, P., et al. (2018). The predictive coding account of psychosis
"""

from __future__ import annotations

import math
import time
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import deque
import threading

logger = logging.getLogger("ara.neuro.remodulator")


# =============================================================================
# Enums and Constants
# =============================================================================

class DisorderPattern(Enum):
    """Recognized aberrant precision patterns."""
    HEALTHY = "healthy"
    SCHIZOPHRENIA_LIKE = "schizophrenia_like"  # D >> 1, prior-dominated
    ASD_LIKE = "asd_like"                       # D << 1, sensory-dominated
    MIXED = "mixed"                             # Complex/unstable pattern
    UNKNOWN = "unknown"


class CriticalityRegime(Enum):
    """Criticality operating regime."""
    SUBCRITICAL = "subcritical"     # ρ < 0.7: too ordered
    CRITICAL = "critical"           # 0.7 ≤ ρ ≤ 1.1: optimal
    SUPERCRITICAL = "supercritical" # ρ > 1.1: too chaotic


class InterventionType(Enum):
    """Types of corrective interventions."""
    NONE = auto()
    BOOST_SENSORY_PRECISION = auto()    # For schizophrenia-like
    BOOST_PRIOR_PRECISION = auto()       # For ASD-like
    DAMPEN_CRITICALITY = auto()          # For supercritical
    INCREASE_CRITICALITY = auto()        # For subcritical
    STABILIZE = auto()                   # General stabilization


# Thresholds
D_HEALTHY_LOW = 0.5      # Below this = sensory-dominated
D_HEALTHY_HIGH = 2.0     # Above this = prior-dominated
D_CRISIS_LOW = 0.1       # Severe sensory-domination
D_CRISIS_HIGH = 10.0     # Severe prior-domination

RHO_SUBCRITICAL = 0.7
RHO_SUPERCRITICAL = 1.1
RHO_TARGET = 0.88        # Optimal operating point


# =============================================================================
# Core Data Structures
# =============================================================================

@dataclass
class PrecisionState:
    """Current precision weighting state."""
    pi_prior: float = 1.0           # Precision on priors/predictions
    pi_sensory: float = 1.0         # Precision on sensory evidence
    pi_action: float = 1.0          # Precision on action/motor predictions

    @property
    def D(self) -> float:
        """Delusion Index: prior/sensory precision ratio."""
        return self.pi_prior / max(self.pi_sensory, 0.001)

    @property
    def log_D(self) -> float:
        """Log10 of Delusion Index (for visualization)."""
        return math.log10(max(self.D, 0.001))

    def to_dict(self) -> Dict[str, float]:
        return {
            "pi_prior": round(self.pi_prior, 4),
            "pi_sensory": round(self.pi_sensory, 4),
            "pi_action": round(self.pi_action, 4),
            "D": round(self.D, 4),
            "log_D": round(self.log_D, 3),
        }


@dataclass
class CriticalityState:
    """Current criticality state."""
    rho: float = 0.88               # Branching ratio
    tau: float = 1.5                # Power-law exponent
    lambda_coupling: float = 1.0    # Global coupling strength

    @property
    def regime(self) -> CriticalityRegime:
        if self.rho < RHO_SUBCRITICAL:
            return CriticalityRegime.SUBCRITICAL
        elif self.rho > RHO_SUPERCRITICAL:
            return CriticalityRegime.SUPERCRITICAL
        else:
            return CriticalityRegime.CRITICAL

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rho": round(self.rho, 4),
            "tau": round(self.tau, 3),
            "lambda": round(self.lambda_coupling, 4),
            "regime": self.regime.value,
        }


@dataclass
class BrainState:
    """Complete brain state snapshot."""
    timestamp: float = field(default_factory=time.time)
    precision: PrecisionState = field(default_factory=PrecisionState)
    criticality: CriticalityState = field(default_factory=CriticalityState)

    # Derived pattern
    pattern: DisorderPattern = DisorderPattern.HEALTHY
    pattern_confidence: float = 0.0

    # Symptom indicators
    hallucination_risk: float = 0.0     # 0-1, high when D >> 1
    sensory_overwhelm_risk: float = 0.0 # 0-1, high when D << 1
    instability_risk: float = 0.0       # 0-1, high when ρ > 1

    def compute_pattern(self) -> DisorderPattern:
        """Classify current pattern based on state."""
        D = self.precision.D
        rho = self.criticality.rho

        if D_HEALTHY_LOW <= D <= D_HEALTHY_HIGH and RHO_SUBCRITICAL <= rho <= RHO_SUPERCRITICAL:
            return DisorderPattern.HEALTHY
        elif D > D_HEALTHY_HIGH:
            return DisorderPattern.SCHIZOPHRENIA_LIKE
        elif D < D_HEALTHY_LOW:
            return DisorderPattern.ASD_LIKE
        else:
            return DisorderPattern.MIXED

    def compute_risks(self):
        """Compute symptom risk scores."""
        D = self.precision.D
        rho = self.criticality.rho

        # Hallucination risk: increases as D >> 1
        if D > 1:
            self.hallucination_risk = min(1.0, (D - 1) / (D_CRISIS_HIGH - 1))
        else:
            self.hallucination_risk = 0.0

        # Sensory overwhelm risk: increases as D << 1
        if D < 1:
            self.sensory_overwhelm_risk = min(1.0, (1 - D) / (1 - D_CRISIS_LOW))
        else:
            self.sensory_overwhelm_risk = 0.0

        # Instability risk: increases as ρ > 1
        if rho > 1:
            self.instability_risk = min(1.0, (rho - 1) / 0.3)
        else:
            self.instability_risk = 0.0

    def update(self):
        """Update derived fields."""
        self.pattern = self.compute_pattern()
        self.compute_risks()
        self.pattern_confidence = self._compute_confidence()

    def _compute_confidence(self) -> float:
        """Compute confidence in pattern classification."""
        D = self.precision.D

        if self.pattern == DisorderPattern.HEALTHY:
            # Higher confidence when closer to D=1
            return 1.0 - min(1.0, abs(math.log10(D)) / 0.5)
        elif self.pattern == DisorderPattern.SCHIZOPHRENIA_LIKE:
            # Higher confidence when D is clearly elevated
            return min(1.0, (D - D_HEALTHY_HIGH) / D_HEALTHY_HIGH)
        elif self.pattern == DisorderPattern.ASD_LIKE:
            # Higher confidence when D is clearly reduced
            return min(1.0, (D_HEALTHY_LOW - D) / D_HEALTHY_LOW)
        else:
            return 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "precision": self.precision.to_dict(),
            "criticality": self.criticality.to_dict(),
            "pattern": self.pattern.value,
            "pattern_confidence": round(self.pattern_confidence, 3),
            "risks": {
                "hallucination": round(self.hallucination_risk, 3),
                "sensory_overwhelm": round(self.sensory_overwhelm_risk, 3),
                "instability": round(self.instability_risk, 3),
            },
        }


# =============================================================================
# Intervention Commands
# =============================================================================

@dataclass
class Intervention:
    """A corrective intervention command."""
    type: InterventionType
    magnitude: float                # 0-1 intensity
    target_parameter: str           # Which parameter to adjust
    delta: float                    # Amount to adjust by
    rationale: str                  # Human-readable explanation
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.name,
            "magnitude": round(self.magnitude, 3),
            "target": self.target_parameter,
            "delta": round(self.delta, 4),
            "rationale": self.rationale,
            "timestamp": self.timestamp,
        }


# =============================================================================
# Control Laws
# =============================================================================

class PrecisionControlLaw:
    """
    Control law for precision rebalancing.

    Implements a proportional-integral controller that:
    - Monitors D (delusion index) deviation from target
    - Computes corrective adjustments to precision weights
    - Applies smoothed, bounded interventions
    """

    def __init__(
        self,
        D_target: float = 1.0,          # Target delusion index
        rho_target: float = 0.88,       # Target criticality
        Kp_precision: float = 0.1,      # Proportional gain for precision
        Ki_precision: float = 0.01,     # Integral gain for precision
        Kp_criticality: float = 0.05,   # Proportional gain for criticality
        max_intervention: float = 0.2,  # Maximum single-step adjustment
        intervention_rate: float = 0.1, # Smoothing factor for interventions
    ):
        self.D_target = D_target
        self.rho_target = rho_target
        self.Kp_precision = Kp_precision
        self.Ki_precision = Ki_precision
        self.Kp_criticality = Kp_criticality
        self.max_intervention = max_intervention
        self.intervention_rate = intervention_rate

        # Integral error accumulator
        self.D_error_integral = 0.0
        self.rho_error_integral = 0.0

        # Last intervention (for smoothing)
        self.last_pi_prior_delta = 0.0
        self.last_pi_sensory_delta = 0.0
        self.last_lambda_delta = 0.0

    def compute(self, state: BrainState) -> List[Intervention]:
        """
        Compute corrective interventions for current state.

        Returns list of interventions to apply.
        """
        interventions = []

        D = state.precision.D
        rho = state.criticality.rho

        # === Precision Control (D → D_target) ===
        D_error = math.log(D) - math.log(self.D_target)  # Log-space error
        self.D_error_integral += D_error

        # Clamp integral to prevent windup
        self.D_error_integral = max(-10, min(10, self.D_error_integral))

        # PI control
        D_correction = (self.Kp_precision * D_error +
                        self.Ki_precision * self.D_error_integral)

        if abs(D_correction) > 0.01:
            if D > self.D_target:
                # Prior-dominated: boost sensory, reduce prior
                interventions.append(Intervention(
                    type=InterventionType.BOOST_SENSORY_PRECISION,
                    magnitude=min(abs(D_correction), self.max_intervention),
                    target_parameter="pi_sensory",
                    delta=self._smooth(abs(D_correction) * 0.5, "pi_sensory"),
                    rationale=f"D={D:.2f} > target: boosting sensory precision to counter prior-domination",
                ))
                interventions.append(Intervention(
                    type=InterventionType.BOOST_SENSORY_PRECISION,
                    magnitude=min(abs(D_correction), self.max_intervention),
                    target_parameter="pi_prior",
                    delta=self._smooth(-abs(D_correction) * 0.3, "pi_prior"),
                    rationale=f"D={D:.2f} > target: reducing prior precision",
                ))
            else:
                # Sensory-dominated: boost prior, reduce sensory
                interventions.append(Intervention(
                    type=InterventionType.BOOST_PRIOR_PRECISION,
                    magnitude=min(abs(D_correction), self.max_intervention),
                    target_parameter="pi_prior",
                    delta=self._smooth(abs(D_correction) * 0.5, "pi_prior"),
                    rationale=f"D={D:.2f} < target: boosting prior precision to counter sensory-domination",
                ))
                interventions.append(Intervention(
                    type=InterventionType.BOOST_PRIOR_PRECISION,
                    magnitude=min(abs(D_correction), self.max_intervention),
                    target_parameter="pi_sensory",
                    delta=self._smooth(-abs(D_correction) * 0.3, "pi_sensory"),
                    rationale=f"D={D:.2f} < target: reducing sensory precision",
                ))

        # === Criticality Control (ρ → ρ_target) ===
        rho_error = rho - self.rho_target
        self.rho_error_integral += rho_error
        self.rho_error_integral = max(-5, min(5, self.rho_error_integral))

        rho_correction = self.Kp_criticality * rho_error

        if abs(rho_correction) > 0.01:
            if rho > self.rho_target:
                # Supercritical: reduce coupling
                interventions.append(Intervention(
                    type=InterventionType.DAMPEN_CRITICALITY,
                    magnitude=min(abs(rho_correction), self.max_intervention),
                    target_parameter="lambda",
                    delta=self._smooth(-abs(rho_correction), "lambda"),
                    rationale=f"ρ={rho:.3f} > target: dampening coupling to reduce chaos",
                ))
            else:
                # Subcritical: increase coupling
                interventions.append(Intervention(
                    type=InterventionType.INCREASE_CRITICALITY,
                    magnitude=min(abs(rho_correction), self.max_intervention),
                    target_parameter="lambda",
                    delta=self._smooth(abs(rho_correction), "lambda"),
                    rationale=f"ρ={rho:.3f} < target: increasing coupling to improve dynamics",
                ))

        return interventions

    def _smooth(self, value: float, param: str) -> float:
        """Apply exponential smoothing to intervention."""
        if param == "pi_prior":
            smoothed = (self.intervention_rate * value +
                        (1 - self.intervention_rate) * self.last_pi_prior_delta)
            self.last_pi_prior_delta = smoothed
        elif param == "pi_sensory":
            smoothed = (self.intervention_rate * value +
                        (1 - self.intervention_rate) * self.last_pi_sensory_delta)
            self.last_pi_sensory_delta = smoothed
        elif param == "lambda":
            smoothed = (self.intervention_rate * value +
                        (1 - self.intervention_rate) * self.last_lambda_delta)
            self.last_lambda_delta = smoothed
        else:
            smoothed = value

        # Clamp
        return max(-self.max_intervention, min(self.max_intervention, smoothed))

    def reset(self):
        """Reset controller state."""
        self.D_error_integral = 0.0
        self.rho_error_integral = 0.0
        self.last_pi_prior_delta = 0.0
        self.last_pi_sensory_delta = 0.0
        self.last_lambda_delta = 0.0


# =============================================================================
# Intervention Modalities
# =============================================================================

class InterventionModality:
    """Base class for intervention modalities."""

    def apply(self, intervention: Intervention, state: BrainState) -> BrainState:
        """Apply intervention and return modified state."""
        raise NotImplementedError


class DirectPrecisionModality(InterventionModality):
    """
    Direct precision adjustment (theoretical/simulation).

    Directly modifies precision weights - used for simulation.
    In a real system, this would map to some physical intervention.
    """

    def apply(self, intervention: Intervention, state: BrainState) -> BrainState:
        new_state = BrainState(
            precision=PrecisionState(
                pi_prior=state.precision.pi_prior,
                pi_sensory=state.precision.pi_sensory,
                pi_action=state.precision.pi_action,
            ),
            criticality=CriticalityState(
                rho=state.criticality.rho,
                tau=state.criticality.tau,
                lambda_coupling=state.criticality.lambda_coupling,
            ),
        )

        if intervention.target_parameter == "pi_prior":
            new_state.precision.pi_prior = max(0.1, state.precision.pi_prior + intervention.delta)
        elif intervention.target_parameter == "pi_sensory":
            new_state.precision.pi_sensory = max(0.1, state.precision.pi_sensory + intervention.delta)
        elif intervention.target_parameter == "pi_action":
            new_state.precision.pi_action = max(0.1, state.precision.pi_action + intervention.delta)
        elif intervention.target_parameter == "lambda":
            new_state.criticality.lambda_coupling = max(0.1, state.criticality.lambda_coupling + intervention.delta)
            # Lambda affects rho
            new_state.criticality.rho = state.criticality.rho + intervention.delta * 0.5

        new_state.update()
        return new_state


class NeurofeedbackModality(InterventionModality):
    """
    Neurofeedback intervention modality.

    Maps precision adjustments to neurofeedback targets:
    - Boost sensory precision → train alpha suppression, increase gamma
    - Boost prior precision → train alpha enhancement, theta coherence
    """

    def __init__(self):
        self.feedback_targets: Dict[str, float] = {
            "alpha_power": 0.0,      # 8-12 Hz (relaxation, inhibition)
            "theta_power": 0.0,      # 4-8 Hz (memory, prediction)
            "gamma_power": 0.0,      # 30-100 Hz (attention, binding)
            "alpha_theta_ratio": 0.0,
        }

    def apply(self, intervention: Intervention, state: BrainState) -> BrainState:
        """Compute neurofeedback targets (doesn't directly modify state)."""

        if intervention.type == InterventionType.BOOST_SENSORY_PRECISION:
            # Target: increase gamma (attention), decrease alpha (open to input)
            self.feedback_targets["gamma_power"] += intervention.magnitude * 0.5
            self.feedback_targets["alpha_power"] -= intervention.magnitude * 0.3
            logger.info(f"Neurofeedback: Target gamma↑ alpha↓ for sensory boost")

        elif intervention.type == InterventionType.BOOST_PRIOR_PRECISION:
            # Target: increase theta (predictive), increase alpha (filter noise)
            self.feedback_targets["theta_power"] += intervention.magnitude * 0.5
            self.feedback_targets["alpha_power"] += intervention.magnitude * 0.3
            logger.info(f"Neurofeedback: Target theta↑ alpha↑ for prior boost")

        # Return unmodified state (real change comes from user training)
        return state

    def get_targets(self) -> Dict[str, float]:
        return self.feedback_targets.copy()


class PharmacologicalModality(InterventionModality):
    """
    Pharmacological intervention model.

    Maps precision adjustments to simulated drug effects:
    - Boost sensory precision → simulate NMDA enhancer (e.g., D-serine)
    - Boost prior precision → simulate dopamine modulator
    - Dampen criticality → simulate GABAergic agent
    """

    def __init__(self):
        self.drug_levels: Dict[str, float] = {
            "nmda_enhancer": 0.0,    # Boosts sensory PE
            "d2_antagonist": 0.0,    # Reduces aberrant salience
            "gaba_agonist": 0.0,     # Stabilizes, reduces criticality
            "acetylcholine": 0.0,    # Modulates attention
        }

    def apply(self, intervention: Intervention, state: BrainState) -> BrainState:
        """Compute drug effect and modify state."""
        new_state = BrainState(
            precision=PrecisionState(
                pi_prior=state.precision.pi_prior,
                pi_sensory=state.precision.pi_sensory,
                pi_action=state.precision.pi_action,
            ),
            criticality=CriticalityState(
                rho=state.criticality.rho,
                tau=state.criticality.tau,
                lambda_coupling=state.criticality.lambda_coupling,
            ),
        )

        if intervention.type == InterventionType.BOOST_SENSORY_PRECISION:
            # NMDA enhancer effect
            self.drug_levels["nmda_enhancer"] += intervention.magnitude
            effect = intervention.magnitude * 0.3
            new_state.precision.pi_sensory += effect
            logger.info(f"Pharma: NMDA enhancer +{effect:.3f} → sensory precision")

        elif intervention.type == InterventionType.BOOST_PRIOR_PRECISION:
            # D2 antagonist (counterintuitively can sharpen priors by reducing noise)
            self.drug_levels["d2_antagonist"] += intervention.magnitude
            effect = intervention.magnitude * 0.2
            new_state.precision.pi_prior += effect
            logger.info(f"Pharma: D2 modulation +{effect:.3f} → prior precision")

        elif intervention.type == InterventionType.DAMPEN_CRITICALITY:
            # GABAergic stabilization
            self.drug_levels["gaba_agonist"] += intervention.magnitude
            effect = intervention.magnitude * 0.15
            new_state.criticality.rho -= effect
            new_state.criticality.lambda_coupling -= effect * 0.5
            logger.info(f"Pharma: GABA agonist → ρ-{effect:.3f}")

        new_state.update()
        return new_state


# =============================================================================
# Main Brain Remodulator
# =============================================================================

class BrainRemodulator:
    """
    The Brain Remodulator - Precision Thermostat for Neural Dynamics.

    Monitors brain state and applies corrective interventions to maintain
    healthy precision weighting and criticality.

    Usage:
        remodulator = BrainRemodulator()

        # Update with current measurements
        remodulator.update(pi_prior=1.5, pi_sensory=0.8, rho=0.95)

        # Get current state and interventions
        state = remodulator.get_state()
        interventions = remodulator.get_pending_interventions()

        # Apply interventions (simulation mode)
        remodulator.apply_interventions()
    """

    def __init__(
        self,
        D_target: float = 1.0,
        rho_target: float = 0.88,
        modality: Optional[InterventionModality] = None,
        history_length: int = 100,
    ):
        self.D_target = D_target
        self.rho_target = rho_target

        # Current state
        self.state = BrainState()
        self.state.update()

        # Control law
        self.control = PrecisionControlLaw(
            D_target=D_target,
            rho_target=rho_target,
        )

        # Intervention modality
        self.modality = modality or DirectPrecisionModality()

        # Pending interventions
        self.pending_interventions: List[Intervention] = []

        # History
        self.history: deque = deque(maxlen=history_length)
        self.intervention_history: deque = deque(maxlen=history_length)

        # Callbacks
        self._on_pattern_change: Optional[Callable[[DisorderPattern, DisorderPattern], None]] = None
        self._on_intervention: Optional[Callable[[Intervention], None]] = None

        # Threading
        self._lock = threading.Lock()

        logger.info(f"BrainRemodulator initialized: D_target={D_target}, ρ_target={rho_target}")

    def update(
        self,
        pi_prior: Optional[float] = None,
        pi_sensory: Optional[float] = None,
        pi_action: Optional[float] = None,
        rho: Optional[float] = None,
        tau: Optional[float] = None,
        lambda_coupling: Optional[float] = None,
    ):
        """
        Update brain state with new measurements.

        Call this with real-time sensor data.
        """
        with self._lock:
            old_pattern = self.state.pattern

            # Update precision
            if pi_prior is not None:
                self.state.precision.pi_prior = pi_prior
            if pi_sensory is not None:
                self.state.precision.pi_sensory = pi_sensory
            if pi_action is not None:
                self.state.precision.pi_action = pi_action

            # Update criticality
            if rho is not None:
                self.state.criticality.rho = rho
            if tau is not None:
                self.state.criticality.tau = tau
            if lambda_coupling is not None:
                self.state.criticality.lambda_coupling = lambda_coupling

            self.state.timestamp = time.time()
            self.state.update()

            # Record history
            self.history.append(self.state.to_dict())

            # Check for pattern change
            if self.state.pattern != old_pattern:
                logger.warning(f"Pattern changed: {old_pattern.value} → {self.state.pattern.value}")
                if self._on_pattern_change:
                    self._on_pattern_change(old_pattern, self.state.pattern)

            # Compute interventions
            self.pending_interventions = self.control.compute(self.state)

            for intervention in self.pending_interventions:
                self.intervention_history.append(intervention.to_dict())
                if self._on_intervention:
                    self._on_intervention(intervention)

    def apply_interventions(self) -> BrainState:
        """
        Apply pending interventions and update state.

        Returns the new state after interventions.
        """
        with self._lock:
            for intervention in self.pending_interventions:
                self.state = self.modality.apply(intervention, self.state)
                logger.debug(f"Applied: {intervention.type.name} δ={intervention.delta:.4f}")

            self.pending_interventions.clear()
            return self.state

    def get_state(self) -> BrainState:
        """Get current brain state."""
        with self._lock:
            return self.state

    def get_pending_interventions(self) -> List[Intervention]:
        """Get list of pending interventions."""
        with self._lock:
            return self.pending_interventions.copy()

    def get_diagnosis(self) -> Dict[str, Any]:
        """
        Get comprehensive diagnosis of current state.

        Returns human-readable assessment and recommendations.
        """
        with self._lock:
            state = self.state
            D = state.precision.D
            rho = state.criticality.rho

            diagnosis = {
                "timestamp": time.time(),
                "pattern": state.pattern.value,
                "pattern_confidence": state.pattern_confidence,
                "D": D,
                "rho": rho,
                "risks": {
                    "hallucination": state.hallucination_risk,
                    "sensory_overwhelm": state.sensory_overwhelm_risk,
                    "instability": state.instability_risk,
                },
                "assessment": "",
                "recommendations": [],
            }

            # Generate assessment
            if state.pattern == DisorderPattern.HEALTHY:
                diagnosis["assessment"] = (
                    f"Operating in healthy range. D={D:.2f} indicates balanced "
                    f"precision weighting. ρ={rho:.3f} is near-critical optimal."
                )
            elif state.pattern == DisorderPattern.SCHIZOPHRENIA_LIKE:
                diagnosis["assessment"] = (
                    f"ALERT: Prior-dominated pattern detected. D={D:.2f} indicates "
                    f"excessive weight on predictions vs sensory evidence. "
                    f"Risk of false perceptions being treated as real."
                )
                diagnosis["recommendations"] = [
                    "Increase sensory precision weighting",
                    "Reduce prior/prediction confidence",
                    "Ground in concrete sensory experience",
                    "Reality testing exercises",
                ]
            elif state.pattern == DisorderPattern.ASD_LIKE:
                diagnosis["assessment"] = (
                    f"ALERT: Sensory-dominated pattern detected. D={D:.2f} indicates "
                    f"excessive weight on sensory input vs predictions. "
                    f"Risk of overwhelm and difficulty with abstraction."
                )
                diagnosis["recommendations"] = [
                    "Increase prior/prediction precision",
                    "Reduce raw sensory gain",
                    "Practice predictive/abstract thinking",
                    "Gradual sensory exposure with support",
                ]

            if rho > RHO_SUPERCRITICAL:
                diagnosis["recommendations"].append(
                    f"ρ={rho:.3f} is supercritical - apply stabilization"
                )
            elif rho < RHO_SUBCRITICAL:
                diagnosis["recommendations"].append(
                    f"ρ={rho:.3f} is subcritical - increase dynamic range"
                )

            return diagnosis

    def set_modality(self, modality: InterventionModality):
        """Set the intervention modality."""
        with self._lock:
            self.modality = modality

    def set_on_pattern_change(self, callback: Callable[[DisorderPattern, DisorderPattern], None]):
        """Set callback for pattern changes."""
        self._on_pattern_change = callback

    def set_on_intervention(self, callback: Callable[[Intervention], None]):
        """Set callback for interventions."""
        self._on_intervention = callback

    def reset(self):
        """Reset to default state."""
        with self._lock:
            self.state = BrainState()
            self.state.update()
            self.control.reset()
            self.pending_interventions.clear()
            self.history.clear()
            self.intervention_history.clear()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "DisorderPattern",
    "CriticalityRegime",
    "InterventionType",
    # Data structures
    "PrecisionState",
    "CriticalityState",
    "BrainState",
    "Intervention",
    # Control
    "PrecisionControlLaw",
    # Modalities
    "InterventionModality",
    "DirectPrecisionModality",
    "NeurofeedbackModality",
    "PharmacologicalModality",
    # Main class
    "BrainRemodulator",
]
