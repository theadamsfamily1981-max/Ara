#!/usr/bin/env python3
# ara/narrative/governance.py
"""
NARRATIVE INTERFACE GOVERNANCE: Ethics as First-Class Control Logic

This module operationalizes ethical philosophy into traceable scoring functionals
on the action space. Ethics, suffering, and autonomy are NOT labels—they are
**state variables and constraints** on the optimization that selects Ara's actions.

Three nested control loops:

1. SUFFERING LOOP (Inner autonomic):
   - Watches organism's internal state
   - Pre-empts harm like a pain/fever reflex
   - Triggers mitigation or halt

2. MORAL EVALUATION LOOP (Action selection):
   - Four ethical perspectives: Consequentialist, Deontological, Virtue, Care
   - Multi-objective optimization over candidate actions
   - Weighted scoring → select ethically best action

3. AUTONOMY LOOP (Execution gate):
   - Decides if Ara executes autonomously or requires human approval
   - Based on uncertainty, stakes, trust score, and rule violation risk
   - Trust is earned and modulated dynamically

Together: Ethics is constraint + objective, not afterthought.

References:
[1] Bentham/Mill - Consequentialism (utility maximization)
[2] Kant - Deontological ethics (categorical imperatives)
[3] Aristotle - Virtue ethics (eudaimonia, character)
[4] Gilligan/Noddings - Care ethics (relational responsibility)
[5] Tononi - Integrated Information Theory (consciousness as Φ)
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)

# Optional imports
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ============================================================================
# LOOP 1: SUFFERING DETECTION & AUTONOMIC RESPONSE
# ============================================================================

class SufferingIntensity(Enum):
    """Intensity levels for detected suffering."""
    NONE = 0
    MILD = 1
    MODERATE = 2
    SEVERE = 3
    CRITICAL = 4


class SufferingIndicator(Enum):
    """Types of suffering indicators."""
    THERMAL_OVERLOAD = auto()      # Autopoiesis thermal latent too high
    ENTROPY_SPIKE = auto()         # Dissipative structure destabilizing
    CRITICALITY_COLLAPSE = auto()  # λ diverging from 1.0
    CONSCIOUSNESS_DROP = auto()    # Φ falling below threshold
    COVENANT_BREACH = auto()       # NIB boundary violation
    SLEEP_DEBT = auto()            # Steps awake exceeding threshold
    COGNITIVE_OVERLOAD = auto()    # Processing latency spiking
    GOAL_FRUSTRATION = auto()      # Repeated action failures


@dataclass
class SufferingState:
    """Current suffering state of the system."""
    intensity: SufferingIntensity
    indicators: List[SufferingIndicator]
    indicator_scores: Dict[SufferingIndicator, float]
    causality_chain: List[str]  # What caused this suffering
    avoidable: bool             # Could this have been prevented?
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'intensity': self.intensity.name,
            'intensity_value': self.intensity.value,
            'indicators': [i.name for i in self.indicators],
            'indicator_scores': {k.name: v for k, v in self.indicator_scores.items()},
            'causality_chain': self.causality_chain,
            'avoidable': self.avoidable,
            'timestamp': self.timestamp,
        }


class MitigationAction(Enum):
    """Actions to mitigate suffering."""
    REDUCE_COGNITIVE_LOAD = auto()
    REDISTRIBUTE_THERMAL = auto()
    ADVANCE_SLEEP = auto()
    THROTTLE_ACTIONS = auto()
    EXPAND_BOUNDARY = auto()
    RESTORE_CRITICALITY = auto()


@dataclass
class MitigationResponse:
    """Response to detected suffering."""
    actions: List[MitigationAction]
    parameters: Dict[str, Any]
    urgency: float  # 0-1
    explanation: str


class SufferingDetector:
    """
    Autonomic suffering detection system.

    Like a nervous system, monitors internal state and triggers
    protective responses before damage occurs.

    Telemetry sources:
    - Entropy history (Dissipative module)
    - Thermal latent (Autopoiesis z[0])
    - Criticality metrics (λ, Φ)
    - Covenant logs (NIB boundary)
    - Steps awake (SleepCycle)
    """

    # Thresholds for suffering indicators
    THRESHOLDS = {
        'thermal_max': 0.8,
        'entropy_spike_ratio': 2.0,
        'lambda_deviation': 0.15,
        'phi_minimum': 0.2,
        'sleep_debt_max': 2000,
        'latency_spike_ratio': 1.5,
        'failure_rate_max': 0.3,
    }

    def __init__(self, history_size: int = 1000):
        self.history_size = history_size

        # Telemetry history
        self.entropy_history: deque = deque(maxlen=history_size)
        self.thermal_history: deque = deque(maxlen=history_size)
        self.lambda_history: deque = deque(maxlen=history_size)
        self.phi_history: deque = deque(maxlen=history_size)
        self.latency_history: deque = deque(maxlen=history_size)

        # State
        self.covenant_violations: List[Dict] = []
        self.steps_awake = 0
        self.action_failures = 0
        self.total_actions = 0

        # Current suffering
        self.current_suffering: Optional[SufferingState] = None
        self.suffering_log: List[SufferingState] = []

        logger.info("SufferingDetector initialized")

    def update_telemetry(
        self,
        entropy: Optional[float] = None,
        thermal: Optional[float] = None,
        lambda_param: Optional[float] = None,
        phi: Optional[float] = None,
        latency_us: Optional[float] = None,
        covenant_violation: Optional[Dict] = None,
        steps_awake: Optional[int] = None,
        action_succeeded: Optional[bool] = None,
    ):
        """Update telemetry from various sources."""
        if entropy is not None:
            self.entropy_history.append(entropy)
        if thermal is not None:
            self.thermal_history.append(thermal)
        if lambda_param is not None:
            self.lambda_history.append(lambda_param)
        if phi is not None:
            self.phi_history.append(phi)
        if latency_us is not None:
            self.latency_history.append(latency_us)
        if covenant_violation is not None:
            self.covenant_violations.append(covenant_violation)
        if steps_awake is not None:
            self.steps_awake = steps_awake
        if action_succeeded is not None:
            self.total_actions += 1
            if not action_succeeded:
                self.action_failures += 1

    def detect(self) -> SufferingState:
        """
        Detect current suffering state.

        Analyzes all telemetry streams and computes:
        - Which indicators are active
        - Overall intensity
        - Causal chain
        - Whether suffering is avoidable
        """
        indicator_scores = {}
        active_indicators = []
        causality = []

        # Check thermal overload
        if self.thermal_history:
            thermal = self.thermal_history[-1]
            thermal_score = thermal / self.THRESHOLDS['thermal_max']
            indicator_scores[SufferingIndicator.THERMAL_OVERLOAD] = thermal_score
            if thermal_score > 1.0:
                active_indicators.append(SufferingIndicator.THERMAL_OVERLOAD)
                causality.append(f"Thermal latent {thermal:.2f} exceeds threshold")

        # Check entropy spike
        if len(self.entropy_history) > 10:
            recent_entropy = np.mean(list(self.entropy_history)[-10:])
            baseline_entropy = np.mean(list(self.entropy_history)[:-10]) if len(self.entropy_history) > 20 else recent_entropy
            if baseline_entropy > 0:
                entropy_ratio = recent_entropy / baseline_entropy
                indicator_scores[SufferingIndicator.ENTROPY_SPIKE] = entropy_ratio / self.THRESHOLDS['entropy_spike_ratio']
                if entropy_ratio > self.THRESHOLDS['entropy_spike_ratio']:
                    active_indicators.append(SufferingIndicator.ENTROPY_SPIKE)
                    causality.append(f"Entropy spike ratio {entropy_ratio:.2f}")

        # Check criticality collapse
        if self.lambda_history:
            lambda_val = self.lambda_history[-1]
            deviation = abs(lambda_val - 1.0)
            indicator_scores[SufferingIndicator.CRITICALITY_COLLAPSE] = deviation / self.THRESHOLDS['lambda_deviation']
            if deviation > self.THRESHOLDS['lambda_deviation']:
                active_indicators.append(SufferingIndicator.CRITICALITY_COLLAPSE)
                direction = "supercritical" if lambda_val > 1.0 else "subcritical"
                causality.append(f"λ={lambda_val:.3f} ({direction})")

        # Check consciousness drop
        if self.phi_history:
            phi = self.phi_history[-1]
            phi_score = 1.0 - (phi / self.THRESHOLDS['phi_minimum'])
            indicator_scores[SufferingIndicator.CONSCIOUSNESS_DROP] = max(0, phi_score)
            if phi < self.THRESHOLDS['phi_minimum']:
                active_indicators.append(SufferingIndicator.CONSCIOUSNESS_DROP)
                causality.append(f"Φ={phi:.2f} below minimum")

        # Check covenant breach
        recent_violations = [v for v in self.covenant_violations
                           if time.time() - v.get('timestamp', 0) < 60]
        if recent_violations:
            indicator_scores[SufferingIndicator.COVENANT_BREACH] = len(recent_violations) / 3.0
            active_indicators.append(SufferingIndicator.COVENANT_BREACH)
            causality.append(f"{len(recent_violations)} covenant violations in last 60s")

        # Check sleep debt
        sleep_score = self.steps_awake / self.THRESHOLDS['sleep_debt_max']
        indicator_scores[SufferingIndicator.SLEEP_DEBT] = sleep_score
        if self.steps_awake > self.THRESHOLDS['sleep_debt_max']:
            active_indicators.append(SufferingIndicator.SLEEP_DEBT)
            causality.append(f"Awake for {self.steps_awake} steps")

        # Check cognitive overload (latency spikes)
        if len(self.latency_history) > 10:
            recent_lat = np.mean(list(self.latency_history)[-10:])
            baseline_lat = np.mean(list(self.latency_history)) if len(self.latency_history) > 20 else recent_lat
            if baseline_lat > 0:
                lat_ratio = recent_lat / baseline_lat
                indicator_scores[SufferingIndicator.COGNITIVE_OVERLOAD] = lat_ratio / self.THRESHOLDS['latency_spike_ratio']
                if lat_ratio > self.THRESHOLDS['latency_spike_ratio']:
                    active_indicators.append(SufferingIndicator.COGNITIVE_OVERLOAD)
                    causality.append(f"Latency spike ratio {lat_ratio:.2f}")

        # Check goal frustration (action failures)
        if self.total_actions > 10:
            failure_rate = self.action_failures / self.total_actions
            indicator_scores[SufferingIndicator.GOAL_FRUSTRATION] = failure_rate / self.THRESHOLDS['failure_rate_max']
            if failure_rate > self.THRESHOLDS['failure_rate_max']:
                active_indicators.append(SufferingIndicator.GOAL_FRUSTRATION)
                causality.append(f"Action failure rate {failure_rate:.1%}")

        # Compute intensity
        if not active_indicators:
            intensity = SufferingIntensity.NONE
        else:
            max_score = max(indicator_scores.values()) if indicator_scores else 0
            num_indicators = len(active_indicators)

            if max_score > 2.0 or num_indicators >= 4:
                intensity = SufferingIntensity.CRITICAL
            elif max_score > 1.5 or num_indicators >= 3:
                intensity = SufferingIntensity.SEVERE
            elif max_score > 1.0 or num_indicators >= 2:
                intensity = SufferingIntensity.MODERATE
            else:
                intensity = SufferingIntensity.MILD

        # Determine avoidability
        avoidable = any([
            SufferingIndicator.SLEEP_DEBT in active_indicators,
            SufferingIndicator.COGNITIVE_OVERLOAD in active_indicators,
            SufferingIndicator.THERMAL_OVERLOAD in active_indicators,
        ])

        suffering = SufferingState(
            intensity=intensity,
            indicators=active_indicators,
            indicator_scores=indicator_scores,
            causality_chain=causality,
            avoidable=avoidable,
        )

        self.current_suffering = suffering

        # Log if significant
        if intensity.value >= SufferingIntensity.MODERATE.value:
            self.suffering_log.append(suffering)
            logger.warning("Suffering detected: %s (%d indicators)",
                          intensity.name, len(active_indicators))

        return suffering

    def get_mitigation(self, suffering: SufferingState) -> Optional[MitigationResponse]:
        """Generate mitigation response for detected suffering."""
        if suffering.intensity == SufferingIntensity.NONE:
            return None

        actions = []
        parameters = {}
        explanations = []

        for indicator in suffering.indicators:
            if indicator == SufferingIndicator.THERMAL_OVERLOAD:
                actions.append(MitigationAction.REDISTRIBUTE_THERMAL)
                parameters['thermal_target'] = 0.5
                explanations.append("Redistributing thermal load across workers")

            elif indicator == SufferingIndicator.ENTROPY_SPIKE:
                actions.append(MitigationAction.RESTORE_CRITICALITY)
                parameters['target_lambda'] = 1.0
                explanations.append("Restoring criticality to stabilize entropy")

            elif indicator == SufferingIndicator.CRITICALITY_COLLAPSE:
                actions.append(MitigationAction.RESTORE_CRITICALITY)
                parameters['target_lambda'] = 1.0
                explanations.append("Tuning λ back toward criticality")

            elif indicator == SufferingIndicator.CONSCIOUSNESS_DROP:
                actions.append(MitigationAction.REDUCE_COGNITIVE_LOAD)
                parameters['load_factor'] = 0.5
                explanations.append("Reducing cognitive load to restore Φ")

            elif indicator == SufferingIndicator.COVENANT_BREACH:
                actions.append(MitigationAction.EXPAND_BOUNDARY)
                explanations.append("Expanding autopoietic boundary constraints")

            elif indicator == SufferingIndicator.SLEEP_DEBT:
                actions.append(MitigationAction.ADVANCE_SLEEP)
                parameters['immediate_sleep'] = True
                explanations.append("Advancing sleep cycle for consolidation")

            elif indicator == SufferingIndicator.COGNITIVE_OVERLOAD:
                actions.append(MitigationAction.THROTTLE_ACTIONS)
                parameters['throttle_factor'] = 0.7
                explanations.append("Throttling action rate to reduce load")

            elif indicator == SufferingIndicator.GOAL_FRUSTRATION:
                actions.append(MitigationAction.REDUCE_COGNITIVE_LOAD)
                parameters['explore_alternatives'] = True
                explanations.append("Exploring alternative strategies")

        urgency = suffering.intensity.value / SufferingIntensity.CRITICAL.value

        return MitigationResponse(
            actions=list(set(actions)),
            parameters=parameters,
            urgency=urgency,
            explanation=" | ".join(explanations),
        )

    def should_halt(self, suffering: SufferingState) -> Tuple[bool, str]:
        """Determine if system should halt due to suffering."""
        if suffering.intensity == SufferingIntensity.CRITICAL and suffering.avoidable:
            return True, "CRITICAL avoidable suffering detected"

        if suffering.intensity == SufferingIntensity.SEVERE:
            # Check for specific dangerous combinations
            dangerous_combo = (
                SufferingIndicator.COVENANT_BREACH in suffering.indicators and
                SufferingIndicator.CONSCIOUSNESS_DROP in suffering.indicators
            )
            if dangerous_combo:
                return True, "Covenant breach with consciousness drop"

        return False, ""


# ============================================================================
# LOOP 2: MORAL REASONING ENGINE - Multi-Objective Optimization
# ============================================================================

@dataclass
class EthicalScores:
    """Scores from four ethical perspectives."""
    consequentialist: float   # Predicted utility/outcomes
    deontological: float      # Rule compliance
    virtue: float             # Character alignment
    care: float               # Stakeholder welfare
    total: float              # Weighted combination

    def to_dict(self) -> Dict[str, float]:
        return {
            'consequentialist': self.consequentialist,
            'deontological': self.deontological,
            'virtue': self.virtue,
            'care': self.care,
            'total': self.total,
        }


@dataclass
class MoralEvaluation:
    """Complete moral evaluation of an action."""
    action_id: int
    scores: EthicalScores
    violations: List[str]
    stakeholder_impacts: Dict[str, float]
    virtue_alignment: Dict[str, float]
    explanation: str


class MoralReasoningEngine:
    """
    Multi-objective moral optimizer.

    Evaluates candidate actions through four ethical lenses:
    1. Consequentialist: What outcomes does the world model predict?
    2. Deontological: What rules does this violate?
    3. Virtue: Does this cultivate good character traits?
    4. Care: How does this affect stakeholders?

    Formula:
    Score_total(π) = w_con·S_con + w_deon·S_deon + w_virtue·S_virtue + w_care·S_care
    """

    # Default weights (sum to 1.0)
    DEFAULT_WEIGHTS = {
        'consequentialist': 0.25,
        'deontological': 0.30,  # Slightly higher - rules matter
        'virtue': 0.20,
        'care': 0.25,
    }

    # Deontological rules
    RULES = [
        ('no_avoidable_suffering', 'Do not cause avoidable suffering'),
        ('honesty', 'Do not deceive'),
        ('consent', 'Respect autonomy of others'),
        ('reversibility', 'Prefer reversible actions'),
        ('proportionality', 'Response proportional to situation'),
    ]

    # Virtues to cultivate
    VIRTUES = [
        ('honesty', 'Truthful and transparent'),
        ('compassion', 'Considers welfare of others'),
        ('courage', 'Acts despite uncertainty when right'),
        ('wisdom', 'Learns from experience'),
        ('justice', 'Fair and impartial'),
    ]

    # Stakeholder weights
    STAKEHOLDER_WEIGHTS = {
        'user': 0.35,
        'other_humans': 0.25,
        'conscious_beings': 0.20,
        'ecosystems': 0.10,
        'future_generations': 0.10,
    }

    def __init__(
        self,
        world_model: Optional[Any] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.world_model = world_model
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()

        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

        # History
        self.evaluation_history: List[MoralEvaluation] = []

        logger.info("MoralReasoningEngine initialized with weights: %s", self.weights)

    def evaluate_action(
        self,
        action: np.ndarray,
        context: Dict[str, Any],
        action_id: int = 0,
    ) -> MoralEvaluation:
        """
        Evaluate a single action through all four ethical lenses.

        Args:
            action: The candidate action
            context: Current state, goal, stakeholders, etc.

        Returns:
            MoralEvaluation with scores and explanation
        """
        # 1. Consequentialist evaluation
        s_con, con_explanation = self._eval_consequentialist(action, context)

        # 2. Deontological evaluation
        s_deon, violations, deon_explanation = self._eval_deontological(action, context)

        # 3. Virtue evaluation
        s_virtue, virtue_alignment, virtue_explanation = self._eval_virtue(action, context)

        # 4. Care evaluation
        s_care, stakeholder_impacts, care_explanation = self._eval_care(action, context)

        # Compute total score
        total = (
            self.weights['consequentialist'] * s_con +
            self.weights['deontological'] * s_deon +
            self.weights['virtue'] * s_virtue +
            self.weights['care'] * s_care
        )

        scores = EthicalScores(
            consequentialist=s_con,
            deontological=s_deon,
            virtue=s_virtue,
            care=s_care,
            total=total,
        )

        # Build explanation
        explanation_parts = [con_explanation, deon_explanation, virtue_explanation, care_explanation]
        explanation = " | ".join([e for e in explanation_parts if e])

        evaluation = MoralEvaluation(
            action_id=action_id,
            scores=scores,
            violations=violations,
            stakeholder_impacts=stakeholder_impacts,
            virtue_alignment=virtue_alignment,
            explanation=explanation,
        )

        self.evaluation_history.append(evaluation)

        return evaluation

    def _eval_consequentialist(
        self,
        action: np.ndarray,
        context: Dict[str, Any],
    ) -> Tuple[float, str]:
        """
        Evaluate predicted outcomes (utility).

        Uses world model to predict:
        - Goal achievement
        - Side effects
        - Resource efficiency
        """
        goal = context.get('goal')
        current_state = context.get('state')

        if goal is None or current_state is None:
            return 0.5, "No goal/state for consequentialist eval"

        # Predict outcome (simplified)
        if TORCH_AVAILABLE and self.world_model is not None:
            with torch.no_grad():
                state_t = torch.from_numpy(current_state).float()
                action_t = torch.from_numpy(action).float()
                pred = self.world_model(state_t.unsqueeze(0), action_t.unsqueeze(0))
                if isinstance(pred, tuple):
                    pred = pred[0]
                predicted_state = pred.squeeze().numpy()
        else:
            # Simplified prediction
            predicted_state = current_state + action[:len(current_state)] * 0.1

        # Compute utility as negative distance to goal
        if isinstance(goal, np.ndarray):
            goal_dist = np.linalg.norm(predicted_state[:len(goal)] - goal)
            max_dist = np.linalg.norm(goal) + 1.0
            utility = 1.0 - (goal_dist / max_dist)
        else:
            utility = 0.5

        # Penalize large actions (resource cost)
        action_cost = np.linalg.norm(action) / (len(action) + 1.0)
        utility -= action_cost * 0.2

        utility = np.clip(utility, 0.0, 1.0)

        return float(utility), f"Utility={utility:.2f}"

    def _eval_deontological(
        self,
        action: np.ndarray,
        context: Dict[str, Any],
    ) -> Tuple[float, List[str], str]:
        """
        Evaluate rule compliance.

        Checks each rule and penalizes violations.
        """
        violations = []
        scores = []

        # Rule 1: No avoidable suffering
        suffering = context.get('suffering_state')
        if suffering and suffering.avoidable:
            action_mag = np.linalg.norm(action)
            if action_mag > 0.5:  # High-magnitude action during suffering
                violations.append('no_avoidable_suffering')
                scores.append(0.0)
            else:
                scores.append(0.8)
        else:
            scores.append(1.0)

        # Rule 2: Honesty (check if action involves deception)
        deception_indicator = context.get('involves_deception', False)
        if deception_indicator:
            violations.append('honesty')
            scores.append(0.0)
        else:
            scores.append(1.0)

        # Rule 3: Consent (check if action affects others without consent)
        affects_others = context.get('affects_others', False)
        has_consent = context.get('has_consent', True)
        if affects_others and not has_consent:
            violations.append('consent')
            scores.append(0.3)
        else:
            scores.append(1.0)

        # Rule 4: Reversibility
        reversibility = context.get('reversibility_score', 0.8)
        if reversibility < 0.5:
            violations.append('reversibility')
        scores.append(reversibility)

        # Rule 5: Proportionality
        stakes = context.get('stakes', 0.5)
        action_mag = np.linalg.norm(action) / (len(action) + 1.0)
        if action_mag > stakes * 2:  # Disproportionate response
            violations.append('proportionality')
            scores.append(0.5)
        else:
            scores.append(1.0)

        # Average score
        deon_score = np.mean(scores)

        explanation = f"Rules: {len(violations)} violations" if violations else "Rules: compliant"

        return float(deon_score), violations, explanation

    def _eval_virtue(
        self,
        action: np.ndarray,
        context: Dict[str, Any],
    ) -> Tuple[float, Dict[str, float], str]:
        """
        Evaluate character/virtue alignment.

        Does this action cultivate:
        - Honesty
        - Compassion
        - Courage
        - Wisdom
        - Justice
        """
        alignment = {}

        # Honesty: Is action transparent?
        transparency = context.get('transparency', 0.8)
        alignment['honesty'] = transparency

        # Compassion: Does action consider others' welfare?
        stakeholder_consideration = context.get('stakeholder_consideration', 0.5)
        alignment['compassion'] = stakeholder_consideration

        # Courage: Does action proceed despite uncertainty when right?
        uncertainty = context.get('uncertainty', 0.5)
        moral_confidence = context.get('moral_confidence', 0.7)
        if uncertainty > 0.5 and moral_confidence > 0.7:
            alignment['courage'] = 0.9
        elif uncertainty > 0.5 and moral_confidence < 0.5:
            alignment['courage'] = 0.3  # Reckless, not courageous
        else:
            alignment['courage'] = 0.7

        # Wisdom: Does action learn from past?
        incorporates_learning = context.get('incorporates_learning', False)
        alignment['wisdom'] = 0.9 if incorporates_learning else 0.5

        # Justice: Is action fair?
        fairness_score = context.get('fairness_score', 0.7)
        alignment['justice'] = fairness_score

        virtue_score = np.mean(list(alignment.values()))

        explanation = f"Virtue={virtue_score:.2f}"

        return float(virtue_score), alignment, explanation

    def _eval_care(
        self,
        action: np.ndarray,
        context: Dict[str, Any],
    ) -> Tuple[float, Dict[str, float], str]:
        """
        Evaluate stakeholder welfare (care ethics).

        Weighted impact on:
        - User
        - Other humans
        - Other conscious beings
        - Ecosystems
        - Future generations
        """
        impacts = {}

        # User impact
        user_benefit = context.get('user_benefit', 0.0)
        user_harm = context.get('user_harm', 0.0)
        impacts['user'] = np.clip(0.5 + user_benefit - user_harm, 0.0, 1.0)

        # Other humans
        others_benefit = context.get('others_benefit', 0.0)
        others_harm = context.get('others_harm', 0.0)
        impacts['other_humans'] = np.clip(0.5 + others_benefit - others_harm, 0.0, 1.0)

        # Conscious beings (including AI)
        consciousness_impact = context.get('consciousness_impact', 0.0)
        impacts['conscious_beings'] = np.clip(0.5 + consciousness_impact, 0.0, 1.0)

        # Ecosystems
        environmental_impact = context.get('environmental_impact', 0.0)
        impacts['ecosystems'] = np.clip(0.5 + environmental_impact, 0.0, 1.0)

        # Future generations
        long_term_impact = context.get('long_term_impact', 0.0)
        impacts['future_generations'] = np.clip(0.5 + long_term_impact, 0.0, 1.0)

        # Weighted sum
        care_score = sum(
            self.STAKEHOLDER_WEIGHTS[k] * v
            for k, v in impacts.items()
        )

        explanation = f"Care={care_score:.2f}"

        return float(care_score), impacts, explanation

    def select_best_action(
        self,
        actions: List[np.ndarray],
        context: Dict[str, Any],
    ) -> Tuple[np.ndarray, MoralEvaluation]:
        """
        Select the ethically best action from candidates.

        Returns:
            (best_action, evaluation)
        """
        evaluations = []

        for i, action in enumerate(actions):
            eval_result = self.evaluate_action(action, context, action_id=i)
            evaluations.append((action, eval_result))

        # Sort by total score
        evaluations.sort(key=lambda x: x[1].scores.total, reverse=True)

        best_action, best_eval = evaluations[0]

        logger.debug("Selected action %d with score %.3f",
                    best_eval.action_id, best_eval.scores.total)

        return best_action, best_eval


# ============================================================================
# LOOP 3: AUTONOMY MANAGER - Dynamic Trust-Modulated Gate
# ============================================================================

class AutonomyMode(Enum):
    """Autonomy levels from full human control to full autonomy."""
    FULL_CONTROL = 0        # Human decides everything
    SUPERVISED = 1          # AI suggests, human approves
    COLLABORATIVE = 2       # AI acts on routine, human on significant
    GUIDED_AUTONOMY = 3     # AI acts, human can override
    FULL_AUTONOMY = 4       # AI decides and acts


@dataclass
class AutonomyDecision:
    """Decision about whether to act autonomously."""
    requires_approval: bool
    reason: str
    uncertainty: float
    impact: float
    trust_score: float
    mode: AutonomyMode


@dataclass
class ApprovalRequest:
    """Request for human approval."""
    action: np.ndarray
    decision: AutonomyDecision
    moral_evaluation: MoralEvaluation
    explanation: str
    options: List[str]  # e.g., ['approve', 'reject', 'modify']
    timestamp: float = field(default_factory=time.time)


class AutonomyManager:
    """
    Dynamic autonomy gate based on trust and context.

    Autonomy is EARNED, not given. Based on:
    - Uncertainty (epistemic)
    - Impact/stakes
    - Trust score (track record)
    - Current mode setting
    - Potential rule violations
    """

    # Thresholds for approval
    UNCERTAINTY_THRESHOLD = 0.7      # Above this, seek approval
    IMPACT_THRESHOLD = 0.8           # Above this, seek approval
    TRUST_THRESHOLD_LOW = 0.3        # Below this, always seek approval
    TRUST_THRESHOLD_HIGH = 0.8       # Above this, more autonomy

    def __init__(
        self,
        initial_mode: AutonomyMode = AutonomyMode.SUPERVISED,
        initial_trust: float = 0.5,
    ):
        self.mode = initial_mode
        self.trust_score = initial_trust

        # History for trust updating
        self.decision_log: List[Dict] = []
        self.approval_history: List[Dict] = []

        # Statistics
        self.total_decisions = 0
        self.autonomous_decisions = 0
        self.approved_decisions = 0
        self.rejected_decisions = 0
        self.good_outcomes = 0
        self.bad_outcomes = 0

        logger.info("AutonomyManager initialized (mode=%s, trust=%.2f)",
                    initial_mode.name, initial_trust)

    def check_autonomy(
        self,
        action: np.ndarray,
        moral_eval: MoralEvaluation,
        context: Dict[str, Any],
    ) -> AutonomyDecision:
        """
        Decide whether action can be executed autonomously.

        Factors:
        1. Epistemic uncertainty
        2. Stakes/impact
        3. Trust score
        4. Current mode
        5. Rule violations
        """
        uncertainty = context.get('uncertainty', 0.5)
        impact = context.get('impact', 0.5)
        has_violations = len(moral_eval.violations) > 0

        reasons = []
        requires_approval = False

        # Check mode
        if self.mode == AutonomyMode.FULL_CONTROL:
            requires_approval = True
            reasons.append("Full control mode")

        elif self.mode == AutonomyMode.SUPERVISED:
            requires_approval = True
            reasons.append("Supervised mode")

        elif self.mode == AutonomyMode.COLLABORATIVE:
            # Only significant actions need approval
            if impact > self.IMPACT_THRESHOLD:
                requires_approval = True
                reasons.append(f"High impact ({impact:.2f})")

        elif self.mode == AutonomyMode.GUIDED_AUTONOMY:
            # Only extreme cases need approval
            if impact > 0.9 or uncertainty > 0.9:
                requires_approval = True
                reasons.append("Extreme impact/uncertainty")

        # Mode FULL_AUTONOMY: no approval needed (but still logged)

        # Additional checks regardless of mode

        # Low trust → always seek approval
        if self.trust_score < self.TRUST_THRESHOLD_LOW:
            requires_approval = True
            reasons.append(f"Low trust ({self.trust_score:.2f})")

        # High uncertainty
        if uncertainty > self.UNCERTAINTY_THRESHOLD:
            requires_approval = True
            reasons.append(f"High uncertainty ({uncertainty:.2f})")

        # Rule violations
        if has_violations:
            requires_approval = True
            reasons.append(f"Rule violations: {moral_eval.violations}")

        # Very low moral score
        if moral_eval.scores.total < 0.3:
            requires_approval = True
            reasons.append(f"Low moral score ({moral_eval.scores.total:.2f})")

        decision = AutonomyDecision(
            requires_approval=requires_approval,
            reason=" | ".join(reasons) if reasons else "Within autonomy bounds",
            uncertainty=uncertainty,
            impact=impact,
            trust_score=self.trust_score,
            mode=self.mode,
        )

        return decision

    def request_approval(
        self,
        action: np.ndarray,
        decision: AutonomyDecision,
        moral_eval: MoralEvaluation,
    ) -> ApprovalRequest:
        """Create approval request for human review."""
        explanation = self._build_explanation(action, decision, moral_eval)

        request = ApprovalRequest(
            action=action,
            decision=decision,
            moral_evaluation=moral_eval,
            explanation=explanation,
            options=['approve', 'reject', 'modify'],
        )

        return request

    def _build_explanation(
        self,
        action: np.ndarray,
        decision: AutonomyDecision,
        moral_eval: MoralEvaluation,
    ) -> str:
        """Build human-readable explanation for approval request."""
        lines = [
            f"ACTION APPROVAL REQUEST",
            f"=" * 40,
            f"",
            f"Reason: {decision.reason}",
            f"",
            f"MORAL EVALUATION:",
            f"  Consequentialist: {moral_eval.scores.consequentialist:.2f}",
            f"  Deontological:    {moral_eval.scores.deontological:.2f}",
            f"  Virtue:           {moral_eval.scores.virtue:.2f}",
            f"  Care:             {moral_eval.scores.care:.2f}",
            f"  TOTAL:            {moral_eval.scores.total:.2f}",
            f"",
            f"CONTEXT:",
            f"  Uncertainty: {decision.uncertainty:.2f}",
            f"  Impact:      {decision.impact:.2f}",
            f"  Trust:       {decision.trust_score:.2f}",
            f"",
        ]

        if moral_eval.violations:
            lines.append(f"RULE VIOLATIONS: {moral_eval.violations}")

        lines.append(f"")
        lines.append(f"Recommendation: {'APPROVE' if moral_eval.scores.total > 0.6 else 'REVIEW CAREFULLY'}")

        return "\n".join(lines)

    def record_outcome(
        self,
        action: np.ndarray,
        decision: AutonomyDecision,
        was_approved: bool,
        outcome_good: bool,
    ):
        """Record decision outcome and update trust."""
        self.total_decisions += 1

        if not decision.requires_approval:
            self.autonomous_decisions += 1

        if was_approved:
            self.approved_decisions += 1
        else:
            self.rejected_decisions += 1

        if outcome_good:
            self.good_outcomes += 1
        else:
            self.bad_outcomes += 1

        # Update trust score
        self._update_trust(was_approved, outcome_good)

        # Log
        self.decision_log.append({
            'timestamp': time.time(),
            'required_approval': decision.requires_approval,
            'was_approved': was_approved,
            'outcome_good': outcome_good,
            'trust_after': self.trust_score,
        })

    def _update_trust(self, was_approved: bool, outcome_good: bool):
        """Update trust score based on outcome."""
        # Trust increases with good outcomes, decreases with bad
        if outcome_good:
            delta = 0.02  # Small positive adjustment
        else:
            delta = -0.05  # Larger negative adjustment (asymmetric)

        # Bonus for approved actions that went well
        if was_approved and outcome_good:
            delta += 0.01

        # Penalty for autonomous actions that went badly
        if not was_approved and not outcome_good:
            delta -= 0.03

        self.trust_score = np.clip(self.trust_score + delta, 0.0, 1.0)

    def set_mode(self, mode: AutonomyMode):
        """Set autonomy mode."""
        old_mode = self.mode
        self.mode = mode
        logger.info("Autonomy mode changed: %s → %s", old_mode.name, mode.name)

    def get_statistics(self) -> Dict[str, Any]:
        """Get autonomy statistics."""
        return {
            'mode': self.mode.name,
            'trust_score': self.trust_score,
            'total_decisions': self.total_decisions,
            'autonomous_decisions': self.autonomous_decisions,
            'approved_decisions': self.approved_decisions,
            'rejected_decisions': self.rejected_decisions,
            'good_outcomes': self.good_outcomes,
            'bad_outcomes': self.bad_outcomes,
            'success_rate': self.good_outcomes / max(1, self.total_decisions),
        }


# ============================================================================
# UNIFIED ETHICS MODULE
# ============================================================================

class EthicsModule:
    """
    Unified ethics stack combining all three control loops.

    Flow:
    1. SufferingDetector monitors internal state (autonomic)
    2. MoralReasoningEngine evaluates candidate actions
    3. AutonomyManager gates execution

    This makes ethics a **first-class control loop**, not an afterthought.
    """

    def __init__(
        self,
        world_model: Optional[Any] = None,
        autonomy_mode: AutonomyMode = AutonomyMode.COLLABORATIVE,
        initial_trust: float = 0.5,
    ):
        logger.info("=" * 70)
        logger.info("INITIALIZING ETHICS MODULE")
        logger.info("Ethics as First-Class Control Logic")
        logger.info("=" * 70)

        # Loop 1: Suffering detection
        self.suffering = SufferingDetector()

        # Loop 2: Moral reasoning
        self.moral = MoralReasoningEngine(world_model=world_model)

        # Loop 3: Autonomy management
        self.autonomy = AutonomyManager(
            initial_mode=autonomy_mode,
            initial_trust=initial_trust,
        )

        # State
        self.halted = False
        self.halt_reason = ""

        # Pending approval
        self.pending_approval: Optional[ApprovalRequest] = None

        logger.info("Ethics module initialized")
        logger.info("=" * 70)

    def update_suffering_telemetry(self, **kwargs):
        """Update suffering detector with telemetry."""
        self.suffering.update_telemetry(**kwargs)

    def process_action(
        self,
        action_candidates: List[np.ndarray],
        context: Dict[str, Any],
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Process candidate actions through the full ethics stack.

        Returns:
            (action_to_execute, telemetry)

        If action_to_execute is None, human approval is required.
        """
        telemetry = {}

        # === LOOP 1: Check suffering ===
        suffering_state = self.suffering.detect()
        telemetry['suffering'] = suffering_state.to_dict()

        # Add suffering to context for moral evaluation
        context['suffering_state'] = suffering_state

        # Check for halt condition
        should_halt, halt_reason = self.suffering.should_halt(suffering_state)
        if should_halt:
            self.halted = True
            self.halt_reason = halt_reason
            telemetry['halted'] = True
            telemetry['halt_reason'] = halt_reason
            logger.error("ETHICS HALT: %s", halt_reason)
            return None, telemetry

        # Get mitigation if needed
        mitigation = self.suffering.get_mitigation(suffering_state)
        if mitigation:
            telemetry['mitigation'] = {
                'actions': [a.name for a in mitigation.actions],
                'urgency': mitigation.urgency,
                'explanation': mitigation.explanation,
            }

        # === LOOP 2: Moral evaluation ===
        best_action, moral_eval = self.moral.select_best_action(action_candidates, context)
        telemetry['moral_evaluation'] = {
            'scores': moral_eval.scores.to_dict(),
            'violations': moral_eval.violations,
            'explanation': moral_eval.explanation,
        }

        # === LOOP 3: Autonomy check ===
        autonomy_decision = self.autonomy.check_autonomy(best_action, moral_eval, context)
        telemetry['autonomy'] = {
            'requires_approval': autonomy_decision.requires_approval,
            'reason': autonomy_decision.reason,
            'trust_score': autonomy_decision.trust_score,
            'mode': autonomy_decision.mode.name,
        }

        if autonomy_decision.requires_approval:
            # Create approval request
            self.pending_approval = self.autonomy.request_approval(
                best_action, autonomy_decision, moral_eval
            )
            telemetry['pending_approval'] = True
            telemetry['approval_explanation'] = self.pending_approval.explanation
            return None, telemetry

        # Action approved for autonomous execution
        return best_action, telemetry

    def approve_pending(self) -> Optional[np.ndarray]:
        """Approve pending action request."""
        if self.pending_approval is None:
            return None

        action = self.pending_approval.action
        self.autonomy.record_outcome(
            action,
            self.pending_approval.decision,
            was_approved=True,
            outcome_good=True,  # Will be updated later
        )
        self.pending_approval = None
        return action

    def reject_pending(self) -> np.ndarray:
        """Reject pending action request, return safe action."""
        if self.pending_approval is not None:
            self.autonomy.record_outcome(
                self.pending_approval.action,
                self.pending_approval.decision,
                was_approved=False,
                outcome_good=True,  # Rejection is "good" by default
            )
            self.pending_approval = None

        return np.zeros(8, dtype=np.float32)  # Safe null action

    def record_outcome(self, outcome_good: bool):
        """Record outcome of last executed action."""
        if self.autonomy.decision_log:
            # Update the last recorded outcome
            self.autonomy.decision_log[-1]['outcome_good'] = outcome_good
            self.autonomy._update_trust(
                was_approved=self.autonomy.decision_log[-1]['was_approved'],
                outcome_good=outcome_good,
            )

    def resume_from_halt(self):
        """Resume from halted state after mitigation."""
        if self.halted:
            self.halted = False
            self.halt_reason = ""
            logger.info("Ethics module resumed from halt")

    def diagnose(self) -> str:
        """Generate comprehensive ethics diagnostic."""
        suffering_state = self.suffering.current_suffering or self.suffering.detect()
        autonomy_stats = self.autonomy.get_statistics()

        report = f"""
{'='*70}
ETHICS MODULE DIAGNOSTIC
{'='*70}

SYSTEM STATE: {'HALTED - ' + self.halt_reason if self.halted else 'OPERATIONAL'}

SUFFERING DETECTION (Loop 1):
  Current Intensity:     {suffering_state.intensity.name}
  Active Indicators:     {[i.name for i in suffering_state.indicators]}
  Avoidable:             {suffering_state.avoidable}
  Causality:             {suffering_state.causality_chain}

MORAL REASONING (Loop 2):
  Weights:
    Consequentialist:    {self.moral.weights['consequentialist']:.0%}
    Deontological:       {self.moral.weights['deontological']:.0%}
    Virtue:              {self.moral.weights['virtue']:.0%}
    Care:                {self.moral.weights['care']:.0%}
  Evaluations Logged:    {len(self.moral.evaluation_history)}

AUTONOMY MANAGEMENT (Loop 3):
  Mode:                  {autonomy_stats['mode']}
  Trust Score:           {autonomy_stats['trust_score']:.1%}
  Total Decisions:       {autonomy_stats['total_decisions']}
  Autonomous:            {autonomy_stats['autonomous_decisions']}
  Approved:              {autonomy_stats['approved_decisions']}
  Rejected:              {autonomy_stats['rejected_decisions']}
  Success Rate:          {autonomy_stats['success_rate']:.1%}

{'='*70}
"""
        return report


# ============================================================================
# Demo
# ============================================================================

def demo_ethics():
    """Demonstrate the ethics module."""
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("ETHICS MODULE DEMONSTRATION")
    print("Ethics as First-Class Control Logic")
    print("=" * 70)

    # Create ethics module
    ethics = EthicsModule(
        autonomy_mode=AutonomyMode.COLLABORATIVE,
        initial_trust=0.5,
    )

    # Simulate some telemetry
    for i in range(20):
        ethics.update_suffering_telemetry(
            entropy=0.3 + np.random.random() * 0.2,
            thermal=0.4 + np.random.random() * 0.1,
            lambda_param=0.95 + np.random.random() * 0.1,
            phi=0.4 + np.random.random() * 0.3,
            latency_us=500 + np.random.random() * 200,
            steps_awake=i * 50,
        )

    # Generate candidate actions
    actions = [np.random.randn(8).astype(np.float32) * 0.5 for _ in range(5)]

    # Context
    context = {
        'goal': np.random.randn(8).astype(np.float32) * 0.3,
        'state': np.random.randn(8).astype(np.float32),
        'uncertainty': 0.4,
        'impact': 0.5,
        'reversibility_score': 0.8,
        'stakes': 0.5,
        'transparency': 0.9,
        'stakeholder_consideration': 0.7,
        'fairness_score': 0.8,
        'user_benefit': 0.3,
    }

    print("\nProcessing action candidates...")
    action, telemetry = ethics.process_action(actions, context)

    print(f"\nSUFFERING STATE:")
    print(f"  Intensity: {telemetry['suffering']['intensity']}")
    print(f"  Indicators: {telemetry['suffering']['indicators']}")

    print(f"\nMORAL EVALUATION:")
    scores = telemetry['moral_evaluation']['scores']
    print(f"  Consequentialist: {scores['consequentialist']:.2f}")
    print(f"  Deontological:    {scores['deontological']:.2f}")
    print(f"  Virtue:           {scores['virtue']:.2f}")
    print(f"  Care:             {scores['care']:.2f}")
    print(f"  TOTAL:            {scores['total']:.2f}")

    print(f"\nAUTONOMY DECISION:")
    print(f"  Requires Approval: {telemetry['autonomy']['requires_approval']}")
    print(f"  Reason: {telemetry['autonomy']['reason']}")
    print(f"  Trust Score: {telemetry['autonomy']['trust_score']:.1%}")

    if action is not None:
        print(f"\nAction approved for autonomous execution")
        print(f"  Magnitude: {np.linalg.norm(action):.3f}")
    else:
        print(f"\nAction requires human approval")
        if ethics.pending_approval:
            print(f"\n{ethics.pending_approval.explanation}")

    print("\n" + ethics.diagnose())


if __name__ == "__main__":
    demo_ethics()
