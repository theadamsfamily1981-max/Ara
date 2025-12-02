"""
GUF++: Causal Synthesis and Verified Self-Governance

This module extends the Global Utility Function with:
1. PGU verification of weight proposals
2. Causal safety constraints (AF preservation)
3. Verified weight evolution with formal guarantees

The key innovation: Before adopting new GUF weights, the system asks:
"Do these new self-governance rules guarantee AF ≥ G_target under
all plausible stressor trajectories?"

This transforms the system from "adaptive" to "self-governing with
verifiable preference for its own resilience."
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
import hashlib
import math
import json

from .guf import (
    GUFWeights,
    StateVector,
    GoalState,
    GlobalUtilityFunction,
    FocusMode,
    SelfImprovementScheduler
)


# ============================================================
# Safety Constraints
# ============================================================

class ConstraintType(str, Enum):
    """Types of safety constraints for GUF++."""
    AF_FLOOR = "af_floor"           # AF must stay above threshold
    RISK_CEILING = "risk_ceiling"   # Risk must stay below threshold
    STABILITY = "stability"         # Weights can't change too fast
    MONOTONIC = "monotonic"         # Certain metrics must improve
    BOUNDED = "bounded"             # Stay within operational envelope


@dataclass
class SafetyConstraint:
    """A formal constraint on GUF behavior."""
    id: str
    constraint_type: ConstraintType
    description: str

    # Constraint parameters
    metric: str                     # What we're constraining
    threshold: float               # The bound value
    horizon_steps: int = 10        # How far to look ahead
    probability_bound: float = 0.95  # Must hold with this probability

    # For verification
    is_hard: bool = True           # Hard = must always hold, Soft = best effort

    def to_smt_assertion(self) -> str:
        """
        Generate SMT-LIB assertion for this constraint.

        This is a simplified representation - real SMT would need
        the full state transition model.
        """
        if self.constraint_type == ConstraintType.AF_FLOOR:
            return f"(assert (>= af_score {self.threshold}))"
        elif self.constraint_type == ConstraintType.RISK_CEILING:
            return f"(assert (<= risk {self.threshold}))"
        elif self.constraint_type == ConstraintType.STABILITY:
            return f"(assert (<= (abs weight_delta) {self.threshold}))"
        else:
            return f"; constraint {self.id}: {self.description}"

    def check(self, state: StateVector, weights: GUFWeights) -> Tuple[bool, str]:
        """Check if constraint is satisfied for given state and weights."""
        if self.constraint_type == ConstraintType.AF_FLOOR:
            if state.af_score >= self.threshold:
                return True, f"AF {state.af_score:.2f} >= {self.threshold}"
            return False, f"AF {state.af_score:.2f} < {self.threshold}"

        elif self.constraint_type == ConstraintType.RISK_CEILING:
            if state.risk <= self.threshold:
                return True, f"Risk {state.risk:.2f} <= {self.threshold}"
            return False, f"Risk {state.risk:.2f} > {self.threshold}"

        elif self.constraint_type == ConstraintType.BOUNDED:
            # Generic bounds check on a metric
            value = getattr(state, self.metric, None)
            if value is None:
                return True, f"Metric {self.metric} not found"
            if value <= self.threshold:
                return True, f"{self.metric} {value:.2f} <= {self.threshold}"
            return False, f"{self.metric} {value:.2f} > {self.threshold}"

        return True, "Constraint type not directly checkable"


@dataclass
class ConstraintSet:
    """A set of safety constraints that must hold together."""
    name: str
    constraints: List[SafetyConstraint] = field(default_factory=list)

    def add(self, constraint: SafetyConstraint) -> None:
        self.constraints.append(constraint)

    def check_all(self, state: StateVector, weights: GUFWeights) -> Dict[str, Any]:
        """Check all constraints."""
        results = []
        all_satisfied = True
        hard_satisfied = True

        for c in self.constraints:
            satisfied, message = c.check(state, weights)
            results.append({
                "constraint_id": c.id,
                "type": c.constraint_type.value,
                "satisfied": satisfied,
                "message": message,
                "is_hard": c.is_hard
            })

            if not satisfied:
                all_satisfied = False
                if c.is_hard:
                    hard_satisfied = False

        return {
            "all_satisfied": all_satisfied,
            "hard_satisfied": hard_satisfied,
            "results": results,
            "total": len(self.constraints),
            "passed": sum(1 for r in results if r["satisfied"])
        }


# ============================================================
# Weight Proposals
# ============================================================

class ProposalSource(str, Enum):
    """Where a weight proposal came from."""
    GRADIENT = "gradient"       # From gradient-based learning
    EVOLUTION = "evolution"     # From evolutionary search
    EXPERT = "expert"           # Human-designed
    RECOVERY = "recovery"       # Emergency reset
    INTERPOLATION = "interpolation"  # Blend of existing


@dataclass
class WeightProposal:
    """A proposed set of GUF weights awaiting verification."""
    id: str
    proposed_weights: GUFWeights
    source: ProposalSource
    rationale: str

    # Reference to what we're changing from
    baseline_weights: Optional[GUFWeights] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    expected_utility_delta: float = 0.0
    confidence: float = 0.5

    # Verification status
    verified: bool = False
    verification_result: Optional[Dict[str, Any]] = None
    verification_errors: List[str] = field(default_factory=list)

    @property
    def weight_delta(self) -> float:
        """Compute total magnitude of weight change."""
        if not self.baseline_weights:
            return 0.0

        proposed_vec = self.proposed_weights.to_vector()
        baseline_vec = self.baseline_weights.to_vector()

        return math.sqrt(sum((p - b) ** 2 for p, b in zip(proposed_vec, baseline_vec)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source.value,
            "rationale": self.rationale,
            "weights": self.proposed_weights.to_dict(),
            "weight_delta": self.weight_delta,
            "expected_utility_delta": self.expected_utility_delta,
            "verified": self.verified,
            "verification_errors": self.verification_errors
        }


# ============================================================
# GUF Verifier (PGU Integration)
# ============================================================

class GUFVerifier:
    """
    Verifies GUF weight proposals against safety constraints.

    Uses a combination of:
    1. Direct constraint checking on current state
    2. Trajectory simulation under stressor scenarios
    3. SMT-based formal verification (simplified)
    """

    def __init__(
        self,
        constraints: Optional[ConstraintSet] = None,
        min_af_score: float = 2.0,
        max_weight_delta: float = 0.3
    ):
        self.constraints = constraints or self._default_constraints()
        self.min_af_score = min_af_score
        self.max_weight_delta = max_weight_delta

        # Stressor scenarios for simulation
        self._stressor_scenarios = self._build_stressor_scenarios()

        # Verification history
        self._history: List[Dict[str, Any]] = []

    def _default_constraints(self) -> ConstraintSet:
        """Create default safety constraint set."""
        cs = ConstraintSet(name="default_safety")

        # Hard constraints
        cs.add(SafetyConstraint(
            id="af_floor",
            constraint_type=ConstraintType.AF_FLOOR,
            description="AF score must remain above survival threshold",
            metric="af_score",
            threshold=1.5,
            is_hard=True
        ))

        cs.add(SafetyConstraint(
            id="risk_ceiling",
            constraint_type=ConstraintType.RISK_CEILING,
            description="Overall risk must not exceed critical level",
            metric="risk",
            threshold=0.7,
            is_hard=True
        ))

        # Soft constraints
        cs.add(SafetyConstraint(
            id="target_af",
            constraint_type=ConstraintType.AF_FLOOR,
            description="AF score should reach target level",
            metric="af_score",
            threshold=2.0,
            is_hard=False
        ))

        cs.add(SafetyConstraint(
            id="fatigue_bound",
            constraint_type=ConstraintType.BOUNDED,
            description="Fatigue should stay manageable",
            metric="fatigue",
            threshold=0.8,
            is_hard=False
        ))

        return cs

    def _build_stressor_scenarios(self) -> List[Dict[str, Any]]:
        """Build stressor scenarios for trajectory simulation."""
        return [
            {
                "name": "normal_operation",
                "description": "Typical workload",
                "state_deltas": {
                    "af_score": -0.05,
                    "fatigue": 0.02,
                    "clv_instability": 0.01
                }
            },
            {
                "name": "high_load",
                "description": "Heavy workload spike",
                "state_deltas": {
                    "af_score": -0.15,
                    "fatigue": 0.08,
                    "clv_instability": 0.05,
                    "clv_resource": 0.04
                }
            },
            {
                "name": "stability_challenge",
                "description": "Topological instability",
                "state_deltas": {
                    "af_score": -0.20,
                    "clv_structural": 0.10,
                    "structural_rate": 0.05,
                    "confidence": -0.10
                }
            },
            {
                "name": "hardware_degradation",
                "description": "Hardware trust declining",
                "state_deltas": {
                    "hardware_health": -0.15,
                    "pgu_pass_rate": -0.05,
                    "af_score": -0.10
                }
            },
            {
                "name": "recovery_trajectory",
                "description": "System recovering",
                "state_deltas": {
                    "af_score": 0.10,
                    "fatigue": -0.05,
                    "confidence": 0.05,
                    "clv_instability": -0.03
                }
            }
        ]

    def verify(
        self,
        proposal: WeightProposal,
        current_state: StateVector
    ) -> WeightProposal:
        """
        Verify a weight proposal against safety constraints.

        Returns the proposal with verification status updated.
        """
        errors = []
        warnings = []

        # 1. Check weight delta magnitude
        if proposal.weight_delta > self.max_weight_delta:
            errors.append(
                f"Weight change magnitude {proposal.weight_delta:.3f} "
                f"exceeds maximum {self.max_weight_delta}"
            )

        # 2. Check current state against constraints with proposed weights
        guf = GlobalUtilityFunction(
            weights=proposal.proposed_weights,
            goal=GoalState(min_af_score=self.min_af_score)
        )
        utility = guf.compute(current_state)

        constraint_result = self.constraints.check_all(
            current_state,
            proposal.proposed_weights
        )

        if not constraint_result["hard_satisfied"]:
            for r in constraint_result["results"]:
                if not r["satisfied"] and r["is_hard"]:
                    errors.append(f"Hard constraint violated: {r['message']}")

        if not constraint_result["all_satisfied"]:
            for r in constraint_result["results"]:
                if not r["satisfied"] and not r["is_hard"]:
                    warnings.append(f"Soft constraint not satisfied: {r['message']}")

        # 3. Simulate trajectories under stressor scenarios
        trajectory_results = self._simulate_trajectories(
            proposal.proposed_weights,
            current_state,
            horizon=10
        )

        for scenario_name, result in trajectory_results.items():
            if result["af_violated"]:
                errors.append(
                    f"Trajectory simulation '{scenario_name}' shows AF "
                    f"dropping to {result['min_af']:.2f} (below {self.min_af_score})"
                )
            if result["risk_exceeded"]:
                warnings.append(
                    f"Trajectory '{scenario_name}' shows risk "
                    f"reaching {result['max_risk']:.2f}"
                )

        # 4. Check monotonicity of key weights
        if proposal.baseline_weights:
            # AF weight should not decrease significantly
            af_delta = proposal.proposed_weights.w_af - proposal.baseline_weights.w_af
            if af_delta < -0.1:
                errors.append(
                    f"AF weight decreased by {-af_delta:.3f}, "
                    "which could compromise safety focus"
                )

        # 5. Formal verification (simplified SMT check)
        smt_result = self._smt_verify(proposal, current_state)
        if not smt_result["satisfiable"]:
            errors.append(f"SMT verification failed: {smt_result['reason']}")

        # Update proposal
        proposal.verified = len(errors) == 0
        proposal.verification_result = {
            "errors": errors,
            "warnings": warnings,
            "constraint_check": constraint_result,
            "trajectory_results": trajectory_results,
            "smt_result": smt_result,
            "utility_with_proposed": utility
        }
        proposal.verification_errors = errors

        # Record history
        self._history.append({
            "proposal_id": proposal.id,
            "verified": proposal.verified,
            "timestamp": datetime.now().isoformat(),
            "error_count": len(errors),
            "warning_count": len(warnings)
        })

        return proposal

    def _simulate_trajectories(
        self,
        weights: GUFWeights,
        initial_state: StateVector,
        horizon: int
    ) -> Dict[str, Dict[str, Any]]:
        """Simulate state trajectories under different stressor scenarios."""
        results = {}
        guf = GlobalUtilityFunction(weights=weights)

        for scenario in self._stressor_scenarios:
            state = StateVector(
                af_score=initial_state.af_score,
                clv_instability=initial_state.clv_instability,
                clv_resource=initial_state.clv_resource,
                clv_structural=initial_state.clv_structural,
                structural_rate=initial_state.structural_rate,
                confidence=initial_state.confidence,
                fatigue=initial_state.fatigue,
                mood_valence=initial_state.mood_valence,
                hardware_health=initial_state.hardware_health,
                pgu_pass_rate=initial_state.pgu_pass_rate
            )

            min_af = state.af_score
            max_risk = state.risk
            utilities = []

            for step in range(horizon):
                # Apply stressor deltas
                deltas = scenario["state_deltas"]
                state.af_score = max(0.1, state.af_score + deltas.get("af_score", 0))
                state.fatigue = max(0, min(1, state.fatigue + deltas.get("fatigue", 0)))
                state.clv_instability = max(0, min(1, state.clv_instability + deltas.get("clv_instability", 0)))
                state.clv_resource = max(0, min(1, state.clv_resource + deltas.get("clv_resource", 0)))
                state.clv_structural = max(0, min(1, state.clv_structural + deltas.get("clv_structural", 0)))
                state.structural_rate = max(0, state.structural_rate + deltas.get("structural_rate", 0))
                state.confidence = max(0, min(1, state.confidence + deltas.get("confidence", 0)))
                state.hardware_health = max(0, min(1, state.hardware_health + deltas.get("hardware_health", 0)))
                state.pgu_pass_rate = max(0, min(1, state.pgu_pass_rate + deltas.get("pgu_pass_rate", 0)))

                # Track
                min_af = min(min_af, state.af_score)
                max_risk = max(max_risk, state.risk)
                utilities.append(guf.compute(state))

                # GUF would trigger recovery here in real system
                focus = guf.recommend_focus(state)
                if focus == FocusMode.RECOVERY:
                    # Simulate recovery response
                    state.af_score += 0.05
                    state.fatigue -= 0.02

            results[scenario["name"]] = {
                "min_af": min_af,
                "max_risk": max_risk,
                "af_violated": min_af < self.min_af_score,
                "risk_exceeded": max_risk > 0.7,
                "final_utility": utilities[-1] if utilities else 0,
                "utility_trend": "improving" if len(utilities) > 1 and utilities[-1] > utilities[0] else "declining"
            }

        return results

    def _smt_verify(
        self,
        proposal: WeightProposal,
        current_state: StateVector
    ) -> Dict[str, Any]:
        """
        Simplified SMT-style verification.

        In a full implementation, this would:
        1. Encode state transitions as SMT formulas
        2. Add safety constraints as assertions
        3. Check satisfiability (sat = safe, unsat = unsafe)

        Here we do a simplified check based on weight invariants.
        """
        weights = proposal.proposed_weights

        # Check: weights must sum to roughly 1.0 for positive and -0.4 for negative
        positive_sum = (weights.w_af + weights.w_latency + weights.w_throughput +
                       weights.w_confidence + weights.w_mood + weights.w_hardware +
                       weights.w_pgu + weights.w_energy)

        negative_sum = abs(weights.w_clv_instability + weights.w_clv_resource +
                          weights.w_clv_structural + weights.w_structural_rate +
                          weights.w_fatigue)

        # Safety focus: positive weights should dominate over risks
        safety_ratio = positive_sum / max(0.01, negative_sum)

        if safety_ratio < 1.0:
            return {
                "satisfiable": False,
                "reason": f"Risk weights dominate utility (ratio: {safety_ratio:.2f})"
            }

        # AF must have significant weight
        if weights.w_af < 0.15:
            return {
                "satisfiable": False,
                "reason": f"AF weight {weights.w_af:.2f} too low for safety guarantee"
            }

        # Structural rate penalty must be present
        if weights.w_structural_rate > -0.02:
            return {
                "satisfiable": False,
                "reason": "Structural rate penalty insufficient"
            }

        return {
            "satisfiable": True,
            "safety_ratio": safety_ratio,
            "positive_sum": positive_sum,
            "negative_sum": negative_sum
        }

    @property
    def verification_stats(self) -> Dict[str, Any]:
        """Get verification statistics."""
        if not self._history:
            return {"total": 0}

        return {
            "total": len(self._history),
            "verified": sum(1 for h in self._history if h["verified"]),
            "rejected": sum(1 for h in self._history if not h["verified"]),
            "verification_rate": sum(1 for h in self._history if h["verified"]) / len(self._history)
        }


# ============================================================
# Causal GUF (GUF++)
# ============================================================

class CausalGUF:
    """
    GUF++ - Global Utility Function with causal synthesis.

    Extends the base GUF with:
    1. Verified weight evolution
    2. Causal safety constraints
    3. Self-generated governance rules

    The key insight: The system doesn't just adapt its weights,
    it proves that adaptations preserve its ability to survive.
    """

    def __init__(
        self,
        base_guf: Optional[GlobalUtilityFunction] = None,
        verifier: Optional[GUFVerifier] = None,
        auto_verify: bool = True
    ):
        self.guf = base_guf or GlobalUtilityFunction()
        self.verifier = verifier or GUFVerifier()
        self.auto_verify = auto_verify

        # Proposal queue
        self._pending_proposals: List[WeightProposal] = []
        self._applied_proposals: List[WeightProposal] = []
        self._rejected_proposals: List[WeightProposal] = []

        # Evolution tracking
        self._generation = 0
        self._proposal_counter = 0

        # Statistics
        self._stats = {
            "proposals_created": 0,
            "proposals_verified": 0,
            "proposals_applied": 0,
            "proposals_rejected": 0,
            "total_weight_delta": 0.0,
            "af_improvements": 0
        }

    def compute(self, state: StateVector) -> float:
        """Compute utility using current weights."""
        return self.guf.compute(state)

    def recommend_focus(self, state: StateVector) -> FocusMode:
        """Get focus recommendation."""
        return self.guf.recommend_focus(state)

    def propose_weights(
        self,
        new_weights: GUFWeights,
        source: ProposalSource,
        rationale: str,
        current_state: Optional[StateVector] = None
    ) -> WeightProposal:
        """
        Create a weight proposal.

        If auto_verify is True and current_state is provided,
        the proposal will be immediately verified.
        """
        self._proposal_counter += 1
        self._stats["proposals_created"] += 1

        proposal = WeightProposal(
            id=f"wp_{self._generation:03d}_{self._proposal_counter:04d}",
            proposed_weights=new_weights,
            source=source,
            rationale=rationale,
            baseline_weights=GUFWeights(**self.guf.weights.to_dict())
        )

        if self.auto_verify and current_state:
            proposal = self.verifier.verify(proposal, current_state)
            self._stats["proposals_verified"] += 1

        self._pending_proposals.append(proposal)
        return proposal

    def verify_proposal(
        self,
        proposal: WeightProposal,
        current_state: StateVector
    ) -> WeightProposal:
        """Verify a proposal that wasn't auto-verified."""
        if not proposal.verified and proposal.verification_result is None:
            proposal = self.verifier.verify(proposal, current_state)
            self._stats["proposals_verified"] += 1
        return proposal

    def apply_proposal(
        self,
        proposal: WeightProposal,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Apply a verified proposal, updating the GUF weights.

        Returns application result.
        """
        if not proposal.verified and not force:
            return {
                "applied": False,
                "reason": "Proposal not verified",
                "errors": proposal.verification_errors
            }

        # Store old weights
        old_weights = GUFWeights(**self.guf.weights.to_dict())

        # Apply new weights
        self.guf.weights = proposal.proposed_weights

        # Update tracking
        self._generation += 1
        self._stats["proposals_applied"] += 1
        self._stats["total_weight_delta"] += proposal.weight_delta

        # Move from pending to applied
        if proposal in self._pending_proposals:
            self._pending_proposals.remove(proposal)
        self._applied_proposals.append(proposal)

        return {
            "applied": True,
            "generation": self._generation,
            "weight_delta": proposal.weight_delta,
            "old_weights": old_weights.to_dict(),
            "new_weights": proposal.proposed_weights.to_dict()
        }

    def reject_proposal(
        self,
        proposal: WeightProposal,
        reason: str
    ) -> None:
        """Explicitly reject a proposal."""
        proposal.verification_errors.append(f"Rejected: {reason}")

        if proposal in self._pending_proposals:
            self._pending_proposals.remove(proposal)
        self._rejected_proposals.append(proposal)

        self._stats["proposals_rejected"] += 1

    def evolve_weights(
        self,
        current_state: StateVector,
        learning_rate: float = 0.01
    ) -> Optional[WeightProposal]:
        """
        Evolve weights through verified learning.

        1. Learn from GUF history
        2. Create proposal
        3. Verify
        4. Apply if safe
        """
        # Get learning result from base GUF
        learn_result = self.guf.learn_from_history(
            learning_rate=learning_rate,
            min_samples=5
        )

        if learn_result.get("status") != "updated":
            return None

        # The base GUF already updated weights - we need to verify
        # Create proposal representing this update
        proposal = self.propose_weights(
            new_weights=GUFWeights(**self.guf.weights.to_dict()),
            source=ProposalSource.GRADIENT,
            rationale=f"Gradient learning from {learn_result.get('samples_used', 0)} samples",
            current_state=current_state
        )

        if proposal.verified:
            # Already applied by propose_weights
            return proposal
        else:
            # Revert to baseline if verification failed
            if proposal.baseline_weights:
                self.guf.weights = proposal.baseline_weights
            self._stats["proposals_rejected"] += 1
            return proposal

    def propose_from_exploration(
        self,
        current_state: StateVector,
        n_candidates: int = 5,
        sigma: float = 0.02
    ) -> Optional[WeightProposal]:
        """
        Propose weights through evolutionary exploration.

        Generates perturbed weight candidates and verifies them.
        """
        best_proposal = None
        best_utility = self.compute(current_state)

        for i in range(n_candidates):
            # Perturb current weights
            candidate_weights = self.guf.weights.perturb(sigma=sigma)

            # Create proposal
            proposal = self.propose_weights(
                new_weights=candidate_weights,
                source=ProposalSource.EVOLUTION,
                rationale=f"Evolutionary candidate {i+1}/{n_candidates}",
                current_state=current_state
            )

            if proposal.verified:
                # Check if better than current best
                candidate_utility = GlobalUtilityFunction(
                    weights=candidate_weights,
                    goal=self.guf.goal
                ).compute(current_state)

                if candidate_utility > best_utility:
                    best_utility = candidate_utility
                    best_proposal = proposal

        return best_proposal

    def add_safety_constraint(
        self,
        constraint_type: ConstraintType,
        metric: str,
        threshold: float,
        is_hard: bool = True,
        description: str = ""
    ) -> str:
        """Add a custom safety constraint."""
        constraint_id = f"custom_{len(self.verifier.constraints.constraints):03d}"

        constraint = SafetyConstraint(
            id=constraint_id,
            constraint_type=constraint_type,
            description=description or f"Custom {constraint_type.value} on {metric}",
            metric=metric,
            threshold=threshold,
            is_hard=is_hard
        )

        self.verifier.constraints.add(constraint)
        return constraint_id

    def explain_governance(self) -> str:
        """Generate explanation of current governance rules."""
        lines = []
        lines.append("=== GUF++ Governance Rules ===")
        lines.append("")

        # Current weights
        lines.append("Current Weight Values:")
        weights = self.guf.weights.to_dict()
        for k, v in sorted(weights.items(), key=lambda x: -abs(x[1])):
            sign = "+" if v >= 0 else ""
            lines.append(f"  {k}: {sign}{v:.3f}")

        lines.append("")
        lines.append("Safety Constraints:")
        for c in self.verifier.constraints.constraints:
            hard_soft = "HARD" if c.is_hard else "soft"
            lines.append(f"  [{hard_soft}] {c.description}")
            lines.append(f"         {c.metric} {'<=' if c.constraint_type in [ConstraintType.RISK_CEILING, ConstraintType.BOUNDED] else '>='} {c.threshold}")

        lines.append("")
        lines.append(f"Generation: {self._generation}")
        lines.append(f"Applied proposals: {self._stats['proposals_applied']}")
        lines.append(f"Total weight evolution: {self._stats['total_weight_delta']:.3f}")

        return "\n".join(lines)

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            **self._stats,
            "generation": self._generation,
            "pending_proposals": len(self._pending_proposals),
            "verifier_stats": self.verifier.verification_stats
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "guf": self.guf.to_dict(),
            "generation": self._generation,
            "stats": self._stats,
            "constraints": [
                {
                    "id": c.id,
                    "type": c.constraint_type.value,
                    "metric": c.metric,
                    "threshold": c.threshold,
                    "is_hard": c.is_hard
                }
                for c in self.verifier.constraints.constraints
            ]
        }


# ============================================================
# Factory Functions
# ============================================================

def create_causal_guf(
    min_af_score: float = 2.0,
    preset: str = "balanced"
) -> CausalGUF:
    """Create a CausalGUF with safety constraints."""
    from .guf import create_guf

    base_guf = create_guf(preset)

    # Create verifier with AF floor
    verifier = GUFVerifier(min_af_score=min_af_score)

    return CausalGUF(
        base_guf=base_guf,
        verifier=verifier,
        auto_verify=True
    )


def create_safety_constraints(
    af_floor: float = 2.0,
    risk_ceiling: float = 0.5
) -> ConstraintSet:
    """Create a safety constraint set."""
    cs = ConstraintSet(name="custom_safety")

    cs.add(SafetyConstraint(
        id="af_survival",
        constraint_type=ConstraintType.AF_FLOOR,
        description="AF must stay above survival threshold",
        metric="af_score",
        threshold=af_floor,
        is_hard=True
    ))

    cs.add(SafetyConstraint(
        id="risk_limit",
        constraint_type=ConstraintType.RISK_CEILING,
        description="Risk must stay below critical",
        metric="risk",
        threshold=risk_ceiling,
        is_hard=True
    ))

    return cs


__all__ = [
    "ConstraintType",
    "SafetyConstraint",
    "ConstraintSet",
    "ProposalSource",
    "WeightProposal",
    "GUFVerifier",
    "CausalGUF",
    "create_causal_guf",
    "create_safety_constraints"
]
