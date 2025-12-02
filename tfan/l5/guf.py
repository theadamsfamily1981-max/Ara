"""
Global Utility Function (GUF): Deep Self-Modeling

This module implements the system's learned values - a parameterized utility
function that the system uses to evaluate its own state and decide when to
prioritize self-improvement vs external productivity.

The GUF represents "what the system cares about" in a learnable form:
- Weights on CLV components (instability, resource, structural)
- Weights on performance metrics (AF score, latency, throughput)
- A goal state G that defines "good enough to serve the world"

Key insight: The system learns its own values through experience,
discovering which internal signals best predict good vs bad futures.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import json
import math


class FocusMode(str, Enum):
    """Where the system should focus its resources."""
    INTERNAL = "internal"      # Self-improvement, recalibration
    BALANCED = "balanced"      # Normal operation
    EXTERNAL = "external"      # Maximum external throughput
    RECOVERY = "recovery"      # Emergency self-repair


@dataclass
class StateVector:
    """
    Complete state representation for GUF evaluation.

    This is the "observation" the system uses to compute its utility.
    """
    # Core metrics
    af_score: float = 1.0           # Antifragility score (target: > 2.0)
    delta_p99: float = 0.0          # Latency improvement %
    throughput: float = 1.0         # Normalized throughput (0-1)

    # CLV components
    clv_instability: float = 0.0    # 0-1
    clv_resource: float = 0.0       # 0-1
    clv_structural: float = 0.0     # 0-1
    structural_rate: float = 0.0    # Ṡ from L7

    # Self-state
    confidence: float = 0.8         # Belief in world model
    fatigue: float = 0.0            # Accumulated load
    mood_valence: float = 0.0       # From PAD

    # Hardware
    hardware_health: float = 1.0    # Average hardware trust
    pgu_pass_rate: float = 1.0      # Recent PGU success rate

    # Energy
    energy_efficiency: float = 1.0  # Normalized (0-1)

    def to_vector(self) -> List[float]:
        """Convert to flat vector for computation."""
        return [
            self.af_score,
            self.delta_p99 / 100.0,  # Normalize to 0-1 range
            self.throughput,
            self.clv_instability,
            self.clv_resource,
            self.clv_structural,
            self.structural_rate,
            self.confidence,
            1.0 - self.fatigue,  # Invert so higher is better
            (self.mood_valence + 1.0) / 2.0,  # Map -1,1 to 0,1
            self.hardware_health,
            self.pgu_pass_rate,
            self.energy_efficiency
        ]

    @property
    def risk(self) -> float:
        """Compute overall risk from CLV."""
        return 0.5 * self.clv_instability + 0.3 * self.clv_resource + 0.2 * self.clv_structural


@dataclass
class GUFWeights:
    """
    Learnable weights for the Global Utility Function.

    These represent the system's "values" - what it cares about.
    L5 meta-learning can adjust these within constrained bounds.
    """
    # Performance weights (positive = good)
    w_af: float = 0.30              # Antifragility score
    w_latency: float = 0.15         # Latency improvement
    w_throughput: float = 0.15      # Throughput

    # Risk weights (negative = bad)
    w_clv_instability: float = -0.15
    w_clv_resource: float = -0.08
    w_clv_structural: float = -0.07
    w_structural_rate: float = -0.10  # Ṡ penalty

    # Self-state weights
    w_confidence: float = 0.05
    w_fatigue: float = -0.05
    w_mood: float = 0.02

    # Hardware/safety weights
    w_hardware: float = 0.05
    w_pgu: float = 0.05

    # Energy weight
    w_energy: float = 0.03

    # Bounds for L5 learning (min, max)
    BOUNDS: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "w_af": (0.1, 0.5),
        "w_latency": (0.05, 0.3),
        "w_throughput": (0.05, 0.3),
        "w_clv_instability": (-0.3, -0.05),
        "w_clv_resource": (-0.2, -0.02),
        "w_clv_structural": (-0.2, -0.02),
        "w_structural_rate": (-0.25, -0.02),
        "w_confidence": (0.01, 0.15),
        "w_fatigue": (-0.15, -0.01),
        "w_mood": (0.0, 0.1),
        "w_hardware": (0.02, 0.15),
        "w_pgu": (0.02, 0.15),
        "w_energy": (0.01, 0.1)
    })

    def to_vector(self) -> List[float]:
        """Get weights as vector (matching StateVector order)."""
        return [
            self.w_af,
            self.w_latency,
            self.w_throughput,
            self.w_clv_instability,
            self.w_clv_resource,
            self.w_clv_structural,
            self.w_structural_rate,
            self.w_confidence,
            self.w_fatigue,
            self.w_mood,
            self.w_hardware,
            self.w_pgu,
            self.w_energy
        ]

    def update(self, param: str, delta: float) -> bool:
        """
        Update a weight with bounds checking.

        Returns True if update was applied, False if bounded.
        """
        if not hasattr(self, param):
            return False

        current = getattr(self, param)
        bounds = self.BOUNDS.get(param, (-1.0, 1.0))
        new_value = max(bounds[0], min(bounds[1], current + delta))
        setattr(self, param, new_value)
        return new_value != current + delta  # True if we hit a bound

    def perturb(self, sigma: float = 0.01) -> 'GUFWeights':
        """Create a perturbed copy for exploration."""
        import random
        new_weights = GUFWeights()
        for param in ["w_af", "w_latency", "w_throughput", "w_clv_instability",
                      "w_clv_resource", "w_clv_structural", "w_structural_rate",
                      "w_confidence", "w_fatigue", "w_mood", "w_hardware",
                      "w_pgu", "w_energy"]:
            current = getattr(self, param)
            bounds = self.BOUNDS.get(param, (-1.0, 1.0))
            perturbed = current + random.gauss(0, sigma)
            setattr(new_weights, param, max(bounds[0], min(bounds[1], perturbed)))
        return new_weights

    def to_dict(self) -> Dict[str, float]:
        return {
            "w_af": self.w_af,
            "w_latency": self.w_latency,
            "w_throughput": self.w_throughput,
            "w_clv_instability": self.w_clv_instability,
            "w_clv_resource": self.w_clv_resource,
            "w_clv_structural": self.w_clv_structural,
            "w_structural_rate": self.w_structural_rate,
            "w_confidence": self.w_confidence,
            "w_fatigue": self.w_fatigue,
            "w_mood": self.w_mood,
            "w_hardware": self.w_hardware,
            "w_pgu": self.w_pgu,
            "w_energy": self.w_energy
        }


@dataclass
class GoalState:
    """
    The system's goal state G - defines "good enough".

    When the system is above G, it can focus on external tasks.
    When below G, it should prioritize self-improvement.
    """
    # Minimum acceptable AF score
    min_af_score: float = 1.8

    # Maximum acceptable risk
    max_risk: float = 0.4

    # Minimum confidence
    min_confidence: float = 0.6

    # Maximum fatigue before recovery needed
    max_fatigue: float = 0.7

    # Minimum PGU pass rate
    min_pgu_rate: float = 0.9

    # Utility threshold for "good enough"
    utility_threshold: float = 0.6

    def is_satisfied(self, state: StateVector, utility: float) -> bool:
        """Check if current state satisfies goal."""
        return (
            state.af_score >= self.min_af_score and
            state.risk <= self.max_risk and
            state.confidence >= self.min_confidence and
            state.fatigue <= self.max_fatigue and
            state.pgu_pass_rate >= self.min_pgu_rate and
            utility >= self.utility_threshold
        )

    def get_violations(self, state: StateVector, utility: float) -> List[str]:
        """Get list of goal violations."""
        violations = []
        if state.af_score < self.min_af_score:
            violations.append(f"AF score {state.af_score:.2f} < {self.min_af_score}")
        if state.risk > self.max_risk:
            violations.append(f"Risk {state.risk:.2f} > {self.max_risk}")
        if state.confidence < self.min_confidence:
            violations.append(f"Confidence {state.confidence:.2f} < {self.min_confidence}")
        if state.fatigue > self.max_fatigue:
            violations.append(f"Fatigue {state.fatigue:.2f} > {self.max_fatigue}")
        if state.pgu_pass_rate < self.min_pgu_rate:
            violations.append(f"PGU rate {state.pgu_pass_rate:.2f} < {self.min_pgu_rate}")
        if utility < self.utility_threshold:
            violations.append(f"Utility {utility:.2f} < {self.utility_threshold}")
        return violations

    def distance_to_goal(self, state: StateVector, utility: float) -> float:
        """Compute normalized distance to goal state."""
        distances = []

        # Each dimension's distance (0 = satisfied, positive = how far off)
        if state.af_score < self.min_af_score:
            distances.append((self.min_af_score - state.af_score) / self.min_af_score)
        if state.risk > self.max_risk:
            distances.append((state.risk - self.max_risk) / (1.0 - self.max_risk))
        if state.confidence < self.min_confidence:
            distances.append((self.min_confidence - state.confidence) / self.min_confidence)
        if state.fatigue > self.max_fatigue:
            distances.append((state.fatigue - self.max_fatigue) / (1.0 - self.max_fatigue))
        if utility < self.utility_threshold:
            distances.append((self.utility_threshold - utility) / self.utility_threshold)

        return sum(distances) / max(len(distances), 1)


class GlobalUtilityFunction:
    """
    The core GUF that computes the system's utility from its state.

    U(state) = Σ w_i * s_i

    Where w_i are learnable weights and s_i are state components.
    """

    def __init__(
        self,
        weights: Optional[GUFWeights] = None,
        goal: Optional[GoalState] = None
    ):
        self.weights = weights or GUFWeights()
        self.goal = goal or GoalState()

        # History for learning
        self._history: List[Dict[str, Any]] = []
        self._weight_history: List[GUFWeights] = []

    def compute(self, state: StateVector) -> float:
        """
        Compute utility of a state.

        Returns a value roughly in [0, 1] where higher is better.
        """
        state_vec = state.to_vector()
        weight_vec = self.weights.to_vector()

        # Dot product
        raw_utility = sum(s * w for s, w in zip(state_vec, weight_vec))

        # Sigmoid to normalize to (0, 1)
        utility = 1.0 / (1.0 + math.exp(-raw_utility * 3))

        return utility

    def compute_with_breakdown(self, state: StateVector) -> Dict[str, Any]:
        """Compute utility with detailed breakdown."""
        state_vec = state.to_vector()
        weight_vec = self.weights.to_vector()

        contributions = {}
        labels = ["af_score", "latency", "throughput", "clv_instability",
                  "clv_resource", "clv_structural", "structural_rate",
                  "confidence", "fatigue", "mood", "hardware", "pgu", "energy"]

        for i, (s, w, label) in enumerate(zip(state_vec, weight_vec, labels)):
            contributions[label] = {"value": s, "weight": w, "contribution": s * w}

        raw_utility = sum(c["contribution"] for c in contributions.values())
        utility = 1.0 / (1.0 + math.exp(-raw_utility * 3))

        return {
            "utility": utility,
            "raw_utility": raw_utility,
            "contributions": contributions,
            "goal_satisfied": self.goal.is_satisfied(state, utility),
            "goal_violations": self.goal.get_violations(state, utility)
        }

    def recommend_focus(self, state: StateVector) -> FocusMode:
        """
        Recommend where the system should focus based on state and goal.

        This is the key "self vs world" decision.
        """
        utility = self.compute(state)

        # Check for emergency conditions
        if state.af_score < 1.0 or state.risk > 0.8:
            return FocusMode.RECOVERY

        # Check if goal is satisfied
        if self.goal.is_satisfied(state, utility):
            # Good enough - can focus on external tasks
            if utility > 0.8:
                return FocusMode.EXTERNAL
            return FocusMode.BALANCED

        # Goal not satisfied - need self-improvement
        distance = self.goal.distance_to_goal(state, utility)
        if distance > 0.5:
            return FocusMode.RECOVERY  # Far from goal
        return FocusMode.INTERNAL  # Some work needed

    def record_outcome(
        self,
        state_before: StateVector,
        state_after: StateVector,
        action_taken: str,
        external_reward: float = 0.0
    ) -> Dict[str, Any]:
        """
        Record an outcome for learning.

        The system learns which states and actions lead to good futures.
        """
        utility_before = self.compute(state_before)
        utility_after = self.compute(state_after)
        delta_utility = utility_after - utility_before

        record = {
            "timestamp": datetime.now().isoformat(),
            "utility_before": utility_before,
            "utility_after": utility_after,
            "delta_utility": delta_utility,
            "external_reward": external_reward,
            "action": action_taken,
            "state_before": state_before.to_vector(),
            "state_after": state_after.to_vector(),
            "goal_satisfied_before": self.goal.is_satisfied(state_before, utility_before),
            "goal_satisfied_after": self.goal.is_satisfied(state_after, utility_after)
        }

        self._history.append(record)
        return record

    def learn_from_history(
        self,
        learning_rate: float = 0.01,
        min_samples: int = 10
    ) -> Dict[str, Any]:
        """
        Update weights based on recorded outcomes.

        Simple gradient-free learning: adjust weights in the direction
        that correlates with positive outcomes.
        """
        if len(self._history) < min_samples:
            return {"status": "insufficient_data", "samples": len(self._history)}

        # Compute correlation of each state dimension with positive outcomes
        recent = self._history[-min_samples:]

        dimension_effects = {i: 0.0 for i in range(13)}

        for record in recent:
            outcome = record["delta_utility"] + 0.5 * record["external_reward"]
            state_vec = record["state_before"]

            for i, s in enumerate(state_vec):
                dimension_effects[i] += s * outcome

        # Normalize
        for i in dimension_effects:
            dimension_effects[i] /= len(recent)

        # Apply updates
        params = ["w_af", "w_latency", "w_throughput", "w_clv_instability",
                  "w_clv_resource", "w_clv_structural", "w_structural_rate",
                  "w_confidence", "w_fatigue", "w_mood", "w_hardware",
                  "w_pgu", "w_energy"]

        updates = {}
        for i, param in enumerate(params):
            delta = learning_rate * dimension_effects[i]
            bounded = self.weights.update(param, delta)
            updates[param] = {
                "delta": delta,
                "new_value": getattr(self.weights, param),
                "bounded": bounded
            }

        # Save weight snapshot
        self._weight_history.append(GUFWeights(**self.weights.to_dict()))

        return {
            "status": "updated",
            "samples_used": len(recent),
            "updates": updates
        }

    def explain_state(self, state: StateVector) -> str:
        """Generate natural language explanation of current state and utility."""
        breakdown = self.compute_with_breakdown(state)
        utility = breakdown["utility"]
        focus = self.recommend_focus(state)

        lines = []
        lines.append(f"Current utility: {utility:.2%}")
        lines.append(f"Recommended focus: {focus.value}")

        # Top positive contributors
        positive = [(k, v["contribution"]) for k, v in breakdown["contributions"].items()
                    if v["contribution"] > 0]
        positive.sort(key=lambda x: x[1], reverse=True)

        if positive:
            lines.append("Strengths:")
            for k, c in positive[:3]:
                lines.append(f"  - {k}: +{c:.3f}")

        # Top negative contributors
        negative = [(k, v["contribution"]) for k, v in breakdown["contributions"].items()
                    if v["contribution"] < 0]
        negative.sort(key=lambda x: x[1])

        if negative:
            lines.append("Areas needing attention:")
            for k, c in negative[:3]:
                lines.append(f"  - {k}: {c:.3f}")

        # Goal status
        if breakdown["goal_satisfied"]:
            lines.append("Goal state: SATISFIED - can focus on external tasks")
        else:
            lines.append("Goal state: NOT SATISFIED - prioritize self-improvement")
            for v in breakdown["goal_violations"][:3]:
                lines.append(f"  - {v}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weights": self.weights.to_dict(),
            "goal": {
                "min_af_score": self.goal.min_af_score,
                "max_risk": self.goal.max_risk,
                "min_confidence": self.goal.min_confidence,
                "max_fatigue": self.goal.max_fatigue,
                "utility_threshold": self.goal.utility_threshold
            },
            "history_size": len(self._history)
        }


# ============================================================
# Self-Improvement Scheduler
# ============================================================

@dataclass
class SchedulerDecision:
    """A scheduling decision with explanation."""
    focus_mode: FocusMode
    internal_allocation: float  # 0-1, fraction of resources for self
    external_allocation: float  # 0-1, fraction for external tasks
    priority_tasks: List[str]   # What to focus on
    rationale: str
    utility: float
    goal_distance: float


class SelfImprovementScheduler:
    """
    Scheduler that allocates resources between self-improvement and external tasks.

    Uses the GUF to decide when the system should "work on itself" vs
    "serve the world".
    """

    def __init__(
        self,
        guf: Optional[GlobalUtilityFunction] = None,
        min_internal: float = 0.1,  # Always reserve 10% for self
        max_internal: float = 0.8   # Never more than 80% on self
    ):
        self.guf = guf or GlobalUtilityFunction()
        self.min_internal = min_internal
        self.max_internal = max_internal

        # Track self-improvement episodes
        self._episodes: List[Dict[str, Any]] = []

    def decide(self, state: StateVector) -> SchedulerDecision:
        """Make a scheduling decision based on current state."""
        utility = self.guf.compute(state)
        focus = self.guf.recommend_focus(state)
        goal_distance = self.guf.goal.distance_to_goal(state, utility)

        # Compute allocation
        if focus == FocusMode.RECOVERY:
            internal = self.max_internal
            priority = self._get_recovery_priorities(state)
            rationale = "System below safe thresholds - emergency self-repair needed"

        elif focus == FocusMode.INTERNAL:
            internal = 0.5 + 0.3 * goal_distance  # More internal as we're further from goal
            internal = min(internal, self.max_internal)
            priority = self._get_improvement_priorities(state)
            rationale = f"Goal not satisfied (distance: {goal_distance:.2f}) - prioritizing self-improvement"

        elif focus == FocusMode.EXTERNAL:
            internal = self.min_internal
            priority = ["external_throughput", "user_requests"]
            rationale = "Goal satisfied - maximizing external productivity"

        else:  # BALANCED
            internal = 0.3
            priority = self._get_balanced_priorities(state)
            rationale = "Balanced operation - maintaining stability while serving requests"

        return SchedulerDecision(
            focus_mode=focus,
            internal_allocation=internal,
            external_allocation=1.0 - internal,
            priority_tasks=priority,
            rationale=rationale,
            utility=utility,
            goal_distance=goal_distance
        )

    def _get_recovery_priorities(self, state: StateVector) -> List[str]:
        """Determine priorities for recovery mode."""
        priorities = []

        if state.af_score < 1.0:
            priorities.append("restore_antifragility")
        if state.risk > 0.7:
            priorities.append("reduce_clv_risk")
        if state.pgu_pass_rate < 0.8:
            priorities.append("fix_pgu_failures")
        if state.hardware_health < 0.7:
            priorities.append("hardware_diagnostics")
        if state.fatigue > 0.8:
            priorities.append("cooldown_period")

        return priorities or ["general_recovery"]

    def _get_improvement_priorities(self, state: StateVector) -> List[str]:
        """Determine priorities for self-improvement mode."""
        priorities = []

        # Check what's furthest from goal
        if state.af_score < self.guf.goal.min_af_score:
            priorities.append("aepo_optimization")
        if state.confidence < self.guf.goal.min_confidence:
            priorities.append("world_model_update")
        if state.risk > self.guf.goal.max_risk:
            priorities.append("structural_stabilization")
        if state.fatigue > self.guf.goal.max_fatigue:
            priorities.append("load_shedding")

        return priorities or ["general_optimization"]

    def _get_balanced_priorities(self, state: StateVector) -> List[str]:
        """Priorities for balanced mode."""
        return [
            "external_requests",
            "background_optimization",
            "episodic_memory_update"
        ]

    def start_improvement_episode(
        self,
        state: StateVector,
        focus: str
    ) -> str:
        """Start a self-improvement episode."""
        episode_id = f"imp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        utility = self.guf.compute(state)

        self._episodes.append({
            "id": episode_id,
            "started_at": datetime.now().isoformat(),
            "focus": focus,
            "utility_start": utility,
            "state_start": state.to_vector(),
            "ended_at": None,
            "utility_end": None,
            "success": None
        })

        return episode_id

    def end_improvement_episode(
        self,
        episode_id: str,
        state: StateVector,
        success: bool
    ) -> Dict[str, Any]:
        """End a self-improvement episode."""
        utility = self.guf.compute(state)

        for ep in self._episodes:
            if ep["id"] == episode_id:
                ep["ended_at"] = datetime.now().isoformat()
                ep["utility_end"] = utility
                ep["state_end"] = state.to_vector()
                ep["success"] = success
                ep["utility_delta"] = utility - ep["utility_start"]

                # Record for learning
                state_before = StateVector()
                for i, val in enumerate(ep["state_start"]):
                    # Reconstruct (simplified)
                    pass

                return {
                    "episode_id": episode_id,
                    "utility_delta": ep["utility_delta"],
                    "success": success,
                    "duration": ep["ended_at"]
                }

        return {"error": "episode not found"}

    def get_improvement_stats(self) -> Dict[str, Any]:
        """Get statistics on self-improvement episodes."""
        completed = [ep for ep in self._episodes if ep["ended_at"]]

        if not completed:
            return {"total_episodes": 0}

        successful = [ep for ep in completed if ep["success"]]
        avg_delta = sum(ep["utility_delta"] for ep in completed) / len(completed)

        return {
            "total_episodes": len(completed),
            "success_rate": len(successful) / len(completed),
            "avg_utility_delta": avg_delta,
            "total_utility_gained": sum(ep["utility_delta"] for ep in completed if ep["utility_delta"] > 0),
            "recent_episodes": completed[-5:]
        }

    def explain_decision(self, decision: SchedulerDecision) -> str:
        """Generate natural language explanation of scheduling decision."""
        lines = [
            f"Focus mode: {decision.focus_mode.value}",
            f"Resource allocation: {decision.internal_allocation:.0%} internal, {decision.external_allocation:.0%} external",
            f"Rationale: {decision.rationale}",
            f"Priority tasks: {', '.join(decision.priority_tasks)}",
            f"Current utility: {decision.utility:.2%}",
        ]

        if decision.goal_distance > 0:
            lines.append(f"Distance to goal: {decision.goal_distance:.2f}")

        return "\n".join(lines)


# ============================================================
# Convenience Functions
# ============================================================

def create_guf(preset: str = "balanced") -> GlobalUtilityFunction:
    """Create a GUF with preset weights."""
    if preset == "safety_first":
        weights = GUFWeights(
            w_af=0.35,
            w_clv_instability=-0.20,
            w_pgu=0.10
        )
        goal = GoalState(min_af_score=2.0, max_risk=0.3)
    elif preset == "performance":
        weights = GUFWeights(
            w_throughput=0.25,
            w_latency=0.25,
            w_af=0.20
        )
        goal = GoalState(min_af_score=1.5, max_risk=0.5)
    else:  # balanced
        weights = GUFWeights()
        goal = GoalState()

    return GlobalUtilityFunction(weights=weights, goal=goal)


def create_scheduler(
    guf: Optional[GlobalUtilityFunction] = None
) -> SelfImprovementScheduler:
    """Create a self-improvement scheduler."""
    return SelfImprovementScheduler(guf=guf)


def compute_state_from_metrics(
    af_score: float,
    clv: Dict[str, float],
    self_state: Dict[str, float],
    throughput: float = 1.0,
    delta_p99: float = 0.0
) -> StateVector:
    """Helper to create StateVector from typical system metrics."""
    return StateVector(
        af_score=af_score,
        delta_p99=delta_p99,
        throughput=throughput,
        clv_instability=clv.get("instability", 0.0),
        clv_resource=clv.get("resource", 0.0),
        clv_structural=clv.get("structural", 0.0),
        structural_rate=clv.get("structural_dynamics", 0.0),
        confidence=self_state.get("confidence", 0.8),
        fatigue=self_state.get("fatigue", 0.0),
        mood_valence=self_state.get("valence", 0.0),
        hardware_health=self_state.get("hardware_health", 1.0),
        pgu_pass_rate=self_state.get("pgu_pass_rate", 1.0),
        energy_efficiency=self_state.get("energy", 1.0)
    )
