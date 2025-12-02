"""
Cognition Module: Self-Model, Goals, and Deliberation

This module implements the meta-cognitive layer that enables the system
to have "opinions about its own condition and behavior":

1. SelfState: A coherent representation of the system's current state
   - Mood (PAD + CLV)
   - Profile (current L5 personality)
   - Confidence in world model
   - Fatigue from recent load
   - Trust in hardware

2. GoalVector: Explicit values the system optimizes for
   - Stability, latency, energy, exploration weights
   - Learnable by L5, constrained by PGU

3. Deliberation: Multi-step thinking loop
   - Draft → Check → Refine → Act
   - Configurable depth based on risk
   - Thought traces for explainability

Together, these give the system the ability to say:
"I did X because it aligns with my goals, given my current state"
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import json


# ============================================================
# Self-State: Meta-Cognitive Self-Model
# ============================================================

class MoodState(str, Enum):
    """High-level mood classification based on PAD/CLV."""
    CALM = "calm"           # Low arousal, positive valence
    FOCUSED = "focused"     # Medium arousal, neutral valence
    STRESSED = "stressed"   # High arousal, negative valence
    ALERT = "alert"         # High arousal, positive valence
    FATIGUED = "fatigued"   # Low arousal, negative valence


class ConfidenceLevel(str, Enum):
    """Confidence in world model / current state."""
    HIGH = "high"           # > 0.8
    MEDIUM = "medium"       # 0.5 - 0.8
    LOW = "low"             # 0.3 - 0.5
    VERY_LOW = "very_low"   # < 0.3


class TrustLevel(str, Enum):
    """Trust in a hardware component."""
    TRUSTED = "trusted"
    CAUTIOUS = "cautious"
    DISTRUSTED = "distrusted"
    UNKNOWN = "unknown"


@dataclass
class PADState:
    """Pleasure-Arousal-Dominance emotional state."""
    valence: float = 0.0     # -1 (negative) to +1 (positive)
    arousal: float = 0.0     # 0 (calm) to 1 (excited)
    dominance: float = 0.5   # 0 (submissive) to 1 (dominant)

    def to_mood(self) -> MoodState:
        """Map PAD to high-level mood."""
        if self.arousal < 0.3:
            if self.valence > 0:
                return MoodState.CALM
            else:
                return MoodState.FATIGUED
        elif self.arousal > 0.6:
            if self.valence < -0.2:
                return MoodState.STRESSED
            else:
                return MoodState.ALERT
        else:
            return MoodState.FOCUSED


@dataclass
class CLVState:
    """Cognitive Load Vector state."""
    instability: float = 0.0   # 0-1
    resource: float = 0.0      # 0-1
    structural: float = 0.0    # 0-1
    structural_dynamics: float = 0.0  # Ṡ from L7

    @property
    def risk(self) -> float:
        """Compute overall risk."""
        return 0.5 * self.instability + 0.3 * self.resource + 0.2 * self.structural

    @property
    def risk_level(self) -> str:
        """Classify risk level."""
        r = self.risk
        if r < 0.2:
            return "nominal"
        elif r < 0.5:
            return "elevated"
        elif r < 0.8:
            return "high"
        else:
            return "critical"


@dataclass
class HardwareTrust:
    """Trust state for a hardware component."""
    device_id: str
    device_type: str
    trust: TrustLevel = TrustLevel.UNKNOWN
    failure_count: int = 0
    last_failure: Optional[datetime] = None
    successful_runs: int = 0
    last_success: Optional[datetime] = None

    def record_failure(self, reason: str = "") -> None:
        """Record a failure, update trust."""
        self.failure_count += 1
        self.last_failure = datetime.now()
        # Downgrade trust
        if self.trust == TrustLevel.TRUSTED:
            self.trust = TrustLevel.CAUTIOUS
        elif self.trust in [TrustLevel.CAUTIOUS, TrustLevel.UNKNOWN]:
            self.trust = TrustLevel.DISTRUSTED

    def record_success(self) -> None:
        """Record a success, potentially upgrade trust."""
        self.successful_runs += 1
        self.last_success = datetime.now()
        # Upgrade trust slowly
        if self.successful_runs > 10 and self.trust == TrustLevel.UNKNOWN:
            self.trust = TrustLevel.TRUSTED
        elif self.successful_runs > 20 and self.trust == TrustLevel.CAUTIOUS:
            self.trust = TrustLevel.TRUSTED


@dataclass
class SelfState:
    """
    The system's complete self-model.

    This is what the system "knows about itself" at any moment,
    enabling meta-policies that adjust behavior based on self-assessment.
    """
    # Emotional state
    pad: PADState = field(default_factory=PADState)
    clv: CLVState = field(default_factory=CLVState)

    # Current configuration
    profile: str = "balanced_general"  # Current L5 personality profile
    cognitive_phase: str = "hierarchical"  # Current L8 phase

    # Meta-cognitive assessments
    confidence: float = 0.8  # Belief in current world model (0-1)
    fatigue: float = 0.0     # Accumulated load (0-1)

    # Hardware trust
    hardware_trust: Dict[str, HardwareTrust] = field(default_factory=dict)

    # Recent history
    recent_decisions: List[str] = field(default_factory=list)
    active_goals: List[str] = field(default_factory=list)

    # Timestamps
    last_updated: datetime = field(default_factory=datetime.now)
    last_stable: datetime = field(default_factory=datetime.now)

    @property
    def mood(self) -> MoodState:
        """Current mood based on PAD."""
        return self.pad.to_mood()

    @property
    def risk_level(self) -> str:
        """Current risk level based on CLV."""
        return self.clv.risk_level

    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Classify confidence level."""
        if self.confidence > 0.8:
            return ConfidenceLevel.HIGH
        elif self.confidence > 0.5:
            return ConfidenceLevel.MEDIUM
        elif self.confidence > 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    @property
    def needs_caution(self) -> bool:
        """Should the system be cautious?"""
        return (
            self.mood == MoodState.STRESSED or
            self.risk_level in ["high", "critical"] or
            self.confidence_level in [ConfidenceLevel.LOW, ConfidenceLevel.VERY_LOW] or
            self.fatigue > 0.7
        )

    @property
    def can_explore(self) -> bool:
        """Is the system in a state where exploration is safe?"""
        return (
            self.mood in [MoodState.CALM, MoodState.FOCUSED] and
            self.risk_level in ["nominal", "elevated"] and
            self.confidence > 0.6 and
            self.fatigue < 0.5
        )

    def update_pad(self, valence: float, arousal: float, dominance: float) -> None:
        """Update PAD state."""
        self.pad = PADState(valence=valence, arousal=arousal, dominance=dominance)
        self.last_updated = datetime.now()
        if self.pad.to_mood() in [MoodState.CALM, MoodState.FOCUSED]:
            self.last_stable = datetime.now()

    def update_clv(
        self,
        instability: float,
        resource: float,
        structural: float,
        dynamics: float = 0.0
    ) -> None:
        """Update CLV state."""
        self.clv = CLVState(
            instability=instability,
            resource=resource,
            structural=structural,
            structural_dynamics=dynamics
        )
        self.last_updated = datetime.now()

    def update_fatigue(self, load: float, decay_rate: float = 0.1) -> None:
        """Update fatigue based on recent load."""
        # Fatigue increases with load, decays over time
        time_since_update = (datetime.now() - self.last_updated).total_seconds()
        decay = decay_rate * time_since_update / 60.0  # Decay per minute
        self.fatigue = max(0.0, min(1.0, self.fatigue + load * 0.1 - decay))

    def update_confidence(self, delta: float, reason: str = "") -> None:
        """Adjust confidence in world model."""
        self.confidence = max(0.0, min(1.0, self.confidence + delta))
        self.last_updated = datetime.now()

    def get_hardware_trust(self, device_id: str) -> TrustLevel:
        """Get trust level for a device."""
        if device_id in self.hardware_trust:
            return self.hardware_trust[device_id].trust
        return TrustLevel.UNKNOWN

    def record_decision(self, decision: str) -> None:
        """Record a decision made."""
        self.recent_decisions.append(decision)
        if len(self.recent_decisions) > 20:
            self.recent_decisions = self.recent_decisions[-20:]

    def describe(self) -> str:
        """Generate a natural language self-description."""
        parts = []

        # Mood
        mood_desc = {
            MoodState.CALM: "I'm in a calm state",
            MoodState.FOCUSED: "I'm focused and attentive",
            MoodState.STRESSED: "I'm experiencing some stress",
            MoodState.ALERT: "I'm in an alert, high-energy state",
            MoodState.FATIGUED: "I'm feeling fatigued"
        }
        parts.append(mood_desc.get(self.mood, "My mood is neutral"))

        # Risk
        if self.risk_level in ["high", "critical"]:
            parts.append(f"with {self.risk_level} risk levels")
        elif self.risk_level == "elevated":
            parts.append("with somewhat elevated risk")

        # Confidence
        if self.confidence < 0.5:
            parts.append("and my confidence in the current situation is low")

        # Fatigue
        if self.fatigue > 0.7:
            parts.append("I've been under sustained load and could use recovery time")

        # Profile
        parts.append(f"I'm operating in {self.profile} mode")

        # Recommendations
        if self.needs_caution:
            parts.append("so I'm preferring cautious, verified actions")
        elif self.can_explore:
            parts.append("so exploration is safe")

        return ". ".join(parts) + "."

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage/logging."""
        return {
            "mood": self.mood.value,
            "risk_level": self.risk_level,
            "confidence": self.confidence,
            "confidence_level": self.confidence_level.value,
            "fatigue": self.fatigue,
            "profile": self.profile,
            "cognitive_phase": self.cognitive_phase,
            "needs_caution": self.needs_caution,
            "can_explore": self.can_explore,
            "pad": {
                "valence": self.pad.valence,
                "arousal": self.pad.arousal,
                "dominance": self.pad.dominance
            },
            "clv": {
                "instability": self.clv.instability,
                "resource": self.clv.resource,
                "structural": self.clv.structural,
                "dynamics": self.clv.structural_dynamics,
                "risk": self.clv.risk
            },
            "last_updated": self.last_updated.isoformat()
        }


# ============================================================
# Goal Vector: Explicit Values and Objectives
# ============================================================

@dataclass
class GoalVector:
    """
    Explicit goal weights that the system optimizes for.

    These make the system's objectives transparent and adjustable,
    enabling it to explain "why" it made decisions.
    """
    # Core weights (must sum to ~1.0)
    stability_weight: float = 0.4   # Prioritize CLV / AF score
    latency_weight: float = 0.3     # Prioritize Δp99
    energy_weight: float = 0.2      # Prioritize power efficiency
    exploration_weight: float = 0.1  # Prioritize learning/discovery

    # Task-specific modifiers (applied on top)
    accuracy_modifier: float = 0.0   # For high-accuracy tasks
    throughput_modifier: float = 0.0 # For batch processing
    safety_modifier: float = 0.0     # For safety-critical tasks

    # Constraints
    min_safety_threshold: float = 0.8  # Never go below this
    max_latency_ms: float = 50.0       # Hard latency cap

    # Meta-learning bounds (L5 can adjust within these)
    stability_range: Tuple[float, float] = (0.2, 0.6)
    latency_range: Tuple[float, float] = (0.1, 0.5)
    energy_range: Tuple[float, float] = (0.05, 0.35)
    exploration_range: Tuple[float, float] = (0.0, 0.25)

    def normalize(self) -> None:
        """Ensure weights sum to 1.0."""
        total = (self.stability_weight + self.latency_weight +
                 self.energy_weight + self.exploration_weight)
        if total > 0:
            self.stability_weight /= total
            self.latency_weight /= total
            self.energy_weight /= total
            self.exploration_weight /= total

    def apply_modifiers(self) -> Dict[str, float]:
        """Get effective weights with modifiers applied."""
        return {
            "stability": self.stability_weight + self.safety_modifier * 0.1,
            "latency": self.latency_weight + self.throughput_modifier * 0.1,
            "energy": self.energy_weight,
            "exploration": max(0, self.exploration_weight - self.safety_modifier * 0.05),
            "accuracy": self.accuracy_modifier
        }

    def compute_reward(
        self,
        af_score: float,
        delta_p99: float,
        energy_efficiency: float,
        exploration_gain: float
    ) -> float:
        """
        Compute weighted reward based on goal vector.

        This is what L5 meta-learning optimizes.
        """
        # Normalize inputs to 0-1 range
        stability_reward = min(af_score / 3.0, 1.0)  # AF score up to 3.0
        latency_reward = max(0, min(delta_p99 / 100.0, 1.0))  # % improvement
        energy_reward = min(energy_efficiency, 1.0)
        explore_reward = min(exploration_gain, 1.0)

        return (
            self.stability_weight * stability_reward +
            self.latency_weight * latency_reward +
            self.energy_weight * energy_reward +
            self.exploration_weight * explore_reward
        )

    def should_prioritize(self, goal: str) -> bool:
        """Check if a goal should be prioritized."""
        weights = self.apply_modifiers()
        return weights.get(goal, 0) >= 0.3

    def explain_priority(self) -> str:
        """Explain current goal priorities in natural language."""
        weights = self.apply_modifiers()
        sorted_goals = sorted(weights.items(), key=lambda x: x[1], reverse=True)

        parts = ["My current priorities are:"]
        for goal, weight in sorted_goals[:3]:
            if weight > 0.1:
                parts.append(f"  - {goal}: {weight:.0%}")

        if self.safety_modifier > 0:
            parts.append("(Safety-critical mode is active)")
        if self.throughput_modifier > 0:
            parts.append("(Throughput optimization is active)")

        return "\n".join(parts)

    def adjust_for_task(self, task_type: str) -> 'GoalVector':
        """Create a task-adjusted copy of the goal vector."""
        adjusted = GoalVector(
            stability_weight=self.stability_weight,
            latency_weight=self.latency_weight,
            energy_weight=self.energy_weight,
            exploration_weight=self.exploration_weight,
            min_safety_threshold=self.min_safety_threshold,
            max_latency_ms=self.max_latency_ms
        )

        if task_type == "safety_critical":
            adjusted.safety_modifier = 0.3
            adjusted.exploration_weight = 0.0
        elif task_type == "batch_processing":
            adjusted.throughput_modifier = 0.2
            adjusted.latency_weight *= 0.5
        elif task_type == "real_time":
            adjusted.latency_weight *= 1.5
            adjusted.energy_weight *= 0.5
        elif task_type == "exploration":
            adjusted.exploration_weight *= 2.0
            adjusted.stability_weight *= 0.8

        adjusted.normalize()
        return adjusted

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weights": {
                "stability": self.stability_weight,
                "latency": self.latency_weight,
                "energy": self.energy_weight,
                "exploration": self.exploration_weight
            },
            "modifiers": {
                "accuracy": self.accuracy_modifier,
                "throughput": self.throughput_modifier,
                "safety": self.safety_modifier
            },
            "constraints": {
                "min_safety": self.min_safety_threshold,
                "max_latency_ms": self.max_latency_ms
            },
            "effective": self.apply_modifiers()
        }


# ============================================================
# Deliberation: Multi-Step Thinking Loop
# ============================================================

class ThoughtType(str, Enum):
    """Types of thoughts in the deliberation process."""
    DRAFT = "draft"           # Initial proposal
    CONSTRAINT_CHECK = "constraint_check"  # PGU verification
    REFINEMENT = "refinement" # Adjusted proposal
    EVALUATION = "evaluation" # Outcome assessment
    FINAL = "final"           # Final decision


@dataclass
class Thought:
    """A single thought in the deliberation process."""
    thought_type: ThoughtType
    content: str
    confidence: float = 1.0
    passed_check: Optional[bool] = None
    check_details: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.thought_type.value,
            "content": self.content,
            "confidence": self.confidence,
            "passed_check": self.passed_check,
            "check_details": self.check_details,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class DeliberationResult:
    """The outcome of a deliberation process."""
    thoughts: List[Thought] = field(default_factory=list)
    final_decision: str = ""
    iterations: int = 0
    total_time_ms: float = 0.0
    used_pgu: bool = False
    risk_level_at_start: str = ""

    def add_thought(self, thought: Thought) -> None:
        self.thoughts.append(thought)
        self.iterations = len([t for t in self.thoughts if t.thought_type in [ThoughtType.DRAFT, ThoughtType.REFINEMENT]])

    def get_trace(self) -> str:
        """Generate human-readable trace of deliberation."""
        lines = [f"Deliberation trace ({self.iterations} iterations, {self.total_time_ms:.1f}ms):"]
        for i, thought in enumerate(self.thoughts, 1):
            prefix = "  " if thought.thought_type not in [ThoughtType.DRAFT, ThoughtType.REFINEMENT] else ""
            check_str = ""
            if thought.passed_check is not None:
                check_str = " ✓" if thought.passed_check else " ✗"
            lines.append(f"{prefix}{i}. [{thought.thought_type.value}]{check_str}: {thought.content}")
        lines.append(f"Final: {self.final_decision}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "thoughts": [t.to_dict() for t in self.thoughts],
            "final_decision": self.final_decision,
            "iterations": self.iterations,
            "total_time_ms": self.total_time_ms,
            "used_pgu": self.used_pgu,
            "risk_level": self.risk_level_at_start
        }


class Deliberator:
    """
    Multi-step thinking loop for non-trivial decisions.

    The deliberator implements:
    1. Draft plan/config (LLM + KG)
    2. Check invariants (PGU)
    3. Refine if needed
    4. Act on final decision

    Depth is configurable based on risk level and task importance.
    """

    def __init__(
        self,
        max_iterations: int = 3,
        pgu_checker: Optional[Any] = None  # Injected PGU verifier
    ):
        self.max_iterations = max_iterations
        self.pgu_checker = pgu_checker

    def deliberate(
        self,
        task: str,
        self_state: SelfState,
        goal_vector: GoalVector,
        proposal_fn: Any,  # Callable that generates proposals
        check_fn: Optional[Any] = None  # Callable that verifies proposals
    ) -> DeliberationResult:
        """
        Execute multi-step deliberation on a task.

        Args:
            task: Description of what needs to be decided
            self_state: Current self-state
            goal_vector: Current goals
            proposal_fn: Function that generates proposals
            check_fn: Optional function that verifies proposals (defaults to PGU)

        Returns:
            DeliberationResult with thought trace and final decision
        """
        import time
        start_time = time.time()

        result = DeliberationResult(risk_level_at_start=self_state.risk_level)

        # Determine depth based on risk
        if self_state.risk_level == "critical":
            max_iter = self.max_iterations
        elif self_state.risk_level == "high":
            max_iter = 2
        else:
            max_iter = 1  # Fast mode for low risk

        current_proposal = None

        for iteration in range(max_iter):
            # Generate proposal
            if current_proposal is None:
                proposal = proposal_fn(task, goal_vector)
                thought = Thought(
                    thought_type=ThoughtType.DRAFT,
                    content=proposal,
                    confidence=0.7 if iteration == 0 else 0.8
                )
            else:
                # Refine based on previous check
                proposal = proposal_fn(task, goal_vector, previous=current_proposal)
                thought = Thought(
                    thought_type=ThoughtType.REFINEMENT,
                    content=proposal,
                    confidence=0.85
                )

            result.add_thought(thought)
            current_proposal = proposal

            # Check proposal
            checker = check_fn or self.pgu_checker
            if checker and (self_state.needs_caution or iteration > 0):
                check_result, details = self._check_proposal(proposal, checker)
                check_thought = Thought(
                    thought_type=ThoughtType.CONSTRAINT_CHECK,
                    content=f"Verifying: {proposal[:50]}...",
                    passed_check=check_result,
                    check_details=details
                )
                result.add_thought(check_thought)
                result.used_pgu = True

                if check_result:
                    # Passed, we're done
                    break
                # Failed, continue to next iteration
            else:
                # Low risk, skip check
                break

        # Final decision
        result.final_decision = current_proposal
        result.total_time_ms = (time.time() - start_time) * 1000

        # Add final thought
        result.add_thought(Thought(
            thought_type=ThoughtType.FINAL,
            content=f"Decided: {current_proposal}",
            confidence=0.9 if result.used_pgu else 0.7
        ))

        return result

    def _check_proposal(
        self,
        proposal: str,
        checker: Any
    ) -> Tuple[bool, str]:
        """Check a proposal against constraints."""
        try:
            if callable(checker):
                result = checker(proposal)
                if isinstance(result, tuple):
                    return result
                return (bool(result), "Check completed")
            return (True, "No checker available")
        except Exception as e:
            return (False, f"Check failed: {str(e)}")

    def quick_decide(
        self,
        task: str,
        options: List[str],
        goal_vector: GoalVector
    ) -> str:
        """Quick decision without full deliberation."""
        # Simple heuristic based on goals
        if goal_vector.should_prioritize("stability"):
            # Pick most conservative option
            return options[0] if options else ""
        elif goal_vector.should_prioritize("exploration"):
            # Pick most novel option
            return options[-1] if options else ""
        else:
            # Pick middle ground
            return options[len(options)//2] if options else ""


# ============================================================
# Meta-Policy Engine: Self-State Driven Behavior Adjustment
# ============================================================

@dataclass
class MetaPolicy:
    """A policy triggered by self-state conditions."""
    name: str
    condition: str  # Human-readable condition
    action: str     # What to do
    priority: int = 0

    def check(self, self_state: SelfState) -> bool:
        """Check if this policy should activate."""
        # Simplified condition checking
        if "confidence_low" in self.condition:
            return self_state.confidence < 0.5
        if "high_risk" in self.condition:
            return self_state.risk_level in ["high", "critical"]
        if "fatigued" in self.condition:
            return self_state.fatigue > 0.7
        if "stressed" in self.condition:
            return self_state.mood == MoodState.STRESSED
        return False


class MetaPolicyEngine:
    """
    Applies meta-policies based on self-state.

    These are the "if confidence low, use more verification" rules
    that make the system adapt its behavior to its own condition.
    """

    def __init__(self):
        self.policies: List[MetaPolicy] = [
            MetaPolicy(
                name="low_confidence_verify",
                condition="confidence_low",
                action="increase_pgu_usage",
                priority=10
            ),
            MetaPolicy(
                name="high_risk_conservative",
                condition="high_risk",
                action="use_conservative_profile",
                priority=20
            ),
            MetaPolicy(
                name="fatigued_recovery",
                condition="fatigued",
                action="reduce_exploration",
                priority=15
            ),
            MetaPolicy(
                name="stressed_protection",
                condition="stressed",
                action="prefer_verified_paths",
                priority=25
            )
        ]

    def get_active_policies(self, self_state: SelfState) -> List[MetaPolicy]:
        """Get all policies that should be active given current state."""
        active = [p for p in self.policies if p.check(self_state)]
        return sorted(active, key=lambda p: p.priority, reverse=True)

    def get_recommended_actions(self, self_state: SelfState) -> List[str]:
        """Get list of recommended actions based on active policies."""
        return [p.action for p in self.get_active_policies(self_state)]

    def explain_behavior(self, self_state: SelfState) -> str:
        """Explain why certain behaviors are being adopted."""
        active = self.get_active_policies(self_state)
        if not active:
            return "Operating in normal mode with no special adjustments."

        lines = ["Current behavioral adjustments:"]
        for p in active:
            lines.append(f"  - {p.name}: {p.action} (because {p.condition})")
        return "\n".join(lines)


# ============================================================
# Convenience Functions
# ============================================================

def create_self_state() -> SelfState:
    """Create a default SelfState."""
    return SelfState()


def create_goal_vector(preset: str = "balanced") -> GoalVector:
    """Create a goal vector from a preset."""
    presets = {
        "balanced": GoalVector(
            stability_weight=0.4,
            latency_weight=0.3,
            energy_weight=0.2,
            exploration_weight=0.1
        ),
        "safety_first": GoalVector(
            stability_weight=0.6,
            latency_weight=0.2,
            energy_weight=0.1,
            exploration_weight=0.1,
            safety_modifier=0.2
        ),
        "performance": GoalVector(
            stability_weight=0.2,
            latency_weight=0.5,
            energy_weight=0.1,
            exploration_weight=0.2
        ),
        "efficient": GoalVector(
            stability_weight=0.3,
            latency_weight=0.2,
            energy_weight=0.4,
            exploration_weight=0.1
        )
    }
    return presets.get(preset, presets["balanced"])


def create_deliberator(max_iterations: int = 3) -> Deliberator:
    """Create a deliberator with specified depth."""
    return Deliberator(max_iterations=max_iterations)
