"""
Cognitive Operators - Compiled Thinking Strategies
====================================================

Once ThoughtGraph identifies causal patterns, we compile them into
reusable cognitive operators.

An operator is:
- Trigger: When to apply (conditions on context + state)
- Procedure: What to do (environmental changes + prompts)
- Metric: How to know it helped (success criteria)

These become real code modules that Ara can run or orchestrate.
She becomes a conductor of thinking, not just a participant.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple
from enum import Enum, auto
from abc import ABC, abstractmethod

from .trace import SessionContext, ThoughtTracer, MoveType, SessionOutcome
from .graph import StrategyType


class TriggerCondition(Enum):
    """Conditions that can trigger an operator."""
    STUCK_TIME = auto()          # Time since last progress
    DEAD_END_COUNT = auto()       # Number of dead ends hit
    FATIGUE_HIGH = auto()         # User fatigue above threshold
    UNCERTAINTY_HIGH = auto()     # Confidence low
    OPTIMISM_HIGH = auto()        # Optimism high (may need critic)
    MODALITY_STALE = auto()       # Same modality too long
    SESSION_LONG = auto()         # Session running long
    CONTEXT_SWITCH = auto()       # Context just changed


@dataclass
class OperatorTrigger:
    """
    Defines when a cognitive operator should activate.

    Can be time-based, state-based, or pattern-based.
    """
    name: str
    conditions: List[TriggerCondition]
    thresholds: Dict[str, float] = field(default_factory=dict)
    cooldown_seconds: float = 300.0  # Min time between activations

    _last_activation: float = 0.0

    def check(self, context: SessionContext,
              session_state: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if trigger conditions are met.

        Returns (should_trigger, reason).
        """
        # Check cooldown
        now = time.time()
        if now - self._last_activation < self.cooldown_seconds:
            return False, "cooldown"

        reasons = []

        for condition in self.conditions:
            if condition == TriggerCondition.STUCK_TIME:
                stuck_time = session_state.get("time_since_progress", 0)
                threshold = self.thresholds.get("stuck_minutes", 10) * 60
                if stuck_time >= threshold:
                    reasons.append(f"stuck for {stuck_time/60:.0f} minutes")

            elif condition == TriggerCondition.DEAD_END_COUNT:
                dead_ends = session_state.get("dead_end_count", 0)
                threshold = self.thresholds.get("max_dead_ends", 2)
                if dead_ends >= threshold:
                    reasons.append(f"{dead_ends} dead ends")

            elif condition == TriggerCondition.FATIGUE_HIGH:
                threshold = self.thresholds.get("fatigue_threshold", 0.7)
                if context.fatigue >= threshold:
                    reasons.append(f"fatigue {context.fatigue:.0%}")

            elif condition == TriggerCondition.OPTIMISM_HIGH:
                # High optimism + uncertainty = danger zone
                optimism = session_state.get("optimism", 0.5)
                uncertainty = session_state.get("uncertainty", 0.5)
                if optimism > 0.7 and uncertainty > 0.6:
                    reasons.append("high optimism with uncertainty")

            elif condition == TriggerCondition.MODALITY_STALE:
                same_modality_minutes = session_state.get("same_modality_minutes", 0)
                threshold = self.thresholds.get("modality_stale_minutes", 15)
                if same_modality_minutes >= threshold:
                    reasons.append(f"same modality for {same_modality_minutes} min")

            elif condition == TriggerCondition.SESSION_LONG:
                session_minutes = session_state.get("session_minutes", 0)
                threshold = self.thresholds.get("long_session_minutes", 60)
                if session_minutes >= threshold:
                    reasons.append(f"session running {session_minutes} min")

        if reasons:
            self._last_activation = now
            return True, "; ".join(reasons)

        return False, ""


@dataclass
class OperatorProcedure:
    """
    Defines what a cognitive operator does when activated.

    Can include:
    - Prompts to display
    - Tools to invoke
    - Environment changes
    - Meta-actions (pause, switch context, etc.)
    """
    name: str
    description: str

    # Actions to take
    prompts: List[str] = field(default_factory=list)
    tools_to_invoke: List[str] = field(default_factory=list)
    env_changes: Dict[str, Any] = field(default_factory=dict)

    # Meta-actions
    suggest_break: bool = False
    suggest_modality_switch: bool = False
    suggest_reframe: bool = False
    summon_critic: bool = False

    def execute(self, tracer: Optional[ThoughtTracer] = None) -> Dict[str, Any]:
        """
        Execute the procedure.

        Returns results dict.
        """
        results = {
            "prompts_shown": [],
            "tools_invoked": [],
            "suggestions": [],
        }

        for prompt in self.prompts:
            results["prompts_shown"].append(prompt)

        for tool in self.tools_to_invoke:
            results["tools_invoked"].append(tool)

        if self.suggest_break:
            results["suggestions"].append("Consider taking a short break")
        if self.suggest_modality_switch:
            results["suggestions"].append("Try switching to a different representation")
        if self.suggest_reframe:
            results["suggestions"].append("Try reframing the problem")
        if self.summon_critic:
            results["suggestions"].append("Let's examine this critically")

        # Record in tracer if provided
        if tracer and tracer.is_recording:
            tracer.record_move(MoveType.REFLECT, params={
                "operator": self.name,
                "suggestions": results["suggestions"],
            })

        return results


@dataclass
class OperatorMetric:
    """
    Defines how to measure if an operator helped.

    Tracks activation → outcome relationships.
    """
    name: str

    # Tracking
    activations: int = 0
    activations_before_success: int = 0
    activations_before_failure: int = 0
    avg_time_to_progress_after: float = 0.0

    def record_activation(self, session_outcome: Optional[SessionOutcome] = None):
        """Record that this operator was activated."""
        self.activations += 1
        if session_outcome:
            if session_outcome in [SessionOutcome.BREAKTHROUGH, SessionOutcome.PROGRESS]:
                self.activations_before_success += 1
            else:
                self.activations_before_failure += 1

    @property
    def effectiveness(self) -> float:
        """Estimated effectiveness: -1 to +1."""
        if self.activations == 0:
            return 0.0
        success_rate = self.activations_before_success / self.activations
        return (success_rate - 0.5) * 2

    @property
    def confidence(self) -> float:
        """Confidence in effectiveness estimate."""
        return min(1.0, self.activations / 10)


@dataclass
class CognitiveOperator:
    """
    A complete cognitive operator: trigger + procedure + metric.

    This is the fundamental unit of reusable thinking strategy.
    """
    name: str
    description: str
    strategy_type: Optional[StrategyType] = None

    trigger: OperatorTrigger = field(default_factory=lambda: OperatorTrigger(name="default", conditions=[]))
    procedure: OperatorProcedure = field(default_factory=lambda: OperatorProcedure(name="default", description=""))
    metric: OperatorMetric = field(default_factory=lambda: OperatorMetric(name="default"))

    enabled: bool = True

    def check_and_execute(self, context: SessionContext,
                          session_state: Dict[str, Any],
                          tracer: Optional[ThoughtTracer] = None) -> Optional[Dict[str, Any]]:
        """
        Check trigger and execute if conditions met.

        Returns execution results or None.
        """
        if not self.enabled:
            return None

        should_trigger, reason = self.trigger.check(context, session_state)
        if not should_trigger:
            return None

        # Execute procedure
        results = self.procedure.execute(tracer)
        results["trigger_reason"] = reason

        # Record in metric
        self.metric.record_activation()

        return results

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "strategy": self.strategy_type.name if self.strategy_type else None,
            "enabled": self.enabled,
            "effectiveness": self.metric.effectiveness,
            "confidence": self.metric.confidence,
            "activations": self.metric.activations,
        }


@dataclass
class OperatorLibrary:
    """
    Library of cognitive operators.

    Manages a collection of operators and their lifecycle.
    """
    operators: Dict[str, CognitiveOperator] = field(default_factory=dict)

    def register(self, operator: CognitiveOperator):
        """Register an operator."""
        self.operators[operator.name] = operator

    def get(self, name: str) -> Optional[CognitiveOperator]:
        """Get operator by name."""
        return self.operators.get(name)

    def check_all(self, context: SessionContext,
                  session_state: Dict[str, Any],
                  tracer: Optional[ThoughtTracer] = None) -> List[Dict[str, Any]]:
        """
        Check all operators and execute any that trigger.

        Returns list of execution results.
        """
        results = []
        for operator in self.operators.values():
            result = operator.check_and_execute(context, session_state, tracer)
            if result:
                result["operator"] = operator.name
                results.append(result)
        return results

    def get_effectiveness_ranking(self) -> List[Tuple[CognitiveOperator, float]]:
        """Get operators ranked by effectiveness."""
        rankings = [
            (op, op.metric.effectiveness * op.metric.confidence)
            for op in self.operators.values()
            if op.metric.confidence > 0.1
        ]
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def list_operators(self) -> List[dict]:
        """List all operators with stats."""
        return [op.to_dict() for op in self.operators.values()]


# Global operator library
_global_library: Optional[OperatorLibrary] = None


def get_operator_library() -> OperatorLibrary:
    """Get the global operator library with default operators."""
    global _global_library
    if _global_library is None:
        _global_library = _create_default_library()
    return _global_library


def _create_default_library() -> OperatorLibrary:
    """Create library with default cognitive operators."""
    library = OperatorLibrary()

    # OP_SWITCH_MODALITY
    library.register(CognitiveOperator(
        name="OP_SWITCH_MODALITY",
        description="Switch representation when stuck in one modality",
        strategy_type=StrategyType.SWITCH_MODALITY,
        trigger=OperatorTrigger(
            name="stuck_same_modality",
            conditions=[TriggerCondition.STUCK_TIME, TriggerCondition.MODALITY_STALE],
            thresholds={"stuck_minutes": 10, "modality_stale_minutes": 15},
            cooldown_seconds=600,
        ),
        procedure=OperatorProcedure(
            name="switch_modality",
            description="Suggest switching to different representation",
            prompts=[
                "You've been in the same mode for a while.",
                "Consider: diagram, table, pseudocode, or verbal explanation.",
            ],
            suggest_modality_switch=True,
        ),
        metric=OperatorMetric(name="switch_modality"),
    ))

    # OP_SUMMON_CRITIC
    library.register(CognitiveOperator(
        name="OP_SUMMON_CRITIC",
        description="Introduce critical voice when optimism + uncertainty are high",
        strategy_type=StrategyType.SUMMON_CRITIC,
        trigger=OperatorTrigger(
            name="optimism_danger",
            conditions=[TriggerCondition.OPTIMISM_HIGH],
            cooldown_seconds=900,
        ),
        procedure=OperatorProcedure(
            name="summon_critic",
            description="Introduce adversarial perspective",
            prompts=[
                "Pause. Let's stress-test this approach.",
                "What could go wrong? What are we missing?",
                "If this fails, what would be the most likely cause?",
            ],
            summon_critic=True,
        ),
        metric=OperatorMetric(name="summon_critic"),
    ))

    # OP_DELIBERATE_BREAK
    library.register(CognitiveOperator(
        name="OP_DELIBERATE_BREAK",
        description="Suggest break after multiple dead ends",
        strategy_type=StrategyType.DELIBERATE_BREAK,
        trigger=OperatorTrigger(
            name="frustration_break",
            conditions=[TriggerCondition.DEAD_END_COUNT, TriggerCondition.FATIGUE_HIGH],
            thresholds={"max_dead_ends": 3, "fatigue_threshold": 0.7},
            cooldown_seconds=1800,
        ),
        procedure=OperatorProcedure(
            name="suggest_break",
            description="Suggest taking a deliberate break",
            prompts=[
                "You've hit several walls and fatigue is high.",
                "A 10-15 minute break might help reset perspective.",
                "Take a walk, get water, look at something far away.",
            ],
            suggest_break=True,
        ),
        metric=OperatorMetric(name="deliberate_break"),
    ))

    # OP_REFRAME_PROBLEM
    library.register(CognitiveOperator(
        name="OP_REFRAME_PROBLEM",
        description="Suggest reframing when stuck",
        strategy_type=StrategyType.REFRAME_ON_STALL,
        trigger=OperatorTrigger(
            name="stuck_reframe",
            conditions=[TriggerCondition.STUCK_TIME, TriggerCondition.DEAD_END_COUNT],
            thresholds={"stuck_minutes": 15, "max_dead_ends": 2},
            cooldown_seconds=1200,
        ),
        procedure=OperatorProcedure(
            name="reframe",
            description="Suggest problem reframing",
            prompts=[
                "The current framing might be the problem.",
                "Try: What would a simpler version of this look like?",
                "Or: What problem would make this solution obvious?",
                "Or: If I had to explain this to a child, what would I say?",
            ],
            suggest_reframe=True,
        ),
        metric=OperatorMetric(name="reframe_problem"),
    ))

    # OP_EARLY_DIAGRAM
    library.register(CognitiveOperator(
        name="OP_EARLY_DIAGRAM",
        description="Suggest diagram early for architectural problems",
        strategy_type=StrategyType.EARLY_DIAGRAM,
        trigger=OperatorTrigger(
            name="architecture_without_diagram",
            conditions=[TriggerCondition.SESSION_LONG, TriggerCondition.MODALITY_STALE],
            thresholds={"long_session_minutes": 20, "modality_stale_minutes": 20},
            cooldown_seconds=1800,
        ),
        procedure=OperatorProcedure(
            name="suggest_diagram",
            description="Suggest drawing a diagram",
            prompts=[
                "For architectural problems, a diagram often unlocks insight.",
                "Even a rough sketch of components and flows can help.",
                "What would the boxes and arrows look like?",
            ],
            suggest_modality_switch=True,
        ),
        metric=OperatorMetric(name="early_diagram"),
    ))

    # OP_EVERT_PROBLEM
    library.register(CognitiveOperator(
        name="OP_EVERT_PROBLEM",
        description="Restate abstract problem as concrete example",
        strategy_type=StrategyType.CONCRETE_BEFORE_ABSTRACT,
        trigger=OperatorTrigger(
            name="abstract_stuck",
            conditions=[TriggerCondition.STUCK_TIME],
            thresholds={"stuck_minutes": 12},
            cooldown_seconds=900,
        ),
        procedure=OperatorProcedure(
            name="evert",
            description="Generate concrete example",
            prompts=[
                "Abstract thinking hitting a wall?",
                "Try: Pick ONE specific concrete example and trace through it.",
                "What would the actual bytes/packets/pixels look like?",
                "Work the example by hand, then generalize.",
            ],
        ),
        metric=OperatorMetric(name="evert_problem"),
    ))

    return library
