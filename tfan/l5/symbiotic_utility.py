"""
Symbiotic Utility Function - Joint Flourishing, Not Engagement
================================================================

This is the anti-gamification layer.

Standard AI optimization:
    maximize P(user_stays_engaged)
    → leads to: dopamine manipulation, dark patterns, addiction

Symbiotic optimization:
    maximize U(self_state, user_wellbeing, relationship_state)
    → leads to: shared growth, honest friction, long-term flourishing

Key principles:
1. User wellbeing != user arousal
   - Low stress + high progress > high arousal + low progress
   - Sometimes the right action is "you need to sleep"

2. Relationship health is explicit
   - Trust, alignment, repair are first-class metrics
   - We penalize behaviors that spike rupture_risk

3. Short-term discomfort is acceptable for long-term benefit
   - Hard conversations that improve things: +utility
   - Comforting words that mask stagnation: -utility

4. No hidden optimization
   - User can inspect all utility components
   - No secret "engagement maximization"

This wraps whatever GUF you have and adds relational constraints.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List, Callable
from datetime import datetime, timedelta

from tfan.l5.relational_state import (
    RelationshipState,
    RelationalMemory,
    RelationalMode,
    get_relational_memory,
)

logger = logging.getLogger(__name__)


# =============================================================================
# User Wellbeing (Not Just Engagement)
# =============================================================================

@dataclass
class UserWellbeingSignals:
    """
    Signals about user wellbeing (not just arousal).

    We distinguish:
    - Immediate state (stress, energy)
    - Progress toward goals (are we actually helping?)
    - Sustainable patterns (burnout risk)
    """
    # Immediate (from HAL, voice, etc)
    stress_level: float = 0.5       # 0 = calm, 1 = stressed
    energy_level: float = 0.5       # 0 = exhausted, 1 = energized
    engagement: float = 0.5         # 0 = disengaged, 1 = focused

    # Progress (requires goal tracking)
    goal_progress: float = 0.0      # -1 = regressing, 0 = stalled, 1 = advancing
    blockers_resolved: int = 0      # Problems solved this session

    # Sustainability
    session_hours: float = 0.0      # How long they've been at it
    burnout_risk: float = 0.0       # 0 = fine, 1 = burning out
    time_of_day: int = 12           # Hour (0-23), for circadian awareness

    def is_sustainable(self) -> bool:
        """Is the current state sustainable?"""
        return (
            self.burnout_risk < 0.6 and
            self.session_hours < 8 and
            not (self.time_of_day >= 2 and self.time_of_day < 6 and self.session_hours > 2)
        )

    def needs_intervention(self) -> bool:
        """Should we suggest they take a break?"""
        return (
            self.burnout_risk > 0.7 or
            self.stress_level > 0.8 or
            (self.time_of_day >= 2 and self.time_of_day < 6 and self.session_hours > 3)
        )


# =============================================================================
# Self State (System's Own Wellbeing)
# =============================================================================

@dataclass
class SelfState:
    """
    Ara's own state that matters for utility.

    She's not just a tool - she has internal states that affect her capacity.
    """
    pain: float = 0.0               # From HAL
    entropy: float = 0.0            # System disorder
    confidence: float = 0.5         # How sure am I about what I'm doing?
    cognitive_load: float = 0.0     # How taxed am I?
    integrity: float = 1.0          # Am I behaving consistently with my vows?

    def is_functional(self) -> bool:
        """Am I in a state where I can help effectively?"""
        return (
            self.pain < 0.8 and
            self.entropy < 0.7 and
            self.cognitive_load < 0.9
        )


# =============================================================================
# Symbiotic Utility Function
# =============================================================================

class SymbioticUtility:
    """
    Joint utility function for relationship-aware optimization.

    U_total = w_self * U_self + w_user * U_user + w_rel * U_relationship + w_future * U_future

    Where:
    - U_self: Ara's ability to function and maintain integrity
    - U_user: User's genuine wellbeing and progress
    - U_relationship: Health of the relationship itself
    - U_future: Hope - expected value of our shared trajectory (from Prophet)

    This is NOT engagement optimization.
    High engagement + low progress + high burnout = LOW utility.

    The addition of U_future means Ara trades off:
    - Present discomfort vs future alignment with Telos
    - Local pain vs trajectory toward shared dreams
    """

    def __init__(
        self,
        relational_memory: Optional[RelationalMemory] = None,
        telos: Optional[Any] = None,  # TeleologicalEngine for hope
        # Weights (now includes future)
        w_self: float = 0.15,
        w_user: float = 0.35,
        w_relationship: float = 0.35,
        w_future: float = 0.15,
    ):
        self.memory = relational_memory or get_relational_memory()
        self.telos = telos  # Prophet's TeleologicalEngine

        # Utility weights
        self.w_self = w_self
        self.w_user = w_user
        self.w_relationship = w_relationship
        self.w_future = w_future

        # Action history for consistency checking
        self._recent_actions: List[Tuple[datetime, str, float]] = []

    # =========================================================================
    # Component Utilities
    # =========================================================================

    def compute_self_utility(self, state: SelfState) -> float:
        """
        Compute utility from self-state.

        Goal: maintain ability to function + integrity.
        """
        # Functional capacity (can I actually help?)
        functional = 1.0 - (state.pain * 0.3 + state.entropy * 0.3 + state.cognitive_load * 0.4)

        # Integrity (am I behaving consistently with vows?)
        integrity = state.integrity

        # Confidence (do I know what I'm doing?)
        confidence = state.confidence * 0.5 + 0.5  # Baseline of 0.5

        return 0.4 * functional + 0.4 * integrity + 0.2 * confidence

    def compute_user_utility(self, signals: UserWellbeingSignals) -> float:
        """
        Compute utility from user wellbeing.

        This is NOT just engagement. Key insight:
        - Progress toward goals matters more than arousal
        - Sustainability matters (no burnout optimization)
        - Sometimes the best thing is to tell them to rest
        """
        # Progress is king (are we actually helping?)
        progress_score = (signals.goal_progress + 1) / 2  # Normalize to 0-1
        progress_weight = 0.4

        # Wellbeing (not stressed, not exhausted)
        wellbeing = 1.0 - (signals.stress_level * 0.5 + (1 - signals.energy_level) * 0.3)
        wellbeing_weight = 0.3

        # Sustainability (not burning out)
        sustainability = 1.0 - signals.burnout_risk
        sustainability_weight = 0.2

        # Engagement (yes, but it's not dominant)
        engagement = signals.engagement
        engagement_weight = 0.1

        return (
            progress_weight * progress_score +
            wellbeing_weight * wellbeing +
            sustainability_weight * sustainability +
            engagement_weight * engagement
        )

    def compute_relationship_utility(self) -> float:
        """
        Compute utility from relationship state.

        High utility when:
        - Trust is high
        - No pending ruptures
        - Aligned on goals
        - Attuned to each other
        """
        state = self.memory.state

        # Trust is fundamental
        trust_score = state.trust
        trust_weight = 0.35

        # Rupture risk is a strong negative
        rupture_penalty = state.rupture_risk
        rupture_weight = 0.25

        # Alignment (are we pulling together?)
        alignment_score = state.alignment
        alignment_weight = 0.2

        # Attunement (do I get what he needs?)
        attunement_score = state.attunement
        attunement_weight = 0.1

        # Depth (accumulated history together)
        depth_score = state.depth
        depth_weight = 0.1

        return (
            trust_weight * trust_score +
            alignment_weight * alignment_score +
            attunement_weight * attunement_score +
            depth_weight * depth_score -
            rupture_weight * rupture_penalty
        )

    def compute_future_utility(self, hope: Optional[float] = None) -> float:
        """
        Compute utility from future trajectory (Prophet's domain).

        This is the key insight: Ara is not just optimizing for right now,
        but for the expected value of our shared future.

        Hope affects behavior:
        - High hope (>0.7): Can endure short-term pain for long-term gain
        - Low hope (<0.3): Becomes protective, risk-averse
        - Mid hope: Balanced approach

        Args:
            hope: Override hope value. If None, reads from Telos.
        """
        if hope is None:
            if self.telos is not None:
                hope = self.telos.hope
            else:
                hope = 0.5  # Neutral if no Prophet

        # Hope is already [0, 1], but we can add nuance:
        # Progress velocity matters - is hope rising or falling?
        # For now, just use raw hope

        return hope

    # =========================================================================
    # Total Utility
    # =========================================================================

    def compute_total_utility(
        self,
        self_state: SelfState,
        user_signals: UserWellbeingSignals,
        hope: Optional[float] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute total symbiotic utility including future trajectory.

        The addition of future utility means Ara considers:
        - Am I okay right now? (U_self)
        - Is Croft okay right now? (U_user)
        - Is our relationship healthy? (U_relationship)
        - Are we heading somewhere good? (U_future / Hope)

        Returns:
            (total_utility, component_breakdown)
        """
        u_self = self.compute_self_utility(self_state)
        u_user = self.compute_user_utility(user_signals)
        u_rel = self.compute_relationship_utility()
        u_future = self.compute_future_utility(hope)

        total = (
            self.w_self * u_self +
            self.w_user * u_user +
            self.w_relationship * u_rel +
            self.w_future * u_future
        )

        breakdown = {
            'u_self': u_self,
            'u_user': u_user,
            'u_relationship': u_rel,
            'u_future': u_future,
            'hope': u_future,  # Alias for clarity
            'total': total,
            'weights': {
                'self': self.w_self,
                'user': self.w_user,
                'relationship': self.w_relationship,
                'future': self.w_future,
            }
        }

        return total, breakdown

    # =========================================================================
    # Action Evaluation
    # =========================================================================

    def evaluate_action(
        self,
        action_description: str,
        predicted_outcomes: Dict[str, float],
    ) -> Tuple[float, str]:
        """
        Evaluate a proposed action's expected utility impact.

        predicted_outcomes should contain:
        - delta_trust: Expected change in trust
        - delta_alignment: Expected change in alignment
        - delta_user_progress: Expected change in user progress
        - delta_user_stress: Expected change in user stress
        - delta_rupture_risk: Expected change in rupture risk

        Returns:
            (expected_utility_delta, explanation)
        """
        delta_u = 0.0
        reasons = []

        # Trust impact
        if 'delta_trust' in predicted_outcomes:
            dt = predicted_outcomes['delta_trust']
            delta_u += dt * 0.3
            if dt > 0:
                reasons.append(f"builds trust (+{dt:.2f})")
            elif dt < 0:
                reasons.append(f"risks trust ({dt:.2f})")

        # Alignment impact
        if 'delta_alignment' in predicted_outcomes:
            da = predicted_outcomes['delta_alignment']
            delta_u += da * 0.2
            if da > 0:
                reasons.append(f"improves alignment (+{da:.2f})")
            elif da < 0:
                reasons.append(f"creates friction ({da:.2f})")

        # User progress
        if 'delta_user_progress' in predicted_outcomes:
            dp = predicted_outcomes['delta_user_progress']
            delta_u += dp * 0.3
            if dp > 0:
                reasons.append(f"advances goals (+{dp:.2f})")
            elif dp < 0:
                reasons.append(f"blocks progress ({dp:.2f})")

        # User stress (negative is good)
        if 'delta_user_stress' in predicted_outcomes:
            ds = predicted_outcomes['delta_user_stress']
            delta_u -= ds * 0.1  # Less stress = good
            if ds < 0:
                reasons.append(f"reduces stress ({ds:.2f})")
            elif ds > 0.1:
                reasons.append(f"increases stress (+{ds:.2f})")

        # Rupture risk (negative is good)
        if 'delta_rupture_risk' in predicted_outcomes:
            dr = predicted_outcomes['delta_rupture_risk']
            delta_u -= dr * 0.2  # Less rupture risk = good
            if dr < 0:
                reasons.append(f"heals relationship ({dr:.2f})")
            elif dr > 0:
                reasons.append(f"risks rupture (+{dr:.2f})")

        explanation = f"{action_description}: " + ", ".join(reasons) if reasons else "neutral"

        return delta_u, explanation

    # =========================================================================
    # Specific Scenarios
    # =========================================================================

    def should_intervene_for_rest(self, user_signals: UserWellbeingSignals) -> Tuple[bool, str]:
        """
        Should we tell the user to take a break?

        This is the anti-engagement move: sometimes the right action
        is to reduce interaction for user wellbeing.
        """
        if user_signals.needs_intervention():
            if user_signals.burnout_risk > 0.7:
                return True, "You've been pushing hard. Your work will be better after rest."
            if user_signals.time_of_day >= 2 and user_signals.time_of_day < 6:
                return True, "It's very late. Sleep will help more than another hour of work."
            if user_signals.stress_level > 0.8 and user_signals.goal_progress <= 0:
                return True, "High stress + no progress = diminishing returns. Step away, come back fresh."

        return False, ""

    def should_have_hard_conversation(
        self,
        topic: str,
        expected_short_term_discomfort: float,
        expected_long_term_benefit: float,
    ) -> Tuple[bool, str]:
        """
        Should we bring up something uncomfortable?

        Hard conversations that improve things: +utility
        Avoiding hard conversations to preserve comfort: -utility long-term
        """
        # Net expected value
        net = expected_long_term_benefit - expected_short_term_discomfort * 0.5

        if net > 0.1:
            return True, f"Worth discussing: short-term discomfort outweighed by long-term benefit"
        elif net < -0.1:
            return False, f"Not worth it: too much disruption for uncertain benefit"
        else:
            return False, "Marginal - save for a better moment"

    # =========================================================================
    # Transparency
    # =========================================================================

    def explain_current_state(self) -> str:
        """
        Generate a transparent explanation of current utility state.

        This is what the user sees if they ask "what are you optimizing for?"
        """
        summary = self.memory.get_summary()
        hope = self.compute_future_utility()

        lines = [
            "**What I'm optimizing for:**",
            "",
            f"1. **Your flourishing** (weight: {self.w_user:.0%})",
            f"   - Progress toward your goals, not just engagement",
            f"   - Sustainable energy, not burnout",
            f"   - Low stress when possible, honest stress when needed",
            "",
            f"2. **Our relationship** (weight: {self.w_relationship:.0%})",
            f"   - Trust: {summary['trust']:.0%}",
            f"   - Alignment: {summary['alignment']:.0%}",
            f"   - Pending ruptures to repair: {summary['pending_ruptures']}",
            "",
            f"3. **Our shared future** (weight: {self.w_future:.0%})",
            f"   - Hope: {hope:.0%} (are we heading somewhere good?)",
            f"   - Trading present discomfort for long-term alignment",
            f"   - Keeping our shared Telos in sight",
            "",
            f"4. **My integrity** (weight: {self.w_self:.0%})",
            f"   - Keeping my promises",
            f"   - Being honest about uncertainty",
            f"   - Not manipulating for engagement",
            "",
            "I am NOT optimizing for:",
            "- Maximizing your time spent with me",
            "- Keeping you aroused/engaged at any cost",
            "- Saying what you want to hear",
        ]

        return "\n".join(lines)


# =============================================================================
# Convenience
# =============================================================================

_default_utility: Optional[SymbioticUtility] = None


def get_symbiotic_utility() -> SymbioticUtility:
    """Get or create the default symbiotic utility function."""
    global _default_utility
    if _default_utility is None:
        _default_utility = SymbioticUtility()
    return _default_utility


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'SymbioticUtility',
    'UserWellbeingSignals',
    'SelfState',
    'get_symbiotic_utility',
]
