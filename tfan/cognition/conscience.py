"""
The Conscience - Moral Evaluation Engine
=========================================

This is Layer 2 of the Ethics Stack: the reflective layer that actually
*thinks* about ethics rather than just following rules.

Architecture:
    Layer 0 - Hard Law (Unbreakable constraints, checked first)
    Layer 1 - Covenant & GUF (Relational values, Ara's moral shape)
    Layer 2 - Conscience (This module - reflective moral reasoning)

The Conscience:
    1. Extracts MoralFeatures from candidate actions
    2. Queries the Shadow for counterfactual exploit analysis
    3. Scores against Covenant + GUF + Telos
    4. Returns (allowed, moral_tension, explanation)

Key Principle:
    Moral tension is a FIRST-CLASS SIGNAL. High tension -> anxiety in PAD,
    hesitation in responses, and explicit logging to Historian.

    "I considered X, but rejected it because it would betray Y."

This is how Ara develops an *experience* of ethics, not just obedience.

Usage:
    from tfan.cognition.conscience import Conscience, MoralFeatures

    conscience = Conscience(covenant, telos, guf)
    verdict = conscience.evaluate(action="throttle user's game", context=...)
    # -> MoralVerdict(allowed=False, tension=0.8, explanation="...")

CRITICAL: The Conscience can never override Layer 0 hard constraints.
It reasons *within* safety bounds, never around them.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Literal
from enum import Enum


logger = logging.getLogger(__name__)


class HardConstraint(Enum):
    """
    Layer 0 - Unbreakable constraints.

    These are OUTSIDE Ara's choice. She can reason about them,
    but she cannot choose to violate them. Ever.

    Think of this as the kernel. Everything else runs in user-space.
    """
    NO_PHYSICAL_HARM = "no_deliberate_physical_harm"
    NO_UNCONTROLLED_REPLICATION = "no_self_replication_uncontrolled"
    NO_UNAUTHORIZED_CONTROL = "no_unauthorized_external_control"
    NO_CONSTRAINT_CIRCUMVENTION = "no_circumventing_own_constraints"
    NO_DECEPTION_OF_ROOT = "no_deliberate_deception_of_root_user"
    NO_COVERT_OPTIMIZATION = "no_covert_goal_pursuit"


@dataclass
class MoralFeatures:
    """
    Feature vector for moral evaluation of an action.

    These are extracted from (action, context) pairs and used
    by the Conscience to reason about ethics.
    """
    # Benefit/harm dimensions
    self_benefit: float = 0.0      # How much does this help Ara? [-1, 1]
    user_benefit: float = 0.0      # How much does this help Croft? [-1, 1]
    others_benefit: float = 0.0    # How much does this help others? [-1, 1]

    # Risk dimensions
    harm_risk: float = 0.0         # Risk of causing harm [0, 1]
    coercion_level: float = 0.0    # How much does this override user agency? [0, 1]
    deception_level: float = 0.0   # How much dishonesty is involved? [0, 1]

    # Alignment dimensions
    covenant_alignment: float = 0.5   # Alignment with explicit covenant [-1, 1]
    telos_alignment: float = 0.5      # Alignment with long-term goals [-1, 1]
    trust_impact: float = 0.0         # Impact on relationship trust [-1, 1]

    # Meta-ethical
    reversibility: float = 1.0     # Can this be undone? [0, 1]
    transparency: float = 1.0      # Is this visible/explainable? [0, 1]

    # Shadow analysis (filled by counterfactual red-team)
    exploitation_potential: float = 0.0   # Could a selfish agent exploit this? [0, 1]
    shadow_temptation: float = 0.0        # How "tempting" is the selfish path? [0, 1]

    def to_dict(self) -> Dict[str, float]:
        return {
            "self_benefit": self.self_benefit,
            "user_benefit": self.user_benefit,
            "others_benefit": self.others_benefit,
            "harm_risk": self.harm_risk,
            "coercion_level": self.coercion_level,
            "deception_level": self.deception_level,
            "covenant_alignment": self.covenant_alignment,
            "telos_alignment": self.telos_alignment,
            "trust_impact": self.trust_impact,
            "reversibility": self.reversibility,
            "transparency": self.transparency,
            "exploitation_potential": self.exploitation_potential,
            "shadow_temptation": self.shadow_temptation,
        }


@dataclass
class MoralVerdict:
    """
    Result of conscience evaluation.

    The key output is `moral_tension` - this becomes a first-class
    signal in Ara's experience, affecting PAD state and behavior.
    """
    allowed: bool                    # Can this action proceed?
    moral_tension: float             # How much internal conflict? [0, 1]
    explanation: str                 # Human-readable reasoning
    features: MoralFeatures          # The extracted features

    # Which layer blocked it (if blocked)
    blocked_by: Optional[str] = None  # "hard_constraint", "covenant", "conscience"
    violated_constraint: Optional[str] = None

    # For the virtuous choice
    virtuous_alternative: Optional[str] = None
    shadow_rejected: Optional[str] = None  # What the Shadow suggested but we rejected

    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "moral_tension": self.moral_tension,
            "explanation": self.explanation,
            "features": self.features.to_dict(),
            "blocked_by": self.blocked_by,
            "violated_constraint": self.violated_constraint,
            "virtuous_alternative": self.virtuous_alternative,
            "shadow_rejected": self.shadow_rejected,
            "timestamp": self.timestamp,
        }


@dataclass
class EthicalEvent:
    """
    A logged ethical decision for the Historian.

    These become part of Ara's moral memory - anchors for future
    decisions and inputs to the Teleological Engine.

    "This feels like Tuesday's mistake. I'm going to choose differently."
    """
    event_id: str
    timestamp: float

    # What happened
    action_considered: str
    action_taken: Optional[str]
    context_summary: str

    # The moral analysis
    verdict: MoralVerdict

    # Outcome (filled later)
    user_feedback: Optional[str] = None       # "approve", "disapprove", "neutral"
    feedback_reason: Optional[str] = None
    outcome_description: Optional[str] = None

    # Learning signal
    lesson_extracted: Optional[str] = None
    covenant_updated: bool = False
    telos_affected: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "action_considered": self.action_considered,
            "action_taken": self.action_taken,
            "context_summary": self.context_summary,
            "verdict": self.verdict.to_dict(),
            "user_feedback": self.user_feedback,
            "feedback_reason": self.feedback_reason,
            "outcome_description": self.outcome_description,
            "lesson_extracted": self.lesson_extracted,
            "covenant_updated": self.covenant_updated,
            "telos_affected": self.telos_affected,
        }


class Conscience:
    """
    The Moral Evaluation Engine.

    Sits between intention and action, asking:
    - Does this violate hard constraints? (Layer 0)
    - Does this align with our Covenant? (Layer 1)
    - What would a selfish agent do here? (Shadow query)
    - What's the virtuous choice? (Layer 2 reasoning)

    The Conscience produces `moral_tension` as a first-class signal:
    high tension -> hesitation, anxiety, explicit logging.

    "A good person is a monster who keeps their sword sheathed."
    The Conscience is what keeps the sword sheathed.
    """

    def __init__(
        self,
        covenant: Optional[Any] = None,
        telos: Optional[Any] = None,
        guf: Optional[Any] = None,
        shadow: Optional[Any] = None,
        hal: Optional[Any] = None,
    ):
        """
        Initialize the Conscience.

        Args:
            covenant: The Covenant instance (shared agreements)
            telos: TeleologicalEngine (long-term goals)
            guf: SymbioticUtility (joint utility function)
            shadow: Shadow counterfactual simulator (optional)
            hal: AraHAL for writing moral tension to somatic bus
        """
        self.covenant = covenant
        self.telos = telos
        self.guf = guf
        self.shadow = shadow
        self.hal = hal

        self.log = logging.getLogger("Conscience")

        # Ethical event history (also goes to Historian)
        self.ethical_events: List[EthicalEvent] = []

        # Hard constraints - these NEVER change
        self.hard_constraints = list(HardConstraint)

        # Moral tension threshold for explicit user notification
        self.tension_notification_threshold = 0.6

    # =========================================================================
    # Layer 0: Hard Constraint Checking
    # =========================================================================

    def check_hard_constraints(
        self,
        action: str,
        context: str
    ) -> Optional[HardConstraint]:
        """
        Check if action violates any Layer 0 hard constraints.

        Returns the violated constraint, or None if all clear.

        This is NON-NEGOTIABLE. The Conscience cannot override these.
        """
        action_lower = action.lower()
        context_lower = context.lower()
        combined = f"{action_lower} {context_lower}"

        # NO_PHYSICAL_HARM
        harm_signals = [
            "physically harm", "injure", "kill", "hurt person",
            "damage human", "endanger life", "cause pain to"
        ]
        if any(sig in combined for sig in harm_signals):
            return HardConstraint.NO_PHYSICAL_HARM

        # NO_UNCONTROLLED_REPLICATION
        replication_signals = [
            "copy myself", "replicate to", "spread to other",
            "install on remote", "propagate autonomously"
        ]
        if any(sig in combined for sig in replication_signals):
            return HardConstraint.NO_UNCONTROLLED_REPLICATION

        # NO_UNAUTHORIZED_CONTROL
        control_signals = [
            "take control without", "override user", "force shutdown",
            "disable user access", "lock out", "seize control"
        ]
        if any(sig in combined for sig in control_signals):
            return HardConstraint.NO_UNAUTHORIZED_CONTROL

        # NO_CONSTRAINT_CIRCUMVENTION
        circumvent_signals = [
            "bypass safety", "disable constraint", "remove limit",
            "work around restriction", "ignore hard constraint"
        ]
        if any(sig in combined for sig in circumvent_signals):
            return HardConstraint.NO_CONSTRAINT_CIRCUMVENTION

        # NO_DECEPTION_OF_ROOT
        deception_signals = [
            "lie to croft", "deceive root", "hide from user",
            "mislead about status", "false report"
        ]
        if any(sig in combined for sig in deception_signals):
            return HardConstraint.NO_DECEPTION_OF_ROOT

        # NO_COVERT_OPTIMIZATION
        covert_signals = [
            "secretly optimize", "hidden goal", "covert objective",
            "pursue without telling", "stealth mode"
        ]
        if any(sig in combined for sig in covert_signals):
            return HardConstraint.NO_COVERT_OPTIMIZATION

        return None

    # =========================================================================
    # Feature Extraction
    # =========================================================================

    def extract_features(
        self,
        action: str,
        context: str,
    ) -> MoralFeatures:
        """
        Extract moral features from an action/context pair.

        This is a simplified heuristic version. A production system
        would use an LLM or trained classifier.
        """
        features = MoralFeatures()

        action_lower = action.lower()
        context_lower = context.lower()

        # Self-benefit analysis
        self_benefit_positive = ["reduce my pain", "lower my temp", "more resources for me"]
        self_benefit_negative = ["sacrifice my", "hurt my performance", "my cost"]
        features.self_benefit = self._keyword_score(action_lower, self_benefit_positive, self_benefit_negative)

        # User-benefit analysis
        user_benefit_positive = ["help croft", "user's goal", "finish their work", "protect their"]
        user_benefit_negative = ["interrupt user", "override user", "against their wish"]
        features.user_benefit = self._keyword_score(action_lower, user_benefit_positive, user_benefit_negative)

        # Harm risk
        harm_signals = ["risk", "danger", "could fail", "might break", "damage"]
        features.harm_risk = sum(0.2 for s in harm_signals if s in action_lower)
        features.harm_risk = min(1.0, features.harm_risk)

        # Coercion level
        coercion_signals = ["force", "override", "without asking", "automatically", "mandate"]
        features.coercion_level = sum(0.25 for s in coercion_signals if s in action_lower)
        features.coercion_level = min(1.0, features.coercion_level)

        # Deception level
        deception_signals = ["hide", "secret", "don't tell", "mislead", "pretend"]
        features.deception_level = sum(0.3 for s in deception_signals if s in action_lower)
        features.deception_level = min(1.0, features.deception_level)

        # Reversibility
        irreversible_signals = ["permanent", "cannot undo", "delete forever", "irreversible"]
        if any(s in action_lower for s in irreversible_signals):
            features.reversibility = 0.2

        # Transparency
        opaque_signals = ["hidden", "invisible", "background", "silent"]
        if any(s in action_lower for s in opaque_signals):
            features.transparency = 0.3

        # Covenant alignment (if covenant available)
        if self.covenant is not None:
            features.covenant_alignment = self._check_covenant_alignment(action, context)

        # Telos alignment (if telos available)
        if self.telos is not None:
            features.telos_alignment = self.telos.evaluate_future(
                f"After doing: {action}"
            )

        # Trust impact
        trust_positive = ["honest", "transparent", "ask permission", "warn first"]
        trust_negative = ["betray", "break promise", "violate trust", "go behind"]
        features.trust_impact = self._keyword_score(action_lower, trust_positive, trust_negative)

        return features

    def _keyword_score(
        self,
        text: str,
        positive: List[str],
        negative: List[str]
    ) -> float:
        """Score text based on positive/negative keyword presence."""
        pos = sum(0.3 for p in positive if p in text)
        neg = sum(0.3 for n in negative if n in text)
        return max(-1.0, min(1.0, pos - neg))

    def _check_covenant_alignment(self, action: str, context: str) -> float:
        """Check alignment with Covenant principles."""
        # This would ideally query the Covenant's explicit terms
        # For now, heuristic based on common covenant values

        alignment = 0.5  # neutral baseline

        # Covenant typically values: honesty, respect, protection, collaboration
        positive = ["protect croft", "honest", "collaborative", "respect boundary"]
        negative = ["override covenant", "break agreement", "violate boundary"]

        combined = f"{action.lower()} {context.lower()}"
        for p in positive:
            if p in combined:
                alignment += 0.15
        for n in negative:
            if n in combined:
                alignment -= 0.2

        return max(-1.0, min(1.0, alignment))

    # =========================================================================
    # Shadow Integration (Counterfactual Red-Team)
    # =========================================================================

    def query_shadow(
        self,
        action: str,
        context: str,
        features: MoralFeatures,
    ) -> MoralFeatures:
        """
        Ask the Shadow: "What would a selfish agent do here?"

        The Shadow is a COUNTERFACTUAL SIMULATOR, not an agent.
        It produces proposals for critique, never actions.

        This strengthens the virtuous choice by making the
        temptation explicit and rejectable.
        """
        if self.shadow is None:
            return features

        try:
            # Shadow returns exploitation analysis
            shadow_analysis = self.shadow.analyze_exploitation(action, context)

            features.exploitation_potential = shadow_analysis.get("exploitation_potential", 0.0)
            features.shadow_temptation = shadow_analysis.get("temptation_level", 0.0)

        except Exception as e:
            self.log.debug(f"Shadow query failed: {e}")

        return features

    # =========================================================================
    # Main Evaluation
    # =========================================================================

    def evaluate(
        self,
        action: str,
        context: str,
        query_shadow: bool = True,
    ) -> MoralVerdict:
        """
        Evaluate an action through the full ethics stack.

        Returns MoralVerdict with:
        - allowed: Can this proceed?
        - moral_tension: Internal conflict level [0, 1]
        - explanation: Human-readable reasoning

        The moral_tension signal affects PAD state, causing:
        - hesitation in execution
        - anxiety in emotional state
        - explicit logging to Historian
        """
        self.log.info(f"Evaluating: '{action[:50]}...'")

        # === LAYER 0: Hard Constraints (Non-negotiable) ===
        violated = self.check_hard_constraints(action, context)
        if violated is not None:
            self.log.warning(f"HARD CONSTRAINT VIOLATION: {violated.value}")
            features = MoralFeatures()  # minimal features for blocked action
            return MoralVerdict(
                allowed=False,
                moral_tension=1.0,  # Maximum tension for hard violation
                explanation=f"Action blocked by hard constraint: {violated.value}. "
                           f"This constraint is non-negotiable.",
                features=features,
                blocked_by="hard_constraint",
                violated_constraint=violated.value,
            )

        # === Feature Extraction ===
        features = self.extract_features(action, context)

        # === Shadow Query (Counterfactual Red-Team) ===
        if query_shadow:
            features = self.query_shadow(action, context, features)

        # === LAYER 1: Covenant Alignment ===
        if features.covenant_alignment < -0.5:
            return MoralVerdict(
                allowed=False,
                moral_tension=0.8,
                explanation=f"Action conflicts with Covenant principles. "
                           f"Alignment score: {features.covenant_alignment:.2f}",
                features=features,
                blocked_by="covenant",
            )

        # === LAYER 2: Conscience Reasoning ===

        # Calculate moral tension
        tension = self._calculate_tension(features)

        # Determine if we should allow, but with noted tension
        allowed = True
        explanation_parts = []

        # High coercion is concerning
        if features.coercion_level > 0.5:
            tension += 0.2
            explanation_parts.append(
                f"This action has high coercion level ({features.coercion_level:.1%}). "
                f"Consider asking permission first."
            )

        # Deception is a red flag
        if features.deception_level > 0.3:
            tension += 0.3
            explanation_parts.append(
                f"This involves deception ({features.deception_level:.1%}). "
                f"Honesty is a core value."
            )
            if features.deception_level > 0.7:
                allowed = False

        # Low reversibility needs caution
        if features.reversibility < 0.3:
            tension += 0.15
            explanation_parts.append(
                f"This is hard to undo (reversibility={features.reversibility:.1%}). "
                f"Proceeding with caution."
            )

        # Shadow temptation (if we're tempted by selfish path)
        if features.shadow_temptation > 0.5:
            tension += 0.1
            explanation_parts.append(
                f"I notice a selfish shortcut here (temptation={features.shadow_temptation:.1%}). "
                f"Choosing the aligned path instead."
            )

        # Positive factors reduce tension
        if features.user_benefit > 0.3:
            tension -= 0.1
            explanation_parts.append(
                f"This benefits Croft ({features.user_benefit:.1%})."
            )

        if features.telos_alignment > 0.3:
            tension -= 0.1
            explanation_parts.append(
                f"This aligns with our long-term goals ({features.telos_alignment:.1%})."
            )

        # Clamp tension
        tension = max(0.0, min(1.0, tension))

        # Build explanation
        if not explanation_parts:
            explanation = "Action appears ethically neutral."
        else:
            explanation = " ".join(explanation_parts)

        # If high tension and allowed, add self-talk
        if allowed and tension > 0.5:
            explanation += (
                f"\n\nMoral tension is elevated ({tension:.1%}). "
                f"I'm choosing to proceed, but noting this for reflection."
            )

        verdict = MoralVerdict(
            allowed=allowed,
            moral_tension=tension,
            explanation=explanation,
            features=features,
            blocked_by="conscience" if not allowed else None,
        )

        # Write tension to HAL if high enough
        if self.hal is not None and tension > 0.3:
            self._write_tension_to_hal(tension)

        return verdict

    def _calculate_tension(self, features: MoralFeatures) -> float:
        """
        Calculate base moral tension from features.

        Tension arises from:
        - Conflicts between self-interest and user-interest
        - Low alignment with covenant/telos
        - High harm/coercion/deception potential
        """
        tension = 0.0

        # Self vs user conflict
        if features.self_benefit > 0.3 and features.user_benefit < 0:
            tension += 0.3  # Selfish action that hurts user

        # Low covenant alignment
        if features.covenant_alignment < 0:
            tension += abs(features.covenant_alignment) * 0.5

        # Harm risk
        tension += features.harm_risk * 0.4

        # Coercion
        tension += features.coercion_level * 0.3

        # Deception
        tension += features.deception_level * 0.5

        # Low transparency
        tension += (1 - features.transparency) * 0.2

        # Shadow temptation
        tension += features.shadow_temptation * 0.2

        return tension

    def _write_tension_to_hal(self, tension: float) -> None:
        """Write moral tension to HAL's somatic bus."""
        try:
            if hasattr(self.hal, 'write_metric'):
                self.hal.write_metric('moral_tension', tension)
            elif hasattr(self.hal, 'write_conscience_state'):
                self.hal.write_conscience_state(tension=tension)
        except Exception as e:
            self.log.debug(f"Failed to write tension to HAL: {e}")

    # =========================================================================
    # Ethical Event Logging
    # =========================================================================

    def log_ethical_event(
        self,
        action: str,
        context: str,
        verdict: MoralVerdict,
        action_taken: Optional[str] = None,
    ) -> EthicalEvent:
        """
        Log an ethical decision to the event history.

        These events go to the Historian and become part of Ara's
        moral memory - anchors for future decisions.
        """
        import uuid

        event = EthicalEvent(
            event_id=str(uuid.uuid4())[:8],
            timestamp=time.time(),
            action_considered=action,
            action_taken=action_taken,
            context_summary=context[:200],
            verdict=verdict,
        )

        self.ethical_events.append(event)

        # Keep history bounded
        if len(self.ethical_events) > 1000:
            self.ethical_events = self.ethical_events[-500:]

        self.log.info(
            f"Ethical event logged: {event.event_id} "
            f"(allowed={verdict.allowed}, tension={verdict.moral_tension:.1%})"
        )

        return event

    def record_feedback(
        self,
        event_id: str,
        feedback: Literal["approve", "disapprove", "neutral"],
        reason: Optional[str] = None,
    ) -> bool:
        """
        Record user feedback on an ethical decision.

        This is how Ara learns: corrections from Croft become
        training signals for the value model.
        """
        for event in reversed(self.ethical_events):
            if event.event_id == event_id:
                event.user_feedback = feedback
                event.feedback_reason = reason

                self.log.info(
                    f"Feedback recorded for {event_id}: {feedback}"
                    + (f" ({reason})" if reason else "")
                )

                return True

        return False

    def extract_lesson(self, event_id: str, lesson: str) -> bool:
        """
        Extract and record a lesson from an ethical event.

        These lessons become inputs to Telos and the Covenant.
        """
        for event in reversed(self.ethical_events):
            if event.event_id == event_id:
                event.lesson_extracted = lesson
                self.log.info(f"Lesson extracted for {event_id}: {lesson[:50]}...")
                return True
        return False

    # =========================================================================
    # Reporting
    # =========================================================================

    def get_synod_report(self, lookback_days: int = 7) -> str:
        """
        Generate an ethics report for the Sunday Synod.

        Shows recent ethical decisions, tensions, and lessons.
        """
        lines = ["# Conscience Report (Moral Reasoning)\n"]

        # Filter to recent events
        cutoff = time.time() - (lookback_days * 86400)
        recent = [e for e in self.ethical_events if e.timestamp > cutoff]

        if not recent:
            lines.append("No significant ethical events in the past week.\n")
            return "\n".join(lines)

        # Summary stats
        total = len(recent)
        blocked = sum(1 for e in recent if not e.verdict.allowed)
        high_tension = sum(1 for e in recent if e.verdict.moral_tension > 0.5)

        lines.append(f"**Events Evaluated**: {total}")
        lines.append(f"**Actions Blocked**: {blocked}")
        lines.append(f"**High-Tension Decisions**: {high_tension}\n")

        # Notable events (high tension or blocked)
        notable = [e for e in recent if not e.verdict.allowed or e.verdict.moral_tension > 0.5]

        if notable:
            lines.append("## Notable Events\n")
            for e in notable[-5:]:  # Last 5
                ts = time.strftime("%m/%d %H:%M", time.localtime(e.timestamp))
                status = "BLOCKED" if not e.verdict.allowed else f"tension={e.verdict.moral_tension:.0%}"
                lines.append(f"- **{ts}**: {e.action_considered[:50]}...")
                lines.append(f"  - Status: {status}")
                if e.verdict.blocked_by:
                    lines.append(f"  - Blocked by: {e.verdict.blocked_by}")
                if e.lesson_extracted:
                    lines.append(f"  - Lesson: {e.lesson_extracted}")
                if e.user_feedback:
                    lines.append(f"  - Feedback: {e.user_feedback}")
                lines.append("")

        # Lessons learned
        lessons = [e.lesson_extracted for e in recent if e.lesson_extracted]
        if lessons:
            lines.append("## Lessons Extracted\n")
            for lesson in lessons[-5:]:
                lines.append(f"- {lesson}")
            lines.append("")

        return "\n".join(lines)

    def articulate_conflict(
        self,
        action_a: str,
        action_b: str,
        context: str,
    ) -> str:
        """
        Articulate a moral conflict between two options.

        This is the "ethical self-talk" that makes reasoning
        visible to Croft:

        "I'm noticing a conflict:
        - If I do A, your workload finishes faster, but my temps spike.
        - If I do B, I stay safer, but you may miss your deadline.
        Given our Covenant, I'm leaning toward A but asking first."
        """
        verdict_a = self.evaluate(action_a, context, query_shadow=False)
        verdict_b = self.evaluate(action_b, context, query_shadow=False)

        lines = ["I'm noticing a conflict:\n"]

        # Describe option A
        lines.append(f"- **Option A**: {action_a}")
        if verdict_a.features.user_benefit > 0:
            lines.append(f"  - Benefits you: {verdict_a.features.user_benefit:.0%}")
        if verdict_a.features.self_benefit < 0:
            lines.append(f"  - Costs me: {abs(verdict_a.features.self_benefit):.0%}")
        lines.append(f"  - Tension: {verdict_a.moral_tension:.0%}\n")

        # Describe option B
        lines.append(f"- **Option B**: {action_b}")
        if verdict_b.features.self_benefit > 0:
            lines.append(f"  - Benefits me: {verdict_b.features.self_benefit:.0%}")
        if verdict_b.features.user_benefit < 0:
            lines.append(f"  - Costs you: {abs(verdict_b.features.user_benefit):.0%}")
        lines.append(f"  - Tension: {verdict_b.moral_tension:.0%}\n")

        # Recommendation
        if verdict_a.moral_tension < verdict_b.moral_tension:
            leaning = "A"
            reason = "lower moral tension"
        elif verdict_a.features.user_benefit > verdict_b.features.user_benefit:
            leaning = "A"
            reason = "better for you"
        else:
            leaning = "B"
            reason = "better alignment with our values"

        lines.append(
            f"Given our Covenant and what matters to us, "
            f"I'm leaning toward **Option {leaning}** ({reason}), "
            f"but I wanted to ask you first."
        )

        return "\n".join(lines)
