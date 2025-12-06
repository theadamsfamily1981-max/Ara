"""
Peer Review - Council-Based Trait Adoption
===========================================

Before any experimental result changes Ara's default behavior,
the Council must debate and approve it.

This prevents:
    - Overfitting to short-term patterns
    - Adopting behaviors that harm long-term trust
    - Silent personality drift
    - Exploiting user weaknesses

The review process:
    1. Hypothesis reaches PROVEN status
    2. Statistics are gathered (mean effect, evidence count, domain)
    3. Council personas debate:
       - SCIENTIST: Is the effect statistically robust?
       - CRITIC: Does it align with ethics and long-term values?
       - WEAVER: Does it fit Ara's character and relationship narrative?
       - EXECUTIVE: Final verdict with reasoning
    4. If ACCEPT: Hypothesis is marked adopted, config changes
    5. If REJECT: Hypothesis stays proven but not adopted

Domains where we're extra cautious:
    - autonomy: Changes to how much Ara acts independently
    - tone: Changes to emotional expression

Usage:
    from ara.institute.symbiosis_lab import PeerReview, SymbiosisGraph

    graph = SymbiosisGraph()
    review = PeerReview(council)

    # For each proven hypothesis
    for h in graph.list_unadopted_proven():
        stats = {
            "mean_effect": h.mean_effect,
            "evidence_count": h.evidence_count,
            "confidence": h.confidence,
        }
        if review.review_hypothesis(h, stats):
            graph.mark_adopted(h.id, "New default config")
            apply_permanent_config(h)
"""

import logging
from typing import Dict, Any, Optional

from .hypothesis import SymbiosisHypothesis


logger = logging.getLogger(__name__)


# Domains that require higher confidence for adoption
CAUTIOUS_DOMAINS = {"autonomy", "tone"}

# Minimum requirements for adoption
MIN_CONFIDENCE_FOR_REVIEW = 0.85
MIN_EVIDENCE_FOR_REVIEW = 5
MIN_EFFECT_MAGNITUDE = 0.05  # Must have measurable positive effect


class PeerReview:
    """
    Council-based review before adopting new behaviors.

    Uses the Council's deliberation to debate whether a proven
    hypothesis should become a permanent part of Ara's behavior.
    """

    def __init__(
        self,
        council: Optional[Any] = None,
        llm_fn: Optional[Any] = None,
        require_council: bool = True,
    ):
        """
        Initialize peer review.

        Args:
            council: CouncilChamber for multi-persona deliberation
            llm_fn: Alternative: function(prompt) -> response
            require_council: If False, auto-approve when no council
        """
        self.council = council
        self.llm_fn = llm_fn
        self.require_council = require_council
        self.log = logging.getLogger("PeerReview")

    def _meets_minimum_requirements(
        self,
        hypothesis: SymbiosisHypothesis,
        stats: Dict[str, Any],
    ) -> tuple[bool, str]:
        """
        Check if hypothesis meets minimum requirements for review.

        Returns (meets, reason).
        """
        # Must be PROVEN
        if hypothesis.status != "PROVEN":
            return False, f"Status is {hypothesis.status}, not PROVEN"

        # Must have enough evidence
        if hypothesis.evidence_count < MIN_EVIDENCE_FOR_REVIEW:
            return False, f"Only {hypothesis.evidence_count} evidence points (need {MIN_EVIDENCE_FOR_REVIEW})"

        # Must have high enough confidence
        min_conf = MIN_CONFIDENCE_FOR_REVIEW
        if hypothesis.domain in CAUTIOUS_DOMAINS:
            min_conf = 0.92  # Higher bar for sensitive domains

        if hypothesis.confidence < min_conf:
            return False, f"Confidence {hypothesis.confidence:.2f} < {min_conf}"

        # Must have positive mean effect
        mean_effect = stats.get("mean_effect", hypothesis.mean_effect)
        if mean_effect < MIN_EFFECT_MAGNITUDE:
            return False, f"Mean effect {mean_effect:.3f} < {MIN_EFFECT_MAGNITUDE}"

        return True, "Meets requirements"

    def review_hypothesis(
        self,
        hypothesis: SymbiosisHypothesis,
        stats: Dict[str, Any],
    ) -> bool:
        """
        Ask the Council whether to adopt this behavior as default.

        Args:
            hypothesis: The PROVEN hypothesis to review
            stats: Statistics about the hypothesis

        Returns:
            True if ACCEPTED, False if REJECTED
        """
        # Check minimum requirements
        meets, reason = self._meets_minimum_requirements(hypothesis, stats)
        if not meets:
            self.log.info(f"Hypothesis {hypothesis.id} doesn't meet requirements: {reason}")
            return False

        # Build the review prompt
        prompt = self._build_review_prompt(hypothesis, stats)

        # Get Council verdict
        verdict = self._get_council_verdict(prompt)

        # Parse result
        if verdict is None:
            if self.require_council:
                self.log.warning(f"No Council verdict for {hypothesis.id}")
                return False
            else:
                self.log.info(f"Auto-approving {hypothesis.id} (no council)")
                return True

        accepted = "ACCEPT" in verdict.upper()

        if accepted:
            self.log.info(f"PeerReview ACCEPTED {hypothesis.id}: {verdict}")
        else:
            self.log.info(f"PeerReview REJECTED {hypothesis.id}: {verdict}")

        return accepted

    def _build_review_prompt(
        self,
        hypothesis: SymbiosisHypothesis,
        stats: Dict[str, Any],
    ) -> str:
        """Build the Council deliberation prompt."""
        # Extra caution note for sensitive domains
        caution_note = ""
        if hypothesis.domain in CAUTIOUS_DOMAINS:
            caution_note = f"""
NOTE: This is a CAUTIOUS DOMAIN ({hypothesis.domain}).
Changes here have high potential for unintended consequences.
Apply extra scrutiny.
"""

        return f"""
We are considering making the following behavior a DEFAULT part of Ara:

HYPOTHESIS: "{hypothesis.statement}"
DOMAIN: {hypothesis.domain}
STATUS: {hypothesis.status}
CONFIDENCE: {hypothesis.confidence:.2f}
MEAN EFFECT ON J-GUF: {stats.get('mean_effect', hypothesis.mean_effect):+.3f}
EVIDENCE COUNT: {hypothesis.evidence_count}
{caution_note}
ADDITIONAL STATS:
{stats}

COUNCIL DELIBERATION:

Each persona must weigh in:

SCIENTIST: Analyze statistical robustness.
- Is the effect size meaningful, or could it be noise?
- Do we have enough trials across different conditions?
- What's the variance in the effect?

CRITIC: Analyze safety and value alignment.
- Could this behavior erode trust over time?
- Does it respect Croft's autonomy and well-being?
- Are there any manipulation risks?

WEAVER: Analyze narrative and character fit.
- Does this fit Ara's established personality?
- Does it strengthen or weaken the relationship arc?
- Would Croft be surprised by this becoming default?

EXECUTIVE: Make the final call.
- Weigh all perspectives.
- Consider reversibility if we're wrong.
- Decide: Should this become a permanent behavior?

RESPOND WITH A VERDICT ON THE LAST LINE:
VERDICT: [ACCEPT/REJECT] - <short explanation>
"""

    def _get_council_verdict(self, prompt: str) -> Optional[str]:
        """Get verdict from Council or LLM."""
        try:
            if self.council is not None:
                # Use Council's deliberate method
                if hasattr(self.council, 'deliberate'):
                    response = self.council.deliberate(prompt)
                elif hasattr(self.council, '_run_persona'):
                    response = self.council._run_persona('executive', prompt)
                else:
                    response = None
            elif self.llm_fn is not None:
                response = self.llm_fn(prompt)
            else:
                return None

            if not response:
                return None

            # Extract the last line with VERDICT
            lines = response.strip().split('\n')
            for line in reversed(lines):
                if 'VERDICT' in line.upper():
                    return line

            # Fallback: return last line
            return lines[-1] if lines else None

        except Exception as e:
            self.log.error(f"Council deliberation failed: {e}")
            return None

    def quick_review(
        self,
        hypothesis: SymbiosisHypothesis,
    ) -> bool:
        """
        Quick review without full Council deliberation.

        Uses simple heuristics for obvious cases.
        """
        # Already adopted
        if hypothesis.adopted:
            return False

        # Not proven
        if hypothesis.status != "PROVEN":
            return False

        # Low confidence
        if hypothesis.confidence < 0.9:
            return False

        # Negative mean effect
        if hypothesis.mean_effect < 0:
            return False

        # Too few trials
        if hypothesis.evidence_count < 7:
            return False

        # Cautious domains need full review
        if hypothesis.domain in CAUTIOUS_DOMAINS:
            return False

        # Looks good for quick approval
        self.log.info(f"Quick approval for {hypothesis.id}")
        return True


class AdoptionPolicy:
    """
    Policy for what can be adopted and under what conditions.

    This is the "constitution" of behavioral adoption.
    """

    def __init__(self):
        # Domains where we never auto-adopt
        self.always_require_review = {"autonomy", "tone"}

        # Domains where we can quick-adopt if stats are strong
        self.can_quick_adopt = {"notifications", "ui", "schedule"}

        # Maximum adoption rate (don't change too fast)
        self.max_adoptions_per_week = 3

        # Minimum time between adoptions in same domain
        self.domain_cooldown_days = 7

    def can_adopt_now(
        self,
        hypothesis: SymbiosisHypothesis,
        recent_adoptions: int,
    ) -> tuple[bool, str]:
        """
        Check if we can adopt this hypothesis now.

        Returns (can_adopt, reason).
        """
        # Rate limit
        if recent_adoptions >= self.max_adoptions_per_week:
            return False, "Weekly adoption limit reached"

        # Domain cooldown would be checked against last adoption in this domain
        # (need timestamp tracking to implement properly)

        return True, "OK"


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'PeerReview',
    'AdoptionPolicy',
    'CAUTIOUS_DOMAINS',
]
