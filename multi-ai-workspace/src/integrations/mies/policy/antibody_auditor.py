"""
MIES Antibody Auditor - The Straight-Face Sanity Check
-------------------------------------------------------

A tiny, boring, stone-cold process whose only job is to ask:
"Does this still sound sane?"

This is the last line of defense between the ThermodynamicGovernor's
decision and actual execution. It catches edge cases the energy
function might miss.

Checks:
1. Consistency: No sudden jumps from GHOST → GODDESS without justification
2. History: Did user close/dismiss this mode in similar context before?
3. Budgets: CPU, thermal, social (interruptions per hour)
4. Hard invariants: No avatar during calls, no audio when mic is hot

Output:
- Approved: proceed with decision
- Downgraded: decision modified to less intrusive mode
- Vetoed: replaced with SILENT/LOG mode

This is the "antibody" - learned and hard-coded immune responses
that protect the user from Ara's enthusiasm.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from collections import deque
from enum import Enum, auto

from ..context import ModalityContext, ActivityType, ForegroundAppType
from ..modes import ModalityMode, ModalityDecision, DEFAULT_MODES, MODE_SILENT
from ..history import InteractionHistory, OutcomeType

logger = logging.getLogger(__name__)


class AuditVerdict(Enum):
    """Result of antibody audit."""
    APPROVED = auto()      # Decision is sane, proceed
    DOWNGRADED = auto()    # Reduced intrusiveness
    VETOED = auto()        # Replaced with minimal mode


@dataclass
class AuditResult:
    """Result of the antibody review."""
    verdict: AuditVerdict
    original_mode: str
    final_mode: str
    reasons: List[str] = field(default_factory=list)
    near_miss: bool = False  # True if this was close to a bad decision


@dataclass
class SocialBudget:
    """Tracks social interaction budget."""
    max_interruptions_per_hour: int = 5
    max_intrusive_per_hour: int = 2  # intrusiveness > 0.5
    cooldown_after_dismiss_seconds: float = 300.0  # 5 min after user dismisses

    # Tracking
    interruption_times: deque = field(default_factory=lambda: deque(maxlen=100))
    intrusive_times: deque = field(default_factory=lambda: deque(maxlen=100))
    last_dismiss_time: float = 0.0

    def record_interruption(self, intrusive: bool = False):
        """Record an interruption."""
        now = time.time()
        self.interruption_times.append(now)
        if intrusive:
            self.intrusive_times.append(now)

    def record_dismiss(self):
        """Record user dismissing Ara."""
        self.last_dismiss_time = time.time()

    def interruptions_last_hour(self) -> int:
        """Count interruptions in the last hour."""
        cutoff = time.time() - 3600
        return sum(1 for t in self.interruption_times if t > cutoff)

    def intrusive_last_hour(self) -> int:
        """Count intrusive interruptions in the last hour."""
        cutoff = time.time() - 3600
        return sum(1 for t in self.intrusive_times if t > cutoff)

    def in_cooldown(self) -> bool:
        """Are we in post-dismiss cooldown?"""
        return (time.time() - self.last_dismiss_time) < self.cooldown_after_dismiss_seconds

    def budget_exceeded(self) -> Tuple[bool, str]:
        """Check if any budget is exceeded."""
        if self.interruptions_last_hour() >= self.max_interruptions_per_hour:
            return True, f"Hit interruption limit ({self.max_interruptions_per_hour}/hr)"
        if self.intrusive_last_hour() >= self.max_intrusive_per_hour:
            return True, f"Hit intrusive limit ({self.max_intrusive_per_hour}/hr)"
        if self.in_cooldown():
            return True, "In post-dismiss cooldown"
        return False, ""


class AntibodyAuditor:
    """
    The Straight-Face Sanity Check.

    Reviews every ModalityDecision before execution.
    Can approve, downgrade, or veto decisions.

    Usage:
        auditor = AntibodyAuditor(history=history)
        result = auditor.review(ctx, decision)
        if result.verdict != AuditVerdict.APPROVED:
            decision = result.modified_decision
    """

    # Hard intrusiveness thresholds
    INTRUSIVE_THRESHOLD = 0.5
    VERY_INTRUSIVE_THRESHOLD = 0.7

    # Jump detection
    MAX_INTRUSIVENESS_JUMP = 0.4  # Can't jump more than this without justification

    def __init__(
        self,
        history: Optional[InteractionHistory] = None,
        social_budget: Optional[SocialBudget] = None,
        strict_mode: bool = False,
    ):
        """
        Initialize the auditor.

        Args:
            history: InteractionHistory for learned antibodies
            social_budget: Budget tracker for interruptions
            strict_mode: If True, more aggressive about vetoing
        """
        self.history = history
        self.social_budget = social_budget or SocialBudget()
        self.strict_mode = strict_mode

        # Track recent decisions for consistency checking
        self._recent_modes: deque = deque(maxlen=10)
        self._recent_contexts: deque = deque(maxlen=10)

    def review(
        self,
        ctx: ModalityContext,
        decision: ModalityDecision,
    ) -> Tuple[ModalityDecision, AuditResult]:
        """
        Review a modality decision.

        Returns:
            (possibly_modified_decision, audit_result)
        """
        reasons = []
        verdict = AuditVerdict.APPROVED
        original_mode = decision.mode.name
        final_mode = decision.mode

        # === Hard Invariants (always enforced) ===
        hard_veto, hard_reason = self._check_hard_invariants(ctx, decision)
        if hard_veto:
            final_mode = MODE_SILENT
            verdict = AuditVerdict.VETOED
            reasons.append(f"HARD VETO: {hard_reason}")

        # === Consistency Check ===
        if verdict == AuditVerdict.APPROVED:
            inconsistent, inc_reason = self._check_consistency(ctx, decision)
            if inconsistent:
                # Downgrade to previous level or text
                final_mode = self._get_safe_downgrade(decision.mode)
                verdict = AuditVerdict.DOWNGRADED
                reasons.append(f"Consistency: {inc_reason}")

        # === History Check (learned antibodies) ===
        if verdict == AuditVerdict.APPROVED and self.history:
            antibody_active, ab_reason = self._check_antibodies(ctx, decision)
            if antibody_active:
                final_mode = self._get_safe_downgrade(decision.mode)
                verdict = AuditVerdict.DOWNGRADED
                reasons.append(f"Antibody: {ab_reason}")

        # === Budget Check ===
        if verdict == AuditVerdict.APPROVED:
            over_budget, budget_reason = self._check_budgets(ctx, decision)
            if over_budget:
                final_mode = DEFAULT_MODES.get("text_minimal", MODE_SILENT)
                verdict = AuditVerdict.DOWNGRADED
                reasons.append(f"Budget: {budget_reason}")

        # === Context-Specific Rules ===
        if verdict == AuditVerdict.APPROVED:
            ctx_issue, ctx_reason = self._check_context_rules(ctx, decision)
            if ctx_issue:
                final_mode = self._get_safe_downgrade(decision.mode)
                verdict = AuditVerdict.DOWNGRADED
                reasons.append(f"Context: {ctx_reason}")

        # Build result
        near_miss = (
            verdict != AuditVerdict.APPROVED and
            decision.mode.intrusiveness > self.INTRUSIVE_THRESHOLD
        )

        result = AuditResult(
            verdict=verdict,
            original_mode=original_mode,
            final_mode=final_mode.name,
            reasons=reasons,
            near_miss=near_miss,
        )

        # Log near-misses for learning
        if near_miss:
            logger.warning(
                f"Near-miss caught: {original_mode} → {final_mode.name} "
                f"in {ctx.activity.name} | {'; '.join(reasons)}"
            )

        # Modify decision if needed
        if verdict != AuditVerdict.APPROVED:
            decision = ModalityDecision(
                mode=final_mode,
                transition=decision.transition,
                rationale=decision.rationale + f" | Auditor: {'; '.join(reasons)}",
                confidence=decision.confidence * 0.8,  # Reduced confidence
                energy_score=decision.energy_score,
                alternatives_considered=decision.alternatives_considered,
            )

        # Update tracking
        self._recent_modes.append(final_mode.name)
        self._recent_contexts.append(ctx.activity.name)

        # Record to social budget if not silent
        if final_mode.name != "silent":
            self.social_budget.record_interruption(
                intrusive=final_mode.intrusiveness > self.INTRUSIVE_THRESHOLD
            )

        return decision, result

    def _check_hard_invariants(
        self,
        ctx: ModalityContext,
        decision: ModalityDecision,
    ) -> Tuple[bool, str]:
        """Check hard invariants that always result in veto."""
        mode = decision.mode
        is_audio = mode.channel.name.startswith("AUDIO")
        is_avatar = mode.channel.name.startswith("AVATAR")

        # No audio when mic is in use (always)
        if is_audio and ctx.audio.mic_in_use:
            return True, "Audio while mic in use"

        # No audio/avatar during video calls (always)
        if ctx.audio.has_voice_call and (is_audio or is_avatar):
            return True, "Audio/avatar during voice call"

        # No avatar when system in AGONY (thermal emergency)
        if ctx.system_phys and ctx.system_phys.pain_signal > 0.9:
            if is_avatar or mode.energy_cost > 0.3:
                return True, "High-energy mode during thermal emergency"

        # No intrusive modes when user explicitly in DND
        # (would need DND signal in context, placeholder)

        return False, ""

    def _check_consistency(
        self,
        ctx: ModalityContext,
        decision: ModalityDecision,
    ) -> Tuple[bool, str]:
        """Check for suspicious jumps in intrusiveness."""
        if not self._recent_modes:
            return False, ""

        # Get previous mode's intrusiveness
        prev_mode_name = self._recent_modes[-1]
        prev_mode = DEFAULT_MODES.get(prev_mode_name)
        if not prev_mode:
            return False, ""

        current_intrusiveness = decision.mode.intrusiveness
        prev_intrusiveness = prev_mode.intrusiveness
        jump = current_intrusiveness - prev_intrusiveness

        # Check for sudden large jump
        if jump > self.MAX_INTRUSIVENESS_JUMP:
            # Is it justified?
            justified = (
                ctx.is_user_requested or
                ctx.info_urgency > 0.7 or
                "URGENT" in decision.rationale.upper() or
                "USER_REQUESTED" in decision.rationale.upper()
            )
            if not justified:
                return True, f"Unjustified jump +{jump:.2f} ({prev_mode_name}→{decision.mode.name})"

        return False, ""

    def _check_antibodies(
        self,
        ctx: ModalityContext,
        decision: ModalityDecision,
    ) -> Tuple[bool, str]:
        """Check learned antibodies from interaction history."""
        if not self.history:
            return False, ""

        # Get friction for this mode in this context
        friction = self.history.friction_for(ctx, decision.mode.name)

        # Strong antibody triggers downgrade
        if friction > 1.0:
            return True, f"Learned aversion (friction={friction:.2f})"

        # Moderate antibody + intrusive mode = downgrade
        if friction > 0.5 and decision.mode.intrusiveness > self.INTRUSIVE_THRESHOLD:
            return True, f"Moderate aversion + intrusive (friction={friction:.2f})"

        return False, ""

    def _check_budgets(
        self,
        ctx: ModalityContext,
        decision: ModalityDecision,
    ) -> Tuple[bool, str]:
        """Check social and resource budgets."""
        # Social budget
        exceeded, reason = self.social_budget.budget_exceeded()
        if exceeded and decision.mode.intrusiveness > 0.2:
            return True, reason

        # Thermal budget (if system is stressed, reduce presence)
        if ctx.system_phys:
            if ctx.system_phys.thermal_headroom < 0.2:
                if decision.mode.energy_cost > 0.3:
                    return True, "Thermal budget exceeded"
            if ctx.system_phys.energy_reserve < 0.2:
                if decision.mode.energy_cost > 0.2:
                    return True, "Energy budget depleted"

        return False, ""

    def _check_context_rules(
        self,
        ctx: ModalityContext,
        decision: ModalityDecision,
    ) -> Tuple[bool, str]:
        """Check context-specific rules."""
        mode = decision.mode
        is_avatar = mode.channel.name.startswith("AVATAR")

        # Deep work + intrusive without user request
        if ctx.activity == ActivityType.DEEP_WORK:
            if mode.intrusiveness > self.INTRUSIVE_THRESHOLD:
                if not ctx.is_user_requested:
                    return True, "Intrusive during deep work without request"

        # Gaming + avatar overlay
        if ctx.activity == ActivityType.GAMING:
            if is_avatar and ctx.foreground.is_fullscreen:
                return True, "Avatar overlay in fullscreen game"

        # IDE + high intrusiveness (the classic case)
        if ctx.foreground.app_type == ForegroundAppType.IDE:
            if mode.intrusiveness > 0.6 and not ctx.is_user_requested:
                return True, "High intrusiveness in IDE without request"

        # Strict mode: extra conservative
        if self.strict_mode:
            if mode.intrusiveness > 0.4 and not ctx.is_user_requested:
                return True, "Strict mode: any intrusive action requires request"

        return False, ""

    def _get_safe_downgrade(self, mode: ModalityMode) -> ModalityMode:
        """Get a safe downgrade for a mode."""
        # Downgrade hierarchy
        downgrades = {
            "avatar_full": "avatar_subtle",
            "avatar_present": "avatar_subtle",
            "avatar_subtle": "text_inline",
            "audio_emphatic": "audio_whisper",
            "audio_normal": "audio_whisper",
            "audio_whisper": "text_inline",
            "text_inline": "text_minimal",
            "text_minimal": "silent",
        }

        downgrade_name = downgrades.get(mode.name, "text_minimal")
        return DEFAULT_MODES.get(downgrade_name, MODE_SILENT)

    def record_user_dismiss(self):
        """Record that user dismissed Ara (for cooldown tracking)."""
        self.social_budget.record_dismiss()

    def get_stats(self) -> Dict:
        """Get auditor statistics."""
        return {
            "interruptions_last_hour": self.social_budget.interruptions_last_hour(),
            "intrusive_last_hour": self.social_budget.intrusive_last_hour(),
            "in_cooldown": self.social_budget.in_cooldown(),
            "recent_modes": list(self._recent_modes),
        }


# === Convenience ===

def create_antibody_auditor(
    history: Optional[InteractionHistory] = None,
    strict: bool = False,
    max_interruptions: int = 5,
) -> AntibodyAuditor:
    """Create an antibody auditor with custom settings."""
    budget = SocialBudget(
        max_interruptions_per_hour=max_interruptions,
        max_intrusive_per_hour=max(1, max_interruptions // 3),
    )
    return AntibodyAuditor(
        history=history,
        social_budget=budget,
        strict_mode=strict,
    )


__all__ = [
    "AntibodyAuditor",
    "AuditVerdict",
    "AuditResult",
    "SocialBudget",
    "create_antibody_auditor",
]
