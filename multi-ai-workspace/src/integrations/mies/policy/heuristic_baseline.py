"""MIES Heuristic Baseline Policy.

A rules-based modality policy that approximates the MIES design monograph.
This works immediately without training and serves as:
1. A working baseline for immediate use
2. A reference implementation for the learned policy to match
3. A fallback when the learned policy is uncertain

Key Rules:
- Meetings are sacred: almost never speak, at most silent overlay
- Deep work: prefer text-only or silent
- Casual contexts: allow audio whisper or mini avatar
- High urgency can override context restrictions
- User requests always honored
- Thermodynamic state throttles high-energy modes

The policy computes scores for each candidate mode and selects the best.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import logging

from ..context import ModalityContext, ActivityType, ForegroundAppType
from ..modes import (
    ModalityMode,
    ModalityDecision,
    TransitionParams,
    DEFAULT_MODES,
    MODE_SILENT,
    MODE_TEXT_INLINE,
    MODE_TEXT_MINIMAL,
    MODE_TEXT_SIDE,
    MODE_AUDIO_WHISPER,
    MODE_AUDIO_NORMAL,
    MODE_AVATAR_SUBTLE,
    MODE_AVATAR_FULL,
)

logger = logging.getLogger(__name__)


@dataclass
class PolicyConfig:
    """Configuration for the heuristic policy."""
    # Friction weights
    w_social_friction: float = 1.0
    w_urgency: float = 0.7
    w_autonomy: float = 0.3
    w_continuity: float = 0.2  # Preference for staying in same mode

    # Thresholds
    min_urgency_for_audio: float = 0.5
    min_urgency_for_avatar: float = 0.7
    max_intrusiveness_in_meeting: float = 0.1
    max_intrusiveness_deep_work: float = 0.3

    # Autonomy (liveness) parameters
    autonomy_buildup_minutes: float = 10.0  # Time to max autonomy pressure
    autonomy_max_mode_intrusiveness: float = 0.25  # Cap for autonomous actions

    # Thermodynamic limits
    energy_threshold_for_avatar: float = 0.3
    hot_state_max_intrusiveness: float = 0.4


class HeuristicModalityPolicy:
    """
    Rules-based modality selection policy.

    Implements the MIES etiquette rules through explicit score functions.
    This is the "baby MIES" that works immediately.
    """

    def __init__(self, config: Optional[PolicyConfig] = None):
        self.config = config or PolicyConfig()
        self._last_decision: Optional[ModalityDecision] = None

    def select_modality(
        self,
        ctx: ModalityContext,
        prev_mode: Optional[ModalityMode] = None,
    ) -> ModalityDecision:
        """
        Select the best modality for the current context.

        Args:
            ctx: Current modality context
            prev_mode: Previous mode (for smooth transitions)

        Returns:
            ModalityDecision with chosen mode and rationale
        """
        # Get candidate modes
        candidates = self._get_candidate_modes(ctx)

        if not candidates:
            # Fallback to silent
            return ModalityDecision(
                mode=MODE_SILENT,
                rationale="No valid candidates, defaulting to silent",
                confidence=1.0,
            )

        # Score each candidate
        scores: Dict[str, float] = {}
        for mode in candidates:
            scores[mode.name] = self._score_mode(ctx, mode, prev_mode)

        # Select best
        best_mode_name = min(scores, key=scores.get)  # Lower is better
        best_mode = next(m for m in candidates if m.name == best_mode_name)
        best_score = scores[best_mode_name]

        # Compute confidence from score spread
        all_scores = list(scores.values())
        if len(all_scores) > 1:
            score_range = max(all_scores) - min(all_scores)
            confidence = min(1.0, score_range / 2.0)  # Higher spread = more confident
        else:
            confidence = 1.0

        # Build rationale
        rationale = self._build_rationale(ctx, best_mode, scores)

        # Determine transition parameters
        transition = self._compute_transition(prev_mode, best_mode)

        # Check if we need permission
        should_ask, permission_prompt = self._check_permission_needed(ctx, best_mode)

        decision = ModalityDecision(
            mode=best_mode,
            transition=transition,
            should_ask_permission=should_ask,
            permission_prompt=permission_prompt,
            rationale=rationale,
            confidence=confidence,
            energy_score=best_score,
            alternatives_considered=[m.name for m in candidates if m.name != best_mode_name],
        )

        self._last_decision = decision
        return decision

    def _get_candidate_modes(self, ctx: ModalityContext) -> List[ModalityMode]:
        """Get candidate modes based on context constraints."""
        candidates = []

        # Always consider silent
        candidates.append(MODE_SILENT)

        # If no content to deliver, only silent/background are valid
        if ctx.info_urgency < 0.01 and not ctx.is_user_requested:
            candidates.append(DEFAULT_MODES["avatar_subtle"])  # For liveness
            return candidates

        # Text modes are almost always valid
        candidates.append(MODE_TEXT_INLINE)
        candidates.append(MODE_TEXT_MINIMAL)
        candidates.append(MODE_TEXT_SIDE)

        # Audio modes - gatekept by context
        if self._audio_allowed(ctx):
            candidates.append(MODE_AUDIO_WHISPER)
            if ctx.info_urgency > self.config.min_urgency_for_audio or ctx.is_user_requested:
                candidates.append(MODE_AUDIO_NORMAL)

        # Avatar modes - gatekept by energy and context
        if self._avatar_allowed(ctx):
            candidates.append(DEFAULT_MODES["avatar_subtle"])
            if ctx.info_urgency > self.config.min_urgency_for_avatar or ctx.is_user_requested:
                candidates.append(DEFAULT_MODES["avatar_present"])
                if ctx.energy_remaining > 0.5:
                    candidates.append(MODE_AVATAR_FULL)

        return candidates

    def _audio_allowed(self, ctx: ModalityContext) -> bool:
        """Check if audio output is allowed in current context."""
        # Never during meetings
        if ctx.activity == ActivityType.MEETING:
            return False

        # Not if mic is in use (might be recording)
        if ctx.audio.mic_in_use:
            return False

        # Not during voice calls
        if ctx.audio.has_voice_call:
            return False

        # OK otherwise
        return True

    def _avatar_allowed(self, ctx: ModalityContext) -> bool:
        """Check if avatar overlay is allowed in current context."""
        # Not if energy is too low
        if ctx.energy_remaining < self.config.energy_threshold_for_avatar:
            return False

        # Not if system is overheating
        if ctx.thermal_state == "OVERHEATING":
            return False

        # Not during fullscreen games (unless urgent)
        if ctx.foreground.is_fullscreen and ctx.activity == ActivityType.GAMING:
            if ctx.info_urgency < 0.8:
                return False

        return True

    def _score_mode(
        self,
        ctx: ModalityContext,
        mode: ModalityMode,
        prev_mode: Optional[ModalityMode],
    ) -> float:
        """
        Compute energy score for a mode (lower is better).

        E(M, S) = w_friction * E_friction + w_urgency * E_urgency + w_autonomy * E_autonomy
        """
        friction = self._social_friction(ctx, mode)
        urgency = self._urgency_energy(ctx, mode)
        autonomy = self._autonomy_energy(ctx)
        continuity = self._continuity_bonus(mode, prev_mode)

        total = (
            self.config.w_social_friction * friction +
            self.config.w_urgency * urgency +
            self.config.w_autonomy * autonomy -
            self.config.w_continuity * continuity
        )

        return total

    def _social_friction(self, ctx: ModalityContext, mode: ModalityMode) -> float:
        """
        E_friction(M, S) - Social friction cost.

        Higher when the mode clashes with context (rudeness penalty).
        """
        friction = 0.0

        # === Meeting friction ===
        if ctx.activity == ActivityType.MEETING:
            # Audio is extremely rude
            if mode.channel.name.startswith("AUDIO"):
                friction += 5.0
            # Avatar is quite rude
            elif mode.channel.name.startswith("AVATAR"):
                friction += 3.0
            # Even text can be distracting
            elif mode.intrusiveness > 0.2:
                friction += 1.0

        # === Deep work friction ===
        elif ctx.activity == ActivityType.DEEP_WORK:
            # Audio is rude
            if mode.channel.name.startswith("AUDIO"):
                friction += 2.5
            # Large avatar is rude
            if mode.channel.name.startswith("AVATAR") and mode.avatar_size > 0.2:
                friction += 2.0
            # Any intrusiveness is costly
            friction += mode.intrusiveness * 1.5

        # === Gaming friction ===
        elif ctx.activity == ActivityType.GAMING:
            # Fullscreen: overlays are very costly
            if ctx.foreground.is_fullscreen:
                if mode.channel.name.startswith("AVATAR"):
                    friction += 2.0
            # Audio can be OK if user isn't in voice chat
            if mode.channel.name.startswith("AUDIO") and ctx.audio.has_voice_call:
                friction += 3.0

        # === High cognitive load ===
        if ctx.user_cognitive_load > 0.7:
            # Audio is extra costly
            if mode.channel.name.startswith("AUDIO"):
                friction += 1.5
            # High intrusiveness is costly
            friction += mode.intrusiveness * ctx.user_cognitive_load

        # === Negative valence (user seems upset) ===
        if ctx.valence < -0.5:
            # Be more gentle
            if mode.intrusiveness > 0.4:
                friction += 1.0

        # === Thermodynamic constraints ===
        if ctx.thermal_state == "HOT":
            friction += mode.energy_cost * 1.5
        elif ctx.thermal_state == "OVERHEATING":
            friction += mode.energy_cost * 3.0

        return friction

    def _urgency_energy(self, ctx: ModalityContext, mode: ModalityMode) -> float:
        """
        E_urgency(I, M) - Urgency modifier.

        Negative when urgency is high and mode can deliver (lowers total energy).
        """
        # Base urgency benefit
        urgency_benefit = ctx.info_urgency * 3.0

        # User request bonus
        if ctx.is_user_requested:
            urgency_benefit += 2.0

        # Deadline pressure
        if ctx.deadline_seconds is not None and ctx.deadline_seconds < 60:
            urgency_benefit += 2.0

        # Mode must have sufficient bandwidth to benefit from urgency
        bandwidth_match = min(1.0, mode.bandwidth_cost / max(0.1, ctx.info_urgency))

        return -urgency_benefit * bandwidth_match

    def _autonomy_energy(self, ctx: ModalityContext) -> float:
        """
        E_autonomy(t) - Boredom/liveness term.

        The longer Ara has been quiet, the more pressure to do something gentle.
        But this should only enable very low-intrusiveness actions.
        """
        # Time since last utterance, normalized
        minutes_quiet = ctx.seconds_since_last_utterance / 60.0
        buildup = min(1.0, minutes_quiet / self.config.autonomy_buildup_minutes)

        # This energy is positive (penalty for NOT acting)
        # But we only want this to unlock very gentle actions
        return buildup * 0.5  # Gentle pressure

    def _continuity_bonus(
        self,
        mode: ModalityMode,
        prev_mode: Optional[ModalityMode],
    ) -> float:
        """Bonus for staying in or near the previous mode (smooth transitions)."""
        if prev_mode is None:
            return 0.0

        if mode.name == prev_mode.name:
            return 0.3  # Same mode bonus

        # Proximity bonus
        distance = mode.distance_to(prev_mode)
        return max(0.0, 0.2 - distance * 0.1)

    def _compute_transition(
        self,
        prev_mode: Optional[ModalityMode],
        new_mode: ModalityMode,
    ) -> TransitionParams:
        """Compute smooth transition parameters."""
        if prev_mode is None:
            return TransitionParams(duration_ms=200)

        # Longer transition for bigger jumps
        distance = new_mode.distance_to(prev_mode)
        duration = int(200 + distance * 300)  # 200-500ms

        # Fade out first if switching channels
        fade_out_first = prev_mode.channel != new_mode.channel

        return TransitionParams(
            duration_ms=duration,
            easing="ease-out",
            fade_out_first=fade_out_first,
        )

    def _check_permission_needed(
        self,
        ctx: ModalityContext,
        mode: ModalityMode,
    ) -> tuple[bool, Optional[str]]:
        """Check if we should ask permission before this mode."""
        # High intrusiveness in sensitive contexts
        if mode.intrusiveness > 0.5:
            if ctx.activity == ActivityType.DEEP_WORK:
                return True, "You seem focused. May I interrupt briefly?"

        # Avatar in meeting (rare but possible with high urgency)
        if ctx.activity == ActivityType.MEETING:
            if mode.channel.name.startswith("AVATAR"):
                return True, "I see you're in a meeting. This seems urgent - may I show you?"

        return False, None

    def _build_rationale(
        self,
        ctx: ModalityContext,
        mode: ModalityMode,
        scores: Dict[str, float],
    ) -> str:
        """Build human-readable rationale for logging."""
        parts = []

        # Context summary
        parts.append(f"Activity={ctx.activity.name}")
        if ctx.audio.mic_in_use:
            parts.append("mic_active")
        if ctx.foreground.is_fullscreen:
            parts.append("fullscreen")

        # Decision
        parts.append(f"-> {mode.name}")
        parts.append(f"(score={scores[mode.name]:.2f})")

        return " ".join(parts)


# === Convenience ===

def create_heuristic_policy(
    config: Optional[PolicyConfig] = None,
) -> HeuristicModalityPolicy:
    """Create a heuristic policy with optional custom config."""
    return HeuristicModalityPolicy(config)


__all__ = [
    "HeuristicModalityPolicy",
    "PolicyConfig",
    "create_heuristic_policy",
]
