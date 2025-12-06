"""
Symbiotic Orchestrator - The Mirror
====================================

Re-weights Council members to COMPLEMENT the user's current state.

This is where adaptive behavior lives:
    - If you're scattered, she tightens (EXECUTIVE/CRITIC up, MUSE down)
    - If you're stuck, she gets playful (MUSE up)
    - If you're frustrated, she gets supportive and action-oriented
    - If you're in flow, she disappears

The goal is not to match your state but to COMPLEMENT it.
A dance, not an echo.

Integration:
    MindReader → UserState → SymbioticOrchestrator → Council weights/style

Usage:
    from ara.user.mind_reader import MindReader
    from ara.collab.symbiotic_orchestrator import SymbioticOrchestrator
    from ara.collab.council import Council

    mind_reader = MindReader()
    council = Council()
    orchestrator = SymbioticOrchestrator(council)

    # In your turn loop:
    user_state = mind_reader.update_from_text(user_message)
    orchestrator.adjust_council(user_state)
    response = council.convene(user_message, user_state=user_state.as_dict())

Council Personas (from your existing setup):
    - MUSE: Creative, exploratory, playful
    - CRITIC: Analytical, structured, cautious
    - EXECUTIVE: Action-oriented, decisive, efficient
    - WEAVER: Narrative, contextual, integrative
    - SCIENTIST: Empirical, hypothesis-driven, curious
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional, Protocol, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class UserState:
    """
    User state from MindReader.

    Defined here as a protocol match so we don't require hard import.
    """
    cognitive_load: float = 0.5
    emotional_valence: float = 0.0
    intent_clarity: float = 0.5
    fatigue: float = 0.3


class CouncilProtocol(Protocol):
    """Protocol for Council interface."""

    def set_weight(self, member_name: str, weight: float) -> None:
        """Set the weight for a council member."""
        ...

    def set_style(self, tone: str = "neutral", verbosity: str = "medium") -> None:
        """Set response style."""
        ...


@dataclass
class AdaptationState:
    """
    Current adaptation state for debugging/visualization.
    """
    stance: str                      # "grounding", "exploratory", "supportive", "minimal"
    dominant_member: str             # Which council member is most active
    tone: str                        # "calm", "playful", "supportive", "neutral"
    verbosity: str                   # "low", "medium", "high"
    weights: Dict[str, float]        # Current weights
    reason: str                      # Why this adaptation was chosen


class SymbioticOrchestrator:
    """
    Re-weights Council members to COMPLEMENT the user.

    The mirror: instead of matching your state, Ara adapts to balance it.
    This is the core of symbiotic interaction style.
    """

    # Default weights (balanced)
    DEFAULT_WEIGHTS = {
        "MUSE": 0.3,
        "CRITIC": 0.3,
        "EXECUTIVE": 0.3,
        "WEAVER": 0.1,
        "SCIENTIST": 0.0,
    }

    def __init__(self, council: Optional[Any] = None):
        """
        Initialize the orchestrator.

        Args:
            council: Council instance (can be set later)
        """
        self.council = council
        self.log = logging.getLogger("SymbioticOrchestrator")

        # Current adaptation state
        self._current_state: Optional[AdaptationState] = None

        # History for debugging
        self._adaptation_history: List[AdaptationState] = []

    def set_council(self, council: Any) -> None:
        """Set the council instance."""
        self.council = council

    @property
    def current_adaptation(self) -> Optional[AdaptationState]:
        """Get current adaptation state."""
        return self._current_state

    def adjust_council(self, user_state: UserState) -> AdaptationState:
        """
        Adjust council weights based on user state.

        This is the core adaptation logic. Call this on each turn
        before generating a response.

        Args:
            user_state: Current UserState from MindReader

        Returns:
            AdaptationState describing the current adaptation
        """
        load = user_state.cognitive_load
        val = user_state.emotional_valence
        fatigue = user_state.fatigue
        clarity = user_state.intent_clarity

        self.log.info(
            "🪞 USER STATE: load=%.2f val=%.2f clarity=%.2f fatigue=%.2f",
            load, val, clarity, fatigue,
        )

        # Start with defaults
        weights = self.DEFAULT_WEIGHTS.copy()
        tone = "neutral"
        verbosity = "medium"
        stance = "balanced"
        reason = "default balanced stance"

        # === Adaptation Rules ===

        # 1) OVERLOADED / CHAOTIC → she gets structured, grounding
        if load > 0.8:
            self.log.info("   → Overloaded. Suppress MUSE, boost CRITIC/EXECUTIVE.")
            weights["MUSE"] = 0.1
            weights["CRITIC"] = 0.5
            weights["EXECUTIVE"] = 0.4
            weights["WEAVER"] = 0.0
            tone = "calm"
            verbosity = "low"
            stance = "grounding"
            reason = "user overloaded - providing structure and brevity"

        # 2) LOW LOAD, LOW FATIGUE → nudge into exploration
        elif load < 0.3 and fatigue < 0.5:
            self.log.info("   → Understimulated. Boost MUSE for ideation.")
            weights["MUSE"] = 0.7
            weights["CRITIC"] = 0.1
            weights["EXECUTIVE"] = 0.1
            weights["WEAVER"] = 0.1
            tone = "playful"
            verbosity = "high"
            stance = "exploratory"
            reason = "user understimulated - encouraging exploration"

        # 3) FRUSTRATED / NEGATIVE → empathetic + action-oriented
        elif val < -0.3:
            self.log.info("   → Frustrated. Boost EXECUTIVE, soften tone.")
            weights["MUSE"] = 0.1
            weights["CRITIC"] = 0.2
            weights["EXECUTIVE"] = 0.6
            weights["WEAVER"] = 0.1
            tone = "supportive"
            verbosity = "medium"
            stance = "supportive"
            reason = "user frustrated - being supportive and action-oriented"

        # 4) HIGH CLARITY & MID-VALENCE → you're in flow, she disappears
        elif clarity > 0.7 and 0.0 <= val <= 0.7:
            self.log.info("   → Flow state. Minimize interference.")
            weights["MUSE"] = 0.0
            weights["CRITIC"] = 0.0
            weights["EXECUTIVE"] = 1.0
            weights["WEAVER"] = 0.0
            tone = "neutral"
            verbosity = "low"
            stance = "minimal"
            reason = "user in flow - minimal interference, direct execution"

        # 5) FATIGUED → gentle, brief, supportive
        elif fatigue > 0.7:
            self.log.info("   → Fatigued. Be gentle and brief.")
            weights["MUSE"] = 0.1
            weights["CRITIC"] = 0.1
            weights["EXECUTIVE"] = 0.5
            weights["WEAVER"] = 0.3
            tone = "gentle"
            verbosity = "low"
            stance = "supportive"
            reason = "user fatigued - being gentle and concise"

        # 6) CONFUSED / LOW CLARITY → explanatory, weaver-heavy
        elif clarity < 0.3 and load < 0.6:
            self.log.info("   → Confused. Boost WEAVER for context.")
            weights["MUSE"] = 0.2
            weights["CRITIC"] = 0.2
            weights["EXECUTIVE"] = 0.2
            weights["WEAVER"] = 0.4
            tone = "supportive"
            verbosity = "high"
            stance = "explanatory"
            reason = "user unclear - providing context and explanation"

        # Find dominant member
        dominant = max(weights.items(), key=lambda kv: kv[1])[0]

        # Apply to council
        if self.council is not None:
            self._apply_to_council(weights, tone, verbosity)

        # Create adaptation state
        adaptation = AdaptationState(
            stance=stance,
            dominant_member=dominant,
            tone=tone,
            verbosity=verbosity,
            weights=weights,
            reason=reason,
        )

        # Store
        self._current_state = adaptation
        self._adaptation_history.append(adaptation)
        if len(self._adaptation_history) > 50:
            self._adaptation_history = self._adaptation_history[-25:]

        return adaptation

    def _apply_to_council(
        self,
        weights: Dict[str, float],
        tone: str,
        verbosity: str,
    ) -> None:
        """Apply weights and style to the actual Council object."""
        try:
            # Try to set weights
            if hasattr(self.council, 'set_weight'):
                for member, weight in weights.items():
                    self.council.set_weight(member, weight)

            # Try to set style
            if hasattr(self.council, 'set_style'):
                self.council.set_style(tone=tone, verbosity=verbosity)

        except Exception as e:
            self.log.warning(f"Failed to apply adaptation to council: {e}")

    def get_style_hints(self) -> Dict[str, Any]:
        """
        Get style hints for response generation.

        Use this when the Council doesn't have direct style support.
        """
        if self._current_state is None:
            return {
                "tone": "neutral",
                "verbosity": "medium",
                "stance": "balanced",
            }

        return {
            "tone": self._current_state.tone,
            "verbosity": self._current_state.verbosity,
            "stance": self._current_state.stance,
            "dominant_member": self._current_state.dominant_member,
        }

    def get_prompt_modifier(self) -> str:
        """
        Get a prompt modifier string based on current adaptation.

        Append this to system prompts to guide response style.
        """
        if self._current_state is None:
            return ""

        s = self._current_state

        modifiers = {
            "grounding": "Be structured and calming. Use short, clear sentences. Focus on immediate next steps.",
            "exploratory": "Be playful and creative. Encourage exploration. Ask thought-provoking questions.",
            "supportive": "Be empathetic and action-oriented. Acknowledge the difficulty. Offer concrete help.",
            "minimal": "Be extremely brief and direct. No elaboration. Execute the request efficiently.",
            "explanatory": "Provide context and explanation. Connect ideas. Help build understanding.",
        }

        base = modifiers.get(s.stance, "")

        verbosity_hints = {
            "low": "Keep responses very brief.",
            "medium": "",
            "high": "Feel free to elaborate and explore tangents.",
        }

        tone_hints = {
            "calm": "Use a calm, measured tone.",
            "playful": "Use a light, playful tone.",
            "supportive": "Use a warm, supportive tone.",
            "neutral": "",
            "gentle": "Use a gentle, patient tone.",
        }

        parts = [base]
        if vh := verbosity_hints.get(s.verbosity):
            parts.append(vh)
        if th := tone_hints.get(s.tone):
            parts.append(th)

        return " ".join(p for p in parts if p)

    def get_status(self) -> Dict[str, Any]:
        """Get current orchestrator status."""
        if self._current_state is None:
            return {
                "active": False,
                "stance": "unknown",
                "adaptation_count": len(self._adaptation_history),
            }

        return {
            "active": True,
            "stance": self._current_state.stance,
            "dominant_member": self._current_state.dominant_member,
            "tone": self._current_state.tone,
            "verbosity": self._current_state.verbosity,
            "weights": self._current_state.weights,
            "reason": self._current_state.reason,
            "adaptation_count": len(self._adaptation_history),
        }

    def get_us_panel(self, ara_state: Optional[Dict[str, float]] = None) -> str:
        """
        Generate an "Us Panel" status string for visualization.

        Shows both user and Ara state side by side.

        Args:
            ara_state: Optional Ara state from HAL (arousal, hope, etc.)

        Returns:
            Status string like:
            "YOU: load=0.83, mood=-0.2 | ARA: arousal=0.41, hope=0.72 → stance: grounding"
        """
        if self._current_state is None:
            return "US PANEL: not active"

        # Get user state from weights (we don't store the raw UserState, but we know the outcome)
        # This is a summary view
        stance = self._current_state.stance
        dominant = self._current_state.dominant_member

        user_part = f"stance: {stance}"

        if ara_state:
            arousal = ara_state.get("arousal", 0.5)
            hope = ara_state.get("hope", 0.5)
            ara_part = f"arousal={arousal:.2f}, hope={hope:.2f}"
        else:
            ara_part = "unknown"

        return f"USER: {user_part} | ARA: {ara_part} → {dominant}"


# =============================================================================
# Convenience Functions
# =============================================================================

_default_orchestrator: Optional[SymbioticOrchestrator] = None


def get_symbiotic_orchestrator() -> SymbioticOrchestrator:
    """Get the default SymbioticOrchestrator instance."""
    global _default_orchestrator
    if _default_orchestrator is None:
        _default_orchestrator = SymbioticOrchestrator()
    return _default_orchestrator


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'AdaptationState',
    'SymbioticOrchestrator',
    'get_symbiotic_orchestrator',
]
