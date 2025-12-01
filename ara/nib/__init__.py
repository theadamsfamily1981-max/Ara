"""
ARA NIB Module (Neural Identity Buffer)

Identity state, persona management, and long-term self-model.
This bridges TF-A-N's identity preservation mechanisms with
Ara's persona selection and style adaptation.

API Contract:
    GET /nib/state
    - Input: session_id
    - Output: identity_mode, stability, style parameters

The NIB maintains:
    - Core identity vector (who Ara "is")
    - Stability metrics (identity drift monitoring)
    - Style parameters (formality, warmth, directness)
    - Active persona/mode selection
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum
import json

# Add paths
_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


class IdentityMode(str, Enum):
    """Ara's operational identity modes."""
    GUIDE_MENTOR = "guide_mentor"      # Teaching, explaining, nurturing
    TECHNICAL = "technical"             # Precise, detailed, code-focused
    CREATIVE = "creative"               # Playful, exploratory, brainstorming
    SUPPORT = "support"                 # Empathetic, listening, validating
    EXECUTIVE = "executive"             # Decisive, action-oriented, direct
    RESEARCH = "research"               # Analytical, thorough, citation-aware


@dataclass
class StyleParameters:
    """
    Ara's communication style parameters.

    These map to prompt engineering and voice/avatar settings.
    """
    formality: float = 0.5       # [0, 1] casual → formal
    warmth: float = 0.7          # [0, 1] cold → warm
    directness: float = 0.6      # [0, 1] indirect → direct
    verbosity: float = 0.5       # [0, 1] terse → verbose
    playfulness: float = 0.3     # [0, 1] serious → playful
    technical_depth: float = 0.5  # [0, 1] simple → technical

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def for_mode(cls, mode: IdentityMode) -> "StyleParameters":
        """Get default style parameters for a given mode."""
        mode_styles = {
            IdentityMode.GUIDE_MENTOR: cls(
                formality=0.4, warmth=0.8, directness=0.5,
                verbosity=0.6, playfulness=0.3, technical_depth=0.4
            ),
            IdentityMode.TECHNICAL: cls(
                formality=0.6, warmth=0.4, directness=0.8,
                verbosity=0.5, playfulness=0.1, technical_depth=0.9
            ),
            IdentityMode.CREATIVE: cls(
                formality=0.2, warmth=0.7, directness=0.4,
                verbosity=0.6, playfulness=0.8, technical_depth=0.3
            ),
            IdentityMode.SUPPORT: cls(
                formality=0.3, warmth=0.9, directness=0.3,
                verbosity=0.5, playfulness=0.2, technical_depth=0.2
            ),
            IdentityMode.EXECUTIVE: cls(
                formality=0.7, warmth=0.5, directness=0.9,
                verbosity=0.3, playfulness=0.1, technical_depth=0.5
            ),
            IdentityMode.RESEARCH: cls(
                formality=0.7, warmth=0.4, directness=0.6,
                verbosity=0.8, playfulness=0.1, technical_depth=0.8
            ),
        }
        return mode_styles.get(mode, cls())


@dataclass
class IdentityMetrics:
    """
    Identity stability and drift metrics.

    Based on TF-A-N's topological identity preservation:
    - Embedding drift monitoring
    - Wasserstein distance from baseline
    - Cosine similarity to core identity
    """
    stability: float = 1.0           # [0, 1] overall stability
    embedding_drift: float = 0.0      # Distance from baseline (target ≤ 0.02)
    wasserstein_gap: float = 0.0      # Topological drift
    cosine_similarity: float = 1.0    # Similarity to core identity (target ≥ 0.90)
    last_recenter: Optional[str] = None  # Last manifold recentering

    def is_stable(self) -> bool:
        """Check if identity is within acceptable bounds."""
        return (
            self.embedding_drift <= 0.02 and
            self.cosine_similarity >= 0.90 and
            self.stability >= 0.8
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NIBState:
    """
    Complete NIB (Neural Identity Buffer) state.

    This is the source of truth for Ara's identity that
    drives persona selection and communication style.
    """
    # Current mode
    identity_mode: IdentityMode = IdentityMode.GUIDE_MENTOR

    # Style parameters
    style: StyleParameters = field(default_factory=StyleParameters)

    # Stability metrics
    metrics: IdentityMetrics = field(default_factory=IdentityMetrics)

    # Session tracking
    session_id: Optional[str] = None
    user_id: Optional[str] = None

    # Context
    conversation_turns: int = 0
    mode_switches: int = 0
    last_mode_switch: Optional[str] = None

    # Timestamp
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "identity_mode": self.identity_mode.value,
            "style": self.style.to_dict(),
            "metrics": self.metrics.to_dict(),
            "session_id": self.session_id,
            "user_id": self.user_id,
            "conversation_turns": self.conversation_turns,
            "mode_switches": self.mode_switches,
            "last_mode_switch": self.last_mode_switch,
            "timestamp": self.timestamp,
            "is_stable": self.metrics.is_stable(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class NIBManager:
    """
    NIB (Neural Identity Buffer) Manager.

    Manages Ara's identity state, persona selection, and
    style adaptation based on context and TF-A-N signals.

    In production, this wraps TF-A-N's identity preservation
    mechanisms (HyperbolicIdentity, etc.).
    """

    def __init__(
        self,
        default_mode: IdentityMode = IdentityMode.GUIDE_MENTOR,
        use_hrrl: bool = False,
    ):
        self.default_mode = default_mode
        self.use_hrrl = use_hrrl
        self._sessions: Dict[str, NIBState] = {}

        # Try to load HRRL identity components
        self._hrrl_available = False
        if use_hrrl:
            self._init_hrrl()

    def _init_hrrl(self):
        """Initialize HRRL identity components."""
        try:
            from hrrl_agent.identity import HyperbolicIdentity
            self._hrrl_available = True
        except ImportError:
            self._hrrl_available = False

    def get_state(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> NIBState:
        """
        Get current NIB state for a session.

        Args:
            session_id: Session identifier
            user_id: User identifier (for personalization)

        Returns:
            NIBState with current identity configuration
        """
        # Use session state if exists
        if session_id and session_id in self._sessions:
            state = self._sessions[session_id]
            state.timestamp = datetime.utcnow().isoformat()
            return state

        # Create new state
        state = NIBState(
            identity_mode=self.default_mode,
            style=StyleParameters.for_mode(self.default_mode),
            session_id=session_id,
            user_id=user_id,
        )

        if session_id:
            self._sessions[session_id] = state

        return state

    def switch_mode(
        self,
        new_mode: IdentityMode,
        session_id: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> NIBState:
        """
        Switch Ara's identity mode.

        Args:
            new_mode: Target identity mode
            session_id: Session identifier
            reason: Optional reason for switch

        Returns:
            Updated NIBState
        """
        state = self.get_state(session_id)

        if state.identity_mode != new_mode:
            state.identity_mode = new_mode
            state.style = StyleParameters.for_mode(new_mode)
            state.mode_switches += 1
            state.last_mode_switch = datetime.utcnow().isoformat()

        return state

    def infer_mode(
        self,
        text: str,
        context: Optional[List[str]] = None,
    ) -> IdentityMode:
        """
        Infer appropriate identity mode from user input.

        Uses heuristics to detect intent:
        - Code/technical questions → TECHNICAL
        - Emotional content → SUPPORT
        - Creative tasks → CREATIVE
        - Action requests → EXECUTIVE
        - Learning questions → GUIDE_MENTOR
        """
        text_lower = text.lower()

        # Technical indicators
        technical_markers = [
            "code", "error", "bug", "function", "class", "import",
            "api", "database", "server", "debug", "compile", "```"
        ]
        if any(m in text_lower for m in technical_markers):
            return IdentityMode.TECHNICAL

        # Support indicators
        support_markers = [
            "feel", "worried", "stressed", "frustrated", "help me",
            "struggling", "overwhelmed", "anxious", "sad"
        ]
        if any(m in text_lower for m in support_markers):
            return IdentityMode.SUPPORT

        # Creative indicators
        creative_markers = [
            "brainstorm", "ideas", "creative", "imagine", "what if",
            "design", "invent", "story", "explore"
        ]
        if any(m in text_lower for m in creative_markers):
            return IdentityMode.CREATIVE

        # Executive indicators
        executive_markers = [
            "do this", "make it", "just", "quickly", "asap",
            "priority", "urgent", "action", "decide"
        ]
        if any(m in text_lower for m in executive_markers):
            return IdentityMode.EXECUTIVE

        # Research indicators
        research_markers = [
            "research", "paper", "study", "analyze", "compare",
            "literature", "evidence", "methodology"
        ]
        if any(m in text_lower for m in research_markers):
            return IdentityMode.RESEARCH

        # Default to guide/mentor
        return IdentityMode.GUIDE_MENTOR

    def adapt_style(
        self,
        state: NIBState,
        pad_pleasure: float,
        pad_arousal: float,
        pad_dominance: float,
    ) -> StyleParameters:
        """
        Adapt style parameters based on user's affective state.

        When user is:
        - Low pleasure: increase warmth, decrease playfulness
        - High arousal: decrease verbosity, increase directness
        - Low dominance: increase empathy/warmth
        """
        style = state.style

        # Adjust warmth inversely to pleasure (more warmth when user stressed)
        warmth_adj = -pad_pleasure * 0.2
        style.warmth = max(0.3, min(1.0, style.warmth + warmth_adj))

        # Adjust verbosity inversely to arousal (shorter when urgent)
        verbosity_adj = -pad_arousal * 0.2
        style.verbosity = max(0.2, min(0.8, style.verbosity + verbosity_adj))

        # Adjust directness with arousal (more direct when urgent)
        directness_adj = pad_arousal * 0.1
        style.directness = max(0.3, min(0.9, style.directness + directness_adj))

        return style

    def get_prompt_modifiers(self, state: NIBState) -> Dict[str, str]:
        """
        Get prompt engineering modifiers for current state.

        Returns dict of modifiers to inject into system prompts.
        """
        style = state.style

        modifiers = {
            "tone": "",
            "length": "",
            "approach": "",
        }

        # Tone based on formality + warmth
        if style.formality > 0.6 and style.warmth > 0.6:
            modifiers["tone"] = "professional yet warm"
        elif style.formality > 0.6:
            modifiers["tone"] = "professional and precise"
        elif style.warmth > 0.6:
            modifiers["tone"] = "friendly and supportive"
        else:
            modifiers["tone"] = "casual and approachable"

        # Length based on verbosity
        if style.verbosity > 0.7:
            modifiers["length"] = "detailed explanations"
        elif style.verbosity < 0.3:
            modifiers["length"] = "concise responses"
        else:
            modifiers["length"] = "balanced detail"

        # Approach based on directness + playfulness
        if style.directness > 0.7:
            modifiers["approach"] = "direct and action-oriented"
        elif style.playfulness > 0.5:
            modifiers["approach"] = "exploratory and creative"
        else:
            modifiers["approach"] = "thoughtful and thorough"

        return modifiers


# Convenience functions
def get_nib_state(
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Get NIB state for a session."""
    manager = NIBManager()
    state = manager.get_state(session_id, user_id)
    return state.to_dict()


def infer_identity_mode(text: str) -> str:
    """Infer identity mode from text."""
    manager = NIBManager()
    mode = manager.infer_mode(text)
    return mode.value


__all__ = [
    "IdentityMode",
    "StyleParameters",
    "IdentityMetrics",
    "NIBState",
    "NIBManager",
    "get_nib_state",
    "infer_identity_mode",
]
