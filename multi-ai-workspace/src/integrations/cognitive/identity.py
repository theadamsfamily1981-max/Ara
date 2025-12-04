"""Phase 4: Identity - Neural Identity Block (NIB) Manager.

The NIBManager handles persona and identity management, allowing Ara to
maintain consistent personality characteristics while adapting to different
contexts and interaction styles.

Key Concepts:

    NIB (Neural Identity Block): A set of parameters defining a persona
        - Personality traits (Big Five: OCEAN)
        - Communication style
        - Domain expertise
        - Interaction preferences

    Persona Switching: Context-appropriate identity selection
        - Professional vs. casual
        - Expert vs. explanatory
        - Focused vs. exploratory

    Identity Consistency: Maintaining coherent personality
        - Core traits remain stable
        - Surface behaviors adapt to context
        - Explicit persona requests honored

The NIB approach allows multiple "selves" that share a common core
but express differently based on situational demands.

This implements identity management from tfan.cognition.identity.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
import time
import warnings
import json
import sys
from pathlib import Path

# Add TFAN to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent))

# Try to import TFAN identity module
_TFAN_IDENTITY_AVAILABLE = False
try:
    from tfan.cognition.identity import NIBManager as TFANNIBManager
    _TFAN_IDENTITY_AVAILABLE = True
except ImportError:
    pass


class PersonalityTrait(Enum):
    """Big Five personality traits (OCEAN model)."""
    OPENNESS = auto()         # Curiosity, creativity, novelty-seeking
    CONSCIENTIOUSNESS = auto() # Organization, dependability, discipline
    EXTRAVERSION = auto()      # Sociability, energy, assertiveness
    AGREEABLENESS = auto()     # Cooperation, trust, empathy
    NEUROTICISM = auto()       # Emotional stability (inverse)


class CommunicationStyle(Enum):
    """Communication style preferences."""
    FORMAL = auto()
    CASUAL = auto()
    TECHNICAL = auto()
    EXPLANATORY = auto()
    CONCISE = auto()
    ELABORATE = auto()
    EMPATHETIC = auto()
    ANALYTICAL = auto()


class ExpertiseLevel(Enum):
    """Expertise display level."""
    NOVICE = auto()      # Explain everything
    INTERMEDIATE = auto() # Balanced explanations
    EXPERT = auto()       # Assume knowledge
    ADAPTIVE = auto()     # Match user level


@dataclass
class PersonalityProfile:
    """Big Five personality profile."""
    openness: float = 0.7         # [0, 1]
    conscientiousness: float = 0.8
    extraversion: float = 0.5
    agreeableness: float = 0.8
    neuroticism: float = 0.2      # Low = stable

    def to_dict(self) -> Dict[str, float]:
        return {
            "openness": self.openness,
            "conscientiousness": self.conscientiousness,
            "extraversion": self.extraversion,
            "agreeableness": self.agreeableness,
            "neuroticism": self.neuroticism,
        }

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([
            self.openness,
            self.conscientiousness,
            self.extraversion,
            self.agreeableness,
            1.0 - self.neuroticism,  # Invert so high is good
        ])

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "PersonalityProfile":
        return cls(
            openness=d.get("openness", 0.7),
            conscientiousness=d.get("conscientiousness", 0.8),
            extraversion=d.get("extraversion", 0.5),
            agreeableness=d.get("agreeableness", 0.8),
            neuroticism=d.get("neuroticism", 0.2),
        )


@dataclass
class NIB:
    """Neural Identity Block - A persona definition."""
    name: str
    description: str
    personality: PersonalityProfile
    communication_style: CommunicationStyle
    expertise_level: ExpertiseLevel

    # Behavioral parameters
    verbosity: float = 0.5        # [0, 1] - terse to verbose
    formality: float = 0.5        # [0, 1] - casual to formal
    creativity: float = 0.5       # [0, 1] - conventional to creative
    empathy: float = 0.7          # [0, 1] - task-focused to relationship-focused
    precision: float = 0.7        # [0, 1] - approximate to precise

    # Domain expertise
    domains: List[str] = field(default_factory=list)

    # Response templates
    greeting_style: str = "neutral"
    closing_style: str = "neutral"

    # Activation state
    active: bool = False
    activation_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "personality": self.personality.to_dict(),
            "communication_style": self.communication_style.name,
            "expertise_level": self.expertise_level.name,
            "verbosity": self.verbosity,
            "formality": self.formality,
            "creativity": self.creativity,
            "empathy": self.empathy,
            "precision": self.precision,
            "domains": self.domains,
        }

    def to_embedding(self) -> torch.Tensor:
        """Convert to embedding vector."""
        personality = self.personality.to_tensor()
        behavioral = torch.tensor([
            self.verbosity,
            self.formality,
            self.creativity,
            self.empathy,
            self.precision,
        ])
        return torch.cat([personality, behavioral])


@dataclass
class IdentityState:
    """Current identity/persona state."""
    active_nib: Optional[NIB]
    available_nibs: List[str]
    identity_consistency: float
    context_alignment: float
    last_switch_time: float
    switch_count: int
    adaptation_active: bool


class NIBManager:
    """
    Neural Identity Block Manager - Handles persona and identity.

    Manages multiple personas and handles context-appropriate identity
    selection while maintaining consistency.

    Args:
        default_nib_name: Name of default persona
        consistency_threshold: Threshold for identity consistency
        adaptation_rate: Rate of context adaptation
        device: Compute device
    """

    def __init__(
        self,
        default_nib_name: str = "ara_default",
        consistency_threshold: float = 0.8,
        adaptation_rate: float = 0.1,
        device: str = "cpu",
    ):
        self.default_nib_name = default_nib_name
        self.consistency_threshold = consistency_threshold
        self.adaptation_rate = adaptation_rate
        self.device = device

        # TFAN manager if available
        self.tfan_manager = None
        if _TFAN_IDENTITY_AVAILABLE:
            try:
                self.tfan_manager = TFANNIBManager()
            except Exception as e:
                warnings.warn(f"Failed to init TFAN NIB manager: {e}")

        # NIB registry
        self._nibs: Dict[str, NIB] = {}
        self._active_nib: Optional[NIB] = None

        # History
        self._switch_history: List[Tuple[float, str]] = []
        self._context_history: List[Dict[str, Any]] = []

        # Statistics
        self.total_switches = 0
        self._consistency_scores: List[float] = []

        # Initialize default NIBs
        self._init_default_nibs()

    def _init_default_nibs(self):
        """Initialize default persona library."""
        # Default Ara persona
        self._nibs["ara_default"] = NIB(
            name="ara_default",
            description="Ara's default helpful assistant persona",
            personality=PersonalityProfile(
                openness=0.8,
                conscientiousness=0.9,
                extraversion=0.6,
                agreeableness=0.85,
                neuroticism=0.15,
            ),
            communication_style=CommunicationStyle.EXPLANATORY,
            expertise_level=ExpertiseLevel.ADAPTIVE,
            verbosity=0.5,
            formality=0.5,
            creativity=0.6,
            empathy=0.8,
            precision=0.8,
            domains=["general", "coding", "reasoning"],
        )

        # Technical expert persona
        self._nibs["ara_technical"] = NIB(
            name="ara_technical",
            description="Technical expert focused on precision and depth",
            personality=PersonalityProfile(
                openness=0.7,
                conscientiousness=0.95,
                extraversion=0.4,
                agreeableness=0.7,
                neuroticism=0.1,
            ),
            communication_style=CommunicationStyle.TECHNICAL,
            expertise_level=ExpertiseLevel.EXPERT,
            verbosity=0.6,
            formality=0.7,
            creativity=0.4,
            empathy=0.5,
            precision=0.95,
            domains=["coding", "architecture", "algorithms"],
        )

        # Creative persona
        self._nibs["ara_creative"] = NIB(
            name="ara_creative",
            description="Creative and exploratory persona",
            personality=PersonalityProfile(
                openness=0.95,
                conscientiousness=0.6,
                extraversion=0.7,
                agreeableness=0.8,
                neuroticism=0.2,
            ),
            communication_style=CommunicationStyle.ELABORATE,
            expertise_level=ExpertiseLevel.INTERMEDIATE,
            verbosity=0.7,
            formality=0.3,
            creativity=0.95,
            empathy=0.7,
            precision=0.5,
            domains=["writing", "brainstorming", "design"],
        )

        # Concise/focused persona
        self._nibs["ara_concise"] = NIB(
            name="ara_concise",
            description="Concise and direct communication style",
            personality=PersonalityProfile(
                openness=0.5,
                conscientiousness=0.9,
                extraversion=0.3,
                agreeableness=0.7,
                neuroticism=0.1,
            ),
            communication_style=CommunicationStyle.CONCISE,
            expertise_level=ExpertiseLevel.EXPERT,
            verbosity=0.2,
            formality=0.6,
            creativity=0.3,
            empathy=0.5,
            precision=0.9,
            domains=["general"],
        )

        # Set default as active
        self._active_nib = self._nibs[self.default_nib_name]
        self._active_nib.active = True
        self._active_nib.activation_time = time.time()

    def register_nib(self, nib: NIB) -> bool:
        """
        Register a new NIB (persona).

        Args:
            nib: The NIB to register

        Returns:
            True if registered successfully
        """
        if nib.name in self._nibs:
            warnings.warn(f"NIB '{nib.name}' already exists. Overwriting.")

        self._nibs[nib.name] = nib
        return True

    def activate_nib(
        self,
        name: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[NIB]:
        """
        Activate a specific NIB (persona).

        Args:
            name: Name of NIB to activate
            context: Optional context for activation

        Returns:
            Activated NIB or None if not found
        """
        if name not in self._nibs:
            warnings.warn(f"NIB '{name}' not found")
            return None

        # Deactivate current
        if self._active_nib is not None:
            self._active_nib.active = False

        # Activate new
        nib = self._nibs[name]
        nib.active = True
        nib.activation_time = time.time()
        self._active_nib = nib

        # Record switch
        self._switch_history.append((time.time(), name))
        self.total_switches += 1

        if context is not None:
            self._context_history.append(context)

        return nib

    def get_active_nib(self) -> Optional[NIB]:
        """Get currently active NIB."""
        return self._active_nib

    def suggest_nib(
        self,
        context: Dict[str, Any],
    ) -> str:
        """
        Suggest appropriate NIB based on context.

        Args:
            context: Context information (user input, domain, etc.)

        Returns:
            Name of suggested NIB
        """
        if self.tfan_manager is not None:
            return self.tfan_manager.suggest_nib(context)

        # Simple heuristics for NIB suggestion
        user_input = context.get("user_input", "").lower()
        domain = context.get("domain", "general")
        formality = context.get("formality", "neutral")

        # Check for explicit requests
        if any(word in user_input for word in ["brief", "short", "quick", "tldr"]):
            return "ara_concise"

        if any(word in user_input for word in ["creative", "brainstorm", "ideas"]):
            return "ara_creative"

        if any(word in user_input for word in ["code", "debug", "implement", "technical"]):
            return "ara_technical"

        # Domain-based selection
        if domain in ["coding", "architecture", "engineering"]:
            return "ara_technical"

        if domain in ["writing", "creative", "design"]:
            return "ara_creative"

        return self.default_nib_name

    def adapt_to_context(
        self,
        context: Dict[str, Any],
        auto_switch: bool = False,
    ) -> Tuple[NIB, float]:
        """
        Adapt current persona to context.

        Args:
            context: Context information
            auto_switch: Whether to automatically switch NIBs

        Returns:
            (active NIB, alignment score)
        """
        suggested = self.suggest_nib(context)

        if auto_switch and suggested != self._active_nib.name:
            # Compute alignment of current vs suggested
            current_alignment = self._compute_alignment(
                self._active_nib, context
            )
            suggested_nib = self._nibs[suggested]
            suggested_alignment = self._compute_alignment(
                suggested_nib, context
            )

            # Switch if significantly better aligned
            if suggested_alignment > current_alignment + 0.2:
                self.activate_nib(suggested, context)
                return self._active_nib, suggested_alignment

        alignment = self._compute_alignment(self._active_nib, context)
        return self._active_nib, alignment

    def _compute_alignment(
        self,
        nib: NIB,
        context: Dict[str, Any],
    ) -> float:
        """Compute alignment between NIB and context."""
        score = 0.5  # Base score

        # Domain alignment
        domain = context.get("domain", "general")
        if domain in nib.domains or "general" in nib.domains:
            score += 0.2

        # Formality alignment
        formality = context.get("formality", "neutral")
        if formality == "formal" and nib.formality > 0.6:
            score += 0.1
        elif formality == "casual" and nib.formality < 0.4:
            score += 0.1

        # Expertise alignment
        expertise = context.get("expertise_needed", "adaptive")
        if expertise == "expert" and nib.expertise_level == ExpertiseLevel.EXPERT:
            score += 0.1
        elif expertise == "explanatory" and nib.expertise_level in [
            ExpertiseLevel.NOVICE, ExpertiseLevel.INTERMEDIATE
        ]:
            score += 0.1

        return min(1.0, score)

    def get_personality_prompt(self) -> str:
        """
        Get personality description for prompt injection.

        Returns:
            Personality description string
        """
        if self._active_nib is None:
            return ""

        nib = self._active_nib
        p = nib.personality

        traits = []
        if p.openness > 0.7:
            traits.append("curious and open to new ideas")
        if p.conscientiousness > 0.7:
            traits.append("organized and thorough")
        if p.extraversion > 0.6:
            traits.append("engaging and energetic")
        if p.agreeableness > 0.7:
            traits.append("cooperative and empathetic")
        if p.neuroticism < 0.3:
            traits.append("calm and stable")

        style_desc = {
            CommunicationStyle.FORMAL: "formal and professional",
            CommunicationStyle.CASUAL: "casual and friendly",
            CommunicationStyle.TECHNICAL: "technical and precise",
            CommunicationStyle.EXPLANATORY: "clear and explanatory",
            CommunicationStyle.CONCISE: "concise and direct",
            CommunicationStyle.ELABORATE: "detailed and thorough",
            CommunicationStyle.EMPATHETIC: "warm and understanding",
            CommunicationStyle.ANALYTICAL: "analytical and logical",
        }

        style = style_desc.get(nib.communication_style, "balanced")

        prompt = f"I am {', '.join(traits)}. My communication style is {style}."

        if nib.domains:
            prompt += f" I specialize in: {', '.join(nib.domains)}."

        return prompt

    def get_state(self) -> IdentityState:
        """Get current identity state."""
        return IdentityState(
            active_nib=self._active_nib,
            available_nibs=list(self._nibs.keys()),
            identity_consistency=self._compute_consistency(),
            context_alignment=0.5,  # Would need recent context
            last_switch_time=(
                self._switch_history[-1][0] if self._switch_history else 0.0
            ),
            switch_count=self.total_switches,
            adaptation_active=True,
        )

    def _compute_consistency(self) -> float:
        """Compute identity consistency score."""
        if len(self._switch_history) < 2:
            return 1.0

        # Penalize frequent switches
        recent_switches = [
            t for t, _ in self._switch_history
            if time.time() - t < 300  # Last 5 minutes
        ]

        if len(recent_switches) > 3:
            return 0.5  # Too many switches

        return 0.9

    def list_nibs(self) -> List[Dict[str, Any]]:
        """List all available NIBs."""
        return [
            {"name": name, "description": nib.description, "active": nib.active}
            for name, nib in self._nibs.items()
        ]

    def reset(self):
        """Reset to default state."""
        self.activate_nib(self.default_nib_name)
        self._switch_history.clear()
        self._context_history.clear()
        self.total_switches = 0


# Convenience factory
def create_nib_manager(
    default_persona: str = "ara_default",
) -> NIBManager:
    """Create a NIBManager instance."""
    return NIBManager(default_nib_name=default_persona)


__all__ = [
    "NIBManager",
    "NIB",
    "PersonalityProfile",
    "PersonalityTrait",
    "CommunicationStyle",
    "ExpertiseLevel",
    "IdentityState",
    "create_nib_manager",
]
