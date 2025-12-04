"""MIES Thermodynamic Governor - Energy-Based Model + AEPO Policy.

This module implements the thermodynamic policy layer:
1. EnergyFunction: E(M, S) = E_friction + E_urgency + E_autonomy
2. AEPOSampler: Entropy-controlled policy optimization
3. ThermodynamicGovernor: Full policy with deterministic and stochastic modes

AEPO (Adaptive Entropy Policy Optimization):
- Controls exploration through target entropy
- Loss: L = L_task + lambda * (H(pi) - H_target)^2
- Prevents mode collapse (always picking same mode)
- Allows graceful degradation under uncertainty

This is designed to be trained via IRL/RLHF on user preferences,
but works out-of-box with heuristic energy functions.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum, auto
import logging

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None

from ..context import ModalityContext, ActivityType
from ..modes import (
    ModalityMode,
    ModalityDecision,
    TransitionParams,
    DEFAULT_MODES,
    MODE_SILENT,
)

logger = logging.getLogger(__name__)


@dataclass
class ContentMeta:
    """Metadata about the content Ara wants to deliver."""
    severity: float = 0.0           # 0-1, how serious/important
    urgency: float = 0.0            # 0-1, time sensitivity
    deadline_seconds: Optional[float] = None
    is_user_requested: bool = False
    estimated_tokens: int = 100
    requires_attention: bool = False
    can_be_deferred: bool = True


@dataclass
class PolicyState:
    """Persistent state for the policy across decisions."""
    total_decisions: int = 0
    mode_history: List[str] = field(default_factory=list)
    entropy_history: List[float] = field(default_factory=list)
    last_permission_asked: float = 0.0
    user_preference_overrides: Dict[str, float] = field(default_factory=dict)


class EnergyFunction:
    """
    Energy-Based Model for modality scoring.

    E(M, S) combines multiple energy terms:
    - E_friction: Social/context friction (high = rude/inappropriate)
    - E_urgency: Information urgency (negative = important to deliver)
    - E_autonomy: Liveness/boredom pressure (builds over time)
    - E_thermodynamic: Internal energy constraints

    Lower energy = better mode choice.
    """

    def __init__(
        self,
        w_friction: float = 1.0,
        w_urgency: float = 0.7,
        w_autonomy: float = 0.3,
        w_thermo: float = 0.5,
    ):
        self.w_friction = w_friction
        self.w_urgency = w_urgency
        self.w_autonomy = w_autonomy
        self.w_thermo = w_thermo

        # Friction coefficients (learnable in full version)
        self.friction_coefficients = {
            "meeting_audio": 5.0,
            "meeting_avatar": 3.0,
            "deep_work_audio": 2.5,
            "deep_work_avatar_large": 2.0,
            "gaming_fullscreen_overlay": 2.0,
            "high_load_audio": 1.5,
            "negative_valence_intrusive": 1.0,
        }

    def compute(
        self,
        ctx: ModalityContext,
        mode: ModalityMode,
        content_meta: Optional[ContentMeta] = None,
    ) -> float:
        """Compute total energy for a mode in context."""
        content_meta = content_meta or ContentMeta()

        e_friction = self.e_friction(ctx, mode)
        e_urgency = self.e_urgency(ctx, mode, content_meta)
        e_autonomy = self.e_autonomy(ctx)
        e_thermo = self.e_thermodynamic(ctx, mode)

        total = (
            self.w_friction * e_friction +
            self.w_urgency * e_urgency +
            self.w_autonomy * e_autonomy +
            self.w_thermo * e_thermo
        )

        return total

    def e_friction(self, ctx: ModalityContext, mode: ModalityMode) -> float:
        """Social friction energy - how rude is this mode in context?"""
        friction = 0.0

        is_audio = mode.channel.name.startswith("AUDIO")
        is_avatar = mode.channel.name.startswith("AVATAR")

        # Meeting penalties
        if ctx.activity == ActivityType.MEETING:
            if is_audio:
                friction += self.friction_coefficients["meeting_audio"]
            if is_avatar:
                friction += self.friction_coefficients["meeting_avatar"]
            friction += mode.intrusiveness * 2.0

        # Deep work penalties
        elif ctx.activity == ActivityType.DEEP_WORK:
            if is_audio:
                friction += self.friction_coefficients["deep_work_audio"]
            if is_avatar and mode.avatar_size > 0.2:
                friction += self.friction_coefficients["deep_work_avatar_large"]
            friction += mode.intrusiveness * 1.5

        # Gaming penalties
        elif ctx.activity == ActivityType.GAMING:
            if ctx.foreground.is_fullscreen and is_avatar:
                friction += self.friction_coefficients["gaming_fullscreen_overlay"]
            if ctx.audio.has_voice_call and is_audio:
                friction += self.friction_coefficients["meeting_audio"]

        # Cognitive load penalty
        if ctx.user_cognitive_load > 0.7:
            if is_audio:
                friction += self.friction_coefficients["high_load_audio"]
            friction += mode.intrusiveness * ctx.user_cognitive_load

        # Emotional state penalty
        if ctx.valence < -0.5 and mode.intrusiveness > 0.4:
            friction += self.friction_coefficients["negative_valence_intrusive"]

        return friction

    def e_urgency(
        self,
        ctx: ModalityContext,
        mode: ModalityMode,
        content_meta: ContentMeta,
    ) -> float:
        """Urgency energy - negative when urgent content needs delivery."""
        urgency_pressure = 0.0

        # Base urgency
        urgency_pressure += content_meta.urgency * 3.0

        # User request bonus
        if content_meta.is_user_requested:
            urgency_pressure += 2.0

        # Deadline pressure
        if content_meta.deadline_seconds is not None:
            if content_meta.deadline_seconds < 30:
                urgency_pressure += 3.0
            elif content_meta.deadline_seconds < 60:
                urgency_pressure += 2.0
            elif content_meta.deadline_seconds < 300:
                urgency_pressure += 1.0

        # Severity
        urgency_pressure += content_meta.severity * 1.5

        # Mode bandwidth match (can this mode deliver the content?)
        bandwidth_match = self._bandwidth_match(mode, content_meta)

        # Return negative (lower is better, urgency reduces energy)
        return -urgency_pressure * bandwidth_match

    def e_autonomy(self, ctx: ModalityContext) -> float:
        """Autonomy/liveness energy - pressure to act after being quiet."""
        minutes_quiet = ctx.seconds_since_last_utterance / 60.0
        buildup = min(1.0, minutes_quiet / 10.0)  # Saturates at 10 min

        # This is a positive energy (penalty for silence)
        # But small - autonomy should only enable gentle actions
        return buildup * 0.3

    def e_thermodynamic(self, ctx: ModalityContext, mode: ModalityMode) -> float:
        """Thermodynamic energy - internal constraints on expensive modes."""
        thermo_cost = 0.0

        # Energy budget constraint
        if ctx.energy_remaining < 0.5:
            thermo_cost += mode.energy_cost * (1 - ctx.energy_remaining) * 2

        # Thermal state constraint
        if ctx.thermal_state == "WARM":
            thermo_cost += mode.energy_cost * 0.5
        elif ctx.thermal_state == "HOT":
            thermo_cost += mode.energy_cost * 1.5
        elif ctx.thermal_state == "OVERHEATING":
            thermo_cost += mode.energy_cost * 4.0

        # Ara stress/fatigue
        thermo_cost += mode.energy_cost * ctx.ara_fatigue
        thermo_cost += mode.energy_cost * ctx.ara_stress * 0.5

        return thermo_cost

    def _bandwidth_match(self, mode: ModalityMode, content_meta: ContentMeta) -> float:
        """How well does this mode's bandwidth match content needs?"""
        # Estimated bandwidth need from content
        if content_meta.estimated_tokens < 50:
            needed = 0.2
        elif content_meta.estimated_tokens < 200:
            needed = 0.5
        else:
            needed = 0.8

        # Match score
        if mode.bandwidth_cost >= needed:
            return 1.0
        else:
            return mode.bandwidth_cost / needed


class AEPOSampler:
    """
    Adaptive Entropy Policy Optimization sampler.

    Maintains a target entropy and adjusts sampling temperature
    to balance exploration vs exploitation.

    Loss (for training): L = L_task + lambda * (H(pi) - H_target)^2

    For inference, we sample with temperature adjusted to maintain
    target entropy, preventing mode collapse.
    """

    def __init__(
        self,
        target_entropy: float = 0.4,
        lambda_entropy: float = 0.1,
        temperature: float = 1.0,
        temperature_bounds: Tuple[float, float] = (0.1, 2.0),
    ):
        self.target_entropy = target_entropy
        self.lambda_entropy = lambda_entropy
        self.temperature = temperature
        self.temperature_bounds = temperature_bounds

        # History for entropy tracking
        self._entropy_history: List[float] = []
        self._temperature_history: List[float] = []

    def sample(
        self,
        energies: Dict[str, float],
        deterministic: bool = False,
    ) -> Tuple[str, float, float]:
        """
        Sample a mode from energy distribution.

        Args:
            energies: Dict mapping mode name to energy value
            deterministic: If True, always pick lowest energy

        Returns:
            (selected_mode_name, probability, entropy)
        """
        mode_names = list(energies.keys())
        energy_values = np.array([energies[m] for m in mode_names])

        if deterministic:
            idx = np.argmin(energy_values)
            return mode_names[idx], 1.0, 0.0

        # Convert energies to probabilities via softmax
        # Lower energy = higher probability
        neg_energies = -energy_values / self.temperature
        # Numerical stability
        neg_energies = neg_energies - neg_energies.max()
        probs = np.exp(neg_energies)
        probs = probs / probs.sum()

        # Compute entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10))

        # Sample
        idx = np.random.choice(len(mode_names), p=probs)
        selected = mode_names[idx]
        selected_prob = probs[idx]

        # Update entropy history
        self._entropy_history.append(entropy)
        if len(self._entropy_history) > 100:
            self._entropy_history.pop(0)

        # Adapt temperature based on entropy
        self._adapt_temperature(entropy)

        return selected, selected_prob, entropy

    def _adapt_temperature(self, current_entropy: float):
        """Adapt temperature to maintain target entropy."""
        error = current_entropy - self.target_entropy

        # Simple proportional control
        adjustment = 0.1 * error

        self.temperature = np.clip(
            self.temperature + adjustment,
            self.temperature_bounds[0],
            self.temperature_bounds[1],
        )

        self._temperature_history.append(self.temperature)
        if len(self._temperature_history) > 100:
            self._temperature_history.pop(0)

    def get_entropy(self) -> float:
        """Get current entropy estimate."""
        if not self._entropy_history:
            return self.target_entropy
        return np.mean(self._entropy_history[-10:])

    def compute_loss(
        self,
        logits: "torch.Tensor",
        targets: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Compute AEPO loss for training.

        L = L_task + lambda * (H(pi) - H_target)^2
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for training")

        # Task loss (cross entropy)
        l_task = F.cross_entropy(logits, targets)

        # Entropy of policy
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean()

        # Entropy penalty
        l_entropy = self.lambda_entropy * (entropy - self.target_entropy) ** 2

        return l_task + l_entropy


class PolicyNetwork(nn.Module if TORCH_AVAILABLE else object):
    """
    Small MLP for learned policy.

    Maps context features -> logits over modes.
    Stub implementation - ready for RL training.
    """

    def __init__(
        self,
        input_dim: int = 32,
        hidden_dim: int = 64,
        num_modes: int = 13,
    ):
        if not TORCH_AVAILABLE:
            return

        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_modes),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required")
        return self.net(x)


class ThermodynamicGovernor:
    """
    Full thermodynamic modality policy.

    Combines:
    - EnergyFunction for mode scoring
    - AEPOSampler for entropy-controlled sampling
    - HeuristicBaseline as fallback

    Supports two modes:
    1. Deterministic (heuristic): Always pick lowest energy
    2. Stochastic (AEPO): Sample with entropy control
    """

    def __init__(
        self,
        energy_function: Optional[EnergyFunction] = None,
        sampler: Optional[AEPOSampler] = None,
        use_stochastic: bool = False,
        fallback_to_heuristic: bool = True,
    ):
        self.energy_fn = energy_function or EnergyFunction()
        self.sampler = sampler or AEPOSampler()
        self.use_stochastic = use_stochastic
        self.fallback_to_heuristic = fallback_to_heuristic

        # Policy network (for learned version)
        self.policy_network: Optional[PolicyNetwork] = None

        # State
        self._policy_state = PolicyState()
        self._last_decision: Optional[ModalityDecision] = None

    def select_modality(
        self,
        ctx: ModalityContext,
        content_meta: Optional[ContentMeta] = None,
        prev_mode: Optional[ModalityMode] = None,
        policy_state: Optional[PolicyState] = None,
    ) -> ModalityDecision:
        """
        Select modality using thermodynamic policy.

        Args:
            ctx: Current modality context
            content_meta: Metadata about content to deliver
            prev_mode: Previous mode for smooth transitions
            policy_state: Persistent policy state

        Returns:
            ModalityDecision with chosen mode
        """
        content_meta = content_meta or ContentMeta(
            urgency=ctx.info_urgency,
            severity=ctx.info_severity,
            is_user_requested=ctx.is_user_requested,
            estimated_tokens=ctx.content_length_tokens,
        )

        if policy_state is not None:
            self._policy_state = policy_state

        # Get candidate modes
        candidates = self._get_candidates(ctx)

        if not candidates:
            return ModalityDecision(
                mode=MODE_SILENT,
                rationale="No valid candidates",
                confidence=1.0,
            )

        # Compute energies
        energies = {}
        for mode in candidates:
            e = self.energy_fn.compute(ctx, mode, content_meta)

            # Continuity bonus
            if prev_mode is not None and mode.name == prev_mode.name:
                e -= 0.3  # Prefer staying in same mode

            energies[mode.name] = e

        # Select mode
        if self.use_stochastic:
            selected_name, prob, entropy = self.sampler.sample(energies)
            confidence = prob
        else:
            selected_name, prob, entropy = self.sampler.sample(energies, deterministic=True)
            confidence = 1.0

        selected_mode = next(m for m in candidates if m.name == selected_name)

        # Build decision
        transition = self._compute_transition(prev_mode, selected_mode)
        rationale = self._build_rationale(ctx, selected_mode, energies, entropy)

        decision = ModalityDecision(
            mode=selected_mode,
            transition=transition,
            rationale=rationale,
            confidence=confidence,
            energy_score=energies[selected_name],
            alternatives_considered=[m.name for m in candidates if m.name != selected_name],
        )

        # Update state
        self._policy_state.total_decisions += 1
        self._policy_state.mode_history.append(selected_name)
        if len(self._policy_state.mode_history) > 100:
            self._policy_state.mode_history.pop(0)
        self._policy_state.entropy_history.append(entropy)
        if len(self._policy_state.entropy_history) > 100:
            self._policy_state.entropy_history.pop(0)

        self._last_decision = decision
        return decision

    def _get_candidates(self, ctx: ModalityContext) -> List[ModalityMode]:
        """Get candidate modes based on hard constraints."""
        candidates = [DEFAULT_MODES["silent"]]

        # Text modes almost always valid
        for name in ["text_inline", "text_minimal", "text_side"]:
            candidates.append(DEFAULT_MODES[name])

        # Audio modes
        if self._audio_allowed(ctx):
            candidates.append(DEFAULT_MODES["audio_whisper"])
            candidates.append(DEFAULT_MODES["audio_normal"])

        # Avatar modes
        if self._avatar_allowed(ctx):
            candidates.append(DEFAULT_MODES["avatar_subtle"])
            candidates.append(DEFAULT_MODES["avatar_present"])
            if ctx.energy_remaining > 0.5:
                candidates.append(DEFAULT_MODES["avatar_full"])

        return candidates

    def _audio_allowed(self, ctx: ModalityContext) -> bool:
        """Hard constraint: is audio allowed?"""
        if ctx.activity == ActivityType.MEETING:
            return False
        if ctx.audio.mic_in_use:
            return False
        if ctx.audio.has_voice_call:
            return False
        return True

    def _avatar_allowed(self, ctx: ModalityContext) -> bool:
        """Hard constraint: is avatar allowed?"""
        if ctx.energy_remaining < 0.2:
            return False
        if ctx.thermal_state == "OVERHEATING":
            return False
        return True

    def _compute_transition(
        self,
        prev_mode: Optional[ModalityMode],
        new_mode: ModalityMode,
    ) -> TransitionParams:
        """Compute geodesic transition parameters."""
        if prev_mode is None:
            return TransitionParams(duration_ms=200)

        distance = new_mode.distance_to(prev_mode)
        duration = int(200 + distance * 300)

        return TransitionParams(
            duration_ms=duration,
            easing="ease-out",
            fade_out_first=prev_mode.channel != new_mode.channel,
        )

    def _build_rationale(
        self,
        ctx: ModalityContext,
        mode: ModalityMode,
        energies: Dict[str, float],
        entropy: float,
    ) -> str:
        """Build debug rationale."""
        return (
            f"Activity={ctx.activity.name}, "
            f"mode={mode.name}, "
            f"E={energies[mode.name]:.2f}, "
            f"H={entropy:.2f}, "
            f"T={self.sampler.temperature:.2f}"
        )

    def get_state(self) -> PolicyState:
        """Get current policy state."""
        return self._policy_state

    def get_entropy(self) -> float:
        """Get current entropy estimate."""
        return self.sampler.get_entropy()


# === Convenience ===

def create_thermodynamic_governor(
    stochastic: bool = False,
    target_entropy: float = 0.4,
) -> ThermodynamicGovernor:
    """Create a thermodynamic governor."""
    sampler = AEPOSampler(target_entropy=target_entropy)
    return ThermodynamicGovernor(
        sampler=sampler,
        use_stochastic=stochastic,
    )


__all__ = [
    "EnergyFunction",
    "AEPOSampler",
    "PolicyNetwork",
    "ThermodynamicGovernor",
    "ContentMeta",
    "PolicyState",
    "create_thermodynamic_governor",
]
