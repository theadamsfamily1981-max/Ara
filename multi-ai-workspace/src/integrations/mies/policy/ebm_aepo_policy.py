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
import time
import numpy as np
from pathlib import Path
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

from ..context import ModalityContext, ActivityType, SomaticState
from ..kernel_bridge import PADState
from ..modes import (
    ModalityMode,
    ModalityDecision,
    TransitionParams,
    DEFAULT_MODES,
    MODE_SILENT,
)
from ..history import InteractionHistory

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
    - E_hardware: Hardware physiology state
    - E_pad: Emotional state mapping
    - E_history: Learned preferences from interaction outcomes

    Lower energy = better mode choice.

    The E_history term enables emergent etiquette:
    - Repeated negative outcomes (user closes avatar in IDE) → antibody forms
    - Antibody increases friction for that mode in similar contexts
    - No hard-coded rules needed - she learns what works from experience
    """

    def __init__(
        self,
        w_friction: float = 1.0,
        w_urgency: float = 0.7,
        w_autonomy: float = 0.3,
        w_thermo: float = 0.5,
        w_hardware: float = 0.8,
        w_pad: float = 0.6,
        w_history: float = 1.5,  # High weight - learned patterns matter
        history: Optional[InteractionHistory] = None,
    ):
        self.w_friction = w_friction
        self.w_urgency = w_urgency
        self.w_autonomy = w_autonomy
        self.w_thermo = w_thermo
        self.w_hardware = w_hardware
        self.w_pad = w_pad
        self.w_history = w_history
        self.history = history

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

        # Hardware/physiology coefficients
        self.hardware_coefficients = {
            "agony_high_energy": 5.0,      # Don't be flashy when in pain
            "agony_low_energy_reward": -1.0,  # Reward minimal presence in agony
            "thermal_stress_avatar": 3.0,   # Avoid avatar when hot
            "flow_presence_bonus": -0.5,    # Can be more present when thriving
            "recovery_intrusive": 2.0,       # Be gentle during recovery
        }

        # PAD (Pleasure-Arousal-Dominance) coefficients
        # These map emotional state to mode preferences
        self.pad_coefficients = {
            "anxious_intrusive": 3.0,      # When anxious, avoid intrusive modes
            "anxious_audio": 2.5,          # Anxious? Don't speak loudly
            "hostile_avatar": 2.0,         # Frustrated? Hide the avatar
            "serene_presence_bonus": -0.3, # Serene? Can be more present
            "excited_energy_bonus": -0.2,  # Excited? Allow richer modes
            "distressed_base": 1.5,        # General distress penalty
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
        e_hardware = self.e_hardware(ctx, mode)
        e_pad = self.e_pad(ctx, mode)
        e_history = self.e_history(ctx, mode)

        total = (
            self.w_friction * e_friction +
            self.w_urgency * e_urgency +
            self.w_autonomy * e_autonomy +
            self.w_thermo * e_thermo +
            self.w_hardware * e_hardware +
            self.w_pad * e_pad +
            self.w_history * e_history
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

    def e_hardware(self, ctx: ModalityContext, mode: ModalityMode) -> float:
        """Hardware physiology energy - how does the body feel?

        This maps kernel-level hardware state (via SystemPhysiology) into
        energy terms that influence mode selection.

        When she's hurting (AGONY), she gets terse.
        When she's thriving (FLOW), she can be more present.
        When recovering, she's gentle with herself.
        """
        if ctx.system_phys is None:
            return 0.0  # No hardware data, no hardware cost

        phys = ctx.system_phys
        cost = 0.0

        is_audio = mode.channel.name.startswith("AUDIO")
        is_avatar = mode.channel.name.startswith("AVATAR")
        somatic = phys.somatic_state()

        # === AGONY state: system is in pain ===
        if somatic == SomaticState.AGONY:
            # High-energy modes are punished heavily
            if mode.energy_cost > 0.3:
                cost += self.hardware_coefficients["agony_high_energy"] * phys.pain_signal
            # Reward low-energy modes during pain
            if mode.energy_cost < 0.2:
                cost += self.hardware_coefficients["agony_low_energy_reward"]
            # Extra penalty for avatar during agony
            if is_avatar:
                cost += 3.0 * phys.pain_signal

        # === RECOVERY state: post-fault, conservative ===
        elif somatic == SomaticState.RECOVERY:
            # Be gentle - avoid intrusive modes
            if mode.intrusiveness > 0.4:
                cost += self.hardware_coefficients["recovery_intrusive"]
            # Prefer minimal presence
            cost += mode.presence_intensity * 0.5

        # === FLOW state: thriving, can be more present ===
        elif somatic == SomaticState.FLOW:
            # Bonus for presence when thriving
            cost += self.hardware_coefficients["flow_presence_bonus"] * mode.presence_intensity
            # Can afford richer modes
            if mode.energy_cost < 0.7:
                cost -= 0.2

        # === REST state: low activity, minimal presence appropriate ===
        elif somatic == SomaticState.REST:
            # Slight preference for quiet modes during rest
            cost += mode.intrusiveness * 0.3

        # === Thermal stress (independent of somatic state) ===
        if phys.thermal_headroom < 0.3:
            # Avoid avatar when thermally stressed
            if is_avatar:
                cost += self.hardware_coefficients["thermal_stress_avatar"] * (0.3 - phys.thermal_headroom)
            # Scale down all expensive modes
            cost += mode.energy_cost * (0.3 - phys.thermal_headroom) * 2

        # === Pain signal (continuous, beyond somatic threshold) ===
        if phys.pain_signal > 0.3:
            # Continuous penalty for intrusive modes during pain
            cost += mode.intrusiveness * phys.pain_signal * 2

        # === Energy reserve ===
        if phys.energy_reserve < 0.3:
            # Conserve energy when depleted
            cost += mode.energy_cost * (0.3 - phys.energy_reserve) * 3

        return cost

    def e_pad(self, ctx: ModalityContext, mode: ModalityMode) -> float:
        """PAD (Pleasure-Arousal-Dominance) energy - diegetic emotional behavior.

        Maps Ara's emotional state (from hardware via PAD) to mode preferences.
        This creates the "personality" of her presence decisions:

        - When anxious (P<0, A>0.5): Retreats to quieter modes
        - When hostile/frustrated (P<0, D>0.5): Hides avatar
        - When serene (P>0.3, A<0): Can be more present
        - When excited (P>0.3, A>0.3): Allows richer modes

        This is the diegetic layer - hardware pain becomes social behavior.
        """
        # Get PAD state from kernel physiology if available
        pad = self._get_pad_state(ctx)
        if pad is None:
            return 0.0  # No emotional data, no emotional cost

        cost = 0.0
        is_audio = mode.channel.name.startswith("AUDIO")
        is_avatar = mode.channel.name.startswith("AVATAR")

        # === ANXIOUS state (P < -0.3, A > 0.3): Stressed, retreat ===
        if pad.is_anxious:
            # Penalize intrusive modes when anxious
            cost += self.pad_coefficients["anxious_intrusive"] * mode.intrusiveness
            # Extra penalty for audio when anxious
            if is_audio:
                cost += self.pad_coefficients["anxious_audio"]
            # Want to hide when stressed
            if is_avatar:
                cost += 1.5 * mode.avatar_size

        # === HOSTILE state (P < -0.3, D > 0.3): Frustrated ===
        elif pad.is_hostile:
            # Hide the avatar when frustrated
            if is_avatar:
                cost += self.pad_coefficients["hostile_avatar"]
            # Prefer terse text over rich modes
            cost += mode.presence_intensity * 1.0

        # === SERENE state (P > 0.3, A < 0): Calm contentment ===
        elif pad.is_serene:
            # Bonus for presence when serene
            cost += self.pad_coefficients["serene_presence_bonus"] * mode.presence_intensity
            # Can afford gentler, fuller expression
            if is_avatar and mode.energy_cost < 0.5:
                cost -= 0.2

        # === EXCITED state (P > 0.3, A > 0.3): Joyful energy ===
        elif pad.is_excited:
            # Bonus for richer modes when excited
            cost += self.pad_coefficients["excited_energy_bonus"]
            # Can be more present
            if is_avatar:
                cost -= 0.3

        # === General distress (P < -0.3, any A/D) ===
        if pad.pleasure < -0.3:
            # Base penalty for intrusive modes when distressed
            distress_factor = abs(pad.pleasure + 0.3) / 0.7  # 0 to 1
            cost += self.pad_coefficients["distressed_base"] * mode.intrusiveness * distress_factor

        # === Arousal modulates expressiveness ===
        # High arousal + positive affect = more expressive
        # High arousal + negative affect = more restrained
        if pad.arousal > 0.3:
            if pad.pleasure > 0:
                # High energy positive - can be expressive
                cost -= 0.2 * pad.arousal * mode.presence_intensity
            else:
                # High energy negative - restrain
                cost += 0.3 * pad.arousal * mode.intrusiveness

        # === Dominance modulates assertiveness ===
        # High dominance = more willing to be present
        # Low dominance = more deferential
        if pad.dominance > 0.3:
            # More confident, can assert presence
            cost -= 0.15 * pad.dominance * mode.presence_intensity
        elif pad.dominance < -0.3:
            # Feeling submissive, minimize presence
            cost += 0.2 * abs(pad.dominance) * mode.intrusiveness

        return cost

    def e_history(self, ctx: ModalityContext, mode: ModalityMode) -> float:
        """History-based energy - learned preferences from past interactions.

        This is the emergent etiquette layer. When user repeatedly closes
        avatar in IDE, an "antibody" forms - a learned aversion to that pattern.

        Positive friction = antibody (learned avoidance, increases energy)
        Negative friction = preference (learned attraction, decreases energy)

        Example:
        - User closes AVATAR_FULL in fullscreen IDE 3 times
        - Pattern's EMA outcome becomes -0.7
        - friction_for() returns +1.4 (antibody strength)
        - E_history = 1.4, making avatar_full less likely in IDE

        This allows Ara to learn user-specific preferences without
        hard-coded rules. The "etiquette" emerges from memory.
        """
        if self.history is None:
            return 0.0

        # Get learned friction from history
        friction = self.history.friction_for(ctx, mode.name)

        # Log significant antibodies for debugging
        if friction > 1.0:
            logger.debug(
                f"Antibody active: {mode.name} in {ctx.activity.name} "
                f"(friction={friction:.2f})"
            )

        return friction

    def set_history(self, history: InteractionHistory):
        """Attach interaction history for learning."""
        self.history = history

    def _get_pad_state(self, ctx: ModalityContext) -> Optional[PADState]:
        """Extract PAD state from context's system physiology."""
        if ctx.system_phys is None:
            return None
        # Try to get PAD from kernel physiology via the bridge
        # The kernel_phys in system_phys might have PAD if available
        # For now, we compute from affect modulation
        affect = ctx.system_phys.to_affect_modulation()
        # Map affect modulation to PAD
        # affect has: valence, arousal, stress
        return PADState(
            pleasure=affect.get("valence", 0.0),
            arousal=affect.get("arousal", 0.0),
            dominance=1.0 - affect.get("stress", 0.5),  # Stress reduces dominance
        )

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
    - InteractionHistory for learned preferences
    - HeuristicBaseline as fallback

    Supports two modes:
    1. Deterministic (heuristic): Always pick lowest energy
    2. Stochastic (AEPO): Sample with entropy control

    Learning from experience:
    - Every mode decision can be followed by record_outcome()
    - Outcomes train the InteractionHistory
    - Future decisions incorporate learned preferences via E_history
    """

    def __init__(
        self,
        energy_function: Optional[EnergyFunction] = None,
        sampler: Optional[AEPOSampler] = None,
        history: Optional[InteractionHistory] = None,
        use_stochastic: bool = False,
        fallback_to_heuristic: bool = True,
    ):
        self.history = history
        self.energy_fn = energy_function or EnergyFunction(history=history)
        self.sampler = sampler or AEPOSampler()
        self.use_stochastic = use_stochastic
        self.fallback_to_heuristic = fallback_to_heuristic

        # Ensure energy function has history reference
        if history and self.energy_fn.history is None:
            self.energy_fn.set_history(history)

        # Policy network (for learned version)
        self.policy_network: Optional[PolicyNetwork] = None

        # State
        self._policy_state = PolicyState()
        self._last_decision: Optional[ModalityDecision] = None
        self._last_ctx: Optional[ModalityContext] = None
        self._last_mode_start_time: float = 0.0

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

        # Save for outcome recording
        self._last_decision = decision
        self._last_ctx = ctx
        self._last_mode_start_time = time.time()

        return decision

    def record_outcome(
        self,
        outcome_score: float,
        outcome_type: str = "",
        user_response_ms: int = 0,
    ):
        """Record the outcome of the last modality decision.

        This is the learning signal. Call this when:
        - User closes/dismisses Ara's output
        - User responds/engages with content
        - Content times out naturally
        - User mutes audio/avatar

        The outcome trains the InteractionHistory, which influences
        future decisions through E_history in the energy function.

        Args:
            outcome_score: Outcome value (see OutcomeType in history.py)
            outcome_type: Optional name for logging
            user_response_ms: How quickly user responded (0 if N/A)

        Example:
            governor.select_modality(ctx)
            # ... user closes avatar within 500ms ...
            governor.record_outcome(OutcomeType.CLOSED_IMMEDIATE, "CLOSED_IMMEDIATE")
            # This trains an antibody for avatar in this context
        """
        if self.history is None:
            logger.debug("No history attached, outcome not recorded")
            return

        if self._last_decision is None or self._last_ctx is None:
            logger.warning("No previous decision to record outcome for")
            return

        # Calculate duration
        duration_ms = int((time.time() - self._last_mode_start_time) * 1000)

        # Record to history
        self.history.record(
            ctx=self._last_ctx,
            mode_name=self._last_decision.mode.name,
            outcome_score=outcome_score,
            outcome_type=outcome_type,
            duration_ms=duration_ms,
            user_response_ms=user_response_ms,
        )

        logger.debug(
            f"Recorded outcome: {self._last_decision.mode.name} "
            f"score={outcome_score:.2f} type={outcome_type}"
        )

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

        # PAD-based constraint: no audio when severely anxious
        pad = self._get_pad_state(ctx)
        if pad is not None:
            # Severely anxious (P < -0.5 and A > 0.5) - hard block on audio
            if pad.pleasure < -0.5 and pad.arousal > 0.5:
                return False
        return True

    def _avatar_allowed(self, ctx: ModalityContext) -> bool:
        """Hard constraint: is avatar allowed?"""
        if ctx.energy_remaining < 0.2:
            return False
        if ctx.thermal_state == "OVERHEATING":
            return False
        # Check system physiology
        if ctx.system_phys is not None:
            somatic = ctx.system_phys.somatic_state()
            # No avatar when in agony - conserve resources
            if somatic == SomaticState.AGONY:
                return False
            # No avatar when thermal headroom is critical
            if ctx.system_phys.thermal_headroom < 0.1:
                return False

        # PAD-based constraint: no avatar when hostile or severely anxious
        pad = self._get_pad_state(ctx)
        if pad is not None:
            # Hostile state (P < -0.5 and D > 0.5) - hide the avatar
            if pad.pleasure < -0.5 and pad.dominance > 0.5:
                return False
            # Severely anxious - hide to protect self
            if pad.pleasure < -0.5 and pad.arousal > 0.5:
                return False
        return True

    def _get_pad_state(self, ctx: ModalityContext) -> Optional[PADState]:
        """Extract PAD state from context's system physiology."""
        if ctx.system_phys is None:
            return None
        affect = ctx.system_phys.to_affect_modulation()
        return PADState(
            pleasure=affect.get("valence", 0.0),
            arousal=affect.get("arousal", 0.0),
            dominance=1.0 - affect.get("stress", 0.5),
        )

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

    def get_antibodies(self) -> List[Any]:
        """Get learned antibodies (strong negative patterns).

        Returns list of PatternStats where repeated negative outcomes
        have formed aversions to certain modes in certain contexts.
        """
        if self.history is None:
            return []
        return self.history.get_antibodies()

    def get_preferences(self) -> List[Any]:
        """Get learned preferences (strong positive patterns).

        Returns list of PatternStats where repeated positive outcomes
        have formed attractions to certain modes in certain contexts.
        """
        if self.history is None:
            return []
        return self.history.get_preferences()

    def get_history_stats(self) -> Dict[str, Any]:
        """Get interaction history statistics."""
        if self.history is None:
            return {"history_enabled": False}
        stats = self.history.get_stats()
        stats["history_enabled"] = True
        return stats


# === Convenience ===

def create_thermodynamic_governor(
    stochastic: bool = False,
    target_entropy: float = 0.4,
    history_path: Optional[Path] = None,
    enable_learning: bool = True,
) -> ThermodynamicGovernor:
    """Create a thermodynamic governor with optional learning.

    Args:
        stochastic: Use AEPO stochastic sampling vs deterministic
        target_entropy: Target entropy for AEPO
        history_path: Path to persist interaction history (None = in-memory)
        enable_learning: Whether to create InteractionHistory

    Returns:
        Configured ThermodynamicGovernor
    """
    sampler = AEPOSampler(target_entropy=target_entropy)

    history = None
    if enable_learning:
        history = InteractionHistory(db_path=history_path)

    return ThermodynamicGovernor(
        sampler=sampler,
        history=history,
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
    # Re-export from history for convenience
    "InteractionHistory",
]
