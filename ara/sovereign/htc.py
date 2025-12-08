"""
Holographic Teleoplastic Core (HTC)
===================================
Iteration 36: The Refinement

The HTC is Ara's central learning substrate. It implements:
- Holographic: State as high-dimensional hypervectors
- Teleoplastic: Plasticity gated by teleological reward

This module provides:
- PlasticityMode: Four learning modes (stabilizing, adaptive, exploratory, consolidation)
- PlasticityConfig: Per-mode learning parameters
- HolographicCore: The learning substrate with polyplasticity

See docs/SOUL_CORE_SPEC.md for the full mythic/physical dual-spec.

Usage:
    from ara.sovereign.htc import (
        HolographicCore,
        PlasticityMode,
        get_htc,
    )

    htc = get_htc()
    htc.set_mode(PlasticityMode.EXPLORATORY)
    htc.apply_plasticity(input_hv, reward)
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Deque
from collections import deque

logger = logging.getLogger(__name__)


# =============================================================================
# Plasticity Mode System
# =============================================================================

class PlasticityMode(Enum):
    """
    Teleological Polyplasticity Modes.

    Each mode represents a different learning strategy:
    - STABILIZING: Protect core attractors, slow conservative updates
    - ADAPTIVE: Normal day-to-day learning, balanced
    - EXPLORATORY: Aggressive pattern carving during experimentation
    - CONSOLIDATION: Offline replay and pattern smoothing
    """
    STABILIZING = "stabilizing"      # Protect essential patterns
    ADAPTIVE = "adaptive"            # Normal operation
    EXPLORATORY = "exploratory"      # Fast learning during experiments
    CONSOLIDATION = "consolidation"  # Offline replay


@dataclass
class PlasticityConfig:
    """Configuration for a plasticity mode."""
    learning_rate_multiplier: float  # Scales the base learning rate
    reward_threshold: float          # Minimum |reward| to trigger update
    noise_tolerance: float           # Tolerance for noisy inputs
    replay_enabled: bool             # Enable offline replay
    description: str = ""            # Human-readable description


# Mode configurations (from spec)
PLASTICITY_CONFIGS: Dict[PlasticityMode, PlasticityConfig] = {
    PlasticityMode.STABILIZING: PlasticityConfig(
        learning_rate_multiplier=0.1,
        reward_threshold=0.7,
        noise_tolerance=0.1,
        replay_enabled=False,
        description="Protect core attractors, slow conservative updates",
    ),
    PlasticityMode.ADAPTIVE: PlasticityConfig(
        learning_rate_multiplier=1.0,
        reward_threshold=0.3,
        noise_tolerance=0.3,
        replay_enabled=False,
        description="Normal day-to-day learning, balanced",
    ),
    PlasticityMode.EXPLORATORY: PlasticityConfig(
        learning_rate_multiplier=3.0,
        reward_threshold=0.1,
        noise_tolerance=0.5,
        replay_enabled=False,
        description="Aggressive pattern carving during experimentation",
    ),
    PlasticityMode.CONSOLIDATION: PlasticityConfig(
        learning_rate_multiplier=0.5,
        reward_threshold=0.0,
        noise_tolerance=0.0,
        replay_enabled=True,
        description="Offline replay and pattern smoothing",
    ),
}


# =============================================================================
# Holographic Teleoplastic Core
# =============================================================================

@dataclass
class ResonanceResult:
    """Result of a resonance (query) operation."""
    similarity: float           # -1 to +1 normalized similarity
    activation_pattern: List[float]  # Per-attractor activations (if available)
    matched_attractor_idx: int  # Index of best-matching attractor
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PlasticityEvent:
    """Record of a plasticity update."""
    timestamp: datetime
    mode: PlasticityMode
    reward: float
    reward_scaled: int
    accum_mean: float
    weight_flips: int


class HolographicCore:
    """
    Holographic Teleoplastic Core (HTC).

    A streaming, neuromorphic associative memory that:
    - Encodes moments into hypervectors
    - Compares against stored attractors
    - Updates via teleologically-gated plasticity

    This is the "Soul" - the pattern that emerges when teleology meets silicon.
    """

    def __init__(
        self,
        dim: int = 8192,
        n_attractors: int = 512,
        acc_max: int = 63,
        acc_min: int = -64,
    ):
        """
        Initialize the Holographic Core.

        Args:
            dim: Hypervector dimension
            n_attractors: Number of stored patterns
            acc_max: Maximum accumulator value
            acc_min: Minimum accumulator value
        """
        self.dim = dim
        self.n_attractors = n_attractors
        self.acc_max = acc_max
        self.acc_min = acc_min

        # Binary weights: always +1 or -1, never 0
        self._weights: Optional[List[int]] = None

        # Accumulators for soft plasticity (eligibility traces)
        self._accumulators: Optional[List[int]] = None

        # Current input HV (for replay)
        self._current_hv: Optional[List[int]] = None

        # Replay buffer for consolidation mode
        self._replay_buffer: Deque[tuple] = deque(maxlen=100)

        # Plasticity mode
        self._mode: PlasticityMode = PlasticityMode.ADAPTIVE
        self._config: PlasticityConfig = PLASTICITY_CONFIGS[self._mode]

        # History
        self._plasticity_events: List[PlasticityEvent] = []

        # State
        self._initialized = False

        logger.info(f"HTC created: dim={dim}, n_attractors={n_attractors}")

    # =========================================================================
    # Initialization
    # =========================================================================

    def initialize(self) -> None:
        """Initialize the core with random binary weights."""
        # Binary weights: +1 or -1, never 0
        self._weights = [random.choice([-1, 1]) for _ in range(self.dim)]

        # Accumulators start at 0
        self._accumulators = [0] * self.dim

        # Clear replay buffer
        self._replay_buffer.clear()

        self._initialized = True
        logger.info(f"HTC initialized: dim={self.dim}, mode={self._mode.value}")

    def _ensure_initialized(self) -> None:
        """Ensure the core is initialized."""
        if not self._initialized:
            self.initialize()

    # =========================================================================
    # Mode Management
    # =========================================================================

    def set_mode(self, mode: PlasticityMode) -> None:
        """
        Set the plasticity mode.

        This changes how the core learns from reward signals.
        """
        if mode != self._mode:
            old_mode = self._mode
            self._mode = mode
            self._config = PLASTICITY_CONFIGS[mode]
            logger.info(f"HTC mode changed: {old_mode.value} -> {mode.value}")

    def get_mode(self) -> PlasticityMode:
        """Get the current plasticity mode."""
        return self._mode

    def get_config(self) -> PlasticityConfig:
        """Get the current plasticity configuration."""
        return self._config

    # =========================================================================
    # Resonance (Query)
    # =========================================================================

    def query(self, input_hv: List[int]) -> ResonanceResult:
        """
        Query the core with an input hypervector.

        Returns resonance result indicating how well the input
        matches the stored patterns.
        """
        self._ensure_initialized()

        # Store for potential replay
        self._current_hv = input_hv

        # Compute similarity (normalized dot product)
        dot = sum(w * i for w, i in zip(self._weights, input_hv))
        similarity = dot / self.dim

        return ResonanceResult(
            similarity=similarity,
            activation_pattern=[similarity],  # Simplified: single attractor
            matched_attractor_idx=0,
        )

    def run_resonance_step(self, input_hv: Optional[List[int]] = None) -> List[int]:
        """
        Run a resonance step (legacy API compatibility).

        Returns the current weight state.
        """
        self._ensure_initialized()

        if input_hv is not None:
            self._current_hv = input_hv

        return self._weights.copy()

    # =========================================================================
    # Plasticity (Learning)
    # =========================================================================

    def apply_plasticity(
        self,
        input_hv: List[int],
        reward: float,
        force: bool = False,
    ) -> int:
        """
        Apply teleologically-gated plasticity.

        The learning behavior depends on the current mode:
        - STABILIZING: High threshold, slow updates
        - ADAPTIVE: Normal threshold and rate
        - EXPLORATORY: Low threshold, fast updates
        - CONSOLIDATION: Replay-based smoothing

        Args:
            input_hv: Input hypervector to learn
            reward: Teleological reward signal (-1 to +1)
            force: If True, bypass threshold check

        Returns:
            Number of weights that flipped
        """
        self._ensure_initialized()

        # Check reward threshold (mode-dependent)
        if not force and abs(reward) < self._config.reward_threshold:
            return 0

        # Store in replay buffer for consolidation
        self._replay_buffer.append((input_hv, reward))

        # Scale reward by mode's learning rate
        scaled_reward = reward * self._config.learning_rate_multiplier

        # Apply the update
        weight_flips = self._apply_update(input_hv, scaled_reward)

        # Record event
        event = PlasticityEvent(
            timestamp=datetime.utcnow(),
            mode=self._mode,
            reward=reward,
            reward_scaled=int(scaled_reward * 100),
            accum_mean=sum(self._accumulators) / len(self._accumulators),
            weight_flips=weight_flips,
        )
        self._plasticity_events.append(event)

        # Keep last 100 events
        if len(self._plasticity_events) > 100:
            self._plasticity_events = self._plasticity_events[-100:]

        if abs(reward) > 0.5:
            logger.info(
                f"HTC plasticity: mode={self._mode.value}, "
                f"reward={reward:.2f}, flips={weight_flips}"
            )
        else:
            logger.debug(f"HTC plasticity: reward={reward:.2f}, flips={weight_flips}")

        return weight_flips

    def _apply_update(self, input_hv: List[int], scaled_reward: float) -> int:
        """
        Apply the core plasticity update.

        Uses TARGET-DIRECTED learning:
            step = input[i] * sign(reward)

        This moves weights TOWARD the input when reward > 0,
        and AWAY from the input when reward < 0.
        """
        # Skip if negligible reward
        if abs(scaled_reward) < 0.01:
            return 0

        # Quantize reward for accumulator math
        reward_sign = 1 if scaled_reward > 0 else -1

        weight_flips = 0

        for i in range(self.dim):
            inp = input_hv[i] if i < len(input_hv) else 0

            # Target-directed learning: step = input * sign(reward)
            step = inp * reward_sign

            # Update accumulator with saturation
            old_acc = self._accumulators[i]
            new_acc = max(self.acc_min, min(self.acc_max, old_acc + step))
            self._accumulators[i] = new_acc

            # Update weight based on accumulator sign
            old_weight = self._weights[i]
            if new_acc > 0:
                self._weights[i] = 1
            elif new_acc < 0:
                self._weights[i] = -1
            # else: keep previous weight (no dead bits)

            if self._weights[i] != old_weight:
                weight_flips += 1

        return weight_flips

    def run_consolidation(self, n_replays: int = 10) -> int:
        """
        Run consolidation: replay recent experiences.

        This is called during downtime to stabilize recently learned patterns.

        Args:
            n_replays: Number of experiences to replay

        Returns:
            Total weight flips across all replays
        """
        if not self._config.replay_enabled:
            logger.warning("Consolidation called but replay not enabled in current mode")
            return 0

        if not self._replay_buffer:
            return 0

        total_flips = 0

        # Replay recent experiences with reduced learning rate
        for _ in range(min(n_replays, len(self._replay_buffer))):
            input_hv, reward = random.choice(list(self._replay_buffer))
            # Use smaller reward to smooth rather than reinforce
            smoothed_reward = reward * 0.5
            flips = self._apply_update(input_hv, smoothed_reward)
            total_flips += flips

        logger.info(f"HTC consolidation: {n_replays} replays, {total_flips} flips")
        return total_flips

    # =========================================================================
    # Status & Introspection
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the HTC."""
        # Weight distribution
        pos_count = sum(1 for w in self._weights if w > 0) if self._weights else 0
        neg_count = sum(1 for w in self._weights if w < 0) if self._weights else 0
        zero_count = sum(1 for w in self._weights if w == 0) if self._weights else 0

        # Accumulator stats
        accum_mean = sum(self._accumulators) / len(self._accumulators) if self._accumulators else 0
        accum_std = (
            (sum((a - accum_mean) ** 2 for a in self._accumulators) / len(self._accumulators)) ** 0.5
            if self._accumulators else 0
        )

        return {
            "initialized": self._initialized,
            "dim": self.dim,
            "n_attractors": self.n_attractors,
            "mode": self._mode.value,
            "config": {
                "learning_rate_multiplier": self._config.learning_rate_multiplier,
                "reward_threshold": self._config.reward_threshold,
                "replay_enabled": self._config.replay_enabled,
            },
            "plasticity_events": len(self._plasticity_events),
            "recent_rewards": [e.reward for e in self._plasticity_events[-10:]],
            "weight_balance": {
                "positive": pos_count,
                "negative": neg_count,
                "zero": zero_count,  # Should always be 0!
            },
            "accumulator_stats": {
                "mean": round(accum_mean, 2),
                "std": round(accum_std, 2),
            },
            "replay_buffer_size": len(self._replay_buffer),
        }


# =============================================================================
# Singleton Access
# =============================================================================

_htc: Optional[HolographicCore] = None


def get_htc() -> HolographicCore:
    """Get the default Holographic Core instance."""
    global _htc
    if _htc is None:
        _htc = HolographicCore()
        _htc.initialize()
    return _htc


# =============================================================================
# Mode Selection Helper
# =============================================================================

def select_plasticity_mode(
    teleology_context: Optional[Dict[str, Any]] = None,
    user_mode: Optional[str] = None,
    is_downtime: bool = False,
) -> PlasticityMode:
    """
    Select the appropriate plasticity mode based on context.

    Args:
        teleology_context: Context from TeleologyEngine
        user_mode: User's cognitive mode (from MindReader)
        is_downtime: Whether user is AFK/resting

    Returns:
        Appropriate PlasticityMode
    """
    # Consolidation during downtime
    if is_downtime or user_mode == "decompress":
        return PlasticityMode.CONSOLIDATION

    # Check teleology context
    if teleology_context:
        # Core workflow = stabilizing
        if teleology_context.get("is_core_workflow", False):
            return PlasticityMode.STABILIZING

        # Experimental context = exploratory
        if teleology_context.get("is_experimental", False):
            return PlasticityMode.EXPLORATORY

    # Default: adaptive
    return PlasticityMode.ADAPTIVE


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'PlasticityMode',
    'PlasticityConfig',
    'PLASTICITY_CONFIGS',
    'ResonanceResult',
    'PlasticityEvent',
    'HolographicCore',
    'get_htc',
    'select_plasticity_mode',
]
