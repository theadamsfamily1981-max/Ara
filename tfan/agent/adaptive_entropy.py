"""
CLV-Modulated Entropy: Adaptive Exploration for AEPO

This module implements adaptive entropy modulation based on the
Cognitive Load Vector (CLV). The system's exploration rate "breathes"
with system risk:

- Calm, low risk → High entropy → Explore structure freely
- Stressed, high risk → Low entropy → Exploit, make small safe moves

The entropy coefficient α is computed as:

    risk = 0.5 * instability + 0.3 * resource + 0.2 * structural
    α = α_base × (1 + α_range × (1 - risk))

This gives:
    risk ≈ 0 (calm)  → α ≈ α_base × (1 + range) → more exploration
    risk ≈ 1 (chaos) → α ≈ α_base              → conservative

Integration:
    CLV → AdaptiveEntropyController → AEPO loss computation

Usage:
    from tfan.agent.adaptive_entropy import (
        AdaptiveEntropyController,
        compute_entropy_coef_from_clv,
    )

    controller = AdaptiveEntropyController()
    alpha = controller.compute_from_clv(clv)

    # In AEPO loss:
    loss = policy_loss + value_loss - alpha * entropy.mean()
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging
import math

# Add paths
_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

logger = logging.getLogger("tfan.agent.adaptive_entropy")

try:
    from tfan.system.cognitive_load_vector import (
        CognitiveLoadVector, CLVComputer, RiskLevel
    )
    CLV_AVAILABLE = True
except ImportError:
    CLV_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AdaptiveEntropyConfig:
    """Configuration for CLV-modulated entropy."""
    # Base entropy coefficient (what you'd use without adaptation)
    entropy_coef_base: float = 0.01

    # How much CLV is allowed to modulate the coefficient
    # α = base × (1 + range × (1 - risk))
    # range=2.0 means α can be up to 3× base when calm
    entropy_coef_range: float = 2.0

    # Hard bounds to prevent extreme values
    entropy_coef_min: float = 0.001
    entropy_coef_max: float = 0.1

    # Smoothing factor for entropy updates (0 = instant, 1 = no change)
    smoothing: float = 0.3

    # Risk weights (must sum to 1.0)
    instability_weight: float = 0.5
    resource_weight: float = 0.3
    structural_weight: float = 0.2

    # Max entropy step (prevent sudden jumps)
    max_entropy_step: float = 0.5  # Max 50% change per update

    # Whether to use arousal for additional modulation
    use_arousal_boost: bool = True
    arousal_boost_threshold: float = 0.7  # High arousal threshold
    arousal_boost_factor: float = 0.5     # Reduce entropy when aroused

    def validate(self):
        """Validate configuration."""
        weight_sum = self.instability_weight + self.resource_weight + self.structural_weight
        if abs(weight_sum - 1.0) > 0.001:
            logger.warning(f"Risk weights sum to {weight_sum}, normalizing")
            self.instability_weight /= weight_sum
            self.resource_weight /= weight_sum
            self.structural_weight /= weight_sum


# =============================================================================
# ADAPTIVE ENTROPY CONTROLLER
# =============================================================================

@dataclass
class EntropyState:
    """Current state of the entropy controller."""
    current_alpha: float = 0.01
    target_alpha: float = 0.01
    risk_score: float = 0.0
    risk_level: str = "nominal"
    exploration_mode: str = "balanced"  # conservative/balanced/exploratory
    updates: int = 0
    last_updated: str = ""


class AdaptiveEntropyController:
    """
    Controls AEPO entropy based on CLV risk assessment.

    This implements "breathing" exploration: the system explores more
    when stable and less when under stress.
    """

    def __init__(self, config: Optional[AdaptiveEntropyConfig] = None):
        """
        Initialize adaptive entropy controller.

        Args:
            config: Configuration (uses defaults if None)
        """
        self.config = config or AdaptiveEntropyConfig()
        self.config.validate()

        # State
        self._state = EntropyState(
            current_alpha=self.config.entropy_coef_base,
            target_alpha=self.config.entropy_coef_base,
        )

        # History for analysis
        self._history: List[Tuple[float, float, float]] = []  # (risk, alpha, timestamp)
        self._max_history = 100

        logger.info(
            f"AdaptiveEntropyController initialized: "
            f"base={self.config.entropy_coef_base}, range={self.config.entropy_coef_range}"
        )

    def compute_risk_from_clv(self, clv: "CognitiveLoadVector") -> float:
        """
        Compute scalar risk from CLV.

        Args:
            clv: CognitiveLoadVector

        Returns:
            Risk score in [0, 1]
        """
        risk = (
            self.config.instability_weight * clv.instability +
            self.config.resource_weight * clv.resource +
            self.config.structural_weight * clv.structural
        )

        # Clamp to [0, 1]
        return max(0.0, min(1.0, risk))

    def compute_target_alpha(self, risk: float, arousal: float = 0.5) -> float:
        """
        Compute target entropy coefficient from risk.

        Args:
            risk: Risk score [0, 1]
            arousal: PAD arousal [0, 1]

        Returns:
            Target entropy coefficient
        """
        cfg = self.config

        # Base formula: α = base × (1 + range × (1 - risk))
        # Low risk → high alpha; high risk → base alpha
        alpha = cfg.entropy_coef_base * (1.0 + cfg.entropy_coef_range * (1.0 - risk))

        # Optional arousal modulation
        # High arousal = urgent situation → reduce exploration
        if cfg.use_arousal_boost and arousal > cfg.arousal_boost_threshold:
            arousal_factor = 1.0 - cfg.arousal_boost_factor * (arousal - cfg.arousal_boost_threshold) / (1.0 - cfg.arousal_boost_threshold)
            alpha *= max(0.5, arousal_factor)

        # Clamp to bounds
        alpha = max(cfg.entropy_coef_min, min(cfg.entropy_coef_max, alpha))

        return alpha

    def update_from_clv(self, clv: "CognitiveLoadVector") -> float:
        """
        Update entropy coefficient from CLV.

        This is the main entry point for CLV-modulated entropy.

        Args:
            clv: CognitiveLoadVector

        Returns:
            Current entropy coefficient (smoothed)
        """
        cfg = self.config

        # Compute risk
        risk = self.compute_risk_from_clv(clv)

        # Get arousal from CLV components
        arousal = clv.components.arousal if hasattr(clv, 'components') else 0.5

        # Compute target alpha
        target = self.compute_target_alpha(risk, arousal)

        # Apply max step constraint
        current = self._state.current_alpha
        max_step = cfg.max_entropy_step * current
        delta = target - current
        if abs(delta) > max_step:
            target = current + max_step * (1.0 if delta > 0 else -1.0)

        # Smooth update
        self._state.target_alpha = target
        self._state.current_alpha = (
            cfg.smoothing * self._state.current_alpha +
            (1.0 - cfg.smoothing) * target
        )

        # Update state
        self._state.risk_score = risk
        self._state.risk_level = clv.risk_level.value if hasattr(clv.risk_level, 'value') else str(clv.risk_level)
        self._state.updates += 1
        self._state.last_updated = datetime.utcnow().isoformat()

        # Classify exploration mode
        if self._state.current_alpha < cfg.entropy_coef_base * 0.8:
            self._state.exploration_mode = "conservative"
        elif self._state.current_alpha > cfg.entropy_coef_base * 1.5:
            self._state.exploration_mode = "exploratory"
        else:
            self._state.exploration_mode = "balanced"

        # Record history
        self._history.append((
            risk,
            self._state.current_alpha,
            datetime.utcnow().timestamp(),
        ))
        if len(self._history) > self._max_history:
            self._history.pop(0)

        logger.debug(
            f"Entropy update: risk={risk:.3f} → α={self._state.current_alpha:.4f} "
            f"({self._state.exploration_mode})"
        )

        return self._state.current_alpha

    def update_from_raw(
        self,
        instability: float,
        resource: float,
        structural: float,
        arousal: float = 0.5,
    ) -> float:
        """
        Update from raw CLV components (when full CLV not available).

        Args:
            instability: Instability component [0, 1]
            resource: Resource component [0, 1]
            structural: Structural component [0, 1]
            arousal: PAD arousal [0, 1]

        Returns:
            Current entropy coefficient
        """
        # Compute risk directly
        risk = (
            self.config.instability_weight * instability +
            self.config.resource_weight * resource +
            self.config.structural_weight * structural
        )
        risk = max(0.0, min(1.0, risk))

        # Compute target
        target = self.compute_target_alpha(risk, arousal)

        # Apply smoothing (simplified, no max step)
        self._state.target_alpha = target
        self._state.current_alpha = (
            self.config.smoothing * self._state.current_alpha +
            (1.0 - self.config.smoothing) * target
        )

        self._state.risk_score = risk
        self._state.updates += 1

        # Classify exploration mode
        cfg = self.config
        if self._state.current_alpha < cfg.entropy_coef_base * 0.8:
            self._state.exploration_mode = "conservative"
        elif self._state.current_alpha > cfg.entropy_coef_base * 1.5:
            self._state.exploration_mode = "exploratory"
        else:
            self._state.exploration_mode = "balanced"

        return self._state.current_alpha

    def get_current_alpha(self) -> float:
        """Get current entropy coefficient."""
        return self._state.current_alpha

    def get_state(self) -> Dict[str, Any]:
        """Get controller state."""
        return {
            "current_alpha": self._state.current_alpha,
            "target_alpha": self._state.target_alpha,
            "risk_score": self._state.risk_score,
            "risk_level": self._state.risk_level,
            "exploration_mode": self._state.exploration_mode,
            "updates": self._state.updates,
            "base_alpha": self.config.entropy_coef_base,
            "alpha_range": [self.config.entropy_coef_min, self.config.entropy_coef_max],
        }

    def get_history_stats(self) -> Dict[str, Any]:
        """Get statistics from history."""
        if not self._history:
            return {"count": 0}

        risks = [h[0] for h in self._history]
        alphas = [h[1] for h in self._history]

        return {
            "count": len(self._history),
            "risk_mean": sum(risks) / len(risks),
            "risk_min": min(risks),
            "risk_max": max(risks),
            "alpha_mean": sum(alphas) / len(alphas),
            "alpha_min": min(alphas),
            "alpha_max": max(alphas),
            "exploration_time_pct": sum(1 for a in alphas if a > self.config.entropy_coef_base * 1.5) / len(alphas) * 100,
            "conservative_time_pct": sum(1 for a in alphas if a < self.config.entropy_coef_base * 0.8) / len(alphas) * 100,
        }

    def reset(self):
        """Reset controller to initial state."""
        self._state = EntropyState(
            current_alpha=self.config.entropy_coef_base,
            target_alpha=self.config.entropy_coef_base,
        )
        self._history.clear()
        logger.info("AdaptiveEntropyController reset")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global controller instance
_global_controller: Optional[AdaptiveEntropyController] = None


def get_entropy_controller() -> AdaptiveEntropyController:
    """Get global adaptive entropy controller."""
    global _global_controller
    if _global_controller is None:
        _global_controller = AdaptiveEntropyController()
    return _global_controller


def compute_entropy_coef_from_clv(clv: "CognitiveLoadVector") -> float:
    """
    Convenience function to compute entropy coefficient from CLV.

    Example:
        from tfan.agent.adaptive_entropy import compute_entropy_coef_from_clv

        alpha = compute_entropy_coef_from_clv(clv)
        loss = policy_loss - alpha * entropy
    """
    return get_entropy_controller().update_from_clv(clv)


def compute_entropy_coef(
    instability: float = 0.0,
    resource: float = 0.0,
    structural: float = 0.0,
    arousal: float = 0.5,
) -> float:
    """
    Convenience function to compute entropy coefficient from raw values.

    Example:
        alpha = compute_entropy_coef(
            instability=0.3,
            resource=0.2,
            structural=0.1,
            arousal=0.6
        )
    """
    return get_entropy_controller().update_from_raw(
        instability, resource, structural, arousal
    )


def get_exploration_mode() -> str:
    """Get current exploration mode (conservative/balanced/exploratory)."""
    return get_entropy_controller()._state.exploration_mode


# =============================================================================
# INTEGRATION WITH AEPO
# =============================================================================

class CLVModulatedAEPO:
    """
    Wrapper that adds CLV-modulated entropy to existing AEPO.

    This doesn't replace AEPO, just wraps the loss computation
    to use adaptive entropy.
    """

    def __init__(
        self,
        aepo,  # The underlying AEPO module
        entropy_controller: Optional[AdaptiveEntropyController] = None,
    ):
        """
        Initialize CLV-modulated AEPO wrapper.

        Args:
            aepo: The underlying AEPO policy network
            entropy_controller: Entropy controller (uses global if None)
        """
        self.aepo = aepo
        self.entropy_controller = entropy_controller or get_entropy_controller()

        # Store original entropy coef
        self._original_ent_coef = getattr(aepo, 'ent_coef', 0.01)

    def update_entropy_from_clv(self, clv: "CognitiveLoadVector"):
        """Update AEPO entropy coefficient from CLV."""
        new_alpha = self.entropy_controller.update_from_clv(clv)

        # Update AEPO's entropy coefficient
        if hasattr(self.aepo, 'ent_coef'):
            self.aepo.ent_coef = new_alpha

        return new_alpha

    def compute_loss_with_clv(
        self,
        obs,
        advantages,
        clv: Optional["CognitiveLoadVector"] = None,
    ):
        """
        Compute AEPO loss with CLV-modulated entropy.

        Args:
            obs: Observations
            advantages: Advantage dict
            clv: Optional CLV for entropy modulation

        Returns:
            (loss, info) tuple
        """
        # Update entropy if CLV provided
        if clv is not None:
            self.update_entropy_from_clv(clv)

        # Compute loss using AEPO's normal method
        loss, info = self.aepo.loss(obs, advantages)

        # Add entropy controller info
        info["adaptive_alpha"] = self.entropy_controller.get_current_alpha()
        info["exploration_mode"] = self.entropy_controller._state.exploration_mode
        info["risk_score"] = self.entropy_controller._state.risk_score

        return loss, info

    def get_action(self, obs, deterministic: bool = False):
        """Get action from underlying AEPO."""
        return self.aepo.get_action(obs, deterministic)

    def get_entropy_stats(self) -> Dict[str, Any]:
        """Get entropy controller statistics."""
        return self.entropy_controller.get_history_stats()


__all__ = [
    "AdaptiveEntropyConfig",
    "EntropyState",
    "AdaptiveEntropyController",
    "CLVModulatedAEPO",
    "get_entropy_controller",
    "compute_entropy_coef_from_clv",
    "compute_entropy_coef",
    "get_exploration_mode",
]
