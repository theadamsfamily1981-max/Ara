"""
L5 Meta-Learning: AEPO learns L3 Control Laws

This module implements meta-learning over the L3 metacontrol parameters,
allowing the system to learn optimal emotional control policies rather
than relying on hand-tuned values.

The L5MetaLearner:
- Exposes L3 parameters as AEPO action space
- Uses antifragility metrics as reward signal
- Learns on slow timescale (per session/workload)
- Persists learned profiles for GNOME cockpit display

Key L3 Parameters (Action Space):
- jerk_threshold: [0.05, 0.3] - State change rate sensitivity
- controller_weight: [0.1, 0.5] - Blend weight for PAD gating
- arousal_temp_scale: [0.3, 0.7] - Arousal → Temperature coupling
- valence_mem_scale: [0.3, 0.7] - Valence → Memory coupling
- curvature_c: [0.5, 2.0] - Hyperbolic curvature (geometry)

Reward Signal:
- antifragility_score (weight: 0.4)
- delta_p99_percent (weight: 0.3)
- 1 - clv_risk_normalized (weight: 0.2)
- pgu_pass_rate (weight: 0.1)
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from enum import Enum
import logging
import json
import math

# Add paths
_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

logger = logging.getLogger("tfan.l5.meta_learner")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None


# =============================================================================
# L3 CONTROL PARAMETERS (ACTION SPACE)
# =============================================================================

@dataclass
class L3ControlParams:
    """
    L3 Metacontrol parameters that L5 learns to optimize.

    These are the "control laws" that define emotional policy.
    """
    # Jerk threshold: Higher = more tolerant of rapid state changes
    jerk_threshold: float = 0.1  # [0.05, 0.3]

    # Controller weight: Higher = stronger PAD influence
    controller_weight: float = 0.3  # [0.1, 0.5]

    # Arousal → Temperature coupling scale
    arousal_temp_scale: float = 0.5  # [0.3, 0.7]

    # Valence → Memory coupling scale
    valence_mem_scale: float = 0.5  # [0.3, 0.7]

    # Confidence threshold for full control
    confidence_threshold: float = 0.5  # [0.3, 0.8]

    # Hyperbolic curvature (geometry)
    curvature_c: float = 1.0  # [0.5, 2.0]

    # Profile metadata
    profile_name: str = "default"
    profile_version: int = 1
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_action_vector(self) -> List[float]:
        """Convert to normalized action vector [0, 1] for AEPO."""
        return [
            (self.jerk_threshold - 0.05) / 0.25,      # [0.05, 0.3] → [0, 1]
            (self.controller_weight - 0.1) / 0.4,     # [0.1, 0.5] → [0, 1]
            (self.arousal_temp_scale - 0.3) / 0.4,    # [0.3, 0.7] → [0, 1]
            (self.valence_mem_scale - 0.3) / 0.4,     # [0.3, 0.7] → [0, 1]
            (self.confidence_threshold - 0.3) / 0.5,  # [0.3, 0.8] → [0, 1]
            (self.curvature_c - 0.5) / 1.5,           # [0.5, 2.0] → [0, 1]
        ]

    @classmethod
    def from_action_vector(cls, action: List[float], name: str = "learned") -> "L3ControlParams":
        """Create from normalized action vector [0, 1]."""
        return cls(
            jerk_threshold=0.05 + action[0] * 0.25,
            controller_weight=0.1 + action[1] * 0.4,
            arousal_temp_scale=0.3 + action[2] * 0.4,
            valence_mem_scale=0.3 + action[3] * 0.4,
            confidence_threshold=0.3 + action[4] * 0.5,
            curvature_c=0.5 + action[5] * 1.5,
            profile_name=name,
        )

    def save(self, path: str):
        """Save profile to YAML/JSON."""
        data = self.to_dict()
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved L3 profile to {path}")

    @classmethod
    def load(cls, path: str) -> "L3ControlParams":
        """Load profile from YAML/JSON."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


# =============================================================================
# REWARD COMPUTATION
# =============================================================================

@dataclass
class L5RewardSignal:
    """
    Reward signal for L5 meta-learning.

    Combines multiple objectives into single scalar reward.
    """
    antifragility_score: float = 0.0  # >1.0 = antifragile
    delta_p99_percent: float = 0.0    # Latency improvement %
    clv_risk_level: str = "UNKNOWN"   # LOW/MEDIUM/HIGH/CRITICAL
    pgu_pass_rate: float = 1.0        # PGU verification success rate
    throughput_qps: float = 0.0       # Queries per second

    # Weights for combining objectives
    WEIGHTS = {
        "antifragility": 0.4,
        "delta_p99": 0.3,
        "risk": 0.2,
        "pgu": 0.1,
    }

    def compute_reward(self) -> float:
        """
        Compute scalar reward from multi-objective metrics.

        Returns:
            Normalized reward in [0, 1]
        """
        # Normalize antifragility: 1.0 = baseline, 3.0 = excellent
        af_norm = min(1.0, max(0.0, (self.antifragility_score - 1.0) / 2.0))

        # Normalize delta_p99: 0% = baseline, 100% = excellent
        dp99_norm = min(1.0, max(0.0, self.delta_p99_percent / 100.0))

        # Normalize risk: LOW = 1.0, CRITICAL = 0.0
        risk_map = {"LOW": 1.0, "MEDIUM": 0.66, "HIGH": 0.33, "CRITICAL": 0.0, "UNKNOWN": 0.5}
        risk_norm = risk_map.get(self.clv_risk_level, 0.5)

        # PGU pass rate already in [0, 1]
        pgu_norm = self.pgu_pass_rate

        # Weighted combination
        reward = (
            self.WEIGHTS["antifragility"] * af_norm +
            self.WEIGHTS["delta_p99"] * dp99_norm +
            self.WEIGHTS["risk"] * risk_norm +
            self.WEIGHTS["pgu"] * pgu_norm
        )

        return reward

    def to_dict(self) -> Dict[str, Any]:
        return {
            "antifragility_score": self.antifragility_score,
            "delta_p99_percent": self.delta_p99_percent,
            "clv_risk_level": self.clv_risk_level,
            "pgu_pass_rate": self.pgu_pass_rate,
            "throughput_qps": self.throughput_qps,
            "computed_reward": self.compute_reward(),
        }


# =============================================================================
# L5 META-LEARNER
# =============================================================================

class L5MetaLearner:
    """
    L5 Meta-Learning: Learns optimal L3 control parameters.

    Uses simple evolutionary strategy (ES) for sample-efficient
    optimization over the L3 parameter space.

    The learner:
    1. Proposes L3 parameter perturbations
    2. Evaluates via stress test / certification
    3. Updates parameters based on reward gradient
    4. Persists best profile for deployment

    This operates on a slow timescale (per session/workload class).
    """

    def __init__(
        self,
        initial_params: Optional[L3ControlParams] = None,
        learning_rate: float = 0.05,
        noise_std: float = 0.1,
        population_size: int = 8,
        elite_ratio: float = 0.25,
        max_iterations: int = 20,
        profile_dir: str = "profiles/l3",
    ):
        """
        Initialize L5 meta-learner.

        Args:
            initial_params: Starting L3 parameters
            learning_rate: Step size for parameter updates
            noise_std: Standard deviation for perturbations
            population_size: Number of parallel evaluations
            elite_ratio: Fraction of best samples to use
            max_iterations: Maximum learning iterations
            profile_dir: Directory for persisting profiles
        """
        self.params = initial_params or L3ControlParams()
        self.learning_rate = learning_rate
        self.noise_std = noise_std
        self.population_size = population_size
        self.elite_count = max(1, int(population_size * elite_ratio))
        self.max_iterations = max_iterations
        self.profile_dir = Path(profile_dir)

        # History
        self.reward_history: List[float] = []
        self.params_history: List[L3ControlParams] = []
        self.best_reward: float = 0.0
        self.best_params: Optional[L3ControlParams] = None
        self.iteration: int = 0

        # Create profile directory
        self.profile_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"L5MetaLearner initialized with pop_size={population_size}")

    def propose_candidates(self) -> List[L3ControlParams]:
        """
        Generate candidate L3 parameter sets for evaluation.

        Uses Gaussian perturbation around current best.

        Returns:
            List of L3ControlParams candidates
        """
        if not NUMPY_AVAILABLE:
            # Fallback: return current params with small variations
            candidates = [self.params]
            for i in range(self.population_size - 1):
                action = self.params.to_action_vector()
                perturbed = [max(0, min(1, a + (i * 0.1 - 0.2))) for a in action]
                candidates.append(L3ControlParams.from_action_vector(perturbed, f"candidate_{i}"))
            return candidates

        # Get current action vector
        current = np.array(self.params.to_action_vector())

        # Generate perturbations
        candidates = []
        for i in range(self.population_size):
            noise = np.random.randn(len(current)) * self.noise_std
            perturbed = np.clip(current + noise, 0, 1)
            params = L3ControlParams.from_action_vector(
                perturbed.tolist(),
                name=f"candidate_{i}"
            )
            candidates.append(params)

        return candidates

    def update_from_rewards(
        self,
        candidates: List[L3ControlParams],
        rewards: List[float],
    ) -> L3ControlParams:
        """
        Update parameters based on evaluated rewards.

        Uses elite selection + weighted mean.

        Args:
            candidates: List of evaluated candidates
            rewards: Corresponding reward values

        Returns:
            Updated L3ControlParams
        """
        if not NUMPY_AVAILABLE:
            # Fallback: take best candidate
            best_idx = max(range(len(rewards)), key=lambda i: rewards[i])
            return candidates[best_idx]

        # Sort by reward (descending)
        sorted_indices = np.argsort(rewards)[::-1]
        elite_indices = sorted_indices[:self.elite_count]

        # Weighted mean of elite candidates
        elite_rewards = np.array([rewards[i] for i in elite_indices])
        elite_weights = elite_rewards / (elite_rewards.sum() + 1e-8)

        # Compute weighted mean action vector
        elite_actions = np.array([candidates[i].to_action_vector() for i in elite_indices])
        mean_action = np.sum(elite_actions * elite_weights[:, None], axis=0)

        # Blend with current (momentum)
        current = np.array(self.params.to_action_vector())
        updated = current + self.learning_rate * (mean_action - current)
        updated = np.clip(updated, 0, 1)

        # Create new params
        self.iteration += 1
        new_params = L3ControlParams.from_action_vector(
            updated.tolist(),
            name=f"learned_v{self.iteration}"
        )
        new_params.profile_version = self.iteration

        # Track best
        best_reward = max(rewards)
        if best_reward > self.best_reward:
            self.best_reward = best_reward
            self.best_params = candidates[sorted_indices[0]]
            logger.info(f"New best reward: {best_reward:.4f}")

        # Record history
        self.reward_history.append(best_reward)
        self.params_history.append(new_params)
        self.params = new_params

        return new_params

    def save_best_profile(self, filename: str = "best.json"):
        """Save best discovered profile."""
        if self.best_params:
            path = self.profile_dir / filename
            self.best_params.save(str(path))
            logger.info(f"Saved best profile: {path}")

    def load_profile(self, filename: str = "best.json") -> Optional[L3ControlParams]:
        """Load saved profile."""
        path = self.profile_dir / filename
        if path.exists():
            self.params = L3ControlParams.load(str(path))
            logger.info(f"Loaded profile: {path}")
            return self.params
        return None

    def get_status(self) -> Dict[str, Any]:
        """Get meta-learner status for cockpit display."""
        return {
            "iteration": self.iteration,
            "best_reward": self.best_reward,
            "current_params": self.params.to_dict(),
            "best_params": self.best_params.to_dict() if self.best_params else None,
            "reward_history": self.reward_history[-10:],  # Last 10
            "profile_name": self.params.profile_name,
            "profile_version": self.params.profile_version,
        }


# =============================================================================
# ADAPTIVE PERSONALITY (GNOME UX INTEGRATION)
# =============================================================================

class PersonalityProfile(Enum):
    """
    Named personality profiles derived from learned L3 parameters.

    The system learns to exhibit different "personalities" based
    on workload characteristics and environmental conditions.
    """
    CAUTIOUS_STABLE = "cautious_stable"       # Low jerk, high confidence threshold
    REACTIVE_ADAPTIVE = "reactive_adaptive"   # High jerk, fast response
    BALANCED_GENERAL = "balanced_general"     # Middle ground
    EXPLORATORY_CREATIVE = "exploratory"      # High arousal coupling
    CONSERVATIVE_SAFE = "conservative"        # Low arousal, high memory coupling


def classify_personality(params: L3ControlParams) -> PersonalityProfile:
    """
    Classify L3 parameters into a named personality profile.

    Used for GNOME cockpit display.
    """
    jerk = params.jerk_threshold
    weight = params.controller_weight
    arousal = params.arousal_temp_scale
    valence = params.valence_mem_scale

    # Classification rules
    if jerk < 0.1 and weight > 0.35:
        return PersonalityProfile.CAUTIOUS_STABLE
    elif jerk > 0.2 and arousal > 0.55:
        return PersonalityProfile.REACTIVE_ADAPTIVE
    elif arousal > 0.6 and valence < 0.4:
        return PersonalityProfile.EXPLORATORY_CREATIVE
    elif valence > 0.6 and arousal < 0.4:
        return PersonalityProfile.CONSERVATIVE_SAFE
    else:
        return PersonalityProfile.BALANCED_GENERAL


@dataclass
class CockpitPersonalityDisplay:
    """
    Data structure for GNOME cockpit personality display.
    """
    profile_name: str
    personality_type: str
    version: int

    # Key learned values
    jerk_threshold: float
    arousal_temp_coupling: float
    valence_mem_coupling: float
    curvature: float

    # Performance
    best_reward: float
    antifragility_score: float

    # Timestamps
    last_updated: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def create_cockpit_display(
    params: L3ControlParams,
    meta_learner: L5MetaLearner,
    antifragility_score: float = 0.0,
) -> CockpitPersonalityDisplay:
    """
    Create cockpit display data from current state.
    """
    personality = classify_personality(params)

    return CockpitPersonalityDisplay(
        profile_name=params.profile_name,
        personality_type=personality.value,
        version=params.profile_version,
        jerk_threshold=params.jerk_threshold,
        arousal_temp_coupling=params.arousal_temp_scale,
        valence_mem_coupling=params.valence_mem_scale,
        curvature=params.curvature_c,
        best_reward=meta_learner.best_reward,
        antifragility_score=antifragility_score,
        last_updated=params.last_updated,
    )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global meta-learner instance
_meta_learner: Optional[L5MetaLearner] = None


def get_meta_learner() -> L5MetaLearner:
    """Get or create global meta-learner."""
    global _meta_learner
    if _meta_learner is None:
        _meta_learner = L5MetaLearner()
    return _meta_learner


def propose_l3_candidates() -> List[L3ControlParams]:
    """Propose L3 parameter candidates for evaluation."""
    return get_meta_learner().propose_candidates()


def update_l3_from_rewards(
    candidates: List[L3ControlParams],
    rewards: List[float],
) -> L3ControlParams:
    """Update L3 parameters from evaluated rewards."""
    return get_meta_learner().update_from_rewards(candidates, rewards)


def get_current_personality() -> Dict[str, Any]:
    """Get current personality profile for display."""
    ml = get_meta_learner()
    return create_cockpit_display(ml.params, ml).to_dict()


__all__ = [
    "L3ControlParams",
    "L5RewardSignal",
    "L5MetaLearner",
    "PersonalityProfile",
    "CockpitPersonalityDisplay",
    "classify_personality",
    "create_cockpit_display",
    "get_meta_learner",
    "propose_l3_candidates",
    "update_l3_from_rewards",
    "get_current_personality",
]
