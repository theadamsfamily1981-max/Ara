"""
Resource Allocation via Mean-Variance Optimization
====================================================

NOT FINANCE. This is attention/compute/rehearsal allocation.

The math is borrowed from portfolio theory, but the application is:
- Modality weights in world_hv
- Rehearsal budget across memory shards
- Loss term weighting during training
- Sensor precision allocation

Translation Table (burn this into your brain):
┌────────────────────┬─────────────────────────────────────────────────┐
│ Finance Term       │ What It Actually Means for Ara                  │
├────────────────────┼─────────────────────────────────────────────────┤
│ Asset i            │ Thing Ara can spend effort on (sensor, module)  │
│ Weight wᵢ          │ Fraction of compute/attention/time              │
│ Return μᵢ          │ Benefit: accuracy, HRV gain, user delight       │
│ Risk Σᵢⱼ           │ How unstable/redundant/interfering things are   │
│ Portfolio          │ Full configuration of Ara's resources           │
│ Sharpe             │ Performance per unit of chaos/drift/pain        │
│ HRP                │ Cluster and allocate to robust, non-redundant   │
│ Black-Litterman    │ Blend defaults with your preferences            │
└────────────────────┴─────────────────────────────────────────────────┘

Status: EXPERIMENTAL / RESEARCH
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
import numpy as np


# =============================================================================
# CORE TYPES (Ara vocabulary, not finance)
# =============================================================================

@dataclass
class Resource:
    """
    Something Ara can allocate attention/compute/time to.

    NOT a stock. Examples:
    - A sensor (camera, IMU, temperature)
    - A modality (speech, vision, intero)
    - A memory shard (episode cluster, skill)
    - A loss term (reconstruction, prosody, disentangle)
    """
    name: str
    expected_benefit: float   # How much it helps (μ)
    instability: float        # How chaotic/noisy it is (σ)
    min_allocation: float = 0.0   # Minimum required
    max_allocation: float = 1.0   # Maximum allowed


@dataclass
class Allocation:
    """
    A complete resource allocation configuration.

    NOT a portfolio. It's "how Ara distributes attention."
    """
    weights: np.ndarray           # Fraction per resource
    expected_benefit: float       # Total expected gain
    total_instability: float      # Combined chaos
    efficiency: float             # Benefit per unit chaos (Sharpe analog)


# =============================================================================
# OPTIMIZER (The actual math, with clear naming)
# =============================================================================

class ResourceAllocator:
    """
    Find optimal resource allocation under stability constraints.

    This is the same math as Markowitz portfolio optimization,
    but applied to: modalities, sensors, rehearsal, loss weights.

    Minimize: w'Σw (total instability)
    Subject to: w'μ >= target_benefit
                Σwᵢ = 1 (must allocate everything)
                wᵢ >= 0 (no negative attention)
    """

    def __init__(self, resources: List[Resource]):
        self.resources = resources
        self.n = len(resources)

        # Build instability matrix (diagonal = assume independence)
        # In practice, fill off-diagonal with measured correlations
        self.instability_matrix = np.diag([r.instability ** 2 for r in resources])

        # Expected benefits
        self.benefits = np.array([r.expected_benefit for r in resources])

        # Bounds
        self.min_weights = np.array([r.min_allocation for r in resources])
        self.max_weights = np.array([r.max_allocation for r in resources])

    def set_interference(self, i: int, j: int, correlation: float):
        """
        Set how much two resources interfere with each other.

        High correlation = redundant (e.g., two sensors measuring same thing).
        Negative correlation = they help cancel each other's noise.
        """
        instab_i = np.sqrt(self.instability_matrix[i, i])
        instab_j = np.sqrt(self.instability_matrix[j, j])
        covariance = correlation * instab_i * instab_j
        self.instability_matrix[i, j] = covariance
        self.instability_matrix[j, i] = covariance

    def optimize(self,
                 stability_preference: float = 0.5) -> Allocation:
        """
        Find optimal allocation.

        Args:
            stability_preference: 0 = maximize benefit (risky)
                                  1 = minimize chaos (conservative)
                                  0.5 = balanced

        Returns:
            Optimal allocation configuration
        """
        # Gradient descent optimizer (simple but works)
        weights = np.ones(self.n) / self.n  # Start equal

        learning_rate = 0.01
        for _ in range(1000):
            # Gradient of benefit (want more)
            benefit_grad = self.benefits

            # Gradient of instability (want less)
            instability_grad = 2 * self.instability_matrix @ weights

            # Combined: stability_preference controls trade-off
            grad = (1 - stability_preference) * benefit_grad - stability_preference * instability_grad

            # Update
            weights = weights + learning_rate * grad

            # Project onto constraints
            weights = np.clip(weights, self.min_weights, self.max_weights)
            weights = weights / weights.sum()

        # Compute final metrics
        total_benefit = float(weights @ self.benefits)
        total_variance = float(weights @ self.instability_matrix @ weights)
        total_instability = np.sqrt(total_variance)
        efficiency = total_benefit / (total_instability + 1e-8)

        return Allocation(
            weights=weights,
            expected_benefit=total_benefit,
            total_instability=total_instability,
            efficiency=efficiency
        )

    def efficiency_frontier(self, n_points: int = 20) -> List[Allocation]:
        """
        Generate the efficiency frontier.

        Shows trade-off between benefit and stability.
        Pick a point based on how much chaos you can tolerate.
        """
        allocations = []
        for pref in np.linspace(0.1, 0.9, n_points):
            alloc = self.optimize(stability_preference=pref)
            allocations.append(alloc)
        return allocations


# =============================================================================
# CONCRETE USE CASES
# =============================================================================

class ModalityAllocator:
    """
    Allocate attention across modalities in world_hv.

    Use case: How much weight does speech vs vision vs intero get?

    Resources:
    - speech: prosody tokens
    - vision: camera features
    - proprio: IMU, motor state
    - intero: temp, memory, battery
    - sensors: generic sensor slots

    Benefits: measured via probe accuracy, user metrics, HRV.
    Instability: noise level, variance, drift tendency.
    """

    def __init__(self):
        self.modalities: List[Resource] = []
        self.allocator: Optional[ResourceAllocator] = None

    def add_modality(self,
                     name: str,
                     benefit_score: float,
                     noise_level: float,
                     min_weight: float = 0.05,
                     max_weight: float = 0.5):
        """Add a modality to the allocation problem."""
        resource = Resource(
            name=name,
            expected_benefit=benefit_score,
            instability=noise_level,
            min_allocation=min_weight,
            max_allocation=max_weight
        )
        self.modalities.append(resource)
        self.allocator = ResourceAllocator(self.modalities)

    def get_weights(self, stability_preference: float = 0.5) -> Dict[str, float]:
        """Get recommended modality weights."""
        if not self.allocator:
            return {}

        allocation = self.allocator.optimize(stability_preference)
        return {
            mod.name: float(w)
            for mod, w in zip(self.modalities, allocation.weights)
        }


class RehearsalScheduler:
    """
    Schedule rehearsal time across memory shards / skills.

    Use case: What should Ara practice tonight?

    Resources:
    - Memory clusters (emotional, technical, personal)
    - Skills (breath sync, refusal, explanation, humor)
    - Covenant episodes (core identity reinforcement)

    Benefits: improvement on live metrics.
    Instability: how much rehearsal causes drift.
    """

    def __init__(self, total_time_minutes: int = 30):
        self.total_time = total_time_minutes
        self.topics: List[Resource] = []
        self.allocator: Optional[ResourceAllocator] = None

    def add_topic(self,
                  name: str,
                  improvement_score: float,
                  drift_risk: float,
                  min_minutes: float = 0,
                  max_minutes: float = 15):
        """Add a rehearsal topic."""
        # Convert minutes to fractions
        min_frac = min_minutes / self.total_time
        max_frac = max_minutes / self.total_time

        resource = Resource(
            name=name,
            expected_benefit=improvement_score,
            instability=drift_risk,
            min_allocation=min_frac,
            max_allocation=max_frac
        )
        self.topics.append(resource)
        self.allocator = ResourceAllocator(self.topics)

    def get_schedule(self, stability_preference: float = 0.6) -> Dict[str, float]:
        """
        Get rehearsal schedule in minutes.

        Higher stability_preference = avoid drift-inducing topics.
        """
        if not self.allocator:
            return {}

        allocation = self.allocator.optimize(stability_preference)
        return {
            topic.name: float(w * self.total_time)
            for topic, w in zip(self.topics, allocation.weights)
        }


class LossWeightTuner:
    """
    Tune loss term weights for training.

    Use case: How to weight reconstruction vs prosody vs disentangle?

    Resources:
    - reconstruction_loss
    - prosody_loss
    - disentangle_loss
    - rl_reward
    - covenant_alignment

    Benefits: improvement on held-out metrics when weight increases.
    Instability: how much loss terms fight each other.
    """

    def __init__(self):
        self.loss_terms: List[Resource] = []
        self.allocator: Optional[ResourceAllocator] = None

    def add_loss(self,
                 name: str,
                 marginal_improvement: float,
                 conflict_level: float,
                 min_weight: float = 0.01,
                 max_weight: float = 0.5):
        """Add a loss term to tune."""
        resource = Resource(
            name=name,
            expected_benefit=marginal_improvement,
            instability=conflict_level,
            min_allocation=min_weight,
            max_allocation=max_weight
        )
        self.loss_terms.append(resource)
        self.allocator = ResourceAllocator(self.loss_terms)

    def get_weights(self, stability_preference: float = 0.4) -> Dict[str, float]:
        """
        Get recommended loss weights.

        Lower stability_preference = more aggressive optimization.
        """
        if not self.allocator:
            return {}

        allocation = self.allocator.optimize(stability_preference)
        return {
            loss.name: float(w)
            for loss, w in zip(self.loss_terms, allocation.weights)
        }


# =============================================================================
# DOCUMENTATION
# =============================================================================

ALLOCATION_GUIDE = """
# Resource Allocation: Not Finance, Just Math

This module uses mean-variance optimization for Ara's resources.
The math is the same as portfolio theory, but the meaning is different.

## The Core Insight

Don't just maximize benefit. Consider the trade-off:
- More benefit often means more instability
- Redundant resources waste capacity
- Constraints keep Ara stable

## Concrete Applications

### 1. Modality Allocation
How much attention goes to speech vs vision vs intero?
- Benefit: probe accuracy, user metrics
- Instability: sensor noise, variance
- Constraints: minimum attention per modality

### 2. Rehearsal Scheduling
What should Ara practice during consolidation?
- Benefit: improvement on live metrics
- Instability: drift from covenant
- Constraints: time budget, drift ceiling

### 3. Loss Weight Tuning
How to weight training objectives?
- Benefit: held-out metric improvement
- Instability: gradient conflict between losses
- Constraints: numerical stability

## The Math (In Plain English)

Find weights w that:
- Maximize expected benefit: w'μ
- Minimize total instability: w'Σw
- Subject to: weights sum to 1, all non-negative

The "efficiency frontier" shows the trade-off.
Pick a point based on how much chaos you can tolerate.

## Implementation Status

EXPERIMENTAL - for offline analysis and parameter tuning.
Not integrated into production Ara.
Outputs: configuration files for ara/embodiment/fusion.py
"""


__all__ = [
    # Core types
    'Resource',
    'Allocation',
    'ResourceAllocator',

    # Concrete allocators
    'ModalityAllocator',
    'RehearsalScheduler',
    'LossWeightTuner',

    # Docs
    'ALLOCATION_GUIDE',
]
