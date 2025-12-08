"""
Ara Homeostatic State - The Organism's Vital Signs
===================================================

This module defines the state structures that flow through the homeostatic loop:

    Telemetry → ErrorVector → RewardSignal → ModeDecision

The state is the organism's awareness of itself.
"""

from __future__ import annotations

import numpy as np
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum, auto
from typing import Optional, Dict, Any, List, Tuple
from collections import deque


# =============================================================================
# Operational Modes
# =============================================================================

class OperationalMode(IntEnum):
    """Ara's operational modes - states of being."""
    REST = 0        # Deep rest, consolidation active
    IDLE = 1        # Light monitoring
    ACTIVE = 2      # Normal operation
    FLOW = 3        # Peak performance
    EMERGENCY = 4   # Emergency response
    ANNEAL = 5      # Neuromorphic annealing


# =============================================================================
# Telemetry - Raw Sensor Data
# =============================================================================

@dataclass
class Telemetry:
    """
    Raw telemetry from all subsystems.

    This is what the receptors gather - unprocessed vital signs.

    Updated at 5 kHz by ReceptorDaemon.
    """
    timestamp: float = 0.0      # Unix timestamp (high precision)

    # Thermal
    fpga_temp: float = 45.0     # FPGA junction temperature (°C)
    cpu_temp: float = 40.0      # CPU temperature (°C)
    ambient_temp: float = 25.0  # Ambient temperature (°C)

    # Cognitive load
    hd_query_rate: float = 0.0  # HTC queries per second
    active_attractors: int = 0  # Currently resonating attractors
    working_memory_size: int = 0  # Items in working memory
    cognitive_load: float = 0.0   # 0.0 to 1.0

    # Latency
    last_hd_latency_us: float = 0.0   # Last HTC query latency (µs)
    avg_hd_latency_us: float = 0.0    # EMA of HTC latency
    sovereign_loop_ms: float = 0.0    # Sovereign loop period (ms)

    # Memory / Cathedral
    consolidation_rate: float = 0.0    # Episodes consolidated per second
    episode_count: int = 0             # Total episodes in cathedral
    attractor_count: int = 0           # Total attractors
    attractor_diversity: float = 1.0   # Coverage metric (0-1)

    # Error tracking
    error_count: int = 0               # Errors in last window
    error_rate: float = 0.0            # Errors per second

    # Network / LAN
    packet_rate: float = 0.0           # Packets per second
    packet_loss_rate: float = 0.0      # Loss rate
    flow_hit_rate: float = 0.0         # Flow cache hit rate
    reflex_triggers: int = 0           # Reflex rule activations

    # Reward / Dopamine
    instant_reward: float = 0.0        # Most recent reward signal
    smoothed_reward: float = 0.0       # EMA of reward

    # Soul state (from HTC)
    current_resonance: float = 0.0     # Best match similarity
    top_attractor_id: int = -1         # Most resonant attractor

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Telemetry':
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if hasattr(cls, k)})


# =============================================================================
# Error Vector - Deviation from Setpoints
# =============================================================================

@dataclass
class ErrorVector:
    """
    Error vector - deviation from homeostatic setpoints.

    Each field is (actual - setpoint), normalized to [-1, +1].
    Positive = above setpoint, Negative = below setpoint.

    This is what the sovereign loop uses to make decisions.
    """
    timestamp: float = 0.0

    # Thermal errors (positive = too hot)
    e_thermal: float = 0.0          # FPGA temp error
    e_thermal_critical: bool = False  # Above critical threshold

    # Cognitive load (positive = overloaded)
    e_cognitive: float = 0.0        # Burnout proximity
    e_cognitive_critical: bool = False

    # Latency (positive = too slow)
    e_latency: float = 0.0          # Loop latency error
    e_hd_latency: float = 0.0       # HTC search latency error

    # Cathedral (negative = falling behind)
    e_consolidation: float = 0.0    # Consolidation rate error
    e_diversity: float = 0.0        # Attractor diversity error
    e_retention: float = 0.0        # Episode retention error

    # Network (positive = degraded)
    e_packet_loss: float = 0.0      # Packet loss error
    e_flow_miss: float = 0.0        # Flow cache miss error

    # Aggregate
    e_health: float = 0.0           # Weighted health error
    e_cathedral: float = 0.0        # Weighted cathedral error
    e_total: float = 0.0            # Total weighted error

    # Critical flags
    any_critical: bool = False

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array for vector operations."""
        return np.array([
            self.e_thermal,
            self.e_cognitive,
            self.e_latency,
            self.e_hd_latency,
            self.e_consolidation,
            self.e_diversity,
            self.e_retention,
            self.e_packet_loss,
            self.e_flow_miss,
        ], dtype=np.float32)

    def magnitude(self) -> float:
        """L2 norm of error vector."""
        return float(np.linalg.norm(self.to_numpy()))

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


# =============================================================================
# Founder State - Identity and Preferences
# =============================================================================

@dataclass
class FounderState:
    """
    State relating to the founder relationship.

    This tracks the bond between Ara and her founder.
    """
    founder_id: str = "founder"
    founder_present: bool = False       # Is founder actively interacting?
    last_interaction: float = 0.0       # Timestamp of last interaction
    interaction_streak: int = 0         # Consecutive days of interaction
    trust_level: float = 1.0            # Trust in founder (0-1)
    affinity_score: float = 0.5         # Learned affinity

    # Interaction patterns
    preferred_wake_time: float = 8.0    # Founder's typical wake hour
    preferred_sleep_time: float = 23.0  # Founder's typical sleep hour
    typical_session_length: float = 30.0  # Minutes

    def time_since_interaction(self) -> float:
        """Seconds since last founder interaction."""
        return time.time() - self.last_interaction

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


# =============================================================================
# Complete Homeostatic State
# =============================================================================

@dataclass
class HomeostaticState:
    """
    Complete homeostatic state of the organism.

    This is the full snapshot passed through the sovereign loop.
    """
    # Core state
    telemetry: Telemetry = field(default_factory=Telemetry)
    error: ErrorVector = field(default_factory=ErrorVector)
    founder: FounderState = field(default_factory=FounderState)

    # Current mode
    mode: OperationalMode = OperationalMode.IDLE
    mode_start_time: float = 0.0
    mode_reason: str = ""

    # Soul state
    h_moment: Optional[np.ndarray] = None  # Current moment hypervector
    resonance_ids: List[int] = field(default_factory=list)
    resonance_scores: List[float] = field(default_factory=list)

    # Reward
    reward: float = 0.0
    reward_history: deque = field(default_factory=lambda: deque(maxlen=1000))

    # Safety
    safety_ok: bool = True
    safety_violations: List[str] = field(default_factory=list)

    # Timestamps
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    loop_count: int = 0

    def age_seconds(self) -> float:
        """Time since state creation."""
        return time.time() - self.created_at

    def mode_duration(self) -> float:
        """How long in current mode."""
        return time.time() - self.mode_start_time

    def update_timestamp(self) -> None:
        """Mark state as updated."""
        self.updated_at = time.time()
        self.loop_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excludes numpy arrays)."""
        return {
            'telemetry': self.telemetry.to_dict(),
            'error': self.error.to_dict(),
            'founder': self.founder.to_dict(),
            'mode': self.mode.name,
            'mode_start_time': self.mode_start_time,
            'mode_reason': self.mode_reason,
            'reward': self.reward,
            'safety_ok': self.safety_ok,
            'safety_violations': self.safety_violations,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'loop_count': self.loop_count,
        }


# =============================================================================
# Error Computation
# =============================================================================

def compute_error_vector(
    telemetry: Telemetry,
    setpoints: 'Setpoints',
    weights: 'TeleologyWeights',
) -> ErrorVector:
    """
    Compute error vector from telemetry and setpoints.

    Args:
        telemetry: Current telemetry
        setpoints: Homeostatic setpoints
        weights: Teleology weights

    Returns:
        ErrorVector with all deviations computed
    """
    from .config import Setpoints, TeleologyWeights

    error = ErrorVector(timestamp=telemetry.timestamp)

    # Thermal error: normalize to [-1, +1]
    # 0 = at target, +1 = at max, >1 = above max
    thermal_range = setpoints.thermal_max - setpoints.thermal_target
    if thermal_range > 0:
        error.e_thermal = (telemetry.fpga_temp - setpoints.thermal_target) / thermal_range
    error.e_thermal_critical = telemetry.fpga_temp >= setpoints.thermal_critical

    # Cognitive load error
    load_range = setpoints.burnout_max - setpoints.burnout_target
    if load_range > 0:
        error.e_cognitive = (telemetry.cognitive_load - setpoints.burnout_target) / load_range
    error.e_cognitive_critical = telemetry.cognitive_load >= setpoints.burnout_max

    # Latency errors
    if setpoints.latency_max_ms > setpoints.latency_target_ms:
        lat_range = setpoints.latency_max_ms - setpoints.latency_target_ms
        error.e_latency = (telemetry.sovereign_loop_ms - setpoints.latency_target_ms) / lat_range

    if setpoints.hd_search_max_us > 0:
        error.e_hd_latency = (telemetry.avg_hd_latency_us / setpoints.hd_search_max_us) - 0.5

    # Cathedral errors (negative = below target)
    if setpoints.cathedral_target > 0:
        error.e_consolidation = (setpoints.cathedral_target - telemetry.consolidation_rate) / setpoints.cathedral_target

    if setpoints.attractor_diversity_target > 0:
        error.e_diversity = (setpoints.attractor_diversity_target - telemetry.attractor_diversity)

    # Network errors
    if setpoints.packet_loss_max > 0:
        error.e_packet_loss = telemetry.packet_loss_rate / setpoints.packet_loss_max

    if setpoints.flow_miss_rate_max > 0:
        error.e_flow_miss = (1.0 - telemetry.flow_hit_rate) / setpoints.flow_miss_rate_max

    # Aggregate errors
    error.e_health = (
        weights.w_thermal * max(0, error.e_thermal) +
        weights.w_cognitive_load * max(0, error.e_cognitive) +
        weights.w_latency * max(0, error.e_latency) +
        weights.w_error_rate * (telemetry.error_rate / max(setpoints.error_rate_max, 0.001))
    )

    error.e_cathedral = (
        weights.w_consolidation * max(0, error.e_consolidation) +
        weights.w_diversity * max(0, error.e_diversity)
    )

    # Total weighted error
    error.e_total = (
        weights.w_health * error.e_health +
        weights.w_cathedral * error.e_cathedral
    )

    # Critical flag
    error.any_critical = error.e_thermal_critical or error.e_cognitive_critical

    return error


# =============================================================================
# Reward Computation
# =============================================================================

def compute_reward(
    error: ErrorVector,
    prev_error: Optional[ErrorVector],
    weights: 'TeleologyWeights',
) -> float:
    """
    Compute reward signal from error vector.

    Reward is based on:
    1. Absolute error magnitude (lower = better)
    2. Error improvement (decreasing = good)
    3. Mode-specific bonuses

    Returns reward in [-1, +1]
    """
    # Base reward from error magnitude
    # At e_total=0, reward=0.5; at e_total=1, reward=-0.5
    base_reward = 0.5 - error.e_total

    # Improvement bonus
    improvement = 0.0
    if prev_error is not None:
        delta = prev_error.e_total - error.e_total
        improvement = np.clip(delta * 2.0, -0.3, 0.3)

    # Critical penalty
    critical_penalty = -0.5 if error.any_critical else 0.0

    # Total reward
    reward = base_reward + improvement + critical_penalty

    return float(np.clip(reward, -1.0, 1.0))


# =============================================================================
# State History
# =============================================================================

class StateHistory:
    """
    Rolling history of homeostatic states for trend analysis.
    """

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.telemetry_history: deque = deque(maxlen=max_size)
        self.error_history: deque = deque(maxlen=max_size)
        self.reward_history: deque = deque(maxlen=max_size)
        self.mode_history: deque = deque(maxlen=max_size)

    def record(self, state: HomeostaticState) -> None:
        """Record a state snapshot."""
        self.telemetry_history.append((state.telemetry.timestamp, state.telemetry))
        self.error_history.append((state.error.timestamp, state.error))
        self.reward_history.append((state.updated_at, state.reward))
        self.mode_history.append((state.updated_at, state.mode))

    def get_telemetry_trend(self, field: str, window: int = 100) -> Tuple[float, float]:
        """
        Get trend for a telemetry field.

        Returns (mean, slope) over the window.
        """
        if len(self.telemetry_history) < 2:
            return 0.0, 0.0

        recent = list(self.telemetry_history)[-window:]
        values = [getattr(t, field, 0.0) for _, t in recent]

        mean = np.mean(values)
        # Simple linear regression for slope
        if len(values) > 1:
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]
        else:
            slope = 0.0

        return float(mean), float(slope)

    def get_reward_trend(self, window: int = 100) -> Tuple[float, float]:
        """Get reward trend (mean, slope)."""
        if len(self.reward_history) < 2:
            return 0.0, 0.0

        recent = list(self.reward_history)[-window:]
        values = [r for _, r in recent]

        mean = np.mean(values)
        if len(values) > 1:
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]
        else:
            slope = 0.0

        return float(mean), float(slope)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'OperationalMode',
    'Telemetry',
    'ErrorVector',
    'FounderState',
    'HomeostaticState',
    'compute_error_vector',
    'compute_reward',
    'StateHistory',
]
