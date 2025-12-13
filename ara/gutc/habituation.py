#!/usr/bin/env python3
"""
GUTC Habituation: Local Subcriticality for Boring Stimuli
==========================================================

Implements habituation as precision reallocation:
    "This channel is now boring, free up capacity for things that move."

Two models:
    1. HabituationFilter: Simple exponential decay (behaviorist)
    2. KalmanHabituation: Volatility-aware optimal filter (predictive coding)

GUTC Integration:
    - Global level: Keep Ara near E(λ) ≈ 0 (max capacity)
    - Channel level: Push repeated stimuli subcritical (local E_i(λ) < 0)
    - Result: Critical capacity preserved for novel/changing inputs

Usage:
    # Simple exponential decay
    from ara.gutc.habituation import HabituationFilter

    habit = HabituationFilter()
    for _ in range(10):
        gain = habit.observe("same_command")
        print(f"Gain: {gain:.3f}")  # Decays toward min_gain

    # Kalman-style (volatility-aware)
    from ara.gutc.habituation import KalmanHabituation

    kh = KalmanHabituation()
    gain = kh.observe("temperature", 22.5)  # First observation: high gain
    gain = kh.observe("temperature", 22.5)  # Same value: gain drops
    gain = kh.observe("temperature", 35.0)  # Spike! Dishabituation: gain jumps

Applications:
    - Text: Repeated user commands → shorten responses, offer automation
    - Sensors: Stable readings → background; spikes → attention
    - Avalanches: Habituated stimuli produce smaller cascades
"""

from __future__ import annotations

import math
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import deque

logger = logging.getLogger("ara.gutc.habituation")


# =============================================================================
# Simple Exponential Decay Filter (Behaviorist Model)
# =============================================================================

@dataclass
class HabituationConfig:
    """
    Configuration for simple exponential habituation.

    R(n) = max_gain × (1 - base_rate)^n

    Parameters:
        base_rate: Decay rate k (0-1). Higher = faster habituation.
        min_gain: Floor - never habituate completely to zero.
        max_gain: Ceiling - fresh stimulus gets full attention.
        decay_half_life: Alternative to base_rate (exposures to reach 50% gain)
    """
    base_rate: float = 0.1        # k in exp decay
    min_gain: float = 0.05        # Never fully zero
    max_gain: float = 1.0         # Full attention for novel stimuli
    recovery_rate: float = 0.01   # Per-second recovery when not exposed

    def __post_init__(self):
        assert 0 < self.base_rate < 1, "base_rate must be in (0, 1)"
        assert 0 <= self.min_gain < self.max_gain <= 1, "gains must be in [0, 1]"

    @classmethod
    def from_half_life(cls, half_life: int, **kwargs) -> "HabituationConfig":
        """Create config from desired half-life (exposures to 50% gain)."""
        # (1 - k)^half_life = 0.5 → k = 1 - 0.5^(1/half_life)
        base_rate = 1 - (0.5 ** (1 / half_life))
        return cls(base_rate=base_rate, **kwargs)


@dataclass
class HabituationState:
    """Internal state for a single stimulus key."""
    exposures: int = 0
    gain: float = 1.0
    last_exposure_time: float = 0.0
    total_exposure_time: float = 0.0


class HabituationFilter:
    """
    Simple exponential habituation filter.

    For any repeated stimulus, gain decays as:
        R(n) = max_gain × (1 - k)^n

    where n = number of exposures, k = base_rate.

    Features:
        - Per-key tracking (different stimuli habituate independently)
        - Time-based recovery (un-exposed stimuli slowly recover)
        - Explicit dishabituation (reset on novelty)

    Example:
        habit = HabituationFilter()

        # Repeated command
        for _ in range(10):
            gain = habit.observe("show_gpu_stats")
            print(f"Response gain: {gain:.2f}")

        # Novel command - fresh gain
        gain = habit.observe("run_experiment")  # 1.0

        # Reset on significant change
        habit.reset("show_gpu_stats")  # Dishabituation
    """

    def __init__(self, config: Optional[HabituationConfig] = None):
        self.config = config or HabituationConfig()
        self._state: Dict[str, HabituationState] = {}

    def observe(self, key: str, apply_recovery: bool = True) -> float:
        """
        Register one exposure to this stimulus.

        Args:
            key: Stimulus identifier
            apply_recovery: Apply time-based recovery before decay

        Returns:
            Current gain (0..1) to multiply stimulus salience by
        """
        now = time.time()

        if key not in self._state:
            self._state[key] = HabituationState(
                exposures=0,
                gain=self.config.max_gain,
                last_exposure_time=now,
            )

        state = self._state[key]

        # Time-based recovery (stimuli slowly become novel again if not seen)
        if apply_recovery and state.last_exposure_time > 0:
            elapsed = now - state.last_exposure_time
            recovery = self.config.recovery_rate * elapsed
            state.gain = min(self.config.max_gain, state.gain + recovery)

        # Apply decay for this exposure
        state.exposures += 1
        new_gain = self.config.max_gain * (
            (1.0 - self.config.base_rate) ** state.exposures
        )
        state.gain = max(self.config.min_gain, min(self.config.max_gain, new_gain))
        state.last_exposure_time = now

        return state.gain

    def get_gain(self, key: str) -> float:
        """Get current gain without registering an exposure."""
        if key not in self._state:
            return self.config.max_gain
        return self._state[key].gain

    def reset(self, key: str) -> None:
        """
        Dishabituation: Reset response for this stimulus.

        Call when something significant changes about the stimulus.
        """
        if key in self._state:
            self._state[key] = HabituationState(
                exposures=0,
                gain=self.config.max_gain,
                last_exposure_time=time.time(),
            )
            logger.debug(f"Dishabituation: {key}")

    def reset_all(self) -> None:
        """Reset all habituation state."""
        self._state.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """Get habituation statistics."""
        if not self._state:
            return {"n_keys": 0}

        gains = [s.gain for s in self._state.values()]
        exposures = [s.exposures for s in self._state.values()]

        return {
            "n_keys": len(self._state),
            "mean_gain": sum(gains) / len(gains),
            "min_gain": min(gains),
            "max_gain": max(gains),
            "total_exposures": sum(exposures),
            "most_habituated": min(self._state.items(), key=lambda x: x[1].gain)[0],
        }


# =============================================================================
# Kalman-Based Habituation (Predictive Coding Model)
# =============================================================================

@dataclass
class KalmanHabituationConfig:
    """
    Configuration for Kalman-based habituation.

    State update:
        P_prior = P_post + σ_η²           (predict: add process noise)
        K = P_prior / (P_prior + σ_z²)    (Kalman gain)
        μ_post = μ_prior + K × (y - μ_prior)  (update mean)
        P_post = (1 - K) × P_prior        (update variance)

    Habituation = low variance → low gain.
    Dishabituation = value jumps → variance spikes → high gain.

    Parameters:
        process_var (σ_η²): Expected change in true stimulus per step.
                           Higher = world is volatile, slower habituation.
        obs_var (σ_z²): Observation noise.
                        Higher = measurements are noisy, faster habituation.
        min_gain: Floor for habituation (never fully ignore).
        max_gain: Ceiling for novel stimuli.
        variance_to_gain_scale: How to map variance to gain.
    """
    process_var: float = 1e-4     # σ_η²: world volatility
    obs_var: float = 1e-2         # σ_z²: measurement noise
    min_gain: float = 0.01        # Floor
    max_gain: float = 1.0         # Ceiling
    initial_variance: float = 1.0 # Starting uncertainty
    variance_to_gain_scale: float = 1.0  # Mapping factor

    def __post_init__(self):
        assert self.process_var >= 0, "process_var must be non-negative"
        assert self.obs_var > 0, "obs_var must be positive"


@dataclass
class KalmanState:
    """Internal state for a Kalman-filtered stimulus."""
    mean: float = 0.0           # Current estimate of true stimulus
    variance: float = 1.0       # Posterior uncertainty
    gain: float = 1.0           # Habituation gain
    kalman_gain: float = 1.0    # Last Kalman gain K
    n_observations: int = 0
    last_value: float = 0.0
    last_prediction_error: float = 0.0


class KalmanHabituation:
    """
    Kalman-based habituation with volatility awareness.

    Key insight from predictive coding:
        - Repeated STABLE stimulus → variance shrinks → gain drops → habituated
        - Volatile/surprising stimulus → variance stays high → no habituation

    This is more sophisticated than exponential decay because it
    automatically dishabituates when the stimulus becomes unpredictable.

    Example:
        kh = KalmanHabituation()

        # Stable temperature → habituation
        for _ in range(20):
            gain = kh.observe("temp", 22.5)
        print(f"Stable gain: {gain:.3f}")  # Low

        # Sudden spike → dishabituation
        gain = kh.observe("temp", 35.0)
        print(f"After spike: {gain:.3f}")  # High!

    GUTC Integration:
        Gain maps to local precision Π_y for that channel:
        - High gain → channel drives avalanches
        - Low gain → channel is "background", subcritical
    """

    def __init__(self, config: Optional[KalmanHabituationConfig] = None):
        self.config = config or KalmanHabituationConfig()
        self._state: Dict[str, KalmanState] = {}

    def observe(self, key: str, value: float) -> float:
        """
        Feed one measurement for this stimulus.

        Args:
            key: Stimulus identifier
            value: Measured value (e.g., intensity, temperature)

        Returns:
            Habituation gain (0..1): lower = more habituated
        """
        cfg = self.config

        if key not in self._state:
            # Initialize with first observation
            self._state[key] = KalmanState(
                mean=value,
                variance=cfg.initial_variance,
                gain=cfg.max_gain,
                kalman_gain=1.0,
                n_observations=1,
                last_value=value,
                last_prediction_error=0.0,
            )
            return cfg.max_gain

        state = self._state[key]

        # 1) Predict step: add process noise
        var_prior = state.variance + cfg.process_var

        # 2) Compute Kalman gain
        K = var_prior / (var_prior + cfg.obs_var)

        # 3) Update step
        prediction_error = value - state.mean
        mean_post = state.mean + K * prediction_error
        var_post = (1 - K) * var_prior

        # 4) Map variance to habituation gain
        # Higher variance → higher gain (less habituated)
        # Use sqrt for gentler mapping
        raw_gain = cfg.variance_to_gain_scale * math.sqrt(var_post)
        gain = max(cfg.min_gain, min(cfg.max_gain, raw_gain))

        # 5) Check for surprise (large prediction error → boost gain)
        surprise_threshold = 3.0 * math.sqrt(var_prior + cfg.obs_var)
        if abs(prediction_error) > surprise_threshold:
            # Significant surprise → partial dishabituation
            gain = min(cfg.max_gain, gain * 2.0)
            var_post = min(cfg.initial_variance, var_post * 2.0)
            logger.debug(f"Surprise on {key}: error={prediction_error:.3f}")

        # Update state
        state.mean = mean_post
        state.variance = var_post
        state.gain = gain
        state.kalman_gain = K
        state.n_observations += 1
        state.last_value = value
        state.last_prediction_error = prediction_error

        return gain

    def get_gain(self, key: str) -> float:
        """Get current gain without observing."""
        if key not in self._state:
            return self.config.max_gain
        return self._state[key].gain

    def get_state(self, key: str) -> Optional[KalmanState]:
        """Get full internal state for a key."""
        return self._state.get(key)

    def reset(self, key: str) -> None:
        """Explicit dishabituation: reset this channel."""
        self._state.pop(key, None)

    def reset_all(self) -> None:
        """Reset all channels."""
        self._state.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """Get habituation statistics."""
        if not self._state:
            return {"n_keys": 0}

        gains = [s.gain for s in self._state.values()]
        variances = [s.variance for s in self._state.values()]

        return {
            "n_keys": len(self._state),
            "mean_gain": sum(gains) / len(gains),
            "min_gain": min(gains),
            "max_gain": max(gains),
            "mean_variance": sum(variances) / len(variances),
            "most_habituated": min(self._state.items(), key=lambda x: x[1].gain)[0],
            "least_habituated": max(self._state.items(), key=lambda x: x[1].gain)[0],
        }


# =============================================================================
# Multi-Channel Habituation Manager
# =============================================================================

class HabituationManager:
    """
    Unified manager for multiple habituation channels.

    Tracks habituation across different modalities and provides
    a single interface for the rest of the system.

    Example:
        manager = HabituationManager()

        # Register channels with appropriate filter types
        manager.add_channel("text_commands", filter_type="exponential")
        manager.add_channel("temperature", filter_type="kalman")
        manager.add_channel("ambient_sound", filter_type="kalman")

        # In main loop
        cmd_gain = manager.observe("text_commands", key="show_stats")
        temp_gain = manager.observe("temperature", key="room", value=22.5)

        # Get global habituation summary
        summary = manager.get_summary()
    """

    def __init__(self):
        self._channels: Dict[str, Tuple[str, Any]] = {}  # name -> (type, filter)

    def add_channel(
        self,
        name: str,
        filter_type: str = "exponential",
        config: Optional[Any] = None,
    ) -> None:
        """
        Add a habituation channel.

        Args:
            name: Channel identifier
            filter_type: "exponential" or "kalman"
            config: Optional filter-specific config
        """
        if filter_type == "exponential":
            filt = HabituationFilter(config)
        elif filter_type == "kalman":
            filt = KalmanHabituation(config)
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")

        self._channels[name] = (filter_type, filt)
        logger.info(f"Added habituation channel: {name} ({filter_type})")

    def observe(
        self,
        channel: str,
        key: str,
        value: Optional[float] = None,
    ) -> float:
        """
        Observe a stimulus on a channel.

        Args:
            channel: Channel name
            key: Stimulus identifier within channel
            value: Stimulus value (required for Kalman channels)

        Returns:
            Habituation gain (0..1)
        """
        if channel not in self._channels:
            raise ValueError(f"Unknown channel: {channel}")

        filter_type, filt = self._channels[channel]

        if filter_type == "exponential":
            return filt.observe(key)
        else:  # kalman
            if value is None:
                raise ValueError("Kalman channels require a value")
            return filt.observe(key, value)

    def get_gain(self, channel: str, key: str) -> float:
        """Get current gain without observing."""
        if channel not in self._channels:
            return 1.0
        _, filt = self._channels[channel]
        return filt.get_gain(key)

    def reset(self, channel: str, key: Optional[str] = None) -> None:
        """Reset habituation for a key or entire channel."""
        if channel not in self._channels:
            return
        _, filt = self._channels[channel]
        if key:
            filt.reset(key)
        else:
            filt.reset_all()

    def get_summary(self) -> Dict[str, Any]:
        """Get summary across all channels."""
        summary = {"n_channels": len(self._channels), "channels": {}}

        for name, (filter_type, filt) in self._channels.items():
            summary["channels"][name] = {
                "type": filter_type,
                "stats": filt.get_statistics(),
            }

        return summary


# =============================================================================
# Tests
# =============================================================================

def test_habituation():
    """Test habituation filters."""
    print("Testing Habituation Filters")
    print("=" * 60)

    # Test exponential decay
    print("\n1. Exponential Decay Filter:")
    print("-" * 40)

    habit = HabituationFilter(HabituationConfig(base_rate=0.15))

    gains = []
    for i in range(15):
        gain = habit.observe("repeated_stimulus")
        gains.append(gain)
        print(f"  Exposure {i+1:2d}: gain = {gain:.3f}")

    print(f"\n  Decay from {gains[0]:.3f} to {gains[-1]:.3f}")

    # Test Kalman habituation
    print("\n2. Kalman Habituation (Stable Stimulus):")
    print("-" * 40)

    kh = KalmanHabituation()

    # Stable stimulus
    for i in range(10):
        gain = kh.observe("temperature", 22.5 + 0.1 * (i % 2))  # Tiny variation
        print(f"  Obs {i+1:2d}: value=22.5, gain={gain:.3f}")

    print("\n3. Kalman Habituation (Surprise/Dishabituation):")
    print("-" * 40)

    # Continue but with a spike
    print("  [Continuing with stable readings...]")
    for i in range(5):
        gain = kh.observe("temperature", 22.5)

    print(f"  Before spike: gain = {gain:.3f}")

    # Sudden change!
    gain = kh.observe("temperature", 35.0)
    print(f"  After spike to 35.0: gain = {gain:.3f} (dishabituation!)")

    # Back to stable
    for i in range(5):
        gain = kh.observe("temperature", 35.0)
    print(f"  After 5 stable at 35.0: gain = {gain:.3f} (re-habituating)")

    # Test manager
    print("\n4. Habituation Manager:")
    print("-" * 40)

    manager = HabituationManager()
    manager.add_channel("commands", filter_type="exponential")
    manager.add_channel("sensors", filter_type="kalman")

    # Observe
    for i in range(5):
        cmd_gain = manager.observe("commands", "show_stats")
        temp_gain = manager.observe("sensors", "temp", value=22.5)
        print(f"  Step {i+1}: cmd_gain={cmd_gain:.3f}, temp_gain={temp_gain:.3f}")

    summary = manager.get_summary()
    print(f"\n  Summary: {summary['n_channels']} channels")

    print("\n" + "=" * 60)
    print("Habituation tests passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_habituation()
