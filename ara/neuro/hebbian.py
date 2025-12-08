"""
Hebbian Policy Learner
======================

On-chip replacement for the LLM policy loop.

The three-factor Hebbian rule:
    Δw = η · ρ · pre ⊗ post

Where:
    - pre: Pre-synaptic spikes (from state HPV stream)
    - post: Post-synaptic spikes (from detector/head)
    - ρ: Neuromodulator (policy/reward signal)
    - η: Learning rate

This collapses the entire STATE_HPV_QUERY → LLM → NEW_POLICY_HDC
round-trip into a single on-chip update rule.

The neuromodulator ρ encodes what the LLM used to decide:
    - ρ > 0: "strengthen this mapping" (correct detection)
    - ρ < 0: "weaken this mapping" (false positive)
    - ρ = 0: no learning (neutral observation)

In v1 (2025), ρ comes from a simplified BDH/policy module.
Eventually, ρ could come from a spiking value head trained
via reward signals.

Usage:
    learner = HebbianPolicyLearner(n_pre=1024, n_post=128)

    # Each timestep:
    learner.step(pre_spikes, post_spikes, rho)

    # Get learned weights:
    weights = learner.get_weights()
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Callable
import time


@dataclass
class HebbianConfig:
    """Configuration for Hebbian policy learner."""
    n_pre: int = 1024                    # Pre-synaptic (state) dimension
    n_post: int = 128                    # Post-synaptic (head) dimension
    eta: float = 1e-3                    # Learning rate
    w_init_std: float = 0.1              # Initial weight std
    w_min: float = -1.0                  # Weight clamp min
    w_max: float = 1.0                   # Weight clamp max
    decay: float = 0.0                   # Weight decay per step
    eligibility_tau: float = 0.0         # Eligibility trace decay (0 = no trace)


class HebbianPolicyLearner:
    """
    Three-factor Hebbian learning for on-chip policy adaptation.

    Replaces:
        STATE_HPV_QUERY → LLM → NEW_POLICY_HDC

    With:
        pre × post × ρ → Δw (on-chip, milliseconds)
    """

    def __init__(self, config: Optional[HebbianConfig] = None,
                 n_pre: int = 1024, n_post: int = 128, eta: float = 1e-3):
        if config:
            self.cfg = config
        else:
            self.cfg = HebbianConfig(n_pre=n_pre, n_post=n_post, eta=eta)

        # Initialize weights
        self.w = np.random.randn(self.cfg.n_post, self.cfg.n_pre) * self.cfg.w_init_std

        # Eligibility trace (for delayed reward)
        self._eligibility = np.zeros_like(self.w)

        # Statistics
        self._update_count = 0
        self._total_delta = 0.0
        self._history: List[float] = []  # Track weight norm over time

    def step(self, pre: np.ndarray, post: np.ndarray, rho: float) -> np.ndarray:
        """
        Execute one Hebbian update step.

        Args:
            pre: Pre-synaptic spikes/activations, shape (n_pre,)
            post: Post-synaptic spikes/activations, shape (n_post,)
            rho: Neuromodulator signal (scalar)

        Returns:
            The weight delta applied
        """
        # Validate shapes
        assert pre.shape == (self.cfg.n_pre,), f"pre shape {pre.shape} != ({self.cfg.n_pre},)"
        assert post.shape == (self.cfg.n_post,), f"post shape {post.shape} != ({self.cfg.n_post},)"

        # Compute Hebbian outer product
        outer = np.outer(post, pre)  # (n_post, n_pre)

        # Update eligibility trace if enabled
        if self.cfg.eligibility_tau > 0:
            self._eligibility *= (1 - 1/self.cfg.eligibility_tau)
            self._eligibility += outer
            effective_outer = self._eligibility
        else:
            effective_outer = outer

        # Three-factor update: Δw = η · ρ · (post ⊗ pre)
        if rho != 0:
            dw = self.cfg.eta * rho * effective_outer
            self.w += dw

            # Apply weight decay
            if self.cfg.decay > 0:
                self.w *= (1 - self.cfg.decay)

            # Clamp weights
            np.clip(self.w, self.cfg.w_min, self.cfg.w_max, out=self.w)

            self._update_count += 1
            self._total_delta += np.abs(dw).sum()

            return dw

        return np.zeros_like(self.w)

    def forward(self, pre: np.ndarray) -> np.ndarray:
        """
        Forward pass through the learned weights.

        Args:
            pre: Pre-synaptic input, shape (n_pre,)

        Returns:
            Post-synaptic activation, shape (n_post,)
        """
        return self.w @ pre

    def get_weights(self) -> np.ndarray:
        """Get current weight matrix."""
        return self.w.copy()

    def set_weights(self, w: np.ndarray):
        """Set weight matrix (e.g., from saved state)."""
        assert w.shape == self.w.shape
        self.w = w.copy()

    def reset_eligibility(self):
        """Reset eligibility trace."""
        self._eligibility.fill(0)

    def get_stats(self) -> dict:
        """Get learning statistics."""
        return {
            "update_count": self._update_count,
            "total_delta": self._total_delta,
            "weight_norm": float(np.linalg.norm(self.w)),
            "weight_mean": float(self.w.mean()),
            "weight_std": float(self.w.std()),
        }


# ============================================================================
# Lava-compatible Process (for actual neuromorphic deployment)
# ============================================================================

# This is the Lava-style interface. Actual Lava imports would be:
# from lava.magma.core.process.process import Process
# from lava.magma.core.process.ports.ports import InPort
# from lava.magma.core.process.variable import Var

class HebbianPolicyProcess:
    """
    Lava-style Process definition for HebbianPolicyLearner.

    This is the interface that would run on actual neuromorphic hardware.
    For simulation, use HebbianPolicyLearner directly.

    Ports:
        pre: InPort(shape=(n_pre,))  - State/HPV spikes
        post: InPort(shape=(n_post,)) - Detector/head spikes
        mod: InPort(shape=(1,))       - Neuromodulator ρ

    Variables:
        w: Var(shape=(n_post, n_pre)) - Synaptic weights
        eta: Var(shape=(1,))          - Learning rate
    """

    def __init__(self, n_pre: int, n_post: int, eta: float = 1e-3):
        # In real Lava:
        # super().__init__()
        # self.pre = InPort(shape=(n_pre,))
        # self.post = InPort(shape=(n_post,))
        # self.mod = InPort(shape=(1,))
        # self.w = Var(shape=(n_post, n_pre))
        # self.eta = Var(shape=(1,), init=eta)

        # Simulation version:
        self._learner = HebbianPolicyLearner(n_pre=n_pre, n_post=n_post, eta=eta)
        self._pre_buffer = None
        self._post_buffer = None
        self._mod_buffer = 0.0

    def recv_pre(self, spikes: np.ndarray):
        """Receive pre-synaptic spikes."""
        self._pre_buffer = spikes

    def recv_post(self, spikes: np.ndarray):
        """Receive post-synaptic spikes."""
        self._post_buffer = spikes

    def recv_mod(self, rho: float):
        """Receive neuromodulator signal."""
        self._mod_buffer = rho

    def run_spk(self):
        """Execute one spike-driven update (called each timestep)."""
        if self._pre_buffer is not None and self._post_buffer is not None:
            self._learner.step(self._pre_buffer, self._post_buffer, self._mod_buffer)
            self._pre_buffer = None
            self._post_buffer = None
            self._mod_buffer = 0.0

    @property
    def w(self) -> np.ndarray:
        return self._learner.w


# ============================================================================
# Neuromodulator Sources
# ============================================================================

class SimpleNeuromodulator:
    """
    Simple neuromodulator that generates ρ based on prediction error.

    This is the "on-chip BDH" that replaces LLM policy decisions:
    - Compares detector output to expected/desired output
    - Emits ρ > 0 for correct detections (strengthen)
    - Emits ρ < 0 for false alarms (weaken)
    """

    def __init__(self, baseline: float = 0.5, gain: float = 1.0):
        self.baseline = baseline  # Expected activation level
        self.gain = gain
        self._history: List[Tuple[float, float]] = []

    def compute(self, detector_output: float, was_anomaly: bool) -> float:
        """
        Compute neuromodulator signal.

        Args:
            detector_output: Detector's confidence (0-1)
            was_anomaly: Ground truth (did an anomaly actually occur?)

        Returns:
            ρ: Neuromodulator signal
        """
        if was_anomaly:
            # Should have detected: reward high output, punish low
            rho = self.gain * (detector_output - self.baseline)
        else:
            # Should NOT have detected: punish high output, reward low
            rho = -self.gain * (detector_output - self.baseline)

        self._history.append((detector_output, rho))
        return rho

    def get_history(self) -> List[Tuple[float, float]]:
        return self._history.copy()


class RewardModulatedNeuromodulator:
    """
    Neuromodulator driven by external reward signal.

    For integration with reinforcement learning or user feedback.
    """

    def __init__(self, decay: float = 0.9):
        self.decay = decay
        self._accumulated_reward = 0.0

    def add_reward(self, reward: float):
        """Add external reward signal."""
        self._accumulated_reward += reward

    def compute(self, detector_output: float) -> float:
        """
        Compute neuromodulator from accumulated reward.

        The neuromodulator is modulated by detector output,
        so active detections get credit proportional to reward.
        """
        rho = self._accumulated_reward * detector_output
        self._accumulated_reward *= self.decay
        return rho
