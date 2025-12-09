"""
RL Adaptation Research
======================

Learning to tune precision weights from user feedback.

The core Ara system has fixed ω (modality weights) and κ (stress coupling).
This module experiments with learning these from:
- User dwell time (implicit positive feedback)
- Voice valence (prosody-detected mood)
- HRV improvement (if measurable)

Produces: Updated weight configurations for ara/embodiment/fusion.py

NOT RUNTIME CODE - Training/tuning only.
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class FeedbackSignal:
    """A single feedback signal from user interaction."""
    timestamp: float
    signal_type: str  # 'dwell', 'valence', 'hrv'
    value: float
    context_hv: Optional[np.ndarray] = None


@dataclass
class WeightUpdate:
    """Proposed update to precision weights."""
    omega_deltas: dict  # modality -> delta
    kappa_delta: float
    confidence: float
    episode_count: int


class PrecisionWeightLearner:
    """
    Simple RL for precision weight tuning.

    Uses policy gradient to learn:
    - ω_i (baseline weight for modality i)
    - κ (global stress coupling)

    Reward signal: Composite of dwell, valence, HRV improvement.
    """

    def __init__(self,
                 modalities: List[str],
                 learning_rate: float = 0.01,
                 kappa_lr: float = 0.005):
        self.modalities = modalities
        self.learning_rate = learning_rate
        self.kappa_lr = kappa_lr

        # Current weights (initialized from defaults)
        self.omega = {m: 0.5 for m in modalities}
        self.kappa = 0.3

        # Episode buffer
        self.episodes: List[FeedbackSignal] = []
        self.updates: List[WeightUpdate] = []

    def record_feedback(self, signal: FeedbackSignal):
        """Record a feedback signal for batch update."""
        self.episodes.append(signal)

    def compute_reward(self, signals: List[FeedbackSignal]) -> float:
        """
        Compute reward from feedback signals.

        Weights:
        - Dwell time: +1 per 30 seconds
        - Valence: direct value (-1 to +1)
        - HRV: +2 if improved, -1 if worsened
        """
        reward = 0.0
        for s in signals:
            if s.signal_type == 'dwell':
                reward += s.value / 30.0  # +1 per 30s
            elif s.signal_type == 'valence':
                reward += s.value  # -1 to +1
            elif s.signal_type == 'hrv':
                reward += 2.0 if s.value > 0 else -1.0
        return reward

    def update_weights(self) -> Optional[WeightUpdate]:
        """
        Run policy gradient update from collected episodes.

        Returns proposed weight changes (not automatically applied).
        """
        if len(self.episodes) < 10:
            return None

        reward = self.compute_reward(self.episodes)

        # Simple policy gradient: nudge weights in direction of reward
        omega_deltas = {}
        for m in self.modalities:
            # Estimate gradient via finite differences (simplified)
            # In practice, would use proper policy gradient
            delta = self.learning_rate * reward * np.random.randn() * 0.1
            omega_deltas[m] = np.clip(delta, -0.05, 0.05)

        kappa_delta = self.kappa_lr * reward * np.random.randn() * 0.1
        kappa_delta = np.clip(kappa_delta, -0.02, 0.02)

        update = WeightUpdate(
            omega_deltas=omega_deltas,
            kappa_delta=kappa_delta,
            confidence=min(1.0, len(self.episodes) / 100),
            episode_count=len(self.episodes)
        )

        self.updates.append(update)
        self.episodes = []  # Clear buffer

        return update

    def apply_update(self, update: WeightUpdate):
        """Apply a weight update (call manually after review)."""
        for m, delta in update.omega_deltas.items():
            if m in self.omega:
                self.omega[m] = np.clip(self.omega[m] + delta, 0.1, 0.9)

        self.kappa = np.clip(self.kappa + update.kappa_delta, 0.1, 0.8)

    def export_config(self) -> dict:
        """Export current weights as config dict."""
        return {
            'precision_weights': {
                'omega': dict(self.omega),
                'kappa': self.kappa,
                'learned_from_episodes': sum(u.episode_count for u in self.updates)
            }
        }


__all__ = [
    'FeedbackSignal',
    'WeightUpdate',
    'PrecisionWeightLearner',
]
