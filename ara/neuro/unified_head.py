"""
Unified Head - Reflex Head with Shared Learnable Weights
========================================================

A detector/classifier head whose weights are shared with
the Hebbian policy learner, enabling on-chip adaptation.

This replaces the old pattern:
    1. Detect anomaly with fixed weights
    2. Send STATE_HPV_QUERY to LLM
    3. Receive NEW_POLICY_HDC
    4. Manually update weights

With the new pattern:
    1. Detect with adaptive weights
    2. Neuromodulator ρ computed locally
    3. Hebbian update: Δw = η · ρ · pre ⊗ post
    4. Weights adapt in-place, on-chip

Usage:
    head = UnifiedHead(n_input=1024, n_output=4)

    # Forward pass (detection)
    logits = head.forward(spikes)

    # Hebbian update (learning)
    head.learn(pre_spikes, post_spikes, rho)

    # Or use the integrated step:
    decision, learned = head.step(spikes, rho)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
from enum import Enum, auto

from ara.neuro.hebbian import HebbianPolicyLearner, HebbianConfig
from ara.neuro.spike_encoder import SpikeEncoder, SpikeEncoderConfig, SpikeDecoder


class HeadOutput(Enum):
    """Output types for the unified head."""
    NORMAL = auto()
    WARNING = auto()
    ANOMALY = auto()
    CRITICAL = auto()


@dataclass
class UnifiedHeadConfig:
    """Configuration for unified head."""
    n_input: int = 1024                  # Input dimension (HPV/spike)
    n_hidden: int = 256                  # Hidden layer size
    n_output: int = 4                    # Output classes

    # Hebbian learning
    eta: float = 1e-3                    # Learning rate
    use_eligibility: bool = False        # Use eligibility traces
    eligibility_tau: float = 10.0        # Trace decay

    # Activation
    activation: str = "relu"             # "relu", "sigmoid", "tanh"
    output_softmax: bool = True          # Softmax output?

    # Thresholds for decision
    threshold_warning: float = 0.3
    threshold_anomaly: float = 0.5
    threshold_critical: float = 0.8


class UnifiedHead:
    """
    Adaptive detector head with integrated Hebbian learning.

    Architecture:
        input (n_input) → hidden (n_hidden) → output (n_output)

    The hidden→output weights are shared with a HebbianPolicyLearner,
    enabling on-chip adaptation via three-factor rule.
    """

    def __init__(self, config: Optional[UnifiedHeadConfig] = None, **kwargs):
        if config:
            self.cfg = config
        else:
            self.cfg = UnifiedHeadConfig(**kwargs)

        # Input → Hidden weights (fixed or slowly adapting)
        self.w1 = np.random.randn(self.cfg.n_hidden, self.cfg.n_input) * 0.1
        self.b1 = np.zeros(self.cfg.n_hidden)

        # Hidden → Output: uses Hebbian learner
        hebb_cfg = HebbianConfig(
            n_pre=self.cfg.n_hidden,
            n_post=self.cfg.n_output,
            eta=self.cfg.eta,
            eligibility_tau=self.cfg.eligibility_tau if self.cfg.use_eligibility else 0,
        )
        self._learner = HebbianPolicyLearner(config=hebb_cfg)
        self.b2 = np.zeros(self.cfg.n_output)

        # For tracking activations
        self._last_hidden: Optional[np.ndarray] = None
        self._last_output: Optional[np.ndarray] = None

        # Statistics
        self._forward_count = 0
        self._learn_count = 0
        self._decisions: List[HeadOutput] = []

    @property
    def w2(self) -> np.ndarray:
        """Hidden→output weights (shared with learner)."""
        return self._learner.w

    def _activate(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.cfg.activation == "relu":
            return np.maximum(0, x)
        elif self.cfg.activation == "sigmoid":
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif self.cfg.activation == "tanh":
            return np.tanh(x)
        else:
            return x

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax for output probabilities."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the head.

        Args:
            x: Input (HPV or spike train), shape (n_input,)

        Returns:
            output: Class probabilities/logits, shape (n_output,)
        """
        # Input → Hidden
        hidden = self.w1 @ x + self.b1
        hidden = self._activate(hidden)
        self._last_hidden = hidden

        # Hidden → Output (through learner's weights)
        output = self._learner.forward(hidden) + self.b2

        if self.cfg.output_softmax:
            output = self._softmax(output)

        self._last_output = output
        self._forward_count += 1

        return output

    def decide(self, output: Optional[np.ndarray] = None) -> HeadOutput:
        """
        Make a decision based on output.

        Args:
            output: Class probabilities (uses last output if None)

        Returns:
            decision: HeadOutput enum value
        """
        if output is None:
            output = self._last_output

        if output is None:
            return HeadOutput.NORMAL

        # Use max probability as confidence
        confidence = float(output.max())
        class_idx = int(output.argmax())

        # Map to decision based on thresholds
        if confidence >= self.cfg.threshold_critical:
            decision = HeadOutput.CRITICAL
        elif confidence >= self.cfg.threshold_anomaly:
            decision = HeadOutput.ANOMALY
        elif confidence >= self.cfg.threshold_warning:
            decision = HeadOutput.WARNING
        else:
            decision = HeadOutput.NORMAL

        self._decisions.append(decision)
        return decision

    def learn(self, pre: np.ndarray, post: np.ndarray, rho: float) -> np.ndarray:
        """
        Apply Hebbian learning update.

        Args:
            pre: Pre-synaptic activations (hidden layer)
            post: Post-synaptic activations (output layer)
            rho: Neuromodulator signal

        Returns:
            dw: Weight delta applied
        """
        self._learn_count += 1
        return self._learner.step(pre, post, rho)

    def step(self, x: np.ndarray, rho: float = 0.0,
             ground_truth: Optional[HeadOutput] = None) -> Tuple[HeadOutput, np.ndarray]:
        """
        Combined forward + learn step.

        Args:
            x: Input (HPV or spikes)
            rho: Neuromodulator (0 = no learning)
            ground_truth: Optional ground truth for auto-computing rho

        Returns:
            decision: The head's decision
            dw: Weight delta (zero if rho=0)
        """
        # Forward pass
        output = self.forward(x)
        decision = self.decide(output)

        # Auto-compute rho if ground truth provided
        if ground_truth is not None and rho == 0:
            # Simple error signal
            correct = (decision == ground_truth)
            if decision in (HeadOutput.ANOMALY, HeadOutput.CRITICAL):
                # Detected something
                rho = 1.0 if correct else -0.5
            else:
                # Didn't detect
                rho = -0.5 if ground_truth in (HeadOutput.ANOMALY, HeadOutput.CRITICAL) else 0.1

        # Learn if rho != 0
        dw = np.zeros_like(self.w2)
        if rho != 0 and self._last_hidden is not None and self._last_output is not None:
            dw = self.learn(self._last_hidden, self._last_output, rho)

        return decision, dw

    def get_stats(self) -> Dict[str, Any]:
        """Get head statistics."""
        decision_counts = {h.name: 0 for h in HeadOutput}
        for d in self._decisions:
            decision_counts[d.name] += 1

        return {
            "forward_count": self._forward_count,
            "learn_count": self._learn_count,
            "decisions": decision_counts,
            "learner_stats": self._learner.get_stats(),
        }

    def reset_stats(self):
        """Reset statistics."""
        self._decisions.clear()
        self._forward_count = 0
        self._learn_count = 0


# ============================================================================
# Integrated Reflex Unit
# ============================================================================

class ReflexUnit:
    """
    Complete reflex unit: encoding + head + learning.

    This is the full "protocol collapse" in one class:
    - Takes HPV state as input
    - Encodes to spikes
    - Runs through adaptive head
    - Computes neuromodulator
    - Updates weights

    No GPU, no IPC, no LLM needed for steady-state operation.
    """

    def __init__(self, hpv_dim: int = 1024, n_classes: int = 4,
                 eta: float = 1e-3):
        # Spike encoder
        self.encoder = SpikeEncoder(dim=hpv_dim, method="rate")

        # Adaptive head
        head_cfg = UnifiedHeadConfig(
            n_input=hpv_dim,
            n_hidden=256,
            n_output=n_classes,
            eta=eta,
        )
        self.head = UnifiedHead(config=head_cfg)

        # Spike decoder for output
        self.decoder = SpikeDecoder()

        # Track state
        self._last_spikes: Optional[np.ndarray] = None
        self._step_count = 0

    def process(self, hpv: np.ndarray, ground_truth: Optional[HeadOutput] = None,
                external_rho: float = 0.0) -> Tuple[HeadOutput, Dict[str, Any]]:
        """
        Process one HPV state through the reflex unit.

        Args:
            hpv: State hypervector
            ground_truth: Optional ground truth for learning
            external_rho: External neuromodulator override

        Returns:
            decision: The reflex decision
            info: Debug information
        """
        # Encode to spikes
        spikes = self.encoder.encode(hpv, self._step_count)
        self._last_spikes = spikes

        # Run through head with learning
        decision, dw = self.head.step(
            spikes,
            rho=external_rho,
            ground_truth=ground_truth
        )

        self._step_count += 1

        info = {
            "step": self._step_count,
            "spike_rate": float(spikes.mean()),
            "output": self.head._last_output.tolist() if self.head._last_output is not None else None,
            "dw_norm": float(np.linalg.norm(dw)),
        }

        return decision, info

    def get_stats(self) -> Dict[str, Any]:
        """Get unit statistics."""
        return {
            "step_count": self._step_count,
            "head_stats": self.head.get_stats(),
        }
