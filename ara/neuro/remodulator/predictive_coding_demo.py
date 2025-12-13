#!/usr/bin/env python3
"""
Predictive Coding Demo - Precision Weighting Effects

Demonstrates how the Delusion Index (D = Π_prior / Π_sensory) affects
perception in a simple hierarchical predictive coding network.

This shows:
1. D ≈ 1.0: Balanced inference (healthy)
2. D >> 1: Prior-dominated (hallucination-like)
3. D << 1: Sensory-dominated (overwhelm-like)

Based on Whittington & Bogacz (2017) and Friston's Free Energy formulation.

Usage:
    python -m ara.neuro.remodulator.predictive_coding_demo
    python -m ara.neuro.remodulator.predictive_coding_demo --D 5.0
    python -m ara.neuro.remodulator.predictive_coding_demo --interactive
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

# Optional: Use torch if available for GPU acceleration
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# NumPy Implementation (No Dependencies)
# =============================================================================

@dataclass
class PredictiveCodingLayer:
    """
    A single layer in a predictive coding hierarchy.

    Each layer maintains:
    - mu: Current belief/estimate (mean of posterior)
    - prediction: Top-down prediction from higher layer
    - error: Prediction error (bottom-up signal)
    - precision: Confidence in this layer's signals
    """
    mu: np.ndarray          # Current estimate
    prediction: np.ndarray  # Top-down prediction
    error: np.ndarray       # Prediction error
    precision: float        # Precision (inverse variance)

    @property
    def weighted_error(self) -> np.ndarray:
        """Precision-weighted prediction error."""
        return self.precision * self.error


class PredictiveCodingNetwork:
    """
    Simple hierarchical predictive coding network.

    Implements the core equations:
        ε = y - g(μ)           # Prediction error
        μ̇ = ε̃ᵀ Π ε̃            # Belief update (gradient descent on F)

    Where:
        y = input (sensory or from lower layer)
        g(μ) = prediction from current beliefs
        Π = precision matrix
        ε̃ = precision-weighted error
    """

    def __init__(
        self,
        layer_sizes: List[int],
        learning_rate: float = 0.1,
        pi_prior: float = 1.0,
        pi_sensory: float = 1.0,
    ):
        """
        Initialize network.

        Args:
            layer_sizes: Sizes of each layer [sensory, ..., abstract]
            learning_rate: Step size for belief updates
            pi_prior: Precision on prior/predictions
            pi_sensory: Precision on sensory/errors
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.pi_prior = pi_prior
        self.pi_sensory = pi_sensory

        # Initialize layers
        self.layers: List[PredictiveCodingLayer] = []
        for i, size in enumerate(layer_sizes):
            # Higher layers have prior precision, lower have sensory
            if i < len(layer_sizes) // 2:
                precision = pi_sensory
            else:
                precision = pi_prior

            layer = PredictiveCodingLayer(
                mu=np.zeros(size),
                prediction=np.zeros(size),
                error=np.zeros(size),
                precision=precision,
            )
            self.layers.append(layer)

        # Initialize weights between layers (for predictions)
        self.weights = []
        for i in range(len(layer_sizes) - 1):
            W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1
            self.weights.append(W)

    @property
    def D(self) -> float:
        """Delusion Index: ratio of prior to sensory precision."""
        return self.pi_prior / max(self.pi_sensory, 0.001)

    def set_D(self, D: float):
        """
        Set the Delusion Index by adjusting precisions.

        Keeps total precision constant while changing ratio.
        """
        total = self.pi_prior + self.pi_sensory
        self.pi_prior = total * D / (1 + D)
        self.pi_sensory = total / (1 + D)

        # Update layer precisions
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) // 2:
                layer.precision = self.pi_sensory
            else:
                layer.precision = self.pi_prior

    def predict(self, layer_idx: int) -> np.ndarray:
        """
        Generate top-down prediction for a layer.

        Uses beliefs from layer above to predict this layer's state.
        """
        if layer_idx >= len(self.layers) - 1:
            return self.layers[layer_idx].mu  # Top layer predicts itself

        # Prediction = W @ mu_above
        mu_above = self.layers[layer_idx + 1].mu
        W = self.weights[layer_idx]
        return W @ mu_above

    def compute_error(self, layer_idx: int, observation: Optional[np.ndarray] = None):
        """
        Compute prediction error for a layer.

        For lowest layer: error = observation - prediction
        For higher layers: error = mu - prediction
        """
        layer = self.layers[layer_idx]
        prediction = self.predict(layer_idx)
        layer.prediction = prediction

        if layer_idx == 0 and observation is not None:
            # Sensory layer: compare prediction to observation
            layer.error = observation - prediction
        else:
            # Higher layers: compare belief to prediction
            layer.error = layer.mu - prediction

    def update_beliefs(self, n_iterations: int = 20):
        """
        Update beliefs to minimize free energy.

        Implements gradient descent: μ̇ = -∂F/∂μ
        """
        for _ in range(n_iterations):
            # Compute all errors first
            for i in range(len(self.layers)):
                self.compute_error(i)

            # Update beliefs based on weighted errors
            for i in range(1, len(self.layers)):  # Skip sensory layer
                layer = self.layers[i]

                # Gradient from own prediction error
                grad = layer.precision * layer.error

                # Gradient from lower layer's error (backprop through prediction)
                if i > 0:
                    lower = self.layers[i - 1]
                    W = self.weights[i - 1]
                    grad -= lower.precision * (W.T @ lower.error)

                # Update
                layer.mu += self.learning_rate * grad

    def infer(
        self,
        observation: np.ndarray,
        n_iterations: int = 50,
    ) -> np.ndarray:
        """
        Perform inference given an observation.

        Returns the highest-level belief (abstract representation).
        """
        # Set observation
        self.layers[0].mu = observation.copy()

        # Initialize higher layers with small random values
        for layer in self.layers[1:]:
            layer.mu = np.random.randn(*layer.mu.shape) * 0.01

        # Iterative inference
        for _ in range(n_iterations):
            # Bottom-up: compute errors
            for i in range(len(self.layers)):
                self.compute_error(i, observation if i == 0 else None)

            # Update beliefs
            self.update_beliefs(n_iterations=1)

        return self.layers[-1].mu

    def generate(
        self,
        prior: np.ndarray,
        n_iterations: int = 50,
    ) -> np.ndarray:
        """
        Generate observation from a prior belief.

        Sets top-level belief and propagates down.
        """
        # Set prior at top level
        self.layers[-1].mu = prior.copy()

        # Propagate predictions down
        for i in range(len(self.layers) - 2, -1, -1):
            self.layers[i].mu = self.predict(i)

        return self.layers[0].mu

    def free_energy(self) -> float:
        """
        Compute variational free energy.

        F = Σ_l (ε_l^T Π_l ε_l)
        """
        F = 0.0
        for layer in self.layers:
            F += layer.precision * np.sum(layer.error ** 2)
        return F


# =============================================================================
# Demonstrations
# =============================================================================

def demo_precision_effects():
    """
    Demonstrate how D affects perception.

    Shows the same ambiguous stimulus perceived differently
    depending on precision balance.
    """
    print("\n" + "=" * 70)
    print("DEMO: Precision Effects on Perception")
    print("=" * 70)

    # Create network
    net = PredictiveCodingNetwork(
        layer_sizes=[8, 16, 4],  # Sensory -> Hidden -> Abstract
        learning_rate=0.1,
    )

    # Create ambiguous stimulus (could be pattern A or B)
    # This is like a Necker cube - ambiguous without strong priors
    pattern_A = np.array([1, 1, 0, 0, 1, 1, 0, 0], dtype=float)
    pattern_B = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=float)
    ambiguous = 0.5 * (pattern_A + pattern_B) + np.random.randn(8) * 0.1

    print(f"\nAmbiguous stimulus: {ambiguous.round(2)}")
    print(f"Pattern A:          {pattern_A}")
    print(f"Pattern B:          {pattern_B}")

    # Test different D values
    D_values = [0.2, 1.0, 5.0]

    print("\n" + "-" * 70)
    print(f"{'D Value':<12} {'Interpretation':<20} {'Behavior'}")
    print("-" * 70)

    for D in D_values:
        net.set_D(D)

        # Set a prior belief toward pattern A
        prior_A = np.array([1, 0, 0, 0], dtype=float)  # Belief in pattern A
        net.layers[-1].mu = prior_A * 0.5  # Weak initial prior

        # Infer
        belief = net.infer(ambiguous, n_iterations=100)

        # Interpret result
        if D > 2.0:
            interpretation = "Pattern A (prior wins)"
            behavior = "Ignores sensory ambiguity"
        elif D < 0.5:
            interpretation = "Confused/overwhelmed"
            behavior = "Can't resolve ambiguity"
        else:
            interpretation = "Uncertain"
            behavior = "Balances evidence"

        print(f"D = {D:<8.1f} {interpretation:<20} {behavior}")
        print(f"             Belief: {belief.round(2)}")
        print(f"             Free Energy: {net.free_energy():.2f}")


def demo_hallucination():
    """
    Demonstrate hallucination-like perception (D >> 1).

    When prior precision is very high, the network "sees" its
    expectations even in noise.
    """
    print("\n" + "=" * 70)
    print("DEMO: Hallucination (D >> 1, Prior-Dominated)")
    print("=" * 70)

    net = PredictiveCodingNetwork(
        layer_sizes=[8, 16, 4],
        learning_rate=0.1,
    )

    # Pure noise (no real pattern)
    noise = np.random.randn(8) * 0.3

    # Strong prior for pattern A
    pattern_A_prior = np.array([1, 0.5, 0, 0], dtype=float)

    print("\nInput: Pure noise")
    print(f"Noise: {noise.round(2)}")

    # Test with different D values
    print("\n" + "-" * 70)

    for D in [1.0, 5.0, 20.0]:
        net.set_D(D)
        net.layers[-1].mu = pattern_A_prior

        # What does the network "perceive"?
        belief = net.infer(noise, n_iterations=100)

        # Generate what the network thinks it's seeing
        generated = net.generate(belief)

        print(f"\nD = {D:.1f}:")
        print(f"  Belief (abstract): {belief.round(2)}")
        print(f"  'Sees' (generated): {generated.round(2)}")

        if D >= 5.0:
            print("  → HALLUCINATION: Network sees pattern in noise!")


def demo_sensory_overwhelm():
    """
    Demonstrate sensory overwhelm (D << 1).

    When sensory precision is too high, the network can't
    filter noise or form stable abstractions.
    """
    print("\n" + "=" * 70)
    print("DEMO: Sensory Overwhelm (D << 1, Sensory-Dominated)")
    print("=" * 70)

    net = PredictiveCodingNetwork(
        layer_sizes=[8, 16, 4],
        learning_rate=0.1,
    )

    # Clear pattern with some noise
    clear_pattern = np.array([1, 1, 0, 0, 1, 1, 0, 0], dtype=float)
    noisy_pattern = clear_pattern + np.random.randn(8) * 0.2

    print(f"\nTrue pattern: {clear_pattern}")
    print(f"Noisy input:  {noisy_pattern.round(2)}")

    print("\n" + "-" * 70)

    for D in [1.0, 0.2, 0.05]:
        net.set_D(D)

        belief = net.infer(noisy_pattern, n_iterations=100)
        F = net.free_energy()

        print(f"\nD = {D:.2f}:")
        print(f"  Belief: {belief.round(2)}")
        print(f"  Free Energy: {F:.2f}")

        if D <= 0.2:
            print("  → OVERWHELM: Fails to form stable abstraction!")
            print("  → Raw noise dominates perception")


def demo_remodulator_correction():
    """
    Demonstrate the remodulator correcting aberrant D.

    Shows how adjusting precision can restore healthy perception.
    """
    print("\n" + "=" * 70)
    print("DEMO: Remodulator Correction")
    print("=" * 70)

    from .core import BrainRemodulator, DirectPrecisionModality

    # Create remodulator
    remodulator = BrainRemodulator(
        D_target=1.0,
        rho_target=0.88,
        modality=DirectPrecisionModality(),
    )

    # Simulate aberrant state (schizophrenia-like)
    print("\nSimulating schizophrenia-like state (D >> 1)...")
    remodulator.update(pi_prior=3.0, pi_sensory=0.5, rho=0.95)

    state = remodulator.get_state()
    print(f"  Initial D: {state.precision.D:.2f}")
    print(f"  Pattern: {state.pattern.value}")

    # Apply corrections
    print("\nApplying remodulator corrections...")
    for step in range(5):
        interventions = remodulator.get_pending_interventions()
        if interventions:
            state = remodulator.apply_interventions()
            print(f"  Step {step + 1}: D = {state.precision.D:.2f}")

        if abs(state.precision.D - 1.0) < 0.1:
            print("  → Converged to healthy state!")
            break


def interactive_demo():
    """Interactive exploration of precision effects."""
    print("\n" + "=" * 70)
    print("INTERACTIVE DEMO: Explore Precision Effects")
    print("=" * 70)

    net = PredictiveCodingNetwork(
        layer_sizes=[8, 16, 4],
        learning_rate=0.1,
    )

    print("\nThis demo lets you explore how D affects perception.")
    print("Commands:")
    print("  d <value>  - Set D (e.g., 'd 5.0')")
    print("  n          - New random stimulus")
    print("  i          - Infer (perceive stimulus)")
    print("  g          - Generate (imagine from prior)")
    print("  q          - Quit")

    stimulus = np.random.randn(8) * 0.5

    while True:
        try:
            cmd = input("\n> ").strip().lower()
        except EOFError:
            break

        if cmd.startswith('d '):
            try:
                D = float(cmd[2:])
                net.set_D(D)
                print(f"D set to {D:.2f}")
            except ValueError:
                print("Invalid D value")

        elif cmd == 'n':
            stimulus = np.random.randn(8) * 0.5
            print(f"New stimulus: {stimulus.round(2)}")

        elif cmd == 'i':
            belief = net.infer(stimulus, n_iterations=100)
            print(f"Stimulus: {stimulus.round(2)}")
            print(f"Belief:   {belief.round(2)}")
            print(f"D = {net.D:.2f}, F = {net.free_energy():.2f}")

        elif cmd == 'g':
            prior = np.array([1, 0, 0, 0], dtype=float)
            generated = net.generate(prior)
            print(f"Prior:     {prior}")
            print(f"Generated: {generated.round(2)}")

        elif cmd == 'q':
            break

        else:
            print("Unknown command. Use d/n/i/g/q")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Predictive Coding Demo - Precision Effects"
    )
    parser.add_argument(
        "--D", type=float, default=None,
        help="Set specific D value for demos"
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="Run interactive exploration"
    )
    parser.add_argument(
        "--demo", type=str, default="all",
        choices=["all", "precision", "hallucination", "overwhelm", "correction"],
        help="Which demo to run"
    )

    args = parser.parse_args()

    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║               PREDICTIVE CODING DEMO - PRECISION EFFECTS                       ║
║                                                                                ║
║  Demonstrates how the Delusion Index (D = Π_prior / Π_sensory)                 ║
║  affects perception in a hierarchical predictive coding network.               ║
║                                                                                ║
║  D ≈ 1.0: Balanced inference (healthy)                                         ║
║  D >> 1 : Prior-dominated (hallucination risk)                                 ║
║  D << 1 : Sensory-dominated (overwhelm risk)                                   ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)

    if args.interactive:
        interactive_demo()
        return

    if args.demo in ["all", "precision"]:
        demo_precision_effects()

    if args.demo in ["all", "hallucination"]:
        demo_hallucination()

    if args.demo in ["all", "overwhelm"]:
        demo_sensory_overwhelm()

    if args.demo in ["all", "correction"]:
        try:
            demo_remodulator_correction()
        except ImportError as e:
            print(f"\nSkipping remodulator demo: {e}")

    print("\n" + "=" * 70)
    print("For more, try: --interactive")
    print("=" * 70)


if __name__ == "__main__":
    main()
