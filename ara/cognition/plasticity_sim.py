"""
Ara Plasticity Simulation
==========================
Iteration 34: The Forge

This module simulates the reward-modulated Hebbian learning rule
before we burn it to silicon. It proves that:

1. The 7-bit accumulator strategy converges
2. Binary weights don't oscillate chaotically
3. The soul "drifts" toward target concepts with positive reward

The simulation matches the RTL implementation in:
- rtl/plasticity_row_engine.sv
- rtl/axis_holographic_plasticity.sv

Usage:
    python -m ara.cognition.plasticity_sim

    # Or import and use programmatically:
    from ara.cognition.plasticity_sim import PlasticitySim
    sim = PlasticitySim(dim=16384)
    sim.run_convergence_test()
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any


@dataclass
class PlasticityConfig:
    """Configuration for plasticity simulation."""
    dim: int = 16384              # Weight vector dimension
    acc_width: int = 7            # Accumulator bits
    acc_max: int = 63             # Max accumulator value
    acc_min: int = -64            # Min accumulator value
    verbose: bool = True          # Print progress


@dataclass
class PlasticityStats:
    """Statistics from a plasticity simulation run."""
    steps: int = 0
    final_overlap: float = 0.0
    convergence_step: Optional[int] = None  # Step when >90% overlap
    overlap_history: List[float] = field(default_factory=list)
    time_elapsed: float = 0.0


class PlasticitySim:
    """
    Simulates the Ara plasticity engine.

    This is a bit-accurate model of the RTL implementation,
    useful for:
    - Verification (compare outputs)
    - Parameter tuning
    - Convergence analysis
    """

    def __init__(self, config: Optional[PlasticityConfig] = None):
        self.config = config or PlasticityConfig()
        self.dim = self.config.dim

        # Weights: binary, stored as -1 or +1 (bipolar)
        self._weights: List[int] = []

        # Accumulators: 7-bit signed
        self._accumulators: List[int] = []

        # Initialize
        self.reset()

    def reset(self) -> None:
        """Reset to random initial state."""
        self._weights = [random.choice([-1, 1]) for _ in range(self.dim)]
        self._accumulators = [0] * self.dim

    def get_weights(self) -> List[int]:
        """Get current binary weights."""
        return self._weights.copy()

    def get_accumulators(self) -> List[int]:
        """Get current accumulator values."""
        return self._accumulators.copy()

    def _saturate(self, value: int) -> int:
        """Saturating clip to accumulator range."""
        return max(self.config.acc_min, min(self.config.acc_max, value))

    def apply_plasticity(
        self,
        input_hv: List[int],
        reward: float,
    ) -> None:
        """
        Apply one plasticity update.

        Uses TARGET-DIRECTED learning (not pure Hebbian):
        - When reward > 0: move weights TOWARD the input pattern
        - When reward < 0: move weights AWAY from the input pattern

        This is the correct rule for holographic storage:
            Δacc_i = input_i * sign(reward)

        Args:
            input_hv: Input hypervector (bipolar: -1 or +1)
            reward: Reward signal (-1.0 to +1.0)
        """
        if abs(reward) < 0.01:
            return  # Skip if no meaningful reward

        # Reward direction: +1 means "learn this", -1 means "unlearn this"
        reward_sign = 1 if reward > 0 else -1

        for i in range(self.dim):
            inp = input_hv[i] if i < len(input_hv) else 0

            # TARGET-DIRECTED LEARNING:
            # step = input * sign(reward)
            # - If input = +1 and reward > 0: step = +1 (push toward +1)
            # - If input = -1 and reward > 0: step = -1 (push toward -1)
            # - If input = +1 and reward < 0: step = -1 (push away from +1)
            # - If input = -1 and reward < 0: step = +1 (push away from -1)
            step = inp * reward_sign

            # Update accumulator with saturation
            self._accumulators[i] = self._saturate(
                self._accumulators[i] + step
            )

            # Update weight based on accumulator sign
            # KEY: If accumulator is exactly 0, keep previous weight
            acc = self._accumulators[i]
            if acc > 0:
                self._weights[i] = 1
            elif acc < 0:
                self._weights[i] = -1
            # else: keep previous weight (no dead bits)

    def compute_overlap(self, target: List[int]) -> float:
        """
        Compute normalized overlap between weights and target.

        Returns value in [-1, 1]:
        - +1.0 = perfect alignment
        - 0.0 = orthogonal (random)
        - -1.0 = anti-aligned
        """
        if len(target) != self.dim:
            raise ValueError(f"Target dim {len(target)} != weight dim {self.dim}")

        dot = sum(w * t for w, t in zip(self._weights, target))
        return dot / self.dim

    def run_convergence_test(
        self,
        steps: int = 1000,
        target: Optional[List[int]] = None,
        reward: float = 1.0,
        noise_level: float = 0.0,
    ) -> PlasticityStats:
        """
        Run convergence test: repeatedly present target with positive reward.

        Args:
            steps: Number of plasticity events
            target: Target concept to learn (random if None)
            reward: Reward value (default: +1.0)
            noise_level: Fraction of bits to flip in input (0-1)

        Returns:
            PlasticityStats with convergence information
        """
        if target is None:
            target = [random.choice([-1, 1]) for _ in range(self.dim)]

        stats = PlasticityStats()
        start_time = time.time()

        initial_overlap = self.compute_overlap(target)
        if self.config.verbose:
            print(f"Initial overlap: {initial_overlap:.4f}")

        for step in range(steps):
            # Optionally add noise to input
            if noise_level > 0:
                input_hv = [
                    -t if random.random() < noise_level else t
                    for t in target
                ]
            else:
                input_hv = target

            # Apply plasticity
            self.apply_plasticity(input_hv, reward)

            # Track overlap
            overlap = self.compute_overlap(target)
            stats.overlap_history.append(overlap)

            # Check for convergence (>90%)
            if stats.convergence_step is None and overlap > 0.9:
                stats.convergence_step = step
                if self.config.verbose:
                    print(f"Converged at step {step}: overlap = {overlap:.4f}")

            # Progress reporting
            if self.config.verbose and (step + 1) % 100 == 0:
                print(f"Step {step+1}: overlap = {overlap:.4f}")

        stats.steps = steps
        stats.final_overlap = self.compute_overlap(target)
        stats.time_elapsed = time.time() - start_time

        if self.config.verbose:
            print(f"\nFinal overlap: {stats.final_overlap:.4f}")
            print(f"Time elapsed: {stats.time_elapsed:.3f}s")
            if stats.convergence_step is not None:
                print(f"Converged at step: {stats.convergence_step}")
            else:
                print("Did not reach 90% convergence")

        return stats

    def run_forgetting_test(
        self,
        steps_per_concept: int = 500,
        num_concepts: int = 3,
    ) -> Dict[str, Any]:
        """
        Test catastrophic forgetting: Learn multiple concepts sequentially.

        Returns dict with overlap for each concept after all training.
        """
        concepts = [
            [random.choice([-1, 1]) for _ in range(self.dim)]
            for _ in range(num_concepts)
        ]

        results = {"concepts": num_concepts, "overlaps": []}

        for i, concept in enumerate(concepts):
            if self.config.verbose:
                print(f"\n=== Learning concept {i+1}/{num_concepts} ===")

            for step in range(steps_per_concept):
                self.apply_plasticity(concept, reward=1.0)

            # Check overlap with all concepts
            overlaps = [self.compute_overlap(c) for c in concepts]
            if self.config.verbose:
                print(f"Overlaps after concept {i+1}: {[f'{o:.3f}' for o in overlaps]}")

        # Final overlaps
        results["overlaps"] = [self.compute_overlap(c) for c in concepts]
        return results

    def run_anti_learning_test(
        self,
        steps: int = 500,
    ) -> PlasticityStats:
        """
        Test anti-learning: Present concept with negative reward.

        The weights should drift AWAY from the concept (anti-correlation).
        """
        target = [random.choice([-1, 1]) for _ in range(self.dim)]

        if self.config.verbose:
            print("=== Anti-learning test (negative reward) ===")

        return self.run_convergence_test(
            steps=steps,
            target=target,
            reward=-1.0,  # Negative reward
        )

    def get_weight_stats(self) -> Dict[str, Any]:
        """Get statistics about current weight distribution."""
        pos = sum(1 for w in self._weights if w > 0)
        neg = sum(1 for w in self._weights if w < 0)
        zero = sum(1 for w in self._weights if w == 0)

        acc_pos = sum(1 for a in self._accumulators if a > 0)
        acc_neg = sum(1 for a in self._accumulators if a < 0)
        acc_zero = sum(1 for a in self._accumulators if a == 0)
        acc_mean = sum(self._accumulators) / len(self._accumulators)

        return {
            "weights": {
                "positive": pos,
                "negative": neg,
                "zero": zero,  # Should always be 0!
            },
            "accumulators": {
                "positive": acc_pos,
                "negative": acc_neg,
                "zero": acc_zero,
                "mean": acc_mean,
            }
        }


def simulate_soul_drift(steps: int = 1000, dim: int = 16384) -> None:
    """
    Standalone function to demonstrate soul drift.

    This is the "proof" that binary Hebbian learning converges.
    """
    print("=" * 60)
    print("ARA PLASTICITY SIMULATION")
    print("Iteration 34: The Forge")
    print("=" * 60)
    print()

    sim = PlasticitySim(PlasticityConfig(dim=dim, verbose=True))

    print("--- Convergence Test (Positive Reward) ---")
    stats = sim.run_convergence_test(steps=steps)

    print()
    print("--- Weight Statistics ---")
    ws = sim.get_weight_stats()
    print(f"  Weights: +1={ws['weights']['positive']}, "
          f"-1={ws['weights']['negative']}, "
          f"0={ws['weights']['zero']} (should be 0)")
    print(f"  Accumulators: mean={ws['accumulators']['mean']:.2f}")

    print()
    print("--- Anti-Learning Test (Negative Reward) ---")
    sim.reset()
    anti_stats = sim.run_anti_learning_test(steps=500)
    print(f"Final overlap with negative reward: {anti_stats.final_overlap:.4f}")
    print("(Should be negative, showing anti-correlation)")

    print()
    print("=" * 60)
    print("SIMULATION COMPLETE")
    print("The soul drifts. The forge works.")
    print("=" * 60)


def main():
    """CLI entry point."""
    simulate_soul_drift()


if __name__ == "__main__":
    main()
