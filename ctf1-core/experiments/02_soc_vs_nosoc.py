#!/usr/bin/env python3
"""
Experiment 02: SOC vs No-SOC Learning

Question: Does maintaining λ = 1 (SOC rule) improve learning stability?

Protocol:
1. Create two learners: SOCLearner (λ → 1) and NoSOCLearner (λ drifts)
2. Train both on a delayed-copy task
3. Track λ, E(λ), MSE over epochs
4. Compare: SOC should maintain E ≈ 0, NoSOC should drift

Expected result: SOC maintains criticality and stable performance
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core import CriticalCore, SOCLearner, NoSOCLearner


def generate_delayed_copy_task(T: int, delay: int) -> tuple:
    """Generate delayed copy task data."""
    inputs = np.random.randn(T, 1)
    targets = np.zeros((T, 1))
    targets[delay:, 0] = inputs[:-delay, 0]
    return inputs, targets


def main():
    print("=" * 60)
    print("Experiment 02: SOC vs No-SOC Learning")
    print("=" * 60)

    # Parameters
    n_dims = 100
    T = 1000
    delay = 10
    n_epochs = 20
    seed = 42

    print(f"\nParameters:")
    print(f"  Reservoir dim: {n_dims}")
    print(f"  Sequence length: {T}")
    print(f"  Delay: {delay} steps")
    print(f"  Epochs: {n_epochs}")

    # Generate task
    np.random.seed(seed)
    inputs, targets = generate_delayed_copy_task(T, delay)

    # Create learners
    core_soc = CriticalCore(n_dims=n_dims, lambda_init=1.0, seed=seed)
    core_nosoc = CriticalCore(n_dims=n_dims, lambda_init=1.0, seed=seed)

    learner_soc = SOCLearner(core_soc, n_input=1, n_output=1, lambda_target=1.0)
    learner_nosoc = NoSOCLearner(core_nosoc, n_input=1, n_output=1, lambda_target=1.0)

    # Training loop
    print("\nTraining...")
    print("-" * 70)
    print(f"{'Epoch':>6} | {'SOC MSE':>10} {'SOC λ':>8} {'SOC E':>8} | "
          f"{'NoSOC MSE':>10} {'NoSOC λ':>8} {'NoSOC E':>8}")
    print("-" * 70)

    for epoch in range(n_epochs):
        # Train SOC learner
        mse_soc, metrics_soc = learner_soc.train_epoch_soc(inputs, targets)

        # Train NoSOC learner
        mse_nosoc, metrics_nosoc = learner_nosoc.train_epoch_soc(inputs, targets)

        if epoch % 2 == 0:
            print(f"{epoch+1:>6} | "
                  f"{mse_soc:>10.4f} {metrics_soc['lambda']:>8.3f} {metrics_soc['E']:>+8.4f} | "
                  f"{mse_nosoc:>10.4f} {metrics_nosoc['lambda']:>8.3f} {metrics_nosoc['E']:>+8.4f}")

    print("-" * 70)

    # Final comparison
    final_soc_E = learner_soc.E_hist[-1]
    final_nosoc_E = learner_nosoc.E_hist[-1]
    final_soc_mse = learner_soc.loss_hist[-1]
    final_nosoc_mse = learner_nosoc.loss_hist[-1]

    print(f"\nFinal Results:")
    print(f"  SOC:   E = {final_soc_E:+.4f}, MSE = {final_soc_mse:.4f}")
    print(f"  NoSOC: E = {final_nosoc_E:+.4f}, MSE = {final_nosoc_mse:.4f}")

    # Verify predictions
    print("\nVerification:")

    # SOC should maintain E ≈ 0
    if abs(final_soc_E) < 0.01:
        print("  ✓ SOC maintains criticality (|E| < 0.01)")
    else:
        print(f"  ✗ SOC drifted: E = {final_soc_E:+.4f}")

    # NoSOC might drift
    soc_E_var = np.var(learner_soc.E_hist)
    nosoc_E_var = np.var(learner_nosoc.E_hist)

    print(f"  SOC E variance: {soc_E_var:.6f}")
    print(f"  NoSOC E variance: {nosoc_E_var:.6f}")

    if soc_E_var < nosoc_E_var:
        print("  ✓ SOC has more stable E than NoSOC")
    else:
        print("  ? SOC and NoSOC have similar E stability")

    # Save results
    output_file = os.path.join(os.path.dirname(__file__), '..', 'plots', 'soc_comparison.npz')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.savez(output_file,
             soc_E=np.array(learner_soc.E_hist),
             soc_lambda=np.array(learner_soc.lambda_hist),
             soc_mse=np.array(learner_soc.loss_hist),
             nosoc_E=np.array(learner_nosoc.E_hist),
             nosoc_lambda=np.array(learner_nosoc.lambda_hist),
             nosoc_mse=np.array(learner_nosoc.loss_hist))
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
