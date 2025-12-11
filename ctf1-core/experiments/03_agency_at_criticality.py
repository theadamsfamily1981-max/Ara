#!/usr/bin/env python3
"""
Experiment 03: Agency at Criticality

Question: Do agents at λ = 1 show better exploration-exploitation tradeoff?

Protocol:
1. Create agents at different λ (0.7, 1.0, 1.3)
2. Run k-armed bandit task
3. Compare: final reward, policy entropy, regret

Expected result: λ = 1.0 shows best balance of exploration and exploitation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core import CriticalCore, CriticalAgent, MultiArmedBandit, run_bandit_experiment


def main():
    print("=" * 60)
    print("Experiment 03: Agency at Criticality")
    print("=" * 60)

    # Parameters
    n_dims = 64
    n_arms = 4
    n_episodes = 2000
    lambdas = [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5]
    seed = 42

    print(f"\nParameters:")
    print(f"  Reservoir dim: {n_dims}")
    print(f"  Arms: {n_arms}")
    print(f"  Episodes: {n_episodes}")
    print(f"  λ values: {lambdas}")

    # Run experiments
    print("\nRunning experiments...")
    results = []

    for lam in lambdas:
        exp = run_bandit_experiment(
            lambda_value=lam,
            n_dims=n_dims,
            n_arms=n_arms,
            n_episodes=n_episodes,
            seed=seed,
        )
        results.append(exp)
        print(f"  λ={lam:.1f}: reward={exp['final_reward']:.3f}, "
              f"entropy={exp['final_entropy']:.3f}")

    # Summary table
    print("\n" + "-" * 60)
    print(f"{'λ':>6} {'Final Reward':>14} {'Final Entropy':>14} {'Regret':>10}")
    print("-" * 60)

    best_idx = 0
    best_reward = 0

    for i, exp in enumerate(results):
        lam = exp['lambda']
        reward = exp['final_reward']
        entropy = exp['final_entropy']
        regret = exp['optimal_prob'] - reward

        if reward > best_reward:
            best_reward = reward
            best_idx = i

        marker = " <--" if i == best_idx else ""
        print(f"{lam:>6.1f} {reward:>14.3f} {entropy:>14.3f} {regret:>10.3f}{marker}")

    print("-" * 60)

    best_lambda = results[best_idx]['lambda']
    print(f"\nBest performing λ = {best_lambda:.1f}")
    print(f"Optimal arm probability: {results[0]['optimal_prob']:.3f}")

    # Verify prediction
    print("\nVerification:")
    if 0.9 <= best_lambda <= 1.1:
        print("  ✓ CONFIRMED: Best agent is near criticality (λ ≈ 1)")
    else:
        print(f"  ? Best agent at λ = {best_lambda:.1f}, expected near 1.0")

    # Analyze exploration-exploitation
    ordered_agent = [r for r in results if r['lambda'] == 0.7][0]
    critical_agent = [r for r in results if r['lambda'] == 1.0][0]
    chaotic_agent = [r for r in results if r['lambda'] == 1.3][0]

    print(f"\nExploration-Exploitation Analysis:")
    print(f"  Ordered (λ=0.7):  entropy={ordered_agent['final_entropy']:.3f}")
    print(f"  Critical (λ=1.0): entropy={critical_agent['final_entropy']:.3f}")
    print(f"  Chaotic (λ=1.3):  entropy={chaotic_agent['final_entropy']:.3f}")

    # Critical should have moderate entropy
    if (ordered_agent['final_entropy'] < critical_agent['final_entropy'] <
        chaotic_agent['final_entropy']):
        print("  ✓ Critical agent shows intermediate exploration (balanced)")
    else:
        print("  ? Entropy pattern differs from expected")

    # Save results
    output_file = os.path.join(os.path.dirname(__file__), '..', 'plots', 'agency_comparison.npz')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    np.savez(output_file,
             lambdas=np.array([r['lambda'] for r in results]),
             final_rewards=np.array([r['final_reward'] for r in results]),
             final_entropy=np.array([r['final_entropy'] for r in results]),
             optimal_prob=results[0]['optimal_prob'])
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
