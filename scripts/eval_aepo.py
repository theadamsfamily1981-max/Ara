#!/usr/bin/env python
"""
AEPO Evaluation Script

Evaluate trained AEPO policy and verify hard gates.

Hard gates:
- Tool-call count −50% vs baseline
- Reward within −1% of baseline
- Stable entropy curve (no collapse) across 10 seeds

Usage:
    python scripts/eval_aepo.py --checkpoint artifacts/aepo/final.pt
    python scripts/eval_aepo.py --checkpoint artifacts/aepo/checkpoint_100.pt --seeds 10
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from tfan.agent import AEPO, AEPOConfig, ToolEnv, ToolAction
from tfan.agent.env_tools import ToolOracle


class AEPOEvaluator:
    """
    Evaluator for trained AEPO policies.

    Runs comprehensive evaluation and gate verification.
    """

    def __init__(self, policy: AEPO, env: ToolEnv):
        """
        Initialize evaluator.

        Args:
            policy: Trained AEPO policy
            env: Tool environment
        """
        self.policy = policy
        self.env = env

    def evaluate_policy(
        self,
        num_episodes: int = 100,
        deterministic: bool = True,
        seed: int = None
    ) -> Dict:
        """
        Evaluate policy performance.

        Args:
            num_episodes: Number of episodes
            deterministic: Use deterministic policy
            seed: Random seed

        Returns:
            metrics: Performance metrics
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        episode_rewards = []
        tool_call_counts = []
        tool_call_rates = {i: [] for i in range(self.env.num_tools)}

        for _ in range(num_episodes):
            obs = self.env.reset()
            done = False
            episode_reward = 0.0
            tool_calls = 0
            tool_calls_per_tool = {i: 0 for i in range(self.env.num_tools)}

            while not done:
                # Get action
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                with torch.no_grad():
                    action, _ = self.policy.get_action(obs_tensor, deterministic)
                action = action.item()

                # Step
                tool_idx = self.env.step_count % self.env.num_tools
                next_obs, reward, done, info = self.env.step(action, tool_idx)

                # Track
                episode_reward += reward
                if action == ToolAction.CALL:
                    tool_calls += 1
                    tool_calls_per_tool[tool_idx] += 1

                obs = next_obs

            episode_rewards.append(episode_reward)
            tool_call_counts.append(tool_calls)

            # Track per-tool call rates
            for i in range(self.env.num_tools):
                steps_for_tool = self.env.episode_length // self.env.num_tools
                tool_call_rates[i].append(tool_calls_per_tool[i] / steps_for_tool)

        metrics = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_tool_calls': np.mean(tool_call_counts),
            'std_tool_calls': np.std(tool_call_counts),
            'tool_call_rates': {
                i: {
                    'mean': np.mean(rates),
                    'std': np.std(rates)
                }
                for i, rates in tool_call_rates.items()
            }
        }

        return metrics

    def evaluate_baseline(self, num_episodes: int = 100, seed: int = None) -> Dict:
        """
        Evaluate oracle baseline.

        Args:
            num_episodes: Number of episodes
            seed: Random seed

        Returns:
            metrics: Baseline metrics
        """
        if seed is not None:
            np.random.seed(seed)

        oracle = ToolOracle(self.env)

        episode_rewards = []
        tool_call_counts = []

        for _ in range(num_episodes):
            self.env.reset()
            done = False
            episode_reward = 0.0
            tool_calls = 0

            while not done:
                tool_idx = self.env.step_count % self.env.num_tools
                action = oracle.get_action(tool_idx)

                _, reward, done, _ = self.env.step(action, tool_idx)

                episode_reward += reward
                if action == ToolAction.CALL:
                    tool_calls += 1

            episode_rewards.append(episode_reward)
            tool_call_counts.append(tool_calls)

        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_tool_calls': np.mean(tool_call_counts),
            'std_tool_calls': np.std(tool_call_counts)
        }

    def compute_entropy(self, num_episodes: int = 100) -> float:
        """
        Compute average policy entropy.

        Args:
            num_episodes: Number of episodes

        Returns:
            entropy: Average entropy
        """
        entropies = []

        for _ in range(num_episodes):
            obs = self.env.reset()
            done = False

            while not done:
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)

                with torch.no_grad():
                    logits = self.policy(obs_tensor)
                    probs = torch.softmax(logits, dim=-1)
                    log_probs = torch.log_softmax(logits, dim=-1)
                    entropy = -(probs * log_probs).sum(dim=-1).item()

                entropies.append(entropy)

                # Step
                tool_idx = self.env.step_count % self.env.num_tools
                action, _ = self.policy.get_action(obs_tensor)
                obs, _, done, _ = self.env.step(action.item(), tool_idx)

        return np.mean(entropies)

    def verify_gates(
        self,
        policy_metrics: Dict,
        baseline_metrics: Dict,
        entropy: float
    ) -> Dict:
        """
        Verify hard gates.

        Args:
            policy_metrics: Trained policy metrics
            baseline_metrics: Baseline metrics
            entropy: Policy entropy

        Returns:
            gates: Gate verification results
        """
        # Gate 1: Tool-call reduction ≥ 50%
        tool_reduction = (
            (baseline_metrics['mean_tool_calls'] - policy_metrics['mean_tool_calls'])
            / baseline_metrics['mean_tool_calls']
        )
        gate1_pass = tool_reduction >= 0.5

        # Gate 2: Reward delta ≤ 1%
        reward_delta = abs(
            (baseline_metrics['mean_reward'] - policy_metrics['mean_reward'])
            / baseline_metrics['mean_reward']
        ) if baseline_metrics['mean_reward'] != 0 else 0
        gate2_pass = reward_delta <= 0.01

        # Gate 3: Entropy > 0.3 (no collapse)
        gate3_pass = entropy > 0.3

        gates = {
            'tool_reduction': {
                'value': tool_reduction,
                'threshold': 0.5,
                'pass': gate1_pass
            },
            'reward_delta': {
                'value': reward_delta,
                'threshold': 0.01,
                'pass': gate2_pass
            },
            'entropy': {
                'value': entropy,
                'threshold': 0.3,
                'pass': gate3_pass
            },
            'all_pass': gate1_pass and gate2_pass and gate3_pass
        }

        return gates

    def multi_seed_evaluation(
        self,
        num_seeds: int = 10,
        episodes_per_seed: int = 50
    ) -> Dict:
        """
        Evaluate across multiple seeds for stability.

        Args:
            num_seeds: Number of random seeds
            episodes_per_seed: Episodes per seed

        Returns:
            results: Multi-seed results
        """
        all_rewards = []
        all_tool_calls = []
        all_entropies = []
        all_gates = []

        for seed in range(num_seeds):
            # Evaluate policy
            policy_metrics = self.evaluate_policy(
                num_episodes=episodes_per_seed,
                seed=seed
            )

            # Evaluate baseline
            baseline_metrics = self.evaluate_baseline(
                num_episodes=episodes_per_seed,
                seed=seed
            )

            # Compute entropy
            np.random.seed(seed)
            torch.manual_seed(seed)
            entropy = self.compute_entropy(num_episodes=20)

            # Verify gates
            gates = self.verify_gates(policy_metrics, baseline_metrics, entropy)

            # Collect
            all_rewards.append(policy_metrics['mean_reward'])
            all_tool_calls.append(policy_metrics['mean_tool_calls'])
            all_entropies.append(entropy)
            all_gates.append(gates['all_pass'])

        results = {
            'num_seeds': num_seeds,
            'rewards': {
                'mean': np.mean(all_rewards),
                'std': np.std(all_rewards),
                'values': all_rewards
            },
            'tool_calls': {
                'mean': np.mean(all_tool_calls),
                'std': np.std(all_tool_calls),
                'values': all_tool_calls
            },
            'entropy': {
                'mean': np.mean(all_entropies),
                'std': np.std(all_entropies),
                'values': all_entropies
            },
            'gate_pass_rate': np.mean(all_gates),
            'entropy_stable': np.std(all_entropies) < 0.1
        }

        return results


def load_checkpoint(path: str) -> AEPO:
    """
    Load trained AEPO checkpoint.

    Args:
        path: Path to checkpoint

    Returns:
        policy: Loaded policy
    """
    checkpoint = torch.load(path, map_location='cpu')

    config = checkpoint.get('config', AEPOConfig())
    policy = AEPO(config=config)
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval()

    return policy


def main():
    parser = argparse.ArgumentParser(description='Evaluate AEPO policy')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of evaluation episodes')
    parser.add_argument('--seeds', type=int, default=1,
                        help='Number of seeds for multi-seed evaluation')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results')

    args = parser.parse_args()

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    policy = load_checkpoint(args.checkpoint)
    print("Checkpoint loaded successfully\n")

    # Create environment
    env = ToolEnv(num_tools=3, obs_dim=256, episode_length=100)

    # Create evaluator
    evaluator = AEPOEvaluator(policy, env)

    # Single-seed evaluation
    print("=" * 70)
    print("SINGLE-SEED EVALUATION")
    print("=" * 70)

    print("Evaluating baseline (oracle)...")
    baseline_metrics = evaluator.evaluate_baseline(num_episodes=args.episodes)

    print("Evaluating trained policy...")
    policy_metrics = evaluator.evaluate_policy(num_episodes=args.episodes)

    print("Computing entropy...")
    entropy = evaluator.compute_entropy(num_episodes=100)

    print("\nRESULTS")
    print("-" * 70)
    print(f"Baseline:  Reward={baseline_metrics['mean_reward']:6.3f} ± {baseline_metrics['std_reward']:.3f}, "
          f"Tools={baseline_metrics['mean_tool_calls']:5.1f} ± {baseline_metrics['std_tool_calls']:.1f}")
    print(f"Policy:    Reward={policy_metrics['mean_reward']:6.3f} ± {policy_metrics['std_reward']:.3f}, "
          f"Tools={policy_metrics['mean_tool_calls']:5.1f} ± {policy_metrics['std_tool_calls']:.1f}")
    print(f"Entropy:   {entropy:.3f}")
    print()

    # Verify gates
    gates = evaluator.verify_gates(policy_metrics, baseline_metrics, entropy)

    print("HARD GATE VERIFICATION")
    print("-" * 70)
    print(f"Tool-call reduction: {gates['tool_reduction']['value']*100:5.1f}% "
          f"(≥{gates['tool_reduction']['threshold']*100:.0f}%) "
          f"{'✓ PASS' if gates['tool_reduction']['pass'] else '✗ FAIL'}")
    print(f"Reward delta:        {gates['reward_delta']['value']*100:5.1f}% "
          f"(≤{gates['reward_delta']['threshold']*100:.0f}%) "
          f"{'✓ PASS' if gates['reward_delta']['pass'] else '✗ FAIL'}")
    print(f"Entropy (no collapse): {gates['entropy']['value']:.3f} "
          f"(>{gates['entropy']['threshold']:.1f}) "
          f"{'✓ PASS' if gates['entropy']['pass'] else '✗ FAIL'}")
    print()
    print(f"Overall: {'ALL GATES PASSED ✓✓✓' if gates['all_pass'] else 'SOME GATES FAILED ✗✗✗'}")
    print()

    results = {
        'baseline': baseline_metrics,
        'policy': policy_metrics,
        'entropy': entropy,
        'gates': gates
    }

    # Multi-seed evaluation
    if args.seeds > 1:
        print("=" * 70)
        print(f"MULTI-SEED EVALUATION ({args.seeds} seeds)")
        print("=" * 70)

        multi_seed_results = evaluator.multi_seed_evaluation(
            num_seeds=args.seeds,
            episodes_per_seed=50
        )

        print(f"Rewards:     {multi_seed_results['rewards']['mean']:6.3f} ± {multi_seed_results['rewards']['std']:.3f}")
        print(f"Tool calls:  {multi_seed_results['tool_calls']['mean']:5.1f} ± {multi_seed_results['tool_calls']['std']:.1f}")
        print(f"Entropy:     {multi_seed_results['entropy']['mean']:.3f} ± {multi_seed_results['entropy']['std']:.3f}")
        print(f"Gate pass rate: {multi_seed_results['gate_pass_rate']*100:.0f}%")
        print(f"Entropy stable: {'✓ YES' if multi_seed_results['entropy_stable'] else '✗ NO'}")
        print()

        results['multi_seed'] = multi_seed_results

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to: {output_path}")


if __name__ == '__main__':
    main()
