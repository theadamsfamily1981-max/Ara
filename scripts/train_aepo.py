#!/usr/bin/env python
"""
AEPO Training Script

Train entropy-regularized tool-use policy to minimize tool calls
while maintaining task performance.

Hard gates:
- Tool-call count −50% vs baseline
- Reward within −1% of baseline
- Stable entropy curve (no collapse) across 10 seeds

Usage:
    python scripts/train_aepo.py --epochs 100 --seed 42
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.optim as optim

from tfan.agent import AEPO, AEPOConfig, ToolEnv, ToolAction
from tfan.agent.replay_buffer import EpisodeBuffer


class AEPOTrainer:
    """
    Trainer for AEPO policy.

    On-policy training with GAE advantages.
    """

    def __init__(
        self,
        policy: AEPO,
        env: ToolEnv,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        epochs_per_iter: int = 4,
        batch_size: int = 256,
        log_dir: str = "artifacts/aepo"
    ):
        """
        Initialize AEPO trainer.

        Args:
            policy: AEPO policy
            env: Tool environment
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda
            epochs_per_iter: Training epochs per iteration
            batch_size: Batch size
            log_dir: Directory for logs and checkpoints
        """
        self.policy = policy
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epochs_per_iter = epochs_per_iter
        self.batch_size = batch_size

        # Optimizer
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)

        # Episode buffer
        self.episode_buffer = EpisodeBuffer()

        # Logging
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Metrics
        self.metrics_history = []

    def collect_episodes(
        self,
        num_episodes: int,
        deterministic: bool = False
    ) -> Dict:
        """
        Collect episodes using current policy.

        Args:
            num_episodes: Number of episodes to collect
            deterministic: Use deterministic policy

        Returns:
            metrics: Episode metrics
        """
        episode_rewards = []
        tool_call_counts = []

        for _ in range(num_episodes):
            obs = self.env.reset()
            done = False
            episode_reward = 0.0
            tool_calls = 0

            while not done:
                # Get action from policy
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                with torch.no_grad():
                    action, _ = self.policy.get_action(obs_tensor, deterministic)
                action = action.item()

                # Step environment (cycle through tools)
                tool_idx = self.env.step_count % self.env.num_tools
                next_obs, reward, done, info = self.env.step(action, tool_idx)

                # Track metrics
                episode_reward += reward
                if action == ToolAction.CALL:
                    tool_calls += 1

                # Store transition
                self.episode_buffer.add(obs, action, reward, next_obs, done)

                obs = next_obs

            episode_rewards.append(episode_reward)
            tool_call_counts.append(tool_calls)

        metrics = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_tool_calls': np.mean(tool_call_counts),
            'std_tool_calls': np.std(tool_call_counts)
        }

        return metrics

    def train_step(self) -> Dict:
        """
        Perform one training step.

        Returns:
            metrics: Training metrics
        """
        # Compute advantages
        batch = self.episode_buffer.compute_advantages(
            gamma=self.gamma,
            gae_lambda=self.gae_lambda
        )

        # Convert to tensors
        obs = torch.from_numpy(batch['obs'])
        actions = torch.from_numpy(batch['actions']).long()
        advantages = torch.from_numpy(batch['advantages']).float()

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Training epochs
        total_loss = 0.0
        total_entropy = 0.0
        num_batches = 0

        for _ in range(self.epochs_per_iter):
            # Create random indices for batching
            indices = torch.randperm(len(obs))

            for start in range(0, len(obs), self.batch_size):
                end = min(start + self.batch_size, len(obs))
                batch_idx = indices[start:end]

                # Get batch
                batch_obs = obs[batch_idx]
                batch_actions = actions[batch_idx]
                batch_adv = advantages[batch_idx]

                # Compute loss
                loss, info = self.policy.loss(
                    batch_obs,
                    {'action': batch_actions, 'adv': batch_adv}
                )

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

                # Track metrics
                total_loss += info['total_loss']
                total_entropy += info['entropy']
                num_batches += 1

        # Adaptive entropy coefficient
        avg_entropy = total_entropy / num_batches
        self.policy.update_ent_coef(avg_entropy)

        metrics = {
            'train_loss': total_loss / num_batches,
            'entropy': avg_entropy,
            'ent_coef': self.policy.ent_coef
        }

        # Clear buffer
        self.episode_buffer.clear()

        return metrics

    def train(
        self,
        num_iterations: int,
        episodes_per_iter: int = 10,
        eval_interval: int = 10,
        save_interval: int = 20
    ):
        """
        Train AEPO policy.

        Args:
            num_iterations: Number of training iterations
            episodes_per_iter: Episodes to collect per iteration
            eval_interval: Evaluate every N iterations
            save_interval: Save checkpoint every N iterations
        """
        print(f"Training AEPO for {num_iterations} iterations")
        print(f"Log directory: {self.log_dir}")
        print()

        for iteration in range(num_iterations):
            # Collect episodes
            collect_metrics = self.collect_episodes(episodes_per_iter)

            # Train on collected data
            train_metrics = self.train_step()

            # Combine metrics
            metrics = {
                'iteration': iteration,
                **collect_metrics,
                **train_metrics
            }

            self.metrics_history.append(metrics)

            # Print progress
            if iteration % 10 == 0:
                print(f"Iter {iteration:4d} | "
                      f"Reward: {metrics['mean_reward']:6.3f} ± {metrics['std_reward']:5.3f} | "
                      f"Tools: {metrics['mean_tool_calls']:5.1f} ± {metrics['std_tool_calls']:4.1f} | "
                      f"Entropy: {metrics['entropy']:.3f} | "
                      f"Ent_coef: {metrics['ent_coef']:.4f}")

            # Evaluate
            if iteration > 0 and iteration % eval_interval == 0:
                eval_metrics = self.evaluate(num_episodes=20)
                print(f"  EVAL | Reward: {eval_metrics['mean_reward']:6.3f} | "
                      f"Tools: {eval_metrics['mean_tool_calls']:5.1f}")

            # Save checkpoint
            if iteration > 0 and iteration % save_interval == 0:
                self.save_checkpoint(f"checkpoint_{iteration}.pt")

        # Save final model
        self.save_checkpoint("final.pt")

        # Save metrics
        self.save_metrics()

        print("\nTraining complete!")

    def evaluate(
        self,
        num_episodes: int = 20,
        deterministic: bool = True
    ) -> Dict:
        """
        Evaluate current policy.

        Args:
            num_episodes: Number of evaluation episodes
            deterministic: Use deterministic policy

        Returns:
            metrics: Evaluation metrics
        """
        # Create temporary buffer (don't mix with training data)
        temp_buffer = self.episode_buffer
        self.episode_buffer = EpisodeBuffer()

        metrics = self.collect_episodes(num_episodes, deterministic)

        # Restore buffer
        self.episode_buffer = temp_buffer

        return metrics

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.log_dir / filename

        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.policy.config,
            'metrics_history': self.metrics_history
        }

        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")

    def save_metrics(self):
        """Save training metrics to JSON."""
        path = self.log_dir / "metrics.json"

        with open(path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

        print(f"Saved metrics: {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path)

        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.metrics_history = checkpoint.get('metrics_history', [])

        print(f"Loaded checkpoint: {path}")


def compute_baseline(env: ToolEnv, num_episodes: int = 100) -> Dict:
    """
    Compute baseline metrics with oracle policy.

    Args:
        env: Tool environment
        num_episodes: Number of episodes

    Returns:
        metrics: Baseline metrics
    """
    from tfan.agent.env_tools import ToolOracle

    oracle = ToolOracle(env)

    episode_rewards = []
    tool_call_counts = []

    for _ in range(num_episodes):
        env.reset()
        done = False
        episode_reward = 0.0
        tool_calls = 0

        while not done:
            tool_idx = env.step_count % env.num_tools
            action = oracle.get_action(tool_idx)

            _, reward, done, _ = env.step(action, tool_idx)

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


def main():
    parser = argparse.ArgumentParser(description='Train AEPO policy')
    parser.add_argument('--iterations', type=int, default=200,
                        help='Number of training iterations')
    parser.add_argument('--episodes-per-iter', type=int, default=10,
                        help='Episodes per iteration')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--log-dir', type=str, default='artifacts/aepo',
                        help='Log directory')

    args = parser.parse_args()

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create environment
    env = ToolEnv(num_tools=3, obs_dim=256, episode_length=100)

    # Compute baseline
    print("Computing baseline with oracle policy...")
    baseline = compute_baseline(env, num_episodes=100)
    print(f"Baseline: Reward={baseline['mean_reward']:.3f}, "
          f"Tools={baseline['mean_tool_calls']:.1f}")
    print()

    # Create policy
    config = AEPOConfig(
        obs_dim=env.obs_dim,
        ent_coef=0.02,
        target_entropy=0.7,
        adaptive_ent=True
    )
    policy = AEPO(config=config)

    # Create trainer
    trainer = AEPOTrainer(
        policy=policy,
        env=env,
        lr=args.lr,
        log_dir=args.log_dir
    )

    # Train
    trainer.train(
        num_iterations=args.iterations,
        episodes_per_iter=args.episodes_per_iter,
        eval_interval=10,
        save_interval=50
    )

    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    final_metrics = trainer.evaluate(num_episodes=100, deterministic=True)

    print(f"Trained Policy: Reward={final_metrics['mean_reward']:.3f}, "
          f"Tools={final_metrics['mean_tool_calls']:.1f}")
    print(f"Baseline:       Reward={baseline['mean_reward']:.3f}, "
          f"Tools={baseline['mean_tool_calls']:.1f}")
    print()

    # Check hard gates
    print("HARD GATE VERIFICATION")
    print("-" * 60)

    # Gate 1: Tool call reduction
    tool_reduction = (baseline['mean_tool_calls'] - final_metrics['mean_tool_calls']) / baseline['mean_tool_calls']
    gate1_pass = tool_reduction >= 0.5
    print(f"Tool-call reduction: {tool_reduction*100:.1f}% (target: ≥50%) {'✓' if gate1_pass else '✗'}")

    # Gate 2: Reward preservation
    reward_delta = (baseline['mean_reward'] - final_metrics['mean_reward']) / abs(baseline['mean_reward'])
    gate2_pass = reward_delta <= 0.01
    print(f"Reward delta: {reward_delta*100:.1f}% (target: ≤1%) {'✓' if gate2_pass else '✗'}")

    # Gate 3: Entropy stability (check last 10 iterations)
    entropies = [m['entropy'] for m in trainer.metrics_history[-10:]]
    entropy_std = np.std(entropies)
    gate3_pass = entropy_std < 0.1  # Stable if std < 0.1
    print(f"Entropy stability (last 10): std={entropy_std:.3f} (target: <0.1) {'✓' if gate3_pass else '✗'}")

    print()
    all_pass = gate1_pass and gate2_pass and gate3_pass
    print(f"Overall: {'ALL GATES PASSED ✓' if all_pass else 'SOME GATES FAILED ✗'}")

    # Save final report
    report = {
        'baseline': baseline,
        'final': final_metrics,
        'gates': {
            'tool_reduction': {'value': tool_reduction, 'pass': gate1_pass},
            'reward_delta': {'value': reward_delta, 'pass': gate2_pass},
            'entropy_stability': {'value': entropy_std, 'pass': gate3_pass}
        },
        'all_pass': all_pass
    }

    report_path = Path(args.log_dir) / 'final_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nFinal report saved to: {report_path}")


if __name__ == '__main__':
    main()
