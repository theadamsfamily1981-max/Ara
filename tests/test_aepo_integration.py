#!/usr/bin/env python
"""
Integration tests for AEPO implementation.

Tests the complete training and evaluation pipeline.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from tfan.agent import AEPO, AEPOConfig, ToolEnv, ToolAction
from tfan.agent.replay_buffer import EpisodeBuffer


class TestAEPOPolicy:
    """Test AEPO policy network."""

    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        policy = AEPO(obs_dim=256)

        obs = torch.randn(10, 256)
        logits = policy(obs)

        assert logits.shape == (10, 2), "Should output 2 logits per sample"

    def test_get_action(self):
        """Test action sampling."""
        policy = AEPO(obs_dim=256)
        obs = torch.randn(5, 256)

        # Stochastic
        actions, log_probs = policy.get_action(obs)
        assert actions.shape == (5,), "Should output 1 action per sample"
        assert log_probs.shape == (5,), "Should output 1 log_prob per sample"
        assert torch.all((actions >= 0) & (actions <= 1)), "Actions should be 0 or 1"

        # Deterministic
        actions_det, _ = policy.get_action(obs, deterministic=True)
        assert actions_det.shape == (5,)
        assert torch.all((actions_det >= 0) & (actions_det <= 1))

    def test_loss_computation(self):
        """Test loss computation."""
        policy = AEPO(obs_dim=256)

        obs = torch.randn(32, 256)
        advantages = {
            'action': torch.randint(0, 2, (32,)),
            'adv': torch.randn(32)
        }

        loss, info = policy.loss(obs, advantages)

        assert isinstance(loss, torch.Tensor), "Should return loss tensor"
        assert loss.ndim == 0, "Loss should be scalar"
        assert 'entropy' in info, "Should return entropy"
        assert 'policy_loss' in info, "Should return policy loss"

    def test_entropy_adaptation(self):
        """Test adaptive entropy coefficient."""
        config = AEPOConfig(adaptive_ent=True, target_entropy=0.7)
        policy = AEPO(config=config)

        initial_coef = policy.ent_coef

        # Low entropy -> increase coef
        policy.update_ent_coef(current_entropy=0.3)
        assert policy.ent_coef > initial_coef, "Should increase for low entropy"

        # High entropy -> decrease coef
        policy.ent_coef = initial_coef
        policy.update_ent_coef(current_entropy=1.0)
        assert policy.ent_coef < initial_coef, "Should decrease for high entropy"


class TestToolEnv:
    """Test tool environment."""

    def test_reset(self):
        """Test environment reset."""
        env = ToolEnv(num_tools=3, obs_dim=256)

        obs = env.reset()

        assert obs.shape == (256,), "Should return observation"
        assert env.step_count == 0, "Step count should be 0"
        assert len(env.tool_relevant) == 3, "Should track tool relevance"

    def test_step(self):
        """Test environment step."""
        env = ToolEnv(num_tools=3, obs_dim=256, episode_length=10)

        env.reset()

        # Call tool
        obs, reward, done, info = env.step(ToolAction.CALL, tool_idx=0)

        assert obs.shape == (256,), "Should return next observation"
        assert isinstance(reward, (int, float)), "Should return reward"
        assert isinstance(done, bool), "Should return done flag"
        assert 'tool_called' in info, "Should return info"

        # Skip tool
        obs, reward, done, info = env.step(ToolAction.SKIP, tool_idx=1)

        assert obs.shape == (256,)
        assert not info['tool_called'], "Should not call tool"

    def test_episode_length(self):
        """Test episode terminates at correct length."""
        env = ToolEnv(episode_length=10)

        env.reset()
        done = False
        steps = 0

        while not done:
            _, _, done, _ = env.step(ToolAction.SKIP, tool_idx=0)
            steps += 1

        assert steps == 10, "Should terminate after episode_length steps"


class TestEpisodeBuffer:
    """Test episode buffer."""

    def test_add_transitions(self):
        """Test adding transitions."""
        buffer = EpisodeBuffer()

        obs = np.random.randn(256)
        next_obs = np.random.randn(256)

        buffer.add(obs, action=0, reward=1.0, next_obs=next_obs, done=False)
        buffer.add(next_obs, action=1, reward=0.5, next_obs=obs, done=True)

        assert len(buffer) == 1, "Should have 1 complete episode"
        assert len(buffer.current_episode) == 0, "Current episode should be reset"

    def test_compute_advantages(self):
        """Test GAE computation."""
        buffer = EpisodeBuffer()

        # Add complete episode
        for i in range(10):
            obs = np.random.randn(256)
            buffer.add(obs, action=i % 2, reward=1.0, next_obs=obs, done=(i == 9))

        batch = buffer.compute_advantages(gamma=0.99, gae_lambda=0.95)

        assert 'obs' in batch, "Should return observations"
        assert 'actions' in batch, "Should return actions"
        assert 'advantages' in batch, "Should return advantages"
        assert len(batch['advantages']) == 10, "Should have 10 advantages"


class TestAEPOTraining:
    """Test AEPO training pipeline."""

    def test_training_step(self):
        """Test single training step."""
        # Import here to avoid module-level import issues
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from train_aepo import AEPOTrainer

        policy = AEPO(obs_dim=256)
        env = ToolEnv(num_tools=3, obs_dim=256, episode_length=10)

        trainer = AEPOTrainer(policy, env, lr=1e-3)

        # Collect episodes
        collect_metrics = trainer.collect_episodes(num_episodes=2)

        assert 'mean_reward' in collect_metrics
        assert 'mean_tool_calls' in collect_metrics

        # Train
        train_metrics = trainer.train_step()

        assert 'train_loss' in train_metrics
        assert 'entropy' in train_metrics

    def test_checkpoint_save_load(self):
        """Test checkpoint saving and loading."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from train_aepo import AEPOTrainer

        policy = AEPO(obs_dim=256)
        env = ToolEnv(num_tools=3, obs_dim=256)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = AEPOTrainer(policy, env, log_dir=tmpdir)

            # Save checkpoint
            trainer.save_checkpoint('test.pt')

            checkpoint_path = Path(tmpdir) / 'test.pt'
            assert checkpoint_path.exists(), "Checkpoint should be saved"

            # Load checkpoint
            checkpoint = torch.load(checkpoint_path)
            assert 'policy_state_dict' in checkpoint
            assert 'optimizer_state_dict' in checkpoint

    def test_short_training_run(self):
        """Test short training run completes without errors."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from train_aepo import AEPOTrainer

        torch.manual_seed(42)
        np.random.seed(42)

        policy = AEPO(obs_dim=256)
        env = ToolEnv(num_tools=3, obs_dim=256, episode_length=10)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = AEPOTrainer(policy, env, lr=1e-3, log_dir=tmpdir)

            # Short training run
            trainer.train(
                num_iterations=5,
                episodes_per_iter=2,
                eval_interval=3,
                save_interval=10
            )

            assert len(trainer.metrics_history) == 5, "Should have 5 iterations"

            # Check metrics
            metrics = trainer.metrics_history[0]
            assert 'mean_reward' in metrics
            assert 'entropy' in metrics


class TestAEPOEvaluation:
    """Test AEPO evaluation."""

    def test_evaluation(self):
        """Test policy evaluation."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from eval_aepo import AEPOEvaluator

        policy = AEPO(obs_dim=256)
        env = ToolEnv(num_tools=3, obs_dim=256, episode_length=10)

        evaluator = AEPOEvaluator(policy, env)

        # Evaluate policy
        metrics = evaluator.evaluate_policy(num_episodes=5)

        assert 'mean_reward' in metrics
        assert 'mean_tool_calls' in metrics
        assert 'tool_call_rates' in metrics

    def test_baseline_evaluation(self):
        """Test baseline (oracle) evaluation."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from eval_aepo import AEPOEvaluator

        policy = AEPO(obs_dim=256)
        env = ToolEnv(num_tools=3, obs_dim=256, episode_length=10)

        evaluator = AEPOEvaluator(policy, env)

        baseline = evaluator.evaluate_baseline(num_episodes=5)

        assert 'mean_reward' in baseline
        assert 'mean_tool_calls' in baseline

    def test_gate_verification(self):
        """Test hard gate verification."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from eval_aepo import AEPOEvaluator

        policy = AEPO(obs_dim=256)
        env = ToolEnv(num_tools=3, obs_dim=256)

        evaluator = AEPOEvaluator(policy, env)

        policy_metrics = {'mean_reward': 1.0, 'mean_tool_calls': 10}
        baseline_metrics = {'mean_reward': 1.0, 'mean_tool_calls': 20}
        entropy = 0.5

        gates = evaluator.verify_gates(policy_metrics, baseline_metrics, entropy)

        assert 'tool_reduction' in gates
        assert 'reward_delta' in gates
        assert 'entropy' in gates
        assert 'all_pass' in gates

        # This should pass all gates
        assert gates['tool_reduction']['pass'], "50% reduction should pass"
        assert gates['reward_delta']['pass'], "0% delta should pass"
        assert gates['entropy']['pass'], "0.5 entropy should pass"

    def test_multi_seed_evaluation(self):
        """Test multi-seed evaluation."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from eval_aepo import AEPOEvaluator

        policy = AEPO(obs_dim=256)
        env = ToolEnv(num_tools=3, obs_dim=256, episode_length=10)

        evaluator = AEPOEvaluator(policy, env)

        results = evaluator.multi_seed_evaluation(num_seeds=3, episodes_per_seed=5)

        assert 'num_seeds' in results
        assert results['num_seeds'] == 3
        assert 'rewards' in results
        assert 'entropy' in results
        assert len(results['rewards']['values']) == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
