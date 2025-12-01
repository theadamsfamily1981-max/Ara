#!/usr/bin/env python3
"""
SNN Structural Learning Training Script (Phase 1.1)

Trains the AEPO agent to self-tune SNN structure for Pareto-optimal
performance (Accuracy vs Energy vs Stability).

Uses the ara.aepo module's structural learning environment.

Output: best.yaml config file for runtime Model Selector

Usage:
    python scripts/train_structural.py --epochs 500 --promote
    python scripts/train_structural.py --epochs 1000 --epr-threshold 0.15 --output configs/
"""

import argparse
import sys
import os
import json
import yaml
import time
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add project root to path
_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root))

from ara.aepo import (
    AEPOEnv,
    AEPOAgent,
    ParetoMetrics,
    SNNParameters,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger("train_structural")


@dataclass
class StructuralTrainingResult:
    """Results from structural learning training run."""
    # Best solution found
    best_accuracy: float
    best_energy: float
    best_stability: float
    best_hypervolume: float
    best_epr_cv: float

    # Best SNN parameters
    best_tau: float
    best_density: float
    best_num_neurons: int

    # Training stats
    total_episodes: int
    training_time_seconds: float
    pareto_front_size: int

    # Promotion status
    promoted: bool
    config_path: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def meets_gates(self, epr_threshold: float = 0.15) -> bool:
        """Check if result meets production gates."""
        return (
            self.best_accuracy >= 0.90 and
            self.best_energy <= 0.15 and
            self.best_epr_cv <= epr_threshold
        )


def run_structural_training(
    num_episodes: int = 500,
    max_steps: int = 200,
    num_neurons: int = 128,
    epr_threshold: float = 0.15,
    checkpoint_dir: str = "./checkpoints/structural",
) -> tuple:
    """
    Run full structural learning training.

    Args:
        num_episodes: Number of training episodes
        max_steps: Max steps per episode
        num_neurons: Number of SNN neurons
        epr_threshold: EPR-CV stability threshold
        checkpoint_dir: Directory for checkpoints

    Returns:
        (result, best_params, episode_results)
    """
    logger.info("="*70)
    logger.info("SNN STRUCTURAL LEARNING TRAINING")
    logger.info("="*70)
    logger.info(f"Episodes: {num_episodes}")
    logger.info(f"Max Steps: {max_steps}")
    logger.info(f"Neurons: {num_neurons}")
    logger.info(f"EPR-CV Threshold: {epr_threshold}")
    logger.info("="*70)

    start_time = time.time()
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create environment with production targets
    env = AEPOEnv(
        num_neurons=num_neurons,
        max_steps=max_steps,
        target_accuracy=0.95,
        target_energy=0.1,
        target_latency_us=200.0,
        use_tprl=True,
    )

    # Create agent
    agent = AEPOAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=256,
        learning_rate=0.001,
    )

    # Track best results
    best_metrics: Optional[ParetoMetrics] = None
    best_params: Optional[SNNParameters] = None
    best_hypervolume = 0.0

    episode_results = []

    logger.info("\nStarting training loop...")

    for ep in range(num_episodes):
        state = env.reset()
        episode_rewards = []
        episode_states = []
        episode_actions = []

        while not env.done:
            action = agent.select_action(state, explore=True)
            next_state, reward, done, info = env.step_env(action)

            episode_rewards.append(reward)
            episode_states.append(state)
            episode_actions.append(action)

            state = next_state

        # Update policy
        agent.update(episode_rewards, episode_states, episode_actions)

        # Track best by hypervolume
        hv = env._compute_hypervolume()
        if hv > best_hypervolume:
            best_hypervolume = hv
            best_metrics = env.metrics
            best_params = env.params

        # Also track by EPR-CV gate (prefer stable solutions)
        if env.metrics.epr_cv <= epr_threshold:
            if best_metrics is None or env.metrics.accuracy > best_metrics.accuracy:
                best_metrics = env.metrics
                best_params = env.params

        episode_results.append({
            "episode": ep + 1,
            "accuracy": env.metrics.accuracy,
            "energy": env.metrics.energy,
            "epr_cv": env.metrics.epr_cv,
            "stability": env.metrics.stability,
            "hypervolume": hv,
            "reward": sum(episode_rewards),
            "pareto_front_size": len(env.pareto_front),
        })

        # Progress logging
        if (ep + 1) % 50 == 0:
            logger.info(
                f"Episode {ep + 1}/{num_episodes}: "
                f"acc={env.metrics.accuracy:.3f}, "
                f"energy={env.metrics.energy:.3f}, "
                f"EPR-CV={env.metrics.epr_cv:.3f}, "
                f"HV={hv:.4f}, "
                f"front={len(env.pareto_front)}"
            )

            # Save checkpoint
            checkpoint = {
                "episode": ep + 1,
                "best_accuracy": best_metrics.accuracy if best_metrics else 0,
                "best_energy": best_metrics.energy if best_metrics else 1,
                "best_epr_cv": best_metrics.epr_cv if best_metrics else float('inf'),
                "best_hypervolume": best_hypervolume,
            }
            with open(f"{checkpoint_dir}/checkpoint_{ep+1}.json", 'w') as f:
                json.dump(checkpoint, f, indent=2)

    elapsed = time.time() - start_time

    # Ensure we have best params
    if best_params is None:
        best_params = env.params
    if best_metrics is None:
        best_metrics = env.metrics

    # Build result
    result = StructuralTrainingResult(
        best_accuracy=best_metrics.accuracy,
        best_energy=best_metrics.energy,
        best_stability=best_metrics.stability,
        best_hypervolume=best_hypervolume,
        best_epr_cv=best_metrics.epr_cv,
        best_tau=best_params.tau,
        best_density=best_params.density(),
        best_num_neurons=num_neurons,
        total_episodes=num_episodes,
        training_time_seconds=elapsed,
        pareto_front_size=len(env.pareto_front),
        promoted=False,
        config_path=None,
    )

    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE")
    logger.info("="*70)
    logger.info(f"Best Accuracy: {result.best_accuracy:.3f}")
    logger.info(f"Best Energy: {result.best_energy:.3f}")
    logger.info(f"Best Stability: {result.best_stability:.3f}")
    logger.info(f"Best EPR-CV: {result.best_epr_cv:.3f}")
    logger.info(f"Best Hypervolume: {result.best_hypervolume:.4f}")
    logger.info(f"Pareto Front Size: {result.pareto_front_size}")
    logger.info(f"Training Time: {elapsed:.1f}s")
    logger.info(f"Meets Gates: {result.meets_gates(epr_threshold)}")
    logger.info("="*70)

    return result, best_params, episode_results


def generate_config(
    result: StructuralTrainingResult,
    params: SNNParameters,
    output_dir: str = "./configs",
) -> str:
    """Generate best.yaml configuration file."""
    os.makedirs(output_dir, exist_ok=True)

    config = {
        "version": "1.0",
        "generated_at": datetime.utcnow().isoformat(),
        "type": "structural_learning",
        "training": {
            "episodes": result.total_episodes,
            "training_time_seconds": round(result.training_time_seconds, 2),
            "pareto_front_size": result.pareto_front_size,
        },
        "metrics": {
            "accuracy": round(result.best_accuracy, 4),
            "energy": round(result.best_energy, 4),
            "stability": round(result.best_stability, 4),
            "epr_cv": round(result.best_epr_cv, 4),
            "hypervolume": round(result.best_hypervolume, 4),
        },
        "snn": {
            "num_neurons": result.best_num_neurons,
            "tau": round(params.tau, 6),
            "density": round(params.density(), 4),
            "threshold_mean": round(sum(params.v_th) / len(params.v_th), 4),
            "refractory_mean": round(sum(params.r) / len(params.r), 6),
        },
        "gates": {
            "accuracy_gate": 0.90,
            "energy_gate": 0.15,
            "epr_cv_gate": 0.15,
            "passed": result.meets_gates(0.15),
        },
        "deployment": {
            "recommended_backend": "triton" if result.meets_gates() else "development",
            "cxl_offload": result.best_energy <= 0.1,
            "turbo_cache": result.best_accuracy >= 0.92,
        },
    }

    config_path = os.path.join(output_dir, "best.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Config written to: {config_path}")
    return config_path


def promote_config(config_path: str, production_dir: str = "./production") -> bool:
    """Promote configuration to production."""
    os.makedirs(production_dir, exist_ok=True)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if not config.get("gates", {}).get("passed", False):
        logger.warning("Config does not pass gates, skipping promotion")
        return False

    prod_path = os.path.join(production_dir, "active.yaml")
    with open(prod_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # Create promotion record
    record = {
        "promoted_at": datetime.utcnow().isoformat(),
        "source": config_path,
        "metrics": config["metrics"],
    }

    record_path = os.path.join(production_dir, "promotion_history.json")
    history = []
    if os.path.exists(record_path):
        with open(record_path) as f:
            history = json.load(f)
    history.append(record)
    with open(record_path, 'w') as f:
        json.dump(history, f, indent=2)

    logger.info(f"Config promoted to: {prod_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="SNN Structural Learning Training"
    )
    parser.add_argument(
        "--epochs", type=int, default=500,
        help="Number of training episodes"
    )
    parser.add_argument(
        "--max-steps", type=int, default=200,
        help="Max steps per episode"
    )
    parser.add_argument(
        "--neurons", type=int, default=128,
        help="Number of SNN neurons"
    )
    parser.add_argument(
        "--epr-threshold", type=float, default=0.15,
        help="EPR-CV stability threshold"
    )
    parser.add_argument(
        "--output", type=str, default="./configs",
        help="Output directory for configs"
    )
    parser.add_argument(
        "--promote", action="store_true",
        help="Auto-promote to production if gates pass"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="./checkpoints/structural",
        help="Checkpoint directory"
    )

    args = parser.parse_args()

    # Run training
    result, best_params, episode_results = run_structural_training(
        num_episodes=args.epochs,
        max_steps=args.max_steps,
        num_neurons=args.neurons,
        epr_threshold=args.epr_threshold,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Generate config
    if best_params:
        config_path = generate_config(result, best_params, args.output)
        result.config_path = config_path

        # Auto-promote if requested and gates pass
        if args.promote and result.meets_gates(args.epr_threshold):
            if promote_config(config_path):
                result.promoted = True
                logger.info("✓ Configuration promoted to production")
            else:
                logger.warning("✗ Promotion failed")
        elif args.promote:
            logger.warning("✗ Gates not passed, skipping auto-promotion")

    # Save training results
    results_path = os.path.join(args.output, "structural_training_results.json")
    os.makedirs(args.output, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump({
            "result": result.to_dict(),
            "episodes": episode_results,
        }, f, indent=2)
    logger.info(f"Results saved to: {results_path}")

    return 0 if result.meets_gates(args.epr_threshold) else 1


if __name__ == "__main__":
    sys.exit(main())
