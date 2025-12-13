#!/usr/bin/env python3
"""
EXP-001: Live Avalanche Recording Session
==========================================

This is the "plug it in and go" script for recording Ara's neural
avalanches during actual inference.

Three conditions:
    COLD    - Low temperature, no homeostasis (should be subcritical)
    TARGET  - Homeostatic control active (should be critical)
    HOT     - High temperature, no homeostasis (should be supercritical)

Usage:
    # Quick test with synthetic data
    python scripts/science/run_session.py --synthetic --steps 5000

    # Real run (when model is available)
    python scripts/science/run_session.py --condition TARGET --steps 10000

    # Then analyze:
    python scripts/science/fit_powerlaw.py data/experiments/exp_001/avalanches.csv

Wiring into your actual model:
    Replace `mock_forward()` with your real forward pass.
    The key is extracting `hidden_states` from whatever layer you want to monitor.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

import numpy as np

# Add project root to path
_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_root))

from ara.science.avalanche_logger import AvalancheLogger, SyntheticAvalancheGenerator


# =============================================================================
# Experimental Conditions
# =============================================================================

class Condition(Enum):
    COLD = "cold"       # T=0.1, subcritical expected
    TARGET = "target"   # T=dynamic, critical expected
    HOT = "hot"         # T=1.5, supercritical expected


@dataclass
class SessionConfig:
    """Configuration for a recording session."""
    condition: Condition
    n_steps: int = 10000
    output_dir: str = "data/experiments/exp_001"

    # Model parameters (adjust these for real runs)
    temperature: float = 0.7
    homeostasis_enabled: bool = True

    # Logging parameters
    threshold_sigma: float = 2.0
    baseline_window: int = 100

    @classmethod
    def from_condition(cls, condition: Condition, n_steps: int = 10000) -> "SessionConfig":
        """Create config from experimental condition."""
        if condition == Condition.COLD:
            return cls(
                condition=condition,
                n_steps=n_steps,
                temperature=0.1,
                homeostasis_enabled=False,
            )
        elif condition == Condition.TARGET:
            return cls(
                condition=condition,
                n_steps=n_steps,
                temperature=0.7,  # Will be dynamic
                homeostasis_enabled=True,
            )
        else:  # HOT
            return cls(
                condition=condition,
                n_steps=n_steps,
                temperature=1.5,
                homeostasis_enabled=False,
            )


# =============================================================================
# Model Interface (Replace with your actual model)
# =============================================================================

class MockModel:
    """
    Mock model for testing the pipeline.

    Replace this with your actual model wrapper. The key method is
    `forward()` which should return (output, hidden_states).
    """

    def __init__(self, config: SessionConfig):
        self.config = config
        self.rng = np.random.default_rng(42)
        self._step = 0

        # Simulate different regimes
        if config.condition == Condition.COLD:
            self.noise_scale = 0.1  # Low variance
            self.burst_prob = 0.01  # Rare bursts
        elif config.condition == Condition.TARGET:
            self.noise_scale = 0.5  # Moderate variance
            self.burst_prob = 0.05  # Power-law-ish bursts
        else:  # HOT
            self.noise_scale = 1.5  # High variance
            self.burst_prob = 0.15  # Frequent large bursts

    def forward(self, input_text: Optional[str] = None) -> Tuple[str, np.ndarray]:
        """
        Run one forward pass.

        Args:
            input_text: Optional input (ignored in mock)

        Returns:
            (output_text, hidden_states): Model output and internal activations
        """
        self._step += 1

        # Simulate hidden states (replace with actual extraction)
        # Shape: (hidden_dim,) - flatten whatever you get from the model
        hidden_dim = 4096

        # Base noise
        hidden_states = self.rng.normal(0, self.noise_scale, hidden_dim)

        # Occasional bursts (simulates avalanches)
        if self.rng.random() < self.burst_prob:
            burst_size = int(self.rng.pareto(1.5) * 100) + 10
            burst_size = min(burst_size, hidden_dim)
            burst_indices = self.rng.choice(hidden_dim, burst_size, replace=False)
            hidden_states[burst_indices] += self.rng.normal(3.0, 1.0, burst_size)

        output_text = f"[step {self._step}]"
        return output_text, hidden_states


# =============================================================================
# Session Runner
# =============================================================================

def run_session(config: SessionConfig, use_synthetic: bool = False) -> Path:
    """
    Run a recording session.

    Args:
        config: Session configuration
        use_synthetic: Use synthetic generator instead of model

    Returns:
        Path to saved avalanche CSV
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"EXP-001: Avalanche Recording Session")
    print(f"{'='*60}")
    print(f"Condition: {config.condition.value.upper()}")
    print(f"Steps: {config.n_steps}")
    print(f"Temperature: {config.temperature}")
    print(f"Homeostasis: {'ON' if config.homeostasis_enabled else 'OFF'}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    if use_synthetic:
        # Use synthetic generator for testing
        regime = {
            Condition.COLD: "subcritical",
            Condition.TARGET: "critical",
            Condition.HOT: "supercritical",
        }[config.condition]

        print(f"Using SYNTHETIC data generator (regime={regime})")
        gen = SyntheticAvalancheGenerator(regime=regime)
        logger = gen.generate_session(n_steps=config.n_steps)
        logger.log_dir = output_dir

    else:
        # Real model recording
        print("Initializing model...")
        model = MockModel(config)  # Replace with real model

        print("Initializing avalanche logger...")
        logger = AvalancheLogger(
            log_dir=str(output_dir),
            threshold_sigma=config.threshold_sigma,
            baseline_window=config.baseline_window,
        )

        print(f"\nRecording {config.n_steps} steps...")
        start_time = time.time()

        for step in range(config.n_steps):
            # Forward pass
            output, hidden_states = model.forward()

            # Extract signal and log
            # Option 1: Mean activation (simple)
            # signal = hidden_states

            # Option 2: Just pass the full tensor (logger handles it)
            logger.log_step(hidden_states, step)

            # Progress
            if (step + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                rate = (step + 1) / elapsed
                print(f"  Step {step + 1}/{config.n_steps} ({rate:.1f} steps/s)")

        elapsed = time.time() - start_time
        print(f"\nRecording complete in {elapsed:.1f}s")

    # Save results
    filename = f"avalanches_{config.condition.value}.csv"
    csv_path = logger.save_session(filename)

    # Print summary
    stats = logger.get_statistics()
    print(f"\n{'='*60}")
    print("SESSION SUMMARY")
    print(f"{'='*60}")
    print(f"Total avalanches: {stats['n_avalanches']}")
    if stats['n_avalanches'] > 0:
        print(f"Size range: [{stats.get('size_mean', 0) - stats.get('size_std', 0):.1f}, "
              f"{stats.get('size_mean', 0) + stats.get('size_std', 0):.1f}] (mean ± std)")
        print(f"Duration range: [{stats.get('duration_mean', 0) - stats.get('duration_std', 0):.1f}, "
              f"{stats.get('duration_mean', 0) + stats.get('duration_std', 0):.1f}]")
        print(f"Active fraction: {stats.get('active_fraction', 0):.1%}")

    print(f"\nData saved to: {csv_path}")
    print(f"\nNext step:")
    print(f"  python scripts/science/fit_powerlaw.py {csv_path}")

    return csv_path


def run_full_experiment() -> Dict[str, Path]:
    """
    Run all three conditions for a complete experiment.

    Returns:
        Dict mapping condition name to CSV path
    """
    results = {}

    for condition in [Condition.COLD, Condition.TARGET, Condition.HOT]:
        print(f"\n\n{'#'*60}")
        print(f"# CONDITION: {condition.value.upper()}")
        print(f"{'#'*60}\n")

        config = SessionConfig.from_condition(condition, n_steps=5000)
        csv_path = run_session(config, use_synthetic=True)
        results[condition.value] = csv_path

        print("\n")

    print(f"\n{'='*60}")
    print("FULL EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    print("\nResults saved:")
    for cond, path in results.items():
        print(f"  {cond}: {path}")

    print("\nTo analyze all conditions:")
    for cond, path in results.items():
        print(f"  python scripts/science/fit_powerlaw.py {path}")

    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run avalanche recording session for EXP-001",
        epilog="Example: python run_session.py --condition TARGET --steps 10000",
    )

    parser.add_argument(
        "--condition", "-c",
        choices=["cold", "target", "hot"],
        default="target",
        help="Experimental condition (default: target)",
    )
    parser.add_argument(
        "--steps", "-n",
        type=int,
        default=10000,
        help="Number of steps to record (default: 10000)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="data/experiments/exp_001",
        help="Output directory (default: data/experiments/exp_001)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data generator (for testing pipeline)",
    )
    parser.add_argument(
        "--full-experiment",
        action="store_true",
        help="Run all three conditions",
    )

    args = parser.parse_args()

    if args.full_experiment:
        run_full_experiment()
    else:
        condition = Condition(args.condition)
        config = SessionConfig.from_condition(condition, n_steps=args.steps)
        config.output_dir = args.output_dir
        run_session(config, use_synthetic=args.synthetic)


if __name__ == "__main__":
    main()
