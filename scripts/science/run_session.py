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

Unified Telemetry:
    Each session logs THREE gauges simultaneously:
    - ρ (criticality): Via avalanche exponents (τ, α)
    - D (delusion index): Force ratio from SanityMonitor
    - Π_y/Π_μ (precision ratio): From PrecisionMonitor

Usage:
    # Quick test with synthetic data
    python scripts/science/run_session.py --synthetic --steps 5000

    # Real run (when model is available)
    python scripts/science/run_session.py --condition TARGET --steps 10000

    # Full experiment (all three conditions)
    python scripts/science/run_session.py --full-experiment

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
from ara.science.cognitive_health_trace import CognitiveHealthTrace, CognitiveMode
from ara.gutc.sanity_monitor import SanityMonitor, SanityMode
from ara.gutc.precision_diagnostics import PrecisionMonitor


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

    # Precision weights (for GUTC)
    extrinsic_weight: float = 0.6  # Π_y (sensory precision)
    intrinsic_weight: float = 0.4  # Π_μ (prior precision)

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
                extrinsic_weight=0.7,  # More sensory-reliant (rigid)
                intrinsic_weight=0.3,
            )
        elif condition == Condition.TARGET:
            return cls(
                condition=condition,
                n_steps=n_steps,
                temperature=0.7,  # Will be dynamic
                homeostasis_enabled=True,
                extrinsic_weight=0.5,  # Balanced
                intrinsic_weight=0.5,
            )
        else:  # HOT
            return cls(
                condition=condition,
                n_steps=n_steps,
                temperature=1.5,
                homeostasis_enabled=False,
                extrinsic_weight=0.3,  # More prior-dominated (hallucinatory)
                intrinsic_weight=0.7,
            )


# =============================================================================
# Model Interface (Replace with your actual model)
# =============================================================================

@dataclass
class MockOutput:
    """Output from a mock forward pass with all telemetry signals."""
    output_text: str
    hidden_states: np.ndarray
    sensory_error: float   # |ε_y| - discrepancy from input
    prior_error: float     # |ε_μ| - discrepancy from expectation
    rho: float             # Criticality estimate


class MockModel:
    """
    Mock model for testing the pipeline.

    Replace this with your actual model wrapper. The key method is
    `forward()` which should return MockOutput with all signals.
    """

    def __init__(self, config: SessionConfig):
        self.config = config
        self.rng = np.random.default_rng(42)
        self._step = 0
        self._prev_state: Optional[np.ndarray] = None

        # Simulate different regimes
        if config.condition == Condition.COLD:
            self.noise_scale = 0.1  # Low variance
            self.burst_prob = 0.01  # Rare bursts
            self.rho_target = 0.6   # Subcritical
            self.sensory_error_scale = 0.2  # Low sensory error
            self.prior_error_scale = 0.5    # Higher prior error
        elif config.condition == Condition.TARGET:
            self.noise_scale = 0.5  # Moderate variance
            self.burst_prob = 0.05  # Power-law-ish bursts
            self.rho_target = 0.85  # Near-critical
            self.sensory_error_scale = 0.5  # Balanced
            self.prior_error_scale = 0.5
        else:  # HOT
            self.noise_scale = 1.5  # High variance
            self.burst_prob = 0.15  # Frequent large bursts
            self.rho_target = 1.2   # Supercritical
            self.sensory_error_scale = 0.8  # High sensory error (ignored)
            self.prior_error_scale = 0.2    # Low prior error (prior-dominated)

    def forward(self, input_text: Optional[str] = None) -> MockOutput:
        """
        Run one forward pass.

        Args:
            input_text: Optional input (ignored in mock)

        Returns:
            MockOutput with hidden states and telemetry signals
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

        # Simulate errors for sanity monitor
        sensory_error = abs(self.rng.normal(self.sensory_error_scale, 0.1))
        prior_error = abs(self.rng.normal(self.prior_error_scale, 0.1))

        # Simulate rho (criticality) with noise around target
        rho = self.rho_target + self.rng.normal(0, 0.05)
        rho = max(0.0, min(2.0, rho))  # Clamp to [0, 2]

        output_text = f"[step {self._step}]"

        return MockOutput(
            output_text=output_text,
            hidden_states=hidden_states,
            sensory_error=sensory_error,
            prior_error=prior_error,
            rho=rho,
        )


# =============================================================================
# Session Runner
# =============================================================================

@dataclass
class SessionResults:
    """Results from a recording session."""
    avalanche_csv: Path
    health_json: Path
    avalanche_stats: Dict[str, Any]
    health_summary: Dict[str, Any]
    sanity_stats: Dict[str, Any]
    precision_stats: Dict[str, Any]


def run_session(config: SessionConfig, use_synthetic: bool = False) -> SessionResults:
    """
    Run a recording session with unified telemetry.

    Logs all three gauges simultaneously:
    - ρ (criticality) via avalanche statistics
    - D (delusion index) via SanityMonitor
    - Π_y/Π_μ (precision ratio) via PrecisionMonitor

    Args:
        config: Session configuration
        use_synthetic: Use synthetic generator instead of model

    Returns:
        SessionResults with paths to all output files and summary stats
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"EXP-001: Avalanche Recording Session (Unified Telemetry)")
    print(f"{'='*60}")
    print(f"Condition: {config.condition.value.upper()}")
    print(f"Steps: {config.n_steps}")
    print(f"Temperature: {config.temperature}")
    print(f"Homeostasis: {'ON' if config.homeostasis_enabled else 'OFF'}")
    print(f"Precision: Π_y={config.extrinsic_weight:.2f}, Π_μ={config.intrinsic_weight:.2f}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    # Initialize all monitors
    sanity_monitor = SanityMonitor()
    precision_monitor = PrecisionMonitor()
    health_trace = CognitiveHealthTrace(
        output_dir=str(output_dir),
        session_id=f"exp001_{config.condition.value}",
    )

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

        # Generate synthetic health data
        rng = np.random.default_rng(42)
        for step in range(config.n_steps):
            # Simulate telemetry based on regime
            if regime == "subcritical":
                rho = 0.6 + rng.normal(0, 0.05)
                sensory_err = abs(rng.normal(0.2, 0.1))
                prior_err = abs(rng.normal(0.5, 0.1))
            elif regime == "critical":
                rho = 0.85 + rng.normal(0, 0.05)
                sensory_err = abs(rng.normal(0.5, 0.1))
                prior_err = abs(rng.normal(0.5, 0.1))
            else:  # supercritical
                rho = 1.2 + rng.normal(0, 0.1)
                sensory_err = abs(rng.normal(0.8, 0.1))
                prior_err = abs(rng.normal(0.2, 0.1))

            # Log to monitors
            sanity_reading = sanity_monitor.check(
                sensory_error=sensory_err,
                prior_error=prior_err,
                Pi_y=config.extrinsic_weight,
                Pi_mu=config.intrinsic_weight,
            )
            precision_monitor.update(config.extrinsic_weight, config.intrinsic_weight)

            # Log unified health trace (mode computed internally)
            health_trace.log_step(
                step=step,
                sensory_precision=config.extrinsic_weight,
                prior_precision=config.intrinsic_weight,
                rho=rho,
            )

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
            # Forward pass (returns MockOutput with all signals)
            result = model.forward()

            # Log avalanche data
            logger.log_step(result.hidden_states, step)

            # Log sanity (delusion index)
            sanity_reading = sanity_monitor.check(
                sensory_error=result.sensory_error,
                prior_error=result.prior_error,
                Pi_y=config.extrinsic_weight,
                Pi_mu=config.intrinsic_weight,
            )

            # Log precision ratio
            precision_monitor.update(config.extrinsic_weight, config.intrinsic_weight)

            # Log unified health trace (mode computed internally)
            health_trace.log_step(
                step=step,
                sensory_precision=config.extrinsic_weight,
                prior_precision=config.intrinsic_weight,
                rho=result.rho,
            )

            # Progress
            if (step + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                rate = (step + 1) / elapsed
                print(f"  Step {step + 1}/{config.n_steps} ({rate:.1f} steps/s)")

        elapsed = time.time() - start_time
        print(f"\nRecording complete in {elapsed:.1f}s")

    # Save all results
    filename = f"avalanches_{config.condition.value}.csv"
    csv_path = logger.save_session(filename)

    health_filename = f"cognitive_health_{config.condition.value}.json"
    health_path = health_trace.save(health_filename)

    # Gather statistics
    avalanche_stats = logger.get_statistics()
    health_summary = health_trace.compute_summary().to_dict()
    sanity_stats = sanity_monitor.get_statistics()
    precision_stats = precision_monitor.get_statistics()

    # Print comprehensive summary
    print(f"\n{'='*60}")
    print("SESSION SUMMARY - UNIFIED TELEMETRY")
    print(f"{'='*60}")

    print("\n[Avalanches (ρ proxy)]")
    print(f"  Total avalanches: {avalanche_stats['n_avalanches']}")
    if avalanche_stats['n_avalanches'] > 0:
        print(f"  Size (mean±std): {avalanche_stats.get('size_mean', 0):.1f} ± "
              f"{avalanche_stats.get('size_std', 0):.1f}")
        print(f"  Duration (mean±std): {avalanche_stats.get('duration_mean', 0):.1f} ± "
              f"{avalanche_stats.get('duration_std', 0):.1f}")

    print("\n[Criticality (ρ)]")
    print(f"  Mean ρ: {health_summary.get('mean_rho', 0):.3f}")
    print(f"  Std ρ: {health_summary.get('std_rho', 0):.3f}")

    print("\n[Delusion Index (D)]")
    print(f"  Mean D: {sanity_stats.get('mean_delusion_index', 0):.3f}")
    print(f"  Healthy fraction: {sanity_stats.get('healthy_fraction', 0):.1%}")
    print(f"  Trend: {sanity_stats.get('trend', 'unknown')}")

    print("\n[Precision Ratio (Π_y/Π_μ)]")
    print(f"  Current ratio: {precision_stats.get('current_ratio', 0):.3f}")
    print(f"  Pathology: {precision_stats.get('current_pathology', 'UNKNOWN')}")

    print("\n[Cognitive Mode Distribution]")
    mode_dist = health_summary.get('mode_distribution', {})
    for mode, count in mode_dist.items():
        pct = count / health_summary.get('n_samples', 1) * 100
        print(f"  {mode}: {pct:.1f}%")

    print(f"\n{'='*60}")
    print("OUTPUT FILES")
    print(f"{'='*60}")
    print(f"  Avalanches: {csv_path}")
    print(f"  Health trace: {health_path}")

    print(f"\nNext step:")
    print(f"  python scripts/science/fit_powerlaw.py {csv_path}")

    return SessionResults(
        avalanche_csv=csv_path,
        health_json=health_path,
        avalanche_stats=avalanche_stats,
        health_summary=health_summary,
        sanity_stats=sanity_stats,
        precision_stats=precision_stats,
    )


def run_full_experiment(n_steps: int = 5000, use_synthetic: bool = True) -> Dict[str, SessionResults]:
    """
    Run all three conditions for a complete experiment.

    Args:
        n_steps: Number of steps per condition
        use_synthetic: Use synthetic data (for pipeline testing)

    Returns:
        Dict mapping condition name to SessionResults
    """
    results = {}

    for condition in [Condition.COLD, Condition.TARGET, Condition.HOT]:
        print(f"\n\n{'#'*60}")
        print(f"# CONDITION: {condition.value.upper()}")
        print(f"{'#'*60}\n")

        config = SessionConfig.from_condition(condition, n_steps=n_steps)
        session_results = run_session(config, use_synthetic=use_synthetic)
        results[condition.value] = session_results

        print("\n")

    print(f"\n{'='*60}")
    print("FULL EXPERIMENT COMPLETE")
    print(f"{'='*60}")

    # Summary comparison
    print("\n[Condition Comparison]")
    print(f"{'Condition':<10} {'ρ mean':<10} {'D mean':<10} {'Healthy %':<12} {'Avalanches':<10}")
    print("-" * 52)
    for cond, res in results.items():
        rho = res.health_summary.get('mean_rho', 0)
        d = res.sanity_stats.get('mean_delusion_index', 0)
        healthy = res.sanity_stats.get('healthy_fraction', 0) * 100
        n_aval = res.avalanche_stats.get('n_avalanches', 0)
        print(f"{cond.upper():<10} {rho:<10.3f} {d:<10.3f} {healthy:<12.1f} {n_aval:<10}")

    print("\n[Output Files]")
    for cond, res in results.items():
        print(f"  {cond}:")
        print(f"    Avalanches: {res.avalanche_csv}")
        print(f"    Health: {res.health_json}")

    print("\n[Analysis Commands]")
    for cond, res in results.items():
        print(f"  python scripts/science/fit_powerlaw.py {res.avalanche_csv}")

    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run avalanche recording session for EXP-001 with unified telemetry",
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
        help="Run all three conditions (COLD, TARGET, HOT)",
    )

    args = parser.parse_args()

    if args.full_experiment:
        run_full_experiment(n_steps=args.steps, use_synthetic=args.synthetic)
    else:
        condition = Condition(args.condition)
        config = SessionConfig.from_condition(condition, n_steps=args.steps)
        config.output_dir = args.output_dir
        run_session(config, use_synthetic=args.synthetic)


if __name__ == "__main__":
    main()
