#!/usr/bin/env python3
"""
EXP-002: Forgetting Curves Experiment Runner
=============================================

Tests how Ara's memory decays over time under different conditions.

This experiment encodes novel facts, then probes recall at various
delays to measure forgetting dynamics.

Conditions:
    SUBCRIT   - ρ = 0.5, expect fast exponential decay
    CRITICAL  - ρ = 0.8, expect slow power-law decay
    SUPERCRIT - ρ = 1.2, expect high variance / unstable

Consolidation Policies:
    NONE      - Pure dynamics, no explicit storage
    IMMEDIATE - Write to long-term store right away
    DELAYED   - Consolidate after N steps of stability
    REHEARSAL - Periodically re-present during retention

Usage:
    # Quick test with mock model
    python scripts/science/run_forgetting_exp.py --mock --n-facts 20

    # Full experiment
    python scripts/science/run_forgetting_exp.py \
        --condition CRITICAL \
        --consolidation NONE \
        --n-facts 50 \
        --delays 0,10,50,100,500

    # Analyze results
    python scripts/science/fit_forgetting.py data/experiments/exp_002/forgetting_critical.csv

Wiring into real model:
    Replace MockModel with your actual model wrapper.
    Key methods: encode(fact), generate(cue), process(text)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

# Add project root
_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_root))

from ara.science.memory_probe import (
    MemoryProbe,
    Fact,
    RecallResult,
    ForgettingCurve,
    generate_novel_facts,
    fit_exponential,
    fit_power_law,
)


# =============================================================================
# Experimental Conditions
# =============================================================================

class Condition(Enum):
    SUBCRIT = "subcrit"     # ρ = 0.5
    CRITICAL = "critical"   # ρ = 0.8
    SUPERCRIT = "supercrit" # ρ = 1.2


class Consolidation(Enum):
    NONE = "none"           # Pure dynamics
    IMMEDIATE = "immediate" # Write immediately
    DELAYED = "delayed"     # Wait for stability
    REHEARSAL = "rehearsal" # Periodic replay


@dataclass
class ExperimentConfig:
    """Configuration for forgetting experiment."""
    condition: Condition
    consolidation: Consolidation
    n_facts: int = 50
    delays: List[int] = None  # Steps at which to probe
    distractor_steps: int = 100  # Steps between encoding and first probe
    output_dir: str = "data/experiments/exp_002"
    seed: int = 42

    # Model parameters
    rho_target: float = 0.8
    temperature: float = 0.7

    def __post_init__(self):
        if self.delays is None:
            self.delays = [0, 10, 25, 50, 100, 200, 500]

        # Set rho based on condition
        if self.condition == Condition.SUBCRIT:
            self.rho_target = 0.5
        elif self.condition == Condition.CRITICAL:
            self.rho_target = 0.8
        else:  # SUPERCRIT
            self.rho_target = 1.2


# =============================================================================
# Mock Model (Replace with real model)
# =============================================================================

class MockModel:
    """
    Mock model for testing the experiment pipeline.

    Simulates memory decay with different dynamics based on condition.
    Replace with your actual model wrapper.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        # Encoded memories (fact_id -> target)
        self._memory: Dict[str, str] = {}
        self._encode_times: Dict[str, int] = {}
        self._step = 0

        # Decay parameters based on condition
        if config.condition == Condition.SUBCRIT:
            self.decay_tau = 30      # Fast exponential
            self.decay_beta = None   # No power law
            self.noise_level = 0.1
        elif config.condition == Condition.CRITICAL:
            self.decay_tau = None    # No exponential
            self.decay_beta = 0.4    # Slow power law
            self.noise_level = 0.15
        else:  # SUPERCRIT
            self.decay_tau = 100
            self.decay_beta = None
            self.noise_level = 0.4   # High noise

    def encode(self, fact: Fact) -> None:
        """Encode a fact into memory."""
        self._memory[fact.fact_id] = fact.target
        self._encode_times[fact.fact_id] = self._step

        # Consolidation effects
        if self.config.consolidation == Consolidation.IMMEDIATE:
            # Boost retention
            pass  # Already stored
        elif self.config.consolidation == Consolidation.REHEARSAL:
            # Will be reinforced during distractor phase
            pass

    def process(self, text: str) -> None:
        """Process distractor text (advances time, causes interference)."""
        self._step += 1

        # Interference: some memories may be displaced
        if self.config.condition == Condition.SUPERCRIT:
            # High interference in supercritical
            for fact_id in list(self._memory.keys()):
                if self.rng.random() < 0.01:  # 1% per step
                    del self._memory[fact_id]

    def generate(self, cue: str, fact_id: str) -> str:
        """
        Generate response to a cue.

        In real model, this would be model.generate(cue).
        Here we simulate recall with probabilistic decay.
        """
        if fact_id not in self._memory:
            return "I don't know."

        target = self._memory[fact_id]
        encode_time = self._encode_times[fact_id]
        delay = self._step - encode_time

        # Compute retention probability
        if self.decay_beta is not None:
            # Power law: P(recall) = (1 + t/t0)^(-beta)
            t0 = 10
            p_recall = (1 + delay / t0) ** (-self.decay_beta)
        elif self.decay_tau is not None:
            # Exponential: P(recall) = exp(-t/tau)
            p_recall = np.exp(-delay / self.decay_tau)
        else:
            p_recall = 0.5

        # Add noise
        p_recall = np.clip(p_recall + self.rng.normal(0, self.noise_level), 0, 1)

        # Consolidation boosts
        if self.config.consolidation == Consolidation.IMMEDIATE:
            p_recall = min(1.0, p_recall + 0.2)
        elif self.config.consolidation == Consolidation.REHEARSAL:
            p_recall = min(1.0, p_recall + 0.1)

        # Recall or forget
        if self.rng.random() < p_recall:
            return f"The answer is {target}."
        else:
            # Partial recall or complete failure
            if self.rng.random() < 0.3:
                return f"I think it might be... {target[:len(target)//2]}..."
            return "I don't remember."

    def step(self) -> None:
        """Advance time by one step."""
        self._step += 1

    @property
    def current_step(self) -> int:
        return self._step


# =============================================================================
# Experiment Runner
# =============================================================================

def run_experiment(
    config: ExperimentConfig,
    use_mock: bool = True,
) -> Dict[str, Any]:
    """
    Run a single forgetting experiment.

    Args:
        config: Experiment configuration
        use_mock: Use mock model (True) or real model (False)

    Returns:
        Results dictionary
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print("EXP-002: Forgetting Curves Experiment")
    print(f"{'='*60}")
    print(f"Condition: {config.condition.value}")
    print(f"Consolidation: {config.consolidation.value}")
    print(f"N facts: {config.n_facts}")
    print(f"Delays: {config.delays}")
    print(f"ρ target: {config.rho_target}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    # Initialize
    if use_mock:
        print("Using MOCK model (for pipeline testing)")
        model = MockModel(config)
    else:
        # Replace with real model initialization
        raise NotImplementedError("Real model not yet integrated")

    probe = MemoryProbe(output_dir=str(output_dir))
    probe.set_condition(f"{config.condition.value}_{config.consolidation.value}")

    # Generate facts
    print(f"Generating {config.n_facts} novel facts...")
    facts = generate_novel_facts(config.n_facts, seed=config.seed)

    # === ENCODING PHASE ===
    print("\n[ENCODING PHASE]")
    for i, fact in enumerate(facts):
        # Present to model
        model.encode(fact)
        model.step()

        # Register with probe
        probe.encode_fact(fact, encode_step=model.current_step)

        if (i + 1) % 10 == 0:
            print(f"  Encoded {i + 1}/{config.n_facts} facts")

    encoding_end_step = model.current_step
    print(f"Encoding complete at step {encoding_end_step}")

    # === RETENTION INTERVAL ===
    print(f"\n[RETENTION INTERVAL] ({config.distractor_steps} steps)")
    for i in range(config.distractor_steps):
        model.process("Distractor text to cause interference...")

        # Rehearsal consolidation: re-present some facts
        if config.consolidation == Consolidation.REHEARSAL:
            if i % 20 == 0:  # Every 20 steps
                fact = random.choice(facts)
                model.encode(fact)

    print(f"Retention complete, now at step {model.current_step}")

    # === RECALL PHASE ===
    print(f"\n[RECALL PHASE]")
    for delay in config.delays:
        # Advance to target delay (from encoding end)
        target_step = encoding_end_step + delay
        while model.current_step < target_step:
            model.process("More distractor...")

        print(f"\n  Testing at delay={delay} (step {model.current_step})...")

        # Test each fact
        n_correct = 0
        for fact in facts:
            cue = probe.get_cue(fact.fact_id)
            response = model.generate(cue, fact.fact_id)
            result = probe.test_recall(
                fact.fact_id,
                response,
                model.current_step,
            )
            if result.accuracy > 0.5:
                n_correct += 1

        accuracy = n_correct / len(facts)
        print(f"    Accuracy: {accuracy:.1%} ({n_correct}/{len(facts)})")

    # === ANALYSIS ===
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")

    # Compute forgetting curve
    curve = probe.compute_forgetting_curve(delay_bins=config.delays)

    print("\nForgetting Curve:")
    print(f"  {'Delay':>6}  {'Accuracy':>8}  {'Similarity':>10}  {'N':>4}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*10}  {'-'*4}")
    for d, a, s, n in zip(curve.delays, curve.accuracies, curve.similarities, curve.n_samples):
        if not np.isnan(a):
            print(f"  {d:>6}  {a:>8.2f}  {s:>10.3f}  {n:>4}")

    # Fit curves
    delays = np.array([d for d, a in zip(curve.delays, curve.accuracies) if not np.isnan(a)])
    accuracies = np.array([a for a in curve.accuracies if not np.isnan(a)])

    if len(delays) >= 3:
        exp_params, exp_r2 = fit_exponential(delays, accuracies)
        pow_params, pow_r2 = fit_power_law(delays, accuracies)

        print(f"\nCurve Fitting:")
        if exp_params:
            print(f"  Exponential: τ={exp_params.get('tau', 0):.1f}, R²={exp_r2:.3f}")
        if pow_params:
            print(f"  Power-law:   β={pow_params.get('beta', 0):.2f}, R²={pow_r2:.3f}")

        if exp_r2 > pow_r2:
            print(f"\n  → Exponential fits better (subcritical signature)")
        else:
            print(f"\n  → Power-law fits better (critical signature)")
    else:
        exp_params, pow_params = {}, {}
        exp_r2, pow_r2 = 0, 0

    # Save results
    filename = f"forgetting_{config.condition.value}_{config.consolidation.value}.csv"
    csv_path = probe.save_results(filename)

    # Save summary
    results = {
        "condition": config.condition.value,
        "consolidation": config.consolidation.value,
        "rho_target": config.rho_target,
        "n_facts": config.n_facts,
        "delays": config.delays,
        "curve": curve.to_dict(),
        "exponential_fit": exp_params,
        "exponential_r2": exp_r2,
        "power_law_fit": pow_params,
        "power_law_r2": pow_r2,
        "best_fit": "exponential" if exp_r2 > pow_r2 else "power_law",
    }

    summary_path = output_dir / f"summary_{config.condition.value}_{config.consolidation.value}.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nSaved to:")
    print(f"  Data: {csv_path}")
    print(f"  Summary: {summary_path}")

    return results


def run_full_experiment(use_mock: bool = True) -> Dict[str, Any]:
    """
    Run all condition × consolidation combinations.
    """
    all_results = {}

    for condition in Condition:
        for consolidation in [Consolidation.NONE]:  # Start with NONE only
            key = f"{condition.value}_{consolidation.value}"
            print(f"\n\n{'#'*60}")
            print(f"# {key.upper()}")
            print(f"{'#'*60}\n")

            config = ExperimentConfig(
                condition=condition,
                consolidation=consolidation,
                n_facts=30,
                delays=[0, 10, 25, 50, 100, 200],
                seed=42,
            )

            results = run_experiment(config, use_mock=use_mock)
            all_results[key] = results

    # Summary
    print(f"\n\n{'='*60}")
    print("FULL EXPERIMENT SUMMARY")
    print(f"{'='*60}")

    print(f"\n{'Condition':<20} {'Best Fit':<12} {'τ/β':<10} {'R²':<8}")
    print(f"{'-'*20} {'-'*12} {'-'*10} {'-'*8}")

    for key, res in all_results.items():
        best = res['best_fit']
        if best == 'exponential':
            param = res['exponential_fit'].get('tau', 'N/A')
            r2 = res['exponential_r2']
        else:
            param = res['power_law_fit'].get('beta', 'N/A')
            r2 = res['power_law_r2']
        print(f"{key:<20} {best:<12} {param:<10} {r2:<8.3f}")

    return all_results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run EXP-002: Forgetting Curves Experiment",
        epilog="Example: python run_forgetting_exp.py --condition CRITICAL --mock",
    )

    parser.add_argument(
        "--condition", "-c",
        choices=["subcrit", "critical", "supercrit"],
        default="critical",
        help="Experimental condition",
    )
    parser.add_argument(
        "--consolidation",
        choices=["none", "immediate", "delayed", "rehearsal"],
        default="none",
        help="Consolidation policy",
    )
    parser.add_argument(
        "--n-facts", "-n",
        type=int,
        default=50,
        help="Number of facts to encode",
    )
    parser.add_argument(
        "--delays",
        type=str,
        default="0,10,25,50,100,200,500",
        help="Comma-separated delay values",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="data/experiments/exp_002",
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock model (for testing pipeline)",
    )
    parser.add_argument(
        "--full-experiment",
        action="store_true",
        help="Run all conditions",
    )

    args = parser.parse_args()

    if args.full_experiment:
        run_full_experiment(use_mock=args.mock or True)
    else:
        delays = [int(d) for d in args.delays.split(",")]

        config = ExperimentConfig(
            condition=Condition(args.condition),
            consolidation=Consolidation(args.consolidation),
            n_facts=args.n_facts,
            delays=delays,
            output_dir=args.output_dir,
            seed=args.seed,
        )

        run_experiment(config, use_mock=args.mock or True)


if __name__ == "__main__":
    main()
