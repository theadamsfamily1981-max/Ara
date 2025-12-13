#!/usr/bin/env python3
"""
EXP-001: Avalanche Criticality Experiment Runner

Three-condition experiment measuring neural avalanche statistics
across criticality regimes:
- COLD (ρ ≈ 0.6): Subcritical, ordered, short cascades
- TARGET (ρ ≈ 0.8-0.9): Near-critical, optimal capacity-robustness trade-off
- HOT (ρ ≈ 1.1-1.2): Supercritical, chaotic, long cascades

Integrates with:
- AvalancheLogger for cascade detection
- fit_powerlaw.py for exponent estimation
- Cognitive Cockpit HUD for real-time visualization
- GUTC framework for capacity interpretation

Usage:
    python -m experiments.avalanche_criticality.exp001_runner --condition all
    python -m experiments.avalanche_criticality.exp001_runner --condition target --steps 2000
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("exp001")


# =============================================================================
# Experiment Configuration
# =============================================================================

class Condition(Enum):
    """Experimental conditions mapping to criticality regimes."""
    COLD = "cold"       # Subcritical: ρ ≈ 0.6
    TARGET = "target"   # Near-critical: ρ ≈ 0.85
    HOT = "hot"         # Supercritical: ρ ≈ 1.15


@dataclass
class ConditionConfig:
    """Configuration for a single experimental condition."""
    name: str
    target_rho: float           # Target branching ratio
    temperature: float          # LLM temperature analog
    noise_scale: float          # Input noise magnitude
    damping: float              # Damping factor for stability
    description: str

    @classmethod
    def from_condition(cls, condition: Condition) -> "ConditionConfig":
        configs = {
            Condition.COLD: cls(
                name="COLD",
                target_rho=0.6,
                temperature=0.3,
                noise_scale=0.05,
                damping=0.8,
                description="Subcritical: ordered, short cascades, high stability"
            ),
            Condition.TARGET: cls(
                name="TARGET",
                target_rho=0.85,
                temperature=0.8,
                noise_scale=0.1,
                damping=0.5,
                description="Near-critical: balanced capacity-robustness trade-off"
            ),
            Condition.HOT: cls(
                name="HOT",
                target_rho=1.15,
                temperature=1.2,
                noise_scale=0.2,
                damping=0.2,
                description="Supercritical: chaotic, long cascades, instability risk"
            ),
        }
        return configs[condition]


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    conditions: List[Condition] = field(default_factory=lambda: list(Condition))
    steps_per_condition: int = 1000
    hidden_dim: int = 256
    avalanche_threshold_factor: float = 0.1  # Threshold = factor * std
    output_dir: Path = field(default_factory=lambda: Path("data/experiments/exp001"))
    enable_hud: bool = True
    hud_update_interval: int = 10  # Update HUD every N steps


# =============================================================================
# Simulated Cognition Core
# =============================================================================

class SimulatedCognitionCore:
    """
    Simulates a recurrent neural core with tunable criticality.

    Models thought dynamics as:
        h_t = tanh(W_h @ h_{t-1} + W_x @ x_t + noise)

    Criticality (ρ) controlled via spectral radius of W_h.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        target_rho: float = 0.85,
        temperature: float = 0.8,
        noise_scale: float = 0.1,
        damping: float = 0.5,
    ):
        self.hidden_dim = hidden_dim
        self.target_rho = target_rho
        self.temperature = temperature
        self.noise_scale = noise_scale
        self.damping = damping

        # Initialize weights
        self._init_weights()

        # State
        self.h = np.zeros(hidden_dim)
        self.h_prev = np.zeros(hidden_dim)
        self.step = 0

    def _init_weights(self):
        """Initialize recurrent weights with target spectral radius."""
        # Random matrix
        W = np.random.randn(self.hidden_dim, self.hidden_dim) / np.sqrt(self.hidden_dim)

        # Scale to target spectral radius (controls criticality)
        eigenvalues = np.linalg.eigvals(W)
        current_rho = np.max(np.abs(eigenvalues))
        self.W_h = W * (self.target_rho / current_rho)

        # Input weights
        self.W_x = np.random.randn(self.hidden_dim, self.hidden_dim) * 0.1

        logger.info(f"Initialized core: target_ρ={self.target_rho:.2f}, "
                    f"actual_ρ={np.max(np.abs(np.linalg.eigvals(self.W_h))):.3f}")

    def forward(self, x: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Single forward step of cognition.

        Returns delta_h for avalanche detection.
        """
        self.h_prev = self.h.copy()

        # Input (or random perturbation)
        if x is None:
            x = np.random.randn(self.hidden_dim) * self.noise_scale

        # Recurrent dynamics with temperature-scaled noise
        noise = np.random.randn(self.hidden_dim) * self.noise_scale * self.temperature
        pre_activation = self.W_h @ self.h + self.W_x @ x + noise

        # Activation with damping for stability
        self.h = (1 - self.damping) * np.tanh(pre_activation) + self.damping * self.h_prev

        self.step += 1
        return self.h - self.h_prev

    def get_signal(self) -> float:
        """Get scalar signal for avalanche detection (norm of delta-h)."""
        return float(np.linalg.norm(self.h - self.h_prev))

    def reset(self):
        """Reset hidden state."""
        self.h = np.zeros(self.hidden_dim)
        self.h_prev = np.zeros(self.hidden_dim)
        self.step = 0


# =============================================================================
# Avalanche Logger
# =============================================================================

@dataclass
class Avalanche:
    """Single detected avalanche event."""
    start_step: int
    end_step: int
    size: int           # Number of above-threshold steps
    total_activity: float  # Sum of signals during avalanche
    peak_activity: float   # Max signal during avalanche

    @property
    def duration(self) -> int:
        return self.end_step - self.start_step


class AvalancheLogger:
    """
    Detects and logs neural avalanches from continuous signal.

    An avalanche is a contiguous sequence of steps where signal > threshold.
    """

    def __init__(
        self,
        threshold_factor: float = 0.1,
        warmup_steps: int = 100,
    ):
        self.threshold_factor = threshold_factor
        self.warmup_steps = warmup_steps

        # Signal buffer for threshold estimation
        self.signal_buffer: List[float] = []
        self.threshold: Optional[float] = None

        # Avalanche detection state
        self.in_avalanche = False
        self.current_avalanche_start = 0
        self.current_avalanche_signals: List[float] = []

        # Recorded avalanches
        self.avalanches: List[Avalanche] = []
        self.all_signals: List[float] = []

    def log_step(self, step: int, signal: float) -> Optional[Avalanche]:
        """
        Log a signal value and detect avalanches.

        Returns completed Avalanche if one just ended, else None.
        """
        self.all_signals.append(signal)

        # Warmup: collect signals to estimate threshold
        if len(self.signal_buffer) < self.warmup_steps:
            self.signal_buffer.append(signal)
            if len(self.signal_buffer) == self.warmup_steps:
                std = np.std(self.signal_buffer)
                self.threshold = self.threshold_factor * std
                logger.info(f"Avalanche threshold set: {self.threshold:.4f} "
                            f"(factor={self.threshold_factor}, std={std:.4f})")
            return None

        # Avalanche detection
        above_threshold = signal > self.threshold

        if above_threshold:
            if not self.in_avalanche:
                # Start new avalanche
                self.in_avalanche = True
                self.current_avalanche_start = step
                self.current_avalanche_signals = [signal]
            else:
                # Continue avalanche
                self.current_avalanche_signals.append(signal)
        else:
            if self.in_avalanche:
                # End avalanche
                avalanche = Avalanche(
                    start_step=self.current_avalanche_start,
                    end_step=step,
                    size=len(self.current_avalanche_signals),
                    total_activity=sum(self.current_avalanche_signals),
                    peak_activity=max(self.current_avalanche_signals),
                )
                self.avalanches.append(avalanche)
                self.in_avalanche = False
                self.current_avalanche_signals = []
                return avalanche

        return None

    def get_sizes(self) -> List[int]:
        """Get list of avalanche sizes."""
        return [a.size for a in self.avalanches]

    def get_durations(self) -> List[int]:
        """Get list of avalanche durations."""
        return [a.duration for a in self.avalanches]

    def clear(self):
        """Clear all recorded data."""
        self.signal_buffer.clear()
        self.threshold = None
        self.in_avalanche = False
        self.avalanches.clear()
        self.all_signals.clear()


# =============================================================================
# Power-Law Fitting
# =============================================================================

@dataclass
class PowerLawFit:
    """Results of power-law fitting."""
    alpha: float            # Size exponent
    alpha_err: float        # Uncertainty in alpha
    beta: float             # Duration exponent (if computed)
    xmin: int               # Minimum value for fit
    n_tail: int             # Number of points in tail
    ks_statistic: float     # Kolmogorov-Smirnov statistic
    regime: str             # SUBCRITICAL / NEAR_CRITICAL / SUPERCRITICAL
    scaling_check: float    # (α-1)/(β-1) should ≈ 2 for mean-field

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def fit_powerlaw(sizes: List[int], xmin: int = 1) -> PowerLawFit:
    """
    Fit power-law distribution to avalanche sizes.

    Uses maximum likelihood estimation for discrete power-law.
    P(S) ~ S^(-α)

    Args:
        sizes: List of avalanche sizes
        xmin: Minimum size for fitting (excludes small avalanches)

    Returns:
        PowerLawFit with estimated exponent and diagnostics
    """
    # Filter to tail
    tail = [s for s in sizes if s >= xmin]
    n = len(tail)

    if n < 10:
        return PowerLawFit(
            alpha=float('nan'),
            alpha_err=float('nan'),
            beta=float('nan'),
            xmin=xmin,
            n_tail=n,
            ks_statistic=float('nan'),
            regime="INSUFFICIENT_DATA",
            scaling_check=float('nan'),
        )

    # MLE for discrete power-law: α = 1 + n / Σ ln(s_i / (xmin - 0.5))
    log_sum = sum(math.log(s / (xmin - 0.5)) for s in tail)
    alpha = 1 + n / log_sum
    alpha_err = (alpha - 1) / math.sqrt(n)  # Approximate standard error

    # Simple KS test (comparing empirical CDF to theoretical)
    tail_sorted = sorted(tail)
    empirical_cdf = np.arange(1, n + 1) / n
    theoretical_cdf = 1 - (np.array(tail_sorted) / xmin) ** (1 - alpha)
    ks_statistic = float(np.max(np.abs(empirical_cdf - theoretical_cdf)))

    # Estimate duration exponent (β) from duration data if available
    # For now, use theoretical relation: β ≈ α for mean-field
    beta = alpha  # Placeholder

    # Scaling relation check: (α-1)/(β-1) should ≈ 2 for mean-field universality
    if beta > 1:
        scaling_check = (alpha - 1) / (beta - 1)
    else:
        scaling_check = float('nan')

    # Determine regime
    if alpha > 2.5:
        regime = "SUBCRITICAL"
    elif alpha > 1.8:
        regime = "NEAR_CRITICAL"
    elif alpha > 1.2:
        regime = "SUPERCRITICAL"
    else:
        regime = "CHAOTIC"

    return PowerLawFit(
        alpha=alpha,
        alpha_err=alpha_err,
        beta=beta,
        xmin=xmin,
        n_tail=n,
        ks_statistic=ks_statistic,
        regime=regime,
        scaling_check=scaling_check,
    )


# =============================================================================
# HUD Integration
# =============================================================================

def update_hud(
    step: int,
    rho: float,
    alpha: float,
    regime: str,
    signal: float,
    avalanche_sizes: List[int],
    condition: str,
):
    """Update the Cognitive Cockpit HUD with current state."""
    try:
        from hud.cognitive_cockpit import get_cognitive_telemetry

        telemetry = get_cognitive_telemetry()

        # Update criticality
        telemetry.update_criticality(
            rho=rho,
            tau=alpha,  # Using α as τ for display
            avalanche_count=len(avalanche_sizes),
            largest_avalanche=max(avalanche_sizes) if avalanche_sizes else 0,
        )

        # Add avalanche data for scope
        if avalanche_sizes:
            from collections import Counter
            counts = Counter(avalanche_sizes[-100:])  # Last 100
            total = sum(counts.values())
            for size, count in counts.items():
                if size > 0:
                    telemetry.add_avalanche(size, count / total)

        # Update avalanche fit
        cascade_state = {
            "SUBCRITICAL": "fragmented",
            "NEAR_CRITICAL": "stable",
            "SUPERCRITICAL": "runaway",
            "CHAOTIC": "runaway",
        }.get(regime, "stable")

        telemetry.update_avalanche_fit(
            tau=alpha,
            r_squared=0.9,  # Placeholder
            cascade_state=cascade_state,
        )

        # Emit update
        telemetry.emit(step=step)

    except ImportError:
        pass  # HUD not available


# =============================================================================
# Experiment Runner
# =============================================================================

@dataclass
class ConditionResults:
    """Results from a single condition."""
    condition: str
    config: ConditionConfig
    steps: int
    avalanches: List[Avalanche]
    fit: PowerLawFit
    signals: List[float]
    runtime_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "condition": self.condition,
            "config": asdict(self.config),
            "steps": self.steps,
            "n_avalanches": len(self.avalanches),
            "fit": self.fit.to_dict(),
            "runtime_seconds": self.runtime_seconds,
            "avalanche_sizes": [a.size for a in self.avalanches],
            "avalanche_durations": [a.duration for a in self.avalanches],
        }


class ExperimentRunner:
    """Runs the three-condition avalanche experiment."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results: Dict[str, ConditionResults] = {}

        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def run_condition(self, condition: Condition) -> ConditionResults:
        """Run a single experimental condition."""
        cond_config = ConditionConfig.from_condition(condition)
        logger.info(f"\n{'='*60}")
        logger.info(f"Running condition: {cond_config.name}")
        logger.info(f"Description: {cond_config.description}")
        logger.info(f"Target ρ: {cond_config.target_rho}")
        logger.info(f"{'='*60}")

        # Initialize components
        core = SimulatedCognitionCore(
            hidden_dim=self.config.hidden_dim,
            target_rho=cond_config.target_rho,
            temperature=cond_config.temperature,
            noise_scale=cond_config.noise_scale,
            damping=cond_config.damping,
        )
        avalanche_logger = AvalancheLogger(
            threshold_factor=self.config.avalanche_threshold_factor,
        )

        start_time = time.time()
        all_sizes: List[int] = []

        # Run simulation
        for step in range(self.config.steps_per_condition):
            # Optional structured input (sine-modulated for pattern)
            if step % 100 < 50:
                x = np.sin(2 * np.pi * step / 50) * np.ones(self.config.hidden_dim) * 0.1
            else:
                x = None

            # Forward step
            delta_h = core.forward(x)
            signal = core.get_signal()

            # Log avalanche
            completed = avalanche_logger.log_step(step, signal)
            if completed:
                all_sizes.append(completed.size)

            # Update HUD
            if self.config.enable_hud and step % self.config.hud_update_interval == 0:
                # Fit current data
                if len(all_sizes) >= 10:
                    current_fit = fit_powerlaw(all_sizes)
                    update_hud(
                        step=step,
                        rho=cond_config.target_rho,
                        alpha=current_fit.alpha,
                        regime=current_fit.regime,
                        signal=signal,
                        avalanche_sizes=all_sizes,
                        condition=cond_config.name,
                    )

            # Progress
            if step % 200 == 0:
                logger.info(f"  Step {step}/{self.config.steps_per_condition}, "
                            f"avalanches: {len(avalanche_logger.avalanches)}")

        runtime = time.time() - start_time

        # Final fit
        final_fit = fit_powerlaw(all_sizes) if all_sizes else PowerLawFit(
            alpha=float('nan'), alpha_err=float('nan'), beta=float('nan'),
            xmin=1, n_tail=0, ks_statistic=float('nan'),
            regime="NO_DATA", scaling_check=float('nan')
        )

        results = ConditionResults(
            condition=cond_config.name,
            config=cond_config,
            steps=self.config.steps_per_condition,
            avalanches=avalanche_logger.avalanches,
            fit=final_fit,
            signals=avalanche_logger.all_signals,
            runtime_seconds=runtime,
        )

        # Log results
        logger.info(f"\nResults for {cond_config.name}:")
        logger.info(f"  Avalanches detected: {len(results.avalanches)}")
        logger.info(f"  α (size exponent): {final_fit.alpha:.3f} ± {final_fit.alpha_err:.3f}")
        logger.info(f"  Regime: {final_fit.regime}")
        logger.info(f"  KS statistic: {final_fit.ks_statistic:.4f}")
        logger.info(f"  Runtime: {runtime:.2f}s")

        return results

    def run_all(self) -> Dict[str, ConditionResults]:
        """Run all configured conditions."""
        logger.info(f"\nEXP-001: Avalanche Criticality Experiment")
        logger.info(f"Conditions: {[c.value for c in self.config.conditions]}")
        logger.info(f"Steps per condition: {self.config.steps_per_condition}")
        logger.info(f"Output: {self.config.output_dir}")

        for condition in self.config.conditions:
            results = self.run_condition(condition)
            self.results[condition.value] = results

        self.save_results()
        self.print_summary()

        return self.results

    def save_results(self):
        """Save results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON summary
        summary = {
            "experiment": "EXP-001",
            "timestamp": timestamp,
            "config": {
                "steps_per_condition": self.config.steps_per_condition,
                "hidden_dim": self.config.hidden_dim,
                "avalanche_threshold_factor": self.config.avalanche_threshold_factor,
            },
            "results": {k: v.to_dict() for k, v in self.results.items()},
        }

        summary_path = self.config.output_dir / f"exp001_summary_{timestamp}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Saved summary: {summary_path}")

        # Save CSV for each condition
        for cond_name, results in self.results.items():
            csv_path = self.config.output_dir / f"exp001_{cond_name.lower()}_{timestamp}.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "signal"])
                for i, signal in enumerate(results.signals):
                    writer.writerow([i, signal])
            logger.info(f"Saved signals: {csv_path}")

    def print_summary(self):
        """Print comparative summary."""
        logger.info("\n" + "="*70)
        logger.info("EXP-001 SUMMARY: Three-Condition Avalanche Criticality")
        logger.info("="*70)
        logger.info(f"{'Condition':<12} {'ρ_target':<10} {'α':<12} {'Regime':<15} {'N_aval':<10}")
        logger.info("-"*70)

        for cond_name, results in self.results.items():
            logger.info(
                f"{cond_name:<12} "
                f"{results.config.target_rho:<10.2f} "
                f"{results.fit.alpha:<12.3f} "
                f"{results.fit.regime:<15} "
                f"{len(results.avalanches):<10}"
            )

        logger.info("-"*70)
        logger.info("\nGUTC Interpretation:")
        logger.info("  - COLD (α >> 2): Low capacity, high robustness, no long-range correlations")
        logger.info("  - TARGET (α ≈ 2-3): Near-critical, optimal capacity-robustness trade-off")
        logger.info("  - HOT (α < 2): High capacity but unstable, hallucination risk")
        logger.info("\nScaling Relation: (α-1)/(β-1) ≈ 2 indicates mean-field universality class")


def main():
    parser = argparse.ArgumentParser(description="EXP-001: Avalanche Criticality Experiment")
    parser.add_argument("--condition", type=str, default="all",
                        choices=["cold", "target", "hot", "all"],
                        help="Which condition(s) to run")
    parser.add_argument("--steps", type=int, default=1000,
                        help="Steps per condition")
    parser.add_argument("--output", type=str, default="data/experiments/exp001",
                        help="Output directory")
    parser.add_argument("--no-hud", action="store_true",
                        help="Disable HUD updates")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose logging")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Parse conditions
    if args.condition == "all":
        conditions = list(Condition)
    else:
        conditions = [Condition(args.condition)]

    # Configure experiment
    config = ExperimentConfig(
        conditions=conditions,
        steps_per_condition=args.steps,
        output_dir=Path(args.output),
        enable_hud=not args.no_hud,
    )

    # Run
    runner = ExperimentRunner(config)
    runner.run_all()


if __name__ == "__main__":
    main()
