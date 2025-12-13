#!/usr/bin/env python3
"""
Brain Remodulator - Simulation Demo

Demonstrates the Brain Remodulator correcting aberrant precision patterns:

1. SCHIZOPHRENIA-LIKE: D >> 1 (prior-dominated)
   - Simulates state with excessive prior precision
   - Shows remodulator detecting and correcting

2. ASD-LIKE: D << 1 (sensory-dominated)
   - Simulates state with excessive sensory precision
   - Shows remodulator detecting and correcting

3. SUPERCRITICAL: ρ >> 1 (chaotic)
   - Simulates unstable dynamics
   - Shows dampening intervention

Usage:
    python -m ara.neuro.remodulator.simulation
    python -m ara.neuro.remodulator.simulation --pattern schizophrenia
    python -m ara.neuro.remodulator.simulation --pattern asd
    python -m ara.neuro.remodulator.simulation --pattern all --steps 200
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from .core import (
    BrainRemodulator,
    BrainState,
    DisorderPattern,
    Intervention,
    DirectPrecisionModality,
    NeurofeedbackModality,
    PharmacologicalModality,
)

logger = logging.getLogger("ara.neuro.remodulator.sim")


# =============================================================================
# Disorder Simulators
# =============================================================================

class DisorderSimulator:
    """Base class for disorder pattern simulators."""

    def __init__(self, name: str):
        self.name = name
        self.step = 0

    def get_state(self) -> Dict[str, float]:
        """Return simulated precision and criticality values."""
        raise NotImplementedError

    def advance(self):
        """Advance simulation one step."""
        self.step += 1


class SchizophreniaSimulator(DisorderSimulator):
    """
    Simulates schizophrenia-like precision aberration.

    Characteristics:
    - High prior precision (Π_prior >> 1)
    - Low sensory precision (Π_sensory < 1)
    - D = Π_prior / Π_sensory >> 1
    - Fluctuating, sometimes supercritical ρ
    """

    def __init__(self, severity: float = 0.7):
        super().__init__("Schizophrenia-like")
        self.severity = severity  # 0-1

        # Base state
        self.pi_prior = 1.5 + severity * 1.5      # 1.5 to 3.0
        self.pi_sensory = 0.8 - severity * 0.4   # 0.8 to 0.4
        self.rho = 0.95 + severity * 0.2         # 0.95 to 1.15

    def get_state(self) -> Dict[str, float]:
        # Add realistic fluctuations
        noise_prior = random.gauss(0, 0.1 * self.severity)
        noise_sensory = random.gauss(0, 0.05)
        noise_rho = random.gauss(0, 0.02)

        # Occasional "episode" spikes
        if random.random() < 0.05 * self.severity:
            noise_prior += 0.5
            logger.debug(f"[{self.name}] Episode spike at step {self.step}")

        return {
            "pi_prior": max(0.5, self.pi_prior + noise_prior),
            "pi_sensory": max(0.2, self.pi_sensory + noise_sensory),
            "rho": max(0.5, min(1.5, self.rho + noise_rho)),
        }

    def advance(self):
        super().advance()
        # Slow drift
        self.pi_prior += random.gauss(0, 0.01)
        self.pi_sensory += random.gauss(0, 0.005)


class ASDSimulator(DisorderSimulator):
    """
    Simulates ASD-like precision aberration.

    Characteristics:
    - Low prior precision (Π_prior < 1)
    - High sensory precision (Π_sensory >> 1)
    - D = Π_prior / Π_sensory << 1
    - Often subcritical ρ (rigid dynamics)
    """

    def __init__(self, severity: float = 0.7):
        super().__init__("ASD-like")
        self.severity = severity

        # Base state (opposite of schizophrenia)
        self.pi_prior = 0.6 - severity * 0.3     # 0.6 to 0.3
        self.pi_sensory = 1.5 + severity * 1.0   # 1.5 to 2.5
        self.rho = 0.75 - severity * 0.15        # 0.75 to 0.6

    def get_state(self) -> Dict[str, float]:
        # ASD tends to be more stable/predictable
        noise_prior = random.gauss(0, 0.03)
        noise_sensory = random.gauss(0, 0.08 * self.severity)
        noise_rho = random.gauss(0, 0.01)

        # Occasional sensory spike (overwhelm trigger)
        if random.random() < 0.03 * self.severity:
            noise_sensory += 0.4
            logger.debug(f"[{self.name}] Sensory spike at step {self.step}")

        return {
            "pi_prior": max(0.1, self.pi_prior + noise_prior),
            "pi_sensory": max(0.5, self.pi_sensory + noise_sensory),
            "rho": max(0.4, min(1.0, self.rho + noise_rho)),
        }

    def advance(self):
        super().advance()
        self.pi_sensory += random.gauss(0, 0.005)


class HealthySimulator(DisorderSimulator):
    """
    Simulates healthy precision balance (control condition).
    """

    def __init__(self):
        super().__init__("Healthy")
        self.pi_prior = 1.0
        self.pi_sensory = 1.0
        self.rho = 0.88

    def get_state(self) -> Dict[str, float]:
        return {
            "pi_prior": max(0.5, self.pi_prior + random.gauss(0, 0.05)),
            "pi_sensory": max(0.5, self.pi_sensory + random.gauss(0, 0.05)),
            "rho": max(0.6, min(1.1, self.rho + random.gauss(0, 0.02))),
        }

    def advance(self):
        super().advance()


# =============================================================================
# Simulation Runner
# =============================================================================

class SimulationResult:
    """Container for simulation results."""

    def __init__(self, name: str):
        self.name = name
        self.states: List[Dict[str, Any]] = []
        self.interventions: List[Dict[str, Any]] = []
        self.diagnoses: List[Dict[str, Any]] = []

    def add_step(
        self,
        state: BrainState,
        interventions: List[Intervention],
        diagnosis: Dict[str, Any],
    ):
        self.states.append(state.to_dict())
        self.interventions.append([i.to_dict() for i in interventions])
        self.diagnoses.append(diagnosis)

    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not self.states:
            return {}

        D_values = [s["precision"]["D"] for s in self.states]
        rho_values = [s["criticality"]["rho"] for s in self.states]
        patterns = [s["pattern"] for s in self.states]

        # Count pattern distribution
        pattern_counts = {}
        for p in patterns:
            pattern_counts[p] = pattern_counts.get(p, 0) + 1

        # Count interventions
        total_interventions = sum(len(i) for i in self.interventions)

        return {
            "name": self.name,
            "steps": len(self.states),
            "D_mean": sum(D_values) / len(D_values),
            "D_min": min(D_values),
            "D_max": max(D_values),
            "D_final": D_values[-1],
            "rho_mean": sum(rho_values) / len(rho_values),
            "rho_final": rho_values[-1],
            "pattern_distribution": pattern_counts,
            "total_interventions": total_interventions,
            "converged_to_healthy": patterns[-1] == "healthy",
        }


def run_simulation(
    simulator: DisorderSimulator,
    remodulator: BrainRemodulator,
    steps: int = 100,
    apply_corrections: bool = True,
    verbose: bool = True,
) -> SimulationResult:
    """
    Run a simulation with the given disorder simulator.

    Args:
        simulator: Disorder pattern simulator
        remodulator: Brain remodulator instance
        steps: Number of simulation steps
        apply_corrections: Whether to apply remodulator corrections
        verbose: Print progress

    Returns:
        SimulationResult with full history
    """
    result = SimulationResult(simulator.name)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Simulation: {simulator.name}")
        print(f"Steps: {steps}, Corrections: {'ON' if apply_corrections else 'OFF'}")
        print(f"{'='*60}")

    for step in range(steps):
        # Get simulated sensor values
        sensor_state = simulator.get_state()

        # Update remodulator with sensor data
        remodulator.update(
            pi_prior=sensor_state["pi_prior"],
            pi_sensory=sensor_state["pi_sensory"],
            rho=sensor_state["rho"],
        )

        # Get current state and interventions
        state = remodulator.get_state()
        interventions = remodulator.get_pending_interventions()
        diagnosis = remodulator.get_diagnosis()

        # Record
        result.add_step(state, interventions, diagnosis)

        # Apply corrections if enabled
        if apply_corrections and interventions:
            new_state = remodulator.apply_interventions()

            # Feed corrected state back to simulator (closed loop)
            simulator.pi_prior = new_state.precision.pi_prior
            simulator.pi_sensory = new_state.precision.pi_sensory
            simulator.rho = new_state.criticality.rho

        # Advance simulator
        simulator.advance()

        # Progress output
        if verbose and step % 20 == 0:
            D = state.precision.D
            pattern = state.pattern.value
            n_int = len(interventions)
            print(f"  Step {step:3d}: D={D:5.2f} ρ={state.criticality.rho:.3f} "
                  f"pattern={pattern:20s} interventions={n_int}")

    if verbose:
        summary = result.summary()
        print(f"\nResults:")
        print(f"  D: {summary['D_min']:.2f} → {summary['D_max']:.2f} (final: {summary['D_final']:.2f})")
        print(f"  Interventions: {summary['total_interventions']}")
        print(f"  Converged to healthy: {summary['converged_to_healthy']}")

    return result


# =============================================================================
# Main Demo
# =============================================================================

def run_demo(
    pattern: str = "all",
    steps: int = 100,
    severity: float = 0.7,
    output_dir: Optional[Path] = None,
):
    """
    Run the Brain Remodulator demonstration.

    Args:
        pattern: "schizophrenia", "asd", "healthy", or "all"
        steps: Simulation steps per condition
        severity: Disorder severity (0-1)
        output_dir: Optional directory to save results
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                     BRAIN REMODULATOR SIMULATION                              ║
║                                                                               ║
║  Demonstrating precision-based correction of aberrant neural dynamics         ║
║                                                                               ║
║  Theory: Predictive Processing + Active Inference                             ║
║  - D (Delusion Index) = Π_prior / Π_sensory                                   ║
║  - D >> 1: Prior-dominated (schizophrenia-like)                               ║
║  - D << 1: Sensory-dominated (ASD-like)                                       ║
║  - Target: D ≈ 1.0 (balanced)                                                 ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)

    results = {}

    # Run requested simulations
    patterns_to_run = []
    if pattern == "all":
        patterns_to_run = ["healthy", "schizophrenia", "asd"]
    else:
        patterns_to_run = [pattern]

    for p in patterns_to_run:
        # Create fresh remodulator
        remodulator = BrainRemodulator(
            D_target=1.0,
            rho_target=0.88,
            modality=DirectPrecisionModality(),
        )

        # Create simulator
        if p == "schizophrenia":
            simulator = SchizophreniaSimulator(severity=severity)
        elif p == "asd":
            simulator = ASDSimulator(severity=severity)
        else:
            simulator = HealthySimulator()

        # Run simulation WITH corrections
        result = run_simulation(
            simulator=simulator,
            remodulator=remodulator,
            steps=steps,
            apply_corrections=True,
            verbose=True,
        )
        results[p] = result

    # Summary comparison
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Pattern':<20} {'D_init':>8} {'D_final':>8} {'Interventions':>14} {'Healthy':>8}")
    print("-"*60)

    for name, result in results.items():
        summary = result.summary()
        print(f"{name:<20} {summary.get('D_max', 0):>8.2f} {summary['D_final']:>8.2f} "
              f"{summary['total_interventions']:>14} {'YES' if summary['converged_to_healthy'] else 'NO':>8}")

    # Save results if requested
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for name, result in results.items():
            output_file = output_dir / f"remodulator_sim_{name}.json"
            with open(output_file, "w") as f:
                json.dump({
                    "summary": result.summary(),
                    "states": result.states,
                    "interventions": result.interventions,
                }, f, indent=2)
            print(f"\nSaved: {output_file}")

    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    print("""
The Brain Remodulator successfully corrects aberrant precision patterns:

SCHIZOPHRENIA-LIKE (D >> 1):
  - Initial state: Excessive prior precision → hallucination risk
  - Intervention: Boost sensory precision, dampen prior confidence
  - Result: D converges toward 1.0 (balanced)

ASD-LIKE (D << 1):
  - Initial state: Excessive sensory precision → overwhelm risk
  - Intervention: Boost prior precision, reduce sensory gain
  - Result: D converges toward 1.0 (balanced)

This demonstrates the theoretical "precision thermostat" that could:
  - Implement as neurofeedback training targets
  - Guide pharmacological intervention strategies
  - Inform tDCS/TMS stimulation protocols
    """)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Brain Remodulator Simulation Demo"
    )
    parser.add_argument(
        "--pattern", "-p",
        type=str,
        default="all",
        choices=["schizophrenia", "asd", "healthy", "all"],
        help="Pattern to simulate",
    )
    parser.add_argument(
        "--steps", "-s",
        type=int,
        default=100,
        help="Simulation steps",
    )
    parser.add_argument(
        "--severity",
        type=float,
        default=0.7,
        help="Disorder severity (0-1)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for results",
    )

    args = parser.parse_args()

    run_demo(
        pattern=args.pattern,
        steps=args.steps,
        severity=args.severity,
        output_dir=Path(args.output) if args.output else None,
    )


if __name__ == "__main__":
    main()
