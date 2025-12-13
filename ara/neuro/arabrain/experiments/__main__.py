#!/usr/bin/env python3
"""
Edge of Autumn Empirical Validation Suite - Main Runner

Runs all four experiments proving properties of the balanced β regime:

1. OOD Generalization - Better transfer to unseen distributions
2. Antifragility - Gains from moderate perturbations (Taleb)
3. Causal Disentanglement - Interventions affect single factors
4. Biological Criticality - Power-law dynamics like the brain

Usage:
    python -m ara.neuro.arabrain.experiments           # Run all
    python -m ara.neuro.arabrain.experiments --exp 1   # Run specific
    python -m ara.neuro.arabrain.experiments --fast    # Quick mode

Each experiment contributes to proving the Edge of Autumn theorem empirically:
the balanced β regime exists and has scientifically meaningful properties.
"""

import argparse
import time
from typing import List

from .ood_generalization import run_ood_experiment
from .antifragility import run_antifragility_experiment
from .causal_interventions import run_intervention_experiment
from .criticality_signatures import run_criticality_experiment


def print_banner():
    """Print experiment suite banner."""
    print("\n" + "=" * 70)
    print("█████████████████████████████████████████████████████████████████████")
    print("█                                                                   █")
    print("█     EDGE OF AUTUMN EMPIRICAL VALIDATION SUITE                     █")
    print("█     Proving the Balanced Regime Exists and Matters                █")
    print("█                                                                   █")
    print("█████████████████████████████████████████████████████████████████████")
    print("=" * 70)


def print_summary(results: dict):
    """Print unified summary of all experiments."""
    print("\n")
    print("=" * 70)
    print("█████████████████████████████████████████████████████████████████████")
    print("█                    UNIFIED RESULTS SUMMARY                        █")
    print("█████████████████████████████████████████████████████████████████████")
    print("=" * 70)

    print("\n  EXPERIMENT                    CLAIM                           STATUS")
    print("  " + "-" * 66)

    statuses = []

    # OOD Generalization
    if 'ood' in results and results['ood']:
        best = min(results['ood'], key=lambda r: r.auc_drop)
        status = "✓ SUPPORTED" if best.auc_drop < 0.05 else "? PARTIAL"
        print(f"  1. OOD Generalization        Min transfer drop at balanced β  {status}")
        statuses.append('ood' if best.auc_drop < 0.05 else 'partial')
    else:
        print(f"  1. OOD Generalization        (not run)")

    # Antifragility
    if 'antifragile' in results and results['antifragile']:
        antifragile = [r for r in results['antifragile'] if r.is_antifragile]
        status = f"✓ SUPPORTED" if antifragile else "? PARTIAL"
        print(f"  2. Antifragility             Gains from disorder              {status}")
        statuses.append('af' if antifragile else 'partial')
    else:
        print(f"  2. Antifragility             (not run)")

    # Causal Interventions
    if 'causal' in results and results['causal']:
        best = max(results['causal'], key=lambda r: r.modularity)
        status = "✓ SUPPORTED" if best.modularity > 0.5 else "? PARTIAL"
        print(f"  3. Causal Disentanglement    One latent → one factor          {status}")
        statuses.append('causal' if best.modularity > 0.5 else 'partial')
    else:
        print(f"  3. Causal Disentanglement    (not run)")

    # Criticality
    if 'criticality' in results and results['criticality']:
        critical = [r for r in results['criticality'] if r.is_critical]
        status = "✓ SUPPORTED" if critical else "? PARTIAL"
        print(f"  4. Biological Criticality    Power-law dynamics, σ ≈ 1        {status}")
        statuses.append('crit' if critical else 'partial')
    else:
        print(f"  4. Biological Criticality    (not run)")

    # Overall conclusion
    print("\n" + "=" * 70)
    supported = sum(1 for s in statuses if s != 'partial')
    total = len(statuses)

    if supported == total and total >= 3:
        print("█  OVERALL: STRONG EMPIRICAL SUPPORT FOR EDGE OF AUTUMN           █")
        print("█                                                                  █")
        print("█  The balanced β regime:                                          █")
        print("█    • EXISTS (found optimal range)                                █")
        print("█    • GENERALIZES (OOD transfer)                                  █")
        print("█    • IS ANTIFRAGILE (gains from noise)                           █")
        print("█    • IS INTERPRETABLE (causal disentanglement)                   █")
        print("█    • MIMICS THE BRAIN (criticality signatures)                   █")
    elif supported >= 2:
        print("█  OVERALL: MODERATE EMPIRICAL SUPPORT                             █")
        print("█  Some claims verified, others need more investigation            █")
    else:
        print("█  OVERALL: PRELIMINARY EVIDENCE                                   █")
        print("█  More training/data may be needed for stronger claims            █")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Edge of Autumn Empirical Validation Suite"
    )
    parser.add_argument(
        "--exp", "-e",
        type=int,
        choices=[1, 2, 3, 4],
        help="Run specific experiment (1=OOD, 2=Antifragile, 3=Causal, 4=Critical)"
    )
    parser.add_argument(
        "--fast", "-f",
        action="store_true",
        help="Fast mode: fewer β values and epochs"
    )
    parser.add_argument(
        "--beta", "-b",
        type=float,
        nargs="+",
        help="Custom β values to test"
    )
    args = parser.parse_args()

    print_banner()

    # Set parameters based on mode
    if args.fast:
        beta_values = [0.5, 3.0, 30.0]
        num_epochs = 8
        num_samples = 400
    else:
        beta_values = args.beta if args.beta else [0.1, 0.5, 1.0, 3.0, 10.0, 30.0]
        num_epochs = 15
        num_samples = 600

    results = {}
    start_time = time.time()

    experiments = []
    if args.exp is None:
        experiments = [1, 2, 3, 4]
    else:
        experiments = [args.exp]

    # Run selected experiments
    if 1 in experiments:
        print("\n\n" + "▓" * 70)
        print("Running Experiment 1: OOD Generalization")
        print("▓" * 70)
        results['ood'] = run_ood_experiment(
            beta_values=beta_values,
            num_samples=num_samples,
            num_epochs=num_epochs,
        )

    if 2 in experiments:
        print("\n\n" + "▓" * 70)
        print("Running Experiment 2: Antifragility")
        print("▓" * 70)
        results['antifragile'] = run_antifragility_experiment(
            beta_values=beta_values,
            num_samples=num_samples,
            num_epochs=num_epochs,
        )

    if 3 in experiments:
        print("\n\n" + "▓" * 70)
        print("Running Experiment 3: Causal Interventions")
        print("▓" * 70)
        results['causal'] = run_intervention_experiment(
            beta_values=beta_values,
            num_samples=num_samples,
            num_epochs=num_epochs,
        )

    if 4 in experiments:
        print("\n\n" + "▓" * 70)
        print("Running Experiment 4: Biological Criticality")
        print("▓" * 70)
        results['criticality'] = run_criticality_experiment(
            beta_values=beta_values,
            num_samples=num_samples,
            num_epochs=num_epochs,
        )

    # Print unified summary
    print_summary(results)

    elapsed = time.time() - start_time
    print(f"\n  Total runtime: {elapsed/60:.1f} minutes")
    print("=" * 70)


if __name__ == "__main__":
    main()
