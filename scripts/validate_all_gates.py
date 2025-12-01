#!/usr/bin/env python
"""
Comprehensive Gate Validation Script

Validates all SNN and TF-A-N gates:
- Parameter reduction >= 97%
- Avg degree <= 0.02*N
- Rank <= 0.02*N
- Sparsity >= 98%
- EPR-CV <= 0.15 (with FDT)

Usage:
    python scripts/validate_all_gates.py --config configs/ci/ci_quick.yaml
    python scripts/validate_all_gates.py --config configs/snn_emu_4096.yaml --verbose
"""

import argparse
import json
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any

from tfan.backends import build_backend
from tfan.snn import (
    verify_all_gates,
    mask_density,
    degree_from_csr,
)


def validate_snn_gates(config: Dict[str, Any], verbose: bool = False) -> Dict[str, bool]:
    """
    Validate SNN gates for a given configuration.

    Args:
        config: Backend configuration
        verbose: Print detailed info

    Returns:
        gates: Dict of gate_name -> passed
    """
    if config.get('backend') != 'snn_emu':
        print("⚠ Skipping SNN gates (not SNN backend)")
        return {}

    print(f"\n{'='*60}")
    print("SNN GATE VALIDATION")
    print(f"{'='*60}")

    # Build backend
    backend = build_backend(config)
    model = backend.model

    # Extract SNN parameters
    if not hasattr(model, 'lif'):
        print("❌ Model does not have 'lif' attribute")
        return {'has_lif': False}

    lif = model.lif
    summary = lif.summary()

    N = summary['N']
    r = summary['rank']
    avg_degree = summary['avg_degree']
    param_reduction_pct = summary['reduction_pct']
    sparsity = summary['sparsity']

    if verbose:
        print(f"\nConfiguration:")
        print(f"  N: {N}")
        print(f"  Rank: {r}")
        print(f"  Avg degree: {avg_degree:.1f}")
        print(f"  Parameter reduction: {param_reduction_pct:.2f}%")
        print(f"  Sparsity: {sparsity:.2%}")

    # Define gates
    gates = {
        'param_reduction >= 97.0%': param_reduction_pct >= 97.0,
        'avg_degree <= 0.02*N': avg_degree <= 0.02 * N,
        'rank <= 0.02*N': r <= 0.02 * N,
        'sparsity >= 0.98': sparsity >= 0.98,
    }

    print(f"\nGate Results:")
    for gate_name, passed in gates.items():
        status = '✓ PASS' if passed else '✗ FAIL'
        print(f"  {status}: {gate_name}")

    if all(gates.values()):
        print(f"\n✅ All SNN gates passed!")
    else:
        print(f"\n❌ Some SNN gates failed!")

    print(f"{'='*60}\n")

    return gates


def validate_fdt_gates(
    config: Dict[str, Any],
    num_steps: int = 500,
    verbose: bool = False
) -> Dict[str, bool]:
    """
    Validate FDT gates by running short training loop.

    Args:
        config: Backend configuration
        num_steps: Number of training steps
        verbose: Print detailed info

    Returns:
        gates: Dict of gate_name -> passed
    """
    use_fdt = config.get('tfan', {}).get('use_fdt', False)

    if not use_fdt:
        print("⚠ Skipping FDT gates (FDT disabled)")
        return {}

    print(f"\n{'='*60}")
    print("FDT GATE VALIDATION")
    print(f"{'='*60}")

    # Build backend
    backend = build_backend(config)
    model = backend.model
    optimizer = backend.optim
    hooks = backend.hooks

    # Move to CPU for testing
    backend.to_device('cpu')

    # Create dummy data
    if hasattr(model, 'lif'):
        N = model.lif.N
    else:
        N = config.get('model', {}).get('N', 4096)

    inputs = torch.randn(100, N)
    labels = torch.randint(0, 2, (100, N)).float()

    # Training loop
    epr_cvs = []
    batch_size = 2

    for step in range(num_steps):
        # Sample batch
        idx = torch.randint(0, 100, (batch_size,))
        batch_input = inputs[idx]
        batch_labels = labels[idx]

        # Forward
        output, aux = model(batch_input)

        # Loss
        loss = ((output - batch_labels) ** 2).mean()
        aux['loss'] = loss.item()

        # Backward
        loss.backward()

        # Hooks
        hooks.before_step(model)
        optimizer.step()
        optimizer.zero_grad()
        hooks.after_step(model, aux)

        # Collect EPR-CV
        if 'epr_cv' in aux:
            epr_cvs.append(aux['epr_cv'])

        if verbose and step % 100 == 0:
            epr_cv = aux.get('epr_cv', None)
            if epr_cv is not None:
                print(f"  Step {step}: EPR-CV={epr_cv:.4f}")

    # Compute final EPR-CV
    if not epr_cvs:
        print("⚠ No EPR-CV values collected (FDT may not be active)")
        return {'epr_cv_available': False}

    final_epr_cv = np.mean(epr_cvs[-100:])

    if verbose:
        print(f"\nFinal EPR-CV: {final_epr_cv:.4f}")

    # Define gates
    target_epr_cv = config.get('tfan', {}).get('fdt', {}).get('target_epr_cv', 0.15)
    gates = {
        f'epr_cv <= {target_epr_cv}': final_epr_cv <= target_epr_cv,
    }

    print(f"\nGate Results:")
    for gate_name, passed in gates.items():
        status = '✓ PASS' if passed else '✗ FAIL'
        print(f"  {status}: {gate_name}")

    if all(gates.values()):
        print(f"\n✅ All FDT gates passed!")
    else:
        print(f"\n❌ Some FDT gates failed!")

    print(f"{'='*60}\n")

    return gates


def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config


def main():
    parser = argparse.ArgumentParser(description="Validate all gates")
    parser.add_argument('--config', type=str, required=True, help="Path to config YAML")
    parser.add_argument('--fdt-steps', type=int, default=500, help="Steps for FDT validation")
    parser.add_argument('--verbose', action='store_true', help="Verbose output")
    parser.add_argument('--output', type=str, default=None, help="Output JSON path")

    args = parser.parse_args()

    # Load config
    try:
        config = load_config_from_yaml(args.config)
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return 1

    print(f"Validating gates for config: {args.config}")
    print(f"Backend: {config.get('backend', 'unknown')}")

    # Validate SNN gates
    snn_gates = validate_snn_gates(config, verbose=args.verbose)

    # Validate FDT gates
    fdt_gates = validate_fdt_gates(config, num_steps=args.fdt_steps, verbose=args.verbose)

    # Combine results
    all_gates = {**snn_gates, **fdt_gates}

    # Summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")

    total_gates = len(all_gates)
    passed_gates = sum(1 for passed in all_gates.values() if passed)

    print(f"Total gates: {total_gates}")
    print(f"Passed: {passed_gates}")
    print(f"Failed: {total_gates - passed_gates}")

    if all(all_gates.values()):
        print(f"\n✅ All {total_gates} gates passed!")
        exit_code = 0
    else:
        print(f"\n❌ {total_gates - passed_gates} gate(s) failed!")
        exit_code = 1

    print(f"{'='*60}\n")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results = {
            'config_path': args.config,
            'backend': config.get('backend'),
            'gates': {k: bool(v) for k, v in all_gates.items()},
            'total_gates': total_gates,
            'passed_gates': passed_gates,
            'all_passed': all(all_gates.values()),
        }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to {args.output}\n")

    return exit_code


if __name__ == '__main__':
    exit(main())
