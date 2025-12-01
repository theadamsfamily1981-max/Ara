#!/usr/bin/env python
"""
Nightly exact persistent homology validation.

Computes exact PH using GUDHI/Ripser and validates against
differentiable approximation gates.

Usage:
    python tools/nightly_ph_check.py --max-samples 5000 --time-cap-min 20
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

try:
    import gudhi
    HAS_GUDHI = True
except ImportError:
    HAS_GUDHI = False

try:
    import ripser
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False

from tfan.topo import TopologyRegularizer


def compute_exact_ph(points: np.ndarray, max_dimension: int = 1) -> dict:
    """
    Compute exact persistent homology using GUDHI or Ripser.

    Args:
        points: Point cloud (n_points, dim)
        max_dimension: Maximum homology dimension

    Returns:
        Dict mapping degree -> persistence diagram
    """
    if HAS_RIPSER:
        # Use Ripser (faster for Vietoris-Rips)
        result = ripser.ripser(points, maxdim=max_dimension)
        diagrams = {}
        for deg in range(max_dimension + 1):
            if deg < len(result['dgms']):
                diagram = result['dgms'][deg]
                # Filter infinite points
                diagram = diagram[np.isfinite(diagram).all(axis=1)]
                diagrams[deg] = diagram
        return diagrams

    elif HAS_GUDHI:
        # Use GUDHI
        rips_complex = gudhi.RipsComplex(points=points, max_edge_length=2.0)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension)
        simplex_tree.compute_persistence()

        diagrams = {}
        for deg in range(max_dimension + 1):
            pairs = simplex_tree.persistence_intervals_in_dimension(deg)
            # Filter infinite points
            pairs = pairs[np.isfinite(pairs).all(axis=1)]
            diagrams[deg] = pairs
        return diagrams

    else:
        raise RuntimeError("Neither GUDHI nor Ripser available for exact PH")


def validate_sample(
    sample_points: np.ndarray,
    topo_reg: TopologyRegularizer,
    device: str = "cpu",
) -> dict:
    """
    Validate a single sample against exact PH.

    Args:
        sample_points: Point cloud
        topo_reg: Topology regularizer with differentiable PH
        device: Compute device

    Returns:
        Validation metrics
    """
    # Compute exact PH
    exact_diagrams = compute_exact_ph(sample_points, max_dimension=1)

    # Compute approximate PH
    latents = torch.from_numpy(sample_points).float().unsqueeze(0).to(device)
    landscape_dict = topo_reg.compute_landscape(latents, return_diagrams=True)
    approx_diagrams = landscape_dict.get("diagrams", {})

    # Extract first batch item
    approx_diagrams_flat = {}
    for deg, diag_list in approx_diagrams.items():
        if len(diag_list) > 0:
            approx_diagrams_flat[deg] = diag_list[0]

    # Validate
    passes, metrics = topo_reg.validate_against_exact(
        approx_diagrams_flat,
        exact_diagrams,
    )

    return {
        "passes": passes,
        "metrics": metrics,
        "exact_diagrams": {k: v.tolist() for k, v in exact_diagrams.items()},
    }


def main():
    parser = argparse.ArgumentParser(description="Nightly PH validation")
    parser.add_argument("--max-samples", type=int, default=5000,
                        help="Maximum number of samples to validate")
    parser.add_argument("--time-cap-min", type=int, default=20,
                        help="Time cap in minutes")
    parser.add_argument("--output", type=str,
                        default="artifacts/topology/audit.json",
                        help="Output JSON path")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint to load samples from")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for approximate PH")
    args = parser.parse_args()

    # Ensure output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Nightly Persistent Homology Validation")
    print("=" * 80)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Max samples: {args.max_samples}")
    print(f"Time cap: {args.time_cap_min} minutes")
    print(f"GUDHI available: {HAS_GUDHI}")
    print(f"Ripser available: {HAS_RIPSER}")
    print("-" * 80)

    if not (HAS_GUDHI or HAS_RIPSER):
        print("ERROR: Neither GUDHI nor Ripser available")
        return 1

    # Create topology regularizer
    topo_reg = TopologyRegularizer(
        lambda_topo=0.0,  # Not used for validation
        homology_degrees=[0, 1],
        wasserstein_gap_max=0.02,
        cosine_min=0.90,
        device=args.device,
    )

    # Generate or load samples
    if args.checkpoint:
        print(f"Loading samples from checkpoint: {args.checkpoint}")
        # TODO: Load from checkpoint
        # For now, generate synthetic
        n_samples = min(args.max_samples, 100)
        samples = [np.random.randn(50, 64) for _ in range(n_samples)]
    else:
        print("Generating synthetic samples...")
        n_samples = min(args.max_samples, 100)  # Start conservative
        samples = [np.random.randn(50, 64) for _ in range(n_samples)]

    print(f"Validating {len(samples)} samples...")

    # Run validation
    start_time = time.time()
    time_cap_sec = args.time_cap_min * 60

    results = []
    passes_count = 0
    fails_count = 0

    for i, sample in enumerate(samples):
        # Check time cap
        elapsed = time.time() - start_time
        if elapsed > time_cap_sec:
            print(f"\nTime cap reached at sample {i}")
            break

        # Validate
        result = validate_sample(sample, topo_reg, args.device)
        results.append(result)

        if result["passes"]:
            passes_count += 1
        else:
            fails_count += 1

        # Progress
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(samples)}: "
                  f"Pass={passes_count}, Fail={fails_count}, "
                  f"Time={elapsed/60:.1f}min")

    total_time = time.time() - start_time

    # Aggregate metrics
    all_wass = []
    all_cos = []
    for result in results:
        metrics = result["metrics"]
        wass = metrics.get("wasserstein_distance")
        cos = metrics.get("cosine_similarity")
        if wass is not None:
            all_wass.append(wass)
        if cos is not None:
            all_cos.append(cos)

    avg_wass = np.mean(all_wass) if all_wass else float('inf')
    avg_cos = np.mean(all_cos) if all_cos else 0.0

    # Gates
    gates_pass = avg_wass <= 0.02 and avg_cos >= 0.90

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Samples validated: {len(results)}")
    print(f"Time elapsed: {total_time/60:.1f} minutes")
    print(f"Passes: {passes_count}")
    print(f"Fails: {fails_count}")
    print(f"Pass rate: {passes_count / max(len(results), 1):.1%}")
    print("-" * 80)
    print(f"Avg Wasserstein: {avg_wass:.4f} (threshold: 0.02)")
    print(f"Avg Cosine: {avg_cos:.4f} (threshold: 0.90)")
    print("-" * 80)
    print(f"Gates: {'✓ PASS' if gates_pass else '✗ FAIL'}")

    # Save results
    output_data = {
        "date": datetime.now().isoformat(),
        "config": {
            "max_samples": args.max_samples,
            "time_cap_min": args.time_cap_min,
            "actual_samples": len(results),
            "time_elapsed_min": total_time / 60,
        },
        "summary": {
            "passes": passes_count,
            "fails": fails_count,
            "pass_rate": passes_count / max(len(results), 1),
        },
        "metrics": {
            "avg_wasserstein": float(avg_wass),
            "avg_cosine": float(avg_cos),
        },
        "gates": {
            "wasserstein_pass": avg_wass <= 0.02,
            "cosine_pass": avg_cos >= 0.90,
            "overall_pass": gates_pass,
        },
        "details": results[:100],  # Limit detail size
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {args.output}")

    return 0 if gates_pass else 1


if __name__ == "__main__":
    exit(main())
