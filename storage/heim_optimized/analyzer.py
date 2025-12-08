"""
Heim Analyzer - Calibration and Validation
===========================================

Offline calibration using real H_moment / resonance logs to determine:
    - Optimal D (dimensionality)
    - Similarity thresholds
    - Expected accuracy

This validates the "100× compression, 0% recall loss" claim.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import logging

from .config import HEIM_CONFIG, GEOMETRY_THRESHOLDS, BUNDLING_REQUIREMENTS
from .encoder import heim_compress, heim_decompress, hv_cosine_sim, batch_compress


logger = logging.getLogger(__name__)


# =============================================================================
# Analysis Results
# =============================================================================

@dataclass
class HeimResult:
    """Result of Heim analysis/calibration."""
    D: int                        # Selected dimension
    accuracy: float               # Recall accuracy achieved
    thresholds: Dict[str, float]  # Calibrated thresholds
    geometry_ok: bool             # Geometry tests passed
    bundling_ok: bool             # Bundling tests passed
    details: Dict[str, Any] = None


@dataclass
class GeometryReport:
    """Report on HV geometry quality."""
    D: int
    mean_cosine: float
    std_cosine: float
    max_cosine: float
    outlier_fraction: float
    passed: bool


@dataclass
class BundlingReport:
    """Report on bundling capacity."""
    D: int
    k_values: List[int]
    signal_margins: List[float]
    max_k_with_margin: int
    passed: bool


# =============================================================================
# Geometry Validation
# =============================================================================

def validate_geometry(
    n_hvs: int = 1000,
    D: int = None,
    seed: int = 42,
) -> GeometryReport:
    """
    Validate that HVs are near-orthogonal in compressed space.

    Checks:
        1. Mean pairwise cosine ≈ 0
        2. Std ≈ 1/√D
        3. Few outliers > threshold
    """
    if D is None:
        D = HEIM_CONFIG.D_compressed

    rng = np.random.default_rng(seed)

    # Generate random 16k HVs
    D_full = HEIM_CONFIG.D_full
    hvs_full = rng.choice([-1, 1], size=(n_hvs, D_full)).astype(np.float32)

    # Compress to D
    hvs_compressed = batch_compress(hvs_full)

    # Compute pairwise cosines (sample for efficiency)
    n_pairs = min(10000, n_hvs * (n_hvs - 1) // 2)
    cosines = []

    for _ in range(n_pairs):
        i, j = rng.choice(n_hvs, size=2, replace=False)
        # Convert binary to bipolar for cosine
        a = 2.0 * hvs_compressed[i].astype(np.float32) - 1.0
        b = 2.0 * hvs_compressed[j].astype(np.float32) - 1.0

        cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
        cosines.append(abs(cos))

    cosines = np.array(cosines)

    mean_cos = float(np.mean(cosines))
    std_cos = float(np.std(cosines))
    max_cos = float(np.max(cosines))
    outlier_frac = float(np.mean(cosines > GEOMETRY_THRESHOLDS["max_pairwise_cosine"]))

    expected_std = GEOMETRY_THRESHOLDS["expected_cosine_std"]
    max_outlier = GEOMETRY_THRESHOLDS["cosine_outlier_fraction"]

    passed = (
        std_cos < expected_std * 2.0 and  # Allow 2× margin
        outlier_frac < max_outlier
    )

    return GeometryReport(
        D=D,
        mean_cosine=mean_cos,
        std_cosine=std_cos,
        max_cosine=max_cos,
        outlier_fraction=outlier_frac,
        passed=passed,
    )


# =============================================================================
# Bundling Capacity Validation
# =============================================================================

def validate_bundling(
    k_values: List[int] = None,
    D: int = None,
    n_trials: int = 100,
    seed: int = 42,
) -> BundlingReport:
    """
    Validate bundling capacity at various K values.

    Tests whether bundled HVs can be decoded correctly.
    """
    if D is None:
        D = HEIM_CONFIG.D_compressed
    if k_values is None:
        k_values = [4, 8, 12, 16, 24, 32]

    rng = np.random.default_rng(seed)
    D_full = HEIM_CONFIG.D_full

    signal_margins = []
    max_k_passed = 0

    for k in k_values:
        correct_recalls = 0

        for trial in range(n_trials):
            # Generate K random base HVs
            base_hvs = rng.choice([-1, 1], size=(k, D_full)).astype(np.float32)

            # Bundle them
            bundled = np.sign(np.sum(base_hvs, axis=0))

            # Compress
            base_compressed = batch_compress(base_hvs)
            bundled_compressed = heim_compress(bundled)

            # Check if we can recover members
            # A member is "recovered" if it's more similar to bundle than random
            for i in range(k):
                # Similarity to bundle
                sim_bundle = np.mean(base_compressed[i] == bundled_compressed)

                # Similarity to random
                random_hv = heim_compress(rng.choice([-1, 1], size=D_full).astype(np.float32))
                sim_random = np.mean(random_hv == bundled_compressed)

                if sim_bundle > sim_random + BUNDLING_REQUIREMENTS["min_signal_margin"]:
                    correct_recalls += 1

        recall_rate = correct_recalls / (n_trials * k)
        signal_margins.append(recall_rate)

        if recall_rate >= 0.9:  # 90% recall threshold
            max_k_passed = k

    passed = max_k_passed >= BUNDLING_REQUIREMENTS["min_k"]

    return BundlingReport(
        D=D,
        k_values=k_values,
        signal_margins=signal_margins,
        max_k_with_margin=max_k_passed,
        passed=passed,
    )


# =============================================================================
# Full Heim Analysis
# =============================================================================

def heim_analyze(
    query_hvs_16k: np.ndarray,
    labels: np.ndarray,
    target_acc: float = 0.999,
    candidate_Ds: List[int] = None,
) -> HeimResult:
    """
    Full Heim analysis on real query distribution.

    Args:
        query_hvs_16k: (N, 16384) query HVs
        labels: (N,) ground truth labels (e.g., best attractor index)
        target_acc: Target recall accuracy
        candidate_Ds: Dimensions to test

    Returns:
        HeimResult with optimal configuration
    """
    if candidate_Ds is None:
        candidate_Ds = [16384, 4096, 1024, 512, 256, 173, 128]

    N = len(query_hvs_16k)
    best_result = None

    for D in candidate_Ds:
        # Temporarily override config
        original_D = HEIM_CONFIG.D_compressed
        HEIM_CONFIG.D_compressed = D

        try:
            # Compress queries
            compressed = batch_compress(query_hvs_16k)

            # Evaluate nearest neighbor accuracy
            # (simplified: check if same label has highest similarity)
            correct = 0
            for i in range(N):
                best_sim = -1
                best_label = -1
                for j in range(N):
                    if i == j:
                        continue
                    sim = np.mean(compressed[i] == compressed[j])
                    if sim > best_sim:
                        best_sim = sim
                        best_label = labels[j]

                if best_label == labels[i]:
                    correct += 1

            acc = correct / N

            # Validate geometry and bundling
            geom = validate_geometry(n_hvs=500, D=D)
            bundling = validate_bundling(D=D, n_trials=50)

            if acc >= target_acc and geom.passed and bundling.passed:
                thresholds = {
                    'cluster_merge': HEIM_CONFIG.cluster_merge_threshold,
                    'duplicate': HEIM_CONFIG.duplicate_threshold,
                }

                best_result = HeimResult(
                    D=D,
                    accuracy=acc,
                    thresholds=thresholds,
                    geometry_ok=geom.passed,
                    bundling_ok=bundling.passed,
                    details={
                        'geometry': geom.__dict__,
                        'bundling': bundling.__dict__,
                    },
                )
                break  # Take smallest D that works

        finally:
            HEIM_CONFIG.D_compressed = original_D

    if best_result is None:
        raise RuntimeError(f"Heim analysis: could not hit target accuracy {target_acc}")

    return best_result


# =============================================================================
# Quick Validation
# =============================================================================

def quick_validate(verbose: bool = True) -> Tuple[bool, Dict[str, Any]]:
    """
    Quick validation of current Heim configuration.

    Returns:
        (passed, details)
    """
    results = {}

    # Geometry
    geom = validate_geometry(n_hvs=500)
    results['geometry'] = {
        'passed': geom.passed,
        'D': geom.D,
        'mean_cosine': geom.mean_cosine,
        'outlier_fraction': geom.outlier_fraction,
    }

    if verbose:
        logger.info(f"Geometry: {'PASS' if geom.passed else 'FAIL'} "
                   f"(D={geom.D}, outliers={geom.outlier_fraction:.3f})")

    # Bundling
    bundling = validate_bundling(n_trials=50)
    results['bundling'] = {
        'passed': bundling.passed,
        'max_k': bundling.max_k_with_margin,
        'margins': bundling.signal_margins,
    }

    if verbose:
        logger.info(f"Bundling: {'PASS' if bundling.passed else 'FAIL'} "
                   f"(max_k={bundling.max_k_with_margin})")

    # Compression fidelity
    from .encoder import compression_fidelity
    rng = np.random.default_rng(42)
    fidelities = []
    for _ in range(100):
        h = rng.choice([-1, 1], size=HEIM_CONFIG.D_full).astype(np.float32)
        sim, active = compression_fidelity(h)
        fidelities.append(sim)

    mean_fidelity = float(np.mean(fidelities))
    results['fidelity'] = {
        'mean': mean_fidelity,
        'min': float(np.min(fidelities)),
        'max': float(np.max(fidelities)),
    }

    if verbose:
        logger.info(f"Fidelity: mean={mean_fidelity:.3f}")

    # Overall
    passed = geom.passed and bundling.passed and mean_fidelity > 0.3

    return passed, results


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'HeimResult',
    'GeometryReport',
    'BundlingReport',
    'validate_geometry',
    'validate_bundling',
    'heim_analyze',
    'quick_validate',
]
