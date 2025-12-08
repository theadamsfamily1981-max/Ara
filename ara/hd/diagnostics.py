"""
Ara HD Diagnostics - Capacity & Interference Testing
====================================================

Test suite for verifying HTC geometry remains healthy.

At D=16,384:
- Random HVs are essentially orthogonal (cos ≈ 0, σ ≈ 0.008)
- |cos| ≥ 0.05-0.1 is meaningful signal
- 2k attractors in 16k space is geometrically safe
- Failure mode is sloppy plasticity, not geometry

This module provides:
1. Base codebook sanity checks
2. Bundling stress tests
3. Attractor diversity monitoring
4. Health threshold enforcement

Usage:
    from ara.hd.diagnostics import (
        check_codebook_geometry,
        stress_test_bundling,
        check_attractor_diversity,
        SoulHealthReport,
    )

    # Check base vocabulary
    report = check_codebook_geometry(vocab)
    assert report.is_healthy

    # Test bundling capacity
    bundling_report = stress_test_bundling(vocab, feature_counts=[8, 16, 32, 64])

    # Monitor attractor health
    diversity_report = check_attractor_diversity(attractors)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import json
import logging

from .ops import DIM, random_hv, bind, bundle, cosine, hamming_similarity
from .vocab import HDVocab, get_vocab

logger = logging.getLogger(__name__)


# =============================================================================
# Health Thresholds (The Contract)
# =============================================================================

@dataclass
class HealthThresholds:
    """
    Capacity & Interference Contract for HTC-16k.

    Any learning rule, sharding scheme, or encoder variant MUST satisfy:
    1. Base codebook with random-like geometry
    2. Stable attractor diversity after training
    3. Robust per-moment bundling
    """
    # Codebook geometry
    codebook_mean_cos_max: float = 0.02      # Mean |cos| should be near 0
    codebook_std_cos_max: float = 0.02       # Std should be ~1/sqrt(D)
    codebook_tail_threshold: float = 0.10    # |cos| above this is concerning
    codebook_tail_fraction_max: float = 0.01 # Max fraction above threshold

    # Attractor diversity
    attractor_mean_cos_max: float = 0.15     # Mean pairwise |cos| after training
    attractor_cluster_threshold: float = 0.4  # Defines "tight cluster"
    attractor_cluster_fraction_max: float = 0.10  # Max 10% in tight clusters
    attractor_usage_min: float = 0.80        # Min 80% rows activated per day

    # Bundling capacity
    max_features_per_moment: int = 50        # Keep bound features ≤ 50
    signal_noise_sigma_min: float = 3.0      # Feature similarity ≥ 3σ above noise

    def to_dict(self) -> Dict[str, Any]:
        return {
            "codebook": {
                "mean_cos_max": self.codebook_mean_cos_max,
                "std_cos_max": self.codebook_std_cos_max,
                "tail_threshold": self.codebook_tail_threshold,
                "tail_fraction_max": self.codebook_tail_fraction_max,
            },
            "attractor": {
                "mean_cos_max": self.attractor_mean_cos_max,
                "cluster_threshold": self.attractor_cluster_threshold,
                "cluster_fraction_max": self.attractor_cluster_fraction_max,
                "usage_min": self.attractor_usage_min,
            },
            "bundling": {
                "max_features_per_moment": self.max_features_per_moment,
                "signal_noise_sigma_min": self.signal_noise_sigma_min,
            },
        }


# Default thresholds
DEFAULT_THRESHOLDS = HealthThresholds()


# =============================================================================
# Diagnostic Reports
# =============================================================================

@dataclass
class CodebookGeometryReport:
    """Report from codebook geometry check."""
    n_samples: int
    n_pairs: int
    mean_cos: float
    std_cos: float
    min_cos: float
    max_cos: float
    tail_fraction: float  # Fraction with |cos| > threshold

    is_healthy: bool = True
    violations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "n_pairs": self.n_pairs,
            "mean_cos": round(self.mean_cos, 6),
            "std_cos": round(self.std_cos, 6),
            "min_cos": round(self.min_cos, 6),
            "max_cos": round(self.max_cos, 6),
            "tail_fraction": round(self.tail_fraction, 6),
            "is_healthy": self.is_healthy,
            "violations": self.violations,
        }


@dataclass
class BundlingStressReport:
    """Report from bundling stress test."""
    feature_counts: List[int]
    signal_means: List[float]      # Mean similarity of true features to bundle
    signal_stds: List[float]
    noise_means: List[float]       # Mean similarity of random HVs to bundle
    noise_stds: List[float]
    separations_sigma: List[float] # (signal_mean - noise_mean) / noise_std

    max_safe_features: int = 0
    is_healthy: bool = True
    violations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_counts": self.feature_counts,
            "signal_means": [round(x, 4) for x in self.signal_means],
            "noise_means": [round(x, 4) for x in self.noise_means],
            "separations_sigma": [round(x, 2) for x in self.separations_sigma],
            "max_safe_features": self.max_safe_features,
            "is_healthy": self.is_healthy,
            "violations": self.violations,
        }


@dataclass
class AttractorDiversityReport:
    """Report from attractor diversity check."""
    n_attractors: int
    mean_cos: float
    std_cos: float
    max_cos: float
    cluster_fraction: float  # Fraction with |cos| > cluster_threshold
    usage_fraction: float    # Fraction of attractors activated

    is_healthy: bool = True
    violations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_attractors": self.n_attractors,
            "mean_cos": round(self.mean_cos, 6),
            "std_cos": round(self.std_cos, 6),
            "max_cos": round(self.max_cos, 6),
            "cluster_fraction": round(self.cluster_fraction, 4),
            "usage_fraction": round(self.usage_fraction, 4),
            "is_healthy": self.is_healthy,
            "violations": self.violations,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SoulHealthReport:
    """Comprehensive soul health report."""
    codebook: Optional[CodebookGeometryReport] = None
    bundling: Optional[BundlingStressReport] = None
    attractors: Optional[AttractorDiversityReport] = None

    is_healthy: bool = True
    summary: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_healthy": self.is_healthy,
            "summary": self.summary,
            "timestamp": self.timestamp.isoformat(),
            "codebook": self.codebook.to_dict() if self.codebook else None,
            "bundling": self.bundling.to_dict() if self.bundling else None,
            "attractors": self.attractors.to_dict() if self.attractors else None,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# =============================================================================
# Diagnostic Functions
# =============================================================================

def compute_pairwise_cosines(
    hvs: List[np.ndarray],
    max_pairs: int = 50000,
) -> np.ndarray:
    """
    Compute pairwise cosine similarities for a sample of HV pairs.

    For large N, samples randomly to avoid O(N²) blowup.
    """
    n = len(hvs)
    total_pairs = n * (n - 1) // 2

    if total_pairs <= max_pairs:
        # Compute all pairs
        cosines = []
        for i in range(n):
            for j in range(i + 1, n):
                cosines.append(cosine(hvs[i], hvs[j]))
        return np.array(cosines)
    else:
        # Random sample
        rng = np.random.default_rng()
        cosines = []
        indices = rng.choice(n, size=(max_pairs, 2), replace=True)
        for i, j in indices:
            if i != j:
                cosines.append(cosine(hvs[i], hvs[j]))
        return np.array(cosines)


def check_codebook_geometry(
    vocab: Optional[HDVocab] = None,
    n_samples: int = 1000,
    thresholds: HealthThresholds = DEFAULT_THRESHOLDS,
) -> CodebookGeometryReport:
    """
    Check that base codebook has random-like geometry.

    Tests:
    - Mean pairwise |cos| ≈ 0
    - Std ≈ 1/sqrt(D) ≈ 0.008
    - Negligible mass beyond |cos| > 0.1
    """
    vocab = vocab or get_vocab()

    # Generate sample HVs from vocabulary
    hvs = []

    # Add all canonical symbols
    for name in vocab.CANONICAL_ROLES:
        hvs.append(vocab.role(name))
    for name in vocab.CANONICAL_FEATURES:
        hvs.append(vocab.feature(name))
    for name in vocab.CANONICAL_BINS:
        hvs.append(vocab.bin(name))
    for name in vocab.CANONICAL_TAGS:
        hvs.append(vocab.tag(name))

    # Add random HVs to reach n_samples
    while len(hvs) < n_samples:
        hvs.append(random_hv(vocab.dim))

    # Compute pairwise cosines
    cosines = compute_pairwise_cosines(hvs)
    abs_cosines = np.abs(cosines)

    # Statistics
    mean_cos = float(np.mean(abs_cosines))
    std_cos = float(np.std(cosines))
    min_cos = float(np.min(cosines))
    max_cos = float(np.max(cosines))
    tail_fraction = float(np.mean(abs_cosines > thresholds.codebook_tail_threshold))

    # Check violations
    violations = []
    is_healthy = True

    if mean_cos > thresholds.codebook_mean_cos_max:
        violations.append(f"Mean |cos| = {mean_cos:.4f} > {thresholds.codebook_mean_cos_max}")
        is_healthy = False

    if std_cos > thresholds.codebook_std_cos_max:
        violations.append(f"Std cos = {std_cos:.4f} > {thresholds.codebook_std_cos_max}")
        is_healthy = False

    if tail_fraction > thresholds.codebook_tail_fraction_max:
        violations.append(
            f"Tail fraction = {tail_fraction:.4f} > {thresholds.codebook_tail_fraction_max} "
            f"(at threshold {thresholds.codebook_tail_threshold})"
        )
        is_healthy = False

    return CodebookGeometryReport(
        n_samples=len(hvs),
        n_pairs=len(cosines),
        mean_cos=mean_cos,
        std_cos=std_cos,
        min_cos=min_cos,
        max_cos=max_cos,
        tail_fraction=tail_fraction,
        is_healthy=is_healthy,
        violations=violations,
    )


def stress_test_bundling(
    vocab: Optional[HDVocab] = None,
    feature_counts: List[int] = None,
    n_trials: int = 100,
    n_noise_samples: int = 100,
    thresholds: HealthThresholds = DEFAULT_THRESHOLDS,
) -> BundlingStressReport:
    """
    Test bundling capacity at various feature counts.

    For each K in feature_counts:
    1. Generate K role-bound feature HVs
    2. Bundle into H_K
    3. Measure similarity of original features to bundle (signal)
    4. Measure similarity of random HVs to bundle (noise)
    5. Compute separation in sigma units
    """
    vocab = vocab or get_vocab()

    if feature_counts is None:
        feature_counts = [8, 16, 24, 32, 48, 64, 96]

    signal_means = []
    signal_stds = []
    noise_means = []
    noise_stds = []
    separations = []

    for K in feature_counts:
        trial_signals = []
        trial_noises = []

        for _ in range(n_trials):
            # Generate K feature HVs (role-bound)
            features = []
            for i in range(K):
                role = vocab.role(f"TEST_ROLE_{i % 8}")
                feat = vocab.feature(f"TEST_FEATURE_{i}")
                val = vocab.bin(f"BIN_{i % 4}")
                bound = bind(role, bind(feat, val))
                features.append(bound)

            # Bundle
            h_bundle = bundle(features)

            # Signal: similarity of true features to bundle
            for f in features:
                trial_signals.append(cosine(f, h_bundle))

            # Noise: similarity of random HVs to bundle
            for _ in range(n_noise_samples // n_trials + 1):
                noise_hv = random_hv(vocab.dim)
                trial_noises.append(cosine(noise_hv, h_bundle))

        signal_mean = np.mean(trial_signals)
        signal_std = np.std(trial_signals)
        noise_mean = np.mean(trial_noises)
        noise_std = np.std(trial_noises)

        # Separation in sigma units
        if noise_std > 0:
            separation = (signal_mean - noise_mean) / noise_std
        else:
            separation = float('inf')

        signal_means.append(float(signal_mean))
        signal_stds.append(float(signal_std))
        noise_means.append(float(noise_mean))
        noise_stds.append(float(noise_std))
        separations.append(float(separation))

    # Find max safe features (where separation >= threshold)
    max_safe = 0
    for i, (K, sep) in enumerate(zip(feature_counts, separations)):
        if sep >= thresholds.signal_noise_sigma_min:
            max_safe = K

    # Check violations
    violations = []
    is_healthy = True

    # Check that recommended max (50) is still safe
    if max_safe < thresholds.max_features_per_moment:
        violations.append(
            f"Max safe features = {max_safe} < recommended {thresholds.max_features_per_moment}"
        )
        is_healthy = False

    return BundlingStressReport(
        feature_counts=feature_counts,
        signal_means=signal_means,
        signal_stds=signal_stds,
        noise_means=noise_means,
        noise_stds=noise_stds,
        separations_sigma=separations,
        max_safe_features=max_safe,
        is_healthy=is_healthy,
        violations=violations,
    )


def check_attractor_diversity(
    attractors: List[np.ndarray],
    usage_counts: Optional[List[int]] = None,
    thresholds: HealthThresholds = DEFAULT_THRESHOLDS,
) -> AttractorDiversityReport:
    """
    Check attractor diversity and distribution.

    A healthy soul has:
    - Mean pairwise |cos| < 0.15
    - No > 10% of attractors in tight clusters (|cos| > 0.4)
    - >= 80% of attractors activated at least once
    """
    n = len(attractors)

    # Compute pairwise cosines
    cosines = compute_pairwise_cosines(attractors)
    abs_cosines = np.abs(cosines)

    mean_cos = float(np.mean(abs_cosines))
    std_cos = float(np.std(cosines))
    max_cos = float(np.max(abs_cosines))

    # Cluster fraction: pairs with |cos| > threshold
    cluster_fraction = float(np.mean(abs_cosines > thresholds.attractor_cluster_threshold))

    # Usage fraction
    if usage_counts is not None:
        usage_fraction = float(np.mean(np.array(usage_counts) > 0))
    else:
        usage_fraction = 1.0  # Assume all used if not tracked

    # Check violations
    violations = []
    is_healthy = True

    if mean_cos > thresholds.attractor_mean_cos_max:
        violations.append(f"Mean |cos| = {mean_cos:.4f} > {thresholds.attractor_mean_cos_max}")
        is_healthy = False

    if cluster_fraction > thresholds.attractor_cluster_fraction_max:
        violations.append(
            f"Cluster fraction = {cluster_fraction:.4f} > {thresholds.attractor_cluster_fraction_max}"
        )
        is_healthy = False

    if usage_fraction < thresholds.attractor_usage_min:
        violations.append(f"Usage fraction = {usage_fraction:.4f} < {thresholds.attractor_usage_min}")
        is_healthy = False

    return AttractorDiversityReport(
        n_attractors=n,
        mean_cos=mean_cos,
        std_cos=std_cos,
        max_cos=max_cos,
        cluster_fraction=cluster_fraction,
        usage_fraction=usage_fraction,
        is_healthy=is_healthy,
        violations=violations,
    )


def run_full_health_check(
    vocab: Optional[HDVocab] = None,
    attractors: Optional[List[np.ndarray]] = None,
    usage_counts: Optional[List[int]] = None,
    thresholds: HealthThresholds = DEFAULT_THRESHOLDS,
) -> SoulHealthReport:
    """
    Run comprehensive soul health check.

    Returns SoulHealthReport with all diagnostics.
    """
    vocab = vocab or get_vocab()

    # Codebook check
    codebook_report = check_codebook_geometry(vocab, thresholds=thresholds)

    # Bundling stress test
    bundling_report = stress_test_bundling(vocab, thresholds=thresholds)

    # Attractor check (if provided)
    attractor_report = None
    if attractors is not None:
        attractor_report = check_attractor_diversity(
            attractors, usage_counts, thresholds=thresholds
        )

    # Aggregate health
    is_healthy = codebook_report.is_healthy and bundling_report.is_healthy
    if attractor_report is not None:
        is_healthy = is_healthy and attractor_report.is_healthy

    # Summary
    violations = []
    violations.extend(codebook_report.violations)
    violations.extend(bundling_report.violations)
    if attractor_report:
        violations.extend(attractor_report.violations)

    if is_healthy:
        summary = "Soul geometry is healthy. All capacity/interference checks passed."
    else:
        summary = f"Soul geometry UNHEALTHY. {len(violations)} violation(s): " + "; ".join(violations)

    return SoulHealthReport(
        codebook=codebook_report,
        bundling=bundling_report,
        attractors=attractor_report,
        is_healthy=is_healthy,
        summary=summary,
    )


# =============================================================================
# Monitoring Utilities
# =============================================================================

def log_health_metrics(report: SoulHealthReport, logger: logging.Logger = logger) -> None:
    """Log health metrics for monitoring."""
    if report.is_healthy:
        logger.info(f"Soul health: OK - {report.summary}")
    else:
        logger.warning(f"Soul health: DEGRADED - {report.summary}")

    if report.codebook:
        logger.debug(
            f"  Codebook: mean_cos={report.codebook.mean_cos:.4f}, "
            f"tail_fraction={report.codebook.tail_fraction:.4f}"
        )

    if report.bundling:
        logger.debug(
            f"  Bundling: max_safe_features={report.bundling.max_safe_features}, "
            f"separations={report.bundling.separations_sigma}"
        )

    if report.attractors:
        logger.debug(
            f"  Attractors: mean_cos={report.attractors.mean_cos:.4f}, "
            f"cluster_fraction={report.attractors.cluster_fraction:.4f}, "
            f"usage={report.attractors.usage_fraction:.2%}"
        )


def save_health_report(report: SoulHealthReport, path: str) -> None:
    """Save health report to JSON file."""
    with open(path, 'w') as f:
        f.write(report.to_json())


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Thresholds
    'HealthThresholds',
    'DEFAULT_THRESHOLDS',
    # Reports
    'CodebookGeometryReport',
    'BundlingStressReport',
    'AttractorDiversityReport',
    'SoulHealthReport',
    # Functions
    'compute_pairwise_cosines',
    'check_codebook_geometry',
    'stress_test_bundling',
    'check_attractor_diversity',
    'run_full_health_check',
    # Utilities
    'log_health_metrics',
    'save_health_report',
]
