"""
Oversample + Rerank Configuration
==================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class OversampleConfig:
    """Configuration for oversample + rerank pipeline."""

    # Target retrieval
    k: int = 8                        # Final top-K results
    oversample_factor: float = 4.0    # Candidates = k × oversample_factor

    # Adaptive bounds
    min_oversample: float = 1.5
    max_oversample: float = 8.0

    # Similarity thresholds
    coarse_threshold: float = 0.41    # Minimum coarse similarity

    # Teleology reranking
    teleo_weight_default: float = 0.2 # Default teleology weight
    teleo_weight_min: float = 0.1
    teleo_weight_max: float = 0.5

    # Latency budgets
    coarse_latency_budget_us: float = 100.0   # Stage 1
    rerank_latency_budget_us: float = 500.0   # Stage 2

    # Tuning
    tune_interval_ticks: int = 100    # Tune every N sovereign ticks
    recall_target: float = 0.999      # Target recall@k

    # Feature flags
    use_gpu_rescore: bool = False     # Use GPU for Stage 2
    use_fpga_rescore: bool = True     # Use FPGA for Stage 2

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


# Default configuration
OVERSAMPLE_CONFIG = OversampleConfig()


def get_oversample_config() -> OversampleConfig:
    """Get the current oversample configuration."""
    return OVERSAMPLE_CONFIG


# Precomputed recall curves (from calibration)
# oversample_factor → expected recall@8
RECALL_CURVE_D173 = {
    1.5: 0.923,
    2.0: 0.978,
    3.0: 0.992,
    4.0: 0.999,
    6.0: 0.999,
    8.0: 0.999,
}


def lookup_expected_recall(factor: float, D: int = 173) -> float:
    """Lookup expected recall for an oversample factor."""
    if D == 173:
        curve = RECALL_CURVE_D173
    else:
        # Fallback: assume recall improves with oversample
        return min(0.999, 0.9 + 0.02 * factor)

    # Interpolate
    factors = sorted(curve.keys())
    if factor <= factors[0]:
        return curve[factors[0]]
    if factor >= factors[-1]:
        return curve[factors[-1]]

    for i in range(len(factors) - 1):
        if factors[i] <= factor <= factors[i+1]:
            t = (factor - factors[i]) / (factors[i+1] - factors[i])
            return curve[factors[i]] * (1-t) + curve[factors[i+1]] * t

    return 0.999


def lookup_factor_for_recall(target_recall: float, D: int = 173) -> float:
    """Find minimum oversample factor to achieve target recall."""
    if D == 173:
        curve = RECALL_CURVE_D173
    else:
        # Conservative fallback
        return 4.0

    for factor, recall in sorted(curve.items()):
        if recall >= target_recall:
            return factor

    return 8.0  # Max if target not achievable
