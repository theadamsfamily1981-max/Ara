"""
Heim Configuration - Soul Compression Parameters
================================================

Version the soul so we can migrate later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any
import json
from pathlib import Path


@dataclass
class HeimConfig:
    """
    Heim compression configuration.

    These parameters define how the 16k soul compresses to 173 bits.
    """
    # Core dimensions
    D_full: int = 16384           # Original HV dimension
    D_compressed: int = 173       # Heim-reduced dimension

    # Sparsity (fraction of zeros in compressed HV)
    sparsity: float = 0.7         # 70% zeros → 30% active bits

    # Binary representation
    binary: bool = True           # Use {0,1} not {-1,+1}

    # Similarity thresholds
    cluster_merge_threshold: float = 0.80   # Join existing cluster
    duplicate_threshold: float = 0.95       # Near-duplicate detection

    # Eviction policy
    min_retention_hours: float = 24.0       # Minimum episode lifetime
    resonance_floor: float = 0.1            # Below this = eviction candidate

    # Oversampling (for retrieval)
    default_oversample: float = 4.0         # 4× candidates for rerank
    min_oversample: float = 1.5
    max_oversample: float = 8.0

    # Performance budgets
    coarse_latency_budget_us: float = 100.0   # Stage 1 budget
    rerank_latency_budget_us: float = 500.0   # Stage 2 budget

    # Versioning
    version: str = "heim-v1.0"

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'HeimConfig':
        return cls(**{k: v for k, v in d.items() if hasattr(cls, k)})

    def save(self, path: Path) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'HeimConfig':
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


# Default configuration
HEIM_CONFIG = HeimConfig()


def get_heim_config() -> HeimConfig:
    """Get the current Heim configuration."""
    return HEIM_CONFIG


# Projection matrix seed (for reproducibility)
PROJECTION_SEED = 0xARA173


# Thresholds for geometry validation
GEOMETRY_THRESHOLDS = {
    "max_pairwise_cosine": 0.25,      # Max acceptable |cos| between base HVs
    "expected_cosine_std": 0.076,      # 1/√173 ≈ 0.076
    "cosine_outlier_fraction": 0.01,   # Max 1% pairs above threshold
}


# Bundling capacity requirements
BUNDLING_REQUIREMENTS = {
    "min_k": 8,                        # Minimum bundle size
    "target_k": 16,                    # Target bundle size
    "max_k": 32,                       # Maximum bundle size
    "min_signal_margin": 0.15,         # Signal above random baseline
}
