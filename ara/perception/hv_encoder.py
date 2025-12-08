"""
Ara HV Encoder - Sensory to Hypervector
=======================================

Encodes sensory snapshots into high-dimensional hypervectors for the HTC.

Architecture:
- Each sense has a base "role" hypervector (fixed, random)
- Numeric features are binned and encoded as feature HVs
- Semantic tags are encoded with weighted emphasis
- Role × Features are bound together per sense
- All sense HVs are bundled into one context HV

This is the bridge between embodied perception and holographic memory.
"""

from __future__ import annotations

import hashlib
import logging
import random
from typing import Dict, List, Optional, Any

from .sensory import SenseReading, SensorySnapshot

logger = logging.getLogger(__name__)


class HVEncoder:
    """
    Hyperdimensional encoder for sensory data.

    Uses VSA/HD computing operations:
    - Binding (XOR for binary, element-wise multiply for bipolar)
    - Bundling (majority vote / thresholded sum)
    - Permutation (circular shift for sequence)

    Each sense has a role vector, and features are bound to create
    a structured representation.
    """

    # Senses to encode
    SENSE_NAMES = [
        "vision", "hearing", "touch", "smell",
        "taste", "vestibular", "proprioception", "interoception"
    ]

    def __init__(self, dim: int = 8192, rng: Optional[random.Random] = None):
        """
        Initialize the HV encoder.

        Args:
            dim: Hypervector dimension (must match HTC)
            rng: Random number generator for reproducibility
        """
        self.dim = dim
        self.rng = rng or random.Random(42)  # Fixed seed for reproducibility

        # Base role hypervectors (one per sense)
        self.role_hv: Dict[str, List[int]] = {}
        for sense in self.SENSE_NAMES:
            self.role_hv[sense] = self._rand_hv(f"role:{sense}")

        # Feature hypervectors (lazily created)
        self._feature_hv_cache: Dict[str, List[int]] = {}

        logger.info(f"HVEncoder initialized: dim={dim}, senses={len(self.SENSE_NAMES)}")

    def _rand_hv(self, seed_str: str) -> List[int]:
        """Generate a deterministic random bipolar hypervector."""
        seed = int(hashlib.sha256(seed_str.encode()).hexdigest()[:8], 16)
        local_rng = random.Random(seed)
        return [local_rng.choice([-1, 1]) for _ in range(self.dim)]

    def _get_feature_hv(self, name: str) -> List[int]:
        """Get or create a feature hypervector."""
        if name not in self._feature_hv_cache:
            self._feature_hv_cache[name] = self._rand_hv(f"feature:{name}")
        return self._feature_hv_cache[name]

    def encode_sensory_snapshot(self, snap: SensorySnapshot) -> List[int]:
        """
        Encode a complete sensory snapshot into a single context HV.

        Each sense is encoded as: role ⊗ (∑ features)
        All senses are bundled: ∑ (role_i ⊗ sense_i)

        Args:
            snap: SensorySnapshot with all sense readings

        Returns:
            Bipolar hypervector (list of +1/-1)
        """
        # Accumulator for bundling
        hv_sum = [0] * self.dim

        for sense_name in self.SENSE_NAMES:
            if sense_name not in snap.readings:
                continue

            reading = snap.readings[sense_name]
            role = self.role_hv[sense_name]

            # Encode this sense
            sense_hv = self._encode_sense_reading(sense_name, reading)

            # Bind with role and add to bundle
            bound = self._bind(role, sense_hv)
            for i in range(self.dim):
                hv_sum[i] += bound[i]

        # Final bipolar sign
        return [1 if x >= 0 else -1 for x in hv_sum]

    def _encode_sense_reading(self, sense_name: str, reading: SenseReading) -> List[int]:
        """
        Encode a single sense reading.

        Combines:
        - Binned numeric features
        - Weighted semantic tags
        """
        sense_sum = [0] * self.dim

        # Encode numeric features (binned)
        for feat_name, val in reading.value.items():
            bin_id = self._bin_feature(sense_name, feat_name, val)
            feat_key = f"{sense_name}:{feat_name}:bin{bin_id}"
            feat_hv = self._get_feature_hv(feat_key)
            for i in range(self.dim):
                sense_sum[i] += feat_hv[i]

        # Encode semantic tags (weighted)
        for tag, weight in reading.tags.items():
            if weight < 0.1:
                continue  # Skip negligible tags

            tag_key = f"{sense_name}:tag:{tag}"
            tag_hv = self._get_feature_hv(tag_key)

            # Scale contribution by weight (1-5 copies)
            copies = max(1, min(5, int(weight * 5)))
            for i in range(self.dim):
                sense_sum[i] += copies * tag_hv[i]

        # Return bipolar
        return [1 if x >= 0 else -1 for x in sense_sum]

    def _bind(self, hv1: List[int], hv2: List[int]) -> List[int]:
        """
        Bind two hypervectors (element-wise multiply for bipolar).

        Binding creates a new HV that is dissimilar to both inputs
        but can be "unbound" by multiplying again.
        """
        return [a * b for a, b in zip(hv1, hv2)]

    def _bin_feature(self, sense: str, feat: str, val: float) -> int:
        """
        Bin a numeric feature into discrete levels.

        Different features have different ranges and semantics.
        """
        # Temperature features
        if "temp" in feat or "deg" in feat:
            if val < 35:
                return 0  # cool
            if val < 50:
                return 1  # warm
            if val < 70:
                return 2  # hot
            if val < 85:
                return 3  # very hot
            return 4  # critical

        # Voltage features
        if "vcore" in feat:
            if val < 0.88:
                return 0  # critical low
            if val < 0.90:
                return 1  # low
            if val < 0.95:
                return 2  # nominal
            return 3  # high

        if "v33" in feat or "v5" in feat:
            if abs(val - 3.3 if "v33" in feat else val - 5.0) > 0.1:
                return 0  # out of spec
            if abs(val - 3.3 if "v33" in feat else val - 5.0) > 0.05:
                return 1  # marginal
            return 2  # good

        # RPM features (fans)
        if "rpm" in feat:
            if val < 1000:
                return 0  # slow/stopped
            if val < 2500:
                return 1  # normal
            if val < 4000:
                return 2  # high
            return 3  # screaming

        # Percentage features
        if "pct" in feat or "percent" in feat:
            if val < 30:
                return 0
            if val < 60:
                return 1
            if val < 85:
                return 2
            return 3

        # Fatigue/stress (0-1 scale)
        if feat in ["fatigue", "stress", "focus"]:
            if val < 0.3:
                return 0
            if val < 0.6:
                return 1
            if val < 0.8:
                return 2
            return 3

        # Default: 3-level binning
        if val < 0.33:
            return 0
        if val < 0.66:
            return 1
        return 2

    def encode_single_sense(self, sense_name: str, reading: SenseReading) -> List[int]:
        """Encode just one sense (for debugging/testing)."""
        role = self.role_hv.get(sense_name)
        if not role:
            raise ValueError(f"Unknown sense: {sense_name}")

        sense_hv = self._encode_sense_reading(sense_name, reading)
        return self._bind(role, sense_hv)

    def similarity(self, hv1: List[int], hv2: List[int]) -> float:
        """Compute cosine similarity between two bipolar HVs."""
        dot = sum(a * b for a, b in zip(hv1, hv2))
        return dot / self.dim


# =============================================================================
# Singleton Access
# =============================================================================

_hv_encoder: Optional[HVEncoder] = None


def get_hv_encoder(dim: int = 8192) -> HVEncoder:
    """Get the default HVEncoder instance."""
    global _hv_encoder
    if _hv_encoder is None or _hv_encoder.dim != dim:
        _hv_encoder = HVEncoder(dim=dim)
    return _hv_encoder


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'HVEncoder',
    'get_hv_encoder',
]
