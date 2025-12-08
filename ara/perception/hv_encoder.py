"""
Ara HV Encoder - Sensory to Hypervector
=======================================

Encodes sensory snapshots into high-dimensional hypervectors for the HTC.

VSA/HD Computing Operations (per Kanerva 2009, Karunaratne 2020):
- Binding (⊗): XOR for binary, element-wise multiply for bipolar
  - Creates HV dissimilar to both inputs, reversible by binding again
- Bundling (+): Majority vote / thresholded sum
  - Creates HV similar to all inputs (superposition)
- Permutation (ρ): Circular shift for sequence encoding

Architecture:
- Each sense has a base "role" hypervector (fixed, random)
- Numeric features are binned and encoded as feature HVs
- Role-Filler binding: H_attr = H_ROLE ⊗ H_FEATURE ⊗ H_VALUE
- Bundling across senses: H_context = sign(∑ H_sense_i)

References:
- Kanerva (2009): Hyperdimensional Computing
- Karunaratne (2020): In-memory HDC with resistive memory
- PMC9189416: HD computing for multimodal encoding
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple

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
# VSA Operations (Research-Grade)
# =============================================================================

def xor_bind(hv1: List[int], hv2: List[int]) -> List[int]:
    """
    XOR binding for binary hypervectors.

    For bipolar {-1, +1}: XOR is equivalent to element-wise multiply.
    For binary {0, 1}: use actual XOR.

    Returns HV dissimilar to both inputs.
    """
    return [a * b for a, b in zip(hv1, hv2)]


def bundle(hvs: List[List[int]], threshold: float = 0.0) -> List[int]:
    """
    Bundle multiple HVs via majority vote (thresholded sum).

    The result is similar to all inputs (superposition).

    Args:
        hvs: List of bipolar hypervectors
        threshold: Tie-breaking threshold (default 0 = random on tie)

    Returns:
        Bundled bipolar HV
    """
    if not hvs:
        raise ValueError("Cannot bundle empty list")

    dim = len(hvs[0])
    sums = [0] * dim

    for hv in hvs:
        for i in range(dim):
            sums[i] += hv[i]

    # Majority vote with threshold
    return [1 if s > threshold else -1 if s < -threshold else random.choice([-1, 1])
            for s in sums]


def permute(hv: List[int], shift: int = 1) -> List[int]:
    """
    Circular shift permutation for sequence encoding.

    Used to encode temporal order: ρ(H_t-1) ⊗ H_t
    """
    shift = shift % len(hv)
    return hv[-shift:] + hv[:-shift]


def hamming_similarity(hv1: List[int], hv2: List[int]) -> float:
    """
    Normalized Hamming similarity for bipolar HVs.

    Returns value in [-1, 1] where 1 = identical, -1 = opposite.
    """
    matches = sum(1 for a, b in zip(hv1, hv2) if a == b)
    return (2 * matches / len(hv1)) - 1


# =============================================================================
# Affect Decoder (Graphics as Cognition)
# =============================================================================

@dataclass
class AffectState:
    """
    Emotional/cognitive state derived from HTC resonance.

    Maps soul state to animation parameters for the avatar.
    """
    valence: float       # -1 (negative) to +1 (positive)
    arousal: float       # 0 (calm) to 1 (activated)
    certainty: float     # 0 (confused) to 1 (confident)
    focus: float         # 0 (scattered) to 1 (concentrated)

    def to_animation_params(self) -> Dict[str, float]:
        """
        Convert affect state to avatar animation parameters.

        Returns dict with:
        - eye_aperture: How open the eyes are
        - color_warmth: Blue (calm) to orange (aroused)
        - jitter: Amount of micro-movement
        - breath_rate: Simulated breathing speed
        """
        # Eye aperture: more open when aroused, less when uncertain
        eye_aperture = 0.5 + 0.3 * self.arousal - 0.2 * (1 - self.certainty)
        eye_aperture = max(0.1, min(1.0, eye_aperture))

        # Color warmth: blue=0, orange=1
        color_warmth = (self.arousal + 1) / 2  # Map [-1,1] to [0,1]
        if self.valence < -0.5:
            color_warmth = min(color_warmth, 0.3)  # Stay cool when distressed

        # Jitter: high arousal + low certainty = jittery
        jitter = max(0.0, self.arousal - self.certainty) * 0.5

        # Breath rate: faster when aroused
        breath_rate = 0.5 + 0.5 * self.arousal

        return {
            "eye_aperture": eye_aperture,
            "color_warmth": color_warmth,
            "jitter": jitter,
            "breath_rate": breath_rate,
            "valence": self.valence,
            "focus": self.focus,
        }


def decode_affect(
    resonance: float,
    reward_history: List[float],
    attractor_entropy: float = 0.5,
) -> AffectState:
    """
    Decode affect state from HTC metrics.

    Args:
        resonance: Max similarity to any attractor [0, 1]
        reward_history: Recent reward values [-1, 1]
        attractor_entropy: Entropy of resonance distribution [0, 1]

    Returns:
        AffectState for avatar animation
    """
    # Valence: average recent reward, squashed
    if reward_history:
        avg_reward = sum(reward_history) / len(reward_history)
        valence = math.tanh(avg_reward * 2)  # Squash to [-1, 1]
    else:
        valence = 0.0

    # Arousal: variability of rewards
    if len(reward_history) > 1:
        mean = sum(reward_history) / len(reward_history)
        variance = sum((r - mean) ** 2 for r in reward_history) / len(reward_history)
        arousal = math.tanh(math.sqrt(variance) * 3)
    else:
        arousal = 0.3  # Default mild arousal

    # Certainty: inverse of attractor entropy
    certainty = 1.0 - attractor_entropy

    # Focus: resonance strength
    focus = resonance

    return AffectState(
        valence=valence,
        arousal=arousal,
        certainty=certainty,
        focus=focus,
    )


# =============================================================================
# Flow HV Encoder (Network Layer)
# =============================================================================

@dataclass
class FlowFeatures:
    """Network flow features for HV encoding."""
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    bytes_per_sec: float
    packets_per_sec: float
    latency_ms: float
    service: str = "unknown"


class FlowHVEncoder:
    """
    Encodes network flows into hypervectors for the LAN cortex.

    Used for:
    - Flow classification (normal vs suspicious)
    - Traffic pattern learning
    - SmartNIC/eBPF policy encoding
    """

    def __init__(self, dim: int = 8192):
        self.dim = dim
        self._cache: Dict[str, List[int]] = {}

        # Base role HVs
        self.role_src = self._rand_hv("role:src_node")
        self.role_dst = self._rand_hv("role:dst_node")
        self.role_service = self._rand_hv("role:service")
        self.role_latency = self._rand_hv("role:latency")
        self.role_rate = self._rand_hv("role:rate")

    def _rand_hv(self, seed_str: str) -> List[int]:
        seed = int(hashlib.sha256(seed_str.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        return [rng.choice([-1, 1]) for _ in range(self.dim)]

    def _get_hv(self, key: str) -> List[int]:
        if key not in self._cache:
            self._cache[key] = self._rand_hv(key)
        return self._cache[key]

    def encode_flow(self, flow: FlowFeatures) -> List[int]:
        """
        Encode a network flow into a hypervector.

        H_flow = H_SRC ⊗ H_DST ⊗ H_SERVICE ⊗ H_LATENCY_BIN ⊗ H_RATE_BIN
        """
        # Get node HVs
        h_src = self._get_hv(f"node:{flow.src_ip}")
        h_dst = self._get_hv(f"node:{flow.dst_ip}")
        h_service = self._get_hv(f"service:{flow.service}")

        # Bin latency
        if flow.latency_ms < 10:
            lat_bin = "low"
        elif flow.latency_ms < 50:
            lat_bin = "medium"
        elif flow.latency_ms < 200:
            lat_bin = "high"
        else:
            lat_bin = "critical"
        h_latency = self._get_hv(f"latency:{lat_bin}")

        # Bin rate
        if flow.bytes_per_sec < 1e6:
            rate_bin = "low"
        elif flow.bytes_per_sec < 100e6:
            rate_bin = "medium"
        else:
            rate_bin = "high"
        h_rate = self._get_hv(f"rate:{rate_bin}")

        # Bind with roles
        components = [
            xor_bind(self.role_src, h_src),
            xor_bind(self.role_dst, h_dst),
            xor_bind(self.role_service, h_service),
            xor_bind(self.role_latency, h_latency),
            xor_bind(self.role_rate, h_rate),
        ]

        # Bundle all components
        return bundle(components)


# Singleton for flow encoder
_flow_encoder: Optional[FlowHVEncoder] = None


def get_flow_encoder(dim: int = 8192) -> FlowHVEncoder:
    """Get the default FlowHVEncoder instance."""
    global _flow_encoder
    if _flow_encoder is None or _flow_encoder.dim != dim:
        _flow_encoder = FlowHVEncoder(dim=dim)
    return _flow_encoder


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'HVEncoder',
    'get_hv_encoder',
    # VSA operations
    'xor_bind',
    'bundle',
    'permute',
    'hamming_similarity',
    # Affect decoder
    'AffectState',
    'decode_affect',
    # Flow encoder
    'FlowFeatures',
    'FlowHVEncoder',
    'get_flow_encoder',
]
