#!/usr/bin/env python3
"""
Hyperdimensional VSA - Core Implementation
==========================================

16kD hypervectors for holographic identity representation.
Implements NIB structural compression and multimodal fusion.

Theory:
    - Dimension D = 16384 provides 10^1235 unique states
    - Practical limit: N = 12 modalities before collapse
    - T_s preserved across fusion operations

Vector Types:
    - Binary: {-1, +1}^D (MAP architecture)
    - Bipolar: [-1, +1]^D (continuous relaxation)
    - Sparse: k-hot binary (memory efficient)

Key Operations:
    - Bundle (⊕): Element-wise majority/sum → superposition
    - Bind (⊗): Element-wise XOR/multiply → association
    - Permute (ρ): Circular shift → sequence encoding
    - Similarity: Cosine or Hamming distance
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
import hashlib


class VectorType(str, Enum):
    """Type of hypervector representation."""
    BINARY = "binary"       # {-1, +1}
    BIPOLAR = "bipolar"     # [-1, +1] continuous
    SPARSE = "sparse"       # k-hot


@dataclass
class HyperVector:
    """
    A single hyperdimensional vector.

    Default: D = 16384 dimensions, bipolar values.
    """
    vector: np.ndarray
    dim: int = 16384
    vtype: VectorType = VectorType.BIPOLAR
    name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.vector is None:
            self.vector = np.random.choice([-1, 1], size=self.dim).astype(np.float32)
        self.dim = len(self.vector)

    @classmethod
    def random(cls, dim: int = 16384, vtype: VectorType = VectorType.BIPOLAR,
               name: str = "") -> 'HyperVector':
        """Create a random hypervector."""
        if vtype == VectorType.BINARY:
            vec = np.random.choice([-1, 1], size=dim).astype(np.float32)
        elif vtype == VectorType.BIPOLAR:
            vec = np.random.uniform(-1, 1, size=dim).astype(np.float32)
        else:  # SPARSE
            vec = np.zeros(dim, dtype=np.float32)
            k = dim // 100  # 1% sparsity
            indices = np.random.choice(dim, k, replace=False)
            vec[indices] = np.random.choice([-1, 1], size=k)

        return cls(vector=vec, dim=dim, vtype=vtype, name=name)

    @classmethod
    def from_seed(cls, seed: str, dim: int = 16384) -> 'HyperVector':
        """Create deterministic hypervector from seed string."""
        # Use hash as RNG seed
        seed_int = int(hashlib.sha256(seed.encode()).hexdigest()[:16], 16)
        rng = np.random.RandomState(seed_int % (2**31))
        vec = rng.choice([-1, 1], size=dim).astype(np.float32)
        return cls(vector=vec, dim=dim, name=seed)

    @classmethod
    def from_features(cls, features: np.ndarray, dim: int = 16384,
                      name: str = "") -> 'HyperVector':
        """
        Encode feature vector into hypervector.

        Uses random projection (Johnson-Lindenstrauss).
        """
        if features.ndim > 1:
            features = features.flatten()

        # Random projection matrix (deterministic based on feature dim)
        seed = len(features) * 12345
        rng = np.random.RandomState(seed)
        projection = rng.randn(len(features), dim).astype(np.float32)
        projection /= np.sqrt(len(features))

        # Project and binarize
        projected = features @ projection
        vec = np.sign(projected).astype(np.float32)
        vec[vec == 0] = 1  # No zeros

        return cls(vector=vec, dim=dim, vtype=VectorType.BIPOLAR, name=name)

    def binarize(self) -> 'HyperVector':
        """Convert to binary {-1, +1}."""
        binary_vec = np.sign(self.vector).astype(np.float32)
        binary_vec[binary_vec == 0] = 1
        return HyperVector(
            vector=binary_vec, dim=self.dim,
            vtype=VectorType.BINARY, name=self.name
        )

    def normalize(self) -> 'HyperVector':
        """L2 normalize the vector."""
        norm = np.linalg.norm(self.vector)
        if norm > 0:
            return HyperVector(
                vector=self.vector / norm, dim=self.dim,
                vtype=self.vtype, name=self.name
            )
        return self

    def to_bytes(self) -> bytes:
        """Compact binary serialization."""
        # Pack as int8 for bipolar, bit-pack for binary
        if self.vtype == VectorType.BINARY:
            # Bit-pack: 8 values per byte
            bits = (self.vector > 0).astype(np.uint8)
            packed = np.packbits(bits)
            return packed.tobytes()
        else:
            # Quantize to int8
            quantized = (self.vector * 127).astype(np.int8)
            return quantized.tobytes()

    @classmethod
    def from_bytes(cls, data: bytes, dim: int = 16384,
                   vtype: VectorType = VectorType.BIPOLAR) -> 'HyperVector':
        """Deserialize from bytes."""
        if vtype == VectorType.BINARY:
            packed = np.frombuffer(data, dtype=np.uint8)
            bits = np.unpackbits(packed)[:dim]
            vec = bits.astype(np.float32) * 2 - 1  # 0,1 → -1,+1
        else:
            quantized = np.frombuffer(data, dtype=np.int8)[:dim]
            vec = quantized.astype(np.float32) / 127

        return cls(vector=vec, dim=dim, vtype=vtype)


def bind(a: HyperVector, b: HyperVector) -> HyperVector:
    """
    Bind operation (⊗): Element-wise multiplication.

    Creates association between concepts.
    Self-inverse: bind(bind(a, b), b) ≈ a
    """
    result = a.vector * b.vector
    return HyperVector(
        vector=result, dim=a.dim,
        vtype=a.vtype, name=f"({a.name}⊗{b.name})"
    )


def bundle(vectors: List[HyperVector], weights: List[float] = None) -> HyperVector:
    """
    Bundle operation (⊕): Weighted sum with thresholding.

    Creates superposition of concepts.
    Limit: ~12 vectors before interference.
    """
    if not vectors:
        raise ValueError("Cannot bundle empty list")

    if weights is None:
        weights = [1.0] * len(vectors)

    # Weighted sum
    result = np.zeros(vectors[0].dim, dtype=np.float32)
    for v, w in zip(vectors, weights):
        result += w * v.vector

    # Threshold to bipolar (majority vote)
    result = np.sign(result)
    result[result == 0] = 1

    names = "+".join(v.name for v in vectors if v.name)
    return HyperVector(
        vector=result, dim=vectors[0].dim,
        vtype=VectorType.BIPOLAR, name=f"⊕({names})"
    )


def permute(v: HyperVector, shifts: int = 1) -> HyperVector:
    """
    Permute operation (ρ): Circular shift.

    Encodes sequence position or temporal order.
    """
    result = np.roll(v.vector, shifts)
    return HyperVector(
        vector=result, dim=v.dim,
        vtype=v.vtype, name=f"ρ^{shifts}({v.name})"
    )


def similarity(a: HyperVector, b: HyperVector) -> float:
    """
    Similarity measure: Cosine similarity.

    Range: [-1, 1], where 1 = identical, -1 = opposite
    """
    dot = np.dot(a.vector, b.vector)
    norm_a = np.linalg.norm(a.vector)
    norm_b = np.linalg.norm(b.vector)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot / (norm_a * norm_b))


def hamming_similarity(a: HyperVector, b: HyperVector) -> float:
    """
    Hamming similarity for binary vectors.

    Returns fraction of matching bits.
    """
    matches = np.sum(np.sign(a.vector) == np.sign(b.vector))
    return float(matches / a.dim)


class ModalityEncoder:
    """
    Encoder for a specific modality.

    Maps modality-specific features to hypervector space.
    """

    def __init__(self, name: str, dim: int = 16384,
                 feature_dim: int = None):
        self.name = name
        self.dim = dim
        self.feature_dim = feature_dim

        # Base vector for this modality (deterministic)
        self.base = HyperVector.from_seed(f"modality:{name}", dim)

        # Level vectors for quantization (if needed)
        self.levels: Dict[int, HyperVector] = {}

    def encode(self, features: Union[np.ndarray, float, int]) -> HyperVector:
        """Encode features into hypervector."""
        if isinstance(features, (int, float)):
            # Scalar encoding using levels
            return self._encode_scalar(float(features))
        else:
            # Vector encoding using projection
            hv = HyperVector.from_features(features, self.dim, self.name)
            # Bind with modality base to mark source
            return bind(hv, self.base)

    def _encode_scalar(self, value: float, levels: int = 100) -> HyperVector:
        """Encode scalar value using thermometer encoding."""
        # Quantize to level
        level = int(np.clip(value * levels, 0, levels - 1))

        if level not in self.levels:
            # Generate level vector
            self.levels[level] = HyperVector.from_seed(
                f"{self.name}:level:{level}", self.dim
            )

        # Bind level with base modality vector
        return bind(self.levels[level], self.base)

    def decode(self, hv: HyperVector, reference_vectors: List[HyperVector]) -> int:
        """
        Decode hypervector by finding closest reference.

        Returns index of best matching reference.
        """
        similarities = [similarity(hv, ref) for ref in reference_vectors]
        return int(np.argmax(similarities))


class SoulBundle:
    """
    The Soul Substrate: Unified multimodal identity.

    Bundles up to 12 modalities into holographic representation.
    T_s (topological similarity) preserved across fusion.
    """

    # Supported modalities
    MODALITIES = [
        "voice",        # AraVoice prosody
        "vision",       # Scene understanding
        "imu",          # Posture/motion
        "intero",       # Interoception (internal state)
        "hive",         # GPU/memory pressure
        "text",         # Language embeddings
        "audio",        # Raw audio features
        "emotion",      # Emotional vectors
        "motor",        # Motor commands
        "reward",       # Reward signals
        "memory",       # Memory traces
        "temporal",     # Time context
    ]

    MAX_MODALITIES = 12  # Practical limit before interference

    def __init__(self, dim: int = 16384):
        self.dim = dim
        self.encoders: Dict[str, ModalityEncoder] = {}
        self.modality_vectors: Dict[str, HyperVector] = {}
        self.soul: Optional[HyperVector] = None
        self.active_modalities: List[str] = []

        # Initialize encoders for all modalities
        for mod in self.MODALITIES:
            self.encoders[mod] = ModalityEncoder(mod, dim)

    def update_modality(self, modality: str, features: np.ndarray):
        """Update a single modality's encoding."""
        if modality not in self.encoders:
            raise ValueError(f"Unknown modality: {modality}")

        self.modality_vectors[modality] = self.encoders[modality].encode(features)

        if modality not in self.active_modalities:
            self.active_modalities.append(modality)

        # Rebuild soul bundle
        self._rebuild_soul()

    def _rebuild_soul(self):
        """Rebuild the unified soul bundle from active modalities."""
        if not self.modality_vectors:
            self.soul = None
            return

        vectors = list(self.modality_vectors.values())

        if len(vectors) > self.MAX_MODALITIES:
            # Warning: interference may occur
            vectors = vectors[:self.MAX_MODALITIES]

        self.soul = bundle(vectors)

    def get_soul(self) -> Optional[HyperVector]:
        """Get the current unified soul vector."""
        return self.soul

    def similarity_to(self, other: 'SoulBundle') -> float:
        """Compute soul-to-soul similarity (T_s analog)."""
        if self.soul is None or other.soul is None:
            return 0.0
        return similarity(self.soul, other.soul)

    def modality_contribution(self, modality: str) -> float:
        """
        Measure how much a modality contributes to the soul.

        Uses binding and similarity to extract.
        """
        if modality not in self.modality_vectors or self.soul is None:
            return 0.0

        # Unbind modality and measure residual similarity
        mod_vec = self.modality_vectors[modality]
        unbound = bind(self.soul, mod_vec)  # Self-inverse of bind

        # High similarity to modality base means high contribution
        return similarity(unbound, self.encoders[modality].base)

    def interference_level(self) -> float:
        """
        Measure interference between bundled modalities.

        High value = approaching capacity limit.
        """
        if len(self.active_modalities) < 2:
            return 0.0

        # Measure cross-similarities between modality vectors
        vectors = list(self.modality_vectors.values())
        cross_sims = []

        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                cross_sims.append(abs(similarity(vectors[i], vectors[j])))

        # Theoretical quasi-orthogonal similarity ≈ 0
        # As we approach capacity, cross-similarity increases
        return float(np.mean(cross_sims))

    def health_metrics(self) -> Dict[str, Any]:
        """Get health metrics for Cathedral monitoring."""
        interference = self.interference_level()

        return {
            "active_modalities": len(self.active_modalities),
            "max_modalities": self.MAX_MODALITIES,
            "utilization": len(self.active_modalities) / self.MAX_MODALITIES,
            "interference": interference,
            "interference_ok": interference < 0.1,  # Below 10%
            "has_soul": self.soul is not None,
            "modalities": self.active_modalities,
        }

    def to_bytes(self) -> bytes:
        """Serialize soul to compact format."""
        if self.soul is None:
            return b''
        return self.soul.to_bytes()

    def save(self, path: str):
        """Save soul bundle to file."""
        import json
        with open(path, 'wb') as f:
            # Write modality vectors
            data = {
                "dim": self.dim,
                "active": self.active_modalities,
            }
            header = json.dumps(data).encode()
            f.write(len(header).to_bytes(4, 'little'))
            f.write(header)

            # Write soul vector
            if self.soul:
                soul_bytes = self.soul.to_bytes()
                f.write(len(soul_bytes).to_bytes(4, 'little'))
                f.write(soul_bytes)


class VSASpace:
    """
    Complete VSA (Vector Symbolic Architecture) space.

    Manages symbol → hypervector mappings and operations.
    """

    def __init__(self, dim: int = 16384):
        self.dim = dim
        self.symbols: Dict[str, HyperVector] = {}
        self.encoders: Dict[str, ModalityEncoder] = {}

    def get_symbol(self, name: str) -> HyperVector:
        """Get or create symbol hypervector."""
        if name not in self.symbols:
            self.symbols[name] = HyperVector.from_seed(name, self.dim)
        return self.symbols[name]

    def encode(self, modality: str, features: np.ndarray) -> HyperVector:
        """Encode features from a modality."""
        if modality not in self.encoders:
            self.encoders[modality] = ModalityEncoder(modality, self.dim)
        return self.encoders[modality].encode(features)

    def create_sequence(self, items: List[str]) -> HyperVector:
        """
        Encode a sequence of symbols.

        Uses permutation to encode position.
        """
        if not items:
            return HyperVector.random(self.dim)

        # Encode each item with its position
        encoded = []
        for i, item in enumerate(items):
            symbol = self.get_symbol(item)
            positioned = permute(symbol, i)
            encoded.append(positioned)

        return bundle(encoded)

    def create_record(self, fields: Dict[str, str]) -> HyperVector:
        """
        Encode a structured record (key-value pairs).

        Uses binding for key-value association.
        """
        if not fields:
            return HyperVector.random(self.dim)

        bindings = []
        for key, value in fields.items():
            key_hv = self.get_symbol(key)
            val_hv = self.get_symbol(value)
            bindings.append(bind(key_hv, val_hv))

        return bundle(bindings)

    def query_record(self, record: HyperVector, key: str) -> Tuple[str, float]:
        """
        Query a record for a key's value.

        Returns (best_match_symbol, similarity_score).
        """
        key_hv = self.get_symbol(key)
        unbound = bind(record, key_hv)

        # Find best matching symbol
        best_symbol = None
        best_sim = -1.0

        for name, sym_hv in self.symbols.items():
            sim = similarity(unbound, sym_hv)
            if sim > best_sim:
                best_sim = sim
                best_symbol = name

        return best_symbol, best_sim


# =============================================================================
# SINGLETON AND CONVENIENCE
# =============================================================================

_vsa_space: Optional[VSASpace] = None
_soul_bundle: Optional[SoulBundle] = None


def get_vsa_space(dim: int = 16384) -> VSASpace:
    """Get the global VSA space instance."""
    global _vsa_space
    if _vsa_space is None:
        _vsa_space = VSASpace(dim)
    return _vsa_space


def encode_modality(modality: str, features: np.ndarray) -> HyperVector:
    """Encode features from a modality."""
    return get_vsa_space().encode(modality, features)


def create_soul_bundle(dim: int = 16384) -> SoulBundle:
    """Get or create the global soul bundle."""
    global _soul_bundle
    if _soul_bundle is None:
        _soul_bundle = SoulBundle(dim)
    return _soul_bundle
