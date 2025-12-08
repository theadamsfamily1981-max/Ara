"""
Hyperdimensional Computing (HDC) Codec
======================================

The symbolic computation layer for the Holographic Spike architecture.

HDC represents concepts as high-dimensional binary vectors (hypervectors).
Operations on these vectors implement symbolic reasoning:

    BIND (⊗):    A ⊗ B = A XOR B
                 Combines two concepts into a bound pair
                 "Red" ⊗ "Apple" = "Red Apple"

    BUNDLE (+):  A + B + C → majority vote
                 Superimposes concepts (like set union)
                 "Apple" + "Banana" + "Orange" = "Fruits"

    PERMUTE (ρ): ρ(A) = circular shift
                 Creates sequences / positional encoding
                 ρ("Word1") + ρρ("Word2") = sequence

    SIMILARITY:  cos(A, B) or Hamming distance
                 How related are two concepts?

The magic: All operations preserve similarity relationships!
If A ≈ B, then A ⊗ C ≈ B ⊗ C (binding preserves similarity).

Usage:
    from ara.cognition.hdc_codec import HDCCodec

    codec = HDCCodec(dim=8192)

    # Create atomic concepts
    red = codec.random_hv("red")
    apple = codec.random_hv("apple")

    # Bind them
    red_apple = codec.bind(red, apple)

    # Query: "What is bound with apple?"
    query = codec.bind(red_apple, apple)  # Should resemble 'red'
    similarity = codec.similarity(query, red)  # High!

    # Bundle multiple items
    fruits = codec.bundle([apple, codec.random_hv("banana"), codec.random_hv("orange")])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class ConceptRecord:
    """Record of a concept in the item memory."""
    name: str
    vector: np.ndarray
    created_at: float
    access_count: int = 0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class HDCCodec:
    """
    Hyperdimensional Computing encoder/decoder.

    Provides the full HDC algebra:
        - Item Memory: Named hypervector storage
        - Bind: XOR operation for role-filler binding
        - Bundle: Majority vote for superposition
        - Permute: Circular shift for sequence encoding
        - Similarity: Cosine / Hamming distance

    This is the "symbolic" side of the Holographic Spike -
    the same vectors that flow through the SNN fabric.
    """

    def __init__(
        self,
        dim: int = 8192,
        seed: Optional[int] = None,
    ):
        """
        Initialize the HDC codec.

        Args:
            dim: Hypervector dimension (should be large, e.g., 8192)
            seed: Random seed for reproducibility
        """
        self.dim = dim
        self.rng = np.random.default_rng(seed)

        # Item Memory: named hypervectors
        self.item_memory: Dict[str, ConceptRecord] = {}

        # Level hypervectors for encoding scalars
        self._level_hvs: Optional[np.ndarray] = None

        log.info(f"HDCCodec initialized: dim={dim}")

    # =========================================================================
    # BASIC OPERATIONS
    # =========================================================================

    def random_hv(self, name: Optional[str] = None) -> np.ndarray:
        """
        Generate a random hypervector.

        Args:
            name: If provided, store in item memory

        Returns:
            Binary hypervector of shape (dim,) with values {0, 1}
        """
        hv = self.rng.integers(0, 2, size=self.dim, dtype=np.uint8)

        if name and name not in self.item_memory:
            import time
            self.item_memory[name] = ConceptRecord(
                name=name,
                vector=hv.copy(),
                created_at=time.time(),
            )
            log.debug(f"HDC: Created concept '{name}'")

        return hv

    def get_or_create(self, name: str) -> np.ndarray:
        """Get existing hypervector or create new one."""
        if name in self.item_memory:
            record = self.item_memory[name]
            record.access_count += 1
            return record.vector.copy()
        return self.random_hv(name)

    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Bind two hypervectors: A ⊗ B = A XOR B

        Binding creates associations:
            "role" ⊗ "filler" = "role-filler pair"

        Properties:
            - Self-inverse: A ⊗ A = 0 (identity)
            - Commutative: A ⊗ B = B ⊗ A
            - Preserves similarity

        Args:
            a, b: Hypervectors to bind

        Returns:
            Bound hypervector
        """
        return np.bitwise_xor(a.astype(np.uint8), b.astype(np.uint8))

    def unbind(self, bound: np.ndarray, key: np.ndarray) -> np.ndarray:
        """
        Unbind to recover the other component.

        If bound = A ⊗ B, then:
            unbind(bound, A) ≈ B
            unbind(bound, B) ≈ A

        XOR is self-inverse, so unbind = bind.
        """
        return self.bind(bound, key)

    def bundle(self, vectors: List[np.ndarray]) -> np.ndarray:
        """
        Bundle multiple hypervectors: majority vote.

        Bundling creates superposition (set union):
            bundle([A, B, C]) ≈ "something like A, B, or C"

        Properties:
            - Result is similar to all inputs
            - Order doesn't matter
            - Capacity: sqrt(D) items before confusion

        Args:
            vectors: List of hypervectors to bundle

        Returns:
            Bundled hypervector
        """
        if not vectors:
            return np.zeros(self.dim, dtype=np.uint8)

        # Convert to bipolar for voting
        bipolar = [np.where(v > 0.5, 1, -1) for v in vectors]
        summed = np.sum(bipolar, axis=0)

        # Majority vote (break ties randomly)
        ties = (summed == 0)
        result = (summed > 0).astype(np.uint8)
        result[ties] = self.rng.integers(0, 2, size=np.sum(ties), dtype=np.uint8)

        return result

    def permute(self, v: np.ndarray, shift: int = 1) -> np.ndarray:
        """
        Permute (circular shift) a hypervector.

        Permutation creates sequence/position encoding:
            ρ("word1") + ρρ("word2") + ρρρ("word3") = sentence

        Args:
            v: Hypervector to permute
            shift: Number of positions to shift (default: 1)

        Returns:
            Permuted hypervector
        """
        return np.roll(v, shift)

    def inverse_permute(self, v: np.ndarray, shift: int = 1) -> np.ndarray:
        """Inverse permutation (shift in opposite direction)."""
        return np.roll(v, -shift)

    # =========================================================================
    # SIMILARITY MEASURES
    # =========================================================================

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute similarity between two hypervectors.

        Uses normalized Hamming similarity:
            sim = 1 - (hamming_distance / dim)

        Returns value in [0, 1] where:
            1.0 = identical
            0.5 = orthogonal (random)
            0.0 = anti-correlated

        Args:
            a, b: Hypervectors to compare

        Returns:
            Similarity score in [0, 1]
        """
        hamming = np.sum(a != b)
        return 1.0 - (hamming / self.dim)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine similarity (bipolar interpretation).

        Converts {0,1} to {-1,+1} then computes cos(a, b).
        """
        a_bipolar = np.where(a > 0.5, 1.0, -1.0)
        b_bipolar = np.where(b > 0.5, 1.0, -1.0)

        dot = np.dot(a_bipolar, b_bipolar)
        return float(dot / self.dim)

    def hamming_distance(self, a: np.ndarray, b: np.ndarray) -> int:
        """Compute Hamming distance (number of differing bits)."""
        return int(np.sum(a != b))

    # =========================================================================
    # ENCODING PRIMITIVES
    # =========================================================================

    def encode_sequence(self, items: List[str]) -> np.ndarray:
        """
        Encode an ordered sequence of items.

        Uses permutation to encode position:
            encode(["a", "b", "c"]) = ρ⁰(a) + ρ¹(b) + ρ²(c)

        Args:
            items: List of item names (will be looked up / created)

        Returns:
            Sequence hypervector
        """
        if not items:
            return np.zeros(self.dim, dtype=np.uint8)

        vectors = []
        for i, item in enumerate(items):
            hv = self.get_or_create(item)
            hv_shifted = self.permute(hv, shift=i)
            vectors.append(hv_shifted)

        return self.bundle(vectors)

    def encode_set(self, items: List[str]) -> np.ndarray:
        """
        Encode an unordered set of items.

        Simply bundles without permutation:
            encode_set(["a", "b", "c"]) = a + b + c

        Args:
            items: List of item names

        Returns:
            Set hypervector
        """
        vectors = [self.get_or_create(item) for item in items]
        return self.bundle(vectors)

    def encode_key_value(self, key: str, value: str) -> np.ndarray:
        """
        Encode a key-value pair.

        Uses binding: key ⊗ value

        Args:
            key: Key/role name
            value: Value/filler name

        Returns:
            Bound hypervector
        """
        k_hv = self.get_or_create(key)
        v_hv = self.get_or_create(value)
        return self.bind(k_hv, v_hv)

    def encode_record(self, **fields) -> np.ndarray:
        """
        Encode a record with multiple fields.

        Example:
            encode_record(color="red", fruit="apple", size="large")
            = (color ⊗ red) + (fruit ⊗ apple) + (size ⊗ large)

        Args:
            **fields: Field name/value pairs

        Returns:
            Record hypervector
        """
        bindings = []
        for key, value in fields.items():
            binding = self.encode_key_value(key, value)
            bindings.append(binding)

        return self.bundle(bindings)

    def encode_scalar(
        self,
        value: float,
        min_val: float = 0.0,
        max_val: float = 1.0,
        n_levels: int = 32,
    ) -> np.ndarray:
        """
        Encode a scalar value using level hypervectors.

        Creates a thermometer-like encoding where
        similar values have similar hypervectors.

        Args:
            value: Scalar to encode
            min_val, max_val: Value range
            n_levels: Number of quantization levels

        Returns:
            Level hypervector
        """
        # Initialize level HVs if needed
        if self._level_hvs is None or len(self._level_hvs) != n_levels:
            self._init_level_hvs(n_levels)

        # Clamp and quantize
        normalized = (value - min_val) / (max_val - min_val)
        normalized = np.clip(normalized, 0, 1)
        level = int(normalized * (n_levels - 1))

        return self._level_hvs[level].copy()

    def _init_level_hvs(self, n_levels: int) -> None:
        """Initialize level hypervectors with gradual similarity."""
        # Start with random base
        base = self.rng.integers(0, 2, size=self.dim, dtype=np.uint8)

        levels = [base.copy()]
        flip_per_level = self.dim // (n_levels * 2)

        for i in range(1, n_levels):
            prev = levels[-1].copy()
            # Flip some bits to gradually change
            flip_indices = self.rng.choice(self.dim, size=flip_per_level, replace=False)
            prev[flip_indices] = 1 - prev[flip_indices]
            levels.append(prev)

        self._level_hvs = np.array(levels)

    # =========================================================================
    # DECODING / QUERY
    # =========================================================================

    def query_item_memory(
        self,
        query: np.ndarray,
        k: int = 5,
        threshold: float = 0.6,
    ) -> List[Tuple[str, float]]:
        """
        Find items in memory most similar to query.

        Args:
            query: Query hypervector
            k: Number of results
            threshold: Minimum similarity

        Returns:
            List of (name, similarity) tuples
        """
        results = []

        for name, record in self.item_memory.items():
            sim = self.similarity(query, record.vector)
            if sim >= threshold:
                results.append((name, sim))

        results.sort(key=lambda x: -x[1])
        return results[:k]

    def decode_key_value(
        self,
        bound: np.ndarray,
        key: str,
    ) -> List[Tuple[str, float]]:
        """
        Decode a key-value pair to find the value.

        Args:
            bound: Bound hypervector (key ⊗ value)
            key: Known key

        Returns:
            Candidate values with similarities
        """
        k_hv = self.get_or_create(key)
        unbound = self.unbind(bound, k_hv)
        return self.query_item_memory(unbound)

    # =========================================================================
    # INTEGRATION WITH BINARY FRONT-END
    # =========================================================================

    def to_packed(self, hv: np.ndarray) -> np.ndarray:
        """Pack hypervector into uint64 words for binary ops."""
        # Pad to multiple of 64
        n_words = (self.dim + 63) // 64
        padded = np.zeros(n_words * 64, dtype=np.uint8)
        padded[:self.dim] = hv

        # Pack into uint64
        packed = np.zeros(n_words, dtype=np.uint64)
        for w in range(n_words):
            word_bits = padded[w * 64:(w + 1) * 64]
            for i, bit in enumerate(word_bits):
                if bit:
                    packed[w] |= np.uint64(1 << i)

        return packed

    def from_packed(self, packed: np.ndarray) -> np.ndarray:
        """Unpack uint64 words back to hypervector."""
        n_words = len(packed)
        unpacked = np.zeros(n_words * 64, dtype=np.uint8)

        for w in range(n_words):
            word = packed[w]
            for i in range(64):
                unpacked[w * 64 + i] = (word >> i) & 1

        return unpacked[:self.dim]

    def bind_packed(
        self,
        a_packed: np.ndarray,
        b_packed: np.ndarray,
    ) -> np.ndarray:
        """Bind packed hypervectors (fast XOR)."""
        return np.bitwise_xor(a_packed, b_packed)


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_global_codec: Optional[HDCCodec] = None


def get_hdc_codec(dim: int = 8192) -> HDCCodec:
    """Get or create the global HDC codec."""
    global _global_codec
    if _global_codec is None:
        _global_codec = HDCCodec(dim=dim)
    return _global_codec


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'ConceptRecord',
    'HDCCodec',
    'get_hdc_codec',
]
