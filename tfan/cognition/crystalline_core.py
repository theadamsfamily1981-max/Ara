"""
Crystalline Core - Hyperdimensional Computing for Ara
======================================================

This is NOT mystical. This is Vector Symbolic Architecture (VSA), a real
computational framework from Pentti Kanerva's work on sparse distributed memory.

Core idea:
    - Represent concepts as high-dimensional vectors (8192+ bits)
    - Operations on these vectors are:
        BIND (XOR/multiply): "A is related to B"
        BUNDLE (majority/sum): "Store A, B, C together"
        PERMUTE (shift): "A comes before B"
    - Similarity is just Hamming/cosine distance

Why this matters for Ara:
    - One-shot memory: encode an experience, add it to the bundle, done
    - Robust to noise: a few flipped bits don't destroy the memory
    - Hardware-friendly: XOR, popcount, shift are FPGA primitives
    - Compositional: you can query "things like X but with emotion Y"

This is the "wisdom backbone" - fast déjà-vu for the Historian.

References:
    - Kanerva, "Hyperdimensional Computing" (2009)
    - Rahimi et al., "Classification and Regression with Binary HDC" (2016)
    - Neubert et al., "Vector Symbolic Architectures as a Cognitive Model" (2019)
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import hashlib
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

# Default dimensionality - 8192 is a good balance of capacity vs. compute
# Can be 4096 for faster ops, or 10000 for more capacity
DEFAULT_DIM = 8192

# We use bipolar vectors (-1, +1) not binary (0, 1)
# Reason: math is cleaner, cosine similarity works directly
# Binary: XOR for bind, majority for bundle
# Bipolar: multiply for bind, sign(sum) for bundle


# =============================================================================
# Core HDC Operations
# =============================================================================

class HypervectorSpace:
    """
    A hyperdimensional vector space for symbolic computing.

    This is the "algebra" of the Crystalline Core:
    - Each concept gets a random hypervector
    - BIND links concepts together
    - BUNDLE combines multiple concepts
    - PERMUTE encodes order/sequence
    """

    def __init__(self, dim: int = DEFAULT_DIM, seed: Optional[int] = None):
        """
        Initialize a hypervector space.

        Args:
            dim: Dimensionality of vectors (8192 recommended)
            seed: Random seed for reproducibility
        """
        self.dim = dim
        self.rng = np.random.default_rng(seed)

        # Cache of concept -> hypervector mappings
        # Once a concept gets a vector, it keeps that vector forever
        self._concept_cache: Dict[str, np.ndarray] = {}

        # Pre-computed role vectors (these are "slots" in structured representations)
        self._role_vectors: Dict[str, np.ndarray] = {}

        # Permutation matrices for sequence encoding
        self._perm_forward: Optional[np.ndarray] = None
        self._perm_backward: Optional[np.ndarray] = None

        self._init_role_vectors()

    def _init_role_vectors(self) -> None:
        """Initialize standard role vectors for structured encoding."""
        # These are the "slots" we use to encode structured episodes
        roles = [
            "CONTEXT",      # What situation is this?
            "ACTION",       # What did we do?
            "OUTCOME",      # What happened?
            "EMOTION",      # How did it feel?
            "AGENT",        # Who was involved?
            "TIME",         # When (coarse: morning/evening/etc)
            "INTENSITY",    # How strong was this experience?
            "VALENCE",      # Positive or negative?
        ]
        for role in roles:
            self._role_vectors[role] = self._random_vector()

    def _random_vector(self) -> np.ndarray:
        """Generate a random bipolar hypervector."""
        # Bipolar: each element is -1 or +1 with equal probability
        return self.rng.choice([-1, 1], size=self.dim).astype(np.int8)

    def _hash_to_vector(self, key: str) -> np.ndarray:
        """
        Deterministically generate a vector from a string key.

        This ensures the same concept always maps to the same vector,
        even across restarts (if we use the same seed logic).
        """
        # Use SHA-256 to generate enough random bits
        h = hashlib.sha256(key.encode()).digest()

        # Expand hash to fill our dimension
        expanded = bytearray()
        counter = 0
        while len(expanded) < self.dim // 8:
            h2 = hashlib.sha256(key.encode() + counter.to_bytes(4, 'little')).digest()
            expanded.extend(h2)
            counter += 1

        # Convert bytes to bipolar
        bits = np.unpackbits(np.frombuffer(bytes(expanded[:self.dim // 8]), dtype=np.uint8))
        return (bits.astype(np.int8) * 2 - 1)[:self.dim]

    # =========================================================================
    # Core Operations
    # =========================================================================

    def get_vector(self, concept: str, deterministic: bool = True) -> np.ndarray:
        """
        Get the hypervector for a concept.

        First call for a concept generates its vector.
        Subsequent calls return the same vector.

        Args:
            concept: String identifier for the concept
            deterministic: If True, use hash-based generation (reproducible)
        """
        if concept in self._concept_cache:
            return self._concept_cache[concept]

        if deterministic:
            vec = self._hash_to_vector(concept)
        else:
            vec = self._random_vector()

        self._concept_cache[concept] = vec
        return vec

    def get_role(self, role: str) -> np.ndarray:
        """Get a role vector (for structured encoding)."""
        if role not in self._role_vectors:
            self._role_vectors[role] = self._random_vector()
        return self._role_vectors[role]

    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        BIND operation: create a vector representing "A related to B".

        For bipolar vectors, this is element-wise multiplication.
        Properties:
        - bind(A, B) is dissimilar to both A and B
        - bind(A, bind(A, B)) ≈ B (self-inverse, can "unbind")
        - bind(A, B) = bind(B, A) (commutative)

        Use case: "GPU" bound with "OVERHEAT" = concept of "GPU overheating"
        """
        return (a * b).astype(np.int8)

    def bundle(self, vectors: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray:
        """
        BUNDLE operation: combine multiple vectors into one.

        For bipolar vectors, this is sum + sign.
        Properties:
        - Result is similar to ALL inputs
        - Can store many concepts in one vector
        - Robust: a few corrupted inputs don't destroy the bundle

        Use case: Bundle [context, emotion, outcome] into one episode vector
        """
        if not vectors:
            return self._random_vector()

        if weights is None:
            weights = [1.0] * len(vectors)

        # Weighted sum
        total = np.zeros(self.dim, dtype=np.float32)
        for vec, w in zip(vectors, weights):
            total += vec.astype(np.float32) * w

        # Sign function (0 → random choice to break ties)
        result = np.sign(total)
        zeros = result == 0
        if zeros.any():
            result[zeros] = self.rng.choice([-1, 1], size=zeros.sum())

        return result.astype(np.int8)

    def permute(self, vec: np.ndarray, steps: int = 1) -> np.ndarray:
        """
        PERMUTE operation: shift vector to encode sequence position.

        permute(A, 1) means "A at position 1"
        permute(A, 2) means "A at position 2"

        Properties:
        - permute(A, k) is dissimilar to A (for k > 0)
        - Can encode sequences: bundle([permute(A,0), permute(B,1), permute(C,2)])

        Use case: Encode order of events in an episode
        """
        return np.roll(vec, steps)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute similarity between two hypervectors.

        Returns value in [-1, 1]:
        - 1.0 = identical
        - 0.0 = orthogonal (unrelated)
        - -1.0 = opposite

        For bipolar vectors, this is cosine similarity.
        """
        # Cosine similarity (vectors are already unit-ish since all elements are ±1)
        dot = np.dot(a.astype(np.float32), b.astype(np.float32))
        return float(dot / self.dim)

    def hamming_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Hamming-based similarity (for binary interpretation).

        Returns value in [0, 1]:
        - 1.0 = all bits match
        - 0.5 = random/unrelated
        - 0.0 = all bits opposite
        """
        # Convert bipolar to binary for hamming
        a_bin = (a + 1) // 2
        b_bin = (b + 1) // 2
        matches = np.sum(a_bin == b_bin)
        return float(matches / self.dim)

    # =========================================================================
    # Structured Encoding
    # =========================================================================

    def encode_pair(self, role: str, value: str) -> np.ndarray:
        """
        Encode a role-value pair.

        Example: encode_pair("EMOTION", "anxiety")
                 = bind(ROLE_EMOTION, vec("anxiety"))
        """
        role_vec = self.get_role(role)
        value_vec = self.get_vector(value)
        return self.bind(role_vec, value_vec)

    def encode_record(self, **kwargs) -> np.ndarray:
        """
        Encode a structured record as a hypervector.

        Example:
            encode_record(
                CONTEXT="investor_demo",
                EMOTION="stress",
                OUTCOME="success"
            )

        This bundles: bind(CONTEXT, "investor_demo") + bind(EMOTION, "stress") + ...
        """
        pairs = []
        for role, value in kwargs.items():
            pairs.append(self.encode_pair(role.upper(), str(value)))

        return self.bundle(pairs)

    def query_role(self, record: np.ndarray, role: str) -> np.ndarray:
        """
        Extract a role's value from an encoded record.

        Because bind is self-inverse:
        query_role(encode_record(EMOTION="joy"), "EMOTION") ≈ vec("joy")

        The result can then be compared to known concept vectors to decode.
        """
        role_vec = self.get_role(role)
        return self.bind(record, role_vec)

    def decode_role(self, record: np.ndarray, role: str, candidates: List[str]) -> Tuple[str, float]:
        """
        Decode what value a role has in a record.

        Args:
            record: An encoded record
            role: Which role to decode
            candidates: Possible values to check against

        Returns:
            (best_match, similarity_score)
        """
        extracted = self.query_role(record, role)

        best_match = None
        best_sim = -2.0

        for candidate in candidates:
            candidate_vec = self.get_vector(candidate)
            sim = self.similarity(extracted, candidate_vec)
            if sim > best_sim:
                best_sim = sim
                best_match = candidate

        return best_match, best_sim

    # =========================================================================
    # Persistence
    # =========================================================================

    def export_state(self) -> Dict:
        """Export state for persistence."""
        return {
            'dim': self.dim,
            'concepts': {k: v.tolist() for k, v in self._concept_cache.items()},
            'roles': {k: v.tolist() for k, v in self._role_vectors.items()},
        }

    def import_state(self, state: Dict) -> None:
        """Import state from persistence."""
        if state.get('dim') != self.dim:
            raise ValueError(f"Dimension mismatch: {state.get('dim')} vs {self.dim}")

        for k, v in state.get('concepts', {}).items():
            self._concept_cache[k] = np.array(v, dtype=np.int8)
        for k, v in state.get('roles', {}).items():
            self._role_vectors[k] = np.array(v, dtype=np.int8)


# =============================================================================
# Convenience Functions
# =============================================================================

# Global default space (can be replaced with custom instance)
_default_space: Optional[HypervectorSpace] = None


def get_default_space() -> HypervectorSpace:
    """Get the default hypervector space (lazily initialized)."""
    global _default_space
    if _default_space is None:
        _default_space = HypervectorSpace(seed=42)  # Reproducible default
    return _default_space


def vec(concept: str) -> np.ndarray:
    """Shorthand: get vector for a concept."""
    return get_default_space().get_vector(concept)


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Shorthand: bind two vectors."""
    return get_default_space().bind(a, b)


def bundle(vectors: List[np.ndarray]) -> np.ndarray:
    """Shorthand: bundle vectors."""
    return get_default_space().bundle(vectors)


def sim(a: np.ndarray, b: np.ndarray) -> float:
    """Shorthand: similarity between vectors."""
    return get_default_space().similarity(a, b)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'HypervectorSpace',
    'DEFAULT_DIM',
    'get_default_space',
    'vec',
    'bind',
    'bundle',
    'sim',
]
