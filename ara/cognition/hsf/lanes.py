"""
HSF Lane Encoders
==================

Each lane encodes one subsystem's telemetry as a hypervector.

Key concepts:
- ItemMemory: Random seed → consistent hypervector for each feature name
- LaneEncoder: Combines feature HVs with value encoding
- TelemetryLane: Full lane with history and baseline learning

Encoding scheme:
    For each feature (e.g., "gpu_temp"):
    1. Get base HV from item memory: base = ItemMemory["gpu_temp"]
    2. Encode value level: level_hv = permute(base, quantized_level)
    3. Bind feature + value: bound = base XOR level_hv
    4. Bundle all features: lane_hv = majority(bound_1, bound_2, ...)

This gives us:
- Holographic: can recover individual features via resonance
- Compositional: similar values → similar hypervectors
- Cheap: just XOR, shift, popcount
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
import hashlib


@dataclass
class ItemMemory:
    """
    Consistent random hypervector generation for feature names.

    Same name → same HV every time (seeded by name hash).
    This is the "symbol table" of HDC.
    """
    dim: int = 8192
    _cache: Dict[str, np.ndarray] = field(default_factory=dict)

    def __getitem__(self, name: str) -> np.ndarray:
        """Get or create hypervector for name."""
        if name not in self._cache:
            # Seed RNG with name hash for reproducibility
            seed = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            # Binary hypervector: {0, 1}^dim
            self._cache[name] = rng.integers(0, 2, size=self.dim, dtype=np.uint8)
        return self._cache[name]

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Cosine-ish similarity for binary vectors.

        Returns: 1.0 = identical, 0.0 = orthogonal (50% match), -1.0 = opposite
        """
        # Hamming similarity: (matches - mismatches) / dim
        matches = np.sum(a == b)
        return (2 * matches - self.dim) / self.dim

    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """XOR binding: A ⊗ B"""
        return np.bitwise_xor(a, b)

    def permute(self, hv: np.ndarray, shift: int) -> np.ndarray:
        """Cyclic permutation: ρ^k(A)"""
        return np.roll(hv, shift)

    def bundle(self, hvs: List[np.ndarray]) -> np.ndarray:
        """Majority vote bundling: A + B + C + ..."""
        if len(hvs) == 0:
            return np.zeros(self.dim, dtype=np.uint8)
        if len(hvs) == 1:
            return hvs[0].copy()
        # Sum and threshold at majority
        total = np.sum(hvs, axis=0)
        threshold = len(hvs) / 2
        # Tie-break randomly for even counts
        ties = (total == threshold)
        result = (total > threshold).astype(np.uint8)
        if np.any(ties):
            rng = np.random.default_rng()
            result[ties] = rng.integers(0, 2, size=np.sum(ties), dtype=np.uint8)
        return result


@dataclass
class LaneEncoder:
    """
    Encodes a dictionary of feature values into a single hypervector.

    Uses level encoding for continuous values:
    - Quantize value into N levels
    - Permute base HV by level index
    - Bind with base HV

    This gives graceful degradation: similar values → similar HVs.
    """
    item_memory: ItemMemory
    num_levels: int = 32  # Quantization levels for continuous values

    def encode_value(self, feature: str, value: float,
                     min_val: float = 0.0, max_val: float = 1.0) -> np.ndarray:
        """
        Encode a single feature-value pair.

        Returns: bound HV representing (feature, value)
        """
        base = self.item_memory[feature]

        # Normalize and quantize
        normalized = np.clip((value - min_val) / (max_val - min_val + 1e-10), 0, 1)
        level = int(normalized * (self.num_levels - 1))

        # Level encoding: permute by level, then bind
        level_hv = self.item_memory.permute(base, level)
        return self.item_memory.bind(base, level_hv)

    def encode_dict(self, values: Dict[str, float],
                    ranges: Optional[Dict[str, Tuple[float, float]]] = None) -> np.ndarray:
        """
        Encode a dictionary of feature values into one hypervector.

        Args:
            values: {feature_name: value}
            ranges: {feature_name: (min, max)} for normalization

        Returns: bundled HV representing all features
        """
        ranges = ranges or {}
        bound_hvs = []

        for feature, value in values.items():
            min_val, max_val = ranges.get(feature, (0.0, 1.0))
            bound_hv = self.encode_value(feature, value, min_val, max_val)
            bound_hvs.append(bound_hv)

        return self.item_memory.bundle(bound_hvs)


@dataclass
class TelemetryLane:
    """
    A complete lane for one subsystem's telemetry.

    Features:
    - Encodes current state as hypervector
    - Maintains rolling history for baseline learning
    - Computes "baseline" HV from history
    - Reports deviation from baseline
    """
    name: str
    features: List[str]
    dim: int = 8192
    history_size: int = 100

    # Internal state
    item_memory: ItemMemory = field(default_factory=lambda: ItemMemory())
    encoder: LaneEncoder = field(init=False)
    _history: deque = field(default_factory=lambda: deque(maxlen=100))
    _baseline: Optional[np.ndarray] = field(default=None)
    _current: Optional[np.ndarray] = field(default=None)
    _ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self):
        self.item_memory = ItemMemory(dim=self.dim)
        self.encoder = LaneEncoder(self.item_memory)
        self._history = deque(maxlen=self.history_size)

        # Initialize feature name HVs (for later querying)
        for feature in self.features:
            _ = self.item_memory[f"{self.name}:{feature}"]

    def set_ranges(self, ranges: Dict[str, Tuple[float, float]]):
        """Set normalization ranges for features."""
        self._ranges = {f"{self.name}:{k}": v for k, v in ranges.items()}

    def update(self, values: Dict[str, float]) -> np.ndarray:
        """
        Encode new telemetry values.

        Returns: current state hypervector
        """
        # Prefix feature names with lane name
        prefixed = {f"{self.name}:{k}": v for k, v in values.items()}

        # Encode
        self._current = self.encoder.encode_dict(prefixed, self._ranges)

        # Add to history
        self._history.append(self._current.copy())

        return self._current

    def compute_baseline(self) -> np.ndarray:
        """
        Compute baseline HV from history.

        This is the "what normal looks like" vector.
        """
        if len(self._history) == 0:
            self._baseline = np.zeros(self.dim, dtype=np.uint8)
        else:
            self._baseline = self.item_memory.bundle(list(self._history))
        return self._baseline

    def deviation(self) -> float:
        """
        How far is current state from baseline?

        Returns: 0.0 = exactly baseline, 1.0 = maximally different
        """
        if self._current is None or self._baseline is None:
            return 0.0

        sim = self.item_memory.similarity(self._current, self._baseline)
        # Convert from [-1, 1] to [0, 1] deviation
        return (1.0 - sim) / 2.0

    def query_feature(self, feature: str) -> float:
        """
        Query how much a specific feature is present in current state.

        Uses resonance: unbind feature HV, check similarity to level HVs.
        """
        if self._current is None:
            return 0.0

        feature_key = f"{self.name}:{feature}"
        base = self.item_memory[feature_key]

        # Unbind feature from current state
        unbound = self.item_memory.bind(self._current, base)

        # Check resonance with each level
        best_level = 0
        best_sim = -1.0
        for level in range(self.encoder.num_levels):
            level_hv = self.item_memory.permute(base, level)
            sim = self.item_memory.similarity(unbound, level_hv)
            if sim > best_sim:
                best_sim = sim
                best_level = level

        # Convert level back to normalized value
        return best_level / (self.encoder.num_levels - 1)

    @property
    def current(self) -> Optional[np.ndarray]:
        return self._current

    @property
    def baseline(self) -> Optional[np.ndarray]:
        return self._baseline
