"""
Hyperdimensional Encoder
========================

Encode telemetry, logs, and events into hypervectors (HPVs).

The encoder provides consistent, composable representations:
- Metrics → quantized level HPVs
- Logs → n-gram HPVs
- Time → phase-encoded HPVs
- Events → bound combinations

All encodings are D-dimensional binary or bipolar vectors
that can be combined via HDC operations (bind, bundle, permute).
"""

from __future__ import annotations
import numpy as np
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class HDEncoderConfig:
    """Configuration for the HD encoder."""
    dim: int = 1024                    # Hypervector dimension
    n_levels: int = 16                 # Quantization levels for metrics
    ngram_size: int = 3                # N-gram size for text
    time_phases: int = 24              # Time-of-day phases
    bipolar: bool = True               # Use {-1, +1} vs {0, 1}
    seed: int = 42                     # Random seed for reproducibility


class HDEncoder:
    """
    Encode various data types into hypervectors.

    Uses item memory (iM) for atomic concepts and
    compositional encoding for structured data.
    """

    def __init__(self, config: Optional[HDEncoderConfig] = None):
        self.cfg = config or HDEncoderConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

        # Item memories (learned/fixed base vectors)
        self._level_hvs: Optional[np.ndarray] = None      # [n_levels, dim]
        self._char_hvs: Dict[str, np.ndarray] = {}        # char → hv
        self._phase_hvs: Optional[np.ndarray] = None      # [time_phases, dim]
        self._field_hvs: Dict[str, np.ndarray] = {}       # field_name → hv

        self._init_level_hvs()
        self._init_phase_hvs()

    def _init_level_hvs(self):
        """Initialize level hypervectors for metric quantization."""
        # Create thermometer-like encoding: level k shares bits with k-1
        # This preserves similarity between adjacent levels
        dim = self.cfg.dim
        n = self.cfg.n_levels

        if self.cfg.bipolar:
            base = self.rng.choice([-1, 1], size=dim)
        else:
            base = self.rng.integers(0, 2, size=dim)

        self._level_hvs = np.zeros((n, dim), dtype=np.int8)
        self._level_hvs[0] = base

        # Each level flips a fraction of bits from previous
        flip_per_level = dim // (n * 2)
        for i in range(1, n):
            self._level_hvs[i] = self._level_hvs[i - 1].copy()
            flip_idx = self.rng.choice(dim, size=flip_per_level, replace=False)
            self._level_hvs[i, flip_idx] *= -1 if self.cfg.bipolar else (1 - self._level_hvs[i, flip_idx])

    def _init_phase_hvs(self):
        """Initialize time phase hypervectors."""
        dim = self.cfg.dim
        n = self.cfg.time_phases

        if self.cfg.bipolar:
            self._phase_hvs = self.rng.choice([-1, 1], size=(n, dim)).astype(np.int8)
        else:
            self._phase_hvs = self.rng.integers(0, 2, size=(n, dim), dtype=np.int8)

    def _get_char_hv(self, char: str) -> np.ndarray:
        """Get or create hypervector for a character."""
        if char not in self._char_hvs:
            # Deterministic from char code
            seed = ord(char) + self.cfg.seed
            rng = np.random.default_rng(seed)
            if self.cfg.bipolar:
                self._char_hvs[char] = rng.choice([-1, 1], size=self.cfg.dim).astype(np.int8)
            else:
                self._char_hvs[char] = rng.integers(0, 2, size=self.cfg.dim, dtype=np.int8)
        return self._char_hvs[char]

    def _get_field_hv(self, field: str) -> np.ndarray:
        """Get or create hypervector for a field name."""
        if field not in self._field_hvs:
            # Deterministic from field name hash
            h = int(hashlib.md5(field.encode()).hexdigest(), 16)
            rng = np.random.default_rng(h % (2**31))
            if self.cfg.bipolar:
                self._field_hvs[field] = rng.choice([-1, 1], size=self.cfg.dim).astype(np.int8)
            else:
                self._field_hvs[field] = rng.integers(0, 2, size=self.cfg.dim, dtype=np.int8)
        return self._field_hvs[field]

    # =========================================================================
    # Core HDC Operations
    # =========================================================================

    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Bind two hypervectors (XOR for binary, multiply for bipolar)."""
        if self.cfg.bipolar:
            return (a * b).astype(np.int8)
        else:
            return np.bitwise_xor(a, b).astype(np.int8)

    def bundle(self, hvs: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray:
        """Bundle (superpose) hypervectors with optional weights."""
        if not hvs:
            return np.zeros(self.cfg.dim, dtype=np.int8)

        if weights is None:
            weights = [1.0] * len(hvs)

        # Weighted sum
        result = np.zeros(self.cfg.dim, dtype=np.float32)
        for hv, w in zip(hvs, weights):
            result += w * hv.astype(np.float32)

        # Threshold back to binary/bipolar
        if self.cfg.bipolar:
            return np.sign(result).astype(np.int8)
        else:
            return (result > 0).astype(np.int8)

    def permute(self, hv: np.ndarray, k: int = 1) -> np.ndarray:
        """Permute hypervector (cyclic shift) for position encoding."""
        return np.roll(hv, k)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two hypervectors."""
        if self.cfg.bipolar:
            # For bipolar: cos = (a · b) / dim
            return float(np.dot(a.astype(np.float32), b.astype(np.float32))) / self.cfg.dim
        else:
            # For binary: Hamming similarity = 1 - (hamming_dist / dim)
            return 1.0 - float(np.sum(a != b)) / self.cfg.dim

    # =========================================================================
    # Encoding Methods
    # =========================================================================

    def encode_metric(self, value: float, min_val: float = 0.0, max_val: float = 1.0) -> np.ndarray:
        """
        Encode a numeric metric into a hypervector.

        Uses thermometer encoding: similar values have similar HPVs.
        """
        # Clamp and quantize
        normalized = (value - min_val) / (max_val - min_val + 1e-8)
        normalized = np.clip(normalized, 0, 1)
        level = int(normalized * (self.cfg.n_levels - 1))

        return self._level_hvs[level].copy()

    def encode_metrics(self, metrics: Dict[str, float],
                       ranges: Optional[Dict[str, Tuple[float, float]]] = None) -> np.ndarray:
        """
        Encode a dictionary of metrics into a single hypervector.

        Each field is bound with its value, then all are bundled.
        """
        if ranges is None:
            ranges = {}

        hvs = []
        for field, value in metrics.items():
            min_val, max_val = ranges.get(field, (0.0, 1.0))
            field_hv = self._get_field_hv(field)
            value_hv = self.encode_metric(value, min_val, max_val)
            hvs.append(self.bind(field_hv, value_hv))

        return self.bundle(hvs)

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text into a hypervector using n-gram encoding.

        Characters are position-encoded within each n-gram,
        then all n-grams are bundled.
        """
        if not text:
            return np.zeros(self.cfg.dim, dtype=np.int8)

        n = self.cfg.ngram_size
        ngram_hvs = []

        # Pad text
        padded = " " * (n - 1) + text.lower() + " " * (n - 1)

        for i in range(len(padded) - n + 1):
            ngram = padded[i:i + n]

            # Bind characters with position permutation
            ngram_hv = self._get_char_hv(ngram[0])
            for j in range(1, n):
                char_hv = self.permute(self._get_char_hv(ngram[j]), j)
                ngram_hv = self.bind(ngram_hv, char_hv)

            ngram_hvs.append(ngram_hv)

        return self.bundle(ngram_hvs)

    def encode_log(self, log_line: str) -> np.ndarray:
        """
        Encode a log line into a hypervector.

        Extracts key patterns: level, component, message signature.
        """
        # Simple: just use text encoding
        # Could be enhanced with structured extraction
        return self.encode_text(log_line)

    def encode_time_phase(self, hour: float) -> np.ndarray:
        """
        Encode time of day into a hypervector.

        Uses circular phase encoding (hour 23 is similar to hour 0).
        """
        # Map to phase index
        phase = int(hour * self.cfg.time_phases / 24) % self.cfg.time_phases

        # Blend with adjacent phases for smooth transitions
        phase_next = (phase + 1) % self.cfg.time_phases
        blend = (hour * self.cfg.time_phases / 24) % 1.0

        return self.bundle(
            [self._phase_hvs[phase], self._phase_hvs[phase_next]],
            weights=[1.0 - blend, blend]
        )

    def encode_event(self, event_type: str, data: Dict[str, Any],
                     timestamp_hour: Optional[float] = None) -> np.ndarray:
        """
        Encode a structured event into a hypervector.

        Combines event type, data fields, and optional time context.
        """
        hvs = []

        # Event type
        hvs.append(self.encode_text(event_type))

        # Data fields
        for key, value in data.items():
            field_hv = self._get_field_hv(key)
            if isinstance(value, (int, float)):
                value_hv = self.encode_metric(float(value))
            else:
                value_hv = self.encode_text(str(value))
            hvs.append(self.bind(field_hv, value_hv))

        # Time context
        if timestamp_hour is not None:
            hvs.append(self.encode_time_phase(timestamp_hour))

        return self.bundle(hvs)


# Convenience function
def create_encoder(dim: int = 1024) -> HDEncoder:
    """Create an HD encoder with the specified dimension."""
    return HDEncoder(HDEncoderConfig(dim=dim))
