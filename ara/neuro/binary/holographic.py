"""
Holographic Integration Layer
==============================

Bridges the Binary Neural Network stack with Hyperdimensional Computing.

This module unifies:
    - BinaryFrontEnd: XNOR+popcount encoder (cheap, massive)
    - HDCCodec: Symbolic hypervector operations (bind, bundle, permute)
    - ResonantStream: Rolling superposition memory (infinite context)

The key insight: A hypervector streamed bit-by-bit IS a spike train.
Space becomes Time. The same neurons do both SNN and HDC.

Pipeline:
    Raw Input → BinaryFrontEnd → Binary Code → HDCCodec → Symbolic Vector
                                     ↓                          ↓
                            BinaryMemory           ResonantStream
                            (pattern recall)    (infinite context)

Usage:
    from ara.neuro.binary.holographic import HolographicProcessor

    proc = HolographicProcessor(dim=8192)

    # Encode sensory input to hypervector
    hv = proc.encode_sensory(image_tensor)

    # Add to context stream
    proc.add_to_context(hv, label="saw_red_apple")

    # Query: "Have I seen something like this before?"
    resonance = proc.query_context(hv)

    # Symbolic binding: combine concepts
    red = proc.get_concept("red")
    apple = proc.get_concept("apple")
    red_apple = proc.bind(red, apple)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ara.cognition.stream_binding import ResonantStream, get_resonant_stream
from ara.cognition.hdc_codec import HDCCodec, get_hdc_codec
from ara.neuro.binary.memory import BinaryMemory, MoodCodeMemory

log = logging.getLogger(__name__)

# Check for torch
try:
    import torch
    from ara.neuro.binary.frontend import BinaryFrontEnd
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    BinaryFrontEnd = None


@dataclass
class HolographicConfig:
    """Configuration for HolographicProcessor."""
    hv_dim: int = 8192                  # Hypervector dimension
    encoder_input_dim: int = 1024       # BinaryFrontEnd input
    encoder_output_dim: int = 512       # BinaryFrontEnd output (code)
    memory_capacity: int = 100000       # BinaryMemory capacity
    stream_decay: float = 0.9           # ResonantStream decay rate


class HolographicProcessor:
    """
    Unified processor for holographic neural computation.

    Combines:
        1. BinaryFrontEnd: Sensory → Binary code (XNOR+popcount)
        2. HDCCodec: Symbolic operations (bind, bundle, permute)
        3. ResonantStream: Infinite context via rolling superposition
        4. BinaryMemory: Fast pattern recall via Hamming distance

    Modes:
        - SNN Mode: Pattern recognition, sensory encoding
        - HDC Mode: Symbolic reasoning, concept binding
        - Hybrid: Both simultaneously (different time slices)
    """

    def __init__(
        self,
        config: Optional[HolographicConfig] = None,
        dim: int = 8192,
    ):
        """
        Initialize the holographic processor.

        Args:
            config: Full configuration (overrides dim)
            dim: Hypervector dimension (if config not provided)
        """
        if config is None:
            config = HolographicConfig(hv_dim=dim)

        self.config = config
        self.dim = config.hv_dim

        # Components
        self._encoder: Optional[Any] = None  # BinaryFrontEnd (lazy init)
        self._codec = HDCCodec(dim=config.hv_dim)
        self._stream = ResonantStream(dim=config.hv_dim, decay_rate=config.stream_decay)
        self._memory = BinaryMemory(code_dim=config.encoder_output_dim,
                                     capacity=config.memory_capacity)
        self._mood_memory = MoodCodeMemory(code_dim=256)

        # Statistics
        self.encode_count = 0
        self.bind_count = 0
        self.query_count = 0

        log.info(
            f"HolographicProcessor initialized: dim={config.hv_dim}, "
            f"encoder={config.encoder_input_dim}→{config.encoder_output_dim}"
        )

    @property
    def encoder(self):
        """Lazy-load BinaryFrontEnd (requires torch)."""
        if self._encoder is None and TORCH_AVAILABLE:
            self._encoder = BinaryFrontEnd(
                input_dim=self.config.encoder_input_dim,
                output_dim=self.config.encoder_output_dim,
            )
        return self._encoder

    # =========================================================================
    # SENSORY ENCODING (SNN Mode)
    # =========================================================================

    def encode_sensory(
        self,
        x: Union[np.ndarray, "torch.Tensor"],
        to_hypervector: bool = True,
    ) -> np.ndarray:
        """
        Encode sensory input to binary code (optionally expand to hypervector).

        Args:
            x: Sensory input tensor
            to_hypervector: If True, expand code to full hypervector dimension

        Returns:
            Binary code or hypervector
        """
        self.encode_count += 1

        if self.encoder is None:
            # Fallback: random projection
            log.warning("HolographicProcessor: No torch, using random projection")
            if isinstance(x, np.ndarray):
                flat = x.flatten()
            else:
                flat = np.array(x).flatten()

            # Simple hash-based projection
            code = np.zeros(self.config.encoder_output_dim, dtype=np.uint8)
            for i, val in enumerate(flat[:self.config.encoder_output_dim]):
                code[i] = 1 if val > 0 else 0

            if to_hypervector:
                return self._expand_to_hv(code)
            return code

        # Use BinaryFrontEnd
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32))

        with torch.no_grad():
            code, sketch = self.encoder(x)
            code_np = code.numpy()

        if to_hypervector:
            return self._expand_to_hv(code_np)

        return code_np

    def _expand_to_hv(self, code: np.ndarray) -> np.ndarray:
        """Expand binary code to full hypervector dimension."""
        if code.ndim == 1:
            code = code.reshape(1, -1)

        batch_size, code_dim = code.shape

        # Replicate and permute to fill hypervector
        n_reps = (self.dim + code_dim - 1) // code_dim
        expanded = np.zeros((batch_size, self.dim), dtype=np.uint8)

        for i in range(n_reps):
            start = i * code_dim
            end = min(start + code_dim, self.dim)
            length = end - start

            # Permute each replication for diversity
            shifted = np.roll(code, shift=i * 7, axis=1)
            expanded[:, start:end] = shifted[:, :length]

        if batch_size == 1:
            return expanded[0]
        return expanded

    # =========================================================================
    # SYMBOLIC OPERATIONS (HDC Mode)
    # =========================================================================

    def get_concept(self, name: str) -> np.ndarray:
        """Get or create a concept hypervector."""
        return self._codec.get_or_create(name)

    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Bind two hypervectors (XOR).

        "Red" ⊗ "Apple" = "Red Apple"
        """
        self.bind_count += 1
        return self._codec.bind(a, b)

    def unbind(self, bound: np.ndarray, key: np.ndarray) -> np.ndarray:
        """Unbind to recover the other component."""
        return self._codec.unbind(bound, key)

    def bundle(self, vectors: List[np.ndarray]) -> np.ndarray:
        """
        Bundle multiple hypervectors (majority vote).

        "Apple" + "Banana" + "Orange" ≈ "Fruits"
        """
        return self._codec.bundle(vectors)

    def permute(self, v: np.ndarray, shift: int = 1) -> np.ndarray:
        """Permute (circular shift) for sequence encoding."""
        return self._codec.permute(v, shift)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute similarity between hypervectors."""
        return self._codec.similarity(a, b)

    def encode_sequence(self, items: List[str]) -> np.ndarray:
        """Encode an ordered sequence."""
        return self._codec.encode_sequence(items)

    def encode_record(self, **fields) -> np.ndarray:
        """Encode a key-value record."""
        return self._codec.encode_record(**fields)

    # =========================================================================
    # CONTEXT STREAM (Rolling Superposition)
    # =========================================================================

    def add_to_context(
        self,
        vector: np.ndarray,
        label: str = "",
    ) -> float:
        """
        Add a thought/perception to the context stream.

        Returns pre-resonance (how familiar this was).
        """
        return self._stream.add_thought(vector, label=label)

    def query_context(self, vector: np.ndarray) -> float:
        """
        Query: "Does my context contain this concept?"

        Returns resonance score in [-1, 1].
        """
        self.query_count += 1
        return self._stream.query(vector)

    def get_context_activity(self) -> float:
        """Get current context stream activity level."""
        return self._stream.get_activity()

    def decay_context(self, factor: Optional[float] = None) -> None:
        """Apply decay to context (forgetting curve)."""
        self._stream.apply_decay(factor)

    def snapshot_context(self) -> np.ndarray:
        """Get snapshot of current context state."""
        return self._stream.snapshot()

    def restore_context(self, snapshot: np.ndarray) -> None:
        """Restore context from snapshot."""
        self._stream.load_snapshot(snapshot)

    # =========================================================================
    # PATTERN MEMORY (Fast Recall)
    # =========================================================================

    def store_pattern(
        self,
        code: np.ndarray,
        label: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Store a pattern in binary memory."""
        return self._memory.store(code, label=label, metadata=metadata or {})

    def recall_patterns(
        self,
        query: np.ndarray,
        k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Recall similar patterns from memory.

        Returns list of (label, similarity) tuples.
        """
        results = self._memory.query(query, k=k)
        return [(r.entry.label, r.similarity) for r in results]

    def is_familiar(self, code: np.ndarray, threshold: float = 0.8) -> bool:
        """Check if pattern is familiar (seen before)."""
        return self._memory.contains_similar(code, threshold=threshold)

    # =========================================================================
    # MOOD ENCODING (Emotional State Signatures)
    # =========================================================================

    def encode_mood(
        self,
        features: np.ndarray,
        mood: str,
        intensity: float = 1.0,
    ) -> np.ndarray:
        """
        Encode emotional state as binary mood code.

        Args:
            features: State features (will be encoded)
            mood: Mood category (flow, frustration, etc.)
            intensity: Mood intensity

        Returns:
            Mood code hypervector
        """
        # Encode features to code
        if features.shape[0] > 256:
            features = features[:256]

        code = (features > 0).astype(np.uint8)

        # Store in mood memory
        self._mood_memory.store_mood(code, mood=mood, intensity=intensity)

        # Also bind with mood concept for stream
        mood_concept = self.get_concept(f"mood_{mood}")
        return self.bind(self._expand_to_hv(code), mood_concept)

    def recognize_mood(
        self,
        features: np.ndarray,
        threshold: float = 0.7,
    ) -> Optional[Tuple[str, float]]:
        """
        Recognize mood from features.

        Returns (mood, confidence) or None.
        """
        code = (features[:256] > 0).astype(np.uint8) if features.shape[0] > 256 else features
        return self._mood_memory.recognize_mood(code, threshold=threshold)

    # =========================================================================
    # HYBRID OPERATIONS
    # =========================================================================

    def perceive_and_bind(
        self,
        sensory_input: np.ndarray,
        concept_name: str,
    ) -> np.ndarray:
        """
        Perceive sensory input and bind with a concept.

        Example: See a red thing, bind with "color" role.
            perceive_and_bind(red_image, "color") = "color:red_percept"
        """
        # Encode sensory input
        percept_hv = self.encode_sensory(sensory_input, to_hypervector=True)

        # Get concept
        concept_hv = self.get_concept(concept_name)

        # Bind
        return self.bind(percept_hv, concept_hv)

    def perceive_to_context(
        self,
        sensory_input: np.ndarray,
        label: str = "",
    ) -> Tuple[np.ndarray, float]:
        """
        Perceive input and add to context stream.

        Returns (hypervector, pre_resonance).
        """
        hv = self.encode_sensory(sensory_input, to_hypervector=True)
        resonance = self.add_to_context(hv, label=label)
        return hv, resonance

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics."""
        return {
            "encode_count": self.encode_count,
            "bind_count": self.bind_count,
            "query_count": self.query_count,
            "context_activity": self.get_context_activity(),
            "context_thoughts": self._stream.thought_count,
            "memory_size": self._memory.size,
            "mood_memories": len(self._mood_memory.entries),
            "concepts": len(self._codec.item_memory),
        }


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_global_processor: Optional[HolographicProcessor] = None


def get_holographic_processor(dim: int = 8192) -> HolographicProcessor:
    """Get or create the global holographic processor."""
    global _global_processor
    if _global_processor is None:
        _global_processor = HolographicProcessor(dim=dim)
    return _global_processor


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'HolographicConfig',
    'HolographicProcessor',
    'get_holographic_processor',
]
