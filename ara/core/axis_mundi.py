"""
AxisMundi: Global Holographic State

The central nervous system of Ara - a hyperdimensional vector space that
unifies all layers of the organism into coherent global state.

Each "layer" (hardware, avatar, mission, emotion, etc.) has its own
random key vector. State is written by binding (XOR/multiply) with the key,
and read by unbinding with the same key.

Usage:
    axis = AxisMundi(dim=4096, layers=["hardware", "avatar", "mission"])

    # Write state to a layer
    axis.write("avatar", avatar_state_hv)

    # Read state from a layer
    world_from_avatar = axis.read("avatar")

    # Measure coherence between layers
    coherence = axis.coherence("hardware", "mission")

    # Get global state vector
    global_state = axis.global_state()
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from threading import RLock
import json
from pathlib import Path


@dataclass
class LayerState:
    """State for a single layer in AxisMundi."""
    name: str
    key: np.ndarray  # Random key vector for binding/unbinding
    value: np.ndarray  # Current bound state (key ⊗ raw_value)
    raw_value: np.ndarray  # Unbound state (for direct comparison)
    write_count: int = 0
    last_coherence: float = 1.0


class AxisMundi:
    """
    Global holographic state manager.

    Uses hyperdimensional computing to maintain a unified world-state
    where all layers can be queried, compared, and coherence-checked.

    Mathematical basis:
    - Write: state[layer] = layer_key ⊗ value (binding via XOR for binary, multiply for bipolar)
    - Read: unbind(state[layer], layer_key) ≈ value
    - Coherence: cosine_similarity(read(layer_a), read(layer_b))
    """

    def __init__(
        self,
        dim: int = 4096,
        layers: Optional[List[str]] = None,
        bipolar: bool = True,
        seed: Optional[int] = None,
    ):
        """
        Initialize AxisMundi.

        Args:
            dim: Dimensionality of hypervectors (power of 2 recommended)
            layers: List of layer names to initialize
            bipolar: If True, use {-1, +1} vectors; if False, use {0, 1}
            seed: Random seed for reproducible key generation
        """
        self.dim = dim
        self.bipolar = bipolar
        self._rng = np.random.default_rng(seed)
        self._lock = RLock()

        # Layer storage
        self._layers: Dict[str, LayerState] = {}

        # Global accumulator (sum of all bound states)
        self._global: np.ndarray = np.zeros(dim, dtype=np.float32)

        # Initialize default layers
        default_layers = layers or ["hardware", "avatar", "mission", "emotion"]
        for layer_name in default_layers:
            self._init_layer(layer_name)

    def _init_layer(self, name: str) -> None:
        """Initialize a new layer with a random key."""
        if self.bipolar:
            key = self._rng.choice([-1.0, 1.0], size=self.dim).astype(np.float32)
        else:
            key = self._rng.choice([0.0, 1.0], size=self.dim).astype(np.float32)

        # Initialize with zero state
        zero_state = np.zeros(self.dim, dtype=np.float32)

        self._layers[name] = LayerState(
            name=name,
            key=key,
            value=zero_state.copy(),
            raw_value=zero_state.copy(),
        )

    def _bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Bind two vectors (XOR for binary, element-wise multiply for bipolar)."""
        if self.bipolar:
            return a * b
        else:
            return np.logical_xor(a > 0.5, b > 0.5).astype(np.float32)

    def _unbind(self, bound: np.ndarray, key: np.ndarray) -> np.ndarray:
        """Unbind a vector with its key (inverse of bind)."""
        # For bipolar: multiply is self-inverse (key * key = 1)
        # For binary XOR: XOR is self-inverse
        return self._bind(bound, key)

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        """Normalize vector to unit length."""
        norm = np.linalg.norm(v)
        if norm < 1e-10:
            return v
        return v / norm

    def write(self, layer: str, value: np.ndarray, accumulate: bool = True) -> None:
        """
        Write state to a layer.

        Args:
            layer: Layer name
            value: Hypervector to store (will be bound with layer key)
            accumulate: If True, add to global state; if False, just store locally
        """
        with self._lock:
            if layer not in self._layers:
                self._init_layer(layer)

            state = self._layers[layer]

            # Normalize input
            value = self._normalize(value.astype(np.float32))

            # Remove old contribution from global state
            if accumulate and state.write_count > 0:
                self._global -= state.value

            # Bind value with layer key
            bound = self._bind(state.key, value)

            # Store both bound and raw
            state.value = bound
            state.raw_value = value.copy()
            state.write_count += 1

            # Add new contribution to global state
            if accumulate:
                self._global += bound

    def read(self, layer: str) -> np.ndarray:
        """
        Read state from a layer.

        Args:
            layer: Layer name

        Returns:
            Unbound state vector (normalized)
        """
        with self._lock:
            if layer not in self._layers:
                return np.zeros(self.dim, dtype=np.float32)

            state = self._layers[layer]
            unbound = self._unbind(state.value, state.key)
            return self._normalize(unbound)

    def read_raw(self, layer: str) -> np.ndarray:
        """Read the raw (never-bound) value from a layer."""
        with self._lock:
            if layer not in self._layers:
                return np.zeros(self.dim, dtype=np.float32)
            return self._layers[layer].raw_value.copy()

    def coherence(self, layer_a: str, layer_b: str) -> float:
        """
        Measure coherence between two layers.

        Uses cosine similarity of the unbound states.
        Returns value in [-1, 1], where 1 = perfectly aligned.
        """
        with self._lock:
            if layer_a not in self._layers or layer_b not in self._layers:
                return 0.0

            a = self.read(layer_a)
            b = self.read(layer_b)

            # Cosine similarity
            dot = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)

            if norm_a < 1e-10 or norm_b < 1e-10:
                return 0.0

            coherence = float(dot / (norm_a * norm_b))

            # Cache for metrics
            self._layers[layer_a].last_coherence = coherence
            self._layers[layer_b].last_coherence = coherence

            return coherence

    def global_state(self) -> np.ndarray:
        """
        Get the global state vector (superposition of all layers).

        This is the holographic "world model" that contains information
        about all layers, accessible by unbinding with any layer key.
        """
        with self._lock:
            return self._normalize(self._global.copy())

    def query_global(self, layer: str) -> np.ndarray:
        """
        Query global state from a specific layer's perspective.

        Unbinds the global state with the layer's key to extract
        what the world looks like from that layer's viewpoint.
        """
        with self._lock:
            if layer not in self._layers:
                return np.zeros(self.dim, dtype=np.float32)

            key = self._layers[layer].key
            return self._normalize(self._unbind(self._global, key))

    def global_coherence(self) -> float:
        """
        Measure overall system coherence.

        Returns average pairwise coherence across all layers.
        """
        with self._lock:
            layer_names = list(self._layers.keys())
            if len(layer_names) < 2:
                return 1.0

            total = 0.0
            count = 0
            for i, a in enumerate(layer_names):
                for b in layer_names[i+1:]:
                    total += abs(self.coherence(a, b))
                    count += 1

            return total / count if count > 0 else 1.0

    def layer_names(self) -> List[str]:
        """Get list of all layer names."""
        with self._lock:
            return list(self._layers.keys())

    def layer_stats(self) -> Dict[str, Dict]:
        """Get statistics for all layers."""
        with self._lock:
            stats = {}
            for name, state in self._layers.items():
                stats[name] = {
                    "write_count": state.write_count,
                    "last_coherence": state.last_coherence,
                    "value_norm": float(np.linalg.norm(state.value)),
                }
            return stats

    def save(self, path: Path) -> None:
        """Save AxisMundi state to disk."""
        with self._lock:
            data = {
                "dim": self.dim,
                "bipolar": self.bipolar,
                "global": self._global.tolist(),
                "layers": {},
            }
            for name, state in self._layers.items():
                data["layers"][name] = {
                    "key": state.key.tolist(),
                    "value": state.value.tolist(),
                    "raw_value": state.raw_value.tolist(),
                    "write_count": state.write_count,
                }

            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(data, f)

    @classmethod
    def load(cls, path: Path) -> "AxisMundi":
        """Load AxisMundi state from disk."""
        with open(path) as f:
            data = json.load(f)

        axis = cls(dim=data["dim"], bipolar=data["bipolar"], layers=[])
        axis._global = np.array(data["global"], dtype=np.float32)

        for name, layer_data in data["layers"].items():
            axis._layers[name] = LayerState(
                name=name,
                key=np.array(layer_data["key"], dtype=np.float32),
                value=np.array(layer_data["value"], dtype=np.float32),
                raw_value=np.array(layer_data["raw_value"], dtype=np.float32),
                write_count=layer_data["write_count"],
            )

        return axis


# =============================================================================
# Convenience Functions
# =============================================================================

def random_hv(dim: int, bipolar: bool = True, seed: Optional[int] = None) -> np.ndarray:
    """Generate a random hypervector."""
    rng = np.random.default_rng(seed)
    if bipolar:
        return rng.choice([-1.0, 1.0], size=dim).astype(np.float32)
    else:
        return rng.choice([0.0, 1.0], size=dim).astype(np.float32)


def encode_text_to_hv(text: str, dim: int = 4096) -> np.ndarray:
    """
    Simple text-to-hypervector encoding.

    Uses character n-grams with position encoding.
    This is a placeholder - production should use learned embeddings.
    """
    hv = np.zeros(dim, dtype=np.float32)

    # Hash each character trigram to a position
    text = text.lower()
    for i in range(len(text) - 2):
        trigram = text[i:i+3]
        # Simple hash to get consistent random vector per trigram
        seed = hash(trigram) % (2**31)
        rng = np.random.default_rng(seed)
        trigram_hv = rng.choice([-1.0, 1.0], size=dim).astype(np.float32)

        # Position encoding via rotation
        shift = i % dim
        trigram_hv = np.roll(trigram_hv, shift)

        hv += trigram_hv

    # Threshold to bipolar
    return np.sign(hv).astype(np.float32)


# =============================================================================
# Demo / Test
# =============================================================================

if __name__ == "__main__":
    print("=== AxisMundi Demo ===\n")

    # Create AxisMundi with default layers
    axis = AxisMundi(dim=4096, layers=["hardware", "avatar", "mission", "emotion"])
    print(f"Created AxisMundi with dim={axis.dim}, layers={axis.layer_names()}\n")

    # Generate some fake state vectors
    hardware_state = encode_text_to_hv("cpu_temp=45 gpu_util=30 memory_free=16gb")
    avatar_state = encode_text_to_hv("user_happy engaged conversation_flowing")
    mission_state = encode_text_to_hv("assist_user maintain_safety learn_preferences")
    emotion_state = encode_text_to_hv("curious calm attentive")

    # Write states
    axis.write("hardware", hardware_state)
    axis.write("avatar", avatar_state)
    axis.write("mission", mission_state)
    axis.write("emotion", emotion_state)
    print("Wrote states to all layers\n")

    # Check coherence
    print("Coherence Matrix:")
    for a in axis.layer_names():
        row = []
        for b in axis.layer_names():
            c = axis.coherence(a, b)
            row.append(f"{c:+.3f}")
        print(f"  {a:10} | {' | '.join(row)}")

    print(f"\nGlobal coherence: {axis.global_coherence():.3f}")

    # Layer stats
    print("\nLayer Stats:")
    for name, stats in axis.layer_stats().items():
        print(f"  {name}: writes={stats['write_count']}, norm={stats['value_norm']:.2f}")

    # Save and load test
    test_path = Path("/tmp/axis_mundi_test.json")
    axis.save(test_path)
    print(f"\nSaved to {test_path}")

    axis2 = AxisMundi.load(test_path)
    print(f"Loaded: dim={axis2.dim}, layers={axis2.layer_names()}")
    print(f"Global coherence after load: {axis2.global_coherence():.3f}")
