"""
Axis Mundi - The World Tree / Holographic Global State Bus
==========================================================

A single circulating holographic state vector that all layers
read/write via HDC binding instead of object graphs.

Architecture:
- Each layer has a unique key HV (random orthogonal)
- Layers WRITE their state as: bind(layer_key, state_hv)
- Layers READ the world by: unbind(layer_key, world_state)
- Coherence = cosine_sim(read(Li), read(Lj))

This creates "vertical superconductivity" - when all layers see
the same world, coherence is high. When L1 is screaming but L9
is blissed out, coherence drops and reflex arcs can trigger.

Hardware mapping:
- FPGA: Can cache key HVs in BRAM, run bind/unbind in single cycle
- Host: Full coherence computation and visualization
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple, Callable
import logging

logger = logging.getLogger(__name__)

# Default dimensions
AXIS_SIZE = 8192  # Global HV dimensionality
NUM_LAYERS = 9    # L1 through L9


def _rand_unit_hv(dim: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Generate a random unit-normalized hypervector."""
    if rng is None:
        rng = np.random.default_rng()
    hv = rng.standard_normal(dim).astype(np.float32)
    hv /= np.linalg.norm(hv) + 1e-8
    return hv


@dataclass
class LayerSlot:
    """State slot for a single layer in the Axis."""
    key: np.ndarray           # Layer key HV (random, fixed)
    last_write: np.ndarray    # Last raw state HV (unbound)
    energy: float = 0.0       # Activity level
    last_write_time: float = 0.0
    write_count: int = 0
    name: str = ""


@dataclass
class AxisEvent:
    """Event emitted by the Axis for monitoring."""
    event_type: str  # "write", "coherence_drop", "reflex_trigger"
    layer_id: int
    data: Dict = field(default_factory=dict)


class AxisMundi:
    """
    The World Tree - Holographic Global State Bus.

    A single circulating holographic state vector that all layers
    read/write via HDC binding instead of object graphs.

    Usage:
        axis = AxisMundi()

        # Layer writes its state
        axis.write_layer_state(layer_id=1, raw_state_hv=my_hv, strength=1.0)

        # Layer reads the world from its perspective
        world_view = axis.read_layer_state(layer_id=1)

        # Check coherence between layers
        coherence = axis.coherence_between(1, 9)

        # Get overall stack alignment
        alignment = stack_alignment(axis)
    """

    def __init__(
        self,
        dim: int = AXIS_SIZE,
        num_layers: int = NUM_LAYERS,
        decay: float = 0.98,
        seed: int = 42,
    ):
        """
        Initialize the Axis Mundi.

        Args:
            dim: Dimension of the holographic state vector
            num_layers: Number of layers (L1 through L{num_layers})
            decay: Global decay per tick (viscosity)
            seed: Random seed for reproducibility
        """
        self.dim = dim
        self.num_layers = num_layers
        self.decay = decay
        self.rng = np.random.default_rng(seed)

        # The global state vector (world hypervector)
        self.state = np.zeros(dim, dtype=np.float32)

        # Layer slots with unique keys
        self.layers: Dict[int, LayerSlot] = {}
        for lid in range(1, num_layers + 1):
            self.layers[lid] = LayerSlot(
                key=_rand_unit_hv(dim, self.rng),
                last_write=np.zeros(dim, dtype=np.float32),
                energy=0.0,
                name=f"L{lid}",
            )

        # Named layer aliases
        self.layer_names = {
            1: "Hardware/Reflex",
            2: "Sensory",
            3: "Perception",
            4: "Affect/PAD",
            5: "Attention",
            6: "Modality",
            7: "Conscience",
            8: "Planning",
            9: "Mission/Autonomy",
        }

        # Event listeners
        self._event_listeners: List[Callable[[AxisEvent], None]] = []

        # Stats
        self.tick_count = 0
        self.total_writes = 0

        logger.info(f"AxisMundi initialized: dim={dim}, layers={num_layers}")

    # =========================================================================
    # HDC Operations
    # =========================================================================

    @staticmethod
    def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Bind two hypervectors (elementwise multiply).

        This is the core HDC operation that creates associations.
        For unit-length vectors, bind(a, b) encodes "a with b".
        """
        return a * b

    @staticmethod
    def unbind(key: np.ndarray, world: np.ndarray) -> np.ndarray:
        """
        Unbind a hypervector from the world state.

        For unit-length keys, unbind(key, world) = world * key
        recovers what was bound with that key.
        """
        return world * key

    @staticmethod
    def normalize(v: np.ndarray) -> np.ndarray:
        """Normalize a vector to unit length."""
        n = np.linalg.norm(v)
        if n < 1e-8:
            return v
        return v / n

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    # =========================================================================
    # Core API
    # =========================================================================

    def tick(self):
        """
        Advance one control cycle.

        Applies global decay to the state vector.
        Call this once per control loop iteration.
        """
        self.state *= self.decay
        self.tick_count += 1

    def write_layer_state(
        self,
        layer_id: int,
        raw_state_hv: np.ndarray,
        strength: float = 1.0,
    ):
        """
        Layer writes its current state as a hypervector.

        The state is bound with the layer's key before being added
        to the global state, creating a superposition of all layer
        contributions.

        Args:
            layer_id: Layer number (1-9)
            raw_state_hv: The layer's state as a hypervector [dim]
            strength: Write strength multiplier [0, 1]
        """
        if layer_id not in self.layers:
            logger.warning(f"Unknown layer {layer_id}")
            return

        slot = self.layers[layer_id]

        # Store raw state
        raw_state_hv = np.asarray(raw_state_hv, dtype=np.float32)
        if raw_state_hv.shape[0] != self.dim:
            # Pad or truncate
            if raw_state_hv.shape[0] < self.dim:
                padded = np.zeros(self.dim, dtype=np.float32)
                padded[:raw_state_hv.shape[0]] = raw_state_hv
                raw_state_hv = padded
            else:
                raw_state_hv = raw_state_hv[:self.dim]

        slot.last_write = raw_state_hv.copy()
        slot.energy = float(np.mean(np.abs(raw_state_hv)))
        slot.write_count += 1

        # Bind with layer key and add to global state
        bound = self.bind(slot.key, raw_state_hv) * strength
        self.state = np.clip(self.state + bound, -1.0, 1.0)

        self.total_writes += 1

        # Emit event
        self._emit_event(AxisEvent(
            event_type="write",
            layer_id=layer_id,
            data={"energy": slot.energy, "strength": strength},
        ))

    def read_layer_state(self, layer_id: int) -> np.ndarray:
        """
        Layer reads its view of the world.

        Unbinds the global state with the layer's key to get
        what the world "looks like" from that layer's perspective.

        Args:
            layer_id: Layer number (1-9)

        Returns:
            The world state from this layer's perspective [dim]
        """
        if layer_id not in self.layers:
            logger.warning(f"Unknown layer {layer_id}")
            return np.zeros(self.dim, dtype=np.float32)

        slot = self.layers[layer_id]
        return self.unbind(slot.key, self.state)

    def get_layer_energy(self, layer_id: int) -> float:
        """Get the current energy (activity level) of a layer."""
        if layer_id not in self.layers:
            return 0.0
        return self.layers[layer_id].energy

    def coherence_between(self, lid_a: int, lid_b: int) -> float:
        """
        Compute coherence between two layers.

        Returns cosine similarity between what layer A thinks the world is
        and what layer B thinks the world is. High coherence means the
        layers are aligned in their worldview.

        Args:
            lid_a: First layer ID
            lid_b: Second layer ID

        Returns:
            Coherence value [-1, 1], typically want > 0.5 for alignment
        """
        view_a = self.read_layer_state(lid_a)
        view_b = self.read_layer_state(lid_b)
        return self.cosine_similarity(view_a, view_b)

    def get_layer_raw(self, layer_id: int) -> np.ndarray:
        """Get the last raw state written by a layer (before binding)."""
        if layer_id not in self.layers:
            return np.zeros(self.dim, dtype=np.float32)
        return self.layers[layer_id].last_write.copy()

    # =========================================================================
    # Analysis & Monitoring
    # =========================================================================

    def get_coherence_matrix(self) -> np.ndarray:
        """
        Compute full coherence matrix between all layers.

        Returns:
            [num_layers, num_layers] matrix of coherence values
        """
        n = self.num_layers
        matrix = np.zeros((n, n), dtype=np.float32)

        for i in range(1, n + 1):
            for j in range(1, n + 1):
                matrix[i-1, j-1] = self.coherence_between(i, j)

        return matrix

    def get_energy_vector(self) -> np.ndarray:
        """Get energy levels for all layers."""
        return np.array([
            self.layers[lid].energy for lid in range(1, self.num_layers + 1)
        ], dtype=np.float32)

    def get_status(self) -> Dict:
        """Get comprehensive status of the Axis."""
        energies = self.get_energy_vector()
        coherence_1_9 = self.coherence_between(1, 9)

        return {
            "dim": self.dim,
            "num_layers": self.num_layers,
            "tick_count": self.tick_count,
            "total_writes": self.total_writes,
            "state_norm": float(np.linalg.norm(self.state)),
            "layer_energies": {
                lid: self.layers[lid].energy
                for lid in self.layers
            },
            "l1_l9_coherence": coherence_1_9,
            "avg_energy": float(np.mean(energies)),
            "max_energy": float(np.max(energies)),
        }

    # =========================================================================
    # Event System
    # =========================================================================

    def on_event(self, callback: Callable[[AxisEvent], None]):
        """Register an event listener."""
        self._event_listeners.append(callback)

    def _emit_event(self, event: AxisEvent):
        """Emit an event to all listeners."""
        for listener in self._event_listeners:
            try:
                listener(event)
            except Exception as e:
                logger.warning(f"Event listener error: {e}")

    # =========================================================================
    # Serialization
    # =========================================================================

    def save_state(self, path: str):
        """Save current state to file."""
        import json
        data = {
            "dim": self.dim,
            "num_layers": self.num_layers,
            "tick_count": self.tick_count,
            "total_writes": self.total_writes,
            "state": self.state.tolist(),
            "layers": {
                str(lid): {
                    "key": slot.key.tolist(),
                    "last_write": slot.last_write.tolist(),
                    "energy": slot.energy,
                    "write_count": slot.write_count,
                }
                for lid, slot in self.layers.items()
            },
        }
        with open(path, 'w') as f:
            json.dump(data, f)

    def load_state(self, path: str):
        """Load state from file."""
        import json
        with open(path, 'r') as f:
            data = json.load(f)

        self.tick_count = data["tick_count"]
        self.total_writes = data["total_writes"]
        self.state = np.array(data["state"], dtype=np.float32)

        for lid_str, slot_data in data["layers"].items():
            lid = int(lid_str)
            if lid in self.layers:
                self.layers[lid].key = np.array(slot_data["key"], dtype=np.float32)
                self.layers[lid].last_write = np.array(slot_data["last_write"], dtype=np.float32)
                self.layers[lid].energy = slot_data["energy"]
                self.layers[lid].write_count = slot_data["write_count"]


# =============================================================================
# Stack Alignment Metric (Vertical Superconductivity)
# =============================================================================

def stack_alignment(axis: AxisMundi, ref_layer: int = 5) -> float:
    """
    Compute overall stack alignment (vertical superconductivity).

    Returns a value in [0, 1] where 1 means all layers see the same world.

    Args:
        axis: The AxisMundi instance
        ref_layer: Reference layer for comparison (default L5 mid-layer)

    Returns:
        Alignment score [0, 1]
    """
    ref = axis.read_layer_state(ref_layer)
    ref_norm = np.linalg.norm(ref)
    if ref_norm < 1e-8:
        return 0.0
    ref = ref / ref_norm

    sims = []
    for lid in axis.layers.keys():
        hv = axis.read_layer_state(lid)
        hv_norm = np.linalg.norm(hv)
        if hv_norm > 1e-8:
            hv = hv / hv_norm
            sims.append(float(np.dot(ref, hv)))
        else:
            sims.append(0.0)

    # sims are in [-1, 1], map to [0, 1]
    avg_sim = sum(sims) / len(sims) if sims else 0.0
    return 0.5 * (avg_sim + 1.0)


def detect_coherence_crisis(
    axis: AxisMundi,
    l1_energy_threshold: float = 0.35,
    coherence_threshold: float = 0.2,
) -> Tuple[bool, Dict]:
    """
    Detect if the stack is in a coherence crisis.

    A crisis occurs when L1 (hardware) has high energy (pain/stress)
    but low coherence with L9 (mission) - meaning the high-level
    planner doesn't "see" what hardware is feeling.

    Returns:
        (is_crisis, details)
    """
    l1_energy = axis.get_layer_energy(1)
    l9_energy = axis.get_layer_energy(9)
    coherence = axis.coherence_between(1, 9)

    is_crisis = l1_energy > l1_energy_threshold and coherence < coherence_threshold

    return is_crisis, {
        "l1_energy": l1_energy,
        "l9_energy": l9_energy,
        "l1_l9_coherence": coherence,
        "threshold_energy": l1_energy_threshold,
        "threshold_coherence": coherence_threshold,
    }


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate AxisMundi functionality."""
    print("=" * 60)
    print("Axis Mundi - Holographic Global State Bus Demo")
    print("=" * 60)

    axis = AxisMundi(dim=1024, num_layers=9)
    rng = np.random.default_rng(42)

    print("\n--- Writing layer states ---")

    # L1 writes hardware state (some stress)
    l1_state = rng.standard_normal(1024).astype(np.float32) * 0.8
    axis.write_layer_state(1, l1_state, strength=1.0)
    print(f"  L1 wrote state, energy={axis.get_layer_energy(1):.3f}")

    # L5 writes mid-level perception
    l5_state = rng.standard_normal(1024).astype(np.float32) * 0.5
    axis.write_layer_state(5, l5_state, strength=1.0)
    print(f"  L5 wrote state, energy={axis.get_layer_energy(5):.3f}")

    # L9 writes mission state (calm, different)
    l9_state = rng.standard_normal(1024).astype(np.float32) * 0.3
    axis.write_layer_state(9, l9_state, strength=1.0)
    print(f"  L9 wrote state, energy={axis.get_layer_energy(9):.3f}")

    print("\n--- Reading layer views ---")
    l1_view = axis.read_layer_state(1)
    l9_view = axis.read_layer_state(9)
    print(f"  L1 view norm: {np.linalg.norm(l1_view):.3f}")
    print(f"  L9 view norm: {np.linalg.norm(l9_view):.3f}")

    print("\n--- Coherence analysis ---")
    coherence_1_9 = axis.coherence_between(1, 9)
    coherence_1_5 = axis.coherence_between(1, 5)
    coherence_5_9 = axis.coherence_between(5, 9)
    print(f"  L1-L9 coherence: {coherence_1_9:.3f}")
    print(f"  L1-L5 coherence: {coherence_1_5:.3f}")
    print(f"  L5-L9 coherence: {coherence_5_9:.3f}")

    print("\n--- Stack alignment ---")
    alignment = stack_alignment(axis)
    print(f"  Overall alignment: {alignment:.3f}")

    print("\n--- Crisis detection ---")
    is_crisis, details = detect_coherence_crisis(axis)
    print(f"  Crisis detected: {is_crisis}")
    print(f"  Details: {details}")

    print("\n--- Status ---")
    status = axis.get_status()
    for k, v in status.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo()
