"""
Cognitive Layers - 4D Thought Space
====================================

Represent 4D space in 3D layers: each layer is an orthogonal
HDC subspace representing a different style/perspective of thought.

Coordinates:
    (x, y, z) = position in HDC/neuron fabric
    w = conceptual layer index

Layer examples:
    Layer 0: Sensory/Heuristic - fast, emotional, spiky (SNN mode)
    Layer 1: Symbolic - HDC bindings, pattern matching
    Layer 2: Meta - evaluation, Institute metrics
    Layer 3: Teleological - Architect/Horizon perspective

Same physical fabric, different interpretation per layer.
Cross-layer binding lets you see "how this event looks from the meta layer."

This gives:
1. More orthogonal axes before concepts collide
2. Natural separation of "what/how/why" of thinking
3. Trackable 4D trajectories through thought space
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum, auto
import hashlib


class LayerRole(Enum):
    """Predefined cognitive layer roles."""
    SENSORY = auto()      # Layer 0: Raw perception, metrics, fast heuristics
    SYMBOLIC = auto()     # Layer 1: HDC bindings, pattern matching, concepts
    META = auto()         # Layer 2: Evaluation, strategy assessment
    TELEOLOGICAL = auto() # Layer 3: Purpose, Horizon alignment, long-term


@dataclass
class CognitiveLayer:
    """
    A single cognitive layer - one 'floor' in the 4D cathedral.

    Each layer has:
    - Its own HDC subspace (orthogonal seed)
    - A defined role/perspective
    - Layer-specific item memory
    """
    layer_id: int        # w coordinate
    name: str
    role: LayerRole
    dim: int = 8192
    description: str = ""

    # Layer-specific RNG seed (ensures orthogonality)
    _seed: int = field(init=False)
    _rng: np.random.Generator = field(init=False)

    # Item memory for this layer
    _items: Dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        # Generate unique seed from layer name
        self._seed = int(hashlib.sha256(
            f"layer:{self.name}:{self.layer_id}".encode()
        ).hexdigest()[:8], 16)
        self._rng = np.random.default_rng(self._seed)

    def get_item(self, name: str) -> np.ndarray:
        """Get or create an item hypervector in this layer's subspace."""
        if name not in self._items:
            # Seed by both layer and name for reproducibility
            seed = int(hashlib.sha256(
                f"{self._seed}:{name}".encode()
            ).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            self._items[name] = rng.integers(0, 2, size=self.dim, dtype=np.uint8)
        return self._items[name]

    def random_hv(self) -> np.ndarray:
        """Generate a random HV in this layer's subspace."""
        return self._rng.integers(0, 2, size=self.dim, dtype=np.uint8)

    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """XOR binding within this layer."""
        return np.bitwise_xor(a, b)

    def bundle(self, hvs: List[np.ndarray]) -> np.ndarray:
        """Majority vote bundling within this layer."""
        if not hvs:
            return np.zeros(self.dim, dtype=np.uint8)
        if len(hvs) == 1:
            return hvs[0].copy()

        total = np.sum(hvs, axis=0)
        threshold = len(hvs) / 2
        result = (total > threshold).astype(np.uint8)

        # Tie-break with layer-specific RNG
        ties = (total == threshold)
        if np.any(ties):
            result[ties] = self._rng.integers(0, 2, size=np.sum(ties), dtype=np.uint8)
        return result

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute similarity within this layer."""
        matches = np.sum(a == b)
        return (2 * matches - self.dim) / self.dim


@dataclass
class LayerProjection:
    """
    A concept projected into a specific layer.

    (hv, layer_id) = minimal 4D coordinate
    """
    hv: np.ndarray
    layer: CognitiveLayer
    label: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def layer_id(self) -> int:
        return self.layer.layer_id

    def similarity_to(self, other: "LayerProjection") -> float:
        """Compute similarity (only meaningful if same layer)."""
        if self.layer_id != other.layer_id:
            return 0.0  # Cross-layer comparison needs projection
        return self.layer.similarity(self.hv, other.hv)


@dataclass
class LayeredSpace:
    """
    The complete 4D cognitive space - a stack of orthogonal HDC layers.

    Provides:
    - Layer management
    - Cross-layer binding ("how does X look from layer Y?")
    - Trajectory tracking
    """
    dim: int = 8192
    layers: Dict[int, CognitiveLayer] = field(default_factory=dict)

    # Cross-layer projection matrices (conceptual)
    _projection_seeds: Dict[Tuple[int, int], int] = field(default_factory=dict)

    def add_layer(self, layer_id: int, name: str, role: LayerRole,
                  description: str = "") -> CognitiveLayer:
        """Add a cognitive layer."""
        layer = CognitiveLayer(
            layer_id=layer_id,
            name=name,
            role=role,
            dim=self.dim,
            description=description,
        )
        self.layers[layer_id] = layer
        return layer

    def get_layer(self, layer_id: int) -> Optional[CognitiveLayer]:
        """Get a layer by ID."""
        return self.layers.get(layer_id)

    def get_layer_by_role(self, role: LayerRole) -> Optional[CognitiveLayer]:
        """Get first layer with given role."""
        for layer in self.layers.values():
            if layer.role == role:
                return layer
        return None

    def _get_cross_layer_hv(self, from_id: int, to_id: int) -> np.ndarray:
        """Get the HV used for cross-layer projection."""
        key = (from_id, to_id)
        if key not in self._projection_seeds:
            seed = int(hashlib.sha256(
                f"cross:{from_id}→{to_id}".encode()
            ).hexdigest()[:8], 16)
            self._projection_seeds[key] = seed

        rng = np.random.default_rng(self._projection_seeds[key])
        return rng.integers(0, 2, size=self.dim, dtype=np.uint8)

    def project(self, hv: np.ndarray, from_layer: int,
                to_layer: int) -> np.ndarray:
        """
        Project a hypervector from one layer to another.

        Uses binding with a cross-layer projection HV.
        This preserves structure while shifting to the target subspace.
        """
        if from_layer == to_layer:
            return hv.copy()

        # Bind with cross-layer projection HV
        proj_hv = self._get_cross_layer_hv(from_layer, to_layer)
        return np.bitwise_xor(hv, proj_hv)

    def project_to_layer(self, proj: LayerProjection,
                         to_layer_id: int) -> LayerProjection:
        """Project a LayerProjection to a different layer."""
        to_layer = self.get_layer(to_layer_id)
        if to_layer is None:
            raise ValueError(f"Layer {to_layer_id} not found")

        new_hv = self.project(proj.hv, proj.layer_id, to_layer_id)
        return LayerProjection(
            hv=new_hv,
            layer=to_layer,
            label=f"{proj.label}@L{to_layer_id}",
            metadata={**proj.metadata, "projected_from": proj.layer_id},
        )

    def encode_in_layer(self, concept: str, layer_id: int) -> LayerProjection:
        """Encode a concept in a specific layer."""
        layer = self.get_layer(layer_id)
        if layer is None:
            raise ValueError(f"Layer {layer_id} not found")

        hv = layer.get_item(concept)
        return LayerProjection(hv=hv, layer=layer, label=concept)

    def cross_layer_similarity(self, proj1: LayerProjection,
                               proj2: LayerProjection) -> float:
        """
        Compute similarity between projections in different layers.

        Projects both to a common reference layer (0).
        """
        ref_layer = 0
        hv1 = self.project(proj1.hv, proj1.layer_id, ref_layer)
        hv2 = self.project(proj2.hv, proj2.layer_id, ref_layer)

        layer0 = self.get_layer(ref_layer)
        if layer0:
            return layer0.similarity(hv1, hv2)
        else:
            # Fallback: direct comparison
            matches = np.sum(hv1 == hv2)
            return (2 * matches - self.dim) / self.dim


def create_default_layers(dim: int = 8192) -> LayeredSpace:
    """
    Create the default 4-layer cognitive space.

    Layer 0: Sensory - fast, emotional, metric-based
    Layer 1: Symbolic - HDC concepts, pattern matching
    Layer 2: Meta - evaluation, strategy assessment
    Layer 3: Teleological - purpose, Horizon alignment
    """
    space = LayeredSpace(dim=dim)

    space.add_layer(
        layer_id=0,
        name="sensory",
        role=LayerRole.SENSORY,
        description="Fast heuristics, raw metrics, emotional valence. SNN-style snap judgments.",
    )

    space.add_layer(
        layer_id=1,
        name="symbolic",
        role=LayerRole.SYMBOLIC,
        description="HDC bindings, concept composition, pattern matching. The 'thinking' layer.",
    )

    space.add_layer(
        layer_id=2,
        name="meta",
        role=LayerRole.META,
        description="Evaluation of thinking. Is this working? Institute metrics, strategy assessment.",
    )

    space.add_layer(
        layer_id=3,
        name="teleological",
        role=LayerRole.TELEOLOGICAL,
        description="Purpose and direction. Does this advance the Horizon? Architect-level reasoning.",
    )

    return space
