"""
CSTP: Cognitive State Transfer Protocol

This module implements Geometric State Transfer - encoding abstract thoughts
as (z, c) pairs where:
- z: Latent vector (the embedding/content of the thought)
- c: Curvature (the geometric context the thought lives in)

Key Innovation: A "thought" is not just a vector, but a vector in a
specific geometric space. Different curvatures enable different kinds
of reasoning:
- c → 0: Flat, Euclidean - simple, unambiguous, local similarity
- c ∈ (0.5, 1.5): Standard hyperbolic - hierarchical, tree-like
- c > 1.5: High curvature - deep abstraction, compressed hierarchies

L3 Metacontrol Integration:
- Low valence / high stress → c → 0 (keep thoughts simple)
- Calm / planning mode → allow c to rise (complex structures ok)

Use Cases:
1. Serialize/deserialize plans with their geometric context
2. Transfer cognitive state between processes or agents
3. Log and replay "thought sequences" for analysis
4. Enable future multi-agent geometric coordination
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union
import hashlib
import json
import math
import base64


# ============================================================
# Thought Representation
# ============================================================

class ThoughtType(str, Enum):
    """Types of thoughts that can be encoded."""
    OBSERVATION = "observation"   # Perception/input
    INFERENCE = "inference"       # Derived conclusion
    PLAN = "plan"                # Action sequence
    MEMORY = "memory"            # Episodic recall
    HYPOTHESIS = "hypothesis"    # Uncertain belief
    GOAL = "goal"                # Desired state
    EMOTION = "emotion"          # Affective state


class CompressionLevel(str, Enum):
    """How compressed is the thought representation?"""
    RAW = "raw"           # Full-dimensional, uncompressed
    STANDARD = "standard" # Moderate compression
    COMPACT = "compact"   # Highly compressed
    MINIMAL = "minimal"   # Maximum compression


@dataclass
class ThoughtMetadata:
    """Metadata about a thought."""
    thought_type: ThoughtType
    created_at: datetime = field(default_factory=datetime.now)
    source: str = "self"        # Where did this thought come from
    confidence: float = 0.8     # How confident are we in this thought
    importance: float = 0.5     # How important is this thought
    compression: CompressionLevel = CompressionLevel.STANDARD
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.thought_type.value,
            "created_at": self.created_at.isoformat(),
            "source": self.source,
            "confidence": self.confidence,
            "importance": self.importance,
            "compression": self.compression.value,
            "tags": self.tags
        }


@dataclass
class CognitiveState:
    """
    A cognitive state represented as (z, c).

    This is the core abstraction: a thought is a point in geometric space.
    """
    # The latent vector (content)
    z: List[float]

    # The curvature (geometric context)
    c: float

    # Metadata
    metadata: ThoughtMetadata

    # Optional: structured content for plans/sequences
    structure: Optional[Dict[str, Any]] = None

    @property
    def dimension(self) -> int:
        """Dimensionality of the latent vector."""
        return len(self.z)

    @property
    def norm(self) -> float:
        """L2 norm of z."""
        return math.sqrt(sum(x * x for x in self.z))

    @property
    def is_euclidean(self) -> bool:
        """Is this thought in approximately Euclidean space?"""
        return self.c < 0.3

    @property
    def is_hyperbolic(self) -> bool:
        """Is this thought in hyperbolic space?"""
        return self.c >= 0.3

    @property
    def geometry_description(self) -> str:
        """Human-readable geometry description."""
        if self.c < 0.3:
            return "flat (Euclidean)"
        elif self.c < 0.7:
            return "low curvature (shallow hierarchy)"
        elif self.c < 1.5:
            return "standard hyperbolic (tree-like)"
        else:
            return "high curvature (deep abstraction)"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "z": self.z,
            "c": self.c,
            "dimension": self.dimension,
            "norm": self.norm,
            "metadata": self.metadata.to_dict(),
            "structure": self.structure,
            "geometry": self.geometry_description
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CognitiveState":
        """Reconstruct from dictionary."""
        metadata = ThoughtMetadata(
            thought_type=ThoughtType(data["metadata"]["type"]),
            source=data["metadata"].get("source", "unknown"),
            confidence=data["metadata"].get("confidence", 0.5),
            importance=data["metadata"].get("importance", 0.5),
            compression=CompressionLevel(data["metadata"].get("compression", "standard")),
            tags=data["metadata"].get("tags", [])
        )

        return cls(
            z=data["z"],
            c=data["c"],
            metadata=metadata,
            structure=data.get("structure")
        )


# ============================================================
# Curvature Controller (L3 Integration)
# ============================================================

class CurvatureMode(str, Enum):
    """Curvature selection modes."""
    FIXED = "fixed"           # Use specified curvature
    ADAPTIVE = "adaptive"     # Adapt to task
    STRESS_AWARE = "stress_aware"  # Reduce under stress


@dataclass
class CurvaturePolicy:
    """Policy for selecting curvature based on state."""
    mode: CurvatureMode = CurvatureMode.STRESS_AWARE

    # Fixed mode
    fixed_curvature: float = 1.0

    # Adaptive mode
    default_curvature: float = 1.0
    planning_curvature: float = 1.5
    retrieval_curvature: float = 0.3

    # Stress-aware thresholds
    stress_high_threshold: float = 0.7
    stress_low_threshold: float = 0.3
    stressed_curvature: float = 0.2  # Keep it simple when stressed


class CurvatureController:
    """
    Controls curvature selection based on L3 metacontrol state.

    Key insight: Under stress, keep thoughts simple (low curvature).
    When calm and planning, allow complex hierarchical thoughts (high curvature).
    """

    def __init__(self, policy: Optional[CurvaturePolicy] = None):
        self.policy = policy or CurvaturePolicy()
        self._current_stress = 0.0
        self._current_valence = 0.0

    def update_state(
        self,
        stress: float = 0.0,
        valence: float = 0.0,
        arousal: float = 0.0
    ) -> None:
        """Update controller with current metacontrol state."""
        self._current_stress = stress
        self._current_valence = valence

    def select_curvature(
        self,
        task_type: Optional[str] = None,
        hierarchy_depth: Optional[int] = None,
        force_curvature: Optional[float] = None
    ) -> float:
        """
        Select appropriate curvature for current context.

        Args:
            task_type: Type of task (planning, retrieval, etc.)
            hierarchy_depth: Known hierarchy depth
            force_curvature: Override curvature selection

        Returns:
            Selected curvature value
        """
        if force_curvature is not None:
            return force_curvature

        if self.policy.mode == CurvatureMode.FIXED:
            return self.policy.fixed_curvature

        # Start with base curvature
        if task_type == "planning":
            base_c = self.policy.planning_curvature
        elif task_type == "retrieval":
            base_c = self.policy.retrieval_curvature
        else:
            base_c = self.policy.default_curvature

        # Adjust for hierarchy depth
        if hierarchy_depth is not None:
            if hierarchy_depth > 5:
                base_c = max(base_c, 1.5)
            elif hierarchy_depth > 3:
                base_c = max(base_c, 1.0)

        # Stress-aware adjustment
        if self.policy.mode == CurvatureMode.STRESS_AWARE:
            if self._current_stress > self.policy.stress_high_threshold:
                # High stress → simple thoughts
                base_c = min(base_c, self.policy.stressed_curvature)
            elif self._current_valence < -0.5:
                # Low valence → slightly simpler
                base_c = min(base_c, 0.5)

        return max(0.01, base_c)  # Keep positive

    def explain_selection(self, curvature: float) -> str:
        """Explain why a curvature was selected."""
        reasons = []

        if self._current_stress > self.policy.stress_high_threshold:
            reasons.append(f"High stress ({self._current_stress:.2f}) → keeping thoughts simple")

        if self._current_valence < -0.5:
            reasons.append(f"Low valence ({self._current_valence:.2f}) → reducing complexity")

        if curvature < 0.3:
            reasons.append("Flat geometry: local similarity, unambiguous")
        elif curvature < 1.0:
            reasons.append("Low curvature: shallow hierarchies")
        elif curvature < 1.5:
            reasons.append("Standard hyperbolic: tree-like structure")
        else:
            reasons.append("High curvature: deep abstraction, compressed")

        return "; ".join(reasons) if reasons else "Default selection"


# ============================================================
# CSTP Encoder/Decoder
# ============================================================

class CSTPEncoder:
    """
    Encodes cognitive content into (z, c) representation.

    Can encode:
    - Raw vectors
    - Plans (sequences of steps)
    - Observations
    - Structured data
    """

    def __init__(
        self,
        default_dim: int = 64,
        curvature_controller: Optional[CurvatureController] = None
    ):
        self.default_dim = default_dim
        self.curvature_controller = curvature_controller or CurvatureController()
        self._encoding_counter = 0

    def encode_vector(
        self,
        vector: List[float],
        thought_type: ThoughtType = ThoughtType.OBSERVATION,
        curvature: Optional[float] = None,
        task_type: Optional[str] = None,
        confidence: float = 0.8,
        tags: Optional[List[str]] = None
    ) -> CognitiveState:
        """
        Encode a raw vector as a cognitive state.

        Args:
            vector: The latent vector
            thought_type: Type of thought
            curvature: Override curvature (None = auto-select)
            task_type: Task context for curvature selection
            confidence: Confidence in this thought
            tags: Optional tags
        """
        # Select curvature
        c = curvature if curvature is not None else self.curvature_controller.select_curvature(
            task_type=task_type
        )

        # Project to ball if hyperbolic
        z = self._project_to_ball(vector, c)

        metadata = ThoughtMetadata(
            thought_type=thought_type,
            confidence=confidence,
            tags=tags or []
        )

        self._encoding_counter += 1

        return CognitiveState(z=z, c=c, metadata=metadata)

    def encode_plan(
        self,
        steps: List[Dict[str, Any]],
        hierarchy_depth: int = 1,
        curvature: Optional[float] = None,
        confidence: float = 0.8
    ) -> CognitiveState:
        """
        Encode a plan (sequence of steps) as a cognitive state.

        Plans naturally benefit from hyperbolic geometry due to
        their hierarchical/sequential structure.
        """
        # Select curvature based on plan structure
        c = curvature if curvature is not None else self.curvature_controller.select_curvature(
            task_type="planning",
            hierarchy_depth=hierarchy_depth
        )

        # Encode plan structure into latent vector
        z = self._plan_to_vector(steps, hierarchy_depth)

        metadata = ThoughtMetadata(
            thought_type=ThoughtType.PLAN,
            confidence=confidence,
            importance=0.8,  # Plans are important
            tags=["plan", f"steps:{len(steps)}", f"depth:{hierarchy_depth}"]
        )

        return CognitiveState(
            z=z,
            c=c,
            metadata=metadata,
            structure={"type": "plan", "steps": steps, "depth": hierarchy_depth}
        )

    def encode_observation(
        self,
        content: str,
        embedding: Optional[List[float]] = None,
        confidence: float = 0.8,
        curvature: Optional[float] = None
    ) -> CognitiveState:
        """Encode an observation."""
        # Use provided embedding or generate simple hash-based one
        if embedding:
            z = embedding
        else:
            z = self._text_to_vector(content)

        # Observations are typically flat/simple, but allow override
        c = curvature if curvature is not None else self.curvature_controller.select_curvature(task_type="retrieval")

        metadata = ThoughtMetadata(
            thought_type=ThoughtType.OBSERVATION,
            confidence=confidence,
            tags=["observation"]
        )

        return CognitiveState(
            z=z,
            c=c,
            metadata=metadata,
            structure={"type": "observation", "content_hash": hashlib.md5(content.encode()).hexdigest()[:8]}
        )

    def _project_to_ball(self, z: List[float], c: float) -> List[float]:
        """Project vector to Poincaré ball if needed."""
        if c < 0.1:
            return z  # Nearly Euclidean, no projection needed

        norm = math.sqrt(sum(x * x for x in z))
        max_norm = 1.0 - 1e-5

        if norm > max_norm:
            scale = max_norm / norm
            return [x * scale for x in z]

        return z

    def _plan_to_vector(self, steps: List[Dict[str, Any]], depth: int) -> List[float]:
        """Convert plan structure to latent vector."""
        # Simple encoding: hash of step names/types, spread across dimensions
        z = [0.0] * self.default_dim

        for i, step in enumerate(steps):
            # Hash step to get index
            step_str = json.dumps(step, sort_keys=True)
            h = int(hashlib.md5(step_str.encode()).hexdigest()[:8], 16)

            # Spread contribution across dimensions
            for d in range(self.default_dim):
                z[d] += math.sin(h * (d + 1) * 0.1 + i * 0.5) * (1.0 / (i + 1))

        # Normalize
        norm = math.sqrt(sum(x * x for x in z)) + 1e-8
        return [x / norm * 0.5 for x in z]  # Scale to stay in ball

    def _text_to_vector(self, text: str) -> List[float]:
        """Simple text to vector encoding (placeholder for real embedding)."""
        z = [0.0] * self.default_dim

        # Hash-based encoding
        h = hashlib.sha256(text.encode()).digest()

        for i in range(min(len(h), self.default_dim)):
            z[i] = (h[i] - 128) / 256.0  # Normalize to [-0.5, 0.5]

        return z


class CSTPDecoder:
    """
    Decodes (z, c) cognitive states.

    Can extract:
    - Raw vectors
    - Plan structures
    - Metadata
    """

    def decode_to_vector(self, state: CognitiveState) -> List[float]:
        """Extract the raw latent vector."""
        return state.z

    def decode_plan(self, state: CognitiveState) -> Optional[Dict[str, Any]]:
        """Extract plan structure if this is a plan."""
        if state.metadata.thought_type != ThoughtType.PLAN:
            return None

        if state.structure and state.structure.get("type") == "plan":
            return {
                "steps": state.structure.get("steps", []),
                "depth": state.structure.get("depth", 1),
                "curvature": state.c,
                "geometry": state.geometry_description
            }

        return None

    def decode_metadata(self, state: CognitiveState) -> Dict[str, Any]:
        """Extract metadata."""
        return state.metadata.to_dict()

    def compare_states(
        self,
        state1: CognitiveState,
        state2: CognitiveState
    ) -> Dict[str, Any]:
        """Compare two cognitive states."""
        # Vector distance
        if len(state1.z) == len(state2.z):
            euclidean_dist = math.sqrt(sum(
                (a - b) ** 2 for a, b in zip(state1.z, state2.z)
            ))
        else:
            euclidean_dist = float('inf')

        # Curvature difference
        curvature_diff = abs(state1.c - state2.c)

        # Same geometry regime?
        same_regime = state1.geometry_description == state2.geometry_description

        return {
            "euclidean_distance": euclidean_dist,
            "curvature_difference": curvature_diff,
            "same_geometry_regime": same_regime,
            "type_match": state1.metadata.thought_type == state2.metadata.thought_type,
            "can_directly_compare": same_regime and len(state1.z) == len(state2.z)
        }


# ============================================================
# CSTP Protocol (Serialization)
# ============================================================

class CSTPProtocol:
    """
    Protocol for serializing and deserializing cognitive states.

    Supports:
    - JSON format (human-readable)
    - Binary format (compact)
    - Versioned format (forward compatibility)
    """

    VERSION = "1.0.0"

    @classmethod
    def serialize_json(cls, state: CognitiveState) -> str:
        """Serialize to JSON string."""
        data = {
            "version": cls.VERSION,
            "state": state.to_dict()
        }
        return json.dumps(data)

    @classmethod
    def deserialize_json(cls, data: str) -> CognitiveState:
        """Deserialize from JSON string."""
        parsed = json.loads(data)

        # Version check
        version = parsed.get("version", "1.0.0")
        if version.split(".")[0] != cls.VERSION.split(".")[0]:
            raise ValueError(f"Incompatible version: {version}")

        return CognitiveState.from_dict(parsed["state"])

    @classmethod
    def serialize_binary(cls, state: CognitiveState) -> bytes:
        """Serialize to compact binary format."""
        # Simple implementation: JSON + compression indicator
        json_str = cls.serialize_json(state)
        return b"CSTP" + json_str.encode('utf-8')

    @classmethod
    def deserialize_binary(cls, data: bytes) -> CognitiveState:
        """Deserialize from binary format."""
        if not data.startswith(b"CSTP"):
            raise ValueError("Invalid CSTP binary format")

        json_str = data[4:].decode('utf-8')
        return cls.deserialize_json(json_str)

    @classmethod
    def serialize_compact(cls, state: CognitiveState) -> str:
        """Serialize to compact base64 string."""
        binary = cls.serialize_binary(state)
        return base64.b64encode(binary).decode('ascii')

    @classmethod
    def deserialize_compact(cls, data: str) -> CognitiveState:
        """Deserialize from compact base64 string."""
        binary = base64.b64decode(data.encode('ascii'))
        return cls.deserialize_binary(binary)


# ============================================================
# Thought Stream
# ============================================================

@dataclass
class ThoughtStreamEntry:
    """An entry in a thought stream."""
    state: CognitiveState
    sequence_number: int
    timestamp: datetime = field(default_factory=datetime.now)
    previous_hash: Optional[str] = None
    entry_hash: Optional[str] = None

    def compute_hash(self) -> str:
        """Compute hash of this entry."""
        data = {
            "z": self.state.z,
            "c": self.state.c,
            "seq": self.sequence_number,
            "prev": self.previous_hash
        }
        return hashlib.sha256(json.dumps(data).encode()).hexdigest()[:16]


class ThoughtStream:
    """
    A sequence of cognitive states forming a "stream of consciousness".

    Useful for:
    - Logging thought sequences
    - Replaying cognitive trajectories
    - Analyzing thought patterns
    """

    def __init__(self, name: str = "default"):
        self.name = name
        self._entries: List[ThoughtStreamEntry] = []
        self._sequence_counter = 0

    def append(self, state: CognitiveState) -> ThoughtStreamEntry:
        """Add a thought to the stream."""
        prev_hash = self._entries[-1].entry_hash if self._entries else None

        entry = ThoughtStreamEntry(
            state=state,
            sequence_number=self._sequence_counter,
            previous_hash=prev_hash
        )
        entry.entry_hash = entry.compute_hash()

        self._entries.append(entry)
        self._sequence_counter += 1

        return entry

    def get_entries(
        self,
        start: int = 0,
        count: Optional[int] = None
    ) -> List[ThoughtStreamEntry]:
        """Get entries from the stream."""
        end = start + count if count else len(self._entries)
        return self._entries[start:end]

    def get_curvature_trajectory(self) -> List[float]:
        """Get curvature values over time."""
        return [e.state.c for e in self._entries]

    def get_thought_types(self) -> Dict[ThoughtType, int]:
        """Count thought types in stream."""
        counts: Dict[ThoughtType, int] = {}
        for e in self._entries:
            t = e.state.metadata.thought_type
            counts[t] = counts.get(t, 0) + 1
        return counts

    def analyze_geometry_shifts(self) -> List[Dict[str, Any]]:
        """Identify significant geometry shifts."""
        shifts = []

        for i in range(1, len(self._entries)):
            prev_c = self._entries[i - 1].state.c
            curr_c = self._entries[i].state.c
            delta = abs(curr_c - prev_c)

            if delta > 0.3:  # Significant shift
                shifts.append({
                    "sequence": i,
                    "from_c": prev_c,
                    "to_c": curr_c,
                    "delta": delta,
                    "from_geometry": self._entries[i - 1].state.geometry_description,
                    "to_geometry": self._entries[i].state.geometry_description
                })

        return shifts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "length": len(self._entries),
            "entries": [
                {
                    "seq": e.sequence_number,
                    "c": e.state.c,
                    "type": e.state.metadata.thought_type.value,
                    "hash": e.entry_hash
                }
                for e in self._entries
            ]
        }


# ============================================================
# Factory Functions
# ============================================================

def create_encoder(
    dimension: int = 64,
    stress_aware: bool = True
) -> CSTPEncoder:
    """Create a CSTP encoder."""
    policy = CurvaturePolicy(
        mode=CurvatureMode.STRESS_AWARE if stress_aware else CurvatureMode.ADAPTIVE
    )
    controller = CurvatureController(policy)
    return CSTPEncoder(default_dim=dimension, curvature_controller=controller)


def create_decoder() -> CSTPDecoder:
    """Create a CSTP decoder."""
    return CSTPDecoder()


def encode_thought(
    content: Union[str, List[float], Dict[str, Any]],
    thought_type: ThoughtType = ThoughtType.OBSERVATION,
    curvature: Optional[float] = None
) -> CognitiveState:
    """
    Convenience function to encode a thought.

    Args:
        content: String, vector, or plan structure
        thought_type: Type of thought
        curvature: Optional override curvature

    Returns:
        Encoded cognitive state
    """
    encoder = create_encoder()

    if isinstance(content, str):
        return encoder.encode_observation(content, curvature=curvature)
    elif isinstance(content, list) and all(isinstance(x, (int, float)) for x in content):
        return encoder.encode_vector(content, thought_type=thought_type, curvature=curvature)
    elif isinstance(content, dict) and "steps" in content:
        return encoder.encode_plan(
            content["steps"],
            hierarchy_depth=content.get("depth", 1),
            curvature=curvature
        )
    else:
        # Default: treat as observation
        return encoder.encode_observation(str(content), curvature=curvature)


def serialize_thought(state: CognitiveState, format: str = "json") -> str:
    """Serialize a cognitive state."""
    if format == "json":
        return CSTPProtocol.serialize_json(state)
    elif format == "compact":
        return CSTPProtocol.serialize_compact(state)
    else:
        raise ValueError(f"Unknown format: {format}")


def deserialize_thought(data: str, format: str = "json") -> CognitiveState:
    """Deserialize a cognitive state."""
    if format == "json":
        return CSTPProtocol.deserialize_json(data)
    elif format == "compact":
        return CSTPProtocol.deserialize_compact(data)
    else:
        raise ValueError(f"Unknown format: {format}")


__all__ = [
    "ThoughtType",
    "CompressionLevel",
    "ThoughtMetadata",
    "CognitiveState",
    "CurvatureMode",
    "CurvaturePolicy",
    "CurvatureController",
    "CSTPEncoder",
    "CSTPDecoder",
    "CSTPProtocol",
    "ThoughtStreamEntry",
    "ThoughtStream",
    "create_encoder",
    "create_decoder",
    "encode_thought",
    "serialize_thought",
    "deserialize_thought"
]
