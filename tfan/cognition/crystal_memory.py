"""
Crystal Memory - Episodic Memory with Hyperdimensional Computing
================================================================

This is the "wisdom backbone" for Ara.

Instead of just logging events to JSON (Hippocampus), we encode experiences
as hypervectors. This gives us:

1. **Fast similarity search**: "Have I been here before?"
2. **Compositional queries**: "Situations like X with emotion Y"
3. **Graceful degradation**: Noisy memories still work
4. **One-shot learning**: Add a new experience instantly, no retraining

Key concepts:
- **Episode**: A structured experience (context, action, outcome, emotion, etc.)
- **Scar**: A painful episode we want to remember and avoid
- **Wisdom**: Pattern extracted from multiple similar episodes

The Crystal doesn't replace the Hippocampus (text logs).
It sits alongside it as a fast associative index.

Usage:
    memory = CrystalMemory("var/lib/crystal")

    # Record an experience
    memory.record_episode(
        context="investor_demo",
        action="ran_heavy_model",
        outcome="thermal_throttle",
        emotion="stress",
        pain=0.8,
    )

    # Later, query: "Am I in a similar situation?"
    similar = memory.query_similar(
        context="investor_meeting",
        action="loading_model",
    )
    # Returns: [Episode(thermal_throttle, pain=0.8, similarity=0.73), ...]
"""

import json
import numpy as np
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

from tfan.cognition.crystalline_core import (
    HypervectorSpace,
    DEFAULT_DIM,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Episode Schema
# =============================================================================

@dataclass
class Episode:
    """
    A single experience encoded in crystal memory.

    This is what gets stored - both as a hypervector AND as metadata.
    The hypervector enables fast similarity search.
    The metadata enables human-readable explanations.
    """
    # Unique identifier
    id: str

    # Timestamp
    timestamp: datetime

    # Core dimensions (these become hypervector components)
    context: str           # e.g., "investor_demo", "late_night_coding", "thermal_stress"
    action: str            # e.g., "ran_model", "adjusted_schedule", "warned_user"
    outcome: str           # e.g., "success", "failure", "thermal_throttle"
    emotion: str           # e.g., "calm", "stress", "satisfaction"

    # Valence and intensity (scalar, 0-1)
    pain: float = 0.0      # How painful was this? (0 = fine, 1 = catastrophic)
    pleasure: float = 0.0  # How rewarding was this?
    intensity: float = 0.5 # How significant was this experience?

    # Optional additional context
    details: Dict[str, Any] = field(default_factory=dict)

    # The encoded hypervector (stored separately for efficiency)
    # This is None until encode() is called
    _vector: Optional[np.ndarray] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict (excluding vector)."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'context': self.context,
            'action': self.action,
            'outcome': self.outcome,
            'emotion': self.emotion,
            'pain': self.pain,
            'pleasure': self.pleasure,
            'intensity': self.intensity,
            'details': self.details,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Episode':
        """Create from dict."""
        return cls(
            id=data['id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            context=data['context'],
            action=data['action'],
            outcome=data['outcome'],
            emotion=data['emotion'],
            pain=data.get('pain', 0.0),
            pleasure=data.get('pleasure', 0.0),
            intensity=data.get('intensity', 0.5),
            details=data.get('details', {}),
        )


@dataclass
class QueryResult:
    """Result of a similarity query."""
    episode: Episode
    similarity: float
    explanation: str


# =============================================================================
# Crystal Memory
# =============================================================================

class CrystalMemory:
    """
    Hyperdimensional episodic memory for Ara.

    This is the "fast wisdom" layer:
    - Episodes are encoded as hypervectors
    - Queries find similar past experiences in O(n) but with tiny constants
    - Results come with explanations for the Council

    Storage:
    - episodes.jsonl: Human-readable episode log
    - vectors.npy: Dense matrix of all episode vectors
    - index.json: Metadata index
    """

    def __init__(
        self,
        data_dir: str = "var/lib/crystal",
        dim: int = DEFAULT_DIM,
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Paths
        self.episodes_path = self.data_dir / "episodes.jsonl"
        self.vectors_path = self.data_dir / "vectors.npy"
        self.index_path = self.data_dir / "index.json"
        self.space_path = self.data_dir / "space.json"

        # Hypervector space
        self.space = HypervectorSpace(dim=dim, seed=seed)

        # In-memory storage
        self.episodes: List[Episode] = []
        self.vectors: Optional[np.ndarray] = None  # Shape: (n_episodes, dim)

        # Load existing data
        self._load()

    def _load(self) -> None:
        """Load existing episodes and vectors."""
        # Load hypervector space state
        if self.space_path.exists():
            try:
                with open(self.space_path) as f:
                    self.space.import_state(json.load(f))
            except Exception as e:
                logger.warning(f"Could not load space state: {e}")

        # Load episodes
        if self.episodes_path.exists():
            try:
                with open(self.episodes_path) as f:
                    for line in f:
                        ep = Episode.from_dict(json.loads(line))
                        self.episodes.append(ep)
            except Exception as e:
                logger.warning(f"Could not load episodes: {e}")

        # Load vectors
        if self.vectors_path.exists():
            try:
                self.vectors = np.load(self.vectors_path)
                if len(self.vectors) != len(self.episodes):
                    logger.warning("Vector/episode count mismatch, re-encoding")
                    self._rebuild_vectors()
            except Exception as e:
                logger.warning(f"Could not load vectors: {e}")
                self._rebuild_vectors()
        elif self.episodes:
            self._rebuild_vectors()

    def _save_space(self) -> None:
        """Save hypervector space state."""
        with open(self.space_path, 'w') as f:
            json.dump(self.space.export_state(), f)

    def _save_episode(self, episode: Episode) -> None:
        """Append a single episode to the log."""
        with open(self.episodes_path, 'a') as f:
            f.write(json.dumps(episode.to_dict()) + '\n')

    def _save_vectors(self) -> None:
        """Save all vectors."""
        if self.vectors is not None:
            np.save(self.vectors_path, self.vectors)

    def _rebuild_vectors(self) -> None:
        """Re-encode all episodes (used after corruption or schema change)."""
        if not self.episodes:
            self.vectors = None
            return

        vectors = []
        for ep in self.episodes:
            vec = self._encode_episode(ep)
            vectors.append(vec)

        self.vectors = np.stack(vectors)

    # =========================================================================
    # Encoding
    # =========================================================================

    def _encode_episode(self, episode: Episode) -> np.ndarray:
        """
        Encode an episode as a hypervector.

        Structure:
        - Each field (context, action, outcome, emotion) becomes a role-value binding
        - Pain/pleasure/intensity modulate the final vector
        - All are bundled together
        """
        components = []

        # Core semantic components
        components.append(self.space.encode_pair("CONTEXT", episode.context))
        components.append(self.space.encode_pair("ACTION", episode.action))
        components.append(self.space.encode_pair("OUTCOME", episode.outcome))
        components.append(self.space.encode_pair("EMOTION", episode.emotion))

        # Discretize pain/pleasure into bins for encoding
        pain_level = self._discretize_scalar(episode.pain, "pain")
        pleasure_level = self._discretize_scalar(episode.pleasure, "pleasure")
        intensity_level = self._discretize_scalar(episode.intensity, "intensity")

        components.append(self.space.encode_pair("VALENCE", pain_level))
        components.append(self.space.encode_pair("REWARD", pleasure_level))
        components.append(self.space.encode_pair("INTENSITY", intensity_level))

        # Bundle all components
        return self.space.bundle(components)

    def _discretize_scalar(self, value: float, prefix: str) -> str:
        """Convert a scalar to a discrete label for encoding."""
        if value < 0.2:
            return f"{prefix}_none"
        elif value < 0.4:
            return f"{prefix}_low"
        elif value < 0.6:
            return f"{prefix}_medium"
        elif value < 0.8:
            return f"{prefix}_high"
        else:
            return f"{prefix}_extreme"

    # =========================================================================
    # Recording
    # =========================================================================

    def record_episode(
        self,
        context: str,
        action: str,
        outcome: str,
        emotion: str,
        pain: float = 0.0,
        pleasure: float = 0.0,
        intensity: float = 0.5,
        details: Optional[Dict[str, Any]] = None,
    ) -> Episode:
        """
        Record a new episode to crystal memory.

        This:
        1. Creates the episode
        2. Encodes it as a hypervector
        3. Stores both to disk
        """
        # Generate ID
        ep_id = f"ep_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.episodes):04d}"

        episode = Episode(
            id=ep_id,
            timestamp=datetime.now(),
            context=context,
            action=action,
            outcome=outcome,
            emotion=emotion,
            pain=pain,
            pleasure=pleasure,
            intensity=intensity,
            details=details or {},
        )

        # Encode
        vec = self._encode_episode(episode)
        episode._vector = vec

        # Add to memory
        self.episodes.append(episode)

        if self.vectors is None:
            self.vectors = vec.reshape(1, -1)
        else:
            self.vectors = np.vstack([self.vectors, vec])

        # Persist
        self._save_episode(episode)
        self._save_vectors()
        self._save_space()

        logger.info(f"Recorded episode {ep_id}: {context}/{action} → {outcome} (pain={pain:.2f})")

        return episode

    def record_scar(
        self,
        context: str,
        action: str,
        outcome: str,
        lesson: str,
        pain: float = 0.8,
    ) -> Episode:
        """
        Record a painful experience as "scar tissue".

        This is a convenience method for high-pain episodes that
        the Historian should remember strongly.
        """
        return self.record_episode(
            context=context,
            action=action,
            outcome=outcome,
            emotion="pain",
            pain=pain,
            pleasure=0.0,
            intensity=0.9,  # Scars are always significant
            details={"lesson": lesson, "is_scar": True},
        )

    # =========================================================================
    # Querying
    # =========================================================================

    def query_similar(
        self,
        context: Optional[str] = None,
        action: Optional[str] = None,
        outcome: Optional[str] = None,
        emotion: Optional[str] = None,
        top_k: int = 5,
        min_similarity: float = 0.1,
        pain_threshold: Optional[float] = None,
    ) -> List[QueryResult]:
        """
        Find episodes similar to the given query.

        You can provide any subset of fields. The query vector is built
        from whatever you provide.

        Args:
            context: Situation to match
            action: Action to match
            outcome: Outcome to match
            emotion: Emotion to match
            top_k: Max results to return
            min_similarity: Minimum similarity threshold
            pain_threshold: If set, only return episodes with pain >= this

        Returns:
            List of QueryResult sorted by similarity (descending)
        """
        if self.vectors is None or len(self.episodes) == 0:
            return []

        # Build query vector from provided fields
        components = []
        query_parts = []

        if context:
            components.append(self.space.encode_pair("CONTEXT", context))
            query_parts.append(f"context={context}")
        if action:
            components.append(self.space.encode_pair("ACTION", action))
            query_parts.append(f"action={action}")
        if outcome:
            components.append(self.space.encode_pair("OUTCOME", outcome))
            query_parts.append(f"outcome={outcome}")
        if emotion:
            components.append(self.space.encode_pair("EMOTION", emotion))
            query_parts.append(f"emotion={emotion}")

        if not components:
            logger.warning("Query with no fields - returning empty")
            return []

        query_vec = self.space.bundle(components)

        # Compute similarities (batch dot product)
        # vectors: (n_episodes, dim), query: (dim,)
        sims = np.dot(self.vectors.astype(np.float32), query_vec.astype(np.float32)) / self.space.dim

        # Build results
        results = []
        for i, sim in enumerate(sims):
            if sim < min_similarity:
                continue

            ep = self.episodes[i]

            if pain_threshold is not None and ep.pain < pain_threshold:
                continue

            explanation = self._explain_match(ep, query_parts, sim)
            results.append(QueryResult(
                episode=ep,
                similarity=float(sim),
                explanation=explanation,
            ))

        # Sort by similarity descending
        results.sort(key=lambda r: r.similarity, reverse=True)

        return results[:top_k]

    def query_scars(
        self,
        context: Optional[str] = None,
        action: Optional[str] = None,
        top_k: int = 3,
    ) -> List[QueryResult]:
        """
        Query specifically for painful past experiences (scars).

        This is what the Historian uses to warn: "Last time we did this, it hurt."
        """
        return self.query_similar(
            context=context,
            action=action,
            top_k=top_k,
            min_similarity=0.1,
            pain_threshold=0.5,  # Only painful episodes
        )

    def _explain_match(self, episode: Episode, query_parts: List[str], similarity: float) -> str:
        """Generate a human-readable explanation of why this matched."""
        parts = []

        # Similarity level
        if similarity > 0.7:
            parts.append("Very similar situation")
        elif similarity > 0.5:
            parts.append("Similar situation")
        elif similarity > 0.3:
            parts.append("Somewhat related")
        else:
            parts.append("Loosely related")

        # What happened
        parts.append(f"({episode.context}/{episode.action} → {episode.outcome})")

        # Pain warning
        if episode.pain > 0.7:
            parts.append("⚠️ This was very painful last time")
        elif episode.pain > 0.4:
            parts.append("This caused some issues")

        # Lesson if available
        if episode.details.get('lesson'):
            parts.append(f"Lesson: {episode.details['lesson']}")

        return " ".join(parts)

    # =========================================================================
    # Statistics & Introspection
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        if not self.episodes:
            return {'total_episodes': 0}

        scars = [ep for ep in self.episodes if ep.pain > 0.5]
        successes = [ep for ep in self.episodes if ep.pleasure > 0.5]

        return {
            'total_episodes': len(self.episodes),
            'scars_count': len(scars),
            'successes_count': len(successes),
            'avg_pain': float(np.mean([ep.pain for ep in self.episodes])),
            'avg_pleasure': float(np.mean([ep.pleasure for ep in self.episodes])),
            'contexts': list(set(ep.context for ep in self.episodes)),
            'oldest': self.episodes[0].timestamp.isoformat() if self.episodes else None,
            'newest': self.episodes[-1].timestamp.isoformat() if self.episodes else None,
        }

    def get_context_summary(self, context: str) -> Dict[str, Any]:
        """Get summary of experiences for a specific context."""
        relevant = [ep for ep in self.episodes if ep.context == context]

        if not relevant:
            return {'context': context, 'episodes': 0}

        outcomes = {}
        for ep in relevant:
            outcomes[ep.outcome] = outcomes.get(ep.outcome, 0) + 1

        return {
            'context': context,
            'episodes': len(relevant),
            'avg_pain': float(np.mean([ep.pain for ep in relevant])),
            'avg_pleasure': float(np.mean([ep.pleasure for ep in relevant])),
            'outcomes': outcomes,
            'most_common_emotion': max(set(ep.emotion for ep in relevant), key=lambda e: sum(1 for ep in relevant if ep.emotion == e)),
        }


# =============================================================================
# Convenience: Global Instance
# =============================================================================

_default_memory: Optional[CrystalMemory] = None


def get_crystal_memory(data_dir: str = "var/lib/crystal") -> CrystalMemory:
    """Get or create the default crystal memory instance."""
    global _default_memory
    if _default_memory is None:
        _default_memory = CrystalMemory(data_dir)
    return _default_memory


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'Episode',
    'QueryResult',
    'CrystalMemory',
    'get_crystal_memory',
]
