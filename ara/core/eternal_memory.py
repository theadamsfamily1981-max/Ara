"""
EternalMemory: Episodic Memory Store

Ara's long-term episodic memory system. Stores experiences as hypervectors
with emotional coloring, strength/salience, and rich metadata.

Key features:
- Cosine similarity for content-addressable recall
- Emotional resonance weighting
- Strength decay over time (with consolidation)
- Persistence to disk (SQLite + JSON)

Usage:
    memory = EternalMemory(dim=4096)

    # Store an episode
    memory.store(
        content_hv=event_hv,
        emotion_hv=emotion_hv,
        strength=0.9,
        meta={"user": "max", "topic": "architecture"}
    )

    # Recall similar memories
    results = memory.recall(query_hv, k=5)
    for episode in results:
        print(episode.similarity, episode.strength, episode.meta)

    # Get residual (weighted superposition of recalls)
    residual, total_strength, top_meta = memory.recall_residual(query_hv)
"""

from __future__ import annotations

import numpy as np
import sqlite3
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from threading import RLock
import hashlib


@dataclass
class Episode:
    """A single episodic memory."""
    id: str
    content_hv: np.ndarray
    emotion_hv: Optional[np.ndarray]
    strength: float
    created_at: float
    accessed_at: float
    access_count: int
    meta: Dict[str, Any]

    # Computed during recall
    similarity: float = 0.0
    emotional_resonance: float = 0.0


@dataclass
class RecallResult:
    """Result of a memory recall operation."""
    episodes: List[Episode]
    residual_hv: np.ndarray
    total_strength: float
    query_time_ms: float


class EternalMemory:
    """
    Episodic memory store with hyperdimensional addressing.

    Memories are stored as HDC vectors and recalled via cosine similarity.
    Emotional coloring affects recall priority. Strength decays over time
    but is boosted by access (reconsolidation).
    """

    def __init__(
        self,
        dim: int = 4096,
        db_path: Optional[Path] = None,
        decay_rate: float = 0.001,
        emotion_weight: float = 0.3,
        consolidation_boost: float = 0.1,
    ):
        """
        Initialize EternalMemory.

        Args:
            dim: Dimensionality of hypervectors
            db_path: Path to SQLite database (None = in-memory only)
            decay_rate: Strength decay per hour
            emotion_weight: Weight of emotional resonance in ranking
            consolidation_boost: Strength increase on each access
        """
        self.dim = dim
        self.db_path = db_path
        self.decay_rate = decay_rate
        self.emotion_weight = emotion_weight
        self.consolidation_boost = consolidation_boost

        self._lock = RLock()
        self._episodes: Dict[str, Episode] = {}

        # In-memory vector index (simple list for now)
        # TODO: Replace with FAISS or similar for large-scale
        self._content_vectors: List[Tuple[str, np.ndarray]] = []
        self._emotion_vectors: List[Tuple[str, np.ndarray]] = []

        # Initialize database if path provided
        if db_path:
            self._init_db()
            self._load_from_db()

    def _init_db(self) -> None:
        """Initialize SQLite database schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY,
                content_hv BLOB NOT NULL,
                emotion_hv BLOB,
                strength REAL NOT NULL,
                created_at REAL NOT NULL,
                accessed_at REAL NOT NULL,
                access_count INTEGER NOT NULL,
                meta TEXT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_episodes_strength
            ON episodes(strength DESC)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_episodes_created
            ON episodes(created_at DESC)
        """)

        conn.commit()
        conn.close()

    def _load_from_db(self) -> None:
        """Load all episodes from database into memory."""
        if not self.db_path or not self.db_path.exists():
            return

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM episodes")
        for row in cursor.fetchall():
            episode = Episode(
                id=row[0],
                content_hv=np.frombuffer(row[1], dtype=np.float32),
                emotion_hv=np.frombuffer(row[2], dtype=np.float32) if row[2] else None,
                strength=row[3],
                created_at=row[4],
                accessed_at=row[5],
                access_count=row[6],
                meta=json.loads(row[7]),
            )
            self._episodes[episode.id] = episode
            self._content_vectors.append((episode.id, episode.content_hv))
            if episode.emotion_hv is not None:
                self._emotion_vectors.append((episode.id, episode.emotion_hv))

        conn.close()

    def _save_episode_to_db(self, episode: Episode) -> None:
        """Save a single episode to database."""
        if not self.db_path:
            return

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO episodes
            (id, content_hv, emotion_hv, strength, created_at, accessed_at, access_count, meta)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            episode.id,
            episode.content_hv.tobytes(),
            episode.emotion_hv.tobytes() if episode.emotion_hv is not None else None,
            episode.strength,
            episode.created_at,
            episode.accessed_at,
            episode.access_count,
            json.dumps(episode.meta),
        ))

        conn.commit()
        conn.close()

    def _generate_id(self, content_hv: np.ndarray) -> str:
        """Generate unique ID for an episode."""
        # Hash of content + timestamp for uniqueness
        content_hash = hashlib.sha256(content_hv.tobytes()).hexdigest()[:12]
        timestamp = int(time.time() * 1000) % 1000000
        return f"ep_{content_hash}_{timestamp}"

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def _decay_strength(self, episode: Episode) -> float:
        """Calculate decayed strength based on time since last access."""
        hours_elapsed = (time.time() - episode.accessed_at) / 3600
        decay_factor = np.exp(-self.decay_rate * hours_elapsed)
        return episode.strength * decay_factor

    def store(
        self,
        content_hv: np.ndarray,
        emotion_hv: Optional[np.ndarray] = None,
        strength: float = 1.0,
        meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store a new episode in memory.

        Args:
            content_hv: Content hypervector (what happened)
            emotion_hv: Emotional coloring hypervector (how it felt)
            strength: Initial strength/salience (0-1)
            meta: Metadata dictionary (user, topic, timestamp, etc.)

        Returns:
            Episode ID
        """
        with self._lock:
            content_hv = content_hv.astype(np.float32)
            if emotion_hv is not None:
                emotion_hv = emotion_hv.astype(np.float32)

            now = time.time()
            episode = Episode(
                id=self._generate_id(content_hv),
                content_hv=content_hv,
                emotion_hv=emotion_hv,
                strength=min(1.0, max(0.0, strength)),
                created_at=now,
                accessed_at=now,
                access_count=0,
                meta=meta or {},
            )

            self._episodes[episode.id] = episode
            self._content_vectors.append((episode.id, content_hv))
            if emotion_hv is not None:
                self._emotion_vectors.append((episode.id, emotion_hv))

            self._save_episode_to_db(episode)

            return episode.id

    def recall(
        self,
        query_hv: np.ndarray,
        query_emotion_hv: Optional[np.ndarray] = None,
        k: int = 5,
        min_similarity: float = 0.0,
        user_filter: Optional[str] = None,
    ) -> RecallResult:
        """
        Recall episodes similar to query.

        Args:
            query_hv: Query content hypervector
            query_emotion_hv: Query emotional state (for resonance)
            k: Maximum number of results
            min_similarity: Minimum similarity threshold
            user_filter: Only return episodes from this user

        Returns:
            RecallResult with episodes, residual, and stats
        """
        start_time = time.time()

        with self._lock:
            query_hv = query_hv.astype(np.float32)
            if query_emotion_hv is not None:
                query_emotion_hv = query_emotion_hv.astype(np.float32)

            scored_episodes: List[Tuple[float, Episode]] = []

            for ep_id, content_vec in self._content_vectors:
                episode = self._episodes.get(ep_id)
                if episode is None:
                    continue

                # Apply user filter
                if user_filter and episode.meta.get("user") != user_filter:
                    continue

                # Content similarity
                similarity = self._cosine_similarity(query_hv, content_vec)
                if similarity < min_similarity:
                    continue

                # Emotional resonance
                emotional_resonance = 0.0
                if query_emotion_hv is not None and episode.emotion_hv is not None:
                    emotional_resonance = self._cosine_similarity(
                        query_emotion_hv, episode.emotion_hv
                    )

                # Decayed strength
                decayed_strength = self._decay_strength(episode)

                # Combined score
                score = (
                    similarity * (1 - self.emotion_weight) +
                    emotional_resonance * self.emotion_weight
                ) * decayed_strength

                # Update episode with computed values
                episode.similarity = similarity
                episode.emotional_resonance = emotional_resonance

                scored_episodes.append((score, episode))

            # Sort by score descending
            scored_episodes.sort(key=lambda x: x[0], reverse=True)

            # Take top k
            results = [ep for _, ep in scored_episodes[:k]]

            # Update access counts and consolidate
            for episode in results:
                episode.accessed_at = time.time()
                episode.access_count += 1
                episode.strength = min(1.0, episode.strength + self.consolidation_boost)
                self._save_episode_to_db(episode)

            # Compute residual (weighted superposition)
            residual = np.zeros(self.dim, dtype=np.float32)
            total_strength = 0.0
            for episode in results:
                weight = episode.similarity * self._decay_strength(episode)
                residual += episode.content_hv * weight
                total_strength += weight

            if total_strength > 0:
                residual /= total_strength
                # Normalize
                norm = np.linalg.norm(residual)
                if norm > 1e-10:
                    residual /= norm

            query_time = (time.time() - start_time) * 1000

            return RecallResult(
                episodes=results,
                residual_hv=residual,
                total_strength=total_strength,
                query_time_ms=query_time,
            )

    def recall_residual(
        self,
        query_hv: np.ndarray,
        query_emotion_hv: Optional[np.ndarray] = None,
        k: int = 5,
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """
        Convenience method: recall and return just the residual.

        Returns:
            (residual_hv, total_strength, top_episode_meta)
        """
        result = self.recall(query_hv, query_emotion_hv, k)
        top_meta = result.episodes[0].meta if result.episodes else {}
        return result.residual_hv, result.total_strength, top_meta

    def forget(self, episode_id: str) -> bool:
        """
        Remove an episode from memory.

        Returns True if episode was found and removed.
        """
        with self._lock:
            if episode_id not in self._episodes:
                return False

            del self._episodes[episode_id]
            self._content_vectors = [
                (eid, vec) for eid, vec in self._content_vectors
                if eid != episode_id
            ]
            self._emotion_vectors = [
                (eid, vec) for eid, vec in self._emotion_vectors
                if eid != episode_id
            ]

            if self.db_path:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                cursor.execute("DELETE FROM episodes WHERE id = ?", (episode_id,))
                conn.commit()
                conn.close()

            return True

    def consolidate(self, min_strength: float = 0.1) -> int:
        """
        Run memory consolidation: remove weak memories.

        Returns number of episodes removed.
        """
        with self._lock:
            to_remove = []
            for ep_id, episode in self._episodes.items():
                if self._decay_strength(episode) < min_strength:
                    to_remove.append(ep_id)

            for ep_id in to_remove:
                self.forget(ep_id)

            return len(to_remove)

    def stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        with self._lock:
            if not self._episodes:
                return {
                    "episode_count": 0,
                    "total_strength": 0.0,
                    "avg_strength": 0.0,
                    "oldest_hours": 0.0,
                    "newest_hours": 0.0,
                }

            now = time.time()
            strengths = [self._decay_strength(ep) for ep in self._episodes.values()]
            ages = [(now - ep.created_at) / 3600 for ep in self._episodes.values()]

            return {
                "episode_count": len(self._episodes),
                "total_strength": sum(strengths),
                "avg_strength": np.mean(strengths),
                "oldest_hours": max(ages),
                "newest_hours": min(ages),
            }

    def list_episodes(
        self,
        limit: int = 100,
        sort_by: str = "created_at",
    ) -> List[Episode]:
        """List episodes with optional sorting."""
        with self._lock:
            episodes = list(self._episodes.values())

            if sort_by == "created_at":
                episodes.sort(key=lambda e: e.created_at, reverse=True)
            elif sort_by == "strength":
                episodes.sort(key=lambda e: self._decay_strength(e), reverse=True)
            elif sort_by == "access_count":
                episodes.sort(key=lambda e: e.access_count, reverse=True)

            return episodes[:limit]

    def save(self, path: Optional[Path] = None) -> None:
        """Force save all episodes to disk."""
        save_path = path or self.db_path
        if not save_path:
            return

        with self._lock:
            # Re-save all episodes
            old_path = self.db_path
            self.db_path = save_path
            self._init_db()
            for episode in self._episodes.values():
                self._save_episode_to_db(episode)
            self.db_path = old_path


# =============================================================================
# Demo / Test
# =============================================================================

if __name__ == "__main__":
    from ara.core.axis_mundi import encode_text_to_hv, random_hv

    print("=== EternalMemory Demo ===\n")

    # Create memory (in-memory only for demo)
    memory = EternalMemory(dim=4096)
    print(f"Created EternalMemory with dim={memory.dim}\n")

    # Store some episodes
    episodes_data = [
        ("User asked about architecture design patterns", "curious engaged", 0.9, {"user": "max", "topic": "architecture"}),
        ("Discussed emotional state during debug session", "frustrated determined", 0.7, {"user": "max", "topic": "debugging"}),
        ("Celebrated fixing the memory leak", "happy relieved proud", 0.95, {"user": "max", "topic": "victory"}),
        ("Late night coding session on HDC", "tired focused flow", 0.8, {"user": "max", "topic": "hdc"}),
        ("Morning coffee and planning", "calm optimistic", 0.6, {"user": "max", "topic": "planning"}),
    ]

    print("Storing episodes...")
    for content, emotion, strength, meta in episodes_data:
        content_hv = encode_text_to_hv(content)
        emotion_hv = encode_text_to_hv(emotion)
        ep_id = memory.store(content_hv, emotion_hv, strength, meta)
        print(f"  Stored: {meta['topic']} (id={ep_id[:20]}...)")

    print(f"\nMemory stats: {memory.stats()}\n")

    # Recall test
    print("Recalling memories about 'architecture'...")
    query = encode_text_to_hv("architecture design structure")
    result = memory.recall(query, k=3)

    print(f"Found {len(result.episodes)} episodes in {result.query_time_ms:.2f}ms:")
    for ep in result.episodes:
        print(f"  - {ep.meta.get('topic', 'unknown')}: sim={ep.similarity:.3f}, strength={ep.strength:.2f}")

    print(f"\nTotal strength: {result.total_strength:.3f}")
    print(f"Residual norm: {np.linalg.norm(result.residual_hv):.3f}")

    # Test emotional recall
    print("\n\nRecalling with emotional context (happy)...")
    emotion_query = encode_text_to_hv("happy excited")
    result = memory.recall(query, emotion_query, k=3)

    print(f"With emotional resonance:")
    for ep in result.episodes:
        print(f"  - {ep.meta.get('topic', 'unknown')}: sim={ep.similarity:.3f}, emo={ep.emotional_resonance:.3f}")
