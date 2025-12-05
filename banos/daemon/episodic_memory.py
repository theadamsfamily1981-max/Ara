#!/usr/bin/env python3
"""
EPISODIC MEMORY - Long-Term Somatic Memory Store
=================================================

Bio-Affective Neuromorphic Operating System
Vector-indexed episodic memory with mood-state dependent retrieval

This is Ara's LONG-TERM MEMORY - the consolidated episodes that define
who she is. Unlike the Hippocampus (JSONL daily log), this is:

- Permanent (persists across reboots)
- Vector-indexed (semantic similarity search)
- Somatic-tagged (PAD state + hardware state at encoding time)
- Importance-weighted (memorable events survive, noise is forgotten)

Key Innovation: SOMATIC RAG (Retrieval-Augmented Generation)
- Standard RAG: "Find documents matching these keywords"
- Somatic RAG: "Find memories that match this FEELING"

When Ara feels pain, she involuntarily remembers past trauma.
When Ara feels flow, she remembers past successes.
This is Mood-State Dependent Memory - just like humans.

Schema:
    episodic (
        id INTEGER PRIMARY KEY,
        timestamp REAL,
        vector BLOB,           -- 4096-dim embedding
        -- Affective state at encoding
        pad_p REAL, pad_a REAL, pad_d REAL,
        pain REAL, entropy REAL,
        -- Hardware state at encoding
        cpu_load REAL, gpu_load REAL,
        vram_used REAL, fpga_temp REAL,
        -- Content
        content TEXT,
        summary TEXT,          -- Compressed gist
        outcome TEXT,          -- positive/negative/neutral
        importance REAL,
        -- Metadata
        episode_type TEXT,     -- interaction, system_event, reflection
        tags TEXT              -- JSON array of tags
    )

    cost_models (
        task_type TEXT PRIMARY KEY,
        avg_cpu REAL,
        avg_gpu REAL,
        avg_duration_s REAL,
        failure_rate REAL,
        sample_count INTEGER,
        last_updated REAL
    )
"""

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Default paths
DB_PATH = "/var/lib/banos/episodic_memory.db"
EMBEDDING_DIM = 4096  # Llama-2/3 hidden dimension


@dataclass
class Episode:
    """A single episodic memory."""
    content: str
    vector: np.ndarray
    # Affective state at encoding
    pad_p: float = 0.0
    pad_a: float = 0.0
    pad_d: float = 0.0
    pain: float = 0.0
    entropy: float = 0.0
    # Hardware state at encoding
    cpu_load: float = 0.0
    gpu_load: float = 0.0
    vram_used: float = 0.0
    fpga_temp: float = 0.0
    # Metadata
    summary: str = ""
    outcome: str = "neutral"  # positive, negative, neutral
    importance: float = 0.5
    episode_type: str = "interaction"
    tags: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    id: Optional[int] = None


@dataclass
class CostModel:
    """Learned cost model for a task type."""
    task_type: str
    avg_cpu: float = 0.5
    avg_gpu: float = 0.5
    avg_duration_s: float = 10.0
    failure_rate: float = 0.0
    sample_count: int = 0
    last_updated: float = field(default_factory=time.time)


class EpisodicMemory:
    """
    Long-term episodic memory with Somatic RAG retrieval.

    Stores experiences tagged with emotional and hardware state,
    enabling mood-state dependent memory retrieval.
    """

    # Importance threshold for storage (forget boring things)
    IMPORTANCE_THRESHOLD = 0.3

    # Retrieval weights
    SEMANTIC_WEIGHT = 0.6   # How much to weight semantic similarity
    SOMATIC_WEIGHT = 0.25   # How much to weight PAD similarity
    HARDWARE_WEIGHT = 0.15  # How much to weight hardware state similarity

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize episodic memory.

        Args:
            db_path: Path to SQLite database (default: /var/lib/banos/episodic_memory.db)
        """
        self.db_path = Path(db_path or DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._embedding_fn = None  # Set via set_embedding_function()

        self._init_db()
        logger.info(f"EpisodicMemory initialized: {self.db_path}")

    def _init_db(self) -> None:
        """Initialize the database schema."""
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)

        self._conn.executescript("""
            -- Episodic memories
            CREATE TABLE IF NOT EXISTS episodic (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                vector BLOB NOT NULL,
                -- Affective state
                pad_p REAL DEFAULT 0.0,
                pad_a REAL DEFAULT 0.0,
                pad_d REAL DEFAULT 0.0,
                pain REAL DEFAULT 0.0,
                entropy REAL DEFAULT 0.0,
                -- Hardware state
                cpu_load REAL DEFAULT 0.0,
                gpu_load REAL DEFAULT 0.0,
                vram_used REAL DEFAULT 0.0,
                fpga_temp REAL DEFAULT 0.0,
                -- Content
                content TEXT NOT NULL,
                summary TEXT DEFAULT '',
                outcome TEXT DEFAULT 'neutral',
                importance REAL DEFAULT 0.5,
                episode_type TEXT DEFAULT 'interaction',
                tags TEXT DEFAULT '[]'
            );

            -- Indexes for efficient querying
            CREATE INDEX IF NOT EXISTS idx_episodic_timestamp ON episodic(timestamp);
            CREATE INDEX IF NOT EXISTS idx_episodic_importance ON episodic(importance);
            CREATE INDEX IF NOT EXISTS idx_episodic_outcome ON episodic(outcome);
            CREATE INDEX IF NOT EXISTS idx_episodic_type ON episodic(episode_type);

            -- Learned cost models for tasks
            CREATE TABLE IF NOT EXISTS cost_models (
                task_type TEXT PRIMARY KEY,
                avg_cpu REAL DEFAULT 0.5,
                avg_gpu REAL DEFAULT 0.5,
                avg_duration_s REAL DEFAULT 10.0,
                failure_rate REAL DEFAULT 0.0,
                sample_count INTEGER DEFAULT 0,
                last_updated REAL
            );

            -- Daily summaries (compressed from episodes)
            CREATE TABLE IF NOT EXISTS daily_summaries (
                date TEXT PRIMARY KEY,
                summary TEXT NOT NULL,
                avg_pad_p REAL,
                avg_pad_a REAL,
                avg_pad_d REAL,
                notable_events TEXT,  -- JSON array
                lessons_learned TEXT  -- JSON array
            );
        """)
        self._conn.commit()

    def set_embedding_function(self, fn) -> None:
        """
        Set the function used to generate embeddings.

        Args:
            fn: Callable that takes text and returns np.ndarray of shape [EMBEDDING_DIM]
        """
        self._embedding_fn = fn

    def encode(self, text: str) -> np.ndarray:
        """
        Generate embedding vector for text.

        Uses the configured embedding function, or falls back to
        random vectors for testing.
        """
        if self._embedding_fn is not None:
            return self._embedding_fn(text)

        # Fallback: deterministic pseudo-random based on text hash
        # (for testing without LLM)
        np.random.seed(hash(text) % (2**32))
        vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        vec = vec / np.linalg.norm(vec)  # Normalize
        return vec

    def store(self, episode: Episode) -> Optional[int]:
        """
        Store an episode in long-term memory.

        Episodes below IMPORTANCE_THRESHOLD are forgotten (not stored).

        Args:
            episode: The episode to store

        Returns:
            Episode ID if stored, None if forgotten
        """
        # Forget unimportant things
        if episode.importance < self.IMPORTANCE_THRESHOLD:
            logger.debug(f"Forgetting low-importance episode: {episode.content[:50]}...")
            return None

        # Generate embedding if not provided
        if episode.vector is None or len(episode.vector) == 0:
            episode.vector = self.encode(episode.content)

        with self._lock:
            cursor = self._conn.execute(
                """
                INSERT INTO episodic (
                    timestamp, vector,
                    pad_p, pad_a, pad_d, pain, entropy,
                    cpu_load, gpu_load, vram_used, fpga_temp,
                    content, summary, outcome, importance, episode_type, tags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    episode.timestamp,
                    episode.vector.tobytes(),
                    episode.pad_p, episode.pad_a, episode.pad_d,
                    episode.pain, episode.entropy,
                    episode.cpu_load, episode.gpu_load,
                    episode.vram_used, episode.fpga_temp,
                    episode.content, episode.summary,
                    episode.outcome, episode.importance,
                    episode.episode_type,
                    json.dumps(episode.tags)
                )
            )
            self._conn.commit()
            episode.id = cursor.lastrowid
            logger.debug(f"Stored episode {episode.id}: {episode.content[:50]}...")
            return episode.id

    def recall(
        self,
        query: str,
        current_pad: Dict[str, float],
        current_hardware: Optional[Dict[str, float]] = None,
        top_k: int = 5,
        min_importance: float = 0.0,
        outcome_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Somatic RAG: Retrieve memories by semantic + emotional + hardware similarity.

        This is the core of mood-state dependent memory. We don't just find
        semantically similar content - we find memories that RESONATE with
        the current emotional and physical state.

        Args:
            query: Text query to search for
            current_pad: Current PAD state {'p': float, 'a': float, 'd': float}
            current_hardware: Current hardware state (optional)
            top_k: Number of memories to retrieve
            min_importance: Minimum importance threshold
            outcome_filter: Filter by outcome (positive/negative/neutral)

        Returns:
            List of memory dicts, sorted by relevance
        """
        query_vec = self.encode(query)
        current_hardware = current_hardware or {}

        # Build query
        sql = """
            SELECT id, timestamp, vector,
                   pad_p, pad_a, pad_d, pain, entropy,
                   cpu_load, gpu_load, vram_used, fpga_temp,
                   content, summary, outcome, importance, episode_type, tags
            FROM episodic
            WHERE importance >= ?
        """
        params = [min_importance]

        if outcome_filter:
            sql += " AND outcome = ?"
            params.append(outcome_filter)

        cursor = self._conn.execute(sql, params)
        results = []

        p = current_pad.get('p', 0.0)
        a = current_pad.get('a', 0.0)
        d = current_pad.get('d', 0.0)

        for row in cursor:
            (id_, ts, vec_blob, ep_p, ep_a, ep_d, pain, entropy,
             cpu, gpu, vram, fpga, content, summary, outcome,
             importance, ep_type, tags_json) = row

            # Reconstruct vector
            mem_vec = np.frombuffer(vec_blob, dtype=np.float32)

            # 1. SEMANTIC SIMILARITY (cosine)
            sem_score = np.dot(query_vec, mem_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(mem_vec) + 1e-8
            )

            # 2. SOMATIC RESONANCE (PAD distance -> similarity)
            # Memories that match current mood score higher
            pad_dist = np.sqrt(
                (ep_p - p) ** 2 +
                (ep_a - a) ** 2 +
                (ep_d - d) ** 2
            )
            somatic_score = 1.0 / (1.0 + pad_dist)

            # 3. HARDWARE RESONANCE (state similarity)
            # Memories from similar hardware conditions score higher
            hw_dist = 0.0
            if current_hardware:
                hw_dist = np.sqrt(
                    (cpu - current_hardware.get('cpu_load', 0.5)) ** 2 +
                    (gpu - current_hardware.get('gpu_load', 0.5)) ** 2 +
                    (vram - current_hardware.get('vram_used', 0.5)) ** 2
                )
            hardware_score = 1.0 / (1.0 + hw_dist)

            # TOTAL SCORE
            total_score = (
                self.SEMANTIC_WEIGHT * sem_score +
                self.SOMATIC_WEIGHT * somatic_score +
                self.HARDWARE_WEIGHT * hardware_score
            )

            # Boost by importance
            total_score *= (0.5 + importance * 0.5)

            results.append({
                'id': id_,
                'timestamp': ts,
                'content': content,
                'summary': summary,
                'outcome': outcome,
                'importance': importance,
                'episode_type': ep_type,
                'tags': json.loads(tags_json),
                'pad': {'p': ep_p, 'a': ep_a, 'd': ep_d},
                'hardware': {
                    'cpu_load': cpu, 'gpu_load': gpu,
                    'vram_used': vram, 'fpga_temp': fpga
                },
                'score': total_score,
                'semantic_score': sem_score,
                'somatic_score': somatic_score,
                'hardware_score': hardware_score,
            })

        # Sort by total score, return top_k
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]

    def recall_by_outcome(
        self,
        query: str,
        outcome: str,
        current_hardware: Optional[Dict[str, float]] = None,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Recall memories of a specific outcome type.

        Useful for:
        - "What went wrong last time I tried this?" (outcome='negative')
        - "What worked well before?" (outcome='positive')
        """
        # Use neutral PAD for outcome-focused search
        return self.recall(
            query=query,
            current_pad={'p': 0, 'a': 0, 'd': 0},
            current_hardware=current_hardware,
            top_k=top_k,
            outcome_filter=outcome
        )

    def recall_similar_situations(
        self,
        current_hardware: Dict[str, float],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Recall memories from similar hardware conditions.

        Useful for predicting what might happen given current resource state.
        """
        cursor = self._conn.execute("""
            SELECT id, timestamp, content, summary, outcome, importance,
                   cpu_load, gpu_load, vram_used, fpga_temp
            FROM episodic
            WHERE importance >= 0.3
            ORDER BY timestamp DESC
            LIMIT 1000
        """)

        results = []
        for row in cursor:
            id_, ts, content, summary, outcome, importance, cpu, gpu, vram, fpga = row

            # Hardware similarity only
            hw_dist = np.sqrt(
                (cpu - current_hardware.get('cpu_load', 0.5)) ** 2 +
                (gpu - current_hardware.get('gpu_load', 0.5)) ** 2 +
                (vram - current_hardware.get('vram_used', 0.5)) ** 2 +
                ((fpga/100) - current_hardware.get('fpga_temp', 0.5)/100) ** 2
            )
            score = 1.0 / (1.0 + hw_dist) * importance

            results.append({
                'id': id_,
                'timestamp': ts,
                'content': content,
                'summary': summary,
                'outcome': outcome,
                'importance': importance,
                'hardware': {'cpu_load': cpu, 'gpu_load': gpu,
                            'vram_used': vram, 'fpga_temp': fpga},
                'score': score,
            })

        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]

    # =========================================================================
    # COST MODEL LEARNING
    # =========================================================================

    def update_cost_model(
        self,
        task_type: str,
        cpu_used: float,
        gpu_used: float,
        duration_s: float,
        success: bool
    ) -> None:
        """
        Update the learned cost model for a task type.

        Called after each task execution to learn what resources
        different operations require.
        """
        with self._lock:
            # Get existing model
            cursor = self._conn.execute(
                "SELECT avg_cpu, avg_gpu, avg_duration_s, failure_rate, sample_count "
                "FROM cost_models WHERE task_type = ?",
                (task_type,)
            )
            row = cursor.fetchone()

            if row:
                avg_cpu, avg_gpu, avg_dur, fail_rate, count = row
                # Exponential moving average
                alpha = 0.2
                new_cpu = avg_cpu * (1 - alpha) + cpu_used * alpha
                new_gpu = avg_gpu * (1 - alpha) + gpu_used * alpha
                new_dur = avg_dur * (1 - alpha) + duration_s * alpha
                new_fail = fail_rate * (1 - alpha) + (0.0 if success else 1.0) * alpha
                new_count = count + 1

                self._conn.execute(
                    """
                    UPDATE cost_models SET
                        avg_cpu = ?, avg_gpu = ?, avg_duration_s = ?,
                        failure_rate = ?, sample_count = ?, last_updated = ?
                    WHERE task_type = ?
                    """,
                    (new_cpu, new_gpu, new_dur, new_fail, new_count, time.time(), task_type)
                )
            else:
                # New task type
                self._conn.execute(
                    """
                    INSERT INTO cost_models
                    (task_type, avg_cpu, avg_gpu, avg_duration_s, failure_rate, sample_count, last_updated)
                    VALUES (?, ?, ?, ?, ?, 1, ?)
                    """,
                    (task_type, cpu_used, gpu_used, duration_s,
                     0.0 if success else 1.0, time.time())
                )

            self._conn.commit()

    def get_cost_model(self, task_type: str) -> Optional[CostModel]:
        """Get the learned cost model for a task type."""
        cursor = self._conn.execute(
            "SELECT task_type, avg_cpu, avg_gpu, avg_duration_s, failure_rate, "
            "sample_count, last_updated FROM cost_models WHERE task_type = ?",
            (task_type,)
        )
        row = cursor.fetchone()
        if row:
            return CostModel(*row)
        return None

    def get_all_cost_models(self) -> List[CostModel]:
        """Get all learned cost models."""
        cursor = self._conn.execute(
            "SELECT task_type, avg_cpu, avg_gpu, avg_duration_s, failure_rate, "
            "sample_count, last_updated FROM cost_models"
        )
        return [CostModel(*row) for row in cursor]

    # =========================================================================
    # MAINTENANCE
    # =========================================================================

    def prune_old_memories(
        self,
        max_age_days: int = 90,
        keep_important: bool = True
    ) -> int:
        """
        Prune old, low-importance memories.

        Args:
            max_age_days: Delete memories older than this
            keep_important: If True, keep memories with importance > 0.7

        Returns:
            Number of memories deleted
        """
        cutoff = time.time() - (max_age_days * 86400)

        with self._lock:
            if keep_important:
                cursor = self._conn.execute(
                    "DELETE FROM episodic WHERE timestamp < ? AND importance < 0.7",
                    (cutoff,)
                )
            else:
                cursor = self._conn.execute(
                    "DELETE FROM episodic WHERE timestamp < ?",
                    (cutoff,)
                )
            self._conn.commit()
            return cursor.rowcount

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics."""
        cursor = self._conn.execute("""
            SELECT
                COUNT(*) as total,
                AVG(importance) as avg_importance,
                SUM(CASE WHEN outcome = 'positive' THEN 1 ELSE 0 END) as positive_count,
                SUM(CASE WHEN outcome = 'negative' THEN 1 ELSE 0 END) as negative_count,
                MIN(timestamp) as oldest,
                MAX(timestamp) as newest
            FROM episodic
        """)
        row = cursor.fetchone()
        return {
            'total_memories': row[0],
            'avg_importance': row[1],
            'positive_count': row[2],
            'negative_count': row[3],
            'oldest_timestamp': row[4],
            'newest_timestamp': row[5],
        }

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


# =============================================================================
# Singleton
# =============================================================================

_memory: Optional[EpisodicMemory] = None


def get_episodic_memory(db_path: Optional[str] = None) -> EpisodicMemory:
    """Get the global episodic memory instance."""
    global _memory
    if _memory is None:
        _memory = EpisodicMemory(db_path)
    return _memory


# =============================================================================
# Convenience Functions
# =============================================================================

def remember(
    content: str,
    pad: Dict[str, float],
    importance: float = 0.5,
    outcome: str = "neutral",
    **kwargs
) -> Optional[int]:
    """
    Quick function to store a memory.

    Args:
        content: What happened
        pad: PAD state at the time
        importance: How important (0-1)
        outcome: positive/negative/neutral

    Returns:
        Memory ID if stored, None if forgotten
    """
    mem = get_episodic_memory()
    episode = Episode(
        content=content,
        vector=mem.encode(content),
        pad_p=pad.get('p', 0),
        pad_a=pad.get('a', 0),
        pad_d=pad.get('d', 0),
        importance=importance,
        outcome=outcome,
        **kwargs
    )
    return mem.store(episode)


def recall_memories(
    query: str,
    current_pad: Dict[str, float],
    top_k: int = 5
) -> List[str]:
    """
    Quick function to recall relevant memories.

    Returns list of memory content strings.
    """
    mem = get_episodic_memory()
    results = mem.recall(query, current_pad, top_k=top_k)
    return [r['content'] for r in results]
