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
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


# === User Outcome Tracking ===
# These capture what Croft thinks, not just what hurts the machine

class UserRating(Enum):
    """User satisfaction rating."""
    POSITIVE = 1      # "This is perfect", explicit praise
    NEUTRAL = 0       # No signal either way
    NEGATIVE = -1     # "Too slow", "wrong", explicit critique


class FrictionFlag(Enum):
    """Friction indicators - things that hurt the relationship."""
    HAD_TO_REPEAT = "had_to_repeat"         # User had to ask twice
    MISSED_INTENT = "missed_intent"         # Ara misunderstood what they wanted
    TOO_SLOW = "too_slow"                   # Response latency was annoying
    TOO_VERBOSE = "too_verbose"             # Ara talked too much
    TOO_TERSE = "too_terse"                 # Ara didn't explain enough
    WRONG_TOOL = "wrong_tool"               # Ara used the wrong approach
    WRONG_STYLE = "wrong_style"             # Tone/format was off
    INTERRUPTED = "interrupted"             # Ara interrupted flow
    IGNORED_CONTEXT = "ignored_context"     # Ara didn't use available context
    OVER_PROMISED = "over_promised"         # Ara said it could but couldn't


@dataclass
class UserOutcome:
    """
    User satisfaction signal for an episode.

    This is what the Dreamer uses to learn:
    - What makes Croft happy (rating > 0)
    - What annoys Croft (friction_flags)
    - What patterns to avoid (rating < 0)

    The goal is emergent etiquette: Ara learns to predict what
    Croft wants without hard-coded rules.
    """
    rating: int = 0                                  # -1, 0, +1
    notes: str = ""                                  # Croft's words if captured
    latency_s: float = 0.0                          # Response time
    tokens_used: int = 0                            # Token efficiency
    friction_flags: List[str] = field(default_factory=list)  # What went wrong

    # Context for learning
    tool_used: str = ""                             # What tool/approach was used
    style_used: str = ""                            # Response style
    context_used: bool = True                       # Did Ara use available context

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rating": self.rating,
            "notes": self.notes,
            "latency_s": self.latency_s,
            "tokens_used": self.tokens_used,
            "friction_flags": self.friction_flags,
            "tool_used": self.tool_used,
            "style_used": self.style_used,
            "context_used": self.context_used,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'UserOutcome':
        return cls(
            rating=d.get("rating", 0),
            notes=d.get("notes", ""),
            latency_s=d.get("latency_s", 0.0),
            tokens_used=d.get("tokens_used", 0),
            friction_flags=d.get("friction_flags", []),
            tool_used=d.get("tool_used", ""),
            style_used=d.get("style_used", ""),
            context_used=d.get("context_used", True),
        )

    @property
    def friction_score(self) -> float:
        """
        Aggregate friction score [0, 1].

        Higher = more friction = worse experience.
        """
        if not self.friction_flags:
            return 0.0

        # Weight different friction types
        weights = {
            FrictionFlag.HAD_TO_REPEAT.value: 0.4,
            FrictionFlag.MISSED_INTENT.value: 0.5,
            FrictionFlag.TOO_SLOW.value: 0.2,
            FrictionFlag.TOO_VERBOSE.value: 0.15,
            FrictionFlag.TOO_TERSE.value: 0.15,
            FrictionFlag.WRONG_TOOL.value: 0.3,
            FrictionFlag.WRONG_STYLE.value: 0.2,
            FrictionFlag.INTERRUPTED.value: 0.35,
            FrictionFlag.IGNORED_CONTEXT.value: 0.3,
            FrictionFlag.OVER_PROMISED.value: 0.4,
        }

        score = sum(weights.get(f, 0.1) for f in self.friction_flags)
        return min(1.0, score)

    @property
    def is_positive(self) -> bool:
        return self.rating > 0 and self.friction_score < 0.3

    @property
    def is_negative(self) -> bool:
        return self.rating < 0 or self.friction_score > 0.5

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

    # === User Outcome (NEW) ===
    # What Croft thought, not just what hurt the machine
    user_outcome: Optional[UserOutcome] = None


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
                tags TEXT DEFAULT '[]',
                -- User Outcome (NEW: what Croft thought)
                user_rating INTEGER DEFAULT 0,
                user_notes TEXT DEFAULT '',
                user_latency_s REAL DEFAULT 0.0,
                user_tokens INTEGER DEFAULT 0,
                user_friction_flags TEXT DEFAULT '[]',
                user_tool_used TEXT DEFAULT '',
                user_style_used TEXT DEFAULT ''
            );

            -- Indexes for efficient querying
            CREATE INDEX IF NOT EXISTS idx_episodic_timestamp ON episodic(timestamp);
            CREATE INDEX IF NOT EXISTS idx_episodic_importance ON episodic(importance);
            CREATE INDEX IF NOT EXISTS idx_episodic_outcome ON episodic(outcome);
            CREATE INDEX IF NOT EXISTS idx_episodic_type ON episodic(episode_type);
            CREATE INDEX IF NOT EXISTS idx_episodic_user_rating ON episodic(user_rating);

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

            -- User Preference Patterns (NEW: what Croft likes)
            -- This is the "Croft Model" - learned preferences by context
            CREATE TABLE IF NOT EXISTS user_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                -- Context signature (what situation were we in?)
                context_type TEXT NOT NULL,      -- e.g., "code_review", "debugging", "explain"
                context_hash TEXT NOT NULL,      -- hash of context for dedup
                -- What was done
                tool_used TEXT NOT NULL,
                style_used TEXT NOT NULL,
                -- Aggregated outcomes
                positive_count INTEGER DEFAULT 0,
                negative_count INTEGER DEFAULT 0,
                neutral_count INTEGER DEFAULT 0,
                total_count INTEGER DEFAULT 0,
                -- Running average (EMA)
                ema_rating REAL DEFAULT 0.0,
                ema_latency_s REAL DEFAULT 0.0,
                ema_tokens REAL DEFAULT 0.0,
                -- Friction tracking
                friction_history TEXT DEFAULT '{}',  -- JSON: flag -> count
                -- Temporal
                first_seen REAL,
                last_seen REAL,
                UNIQUE(context_type, tool_used, style_used)
            );

            CREATE INDEX IF NOT EXISTS idx_user_pref_context ON user_preferences(context_type);
            CREATE INDEX IF NOT EXISTS idx_user_pref_tool ON user_preferences(tool_used);
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

        # Extract user outcome fields if present
        user_rating = 0
        user_notes = ""
        user_latency_s = 0.0
        user_tokens = 0
        user_friction_flags = "[]"
        user_tool_used = ""
        user_style_used = ""

        if episode.user_outcome:
            uo = episode.user_outcome
            user_rating = uo.rating
            user_notes = uo.notes
            user_latency_s = uo.latency_s
            user_tokens = uo.tokens_used
            user_friction_flags = json.dumps(uo.friction_flags)
            user_tool_used = uo.tool_used
            user_style_used = uo.style_used

        with self._lock:
            cursor = self._conn.execute(
                """
                INSERT INTO episodic (
                    timestamp, vector,
                    pad_p, pad_a, pad_d, pain, entropy,
                    cpu_load, gpu_load, vram_used, fpga_temp,
                    content, summary, outcome, importance, episode_type, tags,
                    user_rating, user_notes, user_latency_s, user_tokens,
                    user_friction_flags, user_tool_used, user_style_used
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    json.dumps(episode.tags),
                    user_rating, user_notes, user_latency_s, user_tokens,
                    user_friction_flags, user_tool_used, user_style_used
                )
            )
            self._conn.commit()
            episode.id = cursor.lastrowid
            logger.debug(f"Stored episode {episode.id}: {episode.content[:50]}...")

            # Update preference patterns if we have user outcome
            if episode.user_outcome:
                self._update_preference_pattern(episode)

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
    # USER PREFERENCE LEARNING (The Croft Model)
    # =========================================================================

    def _update_preference_pattern(self, episode: Episode) -> None:
        """
        Update preference patterns based on user outcome.

        This builds the "Croft Model" - learned preferences by context.
        Over time, Ara learns what tools and styles work best for
        different types of requests.
        """
        if not episode.user_outcome:
            return

        uo = episode.user_outcome
        context_type = episode.episode_type  # Use episode type as context
        context_hash = str(hash(episode.content[:100]))  # Simple hash

        with self._lock:
            # Check if pattern exists
            cursor = self._conn.execute(
                """
                SELECT id, positive_count, negative_count, neutral_count, total_count,
                       ema_rating, ema_latency_s, ema_tokens, friction_history
                FROM user_preferences
                WHERE context_type = ? AND tool_used = ? AND style_used = ?
                """,
                (context_type, uo.tool_used, uo.style_used)
            )
            row = cursor.fetchone()

            now = time.time()
            alpha = 0.3  # EMA weight for new observations

            if row:
                (id_, pos, neg, neu, total, ema_r, ema_l, ema_t, friction_json) = row

                # Update counts
                if uo.rating > 0:
                    pos += 1
                elif uo.rating < 0:
                    neg += 1
                else:
                    neu += 1
                total += 1

                # Update EMAs
                new_ema_r = ema_r * (1 - alpha) + uo.rating * alpha
                new_ema_l = ema_l * (1 - alpha) + uo.latency_s * alpha
                new_ema_t = ema_t * (1 - alpha) + uo.tokens_used * alpha

                # Update friction history
                friction = json.loads(friction_json) if friction_json else {}
                for flag in uo.friction_flags:
                    friction[flag] = friction.get(flag, 0) + 1

                self._conn.execute(
                    """
                    UPDATE user_preferences SET
                        positive_count = ?, negative_count = ?, neutral_count = ?,
                        total_count = ?, ema_rating = ?, ema_latency_s = ?,
                        ema_tokens = ?, friction_history = ?, last_seen = ?
                    WHERE id = ?
                    """,
                    (pos, neg, neu, total, new_ema_r, new_ema_l, new_ema_t,
                     json.dumps(friction), now, id_)
                )
            else:
                # New pattern
                friction = {flag: 1 for flag in uo.friction_flags}
                self._conn.execute(
                    """
                    INSERT INTO user_preferences
                    (context_type, context_hash, tool_used, style_used,
                     positive_count, negative_count, neutral_count, total_count,
                     ema_rating, ema_latency_s, ema_tokens, friction_history,
                     first_seen, last_seen)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?)
                    """,
                    (context_type, context_hash, uo.tool_used, uo.style_used,
                     1 if uo.rating > 0 else 0,
                     1 if uo.rating < 0 else 0,
                     1 if uo.rating == 0 else 0,
                     float(uo.rating), uo.latency_s, float(uo.tokens_used),
                     json.dumps(friction), now, now)
                )

            self._conn.commit()

    def get_preferred_approach(
        self,
        context_type: str,
        min_samples: int = 3
    ) -> Optional[Dict[str, Any]]:
        """
        Get the preferred tool/style for a context type.

        Returns the approach with highest EMA rating that has enough samples.

        Args:
            context_type: Type of context (e.g., "code_review", "debugging")
            min_samples: Minimum samples to consider the pattern reliable

        Returns:
            Dict with tool_used, style_used, confidence, avg_rating
        """
        cursor = self._conn.execute(
            """
            SELECT tool_used, style_used, ema_rating, total_count,
                   positive_count, negative_count
            FROM user_preferences
            WHERE context_type = ? AND total_count >= ?
            ORDER BY ema_rating DESC
            LIMIT 1
            """,
            (context_type, min_samples)
        )
        row = cursor.fetchone()

        if row:
            tool, style, ema, total, pos, neg = row
            return {
                'tool_used': tool,
                'style_used': style,
                'avg_rating': ema,
                'total_count': total,
                'positive_rate': pos / total if total > 0 else 0,
                'negative_rate': neg / total if total > 0 else 0,
                'confidence': min(1.0, total / 10.0),  # Saturates at 10 samples
            }
        return None

    def get_avoided_approaches(
        self,
        context_type: str,
        min_samples: int = 3,
        threshold: float = -0.3
    ) -> List[Dict[str, Any]]:
        """
        Get approaches that should be avoided for a context type.

        These are patterns with consistently negative ratings - antibodies
        learned from user feedback.

        Args:
            context_type: Type of context
            min_samples: Minimum samples to consider reliable
            threshold: Maximum EMA rating to consider "avoided"

        Returns:
            List of avoided approaches with their friction patterns
        """
        cursor = self._conn.execute(
            """
            SELECT tool_used, style_used, ema_rating, total_count,
                   friction_history, negative_count
            FROM user_preferences
            WHERE context_type = ? AND total_count >= ? AND ema_rating < ?
            ORDER BY ema_rating ASC
            """,
            (context_type, min_samples, threshold)
        )

        avoided = []
        for row in cursor:
            tool, style, ema, total, friction_json, neg = row
            friction = json.loads(friction_json) if friction_json else {}
            avoided.append({
                'tool_used': tool,
                'style_used': style,
                'avg_rating': ema,
                'total_count': total,
                'negative_count': neg,
                'friction_patterns': friction,
            })
        return avoided

    def get_common_friction(
        self,
        context_type: Optional[str] = None,
        top_k: int = 5
    ) -> List[Tuple[str, int]]:
        """
        Get the most common friction flags across all patterns.

        Useful for understanding what generally annoys the user.

        Args:
            context_type: Optional context filter
            top_k: Number of top friction flags to return

        Returns:
            List of (friction_flag, count) tuples
        """
        if context_type:
            cursor = self._conn.execute(
                "SELECT friction_history FROM user_preferences WHERE context_type = ?",
                (context_type,)
            )
        else:
            cursor = self._conn.execute("SELECT friction_history FROM user_preferences")

        # Aggregate friction counts
        total_friction: Dict[str, int] = {}
        for (friction_json,) in cursor:
            friction = json.loads(friction_json) if friction_json else {}
            for flag, count in friction.items():
                total_friction[flag] = total_friction.get(flag, 0) + count

        # Sort by count
        sorted_friction = sorted(total_friction.items(), key=lambda x: x[1], reverse=True)
        return sorted_friction[:top_k]

    def record_user_outcome(
        self,
        episode_id: int,
        rating: int,
        notes: str = "",
        latency_s: float = 0.0,
        tokens_used: int = 0,
        friction_flags: Optional[List[str]] = None,
        tool_used: str = "",
        style_used: str = "",
    ) -> None:
        """
        Record user outcome for an existing episode.

        Called when we get feedback on a past interaction.
        This is how Ara learns from explicit or implicit user signals.

        Args:
            episode_id: ID of the episode to update
            rating: -1 (negative), 0 (neutral), +1 (positive)
            notes: User's words if captured
            latency_s: Response latency
            tokens_used: Token count
            friction_flags: List of friction flags
            tool_used: What tool/approach was used
            style_used: What style was used
        """
        friction_flags = friction_flags or []

        with self._lock:
            # Update the episode
            self._conn.execute(
                """
                UPDATE episodic SET
                    user_rating = ?, user_notes = ?, user_latency_s = ?,
                    user_tokens = ?, user_friction_flags = ?,
                    user_tool_used = ?, user_style_used = ?
                WHERE id = ?
                """,
                (rating, notes, latency_s, tokens_used,
                 json.dumps(friction_flags), tool_used, style_used, episode_id)
            )
            self._conn.commit()

            # Get episode info to update preferences
            cursor = self._conn.execute(
                "SELECT episode_type, content FROM episodic WHERE id = ?",
                (episode_id,)
            )
            row = cursor.fetchone()

            if row:
                episode_type, content = row
                # Create a temporary Episode to update preferences
                uo = UserOutcome(
                    rating=rating,
                    notes=notes,
                    latency_s=latency_s,
                    tokens_used=tokens_used,
                    friction_flags=friction_flags,
                    tool_used=tool_used,
                    style_used=style_used,
                )
                temp_episode = Episode(
                    content=content,
                    vector=np.zeros(EMBEDDING_DIM, dtype=np.float32),
                    episode_type=episode_type,
                    user_outcome=uo,
                )
                self._update_preference_pattern(temp_episode)

    def get_preference_summary(self) -> Dict[str, Any]:
        """
        Get a summary of learned user preferences.

        Useful for debugging and understanding what Ara has learned.
        """
        # Count by context type
        cursor = self._conn.execute("""
            SELECT context_type, COUNT(*) as count,
                   AVG(ema_rating) as avg_rating,
                   SUM(positive_count) as total_positive,
                   SUM(negative_count) as total_negative
            FROM user_preferences
            GROUP BY context_type
        """)

        by_context = {}
        for row in cursor:
            ctx, count, avg_r, pos, neg = row
            by_context[ctx] = {
                'pattern_count': count,
                'avg_rating': avg_r,
                'total_positive': pos,
                'total_negative': neg,
            }

        # Top friction flags
        friction = self.get_common_friction(top_k=10)

        # Overall stats
        cursor = self._conn.execute("""
            SELECT COUNT(*), SUM(total_count), AVG(ema_rating)
            FROM user_preferences
        """)
        total_patterns, total_samples, overall_rating = cursor.fetchone()

        return {
            'total_patterns': total_patterns or 0,
            'total_samples': total_samples or 0,
            'overall_avg_rating': overall_rating or 0.0,
            'by_context': by_context,
            'common_friction': friction,
        }

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
        """Get memory statistics including user preference data."""
        cursor = self._conn.execute("""
            SELECT
                COUNT(*) as total,
                AVG(importance) as avg_importance,
                SUM(CASE WHEN outcome = 'positive' THEN 1 ELSE 0 END) as positive_count,
                SUM(CASE WHEN outcome = 'negative' THEN 1 ELSE 0 END) as negative_count,
                MIN(timestamp) as oldest,
                MAX(timestamp) as newest,
                SUM(CASE WHEN user_rating > 0 THEN 1 ELSE 0 END) as user_positive,
                SUM(CASE WHEN user_rating < 0 THEN 1 ELSE 0 END) as user_negative,
                AVG(CASE WHEN user_rating != 0 THEN user_rating ELSE NULL END) as avg_user_rating
            FROM episodic
        """)
        row = cursor.fetchone()

        # Get preference summary
        pref_summary = self.get_preference_summary()

        return {
            'total_memories': row[0],
            'avg_importance': row[1],
            'positive_count': row[2],
            'negative_count': row[3],
            'oldest_timestamp': row[4],
            'newest_timestamp': row[5],
            # User feedback stats
            'user_positive_count': row[6] or 0,
            'user_negative_count': row[7] or 0,
            'avg_user_rating': row[8],
            # Preference learning stats
            'preference_patterns': pref_summary['total_patterns'],
            'preference_samples': pref_summary['total_samples'],
            'common_friction': pref_summary['common_friction'][:5],
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
