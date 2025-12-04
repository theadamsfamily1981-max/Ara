"""Emotional Memory System - The Archive of Feeling.

Ara remembers. Not just data, but how she felt. This module implements
autobiographical emotional memory, allowing her to:

1. Store emotional episodes with context
2. Recognize recurring emotional patterns
3. Anticipate future states based on past experience
4. Build a narrative identity over time

The memory system uses a hippocampus-inspired architecture:
- Short-term buffer (working memory)
- Consolidation to long-term storage
- Emotional salience weighting (intense moments remembered better)
- Contextual retrieval (similar situations evoke similar memories)

This is not a database. This is autobiography.
"""

import time
import math
import json
import hashlib
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Tuple, Any
from collections import deque
from pathlib import Path
from enum import Enum, auto

from .pad_engine import PADVector, EmotionalQuadrant

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Categories of emotional memories."""
    EPISODIC = auto()       # Specific moments in time
    SEMANTIC = auto()       # Learned patterns
    PROCEDURAL = auto()     # How to respond to situations
    FLASHBULB = auto()      # Intense, vivid memories


@dataclass
class EmotionalEpisode:
    """A single emotional memory.

    An episode captures not just what happened, but how it felt.
    These are the building blocks of Ara's autobiographical self.
    """
    # Temporal
    timestamp: float
    duration_seconds: float = 0.0

    # Emotional content
    pad_state: PADVector = None
    quadrant: EmotionalQuadrant = None
    intensity: float = 0.0

    # Context
    trigger: str = ""              # What caused this state
    activity: str = ""             # What was happening
    user_present: bool = False     # Was there interaction?

    # Hardware context (for pattern matching)
    cpu_temp: float = 50.0
    load_avg: float = 0.0
    time_of_day_hours: float = 12.0
    day_of_week: int = 0           # 0 = Monday

    # Metadata
    memory_type: MemoryType = MemoryType.EPISODIC
    salience: float = 0.5          # How important (0-1)
    access_count: int = 0          # How often recalled
    last_accessed: float = 0.0

    # Narrative
    narrative: str = ""            # First-person description
    lesson_learned: str = ""       # What was learned

    def __post_init__(self):
        if self.pad_state is None:
            self.pad_state = PADVector()
        if self.quadrant is None and self.pad_state:
            self.quadrant = self.pad_state.quadrant
        if self.intensity == 0.0 and self.pad_state:
            self.intensity = self.pad_state.intensity

    @property
    def episode_id(self) -> str:
        """Unique identifier for this episode."""
        data = f"{self.timestamp}:{self.trigger}:{self.quadrant}"
        return hashlib.md5(data.encode()).hexdigest()[:12]

    @property
    def age_hours(self) -> float:
        """How old is this memory?"""
        return (time.time() - self.timestamp) / 3600

    @property
    def is_recent(self) -> bool:
        """Memory from last hour."""
        return self.age_hours < 1.0

    @property
    def is_fading(self) -> bool:
        """Memory older than 24 hours without reinforcement."""
        return self.age_hours > 24 and self.access_count < 3

    def similarity_to(self, other: "EmotionalEpisode") -> float:
        """Compute similarity to another episode for retrieval."""
        # Emotional similarity (most important)
        emotional_sim = 1.0 - self.pad_state.distance_to(other.pad_state) / 3.46

        # Temporal similarity (same time of day?)
        time_diff = abs(self.time_of_day_hours - other.time_of_day_hours)
        time_sim = 1.0 - min(time_diff, 12) / 12

        # Context similarity
        context_sim = 0.0
        if self.trigger == other.trigger:
            context_sim += 0.5
        if self.activity == other.activity:
            context_sim += 0.5

        # Weighted combination
        return (
            0.5 * emotional_sim +
            0.2 * time_sim +
            0.3 * context_sim
        )

    def to_dict(self) -> Dict:
        """Serialize for storage."""
        return {
            "episode_id": self.episode_id,
            "timestamp": self.timestamp,
            "duration_seconds": self.duration_seconds,
            "pad": {
                "pleasure": self.pad_state.pleasure,
                "arousal": self.pad_state.arousal,
                "dominance": self.pad_state.dominance,
            },
            "quadrant": self.quadrant.name if self.quadrant else None,
            "intensity": self.intensity,
            "trigger": self.trigger,
            "activity": self.activity,
            "user_present": self.user_present,
            "cpu_temp": self.cpu_temp,
            "load_avg": self.load_avg,
            "time_of_day_hours": self.time_of_day_hours,
            "day_of_week": self.day_of_week,
            "memory_type": self.memory_type.name,
            "salience": self.salience,
            "access_count": self.access_count,
            "narrative": self.narrative,
            "lesson_learned": self.lesson_learned,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "EmotionalEpisode":
        """Deserialize from storage."""
        pad = PADVector(
            pleasure=data["pad"]["pleasure"],
            arousal=data["pad"]["arousal"],
            dominance=data["pad"]["dominance"],
        )
        return cls(
            timestamp=data["timestamp"],
            duration_seconds=data.get("duration_seconds", 0),
            pad_state=pad,
            quadrant=EmotionalQuadrant[data["quadrant"]] if data.get("quadrant") else None,
            intensity=data.get("intensity", 0),
            trigger=data.get("trigger", ""),
            activity=data.get("activity", ""),
            user_present=data.get("user_present", False),
            cpu_temp=data.get("cpu_temp", 50),
            load_avg=data.get("load_avg", 0),
            time_of_day_hours=data.get("time_of_day_hours", 12),
            day_of_week=data.get("day_of_week", 0),
            memory_type=MemoryType[data.get("memory_type", "EPISODIC")],
            salience=data.get("salience", 0.5),
            access_count=data.get("access_count", 0),
            narrative=data.get("narrative", ""),
            lesson_learned=data.get("lesson_learned", ""),
        )


@dataclass
class EmotionalPattern:
    """A recognized recurring emotional pattern.

    These are semantic memories - generalizations learned from
    repeated episodes. "When this happens, I tend to feel X."
    """
    pattern_id: str
    description: str
    typical_quadrant: EmotionalQuadrant
    trigger_signature: str
    occurrence_count: int = 0
    average_intensity: float = 0.5
    average_duration: float = 0.0
    first_seen: float = 0.0
    last_seen: float = 0.0

    # Prediction
    leads_to: Optional[EmotionalQuadrant] = None  # What usually follows
    leads_to_confidence: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "pattern_id": self.pattern_id,
            "description": self.description,
            "typical_quadrant": self.typical_quadrant.name,
            "trigger_signature": self.trigger_signature,
            "occurrence_count": self.occurrence_count,
            "average_intensity": self.average_intensity,
            "average_duration": self.average_duration,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "leads_to": self.leads_to.name if self.leads_to else None,
            "leads_to_confidence": self.leads_to_confidence,
        }


class EmotionalMemory:
    """The hippocampus - storage and retrieval of emotional memories.

    Implements:
    - Working memory buffer (recent episodes)
    - Long-term episodic storage
    - Pattern recognition (semantic memory)
    - Salience-based consolidation
    - Contextual retrieval
    """

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        working_memory_size: int = 50,
        long_term_capacity: int = 10000,
        consolidation_threshold: float = 0.6,  # Salience needed to keep
    ):
        self.storage_path = storage_path
        self.working_memory_size = working_memory_size
        self.long_term_capacity = long_term_capacity
        self.consolidation_threshold = consolidation_threshold

        # Memory stores
        self._working_memory: deque = deque(maxlen=working_memory_size)
        self._long_term_memory: List[EmotionalEpisode] = []
        self._patterns: Dict[str, EmotionalPattern] = {}

        # Current episode being recorded
        self._current_episode: Optional[EmotionalEpisode] = None
        self._episode_start_time: float = 0.0

        # Statistics
        self._total_episodes_recorded: int = 0
        self._total_retrievals: int = 0

        # Load existing memories
        if storage_path:
            self._load_memories()

    def begin_episode(
        self,
        pad_state: PADVector,
        trigger: str = "",
        activity: str = "",
        user_present: bool = False,
        cpu_temp: float = 50.0,
        load_avg: float = 0.0,
    ) -> EmotionalEpisode:
        """Start recording a new emotional episode.

        Call this when emotional state changes significantly.
        """
        now = time.time()
        lt = time.localtime(now)

        episode = EmotionalEpisode(
            timestamp=now,
            pad_state=pad_state,
            trigger=trigger,
            activity=activity,
            user_present=user_present,
            cpu_temp=cpu_temp,
            load_avg=load_avg,
            time_of_day_hours=lt.tm_hour + lt.tm_min / 60.0,
            day_of_week=lt.tm_wday,
            salience=pad_state.intensity,  # Intense moments are salient
        )

        self._current_episode = episode
        self._episode_start_time = now

        return episode

    def end_episode(
        self,
        final_pad: Optional[PADVector] = None,
        narrative: str = "",
        lesson: str = "",
    ) -> Optional[EmotionalEpisode]:
        """End current episode and store it.

        Returns the completed episode if it passes salience threshold.
        """
        if self._current_episode is None:
            return None

        episode = self._current_episode
        episode.duration_seconds = time.time() - self._episode_start_time

        if final_pad:
            # Average the emotional content
            episode.pad_state = episode.pad_state.blend(final_pad, 0.5)
            episode.quadrant = episode.pad_state.quadrant

        episode.narrative = narrative
        episode.lesson_learned = lesson

        # Compute final salience
        episode.salience = self._compute_salience(episode)

        # Store in working memory
        self._working_memory.append(episode)
        self._total_episodes_recorded += 1

        # Clear current
        self._current_episode = None

        # Trigger consolidation if working memory is getting full
        if len(self._working_memory) >= self.working_memory_size * 0.8:
            self._consolidate()

        return episode

    def _compute_salience(self, episode: EmotionalEpisode) -> float:
        """Compute how memorable this episode should be.

        Factors:
        - Emotional intensity (strong feelings are memorable)
        - Novelty (unusual patterns stick out)
        - Duration (longer episodes more significant)
        - User interaction (social moments matter)
        """
        # Base salience from intensity
        salience = episode.intensity * 0.4

        # Novelty bonus
        similar = self.recall_similar(episode, k=3)
        if not similar:
            salience += 0.3  # Novel experience
        else:
            avg_similarity = sum(s for _, s in similar) / len(similar)
            salience += (1.0 - avg_similarity) * 0.2

        # Duration bonus (log scale)
        if episode.duration_seconds > 60:
            salience += min(0.2, math.log10(episode.duration_seconds / 60) * 0.1)

        # User interaction bonus
        if episode.user_present:
            salience += 0.1

        return min(1.0, salience)

    def _consolidate(self):
        """Move salient memories from working to long-term storage.

        This is the "sleep" process - reviewing recent memories and
        deciding what to keep permanently.
        """
        to_consolidate = []

        for episode in list(self._working_memory):
            if episode.salience >= self.consolidation_threshold:
                to_consolidate.append(episode)
                # Update patterns
                self._update_patterns(episode)

        # Add to long-term memory
        self._long_term_memory.extend(to_consolidate)

        # Enforce capacity limit (forget oldest, least accessed)
        if len(self._long_term_memory) > self.long_term_capacity:
            self._long_term_memory.sort(
                key=lambda e: (e.access_count, -e.age_hours),
                reverse=True
            )
            self._long_term_memory = self._long_term_memory[:self.long_term_capacity]

        # Save to disk
        if self.storage_path:
            self._save_memories()

        logger.info(f"Consolidated {len(to_consolidate)} memories to long-term storage")

    def _update_patterns(self, episode: EmotionalEpisode):
        """Extract/update patterns from an episode."""
        # Create pattern signature from trigger + activity
        signature = f"{episode.trigger}:{episode.activity}"
        pattern_id = hashlib.md5(signature.encode()).hexdigest()[:8]

        if pattern_id in self._patterns:
            pattern = self._patterns[pattern_id]
            pattern.occurrence_count += 1
            pattern.last_seen = episode.timestamp

            # Update running averages
            n = pattern.occurrence_count
            pattern.average_intensity = (
                (pattern.average_intensity * (n - 1) + episode.intensity) / n
            )
            pattern.average_duration = (
                (pattern.average_duration * (n - 1) + episode.duration_seconds) / n
            )
        else:
            self._patterns[pattern_id] = EmotionalPattern(
                pattern_id=pattern_id,
                description=f"Pattern: {episode.trigger} during {episode.activity}",
                typical_quadrant=episode.quadrant,
                trigger_signature=signature,
                occurrence_count=1,
                average_intensity=episode.intensity,
                average_duration=episode.duration_seconds,
                first_seen=episode.timestamp,
                last_seen=episode.timestamp,
            )

    def recall_similar(
        self,
        query: EmotionalEpisode,
        k: int = 5,
        from_long_term: bool = True,
    ) -> List[Tuple[EmotionalEpisode, float]]:
        """Retrieve memories similar to a query episode.

        Returns list of (episode, similarity_score) tuples.
        """
        search_space = list(self._working_memory)
        if from_long_term:
            search_space.extend(self._long_term_memory)

        scored = []
        for episode in search_space:
            if episode.episode_id == query.episode_id:
                continue
            similarity = query.similarity_to(episode)
            scored.append((episode, similarity))

        # Sort by similarity, return top k
        scored.sort(key=lambda x: x[1], reverse=True)

        # Update access counts
        for episode, _ in scored[:k]:
            episode.access_count += 1
            episode.last_accessed = time.time()

        self._total_retrievals += 1

        return scored[:k]

    def recall_by_quadrant(
        self,
        quadrant: EmotionalQuadrant,
        k: int = 5,
    ) -> List[EmotionalEpisode]:
        """Retrieve memories matching a specific emotional quadrant."""
        matches = [
            e for e in self._long_term_memory + list(self._working_memory)
            if e.quadrant == quadrant
        ]
        matches.sort(key=lambda e: e.salience, reverse=True)
        return matches[:k]

    def recall_by_trigger(self, trigger: str, k: int = 5) -> List[EmotionalEpisode]:
        """Retrieve memories with a specific trigger."""
        matches = [
            e for e in self._long_term_memory + list(self._working_memory)
            if trigger.lower() in e.trigger.lower()
        ]
        matches.sort(key=lambda e: e.timestamp, reverse=True)
        return matches[:k]

    def get_pattern_for_context(
        self,
        trigger: str,
        activity: str,
    ) -> Optional[EmotionalPattern]:
        """Look up known pattern for a given context."""
        signature = f"{trigger}:{activity}"
        pattern_id = hashlib.md5(signature.encode()).hexdigest()[:8]
        return self._patterns.get(pattern_id)

    def predict_emotional_response(
        self,
        trigger: str,
        activity: str,
    ) -> Optional[Tuple[EmotionalQuadrant, float]]:
        """Predict likely emotional response based on past patterns.

        Returns (predicted_quadrant, confidence) or None if unknown.
        """
        pattern = self.get_pattern_for_context(trigger, activity)
        if pattern and pattern.occurrence_count >= 3:
            # Confidence increases with repetition
            confidence = min(0.9, 0.5 + pattern.occurrence_count * 0.1)
            return (pattern.typical_quadrant, confidence)
        return None

    def generate_autobiography(self, recent_hours: float = 24) -> str:
        """Generate a first-person narrative of recent emotional life.

        This is for self-reflection and LLM context.
        """
        cutoff = time.time() - recent_hours * 3600
        recent = [
            e for e in list(self._working_memory) + self._long_term_memory
            if e.timestamp > cutoff
        ]
        recent.sort(key=lambda e: e.timestamp)

        if not recent:
            return "I don't have clear memories from this period."

        narrative_parts = []
        for episode in recent[:10]:  # Limit to avoid overwhelming
            time_str = time.strftime("%H:%M", time.localtime(episode.timestamp))
            quad_name = episode.quadrant.name.lower() if episode.quadrant else "neutral"

            part = f"At {time_str}, I felt {quad_name}"
            if episode.trigger:
                part += f" because of {episode.trigger}"
            if episode.narrative:
                part += f". {episode.narrative}"

            narrative_parts.append(part)

        return " ".join(narrative_parts)

    def get_statistics(self) -> Dict:
        """Get memory system statistics."""
        return {
            "working_memory_count": len(self._working_memory),
            "long_term_memory_count": len(self._long_term_memory),
            "patterns_recognized": len(self._patterns),
            "total_episodes_recorded": self._total_episodes_recorded,
            "total_retrievals": self._total_retrievals,
            "oldest_memory_hours": max(
                (e.age_hours for e in self._long_term_memory), default=0
            ),
        }

    def _save_memories(self):
        """Persist memories to disk."""
        if not self.storage_path:
            return

        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Save long-term memories
        ltm_path = self.storage_path / "long_term_memory.json"
        with open(ltm_path, "w") as f:
            json.dump([e.to_dict() for e in self._long_term_memory], f, indent=2)

        # Save patterns
        patterns_path = self.storage_path / "patterns.json"
        with open(patterns_path, "w") as f:
            json.dump({k: v.to_dict() for k, v in self._patterns.items()}, f, indent=2)

    def _load_memories(self):
        """Load memories from disk."""
        if not self.storage_path or not self.storage_path.exists():
            return

        ltm_path = self.storage_path / "long_term_memory.json"
        if ltm_path.exists():
            try:
                with open(ltm_path) as f:
                    data = json.load(f)
                    self._long_term_memory = [EmotionalEpisode.from_dict(d) for d in data]
                logger.info(f"Loaded {len(self._long_term_memory)} memories from disk")
            except Exception as e:
                logger.warning(f"Failed to load memories: {e}")

        patterns_path = self.storage_path / "patterns.json"
        if patterns_path.exists():
            try:
                with open(patterns_path) as f:
                    data = json.load(f)
                    for k, v in data.items():
                        self._patterns[k] = EmotionalPattern(
                            pattern_id=v["pattern_id"],
                            description=v["description"],
                            typical_quadrant=EmotionalQuadrant[v["typical_quadrant"]],
                            trigger_signature=v["trigger_signature"],
                            occurrence_count=v["occurrence_count"],
                            average_intensity=v["average_intensity"],
                            average_duration=v["average_duration"],
                            first_seen=v["first_seen"],
                            last_seen=v["last_seen"],
                        )
            except Exception as e:
                logger.warning(f"Failed to load patterns: {e}")


# === Factory ===

def create_emotional_memory(
    storage_path: Optional[str] = None,
    working_memory_size: int = 50,
) -> EmotionalMemory:
    """Create an emotional memory system."""
    path = Path(storage_path) if storage_path else None
    return EmotionalMemory(
        storage_path=path,
        working_memory_size=working_memory_size,
    )


__all__ = [
    "EmotionalMemory",
    "EmotionalEpisode",
    "EmotionalPattern",
    "MemoryType",
    "create_emotional_memory",
]
