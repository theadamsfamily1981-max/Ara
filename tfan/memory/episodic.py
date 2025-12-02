"""
Episodic Memory: Temporal Continuity for Cognitive Systems

This module implements autobiographical memory that gives the system
a "history of itself" - enabling it to:
- Remember past episodes (experiments, decisions, outcomes)
- Learn from experience across sessions
- Answer "what happened when I tried X?" queries
- Build narrative understanding of its own behavior

Episodes are structured as:
  task + config snapshot + PAD/CLV trajectory + outcomes + key decisions
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import json
import hashlib
from pathlib import Path


class EpisodeType(str, Enum):
    """Types of episodes the system can remember."""
    CERTIFICATION = "certification"
    WORKLOAD = "workload"
    EXPERIMENT = "experiment"
    FAILURE = "failure"
    RECOVERY = "recovery"
    CONFIGURATION = "configuration"
    DECISION = "decision"
    LEARNING = "learning"


class OutcomeType(str, Enum):
    """How an episode concluded."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    ABORTED = "aborted"
    UNKNOWN = "unknown"


@dataclass
class PADTrajectory:
    """Track PAD (Pleasure-Arousal-Dominance) state over an episode."""
    timestamps: List[float] = field(default_factory=list)
    valence: List[float] = field(default_factory=list)  # Pleasure
    arousal: List[float] = field(default_factory=list)
    dominance: List[float] = field(default_factory=list)

    def add_state(self, t: float, v: float, a: float, d: float) -> None:
        """Record a PAD state at time t."""
        self.timestamps.append(t)
        self.valence.append(v)
        self.arousal.append(a)
        self.dominance.append(d)

    @property
    def mean_valence(self) -> float:
        return sum(self.valence) / len(self.valence) if self.valence else 0.0

    @property
    def mean_arousal(self) -> float:
        return sum(self.arousal) / len(self.arousal) if self.arousal else 0.0

    @property
    def peak_arousal(self) -> float:
        return max(self.arousal) if self.arousal else 0.0

    def was_stressful(self) -> bool:
        """Was this episode characterized by stress (high arousal, low valence)?"""
        return self.mean_arousal > 0.6 and self.mean_valence < -0.2


@dataclass
class CLVTrajectory:
    """Track CLV (Cognitive Load Vector) over an episode."""
    timestamps: List[float] = field(default_factory=list)
    instability: List[float] = field(default_factory=list)
    resource: List[float] = field(default_factory=list)
    structural: List[float] = field(default_factory=list)
    risk_levels: List[str] = field(default_factory=list)

    def add_state(
        self,
        t: float,
        instability: float,
        resource: float,
        structural: float,
        risk_level: str = "nominal"
    ) -> None:
        """Record CLV state at time t."""
        self.timestamps.append(t)
        self.instability.append(instability)
        self.resource.append(resource)
        self.structural.append(structural)
        self.risk_levels.append(risk_level)

    @property
    def peak_risk(self) -> str:
        """What was the highest risk level during this episode?"""
        risk_order = {"nominal": 0, "elevated": 1, "high": 2, "critical": 3}
        if not self.risk_levels:
            return "unknown"
        return max(self.risk_levels, key=lambda r: risk_order.get(r, 0))

    @property
    def mean_instability(self) -> float:
        return sum(self.instability) / len(self.instability) if self.instability else 0.0


@dataclass
class Decision:
    """A key decision made during an episode."""
    timestamp: float
    decision_type: str  # e.g., "backend_selection", "profile_switch", "parameter_update"
    choice: str  # What was decided
    alternatives: List[str] = field(default_factory=list)  # Other options considered
    rationale: str = ""  # Why this was chosen
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "type": self.decision_type,
            "choice": self.choice,
            "alternatives": self.alternatives,
            "rationale": self.rationale,
            "confidence": self.confidence
        }


@dataclass
class Episode:
    """
    A complete episode in the system's autobiographical memory.

    An episode captures everything about a meaningful unit of experience:
    - What task was being done
    - What configuration was used
    - How the system felt (PAD) and performed (CLV) over time
    - What decisions were made
    - What the outcome was
    """
    id: str
    episode_type: EpisodeType
    title: str
    started_at: datetime
    ended_at: Optional[datetime] = None

    # Task & Config
    task_description: str = ""
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    workload_type: str = ""

    # Trajectories
    pad_trajectory: PADTrajectory = field(default_factory=PADTrajectory)
    clv_trajectory: CLVTrajectory = field(default_factory=CLVTrajectory)

    # Decisions & Actions
    decisions: List[Decision] = field(default_factory=list)
    actions_taken: List[str] = field(default_factory=list)

    # Outcomes
    outcome: OutcomeType = OutcomeType.UNKNOWN
    outcome_metrics: Dict[str, float] = field(default_factory=dict)
    lessons_learned: List[str] = field(default_factory=list)

    # Metadata
    tags: List[str] = field(default_factory=list)
    summary: str = ""  # LLM-generated summary
    parent_episode_id: Optional[str] = None  # For hierarchical episodes

    def add_decision(
        self,
        decision_type: str,
        choice: str,
        alternatives: Optional[List[str]] = None,
        rationale: str = "",
        confidence: float = 1.0
    ) -> Decision:
        """Record a decision made during this episode."""
        elapsed = (datetime.now() - self.started_at).total_seconds()
        decision = Decision(
            timestamp=elapsed,
            decision_type=decision_type,
            choice=choice,
            alternatives=alternatives or [],
            rationale=rationale,
            confidence=confidence
        )
        self.decisions.append(decision)
        return decision

    def close(
        self,
        outcome: OutcomeType,
        metrics: Optional[Dict[str, float]] = None,
        lessons: Optional[List[str]] = None
    ) -> None:
        """Close the episode with its final outcome."""
        self.ended_at = datetime.now()
        self.outcome = outcome
        self.outcome_metrics = metrics or {}
        self.lessons_learned = lessons or []

    @property
    def duration(self) -> timedelta:
        """How long did this episode last?"""
        end = self.ended_at or datetime.now()
        return end - self.started_at

    @property
    def was_successful(self) -> bool:
        return self.outcome in [OutcomeType.SUCCESS, OutcomeType.PARTIAL_SUCCESS]

    @property
    def was_stressful(self) -> bool:
        return self.pad_trajectory.was_stressful()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize episode for storage."""
        return {
            "id": self.id,
            "episode_type": self.episode_type.value,
            "title": self.title,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "task_description": self.task_description,
            "config_snapshot": self.config_snapshot,
            "workload_type": self.workload_type,
            "decisions": [d.to_dict() for d in self.decisions],
            "actions_taken": self.actions_taken,
            "outcome": self.outcome.value,
            "outcome_metrics": self.outcome_metrics,
            "lessons_learned": self.lessons_learned,
            "tags": self.tags,
            "summary": self.summary,
            "parent_episode_id": self.parent_episode_id,
            "pad_summary": {
                "mean_valence": self.pad_trajectory.mean_valence,
                "mean_arousal": self.pad_trajectory.mean_arousal,
                "peak_arousal": self.pad_trajectory.peak_arousal,
                "was_stressful": self.was_stressful
            },
            "clv_summary": {
                "mean_instability": self.clv_trajectory.mean_instability,
                "peak_risk": self.clv_trajectory.peak_risk
            }
        }


class EpisodicMemory:
    """
    The system's autobiographical memory store.

    Provides:
    - Episode storage and retrieval
    - Temporal queries ("what happened last week?")
    - Similarity queries ("what's similar to this situation?")
    - Pattern queries ("when have I seen high-risk failures?")
    - Autobiographical narration
    """

    def __init__(self, persist_dir: Optional[str] = None):
        self._episodes: Dict[str, Episode] = {}
        self._timeline: List[str] = []  # Episode IDs in chronological order
        self._persist_dir = Path(persist_dir) if persist_dir else None
        self._current_episode: Optional[Episode] = None

        if self._persist_dir:
            self._persist_dir.mkdir(parents=True, exist_ok=True)

    # ========== Episode Lifecycle ==========

    def start_episode(
        self,
        episode_type: EpisodeType,
        title: str,
        task_description: str = "",
        config: Optional[Dict[str, Any]] = None,
        workload_type: str = "",
        tags: Optional[List[str]] = None
    ) -> Episode:
        """Start a new episode."""
        episode_id = self._generate_id(title)
        episode = Episode(
            id=episode_id,
            episode_type=episode_type,
            title=title,
            started_at=datetime.now(),
            task_description=task_description,
            config_snapshot=config or {},
            workload_type=workload_type,
            tags=tags or []
        )

        self._episodes[episode_id] = episode
        self._timeline.append(episode_id)
        self._current_episode = episode

        return episode

    def get_current_episode(self) -> Optional[Episode]:
        """Get the currently active episode."""
        return self._current_episode

    def close_current_episode(
        self,
        outcome: OutcomeType,
        metrics: Optional[Dict[str, float]] = None,
        lessons: Optional[List[str]] = None,
        summary: str = ""
    ) -> Optional[Episode]:
        """Close the current episode."""
        if self._current_episode:
            self._current_episode.close(outcome, metrics, lessons)
            self._current_episode.summary = summary
            self._save_episode(self._current_episode)
            closed = self._current_episode
            self._current_episode = None
            return closed
        return None

    # ========== Episode Retrieval ==========

    def get_episode(self, episode_id: str) -> Optional[Episode]:
        """Get episode by ID."""
        return self._episodes.get(episode_id)

    def get_recent(self, count: int = 10) -> List[Episode]:
        """Get most recent episodes."""
        recent_ids = self._timeline[-count:]
        return [self._episodes[eid] for eid in reversed(recent_ids) if eid in self._episodes]

    def get_by_type(self, episode_type: EpisodeType, limit: int = 50) -> List[Episode]:
        """Get episodes of a specific type."""
        matching = [
            self._episodes[eid]
            for eid in reversed(self._timeline)
            if eid in self._episodes and self._episodes[eid].episode_type == episode_type
        ]
        return matching[:limit]

    def get_by_tag(self, tag: str, limit: int = 50) -> List[Episode]:
        """Get episodes with a specific tag."""
        matching = [
            self._episodes[eid]
            for eid in reversed(self._timeline)
            if eid in self._episodes and tag in self._episodes[eid].tags
        ]
        return matching[:limit]

    def get_failures(self, limit: int = 20) -> List[Episode]:
        """Get recent failure episodes."""
        failures = [
            self._episodes[eid]
            for eid in reversed(self._timeline)
            if eid in self._episodes and self._episodes[eid].outcome == OutcomeType.FAILURE
        ]
        return failures[:limit]

    def get_high_risk(self, limit: int = 20) -> List[Episode]:
        """Get episodes that had high risk moments."""
        high_risk = [
            self._episodes[eid]
            for eid in reversed(self._timeline)
            if eid in self._episodes and self._episodes[eid].clv_trajectory.peak_risk in ["high", "critical"]
        ]
        return high_risk[:limit]

    def get_stressful(self, limit: int = 20) -> List[Episode]:
        """Get episodes that were stressful (high arousal, low valence)."""
        stressful = [
            self._episodes[eid]
            for eid in reversed(self._timeline)
            if eid in self._episodes and self._episodes[eid].was_stressful
        ]
        return stressful[:limit]

    # ========== Temporal Queries ==========

    def get_in_timeframe(
        self,
        start: datetime,
        end: Optional[datetime] = None
    ) -> List[Episode]:
        """Get episodes within a time range."""
        end = end or datetime.now()
        matching = [
            self._episodes[eid]
            for eid in self._timeline
            if eid in self._episodes
            and start <= self._episodes[eid].started_at <= end
        ]
        return matching

    def get_since(self, hours: int = 24) -> List[Episode]:
        """Get episodes from the last N hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return self.get_in_timeframe(cutoff)

    # ========== Similarity & Pattern Queries ==========

    def find_similar(
        self,
        episode: Episode,
        top_k: int = 5
    ) -> List[Tuple[Episode, float]]:
        """
        Find episodes similar to the given one.

        Similarity based on:
        - Same episode type
        - Similar workload type
        - Overlapping tags
        - Similar PAD/CLV patterns
        """
        scores = []

        for eid, candidate in self._episodes.items():
            if candidate.id == episode.id:
                continue

            score = 0.0

            # Same type
            if candidate.episode_type == episode.episode_type:
                score += 0.3

            # Same workload
            if candidate.workload_type == episode.workload_type:
                score += 0.2

            # Tag overlap
            tag_overlap = len(set(candidate.tags) & set(episode.tags))
            if tag_overlap > 0:
                score += 0.1 * tag_overlap

            # PAD similarity
            pad_diff = abs(candidate.pad_trajectory.mean_arousal - episode.pad_trajectory.mean_arousal)
            score += 0.2 * (1.0 - min(pad_diff, 1.0))

            # CLV similarity
            clv_diff = abs(candidate.clv_trajectory.mean_instability - episode.clv_trajectory.mean_instability)
            score += 0.2 * (1.0 - min(clv_diff, 1.0))

            scores.append((candidate, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def find_with_decision_type(
        self,
        decision_type: str,
        limit: int = 20
    ) -> List[Episode]:
        """Find episodes that included a specific type of decision."""
        matching = []
        for eid in reversed(self._timeline):
            if eid not in self._episodes:
                continue
            episode = self._episodes[eid]
            if any(d.decision_type == decision_type for d in episode.decisions):
                matching.append(episode)
                if len(matching) >= limit:
                    break
        return matching

    # ========== Autobiographical Queries ==========

    def what_happened_when(self, query: str) -> List[Episode]:
        """
        Answer "what happened when..." queries.

        Searches episode titles, descriptions, and lessons.
        """
        query_lower = query.lower()
        matching = []

        for episode in self._episodes.values():
            searchable = f"{episode.title} {episode.task_description} {' '.join(episode.lessons_learned)}".lower()
            if query_lower in searchable:
                matching.append(episode)

        return sorted(matching, key=lambda e: e.started_at, reverse=True)

    def compare_configs(
        self,
        config_key: str,
        value1: Any,
        value2: Any
    ) -> Dict[str, List[Episode]]:
        """
        Compare outcomes when a config key had different values.

        Returns episodes grouped by the config value.
        """
        episodes_v1 = []
        episodes_v2 = []

        for episode in self._episodes.values():
            if config_key in episode.config_snapshot:
                if episode.config_snapshot[config_key] == value1:
                    episodes_v1.append(episode)
                elif episode.config_snapshot[config_key] == value2:
                    episodes_v2.append(episode)

        return {
            str(value1): episodes_v1,
            str(value2): episodes_v2
        }

    def summarize_period(
        self,
        start: datetime,
        end: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Generate summary statistics for a time period."""
        episodes = self.get_in_timeframe(start, end)

        if not episodes:
            return {"episode_count": 0}

        outcomes = {}
        for ep in episodes:
            outcomes[ep.outcome.value] = outcomes.get(ep.outcome.value, 0) + 1

        return {
            "episode_count": len(episodes),
            "timeframe": {
                "start": start.isoformat(),
                "end": (end or datetime.now()).isoformat()
            },
            "outcomes": outcomes,
            "success_rate": (outcomes.get("success", 0) + outcomes.get("partial_success", 0)) / len(episodes),
            "failure_count": outcomes.get("failure", 0),
            "stressful_episodes": len([e for e in episodes if e.was_stressful]),
            "high_risk_episodes": len([e for e in episodes if e.clv_trajectory.peak_risk in ["high", "critical"]]),
            "types": {
                t.value: len([e for e in episodes if e.episode_type == t])
                for t in EpisodeType
            }
        }

    # ========== Narrative Generation ==========

    def generate_narrative(
        self,
        episode: Episode,
        include_decisions: bool = True
    ) -> str:
        """Generate a natural language narrative of an episode."""
        lines = []

        # Opening
        lines.append(f"On {episode.started_at.strftime('%Y-%m-%d at %H:%M')}, I began {episode.title}.")

        # Task
        if episode.task_description:
            lines.append(f"The task was: {episode.task_description}")

        # Stress/Risk narrative
        if episode.was_stressful:
            lines.append(f"This was a stressful episode with high arousal (peak {episode.pad_trajectory.peak_arousal:.2f}).")

        if episode.clv_trajectory.peak_risk in ["high", "critical"]:
            lines.append(f"Risk reached {episode.clv_trajectory.peak_risk} levels during execution.")

        # Decisions
        if include_decisions and episode.decisions:
            lines.append("Key decisions made:")
            for d in episode.decisions:
                lines.append(f"  - Chose '{d.choice}' for {d.decision_type}" +
                            (f" because: {d.rationale}" if d.rationale else ""))

        # Outcome
        outcome_text = {
            OutcomeType.SUCCESS: "The episode concluded successfully.",
            OutcomeType.PARTIAL_SUCCESS: "The episode achieved partial success.",
            OutcomeType.FAILURE: "Unfortunately, the episode ended in failure.",
            OutcomeType.ABORTED: "The episode was aborted before completion."
        }.get(episode.outcome, "The outcome was undetermined.")
        lines.append(outcome_text)

        # Metrics
        if episode.outcome_metrics:
            lines.append("Key metrics:")
            for k, v in episode.outcome_metrics.items():
                lines.append(f"  - {k}: {v:.3f}" if isinstance(v, float) else f"  - {k}: {v}")

        # Lessons
        if episode.lessons_learned:
            lines.append("Lessons learned:")
            for lesson in episode.lessons_learned:
                lines.append(f"  - {lesson}")

        return "\n".join(lines)

    # ========== Persistence ==========

    def _save_episode(self, episode: Episode) -> None:
        """Save an episode to disk."""
        if not self._persist_dir:
            return

        filepath = self._persist_dir / f"{episode.id}.json"
        with open(filepath, 'w') as f:
            json.dump(episode.to_dict(), f, indent=2)

    def save_all(self) -> None:
        """Save all episodes and index."""
        if not self._persist_dir:
            return

        for episode in self._episodes.values():
            self._save_episode(episode)

        # Save timeline index
        index_path = self._persist_dir / "timeline.json"
        with open(index_path, 'w') as f:
            json.dump(self._timeline, f)

    def load(self) -> int:
        """Load episodes from disk. Returns count loaded."""
        if not self._persist_dir or not self._persist_dir.exists():
            return 0

        # Load timeline
        index_path = self._persist_dir / "timeline.json"
        if index_path.exists():
            with open(index_path) as f:
                self._timeline = json.load(f)

        # Load episodes
        count = 0
        for filepath in self._persist_dir.glob("*.json"):
            if filepath.name == "timeline.json":
                continue
            try:
                with open(filepath) as f:
                    data = json.load(f)
                    episode = self._dict_to_episode(data)
                    self._episodes[episode.id] = episode
                    count += 1
            except Exception:
                continue

        return count

    def _dict_to_episode(self, data: Dict[str, Any]) -> Episode:
        """Reconstruct an Episode from dict."""
        episode = Episode(
            id=data["id"],
            episode_type=EpisodeType(data["episode_type"]),
            title=data["title"],
            started_at=datetime.fromisoformat(data["started_at"]),
            ended_at=datetime.fromisoformat(data["ended_at"]) if data.get("ended_at") else None,
            task_description=data.get("task_description", ""),
            config_snapshot=data.get("config_snapshot", {}),
            workload_type=data.get("workload_type", ""),
            actions_taken=data.get("actions_taken", []),
            outcome=OutcomeType(data.get("outcome", "unknown")),
            outcome_metrics=data.get("outcome_metrics", {}),
            lessons_learned=data.get("lessons_learned", []),
            tags=data.get("tags", []),
            summary=data.get("summary", ""),
            parent_episode_id=data.get("parent_episode_id")
        )

        # Reconstruct decisions
        for d_data in data.get("decisions", []):
            episode.decisions.append(Decision(
                timestamp=d_data["timestamp"],
                decision_type=d_data["type"],
                choice=d_data["choice"],
                alternatives=d_data.get("alternatives", []),
                rationale=d_data.get("rationale", ""),
                confidence=d_data.get("confidence", 1.0)
            ))

        return episode

    def _generate_id(self, title: str) -> str:
        """Generate unique episode ID."""
        hash_input = f"{title}:{datetime.now().isoformat()}"
        return f"ep_{hashlib.md5(hash_input.encode()).hexdigest()[:12]}"

    # ========== Statistics ==========

    @property
    def stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        if not self._episodes:
            return {"episode_count": 0}

        return {
            "episode_count": len(self._episodes),
            "oldest": min(e.started_at for e in self._episodes.values()).isoformat(),
            "newest": max(e.started_at for e in self._episodes.values()).isoformat(),
            "success_rate": len([e for e in self._episodes.values() if e.was_successful]) / len(self._episodes),
            "types": {
                t.value: len([e for e in self._episodes.values() if e.episode_type == t])
                for t in EpisodeType
            }
        }


# ============================================================
# Convenience functions
# ============================================================

def create_episodic_memory(persist_dir: str = "episodes/") -> EpisodicMemory:
    """Create an episodic memory store with persistence."""
    return EpisodicMemory(persist_dir=persist_dir)


def record_certification_episode(
    memory: EpisodicMemory,
    cert_name: str,
    config: Dict[str, Any],
    results: Dict[str, float],
    success: bool
) -> Episode:
    """Helper to record a certification run as an episode."""
    episode = memory.start_episode(
        episode_type=EpisodeType.CERTIFICATION,
        title=f"Certification: {cert_name}",
        config=config,
        tags=["certification", cert_name]
    )

    # Record outcome
    episode.close(
        outcome=OutcomeType.SUCCESS if success else OutcomeType.FAILURE,
        metrics=results,
        lessons=[
            f"AF score: {results.get('af_score', 'N/A')}",
            f"All tests passed: {success}"
        ]
    )

    return episode
