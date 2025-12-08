"""
HSF Episode Recording & Homeostasis Mining
===========================================

Episodes are trajectories through field space:
    (H(t0), zone0) → action → (H(t1), zone1) → ... → outcome

Mining episodes lets us learn:
- Which reflexes actually restore homeostasis
- Which reflexes make things worse
- Correlations between subsystems

This is the "dojo" for reflexes: we learn from experience which
moves work and which don't.

Key metric: ΔHomeostasis
    = time_to_return_to_GOOD with action
    - time_to_return_to_GOOD without action

Positive ΔHomeostasis → good reflex (faster recovery)
Negative ΔHomeostasis → bad reflex (slower recovery)
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import numpy as np

from .zones import Zone, ZoneState
from .reflex import ReflexAction, ReflexResult, ActionType


@dataclass
class EpisodeFrame:
    """A single frame in an episode trajectory."""
    timestamp: float
    tick: int
    zone_states: Dict[str, ZoneState]  # Per-subsystem zones
    global_zone: Zone
    field_similarity: float  # Similarity to baseline (0-1)
    actions_taken: List[ReflexResult]

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "tick": self.tick,
            "zones": {k: v.zone.name for k, v in self.zone_states.items()},
            "global_zone": self.global_zone.name,
            "similarity": self.field_similarity,
            "actions": [r.action.action_type.name for r in self.actions_taken],
        }


@dataclass
class Episode:
    """
    A complete episode: anomaly → (optional actions) → resolution.

    Episodes are the raw data for learning which reflexes work.
    """
    episode_id: str
    subsystem: str           # Primary subsystem (or "global" for multi-system)
    start_tick: int
    start_zone: Zone
    frames: List[EpisodeFrame] = field(default_factory=list)

    # Outcome (set when episode completes)
    end_tick: Optional[int] = None
    end_zone: Optional[Zone] = None
    resolved: bool = False   # Did we return to GOOD?
    ticks_to_resolve: Optional[int] = None

    # Actions during episode
    actions_taken: List[Tuple[int, ReflexAction]] = field(default_factory=list)  # (tick, action)

    def add_frame(self, frame: EpisodeFrame):
        """Add a frame to the episode."""
        self.frames.append(frame)

        # Record any actions
        for result in frame.actions_taken:
            if result.executed:
                self.actions_taken.append((frame.tick, result.action))

    def complete(self, end_tick: int, end_zone: Zone, resolved: bool):
        """Mark episode as complete."""
        self.end_tick = end_tick
        self.end_zone = end_zone
        self.resolved = resolved
        if resolved:
            self.ticks_to_resolve = end_tick - self.start_tick

    @property
    def duration(self) -> int:
        """Episode duration in ticks."""
        if self.end_tick is not None:
            return self.end_tick - self.start_tick
        if self.frames:
            return self.frames[-1].tick - self.start_tick
        return 0

    @property
    def worst_zone(self) -> Zone:
        """Worst zone reached during episode."""
        if not self.frames:
            return self.start_zone
        return max(f.global_zone for f in self.frames)


@dataclass
class EpisodeRecorder:
    """
    Records episodes as they happen.

    An episode starts when any subsystem enters WARM or worse.
    It ends when all subsystems return to GOOD (resolved)
    or when a timeout is reached (unresolved).
    """
    max_episode_ticks: int = 200  # Max ticks before forcing episode end
    min_good_ticks: int = 5       # Must be GOOD for this long to count as resolved

    _current_episode: Optional[Episode] = None
    _completed_episodes: List[Episode] = field(default_factory=list)
    _episode_counter: int = 0
    _tick: int = 0
    _good_streak: int = 0

    def tick(self, zone_states: Dict[str, ZoneState], global_zone: Zone,
             field_similarity: float, actions: List[ReflexResult]) -> Optional[Episode]:
        """
        Process a tick and update episode state.

        Returns completed episode if one just finished, else None.
        """
        self._tick += 1
        completed = None

        # Create frame
        frame = EpisodeFrame(
            timestamp=time.time(),
            tick=self._tick,
            zone_states=zone_states.copy(),
            global_zone=global_zone,
            field_similarity=field_similarity,
            actions_taken=actions,
        )

        # Check if we need to start an episode
        if self._current_episode is None:
            if global_zone > Zone.GOOD:
                # Anomaly detected, start episode
                self._episode_counter += 1
                worst_subsystem = max(zone_states.items(),
                                      key=lambda x: x[1].zone)[0]
                self._current_episode = Episode(
                    episode_id=f"ep_{self._episode_counter:06d}",
                    subsystem=worst_subsystem,
                    start_tick=self._tick,
                    start_zone=global_zone,
                )
                self._good_streak = 0

        # If we have an active episode, record the frame
        if self._current_episode is not None:
            self._current_episode.add_frame(frame)

            # Check for resolution
            if global_zone == Zone.GOOD:
                self._good_streak += 1
                if self._good_streak >= self.min_good_ticks:
                    # Resolved!
                    self._current_episode.complete(
                        end_tick=self._tick,
                        end_zone=Zone.GOOD,
                        resolved=True,
                    )
                    completed = self._current_episode
                    self._completed_episodes.append(completed)
                    self._current_episode = None
            else:
                self._good_streak = 0

                # Check for timeout
                if self._current_episode.duration >= self.max_episode_ticks:
                    # Unresolved timeout
                    self._current_episode.complete(
                        end_tick=self._tick,
                        end_zone=global_zone,
                        resolved=False,
                    )
                    completed = self._current_episode
                    self._completed_episodes.append(completed)
                    self._current_episode = None

        return completed

    def get_completed_episodes(self, n: Optional[int] = None) -> List[Episode]:
        """Get completed episodes, optionally limited to last N."""
        if n is None:
            return self._completed_episodes.copy()
        return self._completed_episodes[-n:]

    @property
    def current_episode(self) -> Optional[Episode]:
        return self._current_episode


@dataclass
class ReflexScore:
    """Score for a reflex based on episode analysis."""
    action_type: ActionType
    target: str
    times_used: int = 0
    times_helped: int = 0      # Episode resolved faster than baseline
    times_hurt: int = 0        # Episode resolved slower than baseline
    times_neutral: int = 0     # No significant difference
    avg_delta_homeostasis: float = 0.0  # Average ΔHomeostasis

    @property
    def effectiveness(self) -> float:
        """Effectiveness score: -1 (harmful) to +1 (helpful)."""
        if self.times_used == 0:
            return 0.0
        return (self.times_helped - self.times_hurt) / self.times_used

    @property
    def confidence(self) -> float:
        """Confidence in the score (more uses = higher confidence)."""
        return min(1.0, self.times_used / 10)


@dataclass
class HomeostasisMiner:
    """
    Mines episodes to learn which reflexes restore homeostasis.

    The key insight: we can estimate ΔHomeostasis by comparing:
    - Episodes where a reflex fired
    - Baseline time-to-recovery for similar starting conditions

    This is not a perfect causal analysis, but it's cheap and
    gives useful signal for reflex tuning.
    """
    baseline_recovery_ticks: Dict[Zone, float] = field(default_factory=dict)
    reflex_scores: Dict[Tuple[ActionType, str], ReflexScore] = field(default_factory=dict)

    # Similarity threshold for "similar starting conditions"
    similarity_threshold: float = 0.15

    def analyze_episode(self, episode: Episode):
        """
        Analyze a single episode to update reflex scores.
        """
        if not episode.resolved:
            # Can't learn much from unresolved episodes (yet)
            return

        start_zone = episode.start_zone
        recovery_ticks = episode.ticks_to_resolve or episode.duration

        # Update baseline for this starting zone
        old_baseline = self.baseline_recovery_ticks.get(start_zone, recovery_ticks)
        # Exponential moving average
        self.baseline_recovery_ticks[start_zone] = 0.9 * old_baseline + 0.1 * recovery_ticks

        # Analyze each action taken
        baseline = self.baseline_recovery_ticks[start_zone]

        for tick, action in episode.actions_taken:
            key = (action.action_type, action.target)

            if key not in self.reflex_scores:
                self.reflex_scores[key] = ReflexScore(
                    action_type=action.action_type,
                    target=action.target,
                )

            score = self.reflex_scores[key]
            score.times_used += 1

            # Compute ΔHomeostasis
            delta = baseline - recovery_ticks  # Positive = faster recovery

            # Classify outcome
            if delta > 2:  # More than 2 ticks faster
                score.times_helped += 1
            elif delta < -2:  # More than 2 ticks slower
                score.times_hurt += 1
            else:
                score.times_neutral += 1

            # Update average
            n = score.times_used
            score.avg_delta_homeostasis = (
                (n - 1) * score.avg_delta_homeostasis + delta
            ) / n

    def analyze_batch(self, episodes: List[Episode]):
        """Analyze a batch of episodes."""
        for episode in episodes:
            self.analyze_episode(episode)

    def get_reflex_ranking(self) -> List[Tuple[ReflexScore, float]]:
        """
        Get reflexes ranked by effectiveness.

        Returns: List of (score, weighted_effectiveness) sorted by effectiveness.
        """
        rankings = []
        for score in self.reflex_scores.values():
            # Weight by confidence
            weighted = score.effectiveness * score.confidence
            rankings.append((score, weighted))

        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def get_recommendations(self) -> Dict[str, List[str]]:
        """
        Get recommendations for reflex tuning.

        Returns dict with:
        - "promote": Reflexes that should be kept/enhanced
        - "demote": Reflexes that should be disabled/reviewed
        - "needs_data": Reflexes that need more episodes
        """
        promote = []
        demote = []
        needs_data = []

        for score in self.reflex_scores.values():
            desc = f"{score.action_type.name} → {score.target}"

            if score.confidence < 0.3:
                needs_data.append(f"{desc} (only {score.times_used} uses)")
            elif score.effectiveness > 0.3:
                promote.append(f"{desc} (effectiveness: {score.effectiveness:.2f})")
            elif score.effectiveness < -0.3:
                demote.append(f"{desc} (effectiveness: {score.effectiveness:.2f})")

        return {
            "promote": promote,
            "demote": demote,
            "needs_data": needs_data,
        }

    def suggest_new_reflexes(self, episodes: List[Episode]) -> List[str]:
        """
        Analyze unaddressed anomalies to suggest new reflexes.

        Looks for patterns where:
        - No reflexes fired
        - Recovery was slow
        """
        suggestions = []

        for episode in episodes:
            if not episode.resolved:
                continue

            # Episodes with no actions that took long to resolve
            if not episode.actions_taken:
                baseline = self.baseline_recovery_ticks.get(
                    episode.start_zone, 20
                )
                if episode.ticks_to_resolve and episode.ticks_to_resolve > baseline * 1.5:
                    suggestions.append(
                        f"Consider reflex for {episode.subsystem} in "
                        f"{episode.start_zone.name} zone (took {episode.ticks_to_resolve} "
                        f"ticks to resolve, baseline is {baseline:.0f})"
                    )

        return list(set(suggestions))[:5]  # Dedupe and limit
