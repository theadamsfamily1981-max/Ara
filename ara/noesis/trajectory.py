"""
Thought Trajectories - 4D Paths Through Cognitive Space
=========================================================

A trajectory is a path through 4D thought space:
    (hv, layer, time) sequences

Instead of "we ran this code", we talk about:
    "We moved from (heuristic, hardware perspective) →
     (symbolic, narrative perspective) →
     (meta, optimization perspective) in this order."

Trajectories can be:
- Recorded from actual thinking sessions
- Mined for common patterns
- Compared and clustered
- Re-run as cognitive workflows

This is where "4D on 3D" becomes tractable science on thought.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import time

from .layers import LayeredSpace, LayerProjection, CognitiveLayer, LayerRole


@dataclass
class TrajectoryPoint:
    """
    A single point in a 4D thought trajectory.

    (hv, layer_id, time) = 4D coordinate
    """
    hv: np.ndarray
    layer_id: int
    timestamp: float
    label: str = ""

    # Optional metadata
    move_type: str = ""           # What kind of thinking move
    produced_insight: bool = False
    confidence: float = 0.5

    def to_tuple(self) -> Tuple[int, float]:
        """Return (layer_id, timestamp) for trajectory analysis."""
        return (self.layer_id, self.timestamp)


@dataclass
class ThoughtTrajectory:
    """
    A complete trajectory through 4D thought space.

    Represents a thinking episode as a sequence of layer transitions.
    """
    trajectory_id: str
    points: List[TrajectoryPoint] = field(default_factory=list)
    outcome: str = ""      # "breakthrough", "stalled", etc.
    goal: str = ""
    domain: str = ""

    # Computed features
    _layer_sequence: List[int] = field(default_factory=list)

    def add_point(self, point: TrajectoryPoint):
        """Add a point to the trajectory."""
        self.points.append(point)
        self._layer_sequence.append(point.layer_id)

    def add(self, hv: np.ndarray, layer_id: int, label: str = "",
            move_type: str = "", produced_insight: bool = False):
        """Convenience method to add a point."""
        point = TrajectoryPoint(
            hv=hv,
            layer_id=layer_id,
            timestamp=time.time(),
            label=label,
            move_type=move_type,
            produced_insight=produced_insight,
        )
        self.add_point(point)

    @property
    def layer_sequence(self) -> List[int]:
        """Get the sequence of layer transitions."""
        return self._layer_sequence.copy()

    @property
    def unique_layers(self) -> set:
        """Get unique layers visited."""
        return set(self._layer_sequence)

    @property
    def layer_transitions(self) -> List[Tuple[int, int]]:
        """Get layer transition pairs (bigrams)."""
        if len(self._layer_sequence) < 2:
            return []
        return list(zip(self._layer_sequence[:-1], self._layer_sequence[1:]))

    @property
    def duration(self) -> float:
        """Total duration in seconds."""
        if len(self.points) < 2:
            return 0.0
        return self.points[-1].timestamp - self.points[0].timestamp

    def time_in_layer(self, layer_id: int) -> float:
        """Estimate time spent in a layer."""
        total = 0.0
        for i in range(len(self.points) - 1):
            if self.points[i].layer_id == layer_id:
                total += self.points[i+1].timestamp - self.points[i].timestamp
        return total

    def layer_distribution(self) -> Dict[int, float]:
        """Get fraction of time spent in each layer."""
        if not self.points or self.duration == 0:
            return {}

        times = {lid: self.time_in_layer(lid) for lid in self.unique_layers}
        total = sum(times.values())
        if total == 0:
            return {}
        return {lid: t / total for lid, t in times.items()}


@dataclass
class TrajectoryPattern:
    """
    A common pattern extracted from multiple trajectories.

    Represents a "cognitive workflow" that can be recognized and reused.
    """
    pattern_id: str
    name: str
    layer_sequence: List[int]
    description: str = ""

    # Statistics
    occurrences: int = 0
    success_rate: float = 0.0
    avg_duration: float = 0.0

    # HDC representation of the pattern
    pattern_hv: Optional[np.ndarray] = None

    def matches(self, trajectory: ThoughtTrajectory, min_overlap: float = 0.7) -> bool:
        """Check if a trajectory matches this pattern."""
        traj_seq = trajectory.layer_sequence
        pattern_seq = self.layer_sequence

        if len(traj_seq) < len(pattern_seq):
            return False

        # Sliding window match
        for i in range(len(traj_seq) - len(pattern_seq) + 1):
            window = traj_seq[i:i + len(pattern_seq)]
            matches = sum(a == b for a, b in zip(window, pattern_seq))
            if matches / len(pattern_seq) >= min_overlap:
                return True
        return False


@dataclass
class TrajectoryMiner:
    """
    Mines trajectories to discover common patterns.

    Finds:
    - Frequent layer transition sequences
    - Patterns associated with success
    - Patterns associated with failure
    """
    space: LayeredSpace
    _trajectories: List[ThoughtTrajectory] = field(default_factory=list)
    _patterns: List[TrajectoryPattern] = field(default_factory=list)

    def add_trajectory(self, traj: ThoughtTrajectory):
        """Add a trajectory for mining."""
        self._trajectories.append(traj)

    def mine_patterns(self, min_length: int = 2, max_length: int = 5,
                      min_support: int = 3) -> List[TrajectoryPattern]:
        """
        Mine frequent layer sequence patterns.

        Uses a simple n-gram approach.
        """
        # Count all n-grams of layer sequences
        ngram_counts: Dict[Tuple, Dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "successes": 0, "durations": []}
        )

        for traj in self._trajectories:
            seq = traj.layer_sequence
            success = traj.outcome in ["breakthrough", "progress"]

            for n in range(min_length, min(max_length + 1, len(seq) + 1)):
                for i in range(len(seq) - n + 1):
                    ngram = tuple(seq[i:i + n])
                    ngram_counts[ngram]["count"] += 1
                    if success:
                        ngram_counts[ngram]["successes"] += 1
                    ngram_counts[ngram]["durations"].append(traj.duration)

        # Filter by support and create patterns
        patterns = []
        for ngram, stats in ngram_counts.items():
            if stats["count"] >= min_support:
                avg_duration = np.mean(stats["durations"]) if stats["durations"] else 0
                success_rate = stats["successes"] / stats["count"]

                # Generate name from layer sequence
                layer_names = []
                for lid in ngram:
                    layer = self.space.get_layer(lid)
                    if layer:
                        layer_names.append(layer.name[:3])
                    else:
                        layer_names.append(f"L{lid}")

                pattern = TrajectoryPattern(
                    pattern_id=f"pat_{'_'.join(layer_names)}",
                    name=f"{' → '.join(layer_names)}",
                    layer_sequence=list(ngram),
                    occurrences=stats["count"],
                    success_rate=success_rate,
                    avg_duration=avg_duration,
                )
                patterns.append(pattern)

        # Sort by frequency * success_rate
        patterns.sort(key=lambda p: p.occurrences * p.success_rate, reverse=True)
        self._patterns = patterns

        return patterns

    def get_success_patterns(self, min_success_rate: float = 0.7) -> List[TrajectoryPattern]:
        """Get patterns associated with success."""
        return [p for p in self._patterns if p.success_rate >= min_success_rate]

    def get_failure_patterns(self, max_success_rate: float = 0.3) -> List[TrajectoryPattern]:
        """Get patterns associated with failure."""
        return [p for p in self._patterns if p.success_rate <= max_success_rate]

    def suggest_next_layer(self, current_trajectory: ThoughtTrajectory) -> Optional[int]:
        """
        Suggest the next layer to visit based on successful patterns.

        Looks at current layer sequence and finds the most promising continuation.
        """
        current_seq = current_trajectory.layer_sequence
        if not current_seq:
            # Start in sensory layer
            return 0

        best_next = None
        best_score = -1

        # Look for patterns that match current prefix
        for pattern in self._patterns:
            pattern_seq = pattern.layer_sequence

            # Check if current sequence matches pattern prefix
            if len(current_seq) >= len(pattern_seq):
                continue

            # Check prefix match
            prefix_len = len(current_seq)
            if pattern_seq[:prefix_len] == current_seq:
                # Pattern matches! Suggest next layer
                next_layer = pattern_seq[prefix_len]
                score = pattern.success_rate * pattern.occurrences

                if score > best_score:
                    best_score = score
                    best_next = next_layer

        return best_next


@dataclass
class TrajectoryNavigator:
    """
    Navigates through 4D thought space using trajectory patterns.

    Provides real-time guidance based on:
    - Current position in thought space
    - Known successful patterns
    - Layer-specific insights
    """
    space: LayeredSpace
    miner: TrajectoryMiner

    # Current trajectory being navigated
    _current: Optional[ThoughtTrajectory] = None

    def start_trajectory(self, goal: str, domain: str = "") -> ThoughtTrajectory:
        """Start a new trajectory."""
        self._current = ThoughtTrajectory(
            trajectory_id=f"traj_{int(time.time())}",
            goal=goal,
            domain=domain,
        )
        return self._current

    def record_point(self, concept: str, layer_id: int,
                     move_type: str = "", produced_insight: bool = False):
        """Record current position in the trajectory."""
        if self._current is None:
            raise RuntimeError("No active trajectory")

        layer = self.space.get_layer(layer_id)
        if layer is None:
            raise ValueError(f"Layer {layer_id} not found")

        hv = layer.get_item(concept)
        self._current.add(
            hv=hv,
            layer_id=layer_id,
            label=concept,
            move_type=move_type,
            produced_insight=produced_insight,
        )

    def get_guidance(self) -> Dict[str, Any]:
        """
        Get navigation guidance for current trajectory.

        Returns suggestions based on patterns.
        """
        if self._current is None:
            return {"status": "no_trajectory"}

        guidance = {
            "current_layer": self._current.layer_sequence[-1] if self._current.points else None,
            "layers_visited": list(self._current.unique_layers),
            "trajectory_length": len(self._current.points),
        }

        # Suggest next layer
        next_layer = self.miner.suggest_next_layer(self._current)
        if next_layer is not None:
            layer = self.space.get_layer(next_layer)
            guidance["suggested_layer"] = {
                "id": next_layer,
                "name": layer.name if layer else f"Layer {next_layer}",
            }

        # Check pattern matches
        matching_patterns = []
        for pattern in self.miner._patterns:
            if pattern.matches(self._current):
                matching_patterns.append({
                    "name": pattern.name,
                    "success_rate": pattern.success_rate,
                })
        guidance["matching_patterns"] = matching_patterns

        # Layer distribution
        guidance["layer_distribution"] = self._current.layer_distribution()

        # Warnings
        warnings = []
        dist = self._current.layer_distribution()

        # Warn if stuck in one layer
        for lid, frac in dist.items():
            if frac > 0.7:
                layer = self.space.get_layer(lid)
                name = layer.name if layer else f"Layer {lid}"
                warnings.append(f"Spending {frac:.0%} of time in {name} layer")

        # Warn if missing key layers
        if len(self._current.points) > 5:
            if 2 not in self._current.unique_layers:
                warnings.append("Haven't visited meta layer yet (evaluation)")
            if 3 not in self._current.unique_layers:
                warnings.append("Haven't visited teleological layer (purpose check)")

        guidance["warnings"] = warnings

        return guidance

    def complete_trajectory(self, outcome: str) -> ThoughtTrajectory:
        """Complete the current trajectory."""
        if self._current is None:
            raise RuntimeError("No active trajectory")

        self._current.outcome = outcome
        completed = self._current
        self.miner.add_trajectory(completed)
        self._current = None

        return completed
