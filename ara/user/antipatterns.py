"""
Anti-Pattern Detector - Finding the Invisible Walls
====================================================

This module searches your history for "Invisible Walls" - things that
slow you down that you've accepted as "normal."

It correlates:
    - Frustration spikes (from MindReader)
    - System state (from Hippocampus/HAL)
    - Activity context (what were you doing?)

Output: FrictionPoints that the Steward can fix proactively.

The goal: Find the friction you don't know you have.

Usage:
    from ara.user.antipatterns import AntiPatternDetector, FrictionPoint

    detector = AntiPatternDetector(hippocampus, mind_reader)
    friction_points = detector.scan_for_friction()

    for fp in friction_points:
        print(f"{fp.description}: {fp.root_cause}")
        # → "High cognitive load editing flux_capacitor.py: File is 4000 lines"
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Protocol
from enum import Enum

logger = logging.getLogger(__name__)


class FrictionType(Enum):
    """Types of friction that slow users down."""
    CODE_COMPLEXITY = "code_complexity"      # File too large, tangled
    CONTEXT_SWITCHING = "context_switching"  # Too many files open
    TOOL_FRICTION = "tool_friction"          # Bad tooling, slow builds
    COGNITIVE_OVERLOAD = "cognitive_overload"  # Too much to track
    REPETITIVE_TASK = "repetitive_task"      # Same thing over and over
    DOCUMENTATION_GAP = "documentation_gap"  # Missing docs
    ENVIRONMENT_ISSUE = "environment_issue"  # System problems


@dataclass
class FrictionPoint:
    """
    A discovered point of friction in the user's workflow.

    These are the "invisible walls" - things that slow you down
    that you've accepted as normal.
    """
    id: str
    friction_type: FrictionType
    description: str          # "High cognitive load editing flux_capacitor.py"
    root_cause: str           # "File is 4000 lines long; spaghetti code"
    impact_score: float       # 0-1, how much does this hurt?
    confidence: float         # 0-1, how sure are we?

    # Evidence
    occurrences: int = 1      # How many times observed
    last_seen: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)

    # Resolution
    suggested_fix: Optional[str] = None
    can_auto_fix: bool = False
    estimated_effort: str = "unknown"  # "trivial", "small", "medium", "large"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "friction_type": self.friction_type.value,
            "description": self.description,
            "root_cause": self.root_cause,
            "impact_score": self.impact_score,
            "confidence": self.confidence,
            "occurrences": self.occurrences,
            "last_seen": self.last_seen,
            "context": self.context,
            "suggested_fix": self.suggested_fix,
            "can_auto_fix": self.can_auto_fix,
            "estimated_effort": self.estimated_effort,
        }


class HippocampusProtocol(Protocol):
    """Protocol for memory access."""
    def recall_by_state(
        self,
        valence_threshold: float,
        limit: int,
    ) -> List[Any]:
        ...

    def recall_recent(self, hours: int) -> List[Any]:
        ...


@dataclass
class PainEpisode:
    """A recorded episode of user pain/frustration."""
    id: str
    timestamp: float
    valence: float           # Emotional state (-1 to 1)
    cognitive_load: float    # Mental burden (0 to 1)
    context: Dict[str, Any]  # What was happening
    duration_seconds: float


class AntiPatternDetector:
    """
    Searches user history for invisible friction.

    This is the "Sensor" that identifies what's slowing the user down
    even when they don't realize it.
    """

    # Thresholds for detection
    FRUSTRATION_THRESHOLD = -0.3  # Valence below this = frustration
    OVERLOAD_THRESHOLD = 0.7      # Cognitive load above this = overload
    MIN_CONFIDENCE = 0.5          # Minimum confidence to report

    # File size thresholds (in lines or MB)
    LARGE_FILE_LINES = 500
    LARGE_FILE_MB = 0.5

    def __init__(
        self,
        hippocampus: Optional[HippocampusProtocol] = None,
        mind_reader: Optional[Any] = None,
    ):
        """
        Initialize the detector.

        Args:
            hippocampus: Memory system for historical recall
            mind_reader: MindReader for current state
        """
        self.memory = hippocampus
        self.mind = mind_reader
        self.log = logging.getLogger("AntiPatternDetector")

        # Cache of known friction points
        self._known_friction: Dict[str, FrictionPoint] = {}

    def scan_for_friction(
        self,
        lookback_hours: int = 24,
        limit: int = 20,
    ) -> List[FrictionPoint]:
        """
        Scan history for friction points.

        Args:
            lookback_hours: How far back to look
            limit: Maximum episodes to analyze

        Returns:
            List of discovered FrictionPoints
        """
        self.log.info("🔍 Scanning for anti-patterns...")

        friction_points = []

        # Get pain episodes from memory
        pain_episodes = self._get_pain_episodes(lookback_hours, limit)

        if not pain_episodes:
            self.log.info("No pain episodes found in recent history")
            return friction_points

        # Analyze each episode for patterns
        for ep in pain_episodes:
            detected = self._analyze_episode(ep)
            friction_points.extend(detected)

        # Deduplicate and merge
        friction_points = self._merge_similar(friction_points)

        # Filter by confidence
        friction_points = [
            fp for fp in friction_points
            if fp.confidence >= self.MIN_CONFIDENCE
        ]

        # Sort by impact
        friction_points.sort(key=lambda x: x.impact_score, reverse=True)

        self.log.info(f"🔍 Found {len(friction_points)} friction points")
        return friction_points

    def _get_pain_episodes(
        self,
        lookback_hours: int,
        limit: int,
    ) -> List[PainEpisode]:
        """Get episodes of user pain from memory."""
        if self.memory is None:
            return self._simulate_pain_episodes()

        try:
            # Try to get episodes with low valence
            raw_episodes = self.memory.recall_by_state(
                valence_threshold=self.FRUSTRATION_THRESHOLD,
                limit=limit,
            )
            return [self._to_pain_episode(ep) for ep in raw_episodes]
        except Exception as e:
            self.log.warning(f"Failed to recall from hippocampus: {e}")
            return self._simulate_pain_episodes()

    def _to_pain_episode(self, raw: Any) -> PainEpisode:
        """Convert raw memory to PainEpisode."""
        return PainEpisode(
            id=getattr(raw, 'id', str(id(raw))),
            timestamp=getattr(raw, 'timestamp', time.time()),
            valence=getattr(raw, 'valence', -0.5),
            cognitive_load=getattr(raw, 'cognitive_load', 0.7),
            context=getattr(raw, 'context', {}),
            duration_seconds=getattr(raw, 'duration', 60),
        )

    def _simulate_pain_episodes(self) -> List[PainEpisode]:
        """Generate simulated pain episodes for testing."""
        return [
            PainEpisode(
                id="sim_001",
                timestamp=time.time() - 3600,
                valence=-0.6,
                cognitive_load=0.85,
                context={
                    "activity": "editing",
                    "filename": "legacy_audio_driver.c",
                    "file_size_lines": 4200,
                    "file_size_mb": 0.8,
                },
                duration_seconds=1800,
            ),
            PainEpisode(
                id="sim_002",
                timestamp=time.time() - 7200,
                valence=-0.4,
                cognitive_load=0.75,
                context={
                    "activity": "debugging",
                    "filename": "snn_core.py",
                    "errors_encountered": 12,
                    "missing_docs": True,
                },
                duration_seconds=2400,
            ),
        ]

    def _analyze_episode(self, ep: PainEpisode) -> List[FrictionPoint]:
        """Analyze a pain episode for friction patterns."""
        detected = []
        ctx = ep.context

        # Pattern 1: Large file editing
        if ctx.get("activity") == "editing":
            file_lines = ctx.get("file_size_lines", 0)
            file_mb = ctx.get("file_size_mb", 0)
            filename = ctx.get("filename", "unknown")

            if file_lines > self.LARGE_FILE_LINES or file_mb > self.LARGE_FILE_MB:
                detected.append(FrictionPoint(
                    id=f"friction_largefile_{filename}",
                    friction_type=FrictionType.CODE_COMPLEXITY,
                    description=f"Struggle with large file: {filename}",
                    root_cause=f"File has {file_lines} lines, exceeds cognitive buffer",
                    impact_score=min(0.9, 0.5 + (file_lines / 5000)),
                    confidence=0.8,
                    context={"filename": filename, "lines": file_lines},
                    suggested_fix=f"Split {filename} into smaller modules",
                    can_auto_fix=False,
                    estimated_effort="medium",
                ))

        # Pattern 2: Debugging without docs
        if ctx.get("activity") == "debugging" and ctx.get("missing_docs"):
            filename = ctx.get("filename", "unknown")
            errors = ctx.get("errors_encountered", 0)

            detected.append(FrictionPoint(
                id=f"friction_nodocs_{filename}",
                friction_type=FrictionType.DOCUMENTATION_GAP,
                description=f"Debugging {filename} without documentation",
                root_cause="Missing API documentation makes debugging harder",
                impact_score=min(0.85, 0.4 + (errors * 0.03)),
                confidence=0.75,
                context={"filename": filename, "errors": errors},
                suggested_fix=f"Generate documentation for {filename}",
                can_auto_fix=True,
                estimated_effort="small",
            ))

        # Pattern 3: High cognitive load duration
        if ep.cognitive_load > self.OVERLOAD_THRESHOLD and ep.duration_seconds > 1800:
            activity = ctx.get("activity", "working")
            detected.append(FrictionPoint(
                id=f"friction_overload_{ep.id}",
                friction_type=FrictionType.COGNITIVE_OVERLOAD,
                description=f"Extended cognitive overload during {activity}",
                root_cause="Task complexity exceeds sustainable attention",
                impact_score=0.7,
                confidence=0.6,
                context={"duration": ep.duration_seconds, "load": ep.cognitive_load},
                suggested_fix="Consider breaking task into smaller chunks",
                can_auto_fix=False,
                estimated_effort="varies",
            ))

        # Pattern 4: Context switching (many files)
        if ctx.get("files_touched", 0) > 5:
            files = ctx.get("files_touched", 0)
            detected.append(FrictionPoint(
                id=f"friction_switching_{ep.id}",
                friction_type=FrictionType.CONTEXT_SWITCHING,
                description=f"Excessive context switching ({files} files)",
                root_cause="Working across too many files simultaneously",
                impact_score=min(0.75, 0.3 + (files * 0.05)),
                confidence=0.65,
                context={"files_touched": files},
                suggested_fix="Consider focusing on fewer files at once",
                can_auto_fix=False,
                estimated_effort="behavioral",
            ))

        return detected

    def _merge_similar(
        self,
        points: List[FrictionPoint],
    ) -> List[FrictionPoint]:
        """Merge similar friction points."""
        merged: Dict[str, FrictionPoint] = {}

        for fp in points:
            if fp.id in merged:
                # Update existing
                existing = merged[fp.id]
                existing.occurrences += 1
                existing.confidence = min(0.95, existing.confidence + 0.1)
                existing.impact_score = max(existing.impact_score, fp.impact_score)
                existing.last_seen = max(existing.last_seen, fp.last_seen)
            else:
                merged[fp.id] = fp

        return list(merged.values())

    def get_auto_fixable(self) -> List[FrictionPoint]:
        """Get friction points that can be automatically fixed."""
        all_friction = self.scan_for_friction()
        return [fp for fp in all_friction if fp.can_auto_fix]

    def get_high_impact(self, threshold: float = 0.7) -> List[FrictionPoint]:
        """Get high-impact friction points."""
        all_friction = self.scan_for_friction()
        return [fp for fp in all_friction if fp.impact_score >= threshold]


# =============================================================================
# Convenience Functions
# =============================================================================

_default_detector: Optional[AntiPatternDetector] = None


def get_antipattern_detector() -> AntiPatternDetector:
    """Get the default AntiPatternDetector."""
    global _default_detector
    if _default_detector is None:
        _default_detector = AntiPatternDetector()
    return _default_detector


def scan_for_friction() -> List[FrictionPoint]:
    """Scan for friction points."""
    return get_antipattern_detector().scan_for_friction()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'FrictionType',
    'FrictionPoint',
    'PainEpisode',
    'AntiPatternDetector',
    'get_antipattern_detector',
    'scan_for_friction',
]
