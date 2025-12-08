"""
ThoughtTrace - Recording How Thinking Happens
===============================================

A ThoughtTrace captures a complete thinking episode:
    (Goal, Context, Moves, State-over-Time, Outcome)

This is not just a log - it's a dataset of thinking episodes that
can be mined for causal patterns.

Each trace becomes a hypervector: "the fingerprint of that thinking session."
"""

from __future__ import annotations
import time
import uuid
import hashlib
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum, auto
from collections import deque


class MoveType(Enum):
    """Types of thought moves during a session."""
    # Inquiry
    ASK_AI = auto()           # Query Ara/Claude/etc
    SEARCH_CODE = auto()       # Search codebase
    READ_DOCS = auto()         # Read documentation
    SEARCH_WEB = auto()        # Web search

    # Creation
    WRITE_CODE = auto()        # Write/edit code
    SKETCH_DIAGRAM = auto()    # Draw diagram
    WRITE_NOTES = auto()       # Write notes/thoughts
    CREATE_TABLE = auto()      # Create comparison table

    # Execution
    RUN_TEST = auto()          # Run tests
    RUN_BUILD = auto()         # Build project
    RUN_EXPERIMENT = auto()    # Run experiment
    DEPLOY = auto()            # Deploy something

    # Navigation
    SWITCH_FILE = auto()       # Switch to different file
    SWITCH_CONTEXT = auto()    # Switch problem context
    ABANDON_PATH = auto()      # Abandon current approach
    PIVOT = auto()             # Pivot to new approach

    # Meta
    REFLECT = auto()           # Pause to reflect
    BREAK = auto()             # Take a break
    SUMMON_CRITIC = auto()     # Deliberately seek criticism
    REFRAME = auto()           # Reframe the problem

    # Collaboration
    DISCUSS = auto()           # Discuss with someone
    DELEGATE = auto()          # Delegate to another agent
    MERGE_WORK = auto()        # Merge others' work


@dataclass
class ThoughtMove:
    """A single move in a thinking session."""
    move_type: MoveType
    timestamp: float
    duration_seconds: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)

    # Optional state snapshots
    typing_density: Optional[float] = None      # chars/min
    backspace_ratio: Optional[float] = None     # backspaces/total_chars
    pause_before: Optional[float] = None        # seconds paused before this move

    # Outcome of this specific move
    produced_insight: bool = False
    produced_artifact: bool = False
    led_to_dead_end: bool = False

    def to_dict(self) -> dict:
        return {
            "type": self.move_type.name,
            "timestamp": self.timestamp,
            "duration": self.duration_seconds,
            "params": self.params,
            "insight": self.produced_insight,
            "artifact": self.produced_artifact,
            "dead_end": self.led_to_dead_end,
        }


@dataclass
class SessionContext:
    """
    The context in which a thinking session occurs.

    Captures everything that might influence thinking quality.
    """
    # Task context
    domain: str = ""               # e.g., "fpga_design", "debugging", "architecture"
    difficulty_estimate: float = 0.5  # 0-1 subjective difficulty
    novelty: float = 0.5           # 0-1 how novel is this problem?
    time_pressure: float = 0.0     # 0-1 how urgent?

    # Human state (estimated or self-reported)
    fatigue: float = 0.5           # 0-1
    focus: float = 0.5             # 0-1
    valence: float = 0.5           # 0-1 (negative to positive mood)
    arousal: float = 0.5           # 0-1 (calm to excited)

    # Environment
    time_of_day: str = ""          # morning, afternoon, evening, night
    interruption_level: float = 0.0  # 0-1 how many interruptions expected
    tools_available: List[str] = field(default_factory=list)

    # Codebase/project state
    codebase_hash: str = ""        # Snapshot of codebase state
    open_files: List[str] = field(default_factory=list)
    recent_changes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "domain": self.domain,
            "difficulty": self.difficulty_estimate,
            "novelty": self.novelty,
            "time_pressure": self.time_pressure,
            "fatigue": self.fatigue,
            "focus": self.focus,
            "valence": self.valence,
            "arousal": self.arousal,
            "time_of_day": self.time_of_day,
            "tools": self.tools_available,
        }


class SessionOutcome(Enum):
    """Outcome categories for thinking sessions."""
    BREAKTHROUGH = auto()      # Major insight or solution
    PROGRESS = auto()          # Meaningful progress made
    INCREMENTAL = auto()       # Small steps forward
    STALLED = auto()           # Got stuck, no progress
    ABANDONED = auto()         # Gave up on this approach
    INTERRUPTED = auto()       # External interruption
    DEFERRED = auto()          # Consciously deferred for later


@dataclass
class ThoughtTrace:
    """
    A complete record of a thinking episode.

    This is the fundamental unit of data for the Noetic Engine.
    """
    trace_id: str
    goal: str
    context: SessionContext
    moves: List[ThoughtMove]
    outcome: SessionOutcome

    # Timing
    start_time: float
    end_time: float

    # Quality metrics (can be set after reflection)
    quality_score: float = 0.0     # 0-1 subjective quality of thinking
    insight_count: int = 0         # Number of insights generated
    artifact_count: int = 0        # Number of artifacts produced
    dead_ends: int = 0             # Number of dead ends hit

    # Optional: HDC fingerprint (set by resonance module)
    fingerprint: Optional[np.ndarray] = None

    # Tags for later analysis
    tags: List[str] = field(default_factory=list)
    notes: str = ""

    @property
    def duration_minutes(self) -> float:
        return (self.end_time - self.start_time) / 60

    @property
    def move_count(self) -> int:
        return len(self.moves)

    @property
    def moves_per_minute(self) -> float:
        if self.duration_minutes == 0:
            return 0
        return self.move_count / self.duration_minutes

    def get_move_sequence(self) -> List[MoveType]:
        """Get just the sequence of move types."""
        return [m.move_type for m in self.moves]

    def get_strategy_patterns(self) -> List[Tuple[MoveType, MoveType]]:
        """Extract consecutive move pairs (bigrams)."""
        types = self.get_move_sequence()
        return list(zip(types[:-1], types[1:]))

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "goal": self.goal,
            "context": self.context.to_dict(),
            "moves": [m.to_dict() for m in self.moves],
            "outcome": self.outcome.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_minutes": self.duration_minutes,
            "quality_score": self.quality_score,
            "insight_count": self.insight_count,
            "artifact_count": self.artifact_count,
            "dead_ends": self.dead_ends,
            "tags": self.tags,
            "notes": self.notes,
        }


@dataclass
class ThoughtTracer:
    """
    Records thinking sessions as ThoughtTraces.

    Usage:
        tracer = ThoughtTracer()
        tracer.begin_session(goal="Fix the bug")
        tracer.record_move(MoveType.ASK_AI, params={"query": "..."})
        tracer.record_move(MoveType.WRITE_CODE, duration=120)
        trace = tracer.end_session(outcome=SessionOutcome.PROGRESS)
    """
    # Configuration
    auto_timestamp: bool = True

    # State
    _current_session: Optional[dict] = field(default=None)
    _completed_traces: List[ThoughtTrace] = field(default_factory=list)
    _last_move_time: float = 0.0

    def begin_session(self, goal: str,
                      context: Optional[SessionContext] = None,
                      domain: str = "") -> str:
        """
        Begin a new thinking session.

        Returns session_id.
        """
        session_id = f"trace_{uuid.uuid4().hex[:12]}"
        now = time.time()

        if context is None:
            context = SessionContext(domain=domain)

        self._current_session = {
            "trace_id": session_id,
            "goal": goal,
            "context": context,
            "moves": [],
            "start_time": now,
        }
        self._last_move_time = now

        return session_id

    def record_move(self, move_type: MoveType,
                    params: Optional[Dict[str, Any]] = None,
                    duration: float = 0.0,
                    produced_insight: bool = False,
                    produced_artifact: bool = False,
                    led_to_dead_end: bool = False) -> ThoughtMove:
        """Record a thought move in the current session."""
        if self._current_session is None:
            raise RuntimeError("No active session. Call begin_session() first.")

        now = time.time()
        pause_before = now - self._last_move_time if self._last_move_time > 0 else 0

        move = ThoughtMove(
            move_type=move_type,
            timestamp=now if self.auto_timestamp else 0,
            duration_seconds=duration,
            params=params or {},
            pause_before=pause_before,
            produced_insight=produced_insight,
            produced_artifact=produced_artifact,
            led_to_dead_end=led_to_dead_end,
        )

        self._current_session["moves"].append(move)
        self._last_move_time = now + duration

        return move

    def update_context(self, **kwargs):
        """Update context mid-session (e.g., fatigue changed)."""
        if self._current_session is None:
            return

        context = self._current_session["context"]
        for key, value in kwargs.items():
            if hasattr(context, key):
                setattr(context, key, value)

    def end_session(self, outcome: SessionOutcome,
                    quality_score: float = 0.0,
                    notes: str = "",
                    tags: Optional[List[str]] = None) -> ThoughtTrace:
        """End the current session and create a ThoughtTrace."""
        if self._current_session is None:
            raise RuntimeError("No active session to end.")

        now = time.time()
        session = self._current_session

        # Count insights, artifacts, dead ends from moves
        insight_count = sum(1 for m in session["moves"] if m.produced_insight)
        artifact_count = sum(1 for m in session["moves"] if m.produced_artifact)
        dead_ends = sum(1 for m in session["moves"] if m.led_to_dead_end)

        trace = ThoughtTrace(
            trace_id=session["trace_id"],
            goal=session["goal"],
            context=session["context"],
            moves=session["moves"],
            outcome=outcome,
            start_time=session["start_time"],
            end_time=now,
            quality_score=quality_score,
            insight_count=insight_count,
            artifact_count=artifact_count,
            dead_ends=dead_ends,
            tags=tags or [],
            notes=notes,
        )

        self._completed_traces.append(trace)
        self._current_session = None

        return trace

    def cancel_session(self):
        """Cancel the current session without creating a trace."""
        self._current_session = None

    def get_completed_traces(self, n: Optional[int] = None) -> List[ThoughtTrace]:
        """Get completed traces."""
        if n is None:
            return self._completed_traces.copy()
        return self._completed_traces[-n:]

    @property
    def is_recording(self) -> bool:
        return self._current_session is not None

    @property
    def current_goal(self) -> Optional[str]:
        if self._current_session:
            return self._current_session["goal"]
        return None


# Global tracer instance
_global_tracer: Optional[ThoughtTracer] = None


def get_tracer() -> ThoughtTracer:
    """Get the global ThoughtTracer instance."""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = ThoughtTracer()
    return _global_tracer
