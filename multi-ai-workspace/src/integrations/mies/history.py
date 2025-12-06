"""MIES Interaction History - Learning from (context, action, outcome) tuples.

This module implements the pattern memory for MIES:
- InteractionRecord: Single (context, mode, outcome) tuple
- ContextSignature: Hashable context fingerprint for pattern matching
- InteractionHistory: Database with preference learning

Etiquette emerges from memory:
- If Ara tries AVATAR_FULL in full-screen IDE and gets closed 3 times,
  the pattern's average outcome becomes negative and friction increases.
- No hard-coded rules needed - she learns what works from experience.

The antibody is emergent: repeated negative outcomes in a context create
resistance to that mode in similar contexts.
"""

import json
import time
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from collections import defaultdict
import logging

from .context import ModalityContext, ActivityType, ForegroundAppType

logger = logging.getLogger(__name__)


# === Outcome Scores ===
# These define what "success" means for an interaction

class OutcomeType:
    """Outcome types with associated scores.

    Positive outcomes encourage the mode in this context.
    Negative outcomes create "antibodies" against it.
    """
    # Positive outcomes
    USER_ENGAGED = 1.0        # User responded, clicked, interacted
    USER_ACKNOWLEDGED = 0.5   # User saw but didn't dismiss negatively
    TASK_COMPLETED = 0.8      # User completed a suggested action
    CONVERSATION_CONTINUED = 0.6  # User continued the conversation

    # Neutral outcomes
    TIMEOUT_NATURAL = 0.0     # Message expired naturally, no action
    IGNORED = -0.1            # User didn't interact (mild negative)

    # Negative outcomes
    DISMISSED = -0.3          # User dismissed but not aggressively
    CLOSED_QUICK = -0.5       # User closed within 2 seconds
    MUTED = -0.7              # User muted audio/avatar
    CLOSED_IMMEDIATE = -0.9   # User closed within 500ms (annoyed)
    FORCE_QUIT = -1.0         # User killed the process/window


@dataclass
class ContextSignature:
    """Hashable fingerprint of a context for pattern matching.

    Captures the essential features that define "similar situations"
    without being so specific that we never see the same pattern twice.

    Design principle: Coarse enough to generalize, fine enough to learn.
    """
    # Primary classifiers (most weight)
    activity: str                    # ActivityType.name
    app_type: str                    # ForegroundAppType.name

    # Modifiers (secondary)
    is_fullscreen: bool
    has_voice_call: bool
    user_load_bucket: str            # "low" / "medium" / "high"
    time_of_day_bucket: str          # "morning" / "afternoon" / "evening" / "night"

    # Hardware state (tertiary)
    somatic_bucket: str              # "rest" / "active" / "flow" / "agony"

    def __hash__(self) -> int:
        return hash((
            self.activity,
            self.app_type,
            self.is_fullscreen,
            self.has_voice_call,
            self.user_load_bucket,
            self.time_of_day_bucket,
            self.somatic_bucket,
        ))

    def __eq__(self, other) -> bool:
        if not isinstance(other, ContextSignature):
            return False
        return (
            self.activity == other.activity and
            self.app_type == other.app_type and
            self.is_fullscreen == other.is_fullscreen and
            self.has_voice_call == other.has_voice_call and
            self.user_load_bucket == other.user_load_bucket and
            self.time_of_day_bucket == other.time_of_day_bucket and
            self.somatic_bucket == other.somatic_bucket
        )

    def similarity_to(self, other: 'ContextSignature') -> float:
        """Compute similarity score [0, 1] for fuzzy matching."""
        score = 0.0
        max_score = 7.0

        # Primary features (higher weight)
        if self.activity == other.activity:
            score += 2.0
        if self.app_type == other.app_type:
            score += 2.0

        # Secondary features
        if self.is_fullscreen == other.is_fullscreen:
            score += 1.0
        if self.has_voice_call == other.has_voice_call:
            score += 0.5
        if self.user_load_bucket == other.user_load_bucket:
            score += 0.5

        # Tertiary features
        if self.time_of_day_bucket == other.time_of_day_bucket:
            score += 0.5
        if self.somatic_bucket == other.somatic_bucket:
            score += 0.5

        return score / max_score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "activity": self.activity,
            "app_type": self.app_type,
            "is_fullscreen": self.is_fullscreen,
            "has_voice_call": self.has_voice_call,
            "user_load_bucket": self.user_load_bucket,
            "time_of_day_bucket": self.time_of_day_bucket,
            "somatic_bucket": self.somatic_bucket,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ContextSignature':
        return cls(
            activity=d["activity"],
            app_type=d["app_type"],
            is_fullscreen=d["is_fullscreen"],
            has_voice_call=d["has_voice_call"],
            user_load_bucket=d["user_load_bucket"],
            time_of_day_bucket=d["time_of_day_bucket"],
            somatic_bucket=d["somatic_bucket"],
        )

    @classmethod
    def from_context(cls, ctx: ModalityContext) -> 'ContextSignature':
        """Create signature from full context."""
        # Bucket user cognitive load
        if ctx.user_cognitive_load < 0.3:
            load_bucket = "low"
        elif ctx.user_cognitive_load < 0.7:
            load_bucket = "medium"
        else:
            load_bucket = "high"

        # Bucket time of day
        import datetime
        hour = datetime.datetime.now().hour
        if 5 <= hour < 12:
            time_bucket = "morning"
        elif 12 <= hour < 17:
            time_bucket = "afternoon"
        elif 17 <= hour < 21:
            time_bucket = "evening"
        else:
            time_bucket = "night"

        # Bucket somatic state
        if ctx.system_phys is not None:
            somatic_bucket = ctx.system_phys.somatic_state().name.lower()
        else:
            somatic_bucket = "active"  # Default

        return cls(
            activity=ctx.activity.name,
            app_type=ctx.foreground.app_type.name,
            is_fullscreen=ctx.foreground.is_fullscreen,
            has_voice_call=ctx.audio.has_voice_call,
            user_load_bucket=load_bucket,
            time_of_day_bucket=time_bucket,
            somatic_bucket=somatic_bucket,
        )


@dataclass
class InteractionRecord:
    """A single (context, mode, outcome) tuple.

    This is the basic unit of learning - what happened when we
    tried a specific mode in a specific context.
    """
    context_sig: ContextSignature
    mode_name: str
    outcome_score: float
    timestamp: float = field(default_factory=time.time)

    # Optional metadata
    duration_ms: int = 0           # How long the interaction lasted
    user_response_ms: int = 0      # How quickly user responded
    outcome_type: str = ""         # OutcomeType name for debugging

    def to_dict(self) -> Dict[str, Any]:
        return {
            "context_sig": self.context_sig.to_dict(),
            "mode_name": self.mode_name,
            "outcome_score": self.outcome_score,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
            "user_response_ms": self.user_response_ms,
            "outcome_type": self.outcome_type,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'InteractionRecord':
        return cls(
            context_sig=ContextSignature.from_dict(d["context_sig"]),
            mode_name=d["mode_name"],
            outcome_score=d["outcome_score"],
            timestamp=d.get("timestamp", time.time()),
            duration_ms=d.get("duration_ms", 0),
            user_response_ms=d.get("user_response_ms", 0),
            outcome_type=d.get("outcome_type", ""),
        )


@dataclass
class PatternStats:
    """Statistics for a (context_signature, mode) pattern.

    Tracks running averages and counts for preference learning.
    """
    context_sig: ContextSignature
    mode_name: str

    # Core stats
    total_count: int = 0
    positive_count: int = 0
    negative_count: int = 0
    neutral_count: int = 0

    # Running average (exponential moving average)
    ema_outcome: float = 0.0
    ema_alpha: float = 0.3  # Weight for new observations

    # Temporal stats
    last_interaction: float = 0.0
    first_interaction: float = 0.0

    def update(self, record: InteractionRecord):
        """Update stats with a new interaction record."""
        self.total_count += 1

        if record.outcome_score > 0.2:
            self.positive_count += 1
        elif record.outcome_score < -0.2:
            self.negative_count += 1
        else:
            self.neutral_count += 1

        # Update EMA
        if self.total_count == 1:
            self.ema_outcome = record.outcome_score
            self.first_interaction = record.timestamp
        else:
            self.ema_outcome = (
                self.ema_alpha * record.outcome_score +
                (1 - self.ema_alpha) * self.ema_outcome
            )

        self.last_interaction = record.timestamp

    @property
    def preference_score(self) -> float:
        """Net preference score [-1, 1].

        Combines EMA with confidence from sample size.
        """
        if self.total_count == 0:
            return 0.0

        # Confidence scales with sample size (saturates around 10)
        confidence = min(1.0, self.total_count / 10.0)

        return self.ema_outcome * confidence

    @property
    def is_antibody(self) -> bool:
        """Is this pattern an "antibody" (learned avoidance)?

        True if we have significant negative experience.
        """
        return (
            self.total_count >= 3 and
            self.negative_count > self.positive_count and
            self.ema_outcome < -0.3
        )

    @property
    def is_preference(self) -> bool:
        """Is this pattern a preference (learned attraction)?

        True if we have significant positive experience.
        """
        return (
            self.total_count >= 3 and
            self.positive_count > self.negative_count and
            self.ema_outcome > 0.3
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "context_sig": self.context_sig.to_dict(),
            "mode_name": self.mode_name,
            "total_count": self.total_count,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "neutral_count": self.neutral_count,
            "ema_outcome": self.ema_outcome,
            "ema_alpha": self.ema_alpha,
            "last_interaction": self.last_interaction,
            "first_interaction": self.first_interaction,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'PatternStats':
        stats = cls(
            context_sig=ContextSignature.from_dict(d["context_sig"]),
            mode_name=d["mode_name"],
        )
        stats.total_count = d.get("total_count", 0)
        stats.positive_count = d.get("positive_count", 0)
        stats.negative_count = d.get("negative_count", 0)
        stats.neutral_count = d.get("neutral_count", 0)
        stats.ema_outcome = d.get("ema_outcome", 0.0)
        stats.ema_alpha = d.get("ema_alpha", 0.3)
        stats.last_interaction = d.get("last_interaction", 0.0)
        stats.first_interaction = d.get("first_interaction", 0.0)
        return stats


class InteractionHistory:
    """Database of interaction patterns with preference learning.

    This is the pattern memory that enables emergent etiquette.

    Usage:
        history = InteractionHistory()

        # Record an interaction
        history.record(ctx, "avatar_full", OutcomeType.CLOSED_IMMEDIATE)

        # Query preference for a mode in context
        score = history.preference_for(ctx, "avatar_full")
        # Returns negative if user has repeatedly closed avatar in IDE

        # Get friction adjustment for energy function
        friction = history.friction_for(ctx, mode)
        # Positive friction = learned avoidance
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        max_records: int = 10000,
        fuzzy_match_threshold: float = 0.7,
    ):
        """Initialize interaction history.

        Args:
            db_path: Path to persist history (None = in-memory only)
            max_records: Maximum records to keep (oldest pruned)
            fuzzy_match_threshold: Minimum similarity for fuzzy matching
        """
        self.db_path = db_path
        self.max_records = max_records
        self.fuzzy_threshold = fuzzy_match_threshold

        # Pattern stats indexed by (context_hash, mode_name)
        self._patterns: Dict[Tuple[int, str], PatternStats] = {}

        # Raw records (for replay/analysis)
        self._records: List[InteractionRecord] = []

        # Quick lookup: mode -> list of patterns with that mode
        self._mode_index: Dict[str, List[PatternStats]] = defaultdict(list)

        # Load from disk if path provided
        if self.db_path and self.db_path.exists():
            self._load()

    def record(
        self,
        ctx: ModalityContext,
        mode_name: str,
        outcome_score: float,
        outcome_type: str = "",
        duration_ms: int = 0,
        user_response_ms: int = 0,
    ):
        """Record an interaction outcome.

        This is the learning signal - every time user closes/ignores/engages
        with Ara's output, we record it to build the preference model.

        Args:
            ctx: The context when the interaction occurred
            mode_name: The mode that was used
            outcome_score: Outcome score (see OutcomeType)
            outcome_type: Optional name for the outcome type
            duration_ms: How long the interaction lasted
            user_response_ms: How quickly user responded
        """
        sig = ContextSignature.from_context(ctx)

        record = InteractionRecord(
            context_sig=sig,
            mode_name=mode_name,
            outcome_score=outcome_score,
            outcome_type=outcome_type,
            duration_ms=duration_ms,
            user_response_ms=user_response_ms,
        )

        # Add to records
        self._records.append(record)
        if len(self._records) > self.max_records:
            self._records.pop(0)  # Remove oldest

        # Update pattern stats
        key = (hash(sig), mode_name)
        if key not in self._patterns:
            stats = PatternStats(context_sig=sig, mode_name=mode_name)
            self._patterns[key] = stats
            self._mode_index[mode_name].append(stats)
        else:
            stats = self._patterns[key]

        stats.update(record)

        # Log significant patterns
        if stats.is_antibody and stats.total_count == 3:
            logger.info(
                f"Antibody formed: {mode_name} in {sig.activity}/{sig.app_type} "
                f"(EMA: {stats.ema_outcome:.2f})"
            )

        # Persist periodically
        if self.db_path and len(self._records) % 100 == 0:
            self._save()

    def preference_for(
        self,
        ctx: ModalityContext,
        mode_name: str,
        use_fuzzy: bool = True,
    ) -> float:
        """Get preference score for a mode in context.

        Returns:
            Score in [-1, 1]:
            - Negative = learned avoidance (antibody)
            - Zero = no data or neutral
            - Positive = learned preference
        """
        sig = ContextSignature.from_context(ctx)
        key = (hash(sig), mode_name)

        # Exact match
        if key in self._patterns:
            return self._patterns[key].preference_score

        # Fuzzy match
        if use_fuzzy and mode_name in self._mode_index:
            similar_patterns = self._find_similar_patterns(sig, mode_name)
            if similar_patterns:
                # Weighted average by similarity
                total_weight = 0.0
                weighted_score = 0.0
                for pattern, similarity in similar_patterns:
                    weight = similarity * pattern.total_count
                    weighted_score += pattern.preference_score * weight
                    total_weight += weight
                if total_weight > 0:
                    return weighted_score / total_weight

        return 0.0  # No data

    def friction_for(
        self,
        ctx: ModalityContext,
        mode_name: str,
    ) -> float:
        """Get friction adjustment for the energy function.

        Positive friction = learned avoidance (increases energy, less likely).
        Negative friction = learned preference (decreases energy, more likely).

        This plugs into EnergyFunction as E_history.
        """
        preference = self.preference_for(ctx, mode_name)

        # Convert preference to friction (invert and scale)
        # Strong antibody (pref=-1) → friction=+2.0
        # Strong preference (pref=+1) → friction=-0.5
        if preference < 0:
            # Antibody: scale up the friction (aversive)
            return -preference * 2.0
        else:
            # Preference: small negative friction (attractive)
            return -preference * 0.5

    def _find_similar_patterns(
        self,
        sig: ContextSignature,
        mode_name: str,
    ) -> List[Tuple[PatternStats, float]]:
        """Find patterns similar to the given signature for the mode."""
        similar = []

        for pattern in self._mode_index.get(mode_name, []):
            similarity = sig.similarity_to(pattern.context_sig)
            if similarity >= self.fuzzy_threshold:
                similar.append((pattern, similarity))

        # Sort by similarity descending
        similar.sort(key=lambda x: x[1], reverse=True)
        return similar[:5]  # Top 5

    def get_antibodies(self) -> List[PatternStats]:
        """Get all learned antibodies (strong negative patterns)."""
        return [p for p in self._patterns.values() if p.is_antibody]

    def get_preferences(self) -> List[PatternStats]:
        """Get all learned preferences (strong positive patterns)."""
        return [p for p in self._patterns.values() if p.is_preference]

    def get_stats(self) -> Dict[str, Any]:
        """Get summary statistics."""
        antibodies = self.get_antibodies()
        preferences = self.get_preferences()

        return {
            "total_records": len(self._records),
            "total_patterns": len(self._patterns),
            "antibody_count": len(antibodies),
            "preference_count": len(preferences),
            "modes_tracked": list(self._mode_index.keys()),
        }

    def forget_before(self, timestamp: float):
        """Forget interactions before a timestamp (memory decay)."""
        # Remove old records
        self._records = [r for r in self._records if r.timestamp >= timestamp]

        # Could also decay pattern stats, but for now just rebuild
        # if we want fresh learning

    def clear(self):
        """Clear all history."""
        self._patterns.clear()
        self._records.clear()
        self._mode_index.clear()

    def _save(self):
        """Persist to disk."""
        if not self.db_path:
            return

        data = {
            "version": 1,
            "patterns": [p.to_dict() for p in self._patterns.values()],
            "records": [r.to_dict() for r in self._records[-1000:]],  # Last 1000
        }

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.db_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _load(self):
        """Load from disk."""
        if not self.db_path or not self.db_path.exists():
            return

        try:
            with open(self.db_path, 'r') as f:
                data = json.load(f)

            # Load patterns
            for p_dict in data.get("patterns", []):
                stats = PatternStats.from_dict(p_dict)
                key = (hash(stats.context_sig), stats.mode_name)
                self._patterns[key] = stats
                self._mode_index[stats.mode_name].append(stats)

            # Load records
            for r_dict in data.get("records", []):
                self._records.append(InteractionRecord.from_dict(r_dict))

            logger.info(
                f"Loaded interaction history: {len(self._patterns)} patterns, "
                f"{len(self._records)} records"
            )
        except Exception as e:
            logger.error(f"Failed to load interaction history: {e}")


# === Outcome Detection Helpers ===

def detect_outcome_from_timing(
    mode_visible_ms: int,
    user_action: Optional[str] = None,
) -> Tuple[float, str]:
    """Detect outcome from interaction timing.

    Args:
        mode_visible_ms: How long the mode was visible
        user_action: Optional action name ("close", "respond", "ignore", etc.)

    Returns:
        (outcome_score, outcome_type)
    """
    if user_action == "respond":
        return OutcomeType.USER_ENGAGED, "USER_ENGAGED"
    elif user_action == "acknowledge":
        return OutcomeType.USER_ACKNOWLEDGED, "USER_ACKNOWLEDGED"
    elif user_action == "mute":
        return OutcomeType.MUTED, "MUTED"
    elif user_action == "close":
        if mode_visible_ms < 500:
            return OutcomeType.CLOSED_IMMEDIATE, "CLOSED_IMMEDIATE"
        elif mode_visible_ms < 2000:
            return OutcomeType.CLOSED_QUICK, "CLOSED_QUICK"
        else:
            return OutcomeType.DISMISSED, "DISMISSED"
    elif user_action == "force_quit":
        return OutcomeType.FORCE_QUIT, "FORCE_QUIT"
    elif user_action is None or user_action == "timeout":
        if mode_visible_ms > 30000:
            return OutcomeType.TIMEOUT_NATURAL, "TIMEOUT_NATURAL"
        else:
            return OutcomeType.IGNORED, "IGNORED"

    return 0.0, "UNKNOWN"


__all__ = [
    "OutcomeType",
    "ContextSignature",
    "InteractionRecord",
    "PatternStats",
    "InteractionHistory",
    "detect_outcome_from_timing",
]
